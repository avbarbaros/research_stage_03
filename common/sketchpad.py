"""Visual Sketchpad-style multi-turn reasoning (Hu et al., NeurIPS 2024), minimal version.

The model reasons in turns. In each turn it either writes one action line,

    ACTION: zoom(x1, y1, x2, y2)      crop a region of the first image and show it enlarged
    ACTION: mark("cup")               show the image with numbered boxes around detected cups
    ACTION: draw_line(x1, y1, x2, y2) draw on the first image and show it (draw_point(x, y) too)
    ACTION: depth()                   show a depth map of the first image

which is executed and its result image appended to the conversation, or it ends
with "ANSWER: ...". Coordinates refer to the first image *as the model sees it*
(Qwen2.5-VL's resized frame, whose size is stated in the prompt) and are mapped
back to the original image with the same resize rule validated in E3.2.

All items of a chunk advance turn by turn, so each turn is one batched generation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.tasks import smart_resize

_ACTION = re.compile(r"ACTION\s*:\s*([A-Za-z_]+)\s*\(([^()]*)\)", re.IGNORECASE)
_ANSWER = re.compile(r"ANSWER\s*:", re.IGNORECASE)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

ACTION_FAMILY = {"zoom": "zoom", "mark": "mark", "draw_line": "draw", "draw_point": "draw", "depth": "depth"}

ACTION_DOCS = {
    "zoom": 'ACTION: zoom(x1, y1, x2, y2) - show the region with corners (x1, y1) and (x2, y2) of the first image, enlarged.',
    "mark": 'ACTION: mark("object") - show the first image with numbered boxes around every detected instance of the object, and list their coordinates.',
    "draw": 'ACTION: draw_line(x1, y1, x2, y2) - draw a red line on the first image and show it (for example an auxiliary line in a figure).\n'
            'ACTION: draw_point(x, y) - mark a point on the first image and show it.',
    "depth": "ACTION: depth() - show a depth map of the first image (brighter = closer to the camera).",
}


# =============================================================================
# Parsing a turn
# =============================================================================
def parse_turn(text: str) -> dict:
    """{"kind": "action", "name", "args", "call", "text"} or {"kind": "final", "text"}.

    The first ACTION or ANSWER line decides. Text after an action line is cut:
    the model sometimes continues as if it had already seen the result."""
    a = _ACTION.search(text or "")
    f = _ANSWER.search(text or "")
    if a and (not f or a.start() < f.start()):
        end = text.find("\n", a.end())
        kept = text[: end if end >= 0 else len(text)].rstrip()
        return {"kind": "action", "name": a.group(1).lower(), "args": a.group(2).strip(), "call": a.group(0), "text": kept}
    return {"kind": "final", "text": text or ""}


# =============================================================================
# Image helpers
# =============================================================================
def fit_pixels(img, max_pixels: int, max_upscale: float = 1.0):
    """Resize so that width*height <= max_pixels; upscale at most `max_upscale` times."""
    w, h = img.size
    s = math.sqrt(max_pixels / float(w * h))
    s = min(s, max_upscale)
    if abs(s - 1.0) < 1e-3:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))))


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


# mark() needs a real object name; these return a request for one instead of boxes
GENERIC_NAMES = {"object", "objects", "thing", "things", "item", "items", "anything", "everything", "something", "all"}

_PALETTE = [(255, 0, 0), (0, 160, 255), (0, 200, 0), (255, 140, 0), (200, 0, 200), (0, 200, 200), (255, 220, 0), (140, 90, 255)]


# =============================================================================
# Per-item state and action execution
# =============================================================================
@dataclass
class ItemState:
    item: Any
    messages: list
    model_w: int
    model_h: int
    allowed: set
    turns: list = field(default_factory=list)
    n_actions: int = 0
    done: bool = False
    forced_final: bool = False
    canvas: Any = None          # first image with the drawings made so far
    calls: set = field(default_factory=set)  # (action, normalised args) already executed
    final_text: str = ""
    out_tokens: int = 0

    @property
    def sx(self) -> float:
        return self.item.image.size[0] / self.model_w

    @property
    def sy(self) -> float:
        return self.item.image.size[1] / self.model_h

    def to_orig(self, xs: list[float]) -> list[float]:
        return [v * (self.sx if k % 2 == 0 else self.sy) for k, v in enumerate(xs)]

    def to_model(self, box: list[float]) -> list[int]:
        return [round(v / (self.sx if k % 2 == 0 else self.sy)) for k, v in enumerate(box)]


class ActionRunner:
    def __init__(self, scfg: dict, tools: dict):
        self.cfg = scfg
        self.tools = tools  # {"detector": callable or None, "depth": callable or None}

    def _added(self, img, max_pixels: int | None = None):
        mp = int(max_pixels or self.cfg.get("added_max_pixels", 401408))
        return fit_pixels(img, mp, float(self.cfg.get("zoom_max_upscale", 4.0)))

    def run(self, st: ItemState, name: str, args: str) -> dict:
        """Execute one action. Returns {"image": PIL or None, "note": str, "ok": bool, "family", "box"}."""
        fam = ACTION_FAMILY.get(name)
        if fam is None:
            return {"image": None, "note": f"Unknown action {name}.", "ok": False, "family": None}
        if fam not in st.allowed:
            return {"image": None, "note": f"The action {name} is not available.", "ok": False, "family": fam}
        key = (name, re.sub(r"\s+", "", args.lower()))
        if key in st.calls:  # guard: an exact repeat shows nothing new
            return {"image": None, "note": "You already used this exact action; its result is shown above. "
                    "Do something different, or give the final answer.", "ok": False, "family": fam, "rejected": "repeat"}
        st.calls.add(key)
        try:
            return {"family": fam, **getattr(self, f"_{name}")(st, args)}
        except Exception as e:  # noqa: BLE001 - a bad action is a result, not a crash
            return {"image": None, "note": f"The action failed ({type(e).__name__}: {e}).", "ok": False, "family": fam}

    # -- actions ---------------------------------------------------------------
    def _nums(self, args: str, n: int) -> list[float]:
        xs = [float(v) for v in _NUM.findall(args)]
        if len(xs) < n:
            raise ValueError(f"expected {n} numbers, got {len(xs)}")
        return xs[:n]

    def _zoom(self, st: ItemState, args: str) -> dict:
        x1, y1, x2, y2 = st.to_orig(self._nums(args, 4))
        W, H = st.item.image.size
        x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
        y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
        m = float(self.cfg.get("zoom_min_px", 64))
        if x2 - x1 < m:
            c = (x1 + x2) / 2
            x1, x2 = max(0, c - m / 2), min(W, c + m / 2)
        if y2 - y1 < m:
            c = (y1 + y2) / 2
            y1, y2 = max(0, c - m / 2), min(H, c + m / 2)
        box = [round(v, 1) for v in (x1, y1, x2, y2)]
        if (x2 - x1) * (y2 - y1) > float(self.cfg.get("zoom_max_area_frac", 0.5)) * W * H:
            # guard: a zoom on (most of) the whole image shows less detail than the original, not more
            return {"image": None, "note": "This region covers most of the image, so zooming would not show more detail. "
                    "Zoom into a smaller region around the object you need.", "ok": False, "box": box, "rejected": "too_large"}
        crop = st.item.image.crop((int(x1), int(y1), int(x2), int(y2)))
        return {"image": self._added(crop, self.cfg.get("zoom_max_pixels")), "note": "", "ok": True, "box": box}

    def _mark(self, st: ItemState, args: str) -> dict:
        from PIL import ImageDraw

        name = args.strip().strip("\"'").strip()
        if not name:
            raise ValueError("no object name")
        if name.lower() in GENERIC_NAMES:  # guard: "object" finds arbitrary things
            return {"image": None, "note": 'mark() needs the name of a specific object from the question, e.g. mark("cup").',
                    "ok": False, "boxes": [], "rejected": "generic_name"}
        det = self.tools.get("detector")
        if det is None:
            raise RuntimeError("detector not loaded")
        boxes = det(st.item.image, name)
        if not boxes:
            return {"image": None, "note": f'No "{name}" was detected.', "ok": True, "boxes": []}
        img = st.item.image.copy()
        d = ImageDraw.Draw(img)
        W, H = img.size
        lw = max(2, round(max(W, H) / 250))
        font = _font(max(12, round(max(W, H) / 40)))
        lines = []
        for k, (b, _s) in enumerate(boxes[: int(self.cfg.get("mark_max", 9))], 1):
            col = _PALETTE[(k - 1) % len(_PALETTE)]
            d.rectangle(b, outline=col, width=lw)
            tx, ty = b[0] + lw, b[1] + lw
            tb = d.textbbox((tx, ty), str(k), font=font)
            d.rectangle(tb, fill=col)
            d.text((tx, ty), str(k), fill=(255, 255, 255), font=font)
            mb = st.to_model(b)
            lines.append(f"{k}: ({mb[0]}, {mb[1]}, {mb[2]}, {mb[3]})")
        note = f'Detected {len(lines)} "{name}": ' + "; ".join(lines) + " (coordinates in the first image)."
        return {"image": self._added(img), "note": note, "ok": True, "boxes": [[round(v, 1) for v in b] for b, _ in boxes]}

    def _canvas(self, st: ItemState):
        if st.canvas is None:
            st.canvas = st.item.image.copy()
        return st.canvas

    def _draw_line(self, st: ItemState, args: str) -> dict:
        from PIL import ImageDraw

        p = st.to_orig(self._nums(args, 4))
        c = self._canvas(st)
        lw = max(2, round(max(c.size) / 200))
        ImageDraw.Draw(c).line(p, fill=(255, 0, 0), width=lw)
        return {"image": self._added(c), "note": "", "ok": True, "points": [round(v, 1) for v in p]}

    def _draw_point(self, st: ItemState, args: str) -> dict:
        from PIL import ImageDraw

        x, y = st.to_orig(self._nums(args, 2))
        c = self._canvas(st)
        r = max(3, round(max(c.size) / 120))
        ImageDraw.Draw(c).ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
        return {"image": self._added(c), "note": "", "ok": True, "points": [round(x, 1), round(y, 1)]}

    def _depth(self, st: ItemState, args: str) -> dict:
        import numpy as np
        from PIL import Image

        dep = self.tools.get("depth")
        if dep is None:
            raise RuntimeError("depth model not loaded")
        d = dep(st.item.image)  # 0 = closest, 1 = farthest
        img = Image.fromarray(((1.0 - d) * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        return {"image": self._added(img), "note": "", "ok": True}


# =============================================================================
# Prompts
# =============================================================================
def answer_hint(item) -> str:
    if item.answer_type == "mc":
        return "(X), where X is the option letter"
    return f"N, where N is {item.fmt or 'a single word or phrase'}"


def options_block(item) -> str:
    return f"Options: {item.options_str()}\n" if item.answer_type == "mc" else ""


def first_prompt(item, template: str, allowed: list[str], max_actions: int, mw: int, mh: int) -> str:
    docs = "\n".join(ACTION_DOCS[a] for a in allowed)
    return (template.replace("{question}", item.question).replace("{options_block}", options_block(item))
            .replace("{answer_hint}", answer_hint(item)).replace("{action_list}", docs)
            .replace("{max_actions}", str(max_actions)).replace("{width}", str(mw)).replace("{height}", str(mh)))


# =============================================================================
# The loop
# =============================================================================
def run_chunk(items, vlm, runner: ActionRunner, allowed: list[str], template: str, scfg: dict, gp,
              model_cfg: dict, image_dir: Path | None, condition: str, desc: str = "") -> list[ItemState]:
    from common.vlm import Request

    max_actions = int(scfg.get("max_actions", 4)) if allowed else 0
    min_px, max_px = model_cfg.get("min_pixels") or 56 * 56, model_cfg.get("max_pixels") or 28 * 28 * 1280
    states = []
    for it in items:
        W, H = it.image.size
        mh, mw = smart_resize(H, W, 28, min_px, max_px)
        text = first_prompt(it, template, allowed, max_actions, mw, mh)
        msgs = [{"role": "user", "content": [{"type": "image", "image": it.image}, {"type": "text", "text": text}]}]
        states.append(ItemState(item=it, messages=msgs, model_w=mw, model_h=mh, allowed=set(allowed)))

    for turn in range(max_actions + 1):
        active = [s for s in states if not s.done]
        if not active:
            break
        if turn == max_actions and max_actions > 0:
            for s in active:  # out of actions: ask for the answer
                s.forced_final = True
                s.messages[-1]["content"].append(
                    {"type": "text", "text": scfg["force_final"].replace("{answer_hint}", answer_hint(s.item))})
        outs = vlm.generate([Request(messages=s.messages) for s in active], gp, f"{desc} turn {turn + 1}")
        for s, o in zip(active, outs):
            s.out_tokens += o.n_tokens
            t = parse_turn(o.text)
            if t["kind"] == "final" or not allowed:
                s.final_text, s.done = o.text, True
                s.turns.append({"text": o.text, "finish": o.finish_reason})
                continue
            if s.forced_final:  # still acting after the forced-answer request: ask once more (below)
                s.turns.append({"text": t["text"], "finish": o.finish_reason, "ignored_action": t["name"]})
                s.messages.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
                s.messages.append({"role": "user", "content": [{"type": "text", "text": scfg["force_retry"].replace(
                    "{answer_hint}", answer_hint(s.item))}]})
                continue
            res = runner.run(s, t["name"], t["args"])
            s.n_actions += 1
            rec = {"text": t["text"], "finish": o.finish_reason, "action": t["name"], "family": res.get("family"),
                   "args": t["args"], "ok": res["ok"], "note": res["note"]}
            for k in ("box", "boxes", "points", "rejected"):
                if k in res:
                    rec[k] = res[k]
            if res.get("image") is not None and image_dir is not None:
                image_dir.mkdir(parents=True, exist_ok=True)
                p = image_dir / f"{s.item.id}_{condition}_{s.n_actions}.jpg"
                res["image"].convert("RGB").save(p, quality=int(scfg.get("image_quality", 70)))
                rec["image_file"] = p.name
            s.turns.append(rec)
            s.messages.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
            content = [{"type": "text", "text": f"Result of {t['call']}:" + (f" {res['note']}" if res["note"] else "")}]
            if res.get("image") is not None:
                content.append({"type": "image", "image": res["image"]})
            content.append({"type": "text", "text": scfg["after_action"]})
            s.messages.append({"role": "user", "content": content})
    retry = [s for s in states if not s.done]  # forced, but answered with another action
    if retry:
        outs = vlm.generate([Request(messages=s.messages) for s in retry], gp, f"{desc} final retry")
        for s, o in zip(retry, outs):
            s.out_tokens += o.n_tokens
            s.final_text, s.done = o.text, True
            s.turns.append({"text": o.text, "finish": o.finish_reason, "retry": True})
    return states
