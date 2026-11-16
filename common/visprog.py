"""ViperGPT-style visual programs (Surís et al., ICCV 2023), minimal version.

A planner LLM writes `def execute_command(image) -> str` against the
`ImagePatch` API below. The program is executed with three tools:

    find / exists / count   Grounding DINO (open-vocabulary detector)
    compute_depth           Depth Anything V2 (relative monocular depth)
    simple_query / ...      the VLM itself, asked about a cropped region ("crop and re-query")

Every tool call is appended to a trace, so each answer can be inspected and
debugged: which objects were found where, what was asked of which crop, and
what came back.

Coordinates: pixels of the full image, x to the right, y DOWNWARD (y=0 is the top).
"""
from __future__ import annotations

import math
import re
import signal
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Tool back-ends
# =============================================================================


class Detector:
    """Grounding DINO through transformers' zero-shot object detection API."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg["model"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["model"]).to(device).eval()
        self.torch = torch

    def __call__(self, image, name: str) -> list[tuple[list[float], float]]:
        """Boxes [x1, y1, x2, y2] in `image` pixels with scores, best first."""
        torch = self.torch
        text = name.strip().lower().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target = [image.size[::-1]]
        bt, tt = float(self.cfg.get("box_threshold", 0.35)), float(self.cfg.get("text_threshold", 0.25))
        try:  # transformers >= 4.51
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=bt, text_threshold=tt, target_sizes=target
            )[0]
        except TypeError:  # older transformers
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_threshold=bt, text_threshold=tt, target_sizes=target
            )[0]
        boxes, scores = res["boxes"], res["scores"]
        if len(boxes) == 0:
            return []
        try:
            from torchvision.ops import nms

            keep = nms(boxes.float(), scores.float(), float(self.cfg.get("nms_iou", 0.5)))
            boxes, scores = boxes[keep], scores[keep]
        except ImportError:
            pass
        order = scores.argsort(descending=True)[: int(self.cfg.get("max_detections", 20))]
        return [([float(v) for v in boxes[i].tolist()], float(scores[i])) for i in order]


class DepthEstimator:
    """Depth Anything V2. Returns a map normalised per image to [0, 1], 0 = closest."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(cfg["model"])
        self.model = AutoModelForDepthEstimation.from_pretrained(cfg["model"]).to(device).eval()
        self.device = device
        self.torch = torch

    def __call__(self, image):
        import numpy as np

        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pred = self.model(**inputs).predicted_depth  # relative inverse depth: larger = closer
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False)
        d = pred.squeeze().float().cpu().numpy()
        lo, hi = float(d.min()), float(d.max())
        d = (d - lo) / (hi - lo + 1e-6)
        return (1.0 - d).astype(np.float32)  # 0 = closest, 1 = farthest


# =============================================================================
# Per-program context (trace, budget, caches)
# =============================================================================


class ToolBudgetExceeded(RuntimeError):
    pass


@dataclass
class Context:
    image: Any
    vlm: Any
    detector: Any
    depth: Any
    query_cfg: dict
    max_calls: int = 30
    trace: list = field(default_factory=list)
    n_calls: int = 0
    _det_cache: dict = field(default_factory=dict)
    _depth_map: Any = None

    def tick(self):
        self.n_calls += 1
        if self.n_calls > self.max_calls:
            raise ToolBudgetExceeded(f"more than {self.max_calls} tool calls")

    def ask(self, image, question: str) -> str:
        from common.vlm import GenParams, Request

        prompt = self.query_cfg["prompt"].format(question=question)
        gp = GenParams(max_new_tokens=int(self.query_cfg.get("max_new_tokens", 16)))
        return self.vlm.generate([Request(prompt, image)], gp, "query", progress=False)[0].text.strip()

    def depth_map(self):
        if self._depth_map is None:
            self._depth_map = self.depth(self.image)
        return self._depth_map


_CTX: Context | None = None  # set by run_program for the duration of one program


def _r(b):
    return [round(v, 1) for v in b]


class ImagePatch:
    """A rectangular region of the image (full-image pixel coordinates, y downward)."""

    def __init__(self, image=None, left=None, top=None, right=None, bottom=None, score: float = 1.0):
        ctx = _CTX
        if ctx is None:
            raise RuntimeError("ImagePatch used outside run_program")
        W, H = ctx.image.size
        self.image_width, self.image_height = W, H
        self.left = int(max(0, 0 if left is None else left))
        self.top = int(max(0, 0 if top is None else top))
        self.right = int(min(W, W if right is None else right))
        self.bottom = int(min(H, H if bottom is None else bottom))
        if self.right <= self.left:
            self.right = min(W, self.left + 1)
        if self.bottom <= self.top:
            self.bottom = min(H, self.top + 1)
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.center_x = (self.left + self.right) / 2
        self.center_y = (self.top + self.bottom) / 2
        self.score = float(score)

    # -- helpers ------------------------------------------------------------
    @property
    def box(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]

    def _is_full(self) -> bool:
        return self.left == 0 and self.top == 0 and self.right == self.image_width and self.bottom == self.image_height

    def _crop_image(self, margin: float = 0.0):
        ctx = _CTX
        mx, my = self.width * margin, self.height * margin
        x1, y1 = max(0, self.left - mx), max(0, self.top - my)
        x2, y2 = min(self.image_width, self.right + mx), min(self.image_height, self.bottom + my)
        m = int(ctx.query_cfg.get("min_crop_px", 32))
        if x2 - x1 < m:  # grow tiny crops around their centre
            cx = (x1 + x2) / 2
            x1, x2 = max(0, cx - m / 2), min(self.image_width, cx + m / 2)
        if y2 - y1 < m:
            cy = (y1 + y2) / 2
            y1, y2 = max(0, cy - m / 2), min(self.image_height, cy + m / 2)
        return ctx.image.crop((int(x1), int(y1), int(x2), int(y2)))

    def __repr__(self) -> str:
        return f"ImagePatch(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom}, score={self.score:.2f})"

    # -- detector -----------------------------------------------------------
    def find(self, object_name: str) -> list["ImagePatch"]:
        ctx = _CTX
        ctx.tick()
        key = (tuple(self.box), object_name.strip().lower())
        if key not in ctx._det_cache:
            crop = ctx.image if self._is_full() else self._crop_image(0.0)
            dets = ctx.detector(crop, object_name)
            ctx._det_cache[key] = [
                ([b[0] + self.left, b[1] + self.top, b[2] + self.left, b[3] + self.top], s) for b, s in dets
            ]
        found = [ImagePatch(None, *b, score=s) for b, s in ctx._det_cache[key]]
        ctx.trace.append({"op": "find", "patch": self.box, "name": object_name, "n": len(found),
                          "boxes": [_r(p.box) for p in found[:10]], "scores": [round(p.score, 2) for p in found[:10]]})
        return found

    def exists(self, object_name: str) -> bool:
        return len(self.find(object_name)) > 0

    def count(self, object_name: str) -> int:
        return len(self.find(object_name))

    # -- VLM on a crop ------------------------------------------------------
    def simple_query(self, question: str) -> str:
        ctx = _CTX
        ctx.tick()
        img = ctx.image if self._is_full() else self._crop_image(float(ctx.query_cfg.get("crop_margin", 0.1)))
        ans = ctx.ask(img, question)
        ctx.trace.append({"op": "simple_query", "patch": self.box, "full": self._is_full(), "q": question, "a": ans})
        return ans

    def verify_property(self, object_name: str, attribute: str) -> bool:
        a = self.simple_query(f"Is the {object_name} {attribute}? Answer yes or no.")
        return a.strip().lower().startswith("yes")

    def best_text_match(self, options: list[str]) -> str:
        opts = [str(o) for o in options]
        a = self.simple_query("Which of these best describes the image: " + ", ".join(opts) + "? Answer with one of them exactly.")
        low = a.strip().lower()
        for o in opts:
            if o.lower() == low:
                return o
        for o in opts:
            if o.lower() in low:
                return o
        return opts[0] if opts else a

    # -- depth --------------------------------------------------------------
    def compute_depth(self) -> float:
        import numpy as np

        ctx = _CTX
        ctx.tick()
        d = ctx.depth_map()[self.top : self.bottom, self.left : self.right]
        v = float(np.median(d)) if d.size else 1.0
        ctx.trace.append({"op": "compute_depth", "patch": self.box, "value": round(v, 3)})
        return v

    # -- geometry -----------------------------------------------------------
    def crop(self, left, top, right, bottom) -> "ImagePatch":
        return ImagePatch(None, left, top, right, bottom)

    def overlaps(self, other: "ImagePatch") -> bool:
        return not (self.right <= other.left or other.right <= self.left or self.bottom <= other.top or other.bottom <= self.top)


def distance(a: ImagePatch, b: ImagePatch) -> float:
    """Distance between the centres of two patches, in pixels."""
    return math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)


def bool_to_yesno(b: bool) -> str:
    return "yes" if b else "no"


# =============================================================================
# Program extraction and execution
# =============================================================================

_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "abs all any bool dict enumerate filter float int isinstance len list map max min "
        "print range reversed round set sorted str sum tuple zip ValueError IndexError Exception"
    ).split()
}


def extract_code(text: str) -> str | None:
    """The program from a planner output: a ```python block, else from `def execute_command`."""
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.DOTALL)
    for block in m:
        if "def execute_command" in block:
            return block.strip()
    i = (text or "").find("def execute_command")
    if i >= 0:
        return text[i:].split("```")[0].strip()
    return None


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _to_answer(x) -> str | None:
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, ImagePatch):
        return None
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def run_program(code: str | None, image, vlm, detector, depth, query_cfg: dict, timeout_s: int = 60, max_calls: int = 30) -> dict:
    """Execute one program. Never raises; errors are returned in the result."""
    global _CTX
    res = {"answer": None, "error": None, "error_type": None, "trace": [], "n_calls": 0, "seconds": 0.0}
    if not code:
        res.update(error="no execute_command found in planner output", error_type="NoProgram")
        return res
    ctx = Context(image=image, vlm=vlm, detector=detector, depth=depth, query_cfg=query_cfg, max_calls=max_calls)
    ns = {"__builtins__": _SAFE_BUILTINS, "ImagePatch": ImagePatch, "distance": distance,
          "bool_to_yesno": bool_to_yesno, "math": math}
    t0 = time.time()
    _CTX = ctx
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout_s))
    try:
        exec(compile(code, "<program>", "exec"), ns)
        fn = ns.get("execute_command")
        if fn is None:
            raise NameError("execute_command is not defined")
        out = fn(image)
        ans = _to_answer(out)
        if ans is None:
            res.update(error=f"program returned {type(out).__name__}: {str(out)[:80]}", error_type="EmptyAnswer")
        else:
            res["answer"] = ans
    except _Timeout:
        res.update(error=f"timeout after {timeout_s}s", error_type="Timeout")
    except ToolBudgetExceeded as e:
        res.update(error=str(e), error_type="ToolBudget")
    except SyntaxError as e:
        res.update(error=f"SyntaxError: {e.msg} (line {e.lineno})", error_type="SyntaxError")
    except Exception as e:  # noqa: BLE001 - any program bug is a result, not a crash
        tb = traceback.extract_tb(e.__traceback__)
        src = code.splitlines()
        line = next((f"line {fr.lineno}: {src[fr.lineno - 1].strip() if 0 < fr.lineno <= len(src) else ''}"
                     for fr in reversed(tb) if fr.filename == "<program>"), "")
        res.update(error=f"{type(e).__name__}: {e} [{line}]", error_type=type(e).__name__)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        _CTX = None
    res["trace"], res["n_calls"], res["seconds"] = ctx.trace, ctx.n_calls, round(time.time() - t0, 2)
    return res


def trace_summary(trace: list) -> dict:
    """Which tools a program actually used."""
    ops = [t["op"] for t in trace]
    full_query_only = ops == ["simple_query"] and bool(trace[0].get("full"))
    return {
        "n_find": ops.count("find"),
        "n_query": ops.count("simple_query"),
        "n_depth": ops.count("compute_depth"),
        "used_detector": "find" in ops,
        "used_depth": "compute_depth" in ops,
        "empty_find": any(t["op"] == "find" and t["n"] == 0 for t in trace),
        # the program only asked the VLM one question about the whole image: no decomposition
        "degenerate": full_query_only,
    }
