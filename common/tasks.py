"""Task families for E3.2 (and later experiments): one item format, one scorer.

Every dataset loader returns a list of `TaskItem`. An item's `answer_type`
decides which prompt template is used and how the output is scored:

    mc      multiple choice; gold is an option letter
    number  numeric answer; gold is a float, compared after rounding to `precision` decimals
    text    short free-form text; normalised exact match
    list    list of numbers; element-wise match after rounding
    bbox    referring-expression grounding; gold is [x1, y1, x2, y2] in original pixels,
            correct when IoU >= iou_threshold (the standard RefCOCO Acc@0.5)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from common.parsing import (
    LETTERS,
    extract_bbox,
    extract_choice,
    extract_number,
    extract_number_list,
    answer_span,
    iou,
    normalize_text,
)


@dataclass
class TaskItem:
    id: str
    family: str
    dataset: str
    row: int
    question: str
    answer_type: str                      # mc | number | text | list | bbox
    gold: Any
    image: Any = field(default=None, repr=False)
    choices: list[str] | None = None
    precision: int = 0                    # decimals for number / list
    fmt: str = ""                         # answer-format phrase for open questions
    subgroup: str = "all"
    extra: dict = field(default_factory=dict)

    def options_str(self) -> str:
        return " ".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(self.choices or []))

    def meta(self) -> dict:
        return {
            "id": self.id,
            "family": self.family,
            "dataset": self.dataset,
            "row": self.row,
            "subgroup": self.subgroup,
            "answer_type": self.answer_type,
            "gold": self.gold,
            "question": self.question,
            # everything score() needs, so saved outputs can be re-scored without the dataset
            "choices": self.choices,
            "precision": self.precision,
            "fmt": self.fmt,
            **self.extra,
        }


_EXTRA_KEYS = ("orig_w", "orig_h", "model_w", "model_h")


def item_from_row(row: dict) -> "TaskItem | None":
    """Rebuild the scoring-relevant part of an item from a saved prediction row.

    Returns None for rows written before choices/precision were stored
    (they cannot be re-scored and keep their original score)."""
    if "precision" not in row or (row["answer_type"] == "mc" and not row.get("choices")):
        return None
    if row["answer_type"] == "bbox" and not all(k in row for k in _EXTRA_KEYS):
        return None
    return TaskItem(
        id=row["id"], family=row["family"], dataset=row["dataset"], row=row["row"], question=row["question"],
        answer_type=row["answer_type"], gold=row["gold"], choices=row.get("choices"), precision=int(row["precision"]),
        fmt=row.get("fmt", ""), subgroup=row["subgroup"], extra={k: row[k] for k in _EXTRA_KEYS if k in row},
    )


# =============================================================================
# Qwen2.5-VL image resizing (needed to map predicted boxes back to the original image)
# =============================================================================
def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 1280) -> tuple[int, int]:
    """Same rule as Qwen2.5-VL's image processor: both sides become multiples of
    `factor` and the area is kept within [min_pixels, max_pixels]."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# =============================================================================
# Scoring
# =============================================================================
def score(item: TaskItem, raw: str, iou_threshold: float = 0.5) -> dict:
    """Return {pred, parse_rule, correct, [iou]}. Unparseable output -> pred None, wrong."""
    t = item.answer_type
    if t == "mc":
        pred, rule = extract_choice(raw, item.choices or [])
        return {"pred": pred, "parse_rule": rule, "correct": pred is not None and pred == item.gold}

    if t == "number":
        pred, rule = extract_number(raw)
        ok = pred is not None and round(pred, item.precision) == round(float(item.gold), item.precision)
        return {"pred": pred, "parse_rule": rule, "correct": bool(ok)}

    if t == "list":
        pred, rule = extract_number_list(raw)
        gold = [round(float(g), item.precision) for g in item.gold]
        ok = pred is not None and [round(p, item.precision) for p in pred] == gold
        return {"pred": pred, "parse_rule": rule, "correct": bool(ok)}

    if t == "text":
        span, rule = answer_span(raw)
        pred = normalize_text(span.split("\n")[-1] if rule != "explicit" else span)
        ok = bool(pred) and pred == normalize_text(str(item.gold))
        return {"pred": pred or None, "parse_rule": rule if pred else "none", "correct": ok}

    if t == "bbox":
        box, rule = extract_bbox(raw)
        if box is None:
            return {"pred": None, "parse_rule": "none", "correct": False, "iou": 0.0}
        # Qwen2.5-VL answers in absolute pixels of the *resized* image it was given.
        sx = item.extra["orig_w"] / item.extra["model_w"]
        sy = item.extra["orig_h"] / item.extra["model_h"]
        box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
        v = iou(box, item.gold)
        return {"pred": [round(b, 1) for b in box], "parse_rule": rule, "correct": v >= iou_threshold, "iou": v}

    raise ValueError(f"unknown answer_type {t!r}")


# =============================================================================
# Loaders. Signature: (dataset_cfg, family_cfg, model_cfg) -> list[TaskItem]
# =============================================================================
def _sample_rows(n_total: int, max_samples: int | None, seed: int, rows: list[int] | None = None) -> list[int]:
    rows = list(range(n_total)) if rows is None else rows
    if max_samples is not None and max_samples < len(rows):
        rows = sorted(random.Random(seed).sample(rows, max_samples))
    return rows


def _rgb(img):
    if img is not None and img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_scienceqa_items(dcfg: dict, fcfg: dict, mcfg: dict) -> list[TaskItem]:
    from common.scienceqa import load_scienceqa

    items = load_scienceqa(
        dcfg["hf_id"],
        fcfg.get("split", "test"),
        image_only=True,
        max_samples=fcfg.get("max_samples"),
        sample_seed=int(fcfg.get("sample_seed", 42)),
    )
    out = []
    for it in items:
        q = it.question if not it.hint.strip() else f"{it.question}\nContext: {it.hint.strip()}"
        out.append(
            TaskItem(
                id=it.id, family=fcfg["name"], dataset="scienceqa", row=it.row, question=q,
                answer_type="mc", gold=it.answer_letter, image=it.image, choices=it.choices,
                subgroup={"natural science": "NAT", "social science": "SOC", "language science": "LAN"}.get(it.subject, it.subject),
                extra={"grade": it.grade},
            )
        )
    return out


def _mathvista_fmt(answer_type: str, precision) -> tuple[str, int]:
    if answer_type == "integer":
        return "an integer", 0
    if answer_type == "float":
        p = int(precision or 1)
        return f"a floating-point number with {p} decimal place{'s' if p > 1 else ''}", p
    if answer_type == "list":
        return "a Python list, e.g. [1, 2, 3]", 2
    return "a single word or short phrase", 0


def load_mathvista_items(dcfg: dict, fcfg: dict, mcfg: dict) -> list[TaskItem]:
    from datasets import load_dataset

    ds = load_dataset(dcfg["hf_id"], split=fcfg.get("split", "testmini"))
    rows = _sample_rows(len(ds), fcfg.get("max_samples"), int(fcfg.get("sample_seed", 42)))
    out = []
    for i in rows:
        ex = ds[i]
        md = ex.get("metadata") or {}
        q = ex["question"]
        if ex.get("unit"):
            q = f"{q} (Unit: {ex['unit']})"
        base = dict(
            id=f"mathvista-{ex['pid']}", family=fcfg["name"], dataset="mathvista", row=i,
            image=_rgb(ex["decoded_image"]), subgroup=md.get("task", "unknown"),
            extra={"pid": ex["pid"], "skills": md.get("skills"), "language": md.get("language"), "context": md.get("context")},
        )
        if ex["question_type"] == "multi_choice":
            choices = list(ex["choices"])
            if ex["answer"] not in choices:
                continue  # malformed item; cannot be scored as multiple choice
            out.append(TaskItem(question=q, answer_type="mc", gold=LETTERS[choices.index(ex["answer"])], choices=choices, **base))
            continue
        fmt, prec = _mathvista_fmt(ex["answer_type"], ex.get("precision"))
        if ex["answer_type"] in ("integer", "float"):
            out.append(TaskItem(question=q, answer_type="number", gold=float(ex["answer"]), precision=prec, fmt=fmt, **base))
        elif ex["answer_type"] == "list":
            gold = [float(x) for x in str(ex["answer"]).strip("[]").split(",") if x.strip()]
            out.append(TaskItem(question=q, answer_type="list", gold=gold, precision=prec, fmt=fmt, **base))
        else:
            out.append(TaskItem(question=q, answer_type="text", gold=str(ex["answer"]), fmt=fmt, **base))
    return out


def load_countbench_items(dcfg: dict, fcfg: dict, mcfg: dict) -> list[TaskItem]:
    """CountBenchQA. The `text` column is the alt-text caption and usually spells
    out the count ("three light bulbs"), so it is never shown to the model."""
    from datasets import load_dataset

    ds = load_dataset(dcfg["hf_id"], split=fcfg.get("split", "test"))
    rows = _sample_rows(len(ds), fcfg.get("max_samples"), int(fcfg.get("sample_seed", 42)))
    out = []
    for i in rows:
        ex = ds[i]
        if ex["image"] is None:
            continue
        n = int(ex["number"])
        out.append(
            TaskItem(
                id=f"countbench-{i}", family=fcfg["name"], dataset="countbench", row=i,
                question=ex["question"], answer_type="number", gold=float(n), precision=0, fmt="an integer",
                image=_rgb(ex["image"]), subgroup="2-4" if n <= 4 else ("5-7" if n <= 7 else "8-10"),
            )
        )
    return out


def load_refcoco_items(dcfg: dict, fcfg: dict, mcfg: dict) -> list[TaskItem]:
    """RefCOCO (lmms-lab copy): one row per object with several referring
    expressions; `expression_index` picks which one becomes the query."""
    from datasets import load_dataset

    ds = load_dataset(dcfg["hf_id"], split=fcfg.get("split", "val"))
    ds = ds.remove_columns([c for c in ("segmentation",) if c in ds.column_names])
    rows = _sample_rows(len(ds), fcfg.get("max_samples"), int(fcfg.get("sample_seed", 42)))
    k = int(fcfg.get("expression_index", 0))
    min_px, max_px = mcfg.get("min_pixels"), mcfg.get("max_pixels")
    out = []
    for i in rows:
        ex = ds[i]
        exprs = list(ex["answer"]) if isinstance(ex["answer"], (list, tuple)) else [str(ex["answer"])]
        if not exprs:
            continue
        expr = exprs[min(k, len(exprs) - 1)]
        img = _rgb(ex["image"])
        w, h = img.size
        x, y, bw, bh = [float(v) for v in ex["bbox"]]
        mh, mw = smart_resize(h, w, 28, min_px or 56 * 56, max_px or 28 * 28 * 1280)
        area = (bw * bh) / (w * h)
        out.append(
            TaskItem(
                id=f"refcoco-{ex['question_id']}", family=fcfg["name"], dataset="refcoco", row=i,
                question=expr, answer_type="bbox", gold=[x, y, x + bw, y + bh], image=img,
                subgroup="small" if area < 0.05 else ("medium" if area < 0.2 else "large"),
                extra={"orig_w": w, "orig_h": h, "model_w": mw, "model_h": mh, "area_frac": round(area, 4), "file_name": ex.get("file_name")},
            )
        )
    return out


LOADERS: dict[str, Callable[[dict, dict, dict], list[TaskItem]]] = {
    "scienceqa": load_scienceqa_items,
    "mathvista": load_mathvista_items,
    "countbench": load_countbench_items,
    "refcoco": load_refcoco_items,
}


def load_family(name: str, fcfg: dict, cfg: dict) -> list[TaskItem]:
    ds_key = fcfg["dataset"]
    if ds_key not in LOADERS:
        raise ValueError(f"no loader for dataset {ds_key!r}; available: {sorted(LOADERS)}")
    fcfg = {**fcfg, "name": name}
    return LOADERS[ds_key](cfg["common"]["datasets"][ds_key], fcfg, cfg["common"]["model"])


def render_prompt(item: TaskItem, templates: dict, mode: str) -> str:
    """templates: E3_2.prompts; mode: direct | cot."""
    kind = {"mc": "mc", "bbox": "bbox"}.get(item.answer_type, "open")
    return templates[kind][mode].format(question=item.question, options=item.options_str(), format=item.fmt)
