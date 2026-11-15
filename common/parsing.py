"""Answer extraction for multiple-choice outputs.

Extraction rules matter (see E0.3). Every prediction records which rule fired,
so parse failures and fallbacks can be counted rather than silently scored.
Rules, tried in order:
  1. explicit    "The answer is (B)", "Answer: B", "**Answer:** (B)" — last match wins
  2. bare        the whole output is a single letter, optionally in () or followed by "." / ")"
  3. leading     output starts with "B)" / "(B)" / "B." followed by more text
  4. option_text output contains exactly one option's text (case-insensitive)
  5. none        nothing matched -> prediction is None and scored as wrong
"""
from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJ"

# The keyword part is case-insensitive; the letter must be uppercase unless it is
# parenthesised, so "the answer is a cat" is not read as option A.
_EXPLICIT = re.compile(
    r"(?i:answer\s*(?:is|:))\s*[:\-]?\s*\**\s*(?:\(\s*([A-Ja-j])\s*\)|([A-J]))(?![A-Za-z])"
)
_BARE = re.compile(r"^\s*\**\(?\s*([A-J])\s*\)?\**\s*[.\)]?\s*$")
_LEADING = re.compile(r"^\s*\**\(?([A-J])(?:\)|\.|:)\s+")


# LaTeX-style final answers, e.g. "\\boxed{B}" or "\\boxed{(B) 145°}" (common in maths CoT).
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def last_boxed(text: str) -> str | None:
    hits = _BOXED.findall(text or "")
    return hits[-1].strip() if hits else None


def extract_choice(text: str, choices: list[str]) -> tuple[str | None, str]:
    """Return (letter or None, rule name)."""
    n = len(choices)
    valid = set(LETTERS[:n])
    if text is None:
        return None, "none"
    t = text.strip()

    hits = [(m.group(1) or m.group(2)).upper() for m in _EXPLICIT.finditer(t)]
    hits = [h for h in hits if h in valid]
    if hits:
        return hits[-1], "explicit"

    boxed = last_boxed(t)
    if boxed is not None:
        m = re.match(r"^\(?\s*([A-J])\s*\)?(?![A-Za-z])", boxed)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper(), "boxed"
        low = boxed.lower()
        found = [i for i, c in enumerate(choices) if c and c.strip().lower() == low]
        if len(found) == 1:
            return LETTERS[found[0]], "boxed"

    m = _BARE.match(t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "bare"

    m = _LEADING.match(t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "leading"

    low = t.lower()
    found = [i for i, c in enumerate(choices) if c and c.strip().lower() in low]
    # If one option text contains another (e.g. "red" and "red and blue"),
    # keep only the longest matches.
    if len(found) > 1:
        longest = max(len(choices[i]) for i in found)
        found = [i for i in found if len(choices[i]) == longest]
    if len(found) == 1:
        return LETTERS[found[0]], "option_text"

    return None, "none"


# =============================================================================
# Open-ended answers (numbers, short text, lists) — used from E3.2 on
# =============================================================================
_ANSWER_IS = re.compile(r"(?i:answer\s*(?:is|:))\s*[:\-]?\s*(.+)")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|-?\.\d+")
_WORD_NUMS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_WORD_NUM_RE = re.compile(r"\b(" + "|".join(_WORD_NUMS) + r")\b", re.IGNORECASE)


def answer_span(text: str) -> tuple[str, str]:
    """Isolate the part of an output that holds the final answer.

    Returns (span, rule): rule "explicit" when an "answer is ..." phrase was
    found (the last one wins, cut at the end of its line), else "whole".
    """
    if text is None:
        return "", "none"
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed, "explicit"
    hits = list(_ANSWER_IS.finditer(text))
    if hits:
        span = hits[-1].group(1).split("\n")[0].strip()
        return span.strip(" *`\"'"), "explicit"
    return text.strip(), "whole"


def _to_float(tok: str) -> float | None:
    tok = tok.replace(",", "").replace(" ", "")
    try:
        if "/" in tok:
            a, b = tok.split("/")
            return float(a) / float(b)
        return float(tok)
    except (ValueError, ZeroDivisionError):
        return None


def parse_number(text: str, prefer: str = "first") -> float | None:
    """First (or last) number in text; digits first, then number words."""
    text = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text or "")
    toks = _NUM.findall(text)
    vals = [v for v in (_to_float(t) for t in toks) if v is not None]
    if vals:
        return vals[0] if prefer == "first" else vals[-1]
    words = _WORD_NUM_RE.findall(text or "")
    if words:
        w = words[0] if prefer == "first" else words[-1]
        return float(_WORD_NUMS[w.lower()])
    return None


def extract_number(text: str) -> tuple[float | None, str]:
    """Number answer: inside the "answer is" span if present, else the last number of the output
    (the final number of a reasoning chain; for a bare direct answer first == last)."""
    span, rule = answer_span(text)
    if rule == "explicit":
        v = parse_number(span, "first")
        if v is not None:
            return v, "explicit"
    v = parse_number(text or "", "last")
    return (v, "last_number") if v is not None else (None, "none")


def extract_number_list(text: str) -> tuple[list[float] | None, str]:
    span, rule = answer_span(text)
    m = re.findall(r"\[([^\[\]]*)\]", span) or re.findall(r"\[([^\[\]]*)\]", text or "")
    src = m[-1] if m else span
    vals = [v for v in (_to_float(t) for t in _NUM.findall(src)) if v is not None]
    return (vals, rule if m or rule == "explicit" else "whole") if vals else (None, "none")


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\"'`*]", "", s)
    s = re.sub(r"[.,;:!?]+$", "", s).strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s)


# =============================================================================
# Bounding boxes (Qwen2.5-VL grounding output)
# =============================================================================
_BBOX_KEY = re.compile(r'"?bbox(?:_2d)?"?\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')
_BBOX_ANY = re.compile(r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")
_BBOX_PAIRS = re.compile(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*(?:,|to|-)?\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")


def extract_bbox(text: str) -> tuple[list[float] | None, str]:
    """Last box in the output as [x1, y1, x2, y2] (model coordinates).

    Rules: "bbox_2d": [...] (Qwen JSON) > any 4-number list > (x1,y1),(x2,y2).
    The last match wins, so a reasoning chain that mentions several boxes is
    scored on its final one.
    """
    for rule, rx in (("json", _BBOX_KEY), ("list", _BBOX_ANY), ("pairs", _BBOX_PAIRS)):
        hits = rx.findall(text or "")
        if hits:
            try:
                x1, y1, x2, y2 = (float(v) for v in hits[-1])
            except ValueError:
                continue
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], rule
    return None, "none"


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reasoning_words(text: str) -> int:
    """Words of prose in an output once code blocks, JSON and boxes are removed.

    Used to check that a CoT prompt actually produced reasoning: a grounding
    model may answer a "think step by step" prompt with the bare JSON box.
    """
    t = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    t = re.sub(r"\[[^\[\]]*\]|\{[^{}]*\}", " ", t)
    return len(re.findall(r"[A-Za-z]{2,}", t))
