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
