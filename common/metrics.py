"""Accuracy, bootstrap confidence intervals and paired significance tests."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def accuracy(correct: list[bool]) -> float:
    return float(np.mean(correct)) if correct else float("nan")


def bootstrap_ci(correct: list[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if not correct:
        return float("nan"), float("nan")
    x = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilson_ci(correct: list[bool], alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Unlike the percentile bootstrap it stays informative at 0% / 100% and for
    small n (a bootstrap of 10/10 correct gives the degenerate interval [1, 1]).
    """
    from scipy.stats import norm

    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    p = sum(bool(c) for c in correct) / n
    z = float(norm.ppf(1 - alpha / 2))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap_diff_ci(
    a: list[bool], b: list[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """CI for acc(b) - acc(a) on the same items."""
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xa), size=(n_boot, len(xa)))
    diffs = xb[idx].mean(axis=1) - xa[idx].mean(axis=1)
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact (binomial) McNemar test on paired correctness.

    b01 = a wrong, b right (b gains); b10 = a right, b wrong (b losses).
    """
    from scipy.stats import binomtest

    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(b01, n, 0.5, alternative="two-sided").pvalue)
    return {"gains": b01, "losses": b10, "p_value": p}


def group_accuracy(records: list[dict], group_order: list[str]) -> dict[str, dict]:
    """records: dicts with 'correct' (bool) and 'groups' (list[str])."""
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        for g in r["groups"]:
            buckets[g].append(bool(r["correct"]))
    out = {}
    for g in group_order + sorted(set(buckets) - set(group_order)):
        if g in buckets:
            out[g] = {"n": len(buckets[g]), "acc": accuracy(buckets[g])}
    return out


def verdict(diff: float, p_value: float, alpha: float) -> str:
    if math.isnan(p_value) or p_value >= alpha:
        return "neutral"
    return "helps" if diff > 0 else "hurts"
