#!/usr/bin/env python
"""E3.3 follow-up — did the agent pick the highlighted object, and does it matter?

CV-Bench's 3D tasks (Depth, Distance) mark the objects a question refers to with
red / blue / green boxes drawn on the image, and store those boxes. This script
compares the agent's instance choices (read from the tool traces of an E3.3 run)
with the gold boxes. No VLM is loaded; only Depth Anything for the oracle part.

  1. instance selection   per highlighted object: searched? detected? gold among the
                          candidates? selected? (IoU >= iou_threshold with the gold box)
  2. outline check        find_marked's yes/no answers vs whether the candidate is the gold one
  3. selection vs answer  agent accuracy when every marked object was selected correctly vs not
  4. oracle instances     the agent's depth / distance rules applied to the GOLD boxes:
                          separates "wrong object" from "wrong depth reasoning"

The colour of each gold box is read from the image (pixels along the drawn outline),
because the stored box order does not follow the colour order of the question.

Usage
  python E3_3_marked_eval.py --set E3_3.run_name=full
Outputs in outputs/E3_3/<run_name>/marked_eval/
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from common.config import base_argparser, dump_json, load_config, make_run_dir
from common.io_utils import read_jsonl
from common.parsing import iou

EXP = "E3_3"
COLORS = ("red", "blue", "green")
_MARK_RE = re.compile(r"the ([^(),]+?) \((?:highlighted|annotated) by (?:a|the) (red|blue|green) box\)", re.IGNORECASE)
_CHECK_RE = re.compile(r"outlined by a (\w+) box", re.IGNORECASE)


# =============================================================================
# Gold: which object has which colour, and where its box is
# =============================================================================
def question_marks(question: str) -> dict[str, str]:
    """{color: object name} from 'the chair (highlighted by a red box)'."""
    return {c.lower(): n.strip().lower() for n, c in _MARK_RE.findall(question)}


def box_color(img_arr, box, band: int, min_frac: float) -> tuple[str, float]:
    """Colour of the outline drawn around `box`: majority colour of the pixels in a
    band of +-band px along its four edges."""
    import numpy as np

    H, W = img_arr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
    y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
    parts = []
    for y in (y1, y2):
        parts.append(img_arr[max(0, y - band): y + band + 1, x1: x2 + 1].reshape(-1, 3))
    for x in (x1, x2):
        parts.append(img_arr[y1: y2 + 1, max(0, x - band): x + band + 1].reshape(-1, 3))
    px = np.concatenate([p for p in parts if p.size]).astype(int)
    if not len(px):
        return "unknown", 0.0
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    counts = {
        "red": int(((r > 150) & (g < 90) & (b < 90)).sum()),
        "green": int(((g > 150) & (r < 110) & (b < 110)).sum()),
        "blue": int(((b > 150) & (r < 90) & (g < 130)).sum()),
    }
    c = max(counts, key=counts.get)
    frac = counts[c] / len(px)
    return (c if frac >= min_frac else "unknown"), frac


# =============================================================================
# Agent: what each find / find_marked call returned and chose (from the trace)
# =============================================================================
def trace_segments(trace: list[dict]) -> list[dict]:
    """Group a trace into detector calls: each 'find' plus the outline checks that follow it."""
    segs = []
    for op in trace or []:
        if op["op"] == "find":
            segs.append({"name": op["name"].strip().lower(), "boxes": op.get("boxes") or [], "checks": []})
        elif op["op"] == "marked_check" and segs:
            m = _CHECK_RE.search(op.get("q", ""))
            segs[-1]["checks"].append({"patch": op["patch"], "yes": op.get("a", "").strip().lower().startswith("yes"),
                                       "color": m.group(1).lower() if m else None})
    for s in segs:
        s["kind"] = "find_marked" if s["checks"] else "find"
        s["color"] = s["checks"][0]["color"] if s["checks"] else None
        yes = [c["patch"] for c in s["checks"] if c["yes"]]
        # find_marked returns the first confirmed candidate first, else the detector order;
        # programs then use element [0].
        s["chosen"] = yes[0] if yes else (s["boxes"][0] if s["boxes"] else None)
        s["confirmed"] = bool(yes)
    return segs


def _same_name(a: str, b: str) -> bool:
    a, b = a.rstrip("s"), b.rstrip("s")
    return a == b or a in b or b in a


def match_segment(segs: list[dict], color: str, name: str) -> dict | None:
    for s in segs:  # find_marked with the right colour
        if s["kind"] == "find_marked" and s["color"] == color:
            return s
    for s in segs:  # otherwise a call for the same object name
        if _same_name(s["name"], name):
            return s
    return None


# =============================================================================
# Oracle: the agent's own depth rules on the gold boxes
# =============================================================================
def median_depth(dmap, box) -> float:
    import numpy as np

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    d = dmap[max(0, y1): max(y1 + 1, y2), max(0, x1): max(x1 + 1, x2)]
    return float(np.median(d)) if d.size else 1.0


def oracle_answers(task: str, marks: dict, gold_by_color: dict, choices: list[str], dmap, W: int, H: int) -> dict:
    """Answer (option text) each rule gives when it is handed the gold boxes."""
    def opt(name):
        for c in choices:
            if _same_name(c.lower(), name):
                return c
        return None

    out = {}
    if task == "Depth":
        cols = [c for c in COLORS if c in marks and c in gold_by_color][:2]
        if len(cols) == 2:
            d = {c: median_depth(dmap, gold_by_color[c]) for c in cols}
            out["depth"] = opt(marks[min(cols, key=d.get)])
    elif task == "Distance":
        if all(c in marks and c in gold_by_color for c in COLORS):
            diag = math.hypot(W, H)
            ref = gold_by_color["red"]
            dref = median_depth(dmap, ref)

            def ctr(b):
                return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

            def d2(b):
                (ax, ay), (bx, by) = ctr(b), ctr(ref)
                return math.hypot(ax - bx, ay - by) / diag

            def dd(b):
                return abs(median_depth(dmap, b) - dref)

            rules = {
                "distance_3d": lambda b: math.sqrt(d2(b) ** 2 + dd(b) ** 2),  # what the agent used
                "depth_only": dd,
                "image_plane_only": d2,
            }
            for rule, f in rules.items():
                best = min(("blue", "green"), key=lambda c: f(gold_by_color[c]))
                out[rule] = opt(marks[best])
    return out


# =============================================================================
# Main
# =============================================================================
def _acc(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(x):
    return "—" if x is None or x != x else f"{100 * x:.1f}"


def main() -> None:
    ap = base_argparser(__doc__.splitlines()[0])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ecfg = cfg[EXP]
    mcfg = ecfg.get("marked_eval") or {}
    thr = float(mcfg.get("iou_threshold", 0.5))
    if not ecfg.get("run_name"):
        raise SystemExit("needs --set E3_3.run_name=<existing E3.3 run>")
    run_dir = make_run_dir(cfg, EXP)
    out_dir = run_dir / "marked_eval"
    out_dir.mkdir(exist_ok=True)

    agent = {r["id"]: r for r in read_jsonl(run_dir / "preds_cvbench_agent.jsonl")}
    direct = {r["id"]: r for r in read_jsonl(run_dir / "preds_cvbench_direct.jsonl")}
    rows3d = [r for i, r in agent.items() if r.get("task") in ("Depth", "Distance") and i in direct]
    if not rows3d:
        raise SystemExit(f"no CV-Bench Depth/Distance rows in {run_dir}")
    print(f"{len(rows3d)} Depth/Distance items from {run_dir.name}")

    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(cfg["common"]["datasets"]["cvbench"]["hf_id"], split=ecfg["benchmarks"]["cvbench"].get("split", "test"))
    depth = None
    if mcfg.get("oracle_depth", True):
        from common.visprog import DepthEstimator

        depth = DepthEstimator(ecfg["tools"]["depth"], mcfg.get("device", "cuda"))

    objects, items, checks = [], [], []
    order_patterns = Counter()
    color_frac = []
    for k, r in enumerate(sorted(rows3d, key=lambda x: x["row"]), 1):
        ex = ds[r["row"]]
        img = ex["image"].convert("RGB")
        arr = np.asarray(img)
        marks = question_marks(ex["question"])
        gold_by_color, order = {}, []
        for b in ex["bbox"] or []:
            c, f = box_color(arr, b, int(mcfg.get("border_px", 3)), float(mcfg.get("color_min_frac", 0.25)))
            color_frac.append(f)
            order.append(c)
            if c != "unknown" and c not in gold_by_color:
                gold_by_color[c] = b
        order_patterns[(r["task"], tuple(order))] += 1

        segs = trace_segments(r.get("trace"))
        item_ok = True
        for c, name in marks.items():
            g = gold_by_color.get(c)
            s = match_segment(segs, c, name)
            o = {"id": r["id"], "task": r["task"], "color": c, "name": name, "gold_known": g is not None,
                 "searched": s is not None, "call": s["kind"] if s else None,
                 "detected": bool(s and s["boxes"]), "n_candidates": len(s["boxes"]) if s else 0,
                 "gold_in_candidates": bool(g and s and any(iou(b, g) >= thr for b in s["boxes"])),
                 "confirmed": bool(s and s.get("confirmed")),
                 "selected_iou": round(iou(s["chosen"], g), 3) if (g and s and s["chosen"]) else 0.0}
            o["selected_correct"] = o["selected_iou"] >= thr
            if g is not None:
                item_ok &= o["selected_correct"]
            objects.append(o)
            if s and g:
                for ch in s["checks"]:
                    checks.append({"task": r["task"], "is_gold": iou(ch["patch"], g) >= thr, "yes": ch["yes"]})
        it = {"id": r["id"], "task": r["task"], "gold": r["gold"], "choices": r.get("choices"),
              "agent_correct": bool(r["correct"]) and not r.get("error"), "agent_error": r.get("error_type"),
              "direct_correct": bool(direct[r["id"]]["correct"]), "all_selected_correct": item_ok,
              "all_gold_known": all(c in gold_by_color for c in marks)}
        if depth is not None and it["all_gold_known"]:
            dmap = depth(img)
            ans = oracle_answers(r["task"], marks, gold_by_color, it["choices"] or [], dmap, *img.size)
            gold_text = (it["choices"] or [])[ord(it["gold"]) - 65] if it["choices"] else None
            for rule, a in ans.items():
                it[f"oracle_{rule}"] = a is not None and gold_text is not None and a.lower() == gold_text.lower()
        items.append(it)
        if k % 50 == 0:
            print(f"  {k}/{len(rows3d)}", flush=True)

    summary = _summarize(objects, items, checks, order_patterns, color_frac, thr, run_dir.name)
    dump_json(summary, out_dir / "summary.json")
    (out_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    for name, rows in (("objects.csv", objects), ("items.csv", items)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            keys = sorted({k for x in rows for k in x})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nResults: {out_dir}")


def _summarize(objects, items, checks, order_patterns, color_frac, thr, run_name) -> dict:
    from common.metrics import mcnemar_exact

    s = {"run": run_name, "iou_threshold": thr, "n_items": len(items), "n_objects": len(objects),
         "gold_color_found_rate": _acc(o["gold_known"] for o in objects),
         "box_order_patterns": {f"{t}: {' / '.join(o)}": n for (t, o), n in order_patterns.most_common(8)},
         "selection": {}, "outline_check": {}, "selection_vs_answer": {}, "oracle": {}}
    for task in ("Depth", "Distance"):
        for role in ("all",) + COLORS:
            obs = [o for o in objects if o["task"] == task and o["gold_known"] and (role == "all" or o["color"] == role)]
            if not obs:
                continue
            s["selection"][f"{task}/{role}"] = {
                "n": len(obs),
                "searched": _acc(o["searched"] for o in obs),
                "via_find_marked": _acc(o["call"] == "find_marked" for o in obs),
                "detected": _acc(o["detected"] for o in obs),
                "gold_in_candidates": _acc(o["gold_in_candidates"] for o in obs),
                "selected_correct": _acc(o["selected_correct"] for o in obs),
                # given the gold box was among the candidates, how often was it the one used?
                "selected_given_available": _acc(o["selected_correct"] for o in obs if o["gold_in_candidates"]),
            }
        ch = [c for c in checks if c["task"] == task]
        pos, neg = [c for c in ch if c["is_gold"]], [c for c in ch if not c["is_gold"]]
        s["outline_check"][task] = {"n_checks": len(ch), "n_gold": len(pos), "n_other": len(neg),
                                    "yes_on_gold": _acc(c["yes"] for c in pos), "yes_on_other": _acc(c["yes"] for c in neg)}
        its = [i for i in items if i["task"] == task and i["all_gold_known"]]
        for flag in (True, False):
            sub = [i for i in its if i["all_selected_correct"] == flag]
            s["selection_vs_answer"][f"{task}/{'all selected correctly' if flag else 'at least one wrong'}"] = {
                "n": len(sub), "agent": _acc(i["agent_correct"] for i in sub), "direct": _acc(i["direct_correct"] for i in sub)}
        rules = sorted({k[7:] for i in its for k in i if k.startswith("oracle_")})
        for rule in rules:
            sub = [i for i in its if f"oracle_{rule}" in i]
            a = [i["agent_correct"] for i in sub]
            o = [i[f"oracle_{rule}"] for i in sub]
            dc = [i["direct_correct"] for i in sub]
            vd = mcnemar_exact(dc, o)
            s["oracle"][f"{task}/{rule}"] = {"n": len(sub), "oracle_instances": _acc(o), "agent": _acc(a),
                                             "direct": _acc(dc), **mcnemar_exact(a, o),
                                             "vs_direct": {"gains": vd["gains"], "losses": vd["losses"], "p_value": vd["p_value"]}}
    return s


def _markdown(s: dict) -> str:
    L = [f"# E3.3 follow-up — instance selection on CV-Bench 3D (run `{s['run']}`)", "",
         f"{s['n_items']} Depth/Distance items, {s['n_objects']} highlighted objects; correct = IoU >= {s['iou_threshold']} with the gold box.",
         f"Gold box colour read from the image for {_pct(s['gold_color_found_rate'])}% of objects.", "",
         "Stored box order, by colour read from the image (most common patterns):", ""]
    L += [f"- {k}: {v}" for k, v in s["box_order_patterns"].items()]
    L += ["", "## 1. Did the agent use the highlighted instance?", "",
          "| task / colour | n | searched % | via find_marked % | detected % | gold among candidates % | selected correctly % | selected, when gold was a candidate % |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in s["selection"].items():
        L.append(f"| {k} | {v['n']} | {_pct(v['searched'])} | {_pct(v['via_find_marked'])} | {_pct(v['detected'])} | "
                 f"{_pct(v['gold_in_candidates'])} | {_pct(v['selected_correct'])} | {_pct(v['selected_given_available'])} |")
    L += ["", "## 2. Outline check (\"Is this X outlined by a <colour> box?\")", "",
          "| task | checks | on the gold instance: n · said yes % | on another instance: n · said yes % |", "|---|---|---|---|"]
    for t, v in s["outline_check"].items():
        L.append(f"| {t} | {v['n_checks']} | {v['n_gold']} · {_pct(v['yes_on_gold'])} | {v['n_other']} · {_pct(v['yes_on_other'])} |")
    L += ["", "## 3. Does selecting the right instances matter?", "",
          "| task / items | n | agent % | direct % |", "|---|---|---|---|"]
    for k, v in s["selection_vs_answer"].items():
        L.append(f"| {k} | {v['n']} | {_pct(v['agent'])} | {_pct(v['direct'])} |")
    L += ["", "## 4. Oracle instances: the agent's rules applied to the gold boxes", "",
          "| task / rule | n | oracle instances % | agent % | direct % | oracle vs agent: gains / losses, p | oracle vs direct: gains / losses, p |",
          "|---|---|---|---|---|---|---|"]
    for k, v in s["oracle"].items():
        L.append(f"| {k} | {v['n']} | {_pct(v['oracle_instances'])} | {_pct(v['agent'])} | {_pct(v['direct'])} | "
                 f"{v['gains']} / {v['losses']}, {v['p_value']:.2g} | "
                 f"{v['vs_direct']['gains']} / {v['vs_direct']['losses']}, {v['vs_direct']['p_value']:.2g} |")
    L += ["", "Rules: `depth` = smaller median depth is closer to the camera (Depth task); for Distance, the option whose "
          "box is nearest the red reference by `distance_3d` (what the agent used: image-plane offset as a fraction of the "
          "diagonal combined with depth difference), `depth_only` (depth difference) or `image_plane_only` (2D offset).",
          "Agent errors count as wrong here (no fallback).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
