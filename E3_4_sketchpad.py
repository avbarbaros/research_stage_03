#!/usr/bin/env python
"""E3.4 — Visual Sketchpad: re-observing a modified image mid-reasoning.

The same VLM (Qwen2.5-VL-7B) on the same items, under:
  direct      image + question -> answer
  text        multi-turn-style reasoning prompt, no actions (the control: reasoning without new images)
  sketch      the same prompt plus actions that return new images: zoom, mark (numbered
              detector boxes), draw_line / draw_point, depth map; up to 4 actions per item
  <x>_only    single-action ablations (zoom_only, mark_only, draw_only, depth_only), run only
              on benchmarks listed in E3_4.run_ablations_on

Benchmarks: MathVista geometry (208), CV-Bench (the 1,000 E3.3 items), V*Bench (191).

Success criterion (roadmap): a measured gain from at least one visual action; a log of
which actions fire and how often they help.

Outputs in outputs/E3_4/<run_name>/
  preds_<bench>_<condition>.jsonl   per item: all turns (text, action, arguments, result note,
                                    saved result image), answer, correctness
  action_images/                    the images the model was shown after each action
  summary.md / summary.json, examples_<bench>_{wins,losses}.jsonl, per_item.csv

Usage
  python E3_4_sketchpad.py --set E3_4.run_name=check --set E3_4.benchmarks.cvbench.sample_seed=7 \
      --set E3_4.benchmarks.cvbench.max_per_task=8 --set E3_4.benchmarks.vstar.max_samples=16 \
      --set E3_4.benchmarks.mathvista_geo.max_samples=200 --set "E3_4.benchmarks.mathvista_geo.only_subgroups=[textbook question answering]"
  python E3_4_sketchpad.py --set E3_4.run_name=full
  python E3_4_sketchpad.py --set E3_4.run_name=full --set "E3_4.run_ablations_on=[vstar]"   # adds ablations
  python E3_4_sketchpad.py --analyze-only --set E3_4.run_name=full
"""
from __future__ import annotations

import csv
import gc
import json
import random
from pathlib import Path

from common.config import (
    REPO_ROOT,
    base_argparser,
    dump_json,
    environment_info,
    load_config,
    make_run_dir,
    save_resolved_config,
    set_seed,
)
from common.io_utils import append_jsonl, done_ids, read_jsonl

EXP = "E3_4"


# =============================================================================
# Generation
# =============================================================================
def direct_prompt(item, prompts: dict) -> str:
    if item.answer_type == "mc":
        return prompts["direct_mc"].format(question=item.question, options=item.options_str())
    return prompts["direct_open"].format(question=item.question, format=item.fmt or "a single word or phrase")


def stage_direct(bench, items, vlm, cfg, run_dir) -> None:
    from common.tasks import score
    from common.vlm import GenParams, Request

    e = cfg[EXP]
    path = run_dir / f"preds_{bench}_direct.jsonl"
    todo = [it for it in items if it.id not in done_ids(path)]
    print(f"\n=== {bench} / direct: {len(items) - len(todo)} done, {len(todo)} to run ===")
    gp = GenParams.from_cfg(e["generation"]["direct"], int(cfg["common"]["seed"]))
    for s in range(0, len(todo), 256):
        chunk = todo[s : s + 256]
        outs = vlm.generate([Request(direct_prompt(it, e["prompts"]), it.image) for it in chunk], gp, f"{bench}-direct")
        rows = []
        for it, o in zip(chunk, outs):
            r = it.meta()
            r.update(condition="direct", raw_output=o.text, out_tokens=o.n_tokens, finish_reason=o.finish_reason, **score(it, o.text))
            rows.append(r)
        append_jsonl(path, rows)


def stage_reasoning(bench, cond, items, vlm, tools_getter, cfg, run_dir) -> None:
    from common.sketchpad import ActionRunner, run_chunk
    from common.tasks import score
    from common.vlm import GenParams

    e = cfg[EXP]
    scfg = e["sketchpad"]
    allowed = [] if cond == "text" else list(e["action_sets"][cond])
    template = (REPO_ROOT / e["prompts"]["text_file" if cond == "text" else "sketch_file"]).read_text(encoding="utf-8")
    path = run_dir / f"preds_{bench}_{cond}.jsonl"
    todo = [it for it in items if it.id not in done_ids(path)]
    print(f"\n=== {bench} / {cond} {allowed or ''}: {len(items) - len(todo)} done, {len(todo)} to run ===")
    if not todo:
        return
    tools = tools_getter(allowed)
    runner = ActionRunner(scfg, tools)
    gp = GenParams.from_cfg(e["generation"]["turn"], int(cfg["common"]["seed"]))
    model_cfg = {**cfg["common"]["model"], **(e.get("model_overrides") or {})}
    img_dir = run_dir / "action_images" if scfg.get("save_action_images", True) else None
    size = int(scfg.get("chunk_size", 128))
    for s in range(0, len(todo), size):
        chunk = todo[s : s + size]
        states = run_chunk(chunk, vlm, runner, allowed, template, scfg, gp, model_cfg, img_dir, cond, f"{bench}-{cond}")
        rows = []
        for st in states:
            r = st.item.meta()
            fams = [t.get("family") for t in st.turns if t.get("action")]
            r.update(condition=cond, allowed=allowed, raw_output=st.final_text, turns=st.turns, n_actions=st.n_actions,
                     action_families=sorted({f for f in fams if f}), n_invalid=sum(1 for t in st.turns if t.get("action") and not t.get("ok")),
                     forced_final=st.forced_final, out_tokens=st.out_tokens,
                     finish_reason=st.turns[-1].get("finish") if st.turns else None, **score(st.item, st.final_text))
            rows.append(r)
        append_jsonl(path, rows)
        print(f"[{bench}/{cond}] {min(s + size, len(todo))}/{len(todo)}  chunk acc {sum(r['correct'] for r in rows) / len(rows):.3f}  "
              f"mean actions {sum(r['n_actions'] for r in rows) / len(rows):.2f}", flush=True)


class Tools:
    """Detector / depth model, loaded on first need."""

    def __init__(self, cfg):
        self.tcfg = cfg[EXP]["tools"]
        self.loaded = {"detector": None, "depth": None}

    def __call__(self, allowed: list[str]) -> dict:
        from common.visprog import DepthEstimator, Detector

        dev = self.tcfg.get("device", "cuda")
        if "mark" in allowed and self.loaded["detector"] is None:
            print(f"loading detector {self.tcfg['detector']['model']} on {dev}")
            self.loaded["detector"] = Detector(self.tcfg["detector"], dev)
        if "depth" in allowed and self.loaded["depth"] is None:
            print(f"loading depth model {self.tcfg['depth']['model']} on {dev}")
            self.loaded["depth"] = DepthEstimator(self.tcfg["depth"], dev)
        return dict(self.loaded)


def conditions_for(cfg, bench) -> list[str]:
    e = cfg[EXP]
    conds = list(e["conditions"])
    if bench in (e.get("run_ablations_on") or []):
        conds += [c for c in e.get("ablation_conditions") or [] if c not in conds]
    return conds


# =============================================================================
# Analysis (JSONL only; the grounding check also reads CV-Bench's gold boxes)
# =============================================================================
def _load(path: Path) -> dict:
    from common.tasks import item_from_row, score

    rows = {r["id"]: r for r in read_jsonl(path)}
    for r in rows.values():
        it = item_from_row(r)
        if it is not None:
            r.update(score(it, r["raw_output"]))
    return rows


def _cmp(a, b, n_boot, alpha):
    from common.metrics import accuracy, mcnemar_exact, paired_bootstrap_diff_ci, verdict

    d = accuracy(b) - accuracy(a)
    mc = mcnemar_exact(a, b)
    lo, hi = paired_bootstrap_diff_ci(a, b, n_boot, alpha)
    return {"n": len(a), "acc_a": accuracy(a), "acc_b": accuracy(b), "diff": d, "diff_ci": [lo, hi], **mc,
            "verdict": verdict(d, mc["p_value"], alpha)}


def analyze(cfg, run_dir: Path) -> dict:
    from common.metrics import accuracy, wilson_ci

    e = cfg[EXP]
    a = e.get("analysis", {})
    nb, al = int(a.get("bootstrap_samples", 2000)), float(a.get("alpha", 0.05))
    rng = random.Random(int(a.get("example_seed", 0)))
    k_ex = int(a.get("examples_per_benchmark", 25))
    all_conds = ["direct", "text"] + [c for c in e["action_sets"]]
    S = {"run_dir": str(run_dir), "model": cfg["common"]["model"]["name"], "benchmarks": {}}
    csv_rows = []
    for bench in e["benchmarks"]:
        P = {c: _load(run_dir / f"preds_{bench}_{c}.jsonl") for c in all_conds}
        P = {c: v for c, v in P.items() if v}
        if "direct" not in P or "text" not in P:
            continue
        main = [c for c in ("direct", "text", "sketch") if c in P]
        ids = sorted(set.intersection(*(set(P[c]) for c in main)))
        if not ids:
            continue
        B = {"n": len(ids), "conditions": {}, "vs_direct": {}, "vs_text": {}, "groups": {}, "actions": {}}
        corr = {c: {i: bool(P[c][i]["correct"]) for i in P[c]} for c in P}
        for c in P:
            cid = [i for i in ids if i in P[c]]
            v = [corr[c][i] for i in cid]
            B["conditions"][c] = {"n": len(cid), "acc": accuracy(v), "ci": list(wilson_ci(v, al)),
                                  "mean_out_tokens": sum(P[c][i].get("out_tokens", 0) for i in cid) / max(1, len(cid)),
                                  "no_answer_rate": accuracy([P[c][i].get("pred") is None for i in cid])}
            if c != "direct":
                B["vs_direct"][c] = _cmp([corr["direct"][i] for i in cid], v, nb, al)
            if c not in ("direct", "text"):
                B["vs_text"][c] = _cmp([corr["text"][i] for i in cid], v, nb, al)
        # subsets
        for key in (a.get("group_by") or {}).get(bench, []):
            B["groups"][key] = {}
            for val in sorted({str(P["direct"][i].get(key)) for i in ids}):
                gi = [i for i in ids if str(P["direct"][i].get(key)) == val]
                g = {"n": len(gi), **{c: accuracy([corr[c][i] for i in gi if i in corr[c]]) for c in P}}
                for c in P:
                    if c not in ("direct", "text"):
                        sub = [i for i in gi if i in corr[c]]
                        g[f"{c}_vs_text"] = _cmp([corr["text"][i] for i in sub], [corr[c][i] for i in sub], nb, al)
                        g[f"{c}_vs_direct"] = _cmp([corr["direct"][i] for i in sub], [corr[c][i] for i in sub], nb, al)
                B["groups"][key][val] = g
        # action log: which actions fire, and how often they help (vs text on the same items)
        for c in P:
            if c in ("direct", "text"):
                continue
            cid = [i for i in ids if i in P[c]]
            R = [P[c][i] for i in cid]
            fams = sorted({f for r in R for f in r.get("action_families", [])})
            log = {"n": len(R), "mean_actions": sum(r["n_actions"] for r in R) / len(R),
                   "any_action_rate": accuracy([r["n_actions"] > 0 for r in R]),
                   "invalid_action_rate": (sum(r.get("n_invalid", 0) for r in R) / max(1, sum(r["n_actions"] for r in R))),
                   "forced_final_rate": accuracy([bool(r.get("forced_final")) for r in R]),
                   "families": {}}
            for f in fams + ["(no action)"]:
                fi = [i for i in cid if (f in P[c][i].get("action_families", [])) or (f == "(no action)" and P[c][i]["n_actions"] == 0)]
                if not fi:
                    continue
                calls = sum(1 for i in fi for t in P[c][i]["turns"] if t.get("family") == f)
                log["families"][f] = {"items": len(fi), "fire_rate": len(fi) / len(cid), "calls": calls,
                                      **_cmp([corr["text"][i] for i in fi], [corr[c][i] for i in fi], nb, al)}
            B["actions"][c] = log
        # E3.3 comparison on CV-Bench
        if bench == "cvbench" and a.get("e3_3_run") and "sketch" in P:
            e33 = run_dir.parent.parent / "E3_3" / a["e3_3_run"]
            ag = {r["id"]: r for r in read_jsonl(e33 / "preds_cvbench_agent.jsonl")}
            dr = {r["id"]: r for r in read_jsonl(e33 / "preds_cvbench_direct.jsonl")}
            ci = [i for i in ids if i in ag and i in dr]
            if ci:
                fb = [bool(dr[i]["correct"]) if ag[i].get("error") else bool(ag[i]["correct"]) for i in ci]
                B["vs_e3_3_agent"] = _cmp(fb, [corr["sketch"][i] for i in ci], nb, al)
                B["vs_e3_3_agent"]["by_task"] = {}
                for t in sorted({P["direct"][i].get("task") for i in ci}):
                    ti = [j for j, i in enumerate(ci) if P["direct"][i].get("task") == t]
                    B["vs_e3_3_agent"]["by_task"][t] = _cmp([fb[j] for j in ti], [corr["sketch"][ci[j]] for j in ti], nb, al)
            if a.get("grounding_check", True):
                try:
                    B["grounding"] = grounding_check(cfg, P, corr, ids)
                except Exception as ex:  # noqa: BLE001 - optional analysis
                    print(f"[warn] grounding check skipped: {type(ex).__name__}: {ex}")
        # sensitivity: drop pilot items
        dev = set()
        for r in a.get("dev_runs") or []:
            for c in main:
                dev |= {x["id"] for x in read_jsonl(run_dir.parent / r / f"preds_{bench}_{c}.jsonl")}
        if dev and "sketch" in P:
            keep = [i for i in ids if i not in dev]
            B["excluding_dev_items"] = {"n_excluded": len(ids) - len(keep),
                                        "sketch_vs_text": _cmp([corr["text"][i] for i in keep], [corr["sketch"][i] for i in keep], nb, al),
                                        "sketch_vs_direct": _cmp([corr["direct"][i] for i in keep], [corr["sketch"][i] for i in keep], nb, al)}
        # examples: random sketch-vs-text wins and losses with all turns
        if "sketch" in P:
            for name, pool in (("wins", [i for i in ids if corr["sketch"][i] and not corr["text"][i]]),
                               ("losses", [i for i in ids if corr["text"][i] and not corr["sketch"][i]])):
                with open(run_dir / f"examples_{bench}_{name}.jsonl", "w", encoding="utf-8") as f:
                    for i in sorted(rng.sample(pool, min(k_ex, len(pool)))):
                        f.write(json.dumps({"id": i, "question": P["direct"][i]["question"], "gold": P["direct"][i]["gold"],
                                            "choices": P["direct"][i].get("choices"), "direct_output": P["direct"][i]["raw_output"],
                                            "text_output": P["text"][i]["raw_output"], "sketch_turns": P["sketch"][i]["turns"]},
                                           ensure_ascii=False) + "\n")
        for i in ids:
            csv_rows.append([bench, i, P["direct"][i].get("subgroup")] + [int(corr[c][i]) if c in corr and i in corr[c] else "" for c in all_conds]
                            + [P["sketch"][i]["n_actions"] if "sketch" in P else "", ";".join(P["sketch"][i].get("action_families", [])) if "sketch" in P else ""])
        S["benchmarks"][bench] = B
    dump_json(S, run_dir / "summary.json")
    (run_dir / "summary.md").write_text(_markdown(S), encoding="utf-8")
    with open(run_dir / "per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "id", "subgroup"] + [f"{c}_correct" for c in all_conds] + ["sketch_n_actions", "sketch_action_families"])
        w.writerows(csv_rows)
    return S


def grounding_check(cfg, P, corr, ids) -> dict:
    """CV-Bench 3D: did the sketch condition's zoom / mark actions cover the highlighted objects?"""
    import numpy as np
    from datasets import load_dataset

    from common.parsing import iou
    from E3_3_marked_eval import box_color, question_marks

    a = cfg[EXP]["analysis"]
    cover, thr = float(a.get("cover_frac", 0.5)), float(a.get("iou_threshold", 0.5))
    mcfg = cfg.get("E3_3", {}).get("marked_eval", {})
    ds = load_dataset(cfg["common"]["datasets"]["cvbench"]["hf_id"], split="test")
    objs, items = [], []
    for i in ids:
        r = P["sketch"][i]
        if r.get("task") not in ("Depth", "Distance"):
            continue
        ex = ds[r["row"]]
        arr = np.asarray(ex["image"].convert("RGB"))
        gold = {}
        for b in ex["bbox"] or []:
            c, _ = box_color(arr, b, int(mcfg.get("border_px", 3)), float(mcfg.get("color_min_frac", 0.25)))
            gold.setdefault(c, b)
        zooms = [t["box"] for t in r["turns"] if t.get("family") == "zoom" and t.get("box")]
        marks = [b for t in r["turns"] if t.get("family") == "mark" for b in t.get("boxes", [])]
        all_cov = True
        for c, name in question_marks(r["question"]).items():
            g = gold.get(c)
            if g is None:
                continue
            ga = max(1e-6, (g[2] - g[0]) * (g[3] - g[1]))

            def frac_in(z):
                ix = max(0.0, min(z[2], g[2]) - max(z[0], g[0]))
                iy = max(0.0, min(z[3], g[3]) - max(z[1], g[1]))
                return ix * iy / ga

            zc = any(frac_in(z) >= cover for z in zooms)
            mc = any(iou(b, g) >= thr for b in marks)
            objs.append({"task": r["task"], "zoom_covered": zc, "mark_found": mc, "any": zc or mc, "acted": bool(zooms or marks)})
            all_cov &= (zc or mc)
        if zooms or marks:
            items.append({"id": i, "task": r["task"], "all_covered": all_cov, "sketch": corr["sketch"][i], "text": corr["text"][i]})

    def acc(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    out = {"objects": len(objs), "objects_in_items_with_zoom_or_mark": sum(o["acted"] for o in objs),
           "covered_by_zoom": acc(o["zoom_covered"] for o in objs if o["acted"]),
           "found_by_mark": acc(o["mark_found"] for o in objs if o["acted"]),
           "covered_any": acc(o["any"] for o in objs if o["acted"]), "items": {}}
    for flag in (True, False):
        sub = [x for x in items if x["all_covered"] == flag]
        out["items"]["all highlighted objects covered" if flag else "at least one not covered"] = {
            "n": len(sub), "sketch": acc(x["sketch"] for x in sub), "text": acc(x["text"] for x in sub)}
    return out


def _pct(x):
    return "—" if x is None or x != x else f"{100 * x:.2f}"


def _c(v):
    return f"{100 * v['diff']:+.2f} ({100 * v['diff_ci'][0]:+.1f} to {100 * v['diff_ci'][1]:+.1f}) · {v['gains']}/{v['losses']} · p {v['p_value']:.2g} · **{v['verdict']}**"


def _markdown(S) -> str:
    L = ["# E3.4 — Visual Sketchpad", "", f"Model: `{S['model']}` · `text` = same reasoning prompt without actions (the control) · "
         "Δ cells: pp (95% CI) · gains/losses · McNemar p · verdict", ""]
    wins = []
    for b, B in S["benchmarks"].items():
        L += [f"## {b} (n = {B['n']})", "", "| condition | acc % (95% CI) | Δ vs direct | Δ vs text | no answer % | mean tokens |", "|---|---|---|---|---|---|"]
        for c, v in B["conditions"].items():
            vd = B["vs_direct"].get(c)
            vt = B["vs_text"].get(c)
            L.append(f"| {c} | {_pct(v['acc'])} ({_pct(v['ci'][0])}–{_pct(v['ci'][1])}) | {_c(vd) if vd else '—'} | {_c(vt) if vt else '—'} | "
                     f"{_pct(v['no_answer_rate'])} | {v['mean_out_tokens']:.0f} |")
            if vt and vt["verdict"] == "helps":
                wins.append(f"{b}: {c} vs text {100 * vt['diff']:+.2f} pp (p = {vt['p_value']:.2g})")
        for c, lg in B["actions"].items():
            L += ["", f"**Actions in `{c}`:** {_pct(lg['any_action_rate'])}% of items used at least one action · "
                  f"{lg['mean_actions']:.2f} actions per item · invalid actions {_pct(lg['invalid_action_rate'])}% · "
                  f"answer forced after the last action {_pct(lg['forced_final_rate'])}%", "",
                  "| action | items where it fired | fire rate % | calls | acc on those items: text → sketch | Δ vs text (gains/losses, p) |", "|---|---|---|---|---|---|"]
            for f, v in lg["families"].items():
                L.append(f"| {f} | {v['items']} | {_pct(v['fire_rate'])} | {v['calls']} | {_pct(v['acc_a'])} → {_pct(v['acc_b'])} | "
                         f"{100 * v['diff']:+.2f} ({v['gains']}/{v['losses']}, {v['p_value']:.2g}) |")
        for key, groups in B["groups"].items():
            L += ["", f"**{b} by {key}**", "", "| subset | n | " + " | ".join(f"{c} %" for c in B["conditions"]) + " | sketch vs text | sketch vs direct |",
                  "|---|---|" + "---|" * len(B["conditions"]) + "---|---|"]
            for val, g in groups.items():
                st, sd = g.get("sketch_vs_text"), g.get("sketch_vs_direct")
                L.append(f"| {val} | {g['n']} | " + " | ".join(_pct(g.get(c)) for c in B["conditions"]) + " | "
                         + (f"{100 * st['diff']:+.2f} ({st['gains']}/{st['losses']}, p {st['p_value']:.2g})" if st else "—") + " | "
                         + (f"{100 * sd['diff']:+.2f} ({sd['gains']}/{sd['losses']}, p {sd['p_value']:.2g})" if sd else "—") + " |")
        if "vs_e3_3_agent" in B:
            v = B["vs_e3_3_agent"]
            L += ["", f"**Sketch vs the E3.3 visual-program agent (with fallback), same items:** {_pct(v['acc_a'])} → {_pct(v['acc_b'])}, {_c(v)}", "",
                  "| task | E3.3 agent % | sketch % | Δ (gains/losses, p) |", "|---|---|---|---|"]
            for t, tv in v["by_task"].items():
                L.append(f"| {t} | {_pct(tv['acc_a'])} | {_pct(tv['acc_b'])} | {100 * tv['diff']:+.2f} ({tv['gains']}/{tv['losses']}, {tv['p_value']:.2g}) |")
        if "grounding" in B:
            g = B["grounding"]
            L += ["", f"**Grounding check (CV-Bench 3D, sketch):** of {g['objects_in_items_with_zoom_or_mark']} highlighted objects in items where the model "
                  f"zoomed or marked, {_pct(g['covered_by_zoom'])}% were covered by a zoom, {_pct(g['found_by_mark'])}% found by a mark, "
                  f"{_pct(g['covered_any'])}% by either.", "", "| items with zoom or mark | n | sketch % | text % |", "|---|---|---|---|"]
            for k, v in g["items"].items():
                L.append(f"| {k} | {v['n']} | {_pct(v['sketch'])} | {_pct(v['text'])} |")
        if "excluding_dev_items" in B:
            x = B["excluding_dev_items"]
            L += ["", f"**Excluding {x['n_excluded']} pilot items:** sketch vs text {_c(x['sketch_vs_text'])}; sketch vs direct {_c(x['sketch_vs_direct'])}"]
        L.append("")
    L += ["**Gains from visual actions over the text-only control (p < 0.05):** " + ("; ".join(wins) if wins else "none"), "",
          "Subset p-values are not corrected for multiple comparisons. Action images: `action_images/`; examples: `examples_<bench>_{wins,losses}.jsonl`.", ""]
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--analyze-only", action="store_true", help="recompute summary from existing files; no GPU needed")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    e = cfg[EXP]
    set_seed(int(cfg["common"]["seed"]))
    benches = list(e.get("run_benchmarks") or e["benchmarks"])
    for c in e["conditions"]:
        if c not in ("direct", "text") and c not in e["action_sets"]:
            raise SystemExit(f"condition {c!r} has no action set in E3_4.action_sets")
    if args.analyze_only and not e.get("run_name"):
        raise SystemExit("--analyze-only needs --set E3_4.run_name=<existing run>")
    run_dir = make_run_dir(cfg, EXP)
    print(f"run dir: {run_dir}")
    if args.analyze_only:
        analyze(cfg, run_dir)
        print((run_dir / "summary.md").read_text(encoding="utf-8"))
        return

    save_resolved_config(cfg, run_dir)
    from common.tasks import load_family
    from common.vlm import VLM

    model_cfg = {**cfg["common"]["model"], **(e.get("model_overrides") or {})}
    vlm = VLM(model_cfg, system_prompt=e["prompts"].get("system"))
    dump_json({**environment_info(), "backend": vlm.backend, "model": vlm.name, "model_cfg": model_cfg}, run_dir / "env.json")
    tools = Tools(cfg)
    for b in benches:
        items = load_family(b, e["benchmarks"][b], cfg)
        print(f"\n##### {b}: {len(items)} items")
        stage_direct(b, items, vlm, cfg, run_dir)
        for c in conditions_for(cfg, b):
            if c != "direct":
                stage_reasoning(b, c, items, vlm, tools, cfg, run_dir)
        del items
        gc.collect()
    analyze(cfg, run_dir)
    print()
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
