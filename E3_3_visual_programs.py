#!/usr/bin/env python
"""E3.3 — Visual programs: a ViperGPT-style agent vs end-to-end prompting.

The same VLM (Qwen2.5-VL-7B) is used two ways on the same items:
  direct      image + question -> short answer (end-to-end)
  agent       the VLM, as a planner, writes `execute_command(image)` against an
              ImagePatch API; the program runs with Grounding DINO (find/count),
              Depth Anything V2 (compute_depth) and the VLM on crops (simple_query).
  agent_fb    agent, but a failed program (exception / timeout / no answer)
              falls back to the direct answer (computed at analysis time)

Benchmarks (config: E3_3.benchmarks)
  gqa       GQA testdev-balanced; subsets by structural type, semantic type, spatial vs not
  cvbench   CV-Bench Count / Relation / Depth / Distance (the spatial subset)

Success criterion (roadmap): the agent beats end-to-end prompting on at least
one compositional subset, and every failure can be inspected.

Stages (each resumable; rerun with the same run_name)
  1. direct answers            preds_<bench>_direct.jsonl
  2. planner programs          programs_<bench>.jsonl
  3. program execution         preds_<bench>_agent.jsonl   (answer, error, full tool trace)

Usage
  python E3_3_visual_programs.py --set E3_3.run_name=check \
      --set E3_3.benchmarks.gqa.max_samples=30 --set E3_3.benchmarks.cvbench.max_per_task=10
  python E3_3_visual_programs.py --set E3_3.run_name=full
  python E3_3_visual_programs.py --analyze-only --set E3_3.run_name=full
"""
from __future__ import annotations

import csv
import gc
import json
import random
import time
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

EXP = "E3_3"


# =============================================================================
# Prompts
# =============================================================================
def direct_prompt(item, prompts: dict) -> str:
    if item.answer_type == "mc":
        return prompts["direct_mc"].format(question=item.question, options=item.options_str())
    return prompts["direct_open"].format(question=item.question)


def planner_prompt(item, template: str) -> str:
    if item.answer_type == "mc":
        fmt = "Options: " + ", ".join(item.choices or [])
    else:
        fmt = f"Answer format: {item.fmt or 'a single word or phrase'}"
    # str.replace, not format(): the template contains Python code with braces
    return template.replace("{question}", item.question).replace("{answer_format}", fmt)


# =============================================================================
# Stages
# =============================================================================
def stage_direct(bench: str, items, vlm, cfg: dict, run_dir: Path) -> None:
    from common.tasks import score
    from common.vlm import GenParams, Request

    ecfg = cfg[EXP]
    path = run_dir / f"preds_{bench}_direct.jsonl"
    todo = [it for it in items if it.id not in done_ids(path)]
    print(f"\n=== {bench} / direct: {len(items) - len(todo)} done, {len(todo)} to run ===")
    if not todo:
        return
    gp = GenParams.from_cfg(ecfg["generation"]["direct"], int(cfg["common"]["seed"]))
    for s in range(0, len(todo), 256):
        chunk = todo[s : s + 256]
        outs = vlm.generate([Request(direct_prompt(it, ecfg["prompts"]), it.image) for it in chunk], gp, f"{bench}-direct")
        rows = []
        for it, o in zip(chunk, outs):
            row = it.meta()
            row.update(condition="direct", raw_output=o.text, out_tokens=o.n_tokens, finish_reason=o.finish_reason,
                       **score(it, o.text))
            rows.append(row)
        append_jsonl(path, rows)


def stage_programs(bench: str, items, vlm, cfg: dict, run_dir: Path) -> None:
    from common.visprog import extract_code
    from common.vlm import GenParams, Request

    pcfg = cfg[EXP]["planner"]
    template = (REPO_ROOT / pcfg["prompt_file"]).read_text(encoding="utf-8")
    path = run_dir / f"programs_{bench}.jsonl"
    todo = [it for it in items if it.id not in done_ids(path)]
    print(f"\n=== {bench} / programs: {len(items) - len(todo)} done, {len(todo)} to write ===")
    if not todo:
        return
    gp = GenParams(max_new_tokens=int(pcfg.get("max_new_tokens", 512)), temperature=float(pcfg.get("temperature", 0.0)),
                   seed=int(cfg["common"]["seed"]))
    see = bool(pcfg.get("sees_image", False))
    for s in range(0, len(todo), 256):
        chunk = todo[s : s + 256]
        outs = vlm.generate([Request(planner_prompt(it, template), it.image if see else None) for it in chunk], gp,
                            f"{bench}-planner")
        append_jsonl(path, [
            {"id": it.id, "planner_output": o.text, "code": extract_code(o.text), "planner_tokens": o.n_tokens,
             "planner_finish": o.finish_reason}
            for it, o in zip(chunk, outs)
        ])


def stage_execute(bench: str, items, vlm, tools: dict, cfg: dict, run_dir: Path) -> None:
    from common.tasks import score
    from common.visprog import run_program, trace_summary

    ecfg = cfg[EXP]
    programs = {r["id"]: r for r in read_jsonl(run_dir / f"programs_{bench}.jsonl")}
    path = run_dir / f"preds_{bench}_agent.jsonl"
    xcfg = ecfg["executor"]
    # Infrastructure failures (e.g. CUDA OOM in a tool) say nothing about the program:
    # such items are run again on resume; the newest row per id is the one analysed.
    retry = set(xcfg.get("retry_error_types") or [])
    last = {r["id"]: r for r in read_jsonl(path)}
    todo = [it for it in items if it.id not in last or last[it.id].get("error_type") in retry]
    n_retry = sum(it.id in last for it in todo)
    print(f"\n=== {bench} / execute: {len(items) - len(todo)} done, {len(todo)} to run ({n_retry} retries) ===")
    buf, t0 = [], time.time()
    for k, it in enumerate(todo, 1):
        prog = programs.get(it.id, {})
        res = run_program(prog.get("code"), it.image, vlm, tools["detector"], tools["depth"], ecfg["tools"]["query"],
                          timeout_s=int(xcfg.get("timeout_s", 60)), max_calls=int(xcfg.get("max_tool_calls", 30)))
        row = it.meta()
        row.update(condition="agent", code=prog.get("code"), planner_finish=prog.get("planner_finish"),
                   raw_output=res["answer"] or "", error=res["error"], error_type=res["error_type"],
                   trace=res["trace"], n_calls=res["n_calls"], seconds=res["seconds"], **trace_summary(res["trace"]))
        if res["answer"] is None:
            row.update(pred=None, parse_rule="none", correct=False)
        else:
            row.update(**score(it, res["answer"]))
            if it.answer_type == "mc" and row["pred"] is None:
                # e.g. "towel" for options [mirror, television]: the program failed to answer the question asked
                row.update(error=f"answer {res['answer']!r} is not one of the options {it.choices}",
                           error_type="AnswerNotInOptions")
        buf.append(row)
        if len(buf) >= 20 or k == len(todo):
            append_jsonl(path, buf)
            buf = []
            el = time.time() - t0
            print(f"[{bench}/agent] {k}/{len(todo)}  {el / k:.1f}s/item  ETA {el / k * (len(todo) - k) / 60:.0f} min", flush=True)


def load_tools(cfg: dict) -> dict:
    from common.visprog import DepthEstimator, Detector

    tcfg = cfg[EXP]["tools"]
    dev = tcfg.get("device", "cuda")
    print(f"loading tools on {dev}: {tcfg['detector']['model']}, {tcfg['depth']['model']}")
    return {"detector": Detector(tcfg["detector"], dev), "depth": DepthEstimator(tcfg["depth"], dev)}


# =============================================================================
# Analysis (JSONL only; no GPU)
# =============================================================================
def _load(path: Path) -> dict:
    """Rows by id, re-scored with the current parser where possible."""
    from common.tasks import item_from_row, score

    rows = {r["id"]: r for r in read_jsonl(path)}
    for r in rows.values():
        if r.get("condition") == "agent" and r.get("error"):
            continue  # failed program: stays wrong
        item = item_from_row(r)
        if item is not None:
            r.update(score(item, r["raw_output"]))
    return rows


def _cmp(a: list[bool], b: list[bool], n_boot: int, alpha: float) -> dict:
    from common.metrics import accuracy, mcnemar_exact, paired_bootstrap_diff_ci, verdict

    d = accuracy(b) - accuracy(a)
    mc = mcnemar_exact(a, b)
    lo, hi = paired_bootstrap_diff_ci(a, b, n_boot, alpha)
    return {"diff": d, "diff_ci": [lo, hi], **mc, "verdict": verdict(d, mc["p_value"], alpha)}


def analyze(cfg: dict, run_dir: Path) -> dict:
    from common.metrics import accuracy, wilson_ci

    ecfg = cfg[EXP]
    acfg = ecfg.get("analysis", {})
    n_boot, alpha = int(acfg.get("bootstrap_samples", 2000)), float(acfg.get("alpha", 0.05))
    k_ex = int(acfg.get("examples_per_benchmark", 25))
    rng = random.Random(int(acfg.get("example_seed", 0)))
    summary = {"run_dir": str(run_dir), "model": cfg["common"]["model"]["name"], "benchmarks": {}}
    csv_rows = []

    for bench in ecfg["benchmarks"]:
        d = _load(run_dir / f"preds_{bench}_direct.jsonl")
        a = _load(run_dir / f"preds_{bench}_agent.jsonl")
        ids = sorted(set(d) & set(a))
        if not ids:
            continue
        cd = [bool(d[i]["correct"]) for i in ids]
        ca = [bool(a[i]["correct"]) for i in ids]
        for i in ids:  # rows written before the AnswerNotInOptions check existed
            if not a[i].get("error") and a[i].get("answer_type") == "mc" and a[i].get("pred") is None:
                a[i]["error"], a[i]["error_type"] = f"answer {a[i]['raw_output']!r} is not an option", "AnswerNotInOptions"
        failed = [bool(a[i].get("error")) for i in ids]
        cf = [d_ if f else a_ for d_, a_, f in zip(cd, ca, failed)]
        conds = {"direct": cd, "agent": ca, "agent_fb": cf}
        recs = [a[i] for i in ids]

        entry = {
            "n": len(ids),
            "acc": {c: {"acc": accuracy(v), "ci": list(wilson_ci(v, alpha))} for c, v in conds.items()},
            "agent_vs_direct": _cmp(cd, ca, n_boot, alpha),
            "agent_fb_vs_direct": _cmp(cd, cf, n_boot, alpha),
            "programs": {
                "error_rate": accuracy(failed),
                "error_types": _counts(r["error_type"] for r in recs if r.get("error")),
                "infrastructure_errors": sum(r.get("error_type") in set((ecfg.get("executor") or {}).get("retry_error_types") or [])
                                             for r in recs),
                "planner_truncated_rate": accuracy([r.get("planner_finish") == "length" for r in recs]),
                "degenerate_rate": accuracy([bool(r.get("degenerate")) for r in recs]),
                "used_detector_rate": accuracy([bool(r.get("used_detector")) for r in recs]),
                "used_depth_rate": accuracy([bool(r.get("used_depth")) for r in recs]),
                "empty_find_rate": accuracy([bool(r.get("empty_find")) for r in recs]),
                "mean_tool_calls": sum(r.get("n_calls", 0) for r in recs) / len(recs),
                "mean_seconds": sum(r.get("seconds", 0.0) for r in recs) / len(recs),
            },
            # accuracy of programs that ran, vs direct on the same items
            "on_executed": {},
            # does the agent do better when it actually decomposes the problem?
            "by_program_kind": {},
            "groups": {},
        }
        ok = [i for i, f in zip(ids, failed) if not f]
        if ok:
            entry["on_executed"] = {"n": len(ok), "direct": accuracy([bool(d[i]["correct"]) for i in ok]),
                                    "agent": accuracy([bool(a[i]["correct"]) for i in ok])}
        for kind, sel in (("detector-based", lambda r: r.get("used_detector") and not r.get("error")),
                          ("query-only (degenerate)", lambda r: r.get("degenerate") and not r.get("error"))):
            sub = [i for i in ids if sel(a[i])]
            if sub:
                entry["by_program_kind"][kind] = {
                    "n": len(sub), "direct": accuracy([bool(d[i]["correct"]) for i in sub]),
                    "agent": accuracy([bool(a[i]["correct"]) for i in sub]),
                }

        for key in (acfg.get("group_by") or {}).get(bench, []):
            vals = sorted({str(d[i].get(key)) for i in ids})
            entry["groups"][key] = {}
            for v in vals:
                pos = [j for j, i in enumerate(ids) if str(d[i].get(key)) == v]
                sd, sa, sf = [cd[j] for j in pos], [ca[j] for j in pos], [cf[j] for j in pos]
                entry["groups"][key][v] = {
                    "n": len(pos), "direct": accuracy(sd), "agent": accuracy(sa), "agent_fb": accuracy(sf),
                    "error_rate": accuracy([failed[j] for j in pos]),
                    "agent_vs_direct": _cmp(sd, sa, n_boot, alpha), "agent_fb_vs_direct": _cmp(sd, sf, n_boot, alpha),
                }
        summary["benchmarks"][bench] = entry

        # examples for reading: random (seeded) losses and wins, and every failed program
        def ex(i):
            return {"id": i, "question": d[i]["question"], "gold": d[i]["gold"], "choices": d[i].get("choices"),
                    "groups": {k: d[i].get(k) for k in (acfg.get("group_by") or {}).get(bench, [])},
                    "direct_output": d[i]["raw_output"], "agent_answer": a[i]["raw_output"], "error": a[i].get("error"),
                    "code": a[i].get("code"), "trace": a[i].get("trace")}

        losses = [i for i, x, y in zip(ids, cd, cf) if x and not y]
        wins = [i for i, x, y in zip(ids, cd, cf) if y and not x]
        for name, pool in (("agent_losses", losses), ("agent_wins", wins)):
            pick = rng.sample(pool, min(k_ex, len(pool)))
            with open(run_dir / f"{name}_{bench}.jsonl", "w", encoding="utf-8") as f:
                for i in sorted(pick):
                    f.write(json.dumps(ex(i), ensure_ascii=False) + "\n")
        with open(run_dir / f"program_errors_{bench}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                if a[i].get("error"):
                    f.write(json.dumps({**ex(i), "error_type": a[i].get("error_type")}, ensure_ascii=False) + "\n")
        for j, i in enumerate(ids):
            csv_rows.append([bench, i, json.dumps(d[i]["gold"]), d[i]["pred"], a[i]["pred"], int(cd[j]), int(ca[j]),
                             int(cf[j]), a[i].get("error_type") or "", int(bool(a[i].get("degenerate")))])

    dump_json(summary, run_dir / "summary.json")
    (run_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    with open(run_dir / "per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "id", "gold", "direct_pred", "agent_pred", "direct_correct", "agent_correct",
                    "agent_fb_correct", "agent_error", "degenerate_program"])
        w.writerows(csv_rows)
    return summary


def _counts(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _pct(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.2f}"


def _markdown(s: dict) -> str:
    L = ["# E3.3 — Visual programs vs end-to-end prompting", "",
         f"Model: `{s['model']}` (direct answers, program writing and crop queries) · tools: Grounding DINO, Depth Anything V2", "",
         "`agent` = program answer, failed programs count as wrong · `agent_fb` = failed programs fall back to the direct answer", "",
         "## Overall", "",
         "| benchmark | n | direct % | agent % | agent_fb % | Δ agent (pp, p) | Δ agent_fb (pp, 95% CI, p) | verdict agent_fb | program errors % |",
         "|---|---|---|---|---|---|---|---|---|"]
    wins = []
    for b, e in s["benchmarks"].items():
        acc, va, vf = e["acc"], e["agent_vs_direct"], e["agent_fb_vs_direct"]
        L.append(
            f"| {b} | {e['n']} | {_pct(acc['direct']['acc'])} | {_pct(acc['agent']['acc'])} | {_pct(acc['agent_fb']['acc'])} | "
            f"{100 * va['diff']:+.2f} ({va['p_value']:.2g}) | {100 * vf['diff']:+.2f} ({100 * vf['diff_ci'][0]:+.2f} to "
            f"{100 * vf['diff_ci'][1]:+.2f}, {vf['p_value']:.2g}) | **{vf['verdict']}** | {_pct(e['programs']['error_rate'])} |"
        )
        for key, groups in e["groups"].items():
            for v, g in groups.items():
                for c in ("agent", "agent_fb"):
                    if g[f"{c}_vs_direct"]["verdict"] == "helps":
                        wins.append(f"{b} / {key}={v} ({c}: {100 * g[f'{c}_vs_direct']['diff']:+.2f} pp, "
                                    f"p = {g[f'{c}_vs_direct']['p_value']:.2g}, n = {g['n']})")
    L += ["", "**Subsets where the agent significantly beats direct:** " + ("; ".join(wins) if wins else "none")]

    L += ["", "## Programs", "",
          "| benchmark | errors % | error types | planner cut off % | query-only programs % | used detector % | used depth % | a find() came back empty % | mean tool calls | mean s / item |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b, e in s["benchmarks"].items():
        p = e["programs"]
        et = ", ".join(f"{k} {v}" for k, v in p["error_types"].items()) or "—"
        L.append(f"| {b} | {_pct(p['error_rate'])} | {et} | {_pct(p['planner_truncated_rate'])} | {_pct(p['degenerate_rate'])} | "
                 f"{_pct(p['used_detector_rate'])} | {_pct(p['used_depth_rate'])} | {_pct(p['empty_find_rate'])} | "
                 f"{p['mean_tool_calls']:.1f} | {p['mean_seconds']:.1f} |")
    infra = {b: e["programs"]["infrastructure_errors"] for b, e in s["benchmarks"].items() if e["programs"].get("infrastructure_errors")}
    if infra:
        L += ["", "**Warning:** infrastructure errors remain (" + ", ".join(f"{b}: {n}" for b, n in infra.items())
              + "). They are not program failures; rerun the same command to retry them."]
    L += ["", "| benchmark | programs that ran: n · direct % · agent % | detector-based programs: n · direct % · agent % | query-only programs: n · direct % · agent % |",
          "|---|---|---|---|"]
    for b, e in s["benchmarks"].items():
        def f3(x):
            return f"{x['n']} · {_pct(x['direct'])} · {_pct(x['agent'])}" if x else "—"
        k = e["by_program_kind"]
        L.append(f"| {b} | {f3(e['on_executed'])} | {f3(k.get('detector-based'))} | {f3(k.get('query-only (degenerate)'))} |")

    L += ["", "## Subsets", ""]
    for b, e in s["benchmarks"].items():
        for key, groups in e["groups"].items():
            L += [f"### {b} by {key}", "",
                  "| subset | n | direct % | agent % | agent_fb % | Δ agent_fb pp | gains / losses | p | verdict | program errors % |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for v, g in groups.items():
                c = g["agent_fb_vs_direct"]
                L.append(f"| {v} | {g['n']} | {_pct(g['direct'])} | {_pct(g['agent'])} | {_pct(g['agent_fb'])} | "
                         f"{100 * c['diff']:+.2f} | {c['gains']} / {c['losses']} | {c['p_value']:.3g} | **{c['verdict']}** | "
                         f"{_pct(g['error_rate'])} |")
            L.append("")
    L += ["Verdict: exact McNemar test on paired correctness vs direct; *helps*/*hurts* when p < alpha. "
          "Subset p-values are not corrected for multiple comparisons. Files: `agent_losses_*.jsonl` / `agent_wins_*.jsonl` "
          "(random samples with program and tool trace), `program_errors_*.jsonl` (every failed program).", ""]
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--analyze-only", action="store_true", help="recompute summary from existing files; no GPU needed")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ecfg = cfg[EXP]
    set_seed(int(cfg["common"]["seed"]))
    benches = list(ecfg.get("run_benchmarks") or ecfg["benchmarks"])
    unknown = [b for b in benches if b not in ecfg["benchmarks"]]
    if unknown:
        raise SystemExit(f"unknown benchmark {unknown}; defined: {list(ecfg['benchmarks'])}")
    if args.analyze_only and not ecfg.get("run_name"):
        raise SystemExit("--analyze-only needs --set E3_3.run_name=<existing run>")
    run_dir = make_run_dir(cfg, EXP)
    print(f"run dir: {run_dir}")
    if args.analyze_only:
        analyze(cfg, run_dir)
        print((run_dir / "summary.md").read_text(encoding="utf-8"))
        return

    save_resolved_config(cfg, run_dir)
    from common.tasks import load_family
    from common.vlm import VLM

    model_cfg = {**cfg["common"]["model"], **(ecfg.get("model_overrides") or {})}
    vlm = VLM(model_cfg, system_prompt=ecfg["prompts"].get("system"))
    dump_json({**environment_info(), "backend": vlm.backend, "model": vlm.name, "model_cfg": model_cfg}, run_dir / "env.json")

    items_by_bench = {}
    for b in benches:
        items_by_bench[b] = load_family(b, ecfg["benchmarks"][b], cfg)
        print(f"{b}: {len(items_by_bench[b])} items")
        stage_direct(b, items_by_bench[b], vlm, cfg, run_dir)
        stage_programs(b, items_by_bench[b], vlm, cfg, run_dir)

    tools = load_tools(cfg)
    for b in benches:
        stage_execute(b, items_by_bench[b], vlm, tools, cfg, run_dir)
        del items_by_bench[b]
        gc.collect()

    analyze(cfg, run_dir)
    print()
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
