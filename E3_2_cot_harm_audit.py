#!/usr/bin/env python
"""E3.2 — CoT harm audit: direct vs chain-of-thought prompting across task families.

Same model, same items, two prompts per item: answer directly, or think step by
step first. Four families, one dataset each (config: E3_2.families):

  math       MathVista testmini   multiple choice + numeric / text answers
  science    ScienceQA (img)      multiple choice
  counting   CountBenchQA         integer counts 2-10
  grounding  RefCOCO val          referring expression -> box, correct if IoU >= 0.5

Success criterion (roadmap): a table showing at least one family where CoT
reduces accuracy, with the drop quantified.

Outputs in outputs/E3_2/<run_name>/
  preds_<family>_<condition>.jsonl   one row per item
  summary.json / summary.md          per-family direct vs CoT, paired tests, subgroup breakdown
  cot_losses_<family>.jsonl          items direct got right and CoT got wrong (for reading)
  per_item.csv, config.yaml, env.json, timings.json

Usage
  python E3_2_cot_harm_audit.py --set E3_2.run_name=check --set E3_2.families.math.max_samples=50 ...
  python E3_2_cot_harm_audit.py --set E3_2.run_name=full
  python E3_2_cot_harm_audit.py --set E3_2.run_name=full --set "E3_2.run_families=[counting]"
  python E3_2_cot_harm_audit.py --analyze-only --set E3_2.run_name=full
"""
from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

from common.config import (
    base_argparser,
    dump_json,
    environment_info,
    load_config,
    make_run_dir,
    save_resolved_config,
    set_seed,
)
from common.io_utils import append_jsonl, done_ids, read_jsonl

EXP = "E3_2"
CONDITIONS = ["direct", "cot"]


# =============================================================================
# Generation
# =============================================================================
def run_family_condition(family: str, condition: str, items, vlm, cfg: dict, run_dir: Path) -> float:
    from common.parsing import reasoning_words
    from common.tasks import render_prompt, score
    from common.vlm import GenParams, Request

    ecfg = cfg[EXP]
    gp = GenParams.from_cfg(ecfg["generation"][condition], int(cfg["common"]["seed"]))
    thr = float(ecfg.get("iou_threshold", 0.5))
    path = run_dir / f"preds_{family}_{condition}.jsonl"
    finished = done_ids(path)
    todo = [it for it in items if it.id not in finished]
    print(f"\n=== {family} / {condition}: {len(finished)} done, {len(todo)} to run ===")
    if not todo:
        return 0.0

    t0 = time.time()
    size = int(ecfg.get("chunk_size", 256))
    for s in range(0, len(todo), size):
        chunk = todo[s : s + size]
        reqs = [Request(render_prompt(it, ecfg["prompts"], condition), it.image) for it in chunk]
        outs = vlm.generate(reqs, gp, f"{family}-{condition}")
        rows = []
        for it, o in zip(chunk, outs):
            row = it.meta()
            row.update(
                condition=condition,
                raw_output=o.text,
                out_tokens=o.n_tokens,
                finish_reason=o.finish_reason,
                reasoning_words=reasoning_words(o.text),
                **score(it, o.text, thr),
            )
            rows.append(row)
        append_jsonl(path, rows)
        print(f"[{family}/{condition}] wrote {len(rows)} rows (chunk acc {sum(r['correct'] for r in rows) / len(rows):.3f})")
    return time.time() - t0


# =============================================================================
# Analysis (JSONL only; no GPU)
# =============================================================================
def analyze(cfg: dict, run_dir: Path) -> dict:
    from common.metrics import accuracy, mcnemar_exact, paired_bootstrap_diff_ci, verdict, wilson_ci

    ecfg = cfg[EXP]
    acfg = ecfg.get("analysis", {})
    n_boot = int(acfg.get("bootstrap_samples", 2000))
    alpha = float(acfg.get("alpha", 0.05))
    k_examples = int(acfg.get("examples_per_family", 25))
    base = ecfg.get("baseline_condition", "direct")
    other = [c for c in ecfg["conditions"] if c != base]
    if len(other) != 1:
        raise SystemExit("E3_2 analysis expects exactly two conditions (baseline + one other)")
    alt = other[0]

    from common.parsing import looks_like_refusal

    rescore = bool(acfg.get("rescore", True))
    thr = float(ecfg.get("iou_threshold", 0.5))
    min_words = int(acfg.get("min_reasoning_words", 15))
    summary: dict = {"run_dir": str(run_dir), "model": cfg["common"]["model"]["name"], "baseline": base, "alt": alt,
                     "min_reasoning_words": min_words, "families": {}}
    csv_rows = []
    failures = open(run_dir / "parse_failures.jsonl", "w", encoding="utf-8")
    for fam in ecfg["families"]:
        p_base, rs_b = _load(run_dir / f"preds_{fam}_{base}.jsonl", rescore, thr)
        p_alt, rs_a = _load(run_dir / f"preds_{fam}_{alt}.jsonl", rescore, thr)
        ids = sorted(set(p_base) & set(p_alt))
        if not ids:
            continue
        for cond, preds in ((base, p_base), (alt, p_alt)):
            for i in ids:
                r = preds[i]
                if r["pred"] is None:
                    failures.write(json.dumps({"family": fam, "condition": cond, "id": i, "answer_type": r["answer_type"],
                                               "refusal": looks_like_refusal(r["raw_output"]),
                                               "gold": r["gold"], "choices": r.get("choices"),
                                               "finish_reason": r.get("finish_reason"), "raw_output": r["raw_output"]},
                                              ensure_ascii=False) + "\n")
        a = [bool(p_base[i]["correct"]) for i in ids]
        b = [bool(p_alt[i]["correct"]) for i in ids]
        mc = mcnemar_exact(a, b)
        diff = accuracy(b) - accuracy(a)
        dlo, dhi = paired_bootstrap_diff_ci(a, b, n_boot, alpha)
        losses = [i for i, x, y in zip(ids, a, b) if x and not y]
        entry = {
            "dataset": p_base[ids[0]]["dataset"],
            "n": len(ids),
            "rescored_changed": {base: rs_b, alt: rs_a},
            base: _cond_stats([p_base[i] for i in ids], alpha, min_words),
            alt: _cond_stats([p_alt[i] for i in ids], alpha, min_words),
            "diff": diff,
            "diff_ci": [dlo, dhi],
            **mc,
            "verdict": verdict(diff, mc["p_value"], alpha),
            # A CoT "loss" caused by an unparseable CoT answer is a format failure, not a reasoning failure.
            "losses_alt_parse_fail": sum(p_alt[i]["pred"] is None for i in losses),
            # ... of which the model explicitly declined ("no object matches", "none of the options")
            "losses_alt_refusal": sum(p_alt[i]["pred"] is None and looks_like_refusal(p_alt[i]["raw_output"]) for i in losses),
            "losses_alt_truncated": sum(p_alt[i].get("finish_reason") == "length" for i in losses),
            "subgroups": {},
        }
        for sg in sorted({p_base[i]["subgroup"] for i in ids}):
            sid = [i for i in ids if p_base[i]["subgroup"] == sg]
            sa = [bool(p_base[i]["correct"]) for i in sid]
            sb = [bool(p_alt[i]["correct"]) for i in sid]
            smc = mcnemar_exact(sa, sb)
            entry["subgroups"][sg] = {
                "n": len(sid),
                base: accuracy(sa),
                alt: accuracy(sb),
                "diff": accuracy(sb) - accuracy(sa),
                **smc,
            }
        summary["families"][fam] = entry

        with open(run_dir / f"cot_losses_{fam}.jsonl", "w", encoding="utf-8") as f:
            for i in losses[:k_examples]:
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "subgroup": p_base[i]["subgroup"],
                            "question": p_base[i]["question"],
                            "gold": p_base[i]["gold"],
                            f"{base}_output": p_base[i]["raw_output"],
                            f"{alt}_output": p_alt[i]["raw_output"],
                            f"{alt}_pred": p_alt[i]["pred"],
                            f"{alt}_iou": p_alt[i].get("iou"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for i in ids:
            csv_rows.append(
                [fam, i, p_base[i]["row"], p_base[i]["subgroup"], json.dumps(p_base[i]["gold"]),
                 json.dumps(p_base[i]["pred"]), json.dumps(p_alt[i]["pred"]),
                 int(bool(p_base[i]["correct"])), int(bool(p_alt[i]["correct"])), p_alt[i]["out_tokens"]]
            )

    failures.close()
    dump_json(summary, run_dir / "summary.json")
    (run_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    with open(run_dir / "per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "id", "row", "subgroup", "gold", f"{base}_pred", f"{alt}_pred", f"{base}_correct", f"{alt}_correct", f"{alt}_out_tokens"])
        w.writerows(csv_rows)
    return summary


def _load(path: Path, rescore: bool, thr: float) -> tuple[dict, int]:
    """Read predictions; if possible re-score raw outputs with the current parser.

    Returns ({id: row}, number of rows whose correctness changed by re-scoring)."""
    from common.parsing import reasoning_words
    from common.tasks import item_from_row, score

    rows = {r["id"]: r for r in read_jsonl(path)}
    changed = 0
    for r in rows.values():
        if "reasoning_words" not in r:
            r["reasoning_words"] = reasoning_words(r["raw_output"])
        if not rescore:
            continue
        item = item_from_row(r)
        if item is None:
            continue
        new = score(item, r["raw_output"], thr)
        if bool(new["correct"]) != bool(r["correct"]):
            changed += 1
        r.update(new)
    return rows, changed


def _cond_stats(recs: list[dict], alpha: float, min_words: int = 15) -> dict:
    from common.metrics import accuracy, wilson_ci
    from common.parsing import looks_like_refusal

    corr = [bool(r["correct"]) for r in recs]
    st = {
        "acc": accuracy(corr),
        "ci": list(wilson_ci(corr, alpha)),
        "parse_fail_rate": accuracy([r["pred"] is None for r in recs]),
        "refusal_rate": accuracy([r["pred"] is None and looks_like_refusal(r["raw_output"]) for r in recs]),
        "parse_rules": _counts(r["parse_rule"] for r in recs),
        "truncated_rate": accuracy([r.get("finish_reason") == "length" for r in recs]),
        "mean_out_tokens": sum(r["out_tokens"] for r in recs) / len(recs),
        # share of outputs that contain prose reasoning (>= min_words words outside code / JSON)
        "reasoned_rate": accuracy([r["reasoning_words"] >= min_words for r in recs]),
    }
    if "iou" in recs[0]:
        st["mean_iou"] = sum(r.get("iou") or 0.0 for r in recs) / len(recs)
    return st


def _counts(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _pct(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.2f}"


def _markdown(s: dict) -> str:
    base, alt = s["baseline"], s["alt"]
    L = [
        "# E3.2 — CoT harm audit",
        "",
        f"Model: `{s['model']}` · each family: same items under `{base}` and `{alt}` prompting.",
        "",
        "## Direct vs CoT by task family",
        "",
        f"| family | dataset | n | {base} % (95% CI) | {alt} % (95% CI) | Δ pp | Δ 95% CI | gains / losses | McNemar p | verdict | {alt} losses with no answer (refusals) / truncated | no answer % ({base} / {alt}) | refusals % ({base} / {alt}) | {alt} truncated % | mean {alt} tok | outputs with reasoning % ({base} / {alt}) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam, e in s["families"].items():
        b, c = e[base], e[alt]
        L.append(
            f"| {fam} | {e['dataset']} | {e['n']} | {_pct(b['acc'])} ({_pct(b['ci'][0])}–{_pct(b['ci'][1])}) | "
            f"{_pct(c['acc'])} ({_pct(c['ci'][0])}–{_pct(c['ci'][1])}) | {100 * e['diff']:+.2f} | "
            f"{100 * e['diff_ci'][0]:+.2f} to {100 * e['diff_ci'][1]:+.2f} | {e['gains']} / {e['losses']} | {e['p_value']:.3g} | "
            f"**{e['verdict']}** | {e['losses_alt_parse_fail']} ({e.get('losses_alt_refusal', 0)}) / {e['losses_alt_truncated']} | "
            f"{_pct(b['parse_fail_rate'])} / {_pct(c['parse_fail_rate'])} | {_pct(b['refusal_rate'])} / {_pct(c['refusal_rate'])} | {_pct(c['truncated_rate'])} | {c['mean_out_tokens']:.0f} | "
            f"{_pct(b['reasoned_rate'])} / {_pct(c['reasoned_rate'])} |"
        )
    weak = [f for f, e in s["families"].items() if e[alt]["reasoned_rate"] < 0.9]
    if weak:
        L += ["", f"**Warning:** fewer than 90% of `{alt}` outputs contain reasoning (≥ {s['min_reasoning_words']} words of prose) in: "
              + ", ".join(f"{f} ({_pct(s['families'][f][alt]['reasoned_rate'])}%)" for f in weak)
              + ". For these families the comparison does not test reasoning."]
    hurt = [(f, e) for f, e in s["families"].items() if e["verdict"] == "hurts"]
    L += ["", "**Families where CoT significantly reduces accuracy:** " + (
        ", ".join(f"{f} ({100 * e['diff']:+.2f} pp, p = {e['p_value']:.2g})" for f, e in hurt) if hurt else "none"
    )]

    for fam, e in s["families"].items():
        if "mean_iou" in e[base]:
            L += ["", f"**{fam}: mean IoU** {base} {e[base]['mean_iou']:.3f} · {alt} {e[alt]['mean_iou']:.3f}"]

    L += ["", "## Subgroups", ""]
    for fam, e in s["families"].items():
        L += [f"### {fam} ({e['dataset']})", "", f"| subgroup | n | {base} % | {alt} % | Δ pp | gains / losses | McNemar p |", "|---|---|---|---|---|---|---|"]
        for sg, g in e["subgroups"].items():
            L.append(f"| {sg} | {g['n']} | {_pct(g[base])} | {_pct(g[alt])} | {100 * g['diff']:+.2f} | {g['gains']} / {g['losses']} | {g['p_value']:.3g} |")
        L.append("")
    L += [
        "Verdict: exact McNemar test on paired correctness; *helps*/*hurts* when p < alpha, else *neutral*. "
        "Subgroup p-values are not corrected for multiple comparisons. "
        "Outputs are re-scored with the current parser at analysis time (`rescored_changed` in summary.json); "
        "unparsed outputs are listed in `parse_failures.jsonl`. *No answer* = no valid answer could be extracted; "
        "*refusal* = the output explicitly declines (\"none of the options\", \"no object matches\", empty box list, \"N/A\").",
        "",
    ]
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--analyze-only", action="store_true", help="recompute summary from existing preds; no GPU needed")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ecfg = cfg[EXP]
    set_seed(int(cfg["common"]["seed"]))
    for c in ecfg["conditions"]:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; choose from {CONDITIONS}")
    fams = list(ecfg.get("run_families") or ecfg["families"])
    unknown = [f for f in fams if f not in ecfg["families"]]
    if unknown:
        raise SystemExit(f"unknown family {unknown}; defined: {list(ecfg['families'])}")

    if args.analyze_only and not ecfg.get("run_name"):
        raise SystemExit("--analyze-only needs --set E3_2.run_name=<existing run>")
    run_dir = make_run_dir(cfg, EXP)
    print(f"run dir: {run_dir}")
    if args.analyze_only:
        analyze(cfg, run_dir)
        print((run_dir / "summary.md").read_text(encoding="utf-8"))
        return

    save_resolved_config(cfg, run_dir)
    from common.tasks import load_family
    from common.vlm import VLM

    vlm = VLM(cfg["common"]["model"], system_prompt=ecfg["prompts"].get("system"))
    dump_json({**environment_info(), "backend": vlm.backend, "model": vlm.name}, run_dir / "env.json")
    timings_path = run_dir / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    for fam in fams:
        items = load_family(fam, ecfg["families"][fam], cfg)
        types = _counts(it.answer_type for it in items)
        print(f"\n##### family {fam}: {len(items)} items {types}")
        for cond in ecfg["conditions"]:
            secs = run_family_condition(fam, cond, items, vlm, cfg, run_dir)
            timings[f"{fam}/{cond}"] = timings.get(f"{fam}/{cond}", 0.0) + secs
            dump_json(timings, timings_path)
        del items
        gc.collect()

    analyze(cfg, run_dir)
    print()
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
