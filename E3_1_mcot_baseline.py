#!/usr/bin/env python
"""E3.1 — MCoT baseline on ScienceQA.

Reproduces the two-stage rationale-then-answer setup of Zhang et al. (2023,
arXiv:2302.00923) with a modern 7B VLM used training-free (prompting only), and
compares it against direct answering on the same items.

Conditions (config: E3_1.conditions)
  direct           image + question -> letter
  cot              single call, "think step by step", then the letter
  two_stage        stage 1 writes a rationale; stage 2 answers given image + rationale
  two_stage_noimg  stage 2 without the image (reuses two_stage rationales)
  oracle           stage 2 with ScienceQA's gold solution as the rationale

Outputs in outputs/E3_1/<run_name>/
  preds_<condition>.jsonl   one row per item (raw outputs, rationale, parsed letter, correctness)
  summary.json / summary.md accuracy, 95% CIs, paired McNemar tests vs baseline, ScienceQA breakdown
  per_item.csv              one row per item, one column per condition (for E3.5 sampling)
  config.yaml, env.json, timings.json

Usage
  python E3_1_mcot_baseline.py                                  # full run with config.yaml
  python E3_1_mcot_baseline.py --set E3_1.max_samples=100       # quick check
  python E3_1_mcot_baseline.py --set E3_1.run_name=full_test    # named run; rerun same name to resume
  python E3_1_mcot_baseline.py --analyze-only --set E3_1.run_name=full_test
"""
from __future__ import annotations

import csv
import json
import shutil
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
from common.parsing import extract_choice

EXP = "E3_1"
ALL_CONDITIONS = ["direct", "cot", "two_stage", "two_stage_noimg", "oracle"]


# =============================================================================
# Generation
# =============================================================================
def fill(template: str, item, include_lecture: bool, rationale: str | None = None) -> str:
    fields = dict(question=item.question, context=item.context(include_lecture), options=item.options_str())
    if rationale is not None:
        fields["rationale"] = rationale.strip() or "N/A"
    return template.format(**fields)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def scored_row(item, condition: str, raw: str, out_tokens: int, finish: str | None, **extra) -> dict:
    pred, rule = extract_choice(raw, item.choices)
    row = item.meta()
    row.update(
        condition=condition,
        raw_output=raw,
        out_tokens=out_tokens,
        finish_reason=finish,
        pred=pred,
        parse_rule=rule,
        correct=pred == item.answer_letter,
    )
    row.update(extra)
    return row


def reuse_source(cfg: dict, run_dir: Path) -> Path | None:
    """Run directory named by E3_1.reuse.from_run (a sibling of this run), or None."""
    name = (cfg[EXP].get("reuse") or {}).get("from_run")
    if not name:
        return None
    src = run_dir.parent / name
    if src.resolve() == run_dir.resolve():
        raise SystemExit("E3_1.reuse.from_run must name a different run than E3_1.run_name")
    if not src.is_dir():
        raise SystemExit(f"E3_1.reuse.from_run: {src} does not exist")
    return src


def copy_reused_predictions(cfg: dict, run_dir: Path) -> None:
    """Copy whole prediction files of E3_1.reuse.copy_conditions from the source run (once)."""
    src = reuse_source(cfg, run_dir)
    if src is None:
        return
    for cond in cfg[EXP]["reuse"].get("copy_conditions") or []:
        s_path, d_path = src / f"preds_{cond}.jsonl", run_dir / f"preds_{cond}.jsonl"
        if d_path.exists():
            continue
        if not s_path.exists():
            raise SystemExit(f"cannot copy {cond}: {s_path} not found")
        shutil.copyfile(s_path, d_path)
        print(f"copied {cond} predictions from {src.name}")


def run_condition(condition: str, items, vlm, cfg: dict, run_dir: Path) -> float:
    from common.vlm import GenParams, Request

    ecfg = cfg[EXP]
    prompts = ecfg["prompts"]
    lec = bool(ecfg.get("include_lecture", False))
    seed = int(cfg["common"]["seed"])
    gp_answer = GenParams.from_cfg(ecfg["generation"]["answer"], seed)
    gp_rat = GenParams.from_cfg(ecfg["generation"]["rationale"], seed)
    gp_cot = GenParams.from_cfg(ecfg["generation"]["cot"], seed)

    path = run_dir / f"preds_{condition}.jsonl"
    finished = done_ids(path) if path.exists() else set()
    todo = [it for it in items if it.id not in finished]
    print(f"\n=== {condition}: {len(finished)} done, {len(todo)} to run ===")
    if not todo:
        return 0.0

    rationales_from_two_stage = {}
    if condition == "two_stage_noimg":
        rationales_from_two_stage = {r["id"]: r for r in read_jsonl(run_dir / "preds_two_stage.jsonl")}
        missing = [it.id for it in todo if it.id not in rationales_from_two_stage]
        if missing:
            raise RuntimeError(
                f"two_stage_noimg needs two_stage rationales; {len(missing)} items are missing. "
                "List two_stage before two_stage_noimg in E3_1.conditions."
            )

    saved_rationales = {}
    src = reuse_source(cfg, run_dir)
    if condition == "two_stage" and src is not None and (ecfg.get("reuse") or {}).get("rationales", False):
        saved_rationales = {r["id"]: r for r in read_jsonl(src / "preds_two_stage.jsonl")}
        n_hit = sum(it.id in saved_rationales for it in todo)
        print(f"reusing {n_hit} stage-1 rationales from {src.name}; generating {len(todo) - n_hit}")

    t0 = time.time()
    for chunk in chunks(todo, int(ecfg.get("chunk_size", 256))):
        rows = []
        if condition == "direct":
            outs = vlm.generate([Request(fill(prompts["direct"], it, lec), it.image) for it in chunk], gp_answer, condition)
            rows = [scored_row(it, condition, o.text, o.n_tokens, o.finish_reason) for it, o in zip(chunk, outs)]

        elif condition == "cot":
            outs = vlm.generate([Request(fill(prompts["cot"], it, lec), it.image) for it in chunk], gp_cot, condition)
            rows = [scored_row(it, condition, o.text, o.n_tokens, o.finish_reason) for it, o in zip(chunk, outs)]

        elif condition in ("two_stage", "oracle", "two_stage_noimg"):
            if condition == "two_stage":
                need = [it for it in chunk if it.id not in saved_rationales]
                rat_outs = vlm.generate(
                    [Request(fill(prompts["rationale"], it, lec), it.image) for it in need], gp_rat, "stage1-rationale"
                )
                fresh = {it.id: (o.text, o.n_tokens, o.finish_reason) for it, o in zip(need, rat_outs)}
                rats = [
                    fresh[it.id]
                    if it.id in fresh
                    else (
                        saved_rationales[it.id]["rationale"],
                        saved_rationales[it.id]["rationale_tokens"],
                        saved_rationales[it.id]["rationale_finish"],
                    )
                    for it in chunk
                ]
            elif condition == "oracle":
                rats = [(it.solution, 0, None) for it in chunk]
            else:
                rats = [
                    (
                        rationales_from_two_stage[it.id]["rationale"],
                        rationales_from_two_stage[it.id]["rationale_tokens"],
                        rationales_from_two_stage[it.id]["rationale_finish"],
                    )
                    for it in chunk
                ]
            use_image = condition != "two_stage_noimg"
            ans_outs = vlm.generate(
                [
                    Request(fill(prompts["answer_with_rationale"], it, lec, rationale=r[0]), it.image if use_image else None)
                    for it, r in zip(chunk, rats)
                ],
                gp_answer,
                f"{condition}-stage2",
            )
            for it, (rtext, rtok, rfin), o in zip(chunk, rats, ans_outs):
                # Does the rationale already name an option? (the prompt asks it not to)
                stated, stated_rule = extract_choice(rtext, it.choices)
                stated = stated if stated_rule == "explicit" else None
                rows.append(
                    scored_row(
                        it,
                        condition,
                        o.text,
                        o.n_tokens,
                        o.finish_reason,
                        rationale=rtext,
                        rationale_tokens=rtok,
                        rationale_finish=rfin,
                        rationale_stated_answer=stated,
                        stage2_sees_image=use_image and it.has_image,
                        rationale_reused=condition == "two_stage" and it.id in saved_rationales,
                    )
                )
        else:
            raise ValueError(f"unknown condition {condition!r}; choose from {ALL_CONDITIONS}")

        append_jsonl(path, rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"[{condition}] wrote {len(rows)} rows (chunk acc {acc:.3f})")
    return time.time() - t0


# =============================================================================
# Analysis (reads only the JSONL files, so it can run without a GPU)
# =============================================================================
def analyze(cfg: dict, run_dir: Path, item_ids: list[str] | None) -> dict:
    from common.metrics import (
        accuracy,
        group_accuracy,
        mcnemar_exact,
        paired_bootstrap_diff_ci,
        verdict,
        wilson_ci,
    )
    from common.scienceqa import GROUP_ORDER

    ecfg = cfg[EXP]
    acfg = ecfg.get("analysis", {})
    n_boot = int(acfg.get("bootstrap_samples", 2000))
    alpha = float(acfg.get("alpha", 0.05))
    base = ecfg.get("baseline_condition", "direct")

    preds: dict[str, dict[str, dict]] = {}
    for cond in [c for c in ALL_CONDITIONS if c in ecfg["conditions"]]:
        rows = read_jsonl(run_dir / f"preds_{cond}.jsonl")
        if rows:
            by_id = {r["id"]: r for r in rows}  # last write wins
            if item_ids is not None:
                keep = set(item_ids)
                by_id = {k: v for k, v in by_id.items() if k in keep}
            preds[cond] = by_id
    if not preds:
        raise SystemExit(f"no predictions found in {run_dir}")

    # Paired comparisons only use items every condition has finished.
    common_ids = sorted(set.intersection(*(set(p) for p in preds.values())))
    summary: dict = {
        "run_dir": str(run_dir),
        "model": cfg["common"]["model"]["name"],
        "split": ecfg["split"],
        "baseline": base,
        "n_common": len(common_ids),
        "conditions": {},
    }

    for cond, by_id in preds.items():
        recs = [by_id[i] for i in common_ids]
        corr = [bool(r["correct"]) for r in recs]
        lo, hi = wilson_ci(corr, alpha)
        entry = {
            "n": len(recs),
            "n_total_in_file": len(by_id),
            "acc": accuracy(corr),
            "ci": [lo, hi],
            "parse_fail_rate": accuracy([r["pred"] is None for r in recs]),
            "parse_rules": _counts(r["parse_rule"] for r in recs),
            "mean_out_tokens": _mean(r["out_tokens"] for r in recs),
            "truncated_rate": accuracy([r.get("finish_reason") == "length" for r in recs]),
            "groups": group_accuracy(recs, GROUP_ORDER),
        }
        if "rationale" in (recs[0] if recs else {}):
            stated = [r for r in recs if r.get("rationale_stated_answer")]
            entry["rationale"] = {
                "mean_tokens": _mean(r.get("rationale_tokens") or 0 for r in recs),
                "truncated_rate": accuracy([r.get("rationale_finish") == "length" for r in recs]),
                "empty_rate": accuracy([not (r.get("rationale") or "").strip() for r in recs]),
                "states_answer_rate": len(stated) / len(recs) if recs else float("nan"),
                "final_agrees_with_stated_rate": (
                    accuracy([r["pred"] == r["rationale_stated_answer"] for r in stated]) if stated else None
                ),
                "stated_answer_correct_rate": (
                    accuracy([r["rationale_stated_answer"] == r["gold"] for r in stated]) if stated else None
                ),
            }
        if cond != base and base in preds:
            a = [bool(preds[base][i]["correct"]) for i in common_ids]
            diff = entry["acc"] - accuracy(a)
            mc = mcnemar_exact(a, corr)
            dlo, dhi = paired_bootstrap_diff_ci(a, corr, n_boot, alpha)
            entry["vs_baseline"] = {
                "diff": diff,
                "diff_ci": [dlo, dhi],
                **mc,
                "verdict": verdict(diff, mc["p_value"], alpha),
            }
            # Same comparison restricted to items with an image: the setting E3.2 / E3.5 care about.
            img_ids = [i for i in common_ids if preds[base][i]["has_image"]]
            if img_ids:
                ai = [bool(preds[base][i]["correct"]) for i in img_ids]
                bi = [bool(by_id[i]["correct"]) for i in img_ids]
                mci = mcnemar_exact(ai, bi)
                di = accuracy(bi) - accuracy(ai)
                entry["vs_baseline_img_only"] = {
                    "n": len(img_ids),
                    "diff": di,
                    **mci,
                    "verdict": verdict(di, mci["p_value"], alpha),
                }
        summary["conditions"][cond] = entry

    # ---- additional paired comparisons: [a, b] -> acc(b) - acc(a) on the same items
    summary["pairs"] = []
    for a_name, b_name in acfg.get("extra_pairs") or []:
        if a_name not in preds or b_name not in preds:
            continue
        a = [bool(preds[a_name][i]["correct"]) for i in common_ids]
        b = [bool(preds[b_name][i]["correct"]) for i in common_ids]
        diff = accuracy(b) - accuracy(a)
        mc = mcnemar_exact(a, b)
        dlo, dhi = paired_bootstrap_diff_ci(a, b, n_boot, alpha)
        summary["pairs"].append(
            {"a": a_name, "b": b_name, "n": len(common_ids), "acc_a": accuracy(a), "acc_b": accuracy(b),
             "diff": diff, "diff_ci": [dlo, dhi], **mc, "verdict": verdict(diff, mc["p_value"], alpha)}
        )

    # ---- does two_stage gain only where the rationale already names a letter?
    #      Same-item comparison with the baseline, split by rationale_stated_answer.
    summary["stated_split"] = {}
    if "two_stage" in preds and base in preds:
        ts = preds["two_stage"]
        for label, keep in (("states_letter", True), ("no_letter", False)):
            sub = [i for i in common_ids if bool(ts[i].get("rationale_stated_answer")) == keep]
            if not sub:
                continue
            a = [bool(preds[base][i]["correct"]) for i in sub]
            b = [bool(ts[i]["correct"]) for i in sub]
            mc = mcnemar_exact(a, b)
            summary["stated_split"][label] = {
                "n": len(sub), "acc_base": accuracy(a), "acc_two_stage": accuracy(b),
                "diff": accuracy(b) - accuracy(a), **mc,
                "verdict": verdict(accuracy(b) - accuracy(a), mc["p_value"], alpha),
            }

    # ---- every unparsed output, for reading
    with open(run_dir / "parse_failures.jsonl", "w", encoding="utf-8") as f:
        for cond, by_id in preds.items():
            for i in common_ids:
                r = by_id[i]
                if r["pred"] is None:
                    f.write(json.dumps({"condition": cond, "id": i, "gold": r["gold"], "n_choices": r["n_choices"],
                                        "finish_reason": r.get("finish_reason"), "raw_output": r["raw_output"]},
                                       ensure_ascii=False) + "\n")

    reuse = ecfg.get("reuse") or {}
    if reuse.get("from_run"):
        summary["reused_from"] = {"run": reuse["from_run"], "rationales": bool(reuse.get("rationales")),
                                  "copied_conditions": list(reuse.get("copy_conditions") or [])}
    summary["answer_max_new_tokens"] = ecfg["generation"]["answer"]["max_new_tokens"]

    dump_json(summary, run_dir / "summary.json")
    (run_dir / "summary.md").write_text(_markdown(summary, GROUP_ORDER), encoding="utf-8")
    _write_per_item_csv(preds, common_ids, run_dir / "per_item.csv")
    return summary


def _counts(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _mean(it) -> float:
    xs = list(it)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.2f}"


def _markdown(s: dict, group_order: list[str]) -> str:
    base = s["baseline"]
    L = [
        f"# E3.1 — MCoT baseline on ScienceQA ({s['split']})",
        "",
        f"Model: `{s['model']}` · paired items: **{s['n_common']}** · baseline: `{base}`",
        "",
        "## Accuracy",
        "",
        "| condition | acc % | 95% CI (Wilson) | Δ vs base (pp) | Δ 95% CI (paired bootstrap) | gains / losses | McNemar p | verdict | img-only Δ (verdict) | parse fail % | truncated % | mean out tok |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cond, e in s["conditions"].items():
        vb = e.get("vs_baseline")
        vi = e.get("vs_baseline_img_only")
        L.append(
            "| {c} | {a} | {lo}–{hi} | {d} | {dci} | {gl} | {p} | {v} | {vi} | {pf} | {tr} | {tok:.0f} |".format(
                c=cond,
                a=_pct(e["acc"]),
                lo=_pct(e["ci"][0]),
                hi=_pct(e["ci"][1]),
                d=f"{100 * vb['diff']:+.2f}" if vb else "—",
                dci=f"{100 * vb['diff_ci'][0]:+.2f} to {100 * vb['diff_ci'][1]:+.2f}" if vb else "—",
                gl=f"{vb['gains']} / {vb['losses']}" if vb else "—",
                p=f"{vb['p_value']:.3g}" if vb else "—",
                v=f"**{vb['verdict']}**" if vb else "(baseline)",
                vi=f"{100 * vi['diff']:+.2f} ({vi['verdict']})" if vi else "—",
                pf=_pct(e["parse_fail_rate"]),
                tr=_pct(e["truncated_rate"]),
                tok=e["mean_out_tokens"],
            )
        )
    L += ["", "## ScienceQA breakdown (accuracy %)", ""]
    groups = [g for g in group_order if any(g in e["groups"] for e in s["conditions"].values())]
    L.append("| condition | " + " | ".join(groups) + " |")
    L.append("|---|" + "---|" * len(groups))
    for cond, e in s["conditions"].items():
        L.append("| " + cond + " | " + " | ".join(_pct(e["groups"].get(g, {}).get("acc")) for g in groups) + " |")
    first = next(iter(s["conditions"].values()))
    L.append("| *n* | " + " | ".join(str(first["groups"].get(g, {}).get("n", 0)) for g in groups) + " |")

    rat = {c: e["rationale"] for c, e in s["conditions"].items() if "rationale" in e}
    if rat:
        L += [
            "",
            "## Rationale diagnostics",
            "",
            "| condition | mean rationale tok | truncated % | empty % | states an answer % | final = stated % | stated correct % |",
            "|---|---|---|---|---|---|---|",
        ]
        for c, r in rat.items():
            L.append(
                f"| {c} | {r['mean_tokens']:.0f} | {_pct(r['truncated_rate'])} | {_pct(r['empty_rate'])} | "
                f"{_pct(r['states_answer_rate'])} | {_pct(r['final_agrees_with_stated_rate'])} | {_pct(r['stated_answer_correct_rate'])} |"
            )
    if s.get("pairs"):
        L += [
            "",
            "## Additional paired comparisons (Δ = b − a, same items)",
            "",
            "| a | b | acc a % | acc b % | Δ pp | Δ 95% CI | b gains / losses | McNemar p | verdict (b vs a) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for pr in s["pairs"]:
            L.append(
                f"| {pr['a']} | {pr['b']} | {_pct(pr['acc_a'])} | {_pct(pr['acc_b'])} | {100 * pr['diff']:+.2f} | "
                f"{100 * pr['diff_ci'][0]:+.2f} to {100 * pr['diff_ci'][1]:+.2f} | {pr['gains']} / {pr['losses']} | "
                f"{pr['p_value']:.3g} | **{pr['verdict']}** |"
            )
    if s.get("stated_split"):
        L += [
            "",
            f"## two_stage vs {base}, split by whether the rationale names an option letter (same items)",
            "",
            f"| rationale | n | {base} % | two_stage % | Δ pp | gains / losses | McNemar p | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for label, g in s["stated_split"].items():
            L.append(
                f"| {label} | {g['n']} | {_pct(g['acc_base'])} | {_pct(g['acc_two_stage'])} | {100 * g['diff']:+.2f} | "
                f"{g['gains']} / {g['losses']} | {g['p_value']:.3g} | **{g['verdict']}** |"
            )
    if s.get("reused_from"):
        r = s["reused_from"]
        L += ["", f"Reused from run `{r['run']}`: stage-1 rationales = {r['rationales']}, copied conditions = {r['copied_conditions'] or 'none'}."]
    L += [
        "",
        f"Answer-call limit: {s.get('answer_max_new_tokens')} new tokens. Unparsed outputs are listed in `parse_failures.jsonl`.",
        "Verdict: McNemar exact test vs the baseline on the same items; *helps*/*hurts* when p < alpha, else *neutral*.",
        "",
    ]
    return "\n".join(L)


def _write_per_item_csv(preds: dict, ids: list[str], path: Path) -> None:
    conds = list(preds)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "row", "gold", "has_image", "subject", "grade"] + [f"{c}_pred" for c in conds] + [f"{c}_correct" for c in conds])
        for i in ids:
            r0 = preds[conds[0]][i]
            w.writerow(
                [i, r0["row"], r0["gold"], r0["has_image"], r0["subject"], r0["grade"]]
                + [preds[c][i]["pred"] for c in conds]
                + [int(bool(preds[c][i]["correct"])) for c in conds]
            )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--analyze-only", action="store_true", help="recompute summary from existing preds_*.jsonl; no GPU needed")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ecfg = cfg[EXP]
    set_seed(int(cfg["common"]["seed"]))

    conditions = list(ecfg["conditions"])
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; choose from {ALL_CONDITIONS}")
    if "two_stage_noimg" in conditions and "two_stage" in conditions:
        # make sure rationales exist before they are reused
        conditions.remove("two_stage_noimg")
        conditions.insert(conditions.index("two_stage") + 1, "two_stage_noimg")

    if args.analyze_only and not ecfg.get("run_name"):
        raise SystemExit("--analyze-only needs --set E3_1.run_name=<existing run>")
    run_dir = make_run_dir(cfg, EXP)
    print(f"run dir: {run_dir}")

    if args.analyze_only:
        # Describe the run as it was generated: generation / reuse settings come from
        # the config saved with the run; analysis settings come from the current config.
        stored = run_dir / "config.yaml"
        if stored.exists():
            import yaml

            old = yaml.safe_load(stored.read_text(encoding="utf-8")).get(EXP, {})
            for key in ("generation", "reuse"):
                if key in old:
                    ecfg[key] = old[key]
                else:
                    ecfg.pop(key, None)
        summary = analyze(cfg, run_dir, item_ids=None)
        print((run_dir / "summary.md").read_text(encoding="utf-8"))
        return

    save_resolved_config(cfg, run_dir)

    from common.scienceqa import load_scienceqa

    items = load_scienceqa(
        cfg["common"]["datasets"]["scienceqa"]["hf_id"],
        ecfg["split"],
        image_only=bool(ecfg.get("image_only", False)),
        max_samples=ecfg.get("max_samples"),
        sample_seed=int(ecfg.get("sample_seed", 42)),
    )
    n_img = sum(it.has_image for it in items)
    print(f"ScienceQA {ecfg['split']}: {len(items)} items ({n_img} with image)")

    from common.vlm import VLM

    vlm = VLM(cfg["common"]["model"], system_prompt=ecfg["prompts"].get("system"))
    dump_json({**environment_info(), "backend": vlm.backend, "model": vlm.name}, run_dir / "env.json")

    copy_reused_predictions(cfg, run_dir)
    timings_path = run_dir / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    for cond in conditions:
        secs = run_condition(cond, items, vlm, cfg, run_dir)
        timings[cond] = timings.get(cond, 0.0) + secs
        dump_json(timings, timings_path)

    analyze(cfg, run_dir, item_ids=[it.id for it in items])
    print()
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
