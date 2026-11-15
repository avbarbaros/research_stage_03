# research_stage_03
Multimodal reasoning models Stage-3: multimodal chain-of-thought and compositional reasoning.

## Layout

```
config.yaml              every configurable parameter (setup + each experiment)
setup.py                 prepares a Colab runtime (run it directly; it is not a setuptools file)
E3_1_mcot_baseline.py    E3.1 two-stage rationale-then-answer vs direct answering on ScienceQA
E3_2_cot_harm_audit.py   E3.2 direct vs CoT on math / science / counting / grounding
docs/                    write-ups of each experiment's results (docs/E3_1_results.md)
common/                  shared code: config, dataset loaders + scoring (tasks.py), VLM wrapper (vLLM / transformers),
                         answer parsing, metrics
outputs/<EXP>/<run>/     results (git-ignored)
```

## Colab (L4) quick start

```bash
# optional: keep weights across sessions
#   from google.colab import drive; drive.mount('/content/drive')
#   then set common.hf_cache_dir in config.yaml to /content/drive/MyDrive/hf_cache

git clone <this repo> && cd research_stage_03
python setup.py --smoke-test                 # install, download model + ScienceQA, answer 4 items

python E3_1_mcot_baseline.py --set E3_1.max_samples=100 --set E3_1.run_name=quick   # ~minutes
python E3_1_mcot_baseline.py --set E3_1.run_name=full_test                          # whole test split
```

Any config value can be overridden with `--set key.path=value` (repeatable).

**Resuming:** predictions are written in chunks. If the runtime dies, rerun the same
command with the same `run_name`; finished items are skipped.

**Re-running only the answer calls** (e.g. after changing `E3_1.generation.answer`): start a new run that
reuses the stage-1 rationales and copies the CoT predictions of an earlier run:
`python E3_1_mcot_baseline.py --set E3_1.run_name=full_img_a64 --set E3_1.reuse.from_run=full_img`

**Re-analysing without a GPU:** `python E3_1_mcot_baseline.py --analyze-only --set E3_1.run_name=full_test`

## E3.1 conditions

| condition | calls | what it tests |
|---|---|---|
| `direct` | 1 | baseline: image + question → letter |
| `cot` | 1 | zero-shot "think step by step" in a single call |
| `two_stage` | 2 | Zhang et al. 2023: rationale first, then answer given the rationale |
| `two_stage_noimg` | 1 (+ reused rationales) | stage 2 without the image: is the answer carried by the rationale alone? |
| `oracle` | 1 | stage 2 with the gold ScienceQA solution: ceiling for rationale quality |

`summary.md` reports accuracy with bootstrap 95% CIs, the paired difference against `direct`
with an exact McNemar test (verdict: helps / hurts / neutral), the standard ScienceQA breakdown
(NAT/SOC/LAN, TXT/IMG/NO, G1-6/G7-12), parse-failure and truncation rates, and rationale diagnostics.
`preds_two_stage.jsonl` keeps each rationale with its correctness; E3.5 samples from it.
