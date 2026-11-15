# E3.1 — Multimodal CoT Baseline on ScienceQA: Results

**Stage:** 3 — Multimodal Chain-of-Thought and Compositional Reasoning
**Run:** `full_img` · **Date:** November 2026
**Script:** [`E3_1_mcot_baseline.py`](../E3_1_mcot_baseline.py) · **Config:** [`config.yaml`](../config.yaml) (section `E3_1`)

---

## 1. Summary

The roadmap asks: *reproduce the two-stage rationale-then-answer setup on ScienceQA with a modern 7B VLM, compare it with direct answering, and say whether the rationale helps, hurts or is neutral.*

**The success criterion is met. On the 2,017 image questions of the ScienceQA test split, the rationale helps.** Two-stage accuracy is 88.05% against 85.42% for direct answering. That is +2.63 points (95% CI +1.19 to +4.07; exact McNemar p = 2.8 × 10⁻⁴).

Four further findings shape the next experiments:

1. **Single-call CoT does better than the two-stage pipeline:** +4.21 points over direct, against +2.63. But 2.7% of two-stage answers could not be parsed, which is larger than the 1.6-point gap between them. Which of the two is better is therefore **not yet established** (§5.2).
2. **The second stage barely uses the image.** Removing the image from the answer call costs only 0.25 points (88.05% → 87.80%). The final answer is carried by the rationale (§5.3).
3. **26% of rationales name the option letter** even though the prompt forbids it, and the final answer copies that letter every time (§5.4).
4. **A perfect rationale is worth +12 points** (oracle: 97.47%). Generated rationales recover about a fifth of that headroom (+2.63 of +12.05 points). The oracle figure is inflated by answer leakage in the gold solutions (§5.5).

---

## 2. Setup

### 2.1 Data

| | |
|---|---|
| Dataset | ScienceQA (Lu et al., NeurIPS 2022), Hugging Face copy `derek-thomas/ScienceQA` |
| Split | `test` (4,241 items) |
| Filter | `image_only: true` → **2,017 items with an image**; every condition is scored on the same items |
| Item ID | `test-<row index>` (the HF copy has no question ID; row order is stable) |
| Context field | the item's `hint`, or `N/A` if empty; `lecture` **not** included (`include_lecture: false`) |

Composition of the 2,017 items: NAT 1,209 · SOC 764 · LAN 44 · with text context (TXT) 1,257 · grades 1–6: 1,429 · grades 7–12: 588. The "NO context" group is empty because every item has an image.

### 2.2 Model and inference

| | |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, no fine-tuning (prompting only) |
| Backend | vLLM, `max_model_len` 8192, `gpu_memory_utilization` 0.90, 1 image per prompt |
| Image resolution | `min_pixels` 200,704 (256 visual tokens) · `max_pixels` 1,003,520 (1,280 visual tokens) |
| Decoding | greedy (temperature 0, top_p 1.0), single run |
| Max new tokens | answer calls 16 · rationale calls 512 · single-call CoT 768 |
| Hardware | Google Colab, 1 × NVIDIA L4 (24 GB) |
| Prompt format | Qwen chat template, default system prompt |

Exact library versions and time per condition are in `env.json` and `timings.json` in the run directory.

### 2.3 Conditions

| Condition | Model calls | Stage-2 input | What it tests |
|---|---|---|---|
| `direct` | 1 | — | Baseline: image + question → option letter |
| `cot` | 1 | — | Zero-shot "think step by step" in one call (control) |
| `two_stage` | 2 | image + question + **generated** rationale | Zhang et al. (2023) rationale-then-answer, used without training |
| `two_stage_noimg` | 1 (+ reused rationales) | question + generated rationale, **no image** | Does stage 2 use the image, or only the rationale? |
| `oracle` | 1 | image + question + **gold** `solution` | Ceiling for rationale quality |

**Difference from the original paper.** Zhang et al. fine-tune a model of under 1B parameters on ScienceQA's training rationales. Here a 7B instruction-tuned VLM is used zero-shot with the same two-stage structure. The results show whether the *structure* helps a modern model. They do not reproduce the paper's numbers and should not be compared with its table directly.

### 2.4 Prompts (verbatim)

`{options}` is rendered as `(A) … (B) …`.

**direct**
```
Question: {question}
Context: {context}
Options: {options}
Answer with the option's letter from the given choices directly.
```

**cot**
```
Question: {question}
Context: {context}
Options: {options}
Think step by step, using the image and the context. Then finish with a final line of the form "The answer is (X)." where X is the option letter.
```

**rationale** (stage 1 of `two_stage`)
```
Question: {question}
Context: {context}
Options: {options}
Write a short step-by-step solution that explains how to answer the question, based on the image and the context. Write only the reasoning; do not state which option letter is correct.
```

**answer_with_rationale** (stage 2 of `two_stage`, `two_stage_noimg`, `oracle`)
```
Question: {question}
Context: {context}
Options: {options}
Solution: {rationale}
Using the solution above, answer with the option's letter from the given choices directly.
```

### 2.5 Scoring

**Answer extraction** ([`common/parsing.py`](../common/parsing.py)). The rules are tried in this order, and every prediction records which rule matched:

1. An explicit phrase such as "The answer is (B)" or "Answer: B" (the last one in the output wins).
2. The output is a single letter.
3. The output starts with a letter, such as "B) …".
4. The output contains exactly one option's text.

If none matches, the item counts as a **parse failure** and is scored **wrong**.

**Statistics** ([`common/metrics.py`](../common/metrics.py)):
- **Per-condition 95% CI:** Wilson score interval.
- **Difference from `direct`:** 95% CI from a paired bootstrap with 2,000 resamples.
- **Significance:** exact McNemar test on the paired correct/incorrect outcomes. *Gains* are items direct got wrong and the condition got right; *losses* are the reverse.
- **Verdict:** *helps* or *hurts* when p < 0.05, otherwise *neutral*.

---

## 3. Results

### 3.1 Accuracy (n = 2,017, all items have an image)

| Condition | Correct | Acc. % | 95% CI | Δ vs direct (pp) | Δ 95% CI | Gains / losses | McNemar p | Verdict | Parse fail | Truncated | Mean out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | 1,723 | **85.42** | 83.82–86.90 | — | — | — | — | baseline | 15 (0.74%) | 0 | 2 |
| cot | 1,808 | **89.64** | 88.23–90.89 | +4.21 | +2.93 to +5.55 | 144 / 59 | 2.2e-09 | helps | 1 (0.05%) | 0 | 192 |
| two_stage | 1,776 | **88.05** | 86.56–89.40 | +2.63 | +1.19 to +4.07 | 130 / 77 | 2.8e-04 | helps | 55 (2.73%) | 44 (2.18%) | 3 |
| two_stage_noimg | 1,771 | **87.80** | 86.30–89.16 | +2.38 | +0.84 to +3.87 | 136 / 88 | 1.6e-03 | helps | 51 (2.53%) | 52 (2.58%) | 3 |
| oracle | 1,966 | **97.47** | 96.69–98.07 | +12.05 | +10.71 to +13.44 | 244 / 1 | 8.7e-72 | helps | 1 (0.05%) | 0 | 2 |

The counts were derived from the reported percentages (n = 2,017) and match the gains/losses columns. "Truncated" means the answer output hit the 16-token limit. For two-stage conditions, "Mean out tok" counts the stage-2 answer only; the rationale averages 189 tokens.

### 3.2 ScienceQA breakdown (accuracy %, Δ vs direct in pp)

| Condition | NAT (1,209) | SOC (764) | LAN (44) | TXT (1,257) | G1-6 (1,429) | G7-12 (588) |
|---|---|---|---|---|---|---|
| direct | 83.54 | 87.83 | 95.45 | 80.51 | 89.99 | 74.32 |
| cot | 88.25 (+4.71) | 91.36 (+3.53) | 97.73 (+2.28) | 87.91 (+7.40) | 91.60 (+1.61) | 84.86 (**+10.54**) |
| two_stage | 88.83 (+5.29) | 86.65 (**−1.18**) | 90.91 (**−4.54**) | 86.08 (+5.57) | 90.62 (+0.63) | 81.80 (+7.48) |
| two_stage_noimg | 88.25 (+4.71) | 86.91 (−0.92) | 90.91 (−4.54) | 85.84 (+5.33) | 90.48 (+0.49) | 81.29 (+6.97) |
| oracle | 97.68 (+14.14) | 96.99 (+9.16) | 100.00 (+4.55) | 96.42 (+15.91) | 99.30 (+9.31) | 93.03 (+18.71) |

The groups were not tested separately for significance. The LAN drop is 2 items out of 44 (42 → 40 correct), and the SOC drop is about 9 items out of 764.

### 3.3 Rationale diagnostics (`two_stage`)

| Metric | Value |
|---|---|
| Mean rationale length | 189 tokens (limit 512) |
| Rationales cut off at 512 tokens | 0.20% (~4) |
| Empty rationales | 0 |
| Rationale states an option letter explicitly | **26.13% (527)** |
| … and the stated letter is correct | 96.96% (511 of 527) |
| … and the final answer equals the stated letter | **100%** |

For `oracle`, **8.97% (181) of gold solutions are empty** in the dataset. Those items receive `Solution: N/A`.

---

## 4. Answer to the E3.1 question

| Question | Answer |
|---|---|
| Does the generated rationale help, hurt or do nothing for Qwen2.5-VL-7B on ScienceQA? | **Helps**: +2.63 pp, 95% CI [+1.19, +4.07], p = 2.8e-4 on 2,017 paired items. |
| Is the gain uniform? | **No.** Gains come from natural science (+5.3), items with text context (+5.6) and grades 7–12 (+7.5). Social and language science fall slightly (−1.2, −4.5); neither drop was tested for significance, and LAN has only 44 items. |
| Is two-stage the best way to use reasoning here? | **Not shown.** Single-call CoT scores higher (+4.21), but the gap to two-stage (1.6 pp) is smaller than two-stage's parse-failure rate (2.7 pp). |

---

## 5. Discussion

### 5.1 Both forms of reasoning help on ScienceQA

CoT improves every subject and grade group, and two-stage improves most of them. The largest gains are on the hardest items: grades 7–12 rise from 74.3% to 84.9% with CoT. This matches the reading in Stage 5 that CoT helps on mathematical and scientific questions. ScienceQA is a domain where reasoning is *expected* to help, so this result alone says nothing about where CoT *hurts*. That is the job of E3.2 (counting and visual grounding).

### 5.2 The two-stage result is pulled down by answer parsing

Two-stage has a parse-failure rate of 2.73% (55 items), against 0.74% for direct, which receives the same 16-token limit. In 44 of the stage-2 outputs the model kept writing past 16 tokens instead of giving a letter. A likely cause is that a long "Solution:" block in the prompt leads the model to continue explaining. Every parse failure is scored wrong.

- If all 55 were actually correct, two-stage would reach at most **90.78%**, above CoT's 89.64%.
- If none were, it stays at 88.05%.

So **the order of CoT and two-stage is not established.** Correcting the parse failures can only raise two-stage's score, so the +2.63-point gain over direct is a conservative estimate. Direct also has 15 parse failures of its own, so the exact gap needs a rescore of both.

### 5.3 Stage 2 answers from the rationale, not the image

Removing the image from the answer call changes accuracy by −0.25 points (5 items net), and every subject group moves by less than 0.6 points. Once the rationale is written, the answer step almost ignores the image: what the model *saw* has been converted into text in stage 1.

This has two consequences:
- Any perception error made in stage 1 cannot be corrected in stage 2. Faithfulness of the rationale to the image (E3.5) is therefore the property that matters.
- The two conditions differ on individual items: compared with direct, two-stage has 130 gains / 77 losses and the no-image version 136 / 88, so at least 17 items change. Their paired difference has not been tested yet (see §7).

### 5.4 Rationales leak the answer despite the instruction

One rationale in four ends by naming a letter, even though the prompt says not to. Stage 2 copies that letter every time. Those stated letters are correct 97% of the time (511 of 527).

Derived from the counts above: among the 1,490 rationales that do *not* name a letter, two-stage gets 1,265 correct (84.9%). That is close to the direct baseline. However, the two subsets probably differ in difficulty (the model may state a letter more readily when it is confident), so this is **not** a controlled comparison. It does suggest that much of the two-stage gain comes from items where the rationale has already settled the answer.

### 5.5 The oracle is an upper bound that includes answer leakage

Supplying ScienceQA's gold solution raises accuracy to 97.47% with a single loss (244 gains / 1 loss). Two caveats:
- Many gold solutions state the answer in plain words (e.g. "The colony is New Hampshire."). The oracle therefore measures "rationale that contains the answer", not "perfect reasoning".
- 181 items (9%) have no gold solution, so for those the oracle is effectively the direct condition with an `N/A` placeholder.

The +12-point gap is best read as **headroom for better rationales**. Generated rationales recover about 22% of it (2.63 / 12.05).

### 5.6 Threats to validity

- **One model, one prompt wording, greedy decoding.** No variation across prompts or samples was measured, so the CIs reflect sampling of *items* only.
- **Parse failures** count as errors and are unevenly spread across conditions (§5.2).
- **Possible training contamination.** ScienceQA is public and old. The 85% zero-shot direct accuracy may partly reflect exposure during Qwen2.5-VL training. This affects absolute numbers more than paired differences.
- **Image-only subset.** Results do not cover text-only ScienceQA items.
- **Group-level effects** (§3.2) were not tested for significance.

---

## 6. Reproducing

```bash
python setup.py --skip-install                    # on a fresh Colab runtime, after the first setup
python E3_1_mcot_baseline.py --set E3_1.run_name=full_img        # image_only: true in config.yaml
python E3_1_mcot_baseline.py --analyze-only --set E3_1.run_name=full_img   # rebuild summary, no GPU
```

The output directory is `<output_root>/E3_1/full_img/` (on Drive: `/content/drive/MyDrive/MMRM/research_stage_03/E3_1/full_img/`):

| File | Content |
|---|---|
| `preds_<condition>.jsonl` | one row per item: raw output, parsed letter, rule used, correctness, groups; plus rationale fields for two-stage conditions |
| `summary.json`, `summary.md` | the tables in §3 |
| `per_item.csv` | one row per item, prediction and correctness for each condition |
| `config.yaml`, `env.json`, `timings.json` | exact configuration, library versions, time per condition |

---

## 7. Open items and next steps

1. **Settle the two-stage vs CoT order.** Inspect the 55 two-stage parse failures (`parse_rule == "none"` in `preds_two_stage.jsonl`). Then rerun stage 2 with a larger answer limit (e.g. 64 tokens), reusing the saved rationales.
2. **Paired tests not yet computed:** two-stage vs no-image, and two-stage vs CoT. Both can be computed from `per_item.csv`:
   ```python
   import pandas as pd; from common.metrics import mcnemar_exact
   d = pd.read_csv("per_item.csv")
   print(mcnemar_exact(d.two_stage_noimg_correct.astype(bool), d.two_stage_correct.astype(bool)))
   print(mcnemar_exact(d.cot_correct.astype(bool), d.two_stage_correct.astype(bool)))
   ```
3. **E3.5 input.** 1,776 correct two-stage items are available for the 200-item faithfulness sample. Stratify by `rationale_stated_answer`, because rationales that name a letter and those that do not may differ in faithfulness.
4. **E3.2.** ScienceQA is a domain where CoT helps. The harm audit needs counting and grounding tasks, where the Stage-5 literature predicts the opposite.

### Contribution to the Stage-3 gate

- *"I have my own measurement of when multimodal CoT hurts rather than helps"*: **partial.** On ScienceQA, reasoning helps overall. The only drops are small, untested ones for two-stage on SOC and LAN; E3.2 is needed.
- *"I can explain why a rationale can be wrong while the answer is right"*: E3.1 supplies the inputs, namely rationales that settle the answer before stage 2 and a stage 2 that ignores the image. The measurement itself is E3.5.
