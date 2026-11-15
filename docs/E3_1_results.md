# E3.1 — Multimodal CoT Baseline on ScienceQA: Results

**Stage:** 3 — Multimodal Chain-of-Thought and Compositional Reasoning
**Run:** `full_img` (generation), re-analysed with `--analyze-only` (paired tests, same-item split, parse-failure dump) · **Date:** November 2026
**Script:** [`E3_1_mcot_baseline.py`](../E3_1_mcot_baseline.py) · **Config:** [`config.yaml`](../config.yaml) (section `E3_1`)

---

## 1. Summary

The roadmap asks: *reproduce the two-stage rationale-then-answer setup on ScienceQA with a modern 7B VLM, compare it with direct answering, and say whether the rationale helps, hurts or is neutral.*

**The success criterion is met. On the 2,017 image questions of the ScienceQA test split, the rationale helps.** Two-stage accuracy is 88.05% against 85.42% for direct answering. That is +2.63 points (95% CI +1.19 to +4.07; exact McNemar p = 2.8 × 10⁻⁴).

Further findings:

1. **Single-call CoT beats the two-stage pipeline:** 89.64% against 88.05%, a difference of −1.59 points (95% CI −2.78 to −0.35; p = 0.016). The 55 two-stage answers that could not be parsed are not formatting accidents. In 45 of them the model, after reading its own rationale, **rejects every option** ("None of the options provided are correct"). Scoring them as wrong is correct (§5.2).
2. **The answer step does not use the image.** Removing the image from the second call changes accuracy by −0.25 points (95% CI −0.79 to +0.30; p = 0.49). The final answer is carried by the rationale (§5.3).
3. **The gain is not explained by rationales that give away the letter.** On the same items, two-stage beats direct by +2.66 points where the rationale names a letter, and by +2.62 points where it does not; both are significant (§5.4).
4. **A perfect rationale is worth much more.** Gold solutions add +9.42 points on top of two-stage (210 gains / 20 losses). Generated rationales recover only about a fifth of the headroom over direct (+2.63 of +12.05 points). The oracle figure is inflated by answers stated inside the gold solutions (§5.5).

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

A letter that is not one of the item's options is not accepted. If no rule matches, the output is unparsed (`pred = None`) and scored **wrong**. All unparsed outputs are listed in `parse_failures.jsonl`, and §3.6 classifies them.

**Statistics** ([`common/metrics.py`](../common/metrics.py)):
- **Per-condition 95% CI:** Wilson score interval.
- **Paired differences:** 95% CI from a paired bootstrap with 2,000 resamples.
- **Significance:** exact McNemar test on paired correct/incorrect outcomes. *Gains* are items the reference got wrong and the compared condition got right; *losses* are the reverse.
- **Verdict:** *helps* or *hurts* when p < 0.05, otherwise *neutral*. No correction for multiple comparisons.

---

## 3. Results

### 3.1 Accuracy (n = 2,017, all items have an image)

| Condition | Correct | Acc. % | 95% CI | Δ vs direct (pp) | Δ 95% CI | Gains / losses | McNemar p | Verdict | Unparsed | Truncated | Mean out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | 1,723 | **85.42** | 83.82–86.90 | — | — | — | — | baseline | 15 (0.74%) | 0 | 2 |
| cot | 1,808 | **89.64** | 88.23–90.89 | +4.21 | +2.93 to +5.55 | 144 / 59 | 2.2e-09 | helps | 1 (0.05%) | 0 | 192 |
| two_stage | 1,776 | **88.05** | 86.56–89.40 | +2.63 | +1.19 to +4.07 | 130 / 77 | 2.8e-04 | helps | 55 (2.73%) | 44 (2.18%) | 3 |
| two_stage_noimg | 1,771 | **87.80** | 86.30–89.16 | +2.38 | +0.84 to +3.87 | 136 / 88 | 1.6e-03 | helps | 51 (2.53%) | 52 (2.58%) | 3 |
| oracle | 1,966 | **97.47** | 96.69–98.07 | +12.05 | +10.71 to +13.44 | 244 / 1 | 8.7e-72 | helps | 1 (0.05%) | 0 | 2 |

The counts were derived from the reported percentages (n = 2,017) and match the gains/losses columns. "Truncated" means the answer output hit the 16-token limit. For two-stage conditions, "Mean out tok" counts the stage-2 answer only; the rationale averages 189 tokens.

How answers were extracted:

| Condition | Single letter | Starts with a letter | "The answer is (X)" | Option text | Unparsed |
|---|---|---|---|---|---|
| direct | 2,002 | — | — | — | 15 |
| cot | — | — | 1,983 | 33 | 1 |
| two_stage | 1,893 | 63 | — | 6 | 55 |
| two_stage_noimg | 1,911 | 45 | — | 10 | 51 |
| oracle | 2,015 | 1 | — | — | 1 |

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

### 3.4 Additional paired comparisons (same 2,017 items, Δ = b − a)

| a | b | Acc. a % | Acc. b % | Δ pp | Δ 95% CI | b gains / losses | McNemar p | Verdict (b vs a) |
|---|---|---|---|---|---|---|---|---|
| two_stage_noimg | two_stage | 87.80 | 88.05 | +0.25 | −0.30 to +0.79 | 19 / 14 | 0.49 | neutral |
| cot | two_stage | 89.64 | 88.05 | **−1.59** | −2.78 to −0.35 | 67 / 99 | 0.016 | **hurts** |
| two_stage | oracle | 88.05 | 97.47 | +9.42 | +8.08 to +10.86 | 210 / 20 | 3.9e-41 | helps |

### 3.5 Two-stage vs direct, split by whether the rationale names a letter (same items)

| Rationale | n | direct % | two_stage % | Δ pp | Gains / losses | McNemar p | Verdict |
|---|---|---|---|---|---|---|---|
| names a letter | 527 | 94.31 | 96.96 | +2.66 | 17 / 3 | 0.0026 | helps |
| no letter | 1,490 | 82.28 | 84.90 | +2.62 | 113 / 74 | 0.0053 | helps |

### 3.6 What the unparsed outputs are

Every unparsed output in `parse_failures.jsonl` was read and assigned to one category.

| Category | Example output | direct | cot | two_stage | two_stage_noimg | oracle |
|---|---|---|---|---|---|---|
| Rejects every option | "None of the options provided are correct based on the items listed." · "The correct answer is not listed among the options." | — | — | **43** | 43 | — |
| `N/A` | "N/A" | — | — | 2 | 2 | — |
| Letter that is not an option | "F", "C" on a 2-option item | **15** | 1 | 5 | 5 | 1 |
| Free text matching no option | "The mammal in the image is the sea otter." | — | — | 4 | — | — |
| Two letters | "A, C" | — | — | 1 | 1 | — |
| **Total** | | 15 | 1 | 55 | 51 | 1 |

Notes:
- The 15 direct failures are 10 × "F" and 5 × "C", all on 2-option questions.
- The single CoT failure (test-5) reasons its way to an organism that is not among the options and answers "(C)".
- 44 of the 55 two-stage failures were cut off at 16 tokens. Every one of the 44 was cut off in the middle of a rejection or hedge ("None of the provided options (A or B) …", "The correct answer is not explicitly provided in the options. However, based on the …").
- Many of the rejections are on the 4-option "what can X and Y trade" questions about two lunch-item lists. This is judged from the output text; the item topic is not stored per row.

---

## 4. Answer to the E3.1 question

| Question | Answer |
|---|---|
| Does the generated rationale help, hurt or do nothing for Qwen2.5-VL-7B on ScienceQA? | **Helps**: +2.63 pp, 95% CI [+1.19, +4.07], p = 2.8e-4 on 2,017 paired items. |
| Is the gain uniform? | **No.** Gains come from natural science (+5.3), items with text context (+5.6) and grades 7–12 (+7.5). Social and language science fall slightly (−1.2, −4.5); neither drop was tested for significance, and LAN has only 44 items. |
| Is two-stage the best way to use reasoning here? | **No.** Single-call CoT is better by 1.59 pp (p = 0.016). Splitting reasoning and answering into two calls costs accuracy, mostly because stage 2 sometimes refuses every option after reading the rationale. |
| Does the gain come from rationales that give away the letter? | **No.** The gain over direct is the same (+2.6 pp) with and without a stated letter. |

---

## 5. Discussion

### 5.1 Both forms of reasoning help on ScienceQA

CoT improves every subject and grade group, and two-stage improves most of them. The largest gains are on the hardest items: grades 7–12 rise from 74.3% to 84.9% with CoT. This matches the reading in Stage 5 that CoT helps on mathematical and scientific questions. ScienceQA is a domain where reasoning is *expected* to help, so this result alone says nothing about where CoT *hurts*. That is the job of E3.2 (counting and visual grounding).

### 5.2 Two-stage loses to CoT because stage 2 rejects the options

Two-stage has 55 unparsed answers (2.73%), against 15 for direct and 1 for CoT. Reading them (§3.6) shows that **45 of the 55 are the model declining to answer**: 43 rejections of every option plus 2 `N/A`. In each case the rationale reached a conclusion that matches none of the options, and stage 2, told to use "the solution above", sided with the rationale over the options.

This is a genuine failure of the pipeline, not of the answer extraction. Counting these items as wrong is the right scoring, so CoT's advantage (−1.59 pp, p = 0.016) stands.

Why does the single call not fail the same way? A plausible explanation is that CoT has to end with "The answer is (X)" in the same generation. That forces the reasoning back onto the given options. The two-stage split removes this constraint: the rationale can drift, and the answer step inherits the drift.

**Answer-limit re-run.** A re-run with a 64-token answer limit, reusing the same rationales (`full_img_a64`), was planned to check whether the truncated outputs end in a letter. Every truncated output is a rejection or hedge in progress, so at most a few hedged recoveries ("…however, the closest option is B") can be expected. The re-run is now an optional robustness check rather than a needed correction.

### 5.3 Stage 2 answers from the rationale, not the image

Removing the image from the answer call changes accuracy by +0.25 pp in favour of keeping it. The 95% CI is −0.30 to +0.79 and p = 0.49, with only 33 items changing (19 / 14). Every subject group moves by less than 0.6 points, and the two conditions share most of their failures (§3.6).

Once the rationale is written, the answer step behaves like a text-only reader of it: what the model *saw* has been converted into text in stage 1. Two consequences follow:
- A perception error made in stage 1 cannot be corrected in stage 2. How faithful the rationale is to the image (E3.5) is therefore the property that matters.
- Stage 2 follows the rationale even when that means rejecting every option (§5.2). This is the same dependence seen from the failure side.

### 5.4 The gain is not explained by rationales that give away the letter

One rationale in four ends by naming a letter, even though the prompt says not to, and stage 2 copies it every time. A same-item comparison with direct (§3.5) shows that this does not explain the gain:
- **Rationale names a letter (527 items):** these are easy items, with direct already at 94.3%. Two-stage adds +2.66 pp (17 / 3, p = 0.0026).
- **No letter (1,490 items):** harder items, with direct at 82.3%. Two-stage adds +2.62 pp (113 / 74, p = 0.0053).

The earlier, uncontrolled reading (two-stage at 84.9% on the no-letter items, "close to direct overall") was misleading because these items are harder than average. Compared on the same items, the rationale helps by the same amount whether or not it states the letter. The model also states a letter more often on items it finds easy, which fits stating a letter being a sign of confidence.

### 5.5 The oracle is an upper bound that includes answer leakage

Supplying ScienceQA's gold solution raises accuracy to 97.47%, with a single loss against direct (244 / 1) and +9.42 pp over two-stage (210 / 20). Two caveats:
- Many gold solutions state the answer in plain words (e.g. "The colony is New Hampshire."). The oracle therefore measures "a rationale that contains the answer", not "perfect reasoning".
- 181 items (9%) have no gold solution, so for those the oracle is effectively the direct condition with an `N/A` placeholder.

The +12-point gap over direct is best read as **headroom for better rationales**. Generated rationales recover about 22% of it (2.63 / 12.05).

### 5.6 Threats to validity

- **One model, one prompt wording, greedy decoding.** No variation across prompts or samples was measured, so the CIs reflect sampling of *items* only. The option rejections in §5.2 depend in particular on the stage-2 wording ("Using the solution above…").
- **Unparsed outputs count as errors.** §3.6 shows that this is substantively right for two-stage. For direct, the 15 invalid letters are also genuine errors.
- **Possible training contamination.** ScienceQA is public and old. The 85% zero-shot direct accuracy may partly reflect exposure during Qwen2.5-VL training. This affects absolute numbers more than paired differences.
- **Image-only subset.** Results do not cover text-only ScienceQA items.
- **Many comparisons, no correction.** The main result (two-stage vs direct, p = 2.8e-4) survives any reasonable correction. The CoT vs two-stage result (p = 0.016) would not survive a strict Bonferroni correction over the 7 tests reported, so treat it as strong but not definitive.
- **Group-level effects** (§3.2) were not tested for significance.

---

## 6. Reproducing

```bash
python setup.py --skip-install                    # on a fresh Colab runtime, after the first setup
python E3_1_mcot_baseline.py --set E3_1.run_name=full_img        # image_only: true; generation.answer.max_new_tokens: 16
python E3_1_mcot_baseline.py --analyze-only --set E3_1.run_name=full_img   # rebuild summary, no GPU
# optional robustness check (§5.2): answer calls with a 64-token limit, reusing rationales and CoT
python E3_1_mcot_baseline.py --set E3_1.run_name=full_img_a64 --set E3_1.reuse.from_run=full_img
```

`--analyze-only` takes the generation settings from the run's saved `config.yaml`, so the summary reports the 16-token answer limit the run actually used. The current config's default is 64.

The output directory is `<output_root>/E3_1/full_img/` (on Drive: `/content/drive/MyDrive/MMRM/research_stage_03/E3_1/full_img/`):

| File | Content |
|---|---|
| `preds_<condition>.jsonl` | one row per item: raw output, parsed letter, rule used, correctness, groups; plus rationale fields for two-stage conditions |
| `summary.json`, `summary.md` | the tables in §3.1–3.5 |
| `parse_failures.jsonl` | every unparsed output, by condition (the input to §3.6) |
| `per_item.csv` | one row per item, prediction and correctness for each condition |
| `config.yaml`, `env.json`, `timings.json` | exact configuration, library versions, time per condition |

---

## 7. Open items and next steps

1. **Optional:** run `full_img_a64` (§5.2, §6). Expected outcome: a handful of hedged rejections become letters, and CoT stays ahead.
2. **Optional:** a stage-2 prompt that forbids "none of the options" would show how much of the CoT advantage is this single failure mode. This changes the experiment, so it should be reported as a variant, not as the baseline.
3. **E3.5 input.** 1,776 correct two-stage items are available for the 200-item faithfulness sample.
   - Stratify by `rationale_stated_answer` (527 / 1,490).
   - The 45 rejection cases are a ready-made set of rationales that contradict the options; they may be worth a separate look.
4. **E3.2.** ScienceQA is a domain where CoT helps. The harm audit needs counting and grounding tasks, where the Stage-5 literature predicts the opposite.

### Contribution to the Stage-3 gate

- *"I have my own measurement of when multimodal CoT hurts rather than helps"*: **partial.** On ScienceQA, reasoning helps overall. Splitting it into two calls hurts relative to one call (−1.59 pp, p = 0.016). Drops by task type need E3.2.
- *"I can explain why a rationale can be wrong while the answer is right"*: E3.1 shows the mechanism that E3.5 will measure. The answer step follows the rationale and ignores the image (§5.3), even to the point of rejecting all options (§5.2). Whether a correct answer rests on a faithful rationale has to be checked against the image, which is E3.5.
