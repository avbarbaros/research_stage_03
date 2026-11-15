# E3.1 — Multimodal Chain-of-Thought Baseline on ScienceQA

**Roadmap stage:** 3 — Multimodal Chain-of-Thought and Compositional Reasoning
**Model:** Qwen2.5-VL-7B-Instruct, zero-shot · **Data:** ScienceQA test, 2,017 image questions
**Reported run:** `full_img_a64` · **Date:** November 2026
**Code:** [`E3_1_mcot_baseline.py`](../E3_1_mcot_baseline.py) · [`config.yaml`](../config.yaml) (section `E3_1`)

---

## 1. Summary

**Question (roadmap E3.1).** Reproduce the two-stage "rationale, then answer" setup of Zhang et al. (2023) on ScienceQA with a modern 7B vision-language model, compare it with direct answering, and say whether the rationale helps, hurts or is neutral.

**Answer: the rationale helps.** On the 2,017 image questions of the ScienceQA test split:

| | Accuracy | vs direct |
|---|---|---|
| Direct answer | 85.42% | — |
| Two-stage (rationale → answer) | 88.15% | **+2.73 pp** (95% CI +1.34 to +4.16, p = 1.7 × 10⁻⁴) |
| Single-call chain-of-thought | 89.64% | **+4.21 pp** (p = 1.8 × 10⁻⁹) |
| Two-stage with the gold rationale | 97.47% | +12.05 pp |

**Four findings:**

1. **One call is better than two.** Single-call CoT beats two-stage by 1.49 pp (p = 0.022). Splitting reasoning from answering costs accuracy.
2. **Why the split hurts.** In 40 items the answer step reads the rationale and then **refuses every option** ("None of the options provided are correct"). A single CoT call must end by picking an option, which keeps the reasoning on the given choices.
3. **The answer step ignores the image.** Removing the image from the second call changes nothing (+0.05 pp, p = 1.0). Once the rationale is written, the answer depends on the text alone. A perception error in the rationale therefore cannot be corrected later; this motivates E3.5.
4. **The gain is real reasoning help, not answer leakage.** A quarter of rationales name the answer letter despite being told not to, but the gain over direct is the same with and without such leakage: +2.66 pp and +2.75 pp.

---

## 2. Setup

### 2.1 Data and model

| | |
|---|---|
| Dataset | ScienceQA (Lu et al., NeurIPS 2022), HF `derek-thomas/ScienceQA`, `test` split |
| Items | the **2,017** test questions that have an image; all conditions see the same items |
| Composition | NAT 1,209 · SOC 764 · LAN 44 · with text context 1,257 · grades 1–6: 1,429 · grades 7–12: 588 |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, **no fine-tuning** |
| Inference | vLLM on 1 × NVIDIA L4; greedy decoding (temperature 0); image 256–1,280 visual tokens |
| Token limits | rationale 512 · single-call CoT 768 · answer calls 64 |

### 2.2 Conditions

| Condition | Calls | What the model gets | Purpose |
|---|---|---|---|
| **direct** | 1 | image + question + options → letter | baseline |
| **cot** | 1 | same, plus "think step by step, end with *The answer is (X)*" | single-call reasoning |
| **two_stage** | 2 | call 1 writes a rationale; call 2 answers given image + rationale | Zhang et al. structure |
| **two_stage_noimg** | 1 | call 2 of two_stage **without the image** (same rationales) | does call 2 use the image? |
| **oracle** | 1 | call 2 with ScienceQA's **gold solution** as the rationale | ceiling for rationale quality |

**Difference from the original paper.** Zhang et al. *fine-tune* a model of under 1B parameters on ScienceQA rationales. Here a 7B instruction-tuned model is used *zero-shot* with the same two-stage structure. This tests whether the structure helps a modern model; the paper's numbers are not directly comparable. Exact prompts are in Appendix A.

### 2.3 How answers are scored

- **Answer extraction.** A letter is extracted with fixed rules (Appendix B). An output with no valid option letter is scored **wrong**; §4 shows what those outputs are.
- **Comparisons.** Every comparison is **paired**: the same items under two conditions.
  - Significance: exact McNemar test.
  - Confidence interval of the difference: paired bootstrap, 2,000 resamples.
  - Per-condition interval: Wilson 95% CI.
- **Verdict.** "Helps" or "hurts" when p < 0.05; otherwise "neutral".

---

## 3. Results

### 3.1 Accuracy

| Condition | Correct / 2,017 | Accuracy (95% CI) | Δ vs direct (95% CI) | Gains / losses vs direct | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| direct | 1,723 | **85.42%** (83.82–86.90) | — | — | — | baseline |
| cot | 1,808 | **89.64%** (88.23–90.89) | +4.21 (+2.88 to +5.55) | 143 / 58 | 1.8e-09 | helps |
| two_stage | 1,778 | **88.15%** (86.67–89.49) | +2.73 (+1.34 to +4.16) | 132 / 77 | 1.7e-04 | helps |
| two_stage_noimg | 1,777 | **88.10%** (86.62–89.44) | +2.68 (+1.19 to +4.17) | 139 / 85 | 3.8e-04 | helps |
| oracle | 1,966 | **97.47%** (96.69–98.07) | +12.05 (+10.66 to +13.49) | 244 / 1 | 8.7e-72 | helps |

*Gains* are items direct got wrong and the condition got right; *losses* are the reverse.

### 3.2 Direct comparisons between the reasoning conditions

| Comparison | Δ (pp) | 95% CI | Gains / losses | McNemar p | Reading |
|---|---|---|---|---|---|
| two_stage **vs** cot | **−1.49** | −2.63 to −0.25 | 66 / 96 | 0.022 | one call beats two |
| two_stage **vs** two_stage_noimg | +0.05 | −0.50 to +0.64 | 17 / 16 | 1.00 | the answer step does not use the image |
| oracle **vs** two_stage | +9.32 | +7.98 to +10.71 | 210 / 22 | 1.1e-39 | large headroom for better rationales |

### 3.3 Does answer leakage explain the gain?

27% of rationales (527) name the answer letter despite the instruction not to, and the answer step copies that letter every time. Here direct and two-stage are compared on the same items, split by leakage:

| Rationale | Items | direct | two_stage | Δ (pp) | Gains / losses | p |
|---|---|---|---|---|---|---|
| names a letter | 527 | 94.31% | 96.96% | **+2.66** | 18 / 4 | 0.004 |
| no letter | 1,490 | 82.28% | 85.03% | **+2.75** | 114 / 73 | 0.003 |

The gain is the same in both groups, so it is not produced by leakage. Leaked letters occur mostly on easy items (direct is already at 94%). They are correct 97% of the time, which fits leakage being a sign of the model's confidence.

### 3.4 Results by ScienceQA category (accuracy %, Δ vs direct in pp)

| Condition | Natural sci. (1,209) | Social sci. (764) | Language (44) | With text context (1,257) | Grades 1–6 (1,429) | Grades 7–12 (588) |
|---|---|---|---|---|---|---|
| direct | 83.46 | 87.96 | 95.45 | 80.43 | 89.99 | 74.32 |
| cot | 88.25 (+4.79) | 91.36 (+3.40) | 97.73 (+2.28) | 87.91 (+7.48) | 91.60 (+1.61) | 84.86 (**+10.54**) |
| two_stage | 88.92 (+5.46) | 86.78 (−1.18) | 90.91 (−4.54) | 86.24 (+5.81) | 90.69 (+0.70) | 81.97 (+7.65) |
| two_stage_noimg | 88.75 (+5.29) | 86.91 (−1.05) | 90.91 (−4.54) | 86.32 (+5.89) | 90.83 (+0.84) | 81.46 (+7.14) |
| oracle | 97.68 (+14.22) | 96.99 (+9.03) | 100.00 (+4.55) | 96.50 (+16.07) | 99.23 (+9.24) | 93.20 (+18.88) |

- Reasoning helps most on the **hardest items**: grades 7–12 rise by 10.5 pp with CoT.
- Two-stage is slightly below direct in social and language science. These drops were not tested for significance; the language drop is 2 of 44 items.

---

## 4. Failure analysis: outputs with no valid answer

Every output without a valid option letter was read and assigned to one category.

| Category | Example | direct | cot | two_stage | two_stage_noimg | oracle |
|---|---|---|---|---|---|---|
| **Rejects all options** | "None of the options provided are correct based on the items listed." | — | — | **38** | 38 | — |
| Answers `N/A` | "N/A" | — | — | 2 | 2 | — |
| Letter that is not an option | "F" or "C" on a 2-option question | 14 | 1 | 5 | 5 | 1 |
| Free text, no option | "The mammal in the image is the sea otter." | — | — | 2 | — | — |
| Two letters | "A, D" | — | — | — | 1 | — |
| **Total** (share of 2,017) | | 14 (0.7%) | 1 (0.05%) | **47 (2.3%)** | 46 (2.3%) | 1 (0.05%) |

**Main point.** Direct and CoT fail this way rarely. Two-stage does so in 2.3% of items, and **40 of its 47 failures are refusals** (38 rejections + 2 `N/A`): the answer step concludes, from the rationale, that no option is correct. None of these outputs hit the token limit (they all ended on their own), so this is the model's decision, not truncation.

Judging from the output text, the refusals come mainly from two kinds of questions:
- **"What can X and Y trade?"** questions about two lunch-item lists. The model concludes no listed trade works (e.g. "the correct trade would be Rosa trading her water for Isabella's water").
- **"Which animal has the same adaptation as the one shown?"** questions. The rationale identifies the pictured animal (e.g. a leopard, a bearded dragon), and the answer step then looks for *that* animal among the options instead of an animal with the same adaptation.

The question topic is not stored per item, so these two groups are a qualitative reading, not a count.

---

## 5. Interpretation

1. **Reasoning helps on ScienceQA, and one call is better than two.** Both CoT and two-stage beat direct answering. CoT is better by 1.5 pp, and §4 accounts for most of that gap. Writing the rationale separately lets it drift away from the options, and the answer step then follows the rationale rather than the question. In the single CoT call, the final "The answer is (X)" is part of the same generation and acts as a constraint.

2. **The answer step uses the rationale, not the image.** Removing the image from the answer call changes the result on only 33 items, in both directions (17 / 16). In this pipeline, what the model *saw* is decided once, in the rationale. Whether rationales are faithful to the image is therefore the key property, which is exactly what E3.5 measures.

3. **The rationale's benefit is genuine.** The gain over direct does not depend on the rationale giving away the answer (§3.3).

4. **Better rationales would help a lot.** A gold rationale lifts accuracy to 97.5%. Generated rationales capture about a fifth of the headroom over direct (2.73 of 12.05 pp). This ceiling is inflated, because many gold solutions state the answer in words (e.g. "The colony is New Hampshire."), and 9% of items have no gold solution.

5. **ScienceQA is a case where reasoning is expected to help.** The Stage-5 literature reports that CoT helps on science and maths but hurts on counting and grounding. E3.1 confirms the first half. E3.2 tests the second.

---

## 6. Robustness: answer-token limit

The first run (`full_img`) capped answer calls at 16 tokens, and 2.2% of two-stage answers were cut off. The reported run (`full_img_a64`) repeats all answer calls with a 64-token cap. It reuses the same rationales and copies the CoT predictions, since CoT is not affected by the answer cap.

| | `full_img` (16 tokens) | `full_img_a64` (64 tokens, reported) |
|---|---|---|
| direct | 85.42% | 85.42% |
| cot | 89.64% | 89.64% (same predictions) |
| two_stage | 88.05% | 88.15% |
| two_stage_noimg | 87.80% | 88.10% |
| oracle | 97.47% | 97.47% |
| two_stage: no valid answer / cut off | 55 / 44 | 47 / 3 |
| two_stage vs direct | +2.63 pp, p = 2.8e-4 | +2.73 pp, p = 1.7e-4 |
| two_stage vs cot | −1.59 pp, p = 0.016 | −1.49 pp, p = 0.022 |
| two_stage vs two_stage_noimg | +0.25 pp, p = 0.49 | +0.05 pp, p = 1.00 |

The longer limit removed almost all truncation but changed accuracy by only 2 items. The cut-off outputs were refusals in progress, not answers about to be given. **All conclusions are unchanged.**

Direct was regenerated in the second run. Its accuracy is identical, but a few individual answers differ (one fewer invalid letter; gains/losses vs CoT 144/59 → 143/58). Batched greedy decoding in vLLM is not exactly reproducible at the level of single items. This matters for item-level claims, not for the aggregate results.

---

## 7. Limitations

- **One model, one prompt wording, greedy decoding.** The confidence intervals cover item sampling only, not prompt or sampling variation. The refusal behaviour (§4) may depend on the answer-step wording ("Using the solution above…").
- **Multiple comparisons.** The main result (two-stage vs direct, p = 1.7e-4) survives a Bonferroni correction over the seven reported tests (threshold ≈ 0.007). The CoT-vs-two-stage result (p = 0.022) does not; treat it as strong but not definitive. Its mechanism (§4) is, however, directly observed.
- **Possible training contamination.** ScienceQA is public and widely used. The 85% zero-shot direct accuracy may partly reflect training exposure. This affects absolute levels more than paired differences.
- **Image questions only.** Text-only ScienceQA items were not evaluated.
- **Category effects** (§3.4) were not tested for significance.

---

## 8. Next steps

| Item | Purpose |
|---|---|
| **E3.2** CoT harm audit (implemented) | Find where CoT *hurts*: counting and visual grounding, vs math and science |
| **E3.5** faithfulness probe | 1,778 correct two-stage answers are available. Sample 200, stratified by letter leakage (527 / 1,490). The 40 refusals are a ready-made set of rationales that contradict the options. |
| Optional: stage-2 prompt that forbids "none of the options" | Measures how much of the CoT advantage is due to refusals alone. Report it as a variant, not the baseline. |

**Contribution to the Stage-3 gate**

| Gate item | Status from E3.1 |
|---|---|
| "I have my own measurement of when multimodal CoT hurts rather than helps" | **Partial.** On ScienceQA, reasoning helps; splitting it into two calls costs 1.5 pp. Task-type harm needs E3.2. |
| "I can explain why a rationale can be wrong while the answer is right" | **Mechanism shown, measurement pending.** The answer step follows the rationale and ignores the image (§3.2, §4). E3.5 measures how often a correct answer rests on an unfaithful rationale. |

---

## Appendix A — Prompts (verbatim)

`{options}` is rendered as `(A) … (B) …`; `{context}` is the item's hint text, or `N/A`. The lecture field is not used.

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

**rationale** (two_stage, call 1)
```
Question: {question}
Context: {context}
Options: {options}
Write a short step-by-step solution that explains how to answer the question, based on the image and the context. Write only the reasoning; do not state which option letter is correct.
```

**answer_with_rationale** (call 2 of two_stage, two_stage_noimg, oracle)
```
Question: {question}
Context: {context}
Options: {options}
Solution: {rationale}
Using the solution above, answer with the option's letter from the given choices directly.
```

## Appendix B — Answer extraction and run details

**Extraction rules** ([`common/parsing.py`](../common/parsing.py)), tried in order. Only letters that are valid options for the item are accepted.
1. Explicit phrase: "The answer is (B)", "Answer: B" (the last occurrence wins).
2. The output is a single letter.
3. The output starts with a letter: "B) …".
4. The output contains exactly one option's text.

Which rule matched (reported run):

| Condition | Single letter | Starts with letter | "The answer is" | Option text | None |
|---|---|---|---|---|---|
| direct | 2,003 | — | — | — | 14 |
| cot | — | — | 1,983 | 33 | 1 |
| two_stage | 1,896 | 62 | — | 12 | 47 |
| two_stage_noimg | 1,905 | 50 | 2 | 14 | 46 |
| oracle | 2,015 | 1 | — | — | 1 |

**Other run facts.**
- Rationales average 189 tokens; 0.2% hit the 512-token limit; none are empty.
- CoT outputs average 192 tokens; none hit the limit.
- 181 items (9%) have an empty gold solution; the oracle receives `N/A` for them.
- Library versions and timings are in `env.json` and `timings.json` in each run directory.

**Reproducing.**
```bash
# run 1: everything, 16-token answer calls
python E3_1_mcot_baseline.py --set E3_1.run_name=full_img --set E3_1.generation.answer.max_new_tokens=16
# run 2 (reported): answer calls again with 64 tokens, reusing run 1's rationales and CoT predictions
python E3_1_mcot_baseline.py --set E3_1.run_name=full_img_a64 --set E3_1.reuse.from_run=full_img
# rebuild tables without a GPU
python E3_1_mcot_baseline.py --analyze-only --set E3_1.run_name=full_img_a64
```

Run directories are under `/content/drive/MyDrive/MMRM/research_stage_03/E3_1/`. `full_img_a64` is self-contained: its `preds_two_stage.jsonl` includes the rationales. `full_img` is kept for the robustness comparison in §6.

| File | Content |
|---|---|
| `preds_<condition>.jsonl` | one row per item: raw output, extracted letter, rule, correctness; rationale fields for two-stage conditions |
| `summary.json`, `summary.md` | the tables of §3 |
| `parse_failures.jsonl` | every output without a valid answer (the input to §4) |
| `per_item.csv` | correctness of every condition per item |
| `config.yaml`, `env.json`, `timings.json` | exact configuration, library versions, run times |
