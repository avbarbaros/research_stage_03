# E3.2 — Chain-of-Thought Harm Audit Across Four Task Families

**Roadmap stage:** 3 — Multimodal Chain-of-Thought and Compositional Reasoning
**Model:** Qwen2.5-VL-7B-Instruct, zero-shot · **Families:** math, science, object counting, visual grounding
**Reported run:** `full` · **Date:** November 2026
**Code:** [`E3_2_cot_harm_audit.py`](../E3_2_cot_harm_audit.py) · [`common/tasks.py`](../common/tasks.py) · [`config.yaml`](../config.yaml) (section `E3_2`)

---

## 1. Summary

**Question (roadmap E3.2).** With the same model throughout, compare direct prompting with chain-of-thought (CoT) prompting on four task families: math, science, object counting and visual grounding. Show at least one family where CoT reduces accuracy, and quantify the drop.

**Answer: CoT helps on science, does nothing on math, and hurts on counting and grounding.**

| Family (dataset, n) | Direct | CoT | Δ (95% CI) | McNemar p | Verdict |
|---|---|---|---|---|---|
| Science (ScienceQA, 1,000) | 86.10% | 91.20% | **+5.10 pp** (+3.00 to +7.20) | 2.7e-06 | **helps** |
| Math (MathVista, 1,000) | 69.40% | 70.60% | +1.20 pp (−1.80 to +3.90) | 0.45 | neutral |
| Counting (CountBenchQA, 491) | 86.56% | 75.97% | **−10.59 pp** (−14.26 to −7.13) | 2.1e-08 | **hurts** |
| Grounding (RefCOCO val, 991) | 89.91% | 86.78% | **−3.13 pp** (−4.74 to −1.41) | 3.7e-04 | **hurts** |

**The success criterion is met.** CoT reduces accuracy significantly in two of the four families. All three significant results survive a Bonferroni correction over the four tests (threshold 0.0125).

**Findings:**

1. **The harm grows with perceptual load.** On counting, CoT costs 3.6 pp for 2–4 objects, 11.7 pp for 5–7 and 16.4 pp for 8–10. On grounding it costs 0.5 pp for large objects, 3.8 pp for medium and 27.3 pp for small ones.
2. **In counting, the verbal count loses track.** In the CoT losses we read, the model enumerates objects one by one in text and usually ends up one or two short. Direct answering, which estimates the count in one step, is more accurate for 5–10 objects. In a few cases the reasoning reinterprets the question ("the same man in six poses" → 1).
3. **In grounding, CoT picks the wrong object, loosens boxes, or refuses.** In 1.6% of items the reasoning concludes that no object matches the description and returns no box ("there is no red bus; the bus is pink"). Direct answering resolves the same imperfect descriptions pragmatically.
4. **Reasoning helps where the answer needs knowledge or multi-step inference.** The science gain comes from natural science (+6.7 pp). On math, CoT gains on word problems (+7.0 pp, p = 0.06) and loses on visual-question items (−5.6 pp, p = 0.05), so the two cancel out.
5. **The harm is not a formatting artefact.** In counting, every CoT loss contains a valid, readable answer. In grounding, only 9 of 52 losses have no box, and those 9 are explicit refusals.

---

## 2. Setup

### 2.1 Task families

| Family | Dataset (HF id, split) | Items | Answer type | Scoring | Subgroups |
|---|---|---|---|---|---|
| Math | MathVista `AI4Math/MathVista`, testmini | all 1,000 | multiple choice, or integer / float / list | option letter; numbers compared after rounding to the item's precision | MathVista task type |
| Science | ScienceQA `derek-thomas/ScienceQA`, test | 1,000 sampled (seed 42) from the 2,017 image questions | multiple choice | option letter | subject (NAT / SOC / LAN) |
| Counting | CountBenchQA `vikhyatk/CountBenchQA`, test | all 491 | integer (2–10) | exact match | count 2–4 / 5–7 / 8–10 |
| Grounding | RefCOCO `lmms-lab-encoder/RefCOCO`, val | 1,000 rows sampled (seed 42) → **991 distinct objects** | bounding box | correct if IoU with the gold box ≥ 0.5 (standard Acc@0.5); mean IoU also reported | box area as a share of the image: small < 5%, medium < 20%, large ≥ 20% |

**Dataset notes:**
- **Grounding count.** The RefCOCO val split contains 395 objects stored twice as identical rows. 9 of them were drawn twice in the sample, so 1,000 sampled rows give 991 distinct objects.
- **Grounding query.** The query is the object's first referring expression.
- **Counting.** CountBenchQA's caption field usually states the count ("three light bulbs"), so it is never shown to the model.

### 2.2 Model and conditions

| | |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, no fine-tuning |
| Inference | vLLM on 1 × NVIDIA L4; greedy decoding (temperature 0); image 256–1,280 visual tokens |
| **direct** | question (+ options) → "Answer with the option's letter / with an integer only / output its bbox in JSON"; max 128 new tokens |
| **cot** | the same prompt, plus "Think step by step", ending with "The answer is (X)" (or the JSON box for grounding); max 1,024 new tokens |

The two prompts differ only in the reasoning instruction and the final-answer line. Exact prompts are in Appendix A.

### 2.3 Scoring and statistics

- **Answer extraction.** Answers are extracted with fixed rules (Appendix B). An output with no valid answer is scored wrong.
- **Reasoning check.** For every output we also record whether it contains reasoning: at least 15 words of prose outside JSON or code. This checks that the CoT prompt actually produced reasoning and that direct did not.
- **Refusals.** Unanswered outputs that explicitly decline ("no object matches", "none of the options", an empty box list) are counted as refusals.
- **Statistics.** Same items under both prompts:
  - Significance: exact McNemar test.
  - CI of the difference: paired bootstrap, 2,000 resamples.
  - Per-condition interval: Wilson 95% CI.
  - Verdict: "helps" or "hurts" when p < 0.05.

---

## 3. Results

### 3.1 Direct vs CoT

| Family | n | Direct (95% CI) | CoT (95% CI) | Δ pp (95% CI) | Gains / losses | p | Verdict |
|---|---|---|---|---|---|---|---|
| Science | 1,000 | 86.10 (83.82–88.11) | 91.20 (89.28–92.80) | +5.10 (+3.00 to +7.20) | 84 / 33 | 2.7e-06 | helps |
| Math | 1,000 | 69.40 (66.47–72.18) | 70.60 (67.70–73.34) | +1.20 (−1.80 to +3.90) | 110 / 98 | 0.45 | neutral |
| Counting | 491 | 86.56 (83.26–89.29) | 75.97 (72.00–79.54) | −10.59 (−14.26 to −7.13) | 18 / 70 | 2.1e-08 | hurts |
| Grounding | 991 | 89.91 (87.88–91.63) | 86.78 (84.53–88.75) | −3.13 (−4.74 to −1.41) | 21 / 52 | 3.7e-04 | hurts |

*Gains* are items direct got wrong and CoT got right; *losses* are the reverse. For grounding, mean IoU is 0.826 (direct) against 0.802 (CoT).

### 3.2 Where within each family

**Counting: the harm grows with the number of objects**

| Count | n | Direct | CoT | Δ pp | Gains / losses | p |
|---|---|---|---|---|---|---|
| 2–4 | 166 | 96.39 | 92.77 | −3.61 | 1 / 7 | 0.070 |
| 5–7 | 154 | 81.17 | 69.48 | **−11.69** | 11 / 29 | 0.006 |
| 8–10 | 171 | 81.87 | 65.50 | **−16.37** | 6 / 34 | 8.4e-06 |

**Grounding: the harm grows as the object gets smaller**

| Object size | n | Direct | CoT | Δ pp | Gains / losses | p |
|---|---|---|---|---|---|---|
| large (≥ 20% of image) | 368 | 92.39 | 91.85 | −0.54 | 6 / 8 | 0.79 |
| medium (5–20%) | 601 | 88.52 | 84.69 | **−3.83** | 15 / 38 | 0.002 |
| small (< 5%) | 22 | 86.36 | 59.09 | **−27.27** | 0 / 6 | 0.031 |

The small-object group has only 22 items; RefCOCO objects are mostly medium or large.

**Math: gains and losses cancel out**

| MathVista task | n | Direct | CoT | Δ pp | Gains / losses | p |
|---|---|---|---|---|---|---|
| math word problem | 186 | 73.66 | 80.65 | +6.99 | 27 / 14 | 0.060 |
| geometry problem solving | 208 | 65.38 | 69.23 | +3.85 | 40 / 32 | 0.41 |
| textbook question answering | 158 | 67.72 | 68.99 | +1.27 | 11 / 9 | 0.82 |
| figure question answering | 269 | 77.32 | 76.95 | −0.37 | 26 / 27 | 1.00 |
| visual question answering | 179 | 59.22 | 53.63 | −5.59 | 6 / 16 | 0.053 |

**Science: the gain comes from natural science**

| Subject | n | Direct | CoT | Δ pp | Gains / losses | p |
|---|---|---|---|---|---|---|
| natural science | 615 | 84.72 | 91.38 | **+6.67** | 63 / 22 | 9.8e-06 |
| social science | 366 | 87.98 | 90.71 | +2.73 | 21 / 11 | 0.11 |
| language science | 19 | 94.74 | 94.74 | 0.00 | 0 / 0 | 1.00 |

Subgroup p-values are not corrected for multiple comparisons.

### 3.3 Did the prompts do what they should?

| Family | Direct outputs with reasoning | CoT outputs with reasoning | No answer (direct / CoT) | Refusals (CoT) | CoT cut off at 1,024 tokens | Mean CoT length |
|---|---|---|---|---|---|---|
| Math | 1.9% | 100% | 0.4% / 0.1% | 0 | 0.9% | 270 tokens |
| Science | 0% | 100% | 0.1% / 0% | 0 | 0 | 192 tokens |
| Counting | 0% | 100% | 0% / 0% | 0 | 0 | 125 tokens |
| Grounding | 0% | 99.6% | 0% / 1.6% | **1.6% (16)** | 0 | 217 tokens |

- **The CoT prompt produced reasoning in essentially every case.** Direct produced none, except for 1.9% of math items (mostly geometry), where the model reasoned despite "answer directly". Four of those hit the 128-token limit before answering.
- **Answer extraction is not a confound.** Outside grounding, at most 0.4% of outputs have no answer in any family. Every grounding CoT output without a box is an explicit refusal.
- **Some CoT chains loop.** 0.9% of math CoT outputs (9 items) were cut off at 1,024 tokens. These were chains repeating the same derivation; 2 of them are among the CoT losses.

---

## 4. What goes wrong under CoT (failure analysis)

For each family the run saves 25 CoT losses (items direct got right and CoT got wrong). They are the first 25 by item ID, which is effectively arbitrary but **not a random sample**. We read them all. Counts below refer to these 25 per family, plus the 16 grounding refusals.

### 4.1 Counting: enumeration loses track

| Pattern | Of 25 | Example |
|---|---|---|
| **Under-counts by enumerating** (usually by 1, sometimes 2) | 17 | 7 apples → lists "one in the foreground… a sixth behind the fifth" → "The answer is 6." |
| Over-counts (double-counts or assumes a grid) | 4 | 9 silhouettes → "Top row: 3… fourth row: 3" → 12 |
| **Reinterprets the question** as counting distinct entities | 4 | 6 male nurses → "a single male nurse character depicted in six different poses" → 1 |

The reasoning also contains internal inconsistencies: "1 + 2 + 3 + … + 9 = 45. The answer is 9."

**Reading:** direct answering estimates the count in one step and is accurate up to about 10 objects (82–96%). Writing the count out item by item turns it into a serial bookkeeping task in text, and items get skipped. Error grows with the number of items, which is consistent with §3.2.

### 4.2 Grounding: wrong referent, loose boxes, and refusals

| Pattern | Example |
|---|---|
| **Reasons its way to the wrong object** | "silver car in front" → picks the car in front of the police officer (IoU 0); "top right guy" → describes the top-right panel of a collage but boxes a different region (IoU 0) |
| **Box grows to cover the whole group or scene** | "olive on pizza in front" → boxes the whole pizza (IoU 0.25); "left side just below middle of pizza" → the entire pizza (IoU 0.28) |
| **Refuses: "no object matches"** (16 items, 1.6%; 9 of them are losses) | "red bus" → "the bus is pink… there is no red bus" `[]`; "zebra on the right" → "the image contains giraffes, not zebras" `[]`; "black guy in" (second annotation: "umpire") → `[]` |

The refusals are notable. **RefCOCO expressions are often imperfect**: truncated, colour-shifted, or written from the annotator's viewpoint. Direct answering resolves them pragmatically to the most plausible object. CoT checks the description literally, finds a mismatch, and returns nothing. In 9 of the 16 refusals, direct had found the right object.

The same mechanism appeared in E3.1: in the two-stage pipeline, the answer step rejected every option after reading the rationale. **Explicit reasoning makes the model more willing to say "nothing matches".**

### 4.3 Math: the two effects cancel out

| Pattern | Example |
|---|---|
| Perception errors inside the chain (miscounting, misreading charts) | "How many baseballs?" (20) → "6 + 6 + 6 + 6 = 24"; bar-chart questions read with wrong values |
| Commonsense estimates replacing a direct guess | "how many people can commute on this vehicle?" (50) → "40 seats + 30 standing = 70" |
| Derivation fails, then guesses among the options | "x² = −189… the closest option is (C)" |
| Final letter contradicts the chain's own conclusion | concludes "Color C has the smallest saturation" → "The answer is (A)" |
| Rounding convention | 58.09 and 44 → average 51.045 → "51.05" (gold 51.04) |
| Looping until cut off | the same proportion derivation repeated until 1,024 tokens |

CoT helps on word problems, where an explicit arithmetic chain is useful. It hurts on visual-question and figure items, where the answer depends on reading the image. These two effects cancel out in the overall score.

### 4.4 Science: net gain, with the same perception failures in the losses

The 33 science losses (against 84 gains) are mostly:
- **Map reading:** "which country / ocean / state is highlighted", 8 of 25.
- **Reasoning over a diagram:** magnet poles, heat-transfer tables.
- **Final letter contradicting the stated conclusion:** e.g. concludes "Solution B has a higher concentration" → "The answer is (A)".

---

## 5. Interpretation

1. **Whether CoT helps or hurts depends on the task family, in the direction the Stage-5 literature reports.** CoT helps where the answer needs knowledge or multi-step inference (science, math word problems). It hurts where the answer depends on precise perception (counting, grounding of small objects).
2. **The harm comes from perception, not from formatting.** Every counting loss is a well-formed numeric answer; the reasoning itself is wrong. The harm grows with perceptual load: more objects to count, smaller objects to localise.
3. **Explicit reasoning replaces a strong one-step perceptual estimate with a weaker step-by-step one.** For counting, enumerating objects in text is less accurate than the model's direct estimate. For grounding, describing the scene in words and then choosing an object loses spatial precision (looser boxes, wrong referent). This fits the perception-bottleneck view in Stage 5: the visual information is available in one step but degrades once it is routed through language.
4. **Literal checking produces refusals.** As in E3.1, reasoning makes the model compare the request against what it sees and reject imperfect matches. This could be useful (detecting bad questions) or harmful (refusing answerable ones). On RefCOCO it is harmful, because the annotations are noisy and the target object exists.
5. **Consequence for thesis directions.** A model or controller that decides *per input* whether to reason (roadmap directions 5 and 6) has a measurable target here. Choosing per family (direct for counting and grounding, CoT for science) would avoid both significant harms and keep the science gain.

---

## 6. Validity checks

| Check | Result |
|---|---|
| Direct grounding accuracy matches the published figure | 89.91% on RefCOCO val, against roughly 90% reported for this model. The box-coordinate conversion from the model's resized image back to the original image is therefore correct. |
| Direct MathVista accuracy is plausible | 69.40% on testmini, close to the published figure (about 68%), despite rule-based answer extraction instead of the official GPT-based extraction |
| Science is consistent with E3.1 | Direct 86.10% here vs 85.42% on all 2,017 items in E3.1; CoT helps in both |
| CoT actually reasons, direct does not | 99.6–100% vs 0–1.9% (§3.3) |
| Harm is not caused by answer extraction | Counting: 0 of 70 losses lack an answer. Grounding: 9 of 52, all explicit refusals. |

**Pilot history (disclosed for transparency).** Two 30-item pilots preceded the full run.
- In the first pilot, the grounding CoT prompt ("first think step by step…") was ignored: the model output the bare JSON box, identical to direct. The grounding CoT prompt was therefore strengthened to ask for written reasoning before the box (Appendix A). The second pilot confirmed reasoning in 100% of outputs.
- The pilots also showed that math CoT often writes its final answer as `\boxed{B}`; the answer extraction was extended to read that format.

Neither pilot contributes to the reported numbers.

---

## 7. Limitations

- **One model, one prompt pair, greedy decoding.** The size of the harm may depend on the CoT wording. In particular, the grounding CoT prompt is more directive than the others ("list the candidate objects…").
- **Failure categories (§4) come from 25 non-random examples per family.** They show what happens, not how often. A systematic count would need a random sample of losses.
- **Small subgroups.** Grounding "small" has 22 items, and science "language" has 19.
- **Rule-based answer extraction** may disagree with official evaluators on edge cases, for example rounding in MathVista free-form items.
- **Possible training contamination** of public benchmarks affects absolute levels more than the paired direct-vs-CoT differences.

---

## 8. Next steps

| Item | Purpose |
|---|---|
| **E3.3** visual-program agent (ViperGPT-style) | Counting and grounding are where CoT fails. A detector-backed program (count detections, return a detector box) is a natural alternative. Measure it on CountBench and the spatial subset. |
| **E3.4** visual actions (crop / re-observe) | Small-object grounding (−27 pp under CoT) is the obvious test case for crop-and-requery. |
| **E3.5** faithfulness probe | Counting chains that list objects that aren't there (e.g. "a sixth apple behind the fifth") are directly checkable against the image. |
| Optional: random sample of 100 counting and grounding losses | Turns the §4 patterns into rates |

**Contribution to the Stage-3 gate**

| Gate item | Status |
|---|---|
| "I have my own measurement of when multimodal CoT hurts rather than helps" | **Met.** CoT hurts counting by 10.6 pp and grounding by 3.1 pp, helps science by 5.1 pp, and is neutral on math. The harm grows with perceptual load. |
| "I can explain why a rationale can be wrong while the answer is right" | Supported by E3.1 and E3.2 together. Correct direct answers next to wrong CoT chains on the same item show that the visual information was available but was lost in the verbal chain. |

---

## Appendix A — Prompts (verbatim)

`{options}` is rendered as `(A) … (B) …`. `{format}` is, for example, "an integer" or "a floating-point number with 1 decimal place". For ScienceQA, `{question}` includes a `Context:` line when the item has one. For grounding, `{question}` is the referring expression.

**Multiple choice (ScienceQA, MathVista multiple-choice)**
```
direct:
{question}
Options: {options}
Answer with the option's letter from the given choices directly.

cot:
{question}
Options: {options}
Think step by step. Then finish with a final line of the form "The answer is (X)." where X is the option letter.
```

**Open answer (MathVista free-form, CountBenchQA)**
```
direct:
{question}
Answer with {format} only.

cot:
{question}
Think step by step. Then finish with a final line of the form "The answer is N." where N is {format}.
```

**Grounding (RefCOCO)**: the direct prompt is Qwen2.5-VL's native grounding phrasing.
```
direct:
Locate "{question}" in the image and output its bbox coordinates in JSON format.

cot:
Locate "{question}" in the image.
Before giving the box, reason step by step in plain sentences: list the objects in the image that could match the description, and explain which one matches its category, attributes and position relative to other objects.
Only after this reasoning, output the bbox coordinates of the matching object in JSON format.
```

## Appendix B — Answer extraction

Implemented in [`common/parsing.py`](../common/parsing.py) and [`common/tasks.py`](../common/tasks.py). Outputs are re-scored with the final extraction rules at analysis time.

| Answer type | Rules (in order) |
|---|---|
| Multiple choice | "The answer is (X)" / "Answer: X" (last occurrence) → `\boxed{X}` or `\boxed{option text}` → a single letter → a leading "X)" → exactly one option's text. Only letters that are valid options for the item are accepted. |
| Number | the number inside "The answer is …" or `\boxed{…}` (fractions and `\frac{a}{b}` supported) → otherwise the last number in the output → otherwise number words ("five"). Compared with the gold answer after rounding to the item's precision. |
| Bounding box | the last `"bbox_2d": [x1, y1, x2, y2]` → otherwise the last 4-number list → otherwise `(x1, y1), (x2, y2)`. Coordinates are mapped from the model's resized image back to the original image (Qwen2.5-VL's resize rule), then IoU is computed against the gold box. An empty list `[]` counts as a refusal. |

## Appendix C — Reproducing

```bash
python setup.py --skip-install --skip-model          # datasets: ScienceQA, MathVista, CountBenchQA, RefCOCO val
python E3_2_cot_harm_audit.py --set E3_2.run_name=full
python E3_2_cot_harm_audit.py --analyze-only --set E3_2.run_name=full   # rebuild tables, no GPU
```

The run directory is `/content/drive/MyDrive/MMRM/research_stage_03/E3_2/full/`:

| File | Content |
|---|---|
| `preds_<family>_<condition>.jsonl` | one row per item: raw output, extracted answer, extraction rule, correctness, IoU (grounding), reasoning-word count; includes everything needed to re-score without the dataset |
| `summary.json`, `summary.md` | the tables of §3 |
| `cot_losses_<family>.jsonl` | the first 25 items per family that direct got right and CoT got wrong (the input to §4) |
| `parse_failures.jsonl` | every output with no valid answer, flagged as refusal or not |
| `per_item.csv` | direct and CoT prediction and correctness per item |
| `config.yaml`, `env.json`, `timings.json` | exact configuration, library versions, run times |
