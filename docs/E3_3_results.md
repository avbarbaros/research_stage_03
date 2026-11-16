# E3.3 — Visual Programs (ViperGPT-style Agent) vs End-to-End Prompting

**Roadmap stage:** 3 — Multimodal Chain-of-Thought and Compositional Reasoning
**Model:** Qwen2.5-VL-7B-Instruct as answerer, program writer and crop reader · **Tools:** Grounding DINO (tiny), Depth Anything V2 (small)
**Benchmarks:** GQA testdev-balanced (1,000 questions), CV-Bench (4 tasks × 250)
**Reported run:** `full` · **Date:** November 2026
**Code:** [`E3_3_visual_programs.py`](../E3_3_visual_programs.py) · [`common/visprog.py`](../common/visprog.py) · [`prompts/E3_3_planner.txt`](../prompts/E3_3_planner.txt) · [`config.yaml`](../config.yaml) (section `E3_3`)

---

## 1. Summary

**Question (roadmap E3.3).** Build a small ViperGPT-style agent: an LLM writes Python that calls a detector, a depth estimator and a crop-and-requery function. Evaluate it on GQA and a spatial subset.

**Success criteria:**
1. The agent beats end-to-end prompting on at least one compositional subset.
2. Every failure can be inspected and debugged.

**Answer: criterion 1 is not met. Criterion 2 is met.**

| Benchmark | Direct | Agent | Agent + fallback | Δ (fallback − direct) | p | Verdict |
|---|---|---|---|---|---|---|
| GQA (1,000) | 62.70% | 51.10% | 53.00% | **−9.70 pp** (95% CI −12.50 to −6.90) | 1.7e-11 | **hurts** |
| CV-Bench (1,000) | 80.30% | 73.80% | 77.60% | −2.70 pp (−5.50 to +0.20) | 0.086 | neutral |

- **The agent loses everywhere it differs significantly from direct.** On GQA it is worse on every structural type (verify, query, choose, logical) and on spatial and non-spatial questions alike. On CV-Bench it is worse on Distance (−10.8 pp).
- **Its best case is CV-Bench Count:** 68.4% vs 63.2% (+5.2 pp, p = 0.15, not significant). Counting is the one task where a tool (detection) replaces a step the end-to-end model does poorly.
- **The loss is not caused by tuning on development items.** Excluding the 70 items inspected during prompt development gives the same result: GQA −10.1 pp, CV-Bench −3.0 pp.
- **Every failure can be traced to a specific step:**
  - a detector miss or false positive;
  - the wrong instance taken from the detector's list;
  - a crop that loses the context the question needs;
  - a bug in the generated program;
  - a fallback answer that ignores the options.
  §4 documents each with examples.

- **On CV-Bench 3D, the failing step was measured directly (§4.4).**
  - The detector proposes the highlighted object for only about 55% of objects. When it does, the agent uses it 93–95% of the time.
  - Given the gold boxes, the agent's own depth rules beat direct: **Depth 96.0% vs 87.2%**, **Distance 84.8% vs 80.4%**.

**Main finding.** Breaking a question into tool calls multiplies the places it can fail. With a modern end-to-end VLM as the baseline, those failures cost more than the decomposition gains. The one exception is counting, where the end-to-end model is itself weak.

Where the failing step can be measured (CV-Bench 3D), it is the **perception step**: finding the object the question refers to. The reasoning step (comparing depths) is not the problem. With the right objects, the tool-based answer would beat end-to-end prompting.

---

## 2. Setup

### 2.1 Benchmarks

| Benchmark | Source (HF) | Items | Scoring | Subsets |
|---|---|---|---|---|
| GQA | `lmms-lab-encoder/GQA`, testdev-balanced (12,578 questions over 398 images) | 1,000 sampled (seed 42) | normalised exact match of a short answer (standard GQA accuracy) | structural type (verify / query / choose / logical / compare); semantic type (attr / rel / obj / cat / global); spatial vs non-spatial (from the GQA program: position questions or spatial relations such as "to the left of", "behind") |
| CV-Bench | `nyu-visionx/CV-Bench`, test | 250 per task (seed 42) | multiple-choice letter | Count, Relation (2D left/right/above/below), Depth (closer to the camera), Distance (which object is closer to a reference object) |

CV-Bench marks the objects it asks about with coloured boxes drawn on the image ("the lamp (highlighted by a red box)"). This turns out to matter (§4.2).

### 2.2 Conditions

| Condition | What happens |
|---|---|
| **direct** | Image and question go to Qwen2.5-VL-7B, which answers in one short output (GQA: "Answer the question using a single word or phrase"; CV-Bench: option letter). |
| **agent** | Qwen2.5-VL-7B sees only the question (as in ViperGPT) and writes `execute_command(image)` against the API below. The program runs; its return value is the answer. A failed program counts as wrong. |
| **agent + fallback** | As agent, but a failed program (exception, timeout, no answer, or an answer that is not one of the options) is replaced by the direct answer. |

### 2.3 The program API

| Call | Backed by | What it returns |
|---|---|---|
| `find(name)`, `exists(name)`, `count(name)` | Grounding DINO tiny (box threshold 0.35, text threshold 0.25, NMS IoU 0.5, max 20) | image patches of detected instances, most confident first |
| `find_marked(name, color)` | Grounding DINO + Qwen2.5-VL on each candidate crop ("Is this X outlined by a red box?") | like `find`, with the instance confirmed as outlined in that colour first |
| `simple_query(q)`, `verify_property(name, attr)`, `best_text_match(options)` | Qwen2.5-VL on the patch's crop (10% margin) | short answer / yes-no / one of the options |
| `compute_depth()` | Depth Anything V2 small, normalised per image | median depth of the patch (0 = closest, 1 = farthest) |
| `distance(a, b)`, `distance_3d(a, b)`, box geometry, `crop`, `overlaps` | plain Python | pixel distance; 2D offset combined with depth difference |

**Execution:**
- Programs run in a restricted Python namespace: no imports and no file access.
- Each program has a 60 s timeout and at most 30 tool calls.
- Every tool call is written to a trace, with its arguments and result.
- The planner prompt (API description, rules, 10 worked examples) is in [`prompts/E3_3_planner.txt`](../prompts/E3_3_planner.txt). All examples are invented, not taken from the test sets.

### 2.4 Inference settings

| | |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, greedy decoding |
| vLLM (shared GPU with the tools) | `gpu_memory_utilization` 0.80, eager mode, `max_model_len` 4096 |
| Program writing | text only (the planner does not see the image), max 512 tokens |
| Crop queries | max 16 tokens, prompt "{question} Answer with a single word or short phrase." |
| Detector input | downscaled to at most 1,024 px on the long side |
| Hardware | 1 × NVIDIA L4 (22 GiB usable); 0.6 s per GQA item, 1.1 s per CV-Bench item for program execution |

### 2.5 Development history (disclosed)

The planner prompt and API were revised twice. Both revisions were based on failures read in small pilot runs drawn from the same item pool as the evaluation.

| Run | Items | What was learned | Change made |
|---|---|---|---|
| `check` | 30 GQA + 40 CV-Bench (seed 42) | 20 of 22 errors were CUDA out-of-memory errors in the detector. Programs invented answer lists, called `verify_property("window", "color")`, and could not express object-to-object distance. | Memory settings fixed and infrastructure errors retried; `distance_3d` added; rules against the observed misuse; answers that are not one of the options treated as failures |
| `check2` | same items | Distance programs chose the wrong instance: the detector's top instance, not the one marked by a coloured box. Pixel coordinates were invented. | `find_marked` added; rules "use it for highlighted objects" and "never write coordinates" |
| `check3` | 40 CV-Bench, **different seed (7)** | `find_marked` was used in 65% of programs, with 0% errors | none: prompt frozen |

**Sensitivity check:** the 70 items of `check` and `check2` are excluded and the comparison is repeated (§3.1). `check3` is not excluded because nothing was changed after it.

---

## 3. Results

### 3.1 Overall

| Benchmark | n | Direct (95% CI) | Agent | Agent + fallback (95% CI) | Δ agent vs direct | Δ agent + fallback vs direct | Program errors |
|---|---|---|---|---|---|---|---|
| GQA | 1,000 | 62.70 (59.66–65.64) | 51.10 | 53.00 | −11.60 pp, p = 8.9e-15 | **−9.70 pp** (−12.50 to −6.90), p = 1.7e-11 | 4.1% |
| CV-Bench | 1,000 | 80.30 (77.72–82.65) | 73.80 | 77.60 | −6.50 pp, p = 8.3e-05 | −2.70 pp (−5.50 to +0.20), p = 0.086 | 5.1% |

Wilson intervals for direct were computed from the reported proportions.

**Excluding the 70 development items:**

| Benchmark | Excluded | n | Direct | Agent + fallback | Δ, p |
|---|---|---|---|---|---|
| GQA | 30 | 970 | 62.78 | 52.68 | −10.10 pp, p = 5.5e-12 |
| CV-Bench | 40 | 960 | 80.52 | 77.50 | −3.02 pp, p = 0.057 |

### 3.2 Subsets (agent + fallback vs direct)

**GQA by structural type**

| Type | n | Direct | Agent | Agent + fallback | Δ pp | Gains / losses | p | Verdict |
|---|---|---|---|---|---|---|---|---|
| choose | 97 | 86.60 | 63.92 | 65.98 | **−20.62** | 3 / 23 | 8.8e-05 | hurts |
| verify | 190 | 86.32 | 71.05 | 72.11 | −14.21 | 11 / 38 | 1.4e-04 | hurts |
| logical | 146 | 80.14 | 67.81 | 67.81 | −12.33 | 8 / 26 | 0.003 | hurts |
| compare | 42 | 61.90 | 52.38 | 52.38 | −9.52 | 3 / 7 | 0.34 | neutral |
| query | 525 | 44.95 | 36.76 | 39.62 | −5.33 | 32 / 60 | 0.005 | hurts |

**GQA by semantic type**

| Type | n | Direct | Agent + fallback | Δ pp | Gains / losses | p | Verdict |
|---|---|---|---|---|---|---|---|
| obj | 70 | 87.14 | 72.86 | −14.29 | 5 / 15 | 0.041 | hurts |
| attr | 393 | 71.25 | 59.03 | −12.21 | 21 / 69 | 3.9e-07 | hurts |
| rel | 437 | 52.86 | 44.16 | −8.70 | 25 / 63 | 6.3e-05 | hurts |
| cat | 86 | 55.81 | 52.33 | −3.49 | 4 / 7 | 0.55 | neutral |
| global | 14 | 50.00 | 64.29 | +14.29 | 2 / 0 | 0.50 | neutral |

**GQA spatial vs non-spatial:** spatial questions (327) −8.87 pp (14 / 43, p = 1.5e-04); non-spatial questions (673) −10.10 pp (43 / 111, p = 4.1e-08). Both are hurt by similar amounts.

**CV-Bench by task**

| Task | n | Direct | Agent | Agent + fallback | Δ pp | Gains / losses | p | Verdict | Program errors |
|---|---|---|---|---|---|---|---|---|---|
| Count | 250 | 63.20 | 67.20 | 68.40 | **+5.20** | 41 / 28 | 0.15 | neutral | 4.4% |
| Depth | 250 | 87.20 | 84.80 | 84.80 | −2.40 | 19 / 25 | 0.45 | neutral | 0% |
| Relation | 250 | 90.40 | 74.80 | 87.60 | −2.80 | 13 / 20 | 0.30 | neutral | **14.8%** |
| Distance | 250 | 80.40 | 68.40 | 69.60 | **−10.80** | 28 / 55 | 0.004 | hurts | 1.2% |

Subset p-values are not corrected for multiple comparisons.

### 3.3 What the programs did

| | GQA | CV-Bench |
|---|---|---|
| Programs that failed | 4.1% (41): no return value 24, invented API arguments (TypeError) 12, out-of-memory errors 3, AttributeError 2 | 5.1% (51): answer not one of the options 51 |
| Programs using the detector | 90.8% | 100% |
| … using depth | 1.3% | 31.4% |
| … using `find_marked` | 0.3% | 62.9% |
| Query-only programs (one question to the VLM about the whole image) | 8.9% | 0% |
| At least one `find()` returned nothing | 22.7% | 28.8% |
| Mean tool calls | 2.2 | 4.7 |

**Accuracy by kind of program (programs that ran):**

| | GQA: n · direct · agent | CV-Bench: n · direct · agent |
|---|---|---|
| all programs that ran | 959 · 63.40 · 53.28 | 949 · 80.61 · 77.77 |
| detector-based programs | 870 · 63.91 · **52.87** | 949 · 80.61 · 77.77 |
| query-only programs | 89 · 58.43 · 57.30 | — |

The loss comes from the programs that actually decompose the question. Query-only programs, which amount to direct prompting with slightly different wording, perform about the same as direct.

**Infrastructure note:** 3 GQA items still record an out-of-memory error. They are counted as failed programs, so they fall back to direct; rerunning `full` would retry them. With 3 items out of 1,000, this cannot change any conclusion.

---

## 4. Failure analysis

The run saves a random sample of 25 losses and 25 wins per benchmark (agent + fallback vs direct, seed 0), plus every failed program. All were read. The patterns below are illustrated with these examples; counts refer to these samples, not to the whole run.

### 4.1 GQA: every step adds a failure point

| Failure | Example (question → what the program did) |
|---|---|
| **Answer not constrained to the choices in the question** ("X or Y?") | "What are the pants made of, cloth or leather?" → crop query "material of these pants?" → "Corduroy". Two such cases in 25 losses; this likely drives the −20.6 pp on *choose* questions. |
| **Wrong instance: the first detection is used regardless of qualifiers** | "Are the trousers blue or pink?" → a 23 × 35 px pair of trousers → "black"; "Is the thin girl to the right or left of the glasses?" → first girl, first glasses |
| **Crop loses the context the question needs** | "What is leaning against the wall?" → crop of the wall, "What is this object?" → "couch"; "Who does the boy sit beside?" → `boy.find("friend")` inside the boy's box finds the boy himself → "Man" |
| **Yes/no about a crop is less reliable** | "Does the glass on the table look clear and small?" → crop: "Is the glass clear?" → "No" (gold yes); "Is the white ground wet or dry?" → "no" (gold wet) |
| **Program logic does not match the question** | "Is the bear that is not little brown and tall?" → computes *not little and not brown and not tall*; "Is the cupboard square?" → `width == height`; "How long are the trousers?" → hard-coded "long" |
| **Detector false positive** | "Are there skateboards or cabinets?" → cabinet detected at 0.46 → "yes" (gold no) |
| **Program has no return value** (24 errors) | "Which kind of furniture is brown?" → `furniture[0].simple_query(...)` without `return`; loops that never match a condition |
| **Invented API arguments** (12 errors) | `find("device", above=keyboard)`, `find("lamp", left=book.left, …)`: the planner wants spatial filtering the API does not offer |

**Wins are partly accidental.** Of the 25 random GQA wins:
- 3 come from programs that return a fixed answer, or compare two meaningless `verify_property` calls, and happen to match the gold answer (e.g. "Is the sky clear and high?" → "yes" whenever a sky is detected).
- 7 differ from direct only in word form, which exact-match scoring rewards ("donuts" vs "donut", "racket" vs "tennis racket", "White" vs "white and green").
- The genuine decomposition wins are geometry checks ("Is the book above a bench?" → compare box centres) and attribute comparisons ("same material?" → ask about each object, compare).

### 4.2 CV-Bench: tools help counting, instance selection hurts distances

| Task | What happens |
|---|---|
| **Count** (+5.2 pp, n.s.) | Detection-based counting beats the VLM's own count on many items: 41 gains, e.g. "How many chairs?" → 5 detected (gold 5; direct said 6). Failures: amorphous objects counted as 1 (walls, fences, houses); overlapping and nested boxes inflate counts (16 cars vs 12); missed instances. A count that matches no option counts as a failed program (11 items). |
| **Relation** (14.8% program errors) | When the detector misses the reference object (heater, sconce, bowl, suitcase…), the program falls back to an unconstrained whole-image query, which answers "behind", "On the table" or "above" to a left/right question. The fallback to direct recovers these (74.8% → 87.6%). |
| **Depth** (−2.4 pp, n.s.) | When `find_marked` confirms the right instances, depth comparison works. Losses: no candidate confirmed (falls back to the first detection); the marked object is not detected at all; a large box whose median depth reflects the background (a "books" box covering a whole shelf). |
| **Distance** (−10.8 pp, hurts) | Three compounding issues: (1) `distance_3d` combines a 2D offset (fraction of the image diagonal) with relative depth, two scales that are not comparable; (2) the detector returns the same box for two categories (one box detected as both "car" and "bus"); (3) some programs still compare distance to the camera and return the reference object ("desk", "printer", "box"), which is not an option. |

In the sampled losses, the outline check (`find_marked`) sometimes answers "no" for every candidate (e.g. all three bowls in 2D-1337) and sometimes "yes" to the first candidate it is shown. §4.4 measures how much this matters. Answer: much less than the detector does.

### 4.3 Where the agent wins

- **Counting** by detection (CV-Bench Count: 41 gains vs 28 losses).
- **Explicit geometry** when both objects are detected correctly (GQA "Is X above Y?", CV-Bench Relation when the marked object is found).
- **Attribute comparisons** that ask about each object separately and compare the answers.

These are exactly the cases where a tool computes something the end-to-end model does unreliably, and where only one or two tool calls are involved.

### 4.4 Follow-up: was the highlighted object used? (measured against the gold boxes)

CV-Bench stores the boxes of the highlighted objects for its 3D tasks. [`E3_3_marked_eval.py`](../E3_3_marked_eval.py) compares them with the boxes the agent actually used, read from the tool traces of the `full` run.

- **Scope:** 500 Depth and Distance items, 1,250 highlighted objects.
- **"Used the right object"** means IoU ≥ 0.5 with the gold box.
- **No VLM is involved.** Only Depth Anything is run, for the oracle in table D.
- **Box colours are read from the image.** CV-Bench does not store boxes in colour order: Depth boxes are stored red then blue, Distance boxes blue, green, then red (the reference object last). The script therefore identifies each gold box by the colour of the outline drawn around it, which succeeded for 100% of objects.

**A. Where in the selection pipeline objects are lost**

| | Depth (500 objects) | Distance (750 objects) |
|---|---|---|
| object searched for by the program | 100% | 99.7% |
| detector returned at least one candidate | 81.0% | 83.2% |
| **gold box among the candidates** | **54.6%** | **54.3%** |
| gold box used by the program | 51.6% | 50.4% |
| used, when the gold box was a candidate | 94.5% | 92.9% |

Whenever the detector returned candidates for a highlighted object, the program checked them with `find_marked`. About 45% of highlighted objects are lost before any choice is made:
- in 17–19% of cases the detector returns nothing for the object;
- in 26–29% it returns boxes of that category, but none of them is the highlighted instance.

**B. The outline check ("Is this X outlined by a red box?")**

| | Checks | Gold instance: said yes | Other instance: said yes |
|---|---|---|---|
| Depth | 510 | 98.8% (of 259) | 33.9% (of 251) |
| Distance | 759 | 98.7% (of 381) | 40.5% (of 378) |

The check almost never rejects the right instance, but it accepts a wrong one 34–41% of the time: the suspected yes-bias. It costs relatively little, because a wrong candidate is accepted only when it is checked *before* the gold one. Given the gold box was a candidate, the right instance is still used 93–95% of the time (table A).

**C. Does using the right objects matter?**

| Items | n | Agent | Direct |
|---|---|---|---|
| Depth: every highlighted object used correctly | 65 | 96.9% | 93.8% |
| Depth: at least one wrong | 185 | 80.5% | 84.9% |
| Distance: every highlighted object used correctly | 40 | 80.0% | 87.5% |
| Distance: at least one wrong | 210 | 66.2% | 79.0% |

Only 26% (Depth) and 16% (Distance) of items have all their highlighted objects right. On those, the agent is far more accurate than on the rest. These subsets are small and not chosen at random: easier images are probably over-represented. Table D gives the controlled comparison.

**D. Oracle instances: the agent's own rules applied to the gold boxes**

| Task / rule | n | Gold boxes | Agent | Direct | Gold boxes vs agent: gains / losses, p |
|---|---|---|---|---|---|
| Depth: smaller median depth is closer | 250 | **96.0%** | 84.8% | 87.2% | 33 / 5, p = 4.3e-06 |
| Distance: `distance_3d` (the agent's rule) | 250 | **84.8%** | 68.4% | 80.4% | 57 / 16, p = 1.5e-06 |
| Distance: depth difference only | 250 | 84.4% | 68.4% | 80.4% | 70 / 30, p = 7.9e-05 |
| Distance: image-plane offset only | 250 | 81.2% | 68.4% | 80.4% | 53 / 21, p = 2.6e-04 |

How to read table D:
- **The depth tool and the agent's rules are not the problem.** Given the right objects, Depth reaches 96.0% (direct: 87.2%), and the `distance_3d` rule criticised in §4.2 reaches 84.8% (direct: 80.4%).
- **Depth carries most of the signal for Distance:** depth difference alone gives 84.4%, the image-plane offset alone 81.2%.
- **Significance against direct was not computed in this run.** The script now reports it; rerunning it (one minute) gives the p-values. The descriptive differences are +8.8 pp (Depth) and +4.4 pp (Distance).

**Caveat.** CV-Bench's 3D boxes come from Omni3D annotations (projected 3D boxes), whose extent can differ systematically from a 2D detector's box. IoU ≥ 0.5 may therefore undercount correct detections, so the 55% in table A is a lower bound. Rerunning with `--set E3_3.marked_eval.iou_threshold=0.3` shows how sensitive it is. Table D does not depend on this threshold.

---

## 5. Interpretation

1. **The decomposition is only as reliable as its weakest step, and there are many steps.** A typical GQA program detects one or two objects, picks an instance, crops and asks. In this run at least one detection came back empty in 23–29% of programs, and the first detection was used without checking qualifiers. Errors at each step compound. The end-to-end model answers from the whole image in one pass and avoids all of them.
2. **The end-to-end baseline is much stronger than in 2023.** ViperGPT reported gains over the end-to-end models of its time. Against a 2025 VLM at 63% (GQA) and 80% (CV-Bench), a program built from a small detector and the same 7B model as planner loses. The agent's advantage, where it exists, is confined to tasks where the end-to-end model is weak: counting.
3. **Counting confirms E3.2 from the other side.** In E3.2, verbal chain-of-thought *hurt* counting by 10.6 pp, because enumerating objects in text loses track. Here detection-based counting *helps*, by 5.2 pp (not significant). For counting, computing with a tool beats reasoning in words.
4. **Interpretability is the agent's real strength.** Every one of the 2,000 answers has a trace that shows where it went wrong. The failures in §4 are concrete engineering targets: constrain answers to the options, select instances by qualifier, keep context around crops, return on every path, offer relational filters. End-to-end answers offer none of this.
5. **On CV-Bench 3D, the bottleneck is grounding the referred object, not the reasoning about it (§4.4).** The detector offers the highlighted object as a candidate only about 55% of the time. With the gold objects, the same depth rules beat end-to-end prompting (96.0% vs 87.2% on Depth). This is the Stage-5 perception bottleneck at the level of tools: the failure is in *seeing which object is meant*, not in reasoning about where it is. It also makes a precise target: better candidate generation (a larger detector, lower thresholds, or locating the drawn outline directly) would turn a loss into a gain on these tasks.
6. **Small pilots were misleading.** `check3` (40 items) showed the agent +20 pp ahead, and Distance at +40 pp. The full run shows −2.7 pp overall and −10.8 pp on Distance. Direct accuracy on Distance was 40% in that pilot and 80% in the full run. The pilots were useful for debugging, not for estimating effects.

---

## 6. Validity checks

| Check | Result |
|---|---|
| Infrastructure errors do not drive the result | Out-of-memory errors were fixed after `check` (20 → 3 of 1,000 GQA items, 0 on CV-Bench); the 3 remaining fall back to direct |
| Prompt development does not explain the result | Excluding the 70 development items: GQA −10.1 pp, CV-Bench −3.0 pp, as with all items (§3.1) |
| No test items appear in the prompt | All 10 examples are invented; three examples first drafted from CV-Bench test items were replaced before any run |
| Programs actually decompose | 90.8% (GQA) and 100% (CV-Bench) use the detector; query-only programs are 8.9% and 0% |
| Failed programs are not silently scored | Every failure is classified (§3.3) and reported both strictly (agent) and with fallback |

---

## 7. Limitations

- **The planner is a 7B VLM.** ViperGPT used a much stronger code model. Many failures in §4.1 are program-writing mistakes that a stronger planner would likely avoid. The result is about this configuration, not about visual programs in general.
- **Small, untuned tools.** Grounding DINO *tiny* at default-style thresholds and Depth Anything V2 *small*. Larger tools, or tuned thresholds, may change the balance, especially for counting and distance.
- **The prompt was developed on pilot items.** Two revisions were made after reading failures (§2.5). The sensitivity check shows this does not change the conclusion, but other prompt designs could do better.
- **GQA exact-match scoring** penalises word-form differences ("donut" vs "donuts") for both conditions. Because agent answers come from crop queries with different wording, this adds noise to the comparison in both directions (§4.1).
- **One run, greedy decoding**, 250 items per CV-Bench task, uncorrected subset tests.
- **The instance-selection analysis (§4.4) uses IoU ≥ 0.5 against Omni3D-derived boxes**, which may undercount correct detections. It covers only CV-Bench 3D; the 2D Relation task stores no boxes.

---

## 8. Next steps

| Item | Purpose |
|---|---|
| **E3.4** visual actions (crop / re-observe) | The roadmap's next step adds a cropping or drawing action that the model chooses during reasoning. The E3.3 runtime (`ImagePatch`, traces, sandbox) is the base. Small-object grounding (E3.2: −27 pp under CoT) and counting are the natural targets. |
| ~~Measure `find_marked` accuracy~~ | **Done (§4.4).** Detection, not selection or depth reasoning, is the bottleneck on CV-Bench 3D. Optional: rerun at IoU 0.3, and with the oracle-vs-direct test now in the script. |
| Optional: better candidates | Lower detector thresholds or a larger Grounding DINO for `find_marked`, or locate the drawn coloured outline directly. Table D bounds the possible gain: up to 96.0% on Depth and 84.8% on Distance with these rules. |
| Optional: targeted fixes | Constrain answers to the options (`best_text_match` for choose questions and fallbacks); select instances by qualifier; add a context margin for relational crop queries. Report as a variant, not as a re-tuned result. |
| E3.5 faithfulness probe | Independent of E3.3; uses E3.1's rationales. |

**Contribution to the Stage-3 gate**

| Gate item | Status |
|---|---|
| "I have a working visual-program agent that I wrote and can extend" | **Met.** Runtime, API, sandbox and traces in [`common/visprog.py`](../common/visprog.py); new tools are one method each (as `find_marked` and `distance_3d` were). |
| "I have my own measurement of when multimodal CoT hurts rather than helps" | Extended. Together with E3.2: verbal reasoning hurts counting, tool-based counting helps slightly. Program decomposition hurts general compositional VQA with this planner. |

---

## Appendix — Reproducing

```bash
python setup.py --skip-install                      # Grounding DINO, Depth Anything, GQA, CV-Bench
python E3_3_visual_programs.py --set E3_3.run_name=full
python E3_3_visual_programs.py --analyze-only --set E3_3.run_name=full   # rebuild tables, no GPU
python E3_3_marked_eval.py --set E3_3.run_name=full                       # §4.4 (Depth Anything only; ~1 min on GPU)
```

The run directory is `/content/drive/MyDrive/MMRM/research_stage_03/E3_3/full/`, next to `check`, `check2` and `check3`. The sensitivity check reads `check` and `check2` from there.

| File | Content |
|---|---|
| `preds_<bench>_direct.jsonl` | direct answers |
| `programs_<bench>.jsonl` | planner output and extracted program per item |
| `preds_<bench>_agent.jsonl` | program, answer, error and error type, full tool trace, tool-usage flags |
| `summary.json`, `summary.md` | the tables of §3 |
| `agent_losses_<bench>.jsonl`, `agent_wins_<bench>.jsonl` | random samples (25, seed 0) with program and trace (the input to §4) |
| `program_errors_<bench>.jsonl` | every failed program |
| `per_item.csv` | direct / agent / agent + fallback correctness per item |
| `config.yaml`, `env.json` | exact configuration and library versions |
| `marked_eval/summary.md`, `summary.json`, `objects.csv`, `items.csv` | §4.4: per highlighted object (searched, detected, gold among candidates, used, IoU) and per item (agent, direct, oracle correctness) |
