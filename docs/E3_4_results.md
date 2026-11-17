# E3.4 — Visual Sketchpad: Visual Actions During Reasoning

**Roadmap stage:** 3 — Multimodal Chain-of-Thought and Compositional Reasoning
**Model:** Qwen2.5-VL-7B-Instruct (reasoner and action planner) · **Tools:** image crop and drawing (PIL), Grounding DINO (tiny), Depth Anything V2 (small)
**Benchmarks:** MathVista testmini geometry (208), CV-Bench (4 tasks × 250, the same 1,000 items as E3.3), V\*Bench (191)
**Reported run:** `full` · **Date:** November 2026
**Code:** [`E3_4_sketchpad.py`](../E3_4_sketchpad.py) · [`common/sketchpad.py`](../common/sketchpad.py) · [`prompts/E3_4_sketchpad.txt`](../prompts/E3_4_sketchpad.txt) · [`prompts/E3_4_text.txt`](../prompts/E3_4_text.txt) · [`config.yaml`](../config.yaml) (section `E3_4`)

---

## 1. Summary

**Question (roadmap E3.4).** Let the model act on the image while it reasons: zoom into a region, mark objects, draw auxiliary lines, or request a depth map, and see each result before continuing (Visual Sketchpad style). Does this beat the same reasoning without actions?

**Success criterion:** visual actions beat the text-only reasoning control on at least one benchmark.

**Answer: nominally met on V\* only, and the gain does not survive the robustness checks. Actions hurt on CV-Bench and do nothing on geometry.**

The key comparison is **sketch vs text**: the same step-by-step prompt, with and without actions. Direct answering is reported as a second reference.

| Benchmark | Direct | Text (CoT) | Sketch (CoT + actions) | Sketch − text | p | Verdict |
|---|---|---|---|---|---|---|
| MathVista geometry (208) | 63.94% | 66.35% | 63.94% | −2.40 pp (−9.1 to +3.8) | 0.57 | neutral |
| CV-Bench (1,000) | 80.30% | 78.20% | 74.10% | **−4.10 pp** (−6.9 to −1.3) | 0.0058 | **hurts** |
| V\* (191) | 73.30% | 64.40% | 71.20% | **+6.81 pp** (+1.0 to +13.1) | 0.047 | helps (nominal) |

- **V\* is the only gain, and it is fragile:**
  - Its p = 0.047 does not survive a Bonferroni correction for three benchmarks (threshold 0.017).
  - Excluding the 31 V\* items seen in the pilots gives +7.50 pp, p = 0.05, which is neutral.
  - Sketch does **not** beat direct answering on V\*: 71.2% vs 73.3% (p = 0.62). The gain over text mostly recovers damage that reasoning did: text is 8.9 pp below direct (p = 0.006), largely through "the object is not in the image" refusals.
  - Reading all 25 gains: only **10** come with a zoom description that supports the answer. **13** commit to an option after the model has said the object is absent or not visible (§4.1).
- **On CV-Bench, actions hurt significantly**, on Depth (−8.4 pp vs text) and Distance (−7.6 pp). Sketch is also 3.5 pp below the E3.3 visual-program agent on the same items (p = 0.037).
- **The model finds the right objects but then reasons worse.** Zooms covered 81% of the highlighted CV-Bench 3D objects, far more than the E3.3 detector's ~55%. Yet on items where every highlighted object was zoomed into, sketch scores 66.5% against text's 81.0% (§4.2).
- **The depth map does not help, although it contains the answer.** Items where `depth()` was called lost 7.9 pp against text (p = 0.021). The model often reads the map backwards. In E3.3, the same depth model with the gold boxes scored 96% on Depth.
- **On geometry figures, actions are rare and unhelpful.** Only 13.5% of items used an action. On those items, accuracy fell from 64% (text) to 43% (2 gains, 8 losses).
- **The model over-acts on natural images.** On CV-Bench every item used an action (2.7 per item), 37% hit the 4-action limit, and 21% repeated an action already taken.

**Main finding.** An untrained 7B VLM can operate visual tools: it zooms onto the right region in most CV-Bench and V\* items. It cannot yet turn what the tools show into better answers. Crops strip away the context that spatial comparisons need, and rendered tool outputs such as a depth map must themselves be perceived, which the model does unreliably. The one place where actions look useful, small objects in high-resolution V\* images, is also the one place where the text-only reasoning baseline is weakest.

---

## 2. Setup

### 2.1 Benchmarks

| Benchmark | Source (HF) | Items | Scoring | Subsets |
|---|---|---|---|---|
| MathVista geometry | `AI4Math/MathVista`, testmini, task "geometry problem solving" | all 208 | option letter (203) or number (5), parsed from the final `ANSWER:` line | answer type |
| CV-Bench | `nyu-visionx/CV-Bench`, test | 250 per task (seed 42), **the same 1,000 items as E3.3** | option letter | Count, Relation, Depth, Distance |
| V\*Bench | `craigwu/vstar_bench`, test | all 191 | option letter | direct_attributes (115: colour or material of a small object), relative_position (76: is A left or right of B) |

Benchmarks are chosen to cover the three uses of a sketchpad: auxiliary lines for geometry, depth and marking for spatial questions, and zoom for small objects in large images (V\* images are around 2,000 × 1,500 px; targets are often under 1% of the image).

### 2.2 Conditions

| Condition | What happens |
|---|---|
| **direct** | Image and question; the model answers with the option letter (max 32 tokens). |
| **text** | Image, question and a step-by-step prompt ending in a final `ANSWER:` line. The control: identical to sketch except that no actions are offered. |
| **sketch** | As text, plus a list of actions. The model may write one `ACTION:` line per turn, up to 4 actions; each result image is appended to the conversation and the model continues. |

Ablations with a single action family (zoom-only, mark-only, draw-only, depth-only) were planned for benchmarks where sketch significantly beats text. They were not run; see §8.

### 2.3 Actions

| Action | Implementation | What the model sees |
|---|---|---|
| `zoom(x1, y1, x2, y2)` | crop of the original image, mapped from the model's coordinate frame | the region at up to 0.8 MP (more pixels than the region had in the first image) |
| `mark("object")` | Grounding DINO tiny (box 0.35, text 0.25, NMS 0.5) | the image with up to 9 numbered boxes, plus their coordinates in text |
| `draw_line(...)`, `draw_point(...)` | PIL, on a canvas that keeps earlier drawings | the image with the drawing, in red |
| `depth()` | Depth Anything V2 small, normalised per image | a greyscale map (brighter = closer, stated in the prompt) |

**Guards (added after pilot 1, §2.5).** These refuse an action with an explanation; a refused action counts as invalid and uses up one of the 4 turns:

- a zoom covering more than 50% of the image ("too_large");
- a mark with a generic name such as "object";
- an exact repeat of an earlier action ("repeat").

**Ending.** After the 4th action the model is told to give the final answer. If it writes another `ACTION:` instead, it is asked once more for the `ANSWER:` line only ("forced answer retry").

The full prompt is in [`prompts/E3_4_sketchpad.txt`](../prompts/E3_4_sketchpad.txt). The text prompt ([`prompts/E3_4_text.txt`](../prompts/E3_4_text.txt)) is the same without the action block.

### 2.4 Inference settings

| | |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, greedy decoding |
| vLLM (GPU shared with the tools) | `gpu_memory_utilization` 0.80, eager mode, `max_model_len` 10,240, up to 6 images per conversation |
| Image resolution | first image 200,704–1,003,520 px (Qwen min/max pixels); zoom results ≤ 802,816 px; other result images ≤ 401,408 px |
| Generation | direct: 32 tokens; text and each sketch turn: 768 tokens |
| Hardware | 1 × NVIDIA L4 (Colab) |

### 2.5 Development history (disclosed)

The prompt and tools were revised once, after reading the first pilot's failures. The second pilot was used only to confirm the revision; nothing was changed after it.

| Run | Items | What was learned | Change made |
|---|---|---|---|
| `check` | CV-Bench 32 (seed 7), V\* 16, MathVista from *textbook QA* | The model zoomed into the whole image (which returned a *lower*-resolution copy), called `mark("object")`, repeated actions in 16–25% of items, and answered the forced-answer request with another action. Text and sketch both claimed V\* objects were absent. Sketch no-answer rate: 12.5% / 31% / 0%. | **Revision 1:** the three guards; zoom results at 0.8 MP instead of 0.4 MP; context 8,192 → 10,240 tokens; forced-answer retry; prompt guidance on each action; **one sentence added to both the text and sketch prompts**: the objects mentioned are in the image, so look closer rather than conclude they are absent |
| `check2` | CV-Bench 32 (seed 11), V\* 16 (seed 7), MathVista from *figure QA* | Sketch no-answer 0% / 6% / 0%; 82% of highlighted CV-Bench objects covered by a zoom; refusals handled. Over-acting and V\* absence claims remained. | none: prompt frozen |

The "objects are present" sentence went into both prompts, so the control stays matched.

**Sensitivity check:** pilot items that are also in the full item sets (15 CV-Bench, 31 V\*) are excluded and the comparisons repeated (§3.1). The MathVista pilots used other subsets, so no geometry item was seen.

---

## 3. Results

### 3.1 Overall

| Benchmark | n | Direct (95% CI) | Text (95% CI) | Sketch (95% CI) | Sketch vs text | Sketch vs direct |
|---|---|---|---|---|---|---|
| MathVista geo | 208 | 63.94 (57.2–70.2) | 66.35 (59.7–72.4) | 63.94 (57.2–70.2) | −2.40 pp, 22/27, p = 0.57, neutral | +0.00 pp, 34/34, p = 1, neutral |
| CV-Bench | 1,000 | 80.30 (77.7–82.7) | 78.20 (75.5–80.7) | 74.10 (71.3–76.7) | **−4.10 pp** (−6.9 to −1.3), 85/126, p = 0.0058, **hurts** | **−6.20 pp** (−8.8 to −3.6), 60/122, p = 5e-6, **hurts** |
| V\* | 191 | 73.30 (66.6–79.1) | 64.40 (57.4–70.8) | 71.20 (64.4–77.2) | **+6.81 pp** (+1.0 to +13.1), 25/12, p = 0.047, **helps** | −2.09 pp, 16/20, p = 0.62, neutral |

Gains/losses are items that sketch got right and the comparison got wrong, and the reverse. p is the exact McNemar test.

**Text vs direct** (the cost of reasoning alone): MathVista +2.40 pp (p = 0.64), CV-Bench −2.10 pp (p = 0.096), V\* **−8.90 pp** (p = 0.006).

**Answers that could not be read** (no final answer, or a refusal):

| | Direct | Text | Sketch |
|---|---|---|---|
| MathVista geo | 1.44% | 5.29% | 6.73% |
| CV-Bench | 0.00% | 0.40% | 0.20% |
| V\* | 0.00% | **15.18%** | 3.14% |

**Excluding the pilot items:**

| Benchmark | Excluded | Sketch vs text | Sketch vs direct |
|---|---|---|---|
| MathVista geo | 0 | −2.40 pp, p = 0.57, neutral | +0.00 pp, p = 1, neutral |
| CV-Bench | 15 | −4.16 pp, p = 0.0053, **hurts** | −6.60 pp, p = 1.3e-6, **hurts** |
| V\* | 31 | +7.50 pp (+0.6 to +14.4), 22/10, p = 0.05, **neutral** | +0.00 pp, p = 1, neutral |

### 3.2 Subsets

| Subset | n | Direct | Text | Sketch | Sketch vs text (gains/losses, p) | Sketch vs direct (gains/losses, p) |
|---|---|---|---|---|---|---|
| **CV-Bench** Count | 250 | 63.60 | 65.20 | 64.00 | −1.20 (17/20, 0.74) | +0.40 (18/17, 1) |
| CV-Bench Relation | 250 | 90.40 | 87.60 | 88.40 | +0.80 (16/14, 0.86) | −2.00 (8/13, 0.38) |
| CV-Bench Depth | 250 | 86.80 | 80.40 | 72.00 | **−8.40** (27/48, 0.02) | **−14.80** (14/51, 4.5e-6) |
| CV-Bench Distance | 250 | 80.40 | 79.60 | 72.00 | **−7.60** (25/44, 0.029) | **−8.40** (20/41, 0.0099) |
| **V\*** direct_attributes | 115 | 73.91 | 63.48 | 73.04 | **+9.57** (18/7, 0.043) | −0.87 (12/13, 1) |
| V\* relative_position | 76 | 72.37 | 65.79 | 68.42 | +2.63 (7/5, 0.77) | −3.95 (4/7, 0.55) |
| **MathVista** mc | 203 | 65.02 | 66.50 | 65.52 | −0.99 (22/24, 0.88) | +0.49 (34/33, 1) |
| MathVista number | 5 | 20.00 | 60.00 | 0.00 | −60.00 (0/3, 0.25) | −20.00 (0/1, 1) |

Subset p-values are not corrected for multiple comparisons.

**Sketch vs the E3.3 visual-program agent** (with fallback, same 1,000 CV-Bench items): 77.60% → 74.10%, −3.50 pp (−6.8 to −0.4), 115/150, p = 0.037, **hurts**. By task: Count −4.4 (p = 0.19), Depth **−12.8** (p = 2.6e-4), Distance +2.4 (p = 0.6), Relation +0.8 (p = 0.87).

### 3.3 How the actions were used

| | MathVista geo | CV-Bench | V\* |
|---|---|---|---|
| Items with ≥ 1 action | 13.5% | **100%** | 97.4% |
| Actions per item | 0.30 | 2.69 | 1.96 |
| Invalid actions (refused or unparseable) | 12.9% | 17.9% | 24.5% |
| Refused: too large / repeat | 7 / 1 | 216 / 262 | 36 / 56 |
| Items that hit the 4-action limit | 3.4% | **37.0%** | 19.4% |
| Items with an exact repeat attempted | 0.5% | 21.0% | 19.4% |
| Forced answer needed the retry | 1.0% | 19.7% | 11.0% |

**Accuracy on the items where each action fired** (text → sketch on the same items):

| Benchmark | Action | Items | Calls | Text → sketch | Δ (gains/losses, p) |
|---|---|---|---|---|---|
| MathVista geo | zoom | 28 | 54 | 64.3 → 42.9 | −21.4 (2/8, 0.11) |
| MathVista geo | draw | 7 | 8 | 42.9 → 14.3 | −28.6 (0/2, 0.5) |
| MathVista geo | no action | 180 | 0 | 66.7 → 67.2 | +0.6 (20/19, 1) |
| CV-Bench | zoom | 805 | 1,843 | 77.9 → 74.2 | **−3.7** (64/94, 0.021) |
| CV-Bench | depth | 277 | 277 | 76.9 → 69.0 | **−7.9** (31/53, 0.021) |
| CV-Bench | draw | 274 | 346 | 84.7 → 82.9 | −1.8 (21/26, 0.56) |
| CV-Bench | mark | 147 | 226 | 72.1 → 72.8 | +0.7 (16/15, 1) |
| V\* | zoom | 186 | 361 | 66.1 → 72.0 | +5.9 (23/12, 0.09) |
| V\* | mark | 9 | 11 | 77.8 → 55.6 | −22.2 (0/2, 0.5) |
| V\* | no action | 5 | 0 | 0.0 → 40.0 | +40.0 (2/0, 0.5) |

An item can use several actions, so the rows overlap. These are not causal estimates: the model chooses which items to act on.

- **V\* is a zoom experiment in practice:** 361 of 377 calls are zooms, in 97% of items.
- **On geometry the model mostly declines to act**, unlike in pilot 2 on MathVista figure QA (98% of items zoomed). When it does act on a diagram, it tends to tile it blindly (`zoom(100, 100, 200, 200)`, then `(100, 200, 200, 300)`, …).
- **The guards fired often and the model did not always adapt.** On CV-Bench, 262 exact repeats were refused, often the same action three times in a row until the forced answer.

---

## 4. Failure analysis

The wins and losses files hold up to 25 items per benchmark where sketch and text disagree. For V\* that is **every** disagreement (25 gains, 12 losses).

### 4.1 V\*: where the +6.8 pp comes from

Every one of the 37 V\* items where sketch and text disagree, classified by what the sketch output says:

| Sketch output | Gains (sketch right, text wrong) | Losses (sketch wrong, text right) |
|---|---|---|
| Zoom shows the object and the description supports the answer | **10** (e.g. spider on the sail, white dog, curly hair, polychromatic Apple logo, golden retriever) | — |
| Zoom lands on the wrong object, or the object is seen and misread | — | 3 (the "parachute" read from a red flag on a sailboat; mark finds the tissue box but it is called white; "mailbox is red, therefore (D) yellow") |
| Model says the object is absent or not visible, then picks an option anyway | **13** (e.g. "no red car or police car… ANSWER: (A)"; "no handbag is visible… ANSWER: (D)") | 5 (zoom misses the bucket, trash can, watchband…; a wrong option is picked) |
| Answer given only after the forced-answer prompt, no reasoning | 2 | 4 (one of them never answers) |

- **Zoom genuinely helps in 10 items and genuinely misleads in 3.** That net +7 items is the grounded effect of zooming. It is too small to test on its own, but it is the clearest positive sign in E3.4.
- **Most of the rest is a change of answering behaviour, not of seeing.** In 13 gains the model says it cannot find the object but still commits to a letter, and on V\* a committed guess is often right (half of V\* questions have 2 options). The text condition more often writes "ANSWER: (N/A)" or "none of the options", which is scored wrong: 15.2% of text outputs have no answer, against 3.1% for sketch.
- **The "objects are in the image" sentence (revision 1) did not stop absence claims** in either condition. This is the reasoning-induced refusal of E3.1 and E3.2 again: the longer the model reasons about a small object, the more likely it is to conclude that it is not there. Direct answering, with no reasoning, is the best V\* condition (73.3%).

### 4.2 CV-Bench: the model finds the objects, then reasons worse

**Grounding check (CV-Bench 3D, items where the model zoomed or marked).** Of 893 highlighted objects, 81.1% were covered by a zoom (at least half of the gold box inside the zoomed region). No `mark` call located a highlighted object.

| Items with zoom or mark | n | Sketch | Text |
|---|---|---|---|
| All highlighted objects covered | 200 | **66.5%** | 81.0% |
| At least one not covered | 123 | 80.5% | 80.5% |

Text accuracy is the same on both groups, so the zoomed items are not harder ones. **Zooming onto exactly the right objects goes with a 14.5 pp drop.**

This reverses the E3.3 picture. There, the detector found the highlighted object only ~55% of the time, and with the gold objects the reasoning would have won (§4.4 of [E3.3](E3_3_results.md)). Here, grounding succeeds and the reasoning fails. The losses show why:

- **Separate crops remove the comparison.** A typical Depth or Distance item zooms into object 1, then object 2, then answers from two isolated close-ups: "The pedestrian is closer… it is in the foreground" (3D-1888), "the traffic cone is much smaller and further away" (3D-1937, wrong). Relative size, occlusion and position in the scene are the cues the text condition uses, and a tight crop removes them.
- **The depth map is misread.** In 13 of the 25 sampled CV-Bench losses, `depth()` is followed by "the depth map shows X as closer", and X is wrong (e.g. 3D-1462, 1467, 1508, 1612, 1663, 1911, 1949). In most of these, text had answered correctly from the photo. Twice the model explicitly reads darker as closer, although the prompt says brighter = closer. Depth wins look the same in reverse ("the depth map shows the lamp closer", correct). Reading a rendered map is itself a perception task, and the model's reading is unreliable. The map contains the answer: in E3.3 §4.4, the same depth model and the gold boxes scored **96.0%** on Depth, while sketch scores 72.0%.
- **Draw is used as a ruler.** On Distance, lines are drawn from box to box and then interpreted ("the red line… indicates that the books are closer to the shelves", 3D-2343, wrong). The line carries no depth information.
- **Over-acting wastes the budget.** Whole-image zooms are refused and re-attempted with a slightly smaller box (2D-270: three refused zooms before `mark("rocks")`). Counting items zoom into an arbitrary 100 × 100 region before marking.

**Where it works:** `mark` on Count (2D-764: 8 broccoli boxes → correct "8", where text counted 7) and Relation items where both objects get a box. Mark is the only action that is not negative on CV-Bench (+0.7 pp), but it fired on only 15% of items.

### 4.3 MathVista geometry: actions add nothing a diagram needs

- Geometry differences between text and sketch are mostly **different reasoning chains on the same problem**: neither condition acted in 87% of items, and those items are even (20/19). The prompts differ only by the action block, which is enough to change a greedy chain.
- **Zooms on diagrams are blind tiling**, not targeted (e.g. mathvista-678: four 160 × 160 tiles left to right, then a wrong trigonometric step). Nothing in a clean, low-resolution figure needs enlarging.
- **`draw` was used 8 times in 208 problems**, and never to construct an auxiliary line that a solution needs. Visual Sketchpad's geometry gains came from a much stronger model (GPT-4o) drawing purposeful constructions; a 7B model does not.
- **Unreadable answers come from repetition loops** that hit the 768-token limit ("But since the options do not include 1, we need to recheck…" repeated), in text (5.3%) and sketch (6.7%) alike.

---

## 5. Interpretation

1. **Tool use and tool benefit are different capabilities.** The model learned the action syntax from the prompt alone and zoomed onto the right objects in 81% of CV-Bench 3D cases. That part works. What fails is turning a crop or a depth map into a better answer. With this model, the sketchpad adds perception steps (reading the new image) without adding reliability.
2. **Crops help attribute questions and hurt relational ones.** V\* direct_attributes ("what colour is the X?") is the only subset with a significant gain over text (+9.6 pp, p = 0.043). Its answer is inside the crop. Depth, Distance and left/right questions need two objects and the space between them, which a crop removes. This matches the design intent of zoom (V\*-style search) and marks its limit.
3. **A tool output shown as an image must be perceived again.** In E3.3, depth values went straight into a Python comparison and were reliable (96% with gold boxes). In E3.4, the same depth model's output is shown as a picture, and on the items where the model requested it, accuracy fell from 77% (text, reading the photo) to 69%. Tools whose results are returned **as text or numbers** (E3.3) suit a 7B model better than tools whose results are **images** (E3.4).
4. **The reasoning-induced refusal is the strongest effect on V\*.** Step-by-step reasoning costs 8.9 pp on V\* (text vs direct), mostly through "the object is not in the image". Sketch recovers most of that, but more through committing to an answer than through seeing better (§4.1). This is the third Stage-3 experiment in which reasoning about small or hard-to-see objects backfires (E3.1 refusals, E3.2 grounding −27 pp, E3.4 V\*).
5. **Stage-3 pattern so far.** Verbal reasoning helps knowledge-heavy science questions (E3.2) and hurts perception-heavy ones. Tool decomposition (E3.3) and visual actions (E3.4) do not fix perception-heavy questions with a 7B model and no training. In both, the limiting step is perception: finding the object in E3.3, reading the tool's visual output in E3.4.

---

## 6. Validity checks

| Check | Result |
|---|---|
| The control is matched | Text and sketch prompts are identical except for the action block; the "objects are present" sentence was added to both |
| Prompt development does not explain the result | CV-Bench result unchanged without the 15 pilot items (−4.16 pp, p = 0.0053). The V\* gain weakens to p = 0.05 without the 31 pilot items, and is reported as fragile |
| Actions do what they claim | Zoom coordinates map to the original image (81% of highlighted objects covered); action result images saved under `action_images/` and inspected in both pilots |
| Invalid actions are counted, not hidden | Refusals, repeats, forced answers and retries are logged and reported (§3.3) |
| Unreadable answers are counted as wrong for every condition | Rates reported per condition (§3.1); every case is in `no_answer_<bench>.jsonl` |
| The CV-Bench comparison with E3.3 is paired | Same 1,000 items, same seed |

---

## 7. Limitations

- **One model, no tool-use training.** Visual Sketchpad's gains were reported with GPT-4o. A 7B model following a prompt is a different setting; the result is about this configuration, not about visual actions in general.
- **The prompt was revised once on pilot items** (§2.5). The CV-Bench conclusion does not depend on it; the V\* gain is borderline once pilot items are excluded.
- **The action budget (4) and the guards shape behaviour.** Refused actions consume a turn; a larger budget or no guards would change the over-acting statistics.
- **Depth is rendered as greyscale.** A colour map, or returning depth values as text, might be read better (see §8).
- **Per-action rows are observational:** the model chooses where to act, so "items where depth fired" are not a random sample.
- **One run, greedy decoding**; V\* has 191 items, so ±7 pp confidence intervals; uncorrected subset tests.
- **The classification of V\* disagreements (§4.1) is by reading the outputs**, one rater (the author), not blind.

---

## 8. Next steps

| Item | Purpose |
|---|---|
| ~~Single-action ablations~~ | **Not run.** The rule was to run them where sketch significantly beats text. Only V\* qualifies, nominally (p = 0.047, not significant after Bonferroni or without pilot items), and V\* sketch is already 96% zoom (361 of 377 calls): a zoom-only run would be nearly the same condition. Mark/draw/depth-only on V\* would test actions the model barely uses there. |
| Optional: depth as numbers | Make `depth()` return the median depth of named or marked objects as text, as in E3.3. This would separate "the depth tool is useful" from "the model can read a depth image" on CV-Bench Depth. |
| Optional: zoom with context | Return the crop beside a thumbnail of the whole image with the region outlined, to test whether lost context explains the relational losses. |
| **E3.5** faithfulness probe | Next in the roadmap. Uses E3.1's two-stage rationales. |

**Contribution to the Stage-3 gate**

| Gate item | Status |
|---|---|
| "I have a working visual-action loop that I wrote and can extend" | **Met.** Multi-turn loop with image results, guards and logging in [`common/sketchpad.py`](../common/sketchpad.py); a new action is one method plus one line of documentation. |
| "I have my own measurement of when multimodal CoT hurts rather than helps" | Extended. Visual actions help (nominally) only on small-object attribute questions; they hurt relational 3D questions even when grounding succeeds. |

---

## Appendix — Reproducing

```bash
python setup.py --skip-install                       # model, Grounding DINO, Depth Anything, MathVista, CV-Bench, V*
python E3_4_sketchpad.py --set E3_4.run_name=full
python E3_4_sketchpad.py --analyze-only --set E3_4.run_name=full   # rebuild tables, no GPU
```

The run directory is `/content/drive/MyDrive/MMRM/research_stage_03/E3_4/full/`, next to `check` and `check2`. The sensitivity check reads `check` and `check2`; the comparison with E3.3 reads `../E3_3/full`.

| File | Content |
|---|---|
| `preds_<bench>_<condition>.jsonl` | outputs per condition; for sketch, every turn with its action, arguments, refusal reason and result image name |
| `summary.json`, `summary.md` | the tables of §3 |
| `examples_<bench>_{wins,losses}.jsonl` | sketch-vs-text disagreements (up to 25 each, seed 0) with all turns: the input to §4 |
| `no_answer_<bench>.jsonl` | every output without a readable answer |
| `action_images/` | every action result image (JPEG) |
| `per_item.csv` | correctness per item and condition |
| `config.yaml`, `env.json` | exact configuration and library versions |
