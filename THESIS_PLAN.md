# THESIS PLAN — authoritative roadmap and framing guide

**Written:** 2026-07-31 · **Status:** ACTIVE — this is the document agents and
future sessions consult before doing project work.
**Companions:** `FINDINGS.md` (measured results, updated per-finding),
`MARCH_2026_POSTMORTEM.md` (why pre-July results are unusable),
`INFERENCE_GUIDE.md` (how to run the models), `Thesis_Topic.md` (the claim).

---

## 1. The thesis claim, stated once, correctly

From `Thesis_Topic.md`: a **real-time student behavior analysis system for
spotting attention loss during learning sessions**, from cameras, via these
named indicators: **body language, head pose, gaze direction, facial
expressions** — surfaced to instructors for intervention, with ethics/privacy
respected.

The supervisor's reframing (documented in `to-do list.md`) is the load-bearing
epistemic move and MUST be preserved in all writing:

> We claim "the student's head is down," never "the student is not paying
> attention." The system extracts **visible attention cues and off-task
> behavior events**; "attention loss" is operationalized as sustained visible
> off-task behavior, not as a mental state.

**Rule for all agents:** Branch B (cues, events, real-time) is the thesis
claim. Branch A (grounding detector + caption→box matching) is *methodology
that makes the training labels sound*. Never present Branch A as "the thesis
novelty" — that framing error cost this project two days of misallocated
emphasis and was corrected 2026-07-31 (see FINDINGS.md §9a scoping correction).

---

## 2. What exists and where it goes in the thesis

Every completed result maps to a chapter. Nothing is discarded.

| asset | result | thesis placement |
|---|---|---|
| LLMSTU data pipeline (video recovery 100%, dedup 283k→85k, leak-free video-wise splits) | done | **Ch. 3 Dataset** — a primary contribution; the leakage discovery + fix is a methods finding in its own right |
| Human gold sets (445 frame-level + 754 dense event frames) | done | Ch. 3 — validation methodology |
| **Teacher ceiling**: Qwen3.5-VL vs human gold = 87.4% frame / 75% event; per-cue recall table | done | Ch. 3 + Ch. 6 Limitations — first measurement of the label ceiling; cite as the bound on all downstream numbers |
| Matching ablation: ordinal 0.3230 vs Hungarian **0.4954** vs p90 0.4787 (R@1, identical inputs) | done | **Ch. 4 Method** — justifies the pseudo-label pipeline for the whole-frame case; +17.2/+53% with R@10 as control. Frame as "label-binding quality determines grounding quality," NOT as the thesis claim |
| Sinkhorn/OT + calibration (AUROC 0.760, ECE 0.011) | done | Ch. 4 — with ARM C's negative result (filtering loses to volume) |
| Main detector run (exact labels, 25k iters) | 🔄 running, R@1 0.5946 @ 5k, ETA ~12:35 | **Ch. 5 Table 1** — the citable grounding number replacing the invalid leaked 0.613 |
| Temporal cue model, 6-class, macro-F1 0.384 vs majority 0.144 | done | Ch. 5 — Branch B baseline; weak classes motivate head-pose work |
| Event layer + metrics (segmentation, temporal-IoU, false-alerts/hour) | built; model-vs-teacher 89% miss | Ch. 5 — currently the honest open gap; P0 below |
| Real-time profiling (7.1 FPS real scene, 2.0 @ 30 students, CLIP batching 1.9×) | done | Ch. 5 — supports the "real-time" word in the title |
| Runtime threshold bug (shipped config recall 0.43; fix sweep queued on final ckpt) | fix queued | Ch. 5 + Ch. 6 — report as a deployment-calibration finding |
| March post-mortem (11 failures, all silent-plausible-output) | done | **Ch. 6** — reproducibility/failure analysis; genuinely strong thesis material |
| SCB dataset + ODVG converter (505 imgs / 3,753 regions, 0 errors) | done | Ch. 5 — zero-shot generalization experiment (P1) |
| Caption neutralization (protected attributes stripped) | done | **Ch. 7 Ethics** — concrete, implemented privacy measure |
| Dashboard | **dropped by user decision 2026-07-29** | mention as engineering future work, one paragraph |

**Numbers that must NOT be cited** (see FINDINGS.md for why): 92.3% matching
(GT leak via spatial prior — cite 81.5% until the queued sw=0.0 sweep lands);
0.6442 temporal accuracy (2-class degenerate + batch-mean bug — cite 0.384
macro-F1); R@1 0.613 (leaked split); 0.410 DDP-sharded macro-F1 (cite 0.384);
any 2026 LVIS number (harness returns zeroed metrics).

---

## 3. Gap analysis — thesis promise vs. present system

Ordered by how much each gap threatens the thesis defense:

1. **Event layer: 89% episode miss (model vs teacher).** The thesis promises
   *detecting attention-loss episodes*; this is the direct measure of that and
   it is currently failing. Not yet measured: model events vs the HUMAN gold
   (754 frames / 24 episodes) — that is the number the thesis must report.
2. **Head pose & gaze: the two named indicators the model cannot see.**
   `turned_to_peer` F1 0.18, `looking_away` F1 0.29 — both orientation
   judgements; CLIP+bbox+color features cannot represent them. Gaze also
   carries phone-use when the phone is occluded (FINDINGS §5.4, user-confirmed).
   OpenCV scaffold detects faces in only 25% of crops → a metric backend
   (MediaPipe) is required, installed ONLY when no training run is in flight.
3. **Runtime detection config silently drops ~57% of students** (prompt
   "student" is out-of-vocabulary post-fine-tune at thr 0.45). Sweep on final
   checkpoint queued; `attention_temporal.yaml` must be updated from its output.
4. **Facial expressions: named in the proposal, absent from the system.**
   Resolution below (§5) — argue deliberate exclusion, do not quietly ignore.
5. Failure-mode analysis tooling (to-do item 32) — needed for Ch. 6.
6. Task-context metadata (item 19) — blocked on user knowledge of recordings.

---

## 4. Work plan (priority-ordered; each item states its thesis purpose)

### P0 — the return-to-track core (do these before anything else)

**P0.1 Finish + harvest the main run** *(automated today)*
Runs to 25k (~12:35). Then: copy best+final ckpts to `thesis_bundle/checkpoints/`
(runbook rule — March lost its best checkpoints to pruning); record full R@1
curve in FINDINGS §Table 1; queued jobs fire automatically (clean matching
sweep sw=0.0 → threshold sweep on iter_25000).
→ Thesis: Ch. 5 Table 1.

**P0.2 Fix the runtime detector config** *(30 min after P0.1)*
Take `work_dirs/probe/threshold_sweep_iter25000.json`'s recommendation
(GENERIC prompt only — a content prompt like "using laptop" wins aggregate F1
by naming the majority activity and misses exactly the off-task minorities;
warning is built into the tool). Update `configs/attention_temporal.yaml`
`text_prompt`/`score_thr`; re-run the profiling harness once to confirm FPS
unchanged. → Thesis: Ch. 5 deployment calibration.

**P0.3 Model-vs-HUMAN event evaluation** — evaluator BUILT and CPU-validated
2026-07-31 (`attention/eval_events_vs_human.py`; loader gives 10 tracks, 754
human / 230 rejected). *(needs ~1 GPU-hour; ASK USER before GPU use)*
Run the full Branch B chain (detector→tracker→features→temporal→events) over
the two dense gold segments; score with `evaluate_events` against
`outputs/gold_events.jsonl` (24 human episodes). This replaces the
uninterpretable model-vs-pseudo 89% with the thesis-grade number.
→ Thesis: Ch. 5's most important Branch B table.

**P0.4 Head pose for real: MediaPipe backend** *(GPU-free install AFTER the
main run finishes; never during a live run)*
(a) `pip install mediapipe` when no training is in flight; smoke-test
`MediaPipeHeadPose` (already implemented in `attention/head_pose.py`) on the
same 60-crop sample the OpenCV backend scored 25% on — require ≥70% face-found
or escalate to 6DRepNet. (b) Rebuild sequences at 555-dim
(`SequenceNPZDataset` + builder; ~2-3 h CPU-parallel). (c) Retrain temporal
model (4 GPUs ~1.5 h at 90 epochs — ASK USER). (d) Report per-class deltas;
success criterion: `looking_away` and `turned_to_peer` F1 both improve
materially (target ≥0.35 / ≥0.30); also re-run event eval (P0.3 config) since
orientation cues feed events.
→ Thesis: the head-pose/gaze indicators the proposal names, plus an ablation
(552-dim vs 555-dim) that directly shows their value. This is the single
highest-value remaining experiment for the claim.

**P0.5 Event-layer improvement pass** — ✅ PARTIALLY RESOLVED 2026-07-31,
and the finding REDIRECTS this item.
A 64-config grid search (smoothing x min_duration x max_gap) over the teacher
timeline vs the human gold found **every config matches exactly 18/24 episodes**;
only false-alerts/hour varied (3.9-9.4). Therefore:
  - Segmentation tuning CANNOT improve event recall — the missed episodes are
    absent from the CUE timeline, not lost by the event layer.
  - Temporal smoothing does NOT help at event level (hypothesis refuted).
  - Adopted for free: `max_gap_s = 1.0` (same 18/24, FA/h 4.7 -> 3.9).
  - Remaining P0.5 work is therefore FOLDED INTO P0.4: the only route to better
    event recall is better cues, i.e. head pose / gaze.
→ `outputs/event_config_tuning.json`; FINDINGS.md 6.1.

### P1 — strengthens the thesis, after P0

**P1.1 Zero-shot SCB generalization eval** *(~30 GPU-min — ASK USER)*
Frozen main checkpoint on `outputs/odvg_scb_bowturnhead_val.jsonl` (505 imgs;
phrases verbatim from training vocabulary so domain gap isn't confounded with
vocabulary gap). Report R@1/R@5 vs the LLMSTU val numbers.
→ Thesis: robustness/generalization section; supports the open-vocabulary
design choice.

**P1.2 Failure-mode analysis tooling** *(CPU)*
Per-video error breakdown, worst-crop galleries, confusion structure over the
val split for both branches; feeds Ch. 6. (to-do item 32.)

**P1.3 The matching→claim bridge experiment** *(optional but powerful; mostly
CPU + one short temporal train)*
Build Branch B sequences from whole-frame captions bound by ordinal vs
Hungarian; compare cue macro-F1. Directly measures "what does binding quality
cost the attention system when only whole-frame captions exist" — connects the
+17.2 methodology result to the thesis claim so Table 2 is not orphaned.

**P1.4 Commit + push everything** (user handles pushes; remind at day end.)

### P2 — if time permits / future work section

- Task-context metadata (item 19) — needs user input on which recording is
  which task type.
- sequence_builder CLIP batching (runtime parity TODO).
- Stale `thesis_bundle/manifest.json` pointer (e1_best → point at the
  preserved copy in thesis_bundle/checkpoints/).
- Dashboard: future work paragraph only (user decision).

---

## 5. Facial expressions — the resolution

The proposal names "facial expressions (boredom, perplexity, curiosity)". The
system does not detect them, and SHOULD NOT pretend to. Write it as a
**deliberate, defended exclusion**, on three grounds already in evidence:

1. **Epistemic:** the visible-cue principle (supervisor's reframing) —
   emotion labels are inferred mental states, exactly what the taxonomy
   refuses to claim.
2. **Empirical:** 16.5–61.5% per-video occlusion and a wide-angle camera
   (2812×1050 across a whole lab) — faces are frequently too small/blocked
   for reliable affect recognition; the gold-set annotator (the user)
   rejected 55.5% of the scattered sample as unverifiable.
3. **Ethical:** affect inference from classroom video is the most
   privacy-invasive tier of this technology family; excluding it strengthens
   the Ch. 7 ethics position.

Then: "affect recognition on datasets with per-student close-up capture
(e.g., DIPSER) is future work." One page, honest, defensible. This turns a
missing deliverable into a reasoned scope decision.

---

## 6. DIPSER (a.k.a. DIPSEER): use it, but narrowly — and do not block on it

**Answer to "will it put us back on track": no — P0 puts us back on track,
and none of P0 requires DIPSER.** The return-to-track levers are MediaPipe
head pose on LLMSTU, the event layer, and the runtime fix. DIPSER *strengthens
validation* of the claim; it is not the path to it. Do not let acquisition
(which is blocked on a manual SciDB account registration — the Bitbucket URL
in the paper 404s; data lives at
`scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893`) gate any P0
work.

### 6.1 ACQUISITION FACTS (measured 2026-07-31, supersedes the estimate below)

User registered; link list at `grounding_data/7856c716c0cc4589a23ee4a23d8a0893.txt`
(618 links). One subject zip downloaded and dissected. Hard facts:

| fact | value | consequence |
|---|---|---|
| 483 subject zips @ ~1.1 GB + 135 context_cam @ ~1.7 GB | **~780 GB total** | full download IMPOSSIBLE (302 GB free). Earlier 60–150 GB estimate was wrong |
| zip layout | `images/` (2,469 png), `metadata/` (1 json per image), `watch_sensors/`, `labels/` | labels ARE included, no separate download needed |
| per-frame metadata | `face/headpose/pose{pitch,yaw,roll}`, `facemesh` (**478 pts**), `body_pose`, bboxes | pose is precomputed |
| **478 facemesh points** | MediaPipe FaceMesh + `refine_landmarks=True` (468+10 iris) | **DIPSER's head pose IS MediaPipe output** |
| expert labels | 4 labelers + self, `{datetime, attention, emotion}`, 7–44 entries per ~5-min session | sparse but human — the genuinely irreplaceable asset |
| also present | `age`, `gender`, **`race`** per frame | must be excluded (ethics; we neutralised protected attributes) |

**DECISION CHANGE — do NOT use DIPSER for head-pose supervision.** Its head pose
is MediaPipe FaceMesh output, which P0.4 computes directly on LLMSTU for free.
Downloading hundreds of GB to distill an estimator we can simply run would be
pure waste. P0.4 is unaffected and remains the head-pose route.

**DIPSER's only irreplaceable asset is the human attention/emotion ratings.**
That supports exactly one experiment: the external-validation correlation
(§6 use 1). Everything else it offers, we can already produce.

### 6.2 VARIANCE CHECK — PASSED (4 subjects, all 3 groups, 209 labels)

An earlier note here flagged low rating variance as a possible blocker. It was
based on reading the empty bin as the disengaged end. **User confirmed the
scale: 1 = low engagement, 5 = maximum.** With that, the distribution is
usable and the concern is withdrawn:

| level | meaning | n | share |
|---|---|---|---|
| 1 | lowest | 0 | 0% |
| **2** | **low** | **85** | **41%** |
| 3 | middle | 101 | 48% |
| 4 | high | 19 | 9% |
| 5 | maximum | 4 | 2% |

The disengaged class our detector targets is the PLURALITY (41%), and the
engaged contrast group has 23 samples. Only the extreme-low bin is empty.

**Variance decomposition (the decisive check):**

| | value |
|---|---|
| within-subject sd (avg) | **0.678** |
| between-subject sd of means | 0.164 |
| ratio | **4.1x** |

Ratings vary INSIDE a session, not merely between students. This supports a
**time-resolved** correlation — does our off-task event activity at time t
predict the expert engagement rating at t — which is far stronger than a
subject-level correlation and needs far fewer subjects. ~50 paired
observations per subject; 20-30 subjects gives 1,000-1,500 pairs.

⚠️ Remaining design caveats: labelers annotate at DIFFERENT timestamps (entry
counts 3-44 for the same subject), so combining them needs an explicit
interpolation/nearest-label rule that must be stated, not silently chosen;
and ratings are ordinal, so use rank correlation (Spearman) or ordinal
regression, never Pearson.

**Proposed minimal download** (pending user approval; ~25–35 GB):
- **0 context cams** (we do not train the detector — saves ~230 GB)
- ~20–30 subject zips spread across group_01..03 for subject diversity
- Sufficient for ~2,000–3,000 human-rated timestamps for the correlation

If/when the user registers and downloads it (est. 60–150 GB; 302 GB free):

**USE — in this order of value:**
1. **External validation of the attention-loss operationalization** (the
   highest-value use, and the one that most directly serves the thesis claim):
   run our cue pipeline on DIPSER's per-student streams and **correlate our
   off-task event outputs with DIPSER's expert-rated attention levels**. We
   never train on their attention scale (mental state), but showing our
   visible-cue events *correlate* with independent expert judgments of
   disengagement is the bridge from "we detect head-down episodes" to "these
   episodes indicate attention loss" — the exact inferential step the thesis
   otherwise has to hand-wave.
2. **Head-pose/gaze supervision:** their landmark/pose/gaze processed features
   as ground truth to validate (or fine-tune) our head-pose module — better
   than MediaPipe-only because it is human/expert-anchored.
3. **Emotion labels:** only for the future-work discussion (§5), not import.

**DO NOT:**
- Train the detector on it (per-student 640×480 cameras ≠ our whole-frame
  geometry; R@10 is 0.995 — detection is not the bottleneck).
- Import the 5-level attention scale as training labels (violates the
  visible-cue principle).
- Touch ethnicity/age/gender fields (we deliberately neutralized protected
  attributes).
- Validate on any mixed val set — DIPSER's close-up capture is sharper
  evidence than deployment reality; validate on LLMSTU-only held-out data.
- Recaption 1.31M images with Qwen3.5 (4.6× LLMSTU inference cost, unmeasured
  teacher ceiling on a new domain, and DIPSER's human labels are better than
  pseudo-labels for the fields we need anyway).

---

## 7. Standing rules for agents working this project

1. **No GPU work without explicit user approval; hard cap 4 GPUs, never 8.**
2. **Record every new result/defect in `FINDINGS.md` in the same turn**, with
   evidence paths; ⚠️-mark anything with a methodological caveat.
3. Preserve cited checkpoints to `thesis_bundle/checkpoints/` BEFORE any
   cleanup; byte-verify copies (`cmp`) before deleting originals.
4. Fail loudly: no zeroed-metric fallbacks, no silent label drops, no
   defaulting past missing keys. The March post-mortem exists because silent
   plausibility destroyed four months of results.
5. Watch out for self-matching `pgrep`/`pkill` patterns in background
   watchers (bit us twice: 28 idle GPU-minutes, one killed shell). Match on
   config filenames, or run watchers from script files.
6. Report the honest number (single-process eval, leak-free split), never the
   flattering one. The replacement numbers are lower and mean what they say.
7. The word "attention" in user-facing claims always means *visible cues*,
   per §1.

---

## 8. Current state snapshot (2026-07-31 07:40)

- Main run: iter 7,450/25,000, R@1 0.5946@5k, ETA ~12:35. 4 GPUs busy.
- Auto-queued after it: clean matching sweep (sw=0.0) → threshold sweep on
  iter_25000 (waiter pid verified, matches on config filename).
- SCB downloaded + converted (505/3,753 validated). DIPSER blocked on user's
  SciDB registration.
- Head-pose: opencv scaffold live (25% face-found — insufficient); MediaPipe
  code ready, install pending a training-free window.
- Dense event gold: 24 human episodes ready for P0.3.
- Disk: 302 GB free. Git: uncommitted work pending user's push workflow.
