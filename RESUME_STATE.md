# Resume state — autonomous run started 2026-09-11

Written so a session that is cut off (usage limit, disconnect) can be continued
exactly, by me or by anyone. **Kept current as work lands** — if this file and
`git log` disagree, trust `git log`.

To continue: `claude --resume <session-id>`, or start a fresh session and say
"read RESUME_STATE.md and continue".

## The five tasks, as authorised

| # | task | state |
|---|---|---|
| 1 | Remove the 900-frame cap on video analysis | **DONE** (needs commit) |
| 2 | Prune dropdown to best-per-taxonomy, keep files | **DONE** (needs commit) |
| 3 | More FPS + hardware upgrade | **l4x1 LIVE**. Stride/resolution tuning still pending |
| 4 | VLM model in the dropdown | **WIRED + stub-tested; BLOCKED** by transformers 4.44.2 pin (FINDINGS 20.3) |
| 5 | Diagnose + improve talking_to_peer / phone_use / using_laptop | **DIAGNOSED** (FINDINGS 20.4); retraining next |

Decisions taken by the user, do not re-ask:
* prune = **hide from dropdown, keep the files** (nothing deleted from HF)
* 6-class slot shows **both** `epochs240/mstcn_556_hp` (best mean, replay-capable)
  and `wave2/mstcn_1074_hp_head` (highest score, live-only)
* hardware = **upgrade** (l4x1 requested)
* class work = **diagnose first**, then retrain only what the diagnosis supports
* full authority: run commands, push to git and to the HF Space/repo without asking

## 1. Frame cap — done

`max_frames = 0` now means the whole video, and is the default in
`precompute_session.precompute`, `pipeline_bridge.run_live`, and both server
flags (`--analyse-frames`, `--max-frames`).

Evidence it was real: `sessions/0325` cached **900 of 1931 frames = 47%** of the
lecture, silently. A full re-run is in progress to `sessions/0325_full`
(verified header: `1931 frames (1.1 min at 30.0 fps); target 1931`, 4.2 fps on
an A100).

## 2. Dropdown prune — done

`model_registry._mark_recommended` flags the best live model per **taxonomy**,
restricted to the canonical training target where one exists, plus the
replay-capable/live-only pair for cue6. Shortlist is 5 of 34:

| classes | variant | val |
|---|---|---|
| 9 | `epochs240/mstcn_556_hp:cue9` | 0.4680 |
| 6 | `epochs240/mstcn_556_hp` | 0.5326 |
| 6 | `wave2/mstcn_1074_hp_head` (live-only) | 0.5307 |
| 3 | `epochs240/mstcn_556_hp:coarse3_reliable` | 0.7687 |
| 2 | `epochs240/mstcn_556_hp:onoff_reliable` | 0.8496 |

`index.html` shows only these; a "show all trained variants" checkbox reveals
the other 29. **Nothing deleted** — all 34 remain in the artifact repo.

## 3. FPS + hardware

`request_space_hardware(l4x1)` accepted; `hardware.requested` shows `l4x1`,
`current` still `t4-medium` until it migrates. L4 = 24 GB Ada, native bf16 (the
T4 is Turing and emulates it), and the headroom matters for task 4 — the VLM is
~8 GB on top of the detector.

Measured baselines to beat: **1.07 fps end to end on the t4-medium**; 4.2 fps
for precompute on an A100; `attention_runtime.yaml` records 5.77 fps at
detector_stride 3 / temporal_stride 2 and 7.03 at 5:3 with 92.2% cue agreement.

Still to do: re-measure on L4, then tune strides/resolution against measured
cue agreement rather than guessing.

## 4. VLM model — wired, tested, blocked

Infrastructure already exists and is tested but **not wired**:
* `attention/vlm_grounder.py` — `QwenGrounder` (Qwen3-VL-4B-Instruct, ~8 GB
  fp16, deliberately NOT the 27B that made the labels), `score_students(frame,
  boxes)`, option-likelihood scoring over one token per cue letter.
* `attention/fusion.py` — `fuse_student`, `fuse_frame`, `Policy`,
  `agreement_summary`.

Plan: expose as a synthetic registry entry (temporal model + VLM fusion), run
the VLM on the detector's boxes, fuse per `Policy`, and surface agreement in the
model card. It will be seconds per frame; that is expected and must be labelled.

## 5. Class performance — diagnosed, retraining next

**Read this before spending GPU.** `turned_to_peer` looks label-limited, not
capacity-limited, and the evidence is already in FINDINGS:
* two humans agree on the derived cue only **76%** of the time (kappa 0.711, §19);
* adding the causally-correct feature (head yaw/pitch/roll) moved it **+0.021**;
* 90 -> 240 epochs moved it **+0.01** (§18);
* AUPRC lift over base rate 4.7x against 11-14x for the classes that work.

`phone_use` and `using_laptop` are different: both are **object-presence**
questions currently answered by CLIP inference rather than by detection, and the
annotation booleans `phone_visible` / `laptop_visible` have inter-annotator
kappa 0.934 / 0.992 — i.e. humans agree almost perfectly, so the label is sound
and the feature is the weak link. That asymmetry is the thing to exploit.

Diagnose first: per-class error decomposition (confusion, coverage, whether
errors concentrate on frames where the pseudo-label and the human gold disagree)
then retrain only what it supports.

## Conventions that still bind

* **Never spend the test split** (`TEST_SPLIT_PROTOCOL.md`). `--split val`.
* Max 4 GPUs at once; long jobs in the background.
* Log every result and defect to `FINDINGS.md` as it happens.
* New sweeps go in a NEW `work_dirs/thesis/<name>/` so published checkpoints are
  never overwritten.


## Diagnosis result (2026-09-11) — read before spending GPU on task 5

Both weak classes fail the SAME way: `screen_oriented` is 76% of the data and
absorbs them. 54.4% of `turned_to_peer` frames and 38.9% of `phone_use` frames
are predicted `screen_oriented`, and it supplies 752/969 and 393/436 of their
false positives respectively.

They are NOT the same problem underneath:

* **`phone_use` — feature-limited, and therefore fixable.** `phone_visible` has
  inter-annotator kappa **0.934**: humans agree almost perfectly. The label is
  sound; CLIP at 224x224 over a whole-body crop cannot resolve a phone. The
  supported intervention is **explicit phone/laptop detection added to the
  feature vector**, then retrain. This is the highest-expected-value GPU spend.
* **`turned_to_peer` — partly label-limited.** Human agreement ~81% on the
  stratified subset against the model's 0.239, so there IS headroom, but head
  yaw bought +0.021 and 150 extra epochs bought +0.01. More of the same will not
  do it.

Next concrete step: add phone/laptop presence features (the detector is already
open-vocabulary — "a mobile phone", "a laptop" are free text prompts), rebuild
the affected feature block, retrain cue6 and cue9 at 240 epochs x 3 seeds in a
NEW work_dirs tree, and compare per-class against the epochs240 baseline.


## 2026-09-11 late: the phone_use plan changed shape (FINDINGS §21)

The detector **ignores its text prompt**. Fine-tuning on one category collapsed
the text conditioning: `qwertyuiop` retrieves students as well as
`a student sitting` does (92% of its boxes match a student box at IoU >= 0.9).
R@1 0.6462 is unaffected — it was always measured on students — but the
open-vocabulary description in docs/LABELS.md was wrong and is corrected.

So `phone_use` cannot be fixed by re-prompting. It needs a **second detector**,
and three pretrained open-vocab checkpoints are already on disk and unused:

    huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_*.pth

Obj365 has `cell phone` and `laptop`. Next concrete steps, in order:

1. Verify the pretrained swin-t DOES respond to prompts (same IoU-vs-nonsense
   test as §21 — do not assume it, the fine-tuned one looked fine too).
2. Measure phone/laptop detection rate on frames whose pseudo-label carries
   `phone_visible` / `laptop_visible`. If the objects are not findable at this
   resolution, the whole plan stops here and that is the finding.
3. Only then: per-student presence features (max score, containment, relative
   size, distance to box centre), extend the feature layout with a NAMED block,
   rebuild sequences, retrain cue6 + cue9 at 240 epochs x 3 seeds in a new tree.

Step 1 and 2 are cheap and decisive. Do them before any sequence rebuild.

## Hardware

Space is on **l4x1** (24 GB Ada, native bf16). The `~1 fps` and `125 s cold
start` figures in FINDINGS §16/20 were measured on **t4-medium** and should be
re-measured before being quoted again.


---

# 2026-09-11 end-of-session state

## Environment constraints discovered (important, not in CLAUDE.md)

* **GitHub and huggingface.co ARE reachable** from this box. CLAUDE.md says they
  are not; git push and HF uploads both work.
* **PyPI is NOT reachable.** `pip install` fails with DNS errors, so no
  dependency can be tested here. The Space's Docker build has network and can
  install; it just cannot be rehearsed locally first.

## Task 4 (VLM) — the real blocker, and the two user suggestions

The blocker is a **version conflict, not a capability one**: Qwen3-VL needs
`AutoModelForImageTextToText` (transformers >= 4.45) and both the HPC and the
Space pin **4.44.2** to protect mmcv's compiled `_ext` against torch 2.2.2.

**User suggestion 1 — "LoRA-train a base open-vocab VLM like Qwen."** Good, but
downstream: LoRA does not change what can be *loaded*. Order must be (a) get a
VLM that loads on this stack, (b) measure it, (c) LoRA only if the base is too
weak. Note the labels came from Qwen3.5-27B, so LoRA-ing a Qwen on those labels
makes the "independent second opinion" claim weaker, not stronger — if the point
is independence, adapt a non-Qwen base, or state plainly that it is an ensemble
rather than an independent check.

**User suggestion 2 — "make sure VLM inference is fast enough for live."** This
is the binding constraint and it changes the design. Arithmetic: a 4B VLM doing
one option-likelihood forward pass per student is roughly 0.2-0.5 s/student on an
L4; six students is 1.5-3 s per frame, against a pipeline that manages ~1-4 fps.
**Synchronous VLM cannot be live.** Four levers, in order of value:

1. **Run the VLM ASYNCHRONOUSLY** — its own thread at its own rate, and each
   frame fuses the most recent opinion available. `fusion.fuse_frame` already
   handles a student with no VLM row (fused temporal-only), so the data model
   supports this today; only the threading is missing. This is the fix.
2. **Batch all students into one forward pass** rather than one per student.
3. **A much smaller VLM.** `huggingface/my_llava-onevision-qwen2-0.5b-ov-2` is
   already on disk, already in the artifact repo, 0.5B, and proven to load on
   this exact stack (the detector ships it and never reads it at inference, so
   the weights are free). ~8x faster than a 4B.
4. **VLM stride** — currently 1 opinion per second of source; can be 1 per 2-5 s.

Current state: wiring complete and stub-tested, entry hidden where it cannot
load, refuses at selection with the reason. Nothing is half-applied.

## Task 5 (weak classes) — where it actually stands

Diagnosis done (§20.4), and the first intervention has been **tested and
rejected before spending GPU** (§21.2):

* The fine-tuned detector ignores its prompt entirely (§21) — so phone detection
  needs the pretrained `mm_grounding_dino` swin-t, which IS prompt-sensitive
  (§21.1: nonsense prompt returns 0 detections, phone boxes 10x smaller than
  person boxes, 0% overlap with students).
* But the **naive** presence feature does not discriminate: fires on 96.7% of
  `phone_visible=True` and **90.8%** of `phone_visible=False`. A 15% pad around a
  seated student reaches their neighbour in a dense classroom.

**Next step is NOT a retrain.** It is to make the geometry tighter and re-measure
the same 240-record test, which takes minutes:

* containment of the phone box in the student box, not centre-in-padded-box;
* phone-box area relative to student-box area (a real phone is ~0.1-1% of it);
* vertical position within the student box (a held phone sits low/central);
* the raw max score as a continuous feature, and its **AUROC** against
  `phone_visible` — that is the number that decides whether to proceed.

Only if AUROC clears roughly 0.75 is a sequence rebuild + 12-run retrain worth
it. The script to adapt is in this session's history; it samples 120 positive and
120 negative records one-per-src_frame from `labels_tracked.jsonl`, un-occluded.

## What is DONE and deployed

* frame cap removed (whole video; demo session re-cut 900 -> 1931 frames)
* dropdown shortlist, 5 of 34, nothing deleted
* Space on **l4x1**; the ~1 fps and 125 s cold-start figures were measured on
  t4-medium and must be re-measured before being quoted
* all 29 live models + calibrations on HF; registry taxonomy-aware
* threat C closed (kappa 0.800 / 0.711); label ceiling shown annotator-dependent


---

# Session 2 continuation — VLM made live-capable, phone feature validated

## VLM: the speed problem is SOLVED (task 4)

`AsyncGrounder` in `attention/vlm_grounder.py` runs any grounder on its own
thread. `submit()` returns immediately and replaces unstarted work (newest
wins); `latest()` returns the most recent opinion with its **age**, and expires
anything older than `max_age_s`. `session_replay` submits on a stride and reads
every frame.

Measured with a deliberately slow stub (0.4 s per student, i.e. a realistic 4B
VLM) over the 900-frame session:

    21.8 fps end to end, 95% of student-frames fused,
    opinion age median 2.9 s / max 4.4 s

Synchronously the same workload is ~0.4 fps. **The pipeline is now decoupled
from VLM speed entirely**, which is what "fast enough for live" required.
8 tests in `attention/tests/test_async_grounder.py`.

`LlavaOneVisionGrounder.available()` confirms the 0.5B LLaVA-OneVision already
on disk is loadable (vendored `llava` package, no transformers dependency). Its
`score_students` is NOT yet implemented — that is the remaining VLM work, and it
is the path that avoids the transformers 4.44.2 pin entirely.

## Phone feature: VALIDATED, retrain is justified (task 5)

Matched-pair test (both students from the same frame) gives:

    score x y_frac      AUROC 0.958
    y_frac              AUROC 0.892
    score_contained     AUROC 0.827
    rel_area            0.556   (drop)
    n_contained         0.270   (inverted, drop)

Physically sensible: a real phone sits 66% down the student box, false positives
cluster at 24% (head height).

**Two earlier versions of this test were wrong and both were caught**, which is
why the numbers above should be trusted more than a single run: the naive padded
-centre feature gave 5.9 points of separation (§21.2), and the first tightened
version was confounded by unmatched sampling and a default-value artefact
(§21.3). The tell was `n_contained` coming out inverted.

### Remaining work for phone_use, in order

1. Add a NAMED feature block to `thesis_eval/data.py` LAYOUTS — e.g. `v1080_obj`
   = v1074_head + 6 object dims — following the existing layout discipline
   (a width/layout mismatch must stay fatal by design).
2. Extend `attention/features.py` with the object pass: the PRETRAINED
   `mm_grounding_dino` swin-t, prompt `"cell phone. laptop."`, features
   `score_contained`, `y_frac`, `score*y_frac` per object class.
   **Do not use the fine-tuned detector — it ignores prompts (§21).**
3. Rebuild sequences (~40 GPU-min of CLIP over 284k crops; the object pass adds
   a second detector forward per frame).
4. Retrain cue6 + cue9, 240 epochs x 3 seeds, in a NEW work_dirs tree.
5. Compare per-class against `epochs240`; the number that matters is `phone_use`
   F1 against 0.541 (cue6) / 0.511 (cue9).

Expected cost: ~1 h rebuild + ~2 h training on 3 GPUs.

## Still open

* FPS re-measure on l4x1 (the 1 fps / 125 s figures are t4-medium).
* `turned_to_peer` — no validated intervention yet. It is partly label-limited
  (humans ~81%, model 0.239) and neither head yaw (+0.021) nor 150 extra epochs
  (+0.01) moved it. Do NOT retrain it speculatively.
* `laptop_visible` — no signal on the unmatched test; re-run matched before
  concluding.
