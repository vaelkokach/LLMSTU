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
| 3 | More FPS + hardware upgrade | l4x1 requested and accepted; migrating. Tuning pending |
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
