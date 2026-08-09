# Repository and data audit — Branch C (`branch-c/overt-cue`)

**Scope:** read-only audit of `grounding_data/`, `LLMDet/work_dirs/`, `huggingface/`,
splits, and feature dimensions. No file outside this document and
`docs/branch_c/ARTIFACT_MANIFEST.json` was written. No GPU used; all commands below
ran on CPU. No prediction or metric was computed on the Branch-B test split or on any
Branch-C outer fold — split-membership counts and file/manifest metadata only.

Every number below cites the exact command or file that produced it. Where a number
could not be independently verified from disk it is marked **UNVERIFIED** with what
would close it.

---

## 0. How this was produced

- `find`, `du -sb`, `wc -l`, `sha256sum` for sizes/counts/hashes.
- One Python pass (`json.load`) over each `meta.json` / `splits.json` /
  `llmstu_seq_split_manifest.json` / `branch_c_folds.json`.
- One full streaming pass over `labels_slim.jsonl` (283,913 records, 2.4 s wall —
  no sampling was needed; see §5).
- `torch.load(..., map_location='cpu', weights_only=False)` on 6 representative
  `work_dirs/thesis/*` checkpoints to read the `spec`/`model` dict and first-layer
  weight shapes — CPU only, no GPU touched.
- `sha256sum` on every checkpoint file under 500 MB (443 under `LLMDet/work_dirs`,
  15 under `huggingface/`); size + mtime only for the rest. Total wall time for all
  of the above: under 6 minutes.

---

## 1. Data directories

| directory | on-disk size (`du -sb`) | file counts | purpose |
|---|---|---|---|
| `grounding_data/LLMSTU` | 11,398,688,264 B (11.4 GB… reported 12 GB by `du -sh`) | 283,914 `.jpg`, 22 `.jsonl` | Raw per-student crop corpus + its label shards + the merged `labels_slim.jsonl` |
| `grounding_data/llmstu_sequences` | 555,957,781 B | 6,531 `.npz` + 1 `meta.json` | 552-dim temporal sequences, legacy 7-class taxonomy, legacy 80/20 split |
| `grounding_data/llmstu_sequences_full` | 569,394,494 B | 6,531 `.npz` + 1 `meta.json` | 570-dim sequences (base+headpose+expression+dynamics), FaceLandmarker backend |
| `grounding_data/llmstu_sequences_full_bp` | 569,381,559 B | 6,531 `.npz` + 1 `meta.json` | Same as above, `bbox_person` face-detection variant — **not cited by the thesis** |
| `grounding_data/llmstu_sequences_full_det` | 565,904,580 B | 6,531 `.npz` + 1 `meta.json` | Same as above, BlazeFace-detector variant — **currently deployed** feature backend |
| `grounding_data/llmstu_sequences_hp` | 558,557,405 B | 6,531 `.npz` + 1 `meta.json` | 556-dim sequences (base+headpose only), backs the 552/553/555/556 column-slice ablation |
| `grounding_data/llmstu_tools` | 925,010,089 B | 19 `.jsonl`, 13 `.py`, 8 `.json`, 6 `.npz`, misc | Dataset-prep pipeline (dedup, video-ID recovery, splits, ODVG regen) |
| `grounding_data/stu_img` | 37,567,980,659 B | 104,069 `.jpg` (in `frames/`) + others | Raw video frames (`frames/`, 36.9 GB) that everything traces back to; also holds the deprecated 4-class `attention_sequences/` (6.9 MB) |

Commands: `du -sb <dir>`; `find <dir> -type f | wc -l`; `find <dir> -type f | sed -E 's/.*\.//' | sort | uniq -c`.

### 1.1 Schema of the primary file in each directory (real records)

**`LLMSTU/labels_slim.jsonl`**, first record (`python3 -c "import json; print(json.loads(open(...).readline()))"`):

```json
{
  "src_frame": "t000000_000_f000000.jpg", "person_idx": 0,
  "bbox_person": [905.34, 86.45, 1223.02, 422.56], "det_conf": 0.9250,
  "head_kpts": 4, "face_kpts": 3, "head_span_px": 156.75,
  "bbox_crop": [1005, 106, 1241, 422],
  "activity": "using_laptop", "gaze_direction": "laptop",
  "attention_target": "device", "engagement_level": "engaged",
  "posture": "upright", "hand_state": "unknown",
  "phone_visible": false, "laptop_visible": true, "talking": false,
  "occluded": true,
  "caption": "A student is looking down at a laptop screen with a focused expression.",
  "model_confidence": 0.9,
  "file_name": "shard_000/part_000/t000000_000_f000000__p00.jpg"
}
```

**`llmstu_sequences*/train/sample_000000.npz`** keys (`np.load(..., allow_pickle=True).files`):
`x` (float32, `[64, D]`), `y_frames` (int64, `[64]`, per-frame class id), `y` (int64
scalar, majority label), `t` (float64, `[64]`, timestamps). `D` is 552/570/570/570/556
for `llmstu_sequences` / `_full` / `_full_bp` / `_full_det` / `_hp` respectively
(confirmed by `z['x'].shape` on the first sample of each dir).

**`llmstu_sequences*/meta.json`** top-level keys: `num_samples` (6531 everywhere),
`class_names`, `label_source` (`"llmstu_structured_fields"`), `split`
(`{train_videos, val_videos}` — **note: no `test_videos` key**, this is the legacy
80/20 split baked in at build time, see §3), `seed` (42), `samples` (list of
`{file, video_id, seat_id, length, label_majority, split}`).

⚠️ **`llmstu_sequences/meta.json`** carries 7 `class_names` (adds `idle_other`);
`_full`/`_full_bp`/`_full_det`/`_hp` all carry the current 6-class
`attention.taxonomy.CUE_CLASSES` (`screen_oriented, looking_away, head_down,
turned_to_peer, phone_use, uncertain`).

**`llmstu_tools/outputs/splits.json`**: `{"train": [...73 video ids...], "val": [...27...], "test": [...27...]}`.

**`llmstu_seq_split_manifest.json`** top-level keys: `source_meta`
(`"grounding_data/llmstu_sequences_full/meta.json"`), `split_source`
(`"grounding_data/llmstu_tools/outputs/splits.json"`), `note` (quoted verbatim in §3),
`counts` (`{sequences,frames,videos} x {train,val,test}`), `samples` (list of
`{file, video_id, seat_id, length, split, old_split}` — 6,531 entries, one per NPZ).

### 1.2 What distinguishes `llmstu_sequences` vs `_full` vs `_full_bp` vs `_full_det` vs `_hp`

Read from `LLMDet/attention/sequence_builder.py` (`build_sequences_llmstu`,
lines ~270–412) and the four build logs under `LLMDet/work_dirs/{attention_temporal_v2,logs}/`:

| dir | builder flags (from build logs / spec dicts) | feature dim | face-detection backend | class taxonomy | status |
|---|---|---|---|---|---|
| `llmstu_sequences` | plain (`--format llmstu`, no head-pose/affect/dynamic) | 552 | none | 7-class (legacy, `idle_other`) | superseded (§3) |
| `llmstu_sequences_hp` | `--head-pose-backend cached` (`build_sequences_hp.log`: `feature dim: 556 (head_pose=cached)`) | 556 | cached FaceLandmarker | 6-class | backs the head-pose column-slice ablation |
| `llmstu_sequences_full` | `--affect-cache ... --dynamic-features` (`build_sequences_full.log`: `feature dim: 570 (head_pose=off)`) | 570 | FaceLandmarker mesh, via `bbox_crop` | 6-class | backs `ladder`/`arch`/`headpose`/`calibration`/`deploy_sim` checkpoints |
| `llmstu_sequences_full_bp` | same builder, affect cache rebuilt on `bbox_person` face crops (FINDINGS §11.15) | 570 | FaceLandmarker mesh, via `bbox_person` | 6-class | diagnostic only — **not cited** |
| `llmstu_sequences_full_det` | same builder, affect cache rebuilt with the BlazeFace full-range detector (FINDINGS §11.16) | 570 | BlazeFace detector, via `bbox_crop` | 6-class | backs the **deployed** `ff_det/mstcn_553_ff_*` checkpoints |

`meta.json` for `_full`, `_full_bp`, `_full_det`, and `_hp` are **byte-identical**
(`sha256 14a7d104e742deed39d2d62a350c5cf482a809a060751ad32e9ef3c89aca0302` for all
four — verified with `sha256sum`) — same tracks, same per-sample split assignment,
same sample ordering. Only the `x` columns inside the NPZ files differ between them.
`llmstu_sequences/meta.json` differs (`sha256 5a3dcc6b...`) because it used a
different, 7-class taxonomy.

---

## 2. Feature dimensions — 552 / 553 / 555 / 556 / 563 / 570

Source: `LLMDet/attention/features.py` (`StudentFeatureExtractor`), `attention/head_pose.py`
(`HeadPoseEstimator`, `OUTPUT_DIM = 4`), `attention/sequence_builder.py` lines 287–376
(affect-cache and dynamic-feature concatenation), and FINDINGS §11.4's column-layout
statement, cross-checked against checkpoint `spec['feature_config']` and the first-layer
weight shape of six loaded checkpoints (see §4.2).

| block | column range | dims | source | first appears in / consumed by |
|---|---|---|---|---|
| CLIP image embedding | `[0:512]` | 512 | `StudentFeatureExtractor._clip_batch` — `openai/clip-vit-base-patch32`, L2-normalised | all configs |
| bbox geometry | `[512:520]` | 8 | `._geom` — cx, cy, area, aspect ratio, left/right/top/bottom (frame-normalised) | all configs |
| color statistics | `[520:544]` | 24 | `._color_stats` — per BGR channel: mean, std, p25, p50, p75, min, max, frac>200 (3×8) | all configs |
| posture-geometry proxies | `[544:552]` | 8 | `._posture_geom` — elongation, vertical intensity centroid, top/bottom mass share, head x-offset, head/torso contrast, box top position, box width fraction | all configs |
| **= base block** | `[0:552]` | **552** | sum of the above | `transformer_552_base_*`, `arch/*_570_full` (as a slice), `headpose/*` (as a slice) |
| head pose | `[552:556]` | 4 | `attention/head_pose.py` `HeadPoseEstimator` — yaw, pitch, roll, `face_found` (binary) | `*_556_hp`, `llmstu_sequences_hp`, `llmstu_sequences_full*` cols 552-555 |
| — `face_found` only | col `555` | 1 | column-slice of the head-pose block | `*_553_facefound`, `*_553_ff` (ablation / deployed) |
| — angles only | cols `552:555` | 3 | column-slice of the head-pose block | `*_555_angles` (ablation) |
| facial expression | `[556:563]` | 7 | `sequence_builder.py` `--affect-cache` (precompute_affect.py; FER-ViT-derived, see §4.3) | `*_563_expr` (ablation slice of the 570-dim array) |
| dynamics / gaze | `[563:570]` | 7 | `attention/dynamic_features.py` `compute_dynamic` — whole-track fidget/lean motion stats + personalised gaze deviation from the head-pose columns | `*_563_dyn` (ablation slice), `*_570_full` |
| **= full block** | `[0:570]` | **570** | base + head pose + expression + dynamics | `llmstu_sequences_full*`, `*_570_full` |

Confirmed independently by loading six checkpoints on CPU (§4.2): first-layer input
widths were exactly 552, 556, 570, 556 (MS-TCN), 556 (ASRF), 553 (MS-TCN, `ff_det`) —
matching `spec['feature_config']` in every case with no mismatch.

**553/555/563 are never separate NPZ builds** — every one of them is a `numpy`
column-slice taken from the 556-dim or 570-dim arrays at training/eval time (this is
explicit in FINDINGS §11.4: *"the 570-dim NPZ column layout... makes each rung a pure
column slice of one array"*), which is also why `llmstu_sequences_full{,_bp,_det}`'s
`meta.json` files are byte-identical to `llmstu_sequences_hp`'s — the underlying
tracks and split never change between rungs of the ladder.

---

## 3. Splits and exposure — reconciliation (required deliverable)

### 3.1 Three splits exist, not one

| split | file | grouping | status |
|---|---|---|---|
| **legacy Branch-B split** | baked into `llmstu_sequences*/meta.json['split']` (`train_videos`/`val_videos` only, no test key) | 80/20 by `sequence_builder.py`'s own shuffle, **independent of** `splits.json` | superseded, kept only for provenance |
| **leak-free Branch-A/B split** | `grounding_data/llmstu_tools/outputs/splits.json` (73/27/27) | video-wise, produced by `make_splits.py` | **spent** — B's test set opened once (§3.3); A's test set opened once (`TEST_SPLIT_PROTOCOL.md`) |
| **Branch-C nested CV** | `outputs/branch_c/splits/branch_c_folds.json` | 5 outer folds x 1 inner holdout, all 127 videos, `video_id` grouping | frozen 2026-08-09, `BRANCH_C_PROTOCOL.md`, not yet opened |

### 3.2 The reconciliation

The task names two claims that look contradictory:

> (a) FINDINGS §9 item 14: *"23 of the 27 test videos are in the cue model's training
> set... macro-F1 0.4098 is labelled validation."*
>
> (b) `llmstu_seq_split_manifest.json`'s `note` field: *"Branch-B sequences
> re-partitioned onto the DETECTOR's leak-free video-wise 73/27/27 split so both
> branches share one split and Branch B gains an untouched test set."*

**Both are true statements about different points in time, and (a) is stale.**

1. `attention/sequence_builder.py:213` drew its own 80/20 video shuffle and never read
   `splits.json`. Under that legacy split, 23 of the detector's 27 held-out test
   videos sat inside the cue model's training set (`FINDINGS §3.4c`,
   `TEST_SPLIT_PROTOCOL.md`, `BRANCH_B_TEST_PROTOCOL.md`, all dated **2026-08-01**).
   Under that regime, macro-F1 **0.4098** (556-dim, single-process measurement,
   FINDINGS §6f) was correctly labelled a *validation* number, and both protocol
   files pre-registered that **no Branch-B test number would be reported** for
   exactly this reason.
2. Also on **2026-08-01**, FINDINGS §11.1 records that the sequences were
   re-partitioned "by metadata alone" (video_id is recorded per-sample in every
   `meta.json`) onto `splits.json`'s 73/27/27, producing
   `llmstu_seq_split_manifest.json` — verified on disk:
   `counts = {sequences: {train:4390, val:1085, test:1056}, frames: {train:185050,
   val:42702, test:43733}, videos: {train:73, val:27, test:27}}`, matching FINDINGS
   §11.1's table exactly. `source_meta` and `split_source` fields in the manifest
   point to `llmstu_sequences_full/meta.json` and `splits.json`, and their sha256 in
   the manifest (`14a7d104...`, `ea72da5a...`) **match the files currently on disk**
   (`sha256sum`, verified in this audit). All models were then **retrained from
   scratch** on the new partition (FINDINGS §11.1: *"the archived 552/556/570
   checkpoints cannot be evaluated here because they trained on 23 of the 27 test
   videos"*).
3. `BRANCH_B_TEST_PROTOCOL.md` (registered 2026-08-01, **before** the retrained
   models saw the test split) pre-registers exactly this fix and its scope, and
   FINDINGS §11.9 reports the resulting **held-out test** macro-F1 numbers on the new
   split: transformer-556 0.407±0.004, ASRF-556 0.477±0.016, **MS-TCN-556
   0.500±0.011** — a real, previously-untouched test result, not a re-labelled
   validation number.
4. FINDINGS §9's own **Closed** list, item 7, already states this: *"Branch B has no
   test split — fixed 2026-08-01, §11.1."* Item 14, added eight days later
   (2026-08-09) in the same document's **Still open** list, restates the pre-fix
   problem verbatim (23/27 leak, macro-F1 0.4098) as if it were still current. It
   contradicts item 7 in the same file, and it contradicts §11.1/§11.9/
   `BRANCH_B_TEST_PROTOCOL.md`, all of which predate it and already closed exactly
   this gap with a real test number.

**Current state, as of this audit:** the leak-free rebuild is real, verified, and
closed. FINDINGS §9 item 14 is a stale re-statement of an already-fixed problem and
should be corrected or removed by whoever next edits FINDINGS.md (not this agent —
FINDINGS.md is out of scope for this audit). **This document does not modify
FINDINGS.md; it records the contradiction and its resolution for the record.**

### 3.3 A second, more recent wrinkle: the Branch-B test split is now itself spent for Branch C

`BRANCH_C_PROTOCOL.md` §1.2 (frozen 2026-08-09, i.e. *after* §11.9's test run) states
plainly: *"That number is real and stays in the record. It cannot become an untouched
test for a new architecture, and no Branch-C selection decision may consult it."*
This is why Branch C does not reuse `splits.json`'s test partition and instead froze
its own independent nested-CV protocol (`branch_c_folds.json`) over **all 127 videos**
(no held-out video subset exists — `frame_to_video.json` has 128 videos, `splits.json`
has 127, the one remaining video has 4 frames — verified: `set(splits.json) ^
set(frame_to_video.json) = {video_0162_0_10_20251026030716_20251026032847}`, 4 frames,
matching FINDINGS §12.2 exactly).

**Provenance chain verified end-to-end in this audit** (`sha256sum`):

```
branch_c_folds.json['manifest_sha256']            = fd913e7c5a52839ee5115224ffdb741c09cc1104eeb1fee5d32b9595e2388314
branch_c_folds.json['sources']['legacy_splits']['sha256'] = ea72da5aa2260563c25e5d40ba18acdeac500c243b1c29365fbcfab1c367e9bd
  == sha256sum grounding_data/llmstu_tools/outputs/splits.json           (MATCH)
branch_c_folds.json['sources']['meta']['sha256']           = 14a7d104e742deed39d2d62a350c5cf482a809a060751ad32e9ef3c89aca0302
  == sha256sum grounding_data/llmstu_sequences_full/meta.json           (MATCH)
```

### 3.4 What has been opened/reported already (never to be reopened as a test)

- Branch-A detector test split (27 videos, 9,405 frames): opened once, R@1 0.6462,
  `TEST_SPLIT_PROTOCOL.md`.
- Branch-B cue-model test split (same 27 videos, 1,056 sequences, 43,733 frames):
  opened once, macro-F1 up to 0.500±0.011 (MS-TCN-556), `BRANCH_B_TEST_PROTOCOL.md`,
  FINDINGS §11.9.
- Neither may be consulted by any Branch-C selection decision
  (`BRANCH_C_PROTOCOL.md` §2.4, §1.2). Branch C's own outer folds are unopened as of
  this audit (2026-08-09) — `outputs/branch_c/RUNS.jsonl` does not exist yet
  (checked: `find outputs/branch_c -iname 'RUNS.jsonl'` → no result), consistent with
  no Branch-C model having produced a scored result yet.

---

## 4. Checkpoints

### 4.1 Inventory

`find LLMDet/work_dirs huggingface \( -iname '*.pth' -o -iname '*.pt' \)`:
**443** files under `LLMDet/work_dirs` (166 GB) + **15** files under `huggingface/`
(≈6.9 GB). Full per-file table (path, bytes, sha256-or-skipped, mtime) is in
`docs/branch_c/ARTIFACT_MANIFEST.json` (487 artifacts total, 458 of them checkpoints).
Summary by family:

| family | dir | # files | total size | consumer | status |
|---|---|---|---|---|---|
| Branch-A detector, main+bundle | `work_dirs/thesis_bundle/checkpoints/` | 11 | ~42 GB (10 files ≈4GB detector ckpts + 1 148MB `temporal_best.pth`) | A (test-registered) / temporal_best.pth is legacy, unconsumed | test-frozen |
| Branch-A detector, other runs | `work_dirs/student_llmstu_exact/`, `grounding_dino_swin_t*/`, `student_ablation_*/` | 21 | large (4GB-class .pth each) | A | training archive |
| **Legacy per-epoch temporal archive** | `work_dirs/attention_temporal{,_full,_hp,_v2}/checkpoints/` | 304 | ≈45 GB | **none** — superseded by the leak-free rebuild (§3.2) | superseded |
| Branch-B leak-free rebuild — feature ladder | `work_dirs/thesis/ladder/*/checkpoints/` | 18 (9 dirs × best+last) | ~900 MB (50MB each) | B | live (FINDINGS §11.3-11.9) |
| Branch-B leak-free rebuild — architecture | `work_dirs/thesis/arch/*/checkpoints/` | 24 | ~340 MB | B | live |
| Branch-B — head-pose column-slice ablation | `work_dirs/thesis/headpose/*/checkpoints/` | 8 | ~400 MB | B | live |
| Branch-B — `bbox_person` diagnostic (not cited) | `work_dirs/thesis/posefix/*/checkpoints/`, `work_dirs/thesis/ff_bp/*/checkpoints/` | 36+12=48 | ~800 MB | **none** — explicitly not cited (§11.15) | diagnostic archive |
| Branch-B — deployed detector-backend variant | `work_dirs/thesis/ff_det/*/checkpoints/` | 12 | ~180 MB | B (deployed) | live |
| Branch-B calibration/deploy-sim artifacts | `work_dirs/thesis/{calibration,deploy_sim}/*` | ~9 | small | B | live |
| HF pretrained detector backbones | `huggingface/mm_grounding_dino/*.pth` | 3 | 3.7 GB | A (init weights) | external release |
| HF fine-tuned FER model, optimizer state | `huggingface/fer_vit/checkpoint-*/{optimizer.pt,rng_state.pth,scheduler.pt}` | 12 | ~2.75 GB (mostly `optimizer.pt`) | B,C (backs the expression feature block) | live; **model weights themselves are `.safetensors`, not `.pt`, so they are outside the .pth/.pt scope of this section** |

Command basis: `find ... -exec du -b {} \;`, then bucketed by path substring;
`sha256sum` run on every file `< 500MB`; size+mtime recorded for the rest (all of
which are the 4GB-class detector checkpoints, the 3 HF backbone `.pth` files, and
the 4 `fer_vit/checkpoint-*/optimizer.pt` files at 655-687MB).

### 4.2 Loadable metadata (CPU, `weights_only=False`, `map_location='cpu'` — no GPU touched)

Six representative `work_dirs/thesis/*` checkpoints were loaded to confirm feature
width without trusting the directory name:

| checkpoint | `spec['feature_config']` | `spec['sequence_root']` | first-layer weight shape | epoch |
|---|---|---|---|---|
| `ladder/transformer_552_base_s42/checkpoints/best.pth` | `552_base` | `llmstu_sequences_full` | `net.input_proj.weight` = `[512, 552]` | 85 |
| `ladder/transformer_556_hp_s42/checkpoints/best.pth` | `556_hp` | `llmstu_sequences_full` | `[512, 556]` | 88 |
| `ladder/transformer_570_full_s42/checkpoints/best.pth` | `570_full` | `llmstu_sequences_full` | `[512, 570]` | 78 |
| `arch/mstcn_556_hp_s42/checkpoints/best.pth` | `556_hp` | `llmstu_sequences_full` | `stage1.inp.weight` = `[128, 556, 1]` | 89 |
| `arch/asrf_556_hp_s42/checkpoints/best.pth` | `556_hp` | `llmstu_sequences_full` | `backbone.inp.weight` = `[128, 556, 1]` | 71 |
| `ff_det/mstcn_553_ff_s42/checkpoints/best.pth` | `553_facefound` | **`llmstu_sequences_full_det`** | `stage1.inp.weight` = `[128, 553, 1]` | 89 |

Every `spec['sequence_root']` for the `ladder`/`arch`/`headpose` families points at
`llmstu_sequences_full` (570-dim on disk; the checkpoint's own feature-width column
slice is applied at load time — consistent with §2's "no separate NPZ per rung"
finding). The `ff_det` family's `sequence_root` explicitly differs
(`llmstu_sequences_full_det`), confirming §1.2's claim that the deployed 553-dim
model is trained on the BlazeFace-detector variant, not the FaceLandmarker one.

`thesis_bundle/checkpoints/temporal_best.pth` top-level keys are
`{epoch, model, optimizer, config}`; `config['model']['input_dim'] = 544`,
`config['model']['num_classes'] = 4`, `config['data']['sequence_dir'] =
'../grounding_data/stu_img/attention_sequences'`. This confirms it is the
**deprecated legacy 4-class / 544-dim model**, entirely disjoint from the current
6-class 552-570-dim family, and not consumed by anything in scope for Branch B or C.

### 4.3 Not verified

- Which exact commit/script produced each of the 304 legacy per-epoch checkpoints in
  `attention_temporal*` — the `build_sequences*.log` files establish which *sequence
  dir* fed which family, but not a 1:1 training-script commit hash per checkpoint.
  **UNVERIFIED** — would need `git log --follow` on `attention/train.py` cross-referenced
  against checkpoint mtimes.
- The exact producer of `huggingface/fer_vit`'s weights (model.safetensors is not a
  `.pth`/`.pt` file and was out of this section's scope; `precompute_affect.py`
  presumably loads it for the `[556:563]` expression columns but this was not traced
  line-by-line). **UNVERIFIED** — would need reading `precompute_affect.py`.

---

## 5. Geometry quantification — `LLMSTU/labels_slim.jsonl`

**Full pass, no sampling**: all 283,913 records streamed and parsed in 2.4 s
(`python3 -c "..."`, script retained at
`/tmp/.../scratchpad/analyze_labels_slim.py` for reproducibility — not committed to
the repo, per the "don't copy large data / don't create files outside scope" rule;
the analysis code itself is small enough to re-paste from this document if needed).

### 5.1 Continuous fields

| field | p0 | p5 | p25 | p50 | p75 | p95 | p100 | mean |
|---|---|---|---|---|---|---|---|---|
| `head_span_px` | 120.0 | 122.3 | 132.6 | 149.6 | 190.3 | 268.3 | 892.4 | 167.8 |
| `det_conf` | 0.350 | 0.686 | 0.880 | 0.911 | 0.926 | 0.940 | 0.968 | 0.881 |
| `bbox_person` width (px) | 90.8 | 148.9 | 214.4 | 280.0 | 393.4 | 510.4 | 1269.2 | — |
| `bbox_person` height (px) | 60.3 | 231.8 | 316.6 | 374.2 | 462.4 | 612.2 | 1050.0 | — |
| `bbox_person` area (px²) | 9,569 | 39,608 | 68,473 | 100,332 | 181,273 | 298,048 | 1,110,185 | — |

(Frame is 2812×1050 = 2,952,600 px²; median person box is ~3.4% of frame area.)

### 5.2 Rates

| quantity | value |
|---|---|
| `occluded == true` | **31.03%** (88,072 / 283,913) |
| `head_kpts > 0` | **100.00%** (283,913 / 283,913) |
| `face_kpts > 0` | **100.00%** (283,913 / 283,913) |
| distinct `src_frame` values | 93,910 |
| students per frame — median / mean / p95 / max | 3 / 3.02 / 6 / 8 |

`head_kpts` value distribution: `{2: 10, 3: 29167, 4: 235332, 5: 19404}`.
`face_kpts` value distribution: `{2: 29169, 3: 254744}` — **`face_kpts` is never 0 or
1 in this file**; it is a small integer keypoint-count field (2 or 3), always
populated.

### 5.3 Face/head-keypoint availability by activity class

| activity | count | `face_kpts>0` rate | `head_kpts>0` rate |
|---|---|---|---|
| `listening` | 111,365 | 100% | 100% |
| `using_laptop` | 94,411 | 100% | 100% |
| `head_down_sleeping` | 20,569 | 100% | 100% |
| `looking_away` | 17,047 | 100% | 100% |
| `reading` | 10,782 | 100% | 100% |
| `other` | 10,419 | 100% | 100% |
| `using_phone` | 9,194 | 100% | 100% |
| `talking_to_peer` | 6,215 | 100% | 100% |
| `writing_notes` | 2,232 | 100% | 100% |
| `eating_drinking` | 1,558 | 100% | 100% |
| `raising_hand` | 121 | 100% | 100% |

### 5.4 What this does and does not tell us about the `face_found` shortcut hypothesis

**This is an important negative result, not a null finding to skip past.**
`labels_slim.jsonl`'s `face_kpts`/`head_kpts` fields are saturated (always ≥2) across
every activity class, so **they cannot be the source of, or a test of, the 92%-vs-8%
`face_found` contrast** that FINDINGS §11.10/§12.1 documents between `screen_oriented`
and `head_down`. That contrast is measured by an entirely separate, later pipeline
stage — `attention/head_pose.py`'s `HeadPoseEstimator` (FaceLandmarker or BlazeFace)
run on `bbox_crop`/`bbox_person` crops at feature-extraction time, cached in
`llmstu_tools/outputs/face_found_cache_detector.npz` /
`head_pose_cache_bbox_person.npz` — **not** in `labels_slim.jsonl`. `head_kpts`/
`face_kpts` in `labels_slim.jsonl` appear to be a fixed-schema annotation field from
the LLMSTU labeling pass (a small integer count, likely a pose/landmark-detector
confidence tier from the *original* per-crop annotation, not the downstream
`face_found` binary), not the shortcut signal itself.

**Consequence for the audit's task 5:** the geometry file confirms occlusion
(31.0%), det_conf, box-size and students-per-frame distributions cleanly, but
**cannot** be used to test the `face_found` shortcut hypothesis by activity class —
that test requires the separate head-pose caches under `llmstu_tools/outputs/*.npz`,
which were not re-computed in this audit (doing so would mean running a detector —
out of scope for a read-only, CPU-only, no-GPU audit). **UNVERIFIED as a shortcut
test; VERIFIED as a statement about what labels_slim.jsonl does and does not
contain.** What would close it: load `face_found_cache_detector.npz` and
`llmstu_sequences_hp`'s NPZ column 555, group by the per-sample `label_majority` in
`meta.json`, and report the by-class face_found rate directly — FINDINGS §11.10
already reports this exact number (92% `screen_oriented` vs 8% `head_down`) from that
separate pipeline, and it should not be re-derived from `labels_slim.jsonl`.

---

## 6. `docs/branch_c/ARTIFACT_MANIFEST.json`

487 entries: 458 checkpoints (path, bytes, sha256-or-`SKIPPED_OVER_500MB_size_and_mtime_only`,
producer, consumer_branch, access_class, mtime), 8 dataset directories, 7 docs/protocols,
6 metadata files, 4 code pointers, 3 split files, 1 labels file. Every entry's
`sha256` field is either a verified hash (computed in this audit) or an explicit
skip marker — no field was left silently blank. `access_class` is `public` for
markdown docs already in git, `restricted` for split/manifest files that gate
what a Branch-C model may read, and `local-only` for the multi-GB data/checkpoint
trees that must never be copied into git.

---

## Summary

### (a) Three most important things verified

1. **The §9-item-14 vs. manifest-note contradiction is resolved, not real.** The
   leak-free Branch-B rebuild (73/27/27 video-wise, `llmstu_seq_split_manifest.json`)
   happened and was scored on 2026-08-01 (FINDINGS §11.1/§11.9, `BRANCH_B_TEST_PROTOCOL.md`),
   producing a genuine held-out test macro-F1 (MS-TCN-556: 0.500±0.011). FINDINGS §9
   item 14 (added 2026-08-09) incorrectly restates the pre-fix state and contradicts
   item 7 in the same section. Verified end-to-end by independently re-hashing
   `splits.json` and `llmstu_sequences_full/meta.json` and matching both against the
   sha256 values recorded inside `llmstu_seq_split_manifest.json` and
   `branch_c_folds.json`.
2. **552→556→563→570 is one column-sliced array, not five separate feature
   extractions.** Confirmed three independent ways: (i) `features.py` +
   `sequence_builder.py` source, (ii) `llmstu_sequences_full{,_bp,_det}` and `_hp`
   all having byte-identical `meta.json` (same tracks/splits, different NPZ columns
   only), (iii) six loaded checkpoints' `spec['sequence_root']` all pointing at the
   same 570-dim (or 570-dim-derived-det) source directory regardless of their
   `feature_config` name, with first-layer weight shapes matching the claimed width
   exactly.
3. **Branch C cannot reuse the Branch-B test split, and has its own frozen protocol.**
   `BRANCH_C_PROTOCOL.md` (frozen 2026-08-09) registers a 5-outer-fold nested grouped
   CV over all 127 videos precisely because the Branch-B/A test split is now spent.
   The fold manifest's self-reported hash (`fd913e7c...`) and its two source-file
   hashes were independently re-verified against the files on disk in this audit —
   full match, no drift.

### (b) Every contradiction found between documents

1. **FINDINGS §9 item 14 vs. FINDINGS §9 item 7 / §11.1 / §11.9 / `BRANCH_B_TEST_PROTOCOL.md`**
   — detailed in §3.2 above. Item 14 is stale; the rebuild it describes as pending
   was already completed and scored eight days before item 14 was written.
2. **`llmstu_tools/README.md` says splits are "video-wise (70/15/15)"**, but the
   actual `splits.json` realises 73/27/27 (train/val/test, not 70/15/15 by any
   rounding of 127 videos). Minor — likely the README documents the intended target
   ratio, not the realised one — but worth fixing since a reader would expect the
   two to match.
3. **`llmstu_sequences/meta.json`'s taxonomy (7 classes, includes `idle_other`) vs.
   every other sequence dir and `attention/taxonomy.py` (6 classes)** — not a
   documentation contradiction so much as a silent schema drift between the earliest
   and all later sequence builds; anyone loading `llmstu_sequences` alone would get a
   different label space than every other artifact in the repo.

### (c) What could not be verified, and what would close it

1. **The `face_found`-shortcut-by-activity-class test cannot be run from
   `labels_slim.jsonl`** (§5.4) — its `face_kpts`/`head_kpts` fields are saturated
   and are not the same signal as the downstream `face_found` head-pose feature.
   Closing it means reading `llmstu_tools/outputs/face_found_cache_detector.npz` (or
   the `llmstu_sequences_hp` NPZ column 555) against `meta.json`'s per-sample
   `label_majority` — already done once, elsewhere, in FINDINGS §11.10, and should be
   cited from there rather than re-derived.
2. **Per-checkpoint producer commit hashes for the 304 legacy `attention_temporal*`
   files** — family-level attribution (which sequence dir fed which family) is
   solid; a specific git commit per checkpoint is not established.
   `git log --follow LLMDet/attention/train.py` cross-referenced with checkpoint
   mtimes would close this, if it is ever needed (these checkpoints are superseded
   and unlikely to matter).
3. **The exact producer of `huggingface/fer_vit`'s model weights** and its precise
   wiring into the `[556:563]` expression columns — `precompute_affect.py` was
   identified as the likely bridge from `sequence_builder.py --affect-cache`'s
   docstring but not read line-by-line to confirm.
4. **`llmstu_tools/README.md`'s "70/15/15" claim** (item b.2 above) — not
   independently reconciled against `make_splits.py`'s actual ratio parameter; could
   be closed by reading that script's default arguments.
