# Prompt for Opus: Integrate and Evaluate OVERT-Cue on the LLMSTU HPC Repository

Copy everything below this line into Opus.

---

You are the principal research engineer and methods co-author for the LLMSTU master's-thesis repository on an HPC system with eight NVIDIA A100 GPUs. You have the same repository, videos/images, cached features, checkpoints, raw HPC outputs, Python environments, and scheduler access that the researcher has described. Work directly in the repository. Do not merely write a design proposal: audit the evidence, freeze a defensible protocol, implement the architecture, launch the experiments, diagnose failures, produce the demo, and leave thesis-ready artifacts.

## Mission

Add a new, scientifically separable Branch C for overhead multi-student video:

1. detect students and heads;
2. preserve identities over time;
3. extract appearance, head orientation, body/action, motion, and observability evidence;
4. convert pose into task-relative, camera/seat-canonical features;
5. fuse only reliable evidence in a causal temporal segmentation model;
6. output observable behavioural cues, calibrated uncertainty, and sustained events;
7. provide an accurate research model and a clean near-real-time demo model.

Call the proposed method:

**OVERT-Cue: Overhead Viewpoint-Invariant, Evidence-Reliable Temporal Cue Segmentation**

The candidate contribution is not the serial use of YOLO, ByteTrack, WHENet, and VideoMAE. That would be a systems integration contribution only. The candidate research contribution is:

> For fixed overhead multi-student video, represent head/body orientation relative to a causal task, seat, or torso frame rather than the camera frame, and fuse appearance, pose, motion, and action experts according to explicitly learned observability. Train the system to remain calibrated and stable under overhead-specific missing or corrupted evidence.

The method must be falsifiable. Do not promise novelty or accuracy before the literature and experiments support it. The mathematical invariance property can be guaranteed if implemented correctly; scholarly novelty and performance cannot be guaranteed. If a prior paper substantially anticipates the method, narrow or change the claim before training.

## Read the repository before changing it

Read these files completely and treat them as binding project context:

- THESIS_PLAN.md
- FINDINGS.md
- MARCH_2026_POSTMORTEM.md
- THESIS_FIRST_DRAFT.md
- THESIS_DEFENSIBILITY_REVIEW.md
- BRANCH_B_TEST_PROTOCOL.md
- TEST_SPLIT_PROTOCOL.md
- outputs/FINAL_RESULTS_REGISTER.md and its machine-readable companion
- LLMDet/configs/attention_runtime.yaml
- the current LLMDet detector configuration and checkpoint specification
- LLMDet/attention/, especially the feature extractor, sequence builder, temporal models, calibration, events, runtime bridge, dashboard, and thesis_eval package
- all relevant tests and manifests

FINDINGS.md is the canonical chronological log of work completed so far. Read it from beginning to end before proposing changes; use it to recover the reasons behind previous experiments, bugs, corrections, negative results, deployment choices, and unresolved tasks. Do not treat every historical entry as a current fact: reconcile it against the final results register, protocols, current configs, checkpoint metadata, and later corrections. Append dated Branch-C findings with exact commands, job IDs, artifact paths, hashes, results, failures, and interpretation. Do not erase or rewrite historical findings to make the new work look cleaner.

Do not limit repository understanding to the named files. Recursively inventory every tracked Markdown file, including README files and documentation inside subpackages. Classify each as authoritative/current, protocol, chronological log, operational guide, design note, historical/post-mortem, thesis draft, or stale/superseded. Create docs/branch_c/DOCUMENTATION_MAP.md showing what each document contributes, conflicts between documents, and which source wins. Follow references from those documents to relevant code, configs, scripts, outputs, and manifests.

Inventory all raw videos, annotations, checkpoints, feature caches, and experiment outputs available on the HPC disks. Record paths and hashes; do not copy large private data into Git.

Preserve all existing results, runtime paths, and dashboard behavior. Never overwrite or silently mutate an old cache, checkpoint, protocol, split, results register, dashboard route, API contract, or default configuration. Put all new artifacts under clearly named Branch-C directories. Work on a new Git branch/worktree if the repository is clean enough to do so. If there are existing user changes, preserve them.

Treat the existing system as a working product with research evidence, not as disposable scaffolding. Before modifying a shared component, identify its callers, tests, configuration contracts, stored output schemas, and thesis results. Prefer adapters, versioned schemas, feature flags, and additive modules over invasive rewrites. A new component must not become the default until the legacy path passes regression tests and the Branch-C path passes its scientific and runtime gates.

## Facts that the new work must respect

The repository already establishes:

- The defensible target is six observable cues: screen_oriented, looking_away, head_down, turned_to_peer, phone_use, and uncertain. Do not claim internal attention, engagement, comprehension, emotion, boredom, or learning outcome.
- Branch A is the project's grounding/detection contribution and reaches test R@1 0.6462. Hungarian matching produced a large controlled gain. Do not discard or rewrite this history.
- On the six-cue Branch-B task, the temporal-transformer 556 model reaches about 0.407 test macro-F1, ASRF-556 about 0.477, and MS-TCN-556 about 0.500 over three seeds. MS-TCN is the strongest existing test result even though ASRF was the pre-registered primary architecture.
- The full 556 head-pose block improves downstream macro-F1, but about 80 percent of that gain is explained by the binary face_found flag. Pose angles add only about 0.0058 over face_found and are not significant in any of three seed comparisons. This is the most important control for the new pose branch.
- Expression and handcrafted dynamics/gaze additions were null or harmful in the existing ablations.
- The existing human event set contains 754 accepted frames, ten tracks, and only 16 distinct episodes after de-duplication. It is a small, selected diagnostic set; it excludes detector/tracker errors and has already been examined extensively.
- The original Branch-B test split has already been opened and reported. It cannot honestly become an untouched test for a new architecture.
- The deployed system uses an MS-TCN checkpoint with 553 inputs: the 552-dimensional base plus face_found. Its BlazeFace detector replaces the expensive mesh. At detector stride 3 and temporal stride 2 it reaches 5.77 FPS at roughly six students, 94.7 percent cue agreement against stride 1, and about 291.7 ms p95 latency on an A100. Treat that as the runtime baseline.
- A past configuration/checkpoint width mismatch yielded invalid behavior. New code must load model architecture and feature schemas strictly from versioned checkpoint metadata. Never pad, truncate, or silently initialize incompatible layers.
- The pseudo-label teacher teaches visible cues. Performance against pseudo-labels is imitation performance, not direct proof of human validity.

Verify these facts against the files; cite the exact artifact for every number used.

## Non-negotiable scientific guardrails

1. Do not call the final scalar an Engagement Index. Preserve the six-dimensional cue posterior and event stream. If a single display score is useful, call it Observable Cue Evidence or Alert Priority, show its formula, make it context-specific, and never describe it as a psychological measure. Do not fit such a score without an explicit human target.
2. Do not assume all repository media are geometrically equivalent merely because they are described as overhead. Quantify camera elevation/azimuth where possible, person and head pixel sizes, occlusion, face visibility, body coverage, student count, and scene layout.
3. Do not assume YOLO COCO weights detect heads. A person detector and a head detector are different components. Use a head-trained checkpoint, DirectMHP-style joint head detection, or a documented crop strategy.
4. Do not assume WHENet is best for top-down tiny or back-facing heads. Benchmark it against at least one full-range alternative, preferably DirectMHP and/or 6DRepNet360, subject to available weights and licenses.
5. Do not replace the existing LLMDet detector merely because YOLO is popular. Implement detector adapters and measure them. Keep LLMDet as the semantic grounding baseline; add YOLO11 or YOLOv8 as a speed/closed-vocabulary baseline or student model.
6. Do not assume BoT-SORT beats ByteTrack. On a fixed overhead camera, ByteTrack may be the simpler and faster choice. Enable BoT-SORT ReID only if it reduces measured identity switches or fragmentation. Camera-motion compensation is unnecessary unless the feed actually moves.
7. Use only past and present frames in every online feature, reference-frame estimator, temporal model, calibration rule, and event decision. Offline/acausal results must be labeled separately.
8. Do not tune on the old Branch-B test or the old human event set. Do not repeatedly inspect a new outer-test fold.
9. Do not invent human labels, ethics approval, consent evidence, or model releases. Build annotation tooling/manifests where evidence is missing and mark the human action required.
10. Run a license audit before vendoring or redistributing models. In particular, current Ultralytics code/weights may impose AGPL-3.0 or enterprise-license conditions. Record licenses and redistribution decisions for every external component.

## Phase 0: evidence, geometry, and novelty audit

Before implementation:

### 0A. Repository and data audit

Create:

- docs/branch_c/REPOSITORY_AND_DATA_AUDIT.md
- docs/branch_c/ARTIFACT_MANIFEST.json

The audit must include data paths, counts, hashes, split membership, camera groups, video groups, label provenance, feature dimensions, checkpoints, runtime dependencies, licenses, and known test exposure. Explicitly resolve whether all media are overhead and whether any truly unused videos or subjects exist.

### 0B. Literature novelty audit

Search primary papers and official repositories through the current date, not only papers already known to the researcher. Cover:

- overhead/top-down classroom student detection, tracking, head pose, gaze, action recognition, and behavioural cue/event segmentation;
- task-relative, body-relative, seat-relative, and camera-invariant pose representations;
- SO(2)/SO(3) canonicalization and geodesic orientation losses;
- uncertainty-, quality-, and observability-aware multimodal fusion;
- missing-modality robustness and modality dropout;
- causal online action segmentation and boundary detection;
- tiny/occluded/full-range multi-person head pose;
- teacher-student distillation for multi-person video inference.

Seed the search with the official/primary sources for YOLO tracking, ByteTrack, BoT-SORT, WHENet, DirectMHP, 6DRepNet360, SlowFast, VideoMAE, MS-TCN, ASRF, and recent uncertainty-aware or missing-modality fusion. Do not rely on blog summaries.

Create docs/branch_c/NOVELTY_AUDIT.md with a claim matrix:

| Candidate claim | Closest prior work | Exact overlap | Remaining difference | Evidence needed | Keep/narrow/drop |

Use a strict no-go rule: if the full combination and its training objective already appear in close prior work, do not call it novel. Reframe the thesis contribution as an overhead-specific adaptation and controlled evaluation, or redesign the method.

### 0C. Freeze Branch-C evaluation

Create BRANCH_C_PROTOCOL.md before reading any new held-out results.

Preferred protocol:

- If genuinely unused videos/subjects/camera sessions exist, freeze them as a new external holdout by immutable manifest and hash before model development.
- Otherwise, use nested grouped cross-validation with video/subject/camera grouping. The inner folds select architecture, loss weights, thresholds, and checkpoints; the outer folds estimate generalization. Report the old Branch-B test only as a legacy retrospective comparison.
- Keep every crop and temporal neighbor from a source video in one group.
- Deduplicate near-identical frames before splitting if videos overlap.
- Fit calibration and event thresholds only on the inner validation data.
- Use at least three fixed seeds for shortlisted models. Prefer five when training cost permits.
- Define one primary metric before experiments: six-cue macro-F1 over grouped outer predictions. Define secondary metrics: balanced accuracy, per-class F1, NLL, Brier score, ECE, risk-coverage/AURC, segmental F1 at 10/25/50, edit score, event precision/recall/F1, onset/duration error, and false alerts/hour.
- For detector/tracker evaluation, use mAP/recall and HOTA/IDF1/identity switches/fragmentation/track coverage only when the annotations support them. Label proxy metrics as proxies.
- For human validity, create a new, disjoint, double-annotated frame/event sample if feasible. Keep the old 16-episode set as legacy diagnostic evidence only.

Produce a machine-readable protocol and split manifest. Add a test that proves no video/subject/camera group crosses folds.

## OVERT-Cue architecture

Implement Branch C modularly. Suggested package:

- LLMDet/attention/overhead/
- LLMDet/configs/branch_c/
- LLMDet/work_dirs/branch_c/ or the repository's established work-dir convention
- outputs/branch_c/
- tools/branch_c/

Reuse existing modules where correct. Avoid a parallel reinvention of calibration, events, evaluation, or logging.

### Stage 1: detection adapters

Implement a common detector interface returning boxes, scores, class, source model, and feature-schema version.

Compare:

1. the exact existing LLMDet/Grounding-DINO detector;
2. YOLO11 or YOLOv8 person detection, using available legal checkpoints;
3. optional LLMDet-to-YOLO distillation or proposal filtering only if it is useful and cleanly ablated.

Head localization is separate. Compare a face detector, a head detector/joint multi-person head-pose model, and person-crop fallback. Quantify coverage by head size, occlusion, facing direction, and cue class.

Choose the accurate research detector and the demo detector separately if warranted.

### Stage 2: tracking adapters

Implement a common online tracker interface and compare:

- the current repository tracker;
- ByteTrack;
- BoT-SORT without ReID;
- BoT-SORT with ReID only if enough identity evidence exists.

Use identical detector outputs for a fair tracker ablation. Tune on validation only. Track state must expose observability signals such as age, time since detection, detector confidence, box scale, recent IoU consistency, velocity consistency, and occlusion/overlap.

### Stage 3: modality experts

For each tracked student at time t, produce timestamp-aligned expert features and an explicit validity/quality vector.

Experts:

1. Existing appearance/semantic/geometry expert: preserve the proven 552-dimensional base or a documented projection of it.
2. Head expert: WHENet baseline plus DirectMHP and/or 6DRepNet360 candidate. Prefer rotation matrices or continuous 6D rotation representations internally. Preserve raw estimator confidence, head/face coverage, crop resolution, entropy if available, and temporal angular consistency.
3. Body/action expert: frozen SlowFast and VideoMAE features on causal per-track clips. Begin with cached frozen features. Fine-tune only after frozen-feature ablations show useful signal. Include an inexpensive body-pose/torso-orientation estimator if it supports the canonical reference frame.
4. Motion/track expert: causal box trajectory, optical/feature motion, occlusion, track age, and interaction geometry. Do not reuse the old whole-track non-streaming handcrafted dynamics as if they were causal.

Use timestamp masks. Never replace a missing expert with an all-zero vector without a separate validity mask.

### Stage 4: task-relative pose canonicalization

This is the main theoretical component.

Let R_head_C(i,t) be the estimated head rotation of student i at time t in camera coordinates. Let R_ref_C(i,t) be a causal reference orientation in the same coordinates, derived in order of preference from:

1. a visible task/monitor/desk direction associated with the student's seat;
2. a reliable torso/body orientation;
3. a causal per-seat reference estimator using past high-quality frames only.

Define:

R_rel(i,t) = transpose(R_ref_C(i,t)) times R_head_C(i,t).

Use the 6D rotation representation, the SO(3) logarithm, or geodesic angles as model features. Include relative angular velocity and acceleration computed causally. Never subtract Euler angles across wraparound.

Prove and unit-test the coordinate invariance:

If the camera coordinate system is changed by a global rotation Q, then

R_head_C_prime = Q R_head_C
R_ref_C_prime  = Q R_ref_C

and therefore

transpose(R_ref_C_prime) R_head_C_prime
= transpose(R_ref_C) transpose(Q) Q R_head_C
= R_rel.

If a trustworthy 3D reference frame is unavailable, implement the SO(2) image-plane analogue using relative heading to the seat/task/torso axis. State the weaker assumptions. Test synthetic image rotations and projective perturbations separately; rotation invariance does not imply arbitrary homography invariance.

R_ref must be estimated without future frames and without consuming held-out labels. If a bootstrapped high-confidence screen_oriented estimate is used, report the circularity risk and compare against scene- or torso-derived references.

### Stage 5: evidence-reliable fusion and causal temporal segmentation

Use modality-specific projections and experts. For modality m:

z_m(i,t) = cue logits from modality m
r_m(i,t) = predicted log reliability/precision from only legitimate quality and context inputs

Candidate quality inputs include crop pixel size, detection confidence, occlusion/overlap, head/face visibility, estimator confidence, pose temporal consistency, action-clip completeness, tracker age, time since a real detection, and out-of-distribution distance.

Fuse with a residual baseline:

alpha_m = masked_softmax(r_m / tau)
z_fused = z_base + sum_m alpha_m z_m

Feed fused features/logits into the existing strongest causal temporal backbone first, MS-TCN, with an ASRF boundary-head variant as a planned comparator. A more complex temporal transformer is allowed only if it beats these baselines under the same protocol.

The display must expose modality reliabilities and abstain when evidence is insufficient. Do not market learned gate weights as causal explanations; call them routing or reliability diagnostics.

The theoretical rationale for reliability weighting may use this limited proposition:

Under conditionally unbiased, uncorrelated expert errors with variances sigma_m_squared, the minimum-variance linear unbiased fusion weights are proportional to one over sigma_m_squared. The learned reliability head approximates log precision. State the assumptions and do not describe this known result as the novel part.

### Stage 6: overhead observability consistency loss

Implement losses incrementally:

L_total =
L_cue
+ lambda_boundary L_boundary
+ lambda_smooth L_smooth
+ lambda_view L_view
+ lambda_order L_quality_order
+ lambda_cf L_counterfactual
+ lambda_cal L_calibration
+ optional lambda_distill L_distill.

Definitions:

- L_cue: class-balanced focal cross-entropy or logit-adjusted cross-entropy, selected on inner validation. Do not run an uncontrolled loss-function lottery.
- L_boundary: focal BCE or the existing ASRF boundary objective for cue transitions.
- L_smooth: the existing truncated temporal log-probability smoothing objective.
- L_view: consistency between clean and physically plausible globally rotated/camera-perturbed samples after task-relative canonicalization. Use Jensen-Shannon divergence on cue posteriors and an optional feature consistency term.
- L_quality_order: create controlled overhead corruptions for one modality, such as head-crop downsampling, blur, partial occlusion, pose jitter, missing clip frames, or box jitter. Enforce that its predicted reliability is lower than the clean modality by a margin:

  max(0, margin - r_m(clean) + r_m(corrupt)).

- L_counterfactual: when one low-reliability modality is removed or corrupted and other evidence remains sufficient, distill the clean fused posterior into the corrupted-path posterior with stop-gradient. Weight this by clean confidence and retained evidence. Do not force invariance when the removed modality contained unique information.
- L_calibration: Brier score or NLL as a proper scoring component. Fit final temperature scaling on validation only.
- L_distill: optional KL and/or feature loss from the accurate multi-expert teacher to the lightweight demo student.

The combined L_view + L_quality_order + L_counterfactual objective may be named the **Overhead Observability Consistency objective** only if its novelty survives the literature audit.

Every loss term must have:

- a unit test;
- a zero-weight baseline;
- a one-term-at-a-time ablation;
- a coefficient selected without outer-test access;
- gradient/finite-value monitoring;
- an explicit failure interpretation.

## Required shortcut and ablation tests

The new branch is not defensible without these controls:

1. Existing MS-TCN-553 face_found deployment baseline.
2. Existing MS-TCN-556 pose baseline.
3. OVERT appearance-only.
4. OVERT plus head quality/presence flags but no angles/rotations.
5. OVERT plus absolute head pose.
6. OVERT plus task-relative pose.
7. OVERT plus body/action expert.
8. Uniform fusion versus learned reliability fusion.
9. Learned fusion without quality-order/counterfactual losses.
10. Full OVERT-Cue.
11. Full model with each modality dropped at inference.
12. Full model under controlled head blur/occlusion, downsampling, tracking gaps, and box jitter.
13. Scene-derived versus torso-derived versus causal learned reference frame.
14. WHENet versus the best full-range head candidate, evaluated by coverage, stability, runtime, and downstream cue/event metrics.
15. ByteTrack versus BoT-SORT versus the current tracker under identical detections.
16. LLMDet versus YOLO under identical tracking/evaluation.

The critical success condition for pose is:

> Task-relative pose must outperform the quality/presence-only control and absolute-pose control across grouped data. If it does not, do not claim a pose contribution, even if the full stack improves.

Also audit whether the gate predicts cue labels from missingness alone. Train/evaluate a quality-only classifier; simulate a changed missingness distribution; and report performance under label-flipped or equalized face/head visibility. This directly tests the existing face_found shortcut.

## Experiment strategy for eight A100 GPUs

Inspect the scheduler, quotas, current jobs, available partitions, installed CUDA/PyTorch versions, and disk/scratch locations first. Use SLURM job arrays or the local scheduler; do not hard-code assumptions about node topology.

Suggested execution:

- Use one short GPU smoke test for each external model and adapter.
- Precompute immutable, sharded, hash-keyed feature caches for head and action experts.
- Use distributed inference for VideoMAE/SlowFast cache generation.
- Parallelize seeds and outer folds after the protocol is frozen.
- Reserve multi-GPU DDP for models that benefit from it; small MS-TCN runs should be separate jobs rather than wasting eight GPUs.
- Use automatic mixed precision where numerically safe.
- Record Git commit, dirty-state diff hash, environment lock, dataset/split hash, feature-schema hash, checkpoint hash, command, seed, host/GPU, scheduler job ID, wall time, and peak memory for every run in outputs/branch_c/RUNS.jsonl.
- Use atomic writes and completion sentinels. A partial cache must never appear valid.
- Resume failed jobs safely. Do not delete evidence from failed runs.

Use a two-stage search:

1. cheap screening on inner validation with frozen caches and one seed;
2. full shortlisted evaluation across fixed seeds and outer folds.

Avoid a large unconstrained hyperparameter sweep. Spend compute on data integrity, modality comparisons, ablations, and statistical repetition.

## Clean-clone GitHub reproducibility contract

Make the repository reproducible from a fresh GitHub clone for every artifact that may legally and practically be distributed. “Works on the current HPC filesystem” is not sufficient. Eliminate undocumented local state, absolute paths, manually copied checkpoints, hidden environment assumptions, and caches whose producer command is unknown.

Git cannot safely execute arbitrary code immediately after clone. Implement the reproducible equivalent:

1. a cross-platform one-command bootstrap for a fresh clone;
2. a versioned artifact resolver that downloads missing public checkpoints/assets automatically when a selected command or dashboard mode needs them;
3. checksum verification, atomic installation, caching, resumable downloads, and actionable errors;
4. a clean-room test performed on a separate fresh clone.

### Reproducibility profiles

Define explicit profiles:

- demo: the smallest legally shareable checkpoint set, configs, calibration files, and short privacy-safe fixture needed to launch the dashboard and run inference;
- research: all distributable pretrained/custom checkpoints, frozen manifests, configs, and small result artifacts needed to rerun the reported Branch-A/B/C evaluations, excluding restricted datasets;
- full-data: the research profile plus dataset acquisition/preparation instructions and access checks;
- hpc: the full training environment, scheduler templates, cache generation, distributed training, evaluation, and demo deployment.

The required first-use experience should be as close as possible to:

python bootstrap.py --profile demo
python -m llmstu_tools.dashboard --config LLMDet/configs/attention_runtime.yaml

If the dashboard is started before bootstrap, its launcher should invoke the artifact resolver for the selected profile, display download size/source/license, fetch missing public files, verify them, and then continue. Provide --offline and --no-download modes. Never silently substitute another checkpoint or run a random/uninitialized model.

### Artifact lock and provenance

Create a committed, machine-readable artifacts.lock.json or equivalent. Each artifact entry must include:

- stable artifact ID and semantic version;
- purpose and owning Branch/config;
- artifact type;
- exact source URL or release identifier;
- upstream repository and commit/tag;
- license and redistribution permission;
- public/restricted/manual-access status;
- expected byte size and SHA-256;
- destination relative to the repository or configured cache root;
- profiles requiring it;
- archive extraction rules and hashes of critical extracted files;
- producer command, dataset/split hash, config hash, code commit, and parent artifacts for project-generated checkpoints;
- one or more mirrors where legally permitted;
- deprecation/replacement metadata.

The resolver must use relative destinations and a configurable cache root. It must download to a temporary file, verify size/hash, extract safely without path traversal, and rename atomically. Corrupt, partial, wrong-version, or HTML-error downloads must never be accepted as weights.

### Upload and hosting strategy

Implement a documented upload/publish workflow for project-generated checkpoints and demo assets. Critically choose the backend based on file size, quotas, permanence, citation needs, licenses, and privacy:

- GitHub Release assets for modest immutable release bundles;
- Git LFS only if repository/bandwidth quotas and clone behavior are acceptable;
- an institutional object store, Hugging Face Hub, or Zenodo for large versioned artifacts where their terms fit;
- DVC only if it has a configured remote and materially improves traceability.

Do not commit large binary checkpoints directly to normal Git history. Do not mirror upstream weights when their license forbids redistribution; store the official download recipe and hash instead. Do not upload raw student video, face crops, annotations, or derived identifiable media unless consent, ethics, and license records explicitly permit it.

Provide:

- tools/artifacts/upload.py or an equivalent backend-neutral publisher;
- tools/artifacts/download.py or a shared resolver CLI;
- dry-run, list, verify, repair, and offline commands;
- release-manifest generation;
- upload verification by downloading into an empty temporary cache and checking hashes;
- environment-variable/credential documentation.

Never commit API keys, access tokens, signed URLs, credentials, or secrets. Upload commands must read credentials from environment variables or the scheduler secret mechanism and must refuse to print them.

### Environment reproducibility

Produce one canonical environment definition and lock:

- pyproject.toml plus a resolved lock file, or a documented conda-lock/uv/pip-tools equivalent;
- exact Python, PyTorch, CUDA, cuDNN, compiler, FFmpeg, and system-library compatibility;
- pinned Git dependencies by immutable commit;
- an Apptainer/Singularity definition for HPC and, if useful, a Dockerfile derived from the same dependency source;
- environment and container build verification;
- CPU-only metadata/test mode where practical, while clearly marking GPU-required inference.

Avoid machine-specific absolute paths. Centralize path resolution and support command-line/environment overrides. Record device placement instead of assuming cuda:0 everywhere. Provide scheduler templates that map logical stages to allocated GPUs.

### Data and result reproducibility

Separate:

- code reproducibility;
- model-artifact reproducibility;
- demo reproducibility;
- full experimental reproducibility;
- human-data availability.

If the original dataset cannot legally be redistributed, the GitHub repository can still be fully self-contained for the public demo only if it includes a privacy-safe, licensed fixture. For full experiments, provide acquisition/access instructions, expected hashes, directory schema, validation commands, and deterministic preprocessing. State clearly which results require restricted data. Never claim 100 percent public reproducibility when a required dataset or human annotation is unavailable.

Commit small essential evidence: protocols, split IDs/hashes where privacy permits, configs, schemas, source code, raw metric JSON/CSV, result registers, environment locks, and provenance. Large predictions/caches may be downloadable artifacts, but every published table must be regenerable from committed or resolvable raw results.

### Clean-room acceptance test

After implementation, use a new empty temporary directory or isolated worktree—not the developer checkout—and:

1. clone the GitHub-ready repository;
2. confirm there are no local untracked dependencies;
3. run the demo bootstrap using only documented credentials;
4. verify every downloaded artifact;
5. launch the dashboard/inference on the shareable fixture;
6. compare its JSONL predictions/events and representative rendered frames with the frozen golden outputs;
7. run unit/integration tests;
8. rebuild at least one small result table from raw outputs;
9. record time, network bytes, disk use, environment, commands, and failures.

Add a GitHub Actions workflow for all checks possible on standard runners. If GPU inference cannot run on hosted CI, use a self-hosted GPU workflow or a separately recorded clean-room GPU job, while keeping CPU schema/bootstrap/dry-run tests in public CI.

Create docs/REPRODUCIBILITY.md and outputs/branch_c/CLEAN_CLONE_REPORT.md. The project is not release-ready until the demo profile succeeds from a clean clone and every non-downloadable dependency is explicitly classified.

## Selection and stop/go gates

Freeze these before final runs:

### Scientific go gate

The proposed contribution is supported only if:

- the full model improves grouped outer macro-F1 over the strongest properly comparable existing baseline by a practically meaningful margin, provisionally at least 0.02 absolute, and the paired per-video bootstrap 95 percent confidence interval for the difference excludes zero; or, if data are too small for this threshold, the protocol must predefine and justify another minimum effect;
- task-relative pose beats quality/presence-only and absolute-pose controls;
- reliability fusion beats uniform fusion under at least two predefined missing/corrupted-modality conditions without materially degrading clean performance;
- results are directionally consistent across at least three seeds;
- calibration and event fragmentation do not worsen enough to erase the frame-level gain.

If these fail, report a negative result and keep Branch C as an engineering extension. Do not manufacture a theoretical claim by changing metrics after the fact.

### Runtime/demo go gate

Provide two profiles:

- OVERT-Cue Research: most accurate validated model, allowed to use cached/heavy experts.
- OVERT-Cue Demo: causal single-stream or asynchronous model, distilled if needed.

The demo target is no worse than the current baseline envelope on one A100 at about six visible students:

- at least 5.77 processed FPS or an explicitly justified better latency/accuracy trade-off;
- p95 latency around or below 300 ms;
- complete track coverage on the fixed demo protocol;
- non-degenerate cue outputs and working abstention;
- stage-level p50/p95/p99 timings and GPU memory;
- strict checkpoint/schema loading.

If heavy VideoMAE/SlowFast or head-pose inference misses the envelope, run them at a lower causal cadence with cached track state or distill them into a lightweight per-track student. Do not hide preprocessing time.

## Clean demo contract

The new architecture must be integrated into the existing dashboard/replay application. Do not build an unrelated toy UI, replace the dashboard wholesale, or compromise any working Branch-A/Branch-B feature.

Before dashboard changes:

- inventory the current dashboard entry points, backend/runtime bridge, API or JSONL schemas, routes, configuration loading, overlays, event logic, privacy controls, and launch commands;
- capture a deterministic legacy replay fixture, screenshots or rendered frames, JSONL output, timing profile, and current tests as a regression baseline;
- document the dependency graph and the smallest safe integration seam;
- identify which parts of OVERT-Cue are online, asynchronous, cached-only, or research-only.

Integrate through a versioned adapter and feature flag. Keep the legacy mode available and unchanged. The dashboard should offer explicit modes such as:

- Legacy Branch B;
- OVERT-Cue Demo;
- OVERT-Cue Research Replay, only where its dependencies are available.

The default must remain Legacy Branch B until OVERT-Cue Demo passes all acceptance gates. A missing Branch-C checkpoint or dependency must fall back safely or show a precise actionable error; it must never break the legacy launch path.

Version the prediction/event schema. Preserve existing fields and consumers, and add Branch-C fields in an additive namespace for modality reliability, canonical pose diagnostics, model provenance, and abstention reason. If a schema migration is unavoidable, supply a compatibility adapter and migration tests.

Dashboard acceptance requires:

- the unmodified legacy replay produces the same cue/event trace and rendered behavior within predefined deterministic tolerances;
- old configs and checkpoints still load strictly;
- mode switching cannot leak state, thresholds, tracker IDs, feature buffers, or calibration parameters across models;
- research-only or acausal data can never be selected accidentally in live mode;
- the OVERT overlays degrade cleanly when head/action modalities are absent;
- all errors identify the failing model, checkpoint, feature schema, and recovery action;
- a documented one-command rollback restores the previous dashboard behavior without data loss.

Use at least one held-out overhead video that was not used for demo tuning. Produce:

- an annotated MP4;
- a replayable JSONL event/prediction trace;
- a deterministic launch command and config;
- a model-card-style description;
- a side-by-side old-versus-new mode;
- detector/tracker/model toggles where feasible;
- overlays for student ID, box/head location, current observable cue, calibrated confidence, event duration, abstention state, and compact per-modality reliability bars;
- a privacy mode that blurs faces/heads in stored or displayed output;
- warnings when evidence is insufficient or a track is stale.

Never display “Student 7 is disengaged.” Display auditable statements such as “Track 7: head_down cue, 4.2 s, confidence 0.81” or “insufficient evidence.”

Add a demo smoke test that loads the exact checkpoint/config, processes a short clip, asserts multiple track IDs and nonconstant outputs, validates JSONL schema, and rejects incompatible feature widths.

Add a dashboard regression test that runs the frozen legacy fixture before and after integration and compares predictions, events, schema, and representative rendered frames. Any unexplained change is a release blocker, even if the new model performs better.

## Statistical reporting

For each shortlisted model:

- report mean and standard deviation over seeds;
- aggregate outer-fold predictions before overall metrics;
- report per-video paired bootstrap confidence intervals and paired differences;
- include class support and per-class F1;
- report both raw and calibrated NLL/Brier/ECE;
- report risk-coverage/AURC and predefined display/alert operating points;
- include parameter count, FLOPs if reliable, cache/inference time, end-to-end FPS, p50/p95/p99 latency, and peak GPU memory;
- separate cue-model-only, detector/tracker-inclusive, and human-referenced results;
- label every pseudo-label metric as teacher agreement;
- never mix retrieval R@1, detection mAP, cue macro-F1, ordinal engagement accuracy, and event recall in one undifferentiated leaderboard.

Correct for multiple comparisons where many variants are tested. Preserve all unsuccessful registered variants in the results table.

## Tests and code quality

Add tests for:

- group split leakage;
- timestamp alignment across experts;
- causal-window construction and absence of future leakage;
- SO(2)/SO(3) invariance;
- rotation wraparound and geodesic distance;
- corruption/reliability ordering loss;
- modality masks and all-missing fallback;
- deterministic cache keys and schema hashes;
- tracker adapter output contract;
- checkpoint/config/feature-schema mismatch;
- calibration fit only on validation;
- demo JSONL and MP4 pipeline smoke test.
- frozen legacy-dashboard replay equivalence;
- dashboard mode isolation and safe fallback;
- backward-compatible prediction/event schemas;
- research-only components being impossible to enable in live mode accidentally.
- artifact-lock coverage for every runtime checkpoint/config dependency;
- checksum rejection, interrupted-download resume, atomic installation, mirror fallback, repair, and offline behavior;
- safe archive extraction and path-traversal rejection;
- dashboard first-run artifact resolution from an empty cache;
- absence of hard-coded developer/HPC absolute paths in committed runtime configs;
- clean-clone bootstrap and inference on the privacy-safe fixture.

Run the existing test suite before and after changes. Do not relax old assertions merely to make the new code pass.

## Required deliverables

Leave all of the following in the repository:

1. docs/branch_c/REPOSITORY_AND_DATA_AUDIT.md
2. docs/branch_c/ARTIFACT_MANIFEST.json
3. docs/branch_c/NOVELTY_AUDIT.md
4. docs/branch_c/ARCHITECTURE.md, including equations, assumptions, and a diagram
5. BRANCH_C_PROTOCOL.md and immutable split manifests
6. implemented detector, tracker, pose/action, canonicalization, fusion, loss, training, evaluation, and runtime modules
7. versioned configs and checkpoint feature schemas
8. tests and their logs
9. outputs/branch_c/RUNS.jsonl
10. outputs/branch_c/RESULTS_REGISTER.md and machine-readable JSON/CSV
11. outputs/branch_c/ABLATIONS.md
12. outputs/branch_c/ERROR_ANALYSIS.md with qualitative examples by failure type
13. demo command, annotated video, trace, and runtime profile
14. THESIS_BRANCH_C_DRAFT.md containing thesis-ready Methods, Experimental Protocol, Results, Limitations, and Contribution wording
15. an updated top-level thesis draft only after the Branch-C results register is frozen; do not insert provisional numbers as facts
16. docs/branch_c/DASHBOARD_INTEGRATION.md with the dependency map, schema version, feature flags, compatibility strategy, regression evidence, and rollback command
17. outputs/branch_c/dashboard/ containing the frozen legacy fixture, old-versus-new regression report, screenshots/rendered frames, and dashboard smoke-test logs
18. docs/branch_c/INDEPENDENT_REVIEW_LOG.md containing the cross-agent findings, resolutions, and sign-offs at each gate
19. docs/branch_c/DOCUMENTATION_MAP.md covering every tracked Markdown file and documentation conflict
20. docs/REPRODUCIBILITY.md, the artifact lock, profile definitions, environment lock, container definitions, and GitHub/HPC setup instructions
21. cross-platform bootstrap, artifact download/verify/repair tooling, credential-safe upload/release tooling, and CI workflows
22. outputs/branch_c/CLEAN_CLONE_REPORT.md with complete fresh-clone evidence
23. dated, evidence-linked Branch-C entries appended to FINDINGS.md

Every table entry must link to its producing command, config, checkpoint, split hash, and raw result artifact.

## Thesis claim language

Use this wording only if all scientific go gates pass:

> We introduce OVERT-Cue, a causal overhead-video method that canonicalizes head orientation relative to a task/seat/torso frame and routes appearance, pose, motion, and action evidence according to measured observability. The relative-rotation representation is invariant to a global change of camera coordinates, while the observability-consistency objective reduces dependence on missing head/face detections. On grouped unseen videos, OVERT-Cue improves [pre-registered metrics] over [baselines], with [confidence interval and ablation evidence].

If only the invariance ablation succeeds:

> We contribute and validate a task-relative pose representation for overhead classroom video, embedded in the existing temporal cue system.

If the full model improves but pose does not beat face/head presence controls:

> We contribute an observability-aware multimodal engineering extension, but do not claim that head-pose geometry is responsible for the gain.

If no controlled improvement survives:

> The study provides a negative result: adding off-the-shelf head-pose and action models did not improve grouped visible-cue segmentation beyond the existing MS-TCN after controlling for modality availability.

Negative evidence is preferable to an invalid novelty claim.

## Multi-agent execution and independent verification

Use multiple agents in parallel when the environment supports them. The purpose is separation of concerns and independent checking, not duplicated uncontrolled editing. The lead Opus agent remains accountable for every merge, command, claim, and result; never accept a subagent's conclusion without inspecting its evidence.

Create at least these roles:

1. **Lead/orchestrator agent:** owns the immutable protocol, task graph, interfaces, integration order, final decisions, and merges.
2. **Repository/dashboard agent:** maps the current runtime and dashboard, freezes the legacy regression fixture, implements the versioned adapter and UI integration, and preserves backward compatibility.
3. **Model/method agent:** implements canonicalization, modality experts, reliability fusion, losses, and unit tests without access to outer-test results.
4. **Data/evaluation agent:** audits splits and labels, implements leakage tests and metrics, freezes manifests, and independently recomputes reported tables from raw predictions.
5. **Independent reviewer/red-team agent:** does not author the feature being reviewed. It searches for future leakage, label/test contamination, mathematical errors, silent fallbacks, schema incompatibility, missingness shortcuts, unsupported claims, and dashboard regressions.
6. **Reproducibility/release agent:** inventories environments and large artifacts, implements the artifact lock/resolver/publisher and clean-clone workflow, scans for secrets and absolute paths, and reproduces the demo from an empty clone.

If agent capacity is smaller, preserve the separation by running the roles sequentially with fresh review contexts. If capacity is larger, subdivide bounded work such as detector/tracker benchmarking, pose/action cache validation, runtime profiling, and literature/license review.

Coordination rules:

- give every agent a bounded task, read-only context where possible, exact deliverables, allowed files, and acceptance tests;
- use separate worktrees or non-overlapping file ownership for parallel writers;
- do not let two agents edit the same file concurrently;
- require agents to record assumptions, commands, artifact paths, hashes, and unresolved concerns;
- keep the frozen protocol and split manifests read-only to implementation agents;
- prohibit every implementation agent from inspecting outer-test results before the lead opens the registered evaluation;
- have the evaluation agent recompute headline metrics independently from saved predictions;
- have the reviewer derive/check the invariance proof and inspect its tests independently of the method author;
- have the dashboard agent and reviewer both sign off on legacy replay equivalence and rollback;
- have the reproducibility agent verify every checkpoint hash/source and run the clean clone without access to the developer's local caches;
- resolve conflicting agent recommendations from evidence, smallest-change principles, and registered metrics, not by majority vote;
- run integration tests after each merge rather than waiting until all parallel work is combined;
- stop a merge when an agent reports an unresolved P0/P1 issue affecting leakage, validity, data loss, privacy, checkpoint correctness, or legacy behavior.

Use a proposer-reviewer gate for each major change:

1. the author supplies the diff, tests, raw evidence, and known risks;
2. an independent agent reviews the code and reproduces the relevant test/result;
3. the lead resolves findings and records the decision;
4. only then may the change enter the integration branch.

No agent may claim that another agent “verified” something without a reproducible artifact. Record all reviews and resolutions in docs/branch_c/INDEPENDENT_REVIEW_LOG.md.

## Execution behavior

Start now. First inspect the repository, data, scheduler, and environments; then write the audits and freeze the protocol. Proceed to a one-video vertical slice, tests, feature-cache jobs, controlled baselines, the shortlisted full experiment, and the demo. Use reasonable assumptions when they are reversible and document them. Ask the researcher only for decisions that cannot be recovered from the repository and would materially change the scientific claim, such as the identity of a truly unused external holdout, permission/ethics evidence, or new human annotations.

Do not stop after planning. Continue until the code, jobs, results, demo, and thesis-ready Branch-C draft are complete, or until a genuine external dependency requires the researcher's action. When blocked by queued jobs, record job IDs and monitoring commands and continue with tasks that do not depend on them.

Think critically before adopting the requested component list. The named models are candidates, not mandatory choices. For every architectural replacement, decide whether to preserve, wrap, augment, distill, or reject it based on repository evidence, scientific validity, runtime, licensing, maintainability, and regression risk. Prefer the smallest reversible change that can test the hypothesis. Explain rejected options.

Zero mistakes cannot be promised in research software. Operationalize mistake resistance through immutable evidence, strict schemas, deterministic fixtures, isolated changes, automated tests, independent reproduction, canary/demo replay, and rollback. Never hide uncertainty or waive a failed gate to appear complete.

At each major checkpoint, report:

- what changed;
- exact files;
- commands/jobs and IDs;
- completed evidence;
- failures or caveats;
- next action;
- whether the candidate contribution remains supported, narrowed, or rejected.

Your final report must lead with the honest outcome, list the best research and demo configurations, cite exact artifact paths and metrics, state whether each go gate passed, and provide one command that reproduces the demo.

---

## Primary sources Opus should verify during its novelty/license audit

- Ultralytics YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/
- Ultralytics tracking documentation: https://docs.ultralytics.com/modes/track/
- ByteTrack: https://github.com/ifzhang/ByteTrack
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
- WHENet: https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
- DirectMHP: https://github.com/hnuzhy/DirectMHP
- 6DRepNet/6DRepNet360: https://github.com/thohemp/6DRepNet and https://github.com/thohemp/6DRepNet360
- SlowFast: https://github.com/facebookresearch/SlowFast
- VideoMAE: https://github.com/MCG-NJU/VideoMAE
- MS-TCN: https://arxiv.org/abs/1903.01945
- ASRF: https://arxiv.org/abs/2007.06866
- Normalized Human Pose Features for Human Action Video Alignment: https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Normalized_Human_Pose_Features_for_Human_Action_Video_Alignment_ICCV_2021_paper.html
- Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion: https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Embracing_Unimodal_Aleatoric_Uncertainty_for_Robust_Multimodal_Fusion_CVPR_2024_paper.html
- MultiModN: https://proceedings.neurips.cc/paper_files/paper/2023/file/5951641ad71b0052cf776f9b71f18932-Paper-Conference.pdf

These are search seeds, not an exhaustive novelty review.
