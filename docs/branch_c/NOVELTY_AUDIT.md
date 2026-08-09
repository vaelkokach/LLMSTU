# Branch C — Novelty Audit

**Date:** 2026-08-09. **Scope:** the *narrowed* claim only (FINDINGS.md §12.1).
The data is one fixed corner-mounted oblique camera (~20-30° elevation),
2812×1050, 128 recordings, one room, students seated in rows facing monitors,
rear/three-quarter head views dominant. There is exactly one camera pose in
the corpus. "Overhead" and any cross-camera viewpoint-invariance claim are
already dropped (FINDINGS §12.1, point 2) and are **not** re-litigated here.

**Method:** arXiv, CVF, and official GitHub repos searched directly (not blog
summaries), through 2026-08-09. Sub-claims C1–C5 audited separately, per the
brief. The governing rule is the spec's strict no-go: *if the full combination
and its training objective already appear in close prior work, say so and
recommend DROP.* Where a mechanism is anticipated only in isolation (not in
this exact combination), that is recorded honestly as NARROW, not KEEP.

---

## 1. Claim matrix

| # | Candidate claim | Closest prior work | Exact overlap | Remaining difference | Evidence needed | Keep/Narrow/Drop |
|---|---|---|---|---|---|---|
| C1 | Seat/task-relative canonicalised head-orientation features (R_rel = R_ref^T R_head), 6D rotation / SO(3) log, SO(2) fallback | **3DPCNet**, "Pose Canonicalization for Robust Viewpoint-Invariant 3D Kinematic Analysis from Monocular RGB", arXiv:2509.23455, Sept 2025 (https://arxiv.org/abs/2509.23455) | Predicts a continuous **6D rotation representation**, converts to SO(3) via Gram-Schmidt, trains with a **geodesic SO(3) loss**, canonicalises pose into a fixed body-centred frame — the *exact* representational and loss machinery C1 proposes | 3DPCNet canonicalises full-body 3D kinematics to a **body-centred** frame for health/sports monocular video; it is not head-only, not a **seat/task-relative** frame, and not classroom/multi-person. No paper found (see §2) doing seat- or desk-relative pose normalisation in classroom video specifically | Show the seat-confound prediction (FINDINGS §12.1 point 3: camera-frame yaw for `screen_oriented` differs by seat, seat-frame yaw should not) holds empirically | **NARROW** — the 6D/SO(3)-canonicalisation technique itself is not new; only the seat/task reference-frame choice for classroom head pose is |
| C1b | 6D rotation representation as the underlying parametrisation | **Zhou et al.**, "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019, arXiv:1812.07035 (https://arxiv.org/abs/1812.07035) | This is the foundational method C1 explicitly builds on; full overlap by design | None — this is citation, not novelty | N/A | **N/A** — cite as basis, never as contribution |
| C1c | Canonical pose features disentangled from viewpoint for action alignment | **Liu et al.**, "Normalized Human Pose Features for Human Action Video Alignment", ICCV 2021 (https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Normalized_Human_Pose_Features_for_Human_Action_Video_Alignment_ICCV_2021_paper.html) | Retargets poses onto a reference skeleton to unify global orientation and disentangle viewpoint/anthropometry, exactly the motivation behind C1 | Full-body skeleton retargeting + unsupervised metric embedding for action alignment/retrieval, not head-only orientation, not classroom, not paired with reliability-gated multimodal fusion | Cite as the direct conceptual predecessor; differentiate on head-only + seat-frame + fusion pairing | **NARROW** (same finding as C1) |
| C1d | Full-range head pose in a corpus dominated by rear/three-quarter views | **DirectMHP**, arXiv:2302.01110 (https://arxiv.org/abs/2302.01110); **6DRepNet360**, "Towards Robust and Unconstrained Full Range of Rotation Head Pose Estimation", IEEE TIP 2024 (https://github.com/thohemp/6DRepNet360) | Both already solve full-range (incl. rear-of-head) multi-person head pose regression in camera frame | Neither adds a seat/task canonicalisation step; C1's remaining contribution is the post-hoc relative transform on top of one of these backbones | Ablate camera-frame vs seat-frame features on the same backbone output | **KEEP the backbone choice as motivated** (rear views dominate — genuinely justifies full-range over WHENet), but the canonicalisation on top is the NARROW item above |
| C2 | Learned per-modality reliability α_m = masked_softmax(r_m/τ), r_m predicted from **quality signals only** (crop size, det. confidence, occlusion, tracker age, estimator confidence, temporal consistency) | **Multi-QuAD**, "Multi-Level Quality-Adaptive Dynamic Network for Reliable Multimodal Classification", arXiv:2412.14489, Dec 2024 (https://arxiv.org/abs/2412.14489) | Predicts per-modality **and** per-feature quality/reliability via a classifier-free, noise-free-prototype design, then uses it to gate fusion — the same design principle as C2 | Different signal source (learned quality proxies from the data itself, not external tracking/detection metadata) and different domain (generic multimodal classification, not video/tracking) | Confirm no per-modality gate in Multi-QuAD is driven by *external pipeline metadata* (crop size, tracker age) rather than learned features | **NARROW** — "predict reliability, then softmax-gate" is not new; the specific external-quality-signal vocabulary for a tracked-video pipeline is the only remaining difference |
| C2b | Reliability driven by quality signals rather than raw content features | **"When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence"**, arXiv:2606.26473, June 2026, Dartmouth (https://arxiv.org/html/2606.26473v1) | Computes quality metrics **once per modality from signal properties, not raw features** (audio clipping/spectral spread, video exposure/blur) and routes/weights modalities by them — this is C2's exact design principle, stated almost verbatim | Domain is stress/sentiment (StressID, CMU-MOSEI), not video/classroom/tracking. Crucially, this paper's own finding is a **warning**: "shuffling native quality signals... does not measurably change frozen fusion decisions" unless the signal actually identifies the reliable modality for that instance | Branch C must run this paper's own diagnostic (or equivalent) on its trained model before claiming the reliability gate does anything real — a naive report of "we added a reliability gate" is not evidence it works | **NARROW**, with an added burden of proof: this paper shows the failure mode where quality-aware gating is architecturally present but decision-irrelevant. C2 must be validated against exactly this failure mode, not merely implemented |
| C2c | Reliability estimated from independent diagnostic probes, aggregated by a router | **PRIME**, "Adaptive Modality Reliability Diagnosis and Restoration for Robust Multimodal Intent Recognition", arXiv:2608.03475, Aug 2026 (https://arxiv.org/html/2608.03475) | Diagnoses reliability from four probes (confidence, epistemic uncertainty, cross-modal agreement, embedding norm) via a "Contextual Reliability Estimator," explicitly **not raw features** | Intent recognition (text/audio/video triplets), not tracked multi-person video; probes are model-internal, not pipeline metadata (crop size, occlusion) | Confirm C2's crop-size/tracker-age signals are meaningfully different in kind from PRIME's internal probes | **NARROW** |
| C3 | Quality-ordering **hinge/margin loss** forcing reliability of a deliberately corrupted modality below its clean counterpart by a margin | **RAC — Ranking-Aware Calibration for Reliable Multimodal Reinforcement Learning**, arXiv:2605.16999, May 2026, Tsinghua (https://arxiv.org/html/2605.16999) | RAC's **Clean–Corrupted Pairwise Loss** is: ℓ_corr = max(0, c_k^(s) − c_k^(0) + m_corr + α·s) — a confidence margin loss that penalises a corrupted sample's confidence for being ≥ the clean sample's confidence minus a margin scaled by corruption severity s. This is functionally the **same loss C3 describes**, stated almost identically ("forcing predicted reliability of a corrupted [input] below its clean counterpart by a margin") | RAC operates on a single scalar VLM/RL rollout confidence, not per-modality r_m in a fusion network; domain is vision-language RL post-training, not video action segmentation | None realistically closes this gap other than domain transfer — the loss form itself is already published | **DROP the loss as a contribution.** This is the most damaging single finding in this audit: the exact functional form of the proposed "novel" loss already exists, dated 5 weeks before this narrowing session. C3 may be *retained only as an explicitly-cited adaptation* of RAC's loss to per-modality video reliability, never presented as original |
| C4 | Counterfactual/self-distillation loss aligning corrupted-path posterior to clean fused posterior, stop-gradient | **PRIME**, arXiv:2608.03475 (as above) | Residual restoration loss ℒ_res = ‖Δh_m,i − sg[h_clean_m,i − h_m,i]‖² — a stop-gradient consistency loss between a clean-path target and a corrupted-path reconstruction, same family as C4 | Restoration is on intermediate representations, not the final fused posterior/logits; different task (intent recognition) | Confirm C4 targets posterior/logit alignment specifically, not representation-level residual reconstruction, and that this distinction is defensible as more than a technicality | **NARROW** |
| C4b | Confidence-weighted fusion with self-distillation under incomplete modalities | **"Confidence-Aware Self-Distillation for Multimodal Sentiment Analysis with Incomplete Modalities"**, arXiv:2506.01490, 2025 (https://arxiv.org/html/2506.01490) | Same problem framing (incomplete/corrupted modality path vs. complete path), confidence derived per-modality (Student-t degrees of freedom), multi-term loss including a distillation-style term | No explicit stop-gradient posterior-alignment term found in this paper's exact loss (ℒ_total = ℒ_CE + α·ℒ_logits + β·ℒ_UF); no hinge/margin ordering term | Verify ℒ_logits does not already implement stop-gradient posterior matching before claiming C4 differs | **NARROW** |
| C4c | Self-distillation for corrupted/degraded-input consistency in tracked multi-modal video | **SDSTrack**, "Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking", arXiv:2403.16002, 2024 | Self-distillation between complete and degraded modality paths, applied to *video tracking* — closer to Branch C's domain than the sentiment-analysis papers above | Object tracking (bbox regression), not action segmentation; no explicit per-modality quality-signal reliability gate or hinge ordering loss | Differentiate on the action-segmentation target and the C2/C3 machinery layered on top | **NARROW** |
| C5 | The COMBINATION of C1–C4 inside a causal MS-TCN/ASRF backbone for classroom cue segmentation | No single paper found combining seat/task-relative canonicalisation + quality-signal reliability gating + a corruption-ordering hinge loss + stop-gradient posterior distillation inside a causal action-segmentation backbone, for classroom or any other domain | — | The **combination and application domain** (fixed-camera classroom, causal online segmentation) appear genuinely unoccupied. But three of its four ingredients (C2, C3, C4) each have close, sometimes near-identical, individual precedent from 2024-2026; C1's technique is precedented too. See §3 no-go analysis | Run the leakage-safe diagnostic (arXiv:2606.26473) and RAC-style ablations (does removing the reliability gate / hinge / distillation term change held-out event recall) before claiming the combination earns its complexity | **NARROW hard — see bottom line** |
| C5b | Closest classroom-domain systems (for context, not overlap) | "Multimodal fusion for real-time classroom engagement assessment using YOLOv9 and DeepFace", Visual Computer, 2025; "Enhancing classroom behavior analysis with multimodal data: a cross-attention fusion network approach", Scientific Reports, 2026 (https://www.nature.com/articles/s41598-026-50448-8) | Both fuse multiple modalities for classroom engagement/behaviour, establishing that classroom multimodal fusion per se is an active, crowded area | Neither does seat-relative canonicalisation, quality-signal reliability gating, or causal segmentation with a boundary-aware head (MS-TCN/ASRF); both appear to assume per-student or near-frontal capture, not this project's oblique/rear-view corpus | Confirm camera geometry in both papers (per-student webcam vs. classroom-wide) to be certain the geometric confound this thesis targets does not already apply to their setups | **Context only — not a claim to keep or drop, cite as related work establishing the crowded baseline** |

---

## 2. Two searches the brief specifically demanded

**"Has anyone already done reliability-weighted fusion with a quality-ordering
or corruption-consistency loss?"** Yes, both — separately, and recently:
RAC (arXiv:2605.16999, May 2026) for the quality-ordering hinge/margin loss
in near-identical functional form, and PRIME (arXiv:2608.03475, Aug 2026) /
the 2025 confidence-aware self-distillation paper for the corruption-
consistency / stop-gradient side. **C2+C3 was correctly flagged as the most
anticipated pair, and it is** — RAC in particular should be treated as a
near-miss rediscovery, not an independent invention.

**"Has anyone already done seat-relative or desk-relative pose normalisation
in classroom video?"** No. Direct searches for "seat-relative," "desk-
relative," and "task-relative" head-pose normalisation in classroom settings
returned nothing on-topic (closest hits were generic classroom head-pose-for-
gaze papers using camera-frame Euler angles, and proctoring-system pose
normalisation for a different purpose — compensating exam-camera placement,
not seat identity). This is the one genuinely clear piece of white space
found in this audit.

---

## 3. Strict no-go analysis (per spec)

The rule: *if the full combination AND its training objective already appear
in close prior work, recommend DROP.* No single prior-work paper implements
C1+C2+C3+C4 together in one training objective for one system. On that literal
test, C5 does not trigger an outright DROP.

But the honest picture is worse than "the combination is new, ship it":
**every individual training-objective component has independent close prior
art**, and in C3's case the loss is essentially reproduced, not merely
analogous. A reviewer who reads RAC, Multi-QuAD, and PRIME back-to-back would
reasonably conclude that Branch C reassembled four known 2024-2026 techniques
into one pipeline and applied them to a new (admittedly harder, single-camera,
un-validatable-for-viewpoint) dataset. That is a legitimate but modest
systems/application contribution — it is not a new modelling idea, and must
not be written up as one.

---

## 4. Bottom line

| Claim | Verdict | One-sentence justification |
|---|---|---|
| C1 | **NARROW** | The 6D-rotation/SO(3)-geodesic canonicalisation machinery already exists nearly verbatim (3DPCNet, 2025); only the seat/task-relative reference frame for classroom head orientation, instead of a body-centred or camera frame, is unoccupied. |
| C2 | **NARROW** | Predicting reliability from quality signals (not raw features) and softmax-gating on it is already published at least three times in 2024-2026 (Multi-QuAD, the Dartmouth diagnostic, PRIME); only the specific tracking-pipeline signal vocabulary differs, and one of those papers is itself a warning that this design pattern often does nothing measurable. |
| C3 | **DROP the loss as a contribution; keep only as a cited adaptation** | RAC's Clean–Corrupted Pairwise Loss (May 2026) is functionally the same margin-ordering loss C3 describes, four months before this narrowing session — presenting it as novel would not survive a literature check. |
| C4 | **NARROW** | Stop-gradient consistency/distillation between a corrupted and a clean path is an established 2024-2026 pattern (PRIME, confidence-aware self-distillation, SDSTrack); Branch C's version differs only in operating on the final fused posterior rather than an intermediate representation, which is a design choice, not a new mechanism. |
| C5 | **NARROW hard** | The specific combination and the classroom/causal-segmentation application are not directly anticipated, but three of its four training-objective ingredients are individually anticipated — sometimes almost exactly — so the honest contribution is integration-and-validation in a new, genuinely constrained setting, not new machinery. |

**Recommended honest wording for the contribution section:**

> For a single fixed oblique classroom camera, we adapt established
> techniques — SO(3)/6D-rotation pose canonicalisation (Zhou et al. 2019; cf.
> 3DPCNet 2025), quality-signal-driven reliability gating (cf. Multi-QuAD
> 2024/25, PRIME 2026), a quality-ordering margin loss on corrupted vs. clean
> confidence (cf. Ranking-Aware Calibration 2026), and stop-gradient
> consistency distillation (cf. confidence-aware self-distillation 2025,
> PRIME 2026) — into one causal multi-expert pipeline (MS-TCN/ASRF) for
> classroom behavioural-cue segmentation, using a seat/task-relative
> reference frame rather than the camera frame. We do not claim any of the
> four component techniques as new. We claim: (1) the seat/task-relative
> reference-frame choice is new for this application, is falsifiable by the
> measured seat/camera-frame yaw confound, and is the one piece of this
> design not found in prior work; and (2) the combination is demonstrated to
> survive end-to-end on a genuinely constrained, single-camera-pose classroom
> corpus with no held-out viewpoint — a setting no cited prior work attempts.

This wording should replace any framing that presents C2, C3, or C4 as
methodological contributions in their own right. If the write-up cannot
accept this framing (i.e., if the thesis needs C2/C3/C4 to read as novel),
the correct action is to drop those sub-claims from the contribution
statement entirely and present them as implementation, not to keep the
stronger wording and hope the committee has not read RAC.
