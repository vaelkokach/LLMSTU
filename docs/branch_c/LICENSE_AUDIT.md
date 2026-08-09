# Branch C — License Audit

**Date:** 2026-08-09. Covers every external component named in the brief,
plus everything already vendored under `/home/jovyan/Computer_vision/LLMDet`
and `/home/jovyan/Computer_vision/huggingface`. Licenses below were read from
the actual upstream LICENSE file (fetched live from GitHub/Hugging Face on
2026-08-09), not from memory, except where explicitly marked otherwise.

**Confirmed by grep of the current codebase:** none of Ultralytics
YOLO/YOLOv8/YOLO11, ByteTrack, BoT-SORT, WHENet, DirectMHP, 6DRepNet(360),
SlowFast/PySlowFast, or VideoMAE are currently imported anywhere in
`LLMDet/attention` or `tools/`. The tracker in use is an in-house `IoUTracker`
(`LLMDet/attention/tracking.py`); the detector is LLMDet/MMDetection; the
appearance embedding is `openai/clip-vit-base-patch32` via `transformers`;
head pose currently runs on MediaPipe, with 6DRepNet named in
`head_pose.py` as "needs external weights" (not yet wired in). This audit is
therefore mostly forward-looking: which of the brief's named candidates are
safe to add, and which are traps to avoid.

---

## 1. Component table

| Component | Upstream repo | License (code) | Weights vs. code | Redistribution | Recommendation |
|---|---|---|---|---|---|
| **Ultralytics YOLOv8 / YOLO11** | github.com/ultralytics/ultralytics | **AGPL-3.0** (verified from repo LICENSE, 2026-08-09) | Pretrained weights are distributed under the **same AGPL-3.0** terms as the code per Ultralytics' own documentation — there is no separate permissive weight license | AGPL-3.0 permits redistribution, but its network-use clause requires that the **complete corresponding source of the whole combined work** be made available to anyone who interacts with it over a network — this includes a served dashboard, not just a downloaded binary | **AVOID.** Not currently used — keep it that way. This is the named trap: Ultralytics sells an Enterprise License specifically because AGPL-3.0 is incompatible with closed/mixed-license distribution. Adding YOLO11/YOLOv8 to this repo would put the *entire* combined codebase (including LLMDet's Apache-2.0 code) under AGPL's network-disclosure obligation for anyone who runs the dashboard. If a YOLO-family detector is ever wanted, do not vendor Ultralytics; there is no safe "just for research" carve-out once the repo is public |
| **ByteTrack** | github.com/ifzhang/ByteTrack | **MIT** (verified) | Reference detector is YOLOX (Megvii, Apache-2.0) — no divergence | Permitted | Safe to vendor if a published-algorithm tracker is ever preferred over the in-house `IoUTracker`. Not currently needed |
| **BoT-SORT** | github.com/NirAharon/BoT-SORT | **MIT** (verified) | The tracker code itself is detector-agnostic and MIT; the repo's bundled example detector is **YOLOv7** (WongKinYiu, historically GPL-3.0) | Tracker logic alone: permitted. Bundled YOLOv7: copyleft risk | **Vendor tracker logic only, never the bundled detector.** Same caution as YOLO above — do not pull in the YOLOv7 reference pipeline wholesale |
| **WHENet** | github.com/Ascend-Research/HeadPoseEstimation-WHENet | **No LICENSE file in the official repo** (confirmed 404 on 2026-08-09). A third-party vendor's copy (PINTO0309/DMHead, `LICENSE.WHENet`) states **BSD-3-Clause, Huawei Technologies 2020** | Weights follow whatever the code license turns out to be; unverified at source | Ambiguous — a downstream mirror's claim is not a substitute for the upstream owner's own license grant | **Avoid depending on the official repo's code as-is.** The absence of a LICENSE file in the canonical repo means default copyright applies regardless of what a third party's mirror says. If WHENet is wanted, get written confirmation of terms from Huawei/Ascend-Research, or prefer 6DRepNet360 below, which solves the same full-range problem under a clean MIT license already committed to a LICENSE.MD file |
| **DirectMHP** | github.com/hnuzhy/DirectMHP | **GPL-3.0** (verified — repo explicitly names "GNU General Public License v3.0") | Built on **YOLOv5** (Ultralytics) as the base detector | GPL-3.0 is strong copyleft: any distributed derivative must also be GPL-3.0 | **AVOID vendoring.** Same failure mode as Ultralytics YOLO — GPL-3.0 would force copyleft obligations onto the combined repo if this code is incorporated (not merely cited). Full-range head pose is already achievable via 6DRepNet360 (MIT) without this risk |
| **6DRepNet** | github.com/thohemp/6DRepNet | **MIT** (verified) | No divergence found | Permitted | Safe. Already named in `head_pose.py` as the escalation path if MediaPipe's face-found rate is insufficient |
| **6DRepNet360** | github.com/thohemp/6DRepNet360 | **MIT** (per repo's LICENSE.MD, confirmed via repo license badge) | No divergence found | Permitted | **Preferred full-range head-pose backend** given this corpus's rear/three-quarter view dominance (FINDINGS §12.1 point 4) — same problem WHENet/DirectMHP target, without their licensing risk |
| **SlowFast / PySlowFast** | github.com/facebookresearch/SlowFast | **Apache 2.0** (verified) | Code confirmed Apache-2.0; historically some FAIR model-zoo checkpoints carry an added research-only note in the model zoo docs rather than the top-level LICENSE — **not independently verified this session for SlowFast's specific Kinetics checkpoints** | Code: permitted. Weights: verify before redistributing any specific checkpoint | Not currently used. If added, re-check the model-zoo page for the exact checkpoint intended, not just the repo LICENSE |
| **VideoMAE** | github.com/MCG-NJU/VideoMAE | **CC BY-NC 4.0** (verified) | Same NC license applies to the released pretrained weights | **Non-commercial only** — this restricts any redistribution, including as part of a public repo intended for unrestricted reuse | **Flag prominently if ever added.** Not currently used. A thesis/academic public repo can depend on NC-licensed work for research, but the NC restriction then attaches to that part of the combined repo and must be stated explicitly — do not let "open source repo" framing imply unrestricted reuse if VideoMAE weights are shipped |
| **MS-TCN** (official) | github.com/yabufarha/ms-tcn | **MIT + Commons Clause v1.0** — Commons Clause explicitly prohibits **Selling** the software (offering it as a paid product/service) | Same restriction attaches to any copied code | Copying restricted by the Sell clause | **Not a live issue**: `LLMDet/attention/thesis_eval/models.py`'s MS-TCN implementation is confirmed **independent, from-scratch code**, not copied from this repo — copyright/license attaches to the specific code expression, not to the published architecture. Document this decision explicitly (this file does) so no future contributor pastes code from the official repo and silently imports the Commons Clause restriction |
| **ASRF** | github.com/yiskw713/asrf | **MIT** (verified) | No divergence | Permitted | Not a concern either way — this repo's ASRF is also an independent reimplementation, and MIT would have permitted copying regardless |
| **MMDetection** | github.com/open-mmlab/mmdetection | **Apache 2.0** (verified) | No divergence | Permitted | Safe — already a live dependency (`LLMDet/attention/detector_adapter.py` imports `mmdet`) |
| **mmengine** | github.com/open-mmlab/mmengine | **Apache 2.0** (verified) | No divergence | Permitted | Safe — already a live dependency (`grounding_data/QWEN3-VL/requirements.txt`) |
| **Grounding-DINO** | github.com/IDEA-Research/GroundingDINO | **Apache 2.0** (verified) | Weights (e.g. `groundingdino_swint_ogc.pth`) ship alongside the Apache-2.0 code with no separate license text found, but are trained on Objects365 + GoldG (Flickr30k + Visual Genome + GQA) + Cap4M — those source datasets carry their own provenance/use norms (Flickr30k images in particular remain under original uploaders' individual licenses) | Code: permitted. Weights: standard open-vocab-detection provenance caveat, not a hard blocker | Keep using (already central to Branch A/LLMDet). No raw GoldG/Flickr30k images are redistributed by this repo — it only ships weights and runs them on the private LLMSTU videos, which is the normal, accepted use pattern in this field |
| **MM-Grounding-DINO** | github.com/open-mmlab/mmdetection (configs/mm_grounding_dino) | **Apache 2.0** (verified, same LICENSE as mmdetection) | Three `.pth` files already vendored at `huggingface/mm_grounding_dino/` (Swin-T/B/L). The README states only "we release all our models to the research community," with **no explicit separate weight license** and no license note on the Objects365/GoldG/GRIT/V3Det training data | Code: permitted. Weights: research-use provenance caveat, same category as plain Grounding-DINO, slightly less documented | Fine for continued thesis use; do not present these three checkpoints as cleanly Apache-2.0 in a public-release license summary — note the provenance caveat once, in this file and in any repo-root NOTICE |
| **CLIP** (`openai/clip-vit-base-patch32`) | github.com/openai/CLIP; huggingface.co/openai/clip-vit-base-patch32 | **MIT** (verified, both the GitHub repo and the HF model card) | No license divergence; OpenAI's model card separately describes the model as "primarily intended for research," which is a usage note, not a legal restriction | Permitted | Safe — already the live appearance-embedding backbone (`LLMDet/attention/features.py`). The research-use note is worth one honest sentence in the thesis's ethics/limitations section, but is not a redistribution blocker |
| **MediaPipe / BlazeFace** (`face_landmarker.task`, `blaze_face_short_range.tflite`, `face_detection_full_range.tflite`) | github.com/google-ai-edge/mediapipe | **Apache 2.0** (verified for the repo; model files' own model cards also state Apache 2.0) | No divergence found. The separate "MediaPipe Solutions API Terms of Service" governs on-device API/telemetry behaviour, not redistribution rights, and does not add restrictions to the model files themselves | Permitted | Safe — already vendored at `huggingface/mediapipe/` and in live use as the head-pose backend |
| **LLMDet** (upstream of this repo's `LLMDet/`) | github.com/idea-research/LLMDet; huggingface.co/fushh7/LLMDet | **Apache 2.0** (verified from the repo's own `LLMDet/LICENSE` file and from the HF model card's `apache-2.0` tag) | Weights: same `apache-2.0` tag on the HF model card, no divergence | Permitted | Safe. `LLMDet/LICENSE` is itself a bundle — Apache-2.0 boilerplate followed by NOTICE-style attributions for **DAB-DETR** (Apache 2.0, IDEA), **Conditional DETR** (Apache 2.0, Microsoft), **Deformable DETR** (Apache 2.0, SenseTime), and **DETR** (Apache 2.0, Facebook) — all sub-dependencies are Apache-2.0, no conflicts. Keep this file intact; Apache-2.0 §4(d) requires its attribution notices survive in any redistribution |
| **huggingface/bert-base-uncased** | huggingface.co/bert-base-uncased | **Apache 2.0** (verified from README) | No divergence | Permitted | Safe |
| **huggingface/siglip-so400m-patch14-384** | huggingface.co/google/siglip-so400m-patch14-384 | **Apache 2.0** (verified from README) | No divergence | Permitted | Safe |
| **huggingface/my_llava-onevision-qwen2-0.5b-ov-2** | huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov | **Apache 2.0** (verified from the lmms-lab model card) | Base LM is Qwen2-0.5B; Qwen's license has historically varied by model size/version (some larger Qwen releases used the more restrictive Tongyi Qianwen License) — **not independently re-verified this session for this exact Qwen2-0.5B point release** | Apache-2.0 as declared, pending the Qwen2-0.5B check | Low priority: grep found no evidence this checkpoint is wired into the live Branch B/C attention pipeline (looks like a Branch A caption-model leftover). If Branch C ever calls it directly, re-verify Qwen2-0.5B's exact license file before relying on the lmms-lab card alone |
| **huggingface/fer_vit** (`dima806/facial_emotions_image_detection`, base `google/vit-base-patch16-224-in21k`) | huggingface.co/dima806/facial_emotions_image_detection | **Apache 2.0** (verified from README) | No divergence | Permitted | Safe license-wise. Independently, THESIS_PLAN.md §5 already excludes facial-expression/emotion claims from the thesis on epistemic/empirical/ethical grounds — this is a scoping decision, not a licensing one; recorded here only for completeness |

---

## 2. What would prevent public release of this repository

1. **No top-level LICENSE file exists yet** (`ls /home/jovyan/Computer_vision`
   confirmed none). Every dependency audited above is compatible with
   Apache-2.0 (which `LLMDet/LICENSE` already uses), except the two
   *hypothetical* additions flagged below. Before any public release, add a
   root `LICENSE` (Apache-2.0 is the natural choice, matching LLMDet and
   almost everything else in the stack) and a `NOTICE` file carrying forward
   the attributions already listed in `LLMDet/LICENSE`.
2. **Ultralytics YOLO (YOLOv8/YOLO11), if ever added, is the single biggest
   release risk.** AGPL-3.0's network-use clause would obligate disclosing
   the complete source of the *combined* work (including all the Apache-2.0
   code around it) to anyone who interacts with a served dashboard. This is
   not currently a problem because the component is not used — the finding
   is purely preventive: do not add it without either (a) the Ultralytics
   Enterprise License, or (b) accepting AGPL for the whole repo.
3. **DirectMHP (GPL-3.0), if ever vendored directly** (as opposed to cited),
   is the second-biggest hypothetical risk, for the same copyleft reason.
   6DRepNet360 (MIT) is the drop-in substitute for the same full-range
   head-pose problem and removes this risk entirely.
4. **VideoMAE (CC BY-NC 4.0), if ever added,** would make part of the repo
   non-commercial-only. Not currently used, and not on the current
   implementation path (head pose runs on MediaPipe/6DRepNet, not VideoMAE),
   but named in the brief as a candidate — flagged so it is not added
   casually.
5. **WHENet's official repo has no LICENSE file at all.** Do not depend on
   it without written confirmation from the maintainers; prefer 6DRepNet360.
6. **Grounding-DINO / MM-Grounding-DINO weight provenance** (Objects365,
   GoldG/Flickr30k, GRIT, V3Det) is a standard, field-wide caveat, not a
   blocker — but it should be named once in the repo's NOTICE so the public
   release is not silently overclaiming a clean Apache-2.0 status for those
   three `.pth` files.

**Everything currently imported and running in this codebase — MMDetection,
mmengine, the LLMDet weights, CLIP, MediaPipe, the in-house `IoUTracker`, and
the from-scratch MS-TCN/ASRF implementations — is Apache-2.0/MIT-clean and
release-safe today.** The risk in this audit is entirely about candidates
named in the brief that are not yet in the codebase; the recommendation is to
keep it that way for YOLO-family and DirectMHP specifically, and to prefer
6DRepNet/6DRepNet360 wherever the brief's other candidates (WHENet,
DirectMHP) were being considered for the same job.
