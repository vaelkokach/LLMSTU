# Thesis Tables C and D

Companion to `work_dirs/thesis/tables/table_a_*.md` (visible-cue classification)
and `table_b_gold_dedup.md` (human-gold temporal events).

> **Comparability rule, applied throughout.** These four tables are never merged.
> A 78% engagement accuracy, a 0.06 MSE, a 74% mAP, a 0.50 macro-F1 and a 0.65
> R@1 answer different questions about different tasks on different data. Any
> single leaderboard containing more than one of them would be meaningless.

---

## Table C — phrase-to-student grounding (Branch A)

Open-vocabulary detector (LLMDet / Grounding-DINO-Swin-T backbone) fine-tuned on
LLMSTU ODVG. All rows use the same leak-free video-wise split and the same
`odvg_val.jsonl` evaluation set; only the caption-unit → box binding used to
build the **training** labels differs.

| binding method | training regions | R@1 | R@5 | R@10 | split | matching accuracy of the binding | calibration | retained coverage | caveats |
|---|---|---|---|---|---|---|---|---|---|
| ordinal (detector-confidence order) | 165,013 | 0.3230 | 0.8978 | 0.9811 | val | 0.2607 | — | 100% | the historic pipeline's binding; ~74% of its training phrases are attached to the wrong student |
| **Hungarian (CLIP cost)** | 165,017 | **0.4954** | **0.9595** | **0.9952** | val | **0.7896** | AUROC 0.760 / ECE 0.011 | 100% | selected binding |
| Hungarian + p ≥ 0.9 filter | 98,526 | 0.4787 | 0.9441 | 0.9913 | val | — | as above | 70.1% of assignments, at 96.2% precision | **negative result**: −0.0167 R@1 vs unfiltered |
| **exact LLMSTU correspondences (main run)** | 141,522 | **0.6343** | 0.9808 | 0.9963 | val | 1.000 by construction | — | 100% | 25k iters; upper reference for the binding ladder |
| **exact correspondences — TEST** | 141,522 | **0.6462** | **0.9891** | **0.9982** | **TEST (27 videos)** | 1.000 by construction | — | 100% | single permitted run under `TEST_SPLIT_PROTOCOL.md`, pre-registered at commit `87bb2db`; protocol now closed |

**What Table C supports.** Replacing ordinal binding with Hungarian assignment
raises R@1 by **+0.1724 (+53% relative)** while R@10 moves only +0.0141. The
divergence between top-1 and top-10 on *identical* frames, captions, boxes,
schedule, LR and seed is the evidence that the gain is **attribution** — which
description belongs to which student — and not detection ability.

**What Table C does not support.** R@k is a retrieval metric over the candidate
boxes of one frame. It is not comparable with SCB's mAP, and it says nothing
about behaviour-classification quality. Test exceeding validation by 0.0119 is
ordinary between-split variance and is *not* an improvement to claim.

**Not citable:** the 0.9231 matching value (`spatial_weight = 0.5` leaks the
ground-truth left-to-right ordering into the cost matrix) and the 1.0000
`ordinal_lr` value (correct by construction — the benchmark orders units
left-to-right and the ground truth is built the same way).

---

## Table D — external literature and datasets

Contextual reference points, **not** a leaderboard. No row here was reproduced by
this project unless the "reproduced" column says so.

| dataset | task | label definition | split unit | published metric | published result | reproduced here | comparability limitations |
|---|---|---|---|---|---|---|---|
| **CMOSE / MocoRank** (arXiv 2312.09066) | ordinal engagement classification | 4 ordered engagement levels over 12,193 online-learning segments, 103 subjects | random 70/20/10 **segment** split | overall accuracy; average (balanced) accuracy | 78.14% overall; 60.94% best average accuracy | **YES — trained here, both protocols.** Official split: accuracy **0.718**, average accuracy **0.601**, macro-F1 0.573, MAE 0.312, QWK 0.537, Spearman 0.535. **Subject-disjoint split: accuracy 0.601, average accuracy 0.435, macro-F1 0.411, MAE 0.446, QWK 0.317, Spearman 0.322.** | Different task (internal engagement, not visible cues) and domain (online coaching, one face per clip). ⚠️ **101 of its 103 subjects appear in more than one official split**, and 100 are shared between official train and test — the release assigns *clips*, not people. Re-splitting by subject costs **QWK −0.220** and **average accuracy −0.166**. Any comparison with this project's video-wise numbers must state that the split policies differ by that much. Reports ICC(2,1)=0.84 for annotator agreement. |
| **DAiSEE** | affect / engagement classification | 4 levels each for boredom, confusion, engagement, frustration; 9,068 clips, 112 users | user-wise clip split | accuracy per dimension | varies by baseline paper | ✗ not run | Webcam-style single-person clips vs a wide 2812×1050 lab shot; a body box fills ~60% of a DAiSEE frame vs ~5% here. The domain gap is the same one that collapsed our model on DIPSER (§6c). |
| **EngageWild / EmotiW 2018** (arXiv 1804.00858) | continuous / ordinal engagement regression | engagement intensity as an ordered value | subject-wise | MSE (principal), Pearson | best test MSE ≈ 0.06 vs 0.15 baseline | ✗ not run | A regression metric. It is not defined over this project's six **nominal** cue classes, and computing an MSE over them would be a category error. |
| **EngageNet** (arXiv 2302.00431) | engagement classification | AU / gaze / head-pose features + learned representations | subject-wise | accuracy | see paper | ✗ not run | Useful as a *feature-family* reference (it uses head pose and gaze, as we do), not as a score to beat. |
| **Authentic-classroom engagement** (arXiv 2101.04215) | binary/continuous engagement from facial video | teacher/observer ratings | student-wise | AUC | 0.620 and 0.720 in two grade groups; **+0.084 AUC from limited person-specific personalisation** | ✗ not run | The personalisation result is the directly relevant one: it independently supports this project's **personalised median-pose gaze baseline** design (§6d.2), even though our isolated ablation found the resulting features add nothing to macro-F1 (§11.4). |
| **SCB-Dataset** (arXiv 2304.02488) | classroom behaviour **bounding-box detection** | 2–20 behaviour classes, boxes | image-wise | precision, recall, mAP@0.5, mAP@[0.5:0.95] | e.g. 74.0% mAP@0.5, 56.8% higher-IoU mAP on a 3-class subset; the paper omits bow-head and turn-head as unsatisfactory | **partially** — zero-shot transfer of our frozen detector: R@1 0.0322, localisation recall 0.392, precision 0.187 | ⛔ **The precision is uninterpretable and the R@k are depressed by construction**: SCB annotates only behaviour-exhibiting students, so correctly detecting an ordinary seated student scores as a false positive. The label mapping is also semantically wrong for a lecture hall (a bowed head there usually means note-taking, an *on-task* behaviour). Report as a qualitative domain-shift limitation, never as a transfer score. |
| **DIPSER** (arXiv 2502.20209) | in-person attention + academic emotion | 1–5 attention, 9 academic emotions, head pose, gaze, wearables | subject-wise | — | — | **yes, two experiments** — (a) cue↔engagement Spearman ρ = **+0.172**, permutation p < 0.0001, 25 subjects, 1,825 pairs; (b) basic-expression → expert boredom AUROC **0.544**, 1,176 pairs, held-out subjects | (a) The correlation is **significant and runs backwards**: in a lecture hall, head movement means note-taking and tracking the instructor, i.e. engagement. Same pixels, opposite meaning. This is the project's strongest evidence for the visible-cue framing. ρ = 0.172 is nonetheless *weak*, ratings are skewed (1,539/1,825 at levels 2–3) and rating 1 has n = 1. (b) 0.544 is barely above chance: the 7 basic **Ekman** expressions do not deliver **Pekrun** academic emotions. |
| **This project — Branch A** | phrase-to-student grounding | open-vocabulary phrase → student box | **video-wise** | R@1 / R@5 / R@10 | **0.6462 / 0.9891 / 0.9982** (test) | — | Retrieval over one frame's candidate boxes. Not comparable with SCB mAP. |
| **This project — Branch B** | 6-class visible-cue frame classification + episode localisation | observable cues only; never a mental state | **video-wise** | macro-F1, balanced accuracy, macro-AUPRC, ECE; segmental F1/edit; event P/R/F1 | see Tables A and B | — | Labels are pseudo-labels from a Qwen3-VL-family teacher whose measured agreement with human gold is 87.4% frame-level and 0.625 event recall. Test measures generalisation to unseen **videos**, not agreement with humans. |

### External-dataset scope: what was trained, and what was not

**Trained: CMOSE** (13.6 GB, CC-BY-SA-4.0, ungated). It is the addendum's P2.14
and the best value of the candidates, because the release ships precomputed I3D
and OpenFace features — no video preprocessing at all. A separate four-level
ordinal head was trained under both the published split and a subject-disjoint
one; see the CMOSE row above and `FINDINGS.md §11.11`. Full result:
`work_dirs/thesis/cmose/cmose_results.json`, code `attention/thesis_eval/cmose.py`.

The most useful thing it produced was not the accuracy. It was the measurement
that **the published split shares 100 of 103 subjects between train and test**,
and that closing that leak costs quadratic weighted kappa 0.537 → 0.317. That is
independent, quantified corroboration of this project's own most expensive
lesson — the March pipeline was invalidated by a leaked split — and it is a
concrete methodological contribution rather than another accuracy number.

**Not trained, and why:**

| dataset | reason |
|---|---|
| **DAiSEE** | Requires downloading and preprocessing ~9,000 webcam videos; the task (4-level affect on single-person webcam clips) is a strictly weaker version of what CMOSE already provides, on a domain even further from a wide lab shot. Low marginal value once CMOSE is done. |
| **EngageWild / EmotiW** | Challenge data is access-restricted rather than openly downloadable. Its regression metrics (MSE, Pearson) are reported in Table D as literature reference points only. |
| **EngageNet** | Same task family as CMOSE; would duplicate the engagement experiment without adding a distinct question. |
| **DIPSER (full, 780 GB)** | Already used for the two experiments that matter (cue↔engagement correlation, expert-boredom validation) from a 26 GB subset. The addendum's main remaining DIPSER use was head-pose/gaze *supervision* — and `FINDINGS §11.10` has since shown that ~80% of the head-pose contribution is the binary `face_found` flag, not the metric angles, so better angular supervision is no longer the promising direction it appeared to be. Deliberately redirected, not skipped for cost. |
| **SCB** | Already downloaded (6.3 GB) and converted; used as a zero-shot domain-shift target. Its annotation is non-exhaustive, so its precision is uninterpretable — see the SCB row above. |

Disk is no longer the constraint (652 GB free at the time of writing). The
constraint is that each additional dataset must answer a *distinct* question the
thesis needs answered, and after CMOSE the remaining candidates largely answer
the same one.
