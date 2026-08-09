# Branch C independent review log

Cross-agent findings, resolutions and sign-offs at each gate, per the driving
specification's multi-agent section. The lead is accountable for every entry: no
agent's conclusion was accepted without inspecting its evidence, and where an
agent's claim was wrong or over-stated that is recorded here rather than quietly
corrected.

Roles were separated by **file ownership** — no two writers ever held the same
file — and by **context**, with reviewers given the artifact rather than the
author's summary.

---

## R1 — Data audit vs the specification's premise

**Author:** Data/Evaluation agent · **Reviewer:** lead · **Gate:** Phase 0

The specification asserts "fixed overhead multi-student video" throughout and
names the method for it. The lead independently sampled six recordings across four
capture dates and rendered them before accepting the premise.

**Finding:** the premise is false. One room, one fixed corner-mounted camera at
~20–30° elevation. **Resolution:** claim narrowed before any implementation,
confirmed with the researcher, recorded in FINDINGS §12.1. The repository's own
system diagram already said "corner-camera classroom/lab videos" — only the
Branch-C prompt called it overhead.

**Sign-off:** ✅ lead. The audit and the lead's independent sampling agree.

---

## R2 — Novelty audit: verify before acting

**Author:** Novelty/License agent · **Reviewer:** lead · **Gate:** Phase 0

The agent recommended **dropping** a thesis contribution (C3) on the strength of
five specific arXiv identifiers. An agent recommending a contribution be dropped is
exactly the claim not to take on trust, and fabricated identifiers are the
characteristic failure mode.

**Lead action:** fetched all five. All resolve to real papers whose abstracts match
the descriptions.

**One over-claim corrected:** the agent stated PRIME (arXiv:2608.03475) anticipates
C4's stop-gradient clean/corrupt distillation. Its abstract does not mention that
mechanism. C4 therefore rests on thinner evidence than the audit claimed, and
FINDINGS §12.8 records it as such rather than repeating the agent's wording.

**Sign-off:** ✅ lead, with the C4 correction applied.

---

## R3 — Licence audit gates a design decision

**Author:** Novelty/License agent · **Reviewer:** lead · **Gate:** pre-implementation

Ultralytics YOLOv8/YOLO11 is AGPL-3.0 including weights; DirectMHP is GPL-3.0 via
YOLOv5. Both would impose copyleft on the combined work.

**Resolution:** both rejected before any code was written. 6DRepNet360 (MIT)
adopted as the full-range head-pose candidate. The specification *proposed*
DirectMHP; the licence audit overrode it. Recorded in FINDINGS §12.9.

**Sign-off:** ✅ lead. Verified the MIT licence in the upstream repository directly.

---

## R4 — Protocol and fold construction

**Author:** lead · **Reviewer:** automated tests + the specification's own criteria

The lead authored the protocol, so the review here is mechanical rather than
social: `LLMDet/attention/tests/test_branch_c_splits.py` (8 tests) proves no video
crosses a fold, every fold partitions the corpus exactly, the manifest hash still
describes its contents, and per-class support is balanced within 2× across folds.

**Self-caught defect:** the first fold assignment was degenerate — 40/61/26/0/0
videos — because every video is ~85% `screen_oriented`, so the prevalence cost is
nearly flat and a soft size penalty never bites. Video count is now a hard
constraint with deterministic swap refinement. This mattered: `turned_to_peer` has
63 sequences in the entire corpus and an unrefined fold held one of them.

**Sign-off:** ✅ tests green, manifest frozen and committed before any result was
read.

---

## R5 — Method code review

**Author:** Method agents (fusion/losses) and lead (canonicalisation) ·
**Reviewer:** the test suite, written against the specification's requirements

The canonicalisation agent was killed by a session limit before producing anything;
the lead wrote `canonical.py` directly. `fusion.py` and `losses.py` survived their
author's termination **untested**, and were not trusted on that basis.

**Two defects found by tests written afterwards, neither visible by reading:**

1. `matrix_to_rot6d` broke its own round trip. Reshaping the `(3, 2)` column slice
   is row-major, so it emitted the two columns interleaved while `rot6d_to_matrix`
   reads them as consecutive. It would never have crashed — it would have surfaced
   months later as a pose feature that meant nothing.
2. A single missing expert turned the whole fused output to NaN. `alpha` is exactly
   0 for an unavailable modality, but IEEE `0 * NaN` is `NaN`. One missing
   head-pose estimate would have poisoned every cue logit on that track.

**Sign-off:** ✅ lead, after both were fixed and pinned by regression tests.

---

## R6 — The pose gate (the branch's decisive gate)

**Author:** lead · **Reviewer:** the pre-registered criterion itself

The gate was written into BRANCH_C_PROTOCOL.md amendment A1 **before the cache it
would judge existed**, so the criterion could not be reshaped after seeing the
numbers.

**Instrument validated before use, convention-free:** rotating the input in-plane
by a known angle and measuring the geodesic between predicted rotations recovers
9.7°/19.3°/39.6° for applied 10/20/40°. The upstream 6D→matrix map was checked
numerically against the local one (max deviation 3.5e-6) rather than by reading the
derivation.

**Result:** H1 confirmed, H2 rejected (+0.0083 mean AUC delta). Arms 6 and 13
dropped.

**A judgement flagged rather than made quietly:** A1 says a null gate means arms 5,
6 and 13 are not trained. Read literally that discards arm 5 — absolute full-range
pose — which the gate had just shown is the strong condition. The lead took the
reading that A1's purpose was to decide *canonicalisation*, and that arm 5 is the
control against which canonicalisation is judged, so the negative cannot be
reported without it. This was put to the researcher explicitly and accepted.

**Sign-off:** ✅ researcher, on an explicitly-flagged judgement call.

---

## R7 — Independent recomputation of the headline result

**Author:** lead · **Reviewer:** Data/Evaluation agent (fresh context, own code)

The reviewer was instructed to recompute from the 45 raw `metrics.json` files
rather than from the lead's `inner_val_results.json`, and to report disagreement
loudly.

**Agreement:** the scientific conclusion reproduced independently. arm5 − arm2
recomputed as **−0.0023, CI [−0.0123, +0.0084]**, against the lead's −0.0022,
CI [−0.0119, +0.0085] — bootstrap noise. Claims about per-class CIs and about
calibration confirmed exactly.

**Four integrity findings the lead had not flagged, all valid:**

| # | finding | resolution |
|---|---|---|
| 1 | `outputs/branch_c/eval/` had 46 dirs, not 45 — `arm5_f0_s42` a superseded duplicate | **Fixed.** It was the lead's one-off timing test; deleted. |
| 2 | `RUNS.jsonl` missing 4 of 45 train entries, all from a different commit | **Explained, not fixed.** Those are the four runs that completed before the dispatcher's run-id parsing bug crashed it (FINDINGS §12; `"..._ff_f0_s42".split("_f")` splits on the `ff`). Their `run_record.json` files are intact and are what the register uses. The gap is in the convenience log, not in the evidence. |
| 3 | The §5.1 shortcut audit was dispatched but incomplete | **Valid at the time of review.** It was mid-flight; completed subsequently as arm 0. |
| 4 | Every per-run macro-F1 differs from raw `metrics.json` by ~1e-4, both directions | **Diagnosed — see R8.** Not a recomputation error. |

**One over-reach corrected:** the reviewer treated the standalone pose scalar's AUC
(0.830/0.852) as uncitable because no raw artifact was archived. That is correct and
the lead accepts it: those five numbers come from an unarchived development-only
pre-test and must not be presented as a controlled contrast. They are context for
why the arm was run, not evidence in a results table.

**Sign-off:** ✅ both. Conclusion independently reproduced; four defects found and
resolved.

---

## R8 — Batch-dependent inference, diagnosed

**Raised by:** R7 finding 4 · **Diagnosed by:** lead

The training loop's in-loop validation and the standalone evaluator disagree by
1e-4 to 3e-4 macro-F1, in both directions, **on the same checkpoint at the same
epoch**.

**Cause:** the training loop validates in padded batches; `run_eval` uses batch
size 1. A dilated TCN's kernel reads zero-padding from batch neighbours within its
receptive field at sequence tails. Masking the *output* does not undo the fact that
the convolution *read* those zeros.

**Consequence:** a sequence's prediction depends slightly on what else is in its
batch. The effect is ~1e-4 against contrasts of interest around 2e-3, so no
conclusion moves — but it is a real determinism caveat and is recorded rather than
left as an anomaly.

**Resolution:** the register and every reported number use the **batch-size-1
evaluator**, which is free of cross-sequence contamination. Sign-off ✅ lead.

---

## R9 — Fusion trainer collapse, caught before the sweep

**Author:** lead · **Reviewer:** a smoke test on a known-answer arm

Before dispatching 60 fusion runs, the appearance-only arm was trained as a sanity
check: it should approximately reproduce the MS-TCN baseline (~0.48–0.54).

**It did not.** macro-F1 sat at exactly 0.1436 for every epoch of every arm — a
constant prediction.

**Cause:** the loss was applied only to the final refinement output. In MS-TCN the
loss is applied to *every* stage; without that, the experts and the gate receive
gradient only through the refinement chain's 6-dim softmax bottleneck and the model
collapses.

**Fix:** deep supervision on every stage, including the fused prediction that
precedes refinement. arm 3 then reached 0.5292 on fold 0, in line with the
baselines.

**Why this is logged:** had the sweep been launched first, 60 runs would have
produced a uniformly null fusion result that looked like a scientific finding and
was a bug. Sign-off ✅ lead.

---

## Open items carried to the researcher

Neither is resolvable by any agent:

1. **Per-student behavioural annotations tracked in git** — 4 files, 3,107 records
   keyed to identifiable frames. `event_gold_bundle/` has no `.gitignore` rule while
   its sibling `gold_annotation_bundle*` does. Removal needs a history rewrite on a
   remote whose visibility was not verified. Left untouched deliberately.
2. **No ethics/consent record** — already documented at
   `THESIS_DEFENSIBILITY_REVIEW.md:92` and left as an explicit `[INSERT...]`
   placeholder in the draft. Specification guardrail 9 forbids inventing one, and
   nothing here does.
