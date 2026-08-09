# Release Risk Register — Branch C (Reproducibility/Release)

Audit only, read-only, no GPU, nothing installed or downloaded. Every risk
below is backed by a command actually run against the live repo on
2026-08-09 (branch `branch-c/overt-cue`). Severity scale: **P0** blocks
release outright, **P1** must fix before release, **P2** should fix soon
after, **P3** cosmetic/low-priority.

---

## RISK 1 — Ethics/consent basis for identifiable student video is undocumented (P0)

**Risk:** The project is built on video of identifiable students in a real
classroom/lab, and per-student behavioural annotations derived from it. No
ethics approval, IRB record, consent form, participant-information sheet,
data-management plan, retention schedule, or publication permission exists
anywhere in the repository.

**Evidence:**
```
$ git grep -niE 'consent|\bethics\b|\bIRB\b|GDPR|institutional review' -- .
```
Every non-trivial hit is either unrelated corpus text (LLaVA eval
JSON/JSONL answer files containing the English word "consent"/"ethics" in
model-generated trivia, e.g. `LLMDet/llava/eval/table/results/...json`) or
this project's **own prior documentation already flagging the same gap**:
- `THESIS_DEFENSIBILITY_REVIEW.md:92`: *"The repository contains student
  video/crops but no ethics approval, consent form, participant-information
  sheet, data-management plan, retention schedule, or publication
  permission... the supervisor's notes state that approval was forgotten."*
- `THESIS_FIRST_DRAFT.md:291,778`: *"No ethics approval, consent
  documentation, data-retention plan, or publication permission is present
  in this repository."*
- `THESIS_FIRST_DRAFT.md:47`: a literal `[INSERT THE VERIFIED
  ETHICS/CONSENT BASIS...]` placeholder still unfilled in the draft thesis.

No such record was found anywhere in the tracked tree or in the untracked
working directory paths this audit was permitted to inspect. Per this
task's instructions, no consent/ethics document has been invented or
drafted here — this is reported as a finding, not remedied.

**Smallest fix:** Not a documentation fix — get the institutional
ethics/consent determination from the supervisor/ethics office first (as
`THESIS_DEFENSIBILITY_REVIEW.md` already recommends), then decide whether any
identifiable media/derived-annotation file can legally be published at all.
This blocks release independent of every other item below.

---

## RISK 2 — Per-student behavioural annotation file is tracked in git (P0)

**Risk:** `event_gold_bundle/gold_annotations_Admin.jsonl` (984 lines) and
`event_gold_bundle/events_Admin.jsonl` (1073 lines) are **tracked in git and
would ship in a public clone**. Each record carries a `file_name` that
encodes a video ID, a timestamp, and a per-student seat slot (e.g.
`t000000_000_f000000_video_0127_0_10_20251111140949_20251111142037__p00.jpg`),
plus fine-grained behavioural fields (`activity`, `gaze_direction`,
`attention_target`, `engagement_level`, `posture`, `hand_state`,
`phone_visible`, `talking`, …) for a real, identifiable student-seat/video
combination.

**Evidence:**
```
$ git ls-files event_gold_bundle/
event_gold_bundle/events_Admin.jsonl
event_gold_bundle/gold_annotations_Admin.jsonl
```
```
$ python3 -c "... json.loads(first line) ..."
{'file_name': 'shard_000/part_000/t000000_000_f000000_video_0127_0_10_20251111140949_20251111142037__p00.jpg',
 'index': 0, 'annotator': 'Admin', 'status': 'ok', 'activity': 'using_phone', ...}
```
Note: `annotator: 'Admin'` is the label-taker, not a student — no student
name strings were found in either file. This is a re-identifiable behavioural
record (video+timestamp+seat → linkable to the actual footage if anyone also
holds the (untracked) source video), not a raw image — but it is real
per-person surveillance-style data about minors/students in a classroom, and
it is currently in the tracked tree unconditionally, unlike every other
gold/session artifact in this repo which the `.gitignore` deliberately keeps
out (see RISK 3).

**Smallest fix:** `git rm --cached event_gold_bundle/*.jsonl` and add
`event_gold_bundle/` to `.gitignore` (it is currently the one gold-bundle
sibling directory *not* covered — `.gitignore` covers `gold_annotation_bundle*`
but has no `event_gold_bundle` rule at all). This alone does not scrub git
history (see RISK 6) but stops the bleeding for a fresh public push/rewrite.

---

## RISK 3 — Privacy: everything else identifiable is correctly gitignored, but the ignore rule for `event_gold_bundle/` is the one gap (P1, downgraded from what it could be)

**Risk:** Verified what identifiable media is and is not tracked:

```
$ git ls-files gold_annotation_bundle/         -> (empty)
$ git ls-files tools/dashboard/sessions/       -> (empty)
$ git ls-files LLMDet/*.mp4 LLMDet/*.tar.gz LLMDet/*.pkl -> (empty)
$ git ls-files grounding_data/ | grep -v llmstu_tools
grounding_data/llmstu_seq_split_manifest.json   # video_id/seat_id/split only, no names, no images
```
`.gitignore` (read in full) correctly excludes `grounding_data`,
`gold_annotation_bundle*`, `tools/dashboard/sessions/`, `*.mp4`, `*.pkl`,
`*.tar.gz`, and per-annotator gold outputs (`tools/gold_annotator/
gold_annotations_*.jsonl`, `gold_events_*.jsonl`). No face crops, frames, or
video are tracked anywhere in the current tree. `tools/gold_annotator/
gold_candidates_sample.jsonl` (50 lines, tracked) contains only generic
neutralised captions (e.g. *"A student is looking down at a laptop screen
with a focused expression"*) plus pseudonymous `video_id`/`seat_id` — no
names, consistent with the caption-neutralisation the project's own
`THESIS_PLAN.md` describes as an ethics measure.

**So the privacy picture is good except for the one file class in RISK 2.**
Rated P1 here (distinct from RISK 2's P0) because it's the systemic
gap — the `.gitignore` pattern that should have caught `event_gold_bundle/`
the same way it caught its sibling `gold_annotation_bundle/` — vs RISK 2's
concrete instance.

**Smallest fix:** add `event_gold_bundle/` to `.gitignore` alongside
`gold_annotation_bundle*` (one line), in addition to RISK 2's `git rm --cached`.

---

## RISK 4 — Hard-coded `/home/jovyan` absolute paths in runtime code (P1)

**Risk:** A public clone will not run out of the box on any machine where
the user is not literally named `jovyan` with the repo at this exact path.

**Evidence:**
```
$ git grep -l '/home/jovyan' -- . | wc -l
71
$ git grep -l '/home/jovyan' -- . | sed -E 's/.*\.([a-zA-Z0-9]+)$/\1/' | sort | uniq -c | sort -rn
     32 py
     30 log
      4 md
      2 jsonl
      2 json
      1 (LLMDet/work_dirs/grounding_dino_swin_t/last_checkpoint)
```
Of the 32 `.py` hits, most are auto-generated mmengine config dumps under
`LLMDet/work_dirs/.../vis_data/config.py` (training-run artifacts, force-added
per the `.gitignore` comment "results JSONs force-added selectively" —
non-blocking, these are logs of past runs, not code a new user executes) and
`LLMDet/work_dirs/grounding_dino_swin_t/grounding_dino_swin_t.py`. The
**actually-blocking** ones — real code paths a reproducer would run — are:

| File | Line | Content |
|---|---|---|
| `LLMDet/matching/evaluate_matching.py:32` | `LLMSTU_ROOT = Path("/home/jovyan/Computer_vision/grounding_data/LLMSTU")` | hardcoded default |
| `LLMDet/matching/run_matching_experiment.py:44` | `default=Path("/home/jovyan/Computer_vision/grounding_data/...")` | hardcoded argparse default |
| `grounding_data/llmstu_tools/build_seat_tracks.py:21` | `ROOT = "/home/jovyan/Computer_vision/grounding_data/LLMSTU"` | hardcoded default |
| `grounding_data/llmstu_tools/recover_video_ids.py:38` | `FRAMES_DIR = "/home/jovyan/Computer_vision/grounding_data/stu_img/frames"` | hardcoded default |
| `grounding_data/llmstu_tools/sample_gold_candidates.py:20` | `CROPS_ROOT = "/home/jovyan/Computer_vision/grounding_data/LLMSTU/crops"` | hardcoded default |
| `grounding_data/llmstu_tools/strip_labels.py:21` | `default="/home/jovyan/Computer_vision/grounding_data/LLMSTU"` | hardcoded argparse default |

`LLMDet/tools/compare_all_models.py:12` and `LLMDet/tools/infer_image.py:12`
also match, but only inside a `PYTHONPATH=/home/jovyan/...` usage example in
a docstring/comment — non-blocking (misleading example, not a hardcoded
runtime default), should still be fixed for hygiene but doesn't break a run.

The 4 `.md` hits (`INFERENCE_GUIDE.md`, `MODEL_COMPARISON_GUIDE.md`,
`outputs/thesis_audit_report.md` + its checkpoint) and the 30 `.log` hits are
documentation/logs — non-blocking by this task's own classification, but
still worth a pass since they'll read as odd to an external reader.

**Smallest fix:** replace the 6 blocking hardcoded defaults with something
derived from `Path(__file__).resolve().parents[N]` or an env var
(`os environ.get("LLMSTU_ROOT", <repo-relative default>)`), matching the
pattern `tools/dashboard/*.py` already uses correctly (`REPO = Path(__file__)...`,
no hardcoded username anywhere in that directory — dashboard code is clean).

---

## RISK 5 — No secrets found in the tracked tree (informational, not a blocker)

**Risk checked, not found.** Targeted greps for `api[_-]?key`, `token`,
`secret`, `password`, `hf_[A-Za-z0-9]`, `sk-[A-Za-z0-9]`, `Bearer `, `AWS_`,
and an AKIA-style AWS key regex across all tracked files turned up nothing
resembling a live credential:

```
$ git grep -niE '<pattern>' -- . ...
```
- The only `Bearer`/`API_KEY` hit is `LLMDet/llava/utils.py:108`:
  `"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]` — reads from
  an environment variable at runtime, no literal key committed. This is
  upstream LLaVA code, not project-specific.
- All `hf_`/`sk-`/`secret`/`token`/`password` hits are either (a) unrelated
  English words in config field names (`tokens_positive`, `max_tokens`,
  `lmm_max_token_length` — mmdet/grounding-dino config keys, not credentials)
  or (b) LLaVA's bundled eval corpus JSON/JSONL text (model-generated essay
  answers that happen to contain the word "secret") — not secrets, just text.
- No AWS-style key pattern matched anywhere.

**Git history pollution check** (cheap check only, per instructions — no
full-history content scan run):
```
$ git log --all --oneline | wc -l
39
```
Only 39 commits total across all refs — small enough that a manual
`git log -p` skim would be cheap if anyone wants to go beyond this repo-wide
grep before a public push, but that full pass was intentionally not run here
(out of scope / expensive).

**No fix needed** — recorded here so the register shows this was checked,
not skipped.

---

## RISK 6 — Large tracked files: none currently, but git history was never audited for pruned large blobs (P2)

**Risk:** Nothing over 10 MB is currently tracked, and `.git` itself is tiny:

```
$ du -sh .git .
22M  .git
312G .
```
```
$ git ls-files -z | xargs -0 du -h | sort -rh | head -5
3.8M  LLMDet/llava/eval/table/results/test_sqa_llava_13b_v0.json
3.7M  LLMDet/llava/eval/table/results/test_sqa_llava_lcs_558k_sqa_12e_vicuna_v1_3_13b.json
3.4M  LLMDet/hf_model/mm2groundingdino.pdf
1.9M  overall system.png
1.6M  LLMDet/images/test11.jpg
```
The largest tracked file is 3.8 MB; nothing approaches the ~10 MB concern
threshold. The 312 GB working-tree vs 22 MB `.git` gap confirms the huge
data/checkpoint/video assets sitting on disk (`LLMDet/0325.mp4` 99 MB,
`LLMDet/lvis_100.pkl` 187 MB, the root-level `*_bundle*.tar.gz` archives
18–37 MB each, etc.) are correctly untracked, not merely `.gitignore`d after
having once been committed and later removed (which would still bloat `.git`
— it doesn't: 22 MB total is consistent with "never committed", not
"committed then stripped").

**Residual risk (why this is still P2 and not closed):** this audit did not
walk the full 39-commit history object-by-object to confirm no now-deleted
large blob was committed-then-removed in an earlier commit and is still
sitting in `.git`'s pack (that would be a more expensive scan than
instructed to run: `git log --all --oneline | wc -l` was the intentionally
cheap proxy check, and 22 MB total `.git` size is strong indirect evidence
against it, but is not a direct proof).

**Smallest fix:** if a definitive answer is wanted before release, run
`git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3 -n -r | head -20` once — cheap, read-only, but was intentionally left for a follow-up pass since 22 MB already makes it very unlikely to change the verdict.

---

## RISK 7 — No consent/ethics doc, no dependency manifest, and no LICENSE-compatibility check for the vendored LLMDet/LLaVA/mmdet trees have not been reconciled (P2)

**Risk:** `LLMDet/` vendors mmdetection 3.3.0 (patched, see
`docs/branch_c/ENVIRONMENT_AUDIT.md` §5) and LLaVA, each under their own
upstream licenses, alongside this project's own code, under one repository
that a public push would present as a single unit. `LLMDet/LICENSE` exists
(13.7 KB, tracked) but this audit did not reconcile it against the licenses
of the vendored mmdet/LLaVA trees or against what this fork adds — that
reconciliation is out of scope for this pass (no license text was read) and
is flagged here only so it isn't silently dropped from the register.

**Smallest fix:** a follow-up pass reading `LLMDet/LICENSE` plus upstream
mmdetection's and LLaVA's licenses side by side, before any public push.

---

## RISK 8 — Git remotes point at the author's own GitHub account; public/private status not checked (P3, informational)

**Evidence:**
```
$ git remote -v
legacy  https://github.com/vaelkokach/LLMDET_STU.git (fetch/push)
origin  https://github.com/vaelkokach/LLMSTU.git (fetch/push)
```
Both remotes are `github.com/vaelkokach/...` — no third-party or
organisational remote. Per this task's instructions, no attempt was made to
contact either remote to check public/private visibility; that is a
one-click GitHub settings check for the account owner, not something this
read-only local audit can or should determine.

**No fix needed from this audit** — action item is simply "confirm visibility
in GitHub settings before/at release time," which is on the account owner.

---

## P0 items, in priority order

1. **RISK 1** — no ethics/consent basis exists for the underlying student
   video/annotation data at all. This is an institutional-approval gap, not
   a repo-hygiene gap, and it gates everything else: even a perfectly clean
   repo cannot be published if the underlying data collection itself lacks a
   documented legal/ethical basis.
2. **RISK 2** — `event_gold_bundle/{events,gold_annotations}_Admin.jsonl`
   (per-student behavioural annotations keyed to video ID + timestamp + seat)
   are tracked in git right now and would ship in any clone/push today.

## Verdict

**This repository could not be responsibly published today.** The blocking
reason is not primarily technical — the technical issues (RISK 2's two
tracked annotation files, RISK 4's six hardcoded paths, RISK 6/7's residual
unknowns) are all small and fixable in well under a day. The blocking reason
is RISK 1: there is no documented ethics/consent basis anywhere in the
repository for the identifiable student video this project is built on, and
the project's own prior review (`THESIS_DEFENSIBILITY_REVIEW.md`) already
reached the same conclusion. That determination has to come from the
supervisor/ethics office, not from a repo cleanup pass.
