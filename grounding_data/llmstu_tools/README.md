# llmstu_tools — LLMSTU dataset preparation pipeline

Turns the raw `grounding_data/LLMSTU` per-student crop dataset (283,913 crops,
1 fps sampling, no video IDs, no splits) into leak-free, deduplicated,
class-balanced training assets.

## Pipeline order

| # | script | output (in `outputs/` unless noted) |
|---|--------|--------------------------------------|
| 1 | `strip_labels.py` | `LLMSTU/labels_slim.jsonl` — 21 shards merged, `raw_caption`/`crop_path` dropped |
| 2 | `recover_video_ids.py --extract` then `--assign` | `frame_to_video.json`, `video_recovery_report.json` |
| 3 | `build_seat_tracks.py` | `labels_tracked.jsonl` (+`video_id`, `seat_id`) |
| 4 | `dedup_subsample.py` | `labels_dedup.jsonl`, `dedup_report.json` |
| 5 | `make_splits.py` | `splits.json`, `labels_{train,val,test}.jsonl` |
| 6 | `regen_odvg.py` | `odvg_{train,val,test}.jsonl`, `correspondence_gt.jsonl` |
| 7 | `sample_gold_candidates.py` | `gold_candidates.jsonl` (1,000 crops for the annotation tool) |

## Key design decisions

* **Video identity** is recovered by appearance-chain tracking: all 128 videos
  start at frame index 0 (seeds), consecutive sampled frames are ~1 s apart
  with a constant f-step of 19/20, matched per step with Hungarian assignment
  on 64×24 grayscale thumbnails, then fragment chains are merged back
  (f-disjointness + anchor-name guidance). Validation on the 4,660
  `_video_`-suffixed anchor frames: **100% accuracy, 128 chains, 0 mixed
  chains** (`video_recovery_report.json`).
* **Seat identity**: `person_idx` in LLMSTU is a det-confidence rank, NOT a
  track ID. Seats come from DBSCAN on head-center points per video
  (eps = 0.5 × median head span). 776 seat tracks over 127 videos (one of the
  128 videos contributed no crops that passed the LLMSTU head-size filter).
* **Dedup**: per (video, seat), run-length encoding over the 6-field label
  state; every state transition is kept, plus a 10 s stride inside runs.
* **Occlusion policy**: `occluded=True AND face_kpts==2` crops are dropped
  (visually unverifiable); other occluded crops kept, flag preserved.
* **Class cap**: `listening`+`using_laptop` capped at 50% of the final set.
* **Splits are video-wise (70/15/15)** — never frame/shard/random-line, so
  1 fps near-duplicates cannot leak across splits.
* **ODVG regeneration**: frames chosen by dedup, but every entry carries
  boxes for *all* students in the frame (missing boxes would train false
  negatives). Phrases are ≤3-word templates from the closed `activity`
  vocabulary; `tokens_positive` are char offsets (validated 0 mismatches).
  This yields *exact* box↔phrase correspondence — unlike the old
  `stu_img/annotations` files whose "Student N"→box binding was arbitrary.
* **`correspondence_gt.jsonl`** (93,736 frames, all students each) is the
  matching-evaluation ground truth for the ordinal/Hungarian/OT comparison.
* **Caption neutralisation**: religious dress (hijab/niqab/headscarf/turban),
  gender terms, and beard/moustache descriptors are stripped or neutralised
  (`he/she`→`they` etc.) before captions enter any training asset. Residual
  audit: 0 hits. Minor grammar artifacts ("they listens") are accepted.

## Caveats

* One video never collided in filenames and thus has no name; it appears as
  `video_unk_000`.
* `labels_tracked.jsonl` retains seat noise rows (`seat_id=-1`, n=1,178);
  downstream steps skip them.
* Everything is CPU-only and deterministic (seeded) except DBSCAN ordering,
  which is stable for fixed input order.
