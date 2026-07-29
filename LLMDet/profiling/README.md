# Real-time profiling harness

Measures the numbers required by to-do item 31 before any "real-time" claim goes
into the thesis: per-stage latency (mean/P50/P95/max for detector, tracker,
feature extraction, temporal model, overlay), end-to-end FPS, peak GPU memory,
and per-stage scaling with student count.

**This is a GPU job** — it loads the LLMDet detector, CLIP, and the temporal
transformer. On the shared 8×A100 node, coordinate before running.

## Usage (from `LLMDet/`)

```bash
# Baseline pass on a real lab video (real detections):
python -m profiling.profile_pipeline \
    --config configs/attention_temporal.yaml \
    --video 0325.mp4 --max-frames 300 \
    --out work_dirs/profiling/report.json

# Scaling study: detections tiled/truncated to 5/10/20/30 synthetic students so
# tracker/feature/temporal-model cost is isolated from detector output size:
python -m profiling.profile_pipeline \
    --config configs/attention_temporal.yaml \
    --video 0325.mp4 --max-frames 200 --scaling 5,10,20,30 \
    --out work_dirs/profiling/report_scaling.json
```

Notes:
- The first `--warmup` frames (default 10) are excluded from statistics
  (model/CUDA warmup would otherwise dominate P95).
- CUDA is synchronized around every stage boundary, so per-stage times are
  accurate rather than async-launch times.
- The overlay stage draws boxes/labels but never opens a window or writes video.
- Output JSON contains one report per pass; the `"real"` entry uses the actual
  detector output, the numbered entries the synthetic counts.
