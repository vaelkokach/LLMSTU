"""Deployment-side model loading, calibration and abstention.

Why this file exists
--------------------
The live path used to build the temporal model from **YAML** and then load a
checkpoint into it with ``strict=False`` inside a bare ``try/except``:

    model = AttentionTransformer(input_dim=cfg["model"]["input_dim"], ...)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"[dashboard] temporal checkpoint not loaded ({e})")

``strict=False`` tolerates missing and unexpected keys but still **raises** on a
size mismatch, so a config declaring ``input_dim: 570`` against a 552-dim
checkpoint took the ``except`` branch, printed one line, and ran the dashboard on
a **fully randomly initialised network**. That is precisely the silent-plausibility
failure the March post-mortem exists to prevent, and it is what
`attention_temporal_full.yaml` was configured to do.

The same path also zero-padded feature vectors up to the config's width, so any
block the live extractor could not produce became a run of zeros — indistinguishable
from a genuine measurement, the trap that made the OpenCV head-pose backend
useless (FINDINGS 9a.3).

Both are structurally impossible here:

* the model is built from the **checkpoint's own** ``spec``, so the config cannot
  disagree with the weights;
* the feature width is **asserted**, never padded;
* every failure raises.

Calibration and abstention
--------------------------
An instructor-facing system should be allowed to say nothing. This wraps the
model with a validation-fitted temperature and two thresholds:

``display_threshold``  below it the live chip reads ``uncertain`` rather than a cue
``alert_threshold``    below it no sustained-episode alert may fire

Both are chosen on **validation** and frozen. Raw predictions are always kept
alongside, so abstention never destroys evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from attention.taxonomy import CUE_CLASSES, taxonomy_classes
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import data as D
from attention.thesis_eval.models import build_model

UNCERTAIN = CUE_CLASSES.index("uncertain")

#: What the live chip reads when confidence is below ``display_threshold``.
#: A string, not a class id, because not every taxonomy HAS an abstention
#: class: ``onoff_reliable`` predicts on_task/off_task only. Using the word
#: keeps the UI's grey chip and its "never alert on this" rule working for
#: every taxonomy, and keeps an abstention distinguishable from a prediction
#: (the ``abstained`` flag says which).
ABSTAIN_LABEL = "uncertain"

#: Width of the vector the live extractor produces, by whether the head stream
#: is switched on. ``StudentFeatureExtractor`` emits
#: [base(552) | yaw, pitch, roll, face_found] = 556, and a further 518 columns
#: (512 CLIP over the head crop + 6 head-box geometry) when
#: ``head_stream=True`` (features.py:116).
LIVE_WIDTH_BASE = 556
LIVE_WIDTH_HEAD = 1074

#: Kept for callers that imported it. It is the width of the DEFAULT live
#: vector, not a ceiling on deployability — see :func:`live_input_width`.
LIVE_MAX_COL = LIVE_WIDTH_BASE

#: Blocks no streaming path can produce, and why. These, not a column count,
#: are what makes a feature config undeployable: ``1074_hp_head`` reads column
#: 1073 and is perfectly live-servable, while ``563_expr`` reads column 562 and
#: is not.
UNDEPLOYABLE_BLOCKS = {
    "express": "the 7 expression dims need a second per-crop FER model",
    "dynamic": "the 7 dynamic dims are whole-track statistics (fidget variance, "
               "a personalised gaze baseline), not per-frame quantities",
}


def live_input_width(feature_config: str) -> int:
    """How many columns the live extractor must produce for this config.

    Raises for a config that reads a block a streaming path cannot produce, so
    an undeployable checkpoint fails here rather than at the width assert with
    a message about padding.
    """
    blocks = D.FEATURE_CONFIGS[feature_config]
    bad = [b for b in blocks if b in UNDEPLOYABLE_BLOCKS]
    if bad:
        raise SystemExit(
            f"feature config {feature_config!r} reads {bad}, which a streaming "
            f"path cannot produce: "
            + "; ".join(UNDEPLOYABLE_BLOCKS[b] for b in bad)
            + ". This checkpoint is not deployable in a streaming path.")
    return LIVE_WIDTH_HEAD if "head" in blocks else LIVE_WIDTH_BASE


def bundle_classes(spec: dict) -> "List[str]":
    """The class names this checkpoint predicts, from its own recorded spec.

    A checkpoint trained on ``onoff_reliable`` has a 2-unit output layer. Building
    a 6-class head for it and loading with ``strict=True`` raises a shape error;
    building one without strict would run it on a randomly initialised layer.
    Reading the taxonomy is the only way to get this right, and it is recorded
    in every run the unified trainer produced.
    """
    return taxonomy_classes(str(spec.get("taxonomy", "cue6") or "cue6"))


@dataclass
class RuntimeBundle:
    """Everything the live path needs, derived from the checkpoint itself."""
    model: torch.nn.Module
    experiment_id: str
    model_name: str
    feature_config: str
    input_dim: int
    seed: Optional[int]
    checkpoint: str
    device: torch.device
    temperature: float = 1.0
    display_threshold: float = 0.0
    alert_threshold: float = 0.0
    calibration_evidence: str = ""
    #: columns to take from the live vector, or None when the config already
    #: equals the full live vector
    live_columns: Optional[np.ndarray] = None
    live_input_width: int = LIVE_MAX_COL
    #: What this checkpoint predicts. Read from the checkpoint's own
    #: ``spec.taxonomy``, never assumed to be the 6 cue classes.
    taxonomy: str = "cue6"
    class_names: List[str] = field(default_factory=lambda: list(CUE_CLASSES))
    #: True when the live extractor must be built with ``head_stream=True``.
    needs_head_stream: bool = False

    def describe(self) -> str:
        sel = "" if self.live_columns is None else \
            f", selecting {self.input_dim} of {self.live_input_width} live columns"
        tax = "" if self.taxonomy == "cue6" else \
            f", taxonomy {self.taxonomy} ({len(self.class_names)} classes)"
        return (f"{self.experiment_id} ({self.model_name}, {self.input_dim}-dim{sel}{tax}, "
                f"T={self.temperature:.3f}, display>={self.display_threshold:.2f}, "
                f"alert>={self.alert_threshold:.2f})")


def load_runtime_model(ckpt_path: str, device: str = "cuda:0",
                       calibration: Optional[str] = None) -> RuntimeBundle:
    """Build the model from the checkpoint's recorded spec. Never guesses."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu")
    spec = ck.get("spec")
    if spec is None:
        raise SystemExit(
            f"{ckpt_path} carries no 'spec'. It predates the unified trainer and "
            "its feature layout cannot be recovered from the file. Retrain with "
            "attention.thesis_eval.train, or load it with the legacy runtime and "
            "label every number it produces as legacy.")
    fc = spec["feature_config"]
    dim = D.config_dim(fc)
    cols = D.column_index(fc)
    width = live_input_width(fc)            # raises for express/dynamic configs
    classes = bundle_classes(spec)
    kw = dict(spec.get("model_kwargs") or {})
    if spec["model"] == "transformer":
        kw.setdefault("dropout", spec.get("dropout", 0.1))
    model = build_model(spec["model"], dim, len(classes), **kw).to(dev)
    # strict=True on purpose: a mismatch must stop the process, not print a line
    # and continue on random weights.
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    b = RuntimeBundle(
        model=model, experiment_id=spec.get("experiment_id", "unknown"),
        model_name=spec["model"], feature_config=fc,
        input_dim=dim, seed=spec.get("seed"), checkpoint=str(ckpt_path), device=dev,
        live_columns=None if dim == width else cols,
        live_input_width=width,
        taxonomy=str(spec.get("taxonomy", "cue6") or "cue6"),
        class_names=classes,
        needs_head_stream="head" in D.FEATURE_CONFIGS[fc])

    if calibration:
        c = json.loads(Path(calibration).read_text())
        b.temperature = float(c["temperature"])
        b.display_threshold = float(c["display_threshold"])
        b.alert_threshold = float(c["alert_threshold"])
        b.calibration_evidence = str(calibration)
    return b


@torch.no_grad()
def predict_window(bundle: RuntimeBundle, window: np.ndarray) -> Dict:
    """Predict the cue for the newest frame of a [T, D] feature window.

    Returns raw and abstained decisions side by side. The feature width is
    checked rather than padded: a short vector means the live extractor is not
    producing a block the model was trained on, which is a configuration error,
    not something to paper over with zeros.
    """
    if window.ndim != 2 or window.shape[1] != bundle.live_input_width:
        raise RuntimeError(
            f"live features are {window.shape[-1]}-dim but the extractor must "
            f"produce {bundle.live_input_width} for {bundle.experiment_id} "
            f"({bundle.feature_config}"
            + (", head_stream=True" if bundle.needs_head_stream else "")
            + "). Fix the feature extractor — do NOT pad, a zero block is "
            "indistinguishable from a real measurement.")
    if bundle.live_columns is not None:
        # SELECT the columns the checkpoint was trained on. A detector-only
        # backend still emits 4 head-pose columns, three of them zero; a
        # 553_facefound model must see the flag alone, not the zeros.
        window = window[:, bundle.live_columns]
    x = torch.from_numpy(np.ascontiguousarray(window)[None]).float().to(bundle.device)
    out = bundle.model(x)
    logits = out["logits"][0, -1].float()
    if bundle.temperature != 1.0:
        logits = logits / bundle.temperature
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    raw = int(probs.argmax())
    conf = float(probs[raw])
    names = bundle.class_names
    return {
        "cue": names[raw],
        "cue_id": raw,
        "confidence": conf,
        # ABSTAIN_LABEL rather than a class id: onoff_reliable has no
        # abstention class of its own, and inventing one would put a label the
        # model cannot predict into the same field as its predictions.
        "displayed_cue": names[raw] if conf >= bundle.display_threshold
                         else ABSTAIN_LABEL,
        "abstained": conf < bundle.display_threshold,
        "alert_allowed": conf >= bundle.alert_threshold,
        "probs": probs.tolist(),
        "taxonomy": bundle.taxonomy,
    }


# --------------------------------------------------------------------------
# threshold selection — validation only
# --------------------------------------------------------------------------

def select_thresholds(val_predictions: str, out: str,
                      min_display_coverage: float = 0.90,
                      min_alert_accuracy: float = 0.85) -> Dict:
    """Fit the temperature and pick both thresholds on **validation**.

    Two thresholds, because the dashboard does two different things:

    * the **live view** shows a cue chip for every tracked student, so it wants
      high coverage — the rule is the highest threshold that still labels
      ``min_display_coverage`` of frames;
    * an **alert** interrupts an instructor, so it wants precision — the rule is
      the lowest threshold whose retained frames are correct at least
      ``min_alert_accuracy`` of the time.

    Both are read off the validation curve and then frozen. Neither is tuned on
    the test split or on the human-gold set.
    """
    from attention.thesis_eval import calibrate as C

    v = np.load(val_predictions, allow_pickle=False)
    p, y = v["probs"].astype(np.float64), v["y"]
    T = C.fit_temperature(p, y)
    pc = C.apply_temperature(p, T)
    rows = C.coverage_risk_curve(pc, y)

    disp = max((r for r in rows if r["coverage"] >= min_display_coverage),
               key=lambda r: r["threshold"])
    alert_candidates = [r for r in rows
                        if np.isfinite(r["selective_accuracy"])
                        and r["selective_accuracy"] >= min_alert_accuracy]
    if not alert_candidates:
        raise SystemExit(
            f"no threshold reaches selective accuracy {min_alert_accuracy}; "
            "lower the target or improve the model rather than shipping alerts "
            "the instructor cannot trust")
    alert = min(alert_candidates, key=lambda r: r["threshold"])

    res = {
        "evaluator_version": EVALUATOR_VERSION,
        "fitted_on": val_predictions,
        "temperature": T,
        "display_threshold": disp["threshold"],
        "display_rule": f"highest threshold retaining >= {min_display_coverage:.0%} coverage",
        "display_coverage": disp["coverage"],
        "display_selective_accuracy": disp["selective_accuracy"],
        "alert_threshold": alert["threshold"],
        "alert_rule": f"lowest threshold with selective accuracy >= {min_alert_accuracy:.0%}",
        "alert_coverage": alert["coverage"],
        "alert_selective_accuracy": alert["selective_accuracy"],
        "uncalibrated_accuracy_at_full_coverage": rows[0]["selective_accuracy"],
        "note": ("Chosen on the validation split only and frozen. Temperature "
                 "scaling cannot change any argmax, so accuracy at full coverage "
                 "is unaffected; only confidence and the abstention behaviour move."),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description=select_thresholds.__doc__)
    ap.add_argument("--val-predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-display-coverage", type=float, default=0.90)
    ap.add_argument("--min-alert-accuracy", type=float, default=0.85)
    args = ap.parse_args()
    r = select_thresholds(args.val_predictions, args.out,
                          args.min_display_coverage, args.min_alert_accuracy)
    print(f"temperature        {r['temperature']:.4f}")
    print(f"display threshold  {r['display_threshold']:.2f}  "
          f"(coverage {r['display_coverage']:.3f}, "
          f"selective accuracy {r['display_selective_accuracy']:.3f})")
    print(f"alert threshold    {r['alert_threshold']:.2f}  "
          f"(coverage {r['alert_coverage']:.3f}, "
          f"selective accuracy {r['alert_selective_accuracy']:.3f})")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# striding
# --------------------------------------------------------------------------

class StrideController:
    """Decides which frames pay for the detector and the temporal head.

    Both are re-run every frame by default, which is wasteful for this task:

    * **Detector (45% of the frame budget).** Students are seated and
      stationary — across 115 LLMSTU tracks the centre of a student's box
      jitters by 4.4% of its diagonal (median), 13% at p90. Re-detecting them
      3.7 times a second buys almost nothing; between detections the tracker
      coasts on the last boxes.
    * **Temporal head (18%).** The taxonomy's episodes are sustained by
      construction — ``min_duration_s`` is 3 s and the shortest per-channel
      minimum is 2 s — so a cue re-decided at 2 Hz instead of 4 Hz cannot
      change which episodes form.

    Neither stride is assumed safe: ``tools/verify_stride_equivalence.py``
    replays the same video at stride 1 and at the configured strides and
    reports per-frame cue agreement.

    Both default to 1, i.e. off, so nothing changes unless a config asks for it.
    """

    def __init__(self, detector_stride: int = 1, temporal_stride: int = 1):
        if detector_stride < 1 or temporal_stride < 1:
            raise ValueError("strides must be >= 1")
        self.detector_stride = int(detector_stride)
        self.temporal_stride = int(temporal_stride)
        self._cache: Dict[int, Dict] = {}
        self._last_predicted: Dict[int, int] = {}

    def should_detect(self, frame_idx: int) -> bool:
        return frame_idx % self.detector_stride == 0

    def adjusted_min_hits(self, min_hits: int) -> int:
        """``min_hits`` rescaled so confirmation takes the same wall-clock time.

        The tracker counts hits in *detected* frames, so at a detector stride of
        5 a ``min_hits`` of 8 needs 40 real frames — 1.6 s at 25 fps — before a
        student appears at all. Measured: leaving it unscaled dropped track
        coverage against the stride-1 reference to 82.1% at stride 5 and 65.4%
        at stride 10, almost entirely as start-up latency rather than lost
        tracks. Dividing keeps the confirmation delay constant in real frames.
        """
        return max(1, round(min_hits / self.detector_stride))

    def should_predict(self, track_id: int, frame_idx: int) -> bool:
        """Predict on the first frame a track is seen, then every K frames.

        Keyed per track rather than globally so a newly confirmed track is not
        left with no cue until the next global tick.
        """
        if track_id not in self._cache:
            return True
        return frame_idx - self._last_predicted[track_id] >= self.temporal_stride

    def store(self, track_id: int, frame_idx: int, result: Dict) -> Dict:
        self._cache[track_id] = result
        self._last_predicted[track_id] = frame_idx
        return result

    def cached(self, track_id: int) -> Optional[Dict]:
        return self._cache.get(track_id)

    def drop_missing(self, live_track_ids) -> None:
        """Forget tracks the tracker has dropped, so ids cannot be reused stale."""
        live = set(live_track_ids)
        for tid in [t for t in self._cache if t not in live]:
            self._cache.pop(tid, None)
            self._last_predicted.pop(tid, None)
