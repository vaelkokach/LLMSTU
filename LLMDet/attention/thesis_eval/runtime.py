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

from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import data as D
from attention.thesis_eval.models import build_model

UNCERTAIN = CUE_CLASSES.index("uncertain")


#: Feature configs the live extractor can serve. The live vector is laid out
#: [base(552) | yaw, pitch, roll, face_found], i.e. the first 556 columns of the
#: offline layout, so any config drawing only on those blocks is deployable.
#: 563_* and 570_full are NOT: the expression block needs a second GPU model per
#: crop and the dynamic block is a whole-track statistic that a streaming path
#: cannot produce faithfully.
LIVE_MAX_COL = 556


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
    #: columns to take from the live 556-wide vector, or None when the config
    #: already equals the full live vector
    live_columns: Optional[np.ndarray] = None
    live_input_width: int = LIVE_MAX_COL

    def describe(self) -> str:
        sel = "" if self.live_columns is None else \
            f", selecting {self.input_dim} of {self.live_input_width} live columns"
        return (f"{self.experiment_id} ({self.model_name}, {self.input_dim}-dim{sel}, "
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
    dim = D.config_dim(spec["feature_config"])
    cols = D.column_index(spec["feature_config"])
    if cols.max() >= LIVE_MAX_COL:
        raise SystemExit(
            f"{ckpt_path} uses feature config {spec['feature_config']!r}, which "
            f"needs column {int(cols.max())}. The live extractor produces only "
            f"{LIVE_MAX_COL} columns: the expression block needs a per-crop FER "
            f"model and the dynamic block is a whole-track statistic. This "
            f"checkpoint is not deployable in a streaming path.")
    kw = dict(spec.get("model_kwargs") or {})
    if spec["model"] == "transformer":
        kw.setdefault("dropout", spec.get("dropout", 0.1))
    model = build_model(spec["model"], dim, len(CUE_CLASSES), **kw).to(dev)
    # strict=True on purpose: a mismatch must stop the process, not print a line
    # and continue on random weights.
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    b = RuntimeBundle(
        model=model, experiment_id=spec.get("experiment_id", "unknown"),
        model_name=spec["model"], feature_config=spec["feature_config"],
        input_dim=dim, seed=spec.get("seed"), checkpoint=str(ckpt_path), device=dev,
        live_columns=None if dim == LIVE_MAX_COL else cols,
        live_input_width=LIVE_MAX_COL)

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
            f"produce {bundle.live_input_width} (base + head-pose block) for "
            f"{bundle.experiment_id}. Fix the feature extractor — do NOT pad, a "
            f"zero block is indistinguishable from a real measurement.")
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
    return {
        "cue": CUE_CLASSES[raw],
        "cue_id": raw,
        "confidence": conf,
        "displayed_cue": CUE_CLASSES[raw] if conf >= bundle.display_threshold
                         else CUE_CLASSES[UNCERTAIN],
        "abstained": conf < bundle.display_threshold,
        "alert_allowed": conf >= bundle.alert_threshold,
        "probs": probs.tolist(),
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
