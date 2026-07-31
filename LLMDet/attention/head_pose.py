"""Head-pose backends for the attention pipeline.

WHY THIS MATTERS HERE: the two weakest cue classes are `turned_to_peer`
(F1 0.18) and `looking_away` (F1 0.29). Both are *orientation* judgements,
which CLIP embeddings + bbox geometry + colour statistics cannot represent.
The dense gold audit (FINDINGS.md 5.4) also showed gaze carries the phone-use
signal when the phone itself is hidden behind a monitor. Head pose is the
single highest-value missing feature.

Backends, in order of preference:

``opencv``    Zero new dependencies. Haar frontal + left/right profile
              cascades give a coarse but genuinely informative YAW signal —
              which is precisely the axis that separates screen-oriented from
              turned-to-peer. Pitch is proxied from the face's vertical
              position inside the head crop, roll is not observable this way
              and is returned as 0.
``mediapipe`` Preferred when installed: FaceMesh landmarks + solvePnP give
              metric yaw/pitch/roll. NOT installed in this environment; a
              dry-run install was abandoned rather than risk perturbing a
              live training environment. Install deliberately, then pass
              backend="mediapipe".
``6drepnet``  Direct yaw/pitch/roll regression. Needs external weights.

All backends return ``np.ndarray([yaw, pitch, roll], float32)`` with angles
normalised to roughly [-1, 1] (i.e. degrees/90) so the block is on the same
scale as the other feature groups and needs no separate normalisation.

Contract: ``available()`` must be False unless the backend can actually run,
because StudentFeatureExtractor.output_dim() branches on it — a backend that
claims availability then fails would change the feature width mid-run.
"""
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

OUTPUT_DIM = 3


class OpenCVHeadPose:
    """Coarse yaw/pitch from Haar frontal + profile cascades.

    Yaw is derived from which cascade fires and where the face sits
    horizontally within the head crop:

      * frontal face found        -> yaw from horizontal face-centre offset
      * left-profile only         -> strong negative yaw
      * right-profile only        -> strong positive yaw (mirrored detection)
      * nothing found             -> yaw 0, confidence 0 (see note below)

    NOTE ON FAILURE: when no face is found the vector is zeros. That is
    deliberately indistinguishable from "facing straight ahead", which is a
    real limitation — a fully turned-away student and an undetected face both
    yield 0. The temporal model can partially disambiguate this from context,
    but a metric backend (mediapipe/6DRepNet) is the proper fix.
    """

    def __init__(self):
        if cv2 is None:
            raise RuntimeError("opencv is required for the opencv head-pose backend")
        base = cv2.data.haarcascades
        self._frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self._profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self._frontal.empty() or self._profile.empty():
            raise RuntimeError("Haar cascade XML failed to load")

    def _detect(self, gray: np.ndarray):
        f = self._frontal.detectMultiScale(gray, 1.15, 4, minSize=(24, 24))
        if len(f):
            return "frontal", max(f, key=lambda r: r[2] * r[3])
        p = self._profile.detectMultiScale(gray, 1.15, 4, minSize=(24, 24))
        if len(p):
            return "right", max(p, key=lambda r: r[2] * r[3])
        # The profile cascade is trained on one side only; mirror to find the other.
        pm = self._profile.detectMultiScale(cv2.flip(gray, 1), 1.15, 4, minSize=(24, 24))
        if len(pm):
            x, y, w, h = max(pm, key=lambda r: r[2] * r[3])
            return "left", (gray.shape[1] - x - w, y, w, h)
        return None, None

    def estimate(self, head_crop_bgr: np.ndarray) -> np.ndarray:
        if head_crop_bgr is None or head_crop_bgr.size == 0:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        h, w = head_crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        gray = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kind, box = self._detect(gray)
        if kind is None:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)

        x, y, fw, fh = box
        cx = (x + fw / 2.0) / max(1.0, w)   # 0..1 across the crop
        cy = (y + fh / 2.0) / max(1.0, h)
        if kind == "frontal":
            yaw = float(np.clip((cx - 0.5) * 2.0, -1.0, 1.0)) * 0.45
        elif kind == "left":
            yaw = -0.75 + float(np.clip((cx - 0.5) * 0.5, -0.25, 0.25))
        else:
            yaw = 0.75 + float(np.clip((cx - 0.5) * 0.5, -0.25, 0.25))
        # Face high in the crop => head up; low => head tilted down.
        pitch = float(np.clip((cy - 0.45) * 2.0, -1.0, 1.0))
        return np.array([yaw, pitch, 0.0], dtype=np.float32)


class MediaPipeHeadPose:  # pragma: no cover - dependency not installed here
    """FaceMesh landmarks + solvePnP. Metric angles; preferred when available."""

    # Canonical 3D points (mm) for nose, chin, eye corners, mouth corners.
    _MODEL = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ], dtype=np.float64)
    _IDX = [1, 199, 33, 263, 61, 291]  # FaceMesh indices for the above

    def __init__(self):
        import mediapipe as mp  # raises ImportError if absent
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.3)

    def estimate(self, head_crop_bgr: np.ndarray) -> np.ndarray:
        if head_crop_bgr is None or head_crop_bgr.size == 0:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        h, w = head_crop_bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in self._IDX], dtype=np.float64)
        f = float(w)
        cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(self._MODEL, pts, cam, np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        rmat, _ = cv2.Rodrigues(rvec)
        sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        return (np.array([yaw, pitch, roll], dtype=np.float32) / 90.0).clip(-1, 1)


_BACKENDS = {"opencv": OpenCVHeadPose, "mediapipe": MediaPipeHeadPose}


class HeadPoseEstimator:
    """Drop-in replacement for the stub in features.py.

    ``backend=None`` keeps the previous behaviour (unavailable, feature block
    omitted) so existing 552-dim checkpoints stay loadable. Pass
    ``backend="opencv"`` to enable the 3 extra dims -> 555-dim features, which
    requires rebuilding sequences and retraining the temporal model.
    """

    OUTPUT_DIM = OUTPUT_DIM

    def __init__(self, backend: Optional[str] = None, device: str = "cuda:0"):
        self.backend = backend
        self.device = device
        self._model = None
        if backend is None:
            return
        if backend not in _BACKENDS:
            raise ValueError(f"unknown head-pose backend {backend!r}; "
                             f"choose from {sorted(_BACKENDS)}")
        try:
            self._model = _BACKENDS[backend]()
        except Exception as e:
            # Fail loudly: a silently-disabled backend would change the feature
            # width and train a model that cannot see orientation at all.
            raise RuntimeError(
                f"head-pose backend {backend!r} could not be initialised: {e}"
            ) from e

    def available(self) -> bool:
        return self._model is not None

    def estimate(self, head_crop_bgr: np.ndarray) -> np.ndarray:
        if not self.available():
            raise RuntimeError("No head-pose backend configured.")
        return self._model.estimate(head_crop_bgr)
