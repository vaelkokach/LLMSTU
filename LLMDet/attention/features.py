from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


# Real implementation lives in attention/head_pose.py. Re-exported here so
# existing imports (`from attention.features import HeadPoseEstimator`) keep
# working and there is exactly ONE implementation rather than a stub plus a
# real class that can drift apart.
#
# backend=None  -> unavailable, feature block omitted, 552-dim (unchanged;
#                  existing checkpoints stay loadable)
# backend="opencv"    -> +3 dims = 555-dim; requires rebuilding sequences and
#                  retraining the temporal model
# backend="mediapipe" -> same 555 dims, metric angles, needs mediapipe installed
from .head_pose import HeadPoseEstimator  # noqa: F401  (re-export)


class StudentFeatureExtractor:
    """Per-student feature vector: CLIP embedding + bbox geometry + color
    statistics + posture-geometry proxies.

    CLIP failures are fatal by default: silently zeroing 512 of the feature
    dims previously let a broken environment train a garbage model. Pass
    ``allow_clip_fallback=True`` to opt into zeros explicitly.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda:0",
        allow_clip_fallback: bool = False,
        head_pose: Optional[HeadPoseEstimator] = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_dim = 512
        self.clip_model: Optional["CLIPModel"] = None
        self.clip_processor: Optional["CLIPProcessor"] = None
        self.head_pose = head_pose
        self.clip_enabled = False

        if CLIPModel is None or CLIPProcessor is None:
            if not allow_clip_fallback:
                raise RuntimeError(
                    "transformers CLIP is not importable. Install transformers, or pass "
                    "allow_clip_fallback=True to train without CLIP features (zeros)."
                )
        else:
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
                self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
                self.clip_model.eval()
                self.clip_enabled = True
            except Exception as e:
                if not allow_clip_fallback:
                    raise RuntimeError(
                        f"CLIP weights failed to load ({e}). Fix the environment, or pass "
                        "allow_clip_fallback=True to continue with zeroed CLIP features."
                    ) from e
                print(f"[warn] CLIP disabled by explicit fallback; using zeros: {e}")

    def output_dim(self) -> int:
        # 512 clip + 8 bbox geom + 24 color stats + 8 posture geom (+3 head pose if configured)
        d = self.clip_dim + 8 + 24 + 8
        if self.head_pose is not None and self.head_pose.available():
            d += HeadPoseEstimator.OUTPUT_DIM
        return d

    def extract(self, frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
        return self.extract_batch(frame_bgr, [bbox_xyxy])[0]

    def extract_batch(self, frame_bgr: np.ndarray, bboxes_xyxy: List[List[float]]) -> np.ndarray:
        """Features for every box of one frame with a single CLIP forward pass.

        Per-crop CLIP calls cost ~20 ms/student and dominated the real-time
        budget at high student counts; batching flattens that to one call per
        frame. Returns [N, output_dim()]; degenerate boxes yield zero rows.
        """
        n = len(bboxes_xyxy)
        out = np.zeros((n, self.output_dim()), dtype=np.float32)
        if n == 0:
            return out

        h, w = frame_bgr.shape[:2]
        crops, clipped, valid_idx = [], [], []
        for i, bbox in enumerate(bboxes_xyxy):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(frame_bgr[y1:y2, x1:x2])
            clipped.append((x1, y1, x2, y2))
            valid_idx.append(i)

        if not crops:
            return out

        clip_feats = self._clip_batch(crops)
        for row, (crop, (x1, y1, x2, y2), i) in enumerate(zip(crops, clipped, valid_idx)):
            parts = [
                clip_feats[row],
                self._geom(x1, y1, x2, y2, w, h),
                self._color_stats(crop),
                self._posture_geom(crop, x1, y1, x2, y2, w, h),
            ]
            if self.head_pose is not None and self.head_pose.available():
                head_h = max(1, int(0.35 * (y2 - y1)))
                parts.append(self.head_pose.estimate(crop[:head_h]).astype(np.float32))
            out[i] = np.concatenate(parts, axis=0).astype(np.float32)
        return out

    def _clip_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        if not self.clip_enabled or self.clip_model is None or self.clip_processor is None:
            return np.zeros((len(crops_bgr), self.clip_dim), dtype=np.float32)
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inp = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            f = self.clip_model.get_image_features(**inp)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        return f.detach().float().cpu().numpy()

    def _geom(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> np.ndarray:
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        cx = (x1 + x2) * 0.5 / max(1.0, float(w))
        cy = (y1 + y2) * 0.5 / max(1.0, float(h))
        area = (bw * bh) / max(1.0, float(w * h))
        ar = bw / bh
        left = x1 / max(1.0, float(w))
        right = x2 / max(1.0, float(w))
        top = y1 / max(1.0, float(h))
        bottom = y2 / max(1.0, float(h))
        return np.array([cx, cy, area, ar, left, right, top, bottom], dtype=np.float32)

    def _color_stats(self, crop_bgr: np.ndarray) -> np.ndarray:
        chans = cv2.split(crop_bgr)
        feats = []
        for ch in chans:
            chf = ch.astype(np.float32)
            # one partition-based call for all three percentiles: identical
            # values to separate np.percentile calls at ~1/3 the sort cost
            p25, p50, p75 = np.percentile(chf, (25, 50, 75))
            feats.extend(
                [
                    float(chf.mean()),
                    float(chf.std()),
                    float(p25),
                    float(p50),
                    float(p75),
                    float(chf.min()),
                    float(chf.max()),
                    float((chf > 200).mean()),
                ]
            )
        return np.array(feats, dtype=np.float32)

    def _posture_geom(self, crop_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> np.ndarray:
        """Cheap posture proxies from box shape and crop intensity layout.

        The head occupies the top of a seated person's box; its horizontal
        offset and the vertical mass distribution shift with leaning,
        head-down and turned postures. All values are in [0, 1]-ish ranges so
        they can sit next to the normalized geometry block.
        """
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        elongation = bh / bw  # tall = upright, squat = slumped/leaning
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # vertical mass distribution: center of intensity mass, top-third share
        col_profile = gray.mean(axis=1)
        total = float(col_profile.sum()) + 1e-6
        ys = np.arange(col_profile.shape[0], dtype=np.float32)
        v_center = float((col_profile * ys).sum() / total / max(1.0, gray.shape[0]))
        third = max(1, gray.shape[0] // 3)
        top_share = float(col_profile[:third].sum() / total)
        bottom_share = float(col_profile[-third:].sum() / total)

        # head-region estimate: brightest-motion proxy is unavailable per-frame,
        # so use the horizontal intensity centroid of the top 35% as a head
        # x-offset proxy (turned head/torso shifts it off center).
        head_h = max(1, int(0.35 * gray.shape[0]))
        head_strip = gray[:head_h]
        row_profile = head_strip.mean(axis=0)
        xs = np.arange(row_profile.shape[0], dtype=np.float32)
        head_cx = float((row_profile * xs).sum() / (float(row_profile.sum()) + 1e-6) / max(1.0, gray.shape[1]))

        # contrast between head strip and torso strip (head-down lowers it)
        torso = gray[head_h:] if gray.shape[0] > head_h else gray
        head_torso_contrast = float(
            (head_strip.mean() - torso.mean()) / (head_strip.mean() + torso.mean() + 1e-6)
        )
        # normalized top-edge position of box (standing/leaning forward changes it)
        top_rel = y1 / max(1.0, float(h))

        return np.array(
            [
                elongation,
                v_center,
                top_share,
                bottom_share,
                head_cx,
                head_torso_contrast,
                top_rel,
                min(1.0, bw / max(1.0, float(w))),
            ],
            dtype=np.float32,
        )
