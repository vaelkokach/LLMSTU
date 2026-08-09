"""Full-range head-pose estimation (6DRepNet360) for Branch C.

Why this exists
---------------
``attention/head_pose.py`` documents a ``6drepnet`` backend at line 23 but never
implements one: ``_BACKENDS`` (line 263) contains only ``opencv``, ``mediapipe``
and ``mediapipe_detector``. Every angle the project has ever measured therefore
comes from MediaPipe, which is a *face* model - it returns nothing at all when it
cannot find a face, which on this corpus is 37% of frames, concentrated in
``head_down`` (15.1% coverage) and ``uncertain`` (3.9%). See FINDINGS 12.13a.

The registered hypothesis H1 (BRANCH_C_PROTOCOL.md 5, amendment A1) is that a
full-range estimator flattens that coverage contrast, removing the missingness
signal the existing model leans on and forcing the geometry to earn its keep. This
module is what makes that testable.

6DRepNet360 is chosen over DirectMHP on licence grounds, not accuracy: DirectMHP
is GPL-3.0 via YOLOv5 and would impose copyleft on the combined work, while
6DRepNet360 is MIT. See FINDINGS 12.9 / docs/branch_c/LICENSE_AUDIT.md.

Architecture, read off the checkpoint rather than assumed
--------------------------------------------------------
``6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth`` (sha256 3ee08f1e...) contains
320 tensors: ``conv1``, ``bn1``, ``layer1``-``layer4`` with bottleneck blocks in
a [3, 4, 6, 3] arrangement, and ``linear_reg`` of shape ``(6, 2048)``. That is a
torchvision ResNet-50 with the classifier replaced by a 6D rotation head. The
layer tensor counts (60/78/114/60) were checked against the bottleneck arithmetic
before this class was written; nothing here is inferred from the model's name.

The 6D-to-matrix convention is upstream's ``compute_rotation_matrix_from_ortho6d``
(x = normalize(x_raw); z = normalize(x cross y_raw); y = z cross x; columns
[x|y|z]). That is algebraically identical to the Gram-Schmidt form in
:func:`attention.branch_c.canonical.rot6d_to_matrix`, and
``test_branch_c_head_pose_fullrange.py`` asserts the two agree numerically rather
than trusting the derivation.

Head localisation
-----------------
The corpus stores *person* crops, and a person crop is not a head crop
(specification guardrail 3). ``head_span_px`` cannot substitute: it is floored at
exactly 120 px and runs at a median 0.43x the person-box height, so it is a
loose upper bound rather than a head size.

This module therefore uses an explicit, documented geometric crop - the top
square region of the person box - whose single parameter is calibrated once
against the frames where MediaPipe *does* find a face, and then applied
uniformly, including where MediaPipe fails. The calibration and the agreement
check both live in ``tools/branch_c/calibrate_head_crop.py`` so the number in
:data:`DEFAULT_HEAD_FRACTION` is reproducible rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from attention.branch_c.canonical import matrix_to_euler, rot6d_to_matrix

REPO = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO / "huggingface/sixdrepnet360/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
WEIGHTS_SHA256 = "3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6"
WEIGHTS_URL = (
    "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/"
    "6DRepNet360_Full-Rotation_300W_LP%2BPanoptic.pth"
)
WEIGHTS_LICENSE = "MIT (github.com/thohemp/6DRepNet360)"

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Head box height as a fraction of the person-box height, anchored at the top.
#: Calibrated in tools/branch_c/calibrate_head_crop.py; see module docstring.
DEFAULT_HEAD_FRACTION = 0.42


def head_box_from_person(
    box: Sequence[float],
    fraction: float = DEFAULT_HEAD_FRACTION,
    aspect: float = 1.0,
) -> Tuple[int, int, int, int]:
    """Geometric head box: the top ``fraction`` of a person box, squared up.

    Args:
        box: ``(x0, y0, x1, y1)`` person box in whatever coordinate frame the
            caller is working in.
        fraction: head height as a share of person-box height.
        aspect: width/height of the head box.

    Returns integer ``(x0, y0, x1, y1)``. Deliberately *not* clipped to an image
    here - the caller knows the image bounds and :func:`crop_head` does the
    clipping, so this stays a pure geometric function that can be unit-tested.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    h = max(y1 - y0, 1.0)
    hh = h * fraction
    hw = hh * aspect
    cx = (x0 + x1) * 0.5
    return (
        int(round(cx - hw * 0.5)),
        int(round(y0)),
        int(round(cx + hw * 0.5)),
        int(round(y0 + hh)),
    )


def crop_head(image: np.ndarray, box: Optional[Sequence[float]] = None,
              fraction: float = DEFAULT_HEAD_FRACTION) -> Optional[np.ndarray]:
    """Extract the head region from a person crop (or a full frame plus a box).

    ``box`` in image coordinates; if ``None`` the whole image is treated as the
    person box, which is the case for the stored LLMSTU crops. Returns ``None``
    when the region is degenerate, rather than an empty array that would fail
    somewhere less obvious.
    """
    if image is None or image.size == 0:
        return None
    H, W = image.shape[:2]
    if box is None:
        box = (0, 0, W, H)
    x0, y0, x1, y1 = head_box_from_person(box, fraction)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return image[y0:y1, x0:x1]


class SixDRepNet360(nn.Module):
    """ResNet-50 backbone with a 6D rotation regression head.

    ``forward`` returns rotation matrices ``(B, 3, 3)`` mapping head-local
    coordinates into camera coordinates - the same direction as ``R_head`` in
    :mod:`attention.branch_c.canonical`.

    Unlike the MediaPipe backends this is a pure regressor: it emits a rotation
    for **every** input, with no detection step and therefore no ``face_found``
    analogue. That is the entire point of the H1 experiment, and it also means a
    caller must not read "an angle exists" as "a head was found here".
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50

        net = resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = (
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear_reg = nn.Linear(2048, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return rot6d_to_matrix(self.linear_reg(x))


@dataclass
class FullRangeHeadPose:
    """Batched full-range head-pose estimator over pre-cropped BGR images."""

    device: str = "cuda:0"
    weights: Path = DEFAULT_WEIGHTS
    head_fraction: float = DEFAULT_HEAD_FRACTION
    model: Optional[SixDRepNet360] = None

    def load(self) -> "FullRangeHeadPose":
        """Load weights strictly. A key mismatch is an error, never a warning.

        FINDINGS records a past incident where a config/checkpoint width mismatch
        silently produced a randomly-initialised model that then generated
        results. ``strict=True`` is not optional here.
        """
        if not Path(self.weights).is_file():
            raise FileNotFoundError(
                f"6DRepNet360 weights not found at {self.weights}. "
                f"Download from {WEIGHTS_URL} (licence: {WEIGHTS_LICENSE}) and "
                f"verify sha256 {WEIGHTS_SHA256}."
            )
        sd = torch.load(self.weights, map_location="cpu", weights_only=False)
        for key in ("model_state_dict", "state_dict"):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model = SixDRepNet360()
        model.load_state_dict(sd, strict=True)
        model.eval().to(self.device)
        self.model = model
        return self

    def preprocess(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        """BGR uint8 head crops -> normalised NCHW float tensor."""
        import cv2

        out = np.empty((len(crops), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        for i, c in enumerate(crops):
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            out[i] = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        out /= 255.0
        out -= np.asarray(IMAGENET_MEAN, dtype=np.float32)
        out /= np.asarray(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def estimate_batch(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        """Head crops -> rotation matrices ``(B, 3, 3)`` on CPU."""
        if self.model is None:
            raise RuntimeError("call .load() first")
        if not crops:
            return torch.empty(0, 3, 3)
        x = self.preprocess(crops).to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=str(self.device).startswith("cuda")):
            R = self.model(x)
        return R.float().cpu()

    @staticmethod
    def to_euler_degrees(R: torch.Tensor) -> torch.Tensor:
        """``(B, 3, 3)`` -> ``(B, 3)`` yaw/pitch/roll in **degrees**, unclipped.

        Provided for comparison against the legacy block only. Downstream Branch-C
        features use the rotation matrix directly (6D / SO(3) log), because Euler
        angles reintroduce exactly the wraparound and gimbal problems
        :mod:`attention.branch_c.canonical` exists to avoid.
        """
        yaw, pitch, roll = matrix_to_euler(R)
        return torch.stack([yaw, pitch, roll], dim=-1) * (180.0 / np.pi)
