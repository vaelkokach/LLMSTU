"""Seat/task-relative pose canonicalisation, and the rotation algebra under it.

The one mathematical claim this branch can guarantee, and the only one it will
make about invariance:

    If the camera coordinate system is changed by a global rotation Q, then both
    the head and the reference rotate with it,

        R_head' = Q R_head        R_ref' = Q R_ref

    so the relative rotation is unchanged:

        R_rel' = (Q R_ref)^T (Q R_head) = R_ref^T Q^T Q R_head = R_ref^T R_head

This is proved, unit-tested over random draws, and probed with synthetic
rotations. It is **not** validated across real viewpoints, because the corpus
contains exactly one camera pose (FINDINGS 12.1). Rotation invariance also does
not imply invariance to a homography; a test in the sibling test module asserts
that a projective perturbation *does* change R_rel, so that nobody later reads
more into this than it says.

Conventions, stated because this project has been bitten by leaving them implicit
--------------------------------------------------------------------------------
* **Angles are in radians internally.** The existing extractor
  (``attention/head_pose.py``) emits degrees scaled by 1/90 and clipped to
  [-1, 1]; :func:`from_head_pose_block` converts and documents the loss.
* **Euler order is intrinsic Z-Y-X**, i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
  This matches ``head_pose.py:178-180``, which recovers
  ``yaw = atan2(M[1,0], M[0,0])``, ``pitch = atan2(-M[2,0], sy)``,
  ``roll = atan2(M[2,1], M[2,2])`` from the MediaPipe facial transformation
  matrix. Do not assume any other library's default agrees.
* **Matrices rotate points, not frames.** ``R_head`` maps head-local coordinates
  into camera coordinates.
* ``R_rel = R_ref^T @ R_head`` is the head expressed *in the reference frame*.

A limitation worth stating loudly
---------------------------------
``head_pose.py:181-182`` divides by 90 and clips to [-1, 1], so any |yaw| beyond
90 degrees saturates. In this room students routinely face away from the camera,
which is exactly the regime beyond that clip. The existing 556-dim feature block
therefore **cannot represent a student facing away** - it saturates them all to the
same value. That is an independent reason, beyond the ``face_found`` shortcut, why
the existing angles carry little signal here, and it is why the full-range
candidate (6DRepNet360, MIT-licensed; DirectMHP is GPL-3.0 and rejected on licence
grounds, see FINDINGS 12.9) is worth measuring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# Below this, two rotations are treated as identical and the log map is taken to
# be zero rather than divided by a vanishing sine.
_EPS = 1e-7


# ---------------------------------------------------------------------------
# Rotation representations
# ---------------------------------------------------------------------------


def euler_to_matrix(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Intrinsic Z-Y-X Euler angles (radians) to rotation matrices.

    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, matching ``head_pose.py``'s
    decomposition. Inputs broadcast against each other; output is ``(..., 3, 3)``.
    """
    yaw, pitch, roll = torch.broadcast_tensors(yaw, pitch, roll)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # Written out rather than composed from three matmuls: it is the same
    # arithmetic, and having the closed form here makes the convention auditable.
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_euler(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`euler_to_matrix`. Returns ``(yaw, pitch, roll)`` in radians.

    Gimbal lock (``|pitch| -> pi/2``) is handled by falling back to a
    roll-into-yaw convention rather than producing NaN. Euler angles are only ever
    used here for interfacing with the legacy feature block; everything internal
    stays in matrices or 6D.
    """
    sp = -R[..., 2, 0]
    sp = torch.clamp(sp, -1.0, 1.0)
    pitch = torch.asin(sp)
    cp = torch.cos(pitch)
    locked = cp.abs() < 1e-6

    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    # At lock, yaw and roll are degenerate; pin roll to 0 and put the whole
    # rotation into yaw so the round trip stays continuous.
    yaw_locked = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
    yaw = torch.where(locked, yaw_locked, yaw)
    roll = torch.where(locked, torch.zeros_like(roll), roll)
    return yaw, pitch, roll


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation (Zhou et al. 2019): first two columns.

    The 6D representation exists because Euler angles and quaternions are
    discontinuous as regression targets. Output ``(..., 6)``, laid out as the
    first column followed by the second - **not** the row-major flattening of the
    ``(3, 2)`` slice, which interleaves them and silently breaks the round trip
    with :func:`rot6d_to_matrix`.
    """
    return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """6D representation back to SO(3) by Gram-Schmidt. Output ``(..., 3, 3)``.

    Always returns a proper rotation: orthonormal with det = +1.
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=_EPS)
    a2_proj = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1, eps=_EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle (rotation vector) of R. Output ``(..., 3)``, magnitude in radians.

    Numerically guarded at both ends. Near identity the naive
    ``theta / (2 sin theta)`` factor is 0/0, and near pi the axis is recovered
    from the diagonal because the skew part vanishes.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    skew = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )

    sin_theta = torch.sin(theta)
    small = theta < 1e-4
    near_pi = (math.pi - theta) < 1e-3

    # Generic branch. The clamp keeps the gradient finite where sin_theta -> 0;
    # those entries are overwritten by the branches below.
    denom = torch.where(sin_theta.abs() < _EPS, torch.full_like(sin_theta, _EPS), sin_theta)
    out = skew * (theta / (2.0 * denom)).unsqueeze(-1)

    # theta ~ 0: log(R) ~ skew/2, which is the limit of the expression above.
    out = torch.where(small.unsqueeze(-1), skew * 0.5, out)

    # theta ~ pi: sin(theta) ~ 0 and the skew part is uninformative. Recover the
    # axis from R + I = 2 a a^T, taking the numerically largest column.
    if bool(near_pi.any()):
        eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
        M = (R + eye) * 0.5
        diag = torch.diagonal(M, dim1=-2, dim2=-1)
        k = torch.argmax(diag, dim=-1)
        axis = torch.gather(M, -1, k[..., None, None].expand(*M.shape[:-1], 1)).squeeze(-1)
        axis = torch.nn.functional.normalize(axis, dim=-1, eps=_EPS)
        # Sign is ambiguous at exactly pi; fix it against the skew part where that
        # still carries signal, which keeps the map continuous approaching pi.
        sign = torch.sign((axis * skew).sum(-1, keepdim=True))
        sign = torch.where(sign.abs() < 0.5, torch.ones_like(sign), sign)
        out = torch.where(near_pi.unsqueeze(-1), axis * sign * theta.unsqueeze(-1), out)
    return out


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector to matrix (Rodrigues). Inverse of :func:`so3_log`."""
    theta = v.norm(dim=-1, keepdim=True)
    small = theta < 1e-6
    axis = v / torch.where(small, torch.ones_like(theta), theta)
    t = theta.squeeze(-1)
    ct, st = torch.cos(t), torch.sin(t)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)
    R = eye + st[..., None, None] * K + (1 - ct)[..., None, None] * (K @ K)
    return torch.where(small[..., None], eye, R)


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle in radians of the rotation taking R1 to R2. In [0, pi].

    The correct way to difference two orientations. Subtracting Euler angles is
    not: see :func:`relative_angular_velocity`.
    """
    rel = R1.transpose(-1, -2) @ R2
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))


def relative_rotation(R_ref: torch.Tensor, R_head: torch.Tensor) -> torch.Tensor:
    """``R_ref^T @ R_head`` — the head expressed in the reference frame.

    Invariant to any global rotation Q applied to both arguments. That is the
    branch's whole mathematical claim, and it is tested directly.
    """
    return R_ref.transpose(-1, -2) @ R_head


# ---------------------------------------------------------------------------
# Causal temporal derivatives
# ---------------------------------------------------------------------------


def relative_angular_velocity(R: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Causal angular velocity, in the tangent space, one step per frame.

    ``omega[t] = log(R[t-1]^T R[t])`` — a **backward** difference, so the value at
    t uses no frame after t (BRANCH_C_PROTOCOL.md section 6). ``omega[0]`` is zero.

    This exists because subtracting Euler angles is wrong across wraparound: going
    from +179 to -179 degrees is a 2-degree turn, not a 358-degree one. The tangent
    space has no such seam.

    Args:
        R: ``(..., T, 3, 3)``
        valid: optional ``(..., T)`` boolean. A step touching an invalid frame
            yields zero velocity rather than a spurious jump.
    """
    if R.shape[-3] < 2:
        return torch.zeros(*R.shape[:-2], 3, dtype=R.dtype, device=R.device)
    prev, curr = R[..., :-1, :, :], R[..., 1:, :, :]
    step = so3_log(prev.transpose(-1, -2) @ curr)
    zero = torch.zeros(*step.shape[:-2], 1, 3, dtype=R.dtype, device=R.device)
    omega = torch.cat([zero, step], dim=-2)
    if valid is not None:
        ok = valid[..., 1:] & valid[..., :-1]
        ok = torch.cat([torch.zeros_like(valid[..., :1]), ok], dim=-1)
        omega = omega * ok.unsqueeze(-1).to(omega.dtype)
    return omega


def relative_angular_acceleration(
    R: torch.Tensor, valid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Causal second difference of orientation. ``alpha[t] = omega[t] - omega[t-1]``."""
    omega = relative_angular_velocity(R, valid)
    alpha = torch.zeros_like(omega)
    alpha[..., 1:, :] = omega[..., 1:, :] - omega[..., :-1, :]
    return alpha


# ---------------------------------------------------------------------------
# SO(2) image-plane fallback
# ---------------------------------------------------------------------------


def relative_heading_so2(theta_head: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
    """Image-plane heading of the head relative to a reference axis, wrapped to (-pi, pi].

    The weaker fallback for when no trustworthy 3D reference exists. It assumes
    the seat/task axis and the head heading are measured in the same image plane
    and that out-of-plane components can be ignored - which is a real assumption,
    not a formality, for a camera looking obliquely down at seated people. Use the
    SO(3) path whenever a reference is available; report which path produced each
    number.

    Invariant to a global rotation of the image plane, by the same argument as the
    SO(3) case with Q a planar rotation.
    """
    d = theta_head - theta_ref
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Causal reference-frame estimators
# ---------------------------------------------------------------------------


@dataclass
class ReferenceEstimate:
    """A reference orientation and how much it should be believed.

    ``confidence`` in [0, 1]. A low-confidence reference must not be silently
    consumed as if it were good: the feature assembler propagates it into the
    validity mask.
    """

    R_ref: torch.Tensor
    confidence: torch.Tensor
    source: str


class SceneReferenceEstimator:
    """Reference from fixed scene geometry: the direction this seat's monitor faces.

    Preferred, because it is the only option with no circularity: the monitor
    direction is a property of the room's furniture, measured once per seat, and
    is independent of anything the model predicts. In this corpus the camera and
    the desks never move, so a per-seat constant is defensible - and its
    per-seat variation is exactly the confound that camera-frame yaw suffers from
    and seat-frame yaw does not.

    The estimate is constant in time, hence trivially causal.
    """

    def __init__(self, seat_reference: dict, default_confidence: float = 1.0):
        """Args:
        seat_reference: ``{seat_id: (3, 3) tensor}``, the monitor/task direction
            for each seat as a rotation from seat-local to camera coordinates.
        """
        self.seat_reference = seat_reference
        self.default_confidence = default_confidence

    def __call__(self, seat_id, T: int, dtype=torch.float32, device=None) -> ReferenceEstimate:
        R = self.seat_reference.get(seat_id)
        if R is None:
            eye = torch.eye(3, dtype=dtype, device=device).expand(T, 3, 3).clone()
            return ReferenceEstimate(eye, torch.zeros(T, dtype=dtype, device=device), "scene:missing")
        R = R.to(dtype=dtype, device=device).expand(T, 3, 3).clone()
        conf = torch.full((T,), self.default_confidence, dtype=dtype, device=device)
        return ReferenceEstimate(R, conf, "scene")


class TorsoReferenceEstimator:
    """Reference from the student's own torso orientation, per frame.

    Second preference. It tracks a student who turns their chair, which the scene
    reference cannot, but it inherits the torso estimator's own noise and
    availability. Causal by construction: frame t uses frame t's torso only.
    """

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def __call__(self, R_torso: torch.Tensor, torso_conf: torch.Tensor) -> ReferenceEstimate:
        conf = torch.where(
            torso_conf >= self.min_confidence, torso_conf, torch.zeros_like(torso_conf)
        )
        return ReferenceEstimate(R_torso, conf, "torso")


class CausalSeatReferenceEstimator:
    """Running per-seat reference from past high-quality frames only.

    Last resort, and the one with a circularity risk that must be reported rather
    than buried. If the reference is bootstrapped from frames the system itself
    judged ``screen_oriented``, then "head agrees with the reference" is partly
    true by construction. BRANCH_C_PROTOCOL.md section 5 requires this estimator to
    be compared against the scene- and torso-derived references (arm 13) precisely
    so the size of that effect is measured rather than assumed away.

    Causality is structural, not incidental: :meth:`update` is called once per
    frame in order and :meth:`current` returns the state *before* the current
    frame is folded in, so the reference at t depends on frames < t only. Below
    ``min_observations`` it returns confidence 0 rather than a poorly-determined
    rotation.
    """

    def __init__(self, min_observations: int = 8, momentum: float = 0.05,
                 min_quality: float = 0.5):
        self.min_observations = min_observations
        self.momentum = momentum
        self.min_quality = min_quality
        self._state: dict = {}

    def reset(self) -> None:
        self._state.clear()

    def observations(self, seat_id) -> int:
        return self._state.get(seat_id, {}).get("n", 0)

    def current(self, seat_id, dtype=torch.float32, device=None) -> ReferenceEstimate:
        """The reference implied by frames strictly before the current one."""
        st = self._state.get(seat_id)
        if st is None or st["n"] < self.min_observations:
            return ReferenceEstimate(
                torch.eye(3, dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                "causal_seat:insufficient",
            )
        conf = min(1.0, st["n"] / (4.0 * self.min_observations))
        return ReferenceEstimate(
            st["R"].to(dtype=dtype, device=device),
            torch.tensor(conf, dtype=dtype, device=device),
            "causal_seat",
        )

    def update(self, seat_id, R_obs: torch.Tensor, quality: float) -> None:
        """Fold one observation in. Call **after** reading :meth:`current` for that frame.

        Averaging is done in the tangent space at the current estimate, which is
        the correct way to blend rotations - a componentwise mean of two rotation
        matrices is not a rotation.
        """
        if quality < self.min_quality:
            return
        st = self._state.get(seat_id)
        if st is None:
            self._state[seat_id] = {"R": R_obs.detach().clone(), "n": 1}
            return
        step = so3_log(st["R"].transpose(-1, -2) @ R_obs.detach())
        st["R"] = st["R"] @ so3_exp(step * self.momentum * float(quality))
        st["n"] += 1


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

#: Column layout of :func:`assemble_pose_features`. Recorded as a constant so the
#: feature-schema hash in outputs/branch_c/RUNS.jsonl means something.
POSE_FEATURE_LAYOUT = (
    ("rot6d_rel", 6),        # relative rotation, continuous representation
    ("log_rel", 3),          # axis-angle of the relative rotation
    ("geodesic_rel", 1),     # angle from the reference direction
    ("omega", 3),            # causal angular velocity
    ("alpha", 3),            # causal angular acceleration
    ("ref_confidence", 1),   # how much the reference itself is believed
)
POSE_FEATURE_DIM = sum(n for _, n in POSE_FEATURE_LAYOUT)  # 17


def assemble_pose_features(
    R_head: torch.Tensor,
    R_ref: torch.Tensor,
    head_valid: torch.Tensor,
    ref_confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the seat-relative pose feature block and its validity mask.

    Returns ``(features (T, POSE_FEATURE_DIM), valid (T,))``.

    A frame is valid only if the head estimate is valid **and** the reference is
    believed. Invalid frames are zeroed, but the mask is returned alongside and is
    never optional: a zero vector and a genuinely-zero rotation are different
    things, and the specification forbids conflating them.
    """
    R_rel = relative_rotation(R_ref, R_head)
    valid = head_valid.bool() & (ref_confidence > 0)

    feats = torch.cat(
        [
            matrix_to_rot6d(R_rel),
            so3_log(R_rel),
            geodesic_distance(
                torch.eye(3, dtype=R_rel.dtype, device=R_rel.device).expand_as(R_rel), R_rel
            ).unsqueeze(-1),
            relative_angular_velocity(R_rel, valid),
            relative_angular_acceleration(R_rel, valid),
            ref_confidence.unsqueeze(-1).to(R_rel.dtype),
        ],
        dim=-1,
    )
    feats = feats * valid.unsqueeze(-1).to(feats.dtype)
    return feats, valid


def from_head_pose_block(block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapt the legacy 4-column ``[yaw, pitch, roll, face_found]`` block to matrices.

    The legacy block is degrees/90 clipped to [-1, 1] (``head_pose.py:181``), so
    this inverts that scaling. **The clipping is not invertible**: any true
    |angle| > 90 degrees arrives saturated, and every student facing away from the
    camera lands on the same value. This adapter exists for the controlled
    comparison against the legacy features (arms 4 and 5), not because the
    representation is adequate.

    Returns ``(R (T, 3, 3), face_found (T,) bool)``.
    """
    deg = block[..., :3] * 90.0
    rad = deg * (math.pi / 180.0)
    R = euler_to_matrix(rad[..., 0], rad[..., 1], rad[..., 2])
    return R, block[..., 3] > 0.5
