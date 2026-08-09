"""Reliability-gated fusion of per-modality cue experts, Branch C.

Implements :class:`ReliabilityFusion`:

    z_m(i,t)   = per-modality cue logits  (supplied by the caller — each
                 modality's own encoder/head, computed from that modality's
                 *content* features)
    r_m(i,t)   = predicted log-reliability / log-precision, computed by a
                 small head from QUALITY/CONTEXT inputs only (never the
                 content features that produced z_m)
    alpha_m    = masked_softmax(r_m / tau)         over available modalities
    z_fused    = z_base + sum_m alpha_m * z_m

``z_base`` is supplied by the caller (e.g. the appearance-only baseline
model's logits) and is a residual path, not something this module computes.
That is what makes the degrade-to-baseline property in
BRANCH_C_PROTOCOL.md §5/§7.1(3) hold by construction: if every modality is
masked out, ``alpha`` is exactly zero everywhere (see :func:`masked_softmax`)
and ``z_fused == z_base``, bit for bit.

Diagnostics
-----------
``alpha`` and ``r_m`` are exposed for the dashboard as **routing/reliability
diagnostics** only. Per BRANCH_C_PROTOCOL.md's framing rules, they are never
to be called "explanations" and never described as "causal" — a large
``alpha_m`` says the gate trusted modality ``m`` more at that (i, t), not that
modality ``m`` caused the prediction.

Theoretical note (limited; not the novel part)
-----------------------------------------------
Under the standard assumption of conditionally unbiased, mutually uncorrelated
expert errors with per-modality variances ``sigma_m^2``, the minimum-variance
linear unbiased combination of the experts weights each by ``1/sigma_m^2``
(inverse-variance / precision weighting; e.g. the classical fixed-effects
meta-analysis result). A log-precision head whose softmax output approximates
those normalised precision weights is therefore approximating a known,
textbook estimator — it is not claimed as a novel result. What Branch C adds
(and what *is* being tested) is that the precision estimate is learned from
observable *quality/context* signals rather than assumed constant, that it is
structurally blind to content, and that it is trained jointly with the
quality-order / counterfactual losses in :mod:`attention.branch_c.losses`.
The assumptions (conditional unbiasedness, zero cross-modality error
correlation) are almost certainly violated in practice; the fusion head is a
learned approximation under those assumptions, not a proof they hold here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# Default softmax temperature for the reliability gate. 1.0 (no sharpening or
# flattening of the learned log-precision before the softmax) is the recorded
# default; screening may retune it on inner validation only, per protocol §2.4.
DEFAULT_TAU: float = 1.0


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Softmax over ``dim``, restricted to positions where ``mask`` is True.

    ``mask`` and ``logits`` must broadcast to the same shape. Behaviour at the
    edge cases the fusion gate must hit exactly:

    * **all modalities missing** along ``dim`` for some slice: returns exactly
      zero for every entry in that slice (no NaN, no uniform fallback) — this
      is what makes ``z_fused == z_base`` when nothing is available.
    * **exactly one modality present**: returns 1.0 for that entry and 0.0 for
      the (masked) rest, independent of temperature.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    any_available = mask.any(dim=dim, keepdim=True)

    # Guard the max-subtraction: a fully-masked slice would otherwise compute
    # max == neg_inf and produce 0/0 == NaN below. Substituting 0.0 for the
    # max in that slice keeps every following op finite; the result is zeroed
    # explicitly afterwards regardless of what this branch produces.
    safe_max = torch.where(any_available, masked_logits.max(dim=dim, keepdim=True).values,
                            torch.zeros_like(masked_logits.max(dim=dim, keepdim=True).values))
    exp = torch.exp(masked_logits - safe_max) * mask.to(logits.dtype)
    denom = exp.sum(dim=dim, keepdim=True)
    denom_safe = torch.where(any_available, denom, torch.ones_like(denom))
    alpha = exp / denom_safe
    alpha = alpha * any_available.to(logits.dtype)
    return alpha


class ReliabilityHead(nn.Module):
    """Maps per-modality QUALITY/CONTEXT features to a scalar log-reliability.

    Structurally cannot see content: its only tensor input is ``quality``
    (shape ``[M, ..., quality_dim]``), never the per-modality content features
    that ``z_m`` was computed from. A learned per-modality embedding is
    concatenated in as additional *context* (modality identity) — this is
    still not a content feature, it is a fixed lookup keyed only by modality
    index, so it carries no information about the current sample.
    """

    def __init__(self, quality_dim: int, num_modalities: int, hidden_dim: int = 32):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_embed = nn.Embedding(num_modalities, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(quality_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        """``quality``: [M, ..., quality_dim] -> log-reliability [M, ...]."""
        m = quality.shape[0]
        assert m == self.num_modalities, (
            f"quality has {m} modalities, head was built for {self.num_modalities}")
        idx = torch.arange(m, device=quality.device)
        emb = self.modality_embed(idx)  # [M, H]
        extra = quality.dim() - 2
        emb = emb.view(m, *([1] * extra), -1).expand(*quality.shape[:-1], -1)
        x = torch.cat([quality, emb], dim=-1)
        return self.net(x).squeeze(-1)


class ReliabilityFusion(nn.Module):
    """Gate ``M`` per-modality cue-logit experts onto a residual baseline.

    Call signature is deliberately explicit about which tensor is which, so
    that "the reliability head cannot see content" is a fact about the API,
    not a convention someone can violate by accident:

    ``forward(z_base, z_experts, quality, mask, tau=None)``

    - ``z_base``:    [B, T, C]        baseline (appearance-only) cue logits
    - ``z_experts``: [M, B, T, C]     per-modality cue logits (content-derived)
    - ``quality``:   [M, B, T, Q]     per-modality quality/context features
    - ``mask``:      [M, B, T] bool   True where that modality is available
    - returns ``(z_fused, diagnostics)`` where ``diagnostics`` is a dict with
      keys ``"alpha"`` [M,B,T] and ``"r"`` [M,B,T] — routing/reliability
      diagnostics for the dashboard, not explanations, not causal claims.
    """

    def __init__(self, num_modalities: int, quality_dim: int,
                 hidden_dim: int = 32, tau: float = DEFAULT_TAU):
        super().__init__()
        self.num_modalities = num_modalities
        self.tau = tau
        self.head = ReliabilityHead(quality_dim, num_modalities, hidden_dim)

    def forward(
        self,
        z_base: torch.Tensor,
        z_experts: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        tau_ = self.tau if tau is None else tau
        r = self.head(quality)                       # [M, B, T]
        alpha = masked_softmax(r / tau_, mask, dim=0)  # [M, B, T]

        # Neutralise unavailable experts *before* weighting. A zero alpha is not
        # enough on its own: IEEE says 0 * NaN is NaN, and an unavailable expert's
        # slot legitimately holds whatever placeholder the feature builder left
        # there. Without this, one missing head-pose estimate turns the entire
        # fused output for that track into NaN.
        z_avail = torch.where(
            mask.unsqueeze(-1), z_experts, torch.zeros_like(z_experts)
        )
        z_fused = z_base + (alpha.unsqueeze(-1) * z_avail).sum(dim=0)
        return z_fused, {"alpha": alpha, "r": r}
