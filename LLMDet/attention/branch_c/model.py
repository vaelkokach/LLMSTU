"""Multi-expert cue model with observability-weighted fusion.

Architecture, following BRANCH_C_PROTOCOL.md §4 arms 8-12 and 17:

    appearance (552) ──► expert TCN ──► z_base ─────────────┐
    head       (9)   ──► expert TCN ──► z_head ──┐          │
    motion     (6)   ──► expert TCN ──► z_motion ┤          │
                                                 ▼          ▼
    quality    (8)   ──► ReliabilityHead ──► α ──► z_fused = z_base + Σ αₘ zₘ
                                                            │
                                                            ▼
                                            MS-TCN refinement stages ──► logits

Three design points that are not free choices:

*The reliability head sees only the quality block.* It is a separate tensor
argument all the way down, so "the gate cannot see content" is a property of the
API rather than a convention. ``test_reliability_head_cannot_see_content`` pins it.

*z_base is a residual.* With every expert masked out the fusion returns exactly
``z_base``, so the model degrades to appearance-only rather than to NaN or to
noise. This is what makes the modality-dropout arm (11) meaningful.

*Refinement runs on fused logits.* MS-TCN's later stages consume the previous
stage's softmax; feeding them the fused prediction is what lets the temporal
prior clean up a fusion decision, rather than the two fighting each other.

Everything here is causal in the sense the protocol requires *except* the
dilated convolutions, which are symmetric in the reference MS-TCN and are left
that way deliberately: arms 1-5 use the same backbone, so changing it here would
confound the fusion comparison with an architecture change. The online/causal
variant is a runtime question (protocol §7.2), not a fusion question, and is
recorded as such rather than silently mixed in.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from attention.branch_c.fusion import ReliabilityFusion, masked_softmax
from attention.thesis_eval.models import MSTCN, _SingleStageTCN

#: Column layout of grounding_data/llmstu_sequences_branch_c. Asserted at load.
LAYOUT: Dict[str, Tuple[int, int]] = {
    "appearance": (0, 552),
    "head": (552, 561),
    "quality": (561, 569),
    "motion": (569, 575),
}
TOTAL_DIM = 575
EXPERTS = ("head", "motion")          # gated experts; appearance is the residual base


class MultiExpertCueModel(nn.Module):
    """Appearance baseline plus reliability-gated head and motion experts."""

    def __init__(
        self,
        num_classes: int = 6,
        channels: int = 128,
        expert_channels: int = 64,
        num_layers: int = 10,
        refine_stages: int = 3,
        dropout: float = 0.5,
        fusion: str = "learned",       # "learned" | "uniform" | "none"
        tau: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_mode = fusion
        self.expert_names = list(EXPERTS)

        a0, a1 = LAYOUT["appearance"]
        self.base = _SingleStageTCN(a1 - a0, channels, num_layers, num_classes, dropout)
        self.experts = nn.ModuleDict({
            name: _SingleStageTCN(LAYOUT[name][1] - LAYOUT[name][0],
                                  expert_channels, num_layers, num_classes, dropout)
            for name in self.expert_names
        })
        q0, q1 = LAYOUT["quality"]
        self.fuse = ReliabilityFusion(num_modalities=len(self.expert_names),
                                      quality_dim=q1 - q0, tau=tau)
        self.refine = MSTCN(num_classes, num_classes, channels, num_layers,
                            num_stages=refine_stages, dropout=dropout)

    @staticmethod
    def _slice(x: torch.Tensor, name: str) -> torch.Tensor:
        a, b = LAYOUT[name]
        return x[..., a:b]

    def availability(self, x: torch.Tensor) -> torch.Tensor:
        """[M, B, T] bool. A modality is available when its evidence is non-degenerate.

        Head: the rotation block is exactly zero when the estimator produced
        nothing for that crop. Motion: frame 0 has no backward difference, so it
        is genuinely unavailable rather than zero-valued.
        """
        head = self._slice(x, "head").abs().sum(-1) > 0
        motion = torch.ones_like(head)
        motion[:, 0] = False
        return torch.stack([head, motion], dim=0)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, 575]
        pad_mask: Optional[torch.Tensor] = None,   # [B, T] True where padded
        mask_override: Optional[torch.Tensor] = None,
        permute_reliability: Optional[torch.Generator] = None,
    ) -> Tuple[dict, dict]:
        """Returns (``{"logits": [B,T,C], "aux_logits": [S,B,T,C]}``, diagnostics).

        ``mask_override`` forces modality availability, which is how arm 11
        (drop each modality at inference) is run without retraining.

        ``permute_reliability`` shuffles α across modalities with everything else
        held fixed — the registered arm-17 diagnostic. If performance survives
        this, the gate is decorative and no fusion contribution may be claimed.
        """
        B, T, _ = x.shape
        valid = (torch.ones(B, 1, T, dtype=x.dtype, device=x.device)
                 if pad_mask is None else (~pad_mask).unsqueeze(1).to(x.dtype))

        def run(module, name):
            a, b = LAYOUT[name]
            out, _ = module(x[..., a:b].transpose(1, 2), valid)       # [B, C, T]
            return out.transpose(1, 2)                                # [B, T, C]

        zb = run(self.base, "appearance")                             # [B, T, C]
        ze = torch.stack([run(self.experts[n], n) for n in self.expert_names], 0)

        mask = self.availability(x) if mask_override is None else mask_override
        if pad_mask is not None:
            mask = mask & (~pad_mask).unsqueeze(0)
        q = self._slice(x, "quality").unsqueeze(0).expand(len(self.expert_names), -1, -1, -1)

        if self.fusion_mode == "none":
            zeros = torch.zeros(mask.shape, dtype=zb.dtype, device=zb.device)
            z_fused, diag = zb, {"alpha": zeros, "r": zeros}
        elif self.fusion_mode == "uniform":
            alpha = masked_softmax(
                torch.zeros(mask.shape, dtype=zb.dtype, device=zb.device), mask, dim=0)
            ze_avail = torch.where(mask.unsqueeze(-1), ze, torch.zeros_like(ze))
            z_fused = zb + (alpha.unsqueeze(-1) * ze_avail).sum(0)
            diag = {"alpha": alpha, "r": torch.zeros_like(alpha)}
        else:
            z_fused, diag = self.fuse(zb, ze, q, mask)

        if permute_reliability is not None:
            alpha = diag["alpha"]
            perm = torch.randperm(alpha.shape[0], generator=permute_reliability,
                                  device=alpha.device)
            ze_avail = torch.where(mask.unsqueeze(-1), ze, torch.zeros_like(ze))
            z_fused = zb + (alpha[perm].unsqueeze(-1) * ze_avail).sum(0)
            diag = {**diag, "alpha": alpha[perm], "permuted": True}

        out = self.refine(z_fused, pad_mask)
        # The fused prediction is prepended as a supervised stage. Without this the
        # only gradient reaching the experts and the gate passes through the
        # refinement chain's 6-dim softmax bottleneck, and the whole model collapses
        # to a constant prediction — measured, not hypothesised: macro-F1 sat at
        # 0.1436 for every epoch and every arm until this was added. MS-TCN's
        # reference implementation supervises every stage for the same reason.
        out["aux_logits"] = torch.cat([z_fused.unsqueeze(0), out["aux_logits"]], dim=0)
        diag["z_fused"] = z_fused
        diag["mask"] = mask
        return out, diag
