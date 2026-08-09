"""Branch-C training losses: cue, boundary, smoothing, robustness and fusion
regularisers, each its own callable with its own coefficient, combined by
:func:`total_loss`.

Reused rather than reimplemented (BRANCH_C_PROTOCOL.md instructs against
duplicating existing objectives):

- :func:`smooth_loss` wraps
  ``attention.thesis_eval.train._smoothing_loss`` verbatim — the project's
  existing MS-TCN truncated-MSE (T-MSE) smoothing term on adjacent
  log-probabilities. Not reimplemented.
- :func:`boundary_loss`'s default mode reuses the exact positive-class
  reweighting scheme inlined in ``attention.thesis_eval.train.train()``
  (transitions are ~2% of frames; unweighted BCE collapses to "no boundary
  anywhere"), and reuses
  ``attention.thesis_eval.models.boundary_targets_from_labels`` for the
  transition-target definition used by the project's ASRF model. Not
  reimplemented.
- :func:`calibration_loss` is **not** the project's post-hoc temperature
  scaling in ``attention.thesis_eval.calibrate`` (Guo et al., fitted on the
  validation split only, argmax-preserving). That procedure is a separate,
  frozen, val-only post-processing step and is deliberately left untouched.
  ``calibration_loss`` here is a differentiable *training-time* proper
  scoring regulariser (Brier / NLL) on raw softmax probabilities, added to
  the training objective like any other term; it follows the same Brier/NLL
  conventions as ``attention.thesis_eval.metrics.brier_score`` /
  ``negative_log_likelihood`` (Brier = mean squared error against one-hot,
  range [0, 2]; NLL = mean negative log-probability of the true class) but
  is not the same code path and must never be described as calibrating
  anything — it is a regulariser on the training objective, calibration
  proper still happens once, post-hoc, on validation, exactly where it
  already lives.

Every loss below returns a finite scalar tensor. Degenerate inputs (e.g. a
batch with no valid — non-``ignore_index`` — frames) return an exact
zero that stays attached to the computation graph, rather than raising or
returning NaN, so a term that happens not to apply to a given batch never
poisons ``total_loss``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from attention.thesis_eval.models import boundary_targets_from_labels
from attention.thesis_eval.train import _smoothing_loss as _mstcn_tmse_smoothing_loss

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# cue_loss
# ---------------------------------------------------------------------------

def cue_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mode: str = "focal",
    class_counts: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    beta: float = 0.9999,
    tau: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Cue classification loss, two selectable objectives (choice deferred to
    inner validation, per protocol — this function implements both, it does
    not choose between them):

    ``mode="focal"``: class-balanced focal cross-entropy. Per-class weights
    follow Cui et al. 2019's "effective number of samples":
    ``w_c = (1 - beta) / (1 - beta ** n_c)``, renormalised to mean 1; the
    focal modulating factor is ``(1 - p_t) ** gamma`` (Lin et al. 2017).

    ``mode="logit_adjusted"``: logit-adjusted softmax cross-entropy (Menon
    et al. 2021): logits are shifted by ``-tau * log(pi_c)`` before the
    softmax, where ``pi_c`` is the empirical class prior from
    ``class_counts``.

    ``class_counts`` is the *training*-split per-class frame count (6
    classes here); if omitted both modes fall back to an unweighted /
    uniform-prior loss rather than raising, so the function is always
    callable in a unit test with no class statistics.

    Returns 0.0 (graph-attached) if every target equals ``ignore_index``.
    """
    c = logits.shape[-1]
    flat_logits = logits.reshape(-1, c)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0

    sel_logits = flat_logits[valid]
    sel_targets = flat_targets[valid]

    if mode == "logit_adjusted":
        if class_counts is None:
            log_prior = torch.zeros(c, device=logits.device, dtype=logits.dtype)
        else:
            counts = torch.as_tensor(class_counts, dtype=logits.dtype,
                                     device=logits.device).clamp_min(1.0)
            prior = counts / counts.sum()
            log_prior = torch.log(prior)
        adjusted = sel_logits - tau * log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted, sel_targets)

    if mode == "focal":
        logp = F.log_softmax(sel_logits, dim=-1)
        p = logp.exp()
        logp_t = logp.gather(1, sel_targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, sel_targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - p_t).clamp_min(0.0) ** gamma
        loss = -focal_factor * logp_t
        if class_counts is not None:
            counts = torch.as_tensor(class_counts, dtype=logits.dtype,
                                     device=logits.device).clamp_min(1.0)
            beta_t = torch.as_tensor(beta, dtype=logits.dtype, device=logits.device)
            eff_num = 1.0 - torch.pow(beta_t, counts)
            cb_weight = (1.0 - beta) / eff_num.clamp_min(1e-12)
            cb_weight = cb_weight / cb_weight.mean()
            loss = loss * cb_weight[sel_targets]
        return loss.mean()

    raise ValueError(f"unknown cue_loss mode {mode!r}")


# ---------------------------------------------------------------------------
# boundary_loss
# ---------------------------------------------------------------------------

def boundary_loss(
    boundary_logits: torch.Tensor,
    y: torch.Tensor,
    mode: str = "pos_weighted_bce",
    gamma: float = 2.0,
    pos_weight_clamp: tuple = (1.0, 50.0),
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Boundary-transition loss over the ASRF-style boundary head.

    Transition targets come from
    ``attention.thesis_eval.models.boundary_targets_from_labels`` (reused,
    not reimplemented). ``mode="pos_weighted_bce"`` (default) reuses the
    positive-class reweighting inlined in
    ``attention.thesis_eval.train.train()``. ``mode="focal"`` additionally
    applies a focal modulating factor on top of the same weighting.

    Returns 0.0 if every frame is ``ignore_index``.
    """
    bt = boundary_targets_from_labels(y, ignore_index)
    valid = (y != ignore_index).to(bt.dtype)
    if valid.sum() == 0:
        return boundary_logits.sum() * 0.0

    pos = bt.sum().clamp(min=1.0)
    total = valid.sum().clamp(min=1.0)
    pw = ((total - pos) / pos).clamp(pos_weight_clamp[0], pos_weight_clamp[1])

    bce = F.binary_cross_entropy_with_logits(
        boundary_logits, bt, reduction="none", pos_weight=pw)
    if mode == "pos_weighted_bce":
        bl = bce
    elif mode == "focal":
        p = torch.sigmoid(boundary_logits)
        pt = torch.where(bt > 0.5, p, 1.0 - p)
        focal_factor = (1.0 - pt).clamp_min(0.0) ** gamma
        bl = focal_factor * bce
    else:
        raise ValueError(f"unknown boundary_loss mode {mode!r}")
    return (bl * valid).sum() / valid.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# smooth_loss
# ---------------------------------------------------------------------------

def smooth_loss(logits: torch.Tensor, valid: torch.Tensor, clamp: float = 16.0) -> torch.Tensor:
    """Wraps ``attention.thesis_eval.train._smoothing_loss`` verbatim — the
    project's existing MS-TCN truncated-MSE smoothing between adjacent
    log-probabilities. Not reimplemented; see that function's docstring for
    the definition.
    """
    return _mstcn_tmse_smoothing_loss(logits, valid, clamp=clamp)


# ---------------------------------------------------------------------------
# view_consistency_loss
# ---------------------------------------------------------------------------

def jsd(logits_a: torch.Tensor, logits_b: torch.Tensor, dim: int = -1,
        eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence (natural log) between two softmax posteriors.

    Symmetric in ``(logits_a, logits_b)`` by construction and bounded in
    ``[0, ln 2]`` per row.
    """
    p = F.softmax(logits_a, dim=dim).clamp_min(eps)
    q = F.softmax(logits_b, dim=dim).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=dim)
    kl_qm = (q * (q.log() - m.log())).sum(dim=dim)
    return 0.5 * kl_pm + 0.5 * kl_qm


def view_consistency_loss(
    logits_clean: torch.Tensor,
    logits_perturbed: torch.Tensor,
    feats_clean: Optional[torch.Tensor] = None,
    feats_perturbed: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    feature_weight: float = 0.0,
) -> torch.Tensor:
    """JSD between the cue posteriors of a clean sample and a globally
    rotated/perturbed view of the same sample (BRANCH_C_PROTOCOL §8: rotation
    invariance is unit-tested and probed synthetically), plus an optional
    feature-consistency MSE term when ``feature_weight > 0``.
    """
    d = jsd(logits_clean, logits_perturbed)
    if valid_mask is not None:
        vm = valid_mask.to(d.dtype)
        loss = (d * vm).sum() / vm.sum().clamp(min=1.0)
    else:
        loss = d.mean()
    if feature_weight > 0.0 and feats_clean is not None and feats_perturbed is not None:
        loss = loss + feature_weight * F.mse_loss(feats_clean, feats_perturbed)
    return loss


# ---------------------------------------------------------------------------
# quality_order_loss
# ---------------------------------------------------------------------------

def quality_order_loss(r_clean: torch.Tensor, r_corrupt: torch.Tensor,
                       margin: float = 0.5) -> torch.Tensor:
    """Hinge on the reliability head's ordering: the clean view's predicted
    log-reliability must exceed the corrupted view's by at least ``margin``.

    ``max(0, margin - r_clean + r_corrupt)``, averaged. Zero exactly when the
    ordering already holds with the required margin.
    """
    return F.relu(margin - r_clean + r_corrupt).mean()


# ---------------------------------------------------------------------------
# counterfactual_loss
# ---------------------------------------------------------------------------

def counterfactual_loss(
    logits_clean: torch.Tensor,
    logits_corrupt: torch.Tensor,
    clean_confidence: torch.Tensor,
    retained_evidence: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Distil the clean fused posterior into the corrupted-path posterior.

    ``KL(stopgrad(softmax(logits_clean)) || softmax(logits_corrupt))``,
    weighted per-sample by ``clean_confidence * retained_evidence``.

    The stop-gradient is a hard ``.detach()`` on the clean side: this term
    trains the corrupted path towards the clean one and never the reverse
    (there is no path back to ``logits_clean`` through this loss).

    ``retained_evidence`` (in [0, 1], supplied by the caller — e.g.
    ``1 - <estimate of the removed modality's unique information>``) is the
    explicit gate the spec requires: when the modality removed to produce
    the "corrupted" view carried information the remaining modalities do not
    retain, ``retained_evidence -> 0`` and this term must not force the
    corrupted posterior to match the clean one — forcing that would teach
    the model the modality was irrelevant when it was not.
    """
    p_clean = F.softmax(logits_clean.detach(), dim=-1).clamp_min(eps)
    logp_corrupt = F.log_softmax(logits_corrupt, dim=-1)
    kl = (p_clean * (p_clean.log() - logp_corrupt)).sum(dim=-1)
    gate = clean_confidence * retained_evidence
    weighted = kl * gate
    if valid_mask is not None:
        vm = valid_mask.to(weighted.dtype)
        return (weighted * vm).sum() / vm.sum().clamp(min=1.0)
    return weighted.mean()


# ---------------------------------------------------------------------------
# calibration_loss
# ---------------------------------------------------------------------------

def calibration_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mode: str = "brier",
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Differentiable proper-scoring training regulariser (Brier or NLL) on
    raw softmax probabilities. See module docstring: this is NOT the
    post-hoc, validation-only temperature scaling in
    ``attention.thesis_eval.calibrate`` — that stays untouched.
    """
    c = logits.shape[-1]
    flat_logits = logits.reshape(-1, c)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0

    sel_logits = flat_logits[valid]
    sel_targets = flat_targets[valid]
    probs = F.softmax(sel_logits, dim=-1)

    if mode == "nll":
        logp = F.log_softmax(sel_logits, dim=-1)
        return -logp.gather(1, sel_targets.unsqueeze(1)).squeeze(1).mean()
    if mode == "brier":
        onehot = torch.zeros_like(probs).scatter_(1, sel_targets.unsqueeze(1), 1.0)
        return ((probs - onehot) ** 2).sum(dim=-1).mean()
    raise ValueError(f"unknown calibration_loss mode {mode!r}")


# ---------------------------------------------------------------------------
# distill_loss
# ---------------------------------------------------------------------------

def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_feats: Optional[torch.Tensor] = None,
    teacher_feats: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    kl_weight: float = 1.0,
    feat_weight: float = 0.0,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Teacher -> lightweight-student distillation (Hinton et al. 2015).

    Temperature-softened KL(teacher || student), teacher fully stop-gradient,
    rescaled by ``temperature ** 2`` (standard correction so the gradient
    magnitude is roughly temperature-invariant), plus an optional
    feature-matching MSE term (teacher features also stop-gradient).
    """
    log_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits.detach() / temperature, dim=-1).clamp_min(eps)
    kl = (p_t * (p_t.log() - log_s)).sum(dim=-1) * (temperature ** 2)
    if valid_mask is not None:
        vm = valid_mask.to(kl.dtype)
        kl_term = (kl * vm).sum() / vm.sum().clamp(min=1.0)
    else:
        kl_term = kl.mean()
    loss = kl_weight * kl_term
    if feat_weight > 0.0 and student_feats is not None and teacher_feats is not None:
        loss = loss + feat_weight * F.mse_loss(student_feats, teacher_feats.detach())
    return loss


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------

# Every coefficient defaults to 0.0 except "cue" — the protocol's required
# zero-weight baseline: with these defaults, total_loss reduces to cue_loss
# alone (arm 3, OVERT appearance-only, and the ablations against it, depend
# on this being exact).
DEFAULT_COEFFICIENTS: Dict[str, float] = {
    "cue": 1.0,
    "boundary": 0.0,
    "smooth": 0.0,
    "view_consistency": 0.0,
    "quality_order": 0.0,
    "counterfactual": 0.0,
    "calibration": 0.0,
    "distill": 0.0,
}


def total_loss(
    terms: Dict[str, torch.Tensor],
    coefficients: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Weighted sum of already-computed loss terms.

    ``terms`` maps loss name -> scalar tensor (only compute the terms you
    intend to use — this function does not call the loss functions itself,
    it only combines their outputs, so a screening run that never sets a
    coefficient never has to pay for computing that term either).

    A term whose coefficient is exactly 0.0 is **skipped entirely** rather
    than multiplied by zero, so a term that is degenerate/NaN when unused
    (e.g. a robustness loss computed on a batch with no corrupted view) can
    never poison the total — this is what "zero-weight configuration
    reproduces the plain baseline exactly" means operationally, not just
    numerically.

    ``coefficients`` defaults to :data:`DEFAULT_COEFFICIENTS`. A coefficient
    dict that omits a key falls back to 0.0, except "cue" which falls back
    to 1.0 to match the default baseline.
    """
    if "cue" not in terms:
        raise KeyError("total_loss requires a 'cue' term")
    coeffs = coefficients if coefficients is not None else DEFAULT_COEFFICIENTS

    total = None
    for name, value in terms.items():
        default = 1.0 if name == "cue" else 0.0
        c = coeffs.get(name, default)
        if c == 0.0:
            continue
        contrib = value * c
        total = contrib if total is None else total + contrib
    if total is None:
        total = terms["cue"].sum() * 0.0
    return total
