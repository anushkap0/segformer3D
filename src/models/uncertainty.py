"""Phase 5: uncertainty estimation.

Two approaches are provided, since they suit different parts of this project
differently and it's worth being deliberate about which you pick:

1. MC-Dropout (`enable_mc_dropout`, `mc_dropout_predict`) -- requires NO architecture
   change and NO retraining. The decoder already has `nn.Dropout3d` before the
   classifier head (see `AllMLPDecoder3D.dropout` in segformer3d.py). At test time,
   keep dropout layers stochastic (train-mode) while keeping BatchNorm/InstanceNorm
   in eval mode, run N forward passes, and treat the sample mean/variance as the
   prediction and its epistemic uncertainty. This is the cheapest, most defensible
   starting point -- just run it directly on your Phase 2 checkpoint.

2. Evidential Deep Learning (`EvidentialHead`, `EvidentialLoss`) -- a single forward
   pass predicts Dirichlet concentration parameters instead of raw logits, giving a
   principled per-voxel uncertainty (vacuity) without needing multiple stochastic
   passes at inference time. This is the standard formulation for mutually-exclusive
   softmax classification (Sensoy, Kaplan & Kandemir, NeurIPS 2018); it does NOT map
   directly onto our default sigmoid multi-label BraTS formulation (WT/TC/ET are
   overlapping, non-exclusive regions -- a Dirichlet over mutually exclusive classes
   doesn't apply there as-is). Use `EvidentialHead` only if you retrain with
   `mutually_exclusive: true` on the raw 4-way label set (background, NCR/NET, ED, ET).

Recommendation: start with MC-Dropout (works today, on your existing checkpoint) and
only build out the evidential path if you specifically want single-pass uncertainty
for a latency-sensitive framing.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. MC-Dropout
# --------------------------------------------------------------------------- #
def enable_mc_dropout(model: nn.Module) -> None:
    """Puts the whole model in eval() (so BatchNorm/InstanceNorm use running stats,
    not batch stats -- important since MC-Dropout inference is typically run with
    batch size 1), then flips only Dropout/Dropout3d submodules back to train()
    so they keep sampling instead of becoming a no-op.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout3d, nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 20,
    is_boundary_model: bool = False,
    eval_output: str = "refined_logits",
    mutually_exclusive: bool = False,
) -> Dict[str, torch.Tensor]:
    """Runs `num_samples` stochastic forward passes and returns:
      - mean_prob: (C, D, H, W) mean predicted probability across samples
      - predictive_variance: (C, D, H, W) per-voxel variance across samples
                              (epistemic uncertainty proxy)
      - predictive_entropy: (D, H, W) entropy of the mean prediction
                             (total uncertainty, only meaningful for
                             mutually_exclusive=True softmax outputs; for sigmoid
                             multi-label outputs use predictive_variance instead)

    x is expected as a single-sample batch: (1, C_in, D, H, W).
    """
    enable_mc_dropout(model)
    probs_samples = []

    for _ in range(num_samples):
        outputs = model(x)
        logits = outputs[eval_output] if is_boundary_model else outputs
        if mutually_exclusive:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
        probs_samples.append(probs[0])  # drop batch dim -> (C, D, H, W)

    stacked = torch.stack(probs_samples, dim=0)  # (N, C, D, H, W)
    mean_prob = stacked.mean(dim=0)
    predictive_variance = stacked.var(dim=0, unbiased=False)

    result = {"mean_prob": mean_prob, "predictive_variance": predictive_variance}

    if mutually_exclusive:
        eps = 1e-8
        entropy = -(mean_prob * torch.log(mean_prob + eps)).sum(dim=0)
        result["predictive_entropy"] = entropy

    return result


# --------------------------------------------------------------------------- #
# 2. Evidential Deep Learning head (for mutually_exclusive training only)
# --------------------------------------------------------------------------- #
class EvidentialHead(nn.Module):
    """Replaces a plain softmax classifier with an evidence-predicting head.

    Evidence e_k >= 0 is produced per class via softplus (rather than raw logits),
    Dirichlet parameters alpha_k = e_k + 1, and:
      - expected probability p_k = alpha_k / S   where S = sum_k(alpha_k)
      - uncertainty (vacuity) u  = num_classes / S
    A high S (lots of evidence) means low uncertainty; S == num_classes (all e_k=0)
    is the maximum-uncertainty / "I don't know" state.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.evidence_conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = F.softplus(self.evidence_conv(features))  # (B, C, D, H, W), >= 0
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)  # S
        prob = alpha / strength
        uncertainty = self.num_classes / strength.squeeze(1)  # (B, D, H, W)
        return {"alpha": alpha, "prob": prob, "uncertainty": uncertainty, "evidence": evidence}


class EvidentialLoss(nn.Module):
    """Expected mean-square-error loss + KL-divergence regularizer toward a uniform
    Dirichlet, following Sensoy et al. (NeurIPS 2018). `target` is a one-hot
    (B, C, D, H, W) tensor. `annealing_step` controls how fast the KL regularizer
    is ramped in (starting near 0, since applying full regularization from epoch 0
    can suppress evidence collection before the model has learned anything useful).
    """

    def __init__(self, num_classes: int, annealing_step: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        strength = alpha.sum(dim=1, keepdim=True)
        prob = alpha / strength

        # expected mean square error term
        err = (target - prob) ** 2
        var = prob * (1 - prob) / (strength + 1)
        mse = (err + var).sum(dim=1).mean()

        # KL(Dir(alpha_tilde) || Dir(1)) regularizer, where alpha_tilde removes the
        # evidence that correctly supports the true class (so the regularizer only
        # penalizes *misleading* evidence, not correct evidence)
        alpha_tilde = target + (1 - target) * alpha
        kl = self._kl_dirichlet_uniform(alpha_tilde)

        annealing_coef = min(1.0, epoch / max(self.annealing_step, 1))
        return mse + annealing_coef * kl.mean()

    def _kl_dirichlet_uniform(self, alpha: torch.Tensor) -> torch.Tensor:
        # alpha: (B, C, D, H, W). KL divergence of Dir(alpha) from Dir(all-ones).
        c = alpha.shape[1]
        strength = alpha.sum(dim=1, keepdim=True)
        term1 = torch.lgamma(strength.squeeze(1)) - torch.lgamma(alpha).sum(dim=1) \
            - torch.lgamma(torch.tensor(float(c), device=alpha.device))
        term2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength))).sum(dim=1)
        return term1 + term2
