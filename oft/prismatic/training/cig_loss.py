import math

import torch
import torch.nn.functional as F


def compute_l1_energy(
    predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error per chunk; CIG uses its negative as an OFT score.

    OFT regresses continuous actions. This is not an AR token log-likelihood.
    Reduce in float32 so mixed-precision forwards retain small energy margins.
    """
    if predicted_actions.ndim != 3 or predicted_actions.shape != ground_truth_actions.shape:
        raise ValueError("Actions must have matching [batch, horizon, action_dim] shapes")
    return (predicted_actions.float() - ground_truth_actions.float()).abs().mean(dim=(1, 2))


def compute_cig_loss(
    pos_energy: torch.Tensor, neg_energies: list[torch.Tensor], tau_c: float = 1.0
) -> torch.Tensor:
    """Negative log probability of the positive instruction under energy scores."""
    if not math.isfinite(tau_c) or tau_c <= 0:
        raise ValueError("CIG temperature must be finite and positive")
    if not neg_energies:
        raise ValueError("CIG requires at least one negative instruction")
    if pos_energy.ndim != 1 or any(e.shape != pos_energy.shape for e in neg_energies):
        raise ValueError("Each energy must have shape [batch]")
    scores = [-pos_energy] + [-e for e in neg_energies]
    logits = torch.stack(scores, dim=1).float() / tau_c
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
