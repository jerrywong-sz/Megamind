import torch
import torch.nn.functional as F


def robustness_loss(logits_clean, logits_damaged, labels, lam):
    """Compute classification plus consistency loss for clean/damaged image pairs."""
    if lam < 0:
        raise ValueError("lam must be non-negative")

    loss_clean = F.binary_cross_entropy_with_logits(logits_clean, labels)
    loss_damaged = F.binary_cross_entropy_with_logits(logits_damaged, labels)
    loss_cls = loss_clean + loss_damaged

    prob_clean = torch.sigmoid(logits_clean)
    prob_damaged = torch.sigmoid(logits_damaged)
    loss_con = ((prob_clean - prob_damaged) ** 2).mean()

    loss_total = loss_cls + lam * loss_con
    return loss_total, loss_cls, loss_con
