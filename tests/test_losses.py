import pytest
import torch

from src.losses import robustness_loss


def make_labels():
    return torch.tensor([[0.0], [1.0], [0.0], [1.0]])


def test_identical_logits_have_zero_consistency_loss():
    labels = make_labels()
    logits = torch.tensor([[0.0], [1.0], [-1.0], [2.0]])

    _, _, loss_con = robustness_loss(logits, logits, labels, lam=0.5)

    assert loss_con.item() == pytest.approx(0.0)


def test_different_logits_have_positive_consistency_loss():
    labels = make_labels()
    logits_clean = torch.tensor([[-4.0], [4.0], [-4.0], [4.0]])
    logits_damaged = torch.tensor([[4.0], [-4.0], [4.0], [-4.0]])

    _, _, loss_con = robustness_loss(logits_clean, logits_damaged, labels, lam=0.5)

    assert loss_con.item() > 0


def test_lambda_zero_makes_total_loss_equal_classification_loss():
    labels = make_labels()
    logits_clean = torch.tensor([[-2.0], [2.0], [-1.0], [1.0]])
    logits_damaged = torch.tensor([[2.0], [-2.0], [1.0], [-1.0]])

    loss_total, loss_cls, _ = robustness_loss(logits_clean, logits_damaged, labels, lam=0)

    assert loss_total.item() == pytest.approx(loss_cls.item())


def test_larger_lambda_increases_total_loss_for_different_logits():
    labels = make_labels()
    logits_clean = torch.tensor([[-4.0], [4.0], [-4.0], [4.0]])
    logits_damaged = torch.tensor([[4.0], [-4.0], [4.0], [-4.0]])

    loss_total_small, _, _ = robustness_loss(logits_clean, logits_damaged, labels, lam=0.25)
    loss_total_large, _, _ = robustness_loss(logits_clean, logits_damaged, labels, lam=1.0)

    assert loss_total_large.item() > loss_total_small.item()


def test_negative_lambda_raises_value_error():
    labels = make_labels()
    logits_clean = torch.tensor([[0.0], [1.0], [-1.0], [2.0]])
    logits_damaged = torch.tensor([[0.5], [0.5], [-0.5], [1.5]])

    with pytest.raises(ValueError):
        robustness_loss(logits_clean, logits_damaged, labels, lam=-0.1)


def test_loss_total_backprop_creates_gradients_for_both_logits():
    labels = make_labels()
    logits_clean = torch.tensor([[0.0], [1.0], [-1.0], [2.0]], requires_grad=True)
    logits_damaged = torch.tensor([[0.5], [0.5], [-0.5], [1.5]], requires_grad=True)

    loss_total, _, _ = robustness_loss(logits_clean, logits_damaged, labels, lam=0.5)
    loss_total.backward()

    assert logits_clean.grad is not None
    assert logits_damaged.grad is not None


def test_losses_are_finite():
    labels = make_labels()
    logits_clean = torch.tensor([[0.0], [1.0], [-1.0], [2.0]])
    logits_damaged = torch.tensor([[0.5], [0.5], [-0.5], [1.5]])

    loss_total, loss_cls, loss_con = robustness_loss(
        logits_clean,
        logits_damaged,
        labels,
        lam=0.5,
    )

    assert torch.isfinite(loss_total)
    assert torch.isfinite(loss_cls)
    assert torch.isfinite(loss_con)
