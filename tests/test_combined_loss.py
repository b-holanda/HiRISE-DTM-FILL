import torch

from marsfill.model.combined_loss import CombinedLoss, LossWeights, GradientLoss


def test_gradient_loss_zero_on_identical():
    pred = torch.ones(1, 1, 3, 3)
    target = torch.ones(1, 1, 3, 3)
    loss = GradientLoss()
    assert loss(pred, target).item() == 0.0


def test_combined_loss_components():
    pred = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    weights = LossWeights(l1_weight=1.0, gradient_weight=1.0, ssim_weight=1.0)
    loss_fn = CombinedLoss(loss_weights=weights)
    total, l1, grad, ssim_loss = loss_fn(pred, target)
    assert total.item() >= 0
    assert l1.item() >= 0
    assert grad.item() >= 0
    assert ssim_loss.item() >= 0
