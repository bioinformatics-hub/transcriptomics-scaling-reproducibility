import torch

from core.models import masked_active_mse_loss, masked_binary_cross_entropy_with_logits


def test_masked_bce_ignores_unmasked_positions():
    logits = torch.zeros((1, 4), dtype=torch.float32)
    targets = torch.zeros((1, 4), dtype=torch.bool)
    mask = torch.tensor([[True, False, False, False]])

    loss = masked_binary_cross_entropy_with_logits(logits, targets, mask)

    assert torch.allclose(loss, torch.log(torch.tensor(2.0)))


def test_masked_bce_does_not_add_unmasked_log_two_artifact():
    logits = torch.zeros((1, 4), dtype=torch.float32)
    targets = torch.zeros((1, 4), dtype=torch.bool)
    mask = torch.tensor([[True, False, False, False]])

    correct_loss = masked_binary_cross_entropy_with_logits(logits, targets, mask)
    old_artifact_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits * mask.float(),
            targets.float() * mask.float(),
            reduction="sum",
        )
        / mask.float().sum()
    )

    assert torch.allclose(old_artifact_loss, 4 * correct_loss)


def test_masked_bce_matches_manual_masked_selection():
    logits = torch.tensor([[0.2, -1.5, 3.0], [0.0, 2.0, -0.7]], dtype=torch.float32)
    targets = torch.tensor([[True, False, True], [False, True, False]])
    mask = torch.tensor([[True, False, True], [False, True, False]])

    loss = masked_binary_cross_entropy_with_logits(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask],
        targets[mask].float(),
    )

    assert torch.allclose(loss, expected)


def test_masked_active_mse_uses_true_active_masked_positions():
    predictions = torch.tensor([[10.0, 2.0, 7.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 5.0, 3.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False, True]])

    loss = masked_active_mse_loss(predictions, targets, mask)

    assert torch.allclose(loss, torch.tensor(9.0))


def test_masked_active_mse_returns_zero_when_no_active_masked_targets():
    predictions = torch.tensor([[10.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 5.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])

    loss = masked_active_mse_loss(predictions, targets, mask)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_empty_mask_losses_remain_differentiable():
    predictions = torch.tensor([[1.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 3.0]])
    mask = torch.zeros_like(targets, dtype=torch.bool)

    loss = masked_binary_cross_entropy_with_logits(
        predictions, targets > 0, mask
    ) + masked_active_mse_loss(predictions, targets, mask)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.equal(predictions.grad, torch.zeros_like(predictions))
