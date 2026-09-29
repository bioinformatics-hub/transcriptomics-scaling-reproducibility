"""Regression checks for the three historical training semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core.config import ScalingConfig, TrainingBehavior, training_behavior
from core.datamodule import BinnedMaskedDataset
from core.models import TransformerModel, legacy_binned_losses


def config_for(behavior):
    return ScalingConfig(
        {
            "metadata": {
                "run_name": "test",
                "pipeline": "default",
                "training_behavior": behavior,
            },
            "paths": {},
            "model": {
                "model_class": "transformer",
                "total_genes": 32,
                "context_length": 16,
                "d_model": 8,
                "dropout": 0.0,
                "lr": 0.001,
                "transformer": {
                    "n_layers": 1,
                    "n_heads": 2,
                    "gating": False,
                    "norm_first": True,
                },
            },
            "datamodule": {
                "labels": "none",
                "mask_pct": 0.15,
                "n_bins": 5,
                "mask_strategy": "uniform",
                "selection": "active",
            },
            "trainer": {
                "n_steps": 10,
                "lr_scheduler": "constant",
                "compute_spearman": False,
                "val_check_interval": 1,
                "find_lr": False,
            },
        }
    )


def test_behavior_validation_and_serialization():
    assert training_behavior(ScalingConfig()) == TrainingBehavior.CORRECTED
    assert config_for("legacy").to_dict()["metadata"]["training_behavior"] == "legacy"
    with pytest.raises(ValueError):
        config_for("typo")


@pytest.mark.parametrize("behavior", ["legacy", "legacy_padding", "corrected"])
def test_config_controls_attention_and_training_loss(behavior):
    model = TransformerModel(config_for(behavior)).eval()
    genes = torch.tensor([[0, 1, 32]])
    expressions = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, False]])
    captured = []
    original_forward = model.encoder.forward

    def record_forward(x, attention_mask=None):
        captured.append(attention_mask)
        return original_forward(x, attention_mask=attention_mask)

    model.encoder.forward = record_forward
    result = model.step((expressions, genes, mask, None), 0)
    assert torch.isfinite(result["losses"]["bce"])
    if behavior == "corrected":
        assert torch.equal(captured[0], torch.tensor([[True, True, False]]))
    else:
        assert captured[0] is None
    # Compare all modes at identical decoder predictions; exercise the public step.
    logits = torch.tensor([[1.0, -1.0, 2.0]], requires_grad=True)
    predictions = torch.tensor([[3.0, 4.0, 5.0]], requires_grad=True)
    model.decoder.forward = lambda x: {"mlm": predictions, "active_logits": logits}
    result = model.step((expressions, genes, mask, None), 0)
    expected = F.binary_cross_entropy_with_logits(
        logits[mask], (expressions > 0)[mask].float()
    )
    if behavior == "legacy":
        expected = expected + 2 * torch.log(torch.tensor(2.0))
    assert torch.allclose(
        result["losses"]["bce"], expected.to(result["losses"]["bce"].dtype)
    )


def test_original_mse_selects_predicted_activity_not_target_activity():
    predictions = torch.tensor([[3.0, 4.0]])
    logits = torch.tensor([[1.0, -1.0]])
    targets = torch.tensor([[0.0, 2.0]])
    mse, _ = legacy_binned_losses(
        predictions, logits, targets, torch.ones(1, 2, dtype=torch.bool)
    )
    assert mse == 9


@pytest.mark.parametrize(
    "behavior, length", [("legacy", 15), ("legacy_padding", 15), ("corrected", 16)]
)
def test_config_controls_actual_binned_iterator(behavior, length):
    config = config_for(behavior)
    dataset = BinnedMaskedDataset([], config, {})
    dataset._iter_dataset_batches = lambda: iter(
        [
            (
                0,
                (
                    np.array([[1.0, 2.0] + [0.0] * 30]),
                    SimpleNamespace(cell_type_ontology_term_id=["unknown"]),
                ),
            )
        ]
    )
    dataset.get_y_batch = lambda obs: None
    dataset.get_dataset_id = lambda obs: None
    batch = next(iter(dataset))
    assert batch[0].shape == (1, length)
    expected_pad = 32 if behavior == "corrected" else 33
    assert expected_pad in batch[1]
