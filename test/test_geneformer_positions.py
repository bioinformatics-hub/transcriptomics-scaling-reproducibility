from types import SimpleNamespace

import numpy as np
import torch

from core.config import ScalingConfig
from core.datamodule import RankValueDataset
from core.models import PositionalEncoding


def make_geneformer_config():
    return ScalingConfig(
        {
            "metadata": {
                "run_name": "test_geneformer_global_positions",
                "pipeline": "geneformer",
            },
            "model": {
                "total_genes": 4,
                "context_length": 2,
                "d_model": 4,
                "dropout": 0.0,
                "model_class": "transformer",
                "from_ckpt": False,
                "transformer": {
                    "n_heads": 1,
                    "gating": False,
                    "n_layers": 1,
                    "norm_first": False,
                },
            },
            "datamodule": {
                "labels": "none",
                "mask_pct": 0.0,
                "batch_size": 1,
                "obs_value_filter": "all",
                "normalize_expr_for_ranking": False,
            },
            "trainer": {
                "compute_spearman": False,
                "val_check_interval": 1,
                "find_lr": False,
            },
            "paths": {
                "path_to_means": "missing_means.csv",
            },
        }
    )


class DummyDataset:
    def __iter__(self):
        yield (
            np.array([[1.0, 0.0, 3.0, 2.0]], dtype=float),
            SimpleNamespace(
                dataset_id=["dataset_a"],
                cell_type_ontology_term_id=["CL:0000000"],
            ),
        )


def test_rank_value_dataset_emits_global_positions(monkeypatch):
    config = make_geneformer_config()
    dataset = RankValueDataset(
        DummyDataset(),
        config,
        dataset_id_map={"dataset_a": 0},
    )

    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    expr, gene_ids, mask, _, y_dataset, leaf_ids, position_ids = next(iter(dataset))

    assert gene_ids.tolist() == [[2, 0]]
    assert position_ids.tolist() == [[0, 2]]
    assert mask.tolist() == [[False, False]]
    assert y_dataset.tolist() == [0]
    assert leaf_ids == ["CL:0000000"]
    assert torch.allclose(expr, torch.tensor([[0.5, 1.0 / 6.0]], dtype=torch.float32))


def test_positional_encoding_uses_explicit_positions():
    positional = PositionalEncoding(d_model=4, dropout=0.0, max_len=2)
    x = torch.zeros((1, 2, 4), dtype=torch.float32)
    position_ids = torch.tensor([[0, 2]], dtype=torch.long)

    out = positional(x, positions=position_ids)

    assert torch.allclose(out[0, 0], positional.pe[0])
    assert torch.allclose(out[0, 1, 0::2], torch.sin(torch.tensor([2.0]) * positional.div_term))
    assert torch.allclose(out[0, 1, 1::2], torch.cos(torch.tensor([2.0]) * positional.div_term))
