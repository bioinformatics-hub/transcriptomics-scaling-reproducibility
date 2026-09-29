from collections import Counter

import pytest
import tiledbsoma as soma

from core.datamodule import CensusDataModule
from core.config import Pipeline
from test.test_resumable_datamodule import (
    _config,
    _write_celltypes,
    _write_dataset_ids,
    _write_tiny_soma,
)
from tools.coarse_labels import collect_validation_leaf_ids


@pytest.mark.parametrize("max_samples", [1, 100])
@pytest.mark.parametrize("pipeline", [Pipeline.DEFAULT, Pipeline.GENEFORMER])
def test_metadata_labels_match_validation_without_expression_reads(tmp_path, monkeypatch, max_samples, pipeline):
    soma_path = tmp_path / "census"
    _write_tiny_soma(soma_path)
    celltypes = tmp_path / "celltypes.csv"
    dataset_ids = tmp_path / "dataset_ids.csv"
    _write_celltypes(celltypes)
    _write_dataset_ids(dataset_ids)
    config = _config(tmp_path, soma_path, False, celltypes_path=celltypes, dataset_ids_path=dataset_ids)
    config.datamodule.max_val_samples = max_samples
    config.metadata.pipeline = pipeline
    # Exercise selection from a filtered population, not the entire obs table.
    config.datamodule.obs_value_filter = "is_primary_data == True and soma_joinid >= 7"
    datamodule = CensusDataModule(config)
    datamodule.setup()
    expected = [str(label) for batch in datamodule.val_dataloader() for label in batch[5]]

    def forbid_expression_read(*args, **kwargs):
        raise AssertionError("Coarse labels must not read expression matrices")

    monkeypatch.setattr(soma.SparseNDArray, "read", forbid_expression_read)
    monkeypatch.setattr(soma.DenseNDArray, "read", forbid_expression_read)
    actual = collect_validation_leaf_ids(config)
    assert Counter(actual) == Counter(expected)
    assert actual == collect_validation_leaf_ids(config)
