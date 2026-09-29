import json

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledbsoma.io
import torch

from core.config import ScalingConfig
from core.datamodule import CensusDataModule, _fast_forward_experiment_dataset
from core.trainer import SegmentTimer
from run.train import train


def _write_tiny_soma(path):
    n_cells = 40
    n_genes = 16
    x = np.arange(n_cells * n_genes, dtype=np.float32).reshape(n_cells, n_genes)
    x = (x % 7) + 1
    x[:, :4] = 0
    obs = pd.DataFrame(
        {
            "is_primary_data": [True] * n_cells,
            "cell_type": [f"cell_type_{i % 3}" for i in range(n_cells)],
            "dataset_id": [f"dataset_{i % 2}" for i in range(n_cells)],
            "cell_type_ontology_term_id": [f"CL:{i % 3:07d}" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    tiledbsoma.io.from_anndata(
        str(path),
        adata,
        measurement_name="RNA",
        X_layer_name="normalized",
        raw_X_layer_name="normalized",
    )


def _write_celltypes(path):
    pd.DataFrame({"cell_type": [f"cell_type_{i}" for i in range(3)]}).to_csv(
        path, index=False
    )


def _write_dataset_ids(path):
    pd.DataFrame({"dataset_id": ["dataset_0", "dataset_1"]}).to_csv(path, index=False)


def _config(
    tmp_path,
    soma_path,
    resume_training,
    labels="none",
    celltypes_path=None,
    dataset_ids_path=None,
):
    return ScalingConfig(
        {
            "metadata": {
                "run_name": "resume_cursor_test",
                "title": "resume_cursor_test",
                "pipeline": "default",
            },
            "paths": {
                "path_to_aimrepo": None,
                "path_to_census": str(soma_path),
                "path_to_hvg": None,
                "path_to_means": None,
                "path_to_celltypes": str(celltypes_path) if celltypes_path else None,
                "path_to_dataset_ids": (
                    str(dataset_ids_path) if dataset_ids_path else None
                ),
                "path_to_ckpt_dir": str(tmp_path / "checkpoints"),
                "path_to_ckpt_file": None,
                "path_to_resume_checkpoint": None,
                "path_to_finetune_dir": None,
                "path_to_genecorpus_dir": None,
                "path_to_tokenized_text_dir": None,
                "path_to_tokenizer": None,
            },
            "model": {
                "model_class": "ffnn",
                "total_genes": 16,
                "context_length": 8,
                "d_model": 8,
                "dropout": 0.0,
                "freeze_encoder": False,
                "lr": 1e-3,
                "from_ckpt": False,
            },
            "datamodule": {
                "obs_value_filter": "is_primary_data == True",
                "labels": labels,
                "normalize_expr_for_ranking": True,
                "selection": "active",
                "n_bins": 4,
                "mask_strategy": "uniform",
                "mask_pct": 1.0,
                "batch_size": 4,
                "max_val_samples": 10,
            },
            "trainer": {
                "checkpoint": False,
                "training_state_checkpoint": False,
                "training_state_checkpoint_steps": 2,
                "resume_training": resume_training,
                "resume_dataloader": True,
                "compute_spearman": False,
                "find_lr": False,
                "val_check_interval": 1.0,
                "checkpoint_steps": 2,
                "n_epochs": 1,
                "n_steps": 4,
                "accumulate_grad": 1,
                "lr_scheduler": "constant",
                "precision": "full",
            },
        }
    )


def _consume_joinids(datamodule, n_batches):
    seen = []
    iterator = iter(datamodule.train_dataloader())
    for _ in range(n_batches):
        next(iterator)
        datamodule.train_dataset.data_cursor.commit_next_batch()
        seen.extend(datamodule.train_dataset.data_cursor.last_batch_soma_joinids)
    return seen


def _dataset_joinids(dataset):
    return [
        int(joinid)
        for _, obs_batch in dataset
        for joinid in obs_batch["soma_joinid"].tolist()
    ]


def test_segment_timer_does_not_restore_previous_segment_elapsed_time():
    timer = SegmentTimer("00:00:10:00")
    timer._offset = 599

    timer.load_state_dict({"time_elapsed": {"train": 599}})

    assert timer._offset == 0
    assert timer.state_dict() == {}


def test_datamodule_rejects_total_genes_mismatch_before_cuda_setup(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)
    config = _config(tmp_path, soma_path, resume_training=False)
    config.model.total_genes = 15

    with pytest.raises(
        ValueError,
        match=r"configured 15.*contains 16 RNA features",
    ):
        CensusDataModule(config)


def test_resumable_tiledbsoma_train_stream_does_not_replay_seen_cells(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)

    first_config = _config(tmp_path, soma_path, resume_training=False)
    first_dm = CensusDataModule(first_config)
    first_dm.setup()
    first_seen = _consume_joinids(first_dm, 3)
    first_dm.train_dataset.data_cursor.save_resume_marker(
        checkpoint_path="manual-test.ckpt", global_step=3
    )

    resume_config = _config(tmp_path, soma_path, resume_training=True)
    resume_dm = CensusDataModule(resume_config)
    resume_dm.setup()
    resumed_seen = _consume_joinids(resume_dm, 3)
    resume_dm.train_dataset.data_cursor.save_resume_marker(
        checkpoint_path="manual-test.ckpt", global_step=6
    )

    assert set(first_seen).isdisjoint(resumed_seen)

    state_path = (
        tmp_path
        / "checkpoints"
        / "resume_cursor_test"
        / "resume_cursor_test"
        / "training_state"
        / "data_cursor.json"
    )
    with open(state_path) as f:
        state = json.load(f)
    assert state["consumed_batches"] == 6
    assert state["global_step"] == 6


def test_fast_forward_slices_exact_shuffled_prefix_before_soma_reads(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)

    config = _config(tmp_path, soma_path, resume_training=False)
    datamodule = CensusDataModule(config)
    datamodule.setup()
    source_dataset = datamodule.train_dataset.dataset
    all_joinids = _dataset_joinids(source_dataset)

    consumed_batches = 3
    resumed_dataset = _fast_forward_experiment_dataset(
        source_dataset,
        consumed_batches,
    )
    expected_offset = consumed_batches * source_dataset.batch_size

    assert len(resumed_dataset.query_ids.obs_joinids) == len(all_joinids) - expected_offset
    assert _dataset_joinids(resumed_dataset) == all_joinids[expected_offset:]


def test_resumable_cursor_advances_after_a_complete_epoch(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)

    config = _config(tmp_path, soma_path, resume_training=False)
    datamodule = CensusDataModule(config)
    datamodule.setup()
    total_batches = len(datamodule.train_dataset)

    _consume_joinids(datamodule, total_batches)
    assert datamodule.train_dataset.data_cursor.consumed_batches == total_batches

    _consume_joinids(datamodule, 1)
    cursor = datamodule.train_dataset.data_cursor
    assert cursor.completed_epochs == 1
    assert cursor.consumed_batches == 1


def test_datamodule_uses_cached_dataset_ids_without_census_scan(tmp_path, monkeypatch):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)
    dataset_ids_path = tmp_path / "dataset_ids.csv"
    _write_dataset_ids(dataset_ids_path)

    def fail_build_dataset_id_map(*args, **kwargs):
        raise AssertionError("dataset id map should come from cached CSV")

    monkeypatch.setattr(
        "core.datamodule.build_dataset_id_map", fail_build_dataset_id_map
    )

    config = _config(
        tmp_path,
        soma_path,
        resume_training=False,
        dataset_ids_path=dataset_ids_path,
    )
    datamodule = CensusDataModule(config)
    datamodule.setup()

    assert datamodule.train_dataset.dataset_id_map == {
        "dataset_0": 0,
        "dataset_1": 1,
    }


def test_training_state_checkpoint_is_saved_separately(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)
    celltypes_path = tmp_path / "celltypes.csv"
    _write_celltypes(celltypes_path)

    config = _config(
        tmp_path,
        soma_path,
        resume_training=False,
        labels="cell_type",
        celltypes_path=celltypes_path,
    )
    config.trainer.training_state_checkpoint = True
    config.trainer.training_state_checkpoint_steps = 1
    config.trainer.resume_dataloader = False
    config.trainer.n_steps = 2
    config.trainer.val_check_interval = 2
    config_path = tmp_path / "train.yml"
    config.save(str(config_path))

    train(str(config_path))

    run_dir = tmp_path / "checkpoints" / "resume_cursor_test" / "resume_cursor_test"
    assert (run_dir / "training_checkpoints" / "last.ckpt").exists()
    assert list(run_dir.glob("*.pt"))


def test_segment_deadline_stops_cleanly_with_resumable_checkpoint(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)

    config = _config(tmp_path, soma_path, resume_training=False)
    config.trainer.training_state_checkpoint = True
    config.trainer.training_state_checkpoint_steps = 10
    config.trainer.max_time = "00:00:00:00"
    config.trainer.limit_val_batches = 0
    config_path = tmp_path / "deadline_train.yml"
    config.save(str(config_path))

    train(str(config_path))

    run_dir = tmp_path / "checkpoints" / "resume_cursor_test" / "resume_cursor_test"
    assert (run_dir / "training_checkpoints" / "last.ckpt").exists()
    assert (run_dir / "training_state" / "data_cursor.json").exists()
    assert not (run_dir / "DONE").exists()


def test_cursor_commits_complete_gradient_accumulation_windows(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)

    config = _config(tmp_path, soma_path, resume_training=False)
    config.trainer.training_state_checkpoint = True
    config.trainer.training_state_checkpoint_steps = 1
    config.trainer.accumulate_grad = 2
    config.trainer.n_steps = 2
    config.trainer.val_check_interval = 2
    config.trainer.limit_val_batches = 0
    config_path = tmp_path / "accumulated_train.yml"
    config.save(str(config_path))

    train(str(config_path))

    run_dir = tmp_path / "checkpoints" / "resume_cursor_test" / "resume_cursor_test"
    checkpoint = torch.load(
        run_dir / "training_checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["global_step"] == 2
    assert checkpoint["training_data_cursor"]["consumed_batches"] == 4


def test_training_resumes_from_last_checkpoint_and_cursor(tmp_path):
    soma_path = tmp_path / "tiny_soma"
    _write_tiny_soma(soma_path)
    celltypes_path = tmp_path / "celltypes.csv"
    _write_celltypes(celltypes_path)

    config = _config(
        tmp_path,
        soma_path,
        resume_training=False,
        labels="cell_type",
        celltypes_path=celltypes_path,
    )
    config.trainer.training_state_checkpoint = True
    config.trainer.training_state_checkpoint_steps = 1
    config.trainer.resume_dataloader = True
    config.trainer.n_steps = 2
    config.trainer.val_check_interval = 2
    config_path = tmp_path / "resume_train.yml"
    config.save(str(config_path))

    train(str(config_path))

    run_dir = tmp_path / "checkpoints" / "resume_cursor_test" / "resume_cursor_test"
    state_path = run_dir / "training_state" / "data_cursor.json"
    with open(state_path) as f:
        first_state = json.load(f)
    assert first_state["consumed_batches"] == 2

    # The checkpoint is authoritative. A stale or independently advanced sidecar
    # must not change which batches resume considers committed.
    stale_state = dict(first_state)
    stale_state["consumed_batches"] = 999
    state_path.write_text(json.dumps(stale_state), encoding="utf-8")

    config.trainer.n_steps = 4
    config.save(str(config_path), overwrite=True)

    train(str(config_path), resume=True)

    with open(state_path) as f:
        resumed_state = json.load(f)
    assert resumed_state["consumed_batches"] == 4
    last_checkpoint = run_dir / "training_checkpoints" / "last.ckpt"
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 4
    assert checkpoint["training_data_cursor"]["consumed_batches"] == 4
    assert resumed_state["global_step"] == checkpoint["global_step"]
