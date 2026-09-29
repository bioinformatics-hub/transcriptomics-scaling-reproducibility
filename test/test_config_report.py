import math
import os

import pandas as pd
import yaml

from tools.config_report import build_report_row, generate_report


def test_generate_report_from_master(tmp_path):
    master_folder = os.path.join(os.environ["PATH_TO_TEST_CONFIGS"], "forward_flops")
    output_csv = tmp_path / "report.csv"

    df = generate_report(master_folder=master_folder, output_csv=str(output_csv))

    assert output_csv.exists()
    assert len(df) == 2

    disk_df = pd.read_csv(output_csv)
    assert len(disk_df) == 2
    assert {"pipeline", "model_class", "total_params", "estimated_train_vram_gib"}.issubset(
        disk_df.columns
    )
    assert (disk_df["total_params"] > 0).all()
    assert (disk_df["estimated_train_vram_gib"] > 0).all()
    assert (disk_df["estimated_inference_vram_gib"] > 0).all()
    assert (
        disk_df["estimated_train_vram_gib"] >= disk_df["estimated_inference_vram_gib"]
    ).all()
    assert set(disk_df["pipeline"]) == {"default", "geneformer"}
    assert math.isfinite(float(disk_df["forward_flops"].iloc[0]))


def test_geneformer_like_long_context_vram_estimate_is_conservative(tmp_path):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    config = {
        "metadata": {"run_name": "geneformer_like_large", "pipeline": "geneformer"},
        "paths": {
            "path_to_aimrepo": str(shared_dir),
            "path_to_census": str(shared_dir),
            "path_to_hvg": str(shared_dir / "hvg.csv"),
            "path_to_means": str(shared_dir / "means.csv"),
            "path_to_celltypes": str(shared_dir / "celltypes.csv"),
            "path_to_ckpt_dir": str(shared_dir),
            "path_to_coarse_labels_dir": str(shared_dir),
            "path_to_ckpt_file": None,
            "path_to_finetune_dir": None,
        },
        "model": {
            "model_class": "transformer",
            "total_genes": 61497,
            "context_length": 2048,
            "d_model": 1024,
            "dropout": 0.2,
            "freeze_encoder": True,
            "lr": 1e-3,
            "from_ckpt": False,
            "transformer": {
                "n_layers": 6,
                "n_heads": 4,
                "gating": False,
                "norm_first": True,
            },
            "bioformer": {
                "n_layers": 4,
                "n_heads": 4,
                "d_opm": 32,
                "d_z": 128,
                "pair_updates": True,
                "chunk_size": 200,
                "gating": False,
                "norm_first": True,
            },
        },
        "datamodule": {
            "obs_value_filter": "is_primary_data == True",
            "labels": "cell_type",
            "normalize_expr_for_ranking": True,
            "selection": "active",
            "n_bins": 50,
            "mask_strategy": "uniform",
            "mask_pct": 0.15,
            "batch_size": 32,
            "max_val_samples": 30000,
        },
        "trainer": {
            "checkpoint": True,
            "compute_spearman": False,
            "find_lr": False,
            "val_check_interval": 500,
            "checkpoint_steps": 12500,
            "n_epochs": 1,
            "n_steps": 50000,
            "accumulate_grad": 1,
            "lr_scheduler": "onecycle",
            "precision": "bf16-mixed",
        },
        "downstream": {
            "poll_interval_s": 60,
            "clustering_method": "leiden",
            "resolution": 1.0,
            "n_neighbors": 15,
            "coarse_k": 30,
            "ridge_seed": 0,
            "keep_artifacts": False,
        },
    }

    for filename in ("hvg.csv", "means.csv", "celltypes.csv"):
        (shared_dir / filename).write_text("x\n", encoding="utf-8")

    config_path = tmp_path / "geneformer_like_large.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    row = build_report_row(str(config_path))

    assert row["encoder_activation_memory_gib"] > 0
    assert row["decoder_logit_memory_gib"] > 0
    assert row["train_runtime_overhead_gib"] > 0
    assert row["estimated_train_vram_gib"] > 60
