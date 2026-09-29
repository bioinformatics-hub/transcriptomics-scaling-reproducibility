"""
Finetune each model .ckpt in the checkpoints directory for the cell-type classification task.

Directory structure required:
./checkpoints/
├── model_name_size_1/
│   ├── config.json
│   ├── checkpoint1.ckpt
│   ├── checkpoint2.ckpt
│   └── ... (other checkpoints)
├── model_name_size_2/
│   ├── config.json
│   ├── checkpoint1.ckpt
│   ├── checkpoint2.ckpt
│   └── ... (other checkpoints)
└── ... (other model folders)

Each subfolder "model_name_size" must contain config.json and all intermediate .ckpt files saved.
"""

import os
import re
import glob
import time 
import errno
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl

from core.trainer import ScalingTrainer
from core.datamodule import CensusDataModule
import core.models as models
from core.config import ScalingConfig, Label
from tools.utils import suppress_messages

# Housekeeping
torch.set_float32_matmul_precision("medium")

LOCK_MAX_RETRIES = 60
LOCK_WAIT_SECS = 1.0


def _acquire_lock(lock_path: str, max_retries: int = LOCK_MAX_RETRIES, wait_s: float = LOCK_WAIT_SECS) -> None:
    """
    Simple cross-process mutex using an atomic lock file (O_CREAT | O_EXCL).
    """
    for _ in range(max_retries):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return  # acquired
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            time.sleep(wait_s)
    raise TimeoutError(f"Could not acquire lock {lock_path!r} after {max_retries} retries.")


def _release_lock(lock_path: str) -> None:
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass


def append_result_safely(path_to_results: str, result_dict: dict) -> None:
    """
    Append a result row to path_to_results in a race-free way using pandas,
    protected by a lock file.
    """
    lock_path = path_to_results + ".lock"
    _acquire_lock(lock_path)
    try:
        if os.path.exists(path_to_results) and os.path.getsize(path_to_results) > 0:
            df = pd.read_csv(path_to_results)
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
        else:
            df = pd.DataFrame([result_dict])

        df.to_csv(path_to_results, index=False)
    finally:
        _release_lock(lock_path)


def get_all_embeddings(
    model, dataloader: DataLoader, device, return_expr=False, max_samples: int = -1, desc=""
):
    """
    Generates embeddings for an entire dataset using a given model.
    """
    all_embeddings = []
    all_labels = []
    all_dataset_ids = []
    all_expr = []
    model.to(device)
    model.eval()
    batch_size = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            expr, gene_ids, _, cell_type, dataset_ids, _, *extras = batch
            position_ids = extras[0] if extras else None
            if batch_size is None:
                batch_size, _ = expr.shape

            tensors = [
                expr.to(device),
                gene_ids.to(device),
                cell_type.to(device),
                dataset_ids.to(device),
            ]
            if position_ids is not None:
                tensors.append(position_ids.to(device))
            expr, gene_ids, cell_type, dataset_ids = tensors[:4]
            position_ids = tensors[4] if len(tensors) == 5 else None

            encoder_output = model.get_encoder_output(
                gene_ids, expr, position_ids=position_ids
            )
            cell_embeddings = encoder_output.mean(dim=1)

            all_embeddings.append(cell_embeddings.cpu())
            all_labels.append(cell_type.cpu())
            all_dataset_ids.append(dataset_ids.cpu())
            all_expr.append(expr.cpu())

            if max_samples > 0 and len(all_embeddings) * batch_size >= max_samples:
                break

    return torch.cat(all_embeddings), torch.cat(all_labels), torch.cat(all_dataset_ids), None if not return_expr else torch.cat(all_expr)

def finetune_lightning(
    trainer_config: ScalingConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    ckpt_title: str,
    ckpt_folder: str,
    ckpt_file: str,
    step: int,
):
    trainer = ScalingTrainer(trainer_config)

    # Train
    trainer.fit(model, datamodule)

    # Test
    test_results = trainer.test(model, datamodule, verbose=False)

    # Prepare results dictionary
    result_dict = {
        "run_title": ckpt_title,
        "model_folder": ckpt_folder,
        "checkpoint": ckpt_file,
        "step": step,
        "finetune_type": "lightning",
        "timestamp": str(datetime.now()),
    }
    result_dict.update(test_results[0])

    return result_dict


def finetune_ridge(
    path_to_celltypes: str,
    n_steps: int,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    ckpt_title: str,
    ckpt_folder: str,
    ckpt_file: str,
    step: int,
):
    datamodule.setup()

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate embeddings for train and test sets
    X_train, y_train, _, _ = get_all_embeddings(
        model=model,
        dataloader=datamodule.train_dataloader(),
        device=device,
        max_samples=n_steps,
        desc="Generating train embeddings",
    )
    X_test, y_test, _, _ = get_all_embeddings(
        model=model,
        dataloader=datamodule.test_dataloader(),
        device=device,
        desc="Generating test embeddings",
    )

    # Prepare for closed-form solution
    num_features = X_train.shape[1]
    num_classes = len(pd.read_csv(path_to_celltypes).cell_type.unique())
    y_train_one_hot = F.one_hot(y_train, num_classes=num_classes).float()

    # Move to device for computation
    X_train, y_train_one_hot, X_test = (
        X_train.to(device),
        y_train_one_hot.to(device),
        X_test.to(device),
    )

    # Solve for weights using closed-form ridge regression
    alpha = 1.0
    eye = torch.eye(num_features, device=device)

    # Instead of:
    # weights = torch.inverse(X_train.T @ X_train + alpha * I) @ (X_train.T @ y_train_one_hot)
    # Use a least squares solver (with regularization included in the system):
    # (more numerically stable, hopefully no longer singular matrix)
    A = X_train.T @ X_train + alpha * eye
    B = X_train.T @ y_train_one_hot
    weights = torch.linalg.lstsq(A, B).solution

    # Make predictions on the test set and calculate accuracy
    logits = X_test @ weights
    predictions = torch.argmax(logits, dim=1).cpu()
    accuracy = (predictions == y_test).float().mean().item()

    # Prepare results dictionary
    result_dict = {
        "run_title": ckpt_title,
        "model_folder": ckpt_folder,
        "checkpoint": ckpt_file,
        "step": step,
        "test_accuracy": accuracy,
        "test_ce": 0,
        "finetune_type": "ridge",
        "timestamp": str(datetime.now()),
    }

    return result_dict


def final_plot(path_to_results: str, path_to_plot: str):
    df = pd.read_csv(path_to_results)
    print("\n\n\n Finetuning completed, creating final plot...")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="step",
        y="test_accuracy",
        hue="model_folder",
        marker="o",
        linewidth=2,
        palette="tab10",
    )
    plt.title("Cell Type Classification Accuracy vs Training Step")
    plt.xlabel("Training Step")
    plt.ylabel("Test Accuracy")
    plt.legend(title="Model Args", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(path_to_plot)
    print(f"Plot saved to {path_to_plot}")


def validate_ckpt_dir_structure(path_to_ckpt_dir: str) -> None:
    """
    Validates that each subdirectory in path_to_ckpt_dir contains exactly one config (.yml) file
    and one or more .ckpt files, with no unexpected files.
    """
    for run_title in glob.glob(f"{path_to_ckpt_dir}/*"):
        if not os.path.isdir(run_title):
            continue
        for run_name in glob.glob(f"{run_title}/*"):
            if not os.path.isdir(run_name):
                raise ValueError(
                    f"Expected directory in '{run_title}', but found file: '{os.path.basename(run_name)}'"
                )
            config_files = [f for f in glob.glob(f"{run_name}/*.yml")]
            ckpt_files = [f for f in glob.glob(f"{run_name}/*.ckpt")]
            other_files = [
                f
                for f in glob.glob(f"{run_name}/*")
                if not (f.endswith(".yml") or f.endswith(".ckpt"))
            ]

            if len(config_files) == 0:
                raise FileNotFoundError(
                    f"No '.yml' config file found in '{run_name}'. Each model folder must contain one config file."
                )
            if len(config_files) > 1:
                raise ValueError(
                    f"Multiple '.yml' config files found in '{run_name}': {config_files}. Only one config file is allowed per folder."
                )
            if len(ckpt_files) == 0:
                raise FileNotFoundError(
                    f"No '.ckpt' checkpoint files found in '{run_name}'. At least one checkpoint file is required."
                )
            if other_files:
                raise ValueError(
                    f"Unexpected files found in '{run_name}': {[os.path.basename(f) for f in other_files]}. "
                    "Only '.yml' and '.ckpt' files are allowed."
                )


def finetune(
    path_to_config: str,
    finetune_type: str,
    job_index: int | None = None,
    num_jobs: int | None = None,
    ):
    suppress_messages()

    # Load config
    config = ScalingConfig(path_to_config)

    # Finetune logic
    config.datamodule.mask_pct = 0.0
    config.datamodule.freeze_encoder = True
    config.trainer.checkpoint = False
    config.datamodule.labels = Label.CELL_TYPE
    assert os.path.exists(config.paths.path_to_celltypes)
    assert os.path.exists(config.paths.path_to_ckpt_dir)
    assert len(os.listdir(config.paths.path_to_ckpt_dir)) > 0

    # Ensure expected structure
    validate_ckpt_dir_structure(config.paths.path_to_ckpt_dir)

    # Paths for saving results
    path_to_results = os.path.join(
        config.paths.path_to_ckpt_dir, "finetune_results.csv"
    )
    path_to_plot = os.path.join(config.paths.path_to_ckpt_dir, "finetune_plot.png")
    
    if job_index is not None and num_jobs is not None:
        assert 0 <= job_index < num_jobs, "job_index must be in [0, num_jobs)"
        print(f"[finetune] Running shard {job_index + 1}/{num_jobs}")
    else:
        print("[finetune] Running in single-job mode (no sharding).")

    global_idx = 0
    # For all config.yml and their checkpoints
    for path_to_ckpt_folder in glob.glob(
        f"{config.paths.path_to_ckpt_dir}/*/*"
    ):  # checkpoints/run_title/run_name
        if not os.path.isdir(path_to_ckpt_folder):
            continue

        ckpt_config = ScalingConfig(os.path.join(path_to_ckpt_folder, "config.yml"))

        ckpt_folder = os.path.basename(path_to_ckpt_folder)
        ckpt_title = os.path.basename(os.path.dirname(path_to_ckpt_folder))

        # We need to have the correct pipeline from the ckpt_config
        config.metadata.pipeline == ckpt_config.metadata.pipeline
        config.datamodule.normalize_expr_for_ranking = (
            ckpt_config.datamodule.normalize_expr_for_ranking
        )
        config.datamodule.n_bins = ckpt_config.datamodule.n_bins

        # Loop through all intermediate checkpoints
        for path_to_ckpt_file in glob.glob(f"{path_to_ckpt_folder}/*.ckpt"):
            ckpt_file = os.path.basename(path_to_ckpt_file)

            # --- sharding: decide if this checkpoint belongs to this job ---
            if job_index is not None and num_jobs is not None:
                if (global_idx % num_jobs) != job_index:
                    global_idx += 1
                    continue  # this ckpt handled by another job

            print(
                "\n"
                + "=" * 40
                + f" {ckpt_title} / {ckpt_folder} / {ckpt_file} "
                + "=" * 40
                + "\n"
            )

            config.run_name = f"{ckpt_title}_{ckpt_folder}_{ckpt_file}"

            # Extract the step from the filename
            step = re.findall(r"\d+", ckpt_file)
            assert len(step) == 1
            step = step[0]

            ckpt_model = models.load_model_from_config(ckpt_config, path_to_ckpt_file)

            model = models.CellTypeClassifier(
                trained_encoder=ckpt_model,
                lr=config.model.lr,
                n_steps=config.trainer.n_steps,
                path_to_celltypes=config.paths.path_to_celltypes,
            )

            # Assert model is freezed:
            for p in model.trained_encoder.parameters():
                assert not p.requires_grad

            # Instantiate the datamodule with the correct config
            datamodule = CensusDataModule(config)

            if finetune_type == "ridge":
                result_dict = finetune_ridge(
                    path_to_celltypes=config.paths.path_to_celltypes,
                    n_steps=config.trainer.n_steps,
                    model=ckpt_model,
                    datamodule=datamodule,
                    ckpt_title=ckpt_title,
                    ckpt_folder=ckpt_folder,
                    ckpt_file=ckpt_file,
                    step=step,
                )
            else:
                result_dict = finetune_lightning(
                    trainer_config=config,
                    model=model,
                    datamodule=datamodule,
                    ckpt_title=ckpt_title,
                    ckpt_folder=ckpt_folder,
                    ckpt_file=ckpt_file,
                    step=step,
                )

            # Append results using CSV mutex
            append_result_safely(path_to_results, result_dict)

            global_idx += 1

    # Only create the final plot in single-job mode.
    # In parallel mode, run a separate plotting step once all jobs are done.
    if job_index is None or num_jobs is None:
        final_plot(path_to_results, path_to_plot)
    else:
        print("[finetune] Parallel mode: skipping final_plot. Run once after all jobs complete.")
