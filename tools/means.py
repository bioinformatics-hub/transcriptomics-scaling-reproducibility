"""
Script to compute:
    - mean expressions of genes
    - unique cell types
in a TileDB Experiment.
Saves the results to stats.csv (gene means) and cell_types.csv (unique cell types).

ScalingConfig prevents non-existent path to be specified: before running this script, ensure you set:
  path_to_hvg: ""
  path_to_means: ""
  path_to_celltypes: ""
  path_to_dataset_ids: ""
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from .utils import load_experiment_and_dataset_from_config
from core.config import ScalingConfig
import warnings

EPS = 1e-9


def compute_means(
    config_path: str,
    overwrite: bool = False,
    batch_size_override: int | None = None,
):
    print("=" * 60)
    print("MEANS & CELL TYPES & DATASET ID PIPELINE")
    print("=" * 60)
    # Parse path to config if provided
    config = ScalingConfig(config_path)

    # Skip if already exists
    if (
        os.path.exists(config.paths.path_to_means)
        and os.path.exists(config.paths.path_to_celltypes)
        and os.path.exists(config.paths.path_to_dataset_ids)
        and not overwrite
    ):
        warnings.warn(
            "Skipping means/celltypes/dataset_ids because all output files already exist."
        )
        return

    for path in [
        config.paths.path_to_means,
        config.paths.path_to_celltypes,
        config.paths.path_to_dataset_ids,
    ]:
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Open Soma ML Dataset
    experiment, experiment_ds = load_experiment_and_dataset_from_config(
        config,
        batch_size_override=batch_size_override,
        shuffle=False,
    )

    # Get gene names
    print("Getting gene ids...")
    gene_names = experiment.ms["RNA"].var.read().concat().to_pandas()["feature_id"]
    gene_names.name = "ensembl_id"

    # Get cell types
    print("Getting cell types...")
    cell_types = (
        experiment.obs.read(
            column_names=["cell_type"],
            value_filter=config.datamodule.obs_value_filter,
        )
        .concat()
        .to_pandas()
        .cell_type.unique()
        .tolist()
    )
    cell_types = pd.DataFrame({"cell_type": cell_types})
    cell_types.to_csv(config.paths.path_to_celltypes)
    print("Saved to", config.paths.path_to_celltypes)

    # Get dataset ids
    print("Getting dataset ids...")
    dataset_ids = (
        experiment.obs.read(
            column_names=["dataset_id"],
            value_filter=config.datamodule.obs_value_filter,
        )
        .concat()
        .to_pandas()
        .dataset_id.unique()
        .tolist()
    )
    dataset_ids = pd.DataFrame({"dataset_id": dataset_ids})
    dataset_ids.to_csv(config.paths.path_to_dataset_ids)
    print("Saved to", config.paths.path_to_dataset_ids)

    # Compute means
    print("Computing means...")
    if batch_size_override is not None:
        print(f"Using overridden stats batch size: {batch_size_override}")
    means = None
    non_zero_cells = None
    observed_num_genes = None
    for x, y in tqdm(experiment_ds, desc="Batch"):
        if means is None:
            observed_num_genes = x.shape[1]
            means = np.zeros(shape=(observed_num_genes,), dtype=np.float64)
            non_zero_cells = np.zeros_like(means)
            if len(gene_names) != observed_num_genes:
                warnings.warn(
                    "Gene metadata length does not match expression width "
                    f"({len(gene_names)} vs {observed_num_genes}). Truncating "
                    "gene ids to the observed expression width.",
                    UserWarning,
                )
                gene_names = gene_names.iloc[:observed_num_genes].reset_index(
                    drop=True
                )
        elif x.shape[1] != observed_num_genes:
            raise ValueError(
                "Encountered inconsistent gene dimension across batches: "
                f"expected {observed_num_genes}, found {x.shape[1]}."
            )
        x = x / x.sum(axis=1, keepdims=True)  # Normalize by total counts per cell
        means += x.sum(axis=0)
        non_zero_cells += (x != 0).sum(axis=0)

    if means is None:
        raise ValueError("No expression batches were returned; cannot compute means.")

    means = means / (non_zero_cells + EPS)

    df = pd.DataFrame(data={"means": means}, index=gene_names)
    output_dir = os.path.dirname(config.paths.path_to_means)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(config.paths.path_to_means)
    print("Saved to", config.paths.path_to_means)
