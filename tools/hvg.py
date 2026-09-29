"""
Manually compute the set of HVG of a given census.

```
python hvg.py --config local_config.yml
```

ScalingConfig prevents non-existent path to be specified: before running this script, ensure you set:
  path_to_hvg: ""
  path_to_means: ""
  path_to_celltypes: ""
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import load_experiment_and_dataset_from_config
from core.config import ScalingConfig
import os

# def hyper params
N_HVG = 2_000
# NUM_GENES = 60_530
N_BINS = 20  # not the same bins as for binning
MIN_DISP = 0.5
MAX_DISP = np.inf
MIN_MEAN = 0.0125
MAX_MEAN = 3


def nth_highest(x, n: int) -> float:
    x = x[~np.isnan(x)]
    if n > x.size:
        msg = "`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions."
        warnings.warn(msg, UserWarning)
        n = x.size
    # interestingly, np.argpartition is slightly slower
    x[::-1].sort()
    return x[n - 1]


def postprocess_dispersions_seurat(
    disp_bin_stats: pd.DataFrame, mean_bin: pd.Series
) -> None:
    # retrieve those genes that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized disperion of 1
    one_gene_per_bin = disp_bin_stats["dev"].isnull()
    gen_indices = np.flatnonzero(one_gene_per_bin.loc[mean_bin])
    if len(gen_indices) == 0:
        return
    print(
        f"Gene indices {gen_indices} fell into a single bin: their "
        "normalized dispersion was set to 1.\n    "
        "Decreasing `n_bins` will likely avoid this effect."
    )
    disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[
        one_gene_per_bin, "avg"
    ]
    disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0


def get_disp_stats(df: pd.DataFrame) -> pd.DataFrame:
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]

    disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")
    postprocess_dispersions_seurat(disp_bin_stats, df["mean_bin"])

    return disp_bin_stats.loc[df["mean_bin"]].set_index(df.index)


def subset_genes(mean, dispersion_norm, n_top_genes, num_genes=61_888):
    """Get boolean mask of genes with normalized dispersion in bounds."""
    if n_top_genes > num_genes:
        print("`n_top_genes` > `adata.n_var`, returning all genes.")
        n_top_genes = num_genes
    disp_cut_off = nth_highest(dispersion_norm, n_top_genes)
    return np.nan_to_num(dispersion_norm, nan=-np.inf) >= disp_cut_off


def compute_hvgs(
    config_path,
    overwrite: bool = False,
    batch_size_override: int | None = None,
):
    print("=" * 60)
    print("HVGs PIPELINE")
    print("=" * 60)
    print("Loading config...")
    config = ScalingConfig(config_path)

    if os.path.exists(config.paths.path_to_hvg) and not overwrite:
        warnings.warn(f"Skipping {config.paths.path_to_hvg} as it already exists.")
        return

    output_dir = os.path.dirname(config.paths.path_to_hvg)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_genes = config.model.context_length
    assert num_genes <= N_HVG, (
        f"num_genes ({num_genes}) > N_HVG ({N_HVG}), might cause problems later."
    )

    print("Loading experiment and dataset...")
    experiment, experiment_ds = load_experiment_and_dataset_from_config(
        config,
        batch_size_override=batch_size_override,
        shuffle=False,
    )
    gene_names = experiment.ms["RNA"].var.read().concat().to_pandas()["feature_id"]
    gene_names.name = "ensembl_id"
    if batch_size_override is not None:
        print(f"Using overridden stats batch size: {batch_size_override}")

    print("Calculating gene means...")
    means = None
    observed_num_genes = None
    num_cells = 0
    for x, y in tqdm(experiment_ds, desc="Batch (means)"):
        if means is None:
            observed_num_genes = x.shape[1]
            means = np.zeros(shape=(observed_num_genes,), dtype=np.float64)
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
        means += x.sum(axis=0)
        num_cells += x.shape[0]

    if means is None:
        raise ValueError("No expression batches were returned; cannot compute HVGs.")
    means = means / num_cells

    print("Calculating gene variances...")
    vars = np.zeros(shape=(observed_num_genes,), dtype=np.float64)
    num_cells = 0
    for x, y in tqdm(experiment_ds, desc="Batch (vars)"):
        if x.shape[1] != observed_num_genes:
            raise ValueError(
                "Encountered inconsistent gene dimension across batches: "
                f"expected {observed_num_genes}, found {x.shape[1]}."
            )
        vars += np.square(x - means).sum(axis=0)
        num_cells += x.shape[0]
    vars = vars / (num_cells - 1)  # using unbiased estimator

    print("Preprocessing means and dispersions...")
    means[means == 0] = 1e-12  # set entries equal to zero to small value
    dispersions = vars / means
    # logarithmized mean as in Seurat
    dispersions[dispersions == 0] = np.nan
    dispersions = np.log(dispersions)
    means = np.log1p(means)

    print("Creating DataFrame...")
    df = pd.DataFrame(
        dict(zip(["means", "dispersions"], (means, dispersions))), index=gene_names
    )

    print("Binning means and calculating dispersion statistics...")
    df["mean_bin"] = pd.cut(df["means"], N_BINS)
    disp_stats = get_disp_stats(df)

    print("Normalizing dispersions and selecting highly variable genes...")
    df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
    df["highly_variable"] = subset_genes(
        mean=means,
        dispersion_norm=df["dispersions_norm"].to_numpy(),
        n_top_genes=N_HVG,
        num_genes=observed_num_genes,
    )

    print(f"Saving results to {config.paths.path_to_hvg}...")
    df.to_csv(config.paths.path_to_hvg)
    print("Done.")
