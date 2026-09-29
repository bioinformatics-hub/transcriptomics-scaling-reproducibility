# Clustering metrics used by run.downstream_tasks.

import os
import re
import glob
from datetime import datetime
from contextlib import nullcontext
from tqdm import tqdm
from dataclasses import dataclass
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
)

from core.config import ScalingConfig
from core.datamodule import CensusDataModule
import core.models as models

from run.finetune import (
    validate_ckpt_dir_structure,
)
from tools.hvg import subset_genes
from tools.utils import suppress_messages
from tools.ontology import (
    build_parents_from_cl_owl,
    build_anchor_mapper,
    encode_string_labels_to_int,
    map_with_refinement,
    build_level_set_cut,
    print_random_mappings,
    FORBIDDEN,
    MACRO_ANCHORS,
    NEURAL_SUB,
    IMMUNE_SUB,
)

torch.set_float32_matmul_precision("medium")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CheckpointInfo:
    run_title: str  # parent folder
    run_name: str  # model folder
    ckpt_path: str  # full path to .ckpt
    ckpt_config_path: str
    step: str  # extracted numeric step as string


def list_all_checkpoints(path_to_ckpt_dir: str) -> list[CheckpointInfo]:
    """
    Return a flat list of all checkpoints in path_to_ckpt_dir.
    """
    validate_ckpt_dir_structure(path_to_ckpt_dir)

    ckpt_infos: list[CheckpointInfo] = []

    # sort for reproducibility
    for run_title_dir in sorted(glob.glob(f"{path_to_ckpt_dir}/*")):
        if not os.path.isdir(run_title_dir):
            continue
        run_title = os.path.basename(run_title_dir)
        # IMPORTANT: this is to focus on checkpoints for a single model config
        # The check can be removed/made configurable
        if run_title != "config_1":
            continue

        for run_name_dir in sorted(glob.glob(f"{run_title_dir}/*")):
            if not os.path.isdir(run_name_dir):
                continue
            run_name = os.path.basename(run_name_dir)

            ckpt_config_path = os.path.join(run_name_dir, "config.yml")
            if not os.path.exists(ckpt_config_path):
                raise FileNotFoundError(
                    f"Expected config.yml in '{run_name_dir}', but none found."
                )

            # Sort the checkpoint files based on the step number
            ckpts_sorted = sorted(
                glob.glob(f"{run_name_dir}/*.ckpt"),
                key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]),
            )
            for ckpt_path in ckpts_sorted:
                ckpt_file = os.path.basename(ckpt_path)
                step_list = re.findall(r"\d+", ckpt_file)
                if len(step_list) != 1:
                    raise ValueError(
                        f"Expected exactly one numeric step in '{ckpt_file}', "
                        f"got: {step_list}"
                    )
                step = step_list[0]
                ckpt_infos.append(
                    CheckpointInfo(
                        run_title=run_title,
                        run_name=run_name,
                        ckpt_path=ckpt_path,
                        ckpt_config_path=ckpt_config_path,
                        step=step,
                    )
                )
    if not ckpt_infos:
        raise FileNotFoundError(f"No checkpoints found under '{path_to_ckpt_dir}'.")

    return ckpt_infos


def get_all_embeddings(
    model,
    dataloader: DataLoader,
    max_samples: int = -1,
    desc="",
    amp: bool = True,
):
    """
    Generates embeddings for an entire dataset using a given model.
    """
    model.eval()  # Model is set to device outside the function

    all_embeddings = []
    all_labels = []
    all_leaf_ids = []

    # Set up automatic mixed precision context if applicable
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (device == "cuda" and amp)
        else nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        for batch in tqdm(dataloader, desc=desc):
            expr, gene_ids, _, cell_type, _, leaf_id, *extras = batch
            position_ids = extras[0] if extras else None

            batch_size, _ = expr.shape

            expr = expr.to(device, non_blocking=True)
            gene_ids = gene_ids.to(device, non_blocking=True)
            cell_type = cell_type.to(device, non_blocking=True)
            if position_ids is not None:
                position_ids = position_ids.to(device, non_blocking=True)

            encoder_output = model.get_encoder_output(
                gene_ids, expr, position_ids=position_ids
            )
            cell_embeddings = encoder_output.mean(dim=1)

            all_embeddings.append(cell_embeddings.detach().cpu())
            all_labels.append(cell_type.detach().cpu())

            all_leaf_ids.extend([str(x) for x in leaf_id])

            if max_samples > 0 and len(all_embeddings) * batch_size >= max_samples:
                break

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0), all_leaf_ids


def get_all_raw_gene_space_vectors(
    dataloader: DataLoader,
    total_genes: int,
    max_samples: int = -1,
    desc="",
):
    """
    Builds fixed gene-space vectors directly from (gene_ids, expr) batches.
    """
    all_features = []
    all_labels = []
    all_leaf_ids = []
    observed_min_id = None
    observed_max_id = None
    invalid_count = 0
    pad_like_invalid_count = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=desc):
            expr, gene_ids, _, cell_type, _, leaf_id, *_ = batch
            assert (
                expr.shape == gene_ids.shape
            ), f"Shape mismatch between expr and gene_ids: {expr.shape} vs {gene_ids.shape}"
            batch_size, _ = expr.shape
            expr = expr.float().cpu()
            gene_ids = gene_ids.long().cpu()
            observed_min_id = (
                int(gene_ids.min())
                if observed_min_id is None
                else min(observed_min_id, int(gene_ids.min()))
            )
            observed_max_id = (
                int(gene_ids.max())
                if observed_max_id is None
                else max(observed_max_id, int(gene_ids.max()))
            )

            batch_features = torch.zeros((batch_size, total_genes), dtype=torch.float32)

            for i in range(batch_size):
                valid = (gene_ids[i] >= 0) & (gene_ids[i] < total_genes)
                invalid = ~valid
                invalid_count += int(invalid.sum())
                if invalid.any():
                    invalid_vals = gene_ids[i][invalid]
                    pad_like = (invalid_vals == total_genes) | (
                        invalid_vals == total_genes + 1
                    )
                    pad_like_invalid_count += int(pad_like.sum())
                    if not pad_like.all():
                        bad = invalid_vals[~pad_like]
                        raise ValueError(
                            "Found invalid gene_ids outside known PAD ids "
                            f"(expected {total_genes} or {total_genes + 1}): "
                            f"{torch.unique(bad).tolist()}"
                        )
                idx = gene_ids[i][valid]
                vals = expr[i][valid]
                # Sum values in case the same gene id appears multiple times.
                batch_features[i].index_add_(0, idx, vals)

            all_features.append(batch_features)
            all_labels.append(cell_type.detach().cpu())
            all_leaf_ids.extend([str(x) for x in leaf_id])

            if max_samples > 0 and len(all_features) * batch_size >= max_samples:
                break

    if observed_min_id is None or observed_max_id is None:
        raise ValueError("No samples found in dataloader.")
    print(
        "Gene-id sanity check:",
        {
            "observed_min_id": observed_min_id,
            "observed_max_id": observed_max_id,
            "total_genes": total_genes,
            "invalid_ids": invalid_count,
            "pad_like_invalid_ids": pad_like_invalid_count,
        },
    )

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), all_leaf_ids


def compute_clustering_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    n_neighbors: int = 15,
) -> dict:
    """
    Given cell embeddings and true labels, run clustering (Leiden/Louvain)
    on the embeddings and compute ARI, NMI, homogeneity.

    Parameters
    ----------
    embeddings : torch.Tensor
        Tensor of shape (N_cells, D) with cell embeddings.
    labels : torch.Tensor
        Tensor of shape (N_cells,) with integer cell-type labels.
    clustering_method : {"leiden", "louvain"}
        Clustering algorithm to use on the kNN graph.
    resolution : float
        Resolution parameter for Leiden/Louvain (controls number of clusters).
    n_neighbors : int
        Number of neighbors for kNN graph construction.

    Returns
    -------
    dict
        Dictionary containing ARI, NMI, homogeneity, and num_clusters.
    """
    # Convert to NumPy
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()

    # Build AnnData with embeddings as "expression matrix"
    adata = sc.AnnData(X=X)

    # kNN graph in embedding space
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=None,
        metric="euclidean",
    )

    # Clustering
    cluster_key = "clusters"
    if clustering_method == "leiden":
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
    elif clustering_method == "louvain":
        sc.tl.louvain(adata, resolution=resolution, key_added=cluster_key)
    else:
        raise ValueError(
            f"Unknown clustering_method='{clustering_method}'. "
            "Supported: 'leiden', 'louvain'."
        )

    cluster_labels = adata.obs[cluster_key].to_numpy()

    # Compute clustering quality metrics
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    hom = homogeneity_score(y, cluster_labels)
    num_clusters = len(np.unique(cluster_labels))
    num_cell_types = len(np.unique(y))

    return {
        "ari": ari,
        "nmi": nmi,
        "homogeneity": hom,
        "num_clusters": num_clusters,
        "num_cell_types": num_cell_types,
        "cluster_labels": cluster_labels,
    }


def compute_clustering_metrics_classic(
    features: torch.Tensor,
    labels: torch.Tensor,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    hvg_idx: np.ndarray | None = None,
) -> dict:
    """
    Classic Scanpy baseline:
    log1p -> HVG -> PCA -> neighbors -> Leiden/Louvain.
    """
    X = features.cpu().numpy()
    y = labels.cpu().numpy()

    adata = sc.AnnData(X=X)
    sc.pp.log1p(adata)

    if hvg_idx is not None:
        hvg_idx = np.asarray(hvg_idx, dtype=int)
        hvg_idx = hvg_idx[(hvg_idx >= 0) & (hvg_idx < adata.n_vars)]
        if hvg_idx.size == 0:
            raise ValueError("Provided hvg_idx is empty after bounds filtering.")
        adata = adata[:, hvg_idx].copy()
    else:
        n_hvg = min(n_hvg, adata.n_vars)
        if n_hvg > 0 and adata.n_vars > 1:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
            if "highly_variable" in adata.var and adata.var["highly_variable"].any():
                adata = adata[:, adata.var["highly_variable"]].copy()

    use_rep = None
    if adata.n_vars > 2 and adata.n_obs > 2:
        n_pcs_eff = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
        if n_pcs_eff >= 2:
            sc.tl.pca(adata, n_comps=n_pcs_eff, svd_solver="arpack")
            use_rep = "X_pca"

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        metric="euclidean",
    )

    cluster_key = "clusters"
    if clustering_method == "leiden":
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
    elif clustering_method == "louvain":
        sc.tl.louvain(adata, resolution=resolution, key_added=cluster_key)
    else:
        raise ValueError(
            f"Unknown clustering_method='{clustering_method}'. "
            "Supported: 'leiden', 'louvain'."
        )

    cluster_labels = adata.obs[cluster_key].to_numpy()
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    hom = homogeneity_score(y, cluster_labels)
    num_clusters = len(np.unique(cluster_labels))
    num_cell_types = len(np.unique(y))

    return {
        "ari": ari,
        "nmi": nmi,
        "homogeneity": hom,
        "num_clusters": num_clusters,
        "num_cell_types": num_cell_types,
        "cluster_labels": cluster_labels,
    }


def raw_binned_cluster(
    path_to_config: str,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    n_neighbors: int = 15,
):
    """
    Clustering baseline on raw normalized-binned expression in fixed gene space.
    Does not load any model/checkpoint.
    """
    suppress_messages()

    config = ScalingConfig(path_to_config)
    datamodule = CensusDataModule(config)
    datamodule.setup()
    assert len(next(iter(datamodule.test_dataloader()))) == 5

    print("\n" + "=" * 40 + " RAW-BINNED CLUSTERING (NO MODEL) " + "=" * 40 + "\n")

    X_test, y_test, leaf_ids = get_all_raw_gene_space_vectors(
        dataloader=datamodule.test_dataloader(),
        total_genes=config.model.total_genes,
        desc="Building raw fixed gene-space vectors",
    )
    assert X_test.shape[0] == y_test.shape[0] == len(leaf_ids)

    # Build ontology parents once
    parents, cl_labels = build_parents_from_cl_owl(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl-basic.owl")
    )
    parents = {k: tuple(sorted(v)) for k, v in parents.items()}

    unique_leaf = sorted(set(leaf_ids))
    macro_mapper = build_anchor_mapper(unique_leaf, parents, MACRO_ANCHORS)

    coarse_ids = [
        map_with_refinement(l, parents, macro_mapper, NEURAL_SUB, IMMUNE_SUB)
        for l in leaf_ids
    ]

    print("\n### Coarse label assignment sanity check ###")
    print("Unique coarse labels:", len(set(coarse_ids)))
    print("Cells per anchor (median):", np.median(list(Counter(coarse_ids).values())))

    print_random_mappings(
        leaf_ids=leaf_ids,
        coarse_ids=coarse_ids,
        cl_labels=cl_labels,
        n=30,
    )

    coarse_counts = Counter(coarse_ids)
    for cl_id, n in coarse_counts.most_common():
        name = cl_labels.get(cl_id, cl_id)
        print(f"{cl_id:12s} {n:6d}  {name}")

    y_test_coarse, coarse_vocab = encode_string_labels_to_int(coarse_ids)

    mask = ~np.isin(coarse_ids, ["other", "unknown"])
    metrics = compute_clustering_metrics(
        embeddings=X_test[mask],
        labels=y_test_coarse[mask],
        clustering_method=clustering_method,
        resolution=resolution,
        n_neighbors=n_neighbors,
    )
    print("Coarse metrics:", metrics)

    cluster_labels = metrics["cluster_labels"]

    ct = pd.crosstab(cluster_labels, np.array(coarse_ids)[mask])
    ct_frac = ct.div(ct.sum(axis=1), axis=0)
    cluster_annotation = ct.idxmax(axis=1)
    print("\nCluster × Coarse label contingency (head):")
    print(ct.iloc[:10, :10])

    df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "coarse_id": np.array(coarse_ids)[mask],
        }
    )

    df_cells = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "coarse_id": np.array(coarse_ids)[mask],
            "leaf_id": np.array(leaf_ids)[mask],
        }
    )
    ct_fine = pd.crosstab(df_cells["cluster"], df_cells["leaf_id"])
    ct_fine_frac = ct_fine.div(ct_fine.sum(axis=1), axis=0)
    topk = (
        ct_fine_frac.stack()
        .reset_index(name="frac")
        .sort_values(["cluster", "frac"], ascending=[True, False])
        .groupby("cluster")
        .head(10)
    )
    print(topk)
    topk["leaf_name"] = topk["leaf_id"].map(lambda x: cl_labels.get(x, x))

    cluster_counts = (
        df.groupby(["cluster", "coarse_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )

    result_dict = {
        "run_title": "raw_binned_baseline",
        "model_folder": "none",
        "checkpoint": "none",
        "step": "none",
        "clustering_type": "raw_binned",
        "timestamp": str(datetime.now()),
        "clustering_method": clustering_method,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
    }
    result_dict.update(metrics)

    cluster_log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cluster_logs"
    )
    os.makedirs(cluster_log_dir, exist_ok=True)

    path_to_results = os.path.join(
        cluster_log_dir, f"cluster_results_raw_binned_{clustering_method}.csv"
    )

    if os.path.exists(path_to_results):
        df = pd.read_csv(path_to_results)
        df = pd.concat(
            [df, pd.DataFrame([result_dict])],
            ignore_index=True,
        )
    else:
        df = pd.DataFrame([result_dict])

    df.to_csv(path_to_results, index=False)
    cluster_counts.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_coarse_composition_raw_binned_{clustering_method}.csv",
        ),
        index=False,
    )

    ct.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_coarse_labels_raw_binned_{clustering_method}.csv",
        )
    )

    ct_frac.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_coarse_labels_fraction_raw_binned_{clustering_method}.csv",
        )
    )

    cluster_annotation.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_dominant_coarse_label_raw_binned_{clustering_method}.csv",
        )
    )

    ct_fine.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_leaf_labels_raw_binned_{clustering_method}.csv",
        )
    )
    ct_fine_frac.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_leaf_labels_fraction_raw_binned_{clustering_method}.csv",
        )
    )
    topk.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_top10_leaf_labels_fraction_raw_binned_{clustering_method}.csv",
        ),
        index=False,
    )
    df_cells.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cell_assignments_raw_binned_{clustering_method}.csv",
        ),
        index=False,
    )


def classic_scanpy_cluster(
    path_to_config: str,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    use_config_hvg: bool = False,
):
    """
    Classic baseline: log1p + HVG + PCA + neighbors + Leiden/Louvain.
    """
    suppress_messages()

    config = ScalingConfig(path_to_config)
    datamodule = CensusDataModule(config)
    datamodule.setup()
    assert len(next(iter(datamodule.test_dataloader()))) == 5

    print("\n" + "=" * 40 + " CLASSIC SCANPY CLUSTERING (NO MODEL) " + "=" * 40 + "\n")

    X_test, y_test, leaf_ids = get_all_raw_gene_space_vectors(
        dataloader=datamodule.test_dataloader(),
        total_genes=config.model.total_genes,
        desc="Building fixed gene-space vectors for classic baseline",
    )
    assert X_test.shape[0] == y_test.shape[0] == len(leaf_ids)
    hvg_idx = None
    if use_config_hvg:
        hvg_df = pd.read_csv(config.paths.path_to_hvg)
        hvg_df["highly_variable"] = subset_genes(
            mean=hvg_df["means"],
            dispersion_norm=hvg_df["dispersions_norm"].to_numpy(),
            n_top_genes=config.model.context_length,
        )
        hvg_idx = hvg_df[hvg_df["highly_variable"]].index.to_numpy(dtype=int)
        print(
            f"Using HVGs from config path ({config.paths.path_to_hvg}), "
            f"selected {len(hvg_idx)} genes."
        )
    else:
        print(f"Using Scanpy HVGs computed on-the-fly (n_hvg={n_hvg}).")

    parents, cl_labels = build_parents_from_cl_owl(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl-basic.owl")
    )
    parents = {k: tuple(sorted(v)) for k, v in parents.items()}

    unique_leaf = sorted(set(leaf_ids))
    macro_mapper = build_anchor_mapper(unique_leaf, parents, MACRO_ANCHORS)
    coarse_ids = [
        map_with_refinement(l, parents, macro_mapper, NEURAL_SUB, IMMUNE_SUB)
        for l in leaf_ids
    ]

    print("\n### Coarse label assignment sanity check ###")
    print("Unique coarse labels:", len(set(coarse_ids)))
    print("Cells per anchor (median):", np.median(list(Counter(coarse_ids).values())))

    print_random_mappings(
        leaf_ids=leaf_ids,
        coarse_ids=coarse_ids,
        cl_labels=cl_labels,
        n=30,
    )

    coarse_counts = Counter(coarse_ids)
    for cl_id, n in coarse_counts.most_common():
        name = cl_labels.get(cl_id, cl_id)
        print(f"{cl_id:12s} {n:6d}  {name}")

    y_test_coarse, coarse_vocab = encode_string_labels_to_int(coarse_ids)
    mask = ~np.isin(coarse_ids, ["other", "unknown"])
    metrics = compute_clustering_metrics_classic(
        features=X_test[mask],
        labels=y_test_coarse[mask],
        clustering_method=clustering_method,
        resolution=resolution,
        n_neighbors=n_neighbors,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        hvg_idx=hvg_idx,
    )
    print("Coarse metrics:", metrics)

    cluster_labels = metrics["cluster_labels"]
    ct = pd.crosstab(cluster_labels, np.array(coarse_ids)[mask])
    ct_frac = ct.div(ct.sum(axis=1), axis=0)
    cluster_annotation = ct.idxmax(axis=1)
    print("\nCluster × Coarse label contingency (head):")
    print(ct.iloc[:10, :10])

    df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "coarse_id": np.array(coarse_ids)[mask],
        }
    )
    df_cells = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "coarse_id": np.array(coarse_ids)[mask],
            "leaf_id": np.array(leaf_ids)[mask],
        }
    )
    ct_fine = pd.crosstab(df_cells["cluster"], df_cells["leaf_id"])
    ct_fine_frac = ct_fine.div(ct_fine.sum(axis=1), axis=0)
    topk = (
        ct_fine_frac.stack()
        .reset_index(name="frac")
        .sort_values(["cluster", "frac"], ascending=[True, False])
        .groupby("cluster")
        .head(10)
    )
    print(topk)
    topk["leaf_name"] = topk["leaf_id"].map(lambda x: cl_labels.get(x, x))

    cluster_counts = (
        df.groupby(["cluster", "coarse_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )

    result_dict = {
        "run_title": "classic_scanpy_baseline",
        "model_folder": "none",
        "checkpoint": "none",
        "step": "none",
        "clustering_type": "classic_scanpy",
        "timestamp": str(datetime.now()),
        "clustering_method": clustering_method,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_hvg": n_hvg,
        "n_pcs": n_pcs,
        "use_config_hvg": use_config_hvg,
    }
    result_dict.update(metrics)

    cluster_log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cluster_logs"
    )
    os.makedirs(cluster_log_dir, exist_ok=True)

    path_to_results = os.path.join(
        cluster_log_dir, f"cluster_results_classic_scanpy_{clustering_method}.csv"
    )
    if os.path.exists(path_to_results):
        df = pd.read_csv(path_to_results)
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([result_dict])
    df.to_csv(path_to_results, index=False)

    cluster_counts.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_coarse_composition_classic_scanpy_{clustering_method}.csv",
        ),
        index=False,
    )
    ct.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_coarse_labels_classic_scanpy_{clustering_method}.csv",
        )
    )
    ct_frac.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_coarse_labels_fraction_classic_scanpy_{clustering_method}.csv",
        )
    )
    cluster_annotation.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_dominant_coarse_label_classic_scanpy_{clustering_method}.csv",
        )
    )
    ct_fine.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_leaf_labels_classic_scanpy_{clustering_method}.csv",
        )
    )
    ct_fine_frac.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_vs_leaf_labels_fraction_classic_scanpy_{clustering_method}.csv",
        )
    )
    topk.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cluster_top10_leaf_labels_fraction_classic_scanpy_{clustering_method}.csv",
        ),
        index=False,
    )
    df_cells.to_csv(
        os.path.join(
            cluster_log_dir,
            f"cell_assignments_classic_scanpy_{clustering_method}.csv",
        ),
        index=False,
    )


def zero_shot_cluster(
    path_to_config: str,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    n_neighbors: int = 15,
    job_id: int | None = None,
    ckpts_per_job: int | None = None,
):
    """
    Zero-shot clustering baseline pipeline.
    For each checkpoint in the ckpt directory:
        1/ load the pretrained encoder
        2/ compute cell embeddings for test loader
        3/ run clustering (Leiden/Louvain) on embeddings
        4/ compute ARI, NMI, homogeneity vs true cell-type labels
        5/ save results to file

    Parameters
    ----------
    path_to_config : str
        Path to a YAML config file (same style as used for finetune/train).
    clustering_method : {"leiden", "louvain"}
        Clustering algorithm to use on the embedding kNN graph.
    resolution : float
        Resolution parameter for Leiden/Louvain clustering.
    n_neighbors : int
        Number of neighbors for kNN graph construction.
    """
    suppress_messages()

    # Load base config
    config = ScalingConfig(path_to_config)

    # Basic checks on ckpt directory
    assert os.path.exists(config.paths.path_to_ckpt_dir), (
        f"Checkpoint directory does not exist: " f"{config.paths.path_to_ckpt_dir}"
    )
    assert len(os.listdir(config.paths.path_to_ckpt_dir)) > 0, (
        f"Checkpoint directory is empty: " f"{config.paths.path_to_ckpt_dir}"
    )

    # Ensure directory structure is as expected
    validate_ckpt_dir_structure(config.paths.path_to_ckpt_dir)
    # Build a flat, ordered list of all checkpoints
    ckpt_infos = list_all_checkpoints(config.paths.path_to_ckpt_dir)
    # Instantiate datamodule
    datamodule = CensusDataModule(config)
    datamodule.setup()
    assert len(next(iter(datamodule.test_dataloader()))) == 5

    # If job_id / ckpts_per_job are provided, select subset for this job
    if job_id is not None and ckpts_per_job is not None:
        start_idx = job_id * ckpts_per_job
        end_idx = min(start_idx + ckpts_per_job, len(ckpt_infos))
        if start_idx >= len(ckpt_infos):
            return
        ckpt_infos = ckpt_infos[start_idx:end_idx]
        print(
            f"[Job {job_id}] processing checkpoints {start_idx}..{end_idx-1} "
            f"out of {len(list_all_checkpoints(config.paths.path_to_ckpt_dir))}"
        )
    else:
        # Single-job behavior: process all checkpoints
        print(
            f"Processing all {len(ckpt_infos)} checkpoints in one job "
            "(no job_id/ckpts_per_job slicing)."
        )

    # For all config files
    for info in ckpt_infos:
        ckpt_folder = info.run_name
        ckpt_title = info.run_title
        path_to_ckpt_file = info.ckpt_path
        ckpt_file = os.path.basename(path_to_ckpt_file)
        step = info.step

        # Process only one every five checkpoints since too many (15 mins per checkpoint)
        if int(step) % 500 != 0:
            continue

        ckpt_config = ScalingConfig(info.ckpt_config_path)
        # Ensure compatible pipeline
        assert config.metadata.pipeline == ckpt_config.metadata.pipeline, (
            "Mismatch between pipeline in base config and checkpoint config:\n"
            f"base: {config.metadata.pipeline}\n"
            f"ckpt: {ckpt_config.metadata.pipeline}"
        )

        # Copy over datamodule settings that must match pretraining
        config.datamodule.normalize_expr_for_ranking = (
            ckpt_config.datamodule.normalize_expr_for_ranking
        )
        config.datamodule.n_bins = ckpt_config.datamodule.n_bins

        print(
            "\n"
            + "=" * 40
            + f" ZERO-SHOT CLUSTERING: {ckpt_title} / {ckpt_folder} / {ckpt_file} "
            + "=" * 40
            + "\n"
        )

        # Load pretrained model (encoder)
        ckpt_model = models.load_model_from_config(
            ckpt_config,
            path_to_ckpt_file,
        )
        ckpt_model.to(device)
        ckpt_model.eval()

        print("Loaded ckpt config d_model:", ckpt_config.model.d_model)
        print("Loaded ckpt config context_length:", ckpt_config.model.context_length)

        print("Model embedder dim:", ckpt_model.embedder.embedding_dim)
        print("Positional pe shape:", tuple(ckpt_model.positional.pe.shape))

        # Get embeddings for test set
        X_test, y_test, leaf_ids = get_all_embeddings(
            model=ckpt_model,
            dataloader=datamodule.test_dataloader(),
            desc="Generating test embeddings for clustering",
        )
        assert X_test.shape[0] == y_test.shape[0] == len(leaf_ids)

        # 2) Build ontology parents once
        parents, cl_labels = build_parents_from_cl_owl(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl-basic.owl")
        )
        parents = {k: tuple(sorted(v)) for k, v in parents.items()}

        unique_leaf = sorted(set(leaf_ids))
        macro_mapper = build_anchor_mapper(unique_leaf, parents, MACRO_ANCHORS)

        coarse_ids = [
            map_with_refinement(l, parents, macro_mapper, NEURAL_SUB, IMMUNE_SUB)
            for l in leaf_ids
        ]

        ### UNCOMMENT TO RUN DYNAMIC DEPTH ONTOLOGY MAPPING
        # coarse_ids, anchors, propagated, depth = build_level_set_cut(
        #     leaf_ids=leaf_ids,          # per-cell leaf ids from get_all_embeddings
        #     parents=parents,
        #     # target_depth=3,
        #     min_depth=3,
        #     min_cells=300,
        #     forbidden=FORBIDDEN,
        #     other_label="other",
        # )

        print("\n### Coarse label assignment sanity check ###")
        # How many unique coarse labels
        print("Unique coarse labels:", len(set(coarse_ids)))
        # print("Unique coarse labels:", len(anchors))
        print(
            "Cells per anchor (median):", np.median(list(Counter(coarse_ids).values()))
        )

        print_random_mappings(
            leaf_ids=leaf_ids,
            coarse_ids=coarse_ids,
            cl_labels=cl_labels,
            n=30,
        )

        # Distribution
        coarse_counts = Counter(coarse_ids)
        for cl_id, n in coarse_counts.most_common():
            name = cl_labels.get(cl_id, cl_id)
            print(f"{cl_id:12s} {n:6d}  {name}")

        y_test_coarse, coarse_vocab = encode_string_labels_to_int(coarse_ids)

        mask = ~np.isin(coarse_ids, ["other", "unknown"])
        metrics = compute_clustering_metrics(
            embeddings=X_test[mask],
            labels=y_test_coarse[mask],
            clustering_method=clustering_method,
            resolution=resolution,
            n_neighbors=n_neighbors,
        )
        print("Coarse metrics:", metrics)

        cluster_labels = metrics["cluster_labels"]

        ct = pd.crosstab(cluster_labels, np.array(coarse_ids)[mask])
        ct_frac = ct.div(ct.sum(axis=1), axis=0)
        cluster_annotation = ct.idxmax(axis=1)
        print("\nCluster × Coarse label contingency (head):")
        print(ct.iloc[:10, :10])

        df = pd.DataFrame(
            {
                "cluster": cluster_labels,
                "coarse_id": np.array(coarse_ids)[mask],
            }
        )

        df_cells = pd.DataFrame(
            {
                "cluster": cluster_labels,
                "coarse_id": np.array(coarse_ids)[mask],
                "leaf_id": np.array(leaf_ids)[mask],  # fine-grained label
            }
        )
        ct_fine = pd.crosstab(df_cells["cluster"], df_cells["leaf_id"])
        ct_fine_frac = ct_fine.div(
            ct_fine.sum(axis=1), axis=0
        )  # row-normalize = within cluster
        topk = (
            ct_fine_frac.stack()
            .reset_index(name="frac")
            .sort_values(["cluster", "frac"], ascending=[True, False])
            .groupby("cluster")
            .head(10)
        )
        print(topk)
        topk["leaf_name"] = topk["leaf_id"].map(lambda x: cl_labels.get(x, x))

        cluster_counts = (
            df.groupby(["cluster", "coarse_id"])
            .size()
            .reset_index(name="count")
            .sort_values(["cluster", "count"], ascending=[True, False])
        )

        # Prepare result row
        result_dict = {
            "run_title": ckpt_title,
            "model_folder": ckpt_folder,
            "checkpoint": ckpt_file,
            "step": step,
            "clustering_type": "zero_shot",
            "timestamp": str(datetime.now()),
            "clustering_method": clustering_method,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
        }
        result_dict.update(metrics)

        cluster_log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cluster_logs"
        )
        os.makedirs(cluster_log_dir, exist_ok=True)

        # Determine per-job output path
        if job_id is None:
            path_to_results = os.path.join(
                cluster_log_dir, f"cluster_results_{clustering_method}.csv"
            )
        else:
            # create one CSV per SLURM job, and then merge them in post processing script
            path_to_results = os.path.join(
                cluster_log_dir, f"cluster_results_job_{job_id}_{clustering_method}.csv"
            )

        # Append to CSV (create if needed)
        if os.path.exists(path_to_results):
            df = pd.read_csv(path_to_results)
            df = pd.concat(
                [df, pd.DataFrame([result_dict])],
                ignore_index=True,
            )
        else:
            df = pd.DataFrame([result_dict])

        df.to_csv(path_to_results, index=False)
        cluster_counts.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_coarse_composition_{job_id}_{step}_{clustering_method}.csv",
            ),
            index=False,
        )

        ct.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_vs_coarse_labels_{job_id}_{step}_{clustering_method}.csv",
            )
        )

        ct_frac.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_vs_coarse_labels_fraction_{job_id}_{step}_{clustering_method}.csv",
            )
        )

        cluster_annotation.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_dominant_coarse_label_{job_id}_{step}_{clustering_method}.csv",
            )
        )

        ct_fine.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_vs_leaf_labels_{job_id}_{step}_{clustering_method}.csv",
            )
        )
        ct_fine_frac.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_vs_leaf_labels_fraction_{job_id}_{step}_{clustering_method}.csv",
            )
        )
        topk.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cluster_top10_leaf_labels_fraction_{job_id}_{step}_{clustering_method}.csv",
            ),
            index=False,
        )
        df_cells.to_csv(
            os.path.join(
                cluster_log_dir,
                f"cell_assignments_{job_id}_{step}_{clustering_method}.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot clustering baseline using pretrained encoder embeddings."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="leiden",
        choices=["leiden", "louvain"],
        help="Clustering algorithm to use.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Leiden/Louvain.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for kNN graph.",
    )

    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Job ID for the finetuning task (default: None).",
    )
    parser.add_argument(
        "--ckpts-per-job",
        type=int,
        default=None,
        help="Number of checkpoints to process per job (default: None).",
    )
    parser.add_argument(
        "--feature-space",
        type=str,
        default="embedding",
        choices=["embedding", "raw_binned", "classic_scanpy"],
        help="Input space for clustering.",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Number of HVGs for classic_scanpy baseline.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of PCs for classic_scanpy baseline.",
    )
    parser.add_argument(
        "--classic-use-config-hvg",
        action="store_true",
        help="Use HVGs from config.paths.path_to_hvg (training-consistent) instead of recomputing HVGs in Scanpy.",
    )

    args = parser.parse_args()

    if args.feature_space == "embedding":
        zero_shot_cluster(
            path_to_config=args.config,
            clustering_method=args.clustering_method,
            resolution=args.resolution,
            n_neighbors=args.n_neighbors,
            job_id=args.job_id,
            ckpts_per_job=args.ckpts_per_job,
        )
    elif args.feature_space == "raw_binned":
        raw_binned_cluster(
            path_to_config=args.config,
            clustering_method=args.clustering_method,
            resolution=args.resolution,
            n_neighbors=args.n_neighbors,
        )
    else:
        classic_scanpy_cluster(
            path_to_config=args.config,
            clustering_method=args.clustering_method,
            resolution=args.resolution,
            n_neighbors=args.n_neighbors,
            n_hvg=args.n_hvg,
            n_pcs=args.n_pcs,
            use_config_hvg=args.classic_use_config_hvg,
        )
