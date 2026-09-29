"""Consume saved embedding artifacts from training and compute downstream metrics."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scib
import torch
from sklearn.model_selection import train_test_split

from core.config import ScalingConfig
from run.cluster import compute_clustering_metrics
from run.finetune import append_result_safely
from tools.coarse_labels import DEFAULT_K, save_cell_type_to_coarse_map
from tools.ontology import encode_string_labels_to_int


def _artifact_step(path: Path, payload: dict) -> int:
    if "global_step" in payload:
        return int(payload["global_step"])
    stem = path.stem
    maybe_step = stem.rsplit("_", 1)[-1]
    return int(maybe_step)


def _artifact_step_from_name(path: Path) -> int | None:
    maybe_step = path.stem.rsplit("_", 1)[-1]
    try:
        return int(maybe_step)
    except (TypeError, ValueError):
        return None


def _is_persistent_artifact(path: Path) -> bool:
    return path.name.startswith("checkpoint_")


def _latest_artifact_path(run_dir: Path) -> Path | None:
    artifacts = [
        p
        for p in run_dir.glob("*.pt")
        if not p.name.endswith(".tmp") and not _is_persistent_artifact(p)
    ]
    if not artifacts:
        return None

    with_step = [(p, _artifact_step_from_name(p)) for p in artifacts]
    valid_step = [(p, s) for p, s in with_step if s is not None]
    if valid_step:
        return max(valid_step, key=lambda item: item[1])[0]

    return max(artifacts, key=lambda p: p.stat().st_mtime_ns)


def _load_coarse_mapping(
    config_path: str,
    mapping_dir: Path,
    coarse_k: int,
) -> dict[str, str]:
    mapping_path = mapping_dir / f"cell_type_to_coarse_k{coarse_k}.csv"
    if not mapping_path.exists():
        save_cell_type_to_coarse_map(
            config_path=config_path,
            output_csv=str(mapping_path),
            k=coarse_k,
            save_visualizations=False,
        )
    mapping_df = pd.read_csv(mapping_path)
    return dict(zip(mapping_df["leaf_id"], mapping_df["coarse_id"]))


def _safe_float(value):
    if value is None:
        return np.nan
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(value)


def _safe_int(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        return int(value)
    except Exception:
        return np.nan


def _compute_batch_correction_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    dataset_ids: torch.Tensor | None,
) -> dict[str, float]:
    if dataset_ids is None:
        return {
            "batch_nmi": np.nan,
            "batch_ari": np.nan,
            "batch_asw_label": np.nan,
            "batch_asw_batch": np.nan,
            "batch_graph_conn": np.nan,
        }

    obs = pd.DataFrame(
        {
            "celltype": labels.cpu().numpy().astype(str),
            "batch": dataset_ids.cpu().numpy().astype(str),
        }
    ).astype("category")

    X = embeddings.cpu().numpy()
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsm["X_emb"] = X

    try:
        metrics = scib.metrics.metrics(
            adata,
            adata,
            batch_key="batch",
            label_key="celltype",
            embed="X_emb",
            ari_=True,
            nmi_=True,
            silhouette_=True,
            isolated_labels_=False,
            isolated_labels_asw_=False,
            graph_conn_=True,
        )
    except Exception:
        return {
            "batch_nmi": np.nan,
            "batch_ari": np.nan,
            "batch_asw_label": np.nan,
            "batch_asw_batch": np.nan,
            "batch_graph_conn": np.nan,
        }

    return {
        "batch_nmi": _safe_float(metrics.loc["NMI_cluster/label"]),
        "batch_ari": _safe_float(metrics.loc["ARI_cluster/label"]),
        "batch_asw_label": _safe_float(metrics.loc["ASW_label"]),
        "batch_asw_batch": _safe_float(metrics.loc["ASW_label/batch"]),
        "batch_graph_conn": _safe_float(metrics.loc["graph_conn"]),
    }


def _solve_ridge(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=num_classes).float()
    eye = torch.eye(X_train.shape[1], device=X_train.device, dtype=X_train.dtype)
    A = X_train.T @ X_train + alpha * eye
    B = X_train.T @ y_train_one_hot
    weights = torch.linalg.lstsq(A, B).solution
    return X_eval @ weights


def _ridge_accuracy(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> float:
    classes = torch.unique(y_train)
    if classes.numel() < 2:
        return np.nan
    logits = _solve_ridge(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        num_classes=int(classes.max().item()) + 1,
    )
    predictions = torch.argmax(logits, dim=1).cpu()
    return float((predictions == y_eval.cpu()).float().mean().item())


def _split_indices(y: torch.Tensor, seed: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    idx = np.arange(len(y))
    if len(idx) < 2:
        return None, None
    try:
        train_idx, val_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=seed,
            stratify=y.cpu().numpy(),
        )
    except ValueError:
        return None, None
    return train_idx, val_idx


def _compute_ridge_metrics(
    embeddings: torch.Tensor,
    fine_labels: torch.Tensor,
    coarse_labels: torch.Tensor | None,
    seed: int,
) -> dict[str, float]:
    metrics = {
        "ridge_train_accuracy_fine": np.nan,
        "ridge_val_accuracy_fine": np.nan,
        "ridge_train_accuracy_coarse": np.nan,
        "ridge_val_accuracy_coarse": np.nan,
    }

    X = embeddings.float()
    y = fine_labels.long()

    train_idx, val_idx = _split_indices(y, seed)
    if train_idx is not None:
        metrics["ridge_train_accuracy_fine"] = _ridge_accuracy(
            X[train_idx],
            y[train_idx],
            X[train_idx],
            y[train_idx],
        )
        metrics["ridge_val_accuracy_fine"] = _ridge_accuracy(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
        )

    if coarse_labels is not None:
        valid_mask = coarse_labels >= 0
        if int(valid_mask.sum()) > 0:
            Xc = X[valid_mask]
            yc = coarse_labels[valid_mask]
            train_idx, val_idx = _split_indices(yc, seed)
            if train_idx is not None:
                metrics["ridge_train_accuracy_coarse"] = _ridge_accuracy(
                    Xc[train_idx],
                    yc[train_idx],
                    Xc[train_idx],
                    yc[train_idx],
                )
                metrics["ridge_val_accuracy_coarse"] = _ridge_accuracy(
                    Xc[train_idx],
                    yc[train_idx],
                    Xc[val_idx],
                    yc[val_idx],
                )

    return metrics


def _compute_clustering_bundle(
    embeddings: torch.Tensor,
    fine_labels: torch.Tensor,
    coarse_labels: torch.Tensor | None,
    clustering_method: str,
    resolution: float,
    n_neighbors: int,
) -> dict[str, float]:
    try:
        fine = compute_clustering_metrics(
            embeddings=embeddings,
            labels=fine_labels,
            clustering_method=clustering_method,
            resolution=resolution,
            n_neighbors=n_neighbors,
        )
    except Exception:
        fine = {
            "ari": np.nan,
            "nmi": np.nan,
            "homogeneity": np.nan,
            "num_clusters": np.nan,
        }
    result = {
        "cluster_fine_ari": _safe_float(fine["ari"]),
        "cluster_fine_nmi": _safe_float(fine["nmi"]),
        "cluster_fine_homogeneity": _safe_float(fine["homogeneity"]),
        "cluster_fine_num_clusters": _safe_int(fine["num_clusters"]),
    }

    if coarse_labels is None:
        result.update(
            {
                "cluster_coarse_ari": np.nan,
                "cluster_coarse_nmi": np.nan,
                "cluster_coarse_homogeneity": np.nan,
                "cluster_coarse_num_clusters": np.nan,
            }
        )
        return result

    coarse_mask = coarse_labels >= 0
    if int(coarse_mask.sum()) == 0:
        result.update(
            {
                "cluster_coarse_ari": np.nan,
                "cluster_coarse_nmi": np.nan,
                "cluster_coarse_homogeneity": np.nan,
                "cluster_coarse_num_clusters": np.nan,
            }
        )
        return result

    try:
        coarse = compute_clustering_metrics(
            embeddings=embeddings[coarse_mask],
            labels=coarse_labels[coarse_mask],
            clustering_method=clustering_method,
            resolution=resolution,
            n_neighbors=n_neighbors,
        )
    except Exception:
        coarse = {
            "ari": np.nan,
            "nmi": np.nan,
            "homogeneity": np.nan,
            "num_clusters": np.nan,
        }
    result.update(
        {
            "cluster_coarse_ari": _safe_float(coarse["ari"]),
            "cluster_coarse_nmi": _safe_float(coarse["nmi"]),
            "cluster_coarse_homogeneity": _safe_float(coarse["homogeneity"]),
            "cluster_coarse_num_clusters": _safe_int(coarse["num_clusters"]),
        }
    )
    return result


def _encode_coarse_labels(leaf_ids: list[str], coarse_map: dict[str, str]) -> tuple[torch.Tensor, list[str]]:
    coarse_ids = [coarse_map.get(str(leaf_id), "unknown") for leaf_id in leaf_ids]
    valid_mask = ~np.isin(coarse_ids, ["other", "unknown"])
    coarse_encoded = torch.full((len(coarse_ids),), -1, dtype=torch.long)
    if valid_mask.any():
        encoded_valid, _ = encode_string_labels_to_int(
            [coarse_ids[i] for i, keep in enumerate(valid_mask) if keep]
        )
        coarse_encoded[torch.from_numpy(valid_mask)] = encoded_valid
    return coarse_encoded, coarse_ids


def _artifact_metrics_row(
    artifact_path: Path,
    artifact: dict,
    run_title: str,
    run_name: str,
    split: str,
    coarse_map: dict[str, str],
    clustering_method: str,
    resolution: float,
    n_neighbors: int,
    ridge_seed: int,
) -> dict:
    embeddings = artifact["embeddings"].float()
    fine_labels = artifact["labels"].long()
    dataset_ids = artifact.get("dataset_ids")
    leaf_ids = [str(x) for x in artifact["leaf_ids"]]
    coarse_labels, _ = _encode_coarse_labels(leaf_ids, coarse_map)

    row = {
        "timestamp": str(datetime.now()),
        "split": split,
        "artifact_file": artifact_path.name,
        "run_title": run_title,
        "model_folder": run_name,
        "step": _artifact_step(artifact_path, artifact),
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
    }
    row.update(
        _compute_clustering_bundle(
            embeddings=embeddings,
            fine_labels=fine_labels,
            coarse_labels=coarse_labels,
            clustering_method=clustering_method,
            resolution=resolution,
            n_neighbors=n_neighbors,
        )
    )
    row.update(_compute_batch_correction_metrics(embeddings, fine_labels, dataset_ids))
    if split == "val":
        row.update(
            _compute_ridge_metrics(
                embeddings=embeddings,
                fine_labels=fine_labels,
                coarse_labels=coarse_labels,
                seed=ridge_seed,
            )
        )
    else:
        row.update(
            {
                "ridge_train_accuracy_fine": np.nan,
                "ridge_val_accuracy_fine": np.nan,
                "ridge_train_accuracy_coarse": np.nan,
                "ridge_val_accuracy_coarse": np.nan,
            }
        )
    return row


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"last_processed_artifact": None}
    with open(state_path, "r") as f:
        return json.load(f)


def _save_state(state_path: Path, state: dict) -> None:
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def _process_run_dir(
    run_dir: Path,
    coarse_map: dict[str, str],
    clustering_method: str,
    resolution: float,
    n_neighbors: int,
    ridge_seed: int,
    delete_processed_artifacts: bool,
) -> None:
    state_path = run_dir / "downstream_state.json"
    state = _load_state(state_path)
    last_processed = state.get("last_processed_artifact")

    metrics_csv = run_dir / "downstream_metrics.csv"
    latest_artifact = _latest_artifact_path(run_dir)
    if latest_artifact is None:
        return

    if latest_artifact.name != last_processed:
        artifact = torch.load(latest_artifact, map_location="cpu")
        row = _artifact_metrics_row(
            artifact_path=latest_artifact,
            artifact=artifact,
            run_title=run_dir.parent.name,
            run_name=run_dir.name,
            split="val",
            coarse_map=coarse_map,
            clustering_method=clustering_method,
            resolution=resolution,
            n_neighbors=n_neighbors,
            ridge_seed=ridge_seed,
        )
        row["checkpoint"] = ""
        append_result_safely(str(metrics_csv), row)
        state["last_processed_artifact"] = latest_artifact.name

    if delete_processed_artifacts:
        for artifact_path in run_dir.glob("*.pt"):
            if artifact_path.name.endswith(".tmp") or _is_persistent_artifact(artifact_path):
                continue
            if latest_artifact is not None and artifact_path.name != latest_artifact.name:
                artifact_path.unlink(missing_ok=True)

    _save_state(state_path, state)


def _is_run_fully_processed(run_dir: Path) -> bool:
    if not (run_dir / "DONE").exists():
        return False

    latest_artifact = _latest_artifact_path(run_dir)
    if latest_artifact is None:
        return True

    state = _load_state(run_dir / "downstream_state.json")
    return state.get("last_processed_artifact") == latest_artifact.name


def downstream_tasks(
    path_to_config: str,
) -> None:
    config = ScalingConfig(path_to_config)
    ckpt_root = Path(config.paths.path_to_ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_dir = ckpt_root / config.metadata.title / config.metadata.run_name

    downstream_cfg = getattr(config, "downstream", None)
    poll_interval_s = int(getattr(downstream_cfg, "poll_interval_s", 60))
    clustering_method = str(getattr(downstream_cfg, "clustering_method", "leiden"))
    resolution = float(getattr(downstream_cfg, "resolution", 1.0))
    n_neighbors = int(getattr(downstream_cfg, "n_neighbors", 15))
    ridge_seed = int(getattr(downstream_cfg, "ridge_seed", 0))
    keep_artifacts = bool(getattr(downstream_cfg, "keep_artifacts", False))
    delete_processed_artifacts = not keep_artifacts

    if clustering_method not in {"leiden", "louvain"}:
        raise ValueError(
            "Invalid downstream.clustering_method. Expected 'leiden' or 'louvain'."
        )

    effective_coarse_k = int(getattr(downstream_cfg, "coarse_k", DEFAULT_K))

    config_coarse_dir = getattr(
        getattr(config, "paths", None),
        "path_to_coarse_labels_dir",
        None,
    )
    effective_coarse_dir = (
        Path(config_coarse_dir) if config_coarse_dir is not None else ckpt_root
    )
    effective_coarse_dir.mkdir(parents=True, exist_ok=True)

    coarse_map = _load_coarse_mapping(
        path_to_config,
        effective_coarse_dir,
        effective_coarse_k,
    )

    print(f"[Downstream] Watching run directory: {run_dir}")
    while True:
        _process_run_dir(
            run_dir=run_dir,
            coarse_map=coarse_map,
            clustering_method=clustering_method,
            resolution=resolution,
            n_neighbors=n_neighbors,
            ridge_seed=ridge_seed,
            delete_processed_artifacts=delete_processed_artifacts,
        )

        all_done = _is_run_fully_processed(run_dir)
        if all_done:
            break

        time.sleep(poll_interval_s)

    print("[Downstream] Tasks completed successfully.")
