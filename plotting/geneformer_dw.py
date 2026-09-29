from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    if str(PLOTTING_DIR) in sys.path:
        sys.path.remove(str(PLOTTING_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

from aim import Repo  # noqa: E402

from plotting.training_loss import (  # noqa: E402
    FONT_SIZE_LABELS,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICKS,
    FONT_SIZE_TITLE,
    GENEFORMER_BCE_CONFIG,
    _format_param_count,
    _style_panel_box,
)
from plotting.utils import (  # noqa: E402
    COLOR_SHADES,
    PLOTS_DIR,
    ROOT_DIR,
    apply_plot_style,
    ensure_parent_dir,
    exponential_moving_average,
    flatten_dict,
    remove_bounding_box,
    save_figure,
)


DEFAULT_AIM_REPO = ROOT_DIR / "aim-repo" / "geneformer_dw"
DEFAULT_CSV = PLOTS_DIR / "geneformer_dw" / "01_isoflops_metrics.csv"
DEFAULT_PREPARED_DATA = PLOTS_DIR / "geneformer_dw" / "01_isoflops_prepared.parquet"
DEFAULT_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "01_isoflops_loss_vs_non_embedding_params.svg"
)
DEFAULT_TRAINING_DATA = (
    PLOTS_DIR / "geneformer_dw" / "02_training_loss_flops_prepared.parquet"
)
DEFAULT_TRAINING_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "02_training_loss_vs_flops_by_model_size.svg"
)
DEFAULT_DOWNSTREAM_DIR = ROOT_DIR / "checkpoints" / "geneformer_dw"
DEFAULT_DOWNSTREAM_DATA = (
    PLOTS_DIR / "geneformer_dw" / "03_downstream_metrics_prepared.parquet"
)
DEFAULT_DOWNSTREAM_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "03_downstream_metrics_vs_non_embedding_params.svg"
)
DEFAULT_DEPTH_OUTPUT = (
    PLOTS_DIR
    / "geneformer_dw"
    / "04_isoflops_loss_vs_non_embedding_params_by_depth.svg"
)
DEFAULT_LOG_SPACED_DEPTH_OUTPUT = (
    PLOTS_DIR
    / "geneformer_dw"
    / "05_isoflops_loss_vs_non_embedding_params_log_spaced_by_depth.svg"
)
DEFAULT_DEPTH_PARABOLAS_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "06_depth_parabolas_by_isoflop_budget.svg"
)
DEFAULT_DEPTH_CONNECTED_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "07_depth_connected_by_isoflop_budget.svg"
)
DEFAULT_DEPTH_CONNECTED_MIDDLE_OUTPUT = (
    PLOTS_DIR
    / "geneformer_dw"
    / "07_depth_connected_by_isoflop_budget_middle_slices.svg"
)
DEFAULT_RATIO_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "08_loss_vs_depth_width_ratio_by_isoflop_budget.svg"
)
DEFAULT_RATIO_MIDDLE_OUTPUT = (
    PLOTS_DIR
    / "geneformer_dw"
    / "08_loss_vs_depth_width_ratio_by_isoflop_budget_middle_slices.svg"
)
DEFAULT_WINNER_RATIO_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "09_winner_depth_width_ratio_vs_params.svg"
)
DEFAULT_SURFACE_OPTIMUM_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10_surface_fit_optimal_ratio_vs_params.svg"
)
DEFAULT_SURFACE_GRID_OVERLAY_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10b_surface_optimum_vs_ratio_grid.svg"
)
DEFAULT_SURFACE_RELATIVE_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10c_surface_optimum_relative_to_fixed_depth.svg"
)
DEFAULT_SURFACE_IMPLIED_DEPTH_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10d_surface_implied_optimal_depth.svg"
)
DEFAULT_SURFACE_COMPUTE_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10e_surface_optimal_ratio_vs_compute.svg"
)
DEFAULT_SURFACE_CONTOUR_PARAMS_RATIO_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10f_surface_contours_params_ratio.svg"
)
DEFAULT_SURFACE_CONTOUR_COMPUTE_RATIO_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10g_surface_contours_compute_ratio.svg"
)
DEFAULT_SURFACE_CONTOUR_PARAMS_COMPUTE_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10h_surface_contours_params_compute.svg"
)
DEFAULT_SURFACE_FIT_DIAGNOSTICS_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10i_surface_fit_diagnostics.svg"
)
DEFAULT_SURFACE_CV_DIAGNOSTICS_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10j_surface_cv_diagnostics.svg"
)
DEFAULT_SURFACE_BOOTSTRAP_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "10k_surface_bootstrap_optimum_bands.svg"
)
DEFAULT_SURFACE_DIAGNOSTICS_CSV = (
    PLOTS_DIR / "geneformer_dw" / "10_surface_fit_diagnostics.csv"
)
DEFAULT_SURFACE_COEFFICIENTS = (
    PLOTS_DIR / "geneformer_dw" / "10_surface_fit_coefficients.csv"
)
DEFAULT_RATIO_GRID_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "11_depth_width_ratio_grid.svg"
)
DEFAULT_ISOPARAM_RATIO_OUTPUT = (
    PLOTS_DIR / "geneformer_dw" / "12_isoparam_ratio_sweeps_by_isoflop_budget.svg"
)
DEFAULT_TARGET_TRAINING_FLOPS = 1.0e18
DEFAULT_LOG_SPACED_MIN_TRAINING_FLOPS = 3.0e15
DEFAULT_N_ISOFLOP_CURVES = 7
DEFAULT_MIN_ISOFLOP_FRACTION = 3.0e15 / DEFAULT_TARGET_TRAINING_FLOPS
COMPLETE_POINT_THRESHOLD = 0.99
SURFACE_OPTIMUM_LINE_COLOR = "#D55E00"
DOWNSTREAM_METRICS = (
    "cluster_fine_nmi",
    "cluster_fine_ari",
    "cluster_coarse_nmi",
    "cluster_coarse_ari",
    "ridge_val_accuracy_fine",
    "ridge_val_accuracy_coarse",
    "batch_graph_conn",
    "batch_asw_batch",
)

RUN_TITLE = "geneformer_dw"
RUN_QUERY = f"run.config.metadata.title == '{RUN_TITLE}'"
METRICS_TO_EXTRACT = {"bce"}
CONFIG_KEYS = {
    "metadata.pipeline",
    "metadata.run_name",
    "metadata.title",
    "datamodule.batch_size",
    "model.context_length",
    "model.d_model",
    "model.lr",
    "model.transformer.n_layers",
    "model.bioformer.n_layers",
    "trainer.accumulate_grad",
    "trainer.n_steps",
    "sweep_metadata.training_flops_per_step",
}
RUN_FIELDS = {
    "hparams.parameters_total",
    "hparams.parameters_embedding",
    "hparams.parameters_encoder",
    "hparams.parameters_decoder",
}
REQUIRED_CSV_COLUMNS = {
    "run_hash",
    "metadata.pipeline",
    "metadata.title",
    "metadata.run_name",
    "metric_name",
    "analysis_fraction",
    "target_step",
    "sampled_step",
    "max_step",
    "loss",
    "training_flops_per_step",
    "sampled_training_flops",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Aim metrics for geneformer_dw and plot isoFLOP loss parabolas.",
    )
    parser.add_argument("--aim-repo", type=Path, default=DEFAULT_AIM_REPO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--prepared-data", type=Path, default=DEFAULT_PREPARED_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--training-output", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    parser.add_argument("--downstream-dir", type=Path, default=DEFAULT_DOWNSTREAM_DIR)
    parser.add_argument("--downstream-data", type=Path, default=DEFAULT_DOWNSTREAM_DATA)
    parser.add_argument(
        "--downstream-output", type=Path, default=DEFAULT_DOWNSTREAM_OUTPUT
    )
    parser.add_argument("--depth-output", type=Path, default=DEFAULT_DEPTH_OUTPUT)
    parser.add_argument(
        "--log-spaced-depth-output", type=Path, default=DEFAULT_LOG_SPACED_DEPTH_OUTPUT
    )
    parser.add_argument(
        "--depth-parabolas-output", type=Path, default=DEFAULT_DEPTH_PARABOLAS_OUTPUT
    )
    parser.add_argument(
        "--depth-connected-output", type=Path, default=DEFAULT_DEPTH_CONNECTED_OUTPUT
    )
    parser.add_argument(
        "--depth-connected-middle-output",
        type=Path,
        default=DEFAULT_DEPTH_CONNECTED_MIDDLE_OUTPUT,
    )
    parser.add_argument("--ratio-output", type=Path, default=DEFAULT_RATIO_OUTPUT)
    parser.add_argument(
        "--ratio-middle-output", type=Path, default=DEFAULT_RATIO_MIDDLE_OUTPUT
    )
    parser.add_argument(
        "--winner-ratio-output", type=Path, default=DEFAULT_WINNER_RATIO_OUTPUT
    )
    parser.add_argument(
        "--surface-optimum-output", type=Path, default=DEFAULT_SURFACE_OPTIMUM_OUTPUT
    )
    parser.add_argument(
        "--surface-grid-overlay-output",
        type=Path,
        default=DEFAULT_SURFACE_GRID_OVERLAY_OUTPUT,
    )
    parser.add_argument(
        "--surface-relative-output", type=Path, default=DEFAULT_SURFACE_RELATIVE_OUTPUT
    )
    parser.add_argument(
        "--surface-implied-depth-output",
        type=Path,
        default=DEFAULT_SURFACE_IMPLIED_DEPTH_OUTPUT,
    )
    parser.add_argument(
        "--surface-compute-output", type=Path, default=DEFAULT_SURFACE_COMPUTE_OUTPUT
    )
    parser.add_argument(
        "--surface-contour-params-ratio-output",
        type=Path,
        default=DEFAULT_SURFACE_CONTOUR_PARAMS_RATIO_OUTPUT,
    )
    parser.add_argument(
        "--surface-contour-compute-ratio-output",
        type=Path,
        default=DEFAULT_SURFACE_CONTOUR_COMPUTE_RATIO_OUTPUT,
    )
    parser.add_argument(
        "--surface-contour-params-compute-output",
        type=Path,
        default=DEFAULT_SURFACE_CONTOUR_PARAMS_COMPUTE_OUTPUT,
    )
    parser.add_argument(
        "--surface-fit-diagnostics-output",
        type=Path,
        default=DEFAULT_SURFACE_FIT_DIAGNOSTICS_OUTPUT,
    )
    parser.add_argument(
        "--surface-cv-diagnostics-output",
        type=Path,
        default=DEFAULT_SURFACE_CV_DIAGNOSTICS_OUTPUT,
    )
    parser.add_argument(
        "--surface-bootstrap-output",
        type=Path,
        default=DEFAULT_SURFACE_BOOTSTRAP_OUTPUT,
    )
    parser.add_argument(
        "--surface-diagnostics-csv", type=Path, default=DEFAULT_SURFACE_DIAGNOSTICS_CSV
    )
    parser.add_argument(
        "--surface-coefficients", type=Path, default=DEFAULT_SURFACE_COEFFICIENTS
    )
    parser.add_argument(
        "--ratio-grid-output", type=Path, default=DEFAULT_RATIO_GRID_OUTPUT
    )
    parser.add_argument(
        "--isoparam-ratio-output", type=Path, default=DEFAULT_ISOPARAM_RATIO_OUTPUT
    )
    parser.add_argument(
        "--target-training-flops",
        type=float,
        default=DEFAULT_TARGET_TRAINING_FLOPS,
        help="Matched terminal training FLOP budget for the DW sweep.",
    )
    parser.add_argument(
        "--n-isoflop-curves",
        type=int,
        default=DEFAULT_N_ISOFLOP_CURVES,
        help="Number of equally spaced isoFLOP curves to sample.",
    )
    parser.add_argument(
        "--min-isoflop-fraction",
        type=float,
        default=DEFAULT_MIN_ISOFLOP_FRACTION,
        help="Smallest isoFLOP budget as a fraction of --target-training-flops.",
    )
    parser.add_argument(
        "--log-spaced-min-training-flops",
        type=float,
        default=DEFAULT_LOG_SPACED_MIN_TRAINING_FLOPS,
        help="Smallest FLOP budget for the fifth log-spaced depth plot.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute cached CSV and parquet data."
    )
    return parser.parse_args()


def _default_flops_fractions(n_curves: int, min_fraction: float) -> list[float]:
    if n_curves < 1:
        raise ValueError("--n-isoflop-curves must be at least 1")
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("--min-isoflop-fraction must be in (0, 1]")
    if n_curves == 1:
        return [1.0]
    return np.geomspace(min_fraction, 1.0, n_curves).tolist()


def _log_spaced_depth_flops_fractions(args: argparse.Namespace) -> list[float]:
    if args.log_spaced_min_training_flops <= 0:
        raise ValueError("--log-spaced-min-training-flops must be positive")
    min_fraction = args.log_spaced_min_training_flops / args.target_training_flops
    if min_fraction > 1.0:
        raise ValueError(
            "--log-spaced-min-training-flops must be no larger than --target-training-flops"
        )
    return _default_flops_fractions(args.n_isoflop_curves, min_fraction)


def _requested_flops_fractions(args: argparse.Namespace) -> list[float]:
    fractions = [
        *_default_flops_fractions(args.n_isoflop_curves, args.min_isoflop_fraction),
        *_log_spaced_depth_flops_fractions(args),
    ]
    return sorted({round(float(fraction), 12) for fraction in fractions})


def _context_to_dict(context: object) -> dict[str, object]:
    if context is None:
        return {}
    if hasattr(context, "to_dict"):
        context_dict = context.to_dict()
        if isinstance(context_dict, dict):
            return context_dict
    try:
        return dict(context)
    except (TypeError, ValueError):
        return {}


def _row_value(row: object, key: str) -> object:
    if hasattr(row, "get"):
        return row.get(key)
    return getattr(row, key)


def _has_requested_fractions(path: Path, requested_fractions: list[float]) -> bool:
    if not path.exists():
        return False
    try:
        cached_fractions = (
            pl.read_csv(path, columns=["analysis_fraction"])["analysis_fraction"]
            .unique()
            .to_list()
        )
    except pl.exceptions.PolarsError:
        try:
            cached_fractions = (
                pl.read_parquet(path, columns=["analysis_fraction"])[
                    "analysis_fraction"
                ]
                .unique()
                .to_list()
            )
        except pl.exceptions.PolarsError:
            return False
    rounded_cached = {round(float(value), 6) for value in cached_fractions}
    return {round(value, 6) for value in requested_fractions}.issubset(rounded_cached)


def _ensure_metrics_csv(args: argparse.Namespace) -> tuple[Path, bool]:
    requested_fractions = _requested_flops_fractions(args)
    if args.csv.exists() and not args.force:
        cached_columns = set(pl.read_csv(args.csv, n_rows=0).columns)
        if REQUIRED_CSV_COLUMNS.issubset(cached_columns) and _has_requested_fractions(
            args.csv, requested_fractions
        ):
            return args.csv, False

    if not args.aim_repo.exists():
        raise FileNotFoundError(f"Aim repo not found: {args.aim_repo}")

    _export_sampled_metrics_csv(
        args.aim_repo,
        args.csv,
        flops_fractions=requested_fractions,
        target_training_flops=args.target_training_flops,
        metrics_to_extract=METRICS_TO_EXTRACT,
        run_query=RUN_QUERY,
        config_keys=CONFIG_KEYS,
        run_fields=RUN_FIELDS,
    )
    return args.csv, True


def _export_sampled_metrics_csv(
    aim_repo_path: Path,
    output_csv: Path,
    *,
    flops_fractions: list[float],
    target_training_flops: float,
    metrics_to_extract: set[str],
    run_query: str | None,
    config_keys: set[str],
    run_fields: set[str],
) -> None:
    repo = Repo(str(aim_repo_path))
    rows: list[dict[str, object]] = []
    if repo.list_all_runs():
        rows = _export_sampled_metrics_rows_from_chunks(
            repo,
            flops_fractions=flops_fractions,
            target_training_flops=target_training_flops,
            metrics_to_extract=metrics_to_extract,
            run_query=run_query,
            config_keys=config_keys,
            run_fields=run_fields,
        )
        runs = ()
    else:
        runs = repo.query_runs(run_query).iter_runs() if run_query else repo.iter_runs()

    for run_entry in runs:
        run = run_entry.run if hasattr(run_entry, "run") else run_entry
        config = run.get("config", default={}) or {}
        flat_config = flatten_dict(config) if isinstance(config, dict) else {}
        flat_config = {
            key: value for key, value in flat_config.items() if key in config_keys
        }

        hparams = run.get("hparams", default={}) or {}
        flat_hparams = (
            flatten_dict(hparams, prefix="hparams") if isinstance(hparams, dict) else {}
        )
        flat_hparams = {
            key: value for key, value in flat_hparams.items() if key in run_fields
        }

        run_common = {
            "run_hash": run.hash,
            "run_name": run.name,
            "experiment": run.experiment if run.experiment else "default",
            **flat_config,
            **flat_hparams,
        }
        n_steps = int(flat_config["trainer.n_steps"])
        training_flops_per_step = float(
            flat_config.get(
                "sweep_metadata.training_flops_per_step",
                target_training_flops / n_steps,
            )
        )

        for metric_sequence in run.metrics():
            metric_name = metric_sequence.name
            if metric_name not in metrics_to_extract:
                continue

            context_dict = _context_to_dict(getattr(metric_sequence, "context", None))
            if context_dict.get("subset") not in (None, "train"):
                continue

            metric_df = metric_sequence.dataframe()
            if metric_df.empty:
                continue

            metric_df = metric_df.sort_values("step")
            max_step = int(metric_df["step"].max())
            for fraction in flops_fractions:
                analysis_training_flops = fraction * target_training_flops
                target_step = max(
                    1, int(round(analysis_training_flops / training_flops_per_step))
                )
                reached_df = metric_df[metric_df["step"] <= target_step]
                sampled = metric_df.iloc[0] if reached_df.empty else reached_df.iloc[-1]
                sampled_step = int(_row_value(sampled, "step"))
                rows.append(
                    {
                        **run_common,
                        "metric_name": metric_name,
                        **flatten_dict(context_dict, prefix="context"),
                        "analysis_fraction": fraction,
                        "target_step": target_step,
                        "sampled_step": sampled_step,
                        "max_step": max_step,
                        "loss": _row_value(sampled, "value"),
                        "training_flops_per_step": training_flops_per_step,
                        "analysis_training_flops": analysis_training_flops,
                        "sampled_training_flops": sampled_step
                        * training_flops_per_step,
                        "timestamp": _row_value(sampled, "time")
                        if "time" in metric_df.columns
                        else _row_value(sampled, "timestamp"),
                    }
                )

    if not rows:
        repo.close()
        raise ValueError(
            f"No matching sampled metrics found in Aim repo: {aim_repo_path}"
        )

    ensure_parent_dir(output_csv)
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    repo.close()


def _matches_run_query(flat_config: dict[str, object], run_query: str | None) -> bool:
    if run_query is None:
        return True
    if run_query == RUN_QUERY:
        return flat_config.get("metadata.title") == RUN_TITLE
    raise ValueError(
        f"Chunk-backed Aim export only supports the configured {RUN_TITLE} run query. "
        f"Received: {run_query}"
    )


def _chunk_run_tree(repo: Repo, run_hash: str):
    return (
        repo.request_tree("meta", run_hash, read_only=True)
        .subtree("meta")
        .subtree("chunks")
        .subtree(run_hash)
    )


def _collect_chunk_attrs(run_tree) -> tuple[dict[str, object], dict[str, object]]:
    attrs = run_tree.subtree("attrs")
    config = attrs.collect("config")
    hparams = attrs.collect("hparams")
    if not isinstance(config, dict):
        config = {}
    if not isinstance(hparams, dict):
        hparams = {}
    return config, hparams


def _train_context_indices(run_tree) -> list[int]:
    contexts = run_tree.get("contexts", {})
    train_contexts = [
        int(context_idx)
        for context_idx, context in contexts.items()
        if isinstance(context, dict) and context.get("subset") == "train"
    ]
    if train_contexts:
        return train_contexts
    return [
        int(context_idx)
        for context_idx, context in contexts.items()
        if context in ({}, None)
    ]


def _sequence_pairs_from_chunk(
    repo: Repo, run_hash: str, context_idx: int, metric_name: str
) -> list[tuple[int, float, object]]:
    sequence_tree = (
        repo.request_tree("seqs", run_hash, read_only=True)
        .subtree("seqs")
        .subtree(("v2", "chunks", run_hash, context_idx, metric_name))
    )
    steps = dict(sequence_tree.array("step").items())
    values = dict(sequence_tree.array("val").items())
    try:
        timestamps = dict(sequence_tree.array("time").items())
    except (KeyError, IndexError):
        timestamps = {}
    return sorted(
        (int(step), float(values[step_hash]), timestamps.get(step_hash))
        for step_hash, step in steps.items()
        if step_hash in values
    )


def _sequence_arrays_from_chunk(
    repo: Repo, run_hash: str, context_idx: int, metric_name: str
) -> tuple[np.ndarray, np.ndarray]:
    sequence_pairs = _sequence_pairs_from_chunk(
        repo, run_hash, context_idx, metric_name
    )
    if not sequence_pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    steps = np.array([step for step, _, _ in sequence_pairs], dtype=np.int64)
    values = np.array([value for _, value, _ in sequence_pairs], dtype=float)
    return steps, values


def _merge_sequence_segments(
    segments: list[tuple[str, list[tuple[int, float, object]]]],
) -> list[tuple[int, float, object]]:
    """Merge resumed/retried Aim traces, preferring the most complete segment on overlap."""
    by_step: dict[int, tuple[int, float, object]] = {}
    ranked_segments = sorted(
        segments,
        key=lambda item: (
            len(item[1]),
            max((step for step, _, _ in item[1]), default=-1),
            item[0],
        ),
        reverse=True,
    )
    for _, sequence_pairs in ranked_segments:
        for step, value, timestamp in sequence_pairs:
            by_step.setdefault(step, (step, value, timestamp))
    return [by_step[step] for step in sorted(by_step)]


def _collect_chunk_metric_groups(
    repo: Repo,
    *,
    metric_names: set[str],
    run_query: str | None,
) -> list[dict[str, Any]]:
    """Group trace segments by sweep config so retries count as one logical run."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for run_hash in sorted(repo.list_all_runs()):
        run_tree = _chunk_run_tree(repo, run_hash)
        config, hparams = _collect_chunk_attrs(run_tree)
        flat_config = flatten_dict(config)
        if not _matches_run_query(flat_config, run_query):
            continue

        config_identity = str(flat_config.get("metadata.run_name") or run_hash)
        traces = run_tree.get("traces", {})
        for context_idx in _train_context_indices(run_tree):
            context_traces = traces.get(context_idx, {})
            if not isinstance(context_traces, dict):
                continue
            for metric_name in sorted(metric_names):
                if metric_name not in context_traces:
                    continue
                sequence_pairs = _sequence_pairs_from_chunk(
                    repo, run_hash, context_idx, metric_name
                )
                if not sequence_pairs:
                    continue

                key = (config_identity, metric_name)
                group = groups.setdefault(
                    key,
                    {
                        "config_identity": config_identity,
                        "metric_name": metric_name,
                        "config": config,
                        "hparams": hparams,
                        "representative_rank": (-1, -1),
                        "run_hashes": set(),
                        "segments": [],
                    },
                )
                segment_rank = (len(sequence_pairs), sequence_pairs[-1][0])
                if segment_rank > group["representative_rank"]:
                    group["config"] = config
                    group["hparams"] = hparams
                    group["representative_rank"] = segment_rank
                group["run_hashes"].add(run_hash)
                group["segments"].append((run_hash, sequence_pairs))

    collected: list[dict[str, Any]] = []
    for group in groups.values():
        group["sequence_pairs"] = _merge_sequence_segments(group.pop("segments"))
        group.pop("representative_rank")
        group["run_hashes"] = sorted(group["run_hashes"])
        collected.append(group)
    return sorted(
        collected, key=lambda group: (group["config_identity"], group["metric_name"])
    )


def _export_sampled_metrics_rows_from_chunks(
    repo: Repo,
    *,
    flops_fractions: list[float],
    target_training_flops: float,
    metrics_to_extract: set[str],
    run_query: str | None,
    config_keys: set[str],
    run_fields: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = _collect_chunk_metric_groups(
        repo,
        metric_names=metrics_to_extract,
        run_query=run_query,
    )
    for group in groups:
        config = group["config"]
        hparams = group["hparams"]
        flat_config = flatten_dict(config)
        selected_config = {
            key: value for key, value in flat_config.items() if key in config_keys
        }
        flat_hparams = flatten_dict(hparams, prefix="hparams")
        selected_hparams = {
            key: value for key, value in flat_hparams.items() if key in run_fields
        }
        if "trainer.n_steps" not in selected_config:
            raise ValueError(
                f"Missing trainer.n_steps for run {group['config_identity']}"
            )
        training_flops_per_step = float(
            config["sweep_metadata"]["training_flops_per_step"]
        )

        run_common = {
            "run_hash": "+".join(group["run_hashes"]),
            "run_name": group["config_identity"],
            "experiment": "default",
            **selected_config,
            **selected_hparams,
        }
        sequence_pairs = group["sequence_pairs"]
        max_step = int(sequence_pairs[-1][0])
        steps = [step for step, _, _ in sequence_pairs]

        for fraction in flops_fractions:
            analysis_training_flops = fraction * target_training_flops
            target_step = max(
                1, int(round(analysis_training_flops / training_flops_per_step))
            )
            sample_index = int(np.searchsorted(steps, target_step, side="right") - 1)
            sample_index = max(sample_index, 0)
            sampled_step, sampled_loss, sampled_time = sequence_pairs[sample_index]
            rows.append(
                {
                    **run_common,
                    "metric_name": group["metric_name"],
                    "context.subset": "train",
                    "analysis_fraction": fraction,
                    "target_step": target_step,
                    "sampled_step": sampled_step,
                    "max_step": max_step,
                    "loss": sampled_loss,
                    "training_flops_per_step": training_flops_per_step,
                    "analysis_training_flops": analysis_training_flops,
                    "sampled_training_flops": sampled_step * training_flops_per_step,
                    "timestamp": sampled_time,
                }
            )

    return rows


def _collect_training_curve_data_from_chunks(aim_repo_path: Path) -> pl.DataFrame:
    repo = Repo(str(aim_repo_path))
    frames: list[pl.DataFrame] = []
    try:
        groups = _collect_chunk_metric_groups(
            repo,
            metric_names={GENEFORMER_BCE_CONFIG.metric_name},
            run_query=RUN_QUERY,
        )
        for group in groups:
            config = group["config"]
            hparams = group["hparams"]
            target_params = float(
                config["sweep_metadata"]["target_non_embedding_params"]
            )
            actual_params = float(
                config["sweep_metadata"]["actual_non_embedding_params"]
            )
            target_depth = int(config["sweep_metadata"]["target_depth"])
            training_flops_per_step = float(
                config["sweep_metadata"]["training_flops_per_step"]
            )
            lr_warmup_steps = int(config["trainer"]["lr_warmup_steps"])
            d_model = int(config["model"]["d_model"])
            encoder_params = float(hparams["parameters_encoder"])
            sequence_pairs = group["sequence_pairs"]
            steps = np.array([step for step, _, _ in sequence_pairs], dtype=np.int64)
            losses = np.array([loss for _, loss, _ in sequence_pairs], dtype=float)
            frames.append(
                pl.DataFrame(
                    {
                        "run_hash": "+".join(group["run_hashes"]),
                        "step": steps,
                        "loss": losses,
                        "cumulative_training_flops": steps.astype(float)
                        * training_flops_per_step,
                        "target_non_embedding_params": target_params,
                        "non_embedding_params": actual_params,
                        "backbone_params": encoder_params,
                        "target_depth": target_depth,
                        "model.d_model": d_model,
                        "training_flops_per_step": training_flops_per_step,
                        "trainer.lr_warmup_steps": lr_warmup_steps,
                    }
                )
            )
    finally:
        repo.close()

    if not frames:
        raise ValueError(f"No training curves found in Aim repo: {aim_repo_path}")
    return pl.concat(frames, how="vertical")


def _ensure_training_curve_data(args: argparse.Namespace) -> tuple[pl.DataFrame, bool]:
    if args.training_data.exists() and not args.force:
        training_df = pl.read_parquet(args.training_data)
        if "trainer.lr_warmup_steps" in training_df.columns:
            return training_df, False

    training_df = _collect_training_curve_data_from_chunks(args.aim_repo)
    ensure_parent_dir(args.training_data)
    training_df.write_parquet(args.training_data)
    return training_df, True


def _parse_float_from_name(pattern: str, column: str) -> pl.Expr:
    return pl.col(column).str.extract(pattern, 1).cast(pl.Float64)


def _parse_int_from_name(pattern: str, column: str) -> pl.Expr:
    return pl.col(column).str.extract(pattern, 1).cast(pl.Int64)


def _target_params_expr(column: str = "model_folder") -> pl.Expr:
    return _parse_float_from_name(r"target_params_([0-9.eE+]+)-", column)


def _target_depth_expr(column: str = "model_folder") -> pl.Expr:
    return _parse_int_from_name(r"depth_([0-9]+)", column)


def _with_dw_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "hparams.parameters_encoder" in df.columns:
        backbone_params = pl.col("hparams.parameters_encoder")
    elif {
        "hparams.parameters_total",
        "hparams.parameters_embedding",
        "hparams.parameters_decoder",
    }.issubset(set(df.columns)):
        backbone_params = (
            pl.col("hparams.parameters_total")
            - pl.col("hparams.parameters_embedding")
            - pl.col("hparams.parameters_decoder")
        )
    else:
        raise ValueError(
            "Cannot derive backbone_params: expected hparams.parameters_encoder or "
            "total, embedding, and decoder parameter counts."
        )

    enriched = df.with_columns(
        [
            backbone_params.alias("backbone_params"),
            backbone_params.alias("non_embedding_params"),
            _target_params_expr("metadata.run_name").alias(
                "target_non_embedding_params"
            ),
            _target_depth_expr("metadata.run_name").alias("target_depth"),
            pl.col("trainer.n_steps").cast(pl.Int64).alias("trainer.n_steps"),
            pl.col("sampled_step").cast(pl.Int64).alias("sampled_step"),
            pl.col("target_step").cast(pl.Int64).alias("target_step"),
            pl.col("max_step").cast(pl.Int64).alias("max_step"),
            pl.col("loss").cast(pl.Float64).alias("loss"),
            pl.col("training_flops_per_step")
            .cast(pl.Float64)
            .alias("training_flops_per_step"),
            pl.col("sampled_training_flops")
            .cast(pl.Float64)
            .alias("sampled_training_flops"),
        ]
    )

    missing = enriched.select(
        [
            pl.col("target_non_embedding_params")
            .null_count()
            .alias("missing_target_params"),
            pl.col("target_depth").null_count().alias("missing_target_depth"),
        ]
    ).row(0)
    if any(value > 0 for value in missing):
        raise ValueError(
            "Could not parse target parameter count or target depth from metadata.run_name"
        )

    return enriched


def _ensure_prepared_data(
    args: argparse.Namespace, metrics_csv: Path
) -> tuple[pl.DataFrame, bool]:
    requested_fractions = _requested_flops_fractions(args)
    if args.prepared_data.exists() and not args.force:
        prepared_df = pl.read_parquet(args.prepared_data)
        required = {
            "backbone_params",
            "non_embedding_params",
            "target_non_embedding_params",
            "target_depth",
            "analysis_fraction",
            "analysis_training_flops",
            "completion_fraction",
            "training_flops_per_step",
            "sampled_training_flops",
        }
        if required.issubset(set(prepared_df.columns)) and _has_requested_fractions(
            args.prepared_data, requested_fractions
        ):
            return prepared_df, False

    prepared_df = pl.read_csv(metrics_csv).filter(
        (pl.col("metadata.pipeline") == GENEFORMER_BCE_CONFIG.pipeline)
        & (pl.col("metric_name") == GENEFORMER_BCE_CONFIG.metric_name)
    )
    if "context.subset" in prepared_df.columns:
        prepared_df = prepared_df.filter(pl.col("context.subset") == "train")
    if prepared_df.is_empty():
        raise ValueError(f"No sampled rows found in {metrics_csv}")

    prepared_df = _with_dw_columns(prepared_df)
    prepared_df = prepared_df.with_columns(
        [
            pl.col("analysis_fraction").cast(pl.Float64),
            (
                pl.col("analysis_fraction").cast(pl.Float64)
                * args.target_training_flops
            ).alias("analysis_training_flops"),
            pl.lit(args.target_training_flops).alias("target_training_flops"),
            (pl.col("max_step") / pl.col("target_step"))
            .clip(0.0, 1.0)
            .alias("completion_fraction"),
            (pl.col("sampled_step") * pl.col("training_flops_per_step")).alias(
                "sampled_training_flops"
            ),
        ]
    )
    ensure_parent_dir(args.prepared_data)
    prepared_df.write_parquet(args.prepared_data)
    return prepared_df, True


def _load_downstream_metrics(downstream_dir: Path) -> pl.DataFrame:
    metric_paths = sorted(downstream_dir.glob("**/downstream_metrics.csv"))
    frames: list[pl.DataFrame] = []
    for metric_path in metric_paths:
        try:
            frame = pl.read_csv(metric_path)
        except pl.exceptions.PolarsError:
            continue
        if (
            frame.is_empty()
            or "model_folder" not in frame.columns
            or "step" not in frame.columns
        ):
            continue
        frames.append(frame.with_columns(pl.lit(str(metric_path)).alias("source_csv")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def _ensure_downstream_data(args: argparse.Namespace) -> tuple[pl.DataFrame, bool]:
    if args.downstream_data.exists() and not args.force:
        downstream_df = pl.read_parquet(args.downstream_data)
        if not downstream_df.is_empty():
            return downstream_df, False

    downstream_df = _load_downstream_metrics(args.downstream_dir)
    if downstream_df.is_empty():
        return downstream_df, False

    available_metrics = [
        metric for metric in DOWNSTREAM_METRICS if metric in downstream_df.columns
    ]
    if not available_metrics:
        return pl.DataFrame(), False

    prepared_df = (
        downstream_df.with_columns(
            [
                pl.col("step").cast(pl.Int64),
                _target_params_expr().alias("target_non_embedding_params"),
                _target_depth_expr().alias("target_depth"),
            ]
        )
        .filter(
            pl.col("target_non_embedding_params").is_not_null()
            & pl.col("target_depth").is_not_null()
        )
        .sort(["model_folder", "step"])
        .group_by("model_folder", maintain_order=True)
        .tail(1)
        .with_columns(
            pl.col("target_non_embedding_params").alias("non_embedding_params")
        )
    )
    if prepared_df.is_empty():
        return prepared_df, False

    ensure_parent_dir(args.downstream_data)
    prepared_df.write_parquet(args.downstream_data)
    return prepared_df, True


def _format_flops_value(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}e{exponent}"


def _format_flops_tick(value: float) -> str:
    formatted = _format_flops_value(value)
    mantissa, exponent = formatted.split("e")
    if float(mantissa) == 1.0:
        return f"1e{exponent}"
    return formatted


def _format_compact_param_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.0f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(round(value)))


def _fit_log_parameter_parabola(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return None
    log_x = np.log10(x[mask].astype(float))
    coefficients = np.polyfit(log_x, y[mask].astype(float), deg=2)
    x_fit = np.geomspace(float(np.min(x[mask])), float(np.max(x[mask])), 240)
    y_fit = np.polyval(coefficients, np.log10(x_fit))
    return x_fit, y_fit


def _fit_log_x_parabola(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    return _fit_log_parameter_parabola(x, y)


def _loss_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return 0.0, 1.0
    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    padding = max(0.06 * (data_max - data_min), 0.08)
    return data_min - padding, data_max + padding


def _with_depth_brightness(
    color: tuple[float, float, float, float], normalized_depth: np.ndarray
) -> np.ndarray:
    rgb = np.tile(np.array(color[:3], dtype=float), (len(normalized_depth), 1))
    hsv = mpl.colors.rgb_to_hsv(rgb)
    hsv[:, 1] = np.clip(0.30 + 0.62 * normalized_depth, 0.0, 1.0)
    hsv[:, 2] = np.clip(0.95 - 0.42 * normalized_depth, 0.0, 1.0)
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return np.column_stack([rgb, np.ones(len(normalized_depth))])


def _filter_analysis_fractions(
    df: pl.DataFrame, fractions: list[float]
) -> pl.DataFrame:
    filter_expr = pl.lit(False)
    for fraction in fractions:
        filter_expr = filter_expr | (
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
    return df.filter(filter_expr)


def _with_depth_width_ratio(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("target_depth") / pl.col("model.d_model")).alias("depth_width_ratio")
    )


def create_isoflop_loss_vs_params_plot(
    df: pl.DataFrame, output_path: Path, *, fractions: list[float] | None = None
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _filter_analysis_fractions(df, fractions)
    if len(fractions) == 0:
        raise ValueError("No isoFLOP fractions found in prepared data")

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.6, 5.9),
        gridspec_kw={"width_ratios": [5.2, 1.45], "wspace": 0.30},
    )

    all_x: list[float] = []
    all_y: list[float] = []
    curve_handles: list[mlines.Line2D] = []

    for index, fraction in enumerate(fractions):
        curve_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        ).sort(["non_embedding_params", "target_depth"])
        x = curve_df["non_embedding_params"].to_numpy().astype(float)
        y = curve_df["loss"].to_numpy().astype(float)
        completion = curve_df["completion_fraction"].to_numpy().astype(float)
        complete = completion >= COMPLETE_POINT_THRESHOLD
        color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        label = f"{_format_flops_value(float(curve_df['analysis_training_flops'].drop_nulls().first()))} FLOPs"

        fit = _fit_log_parameter_parabola(x, y)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, color=color, linestyle=":", linewidth=1.8, zorder=2)

        point_alpha = np.where(
            complete, 0.88, np.clip(completion / COMPLETE_POINT_THRESHOLD, 0.18, 0.82)
        )
        ax.scatter(
            x[complete],
            y[complete],
            color=color,
            edgecolors="black",
            linewidths=0.7,
            s=38,
            alpha=point_alpha[complete],
            zorder=4,
        )
        if np.any(~complete):
            ax.scatter(
                x[~complete],
                y[~complete],
                marker="D",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                s=42,
                alpha=point_alpha[~complete],
                zorder=5,
            )

        curve_handles.append(
            mlines.Line2D(
                [], [], color=color, linestyle=":", linewidth=2.0, label=label
            )
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    ax.set_xscale("log")
    if all_x:
        ax.set_xlim(float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
    if all_y:
        ax.set_ylim(*_loss_limits(np.array(all_y, dtype=float)))
    ax.set_xlabel("Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Cross Entropy", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "IsoFLOP\nBudget",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=curve_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 4,
        handlelength=2.2,
        labelspacing=0.72,
    )
    legend_ax.text(
        0.5,
        0.10,
        "diamonds:\n<99% complete",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 5,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "IsoFLOP Scaling Curves", fontsize=FONT_SIZE_TITLE, x=0.48, y=0.96, ha="center"
    )
    fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def create_isoflop_loss_vs_params_by_depth_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    fractions: list[float] | None = None,
    title: str = "IsoFLOP Scaling Curves by Depth",
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _filter_analysis_fractions(df, fractions)
    if len(fractions) == 0:
        raise ValueError("No isoFLOP fractions found in prepared data")

    depth_values = (
        df["target_depth"].drop_nulls().unique().sort().to_numpy().astype(float)
    )
    if len(depth_values) == 0:
        raise ValueError("No target_depth values found in prepared data")

    depth_min = float(np.min(depth_values))
    depth_max = float(np.max(depth_values))
    depth_norm = mpl.colors.Normalize(vmin=depth_min, vmax=depth_max)
    depth_display_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "geneformer_dw_depth_brightness", ["#f0eee8", "#2f2a24"]
    )

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.6, 5.9),
        gridspec_kw={"width_ratios": [5.2, 1.45], "wspace": 0.30},
    )

    all_x: list[float] = []
    all_y: list[float] = []
    curve_handles: list[mlines.Line2D] = []

    for index, fraction in enumerate(fractions):
        curve_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        ).sort(["non_embedding_params", "target_depth"])
        x = curve_df["non_embedding_params"].to_numpy().astype(float)
        y = curve_df["loss"].to_numpy().astype(float)
        depths = curve_df["target_depth"].to_numpy().astype(float)
        completion = curve_df["completion_fraction"].to_numpy().astype(float)
        complete = completion >= COMPLETE_POINT_THRESHOLD
        base_color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        colors = _with_depth_brightness(base_color, depth_norm(depths))
        label = f"{_format_flops_value(float(curve_df['analysis_training_flops'].drop_nulls().first()))} FLOPs"

        fit = _fit_log_parameter_parabola(x, y)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(
                x_fit,
                y_fit,
                color=base_color,
                linestyle=":",
                linewidth=1.6,
                alpha=0.60,
                zorder=2,
            )

        point_alpha = np.where(
            complete, 0.90, np.clip(completion / COMPLETE_POINT_THRESHOLD, 0.25, 0.78)
        )
        ax.scatter(
            x[complete],
            y[complete],
            color=colors[complete],
            edgecolors="black",
            linewidths=0.7,
            s=38,
            alpha=point_alpha[complete],
            zorder=4,
        )
        if np.any(~complete):
            ax.scatter(
                x[~complete],
                y[~complete],
                marker="D",
                color=colors[~complete],
                edgecolors="black",
                linewidths=0.7,
                s=42,
                alpha=point_alpha[~complete],
                zorder=5,
            )

        curve_handles.append(
            mlines.Line2D(
                [], [], color=base_color, linestyle=":", linewidth=2.0, label=label
            )
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    ax.set_xscale("log")
    if all_x:
        ax.set_xlim(float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
    if all_y:
        ax.set_ylim(*_loss_limits(np.array(all_y, dtype=float)))
    ax.set_xlabel("Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Cross Entropy", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.91,
        "IsoFLOP\nBudget",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=curve_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.80),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 6,
        handlelength=2.0,
        labelspacing=0.34,
    )
    legend_ax.text(
        0.5,
        0.29,
        "Number of\nLayers",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 5,
        va="center",
        ha="center",
    )
    colorbar_ax = legend_ax.inset_axes([0.18, 0.19, 0.64, 0.045])
    mappable = mpl.cm.ScalarMappable(norm=depth_norm, cmap=depth_display_cmap)
    colorbar = fig.colorbar(mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 5, length=0)
    colorbar.outline.set_visible(False)
    colorbar.set_ticks([depth_min, depth_max])
    colorbar.set_ticklabels([str(int(depth_min)), str(int(depth_max))])
    colorbar.update_ticks()
    legend_ax.text(
        0.5,
        0.06,
        "diamonds:\n<99% complete",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 5,
        va="center",
        ha="center",
    )

    fig.suptitle(title, fontsize=FONT_SIZE_TITLE, x=0.48, y=0.96, ha="center")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def create_depth_parabolas_by_isoflop_budget_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _filter_analysis_fractions(df, fractions)
    if len(fractions) == 0:
        raise ValueError("No isoFLOP fractions found in prepared data")

    depth_values = df["target_depth"].drop_nulls().unique().sort().to_list()
    if len(depth_values) == 0:
        raise ValueError("No target_depth values found in prepared data")

    depth_norm = mpl.colors.Normalize(
        vmin=float(min(depth_values)), vmax=float(max(depth_values))
    )
    depth_cmap = plt.colormaps["viridis"]
    depth_colors = {
        int(depth): depth_cmap(0.10 + 0.80 * float(depth_norm(float(depth))))
        for depth in depth_values
    }

    ncols = 3
    nrows = int(np.ceil((len(fractions) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.7 * nrows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()
    plot_axes = flat_axes[: len(fractions)]
    legend_ax = flat_axes[len(fractions)]

    all_x = df["non_embedding_params"].to_numpy().astype(float)
    legend_handles: list[mlines.Line2D] = []

    for ax, fraction in zip(plot_axes, fractions, strict=False):
        panel_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        panel_y: list[float] = []
        for depth in depth_values:
            depth_df = panel_df.filter(pl.col("target_depth") == int(depth)).sort(
                "non_embedding_params"
            )
            if depth_df.is_empty():
                continue

            x = depth_df["non_embedding_params"].to_numpy().astype(float)
            y = depth_df["loss"].to_numpy().astype(float)
            completion = depth_df["completion_fraction"].to_numpy().astype(float)
            complete = completion >= COMPLETE_POINT_THRESHOLD
            color = depth_colors[int(depth)]

            fit = _fit_log_parameter_parabola(x[complete], y[complete])
            if fit is not None:
                x_fit, y_fit = fit
                ax.plot(x_fit, y_fit, color=color, linewidth=1.45, alpha=0.86, zorder=2)

            ax.scatter(
                x[complete],
                y[complete],
                color=color,
                edgecolors="black",
                linewidths=0.55,
                s=24,
                alpha=0.88,
                zorder=4,
            )
            if np.any(~complete):
                ax.scatter(
                    x[~complete],
                    y[~complete],
                    marker="D",
                    color=color,
                    edgecolors="black",
                    linewidths=0.55,
                    s=28,
                    alpha=np.clip(
                        completion[~complete] / COMPLETE_POINT_THRESHOLD, 0.20, 0.72
                    ),
                    zorder=5,
                )

            panel_y.extend(y.tolist())

        budget = float(panel_df["analysis_training_flops"].drop_nulls().first())
        ax.set_title(
            f"{_format_flops_value(budget)} FLOPs",
            fontsize=FONT_SIZE_LABELS - 7,
            y=1.02,
        )
        ax.set_xscale("log")
        if panel_y:
            ax.set_ylim(*_loss_limits(np.array(panel_y, dtype=float)))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(
            True,
            which="major",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    if len(all_x) > 0:
        x_limits = (float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
        for ax in plot_axes:
            ax.set_xlim(*x_limits)

    for ax in flat_axes[len(fractions) + 1 :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Non-embedding Parameters", fontsize=FONT_SIZE_LABELS - 4)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel("Cross Entropy", fontsize=FONT_SIZE_LABELS - 4)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Depth",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    for depth in depth_values:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=depth_colors[int(depth)],
                linewidth=2.2,
                label=str(int(depth)),
            )
        )
    legend_ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 5,
        handlelength=1.45,
        columnspacing=0.80,
        labelspacing=0.28,
        ncol=2,
    )
    legend_ax.text(
        0.5,
        0.13,
        "diamonds: <99% complete\nfits use complete points only",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Depth Parabolas at Fixed Compute",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=0.98,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.91, bottom=0.08, wspace=0.26, hspace=0.42
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_depth_connected_by_isoflop_budget_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _filter_analysis_fractions(df, fractions)
    if len(fractions) == 0:
        raise ValueError("No isoFLOP fractions found in prepared data")

    depth_values = df["target_depth"].drop_nulls().unique().sort().to_list()
    if len(depth_values) == 0:
        raise ValueError("No target_depth values found in prepared data")

    depth_norm = mpl.colors.Normalize(
        vmin=float(min(depth_values)), vmax=float(max(depth_values))
    )
    depth_cmap = plt.colormaps["viridis"]
    depth_colors = {
        int(depth): depth_cmap(0.10 + 0.80 * float(depth_norm(float(depth))))
        for depth in depth_values
    }

    ncols = 3
    nrows = int(np.ceil((len(fractions) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.7 * nrows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()
    plot_axes = flat_axes[: len(fractions)]
    legend_ax = flat_axes[len(fractions)]

    all_x = df["non_embedding_params"].to_numpy().astype(float)
    legend_handles: list[mlines.Line2D] = []

    for ax, fraction in zip(plot_axes, fractions, strict=False):
        panel_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        panel_y: list[float] = []
        for depth in depth_values:
            depth_df = panel_df.filter(pl.col("target_depth") == int(depth)).sort(
                "non_embedding_params"
            )
            if depth_df.is_empty():
                continue

            x = depth_df["non_embedding_params"].to_numpy().astype(float)
            y = depth_df["loss"].to_numpy().astype(float)
            completion = depth_df["completion_fraction"].to_numpy().astype(float)
            complete = completion >= COMPLETE_POINT_THRESHOLD
            color = depth_colors[int(depth)]

            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.35,
                alpha=0.74,
                zorder=2,
            )
            ax.scatter(
                x[complete],
                y[complete],
                color=color,
                edgecolors="black",
                linewidths=0.55,
                s=24,
                alpha=0.88,
                zorder=4,
            )
            if np.any(~complete):
                ax.scatter(
                    x[~complete],
                    y[~complete],
                    marker="D",
                    color=color,
                    edgecolors="black",
                    linewidths=0.55,
                    s=28,
                    alpha=np.clip(
                        completion[~complete] / COMPLETE_POINT_THRESHOLD, 0.20, 0.72
                    ),
                    zorder=5,
                )

            panel_y.extend(y.tolist())

        budget = float(panel_df["analysis_training_flops"].drop_nulls().first())
        ax.set_title(
            f"{_format_flops_value(budget)} FLOPs",
            fontsize=FONT_SIZE_LABELS - 7,
            y=1.02,
        )
        ax.set_xscale("log")
        if panel_y:
            ax.set_ylim(*_loss_limits(np.array(panel_y, dtype=float)))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(
            True,
            which="major",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    if len(all_x) > 0:
        x_limits = (float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
        for ax in plot_axes:
            ax.set_xlim(*x_limits)

    for ax in flat_axes[len(fractions) + 1 :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Non-embedding Parameters", fontsize=FONT_SIZE_LABELS - 4)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel("Cross Entropy", fontsize=FONT_SIZE_LABELS - 4)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Depth",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    for depth in depth_values:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=depth_colors[int(depth)],
                linewidth=2.2,
                label=str(int(depth)),
            )
        )
    legend_ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 5,
        handlelength=1.45,
        columnspacing=0.80,
        labelspacing=0.28,
        ncol=2,
    )
    legend_ax.text(
        0.5,
        0.13,
        "diamonds: <99% complete\nlines connect observed points",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    title_y = 0.965 if nrows == 2 else 0.98
    top = 0.84 if nrows == 2 else 0.91
    fig.suptitle(
        "Depth Lines at Fixed Compute",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=title_y,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=top, bottom=0.08, wspace=0.26, hspace=0.42
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_loss_vs_depth_width_ratio_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    fractions: list[float] | None = None,
    model_size_cmap: str = "viridis",
    ncols: int = 3,
    ylabel: str = "Cross Entropy",
    title: str = "Loss by Depth/Width Ratio",
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _with_depth_width_ratio(_filter_analysis_fractions(df, fractions))
    if len(fractions) == 0:
        raise ValueError("No isoFLOP fractions found in prepared data")

    size_values = df["target_non_embedding_params"].to_numpy().astype(float)
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(size_values)), vmax=float(np.max(size_values))
    )
    cmap = plt.colormaps[model_size_cmap]

    nrows = int(np.ceil((len(fractions) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.7 * nrows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()
    plot_axes = flat_axes[: len(fractions)]
    legend_ax = flat_axes[len(fractions)]
    all_x = df["depth_width_ratio"].to_numpy().astype(float)

    for ax, fraction in zip(plot_axes, fractions, strict=False):
        panel_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        ).sort(["target_non_embedding_params", "depth_width_ratio"])
        x = panel_df["depth_width_ratio"].to_numpy().astype(float)
        y = panel_df["loss"].to_numpy().astype(float)
        sizes = panel_df["target_non_embedding_params"].to_numpy().astype(float)
        completion = panel_df["completion_fraction"].to_numpy().astype(float)
        complete = completion >= COMPLETE_POINT_THRESHOLD
        colors = cmap(norm(sizes))

        ax.scatter(
            x[complete],
            y[complete],
            color=colors[complete],
            edgecolors="black",
            linewidths=0.55,
            s=28,
            alpha=0.90,
            zorder=4,
        )
        if np.any(~complete):
            ax.scatter(
                x[~complete],
                y[~complete],
                marker="D",
                color=colors[~complete],
                edgecolors="black",
                linewidths=0.55,
                s=32,
                alpha=np.clip(
                    completion[~complete] / COMPLETE_POINT_THRESHOLD, 0.20, 0.72
                ),
                zorder=5,
            )

        budget = float(panel_df["analysis_training_flops"].drop_nulls().first())
        ax.set_title(
            f"{_format_flops_value(budget)} FLOPs",
            fontsize=FONT_SIZE_LABELS - 7,
            y=1.02,
        )
        ax.set_xscale("log")
        ax.set_ylim(*_loss_limits(y))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:.3g}")
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(
            True,
            which="major",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    if len(all_x) > 0:
        x_limits = (float(np.min(all_x)) / 1.35, float(np.max(all_x)) * 1.35)
        for ax in plot_axes:
            ax.set_xlim(*x_limits)

    for ax in flat_axes[len(fractions) + 1 :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Depth / Width", fontsize=FONT_SIZE_LABELS - 4)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS - 4)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.86,
        "Model Size",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    cax = legend_ax.inset_axes([0.16, 0.48, 0.68, 0.08])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    colorbar_ticks = [1e6, 1e7, 1e8, 1e9]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([_format_param_count(value) for value in colorbar_ticks])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 1, length=0)
    colorbar.outline.set_linewidth(0.9)
    legend_ax.legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="#1f1a16",
                label="<99% complete",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 7,
    )

    title_y = 0.965 if nrows == 2 else 0.98
    top = 0.84 if nrows == 2 else 0.91
    fig.suptitle(
        title,
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=title_y,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=top, bottom=0.08, wspace=0.26, hspace=0.42
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def create_isoparam_ratio_sweeps_by_isoflop_budget_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _with_depth_width_ratio(_filter_analysis_fractions(df, fractions))
    if len(fractions) == 0:
        raise ValueError("No isoFLOP fractions found in prepared data")

    param_values = (
        df["target_non_embedding_params"].drop_nulls().unique().sort().to_list()
    )
    if len(param_values) == 0:
        raise ValueError("No target_non_embedding_params values found in prepared data")

    param_norm = mpl.colors.LogNorm(
        vmin=float(min(param_values)), vmax=float(max(param_values))
    )
    param_cmap = plt.colormaps["viridis"]
    param_colors = {
        float(value): param_cmap(param_norm(float(value))) for value in param_values
    }

    ncols = 3
    nrows = int(np.ceil((len(fractions) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.7 * nrows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()
    plot_axes = flat_axes[: len(fractions)]
    legend_ax = flat_axes[len(fractions)]

    all_x = df["depth_width_ratio"].to_numpy().astype(float)
    all_y = df["loss"].to_numpy().astype(float)

    for ax, fraction in zip(plot_axes, fractions, strict=False):
        panel_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        panel_y: list[float] = []
        for param_value in param_values:
            param_df = panel_df.filter(
                pl.col("target_non_embedding_params") == float(param_value)
            ).sort("depth_width_ratio")
            if param_df.is_empty():
                continue

            x = param_df["depth_width_ratio"].to_numpy().astype(float)
            y = param_df["loss"].to_numpy().astype(float)
            completion = param_df["completion_fraction"].to_numpy().astype(float)
            complete = completion >= COMPLETE_POINT_THRESHOLD
            color = param_colors[float(param_value)]

            fit = _fit_log_x_parabola(x[complete], y[complete])
            if fit is not None:
                x_fit, y_fit = fit
                ax.plot(x_fit, y_fit, color=color, linewidth=1.45, alpha=0.78, zorder=2)
                min_index = int(np.argmin(y_fit))
                ax.scatter(
                    [x_fit[min_index]],
                    [y_fit[min_index]],
                    marker="v",
                    color=color,
                    edgecolors="black",
                    linewidths=0.55,
                    s=32,
                    alpha=0.90,
                    zorder=6,
                )
            else:
                ax.plot(x, y, color=color, linewidth=1.0, alpha=0.40, zorder=2)

            ax.scatter(
                x[complete],
                y[complete],
                color=color,
                edgecolors="black",
                linewidths=0.50,
                s=21,
                alpha=0.86,
                zorder=4,
            )
            if np.any(~complete):
                ax.scatter(
                    x[~complete],
                    y[~complete],
                    marker="D",
                    color=color,
                    edgecolors="black",
                    linewidths=0.50,
                    s=25,
                    alpha=np.clip(
                        completion[~complete] / COMPLETE_POINT_THRESHOLD, 0.20, 0.70
                    ),
                    zorder=5,
                )

            panel_y.extend(y.tolist())

        budget = float(panel_df["analysis_training_flops"].drop_nulls().first())
        ax.set_title(
            f"{_format_flops_value(budget)} FLOPs",
            fontsize=FONT_SIZE_LABELS - 7,
            y=1.02,
        )
        ax.set_xscale("log")
        if panel_y:
            ax.set_ylim(*_loss_limits(np.array(panel_y, dtype=float)))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:.3g}")
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(
            True,
            which="major",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    if len(all_x) > 0:
        x_limits = (float(np.min(all_x)) / 1.35, float(np.max(all_x)) * 1.35)
        for ax in plot_axes:
            ax.set_xlim(*x_limits)
    if len(all_y) == 0:
        raise ValueError("No loss values found for iso-param ratio sweeps")

    for ax in flat_axes[len(fractions) + 1 :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Depth / Width", fontsize=FONT_SIZE_LABELS - 4)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel("Cross Entropy", fontsize=FONT_SIZE_LABELS - 4)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.86,
        "Iso-param\nSize",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    cax = legend_ax.inset_axes([0.20, 0.35, 0.18, 0.36])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=param_norm, cmap=param_cmap), cax=cax
    )
    colorbar_ticks = [1e6, 1e7, 1e8, 1e9]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([_format_param_count(value) for value in colorbar_ticks])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 1, length=0)
    colorbar.outline.set_linewidth(0.9)
    legend_ax.text(
        0.5,
        0.13,
        "triangles: fitted minima\ndiamonds: <99% complete",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Iso-param Ratio Sweeps at Fixed Compute",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=0.98,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.91, bottom=0.08, wspace=0.26, hspace=0.42
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_winner_ratio_vs_params_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _with_depth_width_ratio(_filter_analysis_fractions(df, fractions)).filter(
        pl.col("completion_fraction") >= COMPLETE_POINT_THRESHOLD
    )
    if df.is_empty():
        raise ValueError("No complete isoFLOP points available for winner-ratio plot")

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.3, 1.35], "wspace": 0.28},
    )
    curve_handles: list[mlines.Line2D] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for index, fraction in enumerate(fractions):
        panel_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        if panel_df.is_empty():
            continue
        winners = (
            panel_df.sort(["target_non_embedding_params", "loss"])
            .group_by("target_non_embedding_params", maintain_order=True)
            .first()
            .sort("target_non_embedding_params")
        )
        x = winners["target_non_embedding_params"].to_numpy().astype(float)
        y = winners["depth_width_ratio"].to_numpy().astype(float)
        depths = winners["target_depth"].to_numpy().astype(int)
        color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        label = f"{_format_flops_value(float(winners['analysis_training_flops'].drop_nulls().first()))} FLOPs"
        ax.plot(x, y, color=color, linewidth=1.7, alpha=0.82, zorder=2)
        ax.scatter(
            x, y, color=color, edgecolors="black", linewidths=0.65, s=34, zorder=4
        )
        for x_value, y_value, depth in zip(x, y, depths, strict=False):
            ax.text(
                x_value,
                y_value * 1.10,
                str(depth),
                fontsize=FONT_SIZE_TICKS - 4,
                ha="center",
                va="bottom",
                color="#2f2a24",
                zorder=6,
            )
        curve_handles.append(
            mlines.Line2D([], [], color=color, linewidth=2.0, label=label)
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_x:
        ax.set_xlim(float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
    if all_y:
        ax.set_ylim(float(np.min(all_y)) / 1.70, float(np.max(all_y)) * 1.90)
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Winning Depth / Width", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "IsoFLOP\nBudget",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=curve_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 6,
        handlelength=1.8,
        labelspacing=0.35,
    )
    legend_ax.text(
        0.5,
        0.10,
        "point labels:\nwinning depth\ncomplete points only",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Winning Depth/Width Ratio",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def create_depth_width_ratio_grid_plot(df: pl.DataFrame, output_path: Path) -> None:
    grid_df = (
        _with_depth_width_ratio(df)
        .select(
            [
                "target_non_embedding_params",
                "target_depth",
                "model.d_model",
                "depth_width_ratio",
            ]
        )
        .unique()
    )
    depth_values = grid_df["target_depth"].drop_nulls().unique().sort().to_list()
    depth_norm = mpl.colors.Normalize(
        vmin=float(min(depth_values)), vmax=float(max(depth_values))
    )
    depth_cmap = plt.colormaps["viridis"]

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.3, 1.35], "wspace": 0.28},
    )
    handles: list[mlines.Line2D] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for depth in depth_values:
        depth_df = grid_df.filter(pl.col("target_depth") == int(depth)).sort(
            "target_non_embedding_params"
        )
        x = depth_df["target_non_embedding_params"].to_numpy().astype(float)
        y = depth_df["depth_width_ratio"].to_numpy().astype(float)
        color = depth_cmap(0.10 + 0.80 * float(depth_norm(float(depth))))
        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.84, zorder=2)
        ax.scatter(
            x,
            y,
            color=color,
            edgecolors="black",
            linewidths=0.55,
            s=30,
            alpha=0.90,
            zorder=4,
        )
        handles.append(
            mlines.Line2D([], [], color=color, linewidth=2.2, label=str(int(depth)))
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
    ax.set_ylim(float(np.min(all_y)) / 1.50, float(np.max(all_y)) * 1.50)
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Depth / Width", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Fixed\nDepth",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 5,
        handlelength=1.45,
        columnspacing=0.80,
        labelspacing=0.28,
        ncol=2,
    )
    legend_ax.text(
        0.5,
        0.12,
        "constant depth\nimplies falling ratio\nas width grows",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Depth/Width Ratio Grid", fontsize=FONT_SIZE_TITLE, x=0.48, y=0.96, ha="center"
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


SURFACE_FEATURES = (
    "intercept",
    "log_params",
    "log_compute",
    "log_ratio",
    "log_params^2",
    "log_compute^2",
    "log_ratio^2",
    "log_params*log_compute",
    "log_params*log_ratio",
    "log_compute*log_ratio",
)


def _surface_design_matrix(
    log_params: np.ndarray, log_compute: np.ndarray, log_ratio: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            np.ones_like(log_params),
            log_params,
            log_compute,
            log_ratio,
            log_params**2,
            log_compute**2,
            log_ratio**2,
            log_params * log_compute,
            log_params * log_ratio,
            log_compute * log_ratio,
        ]
    )


def _surface_fit_df(df: pl.DataFrame) -> pl.DataFrame:
    return _with_depth_width_ratio(df).filter(
        pl.col("completion_fraction") >= COMPLETE_POINT_THRESHOLD
    )


def _fit_surface_model_from_fit_df(fit_df: pl.DataFrame) -> dict[str, object]:
    if fit_df.height < len(SURFACE_FEATURES):
        raise ValueError(
            "Not enough complete points to fit the depth/width response surface"
        )

    log_params = np.log10(
        fit_df["target_non_embedding_params"].to_numpy().astype(float)
    )
    log_compute = np.log10(fit_df["analysis_training_flops"].to_numpy().astype(float))
    log_ratio = np.log10(fit_df["depth_width_ratio"].to_numpy().astype(float))
    y = fit_df["loss"].to_numpy().astype(float)
    means = np.array([log_params.mean(), log_compute.mean(), log_ratio.mean()])
    scales = np.array([log_params.std(), log_compute.std(), log_ratio.std()])
    scales[scales == 0] = 1.0
    z_params, z_compute, z_ratio = (
        (log_params - means[0]) / scales[0],
        (log_compute - means[1]) / scales[1],
        (log_ratio - means[2]) / scales[2],
    )
    design = _surface_design_matrix(z_params, z_compute, z_ratio)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "coefficients": coefficients,
        "means": means,
        "scales": scales,
        "r_squared": r_squared,
        "n_train": int(fit_df.height),
        "ratio_min": float(
            np.min(fit_df["depth_width_ratio"].to_numpy().astype(float))
        ),
        "ratio_max": float(
            np.max(fit_df["depth_width_ratio"].to_numpy().astype(float))
        ),
    }


def _fit_surface_model(df: pl.DataFrame) -> dict[str, object]:
    return _fit_surface_model_from_fit_df(_surface_fit_df(df))


def _surface_predict(
    model: dict[str, object], params: np.ndarray, compute: np.ndarray, ratio: np.ndarray
) -> np.ndarray:
    means = model["means"]
    scales = model["scales"]
    assert isinstance(means, np.ndarray)
    assert isinstance(scales, np.ndarray)
    log_params = (np.log10(params) - means[0]) / scales[0]
    log_compute = (np.log10(compute) - means[1]) / scales[1]
    log_ratio = (np.log10(ratio) - means[2]) / scales[2]
    design = _surface_design_matrix(log_params, log_compute, log_ratio)
    coefficients = model["coefficients"]
    assert isinstance(coefficients, np.ndarray)
    return design @ coefficients


def _surface_analytic_best_ratios(
    model: dict[str, object], params: np.ndarray, compute: float | np.ndarray
) -> np.ndarray:
    """Return the unconstrained log-ratio minimum of a quadratic surface."""
    coefficients = model["coefficients"]
    means = model["means"]
    scales = model["scales"]
    assert isinstance(coefficients, np.ndarray)
    assert isinstance(means, np.ndarray)
    assert isinstance(scales, np.ndarray)

    ratio_quadratic = float(coefficients[6])
    if ratio_quadratic <= 0:
        raise ValueError("The fitted surface is not convex in log depth/width ratio")

    params_values, compute_values = np.broadcast_arrays(
        np.asarray(params, dtype=float), np.asarray(compute, dtype=float)
    )
    z_params = (np.log10(params_values) - means[0]) / scales[0]
    z_compute = (np.log10(compute_values) - means[1]) / scales[1]
    z_ratio = -(
        coefficients[3] + coefficients[8] * z_params + coefficients[9] * z_compute
    ) / (2.0 * ratio_quadratic)
    return 10 ** (means[2] + scales[2] * z_ratio)


def write_surface_coefficients(
    df: pl.DataFrame, output_path: Path
) -> dict[str, object]:
    model = _fit_surface_model(df)
    ensure_parent_dir(output_path)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["feature", "coefficient", "n_train", "r_squared"]
        )
        writer.writeheader()
        coefficients = model["coefficients"]
        assert isinstance(coefficients, np.ndarray)
        for feature, coefficient in zip(SURFACE_FEATURES, coefficients, strict=True):
            writer.writerow(
                {
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "n_train": model["n_train"],
                    "r_squared": model["r_squared"],
                }
            )
    return model


def _surface_predictions_for_df(
    model: dict[str, object], df: pl.DataFrame
) -> np.ndarray:
    return _surface_predict(
        model,
        df["target_non_embedding_params"].to_numpy().astype(float),
        df["analysis_training_flops"].to_numpy().astype(float),
        df["depth_width_ratio"].to_numpy().astype(float),
    )


def _prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def _write_surface_diagnostics(
    rows: list[dict[str, object]], output_path: Path
) -> None:
    ensure_parent_dir(output_path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _take_rows(df: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    return df[indices.astype(int).tolist()]


def create_surface_fit_diagnostics_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    figsize: tuple[float, float] = (15.2, 5.2),
    crop_output: bool = True,
    observed_loss_label: str = "Observed Loss",
    fitted_loss_label: str = "Fitted Loss",
    residual_label: str = "Residual",
    title: str = "Surface Fit Diagnostics",
    compact_legend: bool = False,
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> dict[str, float]:
    fit_df = _surface_fit_df(df)
    y = fit_df["loss"].to_numpy().astype(float)
    predicted = _surface_predictions_for_df(model, fit_df)
    residual = y - predicted
    metrics = _prediction_metrics(y, predicted)
    compute_values = fit_df["analysis_training_flops"].to_numpy().astype(float)
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(compute_values)), vmax=float(np.max(compute_values))
    )
    cmap = plt.colormaps["viridis"]

    apply_plot_style()
    if compact_legend:
        fig, (ax_pred, ax_resid) = plt.subplots(1, 2, figsize=figsize)
        legend_ax = None
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1.0, 1.0, 0.42]},
        )
        ax_pred, ax_resid, legend_ax = axes
    colors = cmap(norm(compute_values))
    ax_pred.scatter(
        y,
        predicted,
        color=colors,
        edgecolors="#1f1a16",
        linewidths=0.35,
        s=22,
        alpha=0.78,
        zorder=3,
    )
    limits = (
        float(min(np.min(y), np.min(predicted))),
        float(max(np.max(y), np.max(predicted))),
    )
    pad = 0.04 * (limits[1] - limits[0])
    ax_pred.plot(
        [limits[0] - pad, limits[1] + pad],
        [limits[0] - pad, limits[1] + pad],
        color="#2f2a24",
        linestyle="--",
        linewidth=1.2,
    )
    ax_pred.set_xlim(limits[0] - pad, limits[1] + pad)
    ax_pred.set_ylim(limits[0] - pad, limits[1] + pad)
    ax_pred.set_xlabel(observed_loss_label, fontsize=FONT_SIZE_LABELS - 2)
    ax_pred.set_ylabel(fitted_loss_label, fontsize=FONT_SIZE_LABELS - 2)
    ax_pred.grid(
        True,
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.7,
    )
    remove_bounding_box(ax_pred, fontsize_ticks=FONT_SIZE_TICKS)

    ax_resid.scatter(
        predicted,
        residual,
        color=colors,
        edgecolors="#1f1a16",
        linewidths=0.35,
        s=22,
        alpha=0.78,
        zorder=3,
    )
    ax_resid.axhline(0.0, color="#2f2a24", linestyle="--", linewidth=1.2)
    ax_resid.set_xlabel(fitted_loss_label, fontsize=FONT_SIZE_LABELS - 2)
    ax_resid.set_ylabel(residual_label, fontsize=FONT_SIZE_LABELS - 2)
    ax_resid.grid(
        True,
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.7,
    )
    remove_bounding_box(ax_resid, fontsize_ticks=FONT_SIZE_TICKS)

    if compact_legend:
        ax_pred.text(
            0.05,
            0.93,
            f"n={fit_df.height}\nR2={metrics['r_squared']:.3f}\nRMSE={metrics['rmse']:.3f}",
            transform=ax_pred.transAxes,
            fontsize=FONT_SIZE_TICKS,
            va="top",
            ha="left",
        )
        cax = ax_resid.inset_axes([0.50, 0.12, 0.38, 0.035])
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
        )
        colorbar.ax.set_title("FLOPs", fontsize=FONT_SIZE_TICKS - 1, pad=4)
        colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 2, length=0, pad=2)
        colorbar.outline.set_linewidth(0.8)
    else:
        assert legend_ax is not None
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        _style_panel_box(legend_ax, rounded=True)
        legend_ax.text(
            0.5,
            0.86,
            "Training\nFLOPs",
            transform=legend_ax.transAxes,
            fontsize=FONT_SIZE_LEGEND,
            va="center",
            ha="center",
        )
        cax = legend_ax.inset_axes([0.20, 0.45, 0.20, 0.25])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 1, length=0)
        colorbar.outline.set_linewidth(0.8)
        legend_ax.text(
            0.5,
            0.19,
            f"n={fit_df.height}\nR2={metrics['r_squared']:.3f}\nRMSE={metrics['rmse']:.3f}",
            transform=legend_ax.transAxes,
            fontsize=FONT_SIZE_LEGEND - 6,
            va="center",
            ha="center",
        )

    fig.suptitle(
        title,
        fontsize=FONT_SIZE_TITLE,
        x=0.50 if compact_legend else 0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.84,
        bottom=0.15,
        wspace=0.38 if compact_legend else 0.30,
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )
    return metrics


def create_surface_fit_window_comparison_plot(
    df: pl.DataFrame,
    early_df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    figsize: tuple[float, float] = (10.8, 16.0),
    crop_output: bool = True,
    observed_loss_label: str = "Observed Loss",
    fitted_loss_label: str = "Fitted Loss",
    residual_label: str = "Residual",
    title: str = "Surface Fit Diagnostics",
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> dict[str, dict[str, float]]:
    """Compare the fitted surface in its analysis range and an earlier window."""
    analysis_df = _surface_fit_df(df)
    warmup_df = _surface_fit_df(early_df)
    all_slices_df = pl.concat(
        [analysis_df, warmup_df],
        how="diagonal_relaxed",
    )
    all_slices_model = _fit_surface_model_from_fit_df(all_slices_df)
    rows = (
        ("In-sample fit", analysis_df, model, "Oranges"),
        ("Warm-up extrapolation", warmup_df, model, "Blues"),
        ("All-slice refit", all_slices_df, all_slices_model, "viridis"),
    )

    apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=figsize, squeeze=False)
    metrics_by_window: dict[str, dict[str, float]] = {}

    for row_index, (row_title, plot_df, row_model, cmap_name) in enumerate(rows):
        ax_pred, ax_resid = axes[row_index]
        observed = plot_df["loss"].to_numpy().astype(float)
        predicted = _surface_predictions_for_df(row_model, plot_df)
        residual = observed - predicted
        metrics = _prediction_metrics(observed, predicted)
        metrics_by_window[row_title] = metrics
        base_cmap = plt.colormaps[cmap_name]
        cmap = (
            mpl.colors.LinearSegmentedColormap.from_list(
                f"{cmap_name}_surface_window",
                base_cmap(np.linspace(0.32, 0.92, 256)),
            )
            if cmap_name in {"Oranges", "Blues"}
            else base_cmap
        )

        compute_values = plot_df["analysis_training_flops"].to_numpy().astype(float)
        norm = mpl.colors.LogNorm(
            vmin=float(np.min(compute_values)),
            vmax=float(np.max(compute_values)),
        )
        colors = cmap(norm(compute_values))

        ax_pred.scatter(
            observed,
            predicted,
            color=colors,
            edgecolors="#1f1a16",
            linewidths=0.35,
            s=22,
            alpha=0.78,
            zorder=3,
        )
        limits = (
            float(min(np.min(observed), np.min(predicted))),
            float(max(np.max(observed), np.max(predicted))),
        )
        pad = 0.04 * (limits[1] - limits[0])
        ax_pred.plot(
            [limits[0] - pad, limits[1] + pad],
            [limits[0] - pad, limits[1] + pad],
            color="#2f2a24",
            linestyle="--",
            linewidth=1.2,
        )
        ax_pred.set_xlim(limits[0] - pad, limits[1] + pad)
        ax_pred.set_ylim(limits[0] - pad, limits[1] + pad)
        ax_pred.set_xlabel(observed_loss_label, fontsize=FONT_SIZE_LABELS - 2)
        ax_pred.set_ylabel(fitted_loss_label, fontsize=FONT_SIZE_LABELS - 2)
        ax_pred.set_title(
            row_title,
            fontsize=FONT_SIZE_LABELS - 2,
            loc="left",
            pad=8,
        )
        ax_pred.grid(
            True,
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.7,
        )
        remove_bounding_box(ax_pred, fontsize_ticks=FONT_SIZE_TICKS)
        ax_pred.text(
            0.05,
            0.93,
            (
                f"n={plot_df.height}\n"
                f"R2={metrics['r_squared']:.3f}\n"
                f"RMSE={metrics['rmse']:.3f}"
            ),
            transform=ax_pred.transAxes,
            fontsize=FONT_SIZE_TICKS,
            va="top",
            ha="left",
        )

        ax_resid.scatter(
            predicted,
            residual,
            color=colors,
            edgecolors="#1f1a16",
            linewidths=0.35,
            s=22,
            alpha=0.78,
            zorder=3,
        )
        ax_resid.axhline(0.0, color="#2f2a24", linestyle="--", linewidth=1.2)
        ax_resid.set_xlabel(fitted_loss_label, fontsize=FONT_SIZE_LABELS - 2)
        ax_resid.set_ylabel(residual_label, fontsize=FONT_SIZE_LABELS - 2)
        ax_resid.set_title(
            "Residuals",
            fontsize=FONT_SIZE_LABELS - 2,
            loc="left",
            pad=8,
        )
        ax_resid.grid(
            True,
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.7,
        )
        remove_bounding_box(ax_resid, fontsize_ticks=FONT_SIZE_TICKS)

        cax = ax_resid.inset_axes([0.50, 0.12, 0.38, 0.035])
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
        )
        colorbar.ax.set_title("FLOPs", fontsize=FONT_SIZE_TICKS - 1, pad=4)
        colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 2, length=0, pad=2)
        colorbar.outline.set_linewidth(0.8)

    fig.suptitle(title, fontsize=FONT_SIZE_TITLE, x=0.50, y=0.98, ha="center")
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.93,
        bottom=0.055,
        hspace=0.52,
        wspace=0.38,
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )
    return metrics_by_window


def _surface_cv_predictions(
    fit_df: pl.DataFrame, *, seed: int = 7, n_folds: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(fit_df.height)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    random_pred = np.full(fit_df.height, np.nan, dtype=float)
    for fold in folds:
        train_idx = np.setdiff1d(indices, fold, assume_unique=False)
        model = _fit_surface_model_from_fit_df(_take_rows(fit_df, train_idx))
        random_pred[fold] = _surface_predictions_for_df(model, _take_rows(fit_df, fold))

    compute_pred = np.full(fit_df.height, np.nan, dtype=float)
    fractions = fit_df["analysis_fraction"].unique().sort().to_list()
    for fraction in fractions:
        test_mask = (
            np.abs(
                fit_df["analysis_fraction"].to_numpy().astype(float) - float(fraction)
            )
            < 1e-9
        )
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        model = _fit_surface_model_from_fit_df(_take_rows(fit_df, train_idx))
        compute_pred[test_idx] = _surface_predictions_for_df(
            model, _take_rows(fit_df, test_idx)
        )
    return random_pred, compute_pred


def create_surface_cv_diagnostics_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    diagnostics_csv: Path | None = None,
    model: dict[str, object] | None = None,
    strategies_to_plot: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (15.2, 5.2),
    crop_output: bool = True,
    y_min: float | None = None,
    match_xy_major_ticks: bool = False,
    observed_loss_label: str = "Observed loss (model-specific units)",
    predicted_loss_label: str = "Predicted loss (model-specific units)",
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> list[dict[str, object]]:
    fit_df = _surface_fit_df(df)
    y = fit_df["loss"].to_numpy().astype(float)
    if model is None:
        model = _fit_surface_model_from_fit_df(fit_df)
    in_sample = _surface_predictions_for_df(model, fit_df)
    random_pred, compute_pred = _surface_cv_predictions(fit_df)
    compute_values = fit_df["analysis_training_flops"].to_numpy().astype(float)
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(compute_values)), vmax=float(np.max(compute_values))
    )
    cmap = plt.colormaps["viridis"]
    colors = cmap(norm(compute_values))
    rows: list[dict[str, object]] = []
    for label, pred in [
        ("in_sample", in_sample),
        ("random_5fold", random_pred),
        ("leave_compute_out", compute_pred),
    ]:
        metrics = _prediction_metrics(y, pred)
        rows.append({"strategy": label, "n": fit_df.height, **metrics})
    if diagnostics_csv is not None:
        _write_surface_diagnostics(rows, diagnostics_csv)

    strategy_values = {
        "in_sample": ("In-sample", in_sample),
        "random_5fold": ("Random 5-fold", random_pred),
        "leave_compute_out": ("Leave compute slice out", compute_pred),
    }
    selected_strategies = strategies_to_plot or tuple(strategy_values)
    strategies = [strategy_values[key] for key in selected_strategies]
    apply_plot_style()
    fig, axes = plt.subplots(
        1, len(strategies), figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()
    limits = (
        float(min(np.min(y), np.nanmin(random_pred), np.nanmin(compute_pred))),
        float(max(np.max(y), np.nanmax(random_pred), np.nanmax(compute_pred))),
    )
    pad = 0.04 * (limits[1] - limits[0])
    row_by_strategy = {row["strategy"]: row for row in rows}
    for ax, (strategy_key, (title, pred)) in zip(
        axes, [(key, strategy_values[key]) for key in selected_strategies], strict=True
    ):
        row = row_by_strategy[strategy_key]
        ax.scatter(
            y,
            pred,
            color=colors,
            edgecolors="#1f1a16",
            linewidths=0.35,
            s=20,
            alpha=0.72,
            zorder=3,
        )
        ax.plot(
            [limits[0] - pad, limits[1] + pad],
            [limits[0] - pad, limits[1] + pad],
            color="#2f2a24",
            linestyle="--",
            linewidth=1.2,
        )
        ax.set_title(title, fontsize=FONT_SIZE_LABELS - 5)
        ax.set_xlabel(observed_loss_label, fontsize=FONT_SIZE_LABELS - 3)
        ax.text(
            0.05,
            0.93,
            f"R2={row['r_squared']:.3f}\nRMSE={row['rmse']:.3f}",
            transform=ax.transAxes,
            fontsize=FONT_SIZE_TICKS,
            va="top",
            ha="left",
        )
        ax.grid(
            True,
            which="major",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
    axes[0].set_ylabel(predicted_loss_label, fontsize=FONT_SIZE_LABELS - 3)
    for ax in axes:
        ax.set_xlim(limits[0] - pad, limits[1] + pad)
        ax.set_ylim(y_min if y_min is not None else limits[0] - pad, limits[1] + pad)
    if match_xy_major_ticks:
        y_lower, y_upper = axes[0].get_ylim()
        common_ticks = [
            tick for tick in axes[0].get_yticks() if y_lower <= tick <= y_upper
        ]
        for ax in axes:
            ax.set_xticks(common_ticks)
            ax.set_yticks(common_ticks)
    cax = axes[-1].inset_axes([0.48, 0.12, 0.38, 0.035])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    colorbar.ax.set_title("FLOPs", fontsize=FONT_SIZE_TICKS - 1, pad=4)
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 2, length=0, pad=2)
    colorbar.outline.set_linewidth(0.8)

    fig.suptitle(
        "Surface Held-out Prediction",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.15, wspace=0.20)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )
    return rows


def create_surface_bootstrap_optimum_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
    n_bootstrap: int = 250,
    seed: int = 11,
    figsize: tuple[float, float] = (10.8, 5.9),
    crop_output: bool = True,
    compact_legend: bool = False,
    legend_width_ratio: float = 1.35,
    y_label: str = "Predicted Optimal Depth / Width",
    enforce_y_axis_units: bool = True,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    selected_fractions = [fractions[0], fractions[len(fractions) // 2], fractions[-1]]
    fit_df = _surface_fit_df(df)
    observed_params = (
        fit_df["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    params_grid = np.geomspace(
        float(np.min(observed_params)), float(np.max(observed_params)), 120
    )
    rng = np.random.default_rng(seed)

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [5.3, legend_width_ratio], "wspace": 0.28},
    )
    handles: list[mlines.Line2D | mpl.patches.Patch] = []
    ratio_min = float(fit_df["depth_width_ratio"].min())
    ratio_max = float(fit_df["depth_width_ratio"].max())
    compute_values = fit_df["analysis_training_flops"].to_numpy().astype(float)
    compute_norm = mpl.colors.LogNorm(
        vmin=float(np.min(compute_values)), vmax=float(np.max(compute_values))
    )
    compute_cmap = plt.colormaps["viridis"]
    for fraction in selected_fractions:
        compute = float(fraction) * target_training_flops
        color = compute_cmap(compute_norm(compute))
        curves: list[np.ndarray] = []
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, fit_df.height, size=fit_df.height)
            try:
                boot_model = _fit_surface_model_from_fit_df(
                    _take_rows(fit_df, sample_idx)
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            try:
                curves.append(
                    _surface_analytic_best_ratios(boot_model, params_grid, compute)
                )
            except ValueError:
                # A non-convex quadratic has no finite optimum in log ratio.
                continue
        if not curves:
            raise ValueError(
                "No convex bootstrap surfaces were available for "
                f"{_format_flops_value(compute)} FLOPs"
            )
        curve_array = np.vstack(curves)
        lower = np.nanpercentile(curve_array, 5, axis=0)
        upper = np.nanpercentile(curve_array, 95, axis=0)
        center = _surface_analytic_best_ratios(model, params_grid, compute)
        lower_outside = lower < ratio_min
        upper_outside = upper > ratio_max
        center_below = center < ratio_min
        center_above = center > ratio_max
        ax.fill_between(
            params_grid,
            np.clip(lower, ratio_min, ratio_max),
            np.clip(upper, ratio_min, ratio_max),
            color=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            params_grid,
            np.where(~(center_below | center_above), center, np.nan),
            color=color,
            linewidth=2.0,
            zorder=4,
        )
        boundary_marker_stride = 6
        for mask, boundary, marker in (
            (lower_outside, ratio_min, "v"),
            (upper_outside, ratio_max, "^"),
            (center_below, ratio_min, "v"),
            (center_above, ratio_max, "^"),
        ):
            marker_indices = np.flatnonzero(mask)[::boundary_marker_stride]
            if marker_indices.size:
                ax.scatter(
                    params_grid[marker_indices],
                    np.full(marker_indices.size, boundary),
                    marker=marker,
                    s=10,
                    color=color,
                    edgecolors="none",
                    zorder=5,
                    clip_on=False,
                )
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=f"{_format_flops_value(compute)} FLOPs",
            )
        )

    for boundary in (ratio_min, ratio_max):
        ax.axhline(
            boundary,
            color="#746d66",
            linestyle=(0, (5.0, 4.0)),
            linewidth=1.15,
            alpha=0.85,
            zorder=3,
        )
    handles.extend(
        [
            mpl.patches.Patch(
                facecolor="#746d66",
                edgecolor="none",
                alpha=0.18,
                label="5-95% interval",
            ),
            mlines.Line2D(
                [],
                [],
                color="#746d66",
                linestyle=(0, (5.0, 4.0)),
                linewidth=1.15,
                label="Observed limits" if compact_legend else "Observed ratio limits",
            ),
            mlines.Line2D(
                [],
                [],
                color="#746d66",
                linestyle="none",
                marker="v",
                markersize=4.5,
                label="Below range" if compact_legend else "Below observed range",
            ),
            mlines.Line2D(
                [],
                [],
                color="#746d66",
                linestyle="none",
                marker="^",
                markersize=4.5,
                label="Above range" if compact_legend else "Above observed range",
            ),
        ]
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    log_padding_factor = 10**0.2
    ax.set_xlim(
        float(np.min(params_grid)) / log_padding_factor,
        float(np.max(params_grid)) * log_padding_factor,
    )
    ax.set_ylim(
        ratio_min / log_padding_factor,
        ratio_max * log_padding_factor,
    )
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.86,
        "Bootstrap\nBands",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    compute_legend = legend_ax.legend(
        handles=handles[: len(selected_fractions)],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.72),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - (6 if compact_legend else 7),
        handlelength=1.8,
        labelspacing=0.38,
    )
    legend_ax.add_artist(compute_legend)
    legend_ax.legend(
        handles=handles[len(selected_fractions) :],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - (6 if compact_legend else 7),
        handlelength=1.8,
        labelspacing=0.38,
    )

    fig.suptitle(
        "Bootstrap Stability of Surface Optimum",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def create_surface_fit_optimal_ratio_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    ratio_grid = np.geomspace(float(model["ratio_min"]), float(model["ratio_max"]), 300)
    observed_params = (
        _with_depth_width_ratio(df)["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
    )
    params_grid = np.geomspace(
        float(np.min(observed_params)), float(np.max(observed_params)), 160
    )

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.3, 1.35], "wspace": 0.28},
    )
    handles: list[mlines.Line2D] = []
    all_y: list[float] = []

    for index, fraction in enumerate(fractions):
        compute = float(fraction) * target_training_flops
        best_ratios: list[float] = []
        for params_value in params_grid:
            params_vector = np.full_like(ratio_grid, params_value, dtype=float)
            compute_vector = np.full_like(ratio_grid, compute, dtype=float)
            predictions = _surface_predict(
                model, params_vector, compute_vector, ratio_grid
            )
            best_ratios.append(float(ratio_grid[int(np.argmin(predictions))]))
        color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        ax.plot(params_grid, best_ratios, color=color, linewidth=1.8, zorder=2)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=f"{_format_flops_value(compute)} FLOPs",
            )
        )
        all_y.extend(best_ratios)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(params_grid)), float(np.max(params_grid)))
    ax.set_ylim(float(np.min(all_y)) / 1.40, float(np.max(all_y)) * 1.40)
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Predicted Optimal Depth / Width", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Surface Fit",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 6,
        handlelength=1.8,
        labelspacing=0.35,
    )
    legend_ax.text(
        0.5,
        0.11,
        f"complete points only\nn={model['n_train']}\nR2={float(model['r_squared']):.3f}",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Fitted Optimal Depth/Width Ratio",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def _surface_best_ratios(
    model: dict[str, object],
    params_grid: np.ndarray,
    compute: float,
    *,
    n_ratio_points: int = 300,
) -> np.ndarray:
    ratio_grid = np.geomspace(
        float(model["ratio_min"]), float(model["ratio_max"]), n_ratio_points
    )
    best_ratios: list[float] = []
    for params_value in params_grid:
        params_vector = np.full_like(ratio_grid, params_value, dtype=float)
        compute_vector = np.full_like(ratio_grid, compute, dtype=float)
        predictions = _surface_predict(model, params_vector, compute_vector, ratio_grid)
        best_ratios.append(float(ratio_grid[int(np.argmin(predictions))]))
    return np.array(best_ratios, dtype=float)


def _log_log_slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log10(x[finite]), np.log10(y[finite]), 1)
    return float(slope)


def _plot_slope_guide(
    ax: plt.Axes,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    slope: float,
    label: str,
    color: str,
    label_offset_points: tuple[float, float] = (4.0, 0.0),
    label_va: str = "center",
) -> None:
    x = np.array([x_start, x_end], dtype=float)
    y = y_start * (x / x_start) ** slope
    ax.plot(x, y, color=color, linestyle="--", linewidth=1.35, alpha=0.80, zorder=3)
    ax.annotate(
        label,
        xy=(x_end, y[-1]),
        xytext=label_offset_points,
        textcoords="offset points",
        color=color,
        fontsize=FONT_SIZE_TICKS - 1,
        va=label_va,
        ha="left",
    )


def create_surface_grid_overlay_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
    title: str = "Optimum Relative to Fixed Depth",
    enforce_y_axis_units: bool = True,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    grid_df = (
        _with_depth_width_ratio(df)
        .select(["target_non_embedding_params", "target_depth", "depth_width_ratio"])
        .unique()
    )
    depth_values = grid_df["target_depth"].drop_nulls().unique().sort().to_list()
    observed_params = (
        grid_df["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    params_grid = np.geomspace(
        float(np.min(observed_params)), float(np.max(observed_params)), 160
    )

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.05, 1.60], "wspace": 0.24},
    )
    all_y: list[float] = []
    for depth in depth_values:
        depth_df = grid_df.filter(pl.col("target_depth") == int(depth)).sort(
            "target_non_embedding_params"
        )
        x = depth_df["target_non_embedding_params"].to_numpy().astype(float)
        y = depth_df["depth_width_ratio"].to_numpy().astype(float)
        ax.plot(x, y, color="#b9b2aa", linewidth=1.0, alpha=0.58, zorder=1)
        all_y.extend(y.tolist())

    handles: list[mlines.Line2D] = []
    slopes: list[float] = []
    compute_values = np.array(
        [float(fraction) * target_training_flops for fraction in fractions], dtype=float
    )
    compute_norm = mpl.colors.LogNorm(
        vmin=float(np.min(compute_values)), vmax=float(np.max(compute_values))
    )
    compute_cmap = plt.colormaps["viridis_r"]
    for compute in compute_values:
        best_ratios = _surface_best_ratios(model, params_grid, compute)
        color = compute_cmap(compute_norm(float(compute)))
        ax.plot(params_grid, best_ratios, color=color, linewidth=1.9, zorder=4)
        slopes.append(_log_log_slope(params_grid, best_ratios))
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=f"{_format_flops_value(compute)} FLOPs",
            )
        )
        all_y.extend(best_ratios.tolist())

    if all_y:
        y_anchor = float(np.nanpercentile(np.array(all_y, dtype=float), 68))
        fixed_x_start = float(params_grid[32])
        top_depth_df = grid_df.filter(
            pl.col("target_depth") == int(max(depth_values))
        ).sort("target_non_embedding_params")
        top_depth_x = np.log10(
            top_depth_df["target_non_embedding_params"].to_numpy().astype(float)
        )
        top_depth_y = np.log10(
            top_depth_df["depth_width_ratio"].to_numpy().astype(float)
        )
        fixed_y_start = float(
            10 ** np.interp(np.log10(fixed_x_start), top_depth_x, top_depth_y)
        )
        _plot_slope_guide(
            ax,
            x_start=fixed_x_start,
            x_end=float(params_grid[104]),
            y_start=fixed_y_start,
            slope=-0.5,
            label="fixed depth ~ -0.50",
            color="#5d5750",
            label_offset_points=(4.0, 5.0),
            label_va="bottom",
        )
        fit_slope = float(np.nanmean(slopes))
        _plot_slope_guide(
            ax,
            x_start=float(params_grid[48]),
            x_end=float(params_grid[122]),
            y_start=y_anchor / 2.8,
            slope=fit_slope,
            label=f"fit optimum ~ {fit_slope:.2f}",
            color="#111111",
            label_offset_points=(4.0, -20.0),
            label_va="top",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(params_grid)) / 1.20, float(np.max(params_grid)) * 1.65)
    ax.set_ylim(float(np.min(all_y)) / 1.50, float(np.max(all_y)) * 1.50)
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Depth / Width", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.90,
        "Compute",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.text(
        0.5,
        0.82,
        "Budget (FLOPs)",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.73),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 6,
        handlelength=1.8,
        labelspacing=0.35,
    )
    fig.suptitle(
        title,
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def _grid_ratio_at_depth(
    grid_df: pl.DataFrame, depth: int, params: np.ndarray
) -> np.ndarray:
    depth_df = grid_df.filter(pl.col("target_depth") == int(depth)).sort(
        "target_non_embedding_params"
    )
    x = np.log10(depth_df["target_non_embedding_params"].to_numpy().astype(float))
    y = np.log10(depth_df["depth_width_ratio"].to_numpy().astype(float))
    return 10 ** np.interp(np.log10(params), x, y)


def _implied_depth_from_grid(
    grid_df: pl.DataFrame, params_value: float, ratio_value: float
) -> float:
    param_df = grid_df.filter(
        pl.col("target_non_embedding_params") == float(params_value)
    ).sort("depth_width_ratio")
    ratios = param_df["depth_width_ratio"].to_numpy().astype(float)
    depths = param_df["target_depth"].to_numpy().astype(float)
    log_ratio = np.log10(float(ratio_value))
    return float(10 ** np.interp(log_ratio, np.log10(ratios), np.log10(depths)))


def create_surface_relative_to_fixed_depth_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    grid_df = (
        _with_depth_width_ratio(df)
        .select(["target_non_embedding_params", "target_depth", "depth_width_ratio"])
        .unique()
    )
    observed_params = (
        grid_df["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    reference_depth = 16
    reference_ratio = _grid_ratio_at_depth(grid_df, reference_depth, observed_params)

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.3, 1.35], "wspace": 0.28},
    )
    handles: list[mlines.Line2D] = []
    all_y: list[float] = []
    for index, fraction in enumerate(fractions):
        compute = float(fraction) * target_training_flops
        best_ratios = _surface_best_ratios(model, observed_params, compute)
        relative = best_ratios / reference_ratio
        color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        ax.plot(
            observed_params,
            relative,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            zorder=3,
        )
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=f"{_format_flops_value(compute)}",
            )
        )
        all_y.extend(relative.tolist())

    ax.axhline(
        1.0, color="#2f2a24", linewidth=1.1, linestyle="--", alpha=0.76, zorder=2
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(
        float(np.min(observed_params)) / 1.25, float(np.max(observed_params)) * 1.25
    )
    ax.set_ylim(float(np.min(all_y)) / 1.35, float(np.max(all_y)) * 1.35)
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(f"Optimal / Depth-{reference_depth} Ratio", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Relative\nRatio",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 6,
        handlelength=1.8,
        labelspacing=0.35,
    )
    legend_ax.text(
        0.5,
        0.12,
        f"dashed line:\nsame ratio as fixed depth {reference_depth}",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Optimum Relative to Fixed Depth",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def create_surface_implied_depth_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
    title: str = "Surface Fit as Implied Optimal Depth",
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    grid_df = (
        _with_depth_width_ratio(df)
        .select(["target_non_embedding_params", "target_depth", "depth_width_ratio"])
        .unique()
    )
    observed_params = (
        grid_df["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    depth_values = grid_df["target_depth"].drop_nulls().unique().sort().to_list()

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.3, 1.35], "wspace": 0.28},
    )
    handles: list[mlines.Line2D] = []
    all_y: list[float] = []
    computes = np.array(
        [float(fraction) * target_training_flops for fraction in fractions], dtype=float
    )
    if len(np.unique(computes)) > 1:
        compute_norm = mpl.colors.LogNorm(
            vmin=float(np.min(computes)), vmax=float(np.max(computes))
        )
    else:
        compute_norm = mpl.colors.Normalize(
            vmin=float(np.min(computes)) / 1.25, vmax=float(np.max(computes)) * 1.25
        )
    compute_cmap = plt.colormaps["viridis_r"]
    for depth in depth_values:
        ax.axhline(float(depth), color="#d2ccc4", linewidth=0.8, zorder=1)
    ratio_bounds = np.array(
        [
            (
                max(float(model["ratio_min"]), float(ratios.min())),
                min(float(model["ratio_max"]), float(ratios.max())),
            )
            for params_value in observed_params
            for ratios in [
                grid_df.filter(
                    pl.col("target_non_embedding_params") == float(params_value)
                )["depth_width_ratio"]
            ]
        ]
    )
    boundary_markers: set[str] = set()
    for compute in computes:
        best_ratios = _surface_best_ratios(model, observed_params, compute)
        unconstrained_ratios = _surface_analytic_best_ratios(
            model, observed_params, compute
        )
        below_grid = unconstrained_ratios < ratio_bounds[:, 0]
        above_grid = unconstrained_ratios > ratio_bounds[:, 1]
        implied_depths = np.array(
            [
                _implied_depth_from_grid(
                    grid_df, float(params_value), float(ratio_value)
                )
                for params_value, ratio_value in zip(
                    observed_params, best_ratios, strict=True
                )
            ],
            dtype=float,
        )
        color = compute_cmap(compute_norm(float(compute)))
        ax.plot(
            observed_params,
            implied_depths,
            color=color,
            linewidth=1.8,
            zorder=3,
        )
        for mask, marker in (
            (~(below_grid | above_grid), "o"),
            (below_grid, "v"),
            (above_grid, "^"),
        ):
            if not np.any(mask):
                continue
            ax.scatter(
                observed_params[mask],
                implied_depths[mask],
                color=color,
                marker=marker,
                s=18 if marker == "o" else 38,
                zorder=4,
            )
            if marker != "o":
                boundary_markers.add(marker)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=f"{_format_flops_value(compute)} FLOPs",
            )
        )
        all_y.extend(implied_depths.tolist())

    for marker, label in (("v", "Below grid limit"), ("^", "Above grid limit")):
        if marker in boundary_markers:
            handles.append(
                mlines.Line2D(
                    [], [], color="#5d5750", marker=marker, linestyle="none",
                    markersize=5, label=label,
                )
            )

    ax.set_xscale("log")
    ax.set_xlim(
        float(np.min(observed_params)) / 1.25, float(np.max(observed_params)) * 1.25
    )
    ax.set_ylim(float(min(depth_values)) * 0.8, float(np.max(all_y)) * 1.04)
    ax.set_xlabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Grid-implied Optimal Depth", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(
        mticker.FixedLocator([float(depth) for depth in depth_values])
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: str(int(value)))
    )
    ax.grid(
        True,
        which="major",
        axis="x",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Training Budget",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.64),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 6,
        handlelength=1.8,
        labelspacing=0.48,
    )
    fig.suptitle(
        title,
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def create_surface_optimal_ratio_vs_compute_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    observed_params = (
        _with_depth_width_ratio(df)["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    selected_params = observed_params[[0, 2, 4, 6]]
    computes = np.array(
        [float(fraction) * target_training_flops for fraction in fractions], dtype=float
    )

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.9),
        gridspec_kw={"width_ratios": [5.3, 1.35], "wspace": 0.28},
    )
    cmap = plt.colormaps["viridis"]
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(selected_params)), vmax=float(np.max(selected_params))
    )
    handles: list[mlines.Line2D] = []
    all_y: list[float] = []
    for params_value in selected_params:
        best = np.array(
            [
                _surface_best_ratios(
                    model, np.array([params_value], dtype=float), compute
                )[0]
                for compute in computes
            ],
            dtype=float,
        )
        color = cmap(norm(float(params_value)))
        ax.plot(
            computes,
            best,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            zorder=3,
        )
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=_format_compact_param_count(params_value),
            )
        )
        all_y.extend(best.tolist())

    _plot_slope_guide(
        ax,
        x_start=float(computes[1]),
        x_end=float(computes[-2]),
        y_start=float(np.nanmedian(all_y)) / 1.35,
        slope=0.19,
        label="compute slope ~ +0.19",
        color="#111111",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(computes)) / 1.25, float(np.max(computes)) * 1.25)
    ax.set_ylim(float(np.min(all_y)) / 1.45, float(np.max(all_y)) * 1.45)
    ax.set_xlabel("Training FLOPs", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Predicted Optimal Depth / Width", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_flops_tick(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.88,
        "Fixed\nParams",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 5,
        handlelength=1.5,
        labelspacing=0.42,
    )
    legend_ax.text(
        0.5,
        0.12,
        f"surface fit\nn={model['n_train']}\nR2={float(model['r_squared']):.3f}",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND - 7,
        va="center",
        ha="center",
    )

    fig.suptitle(
        "Surface Optimum Across Compute",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    save_figure(fig, output_path, apply_default_adjust=False)


def _surface_prediction_grid(
    model: dict[str, object],
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_kind: str,
    y_kind: str,
    fixed_kind: str,
    fixed_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    values = {
        x_kind: x_mesh.ravel(),
        y_kind: y_mesh.ravel(),
        fixed_kind: np.full(x_mesh.size, fixed_value, dtype=float),
    }
    z = _surface_predict(
        model, values["params"], values["compute"], values["ratio"]
    ).reshape(x_mesh.shape)
    return x_mesh, y_mesh, z


def _surface_loss_levels(
    model: dict[str, object], params: np.ndarray, compute: np.ndarray, ratio: np.ndarray
) -> np.ndarray:
    samples: list[np.ndarray] = []
    for compute_value in compute:
        _, _, z = _surface_prediction_grid(
            model,
            params,
            ratio,
            x_kind="params",
            y_kind="ratio",
            fixed_kind="compute",
            fixed_value=float(compute_value),
        )
        samples.append(z.ravel())
    values = np.concatenate(samples)
    return np.linspace(
        float(np.nanpercentile(values, 3)), float(np.nanpercentile(values, 97)), 12
    )


def _format_contour_labels(contour_set: mpl.contour.QuadContourSet) -> None:
    plt.clabel(contour_set, inline=True, fontsize=FONT_SIZE_TICKS - 4, fmt="%.1f")


def _format_ratio_value(value: float) -> str:
    if value < 0.01 or value >= 1:
        return f"{value:.2e}"
    return f"{value:.3g}"


def _plot_surface_optimum_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ax.plot(
        x,
        y,
        color=SURFACE_OPTIMUM_LINE_COLOR,
        linestyle=":",
        linewidth=2.2,
        dash_capstyle="round",
        zorder=7,
    )


def _optimal_compute_for_ratio(
    model: dict[str, object],
    params_grid: np.ndarray,
    compute_grid: np.ndarray,
    target_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    log_computes = np.log10(compute_grid)
    log_target = float(np.log10(target_ratio))
    for params_value in params_grid:
        best_ratios = np.array(
            [
                _surface_best_ratios(
                    model, np.array([params_value], dtype=float), float(compute)
                )[0]
                for compute in compute_grid
            ],
            dtype=float,
        )
        log_best = np.log10(best_ratios)
        order = np.argsort(log_best)
        sorted_best = log_best[order]
        sorted_compute = log_computes[order]
        if sorted_best[0] <= log_target <= sorted_best[-1]:
            x_values.append(float(params_value))
            y_values.append(
                float(10 ** np.interp(log_target, sorted_best, sorted_compute))
            )
    return np.array(x_values, dtype=float), np.array(y_values, dtype=float)


def _style_contour_axis(
    ax: plt.Axes, *, xlabel: str | None = None, ylabel: str | None = None
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS - 5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS - 5)
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.35,
        alpha=0.55,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 2)


def _draw_contour_legend(
    fig: plt.Figure,
    legend_ax: plt.Axes,
    contourf: mpl.contour.QuadContourSet,
    *,
    title: str,
    optimum_label: str | None,
) -> None:
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.80,
        title,
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    cax = legend_ax.inset_axes([0.18, 0.43, 0.64, 0.08])
    colorbar = fig.colorbar(contourf, cax=cax, orientation="horizontal")
    if len(contourf.levels) >= 3:
        colorbar_ticks = [
            float(contourf.levels[0]),
            float(contourf.levels[len(contourf.levels) // 2]),
            float(contourf.levels[-1]),
        ]
        colorbar.set_ticks(colorbar_ticks)
        colorbar.set_ticklabels([f"{value:.1f}" for value in colorbar_ticks])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 2, length=0, pad=2)
    colorbar.outline.set_linewidth(0.8)
    if optimum_label is not None:
        handle = mlines.Line2D(
            [],
            [],
            color=SURFACE_OPTIMUM_LINE_COLOR,
            linestyle=":",
            linewidth=2.2,
            dash_capstyle="round",
            label=optimum_label,
        )
        legend_ax.legend(
            handles=[handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.12),
            frameon=False,
            fontsize=FONT_SIZE_LEGEND - 6,
            handlelength=2.0,
        )


def create_surface_contours_params_ratio_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _with_depth_width_ratio(_filter_analysis_fractions(df, fractions))
    params_grid = np.geomspace(
        float(df["target_non_embedding_params"].min()),
        float(df["target_non_embedding_params"].max()),
        120,
    )
    ratio_grid = np.geomspace(float(model["ratio_min"]), float(model["ratio_max"]), 120)
    compute_values = np.array(
        [float(fraction) * target_training_flops for fraction in fractions], dtype=float
    )
    levels = _surface_loss_levels(model, params_grid, compute_values, ratio_grid)

    ncols = 4
    nrows = int(np.ceil((len(fractions) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()
    contourf = None
    for ax, fraction, compute in zip(
        flat_axes, fractions, compute_values, strict=False
    ):
        x_mesh, y_mesh, z = _surface_prediction_grid(
            model,
            params_grid,
            ratio_grid,
            x_kind="params",
            y_kind="ratio",
            fixed_kind="compute",
            fixed_value=float(compute),
        )
        contourf = ax.contourf(
            x_mesh,
            y_mesh,
            z,
            levels=levels,
            cmap="viridis_r",
            alpha=0.68,
            extend="both",
        )
        contour = ax.contour(
            x_mesh,
            y_mesh,
            z,
            levels=levels,
            colors="#26211d",
            linewidths=0.55,
            alpha=0.78,
        )
        _format_contour_labels(contour)
        best_ratios = _surface_best_ratios(model, params_grid, float(compute))
        _plot_surface_optimum_line(ax, params_grid, best_ratios)
        panel_df = df.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        complete_df = panel_df.filter(
            pl.col("completion_fraction") >= COMPLETE_POINT_THRESHOLD
        )
        ax.scatter(
            complete_df["target_non_embedding_params"].to_numpy().astype(float),
            complete_df["depth_width_ratio"].to_numpy().astype(float),
            s=10,
            facecolors="#ffffff",
            edgecolors="#1f1a16",
            linewidths=0.35,
            alpha=0.74,
            zorder=6,
        )
        ax.set_title(
            f"{_format_flops_value(float(compute))} FLOPs",
            fontsize=FONT_SIZE_LABELS - 8,
            y=1.02,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:.3g}")
        )
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        _style_contour_axis(ax)

    legend_ax = flat_axes[len(fractions)]
    for ax in flat_axes[len(fractions) + 1 :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Parameters", fontsize=FONT_SIZE_LABELS - 5)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel("Depth / Width", fontsize=FONT_SIZE_LABELS - 5)
    assert contourf is not None
    _draw_contour_legend(
        fig,
        legend_ax,
        contourf,
        title="Fitted\nLoss",
        optimum_label="predicted optimum",
    )
    fig.suptitle(
        "Surface Contours: Parameters x Ratio",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=0.995,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.08, wspace=0.22, hspace=0.38
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_surface_contours_compute_ratio_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _with_depth_width_ratio(_filter_analysis_fractions(df, fractions))
    params_values = (
        df["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    compute_grid = np.geomspace(
        float(min(fractions) * target_training_flops),
        float(max(fractions) * target_training_flops),
        120,
    )
    ratio_grid = np.geomspace(float(model["ratio_min"]), float(model["ratio_max"]), 120)
    levels = _surface_loss_levels(
        model, params_values, compute_grid[[0, -1]], ratio_grid
    )

    ncols = 4
    nrows = int(np.ceil((len(params_values) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()
    contourf = None
    for ax, params_value in zip(flat_axes, params_values, strict=False):
        x_mesh, y_mesh, z = _surface_prediction_grid(
            model,
            compute_grid,
            ratio_grid,
            x_kind="compute",
            y_kind="ratio",
            fixed_kind="params",
            fixed_value=float(params_value),
        )
        contourf = ax.contourf(
            x_mesh,
            y_mesh,
            z,
            levels=levels,
            cmap="viridis_r",
            alpha=0.68,
            extend="both",
        )
        contour = ax.contour(
            x_mesh,
            y_mesh,
            z,
            levels=levels,
            colors="#26211d",
            linewidths=0.55,
            alpha=0.78,
        )
        _format_contour_labels(contour)
        best_ratios = np.array(
            [
                _surface_best_ratios(
                    model, np.array([params_value], dtype=float), float(compute)
                )[0]
                for compute in compute_grid
            ],
            dtype=float,
        )
        _plot_surface_optimum_line(ax, compute_grid, best_ratios)
        panel_df = df.filter(
            pl.col("target_non_embedding_params") == float(params_value)
        )
        complete_df = panel_df.filter(
            pl.col("completion_fraction") >= COMPLETE_POINT_THRESHOLD
        )
        ax.scatter(
            complete_df["analysis_training_flops"].to_numpy().astype(float),
            complete_df["depth_width_ratio"].to_numpy().astype(float),
            s=10,
            facecolors="#ffffff",
            edgecolors="#1f1a16",
            linewidths=0.35,
            alpha=0.74,
            zorder=6,
        )
        ax.set_title(
            _format_compact_param_count(float(params_value)),
            fontsize=FONT_SIZE_LABELS - 8,
            y=1.02,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_flops_tick(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:.3g}")
        )
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        _style_contour_axis(ax)

    legend_ax = flat_axes[len(params_values)]
    for ax in flat_axes[len(params_values) + 1 :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Training FLOPs", fontsize=FONT_SIZE_LABELS - 5)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel("Depth / Width", fontsize=FONT_SIZE_LABELS - 5)
    assert contourf is not None
    _draw_contour_legend(
        fig,
        legend_ax,
        contourf,
        title="Fitted\nLoss",
        optimum_label="predicted optimum",
    )
    fig.suptitle(
        "Surface Contours: Compute x Ratio",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=0.995,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.08, wspace=0.22, hspace=0.38
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_surface_contours_params_compute_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    model: dict[str, object],
    target_training_flops: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = df["analysis_fraction"].unique().sort().to_list()
    df = _with_depth_width_ratio(_filter_analysis_fractions(df, fractions))
    params_grid = np.geomspace(
        float(df["target_non_embedding_params"].min()),
        float(df["target_non_embedding_params"].max()),
        120,
    )
    compute_grid = np.geomspace(
        float(min(fractions) * target_training_flops),
        float(max(fractions) * target_training_flops),
        120,
    )
    observed_ratios = df["depth_width_ratio"].drop_nulls().to_numpy().astype(float)
    ratio_values = 10 ** np.quantile(
        np.log10(observed_ratios), [0.10, 0.30, 0.50, 0.70, 0.90]
    )
    levels = _surface_loss_levels(
        model, params_grid, compute_grid[[0, -1]], ratio_values
    )

    ncols = 3
    nrows = int(np.ceil((len(ratio_values) + 1) / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.7 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()
    contourf = None
    for ax, ratio_value in zip(flat_axes, ratio_values, strict=False):
        x_mesh, y_mesh, z = _surface_prediction_grid(
            model,
            params_grid,
            compute_grid,
            x_kind="params",
            y_kind="compute",
            fixed_kind="ratio",
            fixed_value=float(ratio_value),
        )
        contourf = ax.contourf(
            x_mesh,
            y_mesh,
            z,
            levels=levels,
            cmap="viridis_r",
            alpha=0.68,
            extend="both",
        )
        contour = ax.contour(
            x_mesh,
            y_mesh,
            z,
            levels=levels,
            colors="#26211d",
            linewidths=0.55,
            alpha=0.78,
        )
        _format_contour_labels(contour)
        optimum_params, optimum_compute = _optimal_compute_for_ratio(
            model, params_grid, compute_grid, float(ratio_value)
        )
        if len(optimum_params) > 0:
            _plot_surface_optimum_line(ax, optimum_params, optimum_compute)
        nearest = df.with_columns(
            (pl.col("depth_width_ratio").log10() - float(np.log10(ratio_value)))
            .abs()
            .alias("d_ratio")
        )
        nearest = nearest.filter(pl.col("d_ratio") <= pl.col("d_ratio").quantile(0.12))
        complete_df = nearest.filter(
            pl.col("completion_fraction") >= COMPLETE_POINT_THRESHOLD
        )
        ax.scatter(
            complete_df["target_non_embedding_params"].to_numpy().astype(float),
            complete_df["analysis_training_flops"].to_numpy().astype(float),
            s=10,
            facecolors="#ffffff",
            edgecolors="#1f1a16",
            linewidths=0.35,
            alpha=0.58,
            zorder=6,
        )
        ax.set_title(
            f"ratio {_format_ratio_value(float(ratio_value))}",
            fontsize=FONT_SIZE_LABELS - 8,
            y=1.02,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_flops_tick(value))
        )
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        _style_contour_axis(ax)

    legend_ax = flat_axes[len(ratio_values)]
    for ax in flat_axes[len(ratio_values) + 1 :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Parameters", fontsize=FONT_SIZE_LABELS - 5)
    for row_axes in axes:
        first_data_ax = next((ax for ax in row_axes if ax.has_data()), None)
        if first_data_ax is not None:
            first_data_ax.set_ylabel("Training FLOPs", fontsize=FONT_SIZE_LABELS - 5)
    assert contourf is not None
    _draw_contour_legend(
        fig,
        legend_ax,
        contourf,
        title="Fitted\nLoss",
        optimum_label="fixed ratio optimum",
    )
    fig.suptitle(
        "Surface Contours: Parameters x Compute",
        fontsize=FONT_SIZE_TITLE,
        x=0.50,
        y=0.995,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.88, bottom=0.08, wspace=0.25, hspace=0.40
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_training_loss_vs_flops_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    compute_slice_flops: list[float] | None = None,
    figsize: tuple[float, float] = (10.8, 5.9),
    crop_output: bool = True,
    ylabel: str | None = None,
    model_size_cmap: str = "viridis",
    compact_legend: bool = False,
    legend_width_ratio: float = 1.25,
    enforce_y_axis_units: bool = True,
) -> None:
    plot_df = df.filter(pl.col("cumulative_training_flops") > 0)
    if plot_df.is_empty():
        raise ValueError("No positive-FLOP training points available")

    size_values = plot_df["target_non_embedding_params"].to_numpy().astype(float)
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(size_values)), vmax=float(np.max(size_values))
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "geneformer_dw_model_size",
        plt.colormaps[model_size_cmap](np.linspace(0.12, 0.90, 256)),
    )

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [5.4, legend_width_ratio], "wspace": 0.28},
    )

    all_x: list[float] = []
    all_y: list[float] = []
    warmup_handles: list[mlines.Line2D] = []
    for run_hash in plot_df["run_hash"].unique().sort().to_list():
        run_df = plot_df.filter(pl.col("run_hash") == run_hash).sort("step")
        x = run_df["cumulative_training_flops"].to_numpy().astype(float)
        y = run_df["loss"].to_numpy().astype(float)
        y_plot = exponential_moving_average(y, alpha=GENEFORMER_BCE_CONFIG.ema_alpha)
        model_size = float(run_df["target_non_embedding_params"][0])
        color = cmap(norm(model_size))
        ax.plot(
            x,
            y_plot,
            color=color,
            linewidth=0.65,
            alpha=0.42,
            rasterized=True,
            zorder=2,
        )
        warmup_steps = int(run_df["trainer.lr_warmup_steps"][0])
        warmup_index = int(
            np.searchsorted(run_df["step"].to_numpy(), warmup_steps, side="left")
        )
        if warmup_index < len(x):
            ax.scatter(
                x[warmup_index],
                y_plot[warmup_index],
                marker="D",
                s=26,
                color=color,
                edgecolors="#1f1a16",
                linewidths=0.55,
                alpha=0.88,
                zorder=5,
            )
        all_x.extend(x.tolist())
        all_y.extend(y_plot.tolist())

    warmup_handles.append(
        mlines.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=6,
            markerfacecolor="#ffffff",
            markeredgecolor="#1f1a16",
            label="Warmup end" if compact_legend else "End of Warmup",
        )
    )

    ax.set_xscale("log")
    if all_x:
        log_padding_factor = 10**0.2
        ax.set_xlim(
            float(np.min(all_x)) / log_padding_factor,
            float(np.max(all_x)) * log_padding_factor,
        )
    if all_y:
        ax.set_ylim(*_loss_limits(np.array(all_y, dtype=float)))
    ax.set_xlabel("Cumulative Training FLOPs", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(ylabel or GENEFORMER_BCE_CONFIG.ylabel, fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_flops_tick(value))
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
    for compute_flops in compute_slice_flops or []:
        ax.axvline(
            compute_flops,
            color="#000000",
            linestyle=":",
            linewidth=1.5,
            alpha=0.55,
            zorder=1,
        )

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.86,
        "Model Size",
        transform=legend_ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        va="center",
        ha="center",
    )
    cax = legend_ax.inset_axes([0.20, 0.34, 0.18, 0.38])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar_ticks = [1e6, 1e7, 1e8, 1e9]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([_format_param_count(value) for value in colorbar_ticks])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
    colorbar.outline.set_edgecolor("#000000")
    colorbar.outline.set_linewidth(0.9)
    legend_handles = (
        [
            *warmup_handles,
            mlines.Line2D(
                [],
                [],
                color="#000000",
                linestyle=":",
                linewidth=1.5,
                label="IsoFLOP slices" if compact_legend else "IsoFLOP Slices",
            ),
        ]
        if compute_slice_flops
        else warmup_handles
    )
    legend_ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 5,
        handlelength=1.0,
    )

    fig.suptitle(
        "Training Loss by Compute",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=0.96,
        ha="center",
    )
    fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.16)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def _format_metric_title(metric: str) -> str:
    parts = metric.split("_")
    replacements = {
        "ari": "ARI",
        "nmi": "NMI",
        "asw": "ASW",
        "val": "Val",
    }
    return " ".join(replacements.get(part, part.title()) for part in parts)


def _metric_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return 0.0, 1.0
    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    lower = min(0.0, data_min - 0.08)
    upper = max(1.0, data_max + 0.08)
    return lower, upper


def create_downstream_metrics_plot(df: pl.DataFrame, output_path: Path) -> None:
    available_metrics = [
        metric
        for metric in DOWNSTREAM_METRICS
        if metric in df.columns and df.select(pl.col(metric).drop_nulls()).height > 0
    ]
    if df.is_empty() or not available_metrics:
        return

    apply_plot_style()
    ncols = 3
    nrows = int(np.ceil(len(available_metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.9 * ncols, 4.7 * nrows),
        squeeze=False,
        sharex=True,
    )
    color = plt.colormaps[COLOR_SHADES[1]](0.76)
    all_x = df["non_embedding_params"].to_numpy().astype(float)

    for ax, metric in zip(axes.flat, available_metrics, strict=False):
        metric_df = (
            df.select(["non_embedding_params", "target_depth", metric])
            .drop_nulls()
            .sort(["non_embedding_params", "target_depth"])
        )
        x = metric_df["non_embedding_params"].to_numpy().astype(float)
        y = metric_df[metric].to_numpy().astype(float)
        ax.scatter(
            x,
            y,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            s=42,
            alpha=0.88,
            zorder=4,
        )
        fit = _fit_log_parameter_parabola(x, y)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, color=color, linestyle=":", linewidth=1.8, zorder=2)
        ax.set_xscale("log")
        ax.set_title(
            _format_metric_title(metric), fontsize=FONT_SIZE_LABELS - 4, y=1.02
        )
        ax.set_ylim(*_metric_limits(y))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_compact_param_count(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(
            True,
            which="major",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    for ax in axes.flat[len(available_metrics) :]:
        ax.axis("off")

    if len(all_x) > 0:
        for ax in axes.flat[: len(available_metrics)]:
            ax.set_xlim(float(np.min(all_x)) / 1.30, float(np.max(all_x)) * 1.30)
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    for row_axes in axes:
        if row_axes[0].has_data():
            row_axes[0].set_ylabel("Score", fontsize=FONT_SIZE_LABELS)

    fig.suptitle(
        "Downstream Metrics", fontsize=FONT_SIZE_TITLE, x=0.50, y=0.98, ha="center"
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.91, bottom=0.09, wspace=0.22, hspace=0.34
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def main() -> None:
    args = parse_args()
    if args.target_training_flops <= 0:
        raise ValueError("--target-training-flops must be positive")

    default_fractions = _default_flops_fractions(
        args.n_isoflop_curves, args.min_isoflop_fraction
    )
    middle_fractions = default_fractions[1:-1]
    if not middle_fractions:
        raise ValueError(
            "Middle-slice 07/08 plots require at least three isoFLOP curves"
        )
    log_spaced_depth_fractions = _log_spaced_depth_flops_fractions(args)
    metrics_csv, exported_from_aim = _ensure_metrics_csv(args)
    prepared_df, wrote_prepared_data = _ensure_prepared_data(args, metrics_csv)
    training_df, wrote_training_data = _ensure_training_curve_data(args)
    downstream_df, wrote_downstream_data = _ensure_downstream_data(args)
    create_isoflop_loss_vs_params_plot(
        prepared_df, args.output, fractions=default_fractions
    )
    create_training_loss_vs_flops_plot(training_df, args.training_output)
    create_isoflop_loss_vs_params_by_depth_plot(
        prepared_df, args.depth_output, fractions=default_fractions
    )
    create_isoflop_loss_vs_params_by_depth_plot(
        prepared_df,
        args.log_spaced_depth_output,
        fractions=log_spaced_depth_fractions,
        title="IsoFLOP Scaling Curves by Depth",
    )
    create_depth_parabolas_by_isoflop_budget_plot(
        prepared_df,
        args.depth_parabolas_output,
        fractions=default_fractions,
    )
    create_depth_connected_by_isoflop_budget_plot(
        prepared_df,
        args.depth_connected_output,
        fractions=default_fractions,
    )
    create_depth_connected_by_isoflop_budget_plot(
        prepared_df,
        args.depth_connected_middle_output,
        fractions=middle_fractions,
    )
    create_loss_vs_depth_width_ratio_plot(
        prepared_df, args.ratio_output, fractions=default_fractions
    )
    create_loss_vs_depth_width_ratio_plot(
        prepared_df, args.ratio_middle_output, fractions=middle_fractions
    )
    create_isoparam_ratio_sweeps_by_isoflop_budget_plot(
        prepared_df,
        args.isoparam_ratio_output,
        fractions=default_fractions,
    )
    create_winner_ratio_vs_params_plot(
        prepared_df, args.winner_ratio_output, fractions=default_fractions
    )
    surface_model = write_surface_coefficients(prepared_df, args.surface_coefficients)
    create_surface_fit_optimal_ratio_plot(
        prepared_df,
        args.surface_optimum_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_grid_overlay_plot(
        prepared_df,
        args.surface_grid_overlay_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_relative_to_fixed_depth_plot(
        prepared_df,
        args.surface_relative_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_implied_depth_plot(
        prepared_df,
        args.surface_implied_depth_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_optimal_ratio_vs_compute_plot(
        prepared_df,
        args.surface_compute_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_contours_params_ratio_plot(
        prepared_df,
        args.surface_contour_params_ratio_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_contours_compute_ratio_plot(
        prepared_df,
        args.surface_contour_compute_ratio_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_contours_params_compute_plot(
        prepared_df,
        args.surface_contour_params_compute_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_surface_fit_diagnostics_plot(
        prepared_df,
        args.surface_fit_diagnostics_output,
        model=surface_model,
    )
    create_surface_cv_diagnostics_plot(
        prepared_df,
        args.surface_cv_diagnostics_output,
        diagnostics_csv=args.surface_diagnostics_csv,
        model=surface_model,
    )
    create_surface_bootstrap_optimum_plot(
        prepared_df,
        args.surface_bootstrap_output,
        model=surface_model,
        target_training_flops=args.target_training_flops,
        fractions=default_fractions,
    )
    create_depth_width_ratio_grid_plot(prepared_df, args.ratio_grid_output)
    if not downstream_df.is_empty():
        create_downstream_metrics_plot(downstream_df, args.downstream_output)

    if exported_from_aim:
        print(f"Exported metrics CSV to {metrics_csv}")
    else:
        print(f"Reused cached metrics CSV at {metrics_csv}")
    if wrote_prepared_data:
        print(f"Wrote prepared plot data to {args.prepared_data}")
    else:
        print(f"Reused prepared plot data at {args.prepared_data}")
    if wrote_training_data:
        print(f"Wrote training curve data to {args.training_data}")
    else:
        print(f"Reused training curve data at {args.training_data}")
    if downstream_df.is_empty():
        print(
            f"No downstream metrics found under {args.downstream_dir}; skipped downstream plot"
        )
    elif wrote_downstream_data:
        print(f"Wrote downstream metric data to {args.downstream_data}")
    else:
        print(f"Reused downstream metric data at {args.downstream_data}")
    print(f"Saved plot to {args.output}")
    print(f"Saved plot to {args.training_output}")
    print(f"Saved plot to {args.depth_output}")
    print(f"Saved plot to {args.log_spaced_depth_output}")
    print(f"Saved plot to {args.depth_parabolas_output}")
    print(f"Saved plot to {args.depth_connected_output}")
    print(f"Saved plot to {args.depth_connected_middle_output}")
    print(f"Saved plot to {args.ratio_output}")
    print(f"Saved plot to {args.ratio_middle_output}")
    print(f"Saved plot to {args.isoparam_ratio_output}")
    print(f"Saved plot to {args.winner_ratio_output}")
    print(f"Wrote surface fit coefficients to {args.surface_coefficients}")
    print(f"Saved plot to {args.surface_optimum_output}")
    print(f"Saved plot to {args.surface_grid_overlay_output}")
    print(f"Saved plot to {args.surface_relative_output}")
    print(f"Saved plot to {args.surface_implied_depth_output}")
    print(f"Saved plot to {args.surface_compute_output}")
    print(f"Saved plot to {args.surface_contour_params_ratio_output}")
    print(f"Saved plot to {args.surface_contour_compute_ratio_output}")
    print(f"Saved plot to {args.surface_contour_params_compute_output}")
    print(f"Saved plot to {args.surface_fit_diagnostics_output}")
    print(f"Saved plot to {args.surface_cv_diagnostics_output}")
    print(f"Saved plot to {args.surface_bootstrap_output}")
    print(f"Wrote surface diagnostics to {args.surface_diagnostics_csv}")
    print(f"Saved plot to {args.ratio_grid_output}")
    if not downstream_df.is_empty():
        print(f"Saved plot to {args.downstream_output}")


if __name__ == "__main__":
    main()
