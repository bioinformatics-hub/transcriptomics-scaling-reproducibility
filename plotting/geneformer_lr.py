from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
    _add_isotropic_rounded_frame,
    _format_param_count,
    _style_panel_box,
)
from plotting.utils import (  # noqa: E402
    COLOR_SHADES,
    PLOTS_DIR,
    ROOT_DIR,
    apply_plot_style,
    ensure_parent_dir,
    flatten_dict,
    remove_bounding_box,
    save_figure,
)


DEFAULT_AIM_REPO = ROOT_DIR / "aim-repo" / "geneformer_lr"
DEFAULT_CSV = PLOTS_DIR / "geneformer_lr" / "01_lr_sampled_metrics.csv"
DEFAULT_PREPARED_DATA = PLOTS_DIR / "geneformer_lr" / "01_lr_sampled_prepared.parquet"
DEFAULT_SWEEP_OUTPUT = PLOTS_DIR / "geneformer_lr" / "01_lr_sweeps.svg"
DEFAULT_PARAM_SCALING_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "02_optimal_lr_vs_parameters.svg"
)
DEFAULT_DEPTH_SCALING_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "03_optimal_lr_vs_depth.svg"
)
DEFAULT_WIDTH_SCALING_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "04_optimal_lr_vs_width.svg"
)
DEFAULT_POWER_LAW_COMPARISON_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "05_lr_power_law_model_comparison.svg"
)
DEFAULT_POWER_LAW_BOOTSTRAP_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "06_lr_power_law_bootstrap.svg"
)
DEFAULT_POWER_LAW_SUMMARY_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "07_lr_power_law_summary.svg"
)
DEFAULT_FORMULA_LR_TABLE_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "08_formula_lr_table.svg"
)
DEFAULT_FORMULA_LR_TABLE_CSV_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "08_formula_lr_table.csv"
)
DEFAULT_POWER_LAWS_COMPUTE_SLICES_OUTPUT = (
    PLOTS_DIR / "geneformer_lr" / "09_power_laws_comparison_across_compute_slices.svg"
)
DEFAULT_TARGET_TRAINING_FLOPS = 7.0e17
DEFAULT_MAX_ANALYSIS_FRACTION = 0.75
DEFAULT_N_ANALYSIS_FLOP_VALUES = 8

FORMULA_ANCHOR_LR = 1e-3
FORMULA_ANCHOR_PARAMS = 1e6
FORMULA_ANCHOR_DEPTH = 20
FORMULA_PARAM_EXPONENT = -0.3
FORMULA_DEPTH_EXPONENT = 0.5
TABLE_LOG10_INTERCEPT = -2.228204550999
TABLE_PARAM_EXPONENT = -0.246328823214
TABLE_DEPTH_EXPONENT = 0.323160960820

RUN_QUERY = "run.config.metadata.title == 'geneformer_lr'"
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
    "trainer.lr_warmup_steps",
    "trainer.lr_scheduler",
}
RUN_FIELDS = {
    "hparams.parameters_total",
    "hparams.parameters_embedding",
    "hparams.parameters_encoder",
    "hparams.parameters_decoder",
}
REQUIRED_CSV_COLUMNS = {
    "run_hash",
    "run_name",
    "metadata.pipeline",
    "metadata.title",
    "model.lr",
    "model.d_model",
    "model.transformer.n_layers",
    "trainer.n_steps",
    "metric_name",
    "analysis_fraction",
    "target_step",
    "sampled_step",
    "max_step",
    "terminal_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Aim metrics for geneformer_lr and plot learning-rate scaling diagnostics.",
    )
    parser.add_argument("--aim-repo", type=Path, default=DEFAULT_AIM_REPO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--prepared-data", type=Path, default=DEFAULT_PREPARED_DATA)
    parser.add_argument("--sweep-output", type=Path, default=DEFAULT_SWEEP_OUTPUT)
    parser.add_argument(
        "--param-scaling-output", type=Path, default=DEFAULT_PARAM_SCALING_OUTPUT
    )
    parser.add_argument(
        "--depth-scaling-output", type=Path, default=DEFAULT_DEPTH_SCALING_OUTPUT
    )
    parser.add_argument(
        "--width-scaling-output", type=Path, default=DEFAULT_WIDTH_SCALING_OUTPUT
    )
    parser.add_argument(
        "--formula-lr-table-output", type=Path, default=DEFAULT_FORMULA_LR_TABLE_OUTPUT
    )
    parser.add_argument(
        "--formula-lr-table-csv-output",
        type=Path,
        default=DEFAULT_FORMULA_LR_TABLE_CSV_OUTPUT,
    )
    parser.add_argument(
        "--power-law-comparison-output",
        type=Path,
        default=DEFAULT_POWER_LAW_COMPARISON_OUTPUT,
    )
    parser.add_argument(
        "--power-law-bootstrap-output",
        type=Path,
        default=DEFAULT_POWER_LAW_BOOTSTRAP_OUTPUT,
    )
    parser.add_argument(
        "--power-law-summary-output",
        type=Path,
        default=DEFAULT_POWER_LAW_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--power-laws-compute-slices-output",
        type=Path,
        default=DEFAULT_POWER_LAWS_COMPUTE_SLICES_OUTPUT,
    )
    parser.add_argument(
        "--min-completion",
        type=float,
        default=0.90,
        help="Minimum max_step / trainer.n_steps used when selecting runs for optimal-LR plots.",
    )
    parser.add_argument(
        "--near-optimal-pct",
        type=float,
        default=1.0,
        help="Loss tolerance used to define the near-optimal LR band, in percent above best terminal loss.",
    )
    parser.add_argument(
        "--flops-fractions",
        type=str,
        default=None,
        help=(
            "Comma-separated fractions of the matched training FLOP budget to sample. "
            "Default: eight linearly spaced values ending at 75%%."
        ),
    )
    parser.add_argument(
        "--target-training-flops",
        type=float,
        default=DEFAULT_TARGET_TRAINING_FLOPS,
        help="Matched training FLOP budget used to label sampled compute points.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute cached CSV and parquet data."
    )
    return parser.parse_args()


def _default_flops_fractions() -> list[float]:
    step = DEFAULT_MAX_ANALYSIS_FRACTION / DEFAULT_N_ANALYSIS_FLOP_VALUES
    return [step * index for index in range(1, DEFAULT_N_ANALYSIS_FLOP_VALUES + 1)]


def _parse_flops_fractions(raw: str | None) -> list[float]:
    if raw is None:
        return _default_flops_fractions()
    fractions = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not fractions:
        raise ValueError("--flops-fractions must contain at least one value")
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError("--flops-fractions values must be in (0, 1]")
    return fractions


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
    requested_fractions = _parse_flops_fractions(args.flops_fractions)
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
        metrics_to_extract=METRICS_TO_EXTRACT,
        run_query=RUN_QUERY,
        config_keys=CONFIG_KEYS,
        run_fields=RUN_FIELDS,
    )
    return args.csv, True


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


def _export_sampled_metrics_csv(
    aim_repo_path: Path,
    output_csv: Path,
    *,
    flops_fractions: list[float],
    metrics_to_extract: set[str],
    run_query: str | None,
    config_keys: set[str],
    run_fields: set[str],
) -> None:
    repo = Repo(str(aim_repo_path))
    rows: list[dict[str, object]] = []
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
            n_steps = int(flat_config["trainer.n_steps"])
            for fraction in flops_fractions:
                target_step = max(1, int(round(n_steps * fraction)))
                reached_df = metric_df[metric_df["step"] <= target_step]
                if reached_df.empty:
                    sampled = metric_df.iloc[0]
                else:
                    sampled = reached_df.iloc[-1]

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
                        "terminal_loss": _row_value(sampled, "value"),
                        "timestamp": _row_value(sampled, "time")
                        if "time" in metric_df.columns
                        else _row_value(sampled, "timestamp"),
                    }
                )

    if not rows:
        raise ValueError(
            f"No matching sampled metrics found in Aim repo: {aim_repo_path}"
        )

    ensure_parent_dir(output_csv)
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_float_from_name(pattern: str, column: str) -> pl.Expr:
    return pl.col(column).str.extract(pattern, 1).cast(pl.Float64)


def _parse_int_from_name(pattern: str, column: str) -> pl.Expr:
    return pl.col(column).str.extract(pattern, 1).cast(pl.Int64)


def _with_lr_columns(df: pl.DataFrame) -> pl.DataFrame:
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
            _parse_float_from_name(
                r"target_params_([0-9.eE+]+)", "metadata.run_name"
            ).alias("target_non_embedding_params"),
            _parse_int_from_name(r"depth_([0-9]+)", "metadata.run_name").alias(
                "target_depth"
            ),
            pl.col("model.lr").cast(pl.Float64).alias("learning_rate"),
            pl.col("model.transformer.n_layers")
            .cast(pl.Int64)
            .alias("model.transformer.n_layers"),
            pl.col("trainer.n_steps").cast(pl.Int64).alias("trainer.n_steps"),
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
    if args.prepared_data.exists() and not args.force:
        prepared_df = pl.read_parquet(args.prepared_data)
        required = {
            "backbone_params",
            "non_embedding_params",
            "target_non_embedding_params",
            "target_depth",
            "learning_rate",
            "analysis_fraction",
            "analysis_training_flops",
            "completion_fraction",
        }
        numeric_dtypes = {
            "terminal_loss": (pl.Float32, pl.Float64),
            "learning_rate": (pl.Float32, pl.Float64),
            "analysis_fraction": (pl.Float32, pl.Float64),
            "completion_fraction": (pl.Float32, pl.Float64),
        }
        has_numeric_dtypes = all(
            column in prepared_df.columns
            and prepared_df.schema[column] in allowed_dtypes
            for column, allowed_dtypes in numeric_dtypes.items()
        )
        if (
            required.issubset(set(prepared_df.columns))
            and has_numeric_dtypes
            and _has_requested_fractions(
                args.prepared_data, _parse_flops_fractions(args.flops_fractions)
            )
        ):
            return prepared_df, False

    prepared_df = (
        pl.read_csv(metrics_csv)
        .with_columns(
            [
                pl.col("analysis_fraction").cast(pl.Float64, strict=False),
                pl.col("max_step").cast(pl.Int64, strict=False),
                pl.col("sampled_step").cast(pl.Int64, strict=False),
                pl.col("target_step").cast(pl.Int64, strict=False),
                pl.col("terminal_loss").cast(pl.Float64, strict=False),
                pl.col("model.d_model").cast(pl.Int64, strict=False),
                pl.col("model.lr").cast(pl.Float64, strict=False),
                pl.col("model.transformer.n_layers").cast(pl.Int64, strict=False),
                pl.col("trainer.n_steps").cast(pl.Int64, strict=False),
            ]
        )
        .filter(
            (pl.col("metadata.pipeline") == GENEFORMER_BCE_CONFIG.pipeline)
            & (pl.col("metric_name") == GENEFORMER_BCE_CONFIG.metric_name)
        )
    )
    if "context.subset" in prepared_df.columns:
        prepared_df = prepared_df.filter(pl.col("context.subset") == "train")
    if prepared_df.is_empty():
        raise ValueError(f"No sampled rows found in {metrics_csv}")
    prepared_df = _with_lr_columns(prepared_df)
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
            (pl.col("sampled_step") / pl.col("trainer.n_steps")).alias(
                "sampled_flops_fraction"
            ),
            (
                pl.col("sampled_step")
                / pl.col("trainer.n_steps")
                * args.target_training_flops
            ).alias("sampled_training_flops"),
        ]
    )
    ensure_parent_dir(args.prepared_data)
    prepared_df.write_parquet(args.prepared_data)
    return prepared_df, True


def _format_lr(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")


def _format_table_lr(value: float) -> str:
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")


def _format_completion(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def _format_fraction_label(value: float) -> str:
    return f"{int(round(100.0 * value))}% FLOPs"


def _format_flops_value(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}e{exponent}"


def _format_flops_label(value: float) -> str:
    return f"{_format_flops_value(value)} FLOPs"


def _fraction_suffix(value: float) -> str:
    return f"flops_{int(round(100.0 * value)):03d}"


def _flops_suffix(value: float, index: int | None = None) -> str:
    value_suffix = _format_flops_value(value).replace(".", "p")
    if index is None:
        return f"flops_{value_suffix}"
    return f"flops_{index:02d}_{value_suffix}"


def _output_for_fraction(output_path: Path, fraction: float) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_{_fraction_suffix(fraction)}{output_path.suffix}"
    )


def _output_for_flops(
    output_path: Path, flops: float, index: int | None = None
) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_{_flops_suffix(flops, index)}{output_path.suffix}"
    )


def _format_power(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.0f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    return _format_param_count(value)


def _formula_lr_table_values(
    *,
    intercept: float = TABLE_LOG10_INTERCEPT,
    param_exponent: float = TABLE_PARAM_EXPONENT,
    depth_exponent: float = TABLE_DEPTH_EXPONENT,
) -> tuple[list[float], list[int], np.ndarray]:
    target_params = [1e6, 3e6, 10e6, 30e6, 100e6, 300e6, 1e9]
    depths = [2, 4, 8, 16, 24, 32, 64]
    lr_matrix = np.array(
        [
            [
                10
                ** (
                    intercept
                    + param_exponent * np.log10(target_param)
                    + depth_exponent * np.log10(depth)
                )
                for depth in depths
            ]
            for target_param in target_params
        ],
        dtype=float,
    )
    return target_params, depths, lr_matrix


def _observed_lr_arrays(
    best_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_df = best_df.sort(["target_non_embedding_params", "target_depth"])
    params = sorted_df["target_non_embedding_params"].to_numpy().astype(float)
    depths = sorted_df["target_depth"].to_numpy().astype(float)
    observed_lr = sorted_df["optimal_lr"].to_numpy().astype(float)
    if "lr_fit_method" in sorted_df.columns:
        fallback = (sorted_df["lr_fit_method"] != "quadratic").to_numpy()
    else:
        fallback = sorted_df["best_on_boundary"].to_numpy()
    return params, depths, observed_lr, fallback.astype(bool)


def _power_law_design(params: np.ndarray, depths: np.ndarray, terms: str) -> np.ndarray:
    columns = [np.ones_like(params, dtype=float)]
    if "N" in terms:
        columns.append(np.log10(params))
    if "D" in terms:
        columns.append(np.log10(depths))
    return np.column_stack(columns)


def _power_law_metrics(
    observed_log_lr: np.ndarray, predicted_log_lr: np.ndarray
) -> dict[str, float]:
    residual = observed_log_lr - predicted_log_lr
    return {
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        "mae_log10": float(np.mean(np.abs(residual))),
        "median_fold_error": float(10 ** np.median(np.abs(residual))),
        "max_fold_error": float(10 ** np.max(np.abs(residual))),
    }


def _fit_power_law_models(best_df: pl.DataFrame) -> dict[str, dict[str, object]]:
    params, depths, observed_lr, _ = _observed_lr_arrays(best_df)
    observed_log_lr = np.log10(observed_lr)
    fixed_feature = FORMULA_PARAM_EXPONENT * np.log10(
        params
    ) + FORMULA_DEPTH_EXPONENT * np.log10(depths)
    fixed_anchor_intercept = np.log10(FORMULA_ANCHOR_LR) - (
        FORMULA_PARAM_EXPONENT * np.log10(FORMULA_ANCHOR_PARAMS)
        + FORMULA_DEPTH_EXPONENT * np.log10(FORMULA_ANCHOR_DEPTH)
    )

    model_specs = [
        ("constant", "Constant LR", ""),
        ("n_only", "Non-embedding-parameter-only power law", "N"),
        ("d_only", "D-only power law", "D"),
        ("free", "Free non-embedding-parameter/depth power law", "ND"),
    ]
    models: dict[str, dict[str, object]] = {}
    for key, label, terms in model_specs:
        design = _power_law_design(params, depths, terms)
        coefficients, *_ = np.linalg.lstsq(design, observed_log_lr, rcond=None)
        predicted_log_lr = design @ coefficients
        param_exponent = float(coefficients[1]) if "N" in terms else 0.0
        depth_index = 1 + int("N" in terms)
        depth_exponent = float(coefficients[depth_index]) if "D" in terms else 0.0
        models[key] = {
            "label": label,
            "predicted_lr": 10**predicted_log_lr,
            "predicted_log_lr": predicted_log_lr,
            "intercept": float(coefficients[0]),
            "param_exponent": param_exponent,
            "depth_exponent": depth_exponent,
            **_power_law_metrics(observed_log_lr, predicted_log_lr),
        }

    fixed_fit_intercept = float(np.mean(observed_log_lr - fixed_feature))
    fixed_models = {
        "fixed_anchor": (
            "Fixed exponents, anchored normalization",
            fixed_anchor_intercept,
        ),
        "fixed_fit_k": ("Fixed exponents, fitted normalization", fixed_fit_intercept),
    }
    for key, (label, intercept) in fixed_models.items():
        predicted_log_lr = intercept + fixed_feature
        models[key] = {
            "label": label,
            "predicted_lr": 10**predicted_log_lr,
            "predicted_log_lr": predicted_log_lr,
            "intercept": intercept,
            "param_exponent": FORMULA_PARAM_EXPONENT,
            "depth_exponent": FORMULA_DEPTH_EXPONENT,
            **_power_law_metrics(observed_log_lr, predicted_log_lr),
        }

    return models


def _bootstrap_power_law_exponents(
    best_df: pl.DataFrame, n_bootstrap: int = 2000
) -> np.ndarray:
    params, depths, observed_lr, _ = _observed_lr_arrays(best_df)
    observed_log_lr = np.log10(observed_lr)
    design = _power_law_design(params, depths, "ND")
    rng = np.random.default_rng(20240527)
    bootstrap = np.full((n_bootstrap, 2), np.nan)
    for index in range(n_bootstrap):
        sample_indices = rng.integers(0, len(observed_log_lr), len(observed_log_lr))
        sample_design = design[sample_indices]
        if np.linalg.matrix_rank(sample_design) < sample_design.shape[1]:
            continue
        coefficients, *_ = np.linalg.lstsq(
            sample_design, observed_log_lr[sample_indices], rcond=None
        )
        bootstrap[index] = [coefficients[1], coefficients[2]]
    return bootstrap[np.isfinite(bootstrap).all(axis=1)]


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Need at least two positive points to fit a power law")
    slope, intercept = np.polyfit(
        np.log10(x[mask].astype(float)), np.log10(y[mask].astype(float)), deg=1
    )
    return float(slope), float(intercept)


def _terminal_loss_df(df: pl.DataFrame) -> pl.DataFrame:
    if {"max_step", "terminal_loss", "completion_fraction"}.issubset(set(df.columns)):
        return df.sort(["target_non_embedding_params", "target_depth", "learning_rate"])

    run_keys = [
        "run_hash",
        "metadata.run_name",
        "target_non_embedding_params",
        "target_depth",
        "model.d_model",
        "model.transformer.n_layers",
        "backbone_params",
        "non_embedding_params",
        "learning_rate",
        "trainer.n_steps",
    ]
    return (
        df.sort(["run_hash", "step"])
        .group_by(run_keys)
        .agg(
            [
                pl.col("step").max().alias("max_step"),
                pl.col("value").last().alias("terminal_loss"),
            ]
        )
        .with_columns(
            [
                (pl.col("max_step") / pl.col("trainer.n_steps"))
                .clip(0.0, 1.0)
                .alias("completion_fraction"),
                (pl.col("target_non_embedding_params").log10())
                .round(6)
                .alias("target_param_log10"),
            ]
        )
        .sort(["target_non_embedding_params", "target_depth", "learning_rate"])
    )


def _best_lr_df(terminal_df: pl.DataFrame, min_completion: float) -> pl.DataFrame:
    eligible = terminal_df.filter(
        (pl.col("completion_fraction") >= min_completion)
        & pl.col("terminal_loss").is_finite()
    )
    if eligible.is_empty():
        raise ValueError(f"No runs reached min_completion={min_completion:.2f}")
    return (
        eligible.sort(
            [
                "target_non_embedding_params",
                "target_depth",
                "terminal_loss",
                "learning_rate",
            ]
        )
        .group_by(["target_non_embedding_params", "target_depth"], maintain_order=True)
        .agg(
            [
                pl.col("learning_rate").first().alias("optimal_lr"),
                pl.col("terminal_loss").first().alias("best_terminal_loss"),
                pl.col("completion_fraction").first().alias("best_completion_fraction"),
                pl.col("backbone_params").first().alias("backbone_params"),
                pl.col("non_embedding_params").first().alias("non_embedding_params"),
                pl.col("model.d_model").first().alias("model.d_model"),
                pl.col("model.transformer.n_layers")
                .first()
                .alias("model.transformer.n_layers"),
                pl.col("learning_rate").min().alias("min_swept_lr"),
                pl.col("learning_rate").max().alias("max_swept_lr"),
                pl.len().alias("n_eligible_lrs"),
            ]
        )
        .with_columns(
            (
                (pl.col("optimal_lr") == pl.col("min_swept_lr"))
                | (pl.col("optimal_lr") == pl.col("max_swept_lr"))
            ).alias("best_on_boundary")
        )
    )


def _quadratic_best_lr_df(
    terminal_df: pl.DataFrame, min_completion: float
) -> pl.DataFrame:
    eligible = terminal_df.filter(
        (pl.col("completion_fraction") >= min_completion)
        & pl.col("terminal_loss").is_finite()
    )
    if eligible.is_empty():
        raise ValueError(f"No runs reached min_completion={min_completion:.2f}")

    sweep_lr_values = (
        terminal_df.filter(pl.col("learning_rate").is_finite())["learning_rate"]
        .unique()
        .sort()
        .to_numpy()
        .astype(float)
    )
    group_cols = ["target_non_embedding_params", "target_depth"]
    rows: list[dict[str, object]] = []
    for key, group_df in terminal_df.partition_by(group_cols, as_dict=True).items():
        target_param, depth = key
        sorted_by_lr = group_df.sort("learning_rate")
        lr_values = sorted_by_lr["learning_rate"].to_numpy().astype(float)
        losses = sorted_by_lr["terminal_loss"].to_numpy().astype(float)
        completion = sorted_by_lr["completion_fraction"].to_numpy().astype(float)
        eligible_mask = (completion >= min_completion) & np.isfinite(losses)
        eligible_indices = np.flatnonzero(eligible_mask)
        if len(eligible_indices) == 0:
            continue

        best_index = int(eligible_indices[int(np.argmin(losses[eligible_mask]))])
        discrete_lr = float(lr_values[best_index])
        optimal_lr = discrete_lr
        sweep_matches = np.flatnonzero(np.isclose(sweep_lr_values, discrete_lr))
        if len(sweep_matches) != 1:
            raise ValueError(
                f"Unable to locate learning rate {discrete_lr:g} in sweep grid"
            )
        sweep_index = int(sweep_matches[0])
        fit_method = "discrete_boundary"
        quadratic_a = np.nan
        quadratic_b = np.nan
        quadratic_c = np.nan

        if 0 < sweep_index < len(sweep_lr_values) - 1:
            window_lrs = sweep_lr_values[sweep_index - 1 : sweep_index + 2]
            window_indices: list[int] = []
            for window_lr in window_lrs:
                matches = np.flatnonzero(np.isclose(lr_values, window_lr))
                if len(matches) != 1:
                    break
                window_indices.append(int(matches[0]))

            complete_window = len(window_indices) == 3 and bool(
                np.all(eligible_mask[np.asarray(window_indices)])
            )
            if not complete_window:
                fit_method = "discrete_incomplete_window"
            else:
                log_lr = np.log10(window_lrs)
                window_losses = losses[np.asarray(window_indices)]
                quadratic_a, quadratic_b, quadratic_c = np.polyfit(
                    log_lr, window_losses, deg=2
                )
                log_optimum = (
                    -quadratic_b / (2.0 * quadratic_a) if quadratic_a > 0 else np.nan
                )
                if np.isfinite(log_optimum) and log_lr[0] <= log_optimum <= log_lr[-1]:
                    optimal_lr = float(10**log_optimum)
                    fit_method = "quadratic"
                elif quadratic_a <= 0:
                    fit_method = "discrete_nonconvex"
                else:
                    fit_method = "discrete_out_of_window"

        best_row = sorted_by_lr.row(best_index, named=True)
        rows.append(
            {
                "target_non_embedding_params": target_param,
                "target_depth": depth,
                "optimal_lr": optimal_lr,
                "discrete_optimal_lr": discrete_lr,
                "best_terminal_loss": float(best_row["terminal_loss"]),
                "best_completion_fraction": float(best_row["completion_fraction"]),
                "backbone_params": float(best_row["backbone_params"]),
                "non_embedding_params": float(best_row["non_embedding_params"]),
                "model.d_model": int(best_row["model.d_model"]),
                "model.transformer.n_layers": int(
                    best_row["model.transformer.n_layers"]
                ),
                "min_swept_lr": float(np.min(lr_values[eligible_mask])),
                "max_swept_lr": float(np.max(lr_values[eligible_mask])),
                "n_eligible_lrs": int(np.count_nonzero(eligible_mask)),
                "best_on_boundary": fit_method != "quadratic",
                "lr_fit_method": fit_method,
                "quadratic_a": float(quadratic_a),
                "quadratic_b": float(quadratic_b),
                "quadratic_c": float(quadratic_c),
            }
        )

    return pl.DataFrame(rows).sort(group_cols)


def _lr_tolerance_df(
    terminal_df: pl.DataFrame, min_completion: float, near_optimal_pct: float
) -> pl.DataFrame:
    group_cols = ["target_non_embedding_params", "target_depth"]
    eligible = terminal_df.filter(
        (pl.col("completion_fraction") >= min_completion)
        & pl.col("terminal_loss").is_finite()
    )
    if eligible.is_empty():
        raise ValueError(f"No runs reached min_completion={min_completion:.2f}")

    best = (
        eligible.sort([*group_cols, "terminal_loss", "learning_rate"])
        .group_by(group_cols, maintain_order=True)
        .agg(
            [
                pl.col("learning_rate").first().alias("best_lr"),
                pl.col("terminal_loss").first().alias("best_terminal_loss"),
            ]
        )
    )
    with_best = eligible.join(best, on=group_cols, how="left").with_columns(
        (
            pl.col("terminal_loss")
            <= pl.col("best_terminal_loss") * (1.0 + near_optimal_pct / 100.0)
        ).alias("near_optimal")
    )
    return (
        with_best.group_by(group_cols)
        .agg(
            [
                pl.col("best_lr").first().alias("best_lr"),
                pl.col("best_terminal_loss").first().alias("best_terminal_loss"),
                pl.col("learning_rate")
                .filter(pl.col("near_optimal"))
                .max()
                .alias("near_optimal_max_lr"),
                pl.col("learning_rate").max().alias("completed_max_lr"),
                pl.col("near_optimal").sum().alias("n_near_optimal_lrs"),
                pl.len().alias("n_completed_lrs"),
            ]
        )
        .sort(group_cols)
    )


def create_completion_overview_plot(
    terminal_df: pl.DataFrame,
    output_path: Path,
    *,
    min_completion: float,
    title_suffix: str = "",
) -> None:
    summary = (
        terminal_df.group_by(["target_non_embedding_params", "target_depth"])
        .agg(
            [
                pl.col("completion_fraction").min().alias("min_completion"),
                pl.col("completion_fraction").max().alias("max_completion"),
                (pl.col("completion_fraction") >= min_completion)
                .sum()
                .alias("n_complete"),
                pl.len().alias("n_runs"),
            ]
        )
        .sort(["target_non_embedding_params", "target_depth"])
    )
    target_params = summary["target_non_embedding_params"].unique().sort().to_list()
    depths = summary["target_depth"].unique().sort().to_list()
    matrix = np.full((len(target_params), len(depths)), np.nan)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "completion", ["#f4e3d1", "#d8d8d8", "#3f6d5a"]
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for row_index, target_param in enumerate(target_params):
        for col_index, depth in enumerate(depths):
            cell = summary.filter(
                (pl.col("target_non_embedding_params") == target_param)
                & (pl.col("target_depth") == depth)
            )
            if cell.is_empty():
                continue
            row = cell.row(0, named=True)
            matrix[row_index, col_index] = float(row["min_completion"])
            ax.text(
                col_index,
                row_index,
                f"{int(row['n_complete'])}/{int(row['n_runs'])}\n{_format_completion(row['min_completion'])}-{_format_completion(row['max_completion'])}",
                ha="center",
                va="center",
                fontsize=FONT_SIZE_TICKS,
                color="#111111",
            )

    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([str(depth) for depth in depths])
    ax.set_yticks(np.arange(len(target_params)))
    ax.set_yticklabels([_format_power(value) for value in target_params])
    ax.set_xlabel("Target Depth", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_title(
        f"Run Completion by Architecture{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        y=1.04,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax = ax.inset_axes([1.05, 0.08, 0.20, 0.62])
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    cax = legend_ax.inset_axes([0.18, 0.16, 0.18, 0.58])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [0.0, min_completion, 1.0]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_completion(value) for value in ticks])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
    colorbar.outline.set_edgecolor("#000000")
    colorbar.outline.set_linewidth(0.9)
    legend_ax.text(
        0.58,
        0.84,
        "Minimum\nCompletion",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 3,
    )
    legend_ax.text(
        0.58,
        0.08,
        "text: eligible / total\nmin-max",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 5,
    )

    fig.subplots_adjust(left=0.16, right=0.78, top=0.88, bottom=0.14)
    save_figure(fig, output_path)


def create_lr_sweep_plot(
    terminal_df: pl.DataFrame,
    output_path: Path,
    *,
    min_completion: float,
    near_optimal_pct: float,
    title_suffix: str = "",
) -> None:
    target_params = terminal_df["target_non_embedding_params"].unique().sort().to_list()
    depths = terminal_df["target_depth"].unique().sort().to_list()
    nrows = len(target_params)
    ncols = len(depths)
    y_values = terminal_df["terminal_loss"].to_numpy().astype(float)
    y_values = y_values[np.isfinite(y_values)]
    if len(y_values) == 0:
        raise ValueError("No finite terminal losses available for LR sweep plot")
    y_pad = max(0.06 * (float(np.max(y_values)) - float(np.min(y_values))), 0.08)
    y_limits = (float(np.min(y_values)) - y_pad, float(np.max(y_values)) + y_pad)
    lr_values = terminal_df["learning_rate"].unique().sort().to_numpy().astype(float)

    apply_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.3 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_index, target_param in enumerate(target_params):
        for col_index, depth in enumerate(depths):
            ax = axes[row_index, col_index]
            panel_df = terminal_df.filter(
                (pl.col("target_non_embedding_params") == target_param)
                & (pl.col("target_depth") == depth)
            ).sort("learning_rate")
            x = panel_df["learning_rate"].to_numpy().astype(float)
            y = panel_df["terminal_loss"].to_numpy().astype(float)
            completion = panel_df["completion_fraction"].to_numpy().astype(float)
            eligible = completion >= min_completion
            color = plt.colormaps[COLOR_SHADES[row_index % len(COLOR_SHADES)]](0.76)

            ax.plot(x, y, color=color, linewidth=1.4, zorder=2)
            ax.scatter(
                x[eligible],
                y[eligible],
                color=color,
                edgecolors="black",
                linewidths=0.7,
                s=38,
                zorder=4,
            )
            if np.any(~eligible):
                ax.scatter(
                    x[~eligible],
                    y[~eligible],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=0.9,
                    s=38,
                    zorder=4,
                )
            if np.any(eligible):
                best_index = np.flatnonzero(eligible)[int(np.argmin(y[eligible]))]
                ax.scatter(
                    x[best_index],
                    y[best_index],
                    marker="D",
                    color="black",
                    s=54,
                    zorder=5,
                )

            ax.set_xscale("log")
            ax.set_ylim(*y_limits)
            ax.xaxis.set_major_locator(mticker.FixedLocator(lr_values))
            ax.xaxis.set_major_formatter(
                mticker.FixedFormatter([_format_lr(value) for value in lr_values])
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
            ax.tick_params(axis="x", rotation=35)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
            if row_index == 0:
                ax.set_title(f"Depth {depth}", fontsize=FONT_SIZE_LABELS, y=1.03)
            if col_index == 0:
                ax.set_ylabel(
                    f"{_format_power(target_param)}\n{GENEFORMER_BCE_CONFIG.ylabel}",
                    fontsize=FONT_SIZE_LABELS,
                )
            if row_index == nrows - 1:
                ax.set_xlabel("Learning Rate", fontsize=FONT_SIZE_LABELS)

    fig.suptitle(
        f"Learning-rate Sweep at Matched Compute Budget{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        x=0.53,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.93, bottom=0.08, wspace=0.16, hspace=0.30
    )
    save_figure(fig, output_path, apply_default_adjust=False)


def create_lr_tolerance_plot(
    terminal_df: pl.DataFrame,
    output_path: Path,
    *,
    min_completion: float,
    near_optimal_pct: float,
    title_suffix: str = "",
) -> None:
    summary = _lr_tolerance_df(terminal_df, min_completion, near_optimal_pct)
    target_params = terminal_df["target_non_embedding_params"].unique().sort().to_list()
    depths = terminal_df["target_depth"].unique().sort().to_list()
    lr_values = terminal_df["learning_rate"].unique().sort().to_numpy().astype(float)
    matrix = np.full((len(target_params), len(depths)), np.nan)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "near_optimal_lr",
        plt.colormaps["Greens"](np.linspace(0.28, 0.95, 256)),
    )
    cmap.set_bad("#f3f0ea")
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(lr_values)), vmax=float(np.max(lr_values))
    )

    for row_index, target_param in enumerate(target_params):
        for col_index, depth in enumerate(depths):
            cell = summary.filter(
                (pl.col("target_non_embedding_params") == target_param)
                & (pl.col("target_depth") == depth)
            )
            if cell.is_empty():
                ax.text(
                    col_index,
                    row_index,
                    "incomplete",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_TICKS,
                )
                continue

            row = cell.row(0, named=True)
            matrix[row_index, col_index] = float(row["near_optimal_max_lr"])
            ax.text(
                col_index,
                row_index,
                "best "
                f"{_format_lr(row['best_lr'])}\n"
                "near <= "
                f"{_format_lr(row['near_optimal_max_lr'])}\n"
                "complete <= "
                f"{_format_lr(row['completed_max_lr'])}",
                ha="center",
                va="center",
                fontsize=FONT_SIZE_TICKS - 1,
                color="#111111",
            )

    masked_matrix = np.ma.masked_invalid(matrix)
    ax.imshow(masked_matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([str(depth) for depth in depths])
    ax.set_yticks(np.arange(len(target_params)))
    ax.set_yticklabels([_format_power(value) for value in target_params])
    ax.set_xlabel("Target Depth", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Target Non-embedding Parameters", fontsize=FONT_SIZE_LABELS)
    ax.set_title(
        f"Highest Near-optimal Learning Rate{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        y=1.04,
    )
    ax.set_xticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_params), 1), minor=True)
    ax.grid(False, which="major")
    ax.grid(which="minor", color="#ffffff", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    legend_ax = ax.inset_axes([1.05, 0.08, 0.24, 0.66])
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    cax = legend_ax.inset_axes([0.18, 0.17, 0.18, 0.58])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_ticks(lr_values)
    colorbar.set_ticklabels([_format_lr(value) for value in lr_values])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
    colorbar.outline.set_edgecolor("#000000")
    colorbar.outline.set_linewidth(0.9)
    legend_ax.text(
        0.62,
        0.86,
        f"Highest LR\nwithin {near_optimal_pct:g}%\nof best loss",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 3,
    )
    legend_ax.text(
        0.62,
        0.07,
        f"eligible: >= {_format_completion(min_completion)} complete",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 5,
    )

    fig.subplots_adjust(left=0.16, right=0.76, top=0.88, bottom=0.14)
    save_figure(fig, output_path)


def create_optimal_lr_vs_parameters_plot(
    best_df: pl.DataFrame,
    output_path: Path,
    *,
    title_suffix: str = "",
    figsize: tuple[float, float] = (10.4, 5.8),
    color_shades: list[str] | tuple[str, ...] | None = None,
    legend_below: bool = False,
    show_best_legend: bool = True,
    show_title: bool = True,
    crop_output: bool = True,
    fit_label_x_multiplier: float = 1.16,
    fit_label_min_log_gap: float = 0.075,
    fit_label_fontsize: float | None = None,
    x_max_multiplier: float = 1.55,
    y_limits: tuple[float, float] | None = None,
    x_label: str = "Non-embedding Parameters",
    y_label: str = "Best Learning Rate",
    enforce_y_axis_units: bool = True,
    subplot_left: float | None = None,
) -> None:
    depths = best_df["target_depth"].unique().sort().to_list()
    panel_color_shades = color_shades or COLOR_SHADES
    apply_plot_style()
    if legend_below:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.45], hspace=0.40)
        ax = fig.add_subplot(grid[0])
        legend_ax = fig.add_subplot(grid[1])
    else:
        fig, (ax, legend_ax) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [5.2, 1.45], "wspace": 0.30},
        )

    all_y: list[float] = []
    all_x: list[float] = []
    fit_labels: list[dict[str, object]] = []
    for index, depth in enumerate(depths):
        panel_df = best_df.filter(pl.col("target_depth") == depth).sort(
            "non_embedding_params"
        )
        x = panel_df["non_embedding_params"].to_numpy().astype(float)
        y = panel_df["optimal_lr"].to_numpy().astype(float)
        boundary = panel_df["best_on_boundary"].to_numpy()
        color = plt.colormaps[panel_color_shades[index % len(panel_color_shades)]](0.76)
        ax.plot(x, y, color=color, linewidth=1.5, zorder=2)
        ax.scatter(
            x[~boundary],
            y[~boundary],
            color=color,
            edgecolors="black",
            linewidths=0.7,
            s=48,
            zorder=4,
        )
        if np.any(boundary):
            ax.scatter(
                x[boundary],
                y[boundary],
                color=color,
                edgecolors="black",
                linewidths=1.0,
                marker="s",
                s=58,
                zorder=5,
            )
        if len(x) >= 2:
            slope, intercept = _fit_power_law(x, y)
            x_fit = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ax.plot(x_fit, y_fit, color=color, linestyle=":", linewidth=1.4, zorder=3)
            fit_labels.append(
                {
                    "slope": slope,
                    "label_y": float(
                        10 ** (slope * np.log10(float(np.max(x))) + intercept)
                    ),
                    "color": color,
                }
            )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_power(value))
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_lr(value))
    )
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if all_x:
        ax.set_xlim(
            float(np.min(all_x)) / 1.25,
            float(np.max(all_x)) * x_max_multiplier,
        )
    if y_limits is not None:
        ax.set_ylim(*y_limits)
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
    if fit_labels:
        y_min, y_max = ax.get_ylim()
        label_x = float(np.max(all_x)) * fit_label_x_multiplier
        sorted_labels = sorted(
            fit_labels, key=lambda value: float(value["label_y"]), reverse=True
        )
        label_logs: list[float] = []
        lower_log = np.log10(y_min * 1.08)
        upper_log = np.log10(y_max / 1.08)
        for item in sorted_labels:
            desired_log = float(
                np.clip(np.log10(float(item["label_y"])), lower_log, upper_log)
            )
            if label_logs:
                desired_log = min(desired_log, label_logs[-1] - fit_label_min_log_gap)
            label_logs.append(max(desired_log, lower_log))

        for label_log, item in zip(label_logs, sorted_labels, strict=True):
            ax.text(
                label_x,
                10**label_log,
                rf"$\alpha_N$={float(item['slope']):.2f}",
                color=item["color"],
                fontsize=fit_label_fontsize or FONT_SIZE_TICKS,
                ha="left",
                va="center",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.8,
                },
                clip_on=False,
            )
    handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps[panel_color_shades[index % len(panel_color_shades)]](
                0.76
            ),
            linewidth=2,
            label=str(depth),
        )
        for index, depth in enumerate(depths)
    ]
    legend_ax.set_axis_off()
    legend_ax.grid(False)
    if legend_below:
        layers_bounds = (0.01, 0.04, 0.65 if show_best_legend else 0.98, 0.92)
        layers_box = legend_ax.inset_axes(layers_bounds)
    else:
        layers_box = legend_ax.inset_axes([0.02, 0.43, 0.96, 0.50])
    layers_box.set_xticks([])
    layers_box.set_yticks([])
    _style_panel_box(layers_box, rounded=True, draw_box=not legend_below)
    if legend_below:
        _add_isotropic_rounded_frame(legend_ax, layers_bounds)
    layers_box.legend(
        handles=handles,
        title="Number of Layers",
        frameon=False,
        loc="center",
        fontsize=FONT_SIZE_LEGEND - 4,
        title_fontsize=FONT_SIZE_LEGEND - 3,
        borderaxespad=0.0,
        labelspacing=0.72,
        handlelength=1.6,
        handletextpad=0.5 if legend_below else 0.65,
        columnspacing=0.9 if legend_below else 2.0,
        borderpad=0.8,
        ncol=len(depths) if legend_below else 1,
    )

    if show_best_legend:
        if legend_below:
            best_bounds = (0.69, 0.04, 0.30, 0.92)
            best_box = legend_ax.inset_axes(best_bounds)
        else:
            best_box = legend_ax.inset_axes([0.02, 0.08, 0.96, 0.28])
        best_box.set_xticks([])
        best_box.set_yticks([])
        _style_panel_box(best_box, rounded=True, draw_box=not legend_below)
        if legend_below:
            _add_isotropic_rounded_frame(legend_ax, best_bounds)
        best_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="s",
                linestyle="None",
                markersize=6,
                markerfacecolor="white",
                label="On Boundary",
            ),
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="#777777",
                label="Quadratic Approx.",
            ),
        ]
        best_box.legend(
            handles=best_handles,
            title="Best Learning Rate",
            frameon=False,
            loc="center",
            fontsize=FONT_SIZE_LEGEND - 5,
            title_fontsize=FONT_SIZE_LEGEND - 4,
            borderaxespad=0.0,
            labelspacing=0.65,
            handlelength=1.4,
            handletextpad=0.6,
            borderpad=0.7,
            ncol=1,
        )
    if show_title:
        fig.suptitle(
            f"Optimal LR Scaling with Model Size{title_suffix}",
            fontsize=FONT_SIZE_TITLE,
            x=0.5 if legend_below else 0.53,
            y=0.96,
        )
    if legend_below:
        fig.subplots_adjust(
            left=(
                subplot_left
                if subplot_left is not None
                else (0.18 if not show_title else 0.12)
            ),
            right=0.89 if not crop_output else 0.98,
            top=0.84 if show_title else 0.95,
            bottom=0.05,
            hspace=0.40,
        )
    else:
        fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.14, wspace=0.30)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def create_optimal_lr_vs_depth_plot(
    best_df: pl.DataFrame,
    output_path: Path,
    *,
    title_suffix: str = "",
    figsize: tuple[float, float] = (10.4, 5.8),
    legend_below: bool = False,
    show_best_legend: bool = True,
    show_title: bool = True,
    crop_output: bool = True,
    y_limits: tuple[float, float] | None = None,
    legend_box_width_scale: float = 1.0,
    fit_label_x_multiplier: float = 1.18,
    fit_label_min_log_gap: float = 0.075,
    fit_label_fontsize: float | None = None,
    x_label: str = "Depth at Matched Parameter Target",
    y_label: str = "Best Learning Rate",
    enforce_y_axis_units: bool = True,
    subplot_left: float | None = None,
) -> None:
    target_params = best_df["target_non_embedding_params"].unique().sort().to_list()
    apply_plot_style()
    if legend_below:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.45], hspace=0.40)
        ax = fig.add_subplot(grid[0])
        legend_ax = fig.add_subplot(grid[1])
    else:
        fig, (ax, legend_ax) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [5.2, 1.45], "wspace": 0.30},
        )

    all_x: list[float] = []
    fit_labels: list[dict[str, object]] = []
    for index, target_param in enumerate(target_params):
        panel_df = best_df.filter(
            pl.col("target_non_embedding_params") == target_param
        ).sort("target_depth")
        x = panel_df["target_depth"].to_numpy().astype(float)
        y = panel_df["optimal_lr"].to_numpy().astype(float)
        boundary = panel_df["best_on_boundary"].to_numpy()
        color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        ax.plot(x, y, color=color, linewidth=1.5, zorder=2)
        ax.scatter(
            x[~boundary],
            y[~boundary],
            color=color,
            edgecolors="black",
            linewidths=0.7,
            s=48,
            zorder=4,
        )
        if np.any(boundary):
            ax.scatter(
                x[boundary],
                y[boundary],
                color=color,
                edgecolors="black",
                linewidths=1.0,
                marker="s",
                s=58,
                zorder=5,
            )
        if len(x) >= 2:
            slope, intercept = _fit_power_law(x, y)
            x_fit = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ax.plot(x_fit, y_fit, color=color, linestyle=":", linewidth=1.4, zorder=3)
            fit_labels.append(
                {
                    "slope": slope,
                    "label_y": float(
                        10 ** (slope * np.log10(float(np.max(x))) + intercept)
                    ),
                    "color": color,
                }
            )
        all_x.extend(x.tolist())

    depths = best_df["target_depth"].unique().sort().to_numpy().astype(float)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_locator(mticker.FixedLocator(depths))
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter([str(int(value)) for value in depths])
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_lr(value))
    )
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if all_x:
        ax.set_xlim(float(np.min(all_x)) / 1.20, float(np.max(all_x)) * 1.65)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
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
    if fit_labels:
        y_min, y_max = ax.get_ylim()
        label_x = float(np.max(all_x)) * fit_label_x_multiplier
        sorted_labels = sorted(
            fit_labels, key=lambda value: float(value["label_y"]), reverse=True
        )
        label_logs: list[float] = []
        lower_log = np.log10(y_min * 1.08)
        upper_log = np.log10(y_max / 1.08)
        for item in sorted_labels:
            desired_log = float(
                np.clip(np.log10(float(item["label_y"])), lower_log, upper_log)
            )
            if label_logs:
                desired_log = min(
                    desired_log,
                    label_logs[-1] - fit_label_min_log_gap,
                )
            label_logs.append(max(desired_log, lower_log))

        for label_log, item in zip(label_logs, sorted_labels, strict=True):
            ax.text(
                label_x,
                10**label_log,
                rf"$\alpha_D$={float(item['slope']):.2f}",
                color=item["color"],
                fontsize=fit_label_fontsize or FONT_SIZE_TICKS,
                ha="left",
                va="center",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.8,
                },
                clip_on=False,
            )
    handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76),
            linewidth=2,
            label=_format_power(target_param),
        )
        for index, target_param in enumerate(target_params)
    ]
    legend_ax.set_axis_off()
    legend_ax.grid(False)
    if legend_below:
        base_width = 0.65 if show_best_legend else 0.98
        target_width = base_width * legend_box_width_scale
        target_bounds = ((1.0 - target_width) / 2.0, 0.04, target_width, 0.92)
        target_box = legend_ax.inset_axes(target_bounds)
    else:
        target_box = legend_ax.inset_axes([0.02, 0.43, 0.96, 0.50])
    target_box.set_xticks([])
    target_box.set_yticks([])
    _style_panel_box(target_box, rounded=True, draw_box=not legend_below)
    if legend_below:
        _add_isotropic_rounded_frame(legend_ax, target_bounds)
    target_box.legend(
        handles=handles,
        title="Target Parameters",
        frameon=False,
        loc="center",
        fontsize=FONT_SIZE_LEGEND - 4,
        title_fontsize=FONT_SIZE_LEGEND - 3,
        borderaxespad=0.0,
        labelspacing=0.72,
        handlelength=1.6,
        handletextpad=0.5 if legend_below else 0.65,
        columnspacing=0.9 if legend_below else 2.0,
        borderpad=0.8,
        ncol=len(target_params) if legend_below else 1,
    )

    if show_best_legend:
        if legend_below:
            best_bounds = (0.69, 0.04, 0.30, 0.92)
            best_box = legend_ax.inset_axes(best_bounds)
        else:
            best_box = legend_ax.inset_axes([0.02, 0.08, 0.96, 0.28])
        best_box.set_xticks([])
        best_box.set_yticks([])
        _style_panel_box(best_box, rounded=True, draw_box=not legend_below)
        if legend_below:
            _add_isotropic_rounded_frame(legend_ax, best_bounds)
        best_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="s",
                linestyle="None",
                markersize=6,
                markerfacecolor="white",
                label="On Boundary",
            ),
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="#777777",
                label="Quadratic Approx.",
            ),
        ]
        best_box.legend(
            handles=best_handles,
            title="Best Learning Rate",
            frameon=False,
            loc="center",
            fontsize=FONT_SIZE_LEGEND - 5,
            title_fontsize=FONT_SIZE_LEGEND - 4,
            borderaxespad=0.0,
            labelspacing=0.65,
            handlelength=1.4,
            handletextpad=0.6,
            borderpad=0.7,
            ncol=1,
        )
    if show_title:
        fig.suptitle(
            f"Optimal LR Scaling with Depth{title_suffix}",
            fontsize=FONT_SIZE_TITLE,
            x=0.5 if legend_below else 0.53,
            y=0.96,
        )
    if legend_below:
        fig.subplots_adjust(
            left=(
                subplot_left
                if subplot_left is not None
                else (0.18 if not show_title else 0.12)
            ),
            right=0.89 if not crop_output else 0.98,
            top=0.84 if show_title else 0.95,
            bottom=0.05,
            hspace=0.40,
        )
    else:
        fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.14, wspace=0.30)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=crop_output,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def create_optimal_lr_vs_width_plot(
    best_df: pl.DataFrame, output_path: Path, *, title_suffix: str = ""
) -> None:
    target_params = best_df["target_non_embedding_params"].unique().sort().to_list()
    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10.4, 5.8),
        gridspec_kw={"width_ratios": [5.2, 1.45], "wspace": 0.30},
    )

    all_x: list[float] = []
    for index, target_param in enumerate(target_params):
        panel_df = best_df.filter(
            pl.col("target_non_embedding_params") == target_param
        ).sort("model.d_model")
        x = panel_df["model.d_model"].to_numpy().astype(float)
        y = panel_df["optimal_lr"].to_numpy().astype(float)
        depths = panel_df["target_depth"].to_numpy().astype(int)
        boundary = panel_df["best_on_boundary"].to_numpy()
        color = plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        ax.plot(x, y, color=color, linewidth=1.5, zorder=2)
        ax.scatter(
            x[~boundary],
            y[~boundary],
            color=color,
            edgecolors="black",
            linewidths=0.7,
            s=48,
            zorder=4,
        )
        if np.any(boundary):
            ax.scatter(
                x[boundary],
                y[boundary],
                color=color,
                edgecolors="black",
                linewidths=1.0,
                marker="s",
                s=58,
                zorder=5,
            )
        for width, lr, depth in zip(x, y, depths, strict=True):
            ax.text(
                width * 1.03,
                lr,
                f"D{depth}",
                color="#2f2a24",
                fontsize=FONT_SIZE_TICKS - 1,
                va="center",
            )
        if len(x) >= 2:
            slope, intercept = _fit_power_law(x, y)
            x_fit = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ax.plot(x_fit, y_fit, color=color, linestyle=":", linewidth=1.4, zorder=3)
        all_x.extend(x.tolist())

    widths = best_df["model.d_model"].unique().sort().to_numpy().astype(float)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width at Matched Parameter Target", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Best Learning Rate", fontsize=FONT_SIZE_LABELS)
    ax.xaxis.set_major_locator(mticker.FixedLocator(widths))
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter([str(int(value)) for value in widths])
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: _format_lr(value))
    )
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if all_x:
        ax.set_xlim(float(np.min(all_x)) / 1.25, float(np.max(all_x)) * 1.55)
    ax.tick_params(axis="x", rotation=35)
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
    handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76),
            linewidth=2,
            label=_format_power(target_param),
        )
        for index, target_param in enumerate(target_params)
    ]
    legend_ax.set_axis_off()
    legend_ax.grid(False)

    target_box = legend_ax.inset_axes([0.02, 0.43, 0.96, 0.50])
    target_box.set_xticks([])
    target_box.set_yticks([])
    _style_panel_box(target_box, rounded=True)
    target_box.legend(
        handles=handles,
        title="Target Parameters",
        frameon=False,
        loc="center",
        fontsize=FONT_SIZE_LEGEND - 4,
        title_fontsize=FONT_SIZE_LEGEND - 3,
        borderaxespad=0.0,
        labelspacing=0.72,
        handlelength=1.6,
        handletextpad=0.65,
        borderpad=0.8,
    )

    best_box = legend_ax.inset_axes([0.02, 0.08, 0.96, 0.28])
    best_box.set_xticks([])
    best_box.set_yticks([])
    _style_panel_box(best_box, rounded=True)
    best_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="s",
            linestyle="None",
            markersize=6,
            markerfacecolor="white",
            label="On Boundary",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="#777777",
            label="Quadratic Approx.",
        ),
    ]
    best_box.legend(
        handles=best_handles,
        title="Best Learning Rate",
        frameon=False,
        loc="center",
        fontsize=FONT_SIZE_LEGEND - 5,
        title_fontsize=FONT_SIZE_LEGEND - 4,
        borderaxespad=0.0,
        labelspacing=0.65,
        handlelength=1.4,
        handletextpad=0.6,
        borderpad=0.7,
    )
    fig.suptitle(
        f"Optimal LR Scaling with Width{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        x=0.53,
        y=0.96,
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.17, wspace=0.30)
    save_figure(fig, output_path, apply_default_adjust=False)


def create_formula_lr_table_plot(
    output_path: Path,
    *,
    intercept: float = TABLE_LOG10_INTERCEPT,
    param_exponent: float = TABLE_PARAM_EXPONENT,
    depth_exponent: float = TABLE_DEPTH_EXPONENT,
) -> None:
    target_params, depths, lr_matrix = _formula_lr_table_values(
        intercept=intercept,
        param_exponent=param_exponent,
        depth_exponent=depth_exponent,
    )

    apply_plot_style()
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.8),
        gridspec_kw={"width_ratios": [5.2, 2.5], "wspace": 0.18},
    )
    cmap = plt.colormaps["viridis_r"]
    norm = mpl.colors.LogNorm(
        vmin=float(np.min(lr_matrix)), vmax=float(np.max(lr_matrix))
    )

    ax.imshow(lr_matrix, cmap=cmap, norm=norm, aspect="auto", alpha=0.82)
    for row_index, target_param in enumerate(target_params):
        for col_index, depth in enumerate(depths):
            value = lr_matrix[row_index, col_index]
            r, g, b, _ = cmap(norm(value))
            text_color = (
                "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.46 else "#111111"
            )
            ax.text(
                col_index,
                row_index,
                _format_table_lr(value),
                ha="center",
                va="center",
                fontsize=FONT_SIZE_TICKS,
                color=text_color,
            )

    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([str(depth) for depth in depths])
    ax.set_yticks(np.arange(len(target_params)))
    ax.set_yticklabels([_format_power(value) for value in target_params])
    ax.set_xlabel("Depth (D)", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(r"Non-embedding Parameters ($N$)", fontsize=FONT_SIZE_LABELS)
    ax.set_xticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_params), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
    ax.grid(False, which="major")

    legend_ax.set_axis_off()
    legend_ax.grid(False)
    coefficient = 10**intercept
    coefficient_exponent = int(np.floor(np.log10(coefficient)))
    coefficient_mantissa = coefficient / (10**coefficient_exponent)

    formula_box = legend_ax.inset_axes([0.02, 0.58, 0.96, 0.34])
    formula_box.set_xticks([])
    formula_box.set_yticks([])
    _style_panel_box(formula_box, rounded=True)
    formula_box.text(
        0.5,
        0.78,
        "Formula",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 3,
    )
    formula_box.text(
        0.5,
        0.52,
        r"$\mathrm{LR} = \lambda_0 N^{\alpha_N}D^{\alpha_D}$",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 4,
    )
    formula_box.text(
        0.5,
        0.27,
        rf"$\lambda_0={coefficient_mantissa:.2f}\times10^{{{coefficient_exponent}}},\ \alpha_N={param_exponent:.3f},\ \alpha_D={depth_exponent:.3f}$",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 8,
    )

    color_box = legend_ax.inset_axes([0.02, 0.08, 0.96, 0.40])
    color_box.set_xticks([])
    color_box.set_yticks([])
    _style_panel_box(color_box, rounded=True)
    cax = color_box.inset_axes([0.12, 0.34, 0.76, 0.18])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    colorbar.set_ticks([float(np.min(lr_matrix)), 1e-4, float(np.max(lr_matrix))])
    colorbar.set_ticklabels(
        [
            _format_table_lr(float(np.min(lr_matrix))),
            "1e-4",
            _format_table_lr(float(np.max(lr_matrix))),
        ]
    )
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
    colorbar.outline.set_edgecolor("#000000")
    colorbar.outline.set_linewidth(0.9)
    color_box.text(
        0.5,
        0.84,
        "Optimal LR",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LEGEND - 3,
    )

    fig.suptitle(
        "Formula-implied Optimal Learning Rate", fontsize=FONT_SIZE_TITLE, x=0.5, y=0.99
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.82, bottom=0.14, wspace=0.18)
    save_figure(fig, output_path, apply_default_adjust=False)


def write_formula_lr_table_csv(
    output_path: Path,
    *,
    intercept: float = TABLE_LOG10_INTERCEPT,
    param_exponent: float = TABLE_PARAM_EXPONENT,
    depth_exponent: float = TABLE_DEPTH_EXPONENT,
) -> None:
    target_params, depths, lr_matrix = _formula_lr_table_values(
        intercept=intercept,
        param_exponent=param_exponent,
        depth_exponent=depth_exponent,
    )
    ensure_parent_dir(output_path)
    fieldnames = [
        "non_embedding_parameters",
        "non_embedding_parameters_label",
        *(f"depth_{depth}" for depth in depths),
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, target_param in enumerate(target_params):
            row: dict[str, object] = {
                "non_embedding_parameters": int(target_param),
                "non_embedding_parameters_label": _format_power(target_param),
            }
            row.update(
                {
                    f"depth_{depth}": f"{lr_matrix[row_index, col_index]:.8g}"
                    for col_index, depth in enumerate(depths)
                }
            )
            writer.writerow(row)


def create_power_law_fit_plot(
    best_df: pl.DataFrame, output_path: Path, *, title_suffix: str = ""
) -> None:
    params, depths, observed_lr, boundary = _observed_lr_arrays(best_df)
    models = _fit_power_law_models(best_df)
    panel_keys = ["fixed_anchor", "fixed_fit_k", "free"]
    depth_values = sorted({int(value) for value in depths})
    markers = ["o", "s", "^", "D", "P", "X"]
    colors = [
        plt.colormaps[COLOR_SHADES[index % len(COLOR_SHADES)]](0.76)
        for index, _ in enumerate(depth_values)
    ]
    min_lr = min(
        float(np.min(observed_lr)),
        *(float(np.min(models[key]["predicted_lr"])) for key in panel_keys),
    )
    max_lr = max(
        float(np.max(observed_lr)),
        *(float(np.max(models[key]["predicted_lr"])) for key in panel_keys),
    )
    limits = (min_lr / 1.8, max_lr * 1.8)

    apply_plot_style()
    fig, axes = plt.subplots(
        1, len(panel_keys), figsize=(13.8, 4.7), sharex=True, sharey=True
    )
    for ax, key in zip(axes, panel_keys, strict=True):
        model = models[key]
        predicted_lr = model["predicted_lr"]
        ax.plot(limits, limits, color="#2f2a24", linewidth=1.1, linestyle=":", zorder=1)
        for depth_index, depth in enumerate(depth_values):
            mask = depths == depth
            ax.scatter(
                observed_lr[mask & ~boundary],
                predicted_lr[mask & ~boundary],
                color=colors[depth_index],
                marker=markers[depth_index % len(markers)],
                edgecolors="black",
                linewidths=0.7,
                s=58,
                zorder=3,
                label=f"D={depth}",
            )
            if np.any(mask & boundary):
                ax.scatter(
                    observed_lr[mask & boundary],
                    predicted_lr[mask & boundary],
                    facecolors="white",
                    marker=markers[depth_index % len(markers)],
                    edgecolors="black",
                    linewidths=1.0,
                    s=68,
                    zorder=4,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_table_lr(value))
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_table_lr(value))
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
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
        ax.set_title(
            f"{model['label']}\nRMSE={model['rmse_log10']:.2f} dex",
            fontsize=FONT_SIZE_LABELS - 5,
            y=1.03,
        )
    axes[0].set_ylabel("Predicted Best LR", fontsize=FONT_SIZE_LABELS)
    for ax in axes:
        ax.set_xlabel("Observed Best LR", fontsize=FONT_SIZE_LABELS)
    handles = [
        mlines.Line2D(
            [],
            [],
            color=colors[index],
            marker=markers[index % len(markers)],
            linestyle="None",
            markeredgecolor="black",
            markersize=7,
            label=f"D={depth}",
        )
        for index, depth in enumerate(depth_values)
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markersize=7,
            label="Discrete fallback",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.52, 0.94),
        fontsize=FONT_SIZE_LEGEND - 4,
    )
    fig.suptitle(
        f"Observed vs Predicted Quadratic Best LR{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        x=0.52,
        y=1.02,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.16, wspace=0.18)
    save_figure(fig, output_path)


def create_power_law_residual_plot(
    best_df: pl.DataFrame, output_path: Path, *, title_suffix: str = ""
) -> None:
    params, depths, observed_lr, _ = _observed_lr_arrays(best_df)
    models = _fit_power_law_models(best_df)
    panel_keys = ["fixed_anchor", "fixed_fit_k", "free"]
    target_params = sorted({float(value) for value in params})
    depth_values = sorted({int(value) for value in depths})
    residuals_by_model = {
        key: np.log10(observed_lr)
        - np.asarray(models[key]["predicted_log_lr"], dtype=float)
        for key in panel_keys
    }
    max_abs_residual = max(
        float(np.max(np.abs(values))) for values in residuals_by_model.values()
    )
    max_abs_residual = max(max_abs_residual, 0.05)
    norm = mpl.colors.TwoSlopeNorm(
        vmin=-max_abs_residual, vcenter=0.0, vmax=max_abs_residual
    )

    apply_plot_style()
    fig, axes = plt.subplots(
        1, len(panel_keys), figsize=(13.6, 4.9), sharex=True, sharey=True
    )
    image = None
    for ax, key in zip(axes, panel_keys, strict=True):
        matrix = np.full((len(target_params), len(depth_values)), np.nan)
        for value_index, (param, depth) in enumerate(zip(params, depths, strict=True)):
            row_index = target_params.index(float(param))
            col_index = depth_values.index(int(depth))
            matrix[row_index, col_index] = residuals_by_model[key][value_index]
        image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        for row_index, target_param in enumerate(target_params):
            for col_index, depth in enumerate(depth_values):
                value = matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                ax.text(
                    col_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_TICKS,
                    color="#111111",
                )
        ax.set_xticks(np.arange(len(depth_values)))
        ax.set_xticklabels([str(depth) for depth in depth_values])
        ax.set_yticks(np.arange(len(target_params)))
        ax.set_yticklabels([_format_power(value) for value in target_params])
        ax.set_xlabel("Depth (D)", fontsize=FONT_SIZE_LABELS - 2)
        ax.set_title(models[key]["label"], fontsize=FONT_SIZE_LABELS - 4, y=1.03)
        ax.set_xticks(np.arange(-0.5, len(depth_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(target_params), 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
    axes[0].set_ylabel(r"Non-embedding Parameters ($N$)", fontsize=FONT_SIZE_LABELS - 2)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.68, pad=0.025)
        colorbar.set_label("log10(observed / predicted)", fontsize=FONT_SIZE_LEGEND - 3)
        colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
    fig.suptitle(
        f"Power-law Residuals for Quadratic Best LR{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        x=0.48,
        y=1.02,
    )
    fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.14, wspace=0.14)
    save_figure(fig, output_path)


def create_power_law_model_comparison_plot(
    best_df: pl.DataFrame, output_path: Path, *, title_suffix: str = ""
) -> None:
    models = _fit_power_law_models(best_df)
    order = ["constant", "n_only", "d_only", "fixed_anchor", "fixed_fit_k", "free"]
    labels = [str(models[key]["label"]) for key in order]
    rmse = [float(models[key]["rmse_log10"]) for key in order]
    mae = [float(models[key]["mae_log10"]) for key in order]
    fitted = models["free"]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    x = np.arange(len(order))
    colors = ["#b8b8b8", "#7aa6c2", "#c78373", "#d7b46a", "#9bbc8f", "#5f8f73"]
    bars = ax.bar(x, rmse, color=colors, edgecolor="#2f2a24", linewidth=0.8, zorder=3)
    ax.scatter(x, mae, color="#2f2a24", marker="D", s=42, zorder=4, label="MAE")
    for bar, value in zip(bars, rmse, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(rmse) * 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_TICKS,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Log10 LR Error (dex)", fontsize=FONT_SIZE_LABELS)
    ax.set_title(
        f"Quadratic Best-LR Power-law Model Comparison{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        y=1.04,
    )
    ax.grid(
        True,
        axis="y",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
    ax.legend(frameon=False, loc="upper right", fontsize=FONT_SIZE_LEGEND - 3)
    ax.text(
        0.02,
        0.96,
        "free fit: "
        f"N^{float(fitted['param_exponent']):+.2f} "
        f"D^{float(fitted['depth_exponent']):+.2f}\n"
        "fixed formula: "
        f"N^{FORMULA_PARAM_EXPONENT:+.2f} "
        f"D^{FORMULA_DEPTH_EXPONENT:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE_LEGEND - 4,
    )
    fig.subplots_adjust(left=0.13, right=0.96, top=0.86, bottom=0.26)
    save_figure(fig, output_path)


def create_power_law_model_comparison_panel_plot(
    best_dfs_by_flops: list[tuple[float, pl.DataFrame]],
    output_path: Path,
) -> None:
    if not best_dfs_by_flops:
        raise ValueError("No best-LR data provided for model-comparison panel plot")

    order = ["constant", "d_only", "n_only", "free"]
    labels = ["const", r"$D$", r"$N$", r"$N+D$"]
    colors = ["#b8b8b8", "#c78373", "#7aa6c2", "#5f8f73"]
    summaries = []
    x_max = 0.0
    for training_flops, best_df in best_dfs_by_flops:
        models = _fit_power_law_models(best_df)
        rmse = [float(models[key]["rmse_log10"]) for key in order]
        mae = [float(models[key]["mae_log10"]) for key in order]
        x_max = max(x_max, max(rmse), max(mae))
        summaries.append((training_flops, models, rmse, mae))

    apply_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(12.4, 13.2), sharex=True)
    for ax, (training_flops, models, rmse, mae) in zip(
        axes.ravel(), summaries, strict=False
    ):
        y = np.arange(len(order))
        ax.barh(y, rmse, color=colors, edgecolor="#2f2a24", linewidth=0.6, zorder=3)
        ax.scatter(mae, y, color="#2f2a24", marker="D", s=30, zorder=4)
        ax.set_title(_format_flops_label(training_flops), fontsize=FONT_SIZE_LABELS - 5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_TICKS - 2)
        ax.invert_yaxis()
        ax.set_xlim(0.0, x_max * 1.18)
        ax.grid(
            True,
            axis="x",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    for ax in axes[-1, :]:
        ax.set_xlabel("Log10 LR Error (dex)", fontsize=FONT_SIZE_LABELS - 5)

    model_handles = [
        mpatches.Patch(
            facecolor=colors[index],
            edgecolor="none",
            label=labels[index].replace("\n", " "),
        )
        for index in range(len(order))
    ]
    metric_handles = [
        mpatches.Patch(facecolor="#d8d8d8", edgecolor="#2f2a24", label="RMSE"),
        mlines.Line2D(
            [],
            [],
            color="#2f2a24",
            marker="D",
            linestyle="None",
            markersize=5,
            label="MAE",
        ),
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.42, 0.965),
        fontsize=FONT_SIZE_LEGEND - 5,
    )
    fig.legend(
        handles=metric_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.78, 0.965),
        fontsize=FONT_SIZE_LEGEND - 5,
    )
    fig.suptitle(
        "Quadratic Best-LR Power-law Model Comparison",
        fontsize=FONT_SIZE_TITLE,
        x=0.52,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.92, bottom=0.06, wspace=0.14, hspace=0.36
    )
    save_figure(fig, output_path)


def create_power_law_bootstrap_plot(
    best_df: pl.DataFrame, output_path: Path, *, title_suffix: str = ""
) -> None:
    models = _fit_power_law_models(best_df)
    bootstrap = _bootstrap_power_law_exponents(best_df)
    fitted = models["free"]
    if bootstrap.size == 0:
        raise ValueError(
            "Could not bootstrap power-law exponents; sampled design matrices were singular"
        )

    param_ci = np.percentile(bootstrap[:, 0], [2.5, 50, 97.5])
    depth_ci = np.percentile(bootstrap[:, 1], [2.5, 50, 97.5])

    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    hist_specs = [
        (
            axes[0],
            bootstrap[:, 0],
            float(fitted["param_exponent"]),
            FORMULA_PARAM_EXPONENT,
            param_ci,
            "Encoder-size exponent",
        ),
        (
            axes[1],
            bootstrap[:, 1],
            float(fitted["depth_exponent"]),
            FORMULA_DEPTH_EXPONENT,
            depth_ci,
            "Depth exponent",
        ),
    ]
    for ax, values, fitted_value, fixed_value, ci, label in hist_specs:
        ax.hist(
            values,
            bins=34,
            color="#d8d8d8",
            edgecolor="#2f2a24",
            linewidth=0.5,
            zorder=3,
        )
        ax.axvspan(ci[0], ci[2], color="#5f8f73", alpha=0.18, zorder=2)
        ax.axvline(fitted_value, color="#2f2a24", linewidth=1.8, label="fitted")
        ax.axvline(
            fixed_value, color="#b24d3e", linewidth=1.8, linestyle=":", label="assumed"
        )
        ax.set_xlabel(label, fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel("Bootstrap Count", fontsize=FONT_SIZE_LABELS)
        ax.grid(
            True,
            axis="y",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
        ax.text(
            0.03,
            0.95,
            f"95% CI [{ci[0]:.2f}, {ci[2]:.2f}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_LEGEND - 4,
        )
    axes[1].legend(frameon=False, loc="upper right", fontsize=FONT_SIZE_LEGEND - 4)
    fig.suptitle(
        f"Bootstrap Uncertainty for Fitted LR Exponents{title_suffix}",
        fontsize=FONT_SIZE_TITLE,
        x=0.52,
        y=1.02,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.16, wspace=0.25)
    save_figure(fig, output_path)


def create_power_law_bootstrap_panel_plot(
    best_dfs_by_flops: list[tuple[float, pl.DataFrame]],
    output_path: Path,
    *,
    title: str = "Bootstrap Uncertainty for Fitted LR Exponents",
    x_limits: tuple[float, float] | None = None,
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> None:
    if not best_dfs_by_flops:
        raise ValueError("No best-LR data provided for bootstrap panel plot")

    panel_data = []
    param_values: list[float] = []
    depth_values: list[float] = []
    for training_flops, best_df in best_dfs_by_flops:
        models = _fit_power_law_models(best_df)
        bootstrap = _bootstrap_power_law_exponents(best_df)
        if bootstrap.size == 0:
            raise ValueError(
                f"Could not bootstrap power-law exponents for {_format_flops_label(training_flops)}"
            )
        panel_data.append((training_flops, models["free"], bootstrap))
        param_values.extend(bootstrap[:, 0].tolist())
        depth_values.extend(bootstrap[:, 1].tolist())

    if x_limits is None:
        x_min = min(min(param_values), min(depth_values)) - 0.08
        x_max = max(max(param_values), max(depth_values)) + 0.08
    else:
        x_min, x_max = x_limits
        if x_min >= x_max:
            raise ValueError("x_limits must be strictly increasing")

    apply_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(12.4, 13.2), sharex=True, sharey=True)
    for ax, (training_flops, fitted, bootstrap) in zip(
        axes.ravel(), panel_data, strict=False
    ):
        bins = np.linspace(x_min, x_max, 34)
        ax.hist(
            bootstrap[:, 0],
            bins=bins,
            color="#7aa6c2",
            alpha=0.58,
            edgecolor="#2f2a24",
            linewidth=0.35,
            label="N",
        )
        ax.hist(
            bootstrap[:, 1],
            bins=bins,
            color="#c78373",
            alpha=0.52,
            edgecolor="#2f2a24",
            linewidth=0.35,
            label="D",
        )
        ax.axvline(float(fitted["param_exponent"]), color="#2b5c75", linewidth=1.5)
        ax.axvline(float(fitted["depth_exponent"]), color="#8b4d43", linewidth=1.5)
        param_ci = np.percentile(bootstrap[:, 0], [2.5, 97.5])
        depth_ci = np.percentile(bootstrap[:, 1], [2.5, 97.5])
        ax.text(
            0.03,
            0.94,
            f"N [{param_ci[0]:.2f}, {param_ci[1]:.2f}]\nD [{depth_ci[0]:.2f}, {depth_ci[1]:.2f}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_TICKS - 1,
        )
        ax.set_title(_format_flops_label(training_flops), fontsize=FONT_SIZE_LABELS - 5)
        ax.grid(
            True,
            axis="y",
            color="#b3b3b3",
            linestyle="-",
            linewidth=0.4,
            alpha=0.8,
            zorder=1,
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)
        ax.set_xlim(x_min, x_max)

    for ax in axes[:, 0]:
        ax.set_ylabel("Bootstrap Count", fontsize=FONT_SIZE_LABELS - 5)
    for ax in axes[-1, :]:
        ax.set_xlabel("Exponent", fontsize=FONT_SIZE_LABELS - 5)

    handles = [
        mpatches.Patch(
            facecolor="#7aa6c2",
            alpha=0.58,
            edgecolor="#2f2a24",
            label="Size exponent",
        ),
        mpatches.Patch(
            facecolor="#c78373", alpha=0.52, edgecolor="#2f2a24", label="Depth exponent"
        ),
        mlines.Line2D([], [], color="#2f2a24", linewidth=1.5, label="fitted"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, 0.965),
        fontsize=FONT_SIZE_LEGEND - 5,
    )
    fig.suptitle(
        title,
        fontsize=FONT_SIZE_TITLE,
        x=0.52,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.92, bottom=0.07, wspace=0.14, hspace=0.36
    )
    save_figure(
        fig,
        output_path,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def _power_law_validation_summary(
    fraction: float, training_flops: float, best_df: pl.DataFrame
) -> dict[str, object]:
    models = _fit_power_law_models(best_df)
    bootstrap = _bootstrap_power_law_exponents(best_df)
    param_ci = np.percentile(bootstrap[:, 0], [2.5, 50, 97.5])
    depth_ci = np.percentile(bootstrap[:, 1], [2.5, 50, 97.5])
    method_counts = (
        best_df["lr_fit_method"].value_counts()
        if "lr_fit_method" in best_df.columns
        else pl.DataFrame()
    )
    n_quadratic = 0
    n_fallback = best_df.height
    if not method_counts.is_empty():
        count_map = {
            row["lr_fit_method"]: int(row["count"])
            for row in method_counts.iter_rows(named=True)
        }
        n_quadratic = count_map.get("quadratic", 0)
        n_fallback = best_df.height - n_quadratic
    summary: dict[str, object] = {
        "fraction": fraction,
        "training_flops": training_flops,
        "n_points": best_df.height,
        "n_quadratic": n_quadratic,
        "n_fallback": n_fallback,
        "free_param_exponent": float(models["free"]["param_exponent"]),
        "free_depth_exponent": float(models["free"]["depth_exponent"]),
        "free_intercept": float(models["free"]["intercept"]),
        "param_ci_low": float(param_ci[0]),
        "param_ci_high": float(param_ci[2]),
        "depth_ci_low": float(depth_ci[0]),
        "depth_ci_high": float(depth_ci[2]),
    }
    for key, model in models.items():
        summary[f"{key}_rmse_log10"] = float(model["rmse_log10"])
        summary[f"{key}_mae_log10"] = float(model["mae_log10"])
    return summary


def create_power_law_summary_plot(
    summary_rows: list[dict[str, object]], output_path: Path
) -> None:
    if not summary_rows:
        raise ValueError("No power-law summaries were provided")

    rows = sorted(summary_rows, key=lambda row: float(row["fraction"]))
    positions = np.arange(len(rows))
    flops_labels = [_format_flops_value(float(row["training_flops"])) for row in rows]
    model_order = ["constant", "n_only", "d_only", "free"]
    model_labels = {
        "constant": "constant",
        "n_only": r"$N$-only",
        "d_only": r"$D$-only",
        "free": r"$N+D$",
    }
    colors = {
        "constant": "#8d8d8d",
        "n_only": "#6b9bb8",
        "d_only": "#bd7567",
        "free": "#2f6f4e",
    }

    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    ax_rmse, ax_n, ax_d, ax_counts = axes.ravel()

    for key in model_order:
        ax_rmse.plot(
            positions,
            [float(row[f"{key}_rmse_log10"]) for row in rows],
            marker="o",
            linewidth=1.6,
            color=colors[key],
            label=model_labels[key],
        )
    ax_rmse.set_ylabel("RMSE in log10 LR (dex)", fontsize=FONT_SIZE_LABELS - 4)
    ax_rmse.set_title(
        "Model comparison across compute slices", fontsize=FONT_SIZE_LABELS - 3
    )
    ax_rmse.legend(frameon=False, fontsize=FONT_SIZE_LEGEND - 6, ncol=2)

    param_exponent = np.array([float(row["free_param_exponent"]) for row in rows])
    param_ci_low = np.array([float(row["param_ci_low"]) for row in rows])
    param_ci_high = np.array([float(row["param_ci_high"]) for row in rows])
    ax_n.fill_between(
        positions, param_ci_low, param_ci_high, color="#5f8f73", alpha=0.18, linewidth=0
    )
    ax_n.plot(
        positions,
        param_exponent,
        color="#2f6f4e",
        marker="o",
        linewidth=1.6,
        label=r"$N+D$ fit",
    )
    ax_n.set_ylabel(r"$N$ exponent", fontsize=FONT_SIZE_LABELS - 4)
    ax_n.set_title(
        "Size exponent with bootstrap 95% interval", fontsize=FONT_SIZE_LABELS - 3
    )
    ax_n.legend(frameon=False, fontsize=FONT_SIZE_LEGEND - 6)

    depth_exponent = np.array([float(row["free_depth_exponent"]) for row in rows])
    depth_ci_low = np.array([float(row["depth_ci_low"]) for row in rows])
    depth_ci_high = np.array([float(row["depth_ci_high"]) for row in rows])
    ax_d.fill_between(
        positions, depth_ci_low, depth_ci_high, color="#5f8f73", alpha=0.18, linewidth=0
    )
    ax_d.plot(
        positions,
        depth_exponent,
        color="#2f6f4e",
        marker="o",
        linewidth=1.6,
        label=r"$N+D$ fit",
    )
    ax_d.set_ylabel("Depth exponent", fontsize=FONT_SIZE_LABELS - 4)
    ax_d.set_title(
        "Depth exponent with bootstrap 95% interval", fontsize=FONT_SIZE_LABELS - 3
    )

    quadratic_counts = np.array([int(row["n_quadratic"]) for row in rows])
    fallback_counts = np.array([int(row["n_fallback"]) for row in rows])
    ax_counts.bar(
        positions,
        quadratic_counts,
        width=0.72,
        color="#5f8f73",
        edgecolor="#2f2a24",
        linewidth=0.6,
        label="quadratic",
    )
    ax_counts.bar(
        positions,
        fallback_counts,
        width=0.72,
        bottom=quadratic_counts,
        color="#d8d8d8",
        edgecolor="#2f2a24",
        linewidth=0.6,
        label="fallback",
    )
    ax_counts.set_ylabel("Architecture points", fontsize=FONT_SIZE_LABELS - 4)
    ax_counts.set_title(
        "Continuous-optimum fit availability", fontsize=FONT_SIZE_LABELS - 3
    )
    counts_legend = ax_counts.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        fontsize=FONT_SIZE_LEGEND - 6,
    )
    counts_legend.get_frame().set_facecolor("#ffffff")
    counts_legend.get_frame().set_edgecolor("#2f2a24")
    counts_legend.get_frame().set_linewidth(0.8)

    for ax in axes.ravel():
        ax.set_xlabel("Training FLOPs", fontsize=FONT_SIZE_LABELS - 4)
        ax.set_xticks(positions)
        ax.set_xticklabels(flops_labels, rotation=35, ha="right")
        ax.grid(
            True, color="#b3b3b3", linestyle="-", linewidth=0.4, alpha=0.8, zorder=1
        )
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    fig.suptitle(
        "Learning-rate Power-law Validation Summary",
        fontsize=FONT_SIZE_TITLE,
        x=0.52,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.11, wspace=0.24, hspace=0.55
    )
    save_figure(fig, output_path)


def create_power_laws_compute_slices_plot(
    summary_rows: list[dict[str, object]],
    output_path: Path,
    *,
    figsize: tuple[float, float] = (8.8, 5.8),
    y_limits: tuple[float, float] | None = None,
    show_title: bool = True,
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> None:
    """Render panel 1 of the validation summary as a standalone figure."""
    if not summary_rows:
        raise ValueError("No power-law summaries were provided")

    rows = sorted(summary_rows, key=lambda row: float(row["fraction"]))
    positions = np.arange(len(rows))
    flops_labels = [_format_flops_value(float(row["training_flops"])) for row in rows]
    model_order = ["constant", "n_only", "d_only", "free"]
    model_labels = {
        "constant": "constant",
        "n_only": r"$N$-only",
        "d_only": r"$D$-only",
        "free": r"$N+D$",
    }
    colors = {
        "constant": "#8d8d8d",
        "n_only": "#6b9bb8",
        "d_only": "#bd7567",
        "free": "#2f6f4e",
    }

    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    for key in model_order:
        ax.plot(
            positions,
            [float(row[f"{key}_rmse_log10"]) for row in rows],
            marker="o",
            linewidth=1.6,
            color=colors[key],
            label=model_labels[key],
        )

    ax.set_xlabel("Training FLOPs", fontsize=FONT_SIZE_LABELS - 4)
    ax.set_ylabel("RMSE in log10 LR (dex)", fontsize=FONT_SIZE_LABELS - 4)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels(flops_labels, rotation=35, ha="right")
    ax.grid(True, color="#b3b3b3", linestyle="-", linewidth=0.4, alpha=0.8, zorder=1)
    ax.legend(frameon=False, fontsize=FONT_SIZE_LEGEND - 4, ncol=2)
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS - 1)

    if show_title:
        fig.suptitle(
            "Power Laws Comparison across Compute Slices",
            fontsize=FONT_SIZE_TITLE,
            x=0.52,
            y=0.98,
        )
    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.84 if show_title else 0.96,
        bottom=0.20,
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def write_power_law_methods_note(
    output_path: Path,
    summary_rows: list[dict[str, object]],
    *,
    table_model: dict[str, object],
) -> None:
    ensure_parent_dir(output_path)
    rows = sorted(summary_rows, key=lambda row: float(row["fraction"]))
    lines = [
        "# Learning-rate power-law validation methods",
        "",
        "The empirical optimum for each architecture is estimated from eligible runs only, where eligibility means the sampled run reached the configured minimum completion threshold.",
        "",
        "For each `(N, D)` learning-rate sweep, the script first finds the eligible discrete learning rate with the lowest sampled training loss. It fits a quadratic in `log10(lr)` only when the winner's immediate lower and higher learning rates in the original sweep grid are both present, finite, and eligible. Incomplete points are not removed before defining adjacency, so the fit cannot span across a missing or unfinished run:",
        "",
        "```text",
        "loss = c0 + c1 log10(lr) + c2 log10(lr)^2",
        "```",
        "",
        "The continuous optimum is `log10(lr*) = -c1 / (2 c2)`. This quadratic estimate is accepted only when `c2 > 0` and the optimum lies inside the complete three-point fitting window. Otherwise the script falls back to the best eligible discrete learning rate.",
        "",
        "Fallback points are retained in the power-law fits rather than discarded, because the architecture grid is small. This means boundary winners are treated as exact optima even though they may be censored: if the best swept learning rate is at the largest LR, the true optimum could be higher than the sweep range. At the largest analyzed compute slice, the `N=1M, D=20` point is such a boundary fallback, so the final fitted table should be read as a smooth global fit rather than an anchored guarantee for that architecture.",
        "",
        "Power-law models are fit in log space:",
        "",
        "```text",
        "log10(lr*) = log10(lambda_0) + alpha_N log10(N) + alpha_D log10(D)",
        "```",
        "",
        "The reported error unit is dex, meaning base-10 logarithmic error. An error of `0.3 dex` is approximately a factor-of-two error because `10^0.3 ~= 2`.",
        "",
        "Compared models are: constant LR, non-embedding-parameter-only power law, depth-only power law, fixed exponents with anchored or fitted normalization, and a free non-embedding-parameter/depth power law.",
        "",
        "Bootstrap intervals resample architecture points with replacement and refit the free N,D power law. They are intended as sensitivity intervals for this small grid, not as strong asymptotic guarantees.",
        "",
        "The final recommendation table uses the free N,D power-law fit from the largest analyzed compute slice after quadratic optimum estimation.",
        "",
        (
            "The compute slices are eight linearly spaced nonzero training-FLOP values ending at "
            f"{100 * max(float(row['fraction']) for row in rows):.3g}% of the matched budget. "
            f"With a target budget of `{max(float(row['training_flops']) / float(row['fraction']) for row in rows):.2e}` FLOPs, "
            f"the largest analyzed value is `{max(float(row['training_flops']) for row in rows):.2e}` FLOPs. "
            "Linear spacing is used because these slices are intended to sample training progress evenly in cumulative compute; "
            "log spacing would overweight the arbitrary earliest part of training."
        ),
        "",
        "## Final table fit",
        "",
        "```text",
        f"log10(lr*) = {float(table_model['intercept']):.12f} "
        f"{float(table_model['param_exponent']):+.12f} log10(N) "
        f"{float(table_model['depth_exponent']):+.12f} log10(D)",
        "```",
        "",
        "## Per-slice summary",
        "",
        "| Training FLOPs | FLOP fraction | N exponent | depth exponent | free RMSE dex | quadratic points | fallback points |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {_format_flops_value(float(row['training_flops']))} "
            f"| {100 * float(row['fraction']):.3g}% "
            f"| {float(row['free_param_exponent']):.3f} "
            f"| {float(row['free_depth_exponent']):.3f} "
            f"| {float(row['free_rmse_log10']):.3f} "
            f"| {int(row['n_quadratic'])} "
            f"| {int(row['n_fallback'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.min_completion <= 1.0:
        raise ValueError("--min-completion must be in (0, 1]")
    if args.near_optimal_pct < 0:
        raise ValueError("--near-optimal-pct must be non-negative")

    metrics_csv, exported_from_aim = _ensure_metrics_csv(args)
    prepared_df, wrote_prepared_data = _ensure_prepared_data(args, metrics_csv)
    saved_outputs: list[Path] = []
    validation_summaries: list[dict[str, object]] = []
    best_dfs_by_flops: list[tuple[float, pl.DataFrame]] = []
    table_model: dict[str, object] | None = None

    for flops_index, fraction in enumerate(
        _parse_flops_fractions(args.flops_fractions), start=1
    ):
        fraction_df = prepared_df.filter(
            (pl.col("analysis_fraction") - fraction).abs() < 1e-9
        )
        if fraction_df.is_empty():
            raise ValueError(f"No prepared rows found for FLOP fraction {fraction:g}")

        analysis_training_flops = float(
            fraction_df["analysis_training_flops"].drop_nulls().first()
        )
        terminal_df = _terminal_loss_df(fraction_df)
        best_df = _quadratic_best_lr_df(terminal_df, args.min_completion)
        best_dfs_by_flops.append((analysis_training_flops, best_df))
        validation_summary = _power_law_validation_summary(
            fraction, analysis_training_flops, best_df
        )
        validation_summaries.append(validation_summary)
        if np.isclose(fraction, 0.75):
            table_model = _fit_power_law_models(best_df)["free"]
        title_suffix = f" ({_format_flops_label(analysis_training_flops)})"
        output_paths = [
            _output_for_flops(args.sweep_output, analysis_training_flops, flops_index),
            _output_for_flops(
                args.param_scaling_output, analysis_training_flops, flops_index
            ),
            _output_for_flops(
                args.depth_scaling_output, analysis_training_flops, flops_index
            ),
            _output_for_flops(
                args.width_scaling_output, analysis_training_flops, flops_index
            ),
        ]

        create_lr_sweep_plot(
            terminal_df,
            output_paths[0],
            min_completion=args.min_completion,
            near_optimal_pct=args.near_optimal_pct,
            title_suffix=title_suffix,
        )
        create_optimal_lr_vs_parameters_plot(
            best_df, output_paths[1], title_suffix=title_suffix
        )
        create_optimal_lr_vs_depth_plot(
            best_df, output_paths[2], title_suffix=title_suffix
        )
        create_optimal_lr_vs_width_plot(
            best_df, output_paths[3], title_suffix=title_suffix
        )
        saved_outputs.extend(output_paths)

    if table_model is None:
        table_model = _fit_power_law_models(best_df)["free"]

    create_power_law_model_comparison_panel_plot(
        best_dfs_by_flops, args.power_law_comparison_output
    )
    saved_outputs.append(args.power_law_comparison_output)

    create_power_law_bootstrap_panel_plot(
        best_dfs_by_flops, args.power_law_bootstrap_output
    )
    saved_outputs.append(args.power_law_bootstrap_output)

    create_power_law_summary_plot(validation_summaries, args.power_law_summary_output)
    saved_outputs.append(args.power_law_summary_output)

    create_power_laws_compute_slices_plot(
        validation_summaries, args.power_laws_compute_slices_output
    )
    saved_outputs.append(args.power_laws_compute_slices_output)

    create_formula_lr_table_plot(
        args.formula_lr_table_output,
        intercept=float(table_model["intercept"]),
        param_exponent=float(table_model["param_exponent"]),
        depth_exponent=float(table_model["depth_exponent"]),
    )
    saved_outputs.append(args.formula_lr_table_output)
    write_formula_lr_table_csv(
        args.formula_lr_table_csv_output,
        intercept=float(table_model["intercept"]),
        param_exponent=float(table_model["param_exponent"]),
        depth_exponent=float(table_model["depth_exponent"]),
    )
    saved_outputs.append(args.formula_lr_table_csv_output)

    if exported_from_aim:
        print(f"Exported metrics CSV to {metrics_csv}")
    else:
        print(f"Reused cached metrics CSV at {metrics_csv}")
    if wrote_prepared_data:
        print(f"Wrote prepared plot data to {args.prepared_data}")
    else:
        print(f"Reused prepared plot data at {args.prepared_data}")

    for output in saved_outputs:
        print(f"Saved output to {output}")


if __name__ == "__main__":
    main()
