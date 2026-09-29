from __future__ import annotations

import argparse
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

from plotting import geneformer_like as downstream_plots
from plotting.aim import export_metrics_csv
from plotting.training_loss import (
    FONT_SIZE_LABELS,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICKS,
    FONT_SIZE_TITLE,
    SCGPT_BCE_CONFIG,
    SCGPT_MSE_CONFIG,
    _add_isotropic_rounded_frame,
    _format_param_count,
    _make_subplot_title,
    _style_panel_box,
    _format_loss_tick,
    create_training_loss_plot_from_df,
    load_training_loss_data,
    TrainingLossPlotConfig,
)
from tools.config_report import TRAINING_FLOPS_MULTIPLIER, estimate_forward_flops
from core.config import ScalingConfig, load_yaml
from plotting.utils import COLOR_SHADES, PLOTS_DIR, ROOT_DIR, apply_plot_style, ensure_parent_dir, exponential_moving_average, remove_bounding_box, save_figure


DEFAULT_AIM_REPO = ROOT_DIR / "aim-repo" / "scgpt_like"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "scgpt_like"
DEFAULT_MASTER_CONFIG = ROOT_DIR / "publication" / "recipes" / "scgpt_like.yml"
DEFAULT_CSV = PLOTS_DIR / "scgpt_like" / "01_training_loss_metrics.csv"
DEFAULT_PREPARED_DATA = PLOTS_DIR / "scgpt_like" / "01_training_loss_prepared.parquet"
DEFAULT_OUTPUT = PLOTS_DIR / "scgpt_like" / "01_training_loss.svg"
DEFAULT_POWER_LAW_OUTPUT = PLOTS_DIR / "scgpt_like" / "02_power_law.svg"
DEFAULT_RATIO_OUTPUT = PLOTS_DIR / "scgpt_like" / "03_depth_width_ratio.svg"
DEFAULT_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "04_depth_width_ratio_total_parameters.svg"
DEFAULT_FLOPS_TRAINING_OUTPUT = PLOTS_DIR / "scgpt_like" / "05_training_loss_flops.svg"
DEFAULT_FLOPS_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "06_depth_width_ratio_total_flops.svg"
DEFAULT_FLOPS_CONTEXT_2048_OUTPUT = PLOTS_DIR / "scgpt_like" / "07_training_loss_flops_context_length_2048_loglog.svg"
DEFAULT_OFFSET_POWER_LAW_OUTPUT = PLOTS_DIR / "scgpt_like" / "08_offset_power_law.svg"
DEFAULT_OFFSET_RATIO_OUTPUT = PLOTS_DIR / "scgpt_like" / "09_offset_depth_width_ratio.svg"
DEFAULT_OFFSET_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "10_offset_depth_width_ratio_total_parameters.svg"
DEFAULT_OFFSET_FLOPS_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "11_offset_depth_width_ratio_total_flops.svg"
DEFAULT_DOWNSTREAM_DATA = PLOTS_DIR / "scgpt_like" / "08_downstream_metrics_prepared.parquet"
DEFAULT_DOWNSTREAM_AVGBIO_OUTPUT = PLOTS_DIR / "scgpt_like" / "08_downstream_BIOscore.svg"
DEFAULT_DOWNSTREAM_AVGBIO_FLOPS_OUTPUT = PLOTS_DIR / "scgpt_like" / "09_downstream_BIOscore_flops.svg"
DEFAULT_DOWNSTREAM_TERMINAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "10_downstream_terminal_metrics.svg"
DEFAULT_DOWNSTREAM_METRIC_PLOTS_DIR = PLOTS_DIR / "scgpt_like" / "11_downstream_metric_trajectories"
DEFAULT_BCE_CSV = PLOTS_DIR / "scgpt_like" / "bce" / "01_training_loss_metrics.csv"
DEFAULT_BCE_PREPARED_DATA = PLOTS_DIR / "scgpt_like" / "bce" / "01_training_loss_prepared.parquet"
DEFAULT_BCE_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "01_training_loss.svg"
DEFAULT_BCE_POWER_LAW_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "02_power_law.svg"
DEFAULT_BCE_RATIO_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "03_depth_width_ratio.svg"
DEFAULT_BCE_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "04_depth_width_ratio_total_parameters.svg"
DEFAULT_BCE_FLOPS_TRAINING_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "05_training_loss_flops.svg"
DEFAULT_BCE_FLOPS_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "06_depth_width_ratio_total_flops.svg"
DEFAULT_BCE_OFFSET_POWER_LAW_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "08_offset_power_law.svg"
DEFAULT_BCE_OFFSET_RATIO_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "09_offset_depth_width_ratio.svg"
DEFAULT_BCE_OFFSET_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "10_offset_depth_width_ratio_total_parameters.svg"
DEFAULT_BCE_OFFSET_FLOPS_RATIO_TOTAL_OUTPUT = PLOTS_DIR / "scgpt_like" / "bce" / "11_offset_depth_width_ratio_total_flops.svg"
MSE_Y_MAX = 454.0
RUN_QUERY = "run.config.metadata.title == 'scgpt_like'"
FLOPS_JOIN_COLUMNS = [
    "model.context_length",
    "model.d_model",
    "model.transformer.n_layers",
]
DERIVED_COLUMNS = {
    "forward_flops_per_example",
    "training_flops_per_step",
    "cumulative_training_flops",
    "samples_seen_per_step",
    "cumulative_samples_seen",
    "backbone_params",
    "non_embedding_params",
}
CONFIG_KEYS = {
    "metadata.pipeline",
    "metadata.run_name",
    "metadata.title",
    "datamodule.batch_size",
    "model.context_length",
    "model.d_model",
    "model.transformer.n_layers",
    "model.bioformer.n_layers",
    "sweep_metadata.training_flops_per_step",
    "trainer.accumulate_grad",
}
RUN_FIELDS = {
    "hparams.parameters_total",
    "hparams.parameters_embedding",
    "hparams.parameters_encoder",
    "hparams.parameters_decoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Aim metrics for scgpt_like and recreate the first CIBB-style training-loss plot.",
    )
    parser.add_argument(
        "--aim-repo",
        type=Path,
        default=DEFAULT_AIM_REPO,
        help=f"Path to the Aim repo to read. Default: {DEFAULT_AIM_REPO}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Where to write the exported metrics CSV. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the first plot. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--power-law-output",
        type=Path,
        default=DEFAULT_POWER_LAW_OUTPUT,
        help=f"Where to write the second plot. Default: {DEFAULT_POWER_LAW_OUTPUT}",
    )
    parser.add_argument(
        "--ratio-output",
        type=Path,
        default=DEFAULT_RATIO_OUTPUT,
        help=f"Where to write the third plot. Default: {DEFAULT_RATIO_OUTPUT}",
    )
    parser.add_argument(
        "--ratio-total-output",
        type=Path,
        default=DEFAULT_RATIO_TOTAL_OUTPUT,
        help=f"Where to write the fourth plot. Default: {DEFAULT_RATIO_TOTAL_OUTPUT}",
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=DEFAULT_PREPARED_DATA,
        help=f"Where to write the plot-ready intermediate dataset. Default: {DEFAULT_PREPARED_DATA}",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Path to local checkpoint configs used to estimate FLOPs. Default: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--flops-training-output",
        type=Path,
        default=DEFAULT_FLOPS_TRAINING_OUTPUT,
        help=f"Where to write the fifth plot. Default: {DEFAULT_FLOPS_TRAINING_OUTPUT}",
    )
    parser.add_argument(
        "--flops-ratio-total-output",
        type=Path,
        default=DEFAULT_FLOPS_RATIO_TOTAL_OUTPUT,
        help=f"Where to write the sixth plot. Default: {DEFAULT_FLOPS_RATIO_TOTAL_OUTPUT}",
    )
    parser.add_argument(
        "--flops-context-2048-output",
        type=Path,
        default=DEFAULT_FLOPS_CONTEXT_2048_OUTPUT,
        help=f"Where to write the seventh plot. Default: {DEFAULT_FLOPS_CONTEXT_2048_OUTPUT}",
    )
    parser.add_argument(
        "--downstream-data",
        type=Path,
        default=DEFAULT_DOWNSTREAM_DATA,
        help=f"Where to write the plot-ready downstream dataset. Default: {DEFAULT_DOWNSTREAM_DATA}",
    )
    parser.add_argument(
        "--downstream-avgBIO-output",
        type=Path,
        default=DEFAULT_DOWNSTREAM_AVGBIO_OUTPUT,
        help=f"Where to write the downstream BIOscore-vs-step plot. Default: {DEFAULT_DOWNSTREAM_AVGBIO_OUTPUT}",
    )
    parser.add_argument(
        "--downstream-avgBIO-flops-output",
        type=Path,
        default=DEFAULT_DOWNSTREAM_AVGBIO_FLOPS_OUTPUT,
        help=f"Where to write the downstream BIOscore-vs-FLOPs plot. Default: {DEFAULT_DOWNSTREAM_AVGBIO_FLOPS_OUTPUT}",
    )
    parser.add_argument(
        "--downstream-terminal-output",
        type=Path,
        default=DEFAULT_DOWNSTREAM_TERMINAL_OUTPUT,
        help=f"Where to write terminal downstream metric scaling panels. Default: {DEFAULT_DOWNSTREAM_TERMINAL_OUTPUT}",
    )
    parser.add_argument(
        "--downstream-metric-plots-dir",
        type=Path,
        default=DEFAULT_DOWNSTREAM_METRIC_PLOTS_DIR,
        help=f"Where to write per-metric downstream trajectory plots. Default: {DEFAULT_DOWNSTREAM_METRIC_PLOTS_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all intermediate files, including the expensive Aim export.",
    )
    return parser.parse_args()


def _ensure_metrics_csv(
    aim_repo: Path,
    csv_path: Path,
    *,
    force: bool,
    metrics_to_extract: set[str],
) -> tuple[Path, bool]:
    required_run_field_columns = {f"hparams.{field}" if not field.startswith("hparams.") else field for field in RUN_FIELDS}

    if csv_path.exists() and not force:
        cached_columns = set(pl.read_csv(csv_path, n_rows=0).columns)
        if required_run_field_columns.issubset(cached_columns):
            return csv_path, False

    if not aim_repo.exists():
        raise FileNotFoundError(f"Aim repo not found: {aim_repo}")

    export_metrics_csv(
        aim_repo,
        csv_path,
        metrics_to_extract=metrics_to_extract,
        run_query=RUN_QUERY,
        config_keys=CONFIG_KEYS,
        run_fields=RUN_FIELDS,
    )
    return csv_path, True


def _ensure_prepared_data(
    prepared_data_path: Path,
    metrics_csv: Path,
    checkpoint_dir: Path,
    config: TrainingLossPlotConfig,
    *,
    force: bool,
) -> tuple[pl.DataFrame, bool]:
    if prepared_data_path.exists() and not force:
        prepared_df = pl.read_parquet(prepared_data_path)
        needs_backbone_source = "backbone_params" not in prepared_df.columns
        has_backbone_source = (
            "hparams.parameters_encoder" in prepared_df.columns
            or {
                "hparams.parameters_total",
                "hparams.parameters_embedding",
                "hparams.parameters_decoder",
            }.issubset(set(prepared_df.columns))
        )

        if needs_backbone_source and not has_backbone_source:
            prepared_df = load_training_loss_data(metrics_csv, config)

        # Refresh derived plotting columns on cached prepared data so corrected
        # backbone-based parameter counts propagate without requiring --force.
        prepared_df = _with_cost_columns(prepared_df, checkpoint_dir)
        if DERIVED_COLUMNS.issubset(prepared_df.columns):
            ensure_parent_dir(prepared_data_path)
            prepared_df.write_parquet(prepared_data_path)
            return prepared_df, True

        ensure_parent_dir(prepared_data_path)
        prepared_df.write_parquet(prepared_data_path)
        return prepared_df, True

    prepared_df = load_training_loss_data(metrics_csv, config)
    prepared_df = _with_cost_columns(prepared_df, checkpoint_dir)
    ensure_parent_dir(prepared_data_path)
    prepared_df.write_parquet(prepared_data_path)
    return prepared_df, True


def _collect_forward_flops_lookup(checkpoint_dir: Path) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    if not checkpoint_dir.exists():
        return pl.DataFrame(rows)

    for config_path in sorted(checkpoint_dir.glob("**/config.yml")):
        config = ScalingConfig(str(config_path))
        if getattr(config.metadata, "title", None) != "scgpt_like":
            continue

        rows.append(
            {
                "model.context_length": config.model.context_length,
                "model.d_model": config.model.d_model,
                "model.transformer.n_layers": config.model.transformer.n_layers,
                "forward_flops_per_example": float(estimate_forward_flops(config)),
            }
        )

    if not rows:
        raise ValueError(f"No usable config.yml files found under {checkpoint_dir}")

    return pl.DataFrame(rows).unique(subset=FLOPS_JOIN_COLUMNS)


def _scalarize_master_config(value):
    if isinstance(value, list):
        return _scalarize_master_config(value[0])
    if isinstance(value, dict):
        return {key: _scalarize_master_config(child) for key, child in value.items()}
    return value


def _collect_forward_flops_lookup_from_master(df: pl.DataFrame) -> pl.DataFrame:
    if not DEFAULT_MASTER_CONFIG.exists():
        return pl.DataFrame([])

    base_config = load_yaml(str(DEFAULT_MASTER_CONFIG))["defaults"]
    rows: list[dict[str, float | int]] = []
    unique_configs = df.select(FLOPS_JOIN_COLUMNS).unique().sort(FLOPS_JOIN_COLUMNS)
    for row in unique_configs.iter_rows(named=True):
        config_dict = dict(base_config)
        config_dict["metadata"]["title"] = "scgpt_like"
        config_dict["metadata"]["run_name"] = "flops_estimate"
        config_dict["model"]["context_length"] = int(row["model.context_length"])
        config_dict["model"]["d_model"] = int(row["model.d_model"])
        config_dict["model"]["transformer"]["n_layers"] = int(row["model.transformer.n_layers"])
        config = ScalingConfig(config_dict)
        rows.append(
            {
                **row,
                "forward_flops_per_example": float(estimate_forward_flops(config)),
            }
        )

    return pl.DataFrame(rows).unique(subset=FLOPS_JOIN_COLUMNS)


def _with_cost_columns(df: pl.DataFrame, checkpoint_dir: Path) -> pl.DataFrame:
    if "forward_flops_per_example" in df.columns:
        enriched_df = df
    elif "sweep_metadata.training_flops_per_step" in df.columns:
        enriched_df = df.with_columns(
            (
                pl.col("sweep_metadata.training_flops_per_step")
                / pl.col("datamodule.batch_size")
                / pl.col("trainer.accumulate_grad")
                / TRAINING_FLOPS_MULTIPLIER
            ).alias("forward_flops_per_example")
        )
    else:
        flops_lookup = _collect_forward_flops_lookup(checkpoint_dir)
        if flops_lookup.is_empty():
            flops_lookup = _collect_forward_flops_lookup_from_master(df)
        enriched_df = df.join(flops_lookup, on=FLOPS_JOIN_COLUMNS, how="left")

        if enriched_df["forward_flops_per_example"].null_count() > 0:
            missing = (
                enriched_df.filter(pl.col("forward_flops_per_example").is_null())
                .select(FLOPS_JOIN_COLUMNS)
                .unique()
            )
            raise ValueError(f"Missing FLOPs lookup rows for:\n{missing}")

    if "hparams.parameters_encoder" in enriched_df.columns:
        backbone_params = pl.col("hparams.parameters_encoder")
    elif {
        "hparams.parameters_total",
        "hparams.parameters_embedding",
        "hparams.parameters_decoder",
    }.issubset(set(enriched_df.columns)):
        backbone_params = (
            pl.col("hparams.parameters_total")
            - pl.col("hparams.parameters_embedding")
            - pl.col("hparams.parameters_decoder")
        )
    else:
        raise ValueError(
            "Cannot derive backbone_params: expected hparams.parameters_encoder or "
            "the combination of total, embedding, and decoder parameter counts."
        )

    return enriched_df.with_columns(
        [
            backbone_params.alias("backbone_params"),
            backbone_params.alias("non_embedding_params"),
            (
                pl.col("datamodule.batch_size")
                * pl.col("trainer.accumulate_grad")
            ).alias("samples_seen_per_step"),
            (
                pl.col("step")
                * pl.col("datamodule.batch_size")
                * pl.col("trainer.accumulate_grad")
            ).alias("cumulative_samples_seen"),
            (
                pl.col("forward_flops_per_example")
                * pl.col("datamodule.batch_size")
                * pl.col("trainer.accumulate_grad")
                * TRAINING_FLOPS_MULTIPLIER
            ).alias("training_flops_per_step"),
            (
                pl.col("step")
                * pl.col("forward_flops_per_example")
                * pl.col("datamodule.batch_size")
                * pl.col("trainer.accumulate_grad")
                * TRAINING_FLOPS_MULTIPLIER
            ).alias("cumulative_training_flops"),
        ]
    )


def _run_metadata_for_downstream(prepared_df: pl.DataFrame) -> pl.DataFrame:
    metadata_columns = [
        "model.context_length",
        "model.d_model",
        "model.transformer.n_layers",
        "datamodule.batch_size",
        "trainer.accumulate_grad",
        "hparams.parameters_total",
        "hparams.parameters_embedding",
        "hparams.parameters_encoder",
        "hparams.parameters_decoder",
        "forward_flops_per_example",
    ]
    available_columns = [column for column in metadata_columns if column in prepared_df.columns]
    return prepared_df.select(available_columns).unique(subset=FLOPS_JOIN_COLUMNS)


def _ensure_downstream_data(
    args: argparse.Namespace,
    prepared_df: pl.DataFrame,
) -> tuple[pl.DataFrame, bool]:
    if args.downstream_data.exists() and not args.force:
        downstream_df = pl.read_parquet(args.downstream_data)
        required = {
            "BIOscore",
            "avgBIO",
            "avgBIO_fine",
            "avgBIO_coarse",
            "backbone_params",
            "non_embedding_params",
            "cumulative_training_flops",
            "model.context_length",
            "model.d_model",
            "model.transformer.n_layers",
        }
        if required.issubset(set(downstream_df.columns)):
            return downstream_df, False

    downstream_df = downstream_plots._load_downstream_metrics(args.checkpoint_dir)
    if downstream_df.is_empty():
        return downstream_df, False

    downstream_df = downstream_plots._parse_geneformer_like_architecture(downstream_df)
    downstream_df = downstream_df.filter(
        pl.all_horizontal([pl.col(column).is_not_null() for column in FLOPS_JOIN_COLUMNS])
    )
    if downstream_df.is_empty():
        return downstream_df, False

    metadata_df = _run_metadata_for_downstream(prepared_df)
    downstream_df = downstream_df.join(metadata_df, on=FLOPS_JOIN_COLUMNS, how="left")
    missing_metadata = downstream_df.filter(pl.col("hparams.parameters_total").is_null()).select(FLOPS_JOIN_COLUMNS).unique()
    if not missing_metadata.is_empty():
        raise ValueError(f"Missing training metadata for downstream rows:\n{missing_metadata}")

    downstream_df = downstream_plots._with_avg_bio(downstream_df)
    downstream_df = _with_cost_columns(downstream_df, args.checkpoint_dir)
    downstream_df = downstream_df.with_columns(pl.col("model_folder").alias("run_hash"))
    ensure_parent_dir(args.downstream_data)
    downstream_df.write_parquet(args.downstream_data)
    return downstream_df, True


def _prepare_power_law_data(df: pl.DataFrame, x_param_col: str) -> tuple[pl.DataFrame, int]:
    run_max_steps = df.group_by("run_hash").agg(pl.col("step").max().alias("max_step"))
    common_step = int(run_max_steps["max_step"].min())

    power_law_df = (
        df.filter(pl.col("step") == common_step)
        .group_by(
            [
                "model.context_length",
                "model.d_model",
                "model.transformer.n_layers",
                x_param_col,
            ]
        )
        .agg(
            [
                pl.col("value").mean().alias("value"),
                pl.col("run_hash").n_unique().alias("n_runs"),
            ]
        )
        .with_columns(
            (pl.col("model.transformer.n_layers") / pl.col("model.d_model")).alias("depth_width_ratio")
        )
        .sort(["model.context_length", x_param_col])
    )

    if power_law_df.is_empty():
        raise ValueError("No power-law points available at the common training step")

    return power_law_df, common_step


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        raise ValueError("Need at least two points to fit the power law")
    if np.any(y <= 0):
        raise ValueError("Power-law fitting requires strictly positive y values")
    slope, intercept = np.polyfit(np.log10(x.astype(float)), np.log10(y.astype(float)), deg=1)
    return float(slope), float(intercept)


def _fit_offset_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 3:
        raise ValueError("Need at least three points to fit the offset power law")
    if np.any(y <= 0):
        raise ValueError("Offset power-law fitting requires strictly positive y values")

    min_y = float(np.min(y))
    y_range = max(float(np.max(y) - min_y), 1e-6)
    lower_floor = max(0.0, min_y - 3.0 * y_range)
    upper_floor = min_y - max(1e-6, 1e-4 * min_y)
    if upper_floor <= lower_floor:
        lower_floor = max(0.0, min_y * 0.5)

    best: tuple[float, float, float, float] | None = None
    log_x = np.log10(x.astype(float))
    for floor in np.linspace(lower_floor, upper_floor, 400):
        excess = y.astype(float) - floor
        if np.any(excess <= 0):
            continue
        slope, intercept = np.polyfit(log_x, np.log10(excess), deg=1)
        predicted = 10 ** (slope * log_x + intercept)
        sse = float(np.mean((np.log10(excess) - np.log10(predicted)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(slope), float(intercept), float(floor))

    if best is None:
        raise ValueError("Could not fit offset power law")
    _, slope, intercept, floor = best
    return slope, intercept, floor


def _format_param_tick(value: float) -> str:
    if value >= 1_000_000:
        return f"{int(round(value / 1_000_000.0))}M"
    if value >= 1_000:
        return f"{int(round(value / 1_000.0))}K"
    return str(int(round(value)))


def _format_flops_tick(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2g} × 10{_to_superscript(exponent)}"


def _to_superscript(value: int) -> str:
    superscript_map = str.maketrans(
        {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "-": "⁻",
        }
    )
    return str(value).translate(superscript_map)


def _create_flops_context_2048_loglog_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    context_length: int = 2048,
) -> None:
    panel_df = df.filter(pl.col("model.context_length") == context_length)
    if panel_df.is_empty():
        raise ValueError(f"No rows found for context length {context_length}")

    shade_values = panel_df["non_embedding_params"].to_numpy().astype(float)
    positive_shade_values = shade_values[shade_values > 0]
    if len(positive_shade_values) == 0:
        raise ValueError("non_embedding_params must be positive for logarithmic shading")

    x_values = panel_df["cumulative_training_flops"].to_numpy().astype(float)
    y_values = panel_df["value"].to_numpy().astype(float)
    positive_x_values = x_values[x_values > 0]
    if len(positive_x_values) == 0:
        raise ValueError("Cumulative training FLOPs must be positive for a log-log plot")

    shade_norm = mpl.colors.LogNorm(
        vmin=float(np.min(positive_shade_values)),
        vmax=float(np.max(positive_shade_values)),
    )
    x_ticks = np.geomspace(3.5e15, 1.8e18, 5)
    x_min, x_max = 3.5e15, 1.8e18
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_range = max(y_max - y_min, 1e-6)
    y_min -= 0.05 * y_range
    y_max = min(y_max + 0.05 * y_range, MSE_Y_MAX)
    y_ticks = np.linspace(y_min, y_max, 6)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.3))

    cmap_name = "Purples"
    cmap = plt.colormaps[cmap_name]
    tinted_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_single_panel",
        cmap(np.linspace(0.32, 1.0, 256)),
    )

    for run_hash in panel_df["run_hash"].unique().sort().to_list():
        run_df = (
            panel_df.filter(pl.col("run_hash") == run_hash)
            .sort("cumulative_training_flops")
            .group_by("step")
            .agg(
                [
                    pl.col("cumulative_training_flops").first().alias("cumulative_training_flops"),
                    pl.col("value").mean().alias("value"),
                    pl.col("non_embedding_params").first().alias("non_embedding_params"),
                ]
            )
            .sort("cumulative_training_flops")
        )
        run_x = run_df["cumulative_training_flops"].to_numpy().astype(float)
        run_y = run_df["value"].to_numpy().astype(float)
        shade_value = float(run_df["non_embedding_params"][0])
        normalized = float(shade_norm(shade_value))
        color = cmap(0.32 + (1.0 - 0.32) * normalized)

        positive_mask = run_x > 0
        if np.count_nonzero(positive_mask) == 0:
            continue

        ax.plot(
            run_x[positive_mask],
            exponential_moving_average(run_y[positive_mask], alpha=SCGPT_MSE_CONFIG.ema_alpha),
            color=color,
            linewidth=0.6,
            zorder=3,
        )

    ax.set_title(_make_subplot_title("model.context_length", context_length), fontsize=FONT_SIZE_LABELS, y=1.02)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([_format_flops_tick(value) for value in x_ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"{value:.3g}" for value in y_ticks]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_xlabel("Cumulative Training FLOPs", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Mean Squared Error", fontsize=FONT_SIZE_LABELS)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#b3b3b3", linestyle="-", linewidth=0.4, alpha=0.8, zorder=1)
    remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    color_box = ax.inset_axes([0.78, 0.53, 0.18, 0.34])
    color_box.set_xticks([])
    color_box.set_yticks([])
    _style_panel_box(color_box, rounded=True)
    color_box.text(
        0.5,
        0.84,
        "Non-embedding\nParameters",
        transform=color_box.transAxes,
        fontsize=FONT_SIZE_LEGEND - 3,
        va="center",
        ha="center",
    )
    cax = color_box.inset_axes([0.16, 0.16, 0.20, 0.50])
    sm = mpl.cm.ScalarMappable(norm=shade_norm, cmap=tinted_cmap)
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar_ticks = np.geomspace(shade_norm.vmin, shade_norm.vmax, 4)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([_format_param_count(value) for value in colorbar_ticks])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
    colorbar.outline.set_edgecolor("#000000")
    colorbar.outline.set_linewidth(0.9)

    fig.suptitle("Training Performance on Binned Gene Expression", fontsize=FONT_SIZE_TITLE, x=0.53, y=0.96, ha="center")
    fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.16)
    save_figure(fig, output_path)


def _select_parameter_ticks(x_values: np.ndarray, max_ticks: int = 4) -> tuple[np.ndarray, list[str]]:
    if len(x_values) <= max_ticks:
        tick_values = x_values.astype(float)
    else:
        targets = np.geomspace(float(x_values[0]), float(x_values[-1]), num=max_ticks)
        chosen_indices: list[int] = []
        for target in targets:
            nearest_index = int(np.argmin(np.abs(np.log(x_values.astype(float)) - np.log(target))))
            if nearest_index not in chosen_indices:
                chosen_indices.append(nearest_index)

        if 0 not in chosen_indices:
            chosen_indices.insert(0, 0)
        if len(x_values) - 1 not in chosen_indices:
            chosen_indices.append(len(x_values) - 1)

        chosen_indices = sorted(set(chosen_indices))
        tick_values = x_values[chosen_indices].astype(float)

        while len(tick_values) > max_ticks:
            tick_values = tick_values[:-1]
            tick_values[-1] = float(x_values[-1])

    tick_labels = [_format_param_tick(value) for value in tick_values]
    return tick_values, tick_labels


def _log_limits_from_ticks(ticks: np.ndarray, edge_fraction: float = 0.1) -> tuple[float, float]:
    if len(ticks) < 2:
        tick = float(ticks[0]) if len(ticks) == 1 else 1.0
        return tick / 1.1, tick * 1.1

    log_ticks = np.log10(ticks.astype(float))
    lower_log_step = float(log_ticks[1] - log_ticks[0])
    upper_log_step = float(log_ticks[-1] - log_ticks[-2])
    return (
        float(10 ** (log_ticks[0] - edge_fraction * lower_log_step)),
        float(10 ** (log_ticks[-1] + edge_fraction * upper_log_step)),
    )


def _format_ratio_tick(value: float) -> str:
    return f"{value:.3f}"


def _select_log_loss_ticks(y_limits: tuple[float, float], max_ticks: int = 5) -> tuple[np.ndarray, list[str]]:
    y_min, y_max = y_limits
    if y_min <= 0:
        y_min = max(y_max / 100.0, 1e-6)
    tick_values = np.geomspace(y_min, y_max, max_ticks)
    return tick_values, [_format_loss_tick(value) for value in tick_values]


def create_power_law_plot_from_df(
    df: pl.DataFrame,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    color_by_depth_width_ratio: bool = False,
    offset_power_law: bool = False,
    x_param_col: str = "non_embedding_params",
    x_label: str = "Non-embedding Parameters",
    y_limit_max: float | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    show_title: bool = True,
    hspace: float = 0.52,
    compact_legend: bool = False,
    marker_area_scale: float = 1.0,
    subplot_left: float = 0.08,
    subplot_right: float = 0.98,
    wspace: float = 0.18,
    panel_box_aspect: float = 0.78,
    tick_edge_padding_fraction: float | None = None,
    y_tick_bounds: tuple[float, float] | None = None,
    x_tick_rotation: float = 25.0,
    enforce_y_axis_units: bool = True,
) -> None:
    power_law_df, common_step = _prepare_power_law_data(df, x_param_col)
    context_lengths = power_law_df["model.context_length"].unique().sort().to_list()
    n_panels = len(context_lengths)

    apply_plot_style()

    nrows = int(np.ceil(n_panels / ncols))
    resolved_figsize = figsize or (6.2 * ncols, 6.0 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=resolved_figsize,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    x_values_all = power_law_df[x_param_col].unique().sort().to_numpy()
    tick_values, tick_labels = _select_parameter_ticks(x_values_all)
    x_axis_min, x_axis_max = _log_limits_from_ticks(
        tick_values,
        edge_fraction=0.1 if tick_edge_padding_fraction is None else tick_edge_padding_fraction,
    )
    if "flops" in x_param_col:
        tick_labels = [_format_flops_tick(value) for value in tick_values]
    if "flops" in x_param_col:
        scaling_symbol = "C"
    elif "samples" in x_param_col:
        scaling_symbol = "S"
        tick_labels = [_format_flops_tick(value) for value in tick_values]
    else:
        scaling_symbol = "N"
    ratio_norm = None
    ratio_cmap = None
    if color_by_depth_width_ratio:
        ratio_values = power_law_df["depth_width_ratio"].to_numpy().astype(float)
        ratio_norm = mpl.colors.Normalize(
            vmin=0.0 if compact_legend else float(np.min(ratio_values)),
            vmax=float(np.max(ratio_values)),
        )
        ratio_cmap = plt.colormaps["Greys"]
    fit_predictions: list[float] = []
    actual_values: list[float] = []

    for context_length in context_lengths:
        panel_df = power_law_df.filter(pl.col("model.context_length") == context_length).sort(x_param_col)
        x = panel_df[x_param_col].to_numpy().astype(float)
        y = panel_df["value"].to_numpy().astype(float)

        if len(x) < 3:
            raise ValueError(
                f"Need at least three model sizes for context length {context_length}, found {len(x)}"
            )

        if offset_power_law:
            slope, intercept, floor = _fit_offset_power_law(x[:-1], y[:-1])
            predicted_last = floor + 10 ** (slope * np.log10(x[-1]) + intercept)
            # A zero fitted floor is valid, but it cannot define the lower
            # bound of the logarithmic plot axis.
            if floor > 0:
                fit_predictions.append(float(floor))
        else:
            slope, intercept = _fit_power_law(x[:-1], y[:-1])
            predicted_last = 10 ** (slope * np.log10(x[-1]) + intercept)
        fit_predictions.append(float(predicted_last))
        actual_values.extend([float(np.min(y)), float(np.max(y))])

    y_min = min(actual_values + fit_predictions)
    y_max = max(actual_values + fit_predictions)
    y_axis_max = y_max * 1.10 if y_limit_max is None else min(y_max * 1.10, y_limit_max)
    y_limits = (y_min / 1.10, y_axis_max)
    y_tick_values, y_tick_labels = _select_log_loss_ticks(y_tick_bounds or y_limits)
    y_axis_limits = (
        _log_limits_from_ticks(y_tick_values, edge_fraction=tick_edge_padding_fraction)
        if tick_edge_padding_fraction is not None
        else y_limits
    )

    for panel_index, context_length in enumerate(context_lengths):
        ax = axes[panel_index]
        panel_df = power_law_df.filter(pl.col("model.context_length") == context_length).sort(x_param_col)
        x = panel_df[x_param_col].to_numpy().astype(float)
        y = panel_df["value"].to_numpy().astype(float)

        floor = 0.0
        if offset_power_law:
            slope, intercept, floor = _fit_offset_power_law(x[:-1], y[:-1])
        else:
            slope, intercept = _fit_power_law(x[:-1], y[:-1])
        x_fit = np.geomspace(x[0], x[-1], 200)
        y_fit = floor + 10 ** (slope * np.log10(x_fit) + intercept)
        predicted_last = floor + 10 ** (slope * np.log10(x[-1]) + intercept)
        panel_color = plt.colormaps[COLOR_SHADES[panel_index % len(COLOR_SHADES)]](0.78)
        point_colors = None
        if color_by_depth_width_ratio and ratio_norm is not None and ratio_cmap is not None:
            ratios = panel_df["depth_width_ratio"].to_numpy().astype(float)
            normalized = ratio_norm(ratios)
            point_colors = [ratio_cmap(0.30 + 0.62 * float(value)) for value in normalized]

        ax.plot(x_fit, y_fit, color=panel_color, linewidth=2.0, zorder=2)
        if point_colors is None:
            ax.scatter(x[:-1], y[:-1], color="black", s=42 * marker_area_scale, zorder=4)
        else:
            ax.scatter(
                x[:-1],
                y[:-1],
                color=point_colors[:-1],
                edgecolors="black",
                linewidths=0.6,
                s=44 * marker_area_scale,
                zorder=4,
            )
        ax.scatter(
            x[-1],
            predicted_last,
            color="black",
            marker="x",
            s=72 * marker_area_scale,
            linewidths=2.0,
            zorder=5,
        )
        observed_color = panel_color if point_colors is None else point_colors[-1]
        ax.scatter(
            x[-1],
            y[-1],
            color=observed_color,
            edgecolors="black",
            linewidths=0.9,
            s=82 * marker_area_scale,
            marker="D",
            zorder=6,
        )
        ax.vlines(
            x[-1],
            ymin=min(predicted_last, y[-1]),
            ymax=max(predicted_last, y[-1]),
            colors="black",
            linestyles=":",
            linewidth=1.4,
            zorder=3,
        )

        ax.set_title(_make_subplot_title("model.context_length", context_length), fontsize=FONT_SIZE_LABELS, y=1.02)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(*y_axis_limits)
        ax.xaxis.set_major_locator(mticker.FixedLocator(tick_values))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_tick_values))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(y_tick_labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", rotation=x_tick_rotation)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right" if x_tick_rotation else "center")
        ax.margins(x=0.02, y=0.03)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="#b3b3b3", linestyle="-", linewidth=0.4, alpha=0.8, zorder=1)
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)
        coefficient_box = mpatches.FancyBboxPatch(
            (0.66 if compact_legend else 0.62, 0.80),
            0.22 if compact_legend else 0.30,
            0.12,
            boxstyle="round,pad=0.02",
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#000000",
            linewidth=0.9,
            alpha=0.92,
            zorder=7,
        )
        ax.add_patch(coefficient_box)
        ax.text(
            0.77,
            0.86,
            f"k = {-slope:.3f}" if not offset_power_law else f"k = {-slope:.3f}\n$L_∞$ = {_format_loss_tick(floor)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONT_SIZE_TICKS if offset_power_law else FONT_SIZE_TICKS + 1,
            color="#2f2a24",
            zorder=8,
        )

    for ax in axes[n_panels:]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(*y_axis_limits)
        ax.xaxis.set_major_locator(mticker.FixedLocator(tick_values))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_tick_values))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(y_tick_labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(True)
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    for index, ax in enumerate(axes[:n_panels]):
        col = index % ncols
        ax.set_box_aspect(panel_box_aspect)
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABELS)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=(col == 0))
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS)

    if len(axes) > n_panels:
        legend_ax = axes[n_panels]
        legend_ax.set_box_aspect(panel_box_aspect)
        legend_ax.set_axis_off()
        legend_ax.set_xlabel("")
        legend_ax.set_ylabel("")
        legend_ax.grid(False)

        left_x = 0.00 if compact_legend else 0.05
        bottom_y = -0.15
        total_h = 1.25 if compact_legend else 1.00
        left_w = 0.54 if compact_legend else 0.47
        gap = 0.06 if compact_legend else 0.1
        right_x = left_x + left_w + gap
        right_w = 0.40 if compact_legend else 0.30

        legend_bounds = (left_x, bottom_y, left_w, total_h)
        legend_box = legend_ax.inset_axes(legend_bounds)
        legend_box.set_xticks([])
        legend_box.set_yticks([])
        _style_panel_box(
            legend_box, rounded=True, draw_box=not compact_legend
        )
        if compact_legend:
            _add_isotropic_rounded_frame(legend_ax, legend_bounds)
        observed_handle_color = "#7f7f7f" if color_by_depth_width_ratio else plt.colormaps[COLOR_SHADES[0]](0.78)
        legend_handles = [
            mlines.Line2D([], [], color=observed_handle_color, marker="o", linestyle="None", markersize=5.5 if compact_legend else 6.5, markeredgecolor="black", label="Observed models"),
            mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=6.5 if compact_legend else 8, markeredgewidth=1.6 if compact_legend else 1.8, label="Predicted n-th"),
            mlines.Line2D([], [], color=observed_handle_color, marker="D", linestyle="None", markersize=6.0 if compact_legend else 7.5, markeredgecolor="black", label="Observed n-th"),
            mlines.Line2D([], [], color=plt.colormaps[COLOR_SHADES[0]](0.78), linewidth=2.0, label="Offset fit" if offset_power_law else "Log-log fit"),
            mlines.Line2D([], [], color="black", linestyle=":", linewidth=1.4, label="Vertical gap"),
        ]
        legend = legend_box.legend(
            handles=legend_handles,
            loc="center",
            bbox_to_anchor=(0.54, 0.43) if compact_legend else None,
            title=None if compact_legend else "Scaling Plot",
            fontsize=FONT_SIZE_LEGEND - 3,
            title_fontsize=FONT_SIZE_LEGEND - (2 if compact_legend else 0),
            handlelength=1.2 if compact_legend else 2.1,
            borderaxespad=0.0,
            frameon=False,
            labelspacing=0.55 if compact_legend else 1.0,
            handletextpad=0.45 if compact_legend else 0.7,
        )
        if compact_legend:
            legend_box.text(
                0.56 if compact_legend else 0.5,
                0.88,
                "Scaling Plot",
                transform=legend_box.transAxes,
                fontsize=FONT_SIZE_LEGEND - 2,
                va="center",
                ha="center",
            )
        else:
            legend.get_title().set_y(30)

        if color_by_depth_width_ratio and compact_legend:
            formula_box = None
            vertical_gap = 0.0
            right_top_h = total_h
            right_bottom_h = 0.0
        elif color_by_depth_width_ratio:
            vertical_gap = 0.06
            right_top_h = 0.7
            right_bottom_h = total_h - right_top_h - vertical_gap
            formula_box = legend_ax.inset_axes(
                [right_x, bottom_y - 0.013, right_w, right_bottom_h]
            )
        else:
            formula_box = legend_ax.inset_axes([0.66, 0.26, 0.28, 0.48])
        if formula_box is not None:
            formula_box.set_xticks([])
            formula_box.set_yticks([])
            _style_panel_box(formula_box, rounded=True)
            formula_box.text(
                0.5,
                0.69 if color_by_depth_width_ratio else 0.64,
                "Scaling Law",
                transform=formula_box.transAxes,
                fontsize=FONT_SIZE_LEGEND - (2 if compact_legend else 1),
                va="center",
                ha="center",
            )
            formula_box.text(
                0.5,
                0.29 if color_by_depth_width_ratio else 0.38,
                (
                    rf"$L = L_∞ + A{scaling_symbol}^{{-k}}$"
                    if offset_power_law
                    else rf"$L \propto {scaling_symbol}^{{-k}}$"
                ),
                transform=formula_box.transAxes,
                fontsize=FONT_SIZE_LABELS - (3 if compact_legend else 1),
                va="center",
                ha="center",
            )

        if color_by_depth_width_ratio and ratio_norm is not None and ratio_cmap is not None:
            ratio_bounds = (
                [right_x, bottom_y, right_w, total_h]
                if compact_legend
                else [
                    right_x,
                    bottom_y + right_bottom_h + vertical_gap + 0.013,
                    right_w,
                    right_top_h,
                ]
            )
            ratio_box = legend_ax.inset_axes(ratio_bounds)
            ratio_box.set_xticks([])
            ratio_box.set_yticks([])
            _style_panel_box(
                ratio_box, rounded=True, draw_box=not compact_legend
            )
            if compact_legend:
                _add_isotropic_rounded_frame(legend_ax, tuple(ratio_bounds))
            ratio_box.text(
                0.56 if compact_legend else 0.5,
                0.83,
                "Depth / Width\nRatio",
                transform=ratio_box.transAxes,
                fontsize=FONT_SIZE_LEGEND - (2 if compact_legend else 3),
                va="center",
                ha="center",
                multialignment="center",
            )
            cax_bounds = (
                [0.27, 0.12, 0.18, 0.50]
                if compact_legend
                else [0.20, 0.1, 0.16, 0.55]
            )
            cax = ratio_box.inset_axes(cax_bounds)
            sm = mpl.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)
            colorbar = fig.colorbar(sm, cax=cax, orientation="vertical")
            if compact_legend and colorbar.solids is not None:
                colorbar.solids.set_edgecolor("face")
                colorbar.solids.set_linewidth(0)
            ratio_ticks = np.linspace(float(ratio_norm.vmin), float(ratio_norm.vmax), 3 if compact_legend else 4)
            colorbar.set_ticks(ratio_ticks)
            colorbar.set_ticklabels(
                [f"{value:.2f}" if compact_legend else _format_ratio_tick(value) for value in ratio_ticks]
            )
            colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 1, length=0)
            colorbar.outline.set_edgecolor("#000000")
            colorbar.outline.set_linewidth(0.9)

    if show_title:
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE, x=0.53, y=0.955, ha="center")
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=0.93 if show_title else 0.98,
        bottom=0.08,
        wspace=wspace,
        hspace=hspace,
    )
    save_figure(
        fig,
        output_path,
        enforce_y_axis_units=enforce_y_axis_units,
    )


def main() -> None:
    args = parse_args()
    metrics_csv, exported_from_aim = _ensure_metrics_csv(
        args.aim_repo,
        args.csv,
        force=args.force,
        metrics_to_extract={SCGPT_MSE_CONFIG.metric_name},
    )
    prepared_df, wrote_prepared_data = _ensure_prepared_data(
        args.prepared_data,
        metrics_csv,
        args.checkpoint_dir,
        SCGPT_MSE_CONFIG,
        force=args.force,
    )
    downstream_df, wrote_downstream_data = _ensure_downstream_data(args, prepared_df)

    create_training_loss_plot_from_df(prepared_df, args.output, config=SCGPT_MSE_CONFIG)
    create_training_loss_plot_from_df(
        prepared_df,
        args.flops_training_output,
        config=SCGPT_MSE_CONFIG,
        x_col="cumulative_training_flops",
        x_label="Cumulative Training FLOPs",
        x_tick_formatter=_format_flops_tick,
        x_tick_rotation=25.0,
    )
    _create_flops_context_2048_loglog_plot(
        prepared_df,
        args.flops_context_2048_output,
    )
    create_power_law_plot_from_df(
        prepared_df,
        args.power_law_output,
        title="Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        args.ratio_output,
        title="Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        args.ratio_total_output,
        title="Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        x_param_col="hparams.parameters_total",
        x_label="Total Parameters",
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        args.flops_ratio_total_output,
        title="Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        x_param_col="cumulative_training_flops",
        x_label="Cumulative Training FLOPs",
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        DEFAULT_OFFSET_POWER_LAW_OUTPUT,
        title="Offset Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        offset_power_law=True,
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        DEFAULT_OFFSET_RATIO_OUTPUT,
        title="Offset Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        offset_power_law=True,
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        DEFAULT_OFFSET_RATIO_TOTAL_OUTPUT,
        title="Offset Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        offset_power_law=True,
        x_param_col="hparams.parameters_total",
        x_label="Total Parameters",
        y_limit_max=MSE_Y_MAX,
    )
    create_power_law_plot_from_df(
        prepared_df,
        DEFAULT_OFFSET_FLOPS_RATIO_TOTAL_OUTPUT,
        title="Offset Scaling Trends with Binned Gene Expression",
        ylabel=SCGPT_MSE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        offset_power_law=True,
        x_param_col="cumulative_training_flops",
        x_label="Cumulative Training FLOPs",
        y_limit_max=MSE_Y_MAX,
    )
    if not downstream_df.is_empty():
        bio_score_df = downstream_plots._create_downstream_metric_frame(downstream_df, "BIOscore")
        create_training_loss_plot_from_df(
            bio_score_df,
            args.downstream_avgBIO_output,
            config=downstream_plots.AVGBIO_CONFIG,
            line_width=0.85,
        )
        create_training_loss_plot_from_df(
            bio_score_df,
            args.downstream_avgBIO_flops_output,
            config=downstream_plots.AVGBIO_CONFIG,
            x_col="cumulative_training_flops",
            x_label="Cumulative Training FLOPs",
            x_tick_formatter=_format_flops_tick,
            x_tick_rotation=25.0,
            line_width=0.85,
        )
        downstream_plots.create_downstream_terminal_metrics_plot(
            downstream_df,
            args.downstream_terminal_output,
        )
        downstream_metric_plot_paths = downstream_plots.create_downstream_metric_trajectory_plots(
            downstream_df,
            args.downstream_metric_plots_dir,
        )
    else:
        downstream_metric_plot_paths = []

    bce_metrics_csv, bce_exported_from_aim = _ensure_metrics_csv(
        args.aim_repo,
        DEFAULT_BCE_CSV,
        force=args.force,
        metrics_to_extract={SCGPT_BCE_CONFIG.metric_name},
    )
    bce_prepared_df, bce_wrote_prepared_data = _ensure_prepared_data(
        DEFAULT_BCE_PREPARED_DATA,
        bce_metrics_csv,
        args.checkpoint_dir,
        SCGPT_BCE_CONFIG,
        force=args.force,
    )
    create_training_loss_plot_from_df(bce_prepared_df, DEFAULT_BCE_OUTPUT, config=SCGPT_BCE_CONFIG)
    create_training_loss_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_FLOPS_TRAINING_OUTPUT,
        config=SCGPT_BCE_CONFIG,
        x_col="cumulative_training_flops",
        x_label="Cumulative Training FLOPs",
        x_tick_formatter=_format_flops_tick,
        x_tick_rotation=25.0,
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_POWER_LAW_OUTPUT,
        title="Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_RATIO_OUTPUT,
        title="Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_RATIO_TOTAL_OUTPUT,
        title="Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        x_param_col="hparams.parameters_total",
        x_label="Total Parameters",
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_FLOPS_RATIO_TOTAL_OUTPUT,
        title="Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        x_param_col="cumulative_training_flops",
        x_label="Cumulative Training FLOPs",
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_OFFSET_POWER_LAW_OUTPUT,
        title="Offset Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        offset_power_law=True,
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_OFFSET_RATIO_OUTPUT,
        title="Offset Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        offset_power_law=True,
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_OFFSET_RATIO_TOTAL_OUTPUT,
        title="Offset Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        offset_power_law=True,
        x_param_col="hparams.parameters_total",
        x_label="Total Parameters",
    )
    create_power_law_plot_from_df(
        bce_prepared_df,
        DEFAULT_BCE_OFFSET_FLOPS_RATIO_TOTAL_OUTPUT,
        title="Offset Scaling Trends with Masked Gene Activity Prediction",
        ylabel=SCGPT_BCE_CONFIG.ylabel,
        color_by_depth_width_ratio=True,
        offset_power_law=True,
        x_param_col="cumulative_training_flops",
        x_label="Cumulative Training FLOPs",
    )

    if exported_from_aim:
        print(f"Exported metrics CSV to {metrics_csv}")
    else:
        print(f"Reused cached metrics CSV at {metrics_csv}")

    if wrote_prepared_data:
        print(f"Wrote prepared plot data to {args.prepared_data}")
    else:
        print(f"Reused prepared plot data at {args.prepared_data}")
    if downstream_df.is_empty():
        print(f"No downstream metrics found under {args.checkpoint_dir}; skipped downstream plots")
    elif wrote_downstream_data:
        print(f"Wrote downstream metric data to {args.downstream_data}")
    else:
        print(f"Reused downstream metric data at {args.downstream_data}")
    if bce_exported_from_aim:
        print(f"Exported BCE metrics CSV to {bce_metrics_csv}")
    else:
        print(f"Reused cached BCE metrics CSV at {bce_metrics_csv}")
    if bce_wrote_prepared_data:
        print(f"Wrote BCE prepared plot data to {DEFAULT_BCE_PREPARED_DATA}")
    else:
        print(f"Reused BCE prepared plot data at {DEFAULT_BCE_PREPARED_DATA}")

    print(f"Saved plot to {args.output}")
    print(f"Saved plot to {args.flops_training_output}")
    print(f"Saved plot to {args.flops_context_2048_output}")
    print(f"Saved plot to {args.power_law_output}")
    print(f"Saved plot to {args.ratio_output}")
    print(f"Saved plot to {args.ratio_total_output}")
    print(f"Saved plot to {args.flops_ratio_total_output}")
    if not downstream_df.is_empty():
        print(f"Saved plot to {args.downstream_avgBIO_output}")
        print(f"Saved plot to {args.downstream_avgBIO_flops_output}")
        print(f"Saved plot to {args.downstream_terminal_output}")
        print(f"Saved {len(downstream_metric_plot_paths)} per-metric downstream trajectory plot files to {args.downstream_metric_plots_dir}")
    for path in [
        DEFAULT_OFFSET_POWER_LAW_OUTPUT,
        DEFAULT_OFFSET_RATIO_OUTPUT,
        DEFAULT_OFFSET_RATIO_TOTAL_OUTPUT,
        DEFAULT_OFFSET_FLOPS_RATIO_TOTAL_OUTPUT,
        DEFAULT_BCE_OUTPUT,
        DEFAULT_BCE_FLOPS_TRAINING_OUTPUT,
        DEFAULT_BCE_POWER_LAW_OUTPUT,
        DEFAULT_BCE_RATIO_OUTPUT,
        DEFAULT_BCE_RATIO_TOTAL_OUTPUT,
        DEFAULT_BCE_FLOPS_RATIO_TOTAL_OUTPUT,
        DEFAULT_BCE_OFFSET_POWER_LAW_OUTPUT,
        DEFAULT_BCE_OFFSET_RATIO_OUTPUT,
        DEFAULT_BCE_OFFSET_RATIO_TOTAL_OUTPUT,
        DEFAULT_BCE_OFFSET_FLOPS_RATIO_TOTAL_OUTPUT,
    ]:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
