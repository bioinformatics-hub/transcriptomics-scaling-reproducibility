"""Build supplementary-only downstream and learning-rate diagnostic figures."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

from plotting import geneformer_dw, geneformer_lr
from plotting import training_loss
from plotting.training_loss import GENEFORMER_BCE_CONFIG, SCGPT_MSE_CONFIG
from plotting.utils import (
    GRID_COLOR,
    ROOT_DIR,
    TOL_MUTED_CMAP_NAMES,
    WONG_CMAP_NAMES,
    WONG_PALETTE,
    apply_plot_style,
    remove_bounding_box,
    save_figure,
)
from tools.paper_plots.figure_04 import BASE_PANEL_TYPOGRAPHY
from tools.paper_plots.utils import temporary_plot_settings as _temporary_plot_settings
from tools.paper_plots.data import paper_data
from tools.paper_plots.style import (
    PAPER_FONT_FAMILY,
    PAPER_MAX_FONT_SIZE_PT,
    PAPER_MIN_FONT_SIZE_PT,
    PAPER_TICK_EDGE_PADDING_FRACTION,
    validate_paper_svg,
)


OUTPUT_DIR = ROOT_DIR / "manuscript" / "figures"
RANKED_DOWNSTREAM = (
    paper_data("geneformer_like/08_downstream_metrics_prepared.parquet")
)
BINNED_DOWNSTREAM = (
    paper_data("scgpt_like/08_downstream_metrics_prepared.parquet")
)
RANKED_LR = paper_data("geneformer_lr/01_lr_sampled_prepared.parquet")
BINNED_LR = paper_data("scgpt_lr_mse/01_lr_sampled_prepared.parquet")
RANKED_DW = paper_data("geneformer_dw/01_isoflops_prepared.parquet")
BINNED_DW = paper_data("scgpt_dw_rerun/01_isoflops_prepared.parquet")
RANKED_DW_TRAINING = (
    paper_data("geneformer_dw/02_training_loss_flops_prepared.parquet")
)
BINNED_DW_TRAINING = (
    paper_data("scgpt_dw_rerun/02_training_loss_flops_prepared.parquet")
)
DW_EARLY_FLOPS = (1.0e14, 1.78e14, 3.16e14, 5.62e14, 1.0e15, 1.78e15)

CONTEXT_LENGTH_CMAPS = TOL_MUTED_CMAP_NAMES[:5]
LR_SIZE_CMAPS = WONG_CMAP_NAMES[:4]
# Match the actual display widths in supplementary.tex: A4 with 18 mm margins
# gives a 174 mm text block, and paired panels occupy 0.49 of that width.
FULL_WIDTH_MM = 174.0
HALF_WIDTH_MM = 0.49 * FULL_WIDTH_MM
DW_PANEL_TYPOGRAPHY = {
    "FONT_SIZE_TICKS": 18,
    "FONT_SIZE_LABELS": 21.8,
    "FONT_SIZE_TITLE": 21.8,
    "FONT_SIZE_LEGEND": 21.8,
}
SVG_NS = "http://www.w3.org/2000/svg"

DOWNSTREAM_GROUPS = {
    "fine": (
        ("cluster_fine_nmi", "Fine-label NMI"),
        ("cluster_fine_ari", "Fine-label ARI"),
        ("cluster_fine_homogeneity", "Fine-label homogeneity"),
    ),
    "coarse": (
        ("cluster_coarse_nmi", "Coarse-label NMI"),
        ("cluster_coarse_ari", "Coarse-label ARI"),
        ("cluster_coarse_homogeneity", "Coarse-label homogeneity"),
    ),
    "representation": (
        ("batch_asw_label", "Label ASW"),
        ("batch_graph_conn", "Graph connectivity"),
        ("ridge_val_accuracy_coarse", "Coarse ridge accuracy"),
    ),
    "batch": (
        ("batch_nmi", "Resolution-optimised NMI"),
        ("batch_ari", "Resolution-optimised ARI"),
        ("batch_asw_batch", "Batch ASW"),
    ),
}


def _svg_length_points(value: str) -> float:
    if value.endswith("mm"):
        return float(value[:-2]) * 72.0 / 25.4
    if value.endswith("pt"):
        return float(value[:-2])
    if value.endswith("in"):
        return float(value[:-2]) * 72.0
    return float(value)


def _paperize_svg(source: Path, destination: Path, *, width_mm: float) -> None:
    """Set final physical size and clamp editable text to the paper font band."""
    root = ET.parse(source).getroot()
    view_box = [float(value) for value in root.attrib["viewBox"].split()]
    source_width_pt = _svg_length_points(root.attrib["width"])
    source_height_pt = _svg_length_points(root.attrib["height"])
    height_mm = width_mm * source_height_pt / source_width_pt
    if height_mm > 210.0:
        raise ValueError(
            f"Supplementary figure would be {height_mm:.1f} mm high: {source}"
        )

    root.set("width", f"{width_mm:g}mm")
    root.set("height", f"{height_mm:.8f}mm")
    physical_scale = min(
        width_mm * 72.0 / 25.4 / view_box[2],
        height_mm * 72.0 / 25.4 / view_box[3],
    )
    namespace = f"{{{SVG_NS}}}"
    for element in root.iter(f"{namespace}text"):
        raw_size = element.get("font-size")
        style = element.get("style", "")
        style_parts = [part.strip() for part in style.split(";") if part.strip()]
        style_map = {
            key.strip(): value.strip()
            for part in style_parts
            if ":" in part
            for key, value in [part.split(":", 1)]
        }
        if raw_size is None:
            raw_size = style_map.get("font-size")
        if raw_size is not None:
            authored_size = float(raw_size.removesuffix("px").removesuffix("pt"))
            physical_size = authored_size * physical_scale
            target_size = min(
                max(physical_size, PAPER_MIN_FONT_SIZE_PT),
                PAPER_MAX_FONT_SIZE_PT,
            )
            style_map["font-size"] = f"{target_size / physical_scale:.8f}px"
            element.attrib.pop("font-size", None)
        style_map["font-family"] = PAPER_FONT_FAMILY
        element.set(
            "style",
            "; ".join(f"{key}: {value}" for key, value in style_map.items()),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SVG_NS)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    validate_paper_svg(destination, required_width_mm=width_mm)


def _svg_to_pdf(source: Path, destination: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise FileNotFoundError(
            "rsvg-convert is required to create the supplementary vector PDFs"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [converter, "--format=pdf", "--output", str(destination), str(source)],
        check=True,
    )


def _render_paper_pdf(
    render,
    output_path: Path,
    *,
    width_mm: float,
) -> None:
    with tempfile.TemporaryDirectory(prefix="supplementary_plot_") as directory:
        temporary_dir = Path(directory)
        source = temporary_dir / "source.svg"
        normalized = temporary_dir / "paper.svg"
        render(source)
        _paperize_svg(source, normalized, width_mm=width_mm)
        _svg_to_pdf(normalized, output_path)


def _ema(values: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    if len(values) == 0:
        return values
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _step_formatter(value: float, _position: int | None = None) -> str:
    if value == 0:
        return "0"
    return f"{value / 1_000:g}k"


def _flops_formatter(value: float, _position: int | None = None) -> str:
    if value == 0:
        return "0"
    return f"{value:.1e}".replace("e+", "e")


def _metric_limits(df: pl.DataFrame, metric: str) -> tuple[float, float]:
    values = df[metric].drop_nulls().to_numpy().astype(float)
    values = values[np.isfinite(values)]
    low = float(np.min(values))
    high = float(np.max(values))
    pad = max(0.06 * (high - low), 0.015)
    return max(0.0, low - pad), min(1.0, high + pad)


def create_downstream_group(
    df: pl.DataFrame,
    *,
    formulation: str,
    metrics: tuple[tuple[str, str], ...],
    output_path: Path,
) -> None:
    contexts = df["model.context_length"].drop_nulls().unique().sort().to_list()
    parameter_values = df["non_embedding_params"].drop_nulls().to_numpy().astype(float)
    parameter_norm = plt.Normalize(
        vmin=float(np.log10(np.min(parameter_values))),
        vmax=float(np.log10(np.max(parameter_values))),
    )

    apply_plot_style()
    nrows = len(metrics)
    ncols = len(contexts)
    height = 2.25 if nrows == 1 else 1.82 * nrows + 0.82
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.08, height),
        squeeze=False,
        sharex=True,
    )

    for row_index, (metric, label) in enumerate(metrics):
        y_limits = _metric_limits(df, metric)
        for col_index, context in enumerate(contexts):
            ax = axes[row_index, col_index]
            panel = df.filter(pl.col("model.context_length") == context)
            cmap = plt.colormaps[
                CONTEXT_LENGTH_CMAPS[col_index % len(CONTEXT_LENGTH_CMAPS)]
            ]
            for run in panel.partition_by("model_folder", maintain_order=True):
                run = run.filter(pl.col(metric).is_not_null()).sort("step")
                if run.is_empty():
                    continue
                parameter_count = float(run["non_embedding_params"][0])
                shade = 0.28 + 0.67 * parameter_norm(np.log10(parameter_count))
                ax.plot(
                    run["step"].to_numpy(),
                    _ema(run[metric].to_numpy().astype(float)),
                    color=cmap(shade),
                    linewidth=0.55,
                    alpha=0.92,
                    rasterized=False,
                )

            ax.set_xlim(0, 50_000)
            ax.set_ylim(*y_limits)
            ax.xaxis.set_major_locator(mticker.FixedLocator([0, 25_000, 50_000]))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_step_formatter))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.grid(
                True,
                color=GRID_COLOR,
                linewidth=0.35,
                linestyle="-",
                alpha=0.75,
            )
            remove_bounding_box(ax, fontsize_ticks=5.2)
            if row_index == 0:
                ax.set_title(f"Context Length: {context:,}", fontsize=6.1, pad=4)
            if col_index == 0:
                ax.set_ylabel(label, fontsize=5.8)
            else:
                ax.tick_params(labelleft=False)
            if row_index == nrows - 1:
                ax.set_xlabel("Training steps", fontsize=5.8)

    fig.suptitle(
        f"{formulation}: downstream metric trajectories",
        fontsize=7.0,
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Within each context, darker lines denote more non-embedding parameters; curves use EMA smoothing (alpha = 0.15).",
        ha="center",
        va="bottom",
        fontsize=5.0,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.992,
        top=0.91 if nrows == 1 else 0.945,
        bottom=0.16 if nrows == 1 else 0.095,
        wspace=0.11,
        hspace=0.35,
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=False,
        max_size_mm=(180.0, 210.0),
        enforce_y_axis_units=False,
    )


def _format_parameter_target(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g}B"
    return f"{value / 1e6:g}M"


def _format_lr(value: float, _position: int | None = None) -> str:
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / 10**exponent
    if np.isclose(coefficient, 1.0):
        return rf"$10^{{{exponent}}}$"
    return rf"${coefficient:g}\times10^{{{exponent}}}$"


def create_lr_sweep_with_quadratics(
    df: pl.DataFrame,
    *,
    formulation: str,
    loss_label: str,
    output_path: Path,
    analysis_fraction: float | None = None,
    min_completion: float = 0.90,
) -> tuple[int, int]:
    if analysis_fraction is None:
        analysis_fraction = float(df["analysis_fraction"].max())
    slice_df = df.filter((pl.col("analysis_fraction") - analysis_fraction).abs() < 1e-9)
    if slice_df.is_empty():
        raise ValueError(f"No learning-rate rows at fraction {analysis_fraction:g}")
    terminal = geneformer_lr._terminal_loss_df(slice_df)
    optima = geneformer_lr._quadratic_best_lr_df(terminal, min_completion)
    targets = terminal["target_non_embedding_params"].unique().sort().to_list()
    depths = terminal["target_depth"].unique().sort().to_list()
    lr_values = terminal["learning_rate"].unique().sort().to_numpy().astype(float)
    compute = float(slice_df["analysis_training_flops"].drop_nulls().first())

    apply_plot_style()
    fig, axes = plt.subplots(
        len(targets),
        len(depths),
        figsize=(7.08, 7.0),
        squeeze=False,
        sharex=True,
    )
    quadratic_count = 0
    fallback_count = 0

    for row_index, target in enumerate(targets):
        row_values = (
            terminal.filter(pl.col("target_non_embedding_params") == target)[
                "terminal_loss"
            ]
            .drop_nulls()
            .to_numpy()
            .astype(float)
        )
        row_values = row_values[np.isfinite(row_values)]
        fitted_minima: list[float] = []
        for row in optima.filter(
            (pl.col("target_non_embedding_params") == target)
            & (pl.col("lr_fit_method") == "quadratic")
        ).iter_rows(named=True):
            log_lr = np.log10(float(row["optimal_lr"]))
            fitted_minima.append(
                float(row["quadratic_a"]) * log_lr**2
                + float(row["quadratic_b"]) * log_lr
                + float(row["quadratic_c"])
            )
        if fitted_minima:
            row_values = np.concatenate([row_values, np.asarray(fitted_minima)])
        row_pad = max(
            0.10 * (float(np.max(row_values)) - float(np.min(row_values))), 0.02
        )
        row_limits = (
            float(np.min(row_values)) - row_pad,
            float(np.max(row_values)) + row_pad,
        )

        for col_index, depth in enumerate(depths):
            ax = axes[row_index, col_index]
            panel = terminal.filter(
                (pl.col("target_non_embedding_params") == target)
                & (pl.col("target_depth") == depth)
            ).sort("learning_rate")
            x = panel["learning_rate"].to_numpy().astype(float)
            y = panel["terminal_loss"].to_numpy().astype(float)
            completion = panel["completion_fraction"].to_numpy().astype(float)
            finite = np.isfinite(y)
            eligible = (completion >= min_completion) & finite
            color = plt.colormaps[LR_SIZE_CMAPS[row_index % len(LR_SIZE_CMAPS)]](0.78)
            ax.plot(x[finite], y[finite], color=color, linewidth=0.75, alpha=0.65)
            ax.scatter(
                x[eligible],
                y[eligible],
                color=color,
                edgecolors="black",
                linewidths=0.35,
                s=12,
                zorder=4,
            )
            incomplete = finite & ~eligible
            if np.any(incomplete):
                ax.scatter(
                    x[incomplete],
                    y[incomplete],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=0.45,
                    s=12,
                    zorder=4,
                )

            optimum = optima.filter(
                (pl.col("target_non_embedding_params") == target)
                & (pl.col("target_depth") == depth)
            ).row(0, named=True)
            discrete_lr = float(optimum["discrete_optimal_lr"])
            best_index = int(np.argmin(np.abs(x - discrete_lr)))
            if optimum["lr_fit_method"] == "quadratic":
                quadratic_count += 1
                left = x[max(0, best_index - 1)]
                right = x[min(len(x) - 1, best_index + 1)]
                x_fit = np.geomspace(left, right, 120)
                log_x = np.log10(x_fit)
                y_fit = (
                    float(optimum["quadratic_a"]) * log_x**2
                    + float(optimum["quadratic_b"]) * log_x
                    + float(optimum["quadratic_c"])
                )
                optimal_lr = float(optimum["optimal_lr"])
                optimal_log_lr = np.log10(optimal_lr)
                optimal_loss = (
                    float(optimum["quadratic_a"]) * optimal_log_lr**2
                    + float(optimum["quadratic_b"]) * optimal_log_lr
                    + float(optimum["quadratic_c"])
                )
                ax.plot(
                    x_fit, y_fit, color="black", linestyle="--", linewidth=0.8, zorder=3
                )
                ax.scatter(
                    [optimal_lr],
                    [optimal_loss],
                    marker="*",
                    color="black",
                    s=27,
                    zorder=5,
                )
            else:
                fallback_count += 1
                ax.scatter(
                    [discrete_lr],
                    [float(optimum["best_terminal_loss"])],
                    marker="s",
                    color="black",
                    s=17,
                    zorder=5,
                )

            ax.set_xscale("log")
            ax.set_ylim(*row_limits)
            ax.xaxis.set_major_locator(mticker.FixedLocator(lr_values))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_lr))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.grid(True, color=GRID_COLOR, linewidth=0.35, alpha=0.75)
            remove_bounding_box(ax, fontsize_ticks=5.0)
            ax.tick_params(axis="x", rotation=32)
            if row_index == 0:
                ax.set_title(f"Depth {depth}", fontsize=6.2, pad=4)
            if col_index == 0:
                ax.set_ylabel(
                    f"{_format_parameter_target(float(target))}\n{loss_label}",
                    fontsize=5.8,
                )
            else:
                ax.tick_params(labelleft=False)
            if row_index == len(targets) - 1:
                ax.set_xlabel("Learning rate", fontsize=5.8)

    handles = [
        mlines.Line2D([], [], color="black", linestyle="--", label="local quadratic"),
        mlines.Line2D(
            [], [], color="black", marker="*", linestyle="None", label="inferred vertex"
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="s",
            linestyle="None",
            label="sampled fallback",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=5.2,
        bbox_to_anchor=(0.53, 0.006),
    )
    fig.suptitle(
        f"{formulation}: learning-rate sweeps at {compute:.2e} FLOPs",
        fontsize=7.0,
        y=0.993,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        top=0.945,
        bottom=0.095,
        wspace=0.10,
        hspace=0.30,
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=False,
        max_size_mm=(180.0, 210.0),
        enforce_x_axis_units=False,
        enforce_y_axis_units=False,
    )
    return quadratic_count, fallback_count


def _render_context_training_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    config,
    by_flops: bool,
) -> None:
    settings = {
        "FONT_SIZE_TICKS": 12.8,
        "FONT_SIZE_LABELS": 13.8,
        "FONT_SIZE_TITLE": 13.8,
        "FONT_SIZE_LEGEND": 13.8,
        "COLOR_SHADES": CONTEXT_LENGTH_CMAPS,
    }
    layout: dict[str, object] = {
        "ncols": 2,
        "figsize": (8.2, 8.3),
        "show_title": False,
        "hspace": 0.72,
        "compact_legend": True,
        "subplot_left": 0.04,
        "subplot_right": 0.985,
        "wspace": 0.07,
        "panel_box_aspect": 0.55,
        "tick_edge_padding_fraction": PAPER_TICK_EDGE_PADDING_FRACTION,
        "enforce_y_axis_units": False,
        "compact_colorbar_x": 0.79,
        "compact_shade_title_x": 0.5,
    }
    if config.metric_name == "mse":
        layout.update(
            {
                "y_major_ticks": (200.0, 250.0, 300.0, 350.0, 400.0, 450.0),
                "y_limits_override": (200.0, 470.0),
            }
        )
    else:
        layout.update(
            {
                "y_major_ticks": (4.0, 6.0, 8.0, 10.0, 12.0),
                "y_limits_override": (3.0, 12.0),
            }
        )
    if by_flops:
        layout.update(
            {
                "x_col": "cumulative_training_flops",
                "x_label": "Cumulative compute (FLOPs)",
                "x_tick_formatter": _flops_formatter,
                "x_tick_rotation": 25.0,
                "subplot_left": 0.06,
                "subplot_right": 0.94,
                "wspace": 0.12,
            }
        )
    else:
        layout["x_tick_formatter"] = _step_formatter

    with _temporary_plot_settings(training_loss, settings):
        training_loss.create_training_loss_plot_from_df(
            df,
            output_path,
            config=config,
            line_width=0.85,
            **layout,
        )


def _learning_rate_analysis(
    prepared: pl.DataFrame,
    *,
    min_completion: float = 0.90,
) -> tuple[list[tuple[float, pl.DataFrame]], list[dict[str, object]]]:
    best_by_compute: list[tuple[float, pl.DataFrame]] = []
    summaries: list[dict[str, object]] = []
    fractions = prepared["analysis_fraction"].drop_nulls().unique().sort().to_list()
    for fraction in fractions:
        fraction_df = prepared.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        compute = float(fraction_df["analysis_training_flops"].drop_nulls().first())
        best = geneformer_lr._quadratic_best_lr_df(
            geneformer_lr._terminal_loss_df(fraction_df), min_completion
        )
        best_by_compute.append((compute, best))
        summaries.append(
            geneformer_lr._power_law_validation_summary(float(fraction), compute, best)
        )
    return best_by_compute, summaries


def _render_lr_bootstrap_plot(
    best_by_compute: list[tuple[float, pl.DataFrame]],
    output_path: Path,
    *,
    title: str,
) -> None:
    with _temporary_plot_settings(geneformer_lr, BASE_PANEL_TYPOGRAPHY):
        geneformer_lr.create_power_law_bootstrap_panel_plot(
            best_by_compute,
            output_path,
            title=title,
            x_limits=(-2.0, 2.4),
            enforce_x_axis_units=False,
            enforce_y_axis_units=False,
        )


def _render_dw_observed_plot(
    df: pl.DataFrame,
    output_path: Path,
    *,
    ranked: bool,
) -> None:
    fractions = df["analysis_fraction"].drop_nulls().unique().sort().to_list()
    title = "Ranked Gene Identity" if ranked else "Binned Gene Expression"
    with _temporary_plot_settings(geneformer_dw, DW_PANEL_TYPOGRAPHY):
        geneformer_dw.create_loss_vs_depth_width_ratio_plot(
            df,
            output_path,
            fractions=fractions,
            model_size_cmap="plasma",
            ncols=2,
            ylabel="Cross-entropy" if ranked else "Mean Squared Error",
            title=title,
            enforce_x_axis_units=False,
            enforce_y_axis_units=False,
        )


def _render_dw_fit_diagnostics(
    df: pl.DataFrame,
    early_df: pl.DataFrame,
    output_path: Path,
    *,
    ranked: bool,
) -> None:
    model = geneformer_dw._fit_surface_model_from_fit_df(
        geneformer_dw._surface_fit_df(df)
    )
    if ranked:
        observed_label = "Observed Cross-entropy"
        fitted_label = "Fitted Cross-entropy"
        residual_label = "Cross-entropy residual"
        title = "Ranked Gene Identity"
    else:
        observed_label = "Observed Mean Squared Error"
        fitted_label = "Fitted Mean Squared Error"
        residual_label = "MSE residual"
        title = "Binned Gene Expression"
    with _temporary_plot_settings(geneformer_dw, DW_PANEL_TYPOGRAPHY):
        geneformer_dw.create_surface_fit_window_comparison_plot(
            df,
            early_df,
            output_path,
            model=model,
            figsize=(10.8, 16.0),
            crop_output=False,
            observed_loss_label=observed_label,
            fitted_loss_label=fitted_label,
            residual_label=residual_label,
            title=title,
            enforce_x_axis_units=False,
            enforce_y_axis_units=False,
        )


def _sample_dw_training_curves(
    training_df: pl.DataFrame,
    *,
    compute_budgets: tuple[float, ...] = DW_EARLY_FLOPS,
) -> pl.DataFrame:
    """Sample complete depth/width trajectories at fixed early compute budgets."""
    n_runs = training_df["run_hash"].n_unique()
    frames: list[pl.DataFrame] = []
    sorted_df = training_df.sort(["run_hash", "step"])
    for compute in compute_budgets:
        sampled = (
            sorted_df.with_columns(
                (pl.lit(compute) / pl.col("training_flops_per_step"))
                .round()
                .clip(1)
                .cast(pl.Int64)
                .alias("target_step")
            )
            .filter(pl.col("step") <= pl.col("target_step"))
            .group_by("run_hash", maintain_order=True)
            .tail(1)
            .with_columns(
                [
                    pl.lit(compute / geneformer_dw.DEFAULT_TARGET_TRAINING_FLOPS).alias(
                        "analysis_fraction"
                    ),
                    pl.lit(compute).alias("analysis_training_flops"),
                    pl.lit(1.0).alias("completion_fraction"),
                    (pl.col("step") * pl.col("training_flops_per_step")).alias(
                        "sampled_training_flops"
                    ),
                    (pl.col("target_step") < pl.col("trainer.lr_warmup_steps")).alias(
                        "warmup_ongoing"
                    ),
                ]
            )
        )
        if sampled.height != n_runs:
            raise ValueError(
                f"Expected {n_runs} runs at {compute:.3g} FLOPs, found {sampled.height}"
            )
        frames.append(sampled)
    return pl.concat(frames, how="vertical")


def _render_dw_slice_windows(
    ranked_training: pl.DataFrame,
    binned_training: pl.DataFrame,
    output_path: Path,
    *,
    analysis_flops: tuple[float, ...],
) -> None:
    early_color = WONG_PALETTE[0]
    analysis_color = WONG_PALETTE[1]
    x_min = min(DW_EARLY_FLOPS) / 1.35
    x_max = max(analysis_flops) * 1.10
    all_param_values = np.concatenate(
        [
            ranked_training["target_non_embedding_params"].to_numpy().astype(float),
            binned_training["target_non_embedding_params"].to_numpy().astype(float),
        ]
    )
    param_norm = mpl.colors.LogNorm(
        vmin=float(np.min(all_param_values)),
        vmax=float(np.max(all_param_values)),
    )
    param_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "dw_slice_window_model_size",
        plt.colormaps["Greys"](np.linspace(0.20, 0.88, 256)),
    )

    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.45))
    panels = (
        (
            axes[0],
            ranked_training,
            GENEFORMER_BCE_CONFIG,
            "Ranked Gene Identity",
            "Cross-entropy",
        ),
        (
            axes[1],
            binned_training,
            SCGPT_MSE_CONFIG,
            "Binned Gene Expression",
            "Mean Squared Error",
        ),
    )
    for ax, training_df, config, title, ylabel in panels:
        plotted_losses: list[float] = []
        for run_df in training_df.partition_by("run_hash", maintain_order=True):
            run_df = run_df.sort("step")
            x_full = run_df["cumulative_training_flops"].to_numpy().astype(float)
            y_full = geneformer_dw.exponential_moving_average(
                run_df["loss"].to_numpy().astype(float),
                alpha=config.ema_alpha,
            )
            visible = (x_full >= x_min) & (x_full <= x_max)
            visible_indices = np.flatnonzero(visible)
            if len(visible_indices) == 0:
                continue
            if len(visible_indices) > 900:
                sampled_positions = np.linspace(
                    0,
                    len(visible_indices) - 1,
                    900,
                    dtype=int,
                )
                visible_indices = visible_indices[sampled_positions]
            x = x_full[visible_indices]
            y = y_full[visible_indices]
            plotted_losses.extend(y.tolist())
            model_size = float(run_df["target_non_embedding_params"][0])
            ax.plot(
                x,
                y,
                color=param_cmap(param_norm(model_size)),
                linewidth=0.52,
                alpha=0.72,
                zorder=2,
            )

            warmup_step = int(run_df["trainer.lr_warmup_steps"][0])
            warmup_index = int(
                np.searchsorted(
                    run_df["step"].to_numpy(),
                    warmup_step,
                    side="left",
                )
            )
            if warmup_index < len(x_full) and x_min <= x_full[warmup_index] <= x_max:
                ax.scatter(
                    x_full[warmup_index],
                    y_full[warmup_index],
                    marker="D",
                    s=8,
                    facecolor="white",
                    edgecolor="#4f4a45",
                    linewidth=0.45,
                    alpha=0.72,
                    zorder=4,
                )

        for compute in DW_EARLY_FLOPS:
            ax.axvline(
                compute,
                color=early_color,
                linewidth=0.85,
                alpha=0.68,
                zorder=1,
            )
        for compute in analysis_flops:
            ax.axvline(
                compute,
                color=analysis_color,
                linewidth=0.85,
                alpha=0.68,
                zorder=1,
            )

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        if plotted_losses:
            ax.set_ylim(*geneformer_dw._loss_limits(np.asarray(plotted_losses)))
        ax.set_title(title, fontsize=7.4, pad=5)
        ax.set_xlabel("Cumulative Training FLOPs", fontsize=7.0)
        ax.set_ylabel(ylabel, fontsize=7.0)
        ax.xaxis.set_major_locator(
            mticker.FixedLocator([1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18])
        )
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _: geneformer_dw._format_flops_tick(value)
            )
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(5, integer=ylabel == "Mean Squared Error")
        )
        ax.grid(
            True,
            which="major",
            color=GRID_COLOR,
            linestyle="-",
            linewidth=0.4,
            alpha=0.75,
            zorder=0,
        )
        remove_bounding_box(ax, fontsize_ticks=6.2)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=analysis_color,
            linewidth=1.5,
            label="Analysis slices",
        ),
        mlines.Line2D(
            [],
            [],
            color=early_color,
            linewidth=1.5,
            label="Warm-up slices",
        ),
        mlines.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor="#4f4a45",
            label="Warm-up end",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        frameon=False,
        fontsize=6.4,
        ncol=3,
        handlelength=2.0,
        columnspacing=1.4,
    )
    colorbar_ax = fig.add_axes([0.40, 0.075, 0.20, 0.022])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=param_norm, cmap=param_cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks([1.0e6, 1.0e7, 1.0e8, 1.0e9])
    colorbar.set_ticklabels(["1M", "10M", "100M", "1B"])
    colorbar.ax.set_title("Non-embedding parameters", fontsize=6.2, pad=3)
    colorbar.ax.tick_params(labelsize=5.8, length=0, pad=2)
    colorbar.outline.set_linewidth(0.7)
    fig.suptitle(
        "Training Trajectories and Surface-fit Slices",
        fontsize=8.0,
        x=0.5,
        y=0.985,
        ha="center",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.76, bottom=0.22, wspace=0.24)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=False,
        max_size_mm=(180.0, 210.0),
        enforce_x_axis_units=False,
        enforce_y_axis_units=False,
    )


def _surface_bootstrap_instability(
    df: pl.DataFrame,
    *,
    target_params: float,
    target_compute: float,
    n_bootstrap: int = 250,
    seed: int = 11,
) -> dict[str, object]:
    fit_df = geneformer_dw._surface_fit_df(df)
    rng = np.random.default_rng(seed)
    curvatures: list[float] = []
    optimum_log_ratios: list[float] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, fit_df.height, size=fit_df.height)
        try:
            model = geneformer_dw._fit_surface_model_from_fit_df(
                geneformer_dw._take_rows(fit_df, sample_idx)
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        coefficients = model["coefficients"]
        means = model["means"]
        scales = model["scales"]
        assert isinstance(coefficients, np.ndarray)
        assert isinstance(means, np.ndarray)
        assert isinstance(scales, np.ndarray)
        curvature = float(coefficients[6])
        curvatures.append(curvature)
        if curvature <= 0:
            continue

        z_params = (np.log10(target_params) - means[0]) / scales[0]
        z_compute = (np.log10(target_compute) - means[1]) / scales[1]
        z_ratio = -(
            coefficients[3] + coefficients[8] * z_params + coefficients[9] * z_compute
        ) / (2.0 * curvature)
        optimum_log_ratios.append(float(means[2] + scales[2] * z_ratio))

    if not curvatures or not optimum_log_ratios:
        raise ValueError("Bootstrap surface diagnostics produced no valid optima")
    return {
        "n_fitted": len(curvatures),
        "n_nonconvex": sum(value <= 0 for value in curvatures),
        "optimum_log_ratios": np.asarray(optimum_log_ratios, dtype=float),
        "ratio_min": float(fit_df["depth_width_ratio"].min()),
        "ratio_max": float(fit_df["depth_width_ratio"].max()),
    }


def _render_dw_instability_plot(
    ranked_df: pl.DataFrame,
    binned_df: pl.DataFrame,
    output_path: Path,
    *,
    target_params: float = 1.0e8,
    target_compute: float = 1.0e18,
) -> None:
    summaries = [
        _surface_bootstrap_instability(
            data,
            target_params=target_params,
            target_compute=target_compute,
        )
        for data in (ranked_df, binned_df)
    ]
    labels = ["Ranked", "Binned"]
    positions = np.arange(len(labels), dtype=float)

    apply_plot_style()
    fig, (ax_convexity, ax_optimum) = plt.subplots(
        1,
        2,
        figsize=(7.08, 3.2),
        gridspec_kw={"width_ratios": [0.82, 1.18], "wspace": 0.34},
    )

    nonconvex_counts = np.asarray(
        [int(summary["n_nonconvex"]) for summary in summaries], dtype=int
    )
    fitted_counts = np.asarray(
        [int(summary["n_fitted"]) for summary in summaries], dtype=int
    )
    nonconvex_pct = 100.0 * nonconvex_counts / fitted_counts
    convex_pct = 100.0 - nonconvex_pct
    ax_convexity.bar(
        positions,
        convex_pct,
        width=0.62,
        color=WONG_PALETTE[2],
        label="Convex",
        zorder=3,
    )
    ax_convexity.bar(
        positions,
        nonconvex_pct,
        width=0.62,
        bottom=convex_pct,
        color=WONG_PALETTE[1],
        label="Non-convex",
        zorder=3,
    )
    for position, count, total, pct in zip(
        positions,
        nonconvex_counts,
        fitted_counts,
        nonconvex_pct,
        strict=True,
    ):
        y = 4.0 if count == 0 else 100.0 - pct / 2.0
        ax_convexity.text(
            position,
            y,
            f"{count}/{total}",
            ha="center",
            va="center",
            fontsize=6.2,
            color="white" if count else "#2f2a24",
            zorder=4,
        )
    ax_convexity.set_xticks(positions, labels)
    ax_convexity.set_ylim(0.0, 100.0)
    ax_convexity.set_ylabel("Bootstrap surfaces (%)", fontsize=7.0)
    ax_convexity.set_title("Convexity under resampling", fontsize=7.4, pad=6)
    ax_convexity.grid(
        True,
        axis="y",
        color=GRID_COLOR,
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=1,
    )
    ax_convexity.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        fontsize=6.0,
    )
    remove_bounding_box(ax_convexity, fontsize_ticks=6.2)

    ratio_min = min(float(summary["ratio_min"]) for summary in summaries)
    ratio_max = max(float(summary["ratio_max"]) for summary in summaries)
    ax_optimum.axvspan(
        ratio_min,
        ratio_max,
        color="#b9b2aa",
        alpha=0.18,
        linewidth=0.0,
        zorder=1,
    )
    interval_colors = (WONG_PALETTE[0], WONG_PALETTE[1])
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    for position, summary, color in zip(
        positions,
        summaries,
        interval_colors,
        strict=True,
    ):
        log_values = summary["optimum_log_ratios"]
        assert isinstance(log_values, np.ndarray)
        values = 10**log_values
        q025, q25, q50, q75, q975 = np.percentile(
            values,
            [2.5, 25.0, 50.0, 75.0, 97.5],
        )
        lower_limits.append(float(q025))
        upper_limits.append(float(q975))
        ax_optimum.plot(
            [q025, q975],
            [position, position],
            color=color,
            linewidth=1.2,
            zorder=3,
        )
        ax_optimum.plot(
            [q25, q75],
            [position, position],
            color=color,
            linewidth=5.0,
            solid_capstyle="butt",
            zorder=4,
        )
        ax_optimum.scatter(
            [q50],
            [position],
            color="white",
            edgecolors=color,
            linewidths=1.1,
            s=24,
            zorder=5,
        )
    ax_optimum.set_xscale("log")
    ax_optimum.set_xlim(
        min(lower_limits) / 2.0,
        max(ratio_max, max(upper_limits)) * 1.4,
    )
    ax_optimum.set_yticks(positions, labels)
    ax_optimum.invert_yaxis()
    ax_optimum.set_xlabel("Predicted optimal depth-to-width ratio", fontsize=7.0)
    ax_optimum.set_title(
        r"Optimum at 100M parameters and $10^{18}$ FLOPs",
        fontsize=7.4,
        pad=6,
    )
    ax_optimum.grid(
        True,
        axis="x",
        color=GRID_COLOR,
        linestyle="-",
        linewidth=0.4,
        alpha=0.8,
        zorder=0,
    )
    ax_optimum.text(
        0.98,
        0.05,
        "convex fits only",
        transform=ax_optimum.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#5d5750",
    )
    remove_bounding_box(ax_optimum, fontsize_ticks=6.2)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.24)
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=False,
        max_size_mm=(180.0, 210.0),
        enforce_x_axis_units=False,
        enforce_y_axis_units=False,
    )


def _render_ranked_ratio_grid(df: pl.DataFrame, output_path: Path) -> None:
    model = geneformer_dw._fit_surface_model_from_fit_df(
        geneformer_dw._surface_fit_df(df)
    )
    with _temporary_plot_settings(geneformer_dw, DW_PANEL_TYPOGRAPHY):
        geneformer_dw.create_surface_grid_overlay_plot(
            df,
            output_path,
            model=model,
            target_training_flops=geneformer_dw.DEFAULT_TARGET_TRAINING_FLOPS,
            title="Ranked Gene Identity: Optimum Relative to Fixed Depth",
            enforce_y_axis_units=False,
        )


def _render_ranked_implied_depth(df: pl.DataFrame, output_path: Path) -> None:
    model = geneformer_dw._fit_surface_model_from_fit_df(
        geneformer_dw._surface_fit_df(df)
    )
    with _temporary_plot_settings(geneformer_dw, DW_PANEL_TYPOGRAPHY):
        geneformer_dw.create_surface_implied_depth_plot(
            df,
            output_path,
            model=model,
            target_training_flops=geneformer_dw.DEFAULT_TARGET_TRAINING_FLOPS,
            title="Ranked Gene Identity: Implied Optimal Depth",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranked_training = pl.read_parquet(paper_data("geneformer_like/01_training_loss_prepared.parquet"))
    binned_training = pl.read_parquet(paper_data("scgpt_like/01_training_loss_prepared.parquet"))
    training_sources = (
        ("ranked", ranked_training, GENEFORMER_BCE_CONFIG),
        ("binned", binned_training, SCGPT_MSE_CONFIG),
    )
    for stem, data, config in training_sources:
        for suffix, by_flops in (("steps", False), ("flops", True)):
            output = args.output_dir / f"supp_{stem}_training_{suffix}.pdf"
            _render_paper_pdf(
                lambda source, data=data, config=config, by_flops=by_flops: (
                    _render_context_training_plot(
                        data,
                        source,
                        config=config,
                        by_flops=by_flops,
                    )
                ),
                output,
                width_mm=HALF_WIDTH_MM,
            )
            print(f"Saved {output}")

    downstream_sources = (
        ("ranked", "Ranked Gene Identity", RANKED_DOWNSTREAM),
        ("binned", "Binned Gene Expression", BINNED_DOWNSTREAM),
    )
    for stem, label, source in downstream_sources:
        df = pl.read_parquet(source)
        for group_name, metrics in DOWNSTREAM_GROUPS.items():
            output = args.output_dir / f"supp_{stem}_downstream_{group_name}.pdf"
            _render_paper_pdf(
                lambda source, df=df, label=label, metrics=metrics: (
                    create_downstream_group(
                        df,
                        formulation=label,
                        metrics=metrics,
                        output_path=source,
                    )
                ),
                output,
                width_mm=FULL_WIDTH_MM,
            )
            print(f"Saved {output}")

    lr_sources = (
        ("ranked", "Ranked Gene Identity", "Cross-entropy", RANKED_LR),
        ("binned", "Binned Gene Expression", "Mean Squared Error", BINNED_LR),
    )
    for stem, label, loss_label, source in lr_sources:
        prepared = pl.read_parquet(source)
        output = args.output_dir / f"supp_{stem}_lr_sweeps.pdf"
        _render_paper_pdf(
            lambda temporary, prepared=prepared, label=label, loss_label=loss_label: (
                create_lr_sweep_with_quadratics(
                    prepared,
                    formulation=label,
                    loss_label=loss_label,
                    output_path=temporary,
                )
            ),
            output,
            width_mm=FULL_WIDTH_MM,
        )
        print(f"Saved {output}")

        best_by_compute, _ = _learning_rate_analysis(prepared)

        bootstrap_output = args.output_dir / f"supp_{stem}_lr_bootstrap.pdf"
        _render_paper_pdf(
            lambda temporary, best_by_compute=best_by_compute, label=label: (
                _render_lr_bootstrap_plot(
                    best_by_compute,
                    temporary,
                    title=label,
                )
            ),
            bootstrap_output,
            width_mm=HALF_WIDTH_MM,
        )
        print(f"Saved {bootstrap_output}")

    ranked_dw = pl.read_parquet(RANKED_DW)
    binned_dw = pl.read_parquet(BINNED_DW)
    ranked_dw_training = pl.read_parquet(RANKED_DW_TRAINING)
    binned_dw_training = pl.read_parquet(BINNED_DW_TRAINING)
    ranked_dw_early = _sample_dw_training_curves(ranked_dw_training)
    binned_dw_early = _sample_dw_training_curves(binned_dw_training)

    slice_windows_output = args.output_dir / "supp_dw_slice_windows.pdf"
    analysis_flops = tuple(
        float(value)
        for value in ranked_dw["analysis_training_flops"]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    _render_paper_pdf(
        lambda temporary: _render_dw_slice_windows(
            ranked_dw_training,
            binned_dw_training,
            temporary,
            analysis_flops=analysis_flops,
        ),
        slice_windows_output,
        width_mm=FULL_WIDTH_MM,
    )
    print(f"Saved {slice_windows_output}")
    for stem, data, early_data, ranked in (
        ("ranked", ranked_dw, ranked_dw_early, True),
        ("binned", binned_dw, binned_dw_early, False),
    ):
        observed_output = args.output_dir / f"supp_{stem}_dw_observed.pdf"
        _render_paper_pdf(
            lambda temporary, data=data, ranked=ranked: _render_dw_observed_plot(
                data,
                temporary,
                ranked=ranked,
            ),
            observed_output,
            width_mm=HALF_WIDTH_MM,
        )
        print(f"Saved {observed_output}")

        diagnostics_output = args.output_dir / f"supp_{stem}_dw_fit_diagnostics.pdf"
        _render_paper_pdf(
            lambda temporary, data=data, early_data=early_data, ranked=ranked: (
                _render_dw_fit_diagnostics(
                    data,
                    early_data,
                    temporary,
                    ranked=ranked,
                )
            ),
            diagnostics_output,
            width_mm=HALF_WIDTH_MM,
        )
        print(f"Saved {diagnostics_output}")

    for suffix, renderer in (
        ("ratio_grid", _render_ranked_ratio_grid),
        ("implied_depth", _render_ranked_implied_depth),
    ):
        output = args.output_dir / f"supp_ranked_dw_{suffix}.pdf"
        _render_paper_pdf(
            lambda temporary, renderer=renderer: renderer(ranked_dw, temporary),
            output,
            width_mm=FULL_WIDTH_MM,
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
