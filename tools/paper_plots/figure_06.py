from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

from plotting import geneformer_dw as dw
from plotting.training_loss import _style_panel_box
from plotting.utils import COLOR_SHADES, PLOTS_DIR, save_figure
from tools.paper_plots.data import paper_data
from tools.paper_plots.style import (
    PAPER_MAX_FONT_SIZE_PT,
    use_paper_style,
    validate_paper_dimensions,
    validate_paper_svg,
)


PAPER_WIDTH_MM = 180.0
PAPER_HEIGHT_MM = 150.0
DEFAULT_OUTPUT = PLOTS_DIR / "paper" / "06_geneformer_surface_contours.svg"
PREPARED_DATA = paper_data("geneformer_dw/01_isoflops_prepared.parquet")
GRID_SIZE = 120
CONTOUR_LEVEL_COUNT = 12
OPTIMUM_LINE_WIDTH = 1.25
BLUES_RANGE = (0.25, 0.95)
BLUES_ALPHA_RANGE = (0.68, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper Figure 06 from the Geneformer depth/width response-surface "
            "contours (plots 10f, 10g, and 10h)."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require_data(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached Geneformer surface data: {path}")
    return pl.read_parquet(path)


def _every_other(values: list[float]) -> list[float]:
    return [float(value) for value in values[::2]]


def _format_flops(value: float) -> str:
    formatted = dw._format_flops_value(value)
    mantissa, exponent = formatted.split("e")
    if float(mantissa) == 1.0:
        return f"1e{exponent}"
    return formatted


def _format_axis(
    ax: plt.Axes,
    *,
    x_kind: str,
    y_kind: str,
    show_ylabel: bool,
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    if x_kind == "params":
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _: dw._format_compact_param_count(value)
            )
        )
        xlabel = "Parameters"
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_flops(value))
        )
        xlabel = "Training FLOPs"
    if y_kind == "ratio":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:.3g}")
        )
        ylabel = "Depth-to-width ratio"
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: _format_flops(value))
        )
        ylabel = "Training compute (FLOPs)"
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(xlabel, fontsize=7)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.30,
        alpha=0.55,
        zorder=1,
    )
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=6.5, pad=1.5)


def _draw_contours(
    ax: plt.Axes,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    z: np.ndarray,
    *,
    levels: np.ndarray,
) -> mpl.contour.QuadContourSet:
    blue_colors = plt.colormaps[COLOR_SHADES[0]](
        np.linspace(BLUES_RANGE[0], BLUES_RANGE[1], 256)
    )
    blue_colors[:, 3] = np.linspace(
        BLUES_ALPHA_RANGE[0], BLUES_ALPHA_RANGE[1], len(blue_colors)
    )
    blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "figure_06_accessible_blues",
        blue_colors,
    )
    contourf = ax.contourf(
        x_mesh,
        y_mesh,
        z,
        levels=levels,
        cmap=blue_cmap,
        extend="both",
    )
    ax.contour(
        x_mesh,
        y_mesh,
        z,
        levels=levels,
        colors="#26211d",
        linewidths=0.42,
        alpha=0.75,
    )
    return contourf


def _scatter_observations(
    ax: plt.Axes,
    frame: pl.DataFrame,
    *,
    x_column: str,
    y_column: str,
    alpha: float = 0.74,
) -> None:
    complete = frame.filter(
        pl.col("completion_fraction") >= dw.COMPLETE_POINT_THRESHOLD
    )
    ax.scatter(
        complete[x_column].to_numpy().astype(float),
        complete[y_column].to_numpy().astype(float),
        s=7,
        facecolors="#ffffff",
        edgecolors="#1f1a16",
        linewidths=0.28,
        alpha=alpha,
        zorder=6,
    )


def _plot_surface_optimum_line(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray
) -> None:
    ax.plot(
        x,
        y,
        color=dw.SURFACE_OPTIMUM_LINE_COLOR,
        linestyle=":",
        linewidth=OPTIMUM_LINE_WIDTH,
        dash_capstyle="round",
        zorder=7,
    )


def _analytic_surface_best_ratios(
    model: dict[str, object],
    params: np.ndarray,
    compute: float | np.ndarray,
) -> np.ndarray:
    coefficients = model["coefficients"]
    means = model["means"]
    scales = model["scales"]
    assert isinstance(coefficients, np.ndarray)
    assert isinstance(means, np.ndarray)
    assert isinstance(scales, np.ndarray)

    ratio_quadratic = float(coefficients[6])
    if ratio_quadratic <= 0:
        raise ValueError(
            "The fitted log-ratio quadratic is not convex, so it has no "
            "analytical minimum"
        )

    params_values, compute_values = np.broadcast_arrays(
        np.asarray(params, dtype=float), np.asarray(compute, dtype=float)
    )
    standardized_params = (np.log10(params_values) - means[0]) / scales[0]
    standardized_compute = (np.log10(compute_values) - means[1]) / scales[1]
    standardized_ratio = -(
        coefficients[3]
        + coefficients[8] * standardized_params
        + coefficients[9] * standardized_compute
    ) / (2.0 * ratio_quadratic)
    return 10 ** (means[2] + scales[2] * standardized_ratio)


def _common_loss_levels(
    model: dict[str, object],
    params_grid: np.ndarray,
    compute_grid: np.ndarray,
    ratio_grid: np.ndarray,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    for compute in np.geomspace(compute_grid[0], compute_grid[-1], 12):
        _, _, z = dw._surface_prediction_grid(
            model,
            params_grid,
            ratio_grid,
            x_kind="params",
            y_kind="ratio",
            fixed_kind="compute",
            fixed_value=float(compute),
        )
        samples.append(z.ravel())
    values = np.concatenate(samples)
    return np.linspace(
        float(np.nanpercentile(values, 3)),
        float(np.nanpercentile(values, 97)),
        CONTOUR_LEVEL_COUNT,
    )


def _draw_shared_legend(
    fig: plt.Figure,
    legend_ax: plt.Axes,
    contourf: mpl.contour.QuadContourSet,
) -> None:
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    _style_panel_box(legend_ax, rounded=True)
    legend_ax.text(
        0.5,
        0.84,
        "Fitted loss (nats)",
        transform=legend_ax.transAxes,
        fontsize=7,
        va="center",
        ha="center",
    )
    colorbar_ax = legend_ax.inset_axes([0.08, 0.54, 0.84, 0.085])
    colorbar = fig.colorbar(contourf, cax=colorbar_ax, orientation="horizontal")
    colorbar_ticks = [
        float(contourf.levels[0]),
        float(contourf.levels[len(contourf.levels) // 2]),
        float(contourf.levels[-1]),
    ]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([f"{value:.1f}" for value in colorbar_ticks])
    colorbar.ax.tick_params(labelsize=6.5, length=0, pad=1.5)
    colorbar.outline.set_linewidth(0.7)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=dw.SURFACE_OPTIMUM_LINE_COLOR,
            linestyle=":",
            linewidth=OPTIMUM_LINE_WIDTH,
            dash_capstyle="round",
            label="Predicted surface optimum",
        ),
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor="#1f1a16",
            markeredgewidth=0.6,
            label="Observed configuration",
        ),
    ]
    legend_ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        frameon=False,
        fontsize=6.8,
        handlelength=2.0,
        labelspacing=0.55,
    )


def build_figure(output_path: Path) -> Path:
    validate_paper_dimensions(PAPER_WIDTH_MM, PAPER_HEIGHT_MM)
    raw = _require_data(PREPARED_DATA)
    all_fractions = sorted(
        float(value) for value in raw["analysis_fraction"].unique().to_list()
    )
    selected_fractions = _every_other(all_fractions)
    if len(selected_fractions) != 4:
        raise ValueError(
            "Figure 06 expects seven Geneformer compute slices so every-other "
            f"sampling yields four; found {len(all_fractions)} slices"
        )

    data = dw._with_depth_width_ratio(
        dw._filter_analysis_fractions(raw, all_fractions)
    )
    model = dw._fit_surface_model_from_fit_df(dw._surface_fit_df(data))
    params_values = sorted(
        float(value)
        for value in data["target_non_embedding_params"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    selected_params = _every_other(params_values)
    if len(selected_params) != 4:
        raise ValueError(
            "Figure 06 expects seven Geneformer parameter slices so every-other "
            f"sampling yields four; found {len(params_values)} slices"
        )

    log_padding = 10**0.2
    observed_params_limits = (min(params_values), max(params_values))
    observed_compute_limits = (
        min(all_fractions) * dw.DEFAULT_TARGET_TRAINING_FLOPS,
        max(all_fractions) * dw.DEFAULT_TARGET_TRAINING_FLOPS,
    )
    observed_ratio_limits = (
        float(model["ratio_min"]),
        float(model["ratio_max"]),
    )
    params_limits = (
        observed_params_limits[0] / log_padding,
        observed_params_limits[1] * log_padding,
    )
    compute_limits = (
        observed_compute_limits[0] / log_padding,
        observed_compute_limits[1] * log_padding,
    )
    ratio_limits = (
        observed_ratio_limits[0] / log_padding,
        observed_ratio_limits[1] * log_padding,
    )
    params_grid = np.geomspace(*params_limits, GRID_SIZE)
    compute_grid = np.geomspace(*compute_limits, GRID_SIZE)
    ratio_grid = np.geomspace(
        *ratio_limits,
        GRID_SIZE,
    )
    all_ratio_values = 10 ** np.quantile(
        np.log10(data["depth_width_ratio"].drop_nulls().to_numpy().astype(float)),
        [0.10, 0.30, 0.50, 0.70, 0.90],
    )
    selected_ratios = all_ratio_values[::2]
    levels = _common_loss_levels(model, params_grid, compute_grid, ratio_grid)

    use_paper_style()
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(PAPER_WIDTH_MM / 25.4, PAPER_HEIGHT_MM / 25.4),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    contourf: mpl.contour.QuadContourSet | None = None
    for column, fraction in enumerate(selected_fractions):
        ax = axes[0, column]
        compute = fraction * dw.DEFAULT_TARGET_TRAINING_FLOPS
        x_mesh, y_mesh, z = dw._surface_prediction_grid(
            model,
            params_grid,
            ratio_grid,
            x_kind="params",
            y_kind="ratio",
            fixed_kind="compute",
            fixed_value=float(compute),
        )
        contourf = _draw_contours(ax, x_mesh, y_mesh, z, levels=levels)
        _plot_surface_optimum_line(
            ax,
            params_grid,
            _analytic_surface_best_ratios(model, params_grid, compute),
        )
        panel = data.filter(
            (pl.col("analysis_fraction") - float(fraction)).abs() < 1e-9
        )
        _scatter_observations(
            ax,
            panel,
            x_column="target_non_embedding_params",
            y_column="depth_width_ratio",
        )
        ax.set_title(f"{_format_flops(compute)} FLOPs", fontsize=7, pad=4)
        _format_axis(
            ax, x_kind="params", y_kind="ratio", show_ylabel=column == 0
        )
        ax.set_xlim(*params_limits)
        ax.set_ylim(*ratio_limits)

    for column, params_value in enumerate(selected_params):
        ax = axes[1, column]
        x_mesh, y_mesh, z = dw._surface_prediction_grid(
            model,
            compute_grid,
            ratio_grid,
            x_kind="compute",
            y_kind="ratio",
            fixed_kind="params",
            fixed_value=float(params_value),
        )
        contourf = _draw_contours(ax, x_mesh, y_mesh, z, levels=levels)
        best_ratios = _analytic_surface_best_ratios(
            model,
            np.full_like(compute_grid, params_value),
            compute_grid,
        )
        _plot_surface_optimum_line(ax, compute_grid, best_ratios)
        panel = data.filter(
            pl.col("target_non_embedding_params") == float(params_value)
        )
        _scatter_observations(
            ax,
            panel,
            x_column="analysis_training_flops",
            y_column="depth_width_ratio",
        )
        ax.set_title(
            dw._format_compact_param_count(params_value), fontsize=7, pad=4
        )
        _format_axis(
            ax, x_kind="compute", y_kind="ratio", show_ylabel=column == 0
        )
        ax.set_xlim(*compute_limits)
        ax.set_ylim(*ratio_limits)

    for column, ratio_value in enumerate(selected_ratios):
        ax = axes[2, column]
        x_mesh, y_mesh, z = dw._surface_prediction_grid(
            model,
            params_grid,
            compute_grid,
            x_kind="params",
            y_kind="compute",
            fixed_kind="ratio",
            fixed_value=float(ratio_value),
        )
        contourf = _draw_contours(ax, x_mesh, y_mesh, z, levels=levels)
        nearest = data.with_columns(
            (pl.col("depth_width_ratio").log10() - float(np.log10(ratio_value)))
            .abs()
            .alias("d_ratio")
        ).filter(pl.col("d_ratio") <= pl.col("d_ratio").quantile(0.12))
        _scatter_observations(
            ax,
            nearest,
            x_column="target_non_embedding_params",
            y_column="analysis_training_flops",
            alpha=0.58,
        )
        ax.set_title(
            f"ratio {dw._format_ratio_value(float(ratio_value))}",
            fontsize=7,
            pad=4,
        )
        _format_axis(
            ax, x_kind="params", y_kind="compute", show_ylabel=column == 0
        )
        ax.set_xlim(*params_limits)
        ax.set_ylim(*compute_limits)

    assert contourf is not None
    _draw_shared_legend(fig, axes[2, 3], contourf)

    row_titles = (
        "Parameters × Depth/Width at fixed compute",
        "Compute × Depth/Width at fixed parameter count",
        "Parameters × Compute at fixed depth/width ratio",
    )
    for row, (label, title) in enumerate(zip("ABC", row_titles, strict=True)):
        axes[row, 0].text(
            -0.28,
            1.25,
            label,
            transform=axes[row, 0].transAxes,
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="top",
        )
        axes[row, 1].text(
            1.10,
            1.25,
            title,
            transform=axes[row, 1].transAxes,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="top",
        )

    fig.subplots_adjust(
        left=0.085,
        right=0.970,
        top=0.935,
        bottom=0.075,
        wspace=0.20,
        hspace=0.72,
    )
    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=False,
        max_font_size_pt=PAPER_MAX_FONT_SIZE_PT,
        max_size_mm=(PAPER_WIDTH_MM, 210.0),
        enforce_y_axis_units=False,
    )
    validate_paper_svg(output_path)
    return output_path


def main() -> None:
    svg_path = build_figure(parse_args().output)
    print(f"Wrote {svg_path} ({PAPER_WIDTH_MM:g} mm × {PAPER_HEIGHT_MM:g} mm)")


if __name__ == "__main__":
    main()
