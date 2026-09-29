from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.path import Path as MplPath
import numpy as np
import polars as pl

from plotting.utils import (
    COLOR_SHADES,
    FONT_SIZE_LABELS,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICKS,
    FONT_SIZE_TITLE,
    apply_plot_style,
    exponential_moving_average,
    format_param_name,
    format_param_value,
    remove_bounding_box,
    save_figure,
    series_unique_sorted,
)


@dataclass(frozen=True)
class TrainingLossPlotConfig:
    pipeline: str
    metric_name: str
    title: str
    ylabel: str
    color_param_candidates: tuple[str, ...]
    shade_param_candidates: tuple[str, ...]
    continuous_shade_param_candidates: tuple[str, ...] = ()
    y_limits: tuple[float, float] | None = None
    ema_alpha: float = 0.1


GENEFORMER_BCE_CONFIG = TrainingLossPlotConfig(
    pipeline="geneformer",
    metric_name="bce",
    title="Training Performance on Ranked Gene Identity",
    ylabel="Cross Entropy",
    color_param_candidates=(
        "datamodule.batch_size",
        "model.context_length",
        "model.transformer.n_layers",
        "model.bioformer.n_layers",
    ),
    shade_param_candidates=(
        "model.context_length",
        "model.transformer.n_layers",
        "model.bioformer.n_layers",
        "trainer.accumulate_grad",
    ),
    continuous_shade_param_candidates=("non_embedding_params",),
    y_limits=None,
)


SCGPT_MSE_CONFIG = TrainingLossPlotConfig(
    pipeline="default",
    metric_name="mse",
    title="Training Performance on Binned Gene Expression",
    ylabel="Mean Squared Error",
    color_param_candidates=(
        "datamodule.batch_size",
        "model.context_length",
        "model.transformer.n_layers",
        "model.bioformer.n_layers",
    ),
    shade_param_candidates=(
        "model.context_length",
        "model.transformer.n_layers",
        "model.bioformer.n_layers",
        "trainer.accumulate_grad",
    ),
    continuous_shade_param_candidates=("non_embedding_params",),
    y_limits=(0.0, 454.0),
)


SCGPT_BCE_CONFIG = TrainingLossPlotConfig(
    pipeline="default",
    metric_name="bce",
    title="Training Performance on Masked Gene Activity Prediction",
    ylabel="Binary Cross Entropy",
    color_param_candidates=(
        "datamodule.batch_size",
        "model.context_length",
        "model.transformer.n_layers",
        "model.bioformer.n_layers",
    ),
    shade_param_candidates=(
        "model.context_length",
        "model.transformer.n_layers",
        "model.bioformer.n_layers",
        "trainer.accumulate_grad",
    ),
    continuous_shade_param_candidates=("non_embedding_params",),
    y_limits=None,
)


def _select_style_dimensions(
    df: pl.DataFrame,
    color_candidates: tuple[str, ...],
    shade_candidates: tuple[str, ...],
    continuous_shade_candidates: tuple[str, ...],
) -> tuple[str, str | None, bool]:
    varying_columns = {
        column: len(series_unique_sorted(df, column))
        for column in df.columns
        if len(series_unique_sorted(df, column)) > 1
    }

    color_param = None
    for candidate in color_candidates:
        if varying_columns.get(candidate, 0) > 1:
            color_param = candidate
            break
    if color_param is None:
        raise ValueError("No varying parameter found to drive plot colors")

    for candidate in continuous_shade_candidates:
        if candidate != color_param and varying_columns.get(candidate, 0) > 1:
            return color_param, candidate, True

    shade_param = None
    for candidate in shade_candidates:
        if candidate != color_param and varying_columns.get(candidate, 0) > 1:
            shade_param = candidate
            break

    return color_param, shade_param, False


def _resolve_y_limits(values: np.ndarray, configured_limits: tuple[float, float] | None) -> tuple[float, float]:
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_range = max(data_max - data_min, 1e-6)
    padding = max(0.04 * data_range, 0.08)
    dynamic_limits = (data_min - padding, data_max + padding)

    if configured_limits is None:
        return dynamic_limits

    configured_span = configured_limits[1] - configured_limits[0]
    if configured_span > data_range * 2.5:
        return dynamic_limits
    return configured_limits


def _style_legend(legend: plt.Legend) -> None:
    frame = legend.get_frame()
    frame.set_linewidth(1.2)
    frame.set_edgecolor("#b8ab93")
    frame.set_facecolor("#ffffff")
    frame.set_alpha(0.9)


def _style_panel_box(ax: plt.Axes, *, rounded: bool = False, draw_box: bool = True) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    if not draw_box:
        return

    figure = ax.figure
    bbox = ax.get_position()
    panel_aspect = (bbox.width * figure.get_figwidth()) / (bbox.height * figure.get_figheight())
    boxstyle = "round,pad=0.01" if rounded else "square,pad=0.0"
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle=boxstyle,
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#000000",
            linewidth=0.9,
            clip_on=False,
            zorder=-1,
            mutation_scale=7,
            mutation_aspect=panel_aspect,
        )
    )


def _add_isotropic_rounded_frame(
    parent_ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    radius_mm: float = 1.35,
) -> None:
    """Draw a rounded frame with a fixed physical radius in parent axes coordinates."""
    x, y, width, height = bounds
    bbox = parent_ax.get_position()
    figure = parent_ax.figure
    axes_width_mm = bbox.width * figure.get_figwidth() * 25.4
    axes_height_mm = bbox.height * figure.get_figheight() * 25.4
    radius_x = min(radius_mm / axes_width_mm, width / 2)
    radius_y = min(radius_mm / axes_height_mm, height / 2)
    curve = 0.5522847498

    vertices = [
        (x + radius_x, y),
        (x + width - radius_x, y),
        (x + width - radius_x + curve * radius_x, y),
        (x + width, y + radius_y - curve * radius_y),
        (x + width, y + radius_y),
        (x + width, y + height - radius_y),
        (x + width, y + height - radius_y + curve * radius_y),
        (x + width - radius_x + curve * radius_x, y + height),
        (x + width - radius_x, y + height),
        (x + radius_x, y + height),
        (x + radius_x - curve * radius_x, y + height),
        (x, y + height - radius_y + curve * radius_y),
        (x, y + height - radius_y),
        (x, y + radius_y),
        (x, y + radius_y - curve * radius_y),
        (x + radius_x - curve * radius_x, y),
        (x + radius_x, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]
    parent_ax.add_patch(
        mpatches.PathPatch(
            MplPath(vertices, codes),
            transform=parent_ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#000000",
            linewidth=0.9,
            clip_on=False,
            zorder=-1,
        )
    )


def _with_derived_plot_columns(df: pl.DataFrame) -> pl.DataFrame:
    if {
        "hparams.parameters_total",
        "hparams.parameters_embedding",
    }.issubset(set(df.columns)):
        return df.with_columns(
            (
                pl.col("hparams.parameters_total") - pl.col("hparams.parameters_embedding")
            ).alias("non_embedding_params")
        )
    return df


def _format_param_count(value: float) -> str:
    def compact(scaled_value: float, suffix: str) -> str:
        label = f"{scaled_value:.1f}".rstrip("0").rstrip(".")
        return f"{label}{suffix}"

    if value >= 1_000_000_000:
        return compact(value / 1_000_000_000, "B")
    if value >= 1_000_000:
        return compact(value / 1_000_000, "M")
    if value >= 1_000:
        return compact(value / 1_000, "K")
    return str(int(value))


def _format_shade_value(parameter: str, value: float) -> str:
    if parameter == "non_embedding_params":
        return _format_param_count(value)
    rounded = round(float(value))
    if np.isclose(float(value), rounded):
        return str(int(rounded))
    return format_param_value(float(value))


def _wrapped_param_name(parameter: str) -> str:
    return "\n".join(format_param_name(parameter).split())


def _format_large_number_tick(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000_000_000:
        return f"{value / 1_000_000_000_000_000:.1f}e15"
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(round(value)))


def _select_linear_ticks(x_max: float, max_ticks: int = 6) -> list[float]:
    if x_max <= 0:
        return [0.0]
    return list(np.linspace(0.0, x_max, max_ticks))


def _select_range_ticks(y_min: float, y_max: float, max_ticks: int = 6) -> list[float]:
    if y_max <= y_min:
        return [float(y_min)]
    return list(np.linspace(float(y_min), float(y_max), max_ticks))


def _format_loss_tick(value: float) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    abs_value = abs(float(value))
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _linear_limits_from_ticks(ticks: list[float], edge_fraction: float = 0.1) -> tuple[float, float]:
    if len(ticks) < 2:
        tick = float(ticks[0]) if ticks else 0.0
        return tick - 1.0, tick + 1.0

    lower_step = float(ticks[1] - ticks[0])
    upper_step = float(ticks[-1] - ticks[-2])
    return (
        float(ticks[0]) - edge_fraction * lower_step,
        float(ticks[-1]) + edge_fraction * upper_step,
    )


def _truncated_colormap(name: str, start: float = 0.0, stop: float = 1.0) -> mpl.colors.Colormap:
    base_cmap = plt.colormaps[name]
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{name}_truncated_{start:.2f}_{stop:.2f}",
        base_cmap(np.linspace(start, stop, 256)),
    )


def _make_subplot_title(color_param: str, color_value: object) -> str:
    return f"{format_param_name(color_param)}: {format_param_value(color_value)}"


def _build_context_handles(color_values: list[object]) -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for index, value in enumerate(color_values):
        cmap_name = COLOR_SHADES[index % len(COLOR_SHADES)]
        color = plt.colormaps[cmap_name](0.78)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=3,
                label=str(value),
            )
        )
    return handles


def create_training_loss_plot(
    csv_path: Path,
    output_path: Path,
    config: TrainingLossPlotConfig,
) -> None:
    df = load_training_loss_data(csv_path, config)
    create_training_loss_plot_from_df(df, output_path, config)


def load_training_loss_data(
    csv_path: Path,
    config: TrainingLossPlotConfig,
) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    df = df.filter(
        (pl.col("metadata.pipeline") == config.pipeline)
        & (pl.col("metric_name") == config.metric_name)
    )
    if "context.subset" in df.columns:
        df = df.filter(pl.col("context.subset") == "train")
    df = _with_derived_plot_columns(df)
    if df.is_empty():
        raise ValueError(
            f"No rows found for pipeline={config.pipeline!r} and metric={config.metric_name!r} in {csv_path}"
        )
    return df


def create_training_loss_plot_from_df(
    df: pl.DataFrame,
    output_path: Path,
    config: TrainingLossPlotConfig,
    *,
    x_col: str = "step",
    x_label: str = "Training Steps",
    x_tick_formatter: Callable[[float], str] | None = None,
    x_tick_rotation: float = 0.0,
    line_width: float = 0.3,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    show_title: bool = True,
    hspace: float = 0.52,
    compact_legend: bool = False,
    subplot_left: float = 0.08,
    subplot_right: float = 0.98,
    wspace: float = 0.18,
    panel_box_aspect: float = 0.78,
    tick_edge_padding_fraction: float | None = None,
    y_major_ticks: tuple[float, ...] | None = None,
    y_limits_override: tuple[float, float] | None = None,
    enforce_y_axis_units: bool = True,
    compact_colorbar_x: float = 0.08,
    compact_shade_title_x: float = 0.56,
) -> None:
    if df.is_empty():
        raise ValueError("Cannot create a training-loss plot from an empty dataframe")

    color_param, shade_param, continuous_shade = _select_style_dimensions(
        df,
        config.color_param_candidates,
        config.shade_param_candidates,
        config.continuous_shade_param_candidates,
    )
    y_values = df["value"].to_numpy()
    shade_start = 0.32
    x_data_max = float(df[x_col].max())
    apply_plot_style()

    color_values = series_unique_sorted(df, color_param)
    n_panels = len(color_values)
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
    shade_norm = None
    if continuous_shade and shade_param is not None:
        shade_values = df[shade_param].to_numpy().astype(float)
        positive_shade_values = shade_values[shade_values > 0]
        if len(positive_shade_values) == 0:
            raise ValueError(f"{shade_param} must be positive to use logarithmic shading")
        shade_norm = mpl.colors.LogNorm(
            vmin=float(np.min(positive_shade_values)),
            vmax=float(np.max(positive_shade_values)),
        )

    y_limits = _resolve_y_limits(y_values, config.y_limits)
    if y_limits_override is not None:
        if y_limits_override[0] >= y_limits_override[1]:
            raise ValueError("y_limits_override must be strictly increasing")
        y_limits = y_limits_override
    if x_col == "step":
        x_ticks = [0, 10_000, 20_000, 30_000, 40_000, 50_000]
    else:
        x_ticks = _select_linear_ticks(x_data_max)
    x_min, x_max = _linear_limits_from_ticks(x_ticks)
    x_tick_formatter = x_tick_formatter or (lambda value: str(int(round(value))))
    y_ticks = list(y_major_ticks) if y_major_ticks is not None else _select_range_ticks(*y_limits)
    if y_limits_override is not None:
        y_axis_limits = y_limits_override
    elif tick_edge_padding_fraction is not None:
        y_axis_limits = _linear_limits_from_ticks(
            y_ticks, edge_fraction=tick_edge_padding_fraction
        )
    else:
        y_axis_limits = y_limits
    if tick_edge_padding_fraction is not None:
        x_min, x_max = _linear_limits_from_ticks(x_ticks, edge_fraction=tick_edge_padding_fraction)
    context_handles = _build_context_handles(color_values)

    for color_index, color_value in enumerate(color_values):
        ax = axes[color_index]
        filtered_df = df.filter(pl.col(color_param) == color_value)

        if shade_param is None or continuous_shade:
            grouped_values = [None]
        else:
            grouped_values = series_unique_sorted(filtered_df, shade_param)

        cmap = plt.colormaps[COLOR_SHADES[color_index % len(COLOR_SHADES)]]
        tinted_cmap = _truncated_colormap(
            COLOR_SHADES[color_index % len(COLOR_SHADES)],
            start=shade_start,
            stop=1.0,
        )
        colors = [cmap(value) for value in np.linspace(shade_start, 1, len(grouped_values))]

        for shade_index, shade_value in enumerate(grouped_values):
            df_subset = filtered_df
            if shade_param is not None and not continuous_shade:
                df_subset = df_subset.filter(pl.col(shade_param) == shade_value)

            for run_hash in series_unique_sorted(df_subset, "run_hash"):
                run_df = (
                    df_subset.filter(pl.col("run_hash") == run_hash)
                    .sort(x_col)
                )
                if x_col == "step":
                    run_df = run_df.group_by("step").agg(pl.col("value").mean().alias("value")).sort("step")
                else:
                    run_df = (
                        run_df.group_by("step")
                        .agg(
                            [
                                pl.col(x_col).first().alias(x_col),
                                pl.col("value").mean().alias("value"),
                            ]
                        )
                        .sort(x_col)
                    )
                x = run_df[x_col].to_numpy()
                y = run_df["value"].to_numpy()
                if len(y) == 0:
                    continue

                color = colors[shade_index]
                if continuous_shade and shade_param is not None and shade_norm is not None:
                    shade_value_for_run = (
                        df_subset.filter(pl.col("run_hash") == run_hash)
                        .select(pl.col(shade_param).first())
                        .to_series()
                        .item(0)
                    )
                    normalized = float(shade_norm(float(shade_value_for_run)))
                    color = cmap(shade_start + (1.0 - shade_start) * normalized)

                ax.plot(
                    x,
                    exponential_moving_average(y, alpha=config.ema_alpha),
                    color=color,
                    linewidth=line_width,
                    zorder=3,
                )

        ax.set_title(_make_subplot_title(color_param, color_value), fontsize=FONT_SIZE_LABELS, y=1.02)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*y_axis_limits)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([x_tick_formatter(tick) for tick in x_ticks])
        ax.set_yticklabels([_format_loss_tick(tick) for tick in y_ticks])
        ax.tick_params(axis="x", rotation=x_tick_rotation)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right" if x_tick_rotation else "center")
        ax.margins(x=0.02, y=0.03)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="#b3b3b3", linestyle='-', linewidth=0.4, alpha=0.8, zorder=1)
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

        if continuous_shade and shade_param is not None and shade_norm is not None:
            cax = ax.inset_axes(
                [compact_colorbar_x if compact_legend else 0.79, 0.58, 0.05, 0.32]
            )
            sm = mpl.cm.ScalarMappable(norm=shade_norm, cmap=tinted_cmap)
            colorbar = fig.colorbar(sm, cax=cax)
            tick_values = np.geomspace(shade_norm.vmin, shade_norm.vmax, 4)
            colorbar.set_ticks(tick_values)
            colorbar.set_ticklabels(
                [_format_shade_value(shade_param, value) for value in tick_values]
            )
            colorbar.ax.minorticks_off()
            colorbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
            colorbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            colorbar.ax.tick_params(labelsize=max(FONT_SIZE_TICKS - 2, 8), length=0)
            colorbar.outline.set_edgecolor("#000000")
            colorbar.outline.set_linewidth(0.9)

    for ax in axes[n_panels:]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*y_axis_limits)
        ax.grid(True)
        remove_bounding_box(ax, fontsize_ticks=FONT_SIZE_TICKS)

    for index, ax in enumerate(axes[:n_panels]):
        col = index % ncols
        ax.set_box_aspect(panel_box_aspect)
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABELS)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=(col == 0))
        if col == 0:
            ax.set_ylabel(config.ylabel, fontsize=FONT_SIZE_LABELS)

    if len(axes) > n_panels:
        legend_ax = axes[n_panels]
        legend_ax.set_box_aspect(panel_box_aspect)
        legend_ax.set_axis_off()
        legend_ax.set_xlabel("")
        legend_ax.set_ylabel("")
        legend_ax.grid(False)

        # Follow the compact scaling-legend layout: one shared vertical extent
        # and a single horizontal gap between the two inset boxes.
        left_x = 0.00 if compact_legend else 0.02
        bottom_y = -0.275 if compact_legend else 0.10
        total_h = 1.45 if compact_legend else 0.80
        left_w = 0.44
        gap = 0.04 if compact_legend else 0.10
        right_x = left_x + left_w + gap
        right_w = 0.52 if compact_legend else 0.40

        context_bounds = [left_x, bottom_y, left_w, total_h]
        context_box = legend_ax.inset_axes(context_bounds)
        context_box.set_xticks([])
        context_box.set_yticks([])
        _style_panel_box(context_box, rounded=True, draw_box=not compact_legend)
        if compact_legend:
            _add_isotropic_rounded_frame(legend_ax, tuple(context_bounds))
        context_box.legend(
            handles=context_handles,
            loc="center",
            bbox_to_anchor=(0.56, 0.42) if compact_legend else None,
            title=None if compact_legend else format_param_name(color_param),
            fontsize=FONT_SIZE_LEGEND - 3,
            title_fontsize=FONT_SIZE_LEGEND - 2 if compact_legend else FONT_SIZE_LEGEND,
            handlelength=2.1,
            borderaxespad=0.0,
            frameon=False,
            labelspacing=0.55 if compact_legend else 1.0,
            handletextpad=0.7,
        )
        if compact_legend:
            context_box.text(
                0.56,
                0.84,
                _wrapped_param_name(color_param),
                transform=context_box.transAxes,
                fontsize=FONT_SIZE_LEGEND - 2,
                va="center",
                ha="center",
            )

        if continuous_shade and shade_param is not None and shade_norm is not None:
            shade_bounds = [right_x, bottom_y, right_w, total_h]
            shade_box = legend_ax.inset_axes(shade_bounds)
            shade_box.set_xticks([])
            shade_box.set_yticks([])
            _style_panel_box(shade_box, rounded=True, draw_box=not compact_legend)
            if compact_legend:
                _add_isotropic_rounded_frame(legend_ax, tuple(shade_bounds))
            shade_box.text(
                compact_shade_title_x if compact_legend else 0.5,
                0.84,
                _wrapped_param_name(shade_param),
                transform=shade_box.transAxes,
                fontsize=FONT_SIZE_LEGEND - (2 if compact_legend else 3),
                va="center",
                ha="center",
            )
            cax_bounds = [0.31, 0.12, 0.14, 0.57] if compact_legend else [0.19, 0.18, 0.22, 0.52]
            cax = shade_box.inset_axes(cax_bounds)
            neutral_cmap = _truncated_colormap("Greys", start=shade_start, stop=1.0)
            sm = mpl.cm.ScalarMappable(norm=shade_norm, cmap=neutral_cmap)
            colorbar = fig.colorbar(sm, cax=cax, orientation="vertical")
            if compact_legend and colorbar.solids is not None:
                colorbar.solids.set_edgecolor("face")
                colorbar.solids.set_linewidth(0)
            tick_values = np.geomspace(shade_norm.vmin, shade_norm.vmax, 4)
            colorbar.set_ticks(tick_values)
            colorbar.set_ticklabels(
                [_format_shade_value(shade_param, value) for value in tick_values]
            )
            colorbar.ax.minorticks_off()
            colorbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
            colorbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            colorbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, length=0)
            colorbar.outline.set_edgecolor("#000000")
            colorbar.outline.set_linewidth(0.9)
            if compact_legend:
                shade_box.add_patch(
                    mpatches.Rectangle(
                        (0.31, 0.12),
                        0.14,
                        0.57,
                        transform=shade_box.transAxes,
                        fill=False,
                        edgecolor="#000000",
                        linewidth=0.9,
                        clip_on=False,
                        zorder=10,
                    )
                )

    if show_title:
        fig.suptitle(config.title, fontsize=FONT_SIZE_TITLE, x=0.53, y=0.955, ha="center")
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=0.93 if show_title else 0.98,
        bottom=0.08,
        wspace=wspace,
        hspace=hspace,
    )

    if not continuous_shade and shade_param is not None:
        raise NotImplementedError(
            "Faceted training-loss plots currently require continuous shading."
        )

    save_figure(
        fig,
        output_path,
        enforce_y_axis_units=enforce_y_axis_units,
    )
