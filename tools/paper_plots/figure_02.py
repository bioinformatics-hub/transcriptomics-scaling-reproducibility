from __future__ import annotations

import argparse
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from fontTools.ttLib import TTCollection
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from plotting.scgpt_batch import _select_most_complete_runs
from plotting.utils import (
    COLOR_SHADES,
    PLOTS_DIR,
    exponential_moving_average,
    save_figure,
    series_unique_sorted,
)
from tools.paper_plots.data import paper_data
from tools.paper_plots.style import (
    PAPER_FONT_FAMILY,
    PAPER_MAX_FONT_SIZE_PT,
    despine,
    use_paper_style,
    validate_paper_dimensions,
    validate_paper_svg,
)


PAPER_WIDTH_MM = 180.0
PAPER_HEIGHT_MM = 95.0
DEFAULT_OUTPUT = PLOTS_DIR / "paper" / "02_batch_size_scaling.svg"

GENEFORMER_DATA = paper_data("geneformer_batch/01_training_loss_metrics.csv")
SCGPT_DATA = paper_data("scgpt_batch/01_training_loss_metrics.csv")

PARAM_COLOR = "datamodule.batch_size"
PARAM_SHADE = "model.d_model"
EMA_ALPHA = 0.05
GENEFORMER_Y_LIMITS = (4.2, 10.8)
SCGPT_Y_LIMITS = (160.0, 380.0)
GENEFORMER_LINE_WIDTH = 0.8
SCGPT_LINE_WIDTH = 0.4
BATCH_CMAP_OVERRIDES = {128: "wong_black"}


@contextmanager
def _paper_bold_font():
    normal_properties = FontProperties(
        family=PAPER_FONT_FAMILY,
        weight="normal",
    )
    bold_properties = FontProperties(
        family=PAPER_FONT_FAMILY,
        size=7,
        weight="bold",
    )
    normal_path = Path(
        font_manager.findfont(normal_properties, fallback_to_default=False)
    )
    bold_path = Path(
        font_manager.findfont(bold_properties, fallback_to_default=False)
    )
    if bold_path != normal_path or bold_path.suffix.lower() != ".ttc":
        yield bold_properties
        return

    collection = TTCollection(bold_path)
    try:
        bold_face = next(
            (
                font
                for font in collection.fonts
                if font["name"].getDebugName(2) == "Bold"
            ),
            None,
        )
        if bold_face is None:
            raise FileNotFoundError(f"No Bold face found in {bold_path}")
        with tempfile.TemporaryDirectory(prefix="paper_arial_bold_") as directory:
            extracted_path = Path(directory) / "Arial-Bold.ttf"
            bold_face.save(extracted_path)
            font_manager.fontManager.addfont(extracted_path)
            yield FontProperties(fname=extracted_path, size=7, weight="bold")
    finally:
        collection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper Figure 02 by combining the Geneformer and scGPT "
            "batch-size training-loss sweeps."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--geneformer-data",
        type=Path,
        default=GENEFORMER_DATA,
        help="Cached Geneformer batch-sweep metrics CSV.",
    )
    parser.add_argument(
        "--scgpt-data",
        type=Path,
        default=SCGPT_DATA,
        help="Cached scGPT batch-sweep metrics CSV.",
    )
    return parser.parse_args()


def _load_metrics(path: Path, *, expected_metric: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached batch-sweep metrics: {path}. Run the corresponding "
            "plotting/*_batch.py script first."
        )
    frame = (
        pl.read_parquet(path) if path.suffix == ".parquet"
        else pl.read_csv(path, infer_schema_length=10_000)
    )
    required = {"run_hash", "metric_name", "step", "value", PARAM_COLOR, PARAM_SHADE}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    metrics = set(frame["metric_name"].unique().to_list())
    if metrics != {expected_metric}:
        raise ValueError(
            f"Expected only metric={expected_metric!r} in {path}; found {sorted(metrics)}"
        )
    return _select_most_complete_runs(frame)


def _shade_mapping(frame: pl.DataFrame) -> dict[object, float]:
    embedding_dims = series_unique_sorted(frame, PARAM_SHADE)
    return dict(
        zip(
            embedding_dims,
            np.linspace(0.3, 1.0, len(embedding_dims)),
            strict=True,
        )
    )


def _plot_sweep(
    ax: plt.Axes,
    frame: pl.DataFrame,
    *,
    line_width: float,
) -> None:
    batch_sizes = series_unique_sorted(frame, PARAM_COLOR)
    shade_by_embedding = _shade_mapping(frame)

    for batch_index, batch_size in enumerate(batch_sizes):
        batch_frame = frame.filter(pl.col(PARAM_COLOR) == batch_size)
        cmap_name = BATCH_CMAP_OVERRIDES.get(
            int(batch_size), COLOR_SHADES[batch_index]
        )
        cmap = plt.colormaps[cmap_name]
        for run_hash in series_unique_sorted(batch_frame, "run_hash"):
            run_frame = (
                batch_frame.filter(pl.col("run_hash") == run_hash)
                .group_by("step")
                .agg(
                    pl.col("value").mean().alias("value"),
                    pl.col(PARAM_SHADE).first().alias(PARAM_SHADE),
                )
                .sort("step")
            )
            embedding_dim = run_frame[PARAM_SHADE][0]
            ax.plot(
                run_frame["step"].to_numpy(),
                exponential_moving_average(
                    run_frame["value"].to_numpy(), alpha=EMA_ALPHA
                ),
                color=cmap(shade_by_embedding[embedding_dim]),
                linewidth=line_width,
                zorder=3,
            )


def _format_step(value: float, _: float) -> str:
    if value == 0:
        return "0"
    return f"{value / 1_000:g}k"


def _format_axis(
    ax: plt.Axes,
    *,
    title: str,
    title_font: FontProperties,
    ylabel: str,
    y_limits: tuple[float, float],
    y_ticks: tuple[float, ...],
) -> None:
    ax.set_xlim(-1_000, 16_000)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(mticker.FixedLocator((0, 5_000, 10_000, 15_000)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_step))
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax.set_title(title, fontproperties=title_font, pad=5)
    ax.set_xlabel("Training Steps", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.grid(
        True,
        which="major",
        color="#b3b3b3",
        linestyle="-",
        linewidth=0.3,
        alpha=0.58,
        zorder=1,
    )
    despine(ax, tick_labelsize=6.5)


def _legend_handles(
    frame: pl.DataFrame,
) -> tuple[list[mlines.Line2D], list[mlines.Line2D]]:
    batch_sizes = series_unique_sorted(frame, PARAM_COLOR)
    batch_handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps[
                BATCH_CMAP_OVERRIDES.get(int(batch_size), COLOR_SHADES[index])
            ](0.7),
            linewidth=1.35,
            label=str(int(batch_size)),
        )
        for index, batch_size in enumerate(batch_sizes)
    ]
    embedding_dims = series_unique_sorted(frame, PARAM_SHADE)
    grey_values = np.linspace(0.3, 1.0, len(embedding_dims))
    embedding_handles = [
        mlines.Line2D(
            [],
            [],
            color=plt.colormaps["Greys"](grey_values[index]),
            linewidth=1.35,
            label=str(int(embedding_dim)),
        )
        for index, embedding_dim in enumerate(embedding_dims)
    ]
    return batch_handles, embedding_handles


def _style_legend_box(legend: plt.Legend) -> None:
    frame = legend.get_frame()
    frame.set_linewidth(0.6)
    frame.set_edgecolor("#2f2a24")
    frame.set_facecolor("white")
    frame.set_alpha(0.9)


def build_figure(
    output_path: Path,
    *,
    geneformer_data: Path = GENEFORMER_DATA,
    scgpt_data: Path = SCGPT_DATA,
) -> Path:
    validate_paper_dimensions(PAPER_WIDTH_MM, PAPER_HEIGHT_MM)
    geneformer = _load_metrics(geneformer_data, expected_metric="bce")
    scgpt = _load_metrics(scgpt_data, expected_metric="mse")

    geneformer_batches = series_unique_sorted(geneformer, PARAM_COLOR)
    scgpt_batches = series_unique_sorted(scgpt, PARAM_COLOR)
    geneformer_dims = series_unique_sorted(geneformer, PARAM_SHADE)
    scgpt_dims = series_unique_sorted(scgpt, PARAM_SHADE)
    if geneformer_batches != scgpt_batches or geneformer_dims != scgpt_dims:
        raise ValueError(
            "Geneformer and scGPT batch sweeps must use matching batch sizes and "
            "embedding dimensions"
        )

    use_paper_style()
    with _paper_bold_font() as bold_font:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(PAPER_WIDTH_MM / 25.4, PAPER_HEIGHT_MM / 25.4),
            squeeze=False,
        )
        geneformer_ax, scgpt_ax = axes[0]

        _plot_sweep(
            geneformer_ax,
            geneformer,
            line_width=GENEFORMER_LINE_WIDTH,
        )
        _format_axis(
            geneformer_ax,
            title="Training Performance on Ranked Gene Identity",
            title_font=bold_font,
            ylabel="Cross-entropy loss",
            y_limits=GENEFORMER_Y_LIMITS,
            y_ticks=(5, 6, 7, 8, 9, 10),
        )
        _plot_sweep(scgpt_ax, scgpt, line_width=SCGPT_LINE_WIDTH)
        _format_axis(
            scgpt_ax,
            title="Training Performance on Binned Gene Expression",
            title_font=bold_font,
            ylabel="Mean squared error",
            y_limits=SCGPT_Y_LIMITS,
            y_ticks=(200, 250, 300, 350),
        )

        for label, ax in zip("AB", axes[0], strict=True):
            ax.text(
                -0.11,
                1.08,
                label,
                transform=ax.transAxes,
                fontproperties=bold_font,
                ha="left",
                va="top",
            )

        fig.subplots_adjust(
            left=0.070,
            right=0.985,
            top=0.900,
            bottom=0.225,
            wspace=0.25,
        )
        panel_centers = [
            (ax.get_position().x0 + ax.get_position().x1) / 2 for ax in axes[0]
        ]

        batch_handles, embedding_handles = _legend_handles(geneformer)
        batch_legend = fig.legend(
            handles=batch_handles,
            title="Batch Size",
            loc="lower center",
            bbox_to_anchor=(panel_centers[0], 0.015),
            ncol=len(batch_handles),
            frameon=True,
            fancybox=True,
            fontsize=6.5,
            title_fontsize=7,
            handlelength=2.0,
            columnspacing=1.6,
        )
        _style_legend_box(batch_legend)
        embedding_legend = fig.legend(
            handles=embedding_handles,
            title="Embedding Dimension",
            loc="lower center",
            bbox_to_anchor=(panel_centers[1], 0.015),
            ncol=len(embedding_handles),
            frameon=True,
            fancybox=True,
            fontsize=6.5,
            title_fontsize=7,
            handlelength=2.0,
            columnspacing=1.6,
        )
        _style_legend_box(embedding_legend)

        save_figure(
            fig,
            output_path,
            apply_default_adjust=False,
            crop=False,
            max_font_size_pt=PAPER_MAX_FONT_SIZE_PT,
            max_size_mm=(PAPER_WIDTH_MM, PAPER_HEIGHT_MM),
            enforce_y_axis_units=False,
        )
    validate_paper_svg(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    svg_path = build_figure(
        args.output,
        geneformer_data=args.geneformer_data,
        scgpt_data=args.scgpt_data,
    )
    print(f"Wrote {svg_path} ({PAPER_WIDTH_MM:g} mm × {PAPER_HEIGHT_MM:g} mm)")


if __name__ == "__main__":
    main()
