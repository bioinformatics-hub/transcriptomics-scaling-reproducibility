from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mpl-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import polars as pl
from matplotlib import font_manager


ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "plots"
FONT_NAME = "Arial"
MAX_FIGURE_WIDTH_MM = 180.0
MAX_FIGURE_HEIGHT_MM = 210.0
MIN_FONT_SIZE_PT = 5.0

WONG_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)
WONG_CMAP_NAMES = (
    "wong_blue",
    "wong_vermillion",
    "wong_green",
    "wong_purple",
    "wong_orange",
    "wong_sky_blue",
    "wong_black",
)
TOL_MUTED_PALETTE = (
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
)
TOL_MUTED_CMAP_NAMES = (
    "tol_muted_rose",
    "tol_muted_indigo",
    "tol_muted_sand",
    "tol_muted_green",
    "tol_muted_cyan",
    "tol_muted_wine",
    "tol_muted_teal",
    "tol_muted_olive",
    "tol_muted_purple",
)


def _register_accessible_colormaps() -> None:
    palettes = (
        (WONG_CMAP_NAMES, WONG_PALETTE),
        (TOL_MUTED_CMAP_NAMES, TOL_MUTED_PALETTE),
    )
    for names, colors in palettes:
        for name, color in zip(names, colors, strict=True):
            if name in matplotlib.colormaps:
                continue
            rgb = np.asarray(matplotlib.colors.to_rgb(color))
            light_rgb = 0.72 * np.ones(3) + 0.28 * rgb
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                name,
                (light_rgb, rgb),
            )
            matplotlib.colormaps.register(cmap)


_register_accessible_colormaps()
COLOR_SHADES = list(WONG_CMAP_NAMES)
FONT_SIZE_TICKS = 13
FONT_SIZE_LABELS = 24
FONT_SIZE_TITLE = 30
FONT_SIZE_LEGEND = 18
GRID_COLOR = "#d8d2c4"
TEXT_COLOR = "#2f2a24"

_LABEL_WITH_UNITS_RE = re.compile(
    r"\((?:steps?|FLOPs|parameters?|layers?|features?|nats?|dex|replicates?|"
    r"architectures?|observations?|dimensionless|model-specific units|"
    r"squared expression units)\)\s*$",
    flags=re.IGNORECASE,
)
_AXIS_LABELS_WITH_UNITS = {
    "training steps": "Training progress (steps)",
    "training step": "Training progress (steps)",
    "cumulative training flops": "Cumulative training compute (FLOPs)",
    "training flops": "Training compute (FLOPs)",
    "non-embedding parameters": "Non-embedding model size (parameters)",
    "target non-embedding parameters": "Target non-embedding model size (parameters)",
    "total parameters": "Total model size (parameters)",
    "parameters": "Model size (parameters)",
    "target depth": "Target depth (layers)",
    "depth": "Model depth (layers)",
    "depth (d)": "Model depth, D (layers)",
    "depth at matched parameter target": "Model depth at matched size (layers)",
    "width at matched parameter target": "Model width at matched size (features)",
    "learning rate": "Learning rate (dimensionless)",
    "best learning rate": "Optimal learning rate (dimensionless)",
    "predicted best lr": "Predicted optimal learning rate (dimensionless)",
    "observed best lr": "Observed optimal learning rate (dimensionless)",
    "cross entropy": "Cross-entropy loss (nats)",
    "binary cross entropy": "Binary cross-entropy loss (nats)",
    "mean squared error": "Mean squared error (squared expression units)",
    "mse": "Mean squared error (squared expression units)",
    "observed loss": "Observed loss (model-specific units)",
    "fitted loss": "Fitted loss (model-specific units)",
    "predicted loss": "Predicted loss (model-specific units)",
    "residual": "Loss residual (model-specific units)",
    "depth / width": "Depth-to-width ratio (dimensionless)",
    "winning depth / width": "Winning depth-to-width ratio (dimensionless)",
    "predicted optimal depth / width": (
        "Predicted optimal depth-to-width ratio (dimensionless)"
    ),
    "bioscore": "BIOscore (dimensionless)",
    "score": "Score (dimensionless)",
    "exponent": "Scaling exponent (dimensionless)",
    "bootstrap count": "Bootstrap count (replicates)",
    "architecture points": "Architecture count (architectures)",
}


def axis_label_with_unit(label: str) -> str:
    """Return a publication axis label with an explicit parenthesized unit."""
    stripped = label.strip()
    if not stripped or _LABEL_WITH_UNITS_RE.search(stripped):
        return label

    mapped = _AXIS_LABELS_WITH_UNITS.get(stripped.casefold())
    if mapped is not None:
        return mapped

    lower = stripped.casefold()
    if "log10" in lower and ("lr" in lower or "learning rate" in lower):
        return f"{stripped} (dex)"
    if "flop" in lower:
        return f"{stripped} (FLOPs)"
    if "parameter" in lower:
        return f"{stripped} (parameters)"
    if "step" in lower:
        return f"{stripped} (steps)"
    if "learning rate" in lower or re.search(r"\blr\b", lower):
        return f"{stripped} (dimensionless)"
    if "loss" in lower:
        return f"{stripped} (model-specific units)"
    if "depth" in lower and not any(
        word in lower for word in ("ratio", "/", "exponent")
    ):
        return f"{stripped} (layers)"
    if "width" in lower and not any(
        word in lower for word in ("ratio", "/", "exponent")
    ):
        return f"{stripped} (features)"
    if any(
        word in lower
        for word in (
            "score",
            "accuracy",
            "nmi",
            "ari",
            "homogeneity",
            "connectivity",
            "asw",
            "ratio",
            "exponent",
            "fraction",
        )
    ):
        return f"{stripped} (dimensionless)"
    return f"{stripped} (arbitrary units)"


def flatten_dict(data: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def exponential_moving_average(data: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for index in range(1, len(data)):
        ema[index] = alpha * data[index] + (1.0 - alpha) * ema[index - 1]
    return ema


def remove_bounding_box(ax: plt.Axes, fontsize_ticks: int, remove: bool = True) -> None:
    if not remove:
        return
    style_axis_lines_and_ticks(ax, tick_labelsize=fontsize_ticks)


def style_axis_lines_and_ticks(
    ax: plt.Axes,
    *,
    tick_labelsize: float | None = None,
) -> None:
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    common = {
        "direction": "out",
        "width": 0.6,
        "color": TEXT_COLOR,
        "top": False,
        "right": False,
        "bottom": True,
        "left": True,
    }
    major = {"length": 2.5, **common}
    if tick_labelsize is not None:
        major["labelsize"] = tick_labelsize
    ax.tick_params(axis="both", which="major", **major)
    ax.tick_params(axis="both", which="minor", length=1.5, **common)


def register_plot_font() -> str:
    try:
        font_path = Path(font_manager.findfont(FONT_NAME, fallback_to_default=False))
    except ValueError as error:
        raise FileNotFoundError(
            f"The {FONT_NAME} font is required to render plots. Install it and "
            "refresh Matplotlib's font cache before rebuilding figures."
        ) from error

    font_manager.fontManager.addfont(str(font_path))
    return font_manager.FontProperties(fname=str(font_path)).get_name()


def apply_plot_style() -> None:
    font_name = register_plot_font()
    plt.rcParams.update(
        {
            "agg.path.chunksize": 200,
            "font.family": font_name,
            # Keep SVG labels as editable text and embed TrueType fonts in
            # publication PDF/PS exports (font type 42).
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.fallback": None,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.prop_cycle": matplotlib.cycler(color=WONG_PALETTE),
            "axes.linewidth": 0.6,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.edgecolor": TEXT_COLOR,
            "legend.facecolor": "white",
            "axes.grid": True,  # Turns the grid on globally
            "axes.axisbelow": True,  # Forces grid behind data (fixes the z-order issue globally)
            "grid.color": "#e0e0e0",  # Sets a soft grey so it doesn't clash with TEXT_COLOR
            "grid.linestyle": "--",  # Optional: makes it dashed
            "grid.alpha": 0.7,  # Optional: softens the line
        }
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _enforce_font_size_bounds(
    fig: plt.Figure, *, max_font_size_pt: float | None
) -> None:
    if max_font_size_pt is not None and max_font_size_pt < MIN_FONT_SIZE_PT:
        raise ValueError(
            f"Maximum font size ({max_font_size_pt:g} pt) cannot be smaller than "
            f"the minimum ({MIN_FONT_SIZE_PT:g} pt)"
        )
    for text in fig.findobj(match=mtext.Text):
        font_size = max(text.get_fontsize(), MIN_FONT_SIZE_PT)
        if max_font_size_pt is not None:
            font_size = min(font_size, max_font_size_pt)
        text.set_fontsize(font_size)


def _enforce_editable_arial_text(fig: plt.Figure) -> None:
    register_plot_font()
    for text in fig.findobj(match=mtext.Text):
        text.set_fontfamily(FONT_NAME)


def _enforce_axis_guidelines(
    fig: plt.Figure,
    *,
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> None:
    for ax in fig.axes:
        if not ax.axison or not ax.get_visible():
            continue
        if (
            len(ax.get_xticks()) == 0
            and len(ax.get_yticks()) == 0
            and not ax.get_xlabel()
            and not ax.get_ylabel()
        ):
            continue
        if enforce_x_axis_units and ax.get_xlabel():
            ax.set_xlabel(axis_label_with_unit(ax.get_xlabel()))
        if enforce_y_axis_units and ax.get_ylabel():
            ax.set_ylabel(axis_label_with_unit(ax.get_ylabel()))
        style_axis_lines_and_ticks(ax)


def _constrain_figure_size(
    fig: plt.Figure, *, max_size_mm: tuple[float, float] | None
) -> None:
    if max_size_mm is None:
        return
    max_width_mm, max_height_mm = max_size_mm
    width_in, height_in = fig.get_size_inches()
    scale = min(
        1.0,
        max_width_mm / 25.4 / width_in,
        max_height_mm / 25.4 / height_in,
    )
    if scale < 1.0:
        fig.set_size_inches(width_in * scale, height_in * scale, forward=True)


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    apply_default_adjust: bool = True,
    crop: bool = True,
    max_font_size_pt: float | None = None,
    max_size_mm: tuple[float, float] | None = None,
    enforce_x_axis_units: bool = True,
    enforce_y_axis_units: bool = True,
) -> None:
    ensure_parent_dir(output_path)
    _constrain_figure_size(fig, max_size_mm=max_size_mm)
    _enforce_axis_guidelines(
        fig,
        enforce_x_axis_units=enforce_x_axis_units,
        enforce_y_axis_units=enforce_y_axis_units,
    )
    _enforce_font_size_bounds(fig, max_font_size_pt=max_font_size_pt)
    _enforce_editable_arial_text(fig)
    if apply_default_adjust:
        fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.13)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.08} if crop else {}
    with matplotlib.rc_context(
        {
            "font.family": FONT_NAME,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.fallback": None,
        }
    ):
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def format_param_name(name: str) -> str:
    label_map = {
        "datamodule.batch_size": "Batch Size",
        "model.d_model": "Embedding Dimension",
        "model.context_length": "Context Length",
        "model.transformer.n_layers": "Layers",
        "model.bioformer.n_layers": "Layers",
        "trainer.accumulate_grad": "Gradient Accumulation",
        "non_embedding_params": "Non-embedding Parameters",
    }
    return label_map.get(name, name.split(".")[-1].replace("_", " ").title())


def format_param_value(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def series_unique_sorted(df: pl.DataFrame, column: str) -> list[object]:
    return df.select(column).unique().drop_nulls().to_series().sort().to_list()


def build_legend_handles(
    df: pl.DataFrame,
    color_param: str,
    shade_param: str | None,
) -> tuple[list[object], list[mlines.Line2D], list[mlines.Line2D]]:
    color_values = series_unique_sorted(df, color_param)
    color_handles = []
    for index, value in enumerate(color_values):
        cmap_name = COLOR_SHADES[index % len(COLOR_SHADES)]
        color = plt.colormaps[cmap_name](0.7)
        color_handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=3,
                label=f"{format_param_name(color_param)}: {format_param_value(value)}",
            )
        )

    shade_handles: list[mlines.Line2D] = []
    if shade_param:
        shade_values = series_unique_sorted(df, shade_param)
        cmap_greys = plt.colormaps["Greys"]
        for index, value in enumerate(shade_values):
            shade = cmap_greys(np.linspace(0.3, 1, len(shade_values))[index])
            shade_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=shade,
                    linewidth=3,
                    label=f"{format_param_name(shade_param)}: {format_param_value(value)}",
                )
            )
    return color_values, color_handles, shade_handles
