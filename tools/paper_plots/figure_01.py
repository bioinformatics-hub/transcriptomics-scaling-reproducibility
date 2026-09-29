"""Build the 80 mm-wide pipeline and model-structure schematic."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Rectangle,
)

from plotting.utils import PLOTS_DIR, ROOT_DIR
from tools.paper_plots.style import (
    PAPER_FONT_FAMILY,
    PAPER_MAX_FONT_SIZE_PT,
    PAPER_SCHEMATIC_WIDTH_MM,
    use_paper_style,
    validate_paper_dimensions,
    validate_paper_svg,
)


PAPER_WIDTH_MM = PAPER_SCHEMATIC_WIDTH_MM
PAPER_HEIGHT_MM = 85.0
DRAWING_WIDTH = 80.0
DRAWING_HEIGHT = 110.0
FONT_SIZE_PT = 5.0
PANEL_LABEL_SIZE_PT = PAPER_MAX_FONT_SIZE_PT
DEFAULT_OUTPUT = PLOTS_DIR / "paper" / "01" / "01_schematic.svg"
DEFAULT_PDF_OUTPUT = ROOT_DIR / "manuscript" / "figures" / "01_schematic.pdf"

BLACK = "#202020"
GOLD = "#F0E442"
ORANGE = "#E69F00"
LAVENDER = "#CC79A7"
BLUE = "#56B4E9"
GREY = "#929292"
GREEN = "#009E73"
GREEN_FILL = "#D9F0E8"
RED = "#D55E00"
RED_FILL = "#FBE5D9"
DASHED_ARROW_LINESTYLE = (0, (5, 4))


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.y + self.height / 2

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.width, self.y + self.height / 2

    @property
    def top(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height

    @property
    def bottom(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper Figure 01 as editable SVG and embedded-font PDF."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    return parser.parse_args()


def _text(
    ax: Axes,
    x: float,
    y: float,
    value: str,
    *,
    ha: str = "center",
    va: str = "center",
    color: str = BLACK,
    weight: str = "normal",
    rotation: float = 0,
    size: float = FONT_SIZE_PT,
    zorder: float = 5,
) -> None:
    ax.text(
        x,
        y,
        value,
        ha=ha,
        va=va,
        color=color,
        weight=weight,
        rotation=rotation,
        fontsize=size,
        fontfamily=PAPER_FONT_FAMILY,
        linespacing=1.0,
        zorder=zorder,
    )


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
    rounded: bool = False,
) -> Box:
    if rounded:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=1.1",
            facecolor=facecolor,
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=3,
        )
    else:
        patch = Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=3,
        )
    ax.add_patch(patch)
    _text(ax, x + width / 2, y + height / 2, label)
    return Box(x, y, width, height)


def _region(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edgecolor: str,
    facecolor: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=1.2",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.5,
            linestyle=(0, (5, 4)),
            zorder=1,
        )
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLACK,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=5.0,
            linewidth=0.5,
            linestyle=DASHED_ARROW_LINESTYLE if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def _path(
    ax: Axes,
    points: list[tuple[float, float]],
    *,
    arrow: bool = True,
    color: str = BLACK,
    dashed: bool = False,
    zorder: float = 2,
) -> None:
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    if arrow:
        patch = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=5.0,
            linewidth=0.5,
            linestyle=DASHED_ARROW_LINESTYLE if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    else:
        patch = PathPatch(
            path,
            fill=False,
            linewidth=0.5,
            linestyle=DASHED_ARROW_LINESTYLE if dashed else "-",
            edgecolor=color,
            zorder=zorder,
        )
    ax.add_patch(patch)


def _draw_pipeline(ax: Axes) -> None:
    _text(
        ax,
        1.0,
        108.0,
        "A",
        ha="left",
        va="top",
        weight="bold",
        size=PANEL_LABEL_SIZE_PT,
    )

    census = _box(
        ax, 2.25, 79.75, 14.0, 8.5, "CELLxGENE\nCensus", facecolor=GOLD
    )
    data = _box(
        ax, 19.25, 80.9, 9.0, 6.2, "Data\nModule", facecolor=ORANGE, rounded=True
    )
    expression = _box(
        ax, 32.0, 98.9, 12.5, 7.2, "Gene\nExpression", facecolor=GOLD
    )
    mask = _box(ax, 32.0, 81.5, 12.5, 6.5, "Mask", facecolor=GREY)
    identity = _box(
        ax, 32.0, 61.4, 12.5, 7.2, "Gene\nIdentity", facecolor=GOLD
    )
    masked_expression = _box(
        ax, 47.75, 88.75, 13.5, 8.0, "Masked\nGene Expression", facecolor=GOLD
    )
    masked_identity = _box(
        ax, 47.75, 73.25, 13.5, 8.0, "(Masked)\nGene Identity", facecolor=GOLD
    )
    model = _box(
        ax, 64.25, 81.4, 8.5, 6.2, "Model", facecolor=BLUE, rounded=True
    )

    _text(ax, 9.25, 90.0, "61k genes")
    _text(ax, 0.9, 84.0, "96M cells", rotation=90)
    loss_x = 77.4
    _text(ax, loss_x, 84.5, "Loss")

    _arrow(ax, census.right, data.left)
    _path(ax, [data.right, (29.8, data.right[1]), (29.8, expression.left[1]), expression.left])
    _path(ax, [data.right, (29.8, data.right[1]), (29.8, identity.left[1]), identity.left])
    _path(
        ax,
        [
            expression.bottom,
            (expression.bottom[0], masked_expression.y + 6.0),
            (masked_expression.x, masked_expression.y + 6.0),
        ],
    )
    _path(
        ax,
        [
            mask.right,
            (46.5, mask.right[1]),
            (46.5, masked_expression.y + 2.0),
            (masked_expression.x, masked_expression.y + 2.0),
        ],
    )
    _path(
        ax,
        [
            mask.right,
            (46.5, mask.right[1]),
            (46.5, masked_identity.y + 5.8),
            (masked_identity.x, masked_identity.y + 5.8),
        ],
        color=RED,
        dashed=True,
    )
    _path(
        ax,
        [
            identity.top,
            (identity.top[0], masked_identity.y + 2.0),
            (masked_identity.x, masked_identity.y + 2.0),
        ],
    )
    _path(
        ax,
        [
            masked_expression.right,
            (model.top[0], masked_expression.right[1]),
            model.top,
        ],
    )
    _path(
        ax,
        [
            masked_identity.right,
            (model.bottom[0], masked_identity.right[1]),
            model.bottom,
        ],
    )
    _arrow(ax, model.right, (75.6, model.right[1]))
    _path(
        ax,
        [
            expression.right,
            (loss_x, 102.5),
            (loss_x, 85.7),
        ],
    )
    _path(
        ax,
        [
            identity.right,
            (loss_x, 65.0),
            (loss_x, 83.3),
        ],
    )


def _draw_model(ax: Axes) -> None:
    _text(
        ax,
        1.0,
        59.5,
        "B",
        ha="left",
        va="top",
        weight="bold",
        size=PANEL_LABEL_SIZE_PT,
    )

    _region(ax, 1.7, 47.2, 29.7, 9.3, edgecolor=GREEN, facecolor=GREEN_FILL)
    _region(ax, 47.8, 33.8, 31.5, 22.4, edgecolor=GREEN, facecolor=GREEN_FILL)
    _region(ax, 1.7, 19.7, 29.7, 10.0, edgecolor=RED, facecolor=RED_FILL)
    _region(ax, 47.8, 19.7, 31.5, 10.0, edgecolor=RED, facecolor=RED_FILL)
    _text(ax, 7.0, 58.2, "Gene Expression", ha="left", color=GREEN)
    _text(ax, 77.5, 58.2, "Gene Expression", ha="right", color=GREEN)
    _text(ax, 3.0, 18.2, "Gene Ranking", ha="left", color=RED)
    _text(ax, 77.5, 18.2, "Gene Ranking", ha="right", color=RED)

    masked_expression = _box(
        ax, 2.75, 48.15, 13.0, 7.4, "Masked\nGene Expression", facecolor=GOLD
    )
    expression_ffnn = _box(
        ax, 20.0, 48.65, 9.0, 6.4, "FFNN", facecolor=LAVENDER, rounded=True
    )
    masked_identity = _box(
        ax, 2.75, 34.55, 13.0, 7.4, "(Masked)\nGene Identity", facecolor=GOLD
    )
    embedder = _box(
        ax, 20.0, 35.05, 9.0, 6.4, "Embedder", facecolor=LAVENDER, rounded=True
    )
    backbone = _box(
        ax, 34.1, 35.05, 10.5, 6.4, "Backbone", facecolor=BLUE, rounded=True
    )

    _text(ax, 11.4, 24.65, "Positional Encoding")
    ax.add_patch(
        Ellipse(
            (24.5, 24.65),
            width=4.8,
            height=6.2,
            facecolor="#D55E00",
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=3,
        )
    )
    theta = [index * 0.18 for index in range(36)]
    wave = [
        (22.6 + 3.8 * t / theta[-1], 24.65 + 1.35 * math.sin(t)) for t in theta
    ]
    _path(ax, wave, arrow=False, color=BLACK, zorder=4)

    mse_ffnn = _box(
        ax, 48.5, 48.65, 9.0, 6.4, "FFNN", facecolor=LAVENDER, rounded=True
    )
    bce_ffnn = _box(
        ax, 48.5, 35.05, 9.0, 6.4, "FFNN", facecolor=LAVENDER, rounded=True
    )
    ce_ffnn = _box(
        ax, 48.5, 21.45, 9.0, 6.4, "FFNN", facecolor=LAVENDER, rounded=True
    )
    mse_target = _box(
        ax, 60.2, 48.15, 12.0, 7.4, "Gene\nExpression", facecolor=GOLD
    )
    bce_target = _box(ax, 60.2, 34.55, 12.0, 7.4, "Active", facecolor=GOLD)
    ce_target = _box(
        ax, 60.2, 20.95, 12.0, 7.4, "Gene\nIdentity", facecolor=GOLD
    )

    _arrow(ax, masked_expression.right, expression_ffnn.left)
    _arrow(ax, masked_identity.right, embedder.left)
    _path(
        ax,
        [
            expression_ffnn.right,
            (31.8, expression_ffnn.right[1]),
            (31.8, backbone.left[1]),
            backbone.left,
        ],
    )
    _arrow(ax, embedder.right, backbone.left)
    _path(
        ax,
        [(26.9, 24.65), (31.8, 24.65), (31.8, backbone.left[1]), backbone.left],
    )

    for target in (mse_ffnn, bce_ffnn, ce_ffnn):
        _path(
            ax,
            [
                backbone.right,
                (47.1, backbone.right[1]),
                (47.1, target.left[1]),
                target.left,
            ],
        )
    _arrow(ax, mse_ffnn.right, mse_target.left)
    _arrow(ax, bce_ffnn.right, bce_target.left)
    _arrow(ax, ce_ffnn.right, ce_target.left)

    for target, label in ((mse_target, "MSE"), (bce_target, "BCE"), (ce_target, "CE")):
        _arrow(ax, target.right, (74.5, target.right[1]))
        _text(ax, 78.8, target.right[1], label, ha="right")


def _create_figure() -> Figure:
    validate_paper_dimensions(
        PAPER_WIDTH_MM,
        PAPER_HEIGHT_MM,
        required_width_mm=PAPER_SCHEMATIC_WIDTH_MM,
    )
    use_paper_style()
    fig = plt.figure(
        figsize=(PAPER_WIDTH_MM / 25.4, PAPER_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    ax = fig.add_axes((0, 0, 1, 1))
    # Authored on the original 80-unit grid; the 88 mm page expands the boxes
    # horizontally while point-sized text remains undistorted.
    ax.set_xlim(0, DRAWING_WIDTH)
    ax.set_ylim(10, DRAWING_HEIGHT)
    ax.set_aspect("auto")
    ax.axis("off")
    _draw_pipeline(ax)
    _draw_model(ax)
    return fig


def build_figure(output_path: Path, *, pdf_output: Path | None = None) -> Path:
    fig = _create_figure()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", facecolor="white")
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, format="pdf", facecolor="white")
    plt.close(fig)
    validate_paper_svg(
        output_path,
        required_width_mm=PAPER_SCHEMATIC_WIDTH_MM,
    )
    return output_path


def main() -> None:
    args = parse_args()
    svg_path = build_figure(args.output, pdf_output=args.pdf_output)
    print(
        f"Wrote {svg_path} and {args.pdf_output} "
        f"({PAPER_WIDTH_MM:g} mm × {PAPER_HEIGHT_MM:g} mm)"
    )


if __name__ == "__main__":
    main()
