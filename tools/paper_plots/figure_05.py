from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from plotting import geneformer_dw as dw
from plotting.training_loss import GENEFORMER_BCE_CONFIG, SCGPT_MSE_CONFIG
from plotting.utils import PLOTS_DIR
from tools.paper_plots.utils import (
    prefix_svg_ids as _prefix_svg_ids,
    temporary_plot_settings,
)
from tools.paper_plots.data import paper_data
from tools.paper_plots.style import (
    PAPER_FONT_FAMILY,
    paper_font_size_in_svg_units,
    validate_paper_dimensions,
    validate_paper_svg,
)


PAPER_WIDTH_MM = 180.0
PAPER_HEIGHT_MM = 174.0
DEFAULT_OUTPUT = PLOTS_DIR / "paper" / "05_depth_width_scaling.svg"
GENEFORMER_TRAINING_DATA = (
    paper_data("geneformer_dw/02_training_loss_flops_prepared.parquet")
)
SCGPT_TRAINING_DATA = (
    paper_data("scgpt_dw_rerun/02_training_loss_flops_prepared.parquet")
)
GENEFORMER_SURFACE_DATA = paper_data("geneformer_dw/01_isoflops_prepared.parquet")
SCGPT_SURFACE_DATA = paper_data("scgpt_dw_rerun/01_isoflops_prepared.parquet")
SURFACE_FRACTIONS = ("random_5fold", "leave_compute_out")
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


class Panel:
    def __init__(
        self,
        label: str,
        svg_path: Path,
        x_mm: float,
        y_mm: float,
        width_mm: float,
        height_mm: float,
    ):
        self.label = label
        self.svg_path = svg_path
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.width_mm = width_mm
        self.height_mm = height_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper Figure 05: depth/width scaling diagnostics."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached depth/width data: {path}")
    return pl.read_parquet(path)


def _compute_slices(path: Path) -> list[float]:
    values = (
        _require(path)["analysis_training_flops"].drop_nulls().unique().sort().to_list()
    )
    return [float(value) for value in values]


@contextmanager
def _paper_panel_typography():
    names_and_values = {
        # Source panels are reduced to roughly 32% in the composite SVG.
        "FONT_SIZE_TICKS": 18,
        "FONT_SIZE_LABELS": 21.8,
        "FONT_SIZE_TITLE": 21.8,
        "FONT_SIZE_LEGEND": 21.8,
    }
    with temporary_plot_settings(dw, names_and_values):
        yield


def _append_svg_panel(master: ET.Element, panel: Panel) -> None:
    source_root = ET.parse(panel.svg_path).getroot()
    _prefix_svg_ids(source_root, panel.label.lower())
    nested = ET.SubElement(
        master,
        f"{{{SVG_NS}}}svg",
        {
            "x": str(panel.x_mm),
            "y": str(panel.y_mm),
            "width": str(panel.width_mm),
            "height": str(panel.height_mm),
            "viewBox": source_root.attrib["viewBox"],
            "preserveAspectRatio": "xMidYMid meet",
        },
    )
    for child in list(source_root):
        nested.append(child)


def _source_paths(directory: Path) -> dict[str, Path]:
    names = (
        "geneformer_training",
        "geneformer_cv",
        "geneformer_bootstrap",
        "scgpt_training",
        "scgpt_cv",
        "scgpt_bootstrap",
    )
    return {name: directory / f"{name}.svg" for name in names}


def _render_sources(
    directory: Path,
) -> tuple[dict[str, Path], list[float]]:
    gf_training = _require(GENEFORMER_TRAINING_DATA)
    sc_training = _require(SCGPT_TRAINING_DATA)
    gf_surface = _require(GENEFORMER_SURFACE_DATA)
    sc_surface = _require(SCGPT_SURFACE_DATA)
    slices = _compute_slices(GENEFORMER_SURFACE_DATA)
    if slices != _compute_slices(SCGPT_SURFACE_DATA):
        raise ValueError("Geneformer and scGPT compute slices do not match")

    paths = _source_paths(directory)
    original_config = dw.GENEFORMER_BCE_CONFIG
    try:
        with _paper_panel_typography():
            dw.GENEFORMER_BCE_CONFIG = GENEFORMER_BCE_CONFIG
            dw.create_training_loss_vs_flops_plot(
                gf_training,
                paths["geneformer_training"],
                compute_slice_flops=slices,
                figsize=(10.8, 6.5),
                crop_output=False,
                ylabel="Cross-entropy loss",
                model_size_cmap="plasma",
                compact_legend=True,
                legend_width_ratio=1.65,
                enforce_y_axis_units=False,
            )
            dw.create_surface_cv_diagnostics_plot(
                gf_surface,
                paths["geneformer_cv"],
                strategies_to_plot=SURFACE_FRACTIONS,
                figsize=(10.8, 6.48),
                crop_output=False,
                y_min=5.2,
                match_xy_major_ticks=True,
                observed_loss_label="Observed cross-entropy loss",
                predicted_loss_label="Predicted cross-entropy loss",
                enforce_x_axis_units=False,
                enforce_y_axis_units=False,
            )
            gf_model = dw._fit_surface_model_from_fit_df(dw._surface_fit_df(gf_surface))
            dw.create_surface_bootstrap_optimum_plot(
                gf_surface,
                paths["geneformer_bootstrap"],
                model=gf_model,
                target_training_flops=dw.DEFAULT_TARGET_TRAINING_FLOPS,
                figsize=(10.8, 6.8),
                crop_output=False,
                compact_legend=True,
                legend_width_ratio=2.05,
                y_label="Predicted optimal depth-to-width ratio",
                enforce_y_axis_units=False,
            )

            dw.GENEFORMER_BCE_CONFIG = SCGPT_MSE_CONFIG
            dw.create_training_loss_vs_flops_plot(
                sc_training,
                paths["scgpt_training"],
                compute_slice_flops=slices,
                figsize=(10.8, 6.5),
                crop_output=False,
                ylabel="Mean squared error",
                model_size_cmap="plasma",
                compact_legend=True,
                legend_width_ratio=1.65,
                enforce_y_axis_units=False,
            )
            dw.create_surface_cv_diagnostics_plot(
                sc_surface,
                paths["scgpt_cv"],
                strategies_to_plot=SURFACE_FRACTIONS,
                figsize=(10.8, 6.48),
                crop_output=False,
                match_xy_major_ticks=True,
                observed_loss_label="Observed mean squared error",
                predicted_loss_label="Predicted mean squared error",
                enforce_x_axis_units=False,
                enforce_y_axis_units=False,
            )
            sc_model = dw._fit_surface_model_from_fit_df(dw._surface_fit_df(sc_surface))
            dw.create_surface_bootstrap_optimum_plot(
                sc_surface,
                paths["scgpt_bootstrap"],
                model=sc_model,
                target_training_flops=dw.DEFAULT_TARGET_TRAINING_FLOPS,
                figsize=(10.8, 6.8),
                crop_output=False,
                compact_legend=True,
                legend_width_ratio=2.05,
                y_label="Predicted optimal depth-to-width ratio",
                enforce_y_axis_units=False,
            )
    finally:
        dw.GENEFORMER_BCE_CONFIG = original_config
    return paths, slices


def _svg_text(
    root: ET.Element,
    text: str,
    *,
    x: float,
    y: float,
    size: float,
    anchor: str = "middle",
) -> None:
    element = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": str(y),
            "font-family": PAPER_FONT_FAMILY,
            "font-size": str(paper_font_size_in_svg_units(size)),
            "font-weight": "bold",
            "fill": "#2f2a24",
            "text-anchor": anchor,
        },
    )
    element.text = text


def _compose_svg(panels: list[Panel], output_path: Path) -> None:
    validate_paper_dimensions(PAPER_WIDTH_MM, PAPER_HEIGHT_MM)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{PAPER_WIDTH_MM}mm",
            "height": f"{PAPER_HEIGHT_MM}mm",
            "viewBox": f"0 0 {PAPER_WIDTH_MM} {PAPER_HEIGHT_MM}",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"width": str(PAPER_WIDTH_MM), "height": str(PAPER_HEIGHT_MM), "fill": "white"},
    )
    for panel in panels:
        _append_svg_panel(root, panel)
        _svg_text(
            root,
            panel.label,
            x=panel.x_mm + 1.0,
            y=panel.y_mm + 4.5,
            size=5.0,
            anchor="start",
        )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}line",
        {
            "x1": "90",
            "y1": "3",
            "x2": "90",
            "y2": "173",
            "stroke": "#b8ab93",
            "stroke-width": "0.3",
        },
    )
    _svg_text(root, "Ranked Gene Identity", x=45.0, y=5.5, size=7.0)
    _svg_text(root, "Binned Gene Expression", x=135.0, y=5.5, size=7.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    validate_paper_svg(output_path)


def build_figure(output_path: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="paper_figure_05_") as temporary_directory:
        sources, _ = _render_sources(Path(temporary_directory))
        panels = [
            Panel("A", sources["geneformer_training"], 1.0, 8.0, 87.5, 53.0),
            Panel("B", sources["geneformer_cv"], 1.0, 62.5, 87.5, 53.0),
            Panel("C", sources["geneformer_bootstrap"], 1.0, 117.0, 87.5, 56.0),
            Panel("D", sources["scgpt_training"], 91.0, 8.0, 87.5, 53.0),
            Panel("E", sources["scgpt_cv"], 91.0, 62.5, 87.5, 53.0),
            Panel("F", sources["scgpt_bootstrap"], 91.0, 117.0, 87.5, 56.0),
        ]
        _compose_svg(panels, output_path)
    return output_path


def main() -> None:
    svg_path = build_figure(parse_args().output)
    print(f"Wrote {svg_path} ({PAPER_WIDTH_MM:g} mm × {PAPER_HEIGHT_MM:g} mm)")


if __name__ == "__main__":
    main()
