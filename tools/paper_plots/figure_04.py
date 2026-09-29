from __future__ import annotations

import argparse
import re
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from plotting import geneformer_lr as lr
from plotting.utils import COLOR_SHADES, PLOTS_DIR
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
PAPER_HEIGHT_MM = 134.0
DEFAULT_OUTPUT = PLOTS_DIR / "paper" / "04_learning_rate_scaling.svg"
GENEFORMER_DATA = paper_data("geneformer_lr/01_lr_sampled_prepared.parquet")
# Deliberately use the MSE-derived scGPT cache; the BCE variants are irrelevant.
SCGPT_MSE_DATA = paper_data("scgpt_lr_mse/01_lr_sampled_prepared.parquet")
MIN_COMPLETION = 0.90
SHARED_SERIES_Y_LIMITS = (-0.04, 0.84)
# 20% of one log-decade on either side of the outer 1e-5 and 1e-3 ticks.
SHARED_OPTIMAL_LR_Y_LIMITS = (10**-5.2, 10**-2.8)
BASE_PANEL_TYPOGRAPHY = {
    "FONT_SIZE_TICKS": 17.0,
    "FONT_SIZE_LABELS": 20.0,
    "FONT_SIZE_TITLE": 16.98,
    "FONT_SIZE_LEGEND": 20.96,
}
UPPER_PANEL_TYPOGRAPHY = {
    # Upper sources are reduced to 29.5% in the composite. These values make
    # their final ticks about 6.6 pt and labels/legend titles about 7 pt,
    # matching panels C and F.
    "FONT_SIZE_TICKS": 22.34,
    "FONT_SIZE_LABELS": 23.62,
    "FONT_SIZE_TITLE": 23.62,
    "FONT_SIZE_LEGEND": 20.96,
}
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
_SVG_FONT_SIZE_RE = re.compile(r"(font-size:\s*)([0-9.]+)(?:px|pt)?")


@dataclass(frozen=True)
class ModelData:
    name: str
    fraction: float
    training_flops: float
    best: pl.DataFrame
    summaries: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class Panel:
    label: str
    svg_path: Path
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    align_bottom: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper Figure 04 from the original Geneformer and scGPT MSE LR panels."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_model(name: str, path: Path, *, expected_metric: str) -> ModelData:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached LR data: {path}")
    prepared = pl.read_parquet(path)
    metrics = set(prepared["metric_name"].unique().to_list())
    if metrics != {expected_metric}:
        raise ValueError(
            f"{name} must use only metric={expected_metric!r}; found {sorted(metrics)} in {path}"
        )

    fractions = sorted(
        float(value) for value in prepared["analysis_fraction"].unique().to_list()
    )
    if len(fractions) < 2:
        raise ValueError(f"{path} needs at least two compute slices")
    selected_fraction = fractions[-1]
    selected_best: pl.DataFrame | None = None
    selected_flops: float | None = None
    summaries: list[dict[str, object]] = []
    for fraction in fractions:
        fraction_df = prepared.filter(
            (pl.col("analysis_fraction") - fraction).abs() < 1e-9
        )
        training_flops = float(
            fraction_df["analysis_training_flops"].drop_nulls().first()
        )
        best = lr._quadratic_best_lr_df(
            lr._terminal_loss_df(fraction_df), MIN_COMPLETION
        )
        summaries.append(
            lr._power_law_validation_summary(fraction, training_flops, best)
        )
        if np.isclose(fraction, selected_fraction):
            selected_best = best
            selected_flops = training_flops

    assert selected_best is not None and selected_flops is not None
    return ModelData(
        name, selected_fraction, selected_flops, selected_best, tuple(summaries)
    )


@contextmanager
def _paper_panel_typography(names_and_values: dict[str, float]):
    with temporary_plot_settings(lr, names_and_values):
        yield


def _render_source_panels(
    directory: Path,
) -> tuple[dict[str, Path], tuple[ModelData, ModelData]]:
    geneformer = _load_model("Geneformer", GENEFORMER_DATA, expected_metric="bce")
    scgpt = _load_model("scGPT", SCGPT_MSE_DATA, expected_metric="mse")
    if not np.isclose(geneformer.fraction, scgpt.fraction):
        raise ValueError(
            "The last Geneformer and scGPT slices do not match in relative progress"
        )

    paths = {
        name: directory / f"{name}.svg"
        for name in (
            "geneformer_size",
            "geneformer_depth",
            "geneformer_series",
            "scgpt_size",
            "scgpt_depth",
            "scgpt_series",
        )
    }
    with _paper_panel_typography(UPPER_PANEL_TYPOGRAPHY):
        for prefix, model in (("geneformer", geneformer), ("scgpt", scgpt)):
            suffix = f" ({lr._format_flops_label(model.training_flops)})"
            lr.create_optimal_lr_vs_parameters_plot(
                model.best,
                paths[f"{prefix}_size"],
                title_suffix=suffix,
                figsize=(5.8, 6.8),
                color_shades=tuple(COLOR_SHADES[:3]),
                legend_below=True,
                show_best_legend=False,
                show_title=False,
                crop_output=False,
                fit_label_x_multiplier=1.12,
                fit_label_min_log_gap=0.23,
                fit_label_fontsize=20.0,
                x_max_multiplier=3.0,
                y_limits=SHARED_OPTIMAL_LR_Y_LIMITS,
                x_label="Model size (parameters)",
                y_label="Optimal LR",
                enforce_y_axis_units=False,
                subplot_left=0.22,
            )
            lr.create_optimal_lr_vs_depth_plot(
                model.best,
                paths[f"{prefix}_depth"],
                title_suffix=suffix,
                figsize=(5.8, 6.8),
                legend_below=True,
                show_best_legend=False,
                show_title=False,
                crop_output=False,
                y_limits=SHARED_OPTIMAL_LR_Y_LIMITS,
                legend_box_width_scale=1.25,
                fit_label_x_multiplier=1.12,
                fit_label_min_log_gap=0.14,
                fit_label_fontsize=20.0,
                x_label="Model depth (layers)",
                y_label="Optimal LR",
                enforce_y_axis_units=False,
                subplot_left=0.22,
            )
    with _paper_panel_typography(BASE_PANEL_TYPOGRAPHY):
        for prefix, model in (("geneformer", geneformer), ("scgpt", scgpt)):
            lr.create_power_laws_compute_slices_plot(
                list(model.summaries),
                paths[f"{prefix}_series"],
                figsize=(9.0, 6.5),
                y_limits=SHARED_SERIES_Y_LIMITS,
                show_title=False,
            )

    return paths, (geneformer, scgpt)


def _append_svg_panel(master: ET.Element, panel: Panel) -> None:
    source_root = ET.parse(panel.svg_path).getroot()
    view_box = [float(value) for value in source_root.attrib["viewBox"].split()]
    physical_scale = min(
        panel.width_mm * 72.0 / 25.4 / view_box[2],
        panel.height_mm * 72.0 / 25.4 / view_box[3],
    )
    minimum_authored_size = 5.0 / physical_scale
    for element in source_root.iter():
        style = element.get("style", "")
        match = _SVG_FONT_SIZE_RE.search(style)
        if match and float(match.group(2)) < minimum_authored_size:
            element.set(
                "style",
                _SVG_FONT_SIZE_RE.sub(
                    rf"\g<1>{minimum_authored_size:.8f}px",
                    style,
                    count=1,
                ),
            )
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
            "preserveAspectRatio": (
                "xMidYMax meet" if panel.align_bottom else "xMidYMid meet"
            ),
        },
    )
    for child in list(source_root):
        nested.append(child)


def _svg_text(
    root: ET.Element,
    text: str,
    *,
    x: float,
    y: float,
    size: float,
    anchor: str = "start",
    weight: str = "normal",
) -> None:
    element = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": str(y),
            "font-family": PAPER_FONT_FAMILY,
            "font-size": str(paper_font_size_in_svg_units(size)),
            "font-weight": weight,
            "fill": "#2f2a24",
            "text-anchor": anchor,
        },
    )
    element.text = text


def _compose_svg(panels: list[Panel], output_path: Path) -> None:
    validate_paper_dimensions(PAPER_WIDTH_MM, PAPER_HEIGHT_MM)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    master = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{PAPER_WIDTH_MM}mm",
            "height": f"{PAPER_HEIGHT_MM}mm",
            "viewBox": f"0 0 {PAPER_WIDTH_MM} {PAPER_HEIGHT_MM}",
            "version": "1.1",
        },
    )
    ET.SubElement(
        master,
        f"{{{SVG_NS}}}rect",
        {"width": str(PAPER_WIDTH_MM), "height": str(PAPER_HEIGHT_MM), "fill": "white"},
    )
    for panel in panels:
        _append_svg_panel(master, panel)
        _svg_text(
            master,
            panel.label,
            x=panel.x_mm + 1.0,
            y=panel.y_mm + (2.0 if panel.y_mm < 20 else 4.5),
            size=5.0,
            weight="bold",
        )

    ET.SubElement(
        master,
        f"{{{SVG_NS}}}line",
        {
            "x1": "90",
            "y1": "1",
            "x2": "90",
            "y2": "132",
            "stroke": "#b8ab93",
            "stroke-width": "0.3",
        },
    )
    for x, heading in (
        (45.0, "Ranked Gene Identity"),
        (135.0, "Binned Gene Expression"),
    ):
        _svg_text(master, heading, x=x, y=4.5, size=7.0, anchor="middle", weight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(master).write(output_path, encoding="utf-8", xml_declaration=True)
    validate_paper_svg(output_path)


def build_figure(output_path: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="paper_figure_04_") as temporary_directory:
        sources, _ = _render_source_panels(Path(temporary_directory))
        panels = [
            Panel("A", sources["geneformer_size"], 1.0, 6.5, 43.5, 55.0, True),
            Panel("B", sources["geneformer_depth"], 45.0, 6.5, 43.5, 55.0, True),
            Panel("C", sources["geneformer_series"], 1.0, 63.0, 87.5, 68.5),
            Panel("D", sources["scgpt_size"], 91.0, 6.5, 43.5, 55.0, True),
            Panel("E", sources["scgpt_depth"], 135.0, 6.5, 43.5, 55.0, True),
            Panel("F", sources["scgpt_series"], 91.0, 63.0, 87.5, 68.5),
        ]
        _compose_svg(panels, output_path)
    return output_path


def main() -> None:
    svg_path = build_figure(parse_args().output)
    print(f"Wrote {svg_path} ({PAPER_WIDTH_MM:g} mm × {PAPER_HEIGHT_MM:g} mm)")


if __name__ == "__main__":
    main()
