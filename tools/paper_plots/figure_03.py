from __future__ import annotations

import argparse
import re
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from plotting import geneformer_like, scgpt_like
from plotting import training_loss
from plotting.utils import PLOTS_DIR, TOL_MUTED_CMAP_NAMES
from tools.paper_plots.utils import prefix_svg_ids as _prefix_svg_ids
from tools.paper_plots.data import paper_data
from tools.paper_plots.style import (
    PAPER_FONT_FAMILY,
    PAPER_MIN_FONT_SIZE_PT,
    PAPER_TICK_EDGE_PADDING_FRACTION,
    paper_font_size_in_svg_units,
    validate_paper_dimensions,
    validate_paper_svg,
)


PAPER_WIDTH_MM = 180.0
PAPER_HEIGHT_MM = 190.9
BIOSCORE_MAJOR_TICKS = (0.1, 0.2, 0.3, 0.4, 0.5)
CONTEXT_LENGTH_CMAPS = TOL_MUTED_CMAP_NAMES[:5]
DEFAULT_OUTPUT = PLOTS_DIR / "paper" / "03_scaling_and_BIOscore.svg"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
_SVG_TEXT_SCALE_RE = re.compile(r"scale\([0-9.]+[ ,]+-[0-9.]+\)")
_SVG_LIVE_FONT_SIZE_RE = re.compile(
    r"(font-size:\s*)[0-9.]+(?:px|pt)?"
)
_SVG_AXES_ID_RE = re.compile(r"^axes_([0-9]+)$")
SOURCE_PLOT_AXIS_COUNT = 5


@dataclass(frozen=True)
class Panel:
    label: str
    row_label: str
    column_label: str
    svg_path: Path
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper Figure 03 from geneformer-like and scGPT-like plots 04 and 08.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require_data(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached plot data: {path}. Run the corresponding *_like plotting script first."
        )
    return pl.read_parquet(path)


@contextmanager
def _paper_panel_typography():
    modules = (training_loss, geneformer_like, scgpt_like)
    names_and_values = {
        # The composite scales these sources to about 51% of authored size.
        "FONT_SIZE_TICKS": 12.8,
        "FONT_SIZE_LABELS": 13.8,
        "FONT_SIZE_TITLE": 13.8,
        "FONT_SIZE_LEGEND": 13.8,
    }
    original_values = {
        (module, name): getattr(module, name)
        for module in modules
        for name in names_and_values
    }
    original_color_shades = {module: module.COLOR_SHADES for module in modules}
    try:
        for module in modules:
            for name, value in names_and_values.items():
                setattr(module, name, value)
            module.COLOR_SHADES = CONTEXT_LENGTH_CMAPS
        yield
    finally:
        for (module, name), value in original_values.items():
            setattr(module, name, value)
        for module, color_shades in original_color_shades.items():
            module.COLOR_SHADES = color_shades


def _format_step_tick(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value / 1_000:g}k"


def _render_source_panels(directory: Path) -> dict[str, Path]:
    geneformer_prepared = _require_data(paper_data("geneformer_like/01_training_loss_prepared.parquet"))
    geneformer_downstream = _require_data(paper_data("geneformer_like/08_downstream_metrics_prepared.parquet"))
    scgpt_prepared = _require_data(paper_data("scgpt_like/01_training_loss_prepared.parquet"))
    scgpt_downstream = _require_data(paper_data("scgpt_like/08_downstream_metrics_prepared.parquet"))

    paths = {
        name: directory / f"{name}.svg"
        for name in ("geneformer_03", "geneformer_08", "scgpt_03", "scgpt_08")
    }
    common_layout = {
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
    }

    with _paper_panel_typography():
        geneformer_like.create_power_law_plot_from_df(
            geneformer_prepared,
            paths["geneformer_03"],
            title="",
            color_by_depth_width_ratio=True,
            x_label="Model size (parameters)",
            y_label="Cross-entropy loss",
            x_tick_rotation=0.0,
            enforce_y_axis_units=False,
            marker_area_scale=0.30,
            y_tick_bounds=(3.0, 8.0),
            **common_layout,
        )
        training_loss.create_training_loss_plot_from_df(
            geneformer_like._create_downstream_metric_frame(geneformer_downstream, "BIOscore"),
            paths["geneformer_08"],
            geneformer_like.AVGBIO_CONFIG,
            line_width=0.85,
            x_tick_formatter=_format_step_tick,
            y_major_ticks=BIOSCORE_MAJOR_TICKS,
            enforce_y_axis_units=False,
            **common_layout,
        )
        scgpt_like.create_power_law_plot_from_df(
            scgpt_prepared,
            paths["scgpt_03"],
            title="",
            ylabel="Mean squared error",
            x_label="Model size (parameters)",
            x_tick_rotation=0.0,
            enforce_y_axis_units=False,
            color_by_depth_width_ratio=True,
            marker_area_scale=0.30,
            y_limit_max=scgpt_like.MSE_Y_MAX,
            y_tick_bounds=(230.0, 330.0),
            **common_layout,
        )
        training_loss.create_training_loss_plot_from_df(
            geneformer_like._create_downstream_metric_frame(scgpt_downstream, "BIOscore"),
            paths["scgpt_08"],
            geneformer_like.AVGBIO_CONFIG,
            line_width=0.85,
            x_tick_formatter=_format_step_tick,
            y_major_ticks=BIOSCORE_MAJOR_TICKS,
            enforce_y_axis_units=False,
            **common_layout,
        )

    return paths


def _normalize_panel_text_size(root: ET.Element, panel: Panel) -> None:
    view_box = [float(value) for value in root.attrib["viewBox"].split()]
    physical_scale = min(
        panel.width_mm * 72.0 / 25.4 / view_box[2],
        panel.height_mm * 72.0 / 25.4 / view_box[3],
    )
    encoded_scale = PAPER_MIN_FONT_SIZE_PT / physical_scale / 100.0
    replacement = f"scale({encoded_scale:.8f} -{encoded_scale:.8f})"
    namespace = f"{{{SVG_NS}}}"
    top_level_axes_seen = 0

    def normalize_text(
        element: ET.Element,
        *,
        in_plot_axes: bool = False,
        in_axis_group: bool = False,
        axes_depth: int = 0,
    ) -> None:
        nonlocal top_level_axes_seen
        element_id = element.get("id", "")
        axes_match = _SVG_AXES_ID_RE.fullmatch(element_id)
        if axes_match:
            if axes_depth == 0:
                top_level_axes_seen += 1
                in_plot_axes = top_level_axes_seen <= SOURCE_PLOT_AXIS_COUNT
            else:
                # Inset colour bars are legend elements, not plot axes.
                in_plot_axes = False
            in_axis_group = False
            axes_depth += 1
        elif in_plot_axes and element_id.startswith("matplotlib.axis_"):
            in_axis_group = True

        if element.tag == f"{namespace}text":
            target_size_pt = 7.0 if in_plot_axes and in_axis_group else 5.0
            authored_size = target_size_pt / physical_scale
            if "font-size" in element.attrib:
                element.set("font-size", f"{authored_size:.8f}")
            style = element.get("style")
            if style and _SVG_LIVE_FONT_SIZE_RE.search(style):
                element.set(
                    "style",
                    _SVG_LIVE_FONT_SIZE_RE.sub(
                        rf"\g<1>{authored_size:.8f}px",
                        style,
                        count=1,
                    ),
                )

        for child in element:
            normalize_text(
                child,
                in_plot_axes=in_plot_axes,
                in_axis_group=in_axis_group,
                axes_depth=axes_depth,
            )

    normalize_text(root)

    for group in root.iter(f"{namespace}g"):
        group_id = group.get("id", "")
        if not (group_id.startswith("text_") or "_text_" in group_id):
            continue
        for child in group.iter(f"{namespace}g"):
            transform = child.get("transform")
            if transform and _SVG_TEXT_SCALE_RE.search(transform):
                child.set(
                    "transform",
                    _SVG_TEXT_SCALE_RE.sub(replacement, transform, count=1),
                )


def _append_svg_panel(master: ET.Element, panel: Panel) -> None:
    source_root = ET.parse(panel.svg_path).getroot()
    _normalize_panel_text_size(source_root, panel)
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


def _svg_text(
    root: ET.Element,
    text: str,
    *,
    x: float,
    y: float,
    size: float,
    anchor: str = "start",
    weight: str = "normal",
    transform: str | None = None,
) -> None:
    attributes = {
        "x": str(x),
        "y": str(y),
        "font-family": PAPER_FONT_FAMILY,
        # The master viewBox uses millimetres; SVG font-size uses viewBox units.
        "font-size": str(paper_font_size_in_svg_units(size)),
        "font-weight": weight,
        "fill": "#2f2a24",
        "text-anchor": anchor,
    }
    if transform:
        attributes["transform"] = transform
    element = ET.SubElement(root, f"{{{SVG_NS}}}text", attributes)
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
        panel_label_x = 1.4 if panel.x_mm < 10 else panel.x_mm + 1.5
        _svg_text(master, panel.label, x=panel_label_x, y=panel.y_mm + 4.5, size=7.0, weight="bold")

    ET.SubElement(master, f"{{{SVG_NS}}}line", {"x1": "92.5", "y1": "7", "x2": "92.5", "y2": "189", "stroke": "#b8ab93", "stroke-width": "0.3"})
    ET.SubElement(master, f"{{{SVG_NS}}}line", {"x1": "0.5", "y1": "99.5", "x2": "179.5", "y2": "99.5", "stroke": "#b8ab93", "stroke-width": "0.3"})
    _svg_text(master, "Scaling with non-embedding parameters", x=47.25, y=5.2, size=7.0, anchor="middle", weight="bold")
    _svg_text(master, "Downstream BIOscore", x=136.25, y=5.2, size=7.0, anchor="middle", weight="bold")
    _svg_text(master, "Ranked Gene Identity", x=2.0, y=54.0, size=7.0, anchor="middle", weight="bold", transform="rotate(-90 2.0 54)")
    _svg_text(master, "Binned Gene Expression", x=2.0, y=145.5, size=7.0, anchor="middle", weight="bold", transform="rotate(-90 2.0 145.5)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(master).write(output_path, encoding="utf-8", xml_declaration=True)
    validate_paper_svg(output_path)


def build_figure(output_path: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="paper_figure_03_") as temporary_directory:
        sources = _render_source_panels(Path(temporary_directory))
        left_panel_x = 5.0
        left_panel_width = 87.0
        right_panel_width = 86.5
        panel_height = 93.0
        panels = [
            Panel("A", "Ranked Gene Identity", "Scaling", sources["geneformer_03"], left_panel_x, 7.0, left_panel_width, panel_height),
            Panel("B", "Ranked Gene Identity", "BIOscore", sources["geneformer_08"], 93.0, 7.0, right_panel_width, panel_height),
            Panel("C", "Binned Gene Expression", "Scaling", sources["scgpt_03"], left_panel_x, 99.0, left_panel_width, 91.9),
            Panel("D", "Binned Gene Expression", "BIOscore", sources["scgpt_08"], 93.0, 99.0, right_panel_width, 91.9),
        ]
        _compose_svg(panels, output_path)
    return output_path


def main() -> None:
    svg_path = build_figure(parse_args().output)
    print(f"Wrote {svg_path} ({PAPER_WIDTH_MM:g} mm × {PAPER_HEIGHT_MM:g} mm)")


if __name__ == "__main__":
    main()
