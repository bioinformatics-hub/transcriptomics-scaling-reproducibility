from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.utils import (
    FONT_NAME,
    MIN_FONT_SIZE_PT,
    WONG_PALETTE,
    register_plot_font,
    style_axis_lines_and_ticks,
)


PAPER_FONT_FAMILY = FONT_NAME
PAPER_GRID_COLOR = "#d8d2c4"
PAPER_TEXT_COLOR = "#2f2a24"
PAPER_TICK_EDGE_PADDING_FRACTION = 0.20
PAPER_REQUIRED_WIDTH_MM = 180.0
PAPER_SCHEMATIC_WIDTH_MM = 88.0
PAPER_MAX_HEIGHT_MM = 210.0
PAPER_MIN_FONT_SIZE_PT = MIN_FONT_SIZE_PT
PAPER_MAX_FONT_SIZE_PT = 7.0

_FONT_REGISTERED = False
_SVG_NS = "http://www.w3.org/2000/svg"
_SVG_TEXT_SCALE_RE = re.compile(r"scale\(([0-9.]+)[ ,]+-")
_SVG_STYLE_FONT_SIZE_RE = re.compile(
    r"(?:^|;)\s*font-size:\s*([0-9.]+)(?:px|pt)?(?:;|$)"
)


def register_paper_fonts() -> None:
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    register_plot_font()
    _FONT_REGISTERED = True


def validate_paper_dimensions(
    width_mm: float,
    height_mm: float,
    *,
    required_width_mm: float = PAPER_REQUIRED_WIDTH_MM,
) -> None:
    if not math.isclose(width_mm, required_width_mm, abs_tol=1e-6):
        raise ValueError(
            "Paper figure width must be exactly "
            f"{required_width_mm:g} mm; received {width_mm:g} mm"
        )
    if height_mm > PAPER_MAX_HEIGHT_MM:
        raise ValueError(
            "Paper figure height exceeds the allowed "
            f"{PAPER_MAX_HEIGHT_MM:g} mm maximum: received {height_mm:g} mm"
        )


def paper_font_size_in_svg_units(size_pt: float) -> float:
    """Clamp a point size to the paper band and convert it to mm viewBox units."""
    clamped_pt = min(
        max(size_pt, PAPER_MIN_FONT_SIZE_PT), PAPER_MAX_FONT_SIZE_PT
    )
    return clamped_pt * 25.4 / 72.0


def _svg_length_points(value: str) -> float:
    if value.endswith("mm"):
        return float(value[:-2]) * 72.0 / 25.4
    if value.endswith("pt"):
        return float(value[:-2])
    if value.endswith("in"):
        return float(value[:-2]) * 72.0
    raise ValueError(f"SVG length needs an absolute unit: {value!r}")


def _matplotlib_text_sizes(svg: ET.Element, physical_scale: float) -> list[float]:
    namespace = f"{{{_SVG_NS}}}"
    sizes = []
    for text in svg.iter(f"{namespace}text"):
        raw_size = text.get("font-size")
        if raw_size is None:
            match = _SVG_STYLE_FONT_SIZE_RE.search(text.get("style", ""))
            raw_size = match.group(1) if match else None
        if raw_size is not None:
            sizes.append(float(raw_size.rstrip("pxpt")) * physical_scale)

    # Retain support for older path-based Matplotlib SVGs so the validator can
    # report their sizes before rejecting them as non-editable below.
    for group in svg.iter(f"{namespace}g"):
        group_id = group.get("id", "")
        if not (group_id.startswith("text_") or "_text_" in group_id):
            continue
        for child in list(group):
            if child.tag != f"{namespace}g":
                continue
            match = _SVG_TEXT_SCALE_RE.search(child.get("transform", ""))
            if match:
                # Matplotlib's SVG backend encodes N-point text as scale(N / 100).
                sizes.append(float(match.group(1)) * 100.0 * physical_scale)
                break
    return sizes


def validate_paper_svg(
    path: Path,
    *,
    required_width_mm: float = PAPER_REQUIRED_WIDTH_MM,
) -> None:
    """Validate final dimensions and effective text sizes in a paper SVG."""
    namespace = f"{{{_SVG_NS}}}"
    root = ET.parse(path).getroot()
    view_box = [float(value) for value in root.attrib["viewBox"].split()]
    width_pt = _svg_length_points(root.attrib["width"])
    height_pt = _svg_length_points(root.attrib["height"])
    validate_paper_dimensions(
        width_pt * 25.4 / 72.0,
        height_pt * 25.4 / 72.0,
        required_width_mm=required_width_mm,
    )

    font_sizes: list[float] = []
    nested_svgs = root.findall(f"{namespace}svg")
    if nested_svgs:
        for svg in nested_svgs:
            nested_view_box = [
                float(value) for value in svg.attrib["viewBox"].split()
            ]
            # Composite viewBox units are millimetres, including unitless nested
            # width and height attributes.
            physical_scale = min(
                float(svg.attrib["width"]) * 72.0 / 25.4 / nested_view_box[2],
                float(svg.attrib["height"]) * 72.0 / 25.4 / nested_view_box[3],
            )
            font_sizes.extend(_matplotlib_text_sizes(svg, physical_scale))
    else:
        physical_scale = min(width_pt / view_box[2], height_pt / view_box[3])
        font_sizes.extend(_matplotlib_text_sizes(root, physical_scale))

    if not font_sizes:
        raise ValueError(f"No text sizes found in paper SVG: {path}")

    editable_text = list(root.iter(f"{namespace}text"))
    if not editable_text:
        raise ValueError(
            f"{path} encodes text as paths; paper SVG text must remain editable"
        )
    text_segments = [
        element
        for tag in ("text", "tspan")
        for element in root.iter(f"{namespace}{tag}")
        if element.text and element.text.strip()
    ]
    for text in text_segments:
        font_description = " ".join(
            (text.get("font-family", ""), text.get("style", ""))
        )
        if PAPER_FONT_FAMILY.lower() not in font_description.lower():
            raise ValueError(
                f"{path} contains editable text that does not declare "
                f"{PAPER_FONT_FAMILY}: {font_description!r}"
            )
    minimum = min(font_sizes)
    maximum = max(font_sizes)
    # Composite viewBox scaling and SVG decimal serialization can introduce
    # hundredths of a point of numerical drift around the authored limits.
    tolerance = 0.02
    if minimum < PAPER_MIN_FONT_SIZE_PT - tolerance:
        raise ValueError(
            f"{path} contains {minimum:.3f} pt text, below the "
            f"{PAPER_MIN_FONT_SIZE_PT:g} pt minimum"
        )
    if maximum > PAPER_MAX_FONT_SIZE_PT + tolerance:
        raise ValueError(
            f"{path} contains {maximum:.3f} pt text, above the "
            f"{PAPER_MAX_FONT_SIZE_PT:g} pt maximum"
        )


def use_paper_style() -> None:
    register_paper_fonts()
    plt.rcParams.update(
        {
            "agg.path.chunksize": 200,
            "font.family": PAPER_FONT_FAMILY,
            "font.size": PAPER_MIN_FONT_SIZE_PT,
            # SVG text remains selectable/editable. PDF and PostScript use
            # embedded TrueType (type 42) fonts, as required by the journal.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.fallback": None,
            "text.color": PAPER_TEXT_COLOR,
            "axes.labelcolor": PAPER_TEXT_COLOR,
            "axes.edgecolor": PAPER_TEXT_COLOR,
            "axes.titlecolor": PAPER_TEXT_COLOR,
            "axes.prop_cycle": plt.cycler(color=WONG_PALETTE),
            "axes.linewidth": 0.6,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PAPER_TEXT_COLOR,
            "ytick.color": PAPER_TEXT_COLOR,
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
            "legend.frameon": False,
        }
    )


def despine(ax: plt.Axes, tick_labelsize: float = PAPER_MAX_FONT_SIZE_PT) -> None:
    resolved_size = min(
        max(tick_labelsize, PAPER_MIN_FONT_SIZE_PT), PAPER_MAX_FONT_SIZE_PT
    )
    style_axis_lines_and_ticks(ax, tick_labelsize=resolved_size)
