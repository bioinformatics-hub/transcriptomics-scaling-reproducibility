from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import re

from plotting.utils import (
    MAX_FIGURE_HEIGHT_MM,
    MAX_FIGURE_WIDTH_MM,
    MIN_FONT_SIZE_PT,
    TOL_MUTED_CMAP_NAMES,
    WONG_CMAP_NAMES,
    WONG_PALETTE,
    axis_label_with_unit,
    save_figure,
)
from tools.paper_plots import (
    figure_01,
    figure_02,
    figure_03,
    figure_04,
    figure_05,
    figure_06,
    supplementary,
)
from tools.paper_plots.style import (
    PAPER_FONT_FAMILY,
    PAPER_MAX_FONT_SIZE_PT,
    PAPER_MIN_FONT_SIZE_PT,
    PAPER_SCHEMATIC_WIDTH_MM,
    paper_font_size_in_svg_units,
    validate_paper_dimensions,
    validate_paper_svg,
)


def _svg_length_mm(svg: str, attribute: str) -> float:
    match = re.search(rf'\b{attribute}="([0-9.]+)(mm|pt|in)"', svg)
    assert match is not None
    value = float(match.group(1))
    return {
        "mm": value,
        "pt": value * 25.4 / 72.0,
        "in": value * 25.4,
    }[match.group(2)]


def test_save_figure_writes_only_svg_and_enforces_contract(tmp_path: Path) -> None:
    output_path = tmp_path / "figure.svg"
    fig, ax = plt.subplots(figsize=(14, 10))
    label = ax.text(0.5, 0.5, "small", fontsize=3)
    title = ax.set_title("large", fontsize=20)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cross Entropy")

    save_figure(
        fig,
        output_path,
        apply_default_adjust=False,
        crop=False,
        max_font_size_pt=PAPER_MAX_FONT_SIZE_PT,
        max_size_mm=(MAX_FIGURE_WIDTH_MM, MAX_FIGURE_HEIGHT_MM),
    )

    assert output_path.exists()
    assert not output_path.with_suffix(".png").exists()
    assert label.get_fontsize() == MIN_FONT_SIZE_PT
    assert title.get_fontsize() == PAPER_MAX_FONT_SIZE_PT
    assert ax.get_xlabel() == "Training progress (steps)"
    assert ax.get_ylabel() == "Cross-entropy loss (nats)"
    assert ax.spines["left"].get_visible()
    assert ax.spines["bottom"].get_visible()
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.xaxis.majorTicks[0].tick1line.get_markersize() == pytest.approx(2.5)
    svg = output_path.read_text()
    assert _svg_length_mm(svg, "width") <= MAX_FIGURE_WIDTH_MM + 1e-6
    assert _svg_length_mm(svg, "height") <= MAX_FIGURE_HEIGHT_MM + 1e-6
    validate_paper_svg(output_path)


def test_save_figure_can_keep_y_axis_label_without_a_unit_suffix(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "figure.svg"
    fig, ax = plt.subplots()
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cross-entropy loss")

    save_figure(
        fig,
        output_path,
        crop=False,
        enforce_y_axis_units=False,
    )

    assert ax.get_xlabel() == "Training progress (steps)"
    assert ax.get_ylabel() == "Cross-entropy loss"


@pytest.mark.parametrize(
    ("width_mm", "height_mm"),
    [
        (figure_02.PAPER_WIDTH_MM, figure_02.PAPER_HEIGHT_MM),
        (figure_03.PAPER_WIDTH_MM, figure_03.PAPER_HEIGHT_MM),
        (figure_04.PAPER_WIDTH_MM, figure_04.PAPER_HEIGHT_MM),
        (figure_05.PAPER_WIDTH_MM, figure_05.PAPER_HEIGHT_MM),
        (figure_06.PAPER_WIDTH_MM, figure_06.PAPER_HEIGHT_MM),
    ],
)
def test_paper_figure_dimensions_match_contract(
    width_mm: float, height_mm: float
) -> None:
    assert width_mm == pytest.approx(180.0)
    validate_paper_dimensions(width_mm, height_mm)


def test_schematic_dimensions_match_contract() -> None:
    assert figure_01.PAPER_WIDTH_MM == pytest.approx(PAPER_SCHEMATIC_WIDTH_MM)
    validate_paper_dimensions(
        figure_01.PAPER_WIDTH_MM,
        figure_01.PAPER_HEIGHT_MM,
        required_width_mm=PAPER_SCHEMATIC_WIDTH_MM,
    )


def test_figure_03_uses_a_distinct_accessible_context_palette() -> None:
    assert figure_03.CONTEXT_LENGTH_CMAPS == TOL_MUTED_CMAP_NAMES[:5]


def test_supplementary_reuses_main_figure_color_conventions() -> None:
    assert supplementary.CONTEXT_LENGTH_CMAPS == figure_03.CONTEXT_LENGTH_CMAPS
    assert supplementary.LR_SIZE_CMAPS == WONG_CMAP_NAMES[:4]


def test_paper_style_uses_editable_arial_and_type_42_fonts() -> None:
    from tools.paper_plots.style import use_paper_style

    use_paper_style()
    assert PAPER_FONT_FAMILY == "Arial"
    assert plt.rcParams["font.family"] == ["Arial"]
    assert plt.rcParams["svg.fonttype"] == "none"
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    assert tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"]) == WONG_PALETTE


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Training Steps", "Training progress (steps)"),
        ("Training FLOPs", "Training compute (FLOPs)"),
        ("Non-embedding Parameters", "Non-embedding model size (parameters)"),
        ("Depth / Width", "Depth-to-width ratio (dimensionless)"),
        ("BIOscore", "BIOscore (dimensionless)"),
        ("MSE", "Mean squared error (squared expression units)"),
        ("Log10 LR Error (dex)", "Log10 LR Error (dex)"),
    ],
)
def test_axis_labels_include_parenthesized_units(
    label: str,
    expected: str,
) -> None:
    assert axis_label_with_unit(label) == expected


def test_schematic_svg_is_editable_arial_at_five_to_seven_points(
    tmp_path: Path,
) -> None:
    output = tmp_path / "01_schematic.svg"
    pdf_output = tmp_path / "01_schematic.pdf"
    figure_01.build_figure(output, pdf_output=pdf_output)
    svg = output.read_text()
    pdf = pdf_output.read_bytes()

    assert _svg_length_mm(svg, "width") == pytest.approx(88.0)
    assert "Data Pipeline" not in svg
    assert "Model Structure" not in svg
    assert "font-family: 'Arial'" in svg
    assert "font-size: 5px" in svg
    assert "font-size: 7px" in svg
    assert "<text" in svg
    assert b"/CIDFontType2" in pdf
    assert b"/FontFile2" in pdf
    assert b"Arial" in pdf
    assert b"/Subtype /Type3" not in pdf
    validate_paper_svg(output, required_width_mm=PAPER_SCHEMATIC_WIDTH_MM)


@pytest.mark.parametrize("width_mm", [179.0, 181.0])
def test_paper_dimension_guard_rejects_nonstandard_width(width_mm: float) -> None:
    with pytest.raises(ValueError, match="exactly 180 mm"):
        validate_paper_dimensions(width_mm, 100.0)


def test_paper_dimension_guard_rejects_oversized_height() -> None:
    with pytest.raises(ValueError, match="height exceeds"):
        validate_paper_dimensions(180.0, MAX_FIGURE_HEIGHT_MM + 1)


def test_manual_svg_font_sizes_are_clamped_to_paper_band() -> None:
    points_per_mm = 72.0 / 25.4
    assert paper_font_size_in_svg_units(3) * points_per_mm == pytest.approx(
        PAPER_MIN_FONT_SIZE_PT
    )
    assert paper_font_size_in_svg_units(9) * points_per_mm == pytest.approx(
        PAPER_MAX_FONT_SIZE_PT
    )
