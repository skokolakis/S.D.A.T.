"""Tests for quantity labels and plot palettes used by the UI."""
import re

import pytest

import SDAT


@pytest.mark.parametrize("column, display, axis", [
    ("Chan-1 Resistivity (Ohm-m)", "Chan-1 · Resistivity |ρ| (Ω·m)", "|ρ| (Ω·m)"),
    ("Chan-2 Phase (mRads)", "Chan-2 · Phase −φ (mrad)", "−φ (mrad)"),
    ("Chan-1 Imaginary Conductivity (uS/cm)", "Chan-1 · Imaginary conductivity σ″ (µS/cm)", "σ″ (µS/cm)"),
    ("Frequency[Hz]", "Frequency f (Hz)", "f (Hz)"),
    ("Loop", "Loop", "Loop"),
])
def test_labels_keep_columns_readable(column, display, axis):
    assert SDAT.display_name(column) == display
    assert SDAT.axis_label(column) == axis


def test_every_calculated_quantity_has_a_label():
    import io
    from test_parser import oe_psip_file
    df, ref, _, unit = SDAT.parse_sip_file(io.BytesIO(oe_psip_file().encode()))
    df = SDAT.calculate_physics_properties(df, ref, 0.03, 0.002, unit)
    for column in df.columns:
        if column.startswith("Chan-1 "):
            assert SDAT.display_name(column) != column, column


@pytest.mark.parametrize("palette", SDAT.PALETTES)
def test_palettes_give_hex_colours(palette):
    colors = SDAT.palette_colors(palette, 5)
    assert len(colors) == 5
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colors)
    assert SDAT.palette_colors(palette, 0) == []


def test_publication_preset_is_white_with_boxed_log_axes():
    fig = SDAT.make_subplots(rows=1, cols=1)
    fig.add_scatter(x=[0.1, 1, 10], y=[1, 2, 3], name="a")
    fig.update_xaxes(type="log")
    style = SDAT.get_research_presets()[SDAT.DEFAULT_STYLE_PRESET]
    fig = SDAT.apply_style_to_figure(fig, style)
    assert fig.layout.plot_bgcolor == "#FFFFFF"
    assert fig.layout.xaxis.mirror is True and fig.layout.xaxis.ticks == "outside"
    assert fig.layout.xaxis.dtick == 1 and fig.layout.xaxis.exponentformat == "power"
    assert fig.layout.yaxis.dtick is None
