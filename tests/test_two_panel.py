"""Tests for the two-panel SIP plot (issue #8)."""
import io

import SDAT
from test_parser import oe_psip_file


def processed(loops=("1", "2")):
    text = oe_psip_file()
    header, data = text.split("Phase_Shift[rad],\n")
    rows = data.splitlines()
    data = "\n".join(r.replace("1,", f"{loop},", 1) for loop in loops for r in rows)
    df, ref, _, unit = SDAT.parse_sip_file(io.BytesIO(f"{header}Phase_Shift[rad],\n{data}".encode()))
    df = SDAT.calculate_physics_properties(df, ref, 0.03, 0.002, unit)
    df['Loop'] = df['Loop'].astype(str)
    return df


def test_two_panels_share_log_frequency_axis():
    df = processed()
    fig = SDAT.create_sip_two_panel_plot(
        {"a.csv": df}, "Chan-1", "Resistivity (Ohm-m)", "Frequency[Hz]", log_y_bottom=True
    )
    # One phase + one bottom trace per loop, linked through the legend group
    assert len(fig.data) == 4
    assert [t.yaxis for t in fig.data] == ["y", "y2", "y", "y2"]
    assert fig.data[0].legendgroup == fig.data[1].legendgroup == "Loop 1"
    assert fig.data[0].showlegend and not fig.data[1].showlegend
    assert fig.layout.xaxis.matches == "x2" or fig.layout.xaxis2.matches == "x"
    assert fig.layout.xaxis.type == fig.layout.xaxis2.type == "log"
    assert fig.layout.yaxis2.type == "log"
    assert list(fig.data[0].y) == list(df[df.Loop == "1"]["Chan-1 Phase (mRads)"])


def test_comparison_loop_filter_and_style_apply_to_both_panels():
    fig = SDAT.create_sip_two_panel_plot(
        {"a.csv": processed(), "b.csv": processed(("1",))}, "Chan-1",
        "Fluid Conductivity (uS/cm)", "Frequency[Hz]", selected_loops={"a.csv": ["2"]}
    )
    # Only loop 2 of a.csv is left, so (like the overlay plot) the trace takes the file name
    assert [t.name for t in fig.data[::2]] == ["a.csv", "b.csv"]
    assert len(fig.data[0].x) == 5
    style = SDAT.get_research_presets()["IEEE"]
    fig = SDAT.apply_style_to_figure(fig, style, {"a.csv": "#123456", "b.csv": "#654321"},
                                     {'title_size': 20, 'axis_label_size': 15, 'tick_size': 11, 'legend_size': 9})
    for axis in (fig.layout.xaxis, fig.layout.xaxis2, fig.layout.yaxis, fig.layout.yaxis2):
        assert axis.tickfont.size == 11
        assert axis.title.font.size == 15
    assert fig.data[0].line.color == fig.data[1].line.color == "#123456"


def test_phase_channels():
    assert SDAT.get_phase_channels(["Frequency[Hz]", "Chan-1 Phase (mRads)", "Chan-2 Phase (mRads)"]) == ["Chan-1", "Chan-2"]
