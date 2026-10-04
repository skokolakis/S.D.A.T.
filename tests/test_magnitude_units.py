"""Tests for magnitude unit detection and dB conversion (issue #7)."""
import io

import numpy as np
import pytest

import SDAT

FREQS = [0.1, 1.0, 10.0, 100.0, 1000.0]
RATIOS = [1.52, 1.50, 1.49, 1.47, 1.45]
PHASES = [-0.004, -0.006, -0.008, -0.007, -0.010]


def oe_psip_ratio_file() -> bytes:
    lines = [
        "OE PSIP Measurement",
        "PSIP_Version,2.8.0,",
        "Current Resistor[Ohms],100",
        "***End_Of_Header***",
        ",,Chan-1,Chan-1,",
        "Loop,Frequency[Hz],Magnitude[ratio],Phase_Shift[rad],",
    ]
    lines += [f"1,{f},{m},{p}," for f, m, p in zip(FREQS, RATIOS, PHASES)]
    return "\n".join(lines).encode()


def writer_v2_file(magnitude_header: str) -> bytes:
    """LabVIEW-style Writer_Version 2 file: column names follow the last marker."""
    lines = [
        "LabVIEW Measurement,",
        "Writer_Version,2",
        "Reader_Version,2",
        "Separator,Comma",
        "***End_of_Header***,",
        "Channels,3,",
        "***End_of_Header***,",
        f"X_Value,Frequency[Hz],{magnitude_header},Phase_Shift[rad],Comment",
    ]
    lines += [f"0,{f},{20 * np.log10(m)},{p},"
              for f, m, p in zip(FREQS, RATIOS, PHASES)]
    return "\n".join(lines).encode()


def process(raw: bytes, unit_override=None):
    df, ref, _, unit = SDAT.parse_sip_file(io.BytesIO(raw))
    assert df is not None
    df = SDAT.calculate_physics_properties(df, ref, 0.03, 0.002, unit_override or unit)
    return df, unit


def resistivity(df):
    col = next(c for c in df.columns if c.endswith("Resistivity (Ohm-m)"))
    return df[col].to_numpy()


def test_ratio_file_detected_as_ratio():
    _, unit = process(oe_psip_ratio_file())
    assert unit == "ratio"


@pytest.mark.parametrize("header", ["Magnitude[dB]", "Magnitude"])
def test_writer_v2_db_file_matches_ratio_file(header):
    ratio_df, _ = process(oe_psip_ratio_file())
    db_df, unit = process(writer_v2_file(header))
    assert unit == "dB"
    np.testing.assert_allclose(resistivity(db_df), resistivity(ratio_df), rtol=1e-9)
    assert "Chan-1 Phase (mRads)" in db_df.columns


def test_column_unit_takes_precedence_over_writer_version():
    lines = ["Writer_Version,2"]
    assert SDAT.detect_magnitude_unit(lines, ["Chan-1 Magnitude[ratio]"]) == "ratio"
    assert SDAT.detect_magnitude_unit(lines, ["Magnitude"]) == "dB"
    assert SDAT.detect_magnitude_unit(["PSIP_Version,2.8.0,"], ["Magnitude"]) == "ratio"


def test_manual_override_converts_ratio_labelled_column():
    df, _ = process(oe_psip_ratio_file(), unit_override="dB")
    expected = 100 * 10 ** (np.array(RATIOS) / 20) * 0.002 / 0.03
    np.testing.assert_allclose(resistivity(df), expected)
