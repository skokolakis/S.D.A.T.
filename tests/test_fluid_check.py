"""Tests for the theoretical fluid phase overlay (issue #9)."""
import io

import numpy as np
import pytest

import SDAT


def water_file(sigma_s_per_m: float, freqs) -> bytes:
    """PSIP file for water: Z ∝ 1/(σ + iωε), i.e. a negative raw phase shift."""
    omega = 2 * np.pi * np.asarray(freqs)
    z = 1 / (sigma_s_per_m + 1j * omega * 81 * 8.854e-12)
    ratio = np.abs(z) * 0.03 / 0.002 / 100
    lines = ["OE PSIP Measurement", "Current Resistor[Ohms],100", "***End_Of_Header***",
             ",,Chan-1,Chan-1,", "Loop,Frequency[Hz],Magnitude[ratio],Phase_Shift[rad],"]
    lines += [f"1,{f},{r},{np.angle(zz)}," for f, r, zz in zip(freqs, ratio, z)]
    return "\n".join(lines).encode()


def test_theoretical_phase_value_and_units():
    # 1000 µS/cm = 0.1 S/m; at 10 kHz: arctan(81·ε0·2π·1e4 / 0.1) ≈ 0.4506 mrad
    assert SDAT.theoretical_fluid_phase_mrad(1e4, 1000) == pytest.approx(0.4506, rel=1e-3)
    # Phase grows with frequency and falls with conductivity
    assert SDAT.theoretical_fluid_phase_mrad(1e3, 1000) < SDAT.theoretical_fluid_phase_mrad(1e4, 1000)
    assert SDAT.theoretical_fluid_phase_mrad(1e4, 2000) < SDAT.theoretical_fluid_phase_mrad(1e4, 1000)


def test_water_measurement_lines_up_with_theory():
    freqs = np.logspace(-2, 4, 25)
    df, ref, _, unit = SDAT.parse_sip_file(io.BytesIO(water_file(0.1, freqs)))
    df = SDAT.calculate_physics_properties(df, ref, 0.03, 0.002, unit)
    sigma = SDAT.measured_low_frequency_conductivity(df, "Chan-1", "Frequency[Hz]")
    assert sigma == pytest.approx(1000, rel=1e-6)
    dev = SDAT.fluid_phase_deviation(df["Frequency[Hz]"], df["Chan-1 Phase (mRads)"], sigma, 81, 0.05)
    assert dev['RMS deviation (mrad)'] < 1e-6
    assert dev['Within tolerance (%)'] == 100
    # A wrong sign convention would show up as a large deviation
    flipped = SDAT.fluid_phase_deviation(df["Frequency[Hz]"], -df["Chan-1 Phase (mRads)"], sigma, 81, 0.05)
    assert flipped['Max |deviation| (mrad)'] > 0.8


def test_overlay_adds_band_and_curve_per_row():
    fig = SDAT.make_subplots(rows=2, cols=1, shared_xaxes=True)
    SDAT.add_fluid_phase_overlay(fig, 0.01, 1e4, 1000, 81, 0.05, rows=[1, 2])
    assert len(fig.data) == 6
    assert [t.showlegend for t in fig.data] == [False, True, True, False, False, False]
    curve = fig.data[2]
    np.testing.assert_allclose(fig.data[0].y - curve.y, 0.05)
    np.testing.assert_allclose(curve.y - fig.data[1].y, 0.05)
    assert fig.data[3].yaxis == "y2"
