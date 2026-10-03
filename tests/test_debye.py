"""Tests for the Debye decomposition (issue #6)."""
import numpy as np
import pandas as pd
import pytest

import SDAT

FREQS = np.logspace(-2, 4, 31)


def cole_cole(rho0, m, tau, c, freqs=FREQS):
    iwt = 1j * 2 * np.pi * freqs * tau
    return rho0 * (1 - m * (1 - 1 / (1 + iwt ** c)))


def add_noise(rho, seed=0):
    rng = np.random.default_rng(seed)
    magnitude = np.abs(rho) * (1 + rng.normal(0, 1e-3, rho.size))   # 0.1 %
    phase = np.angle(rho) + rng.normal(0, 1e-4, rho.size)          # 0.1 mrad
    return magnitude, phase


@pytest.mark.parametrize("rho0, m, tau, c", [
    (100.0, 0.10, 1e-2, 0.5),
    (50.0, 0.20, 1e-3, 0.8),
    (20.0, 0.05, 1e-1, 0.6),
])
def test_recovers_cole_cole_parameters(rho0, m, tau, c):
    magnitude, phase = add_noise(cole_cole(rho0, m, tau, c))
    result = SDAT.debye_decomposition(FREQS, magnitude, phase)
    p = result['parameters']
    assert p['rho0 (Ohm-m)'] == pytest.approx(rho0, rel=0.01)
    assert p['m_tot'] == pytest.approx(m, rel=0.05)
    assert p['m_tot_n (S/m)'] == pytest.approx(m / rho0, rel=0.06)
    # A symmetric Cole-Cole distribution is centred on τ: within 0.1 decade
    for key in ('tau_50 (s)', 'tau_mean (s)', 'tau_peak1 (s)'):
        assert abs(np.log10(p[key] / tau)) < 0.1, key
    assert p['n_peaks'] == 1
    # The fit explains the data down to the noise level
    assert p['RMS |rho| misfit (%)'] < 0.2
    assert p['RMS phase misfit (mrad)'] < 0.2


def test_single_debye_term_and_two_peaks():
    debye = SDAT.debye_decomposition(FREQS, np.abs(cole_cole(10, 0.1, 1e-2, 1.0)),
                                     np.angle(cole_cole(10, 0.1, 1e-2, 1.0)))
    assert abs(np.log10(debye['parameters']['tau_peak1 (s)'] / 1e-2)) < 0.1

    rho = 100 * (1 - 0.05 * (1 - 1 / (1 + (1j * 2 * np.pi * FREQS * 1.0) ** 0.7))
                 - 0.05 * (1 - 1 / (1 + (1j * 2 * np.pi * FREQS * 1e-3) ** 0.7)))
    p = SDAT.debye_decomposition(FREQS, *add_noise(rho))['parameters']
    assert p['n_peaks'] == 2
    peaks = [float(v) for v in p['tau_peaks (s)'].split(';')]
    assert abs(np.log10(peaks[0] / 1.0)) < 0.2 and abs(np.log10(peaks[1] / 1e-3)) < 0.2
    assert p['U_tau'] > 10  # a broad, bimodal distribution


def test_integral_parameters_of_known_distribution():
    tau = np.logspace(-4, 0, 41)
    m = np.zeros_like(tau)
    m[[10, 30]] = [0.03, 0.01]   # τ = 1e-3 s and 1e-1 s
    p = SDAT.dd_integral_parameters(tau, m, 50.0)
    assert p['m_tot'] == pytest.approx(0.04)
    assert p['m_tot_n (S/m)'] == pytest.approx(0.04 / 50)
    assert p['tau_mean (s)'] == pytest.approx(10 ** (0.75 * -3 + 0.25 * -1))
    assert p['tau_peak1 (s)'] == pytest.approx(0.1)
    assert p['n_peaks'] == 2


def test_forward_model_matches_definition():
    tau, m = np.array([1e-2]), np.array([0.1])
    rho = SDAT.dd_forward(FREQS, 100, m, tau)
    np.testing.assert_allclose(rho, cole_cole(100, 0.1, 1e-2, 1.0))


def test_fit_dataset_per_loop_from_processed_columns():
    rows = []
    for loop, (m, tau) in {"1": (0.1, 1e-2), "2": (0.05, 1e-1)}.items():
        rho = cole_cole(80, m, tau, 0.6)
        rows.append(pd.DataFrame({
            'Loop': loop, 'Frequency[Hz]': FREQS,
            'Chan-1 Resistivity (Ohm-m)': np.abs(rho), 'Chan-1 Phase_Shift[rad]': np.angle(rho),
        }))
    df = pd.concat(rows, ignore_index=True)
    results = SDAT.fit_debye_decomposition(df, "Chan-1")
    assert list(results) == ["1", "2"]
    assert results["1"]['parameters']['m_tot'] == pytest.approx(0.1, rel=0.05)
    assert results["2"]['parameters']['m_tot'] == pytest.approx(0.05, rel=0.05)
    # Frequency limits restrict the fitted data
    limited = SDAT.fit_debye_decomposition(df, "Chan-1", f_min=0.1, f_max=1000)
    assert limited["1"]['parameters']['n_freqs'] == 21
    table = SDAT.dd_results_table({("a.csv", "Chan-1"): results})
    assert {"File", "Channel", "Loop", "m_tot", "tau_50 (s)", "U_tau"} <= set(table.columns)
    assert len(table) == 2
