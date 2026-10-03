"""Tests for sample geometry handling (issue #10)."""
import io

import numpy as np
import pytest

import SDAT
from test_parser import oe_psip_file


def test_diameter_gives_expected_area():
    assert SDAT.area_from_diameter(0.0508) == pytest.approx(0.0020268, rel=1e-4)


def test_files_with_different_geometry_side_by_side():
    raw = oe_psip_file().encode()
    geometries = {
        "c1.csv": {'length': 0.03, 'area': SDAT.area_from_diameter(0.0508), 'diameter': 0.0508},
        "c3.csv": {'length': 0.12, 'area': SDAT.area_from_diameter(0.0508), 'diameter': 0.0508},
    }
    resistivities = {}
    for name, geom in geometries.items():
        df, ref, _, unit = SDAT.parse_sip_file(io.BytesIO(raw))
        df = SDAT.calculate_physics_properties(df, ref, geom['length'], geom['area'], unit)
        resistivities[name] = df["Chan-1 Resistivity (Ohm-m)"].to_numpy()
        expected = 1.5 * 100 * geom['area'] / geom['length']
        np.testing.assert_allclose(resistivities[name], expected)
    # Same resistance, 4x the length -> a quarter of the resistivity
    np.testing.assert_allclose(resistivities["c3.csv"], resistivities["c1.csv"] / 4)


def test_export_includes_geometry():
    df = SDAT.pd.DataFrame({"Frequency[Hz]": [1.0, 10.0]})
    out = SDAT.with_geometry_columns(df, {'length': 0.06, 'area': 0.002, 'diameter': None})
    assert out["Sample Length (m)"].tolist() == [0.06, 0.06]
    assert out["Sample Area (m²)"].tolist() == [0.002, 0.002]
    assert "Sample Diameter (m)" not in out.columns
    out = SDAT.with_geometry_columns(df, {'length': 0.06, 'area': 0.002027, 'diameter': 0.0508})
    assert out["Sample Diameter (m)"].tolist() == [0.0508, 0.0508]
    assert "Sample Length (m)" not in df.columns
