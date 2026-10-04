# SIP Data Analyzer Tool

A powerful web-based tool for automated analysis of Spectral Induced Polarization (SIP) measurement data. Features automatic format detection, multi-channel support, and interactive visualization.

The tool can be also be used online via **https://sdatsite.streamlit.app**

## Overview

SIP Data Analyzer transforms tedious manual Excel workflows into a seamless, automated analysis pipeline. Upload your SIP measurement files and instantly get:

- Automated data parsing from multiple file formats
- Physics calculations (Resistance, Resistivity, Conductivity)
- Interactive multi-channel visualizations
- Multi-loop comparison and analysis
- CSV export of processed data (including the sample geometry used for each file)

## Features

### Intelligent Format Detection
- **O&E PSIP Format**: Full instrument output with metadata, multi-channel support
- **Simple Table Format**: Basic frequency/magnitude/phase data files
- Automatic detection - no manual format selection needed

### Interface
- Sidebar for data and sample/instrument settings; results in Data, Spectra and Debye decomposition tabs
- Quantities shown with their symbols and units (|ρ| (Ω·m), −φ (mrad), σ′/σ″) while CSV columns keep their names
- Light and dark themes (`.streamlit/config.toml`), switched in the app menu (⋮ → Settings → Choose app theme); figures follow the theme with publication-style presets: boxed axes, decade ticks as powers of ten, colour-blind safe palette (Okabe & Ito)
- Every scientific setting and result column has a ? tooltip explaining it

### Interactive Visualization
- Multi-channel plotting with customizable axes
- Loop filtering and comparison
- SIP two-panel plot: phase on top, magnitude/conductivity below, on a shared log-frequency axis
- Logarithmic scaling options
- Hover tooltips for precise value inspection

### Debye Decomposition
- Fits ρ(ω) = ρ0·[1 − Σ m_k·(1 − 1/(1 + iωτ_k))] to every file/channel/loop spectrum on a log-spaced τ grid (measured range ± 1 decade)
- Smoothness-regularised non-negative least squares on the real and imaginary parts; λ is chosen automatically or set manually
- Shows the fitted curve over the measured phase and magnitude with the RMS misfit, and the relaxation-time distribution m(τ)
- Integral parameters (Weigand & Kemna, 2016): ρ0, m_tot, m_tot_n, τ_mean, τ_10/τ_50/τ_60, U_tau = τ_60/τ_10 and τ peaks, exportable as CSV
- Implemented from the published equations with SciPy (no GPL code, no extra dependencies)

### Fluid Calibration Check
- Overlay the theoretical fluid phase φ(ω) = arctan(ε_r·ε0·ω/σ) (in mrad, same sign as `Phase (mRads)`) with a ± tolerance band
- Fluid conductivity defaults to the measured low-frequency value; ε_r (default 81) and tolerance (default ±0.05 mrad) are adjustable
- Reports the RMS and maximum deviation of each measured phase spectrum from the theory

### Sample Geometry
- Enter the sample cross-section as an area or as a holder diameter (area = π(d/2)²)
- In comparison mode, set length and diameter/area per file, so files measured in different holders get correct resistivities

### Robust Data Handling
- Handles files with multiple header sections
- European decimal format support (comma to dot conversion)
- Flexible separator detection (tabs, spaces)
- Empty column handling
- Automatic data type conversion

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
numpy>=1.24.0
```
## Running the tests

```
pip install pytest
python -m pytest
```

## Supported File Formats
    O&E PSIP Format (.csv)

**Features:**
- Metadata extraction (reference resistor, version, etc.)
- Magnitude as a ratio or in dB (`Magnitude[dB]` columns or `Writer_Version,2` files: R = R_ref · 10^(mag/20)), auto-detected with a manual override in the sidebar
- Multi-channel support (unlimited channels)
- Multiple measurement loops
- Timestamp information
