# SIP Data Analyzer Tool

A powerful web-based tool for automated analysis of Spectral Induced Polarization (SIP) measurement data. Features automatic format detection, multi-channel support, and interactive visualization.

## Overview

SIP Data Analyzer transforms tedious manual Excel workflows into a seamless, automated analysis pipeline. Upload your SIP measurement files and instantly get:

- Automated data parsing from multiple file formats
- Physics calculations (Resistance, Resistivity, Conductivity)
- Interactive multi-channel visualizations
- Multi-loop comparison and analysis
- CSV export of processed data

## Features

### Intelligent Format Detection
- **O&E PSIP Format**: Full instrument output with metadata, multi-channel support
- **Simple Table Format**: Basic frequency/magnitude/phase data files
- Automatic detection - no manual format selection needed

### Comprehensive Physics Calculations
- **Resistance** (Ω): Z = Magnitude × Reference Resistor
- **Resistivity** (Ω·m): ρ = Z × (Area / Length)
- **Conductivity** (mS/m): σ = 1000 / ρ
- Vectorized operations for fast computation

### Interactive Visualization
- Multi-channel plotting with customizable axes
- Loop filtering and comparison
- Dual Y-axis support for comparing different parameters
- Logarithmic scaling options
- Hover tooltips for precise value inspection

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
## Supported File Formats
    O&E PSIP Format (.csv)

**Features:**
- Metadata extraction (reference resistor, version, etc.)
- Multi-channel support (unlimited channels)
- Multiple measurement loops
- Timestamp information
