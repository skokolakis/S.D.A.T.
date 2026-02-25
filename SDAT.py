"""
SIP Data Analyzer - Adaptive Multi-Format Version
Automated analysis tool for Spectral Induced Polarization data
Supports multiple file formats with automatic detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import logging
import re
from typing import Tuple, Optional, Dict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_REF_RESISTOR = 1000.0  # Ohms
DEFAULT_SAMPLE_LENGTH = 0.03   # meters
DEFAULT_SAMPLE_AREA = 0.002    # square meters
HEADER_MARKER = '***End_Of_Header***'
RESISTOR_KEY = 'Current Resistor[Ohms]'
MIN_RESISTIVITY_THRESHOLD = 1e-6  # Ohm-m, minimum value to avoid division issues

# Page configuration
st.set_page_config(
    page_title="SDAT The SIP Data Analyzer Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# FILE FORMAT DETECTION
# ============================================================================

class FileFormat(Enum):
    """Enumeration of supported file formats."""
    OE_PSIP = "OE_PSIP"  # O&E PSIP format with header markers
    SIMPLE_TABLE = "SIMPLE_TABLE"  # Simple frequency/magnitude/phase table
    UNKNOWN = "UNKNOWN"


def detect_file_format(lines: list) -> FileFormat:
    """Detect the format of the uploaded file.
    
    Args:
        lines: List of file lines
        
    Returns:
        FileFormat enum value
    """
    # Check first 20 lines for format indicators
    header_content = '\n'.join(lines[:min(20, len(lines))])
    
    # Check for O&E PSIP format
    if HEADER_MARKER in header_content or 'OE PSIP Measurement' in header_content:
        logger.info("Detected format: OE PSIP")
        return FileFormat.OE_PSIP
    
    # Check for simple table format (has Freq[Hz] or similar in first few lines)
    if any(re.search(r'freq.*hz', line, re.IGNORECASE) for line in lines[:5]):
        logger.info("Detected format: SIMPLE_TABLE")
        return FileFormat.SIMPLE_TABLE
    
    logger.warning("Unknown file format")
    return FileFormat.UNKNOWN


# ============================================================================
# SIMPLE TABLE PARSER
# ============================================================================

def parse_simple_table(lines: list) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Parse simple table format files (like petrelaio_nero_12_10.txt).
    
    Expected format:
        Freq[Hz]    Mag, Resp,[Ratio]    Phase Resp[rad]
        ==========================================================
        1000,000000    0,069830    -0,838229
        ...
    
    Args:
        lines: List of file lines
        
    Returns:
        Tuple of (DataFrame, resistor_value)
    """
    logger.info("Parsing simple table format")
    
    # Find header line (contains "Freq" and "Mag" or "Phase")
    header_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'freq.*hz', line, re.IGNORECASE):
            header_idx = i
            break
    
    if header_idx == -1:
        st.error("Could not find header line with frequency column")
        return None, None
    
    # Find data start (skip separator lines with ===)
    data_start_idx = header_idx + 1
    while data_start_idx < len(lines):
        if '=' in lines[data_start_idx] or not lines[data_start_idx].strip():
            data_start_idx += 1
        else:
            break
    
    # Parse header to get column names
    header_line = lines[header_idx]
    # Split by tabs or multiple spaces
    header_parts = re.split(r'\t+|\s{2,}', header_line.strip())
    header_parts = [h.strip() for h in header_parts if h.strip()]
    
    logger.info(f"Found columns: {header_parts}")
    
    # Read data - handle both comma and dot decimal separators
    data_rows = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line or '=' in line:  # Skip empty lines and separators
            continue
        
        # Replace comma with dot for decimals, then split by tabs or multiple spaces
        line_normalized = line.replace(',', '.')
        values = re.split(r'\t+|\s{2,}', line_normalized)
        values = [v.strip() for v in values if v.strip()]
        
        if len(values) >= len(header_parts):
            data_rows.append(values[:len(header_parts)])
    
    if not data_rows:
        st.error("No data rows found in file")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=header_parts)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Standardize column names for consistency
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'freq' in col_lower:
            column_mapping[col] = 'Frequency[Hz]'
        elif 'mag' in col_lower and 'ratio' in col_lower:
            column_mapping[col] = 'Chan-1 Magnitude[ratio]'
        elif 'phase' in col_lower and 'rad' in col_lower:
            column_mapping[col] = 'Chan-1 Phase_Shift[rad]'
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Add Loop column (simple format typically has only one measurement)
    df['Loop'] = '1'
    
    logger.info(f"Parsed {len(df)} rows with columns: {df.columns.tolist()}")
    
    # No reference resistor in simple format
    return df, DEFAULT_REF_RESISTOR


# ============================================================================
# OE PSIP PARSER
# ============================================================================

def find_header_end(lines: list) -> int:
    """Find the index of the LAST header end marker in the file.
    
    Some files have multiple header sections, so we need the last one
    which precedes the actual data.
    """
    last_marker_idx = -1
    for i, line in enumerate(lines):
        if HEADER_MARKER in line:
            last_marker_idx = i
    return last_marker_idx


def extract_resistor_value(lines: list) -> Tuple[float, bool]:
    """Extract reference resistor value from file metadata."""
    for line in lines:
        if RESISTOR_KEY in line:
            try:
                value = float(line.split(',')[1])
                logger.info(f"Found reference resistor: {value} Ohms")
                return value, True
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse resistor value: {e}")
    
    logger.warning(f"Resistor value not found, using default: {DEFAULT_REF_RESISTOR} Ohms")
    return DEFAULT_REF_RESISTOR, False


def build_column_headers(channel_line: str, column_line: str) -> list:
    """Build combined column headers from channel and column name lines."""
    channel_tokens = channel_line.strip().split(',')
    col_tokens = column_line.strip().split(',')
    
    # Align lengths by padding with empty strings
    max_len = max(len(channel_tokens), len(col_tokens))
    channel_tokens += [''] * (max_len - len(channel_tokens))
    col_tokens += [''] * (max_len - len(col_tokens))
    
    # Merge Channel + Column Name (e.g., "Chan-3 Magnitude[ratio]")
    unique_columns = []
    for ch, col in zip(channel_tokens, col_tokens):
        col = col.strip()
        ch = ch.strip()
        if ch:
            unique_columns.append(f"{ch} {col}")
        else:
            unique_columns.append(col)
    
    # Remove empty column names (caused by trailing commas)
    unique_columns = [c for c in unique_columns if c]
    
    return unique_columns


def parse_oe_psip_format(lines: list) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Parse O&E PSIP format files (like H3_Almyros.csv).
    
    Args:
        lines: List of file lines
        
    Returns:
        Tuple of (DataFrame, resistor_value)
    """
    logger.info("Parsing O&E PSIP format")
    
    # Find header end marker
    header_end_index = find_header_end(lines)
    if header_end_index == -1:
        st.error(f"Could not find '{HEADER_MARKER}' marker in file. "
                "This may not be a valid O&E PSIP file.")
        return None, None
    
    # Extract reference resistor value
    ref_resistor, resistor_found = extract_resistor_value(lines)
    if not resistor_found:
        st.warning(f"Could not find resistor value in file. Using default {DEFAULT_REF_RESISTOR} Ohms")
    
    # Identify key line positions
    channel_line_idx = header_end_index + 1
    col_name_line_idx = header_end_index + 2
    data_start_idx = header_end_index + 3
    
    # Build column headers
    if channel_line_idx >= len(lines) or col_name_line_idx >= len(lines):
        st.error("File structure is incomplete. Missing channel or column name lines.")
        return None, None
    
    unique_columns = build_column_headers(
        lines[channel_line_idx],
        lines[col_name_line_idx]
    )
    
    # Check data consistency - trim headers if needed
    if data_start_idx < len(lines):
        first_data_line = lines[data_start_idx].strip().split(',')
        num_data_cols = len(first_data_line)  # Count all values including empty ones
        
        logger.info(f"Headers: {len(unique_columns)} columns, Data: {num_data_cols} columns")
        
        # If data has fewer columns than headers, trim headers
        if num_data_cols < len(unique_columns):
            logger.warning(f"Trimming headers from {len(unique_columns)} to {num_data_cols}")
            unique_columns = unique_columns[:num_data_cols]
    
    # Load data section
    stringio = io.StringIO('\n'.join(lines))
    try:
        df = pd.read_csv(
            stringio,
            skiprows=data_start_idx,
            header=None,
            names=unique_columns,
            usecols=range(len(unique_columns))
        )
    except Exception as e:
        st.error(f"Error reading CSV data: {e}")
        logger.error(f"CSV parsing error: {e}")
        return None, None
    
    # Convert numeric columns to proper types
    # Skip columns that should remain as strings (Loop, Comment, etc.)
    skip_columns = ['Loop', 'Comment', 'User_Comment']
    
    for col in df.columns:
        if col not in skip_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Parsed {len(df)} rows, {len(df.columns)} columns")
    return df, ref_resistor


# ============================================================================
# UNIFIED PARSER
# ============================================================================

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that the parsed dataframe contains expected SIP data."""
    if df.empty:
        st.error('No valid data rows found in file. The file may be empty or corrupted.')
        return False
    
    # Check for Frequency column (required) - case insensitive
    freq_cols = [col for col in df.columns if 'frequency' in str(col).lower()]
    if not freq_cols:
        st.error(f'No Frequency column found. Available columns: {df.columns.tolist()[:10]}')
        return False
    
    # Check for at least one Magnitude or Phase column
    has_mag = any('magnitude' in str(col).lower() for col in df.columns)
    has_phase = any('phase' in str(col).lower() for col in df.columns)
    
    if not (has_mag or has_phase):
        st.warning('No Magnitude or Phase columns found. Data visualization only (no physics calculations).')
    
    logger.info(f"Validated dataframe: {len(df)} rows, {len(df.columns)} columns")
    return True


@st.cache_data
def parse_sip_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
    """Parse SIP file with automatic format detection.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (DataFrame with parsed data, reference resistor value, format type)
        Returns (None, None, None) if parsing fails
    """
    # Read file content
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    except UnicodeDecodeError:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("latin-1"))
        except Exception as e:
            st.error(
                "Unable to decode file. The file may be corrupted or not a valid text-based file. "
                "Try opening it in a text editor to verify the format."
            )
            logger.error(f"File decoding failed: {e}")
            return None, None, None
    
    lines = stringio.readlines()
    
    if not lines:
        st.error("File is empty")
        return None, None, None
    
    # Detect file format
    file_format = detect_file_format(lines)
    
    # Parse based on detected format
    if file_format == FileFormat.OE_PSIP:
        df, ref_resistor = parse_oe_psip_format(lines)
        format_name = "O&E PSIP Format"
    elif file_format == FileFormat.SIMPLE_TABLE:
        df, ref_resistor = parse_simple_table(lines)
        format_name = "Simple Table Format"
    else:
        st.error(
            "Unknown file format. Supported formats:\n"
            "- O&E PSIP files (with ***End_Of_Header*** marker)\n"
            "- Simple table files (Freq/Magnitude/Phase columns)"
        )
        return None, None, None
    
    if df is None:
        return None, None, None
    
    # Validate the parsed data
    if not validate_dataframe(df):
        return None, None, None
    
    logger.info(f"Successfully parsed file as {format_name}")
    return df, ref_resistor, format_name


# ============================================================================
# PHYSICS CALCULATIONS
# ============================================================================

def calculate_physics_properties(
    sip_data: pd.DataFrame,
    reference_resistor: float,
    sample_length: float,
    sample_area: float
) -> pd.DataFrame:
    """Calculate resistance, resistivity, and conductivity for all channels.
    
    Uses formulas according the Excel worksheet:
    - Resistance = Magnitude × Reference Resistor
    - Resistivity = (Resistance × Area) / Length
    - Fluid Conductivity = (1 / Resistivity) × 10000 (μS/cm)
    - Real Conductivity = (1 / Resistivity) × cos(Phase) (S/m)
    - Imaginary Conductivity = -(1 / Resistivity) × sin(Phase) (S/m)
    - Cell Constant K = 1,000,000 × 0.1606 / Resistance
    - Phase in milliradians = -Phase × 1000
    
    Args:
        sip_data: DataFrame with raw SIP data
        reference_resistor: Reference resistor value in Ohms
        sample_length: Sample length in meters
        sample_area: Sample cross-sectional area in square meters
        
    Returns:
        DataFrame with calculated properties added
    """
    # Find all Magnitude columns to calculate physics for each channel
    magnitude_columns = [c for c in sip_data.columns if 'Magnitude[ratio]' in c]
    
    if not magnitude_columns:
        logger.warning("No magnitude columns found for calculations")
        st.info("No Magnitude[ratio] columns found. Skipping physics calculations. "
               "The file may only contain frequency sweep data.")
        return sip_data
    
    logger.info(f"Calculating physics properties for {len(magnitude_columns)} channels")
    
    # Cell constant factor (geometric factor for the measurement cell)
    CELL_CONSTANT_FACTOR = 0.1606
    
    for mag_col in magnitude_columns:
        # Extract prefix (e.g., "Chan-1")
        prefix = mag_col.replace(' Magnitude[ratio]', '')
        
        # Find corresponding Phase column
        phase_col = f"{prefix} Phase_Shift[rad]"
        has_phase = phase_col in sip_data.columns
        
        # D: Resistance (Ω) = Magnitude × Reference Resistor
        resistance_col = f"{prefix} Resistance (Ohms)"
        sip_data[resistance_col] = sip_data[mag_col] * reference_resistor
        
        # E: Resistivity (Ω·m) = (Resistance × Area) / Length
        resistivity_col = f"{prefix} Resistivity (Ohm-m)"
        sip_data[resistivity_col] = (sip_data[resistance_col] * sample_area) / sample_length
        
        # F: Fluid Conductivity (μS/cm) = (1 / Resistivity) × 10000
        fluid_cond_col = f"{prefix} Fluid Conductivity (uS/cm)"
        sip_data[fluid_cond_col] = np.where(
            sip_data[resistivity_col] > MIN_RESISTIVITY_THRESHOLD,
            (1 / sip_data[resistivity_col]) * 10000,
            np.nan
        )
        
        # J: Cell Constant K = 1,000,000 × 0.1606 / Resistance
        k_col = f"{prefix} Cell Constant K"
        sip_data[k_col] = np.where(
            sip_data[resistance_col] > 1e-10,
            1000000 * CELL_CONSTANT_FACTOR / sip_data[resistance_col],
            np.nan
        )
        
        # Phase-dependent calculations (only if phase data available)
        if has_phase:
            # G: Phase in milliradians = -Phase × 1000
            phase_mrad_col = f"{prefix} Phase (mRads)"
            sip_data[phase_mrad_col] = -sip_data[phase_col] * 1000
            
            # H: Imaginary Conductivity (S/m) = -(1 / Resistivity) × sin(Phase)
            imag_cond_col = f"{prefix} Imaginary Conductivity (S/m)"
            sip_data[imag_cond_col] = np.where(
                sip_data[resistivity_col] > MIN_RESISTIVITY_THRESHOLD,
                -(1 / sip_data[resistivity_col]) * np.sin(sip_data[phase_col]),
                np.nan
            )
            
            # I: Real Conductivity (S/m) = (1 / Resistivity) × cos(Phase)
            real_cond_col = f"{prefix} Real Conductivity (S/m)"
            sip_data[real_cond_col] = np.where(
                sip_data[resistivity_col] > MIN_RESISTIVITY_THRESHOLD,
                (1 / sip_data[resistivity_col]) * np.cos(sip_data[phase_col]),
                np.nan
            )
            
            # K: Imaginary Conductivity in μS/cm = Imaginary Conductivity × 10000
            imag_cond_uscm_col = f"{prefix} Imaginary Conductivity (uS/cm)"
            sip_data[imag_cond_uscm_col] = sip_data[imag_cond_col] * 10000
    
    return sip_data

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic."""
    
    st.title("SDAT The SIP Data Analysis Tool")
    st.markdown("*Automated analysis of Spectral Induced Polarization measurements*")
    st.caption("Supports multiple file formats with automatic detection")
    
    # Sidebar - File Upload and Settings
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader(
            "Upload SIP file",
            type=["csv", "txt"],
            help="Supported formats: O&E PSIP CSV, Simple frequency tables"
        )
        
        st.header("2. Sample Properties")
        
        sample_length = st.number_input(
            "Sample Length (m)",
            value=DEFAULT_SAMPLE_LENGTH,
            format="%.4f",
            min_value=0.0001,
            max_value=10.0,
            help="Length of the sample in meters"
        )
        
        sample_area = st.number_input(
            "Sample Area (m²)",
            value=DEFAULT_SAMPLE_AREA,
            format="%.4f",
            min_value=0.0001,
            max_value=1.0,
            help="Cross-sectional area of the sample in square meters"
        )
        
        # Validate sample dimensions
        if sample_length <= 0 or sample_area <= 0:
            st.error("Sample dimensions must be positive values")
            st.stop()
        
        # Reference resistor override
        manual_ref = st.number_input(
            "Reference Resistor (Ohms)",
            value=0.0,
            help="Leave 0 to use the value from the file. Override only if needed.",
            min_value=0.0,
            max_value=1000000.0
        )
        
        st.divider()
        st.caption("Tip: Hover over any plot point to see exact values")
    
    # Main content area
    if not uploaded_file:
        st.info("Please upload a SIP file to begin analysis")
        
        # Show example/instructions
        with st.expander("About this app"):
            st.markdown("""
            This app automates the analysis of Spectral Induced Polarization (SIP) data:
            
            **Features:**
            - Automatic parsing of multiple file formats
            - Physics calculations (Resistance, Resistivity, Conductivity)
            - Interactive multi-channel visualization
            - Support for multiple measurement loops
            - Customizable dual-axis plots
            
            **Supported File Formats:**
            - **O&E PSIP** (.csv) - Full instrument output with metadata
            - **Simple Tables** (.txt, .csv) - Frequency/Magnitude/Phase data
            
            **How to use:**
            1. Upload your SIP file (CSV or TXT)
            2. Verify/adjust sample properties
            3. Explore the interactive plots
            4. Export processed data if needed
            """)
        return
    
    # Parse the uploaded file
    sip_data, file_ref_resistor, format_name = parse_sip_file(uploaded_file)
    
    if sip_data is None:
        st.error("Failed to parse file. Please check the error messages above.")
        return
    
    # Display detected format
    st.success(f"Detected format: **{format_name}**")
    
    # Determine which resistor value to use
    reference_resistor = manual_ref if manual_ref > 0 else file_ref_resistor
    
    # Store in session state to avoid recalculation
    cache_key = f"{uploaded_file.name}_{sample_length}_{sample_area}_{reference_resistor}"
    
    if 'cache_key' not in st.session_state or st.session_state.cache_key != cache_key:
        st.info(f"Using Reference Resistor: **{reference_resistor:.1f} Ohms**")
        
        # Calculate physics properties
        with st.spinner("Calculating physics properties..."):
            sip_data = calculate_physics_properties(
                sip_data,
                reference_resistor,
                sample_length,
                sample_area
            )
        
        # Convert Loop to string for categorical plotting (if exists)
        if 'Loop' in sip_data.columns:
            sip_data['Loop'] = sip_data['Loop'].astype(str)
        
        # Store in session state
        st.session_state.processed_data = sip_data
        st.session_state.reference_resistor = reference_resistor
        st.session_state.cache_key = cache_key
    else:
        sip_data = st.session_state.processed_data
        reference_resistor = st.session_state.reference_resistor
    
    # Display data summary
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Total Measurements", len(sip_data))
    with col_info2:
        if 'Loop' in sip_data.columns:
            st.metric("Measurement Loops", len(sip_data['Loop'].unique()))
        else:
            st.metric("Measurement Loops", "N/A")
    with col_info3:
        channels = [col.split()[0] for col in sip_data.columns if 'Magnitude' in col]
        st.metric("Channels", len(channels) if channels else 1)
    
    # Visualization Section
    st.divider()
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Plot Settings")
        
        all_cols = sip_data.columns.tolist()
        
        # Safer default selection
        default_x = next(
            (i for i, col in enumerate(all_cols) if 'frequency' in col.lower()),
            0
        )
        default_y = max(0, len(all_cols) - 1)
        
        x_axis = st.selectbox(
            "X-Axis",
            options=all_cols,
            index=default_x,
            help="Select the parameter for the horizontal axis"
        )
        
        y_axis = st.selectbox(
            "Y-Axis",
            options=all_cols,
            index=default_y,
            help="Select the parameter for the vertical axis"
        )
        
        # Loop Selection (if applicable)
        if 'Loop' in sip_data.columns:
            loops = sorted(sip_data['Loop'].unique())
            selected_loops = st.multiselect(
                "Select Loops",
                loops,
                default=loops,
                help="Choose which measurement loops to display"
            )
            
            # Only filter if necessary
            if len(selected_loops) < len(loops):
                plot_data = sip_data[sip_data['Loop'].isin(selected_loops)]
            else:
                plot_data = sip_data
        else:
            plot_data = sip_data
        
        log_x = st.checkbox("Log Scale X-Axis", value=True)
        log_y = st.checkbox("Log Scale Y-Axis", value=False)
    
    with col2:
        if plot_data.empty:
            st.warning("No data selected. Please select at least one loop.")
            return
        
        # Secondary Y-axis toggle
        secondary_enabled = st.checkbox(
            "Enable secondary Y-axis",
            value=False,
            help="Plot additional parameters on a second Y-axis"
        )
        
        # Get numeric columns for secondary axis
        numeric_cols = plot_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Helper to extract channel prefix
        def _prefix(colname):
            return colname.split(' ')[0] if isinstance(colname, str) and ' ' in colname else colname
        
        # Default secondary selection
        default_secondary = [
            c for c in numeric_cols
            if str(_prefix(c)).lower().startswith('chan')
            and not str(_prefix(c)).lower().startswith('chan-1')
        ][:3]  # Limit to 3 for readability
        
        if secondary_enabled:
            secondary_cols = st.multiselect(
                "Secondary Y Columns",
                options=numeric_cols,
                default=default_secondary,
                help="Select additional parameters to plot on the right Y-axis"
            )
        else:
            secondary_cols = []
        
        # Create plot
        if secondary_enabled and secondary_cols:
            # Dual-axis plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Primary trace
            fig.add_trace(
                go.Scatter(
                    x=plot_data[x_axis],
                    y=plot_data[y_axis],
                    mode='lines+markers',
                    name=y_axis,
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
            
            # Secondary traces
            for colname in secondary_cols:
                if colname == y_axis:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=plot_data[x_axis],
                        y=plot_data[colname],
                        mode='lines+markers',
                        name=colname,
                        marker=dict(size=6)
                    ),
                    secondary_y=True
                )
            
            # Apply log scales
            if log_x:
                fig.update_xaxes(type='log')
            if log_y:
                fig.update_yaxes(type='log', secondary_y=False)
                fig.update_yaxes(type='log', secondary_y=True)
            
            fig.update_layout(
                height=600,
                title=f"{y_axis} vs {x_axis}",
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Single-axis plot
            if secondary_enabled and not secondary_cols:
                st.warning("No secondary columns selected; displaying primary axis only.")
            
            fig = px.line(
                plot_data,
                x=x_axis,
                y=y_axis,
                color='Loop' if 'Loop' in plot_data.columns else None,
                markers=True,
                title=f"{y_axis} vs {x_axis}",
                log_x=log_x,
                log_y=log_y
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(
                height=600,
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data Table and Export
    st.divider()
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        with st.expander("View Processed Data Table"):
            st.dataframe(sip_data, use_container_width=True)
    
    with col_b:
        st.download_button(
            label="Download CSV",
            data=sip_data.to_csv(index=False).encode('utf-8'),
            file_name=f"sip_processed_{uploaded_file.name}",
            mime="text/csv",
            help="Download the processed data with all calculated properties"
        )


if __name__ == "__main__":
    main()
