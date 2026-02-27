"""
SIP Data Analyzer - Multi-File Comparison Version
Automated analysis tool for Spectral Induced Polarization data
Supports multiple file formats with automatic detection and comparison
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
from typing import Tuple, Optional, Dict, List
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REF_RESISTOR = 100.0  # Ohms (updated to match Excel)
DEFAULT_SAMPLE_LENGTH = 0.03   # meters
DEFAULT_SAMPLE_AREA = 0.002    # square meters
HEADER_MARKER = '***End_Of_Header***'
RESISTOR_KEY = 'Current Resistor[Ohms]'
MIN_RESISTIVITY_THRESHOLD = 1e-6  # Ohm-m
CELL_CONSTANT_FACTOR = 0.1606  # Geometric factor for K calculation

# Page configuration
st.set_page_config(
    page_title="SIP Data Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)


class FileFormat(Enum):
    """Enumeration of supported file formats."""
    OE_PSIP = "OE_PSIP"
    SIMPLE_TABLE = "SIMPLE_TABLE"
    UNKNOWN = "UNKNOWN"


def detect_file_format(lines: list) -> FileFormat:
    """Detect the format of the uploaded file."""
    header_content = '\n'.join(lines[:min(20, len(lines))])
    
    if HEADER_MARKER in header_content or 'OE PSIP Measurement' in header_content:
        logger.info("Detected format: OE PSIP")
        return FileFormat.OE_PSIP
    
    if any(re.search(r'freq.*hz', line, re.IGNORECASE) for line in lines[:5]):
        logger.info("Detected format: SIMPLE_TABLE")
        return FileFormat.SIMPLE_TABLE
    
    logger.warning("Unknown file format")
    return FileFormat.UNKNOWN


def find_header_end(lines: list) -> int:
    """Find the index of the LAST header end marker in the file."""
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
    
    max_len = max(len(channel_tokens), len(col_tokens))
    channel_tokens += [''] * (max_len - len(channel_tokens))
    col_tokens += [''] * (max_len - len(col_tokens))
    
    unique_columns = []
    for ch, col in zip(channel_tokens, col_tokens):
        col = col.strip()
        ch = ch.strip()
        if ch:
            unique_columns.append(f"{ch} {col}")
        else:
            unique_columns.append(col)
    
    unique_columns = [c for c in unique_columns if c]
    return unique_columns


def parse_simple_table(lines: list) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Parse simple table format files."""
    logger.info("Parsing simple table format")
    
    header_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'freq.*hz', line, re.IGNORECASE):
            header_idx = i
            break
    
    if header_idx == -1:
        st.error("Could not find header line with frequency column")
        return None, None
    
    data_start_idx = header_idx + 1
    while data_start_idx < len(lines):
        if '=' in lines[data_start_idx] or not lines[data_start_idx].strip():
            data_start_idx += 1
        else:
            break
    
    header_line = lines[header_idx]
    header_parts = re.split(r'\t+|\s{2,}', header_line.strip())
    header_parts = [h.strip() for h in header_parts if h.strip()]
    
    logger.info(f"Found columns: {header_parts}")
    
    data_rows = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line or '=' in line:
            continue
        
        line_normalized = line.replace(',', '.')
        values = re.split(r'\t+|\s{2,}', line_normalized)
        values = [v.strip() for v in values if v.strip()]
        
        if len(values) >= len(header_parts):
            data_rows.append(values[:len(header_parts)])
    
    if not data_rows:
        st.error("No data rows found in file")
        return None, None
    
    df = pd.DataFrame(data_rows, columns=header_parts)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    df['Loop'] = '1'
    
    logger.info(f"Parsed {len(df)} rows with columns: {df.columns.tolist()}")
    return df, DEFAULT_REF_RESISTOR


def parse_oe_psip_format(lines: list) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Parse O&E PSIP format files."""
    logger.info("Parsing O&E PSIP format")
    
    header_end_index = find_header_end(lines)
    if header_end_index == -1:
        st.error(f"Could not find '{HEADER_MARKER}' marker in file.")
        return None, None
    
    ref_resistor, resistor_found = extract_resistor_value(lines)
    if not resistor_found:
        st.warning(f"Could not find resistor value in file. Using default {DEFAULT_REF_RESISTOR} Ohms")
    
    channel_line_idx = header_end_index + 1
    col_name_line_idx = header_end_index + 2
    data_start_idx = header_end_index + 3
    
    if channel_line_idx >= len(lines) or col_name_line_idx >= len(lines):
        st.error("File structure is incomplete.")
        return None, None
    
    unique_columns = build_column_headers(
        lines[channel_line_idx],
        lines[col_name_line_idx]
    )
    
    if data_start_idx < len(lines):
        first_data_line = lines[data_start_idx].strip().split(',')
        num_data_cols = len(first_data_line)
        
        logger.info(f"Headers: {len(unique_columns)} columns, Data: {num_data_cols} columns")
        
        if num_data_cols < len(unique_columns):
            logger.warning(f"Trimming headers from {len(unique_columns)} to {num_data_cols}")
            unique_columns = unique_columns[:num_data_cols]
    
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
    
    skip_columns = ['Loop', 'Comment', 'User_Comment']
    for col in df.columns:
        if col not in skip_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Parsed {len(df)} rows, {len(df.columns)} columns")
    return df, ref_resistor


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that the parsed dataframe contains expected SIP data."""
    if df.empty:
        st.error('No valid data rows found in file.')
        return False
    
    freq_cols = [col for col in df.columns if 'frequency' in str(col).lower()]
    if not freq_cols:
        st.error(f'No Frequency column found. Available columns: {df.columns.tolist()[:10]}')
        return False
    
    has_mag = any('magnitude' in str(col).lower() for col in df.columns)
    has_phase = any('phase' in str(col).lower() for col in df.columns)
    
    if not (has_mag or has_phase):
        st.warning('No Magnitude or Phase columns found.')
    
    logger.info(f"Validated dataframe: {len(df)} rows, {len(df.columns)} columns")
    return True


@st.cache_data
def parse_sip_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
    """Parse SIP file with automatic format detection."""
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    except UnicodeDecodeError:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("latin-1"))
        except Exception as e:
            st.error("Unable to decode file.")
            logger.error(f"File decoding failed: {e}")
            return None, None, None
    
    lines = stringio.readlines()
    
    if not lines:
        st.error("File is empty")
        return None, None, None
    
    file_format = detect_file_format(lines)
    
    if file_format == FileFormat.OE_PSIP:
        df, ref_resistor = parse_oe_psip_format(lines)
        format_name = "O&E PSIP Format"
    elif file_format == FileFormat.SIMPLE_TABLE:
        df, ref_resistor = parse_simple_table(lines)
        format_name = "Simple Table Format"
    else:
        st.error("Unknown file format.")
        return None, None, None
    
    if df is None:
        return None, None, None
    
    if not validate_dataframe(df):
        return None, None, None
    
    logger.info(f"Successfully parsed file as {format_name}")
    return df, ref_resistor, format_name


def calculate_physics_properties(
    sip_data: pd.DataFrame,
    reference_resistor: float,
    sample_length: float,
    sample_area: float
) -> pd.DataFrame:
    """Calculate resistance, resistivity, and conductivity for all channels."""
    
    magnitude_columns = [c for c in sip_data.columns if 'Magnitude[ratio]' in c]
    
    if not magnitude_columns:
        logger.warning("No magnitude columns found for calculations")
        st.info("No Magnitude[ratio] columns found. Skipping physics calculations.")
        return sip_data
    
    logger.info(f"Calculating physics properties for {len(magnitude_columns)} channels")
    
    for mag_col in magnitude_columns:
        prefix = mag_col.replace(' Magnitude[ratio]', '')
        phase_col = f"{prefix} Phase_Shift[rad]"
        has_phase = phase_col in sip_data.columns
        
        # Resistance
        resistance_col = f"{prefix} Resistance (Ohms)"
        sip_data[resistance_col] = sip_data[mag_col] * reference_resistor
        
        # Resistivity
        resistivity_col = f"{prefix} Resistivity (Ohm-m)"
        sip_data[resistivity_col] = (sip_data[resistance_col] * sample_area) / sample_length
        
        # Fluid Conductivity
        fluid_cond_col = f"{prefix} Fluid Conductivity (uS/cm)"
        sip_data[fluid_cond_col] = np.where(
            sip_data[resistivity_col] > MIN_RESISTIVITY_THRESHOLD,
            (1 / sip_data[resistivity_col]) * 10000,
            np.nan
        )
        
        # Cell Constant K
        k_col = f"{prefix} Cell Constant K"
        sip_data[k_col] = np.where(
            sip_data[resistance_col] > 1e-10,
            1000000 * CELL_CONSTANT_FACTOR / sip_data[resistance_col],
            np.nan
        )
        
        if has_phase:
            # Phase in milliradians
            phase_mrad_col = f"{prefix} Phase (mRads)"
            sip_data[phase_mrad_col] = -sip_data[phase_col] * 1000
            
            # Imaginary Conductivity
            imag_cond_col = f"{prefix} Imaginary Conductivity (S/m)"
            sip_data[imag_cond_col] = np.where(
                sip_data[resistivity_col] > MIN_RESISTIVITY_THRESHOLD,
                -(1 / sip_data[resistivity_col]) * np.sin(sip_data[phase_col]),
                np.nan
            )
            
            # Real Conductivity
            real_cond_col = f"{prefix} Real Conductivity (S/m)"
            sip_data[real_cond_col] = np.where(
                sip_data[resistivity_col] > MIN_RESISTIVITY_THRESHOLD,
                (1 / sip_data[resistivity_col]) * np.cos(sip_data[phase_col]),
                np.nan
            )
            
            # Imaginary Conductivity in uS/cm
            imag_cond_uscm_col = f"{prefix} Imaginary Conductivity (uS/cm)"
            sip_data[imag_cond_uscm_col] = sip_data[imag_cond_col] * 10000
    
    return sip_data


def create_comparison_plot(datasets: Dict[str, pd.DataFrame], x_col: str, y_col: str, 
                          log_x: bool = False, log_y: bool = False, 
                          plot_type: str = 'overlay',
                          selected_loops: Dict[str, List[str]] = None,
                          color_map: Dict[str, str] = None,
                          custom_title: str = None,
                          custom_x_label: str = None,
                          custom_y_label: str = None) -> go.Figure:
    """Create comparison plot with manual color support and editable titles."""
    
    # Generate default titles from column names if not provided
    if custom_title is None:
        custom_title = f"{y_col} vs {x_col}"
    if custom_x_label is None:
        custom_x_label = x_col
    if custom_y_label is None:
        custom_y_label = y_col
    
    if plot_type == 'overlay':
        fig = go.Figure()
        for name, df in datasets.items():
            if x_col in df.columns and y_col in df.columns:
                if selected_loops and name in selected_loops and 'Loop' in df.columns:
                    if selected_loops[name]:
                        df = df[df['Loop'].isin(selected_loops[name])]
                
                if df.empty: continue
                
                # Apply manual color from the sidebar pickers
                line_color = color_map.get(name) if color_map else None
                
                if 'Loop' in df.columns and len(df['Loop'].unique()) > 1:
                    for loop in sorted(df['Loop'].unique()):
                        loop_data = df[df['Loop'] == loop]
                        fig.add_trace(go.Scatter(
                            x=loop_data[x_col], y=loop_data[y_col],
                            mode='lines+markers', name=f"{name} - L{loop}",
                            line=dict(color=line_color), # Loops of same file share color base
                            marker=dict(size=6)
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=df[y_col],
                        mode='lines+markers', name=name,
                        line=dict(color=line_color),
                        marker=dict(size=6)
                    ))
        
        # Set title and axis labels for overlay plot
        fig.update_layout(
            title=dict(
                text=custom_title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title=custom_x_label,
            yaxis_title=custom_y_label
        )
    
    elif plot_type == 'subplots':
        n_datasets = len(datasets)
        fig = make_subplots(
            rows=n_datasets, cols=1, 
            subplot_titles=list(datasets.keys()),
            vertical_spacing=0.15 / n_datasets if n_datasets > 1 else 0.1
        )
        
        for idx, (name, df) in enumerate(datasets.items(), 1):
            if x_col in df.columns and y_col in df.columns:
                if selected_loops and name in selected_loops and 'Loop' in df.columns:
                    if selected_loops[name]:
                        df = df[df['Loop'].isin(selected_loops[name])]
                
                if df.empty: continue
                
                line_color = color_map.get(name) if color_map else None
                
                if 'Loop' in df.columns and len(df['Loop'].unique()) > 1:
                    for loop in sorted(df['Loop'].unique()):
                        loop_data = df[df['Loop'] == loop]
                        fig.add_trace(
                            go.Scatter(
                                x=loop_data[x_col],
                                y=loop_data[y_col],
                                mode='lines+markers',
                                name=f"Loop {loop}",
                                line=dict(color=line_color),
                                marker=dict(size=6),
                                legendgroup=f"file{idx}",
                                legendgrouptitle_text=name
                            ),
                            row=idx, col=1
                        )
                else:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=df[y_col],
                        mode='lines+markers', name=name,
                        line=dict(color=line_color),
                        marker=dict(size=6)
                    ), row=idx, col=1)
        
        # Set main title and axis labels for subplots
        fig.update_layout(
            title=dict(
                text=custom_title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            )
        )
        
        # Update axis labels for each subplot
        for idx in range(1, n_datasets + 1):
            fig.update_xaxes(title_text=custom_x_label, row=idx, col=1)
            fig.update_yaxes(title_text=custom_y_label, row=idx, col=1)

    fig.update_xaxes(type='log' if log_x else 'linear')
    fig.update_yaxes(type='log' if log_y else 'linear')
    fig.update_layout(
        height=600 if plot_type=='overlay' else 500 * len(datasets), 
        template="plotly_dark",  # Dark template for black background
        paper_bgcolor='black',  # Black background outside plot area
        plot_bgcolor='black',   # Black background inside plot area
        font=dict(color='white'),  # White text
        title=dict(font=dict(color='white')),
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.3)',  # Subtle gray gridlines
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.3)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        ),
        legend=dict(
            bgcolor="rgba(0, 0, 0, 0.8)",  # Black legend background with transparency
            bordercolor="white",  # White border
            borderwidth=1,
            font=dict(color='white')  # White legend text
        )
    )
    return fig


def get_plot_config():
    """Return Plotly config with editability enabled."""
    return {
        'editable': True,  # Enable editing of titles, axis labels, annotations
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'sip_plot',
            'height': 1200,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': []
    }


def main():
    st.title("SIP Data Analysis App")
    
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    
    with st.sidebar:
        st.header("1. Upload Data Files")
        mode = st.radio("Analysis Mode", ["Single File", "Compare Multiple Files"])
        uploaded_files = st.file_uploader("Upload SIP files", type=["csv", "txt"], accept_multiple_files=True)
        
        st.header("2. Sample Properties")
        sample_length = st.number_input("Sample Length (m)", value=DEFAULT_SAMPLE_LENGTH, format="%.4f")
        sample_area = st.number_input("Sample Area (m²)", value=DEFAULT_SAMPLE_AREA, format="%.4f")
        manual_ref = st.number_input("Reference Resistor (Ohms)", value=0.0)

    if not uploaded_files:
        st.info("Please upload SIP file(s) to begin")
        
        with st.expander("About this app"):
            st.markdown("""
            **Features:**
            - Automatic parsing of multiple file formats
            - Physics calculations (Resistance, Resistivity, Conductivity)
            - Interactive multi-channel visualization
            - **Multi-file comparison** with overlay and subplot views
            - Support for multiple measurement loops
            - Customizable dual-axis plots
            
            **Supported File Formats:**
            - O&E PSIP (.csv) - Full instrument output
            - Simple Tables (.txt, .csv) - Frequency/Magnitude/Phase data
            
            **Comparison Mode:**
            Upload multiple files to compare measurements across different samples,
            conditions, or time points. View data overlaid or in separate subplots.
            """)
        return
    
    # Process files
    processed_datasets = {}
    
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        
        # Parse file
        sip_data, file_ref_resistor, format_name = parse_sip_file(uploaded_file)
        
        if sip_data is None:
            st.error(f"Failed to parse {file_key}")
            continue
        
        st.success(f"Loaded: **{file_key}** ({format_name})")
        
        # Determine resistor value
        reference_resistor = manual_ref if manual_ref > 0 else file_ref_resistor
        
        # Calculate physics
        with st.spinner(f"Calculating properties for {file_key}..."):
            sip_data = calculate_physics_properties(
                sip_data,
                reference_resistor,
                sample_length,
                sample_area
            )
        
        # Convert Loop to string if exists
        if 'Loop' in sip_data.columns:
            sip_data['Loop'] = sip_data['Loop'].astype(str)
        
        # Add file identifier column
        sip_data['Source_File'] = file_key
        
        processed_datasets[file_key] = sip_data
    
    if not processed_datasets:
        st.error("No files were successfully processed")
        return
    
    # Display mode selection
    st.divider()
    
    if mode == "Single File":
        # Single file analysis (original behavior)
        file_key = list(processed_datasets.keys())[0]
        sip_data = processed_datasets[file_key]
        
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
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Plot Settings")
            
            all_cols = sip_data.columns.tolist()
            
            default_x = next(
                (i for i, col in enumerate(all_cols) if 'frequency' in col.lower()),
                0
            )
            default_y = max(0, len(all_cols) - 1)
            
            x_axis = st.selectbox("X-Axis", options=all_cols, index=default_x)
            y_axis = st.selectbox("Y-Axis", options=all_cols, index=default_y)
            
            st.markdown("---")
            st.markdown("**Customize Labels:**")
            
            # Generate default values from axis selections
            default_title = f"{y_axis} vs {x_axis}"
            
            custom_title = st.text_input(
                "Plot Title",
                value=default_title,
                help="Edit the plot title"
            )
            
            custom_x_label = st.text_input(
                "X-Axis Label",
                value=x_axis,
                help="Edit the X-axis label"
            )
            
            custom_y_label = st.text_input(
                "Y-Axis Label",
                value=y_axis,
                help="Edit the Y-axis label"
            )
            
            st.markdown("---")
            
            if 'Loop' in sip_data.columns:
                loops = sorted(sip_data['Loop'].unique())
                selected_loops = st.multiselect("Select Loops", loops, default=loops)
                
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
                st.warning("No data selected.")
                return
            
            fig = px.line(
                plot_data,
                x=x_axis,
                y=y_axis,
                color='Loop' if 'Loop' in plot_data.columns else None,
                markers=True,
                title=custom_title,
                log_x=log_x,
                log_y=log_y
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(
                height=600, 
                hovermode='closest',
                title=dict(x=0.5, xanchor='center', font=dict(size=16, color='white')),
                xaxis_title=custom_x_label,
                yaxis_title=custom_y_label,
                template="plotly_dark",  # Dark template for black background
                paper_bgcolor='black',  # Black background outside plot area
                plot_bgcolor='black',   # Black background inside plot area
                font=dict(color='white'),  # White text
                xaxis=dict(
                    gridcolor='rgba(128, 128, 128, 0.3)',  # Subtle gray gridlines
                    zerolinecolor='rgba(128, 128, 128, 0.5)'
                ),
                yaxis=dict(
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    zerolinecolor='rgba(128, 128, 128, 0.5)'
                ),
                legend=dict(
                    bgcolor="rgba(0, 0, 0, 0.8)",  # Black legend background
                    bordercolor="white",  # White border
                    borderwidth=1,
                    font=dict(color='white')  # White legend text
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=get_plot_config())
        
        with st.expander("View Processed Data Table"):
            st.dataframe(sip_data, use_container_width=True)
        
        st.download_button(
            label="Download CSV",
            data=sip_data.to_csv(index=False).encode('utf-8'),
            file_name=f"sip_processed_{file_key}",
            mime="text/csv"
        )
    
    else:
        # Multi-file comparison mode
        st.subheader(f"Comparing {len(processed_datasets)} files")
        
        # Show file summary
        summary_data = []
        for name, df in processed_datasets.items():
            channels = [col.split()[0] for col in df.columns if 'Magnitude' in col]
            summary_data.append({
                "File": name,
                "Rows": len(df),
                "Channels": len(channels) if channels else 1,
                "Loops": len(df['Loop'].unique()) if 'Loop' in df.columns else 1
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Comparison Settings")
            
            # Get common columns across all datasets
            common_cols = set(processed_datasets[list(processed_datasets.keys())[0]].columns)
            for df in processed_datasets.values():
                common_cols = common_cols.intersection(set(df.columns))
            common_cols = sorted(list(common_cols))
            
            if not common_cols:
                st.error("No common columns found across all files")
                return
            
            default_x = next(
                (i for i, col in enumerate(common_cols) if 'frequency' in col.lower()),
                0
            )
            
            x_axis = st.selectbox("X-Axis", options=common_cols, index=default_x)
            
            # Filter Y-axis options to numeric columns only
            numeric_common_cols = []
            for col in common_cols:
                sample_df = processed_datasets[list(processed_datasets.keys())[0]]
                if col in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[col]):
                    numeric_common_cols.append(col)
            
            if not numeric_common_cols:
                st.error("No common numeric columns for Y-axis")
                return
            
            y_axis = st.selectbox("Y-Axis", options=numeric_common_cols)
            
            st.markdown("---")
            st.markdown("**Customize Labels:**")
            
            # Generate default values from axis selections
            default_title = f"{y_axis} vs {x_axis} - Comparison"
            
            custom_title = st.text_input(
                "Plot Title",
                value=default_title,
                help="Edit the plot title"
            )
            
            custom_x_label = st.text_input(
                "X-Axis Label",
                value=x_axis,
                help="Edit the X-axis label"
            )
            
            custom_y_label = st.text_input(
                "Y-Axis Label",
                value=y_axis,
                help="Edit the Y-axis label"
            )
            
            st.markdown("---")
            
            plot_type = st.radio(
                "Plot Type",
                ["overlay", "subplots"],
                format_func=lambda x: "Overlay (Single Plot)" if x == "overlay" else "Subplots (Separate Plots)"
            )
            
            log_x = st.checkbox("Log Scale X-Axis", value=True)
            log_y = st.checkbox("Log Scale Y-Axis", value=False)
            
            st.markdown("---")
            
            # File selection for comparison
            st.markdown("**Files to Compare:**")
            selected_files = {}
            for name in processed_datasets.keys():
                if st.checkbox(name, value=True, key=f"select_{name}"):
                    selected_files[name] = processed_datasets[name]
            
            # Loop filtering section
            st.markdown("---")
            st.markdown("**Loop Filtering:**")
            
            # Check if any selected file has loops
            has_loops = any('Loop' in df.columns for df in selected_files.values())
            
            if has_loops:
                selected_loops = {}
                
                for name, df in selected_files.items():
                    if 'Loop' in df.columns:
                        loops = sorted(df['Loop'].unique())
                        
                        with st.expander(f"Loops for {name}"):
                            # Select all / none buttons
                            col_a, col_b = st.columns(2)
                            with col_a:
                                select_all = st.button("Select All", key=f"all_{name}")
                            with col_b:
                                select_none = st.button("Select None", key=f"none_{name}")
                            
                            if select_all:
                                st.session_state[f"loops_{name}"] = loops
                            elif select_none:
                                st.session_state[f"loops_{name}"] = []
                            
                            # Initialize session state if not exists
                            if f"loops_{name}" not in st.session_state:
                                st.session_state[f"loops_{name}"] = loops
                            
                            selected_file_loops = st.multiselect(
                                f"Select loops",
                                options=loops,
                                default=st.session_state[f"loops_{name}"],
                                key=f"loop_select_{name}"
                            )
                            
                            selected_loops[name] = selected_file_loops
                    else:
                        selected_loops[name] = None
            else:
                selected_loops = None
                st.info("No loop data available in selected files")
        
        with col2:
            if not selected_files:
                st.warning("Please select at least one file to plot")
                return
            
            if len(selected_files) < 2 and mode == "Compare Multiple Files":
                st.info("Select at least 2 files for comparison")
            
            fig = create_comparison_plot(
                selected_files,
                x_axis,
                y_axis,
                log_x=log_x,
                log_y=log_y,
                plot_type=plot_type,
                selected_loops=selected_loops,
                custom_title=custom_title,
                custom_x_label=custom_x_label,
                custom_y_label=custom_y_label
            )
            
            st.plotly_chart(fig, use_container_width=True, config=get_plot_config())
        
        # Export comparison data
        st.divider()
        st.subheader("Export Comparison Data")
        
        if st.button("Combine All Data into Single CSV"):
            combined_df = pd.concat(processed_datasets.values(), ignore_index=True)
            
            st.download_button(
                label="Download Combined CSV",
                data=combined_df.to_csv(index=False).encode('utf-8'),
                file_name="sip_comparison_combined.csv",
                mime="text/csv"
            )
            
            st.success(f"Combined {len(processed_datasets)} files into one CSV with {len(combined_df)} total rows")


if __name__ == "__main__":
    main()
