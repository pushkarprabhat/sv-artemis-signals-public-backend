# core/ui_theme.py
"""
Professional Trading Platform UI Theme
Modern, clean design with excellent readability for financial data
"""

# Color Palette - Professional Trading Theme
COLORS = {
    # Primary Colors
    'primary': '#1E88E5',        # Professional Blue
    'secondary': '#26A69A',      # Teal (success/positive)
    'accent': '#FF6F00',         # Orange (alerts/warnings)
    
    # Status Colors
    'success': '#00C853',        # Green (profits, buy signals)
    'danger': '#D32F2F',         # Red (losses, sell signals)
    'warning': '#FFA000',        # Amber (caution)
    'info': '#0288D1',           # Light Blue (information)
    
    # Background Colors
    'bg_primary': '#0E1117',     # Dark background (Streamlit default dark)
    'bg_secondary': '#262730',   # Card background
    'bg_tertiary': '#1C1E26',    # Sidebar background
    
    # Text Colors
    'text_primary': '#FAFAFA',   # Main text
    'text_secondary': '#B0B0B0', # Secondary text
    'text_muted': '#808080',     # Muted text
    
    # Chart Colors
    'chart_up': '#00E676',       # Bullish candle
    'chart_down': '#FF1744',     # Bearish candle
    'chart_neutral': '#78909C',  # Neutral/volume bars
    
    # Border/Divider
    'border': '#424242',         # Subtle borders
    'divider': '#303030',        # Section dividers
}

# Streamlit Custom CSS
CUSTOM_CSS = f"""
<style>
    /* Main container */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_tertiary']};
    }}
    
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}
    
    [data-testid="stMetricDelta"] {{
        font-size: 1rem;
    }}
    
    /* Success metrics */
    .metric-success {{
        color: {COLORS['success']} !important;
    }}
    
    /* Danger metrics */
    .metric-danger {{
        color: {COLORS['danger']} !important;
    }}
    
    /* Cards */
    .status-card {{
        background-color: {COLORS['bg_secondary']};
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid {COLORS['primary']};
        margin-bottom: 1rem;
    }}
    
    .success-card {{
        border-left-color: {COLORS['success']};
    }}
    
    .warning-card {{
        border-left-color: {COLORS['warning']};
    }}
    
    .danger-card {{
        border-left-color: {COLORS['danger']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: #1976D2;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
    }}
    
    /* Headers */
    h1 {{
        color: {COLORS['text_primary']};
        font-weight: 700;
        margin-bottom: 1.5rem;
    }}
    
    h2 {{
        color: {COLORS['text_primary']};
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid {COLORS['border']};
        padding-bottom: 0.5rem;
    }}
    
    h3 {{
        color: {COLORS['text_secondary']};
        font-weight: 500;
    }}
    
    /* Tables */
    .dataframe {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
    }}
    
    .dataframe th {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        font-weight: 600;
        padding: 0.75rem;
    }}
    
    .dataframe td {{
        padding: 0.5rem;
        border-bottom: 1px solid {COLORS['divider']};
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {COLORS['bg_secondary']};
        border-radius: 6px;
        font-weight: 500;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['bg_secondary']};
        border-radius: 6px 6px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background-color: {COLORS['success']};
    }}
    
    /* Alerts */
    .stAlert {{
        border-radius: 6px;
        padding: 1rem;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom badge */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }}
    
    .badge-success {{
        background-color: {COLORS['success']};
        color: white;
    }}
    
    .badge-danger {{
        background-color: {COLORS['danger']};
        color: white;
    }}
    
    .badge-warning {{
        background-color: {COLORS['warning']};
        color: white;
    }}
    
    .badge-info {{
        background-color: {COLORS['info']};
        color: white;
    }}
    
    /* Provider status indicators */
    .provider-status {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: 4px;
        background-color: {COLORS['bg_secondary']};
        margin-bottom: 0.5rem;
    }}
    
    .status-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }}
    
    .status-online {{
        background-color: {COLORS['success']};
        box-shadow: 0 0 8px {COLORS['success']};
    }}
    
    .status-offline {{
        background-color: {COLORS['danger']};
    }}
    
    .status-degraded {{
        background-color: {COLORS['warning']};
    }}
</style>
"""

def apply_theme():
    """Apply custom theme to Streamlit app"""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def status_badge(text: str, status: str = 'info') -> str:
    """Generate HTML for status badge"""
    return f'<span class="badge badge-{status}">{text}</span>'

def provider_status_indicator(provider: str, is_online: bool, response_time_ms: float = None) -> str:
    """Generate HTML for provider status indicator"""
    status_class = 'status-online' if is_online else 'status-offline'
    status_text = '🟢 Online' if is_online else '🔴 Offline'
    
    response_info = f' • {response_time_ms:.0f}ms' if response_time_ms and is_online else ''
    
    return f'''
    <div class="provider-status">
        <div class="status-dot {status_class}"></div>
        <strong>{provider}</strong>
        <span style="color: {COLORS['text_secondary']}">{status_text}{response_info}</span>
    </div>
    '''
