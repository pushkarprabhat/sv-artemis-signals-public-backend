"""
Phoenix Theme - Vibrant, Rich, and Elegant UI Theme for SV Pair Trading
Features: Warm colors, visual hierarchy, smooth gradients, delightful interactions
"""

import streamlit as st

# ============================================================================
# RUBY GENTLE COLOR PALETTE
# ============================================================================
COLORS = {
    # Primary Ruby Colors
    "ruby_primary": "#E0115F",    # The main Ruby color (Vibrant but elegant)
    "ruby_dark": "#900C3F",       # Deep crimson for backgrounds/gradients
    "rose_gentle": "#D87093",     # Pale Violet Red for softer accents
    
    # Secondary Colors
    "gold": "#FFD700",            # Kept for premium accents
    "warm_grey": "#B0B0B0",       # Gentle text color
    
    # Backgrounds (Deep Burgundy/Dark)
    "bg_dark": "#1A0509",         # Very dark red-black
    "bg_card": "#2D0A14",         # Slightly lighter burgundy for cards
    "bg_sidebar": "#3D1B28",       # Lighter burgundy for sidebar (less dark)
    
    # Status Colors (Softer versions)
    "success": "#4CAF50",
    "warning": "#FFA726",
    "danger": "#EF5350",
    "info": "#42A5F5",
    
    # Text
    "text_primary": "#E8E8E8",
    "text_secondary": "#B0B0B0",

    # Compatibility mapping (for existing code using old keys)
    "orange": "#E0115F",          # Map orange to ruby
    "red": "#C2185B",             # Map red to deep rose
    "purple": "#880E4F",          # Map purple to tyrian purple
    "warm_dark": "#2D0A14",       # Map warm_dark to card bg
    "dark_bg": "#1A0509",         # Map dark_bg to main bg
    "light_text": "#E8E8E8",
    "muted_gold": "#D4AF37",
}

# ============================================================================
# PHOENIX THEME CONFIGURATION (RUBY EDITION)
# ============================================================================
THEME_CSS = f"""
<style>
    /* ========== BASE STYLING ========== */
    :root {{
        --phoenix-primary: {COLORS['ruby_primary']};
        --phoenix-secondary: {COLORS['rose_gentle']};
        --phoenix-gold: {COLORS['gold']};
        --phoenix-bg: {COLORS['bg_dark']};
        --phoenix-card-bg: {COLORS['bg_card']};
    }}
    
    /* ========== MAIN BACKGROUND ========== */
    .main {{
        background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, {COLORS['bg_card']} 100%);
    }}
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {{
        background: {COLORS['bg_sidebar']};
        border-right: 1px solid {COLORS['ruby_dark']};
    }}
    
    /* ========== HEADER STYLING ========== */
    .phoenix-header {{
        background: linear-gradient(90deg, {COLORS['ruby_dark']} 0%, {COLORS['ruby_primary']} 100%);
        color: {COLORS['text_primary']};
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 5px solid {COLORS['gold']};
        box-shadow: 0 4px 20px rgba(224, 17, 95, 0.2);
    }}
    
    .phoenix-header h1 {{
        margin: 0;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: {COLORS['text_primary']};
    }}
    
    .phoenix-header-subtitle {{
        color: {COLORS['rose_gentle']};
        font-size: 15px;
        margin-top: 5px;
        font-weight: 400;
    }}
    
    /* ========== CARD STYLING ========== */
    .phoenix-card {{
        background: linear-gradient(145deg, {COLORS['bg_card']} 0%, {COLORS['bg_dark']} 100%);
        border: 1px solid {COLORS['ruby_dark']};
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }}
    
    .phoenix-card:hover {{
        border-color: {COLORS['ruby_primary']};
        transform: translateY(-2px);
    }}
    
    /* ========== METRIC CARDS ========== */
    .phoenix-metric {{
        background: rgba(45, 10, 20, 0.6);
        border-left: 3px solid {COLORS['ruby_primary']};
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }}
    
    .phoenix-metric-value {{
        font-size: 26px;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin: 5px 0;
    }}
    
    .phoenix-metric-label {{
        font-size: 13px;
        color: {COLORS['rose_gentle']};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* ========== BUTTONS ========== */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['ruby_primary']} 0%, {COLORS['ruby_dark']} 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['ruby_dark']};
        color: {COLORS['gold']};
        border-color: {COLORS['gold']};
    }}

    
    /* ========== PRIMARY BUTTON ========== */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['gold']} 0%, {COLORS['orange']} 100%);
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {COLORS['orange']} 0%, {COLORS['red']} 100%);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }}
    
    /* ========== TABS ========== */
    [data-baseweb="tab"] {{
        background: transparent;
        border-bottom: 2px solid rgba(255, 215, 0, 0.3);
        color: {COLORS['light_text']};
        font-weight: 500;
    }}
    
    [data-baseweb="tab"][aria-selected="true"] {{
        border-bottom-color: {COLORS['gold']};
        color: {COLORS['gold']};
        box-shadow: 0 2px 10px rgba(255, 215, 0, 0.2);
    }}
    
    [data-baseweb="tab"]:hover {{
        color: {COLORS['orange']};
    }}
    
    /* ========== INPUT FIELDS ========== */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select,
    .stMultiSelect [data-baseweb="select"] {{
        background: rgba(255, 215, 0, 0.05);
        border: 2px solid {COLORS['gold']};
        color: {COLORS['light_text']};
        border-radius: 6px;
        padding: 10px 12px;
        transition: all 0.3s ease;
    }}
    
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox select:focus {{
        border-color: {COLORS['orange']};
        box-shadow: 0 0 12px rgba(255, 140, 0, 0.3);
    }}
    
    /* ========== DIVIDERS ========== */
    .stDivider {{
        border: 1px solid transparent;
        background: linear-gradient(90deg, transparent, {COLORS['gold']}, transparent);
        margin: 20px 0;
    }}
    
    /* ========== SUCCESS/WARNING/ERROR ========== */
    .stSuccess {{
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid {COLORS['success']};
        border-radius: 6px;
        padding: 16px;
    }}
    
    .stWarning {{
        background: rgba(255, 152, 0, 0.1);
        border-left: 4px solid {COLORS['warning']};
        border-radius: 6px;
        padding: 16px;
    }}
    
    .stError {{
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid {COLORS['danger']};
        border-radius: 6px;
        padding: 16px;
    }}
    
    .stInfo {{
        background: rgba(33, 150, 243, 0.1);
        border-left: 4px solid {COLORS['info']};
        border-radius: 6px;
        padding: 16px;
    }}
    
    /* ========== DATAFRAME ========== */
    .stDataframe {{
        background: rgba(255, 215, 0, 0.03);
        border: 1px solid {COLORS['gold']};
        border-radius: 8px;
        overflow: hidden;
    }}
    
    .stDataframe thead {{
        background: linear-gradient(90deg, rgba(255, 140, 0, 0.2) 0%, rgba(255, 215, 0, 0.15) 100%);
        border-bottom: 2px solid {COLORS['gold']};
    }}
    
    /* ========== EXPANDER ========== */
    .stExpander {{
        background: rgba(147, 112, 219, 0.05);
        border: 1px solid {COLORS['purple']};
        border-radius: 8px;
    }}
    
    .stExpander > button {{
        color: {COLORS['gold']};
    }}
    
    /* ========== SIDEBAR RADIO/CHECKBOX ========== */
    [data-testid="stSidebar"] [data-baseweb="radio"] {{
        accent-color: {COLORS['gold']};
    }}
    
    [data-testid="stSidebar"] [data-baseweb="checkbox"] {{
        accent-color: {COLORS['gold']};
    }}
    
    /* ========== PAGE LINK ========== */
    [data-testid="stSidebar"] .stPageLink {{
        color: {COLORS['light_text']};
        padding: 12px;
        margin: 6px 0;
        border-radius: 6px;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stSidebar"] .stPageLink:hover {{
        background: rgba(255, 140, 0, 0.15);
        color: {COLORS['gold']};
        padding-left: 16px;
    }}
    
    [data-testid="stSidebar"] .stPageLink.active {{
        background: linear-gradient(90deg, rgba(255, 140, 0, 0.2) 0%, rgba(255, 215, 0, 0.1) 100%);
        color: {COLORS['gold']};
        border-left: 4px solid {COLORS['gold']};
        font-weight: 600;
        padding-left: 12px;
    }}
    
    /* ========== METRIC ========== */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, rgba(255, 140, 0, 0.08) 0%, rgba(255, 215, 0, 0.04) 100%);
        border: 1px solid {COLORS['gold']};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(255, 69, 0, 0.1);
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS['gold']};
        font-weight: 700;
    }}
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {COLORS['orange']} 0%, {COLORS['gold']} 100%);
        box-shadow: 0 0 10px rgba(255, 140, 0, 0.5);
    }}
    
    /* ========== TEXT & HEADINGS ========== */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['light_text']};
    }}
    
    h1 {{
        color: {COLORS['gold']};
        font-size: 32px;
        font-weight: 800;
        text-shadow: 0 2px 8px rgba(255, 69, 0, 0.2);
    }}
    
    h2 {{
        border-bottom: 2px solid {COLORS['orange']};
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}
    
    p {{
        color: {COLORS['light_text']};
        line-height: 1.6;
    }}
    
    /* ========== ANIMATIONS ========== */
    @keyframes phoenix-glow {{
        0%, 100% {{
            box-shadow: 0 0 10px rgba(255, 140, 0, 0.3);
        }}
        50% {{
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        }}
    }}
    
    .phoenix-glow {{
        animation: phoenix-glow 2s ease-in-out infinite;
    }}
    
    @keyframes pulse-subtle {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.8;
        }}
    }}
    
    .pulse-subtle {{
        animation: pulse-subtle 2s ease-in-out infinite;
    }}
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['dark_bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['gold']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['orange']};
    }}
</style>
"""


def apply_phoenix_theme():
    """Apply Phoenix theme to the entire app"""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_phoenix_header(title: str, subtitle: str = "", emoji: str = "🔥"):
    """
    Render a vibrant Phoenix-themed header
    
    Args:
        title: Main header text
        subtitle: Optional subtitle text
        emoji: Icon emoji (default: 🔥)
    """
    header_html = f"""
    <div class="phoenix-header">
        <h1>{emoji} {title}</h1>
        {'<div class="phoenix-header-subtitle">' + subtitle + '</div>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_phoenix_card(content, title: str = None, emoji: str = "✨", expandable: bool = False):
    """
    Render a Phoenix-themed card with optional title
    
    Args:
        content: Content to display (text or function that renders)
        title: Optional card title
        emoji: Icon emoji
        expandable: Whether to show as expander
    """
    if expandable and title:
        with st.expander(f"{emoji} {title}"):
            if callable(content):
                content()
            else:
                st.markdown(content)
    else:
        card_html = f"""
        <div class="phoenix-card">
            {'<h3 style="color: #FFD700; margin-top: 0; display: flex; align-items: center;"><span style="margin-right: 10px;">' + emoji + '</span>' + title + '</h3>' if title else ''}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        if callable(content):
            content()
        else:
            st.markdown(content)


def render_metric_card(label: str, value: str, emoji: str = "📊", delta: str = None):
    """
    Render a Phoenix-themed metric card
    
    Args:
        label: Metric label
        value: Metric value
        emoji: Icon emoji
        delta: Optional change indicator
    """
    delta_html = f' <span style="color: #4CAF50; font-size: 12px;">({delta})</span>' if delta else ''
    
    metric_html = f"""
    <div class="phoenix-metric">
        <div class="phoenix-metric-label">{emoji} {label}</div>
        <div class="phoenix-metric-value">{value}</div>{delta_html}
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)


def phoenix_button(label: str, key: str = None, primary: bool = False, use_container_width: bool = True):
    """
    Render a Phoenix-styled button (wrapper around st.button)
    
    Args:
        label: Button label
        key: Unique key for button
        primary: Whether to use primary styling
        use_container_width: Whether button should expand to container width (deprecated, uses width parameter)
    
    Returns:
        Boolean indicating if button was clicked
    """
    width = "stretch" if use_container_width else "content"
    return st.button(label, key=key, type="primary" if primary else "secondary", width=width)


def get_phoenix_colors() -> dict:
    """Return Phoenix color palette"""
    return COLORS.copy()


def highlight_dataframe(df, column_colors: dict = None):
    """
    Style a dataframe with Phoenix theme
    
    Args:
        df: Pandas DataFrame
        column_colors: Dict mapping column names to colors
    
    Returns:
        Styled DataFrame
    """
    def color_row(row):
        return ['background-color: rgba(255, 140, 0, 0.1)'] * len(row)
    
    styled_df = df.style.apply(color_row, axis=1)
    return styled_df
