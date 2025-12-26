# utils/dashboard_utils.py
# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD UTILITIES & OPTIMIZATION
# Shared utilities for all dashboard pages - caching, validation, styling
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CACHING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_market_data(symbol: str, days: int) -> pd.DataFrame:
    """
    Load market data with caching (1 hour TTL).
    
    Args:
        symbol: Stock symbol
        days: Number of days of historical data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Generate synthetic data (replace with real source)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(hash(symbol) % 2**32)
        
        base_price = 20000
        trend = np.linspace(0, 500, days)
        noise = np.cumsum(np.random.randn(days) * 100)
        close_prices = base_price + trend + noise
        
        df = pd.DataFrame({
            'date': dates,
            'open': close_prices + np.random.randn(days) * 50,
            'high': close_prices + abs(np.random.randn(days) * 80),
            'low': close_prices - abs(np.random.randn(days) * 80),
            'close': close_prices,
            'volume': np.random.randint(50000000, 300000000, days),
        })
        
        df = df.set_index('date')
        return df.sort_index()
    except Exception as e:
        logger.error(f"Error loading market data for {symbol}: {e}")
        raise


@st.cache_resource
def get_analyzer_instance(analyzer_class):
    """
    Get cached analyzer instance (lifetime of app).
    
    Args:
        analyzer_class: Analyzer class to instantiate
        
    Returns:
        Instantiated analyzer object
    """
    try:
        return analyzer_class()
    except Exception as e:
        logger.error(f"Error creating analyzer instance: {e}")
        raise


def cache_computation(ttl_seconds: int = 300):
    """
    Decorator for caching expensive computations.
    
    Args:
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl_seconds)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize default session state variables."""
    defaults = {
        'symbol_history': [],
        'lookback_periods': {},
        'chart_preferences': {},
        'selected_symbols': [],
        'dashboard_state': {},
        'user_filters': {},
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Get value from session state with fallback default.
    
    Args:
        key: Session state key
        default: Default value if key not found
        
    Returns:
        Session value or default
    """
    initialize_session_state()
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Set value in session state.
    
    Args:
        key: Session state key
        value: Value to set
    """
    initialize_session_state()
    st.session_state[key] = value


def add_to_history(key: str, value: Any, max_items: int = 10) -> None:
    """
    Add item to session history, keeping last N items.
    
    Args:
        key: History key (e.g., 'symbol_history')
        value: Value to add
        max_items: Maximum history items to keep
    """
    initialize_session_state()
    
    if key not in st.session_state:
        st.session_state[key] = []
    
    history = st.session_state[key]
    
    # Remove if already in history (to move to top)
    if value in history:
        history.remove(value)
    
    # Add to beginning
    history.insert(0, value)
    
    # Trim to max size
    st.session_state[key] = history[:max_items]


# ═══════════════════════════════════════════════════════════════════════════
# DATA VALIDATION & CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        
    Returns:
        (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for all NaN columns
    for col in required_columns:
        if df[col].isna().all():
            return False, f"Column '{col}' contains all NaN values"
    
    # Check for duplicate indices
    if df.index.duplicated().any():
        return False, "DataFrame index has duplicates"
    
    return True, ""


def clean_data(df: pd.DataFrame, fill_method: str = 'forward') -> pd.DataFrame:
    """
    Clean DataFrame - handle NaN, duplicates, etc.
    
    Args:
        df: DataFrame to clean
        fill_method: 'forward', 'backward', or 'drop'
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle NaN values
    if fill_method == 'forward':
        df = df.fillna(method='ffill')
    elif fill_method == 'backward':
        df = df.fillna(method='bfill')
    elif fill_method == 'drop':
        df = df.dropna()
    
    # Sort by index
    df = df.sort_index()
    
    return df


def validate_symbol(symbol: str, allowed_symbols: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate stock symbol.
    
    Args:
        symbol: Symbol to validate
        allowed_symbols: List of allowed symbols (if any)
        
    Returns:
        (is_valid, error_message)
    """
    if not symbol or not isinstance(symbol, str):
        return False, "Symbol must be a non-empty string"
    
    symbol = symbol.upper().strip()
    
    # Basic format check
    if len(symbol) < 1 or len(symbol) > 10:
        return False, "Symbol must be 1-10 characters"
    
    # Check for invalid characters
    if not symbol.replace('&', '').replace('-', '').isalnum():
        return False, "Symbol contains invalid characters"
    
    # Check against allowed list if provided
    if allowed_symbols and symbol not in allowed_symbols:
        return False, f"Symbol '{symbol}' not in allowed list"
    
    return True, ""


def validate_period(start_date: datetime, end_date: datetime, 
                   min_days: int = 1, max_days: int = 2000) -> Tuple[bool, str]:
    """
    Validate date period.
    
    Args:
        start_date: Start date
        end_date: End date
        min_days: Minimum period length
        max_days: Maximum period length
        
    Returns:
        (is_valid, error_message)
    """
    if start_date >= end_date:
        return False, "Start date must be before end date"
    
    days = (end_date - start_date).days
    
    if days < min_days:
        return False, f"Period must be at least {min_days} day(s)"
    
    if days > max_days:
        return False, f"Period cannot exceed {max_days} days"
    
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════
# STYLING & FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def get_signal_color(signal: str) -> str:
    """Get color for trading signal."""
    signal_colors = {
        'BUY': '#00ff00',
        'SELL': '#ff0000',
        'NEUTRAL': '#ffaa00',
    }
    return signal_colors.get(signal, '#888888')


def get_signal_emoji(signal: str) -> str:
    """Get emoji for trading signal."""
    signal_emojis = {
        'BUY': '🟢',
        'SELL': '🔴',
        'NEUTRAL': '🟡',
    }
    return signal_emojis.get(signal, '❓')


def get_severity_color(severity: str) -> str:
    """Get color based on severity level."""
    severity_colors = {
        'Minor': '#ffaa00',
        'Moderate': '#ff6600',
        'Major': '#ff3300',
        'Critical': '#ff0000',
    }
    return severity_colors.get(severity, '#888888')


def get_severity_emoji(severity: str) -> str:
    """Get emoji based on severity."""
    severity_emojis = {
        'Minor': '⚪',
        'Moderate': '🟡',
        'Major': '🔴',
        'Critical': '⛔',
    }
    return severity_emojis.get(severity, '❓')


def format_currency(value: float, symbol: str = '₹', decimals: int = 2) -> str:
    """Format value as currency."""
    return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value:+.{decimals}f}%"


def format_large_number(value: float, decimals: int = 1) -> str:
    """Format large number with K/M/B suffix."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    return f"{value:.{decimals}f}"


# ═══════════════════════════════════════════════════════════════════════════
# CHART TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

def create_signal_badge(signal: str, confidence: float) -> str:
    """Create HTML badge for signal display."""
    color = get_signal_color(signal)
    emoji = get_signal_emoji(signal)
    confidence_pct = int(confidence * 100) if isinstance(confidence, float) else 0
    
    return f"""
    <div style='
        background: linear-gradient(135deg, {color}33 0%, {color}11 100%);
        border-left: 5px solid {color};
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    '>
        <h3 style='margin: 0; color: {color}; font-size: 24px;'>{emoji} {signal}</h3>
        <p style='margin: 3px 0 0 0; font-size: 12px; color: #aaaaaa;'>Confidence: {confidence_pct}%</p>
    </div>
    """


def create_metric_card(title: str, value: str, detail: str = "", color: str = "#1f77b4") -> str:
    """Create HTML metric card."""
    return f"""
    <div style='
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border-left: 4px solid {color};
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
    '>
        <p style='margin: 0; font-size: 12px; color: #888; font-weight: bold;'>{title}</p>
        <h2 style='margin: 5px 0 0 0; color: {color}; font-size: 24px;'>{value}</h2>
        {f'<p style="margin: 3px 0 0 0; font-size: 11px; color: #666;">{detail}</p>' if detail else ''}
    </div>
    """


def create_empty_chart(title: str) -> go.Figure:
    """Create empty placeholder chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"No data available for {title}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="#888888")
    )
    fig.update_layout(
        title=title,
        template='plotly_dark',
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def safe_execute(func: Callable, *args, fallback: Any = None, 
                log_error: bool = True, **kwargs) -> Any:
    """
    Safely execute function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        fallback: Value to return on error
        log_error: Whether to log the error
        **kwargs: Keyword arguments
        
    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"Error in {func.__name__}: {e}")
        return fallback


def handle_errors(error_context: str = "Operation"):
    """
    Decorator for error handling in Streamlit apps.
    
    Args:
        error_context: Description of operation for error message
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_context} failed: {e}")
                st.error(f"❌ {error_context} failed: {str(e)}")
                return None
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name} took {self.elapsed:.3f} seconds")


def measure_performance(func: Callable) -> Callable:
    """Decorator for measuring function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with PerformanceTimer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_LOOKBACK_MIN = 30
DEFAULT_LOOKBACK_MAX = 250
DEFAULT_LOOKBACK_DEFAULT = 100

AVAILABLE_SYMBOLS = ["NIFTY", "SENSEX", "NIFTY50", "NIFTY_IT", "RELIANCE", 
                     "TCS", "INFY", "WIPRO", "HDFC", "ICICI", "AXIS"]

TIMEFRAMES = ["1min", "5min", "15min", "30min", "60min", "day", "week", "month"]

CHART_TEMPLATE = "plotly_dark"
CHART_HEIGHT_DEFAULT = 500
CHART_HEIGHT_SMALL = 400


def apply_dashboard_styling():
    """Apply consistent styling to all dashboards."""
    st.markdown("""
    <style>
        .stMainBlockContainer {
            padding: 20px;
        }
        
        .signal-card {
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            background-color: #0e1117;
            margin: 10px 0;
        }
        
        .metric-card {
            padding: 15px;
            border-radius: 8px;
            background-color: #161b22;
            text-align: center;
            margin: 10px 0;
        }
        
        h1, h2, h3 {
            color: #58a6ff;
            margin-top: 20px;
        }
        
        .divider {
            border-top: 1px solid #30363d;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Test utilities
    initialize_session_state()
    print("Dashboard utilities initialized successfully")
