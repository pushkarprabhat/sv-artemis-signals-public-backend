"""
LTP Display Component for Streamlit UI
Professional live market price display with institutional presentation
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Optional
from core.ltp_database import get_ltp_database
from utils.market_hours import is_market_open


def display_market_status():
    """
    Display market status banner (OPEN/CLOSED) with live timestamp.
    Returns True if market is open, False otherwise.
    """
    market_open = is_market_open()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if market_open:
        status_color = "#00ff00"  # Green
        status_text = "🟢 MARKET OPEN"
        pulse_animation = "animation: pulse 2s infinite;"
        bg_color = "rgba(0, 255, 0, 0.1)"
    else:
        status_color = "#ff6b6b"  # Red
        status_text = "🔴 MARKET CLOSED"
        pulse_animation = ""
        bg_color = "rgba(255, 107, 107, 0.1)"
    
    st.markdown(f"""
        <style>
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
            100% {{ opacity: 1; }}
        }}
        </style>
        <div style="
            background: {bg_color};
            padding: 15px 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid {status_color};
            display: flex;
            justify-content: space-between;
            align-items: center;
            {pulse_animation}
        ">
            <div style="
                color: {status_color};
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
            ">
                {status_text}
            </div>
            <div style="
                color: #b0b0b0;
                font-size: 14px;
            ">
                🕐 Last Updated: {current_time}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    return market_open


def display_ltp_ticker(symbols: List[str], columns: int = 4):
    """
    Display LTP ticker for multiple symbols in a grid layout.
    
    Args:
        symbols: List of symbols to display
        columns: Number of columns in the grid
    """
    ltp_db = get_ltp_database()
    
    # Create columns
    cols = st.columns(columns)
    
    for idx, symbol in enumerate(symbols):
        col_idx = idx % columns
        
        with cols[col_idx]:
            ltp_data = ltp_db.get_ltp(symbol)
            
            if ltp_data:
                last_price = ltp_data['last_price']
                change_pct = ltp_data.get('change_percent', 0.0)
                timestamp = ltp_data.get('timestamp', 'N/A')
                
                # Color based on change
                if change_pct > 0:
                    color = "#00ff00"  # Green
                    arrow = "▲"
                elif change_pct < 0:
                    color = "#ff0000"  # Red
                    arrow = "▼"
                else:
                    color = "#808080"  # Gray
                    arrow = "━"
                
                # Display card
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_display = dt.strftime("%H:%M:%S")
                    date_display = dt.strftime("%d %b")
                except:
                    time_display = "--:--:--"
                    date_display = "--"
                
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <div style="color: white; font-size: 13px; font-weight: bold;">
                            {symbol}
                        </div>
                        <div style="color: white; font-size: 22px; font-weight: bold; margin: 5px 0;">
                            ₹{last_price:,.2f}
                        </div>
                        <div style="color: {color}; font-size: 13px; font-weight: bold;">
                            {arrow} {abs(change_pct):.2f}%
                        </div>
                        <div style="color: #e0e0e0; font-size: 11px; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 5px;">
                            🕐 {time_display}<br>
                            <span style="font-size: 9px; color: #c0c0c0;">{date_display}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # No data available
                st.markdown(f"""
                    <div style="
                        background: #333;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                        text-align: center;
                    ">
                        <div style="color: white; font-size: 14px; font-weight: bold;">
                            {symbol}
                        </div>
                        <div style="color: #808080; font-size: 12px; margin-top: 10px;">
                            No data available
                        </div>
                    </div>
                """, unsafe_allow_html=True)


def display_ltp_table(symbols: List[str] = None):
    """
    Display LTP data in a table format.
    
    Args:
        symbols: Optional list of symbols. If None, shows all.
    """
    ltp_db = get_ltp_database()
    
    if symbols:
        # Get specific symbols
        data = []
        for symbol in symbols:
            ltp_data = ltp_db.get_ltp(symbol)
            if ltp_data:
                data.append(ltp_data)
        
        if data:
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()
    else:
        # Get all symbols
        df = ltp_db.get_all_ltp()
    
    if not df.empty:
        # Format the dataframe
        display_df = df[['symbol', 'last_price', 'change_percent', 
                         'volume', 'timestamp']].copy()
        display_df.columns = ['Symbol', 'LTP', 'Change %', 'Volume', 'Last Updated']
        
        # Format numbers
        display_df['LTP'] = display_df['LTP'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Change %'] = display_df['Change %'].apply(
            lambda x: f"{'+' if x > 0 else ''}{x:.2f}%"
        )
        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,}")
        
        # Apply color to change column
        def color_change(val):
            if '+' in val:
                return 'background-color: #00ff0020; color: #00ff00'
            elif '-' in val:
                return 'background-color: #ff000020; color: #ff0000'
            return ''
        
        styled_df = display_df.style.applymap(
            color_change, 
            subset=['Change %']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("📊 No LTP data available. Will populate during market hours.")


def display_ltp_single(symbol: str, size: str = "medium"):
    """
    Display LTP for a single symbol (for dashboard headers).
    
    Args:
        symbol: Symbol to display
        size: Size - 'small', 'medium', or 'large'
    """
    ltp_db = get_ltp_database()
    ltp_data = ltp_db.get_ltp(symbol)
    
    if ltp_data:
        last_price = ltp_data['last_price']
        change_pct = ltp_data.get('change_percent', 0.0)
        
        # Color based on change
        if change_pct > 0:
            color = "#00ff00"
            arrow = "▲"
        elif change_pct < 0:
            color = "#ff0000"
            arrow = "▼"
        else:
            color = "#808080"
            arrow = "━"
        
        # Size-based styling
        if size == "small":
            font_size_price = "18px"
            font_size_change = "12px"
        elif size == "large":
            font_size_price = "36px"
            font_size_change = "18px"
        else:  # medium
            font_size_price = "24px"
            font_size_change = "14px"
        
        st.markdown(f"""
            <div style="text-align: center; margin: 10px 0;">
                <div style="font-size: {font_size_price}; font-weight: bold; color: white;">
                    ₹{last_price:,.2f}
                </div>
                <div style="font-size: {font_size_change}; color: {color};">
                    {arrow} {abs(change_pct):.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="text-align: center; margin: 10px 0; color: #808080;">
                No data for {symbol}
            </div>
        """, unsafe_allow_html=True)


def display_ltp_summary_stats():
    """Display summary statistics about LTP database."""
    ltp_db = get_ltp_database()
    stats = ltp_db.get_summary_stats()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Tracked Symbols",
                value=stats.get('total_symbols', 0)
            )
        
        with col2:
            last_update = stats.get('last_update', 'Never')
            if last_update != 'Never':
                try:
                    dt = datetime.fromisoformat(last_update)
                    last_update = dt.strftime("%H:%M:%S")
                except:
                    pass
            st.metric(
                label="🕐 Last Update",
                value=last_update
            )
        
        with col3:
            st.metric(
                label="📈 History Records",
                value=f"{stats.get('total_history_records', 0):,}"
            )


@st.cache_data(ttl=5)  # Cache for 5 seconds (near real-time)
def get_cached_ltp(symbol: str) -> Optional[dict]:
    """
    Get LTP with caching for performance.
    
    Args:
        symbol: Symbol name
        
    Returns:
        LTP data dict or None
    """
    ltp_db = get_ltp_database()
    return ltp_db.get_ltp(symbol)


@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_cached_all_ltp() -> pd.DataFrame:
    """
    Get all LTP data with caching.
    
    Returns:
        DataFrame with all LTP data
    """
    ltp_db = get_ltp_database()
    return ltp_db.get_all_ltp()
