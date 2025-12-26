# utils/dashboard_display.py
# Professional trading dashboard UI components
# Bloomberg-style institutional market dashboard

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List
from utils.index_data import get_index_metadata, get_index_ohlcv, get_constituent_data, format_large_number
from utils.ltp_fallback import get_ltp_with_fallback, get_ohlcv_with_fallback
from utils.logger import logger

def display_index_card(symbol: str, ltp: float, ohlcv: Dict, metadata: Dict):
    """Display beautiful index card with live data
    
    Args:
        symbol: Index symbol
        ltp: Last traded price
        ohlcv: OHLCV dictionary
        metadata: Index metadata
    """
    if not ohlcv:
        st.warning(f"⚠️ No data for {symbol}")
        return
    
    # Calculate changes
    prev_close = ohlcv.get('prev_close', ltp)
    change = ltp - prev_close
    change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
    
    # Color coding
    color = "🟢" if change >= 0 else "🔴"
    arrow = "▲" if change >= 0 else "▼"
    bg_color = "rgba(76, 175, 80, 0.1)" if change >= 0 else "rgba(244, 67, 54, 0.1)"
    border_color = metadata.get('color', '#4CAF50')
    
    # Create card HTML
    html = f"""
    <div style="
        background: {bg_color};
        border-left: 5px solid {border_color};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    ">
        <div style="color: #B0B0B0; font-size: 12px; text-transform: uppercase; margin-bottom: 5px;">
            {metadata.get('full_name', symbol)}
        </div>
        <div style="color: #FFFFFF; font-size: 28px; font-weight: bold; margin: 10px 0;">
            {color} {ltp:,.2f}
        </div>
        <div style="color: {'#4CAF50' if change >= 0 else '#F44336'}; font-size: 16px; font-weight: bold;">
            {arrow} {change:+,.2f} ({change_pct:+.2f}%)
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.1); margin-top: 10px; padding-top: 10px;">
            <div style="display: flex; justify-content: space-between; color: #B0B0B0; font-size: 11px;">
                <span>Open: {ohlcv['open']:,.2f}</span>
                <span>High: {ohlcv['high']:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #B0B0B0; font-size: 11px; margin-top: 3px;">
                <span>Low: {ohlcv['low']:,.2f}</span>
                <span>Prev: {prev_close:,.2f}</span>
            </div>
            <div style="color: #808080; font-size: 10px; margin-top: 5px;">
                Vol: {format_large_number(ohlcv.get('volume', 0))}
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def display_major_indices(symbols: List[str]):
    """Display major indices in a grid layout with robust fallback
    
    Args:
        symbols: List of index symbols to display
        
    Uses multi-source fallback to ensure reliable price display:
    Priority: Live API → Cached LTP → Yesterday's close
    """
    st.markdown("### 📊 MAJOR INDICES — Live Market Data")
    st.markdown("---")
    
    metadata_map = get_index_metadata()
    
    # Create columns based on number of indices
    cols = st.columns(len(symbols))
    
    for idx, symbol in enumerate(symbols):
        with cols[idx]:
            try:
                # Get live price with fallback chain
                ltp = get_ltp_with_fallback(symbol)
                
                # Get OHLCV data with fallback
                ohlcv = get_ohlcv_with_fallback(symbol)
                
                # Get metadata
                metadata = metadata_map.get(symbol, {})
                
                if ltp and ohlcv:
                    display_index_card(symbol, ltp, ohlcv, metadata)
                elif ltp:
                    # Have LTP but no OHLCV - show minimal card
                    st.markdown(f"### {symbol}")
                    st.metric(label=metadata.get('full_name', symbol), 
                             value=f"₹{ltp:,.2f}",
                             delta="Live")
                else:
                    # All sources failed - show warning
                    st.warning(f"⚠️ {symbol}: No data available")
                    
            except Exception as e:
                logger.error(f"Error displaying {symbol}: {e}")
                st.error(f"❌ {symbol}: Error")

def display_constituent_table(index_name: str, full_index_name: str):
    """Display constituent stocks in a sortable table
    
    Args:
        index_name: Short index name
        full_index_name: Full index name for constituent lookup
    """
    st.markdown(f"### 📈 {index_name} CONSTITUENTS")
    
    # Get constituent data
    df = get_constituent_data(full_index_name)
    
    if df.empty:
        st.info(f"📊 Loading constituent data for {index_name}...")
        return
    
    # Add filters
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Change %", "Volume", "LTP", "Symbol"],
            key=f"sort_{index_name}"
        )
    
    with col2:
        filter_type = st.selectbox(
            "Filter:",
            ["All", "Gainers Only", "Losers Only", "Most Active"],
            key=f"filter_{index_name}"
        )
    
    with col3:
        limit = st.number_input(
            "Show top:",
            min_value=5,
            max_value=len(df),
            value=min(20, len(df)),
            step=5,
            key=f"limit_{index_name}"
        )
    
    # Apply filters
    if filter_type == "Gainers Only":
        df = df[df['change_pct'] > 0]
    elif filter_type == "Losers Only":
        df = df[df['change_pct'] < 0]
    elif filter_type == "Most Active":
        df = df.nlargest(limit, 'volume')
    
    # Sort
    sort_column_map = {
        "Change %": "change_pct",
        "Volume": "volume",
        "LTP": "ltp",
        "Symbol": "symbol"
    }
    sort_col = sort_column_map[sort_by]
    df = df.sort_values(sort_col, ascending=(sort_by == "Symbol"))
    
    # Limit rows
    df = df.head(int(limit))
    
    # Format for display
    display_df = pd.DataFrame({
        'Symbol': df['symbol'],
        'Open': df['open'].apply(lambda x: f"₹{x:,.2f}"),
        'High': df['high'].apply(lambda x: f"₹{x:,.2f}"),
        'Low': df['low'].apply(lambda x: f"₹{x:,.2f}"),
        'LTP': df['ltp'].apply(lambda x: f"₹{x:,.2f}"),
        'Volume': df['volume'].apply(format_large_number),
        'Change': df['change_pct'].apply(lambda x: f"{x:+.2f}%")
    })
    
    # Display with color coding
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    advancers = len(df[df['change_pct'] > 0])
    decliners = len(df[df['change_pct'] < 0])
    unchanged = len(df[df['change_pct'] == 0])
    total_volume = df['volume'].sum()
    
    with col1:
        st.metric("🟢 Advancers", advancers)
    with col2:
        st.metric("🔴 Decliners", decliners)
    with col3:
        st.metric("⚪ Unchanged", unchanged)
    with col4:
        st.metric("📊 Total Volume", format_large_number(total_volume))

def display_index_tabs(indices: Dict[str, str]):
    """Display tabbed interface for index constituents
    
    Args:
        indices: Dict mapping short names to full names
    """
    st.markdown("---")
    st.markdown("### 📑 INDEX CONSTITUENTS — Detailed View")
    
    # Create tabs
    tab_names = list(indices.keys())
    tabs = st.tabs(tab_names)
    
    # Display content for each tab
    for tab, (short_name, full_name) in zip(tabs, indices.items()):
        with tab:
            display_constituent_table(short_name, full_name)

def display_quick_metrics():
    """Display quick system metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Capital",
            "₹1,00,000",
            delta=None
        )
    
    with col2:
        st.metric(
            "📊 Open Trades",
            "0",
            delta="Paper Mode"
        )
    
    with col3:
        st.metric(
            "📈 Today's P&L",
            "₹0.00",
            delta="+0.00%"
        )
    
    with col4:
        st.metric(
            "🎯 Win Rate",
            "N/A",
            delta="Not trading yet"
        )
