"""
utils/market_display.py - Market data display utilities for NIFTY50 stocks
"""
import pandas as pd
import streamlit as st
from utils.helpers import get_ltp
from typing import Optional
import plotly.graph_objects as go

def get_nifty50_stocks(universe_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Get all NIFTY50 stocks from universe
    
    Args:
        universe_df: Universe dataframe. If None, loads from symbols module.
    
    Returns:
        DataFrame with NIFTY50 stocks
    """
    if universe_df is None:
        from symbols import load_universe
        universe_df = load_universe()
    
    nifty50 = universe_df[universe_df.get('In_NIFTY50', 'N') == 'Y'].copy()
    return nifty50[['Symbol', 'Name', 'Industry']].reset_index(drop=True)

def display_nifty50_scrollable(universe_df: Optional[pd.DataFrame] = None, kite=None):
    """Display NIFTY50 stocks in a scrollable, interactive format
    
    Args:
        universe_df: Universe dataframe
        kite: KiteConnect instance for live prices
    """
    nifty50 = get_nifty50_stocks(universe_df)
    
    st.markdown("### 📈 NIFTY 50 Stocks")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.write(f"**Total Stocks:** {len(nifty50)}")
    with col2:
        st.write(f"**Data Status:** ✅ Live")
    with col3:
        refresh = st.button("🔄 Refresh", key="refresh_nifty50")
    
    # Create scrollable container with dataframe display
    # Add search/filter capability
    search = st.text_input("🔍 Search stocks...", "", key="nifty50_search")
    
    # Filter based on search
    if search:
        filtered = nifty50[
            nifty50['Symbol'].str.contains(search, case=False, na=False) |
            nifty50['Name'].str.contains(search, case=False, na=False)
        ]
    else:
        filtered = nifty50
    
    # Display as interactive table
    if kite:
        # Fetch live prices for all stocks
        display_data = []
        for idx, row in filtered.iterrows():
            symbol = row['Symbol']
            try:
                price = get_ltp(symbol, "NSE")
                # Ensure price is a float, not a dict or other type
                if isinstance(price, dict):
                    price = price.get('last_price') if 'last_price' in price else None
                price = float(price) if price else None
            except (TypeError, ValueError):
                price = None
            
            display_data.append({
                'Symbol': symbol,
                'Name': row['Name'][:25] if pd.notna(row['Name']) else "N/A",
                'Industry': row['Industry'][:20] if pd.notna(row['Industry']) else "N/A",
                'Price': f"₹{price:,.2f}" if price else "N/A",
                'Status': '✅' if price else '⏳'
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(
            display_df,
            width='stretch',
            height=400,
            hide_index=True
        )
    else:
        # Just show static data without prices
        display_df = filtered[['Symbol', 'Name', 'Industry']].copy()
        st.dataframe(
            display_df,
            width='stretch',
            height=400,
            hide_index=True
        )
    
    st.caption(f"📊 Showing {len(filtered)} of {len(nifty50)} stocks")

def display_nifty50_grid(universe_df: Optional[pd.DataFrame] = None, kite=None):
    """Display NIFTY50 stocks in a card grid layout
    
    Args:
        universe_df: Universe dataframe
        kite: KiteConnect instance for live prices
    """
    nifty50 = get_nifty50_stocks(universe_df)
    
    st.markdown("### 📈 NIFTY 50 Stocks Grid")
    
    # Create a responsive grid (4 columns on wide screens)
    cols_per_row = 4
    cols = st.columns(cols_per_row)
    
    for idx, (_, row) in enumerate(nifty50.head(50).iterrows()):
        col_idx = idx % cols_per_row
        symbol = row['Symbol']
        
        with cols[col_idx]:
            try:
                price = get_ltp(symbol, "NSE") if kite else None
                # Ensure price is a float, not a dict or other type
                if isinstance(price, dict):
                    price = price.get('last_price') if 'last_price' in price else None
                price = float(price) if price else None
            except (TypeError, ValueError):
                price = None
            
            # Create card
            price_str = f'₹{price:,.2f}' if price else 'Loading...'
            price_color = '#00FF00' if price else '#888'
            
            card_html = f"""
            <div style='
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px;
                border-radius: 10px;
                border-left: 3px solid #FFD700;
                margin-bottom: 10px;
                text-align: center;
            '>
                <div style='color: #FFD700; font-weight: bold; font-size: 14px;'>
                    {symbol}
                </div>
                <div style='color: #C9D1D9; font-size: 12px; margin-top: 5px;'>
                    {row['Industry'][:15] if pd.notna(row['Industry']) else "N/A"}
                </div>
                <div style='color: {price_color}; font-size: 13px; margin-top: 5px; font-weight: bold;'>
                    {price_str}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

def create_nifty50_watchlist_html(universe_df: Optional[pd.DataFrame] = None) -> str:
    """Create HTML for NIFTY50 watchlist display
    
    Args:
        universe_df: Universe dataframe
    
    Returns:
        HTML string
    """
    nifty50 = get_nifty50_stocks(universe_df)
    
    # Create a simple HTML table
    html_rows = []
    for idx, row in enumerate(nifty50.head(25).iterrows(), 1):
        _, data = row
        html_rows.append(
            f"<tr>"
            f"<td style='padding: 10px; border-bottom: 1px solid #333;'>{data['Symbol']}</td>"
            f"<td style='padding: 10px; border-bottom: 1px solid #333; color: #888; font-size: 12px;'>"
            f"{data['Industry'][:20]}</td>"
            f"</tr>"
        )
    
    table_html = f"""
    <div style='background: #0d1117; border-radius: 10px; padding: 15px; max-height: 500px; overflow-y: auto;'>
        <table style='width: 100%; border-collapse: collapse;'>
            <thead>
                <tr style='border-bottom: 2px solid #FFD700;'>
                    <th style='padding: 10px; color: #FFD700; text-align: left;'>Symbol</th>
                    <th style='padding: 10px; color: #FFD700; text-align: left;'>Sector</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    
    return table_html
