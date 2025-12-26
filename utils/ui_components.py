"""
ui_components.py - Reusable UI components for static header and footer
Provides consistent header and footer across all Streamlit pages
"""

import streamlit as st
from datetime import datetime

def render_header():
    """
    Renders a consistent header across all pages
    Includes app title, connection status, and user info if logged in
    """
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📈 Artemis Signals by ManekBaba")
    
    with col2:
        # Connection status
        if "kiteconnect_session" in st.session_state and st.session_state.kiteconnect_session:
            st.markdown("<div style='text-align: center; color: green;'><small>🟢 Connected</small></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; color: orange;'><small>🟡 Demo Mode</small></div>", unsafe_allow_html=True)
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"<div style='text-align: right;'><small>{current_time} IST</small></div>", unsafe_allow_html=True)
    
    st.divider()


def render_footer():
    """
    Renders a consistent footer across all pages
    Includes last update time, version info, and support links
    """
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.caption("🔄 Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with col2:
        from config import COMMERCIAL_MODE, PROJECT_NAME
        if not COMMERCIAL_MODE:
            # Personal mode: Professional caption
            st.caption("Systematic Trading | Built with Discipline")
        else:
            st.caption(f"{PROJECT_NAME} | Professional Trading Platform v2.0")
    
    with col3:
        st.caption("Pair Trading | Options Strategies | Live Scanner")


def render_user_badge():
    """
    Renders user information badge in sidebar
    Shows logged in user, capital, and account type
    """
    if "user_name" in st.session_state and st.session_state.user_name:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 **{st.session_state.user_name}**")
        
        if "account_type" in st.session_state:
            st.sidebar.caption(f"Account: {st.session_state.account_type}")
        
        if "capital" in st.session_state:
            st.sidebar.caption(f"Capital: ₹{st.session_state.capital:,.0f}")
        
        st.sidebar.markdown("---")


def render_status_bar(live_positions=0, pending_orders=0, open_pnl=0.0):
    """
    Renders a status bar with key trading metrics
    
    Parameters:
    - live_positions: Number of open positions
    - pending_orders: Number of pending orders
    - open_pnl: Current unrealised P&L
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Open Positions", live_positions, delta=None)
    with col2:
        st.metric("Pending Orders", pending_orders, delta=None)
    with col3:
        if open_pnl >= 0:
            st.metric("Open P&L", f"₹{open_pnl:,.0f}", delta=f"₹{open_pnl:,.0f}")
        else:
            st.metric("Open P&L", f"₹{open_pnl:,.0f}", delta=f"₹{open_pnl:,.0f}")
    with col4:
        total_capital = st.session_state.get("capital", 1_000_000)
        available = total_capital - (live_positions * 50000)  # Mock calculation
        st.metric("Available Margin", f"₹{available:,.0f}", delta=None)


def render_quick_actions():
    """
    Renders quick action buttons for common trading operations
    """
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📞 Support"):
            st.info("Support contact: support@svtrading.com")
    
    with col2:
        if st.button("📖 Help"):
            st.info("Visit docs for detailed guides")
    
    if st.sidebar.button("⚙️ Settings"):
        st.session_state.show_settings = not st.session_state.get("show_settings", False)


def render_market_status():
    """
    Renders current market status and important times
    """
    current_time = datetime.now()
    
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()
    current = current_time.time()
    
    st.sidebar.markdown("### 📊 Market Status")
    
    if market_open <= current <= market_close:
        st.sidebar.markdown("<div style='color: green;'>🟢 **MARKET OPEN**</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div style='color: red;'>🔴 **MARKET CLOSED**</div>", unsafe_allow_html=True)
    
    st.sidebar.caption(f"Current: {current.strftime('%H:%M:%S')}")
    st.sidebar.caption(f"Opens: 09:15 | Closes: 15:30")


def render_data_refresh_indicator(last_refresh_time=None):
    """
    Renders data refresh timestamp and refresh status
    
    Parameters:
    - last_refresh_time: datetime of last data refresh
    """
    if last_refresh_time:
        time_diff = (datetime.now() - last_refresh_time).total_seconds()
        
        if time_diff < 60:
            status = "🟢 Fresh"
        elif time_diff < 300:
            status = "🟡 Recent"
        else:
            status = "🔴 Stale"
        
        st.caption(f"Data Status: {status} | Updated: {last_refresh_time.strftime('%H:%M:%S')}")
    else:
        st.caption("Data Status: 🟡 Not loaded")
