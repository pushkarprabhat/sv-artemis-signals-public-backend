"""
Data Refresh Status Module - Displays background data refresh progress in dashboard
Shows 3-min, 5-min, and daily data refresh status with live metrics
"""

import streamlit as st
import datetime as dt
from pathlib import Path
from typing import Dict, Optional
import json
from utils.logger import logger

# Status file for inter-process communication with background service
STATUS_FILE = Path(__file__).parent.parent / "logs" / "refresh_status.json"


def get_background_refresh_status() -> Dict:
    """
    Get current background data refresh status from background service
    Reads from refresh_status.json file
    
    Returns:
        Dict with keys:
        - '3min': {'last_refresh': timestamp, 'symbols': count, 'status': 'downloading/idle'}
        - '5min': {'last_refresh': timestamp, 'symbols': count, 'status': 'downloading/idle'}
        - 'daily': {'last_refresh': timestamp, 'symbols': count, 'status': 'downloading/idle'}
        - 'overall': {'active': bool, 'last_update': timestamp}
    """
    default_status = {
        '3min': {
            'last_refresh': None,
            'last_refresh_time': 'Never',
            'symbols_downloaded': 0,
            'status': 'pending',
            'coverage_pct': 0
        },
        '5min': {
            'last_refresh': None,
            'last_refresh_time': 'Never',
            'symbols_downloaded': 0,
            'status': 'pending',
            'coverage_pct': 0
        },
        'daily': {
            'last_refresh': None,
            'last_refresh_time': 'Never',
            'symbols_downloaded': 0,
            'status': 'pending',
            'coverage_pct': 0
        },
        'overall': {
            'active': False,
            'last_update': dt.datetime.now().isoformat(),
            'uptime': '0m'
        }
    }
    
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read refresh status file: {e}")
    
    return default_status


def update_refresh_status(interval: str, symbols_count: int, success: bool = True):
    """
    Update refresh status for a specific interval
    Called by background service to report progress
    
    Args:
        interval: '3min', '5min', or 'daily'
        symbols_count: Number of symbols just downloaded
        success: Whether download was successful
    """
    try:
        status = get_background_refresh_status()
        
        status[interval] = {
            'last_refresh': dt.datetime.now().isoformat(),
            'last_refresh_time': dt.datetime.now().strftime('%H:%M:%S'),
            'symbols_downloaded': symbols_count,
            'status': 'success' if success else 'failed',
            'coverage_pct': min(100, (symbols_count / 8575) * 100) if interval == '5min' else 100
        }
        
        status['overall']['last_update'] = dt.datetime.now().isoformat()
        
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not update refresh status: {e}")


def render_data_refresh_status():
    """
    Render data refresh status widget in Streamlit dashboard
    Shows 3-min, 5-min, daily refresh status with progress indicators
    """
    status = get_background_refresh_status()
    
    # Get current status
    refresh_3min = status.get('3min', {})
    refresh_5min = status.get('5min', {})
    refresh_daily = status.get('daily', {})
    overall = status.get('overall', {})
    
    # Determine if service is active
    is_active = overall.get('active', False)
    
    # Calculate time since last refresh
    def time_since_refresh(last_refresh_str: Optional[str]) -> str:
        if not last_refresh_str:
            return "Never"
        try:
            last_refresh = dt.datetime.fromisoformat(last_refresh_str)
            now = dt.datetime.now()
            delta = now - last_refresh
            
            if delta.total_seconds() < 60:
                return "Just now"
            elif delta.total_seconds() < 3600:
                mins = int(delta.total_seconds() / 60)
                return f"{mins}m ago"
            elif delta.total_seconds() < 86400:
                hours = int(delta.total_seconds() / 3600)
                return f"{hours}h ago"
            else:
                days = int(delta.total_seconds() / 86400)
                return f"{days}d ago"
        except:
            return last_refresh_str
    
    # Render main container
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])
        
        # Service Status
        with col1:
            status_emoji = "🟢" if is_active else "🔴"
            st.metric(
                label="📊 Data Refresh Service",
                value=f"{status_emoji} {'ACTIVE' if is_active else 'IDLE'}"
            )
        
        # Overall Status
        with col2:
            last_update = overall.get('last_update')
            st.metric(
                label="⏱️ Last Update",
                value=time_since_refresh(last_update)
            )
        
        # Coverage
        coverage = max(
            refresh_3min.get('coverage_pct', 0),
            refresh_5min.get('coverage_pct', 0)
        )
        with col3:
            st.metric(
                label="📈 Data Coverage",
                value=f"{coverage:.1f}%"
            )
        
        with col4:
            pass  # Spacer
    
    # Detail rows for each interval
    st.markdown("### 📡 Refresh Progress")
    
    # Create three columns for 3min, 5min, daily
    col1, col2, col3 = st.columns(3)
    
    # 3-MINUTE DATA
    with col1:
        with st.container(border=True):
            st.markdown("#### 3-Min Data")
            
            status_3min = refresh_3min.get('status', 'pending')
            if status_3min == 'success':
                st.success("✅ Downloaded", icon="✓")
            elif status_3min == 'failed':
                st.error("❌ Failed", icon="✗")
            else:
                st.info("⏳ Pending", icon="⏱")
            
            st.caption(f"**Last:** {time_since_refresh(refresh_3min.get('last_refresh'))}")
            st.caption(f"**Symbols:** {refresh_3min.get('symbols_downloaded', 0)}")
    
    # 5-MINUTE DATA
    with col2:
        with st.container(border=True):
            st.markdown("#### 5-Min Data")
            
            status_5min = refresh_5min.get('status', 'pending')
            if status_5min == 'success':
                st.success("✅ Downloaded", icon="✓")
            elif status_5min == 'failed':
                st.error("❌ Failed", icon="✗")
            else:
                st.info("⏳ Pending", icon="⏱")
            
            st.caption(f"**Last:** {time_since_refresh(refresh_5min.get('last_refresh'))}")
            st.caption(f"**Symbols:** {refresh_5min.get('symbols_downloaded', 0)}")
            
            # Progress bar
            coverage_5min = refresh_5min.get('coverage_pct', 0) / 100
            st.progress(coverage_5min, text=f"{coverage_5min*100:.0f}%")
    
    # DAILY DATA
    with col3:
        with st.container(border=True):
            st.markdown("#### Daily Data")
            
            status_daily = refresh_daily.get('status', 'pending')
            if status_daily == 'success':
                st.success("✅ Downloaded", icon="✓")
            elif status_daily == 'failed':
                st.error("❌ Failed", icon="✗")
            else:
                st.info("⏳ Pending", icon="⏱")
            
            st.caption(f"**Last:** {time_since_refresh(refresh_daily.get('last_refresh'))}")
            st.caption(f"**Symbols:** {refresh_daily.get('symbols_downloaded', 0)}")


def render_inline_refresh_status():
    """
    Render compact refresh status for sidebar
    Shows quick status indicators for each interval
    """
    status = get_background_refresh_status()
    
    refresh_3min = status.get('3min', {})
    refresh_5min = status.get('5min', {})
    refresh_daily = status.get('daily', {})
    overall = status.get('overall', {})
    
    is_active = overall.get('active', False)
    
    # Quick status indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Refresh Status")
    
    # Service status
    if is_active:
        st.sidebar.success("🟢 Service Active")
    else:
        st.sidebar.info("🔴 Service Idle")
    
    # Interval status
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        status_3min = refresh_3min.get('status', 'pending')
        if status_3min == 'success':
            st.markdown("**3M** ✅")
        elif status_3min == 'failed':
            st.markdown("**3M** ❌")
        else:
            st.markdown("**3M** ⏳")
    
    with col2:
        status_5min = refresh_5min.get('status', 'pending')
        if status_5min == 'success':
            st.markdown("**5M** ✅")
        elif status_5min == 'failed':
            st.markdown("**5M** ❌")
        else:
            st.markdown("**5M** ⏳")
    
    with col3:
        status_daily = refresh_daily.get('status', 'pending')
        if status_daily == 'success':
            st.markdown("**Day** ✅")
        elif status_daily == 'failed':
            st.markdown("**Day** ❌")
        else:
            st.markdown("**Day** ⏳")


if __name__ == "__main__":
    # Test: Display the status widget
    st.title("Data Refresh Status - Test")
    render_data_refresh_status()
