"""
Auto-Scanner Service - Automatically runs live scanner during market hours
Implements three-tier data staleness strategy:
1. Before market: Data from previous trading day (verified by BOD)
2. During market: Max 20 minutes stale (with tiered warnings)
3. After market: EOD process syncs all data
4. Trading holiday: Use previous trading day data
"""

import streamlit as st
import pytz
from datetime import datetime, timedelta
import os
import json
from utils.logger import logger
from utils.market_hours import is_market_open, get_market_status
from utils.helpers import get_kite_instance


def get_latest_data_timestamp():
    """Get the timestamp of the latest data file in data/day/ directory"""
    try:
        data_dir = "data/day"
        if not os.path.exists(data_dir):
            return None
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not files:
            return None
        
        # Get the most recent file's modification time
        latest_file = max(
            [os.path.join(data_dir, f) for f in files],
            key=os.path.getmtime
        )
        
        mtime = os.path.getmtime(latest_file)
        return datetime.fromtimestamp(mtime)
    except Exception as e:
        logger.debug(f"Error getting latest data timestamp: {e}")
        return None


def get_data_age_minutes():
    """Get age of data in minutes"""
    try:
        latest_ts = get_latest_data_timestamp()
        if not latest_ts:
            return None
        
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        age_delta = now - latest_ts.replace(tzinfo=ist)
        age_minutes = int(age_delta.total_seconds() / 60)
        return age_minutes
    except Exception as e:
        logger.debug(f"Error calculating data age: {e}")
        return None


def get_staleness_status():
    """
    Get detailed staleness status with color coding
    Returns: (status, color, message, pause_scanner)
    """
    age_minutes = get_data_age_minutes()
    
    if age_minutes is None:
        return "UNKNOWN", "gray", "No data found", True
    
    # During market hours (9:15 AM - 3:30 PM IST)
    market_status = get_market_status()
    is_market_hours = market_status.get('is_open', False)
    
    if is_market_hours:
        # DURING MARKET - Tiered alerts
        if age_minutes <= 10:
            return "FRESH", "green", f"✅ Fresh ({age_minutes}m old)", False
        elif age_minutes <= 15:
            return "GOOD", "green", f"✅ Good ({age_minutes}m old)", False
        elif age_minutes <= 20:
            return "CAUTION", "orange", f"⚠️ Caution ({age_minutes}m old) - Auto-refreshing", False
        elif age_minutes <= 25:
            return "STALE", "red", f"🔴 Stale ({age_minutes}m old) - Pair trading paused", True
        else:
            return "CRITICAL", "red", f"⛔ Critical ({age_minutes}m old) - All trading paused", True
    
    else:
        # OUTSIDE MARKET HOURS
        if age_minutes <= 60:
            return "FRESH", "green", f"✅ Fresh ({age_minutes}m old)", False
        elif age_minutes <= 240:  # 4 hours
            return "GOOD", "green", f"✅ Good ({age_minutes}m old)", False
        elif age_minutes <= 1440:  # 24 hours
            return "OLD", "orange", f"⚠️ Previous trading day data ({age_minutes//60}h old)", False
        else:
            return "STALE", "red", f"🔴 Stale ({age_minutes//60}h+ old) - Run EOD download", True


def is_data_fresh(max_age_hours=4):
    """Check if we have fresh data"""
    try:
        age_minutes = get_data_age_minutes()
        if age_minutes is None:
            return False
        
        max_age_minutes = max_age_hours * 60
        return age_minutes < max_age_minutes
    except Exception as e:
        logger.debug(f"Error checking data freshness: {e}")
        return False


def should_auto_run_scanner():
    """
    Determine if scanner should auto-run:
    - Market is open (9:15 AM - 3:30 PM IST)
    - Data is fresh (within 4 hours)
    - Session not paused
    """
    try:
        # Check market hours
        market_status = get_market_status()
        if not market_status.get('is_open', False):
            return False, "Market is closed"
        
        # Check data freshness
        if not is_data_fresh():
            return False, "Data not fresh (>4 hours old)"
        
        # Check if user disabled auto-run
        if st.session_state.get('disable_auto_scanner', False):
            return False, "Auto-run disabled"
        
        return True, "Ready to scan"
    except Exception as e:
        logger.debug(f"Error in should_auto_run_scanner: {e}")
        return False, f"Error: {str(e)}"


def render_auto_scanner_status():
    """
    Render auto-scanner status in sidebar with data staleness indicators
    Shows three-tier staleness strategy based on time of day
    """
    try:
        market_status = get_market_status()
        should_run, reason = should_auto_run_scanner()
        staleness_status, color, staleness_msg, pause_scanner = get_staleness_status()
        
        st.markdown("### 📡 AUTO SCANNER STATUS")
        
        # Market Status
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if market_status.get('is_open', False):
                    ist = pytz.timezone('Asia/Kolkata')
                    now = datetime.now(ist)
                    close_time = market_status.get('close_time')
                    
                    if close_time:
                        # Parse close time
                        close_dt = datetime.strptime(close_time, "%H:%M").replace(
                            year=now.year, month=now.month, day=now.day
                        )
                        time_left = close_dt - now
                        hours = int(time_left.total_seconds() // 3600)
                        mins = int((time_left.total_seconds() % 3600) // 60)
                        
                        st.write(f"🟢 **Market: OPEN** | Close: {close_time} ({hours}h {mins}m)")
                    else:
                        st.write("🟢 **Market: OPEN**")
                else:
                    next_open = market_status.get('next_open_time', 'Tomorrow 9:15 AM')
                    st.write(f"🔴 **Market: CLOSED** | Opens: {next_open}")
            
            with col2:
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).strftime("%H:%M:%S")
                st.metric("IST Time", current_time, label_visibility="collapsed")
        
        # Data Staleness Status (Three-tier strategy)
        with st.container(border=True):
            age_minutes = get_data_age_minutes()
            latest_ts = get_latest_data_timestamp()
            
            if latest_ts:
                ist = pytz.timezone('Asia/Kolkata')
                now = datetime.now(ist)
                age_delta = now - latest_ts.replace(tzinfo=ist)
                
                # Show staleness message with color
                if staleness_status == "FRESH":
                    st.success(f"✅ **Data Status:** {staleness_msg}", icon="✅")
                elif staleness_status == "GOOD":
                    st.success(f"✅ **Data Status:** {staleness_msg}", icon="✅")
                elif staleness_status == "CAUTION":
                    st.warning(f"⚠️ **Data Status:** {staleness_msg}", icon="⚠️")
                elif staleness_status == "STALE":
                    st.warning(f"🔴 **Data Status:** {staleness_msg}", icon="⚠️")
                else:
                    st.error(f"⛔ **Data Status:** {staleness_msg}", icon="❌")
                
                # Show age breakdown
                hours = int(age_delta.total_seconds() // 3600)
                mins = int((age_delta.total_seconds() % 3600) // 60)
                if hours > 0:
                    age_text = f"{hours}h {mins}m"
                else:
                    age_text = f"{mins}m"
                
                st.caption(f"Last update: {age_text} ago")
                
                # Show staleness strategy info
                if market_status.get('is_open', False):
                    st.caption("During market: Max 20m staleness allowed")
                else:
                    st.caption("After hours: Relying on EOD sync")
            else:
                st.error("**❌ No data found**", icon="❌")
                st.caption("Go to Data Management to download latest data")
        
        # Auto-run Status
        with st.container(border=True):
            if should_run and not pause_scanner:
                st.success("✅ AUTO SCANNER: READY", icon="✅")
                st.caption(f"Status: {reason}")
                
                # Toggle to disable
                if st.checkbox("Pause auto-run", key="pause_auto_scanner"):
                    st.session_state.disable_auto_scanner = True
                    st.rerun()
            else:
                if pause_scanner:
                    st.error(f"❌ AUTO SCANNER: PAUSED (Data stale)", icon="⛔")
                    st.caption(f"Reason: {staleness_msg}")
                else:
                    st.warning(f"❌ AUTO SCANNER: NOT READY", icon="⚠️")
                    st.caption(f"Reason: {reason}")
                
                # Show why it's not running
                if "Market is closed" in reason:
                    st.info("Scanner will auto-start when market opens at 9:15 AM IST")
                elif "Data not fresh" in reason:
                    st.info("Please download latest data in Data Management")
                
                # Allow re-enabling
                if st.checkbox("Enable when ready", key="enable_auto_scanner"):
                    st.session_state.disable_auto_scanner = False
                    st.rerun()
        
        # Data Staleness Strategy Info
        with st.expander("📊 Data Staleness Strategy", expanded=False):
            st.markdown("""
**Three-Tier Data Freshness Strategy:**

**Before Market (< 9:15 AM):**
- ✅ Data from previous trading day
- ✅ BOD verifies completeness
- Auto-scanner ready when market opens

**During Market (9:15 AM - 3:30 PM):**
- 🟢 0-10 min: Fresh (all trading)
- 🟡 10-15 min: Good (all trading)
- 🟠 15-20 min: Caution (auto-refresh)
- 🔴 20-25 min: Stale (pair trading paused)
- ⛔ > 25 min: Critical (all trading paused)

**After Market (> 3:30 PM):**
- ✅ EOD download at 5:45 PM syncs all data
- ✅ Ready by 8:00 PM for next day

**Trading Holiday:**
- ✅ Use previous trading day data
- ✅ Skip BOD/EOD processes
            """)
        
    except Exception as e:
        logger.error(f"Error rendering auto-scanner status: {e}")
        st.error(f"Error loading scanner status: {str(e)}")


def init_auto_scanner_session():
    """Initialize auto-scanner session state"""
    if "disable_auto_scanner" not in st.session_state:
        st.session_state.disable_auto_scanner = False
    
    if "last_scan_time" not in st.session_state:
        st.session_state.last_scan_time = None
    
    if "live_scan_active" not in st.session_state:
        st.session_state.live_scan_active = False


def trigger_auto_scan():
    """Trigger auto-scan if conditions are met"""
    try:
        should_run, _ = should_auto_run_scanner()
        
        if should_run:
            # Check if enough time has passed since last scan (minimum 60 seconds)
            last_scan = st.session_state.get('last_scan_time')
            if last_scan is None or (datetime.now() - last_scan).total_seconds() > 60:
                st.session_state.live_scan_active = True
                st.session_state.last_scan_time = datetime.now()
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error triggering auto-scan: {e}")
        return False
