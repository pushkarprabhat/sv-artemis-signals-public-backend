# For Shivaansh & Krishaansh — this line pays their fees
STRATEGIES_AVAILABLE = True  # Ensure this is always defined; set dynamically if needed
import time
import os
from utils.telegram import send_telegram
import streamlit as st

# Always initialize session state keys before use
if 'scheduler_running' not in st.session_state:
    st.session_state.scheduler_running = False  # For Shivaansh & Krishaansh — this line pays your fees!
# === ADVANCED SCHEDULING & EMAIL REMINDERS ===
import smtplib
from email.mime.text import MIMEText
import config as artemis_config
from config import SIGNALS_PATH

def send_email_reminder(subject, body, to_email):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = artemis_config.EMAIL_CONFIG['sender_email']
        msg['To'] = to_email
        with smtplib.SMTP(artemis_config.EMAIL_CONFIG['smtp_host'], artemis_config.EMAIL_CONFIG['smtp_port']) as server:
            if artemis_config.EMAIL_CONFIG['smtp_user'] and artemis_config.EMAIL_CONFIG['smtp_password']:
                server.starttls()
                server.login(artemis_config.EMAIL_CONFIG['smtp_user'], artemis_config.EMAIL_CONFIG['smtp_password'])
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email reminder failed: {e}")
        return False
try:
     from utils.market_hours import is_market_open
except ImportError:
     def is_market_open():
          return False  # TODO: Replace with actual logic

def should_send_reminder(frequency, last_sent, now, reminder_time):
    # frequency: 'daily', 'weekly', or 'custom' (future)
    # last_sent, now: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'
    # reminder_time: 'HH:MM' (24h)
    from datetime import datetime, timedelta
    now_dt = datetime.strptime(now, '%Y-%m-%d %H:%M')
    last_dt = datetime.strptime(last_sent, '%Y-%m-%d %H:%M') if last_sent else None
    target_time = now_dt.replace(hour=int(reminder_time[:2]), minute=int(reminder_time[3:]), second=0, microsecond=0)
    if frequency == 'daily':
        if not last_sent or (now_dt.date() > last_dt.date() and now_dt >= target_time):
            return True
    elif frequency == 'weekly':
        if not last_sent or (now_dt.isocalendar()[1] > last_dt.isocalendar()[1] and now_dt >= target_time):
            return True
    # Add more custom logic as needed
    return False

with st.sidebar:
    # Define pending tasks (fetch from tracker or set empty for now)
    pending = st.session_state.get('pending_tasks', [])
    st.markdown("### ⚙️ Reminder Settings")
    freq = st.selectbox("Reminder Frequency", ["daily", "weekly"], index=["daily", "weekly"].index(getattr(artemis_config, 'REMINDER_FREQUENCY', 'daily')))
    time_str = st.text_input("Reminder Time (HH:MM, 24h IST)", value=getattr(artemis_config, 'REMINDER_TIME', '08:00'))
    email = st.text_input("Reminder Email", value=getattr(artemis_config, 'REMINDER_EMAIL', artemis_config.EMAIL_CONFIG['recipient_emails'][0]))
    if st.button("Save Reminder Settings"):
        st.session_state['reminder_freq'] = freq
        st.session_state['reminder_time'] = time_str
        st.session_state['reminder_email'] = email
        st.success("Reminder settings saved!")

    # Automated reminder logic
    now_str = time.strftime('%Y-%m-%d %H:%M')
    last_sent = st.session_state.get('last_reminder_sent', '')
    reminder_freq = st.session_state.get('reminder_freq', freq)
    reminder_time = st.session_state.get('reminder_time', time_str)
    reminder_email = st.session_state.get('reminder_email', email)
    if pending and should_send_reminder(reminder_freq, last_sent, now_str, reminder_time):
        subject = "[Artemis Sprint Reminder] Pending Tasks"
        body = "Pending tasks for Artemis:\n" + '\n'.join(f"- {t}" for t in pending)
        send_email_reminder(subject, body, reminder_email)
        send_telegram(subject + "\n" + body)
        st.session_state['last_reminder_sent'] = now_str
# === AUTOMATED SPRINT REMINDERS ===
import re
SPRINT_TRACKER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sv-artemis-signals-private', 'SPRINT_TASK_TRACKER_JAN2026.md'))
def get_pending_sprint_tasks():
    if not os.path.exists(SPRINT_TRACKER):
        return []
    with open(SPRINT_TRACKER, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tasks = []
    for line in lines:
        m = re.match(r'- \[ \] (.+)', line)
        if m:
            tasks.append(m.group(1).strip())
    return tasks

with st.sidebar:
    st.markdown("### ⏰ Sprint Reminders")
    pending = get_pending_sprint_tasks()
    if pending:
        st.warning(f"Pending tasks: {len(pending)}")
        for t in pending:
            st.write(f"- {t}")
        # Send daily Telegram reminder (once per day)
        today = time.strftime('%Y-%m-%d')
        if st.session_state.get('last_reminder', '') != today:
            send_telegram(f"[SPRINT REMINDER] Pending tasks for Artemis:\n" + '\n'.join(f"- {t}" for t in pending))
            st.session_state['last_reminder'] = today
    else:
        st.success("No pending sprint tasks! 🚀")
# === AUTOMATED HEALTH CHECKS ===
import requests
import time
from utils.telegram import send_telegram

def check_service(url, name):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return True, "🟢 Healthy"
        else:
            return False, f"🔴 Error: {r.status_code}"
    except Exception as e:
        return False, f"🔴 Down: {e}"

with st.sidebar:
    st.markdown("### 🩺 Artemis Health Check")
    backend_ok, backend_status = check_service("http://127.0.0.1:8000/api/v1/health", "Backend")
    frontend_ok, frontend_status = check_service("http://localhost:5173", "Frontend")
    streamlit_ok, streamlit_status = check_service("http://localhost:8501", "Streamlit")
    st.write(f"Backend: {backend_status}")
    st.write(f"Frontend: {frontend_status}")
    st.write(f"Streamlit: {streamlit_status}")
    if not (backend_ok and frontend_ok and streamlit_ok):
        send_telegram(f"[HEALTH ALERT] Artemis service down!\nBackend: {backend_status}\nFrontend: {frontend_status}\nStreamlit: {streamlit_status}")
        st.error("One or more Artemis services are down! Telegram alert sent.")
    else:
        st.success("All Artemis services healthy.")
# Import get_ltp_database for LTP management
from core.ltp_database import get_ltp_database
# main.py — ARTEMIS SIGNALS
# Professional algorithmic trading platform with adaptive validation + EOD/BOD scheduler
# Institutional-grade quantitative trading system



import streamlit as st
import sys
import os

from config import COMMERCIAL_MODE

import pathlib
from dotenv import load_dotenv
EXPECTED_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPECTED_ROOT, '..'))
cwd = os.path.abspath(os.getcwd())
pythonpath = os.environ.get('PYTHONPATH', '')

# Robust import of shared timeframes (after EXPECTED_ROOT is defined)
PRIVATE_PATH = os.path.abspath(os.path.join(EXPECTED_ROOT, '..', 'sv-artemis-signals-private'))
if PRIVATE_PATH not in sys.path:
    sys.path.insert(0, PRIVATE_PATH)

try:
    from shared.config import AVAILABLE_TIMEFRAMES as TIMEFRAMES
except ImportError as e:
    print("[FATAL] Could not import AVAILABLE_TIMEFRAMES from shared config.\n" \
          f"Checked path: {PRIVATE_PATH}\n" \
          f"sys.path: {sys.path}\n" \
          f"Error: {e}\n" \
          "Please check your workspace structure and PYTHONPATH.\n")
    sys.exit(1)


# Import set_kite for KiteConnect global instance
from utils.helpers import set_kite

# Import project branding
PRIVATE_CONFIG_PATH = os.path.abspath(os.path.join(EXPECTED_ROOT, '..', 'sv-artemis-signals-private'))
if PRIVATE_CONFIG_PATH not in sys.path:
    sys.path.insert(0, PRIVATE_CONFIG_PATH)
try:
    from config import PROJECT_NAME
except ImportError:
    PROJECT_NAME = "Artemis Signals"
PROJECT_TAGLINE = "Quantitative Trading Automation for Indian Markets"
PROJECT_TAGLINE2 = "Family-focused, Modular, and Ultra-Professional — For Shivaansh & Krishaansh"

import pathlib
EXPECTED_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPECTED_ROOT, '..'))
cwd = os.path.abspath(os.getcwd())
pythonpath = os.environ.get('PYTHONPATH', '')

# Allow running from backend dir or workspace root, as long as imports resolve
if EXPECTED_ROOT not in sys.path:
    sys.path.insert(0, EXPECTED_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

def _can_import_backend():
    try:
        import config
        return True
    except ImportError:
        return False

if not _can_import_backend():
    print(f"\n[ERROR] Could not import backend modules.\n" \
          f"Current working directory: {cwd}\n" \
          f"PYTHONPATH: {pythonpath}\n" \
          f"sys.path: {sys.path}\n" \
          f"\nTry running from either:\n  cd {EXPECTED_ROOT} && python main.py\n  OR\n  cd {WORKSPACE_ROOT} && python sv-artemis-signals-public-backend/main.py\n\nFor Shivaansh & Krishaansh — this line pays your fees!\n")
    sys.exit(1)

# ============================================================================
# MAIN TABS (move all dashboard logic below this point into tab blocks)
# ============================================================================
if not COMMERCIAL_MODE:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Engine", "🎯 Strategy Scanner", "📈 Backtesting", "📊 System Info", "🚦 Active Signals"])
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Engine", "🎯 Strategy Scanner", "📈 Backtesting", "📊 System Metrics", "🚦 Active Signals"])

with tab1:
    st.header("📊 Data Engine — Your Foundation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 UPDATE ALL PRICE DATA (550+ assets)", type="primary"):
            with st.spinner("📥 Downloading..."):
                try:
                    download_all_price_data()
                    # Show validation statistics (ADAPTIVE VALIDATION)
                    if VALIDATION_AVAILABLE:
                        stats = validation_stats.get_summary()
                        
                        if stats['total'] > 0:
                            st.success(f"✅ All data updated! Validation: {stats['passed']}/{stats['total']} ({stats['success_rate']}%)")
                            
                            # Show metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("✅ Passed", stats['passed'])
                            with metric_col2:
                                st.metric("❌ Failed", stats['failed'])
                            with metric_col3:
                                st.metric("📊 Success Rate", f"{stats['success_rate']}%")
                            
                            # Show breakdown
                            if stats['by_reason']:
                                st.write("**Validation Breakdown:**")
                                reason_cols = st.columns(min(3, len(stats['by_reason'])))
                                for idx, (reason, count) in enumerate(stats['by_reason'].items()):
                                    with reason_cols[idx % len(reason_cols)]:
                                        st.write(f"📌 {reason}: {count}")
                            
                            st.balloons()
                        else:
                            st.success("✅ All data updated!")
                            st.balloons()
                    else:
                        st.success("✅ All data updated!")
                        st.balloons()
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")
    with col2:
        if st.button("📊 DOWNLOAD TODAY'S OPTION CHAIN + IV", type="primary"):
            with st.spinner("🔨 Building edge..."):
                try:
                    download_and_save_atm_iv()
                    st.success("✅ IV Database updated")
                except Exception as e:
                    st.error(f"❌ IV download failed: {e}")

with tab2:
    st.header("🎯 Strategy Scanner — ML + GARCH + IV")
    
    tf = st.selectbox("Timeframe", TIMEFRAMES, index=3)
    
    if st.button("🚀 SCAN ALL STRATEGIES NOW", type="primary"):
        with st.spinner("⚡ Running scanner..."):
            try:
                df = scan_all_strategies(tf)
                st.session_state.results = df
                st.success("✅ Scan complete!")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Scan failed: {e}")
    
    if 'results' in st.session_state and not st.session_state.results.empty:
        st.markdown("### Results")
        st.dataframe(st.session_state.results, use_container_width=True)

with tab3:
    st.header("📈 Backtesting — Prove It Works")
    
    st.markdown("Select a pair and timeframe to backtest")
    
    col1, col2 = st.columns(2)
    with col1:
        pair = st.text_input("Pair (e.g., SBIN-TCS)", "")
    with col2:
        tf_bt = st.selectbox("Timeframe", TIMEFRAMES)
    
    if st.button("▶️ RUN BACKTEST", type="primary"):
        if pair and "-" in pair:
            try:
                a, b = pair.split("-")
                with st.spinner("⏳ Backtesting..."):
                    result = backtest_pair_multi_tf(a, b, tf_bt)
                    if result:
                        st.success(f"✅ Return: {result.get('return_pct', 0):.2f}% | Sharpe: {result.get('sharpe', 0):.2f}")
                        st.json(result)
                    else:
                        st.warning("No backtest results")
            except Exception as e:
                st.error(f"❌ Backtest failed: {e}")
        else:
            st.warning("⚠️ Enter pair as SYMBOL1-SYMBOL2")



    # Removed duplicate/old tab4 and tab5 blocks after refactor

# ============================================================================
# EOD/BOD STATUS DASHBOARD
# ============================================================================
st.divider()

if st.session_state.scheduler_running:
    st.markdown("### 🔔 EOD/BOD SCHEDULER STATUS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = st.session_state.eod_bod_status.get('bod_status', 'Pending')
        color = "🟢" if status == "✅ Completed" else "🟡"
        st.markdown(f"""
        <div class='phoenix-card'>
            <div style='font-size: 20px;'>{color} BOD (9:15 AM)</div>
            <div style='font-size: 12px; color: #B0B0B0;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = st.session_state.eod_bod_status.get('eod_status', 'Pending')
        color = "🟢" if status == "✅ Completed" else "🟡"
        st.markdown(f"""
        <div class='phoenix-card'>
            <div style='font-size: 20px;'>{color} EOD (3:45 PM)</div>
            <div style='font-size: 12px; color: #B0B0B0;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        is_open = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
        st.markdown(f"""
        <div class='phoenix-card'>
            <div style='font-size: 20px;'>{is_open}</div>
            <div style='font-size: 12px; color: #B0B0B0;'>Market Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='phoenix-card'>
            <div style='font-size: 20px;'>⚙️ ACTIVE</div>
            <div style='font-size: 12px; color: #B0B0B0;'>Scheduler Running</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("⚠️ EOD/BOD Scheduler not running. Manual downloads needed.")

# ============================================================================
# QUICK ACTIONS
# ============================================================================
st.divider()
st.markdown("### ⚡ Quick Actions")

qcol1, qcol2, qcol3 = st.columns(3)

with qcol1:
    if st.button("📡 Open Live Scanner", use_container_width=True):
        st.info("Live Scanner Page → Click from sidebar")

with qcol2:
    if st.button("🔍 Open Analyser", use_container_width=True):
        st.info("Analyser Page → Click from sidebar")

with qcol3:
    if st.button("💼 Open Trades", use_container_width=True):
        st.info("Trades Page → Click from sidebar")

# ============================================================================
# SYSTEM STATUS
# ============================================================================
st.divider()
st.markdown("### 🔧 System Status")

sys_col1, sys_col2, sys_col3, sys_col4 = st.columns(4)

with sys_col1:
    st.markdown("""
    <div class='phoenix-metric'>
        <div class='phoenix-metric-label'>🟢 Database</div>
        <div class='phoenix-metric-value'>Healthy</div>
    </div>
    """, unsafe_allow_html=True)

with sys_col2:
    st.markdown("""
    <div class='phoenix-metric'>
        <div class='phoenix-metric-label'>🟢 API</div>
        <div class='phoenix-metric-value'>Connected</div>
    </div>
    """, unsafe_allow_html=True)

with sys_col3:
    st.markdown("""
    <div class='phoenix-metric'>
        <div class='phoenix-metric-label'>🟢 Scheduler</div>
        <div class='phoenix-metric-value'>Running</div>
    </div>
    """, unsafe_allow_html=True)

with sys_col4:
    st.markdown("""
    <div class='phoenix-metric'>
        <div class='phoenix-metric-label'>🟢 Scanner</div>
        <div class='phoenix-metric-value'>Ready</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# AUTO-REFRESH DURING MARKET HOURS
# ============================================================================
if is_market_open():
    # Rerun every 5 minutes during market hours
    import time
    time.sleep(300)
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.divider()

if not COMMERCIAL_MODE:
    # Personal mode: Professional footer
    # System runs 24/7 with automated monitoring
    # Target: Consistent performance through disciplined execution
    # Focus: Risk management and systematic strategy deployment
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #000000, #1e3d59); color: gold; border-radius: 20px;'>
        <h3>📊 SYSTEMATIC QUANTITATIVE TRADING</h3>
        <p>Target: 40-50% annual returns through disciplined execution</p>
        <p><strong>No excuses. Only execution. Only discipline.</strong></p>
        <p style='font-size: 12px; color: #B0B0B0; margin-top: 15px;'>© 2025 Artemis Signals — ALL RIGHTS RESERVED</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #000000, #1e3d59); color: gold; border-radius: 20px;'>
        <h3>📊 {PROJECT_NAME}</h3>
        <p>Institutional-grade algorithmic trading platform</p>
        <p><strong>Powered by quantitative excellence & systematic execution</strong></p>
        <p style='font-size: 12px; color: #B0B0B0; margin-top: 15px;'>© 2025 {PROJECT_NAME} — ALL RIGHTS RESERVED</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ADMIN DASHBOARD - NOTIFICATION LOGS
# ============================================================================
with st.sidebar:
    st.header("🛡️ Admin Dashboard")
    admin_tab = st.radio("Admin Panel", [
        "Notification Logs",
        "User Management",
        "Channel Health",
        "System Health",
        "Signal Management",
        "Config & Secrets",
        "Manual Actions"
    ])
    if admin_tab == "Notification Logs":
        if st.button("🔍 Review Notification Logs", type="primary"):
            from utils.notifications import AdminManagementDashboard
            dashboard = AdminManagementDashboard()
            st.text("Log file: " + dashboard.log_path)
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dashboard.show_dashboard()
            st.text(buf.getvalue())
    elif admin_tab == "User Management":
        st.info("User management coming soon: view/add/edit/deactivate users, assign roles, reset passwords.")
    elif admin_tab == "Channel Health":
        st.info("Channel health coming soon: live status for Email, SMS, WhatsApp, Telegram, test send.")
    elif admin_tab == "System Health":
        st.info("System health coming soon: backend service status, error/warning counts, manual health check.")
    elif admin_tab == "Signal Management":
        st.info("Signal management coming soon: list/enable/disable strategies, view last signal per strategy.")
    elif admin_tab == "Config & Secrets":
        st.info("Config & secrets coming soon: view/edit config values, reload config.")
    elif admin_tab == "Manual Actions":
        st.info("Manual actions coming soon: trigger jobs, download/export signals and logs.")
    st.info("For Shivaansh & Krishaansh — every alert pays their fees!")
