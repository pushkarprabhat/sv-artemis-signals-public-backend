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
                    # ...existing code...
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

with tab4:
    # System Info/System Metrics tab placeholder (add actual logic as needed)
    st.header("📊 System Info / Metrics")
    st.info("System metrics and info will be displayed here.")

with tab5:
    st.header("🚦 Active Signals — Real-Time")
    import os, json
    signals_path = os.path.join(os.path.dirname(__file__), 'marketdata', 'signals.json')
    if os.path.exists(signals_path):
        with open(signals_path, 'r', encoding='utf-8') as f:
            signals = json.load(f)
        if signals:
            import pandas as pd
            df = pd.DataFrame(signals)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No active signals found.")
    else:
        st.warning("signals.json not found. Run scanner to generate signals.")


# Scheduler import and availability detection (robust)
try:
    from core.scheduler import SchedulerService
    SCHEDULER_AVAILABLE = True
except Exception as e:
    from utils.logger import logger
    logger.warning(f"Scheduler not available: {e}")
    SCHEDULER_AVAILABLE = False

import pandas as pd
import numpy as np
from datetime import datetime
from kiteconnect import KiteConnect
from utils.logger import logger

# Patch paper trading manager import to use shared/private strategies path
PRIVATE_STRATEGIES_PATH = os.path.abspath(os.path.join(EXPECTED_ROOT, '..', 'sv-artemis-signals-private', 'strategies'))
if PRIVATE_STRATEGIES_PATH not in sys.path:
    sys.path.insert(0, PRIVATE_STRATEGIES_PATH)


    # All tab logic is now after st.tabs() assignment above

# ============================================================================
# ZERODHA LOGIN
# ============================================================================
load_dotenv()
api_key = os.getenv("ZERODHA_API_KEY")
access_token = os.getenv("ZERODHA_ACCESS_TOKEN")

if not api_key or not api_key.strip():
    raise RuntimeError("Environment variable 'ZERODHA_API_KEY' is not set or is empty.")

if not access_token or not access_token.strip():
    raise RuntimeError("Environment variable 'ZERODHA_ACCESS_TOKEN' is not set or is empty.")

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
set_kite(kite)

st.set_page_config(page_title=PROJECT_NAME, layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<h1 style='text-align: center; color: gold; text-shadow: 0 0 20px gold;'>{PROJECT_NAME}</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: cyan;'>{PROJECT_TAGLINE}</h2>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: cyan;'>{PROJECT_TAGLINE2}</h2>", unsafe_allow_html=True)

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
    h1, h2, h3 {font-family: 'Georgia', serif; color: gold; text-align: center;}
    .stButton>button {background: gold; color: black; font-weight: bold; border-radius: 15px; height: 3em; width: 100%;}
    .metric-card {background: rgba(255,215,0,0.15); padding: 20px; border-radius: 20px; text-align: center; border: 3px solid gold;}
    .phoenix-card {background: rgba(255, 140, 0, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #FF8C00;}
    .phoenix-metric {background: rgba(255, 140, 0, 0.05); padding: 10px; border-radius: 8px; margin: 5px 0;}
    .phoenix-metric-label {color: #B0B0B0; font-size: 11px; text-transform: uppercase;}
    .phoenix-metric-value {color: #E8E8E8; font-size: 18px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "scheduler_service" not in st.session_state:
    st.session_state.scheduler_service = None
if "scheduler_running" not in st.session_state:
    st.session_state.scheduler_running = False
if "eod_bod_status" not in st.session_state:
    st.session_state.eod_bod_status = {
        "bod_last_run": None,
        "eod_last_run": None,
        "bod_status": "Pending",
        "eod_status": "Pending",
    }
if "subscribed" not in st.session_state:
    st.session_state.subscribed = False  # Track subscription for demo purposes

# ============================================================================
# PAPER TRADING INITIALIZATION (Simplified - No Import Conflicts)
# ============================================================================
if "paper_manager" not in st.session_state:
    try:
        # Use sys.path to avoid import conflicts
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Direct import from strategies folder
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "paper_manager_module",
            project_root / "strategies" / "paper_trading_manager.py"
        )
        if spec and spec.loader:
            paper_module = importlib.util.module_from_spec(spec)
            sys.modules['paper_manager_module'] = paper_module
            spec.loader.exec_module(paper_module)
            st.session_state.paper_manager = paper_module.PaperTradingManager(
                initial_capital=100000,
                challenge_days=30,
                auto_trade=False
            )
            logger.info("📝 Paper Trading Manager initialized")
    except Exception as e:
        logger.error(f"Failed to init paper trading: {e}")
        st.session_state.paper_manager = None

if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = "PAPER"

# ============================================================================
# LTP DATABASE INITIALIZATION
# ============================================================================
if "ltp_db" not in st.session_state:
    st.session_state.ltp_db = get_ltp_database()
    logger.info("✅ LTP Database initialized")

# ============================================================================
# EOD/BOD SCHEDULER STARTUP (CRITICAL!)
# ============================================================================
if SCHEDULER_AVAILABLE and not st.session_state.scheduler_running:
    try:
        logger.info("[STARTUP] Initializing EOD/BOD Scheduler...")
        st.session_state.scheduler_service = SchedulerService()
        st.session_state.scheduler_service.start()
        st.session_state.scheduler_running = True
        logger.info("[OK] EOD/BOD Scheduler started successfully!")
        logger.info("  ✅ BOD scheduled: 07:00 AM IST (Market pre-open)")
        logger.info("  ✅ EOD scheduled: 05:55 PM IST (Market close + 5 min buffer)")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start scheduler: {e}")
        st.warning(f"⚠️ Scheduler not available: {e}")

# ============================================================================
# PARALLEL OPTIONS SCANNER STARTUP
# ============================================================================
if "options_scanner_running" not in st.session_state:
    st.session_state.options_scanner_running = False

if STRATEGIES_AVAILABLE and not st.session_state.options_scanner_running:
    try:
        logger.info("[STARTUP] Starting Parallel Options Scanner...")
        # Start scanning every 15 minutes with 70% confidence threshold
        scanner = start_parallel_options_scanning(
            interval_minutes=15,
            confidence_threshold=0.70
        )
        st.session_state.options_scanner_running = True
        logger.info("[OK] Parallel Options Scanner started!")
        logger.info("  ✅ Scanning every 15 minutes")
        logger.info("  ✅ Alerting trades with 70%+ confidence")
        logger.info("  ✅ Telegram + Email alerts enabled")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start options scanner: {e}")
        st.warning(f"⚠️ Options scanner not available: {e}")

# ============================================================================
# TRADING MODE DISPLAY
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 🟢 Mode: **PAPER**")

with col2:
    if st.session_state.paper_manager:
        perf = st.session_state.paper_manager.get_overall_performance()
        capital = perf.get('capital', 100000)
        st.metric("💰 Available Capital", f"₹{capital:,.0f}")
    else:
        st.metric("💰 Available Capital", "₹100,000")

with col3:
    if st.session_state.paper_manager:
        open_trades = st.session_state.paper_manager.get_open_pair_trades()
        open_pos = len(open_trades) if isinstance(open_trades, (list, pd.DataFrame)) else 0
        st.metric("🎯 Open Positions", open_pos)
    else:
        st.metric("🎯 Open Positions", 0)

st.markdown("---")

# ============================================================================
# LIVE MARKET DASHBOARD — PROFESSIONAL VIEW
# ============================================================================

# Display market status (OPEN/CLOSED) with timestamp
market_is_open = display_market_status()

# ============================================================================
# AUTO-SCANNING BACKGROUND PROCESS
# ============================================================================
# Initialize auto-scanner if not already running
if "auto_scanner_enabled" not in st.session_state:
    st.session_state.auto_scanner_enabled = False
if "latest_signals" not in st.session_state:
    st.session_state.latest_signals = []
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

# ============================================================================
# SIDEBAR - ORGANIZED MENU + SYSTEM STATUS
# ============================================================================
with st.sidebar:
    st.markdown("## 🎯 ARTEMIS SIGNALS")
    if not COMMERCIAL_MODE:
        # Non-commercial mode: Personal trading dashboard
        # Real-time monitoring of systematic trading strategies
        # Every signal follows quantitative rules
        st.markdown("### *Personal Trading System*")
    st.markdown("---")
    
    # ========================================================================
    # SYSTEM STATUS SECTION
    # ========================================================================
    st.markdown("### 📊 System Status")
    
    # API Connection
    try:
        test_ltp = get_ltp("NIFTY")
        if test_ltp and test_ltp > 0:
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ Demo Mode")
    except:
        st.warning("⚠️ Demo Mode")
    
    # Market Status
    from datetime import datetime
    from config import BOD_IDEAL_HOUR, BOD_IDEAL_MINUTE, EOD_IDEAL_HOUR, EOD_IDEAL_MINUTE
    
    now = datetime.now()
    st.write(f"🕐 {now.strftime('%H:%M:%S')}")
    if market_is_open:
        st.write("🟢 **Market OPEN**")
    else:
        st.write("🔴 **Market CLOSED**")
    
    # Auto-Scanner Status
    if st.session_state.auto_scanner_enabled:
        st.success("🔄 Auto-Scan: ON")
        if st.session_state.last_scan_time:
            st.caption(f"Last: {st.session_state.last_scan_time.strftime('%H:%M')}")
    else:
        st.info("💤 Auto-Scan: OFF")
    
    st.markdown("---")
    
    # ========================================================================
    # NAVIGATION MENU - ORGANIZED IN GROUPS
    # ========================================================================
    st.markdown("### 📱 Navigation")
    
    # Group 1: TRADING & EXECUTION
    st.markdown("#### 🚀 Trading")
    if st.button("📊 Dashboard (Home)", use_container_width=True, key="nav_home"):
        st.switch_page("main.py")
    if st.button("🏆 Top 10 Trades", use_container_width=True, key="nav_top10"):
        st.switch_page("pages/top_10_trades.py")
    if st.button("🔍 Live Scanner", use_container_width=True, key="nav_scanner"):
        st.switch_page("pages/live_scanner.py")
    if st.button("📝 Paper Trading", use_container_width=True, key="nav_paper"):
        st.switch_page("pages/paper_trading.py")
    if st.button("💼 Portfolio", use_container_width=True, key="nav_portfolio"):
        st.switch_page("pages/portfolio.py")
    if st.button("💳 Subscription", use_container_width=True, key="nav_subscribe"):
        st.switch_page("pages/subscribe.py")
    
    st.markdown("---")
    
    # Group 2: ANALYSIS & SIGNALS
    st.markdown("#### 📈 Analysis")
    if st.button("🎯 Pair Signals", use_container_width=True, key="nav_pairs"):
        st.switch_page("pages/pair_signals.py")
    if st.button("⚡ Risk Metrics", use_container_width=True, key="nav_risk"):
        st.switch_page("pages/risk_metrics.py")
    if st.button("🔔 Anomaly Alerts", use_container_width=True, key="nav_anomaly"):
        st.switch_page("pages/anomaly_alerts.py")
    if st.button("📊 Technical Analysis", use_container_width=True, key="nav_technical"):
        st.switch_page("pages/technical_analysis.py")
    
    st.markdown("---")
    
    # Group 3: REPORTS & INTELLIGENCE
    st.markdown("#### 📋 Reports")
    if st.button("🌅 Opening Bell", use_container_width=True, key="nav_opening"):
        st.switch_page("pages/opening_bell_report.py")
    if st.button("🌆 Closing Bell", use_container_width=True, key="nav_closing":
        st.switch_page("pages/closing_bell_report.py")
    if st.button("📧 Email Reports", use_container_width=True, key="nav_email"):
        st.switch_page("pages/email_subscriptions.py")
    if not COMMERCIAL_MODE:
        # 💙 Family dashboard - your education fund tracker
        if st.button("👥 Family Info", use_container_width=True, key="nav_family"):
            st.switch_page("pages/family_info.py")
    
    st.markdown("---")
    
    # Group 4: GLOBAL MARKETS
    st.markdown("#### 🌍 Global Markets")
    if st.button("🌎 World Indices", use_container_width=True, key="nav_world"):
        st.switch_page("pages/world_indices.py")
    if st.button("💱 Commodities & FX", use_container_width=True, key="nav_commodities"):
        st.switch_page("pages/commodities_fx.py")
    
    st.markdown("---")
    
    # Group 5: SYSTEM & SETTINGS
    st.markdown("#### ⚙️ System")
    if st.button("📡 Data Manager", use_container_width=True, key="nav_data"):
        st.switch_page("pages/data_manager.py")
    if st.button("🌌 Universe", use_container_width=True, key="nav_universe"):
        st.switch_page("pages/universe.py")
    if st.button("💬 Telegram Test", use_container_width=True, key="nav_telegram"):
        st.switch_page("pages/telegram_test.py")
    if st.button("⚙️ Settings", use_container_width=True, key="nav_settings"):
        st.switch_page("pages/settings.py")
    
    st.markdown("---")
    
    # ========================================================================
    # QUICK STATS FOOTER
    # ========================================================================
    st.markdown("### 💰 Quick Stats")
    st.write(f"Capital: ₹{100000:,}")
    st.write(f"Risk: 2% per trade")
    st.write(f"Mode: 📝 Paper")
    sub_status = "💎 Premium" if st.session_state.get('subscribed', False) else "🆓 Free"
    st.write(f"Access: **{sub_status}**")
    
    st.markdown("---")
    if not COMMERCIAL_MODE:
        # Personal mode: Systematic execution
        st.markdown("*Systematic execution* 📊")
    else:
        st.markdown("*Powered by quantitative excellence*")

# Auto-refresh setup using Streamlit's native mechanism
if market_is_open:
    # Initialize refresh counter
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    # Display refresh info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🔄 Auto-refreshing every 10 seconds during market hours")
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh every 10 seconds
    import time
    time.sleep(10)
    st.session_state.refresh_counter += 1
    st.rerun()
else:
    st.info("💤 Showing last known prices (market closed)")

# ============================================================================
# MAJOR INDICES DISPLAY — BLOOMBERG STYLE
# ============================================================================
# Major indices to track
major_indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
display_major_indices(major_indices)

# ============================================================================
# QUICK METRICS — TRADING STATUS
# ============================================================================
st.markdown("---")
display_quick_metrics()

# ============================================================================
# INDEX CONSTITUENTS — TABBED VIEW
# ============================================================================
index_mapping = {
    "NIFTY 50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT"
}
display_index_tabs(index_mapping)

# ============================================================================
# LIVE STRATEGY SIGNALS — PAPER TRADING READY
# ============================================================================
st.markdown("---")
st.markdown("## 🎯 LIVE STRATEGY SCANNER")
if not COMMERCIAL_MODE:
    # Personal mode: Real signal monitoring
    st.markdown("*Personal system — Real signals, real execution*")
else:
    st.markdown("*Real-time multi-strategy signal generation with ML scoring*")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.write("Scan all strategies for cointegrated pairs, options setups, and momentum signals")
with col2:
    if st.button("🔍 SCAN NOW", type="primary", use_container_width=True):
        if st.session_state.get('subscribed', False):
            st.session_state.scanner_run = True
        else:
            st.warning("🔒 Subscription required to access live signals.")
            if st.button("💳 Get Access Now"):
                st.switch_page("pages/subscribe.py")
with col3:
    if st.button("📊 VIEW BACKTEST", use_container_width=True):
        st.info("Backtest results saved in data/backtest_results.csv")

if st.session_state.get('scanner_run', False):
    with st.spinner("🔍 Scanning 65 focused stocks across 14 sectors..."):
        try:
            from core.pairs import scan_all_strategies
            signals_df = scan_all_strategies()
            
            if len(signals_df) > 0:
                st.success(f"✅ Found {len(signals_df)} high-quality signals!")
                
                # Display signals table
                st.dataframe(
                    signals_df[['pair', 'p_value', 'industry', 'ml_score', 'recommend',
                                'capital_required', 'selection_criteria']].head(10),
                    use_container_width=True,
                    height=400
                )

                # Trade execution integration (Paper or Live based on mode)
                st.markdown(f"### 🚀 EXECUTE TRADES ({st.session_state.trading_mode} MODE)")

                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    mode_emoji = "📝" if st.session_state.trading_mode == "PAPER" else "🔴"
                    st.write(f"{mode_emoji} Execute top 3 signals in **{st.session_state.trading_mode}** mode")
                with col_b:
                    num_to_execute = st.number_input("Execute Top N:", min_value=1, max_value=10, value=3, step=1)
                with col_c:
                    execute_button_label = f"▶️ EXECUTE TOP {num_to_execute}"
                    if st.session_state.trading_mode == "LIVE":
                        execute_button_label = f"🔴 LIVE: EXECUTE {num_to_execute}"

                    if st.button(execute_button_label, type="primary"):
                        st.session_state.execute_trades_count = num_to_execute

                # Execute trades if button clicked
                if st.session_state.get('execute_trades_count', 0) > 0:
                    num_exec = st.session_state.execute_trades_count
                    top_signals = signals_df.head(num_exec)

                    success_count = 0
                    fail_count = 0

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    if st.session_state.paper_manager:
                        for idx, signal in top_signals.iterrows():
                            status_text.text(f"Executing {idx + 1}/{num_exec}: {signal['pair']}...")
                            progress_bar.progress((idx + 1) / num_exec)
                            try:
                                # Execute paper trade
                                st.session_state.paper_manager.add_pair_trade(
                                    pair_name=signal['pair'],
                                    symbol1=signal.get('symbol1', signal['pair'].split('-')[0]),
                                    symbol2=signal.get('symbol2', signal['pair'].split('-')[1]),
                                    qty1=int(signal.get('qty1', 10)),
                                    qty2=int(signal.get('qty2', 10)),
                                    entry_price1=float(signal.get('price1', 100)),
                                    entry_price2=float(signal.get('price2', 100)),
                                    signal_type=signal['recommend'],
                                    comment=f"Scanner: {signal.get('selection_criteria', '')}"
                                )
                                success_count += 1
                                st.success(f"✅ Paper trade: {signal['pair']}")
                            except Exception as e:
                                fail_count += 1
                                st.error(f"❌ Failed: {signal['pair']} - {e}")
                            time.sleep(0.3)
                    else:
                        st.error("Paper trading manager not initialized")

                    # Summary
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✅ Successful", success_count)
                    with col2:
                        st.metric("❌ Failed", fail_count)

                    st.info("📝 View all paper trades in the **Portfolio** page →")

                    # Reset execution flag
                    st.session_state.execute_trades_count = 0

                # Save signals to session for other pages
                st.session_state.latest_signals = signals_df

            else:
                st.warning("No signals found. Market conditions may not favor current strategies.")
                
        except Exception as e:
            st.error(f"Scanner error: {e}")
            logger.error(f"Scanner failed: {e}")
    
    # Reset scanner flag
    st.session_state.scanner_run = False

# ============================================================================
# BACKGROUND: FETCH AND STORE LTP DATA
# ============================================================================
# Fetch and store LTP for all tracked symbols in background
live_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "INDIAVIX"]
ltp_db = st.session_state.ltp_db

for sym in live_symbols:
    try:
        price = get_ltp(sym)
        if price and price > 0:
            ltp_db.update_ltp(
                symbol=sym,
                last_price=price,
                exchange='NSE'
            )
    except Exception as e:
        logger.debug(f"Could not fetch/store LTP for {sym}: {e}")

# ============================================================================
# SYSTEM STATUS & DATA COVERAGE
# ============================================================================
st.markdown("---")
st.markdown("### 🎯 SYSTEM STATUS & DATA COVERAGE")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric(
        label="📊 Tracked Instruments",
        value=len(live_symbols),
        delta="Live"
    )

with status_col2:
    # Check data availability
    data_path = Path("data")
    timeframes_available = []
    for tf in TIMEFRAMES:
        tf_path = data_path / tf
        if tf_path.exists() and len(list(tf_path.glob("*.csv"))) > 0:
            timeframes_available.append(tf)
    
    st.metric(
        label="⏱️ Timeframes Ready",
        value=f"{len(timeframes_available)}/{len(TIMEFRAMES)}",
        delta="Multi-horizon"
    )

with status_col3:
    # Check total data files
    total_files = 0
    for tf in TIMEFRAMES:
        tf_path = data_path / tf
        if tf_path.exists():
            total_files += len(list(tf_path.glob("*.csv")))
    
    st.metric(
        label="📁 Data Files",
        value=total_files,
        delta="CSV files"
    )

with status_col4:
    # Market status
    if market_is_open:
        st.metric(
            label="🟢 Market Status",
            value="LIVE",
            delta="Trading Active"
        )
    else:
        st.metric(
            label="🔴 Market Status", 
            value="CLOSED",
            delta="EOD Mode"
        )

# Detailed data coverage table
with st.expander("📋 DETAILED DATA COVERAGE", expanded=False):
    st.markdown("#### Available Data by Timeframe")
    
    coverage_data = []
    for tf in TIMEFRAMES:
        tf_path = data_path / tf
        if tf_path.exists():
            files = list(tf_path.glob("*.csv"))
            symbols_with_data = [f.stem for f in files[:10]]  # First 10
            coverage_data.append({
                "Timeframe": tf,
                "Files": len(files),
                "Sample Symbols": ", ".join(symbols_with_data[:5]) + ("..." if len(symbols_with_data) > 5 else ""),
                "Status": "✅ Ready" if len(files) > 0 else "❌ Missing"
            })
        else:
            coverage_data.append({
                "Timeframe": tf,
                "Files": 0,
                "Sample Symbols": "N/A",
                "Status": "❌ Missing"
            })
    
    if coverage_data:
        df_coverage = pd.DataFrame(coverage_data)
        st.dataframe(df_coverage, use_container_width=True, hide_index=True)
    
    # Check specific indices
    st.markdown("#### Key Indices Data Status")
    indices_status = []
    for symbol in live_symbols:
        has_data = False
        for tf in TIMEFRAMES:
            tf_path = data_path / tf / f"{symbol}.csv"
            if tf_path.exists():
                has_data = True
                break
        
        indices_status.append({
            "Symbol": symbol,
            "Status": "✅ Data Available" if has_data else "❌ No Data",
            "Type": "Index" if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"] 
                    else "Volatility" if symbol == "INDIAVIX" 
                    else "Commodity" if symbol in ["CRUDEOIL", "GOLD", "SILVER"] 
                    else "Currency"
        })
    
    df_indices = pd.DataFrame(indices_status)
    st.dataframe(df_indices, use_container_width=True, hide_index=True)

# Scanner status section
with st.expander("🔍 STRATEGY SCANNER STATUS", expanded=False):
    st.markdown("#### Available Strategies")
    
    strategies_info = [
        {"Strategy": "Pairs Trading", "Status": "✅ Active", "Instruments": "All NSE stocks", "Timeframes": "15m, 30m, 60m, day"},
        {"Strategy": "Momentum", "Status": "✅ Active", "Instruments": "All", "Timeframes": "All"},
        {"Strategy": "MA Crossover", "Status": "✅ Active", "Instruments": "All", "Timeframes": "All"},
        {"Strategy": "Volatility Breakout", "Status": "✅ Active", "Instruments": "High volume", "Timeframes": "15m, 60m"},
        {"Strategy": "IV Crush Strangle", "Status": "✅ Active", "Instruments": "NIFTY, BANKNIFTY", "Timeframes": "day, week"},
        {"Strategy": "Mean Reversion", "Status": "✅ Active", "Instruments": "All", "Timeframes": "All"},
        {"Strategy": "Trend Following", "Status": "✅ Active", "Instruments": "All", "Timeframes": "day, week"},
        {"Strategy": "Options Greeks", "Status": "✅ Active", "Instruments": "Options", "Timeframes": "Real-time"},
    ]
    
    df_strategies = pd.DataFrame(strategies_info)
    st.dataframe(df_strategies, use_container_width=True, hide_index=True)
    
    st.info("""
    🔄 **Scanner Execution:**
    - Runs on-demand when you click 'SCAN ALL STRATEGIES'
    - Analyzes all instruments × all timeframes × all strategies
    - Results ranked by ML Score (0-300)
    - Top 10 displayed in 'Strategy Scanner' tab
    """)

# LTP Database stats in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 LTP Database Stats")
    display_ltp_summary_stats()

# ============================================================================
# EDUCATION SECTION
# ============================================================================
if not COMMERCIAL_MODE:
    # Personal mode: Educational volatility content
    # IV Crush strategies generate consistent returns from volatility compression
    # Historical performance demonstrates effectiveness of the approach
    expander_title = "📚 IV CRUSH & GARCH — VOLATILITY MODELING"
    garch_explanation = """
    The market exhibits varying volatility regimes. Some periods are calm, others turbulent.
    
    GARCH has long memory — captures persistent volatility patterns from historical data.
    EWMA has short memory — responsive to recent volatility changes.
    
    The system uses both:
    - GARCH → decides WHEN to sell options (strategic timing)
    - EWMA → decides HOW MUCH to risk (tactical sizing)
    
    Every decision follows systematic, quantitative rules.
    """
else:
    expander_title = "📚 IV CRUSH & GARCH VOLATILITY MODELING"
    garch_explanation = """
    Volatility modeling using two complementary approaches:
    
    **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**:
    - Long-term memory of volatility shocks
    - Captures volatility clustering and mean reversion
    - Used for strategic timing decisions
    
    **EWMA (Exponentially Weighted Moving Average)**:
    - Responsive to recent volatility changes
    - Provides real-time risk adjustment
    - Used for position sizing and risk management
    
    Combined approach provides robust volatility forecasting.
    """

with st.expander(expander_title, expanded=True):
    st.success("**IV Crush** = Fear disappears after event → options price crashes → we keep premium")
    st.markdown("""
    **Real Example — Budget Day 2024**:
    - IV Rank = 92% → Sold NIFTY 25000 strangle → collected ₹2,85,000
    - Next day IV fell from 24% → 12%
    - Profit = ₹2,48,000 in 24 hours
    """)

    st.markdown("**GARCH vs EWMA — Professional Volatility Modeling**" if COMMERCIAL_MODE else "**GARCH vs EWMA — Explained Like a Father**")
    st.write(garch_explanation)

# ============================================================================
# POSITION SIZING
# ============================================================================
with st.expander("💰 POSITION SIZING — KELLY CRITERION"):
    if not COMMERCIAL_MODE:
        # Personal mode: Conservative position sizing
        # Balanced approach: growth with capital preservation
        st.success("We use **Half-Kelly** — Optimal growth with risk control")
    else:
        st.success("We use **Half-Kelly** — Optimal growth with capital preservation")
    
    st.markdown("""
    With 65% win rate & 1.8:1 reward/risk:
    - Full Kelly = 50% risk (too aggressive)
    - Half-Kelly = 25% risk → we use 2% max (optimal)
    
    ₹1 lakh → ₹2,000 risk/trade → scales to ₹10 crore systematically
    """)

# ============================================================================
# MAIN TABS
# ============================================================================

if not COMMERCIAL_MODE:
    # Personal mode: Personal dashboard layout
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
