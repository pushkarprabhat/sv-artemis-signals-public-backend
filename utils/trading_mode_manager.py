"""
Trading Mode Manager - Handle manual vs auto-trading modes
Syncs signals across pages and handles auto-entry logic
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from utils.logger import logger


def init_trading_mode_session(broker_mode='paper'):
    """Initialize trading mode session state
    
    Args:
        broker_mode: 'paper' for paper trading, 'live' for live trading
    """
    # For paper trading: FORCE AUTO MODE ONLY (for testing strategies)
    # For live trading: DEFAULT to manual, allow switching to auto
    
    if broker_mode.lower() == 'paper':
        # Paper trading: AUTO ONLY
        st.session_state.trading_mode = "auto"
        st.session_state.available_modes = ['auto']  # No choice in paper trading
        st.session_state.auto_entry_threshold = 0.75  # 75% confidence
        st.session_state.trading_mode_locked = True  # Prevent user from switching
    else:
        # Live trading: DEFAULT MANUAL, allow switching to auto
        st.session_state.trading_mode = "manual"
        st.session_state.available_modes = ['manual', 'auto']  # User can switch
        st.session_state.auto_entry_threshold = 0.85  # Higher threshold for live
        st.session_state.trading_mode_locked = False  # Allow user to switch
    
    if "synced_signals" not in st.session_state:
        st.session_state.synced_signals = []
    
    if "auto_entered_trades" not in st.session_state:
        st.session_state.auto_entered_trades = []


def sync_signals_to_session(scan_results):
    """
    Sync signals from scanner to session state for main page display
    Also handles auto-entry if in auto mode
    """
    try:
        if scan_results is None or scan_results.empty:
            st.session_state.synced_signals = []
            return
        
        # Convert results to list of dicts for storage
        signals = []
        for idx, row in scan_results.iterrows():
            signal = {
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                'pair': row.get('pair', row.get('symbol1', 'N/A')),
                'strategy': row.get('strategy', 'Unknown'),
                'ml_score': float(row.get('ml_score', 0)),
                'recommend': row.get('recommend', 'HOLD'),
                'p_value': row.get('p_value', None),
                'correlation': row.get('correlation', None),
                'symbol1': row.get('symbol1', None),
                'symbol2': row.get('symbol2', None),
                'qty1': row.get('qty1', None),
                'qty2': row.get('qty2', None),
                'price1': row.get('price1', None),
                'price2': row.get('price2', None),
                'capital_required': row.get('capital_required', 0),
                'max_profit': row.get('max_profit', None),
                'max_loss': row.get('max_loss', None),
            }
            signals.append(signal)
        
        # Store in session
        st.session_state.synced_signals = signals
        
        # Handle auto-trading if enabled
        if st.session_state.trading_mode == "auto":
            handle_auto_entry(signals)
        
    except Exception as e:
        logger.error(f"Error syncing signals: {e}")


def handle_auto_entry(signals):
    """
    Automatically enter trades based on confidence threshold
    Only for paper trading mode
    """
    try:
        from core.papertrader import PaperTrader
        
        pt = PaperTrader()
        threshold = st.session_state.auto_entry_threshold
        
        for signal in signals:
            # Check if already entered
            signal_id = f"{signal['pair']}_{signal['strategy']}"
            if signal_id in st.session_state.get('auto_entered_trades', []):
                continue
            
            # Check confidence
            confidence = signal['ml_score'] / 100 if signal['ml_score'] > 1 else signal['ml_score']
            if confidence >= threshold:
                try:
                    # Auto-enter the trade
                    if signal['strategy'] == 'Pair Trading' and signal['symbol1'] and signal['symbol2']:
                        # Pair trade
                        trade_id = pt.enter_pair_trade(
                            symbol1=signal['symbol1'],
                            symbol2=signal['symbol2'],
                            qty1=signal.get('qty1', 1),
                            qty2=signal.get('qty2', 1),
                            price1=signal.get('price1', 0),
                            price2=signal.get('price2', 0),
                            stop_loss_pct=2.0
                        )
                        
                        if trade_id:
                            # Track auto-entered trade
                            st.session_state.auto_entered_trades.append(signal_id)
                            logger.info(f"[AUTO-ENTRY] {signal['pair']} entered (confidence: {confidence:.2%})")
                    
                    else:
                        # Single leg trade
                        trade_id = pt.enter_trade(
                            symbol=signal['pair'],
                            qty=signal.get('qty1', 1),
                            entry_price=signal.get('price1', 0),
                            stop_loss_pct=2.0,
                            strategy=signal['strategy']
                        )
                        
                        if trade_id:
                            st.session_state.auto_entered_trades.append(signal_id)
                            logger.info(f"[AUTO-ENTRY] {signal['pair']} entered (confidence: {confidence:.2%})")
                
                except Exception as e:
                    logger.warning(f"Auto-entry failed for {signal['pair']}: {e}")
    
    except Exception as e:
        logger.error(f"Error in auto-entry handler: {e}")


def render_trading_mode_selector():
    """Render trading mode selector in sidebar with mode-based constraints"""
    try:
        st.markdown("### 🤖 TRADING MODE")
        
        with st.container(border=True):
            # Check if mode is locked (paper trading = auto only)
            is_locked = st.session_state.get('trading_mode_locked', False)
            available_modes = st.session_state.get('available_modes', ['manual', 'auto'])
            
            if is_locked:
                # Paper trading: AUTO MODE ONLY
                st.success("🤖 AUTO MODE (Paper Trading)", icon="✅")
                st.info(
                    "Paper trading is in **AUTO MODE ONLY** for testing strategies. "
                    "All signals above the confidence threshold will be auto-entered.",
                    icon="ℹ️"
                )
                
                # Show threshold control
                threshold = st.slider(
                    "Entry Confidence Threshold",
                    min_value=50,
                    max_value=99,
                    value=int(st.session_state.get('auto_entry_threshold', 0.75) * 100),
                    step=5,
                    help="Auto-enter trades above this confidence %",
                    disabled=False  # User can adjust threshold
                )
                st.session_state.auto_entry_threshold = threshold / 100
                
                st.caption(f"Will auto-enter trades with {threshold}%+ confidence")
            else:
                # Live trading: Manual + Auto modes available
                mode_options = []
                mode_captions = []
                
                if 'manual' in available_modes:
                    mode_options.append("Manual Entry")
                    mode_captions.append("Review each signal, confirm manually")
                
                if 'auto' in available_modes:
                    mode_options.append("Auto Entry")
                    mode_captions.append("Auto-enter trades above confidence threshold")
                
                if len(mode_options) > 1:
                    mode = st.radio(
                        "Select Mode:",
                        mode_options,
                        captions=mode_captions,
                        key="trading_mode_radio",
                        horizontal=False
                    )
                    
                    # Map to session
                    st.session_state.trading_mode = "auto" if mode == "Auto Entry" else "manual"
                else:
                    # Only one mode available
                    st.session_state.trading_mode = "manual"
                    st.info(f"Mode: {mode_options[0]} (only option for live trading)", icon="ℹ️")
                
                # Show current mode details
                if st.session_state.trading_mode == "auto":
                    st.warning("⚠️ AUTO MODE (Live Trading)", icon="⚠️")
                    st.markdown(
                        "**Caution**: Auto-mode will enter real trades with real money. "
                        "Review capital settings and risk parameters before enabling.",
                        unsafe_allow_html=False
                    )
                    
                    # Higher threshold for live trading
                    threshold = st.slider(
                        "Entry Confidence Threshold",
                        min_value=70,
                        max_value=99,
                        value=int(st.session_state.get('auto_entry_threshold', 0.85) * 100),
                        step=5,
                        help="Auto-enter trades above this confidence % (higher = safer)"
                    )
                    st.session_state.auto_entry_threshold = threshold / 100
                    st.caption(f"Will auto-enter trades with {threshold}%+ confidence")
                else:
                    st.info("👤 MANUAL MODE (Live Trading)", icon="ℹ️")
                    st.caption("Review each signal and click to enter trades manually. Recommended for live trading.")
    
    except Exception as e:
        logger.error(f"Error rendering trading mode: {e}")


def get_synced_signals():
    """Get currently synced signals for display"""
    return st.session_state.get('synced_signals', [])


def get_auto_entry_status():
    """Get status of auto-entry system"""
    return {
        'mode': st.session_state.get('trading_mode', 'manual'),
        'threshold': st.session_state.get('auto_entry_threshold', 0.75),
        'auto_entered_count': len(st.session_state.get('auto_entered_trades', [])),
        'total_signals': len(st.session_state.get('synced_signals', [])),
    }
