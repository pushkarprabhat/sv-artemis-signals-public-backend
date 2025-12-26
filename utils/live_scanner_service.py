"""
utils/live_scanner_service.py - Background live scanning service
Runs continuously and maintains trade recommendations across the app
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from utils.logger import logger
from utils.market_hours import is_market_open

class LiveScannerService:
    """Manages background live scanning and trade recommendations"""
    
    def __init__(self):
        self.last_scan_time = None
        self.last_scan_interval = 300  # 5 minutes
        self.active_scans = {}  # Track active scans
        self.trade_recommendations = []  # Last N recommendations
        self.max_recommendations = 5
        
    def initialize_session_state(self):
        """Initialize session state for scanner"""
        if 'scanner_active' not in st.session_state:
            st.session_state.scanner_active = False
        if 'trade_recommendations' not in st.session_state:
            st.session_state.trade_recommendations = []
        if 'active_scans' not in st.session_state:
            st.session_state.active_scans = {}
        if 'last_scan_time' not in st.session_state:
            st.session_state.last_scan_time = None
    
    def should_run_scan(self):
        """Check if enough time has passed to run another scan"""
        if st.session_state.last_scan_time is None:
            return True
        
        time_since_scan = datetime.now() - st.session_state.last_scan_time
        return time_since_scan.total_seconds() >= self.last_scan_interval
    
    def run_pair_scans(self):
        """Run pair trading scans for 15min, 30min, 60min, day only"""
        results = []
        timeframes = ["15min", "30min", "60min", "day"]  # Only these for pair trading
        
        # Lazy import
        try:
            from core.pairs import scan_all_strategies
        except Exception as e:
            logger.warning(f"Could not import scan_all_strategies: {e}")
            return None
        
        for tf in timeframes:
            try:
                logger.info(f"[PAIR SCAN] Starting {tf} pair scan")
                
                # Update active scans
                scan_id = f"pairs_{tf}"
                st.session_state.active_scans[scan_id] = {
                    'strategy': 'Pair Trading',
                    'timeframe': tf,
                    'status': 'Running',
                    'start_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'pairs_checked': 0,
                    'signals_found': 0
                }
                
                # Run scan - only pairs
                try:
                    scan_result = scan_all_strategies(
                        tf=tf,
                        include_pairs=True,
                        include_volatility=False,
                        include_momentum=False,
                        include_mean_reversion=False
                    )
                except KeyError as key_err:
                    # Handle missing column errors gracefully
                    logger.warning(f"Pair scan failed for {tf}: Missing column '{key_err}' - likely no data available")
                    scan_result = None
                except Exception as scan_err:
                    logger.warning(f"Pair scan failed for {tf}: {scan_err}")
                    scan_result = None
                
                # Check if result is valid
                if scan_result is not None:
                    if isinstance(scan_result, pd.DataFrame) and not scan_result.empty:
                        scan_result['scan_time'] = datetime.now(pytz.timezone('Asia/Kolkata'))
                        scan_result['timeframe'] = tf
                        results.append(scan_result)
                        
                        # Update scan status
                        st.session_state.active_scans[scan_id]['signals_found'] = len(scan_result)
                        st.session_state.active_scans[scan_id]['status'] = 'Completed'
                        
                        logger.info(f"[PAIR SCAN] Found {len(scan_result)} signals in {tf}")
                    else:
                        st.session_state.active_scans[scan_id]['status'] = 'No Signals'
                else:
                    st.session_state.active_scans[scan_id]['status'] = 'No Signals'
                    
            except Exception as e:
                try:
                    st.session_state.active_scans[scan_id]['status'] = f'Error: {str(e)[:30]}'
                except:
                    pass
                logger.error(f"Pair scan error for {tf}: {e}")
        
        if results:
            try:
                combined = pd.concat(results, ignore_index=True)
                return combined
            except Exception as e:
                logger.error(f"Error combining results: {e}")
                return None
        return None
    
    def run_momentum_scan(self):
        """Run momentum scans"""
        # Lazy import
        try:
            from core.pairs import scan_all_strategies
        except Exception as e:
            logger.warning(f"Could not import scan_all_strategies: {e}")
            return None
        
        try:
            scan_id = "momentum"
            st.session_state.active_scans[scan_id] = {
                'strategy': 'Momentum',
                'timeframe': '60min',
                'status': 'Running',
                'start_time': datetime.now(pytz.timezone('Asia/Kolkata')),
            }
            
            result = scan_all_strategies(
                tf="60min",
                include_pairs=False,
                include_volatility=False,
                include_momentum=True,
                include_mean_reversion=False
            )
            
            if result is not None and not result.empty:
                result['scan_time'] = datetime.now(pytz.timezone('Asia/Kolkata'))
                result['timeframe'] = '60min'
                st.session_state.active_scans[scan_id]['status'] = 'Completed'
                st.session_state.active_scans[scan_id]['signals_found'] = len(result)
                return result
            else:
                st.session_state.active_scans[scan_id]['status'] = 'No Signals'
                
        except Exception as e:
            st.session_state.active_scans[scan_id]['status'] = f'Error: {str(e)[:30]}'
            logger.error(f"Momentum scan error: {e}")
        
        return None
    
    def run_mean_reversion_scan(self):
        """Run mean reversion scans with buy/sell signals"""
        # Lazy import
        try:
            from core.pairs import scan_all_strategies
        except Exception as e:
            logger.warning(f"Could not import scan_all_strategies: {e}")
            return None
        
        try:
            scan_id = "mean_reversion"
            st.session_state.active_scans[scan_id] = {
                'strategy': 'Mean Reversion',
                'timeframe': '60min',
                'status': 'Running',
                'start_time': datetime.now(pytz.timezone('Asia/Kolkata')),
            }
            
            result = scan_all_strategies(
                tf="60min",
                include_pairs=False,
                include_volatility=False,
                include_momentum=False,
                include_mean_reversion=True
            )
            
            if result is not None and not result.empty:
                result['scan_time'] = datetime.now(pytz.timezone('Asia/Kolkata'))
                result['timeframe'] = '60min'
                # Separate buy and sell signals
                result['signal_type'] = result['recommend'].apply(
                    lambda x: 'BUY' if 'BUY' in str(x).upper() else ('SELL' if 'SELL' in str(x).upper() else 'HOLD')
                )
                st.session_state.active_scans[scan_id]['status'] = 'Completed'
                st.session_state.active_scans[scan_id]['signals_found'] = len(result)
                return result
            else:
                st.session_state.active_scans[scan_id]['status'] = 'No Signals'
                
        except Exception as e:
            st.session_state.active_scans[scan_id]['status'] = f'Error: {str(e)[:30]}'
            logger.error(f"Mean reversion scan error: {e}")
        
        return None
    
    def run_all_scans(self):
        """Run all active scans"""
        if not is_market_open():
            st.session_state.active_scans.clear()
            return None
        
        logger.info("[SCANNER] Starting all scans")
        
        all_results = []
        
        # Run pair scans
        pair_results = self.run_pair_scans()
        if pair_results is not None and not pair_results.empty:
            all_results.append(pair_results)
        
        # Run momentum scan
        momentum_results = self.run_momentum_scan()
        if momentum_results is not None and not momentum_results.empty:
            all_results.append(momentum_results)
        
        # Run mean reversion scan
        mr_results = self.run_mean_reversion_scan()
        if mr_results is not None and not mr_results.empty:
            all_results.append(mr_results)
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            st.session_state.last_scan_time = datetime.now()
            
            # Update recommendations
            self.update_recommendations(combined)
            return combined
        
        st.session_state.last_scan_time = datetime.now()
        return None
    
    def update_recommendations(self, scan_results):
        """Update recommendations list with latest signals"""
        # Ensure session state is initialized
        if 'trade_recommendations' not in st.session_state:
            st.session_state.trade_recommendations = []
        
        if scan_results is None:
            return
        
        # Check if DataFrame is empty properly
        try:
            if isinstance(scan_results, pd.DataFrame):
                if scan_results.empty:
                    return
            else:
                return
        except:
            return
        
        # Convert to list format
        new_recs = []
        try:
            for _, row in scan_results.iterrows():
                rec = {
                    'timestamp': row.get('scan_time', datetime.now(pytz.timezone('Asia/Kolkata'))),
                    'pair': row.get('pair', row.get('symbol', 'N/A')),
                    'strategy': row.get('strategy', 'N/A'),
                    'timeframe': row.get('timeframe', 'N/A'),
                    'signal': row.get('recommend', 'HOLD'),
                    'ml_score': row.get('ml_score', 0),
                    'signal_type': row.get('signal_type', 'HOLD'),
                }
                new_recs.append(rec)
        except Exception as e:
            logger.error(f"Error processing scan results: {e}")
            return
        
        # Add to recommendations (keep last 5)
        st.session_state.trade_recommendations = (new_recs + st.session_state.trade_recommendations)[:self.max_recommendations]
    
    def get_recommendations_df(self):
        """Get recommendations as DataFrame"""
        # Ensure session state is initialized
        if 'trade_recommendations' not in st.session_state:
            st.session_state.trade_recommendations = []
        
        if not st.session_state.trade_recommendations:
            return pd.DataFrame(columns=['Timestamp', 'Pair/Stock', 'Strategy', 'Timeframe', 'Signal', 'Score'])
        
        data = []
        for rec in st.session_state.trade_recommendations:
            data.append({
                'Timestamp': rec['timestamp'].strftime('%H:%M:%S'),
                'Pair/Stock': rec['pair'],
                'Strategy': rec['strategy'],
                'Timeframe': rec['timeframe'],
                'Signal': rec['signal'],
                'Score': f"{rec['ml_score']:.2f}" if isinstance(rec['ml_score'], (int, float)) else 'N/A'
            })
        
        return pd.DataFrame(data)
    
    def get_active_scans_df(self):
        """Get active scans as DataFrame"""
        # Ensure session state is initialized
        if 'active_scans' not in st.session_state:
            st.session_state.active_scans = {}
        
        if not st.session_state.active_scans:
            return pd.DataFrame(columns=['Strategy', 'Timeframe', 'Status', 'Signals', 'Start Time'])
        
        data = []
        for scan_id, scan_info in st.session_state.active_scans.items():
            data.append({
                'Strategy': scan_info.get('strategy', 'N/A'),
                'Timeframe': scan_info.get('timeframe', 'N/A'),
                'Status': scan_info.get('status', 'Unknown'),
                'Signals': scan_info.get('signals_found', 0),
                'Start Time': scan_info.get('start_time', datetime.now()).strftime('%H:%M:%S')
            })
        
        return pd.DataFrame(data)

# Global scanner instance
_scanner = None

def get_scanner():
    """Get or create scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = LiveScannerService()
        _scanner.initialize_session_state()
    return _scanner
