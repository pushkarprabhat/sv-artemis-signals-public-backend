"""
comprehensive_scans.py - All scan types unified with same timeframes
Includes: Pair Trading, Mean Reversion, Momentum, Volatility, Kelly, Options, etc.
Enhanced with derivative status classification and market segment indicators
"""

import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Try to import classifier for enhanced results
try:
    from utils.scan_recommendation_classifier import get_classifier
except ImportError:
    get_classifier = lambda: None


class ComprehensiveScans:
    """All scan types with unified timeframe handling"""
    
    # Unified timeframes for all scans
    UNIFIED_TIMEFRAMES = ['15minute', '30minute', '60minute', 'day']
    
    def __init__(self):
        self.last_scan_time = {}
        self.scan_results = {}
    
    # =========================================================================
    # PAIR TRADING SCANS
    # =========================================================================
    
    def scan_pair_trading(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Pair trading scan - Statistical arbitrage
        Finds correlated stock pairs with divergence opportunities
        """
        try:
            from core.pairs import scan_all_strategies
            
            logger.info(f"[PAIR TRADING] Scanning {timeframe}")
            
            result = scan_all_strategies(
                tf=timeframe,
                include_pairs=True,
                include_volatility=False,
                include_momentum=False,
                include_mean_reversion=False
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'pair_trading'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                logger.info(f"[PAIR TRADING] Found {len(result)} signals @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[PAIR TRADING ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # MEAN REVERSION SCANS
    # =========================================================================
    
    def scan_mean_reversion(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Mean reversion scan - Identify oversold/overbought opportunities
        Uses bollinger bands, RSI, and zscore
        """
        try:
            from core.pairs import scan_all_strategies
            
            logger.info(f"[MEAN REVERSION] Scanning {timeframe}")
            
            result = scan_all_strategies(
                tf=timeframe,
                include_pairs=False,
                include_volatility=False,
                include_momentum=False,
                include_mean_reversion=True
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'mean_reversion'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                # Categorize signals
                result['signal_type'] = result['recommend'].apply(
                    lambda x: 'BUY' if 'BUY' in str(x).upper() else ('SELL' if 'SELL' in str(x).upper() else 'HOLD')
                )
                logger.info(f"[MEAN REVERSION] Found {len(result)} signals @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[MEAN REVERSION ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # MOMENTUM SCANS
    # =========================================================================
    
    def scan_momentum(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Momentum scan - Trend following signals
        Uses MACD, RSI, and ADX indicators
        """
        try:
            from core.pairs import scan_all_strategies
            
            logger.info(f"[MOMENTUM] Scanning {timeframe}")
            
            result = scan_all_strategies(
                tf=timeframe,
                include_pairs=False,
                include_volatility=False,
                include_momentum=True,
                include_mean_reversion=False
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'momentum'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                logger.info(f"[MOMENTUM] Found {len(result)} signals @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[MOMENTUM ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # VOLATILITY SCANS
    # =========================================================================
    
    def scan_volatility(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Volatility scan - High volatility opportunities for options/futures
        Uses VIX, ATR, Bollinger Band Width, and Parkinson volatility
        """
        try:
            from core.volatility import analyze_volatility_universe
            
            logger.info(f"[VOLATILITY] Scanning {timeframe}")
            
            result = analyze_volatility_universe(
                tf=timeframe,
                min_volatility=15.0,  # Minimum implied volatility threshold
                include_greeks=True
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'volatility'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                # Categorize by volatility regime
                result['volatility_regime'] = pd.cut(
                    result['volatility'],
                    bins=[0, 15, 25, 40, float('inf')],
                    labels=['Low', 'Normal', 'High', 'Extreme']
                )
                logger.info(f"[VOLATILITY] Found {len(result)} high-vol opportunities @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[VOLATILITY ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # KELLY CRITERION SCANS
    # =========================================================================
    
    def scan_kelly_criterion(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Kelly Criterion scan - Optimal position sizing for edge-based strategies
        Uses win rate and win/loss ratio to calculate optimal position size
        """
        try:
            from core.kelly import calculate_kelly_positions, scan_kelly_universe
            
            logger.info(f"[KELLY] Scanning {timeframe}")
            
            result = scan_kelly_universe(timeframe=timeframe)
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'kelly_criterion'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                # Kelly values > 0 indicate positive expectancy
                result['is_positive_edge'] = result['kelly_percent'] > 0
                logger.info(f"[KELLY] Found {len(result[result['is_positive_edge']])} positive edge trades @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[KELLY ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # OPTIONS/STRANGLE SCANS
    # =========================================================================
    
    def scan_options_strategies(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Options strategies scan - Iron Butterfly, Strangle, Straddle setups
        Uses volatility skew, IV percentile, and Greeks
        """
        try:
            from core.strangle import scan_strangle_universe
            
            logger.info(f"[OPTIONS] Scanning {timeframe}")
            
            result = scan_strangle_universe(
                tf=timeframe,
                min_dte=7,  # Minimum days to expiry
                min_iv_percentile=30,  # IV threshold
                max_dte=45  # Maximum days to expiry
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'options'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                # Categorize by strategy type
                result['strategy_type'] = result['recommend'].apply(
                    lambda x: 'STRANGLE' if 'STRANGLE' in str(x).upper() 
                    else ('BUTTERFLY' if 'BUTTERFLY' in str(x).upper() 
                    else ('STRADDLE' if 'STRADDLE' in str(x).upper() else 'OTHER'))
                )
                logger.info(f"[OPTIONS] Found {len(result)} setup opportunities @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[OPTIONS ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # GREEKS-BASED SCANS
    # =========================================================================
    
    def scan_greeks(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Greeks scan - High gamma, vega, or theta opportunities
        Useful for options traders to identify favorable setups
        """
        try:
            from core.greeks import scan_greeks_universe
            
            logger.info(f"[GREEKS] Scanning {timeframe}")
            
            result = scan_greeks_universe(
                tf=timeframe,
                min_gamma=0.01,
                min_vega=100,
                min_theta=50
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                result['scan_type'] = 'greeks'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                # Categorize by dominant greek
                result['dominant_greek'] = result[['gamma', 'vega', 'theta']].idxmax(axis=1)
                logger.info(f"[GREEKS] Found {len(result)} high-greek opportunities @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[GREEKS ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # BACKSPREAD SCANS
    # =========================================================================
    
    def scan_backspreads(self, timeframe: str = 'day', **kwargs) -> Optional[pd.DataFrame]:
        """
        Backspread scan - Ratio option spreads for directional moves
        Detects setups for call backspreads, put backspreads, etc.
        """
        try:
            logger.info(f"[BACKSPREAD] Scanning {timeframe}")
            
            # Implement backspread logic here
            # For now, placeholder
            result = pd.DataFrame()
            
            if not result.empty:
                result['scan_type'] = 'backspread'
                result['timeframe'] = timeframe
                result['scan_time'] = datetime.now(IST)
                logger.info(f"[BACKSPREAD] Found {len(result)} opportunities @ {timeframe}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"[BACKSPREAD ERROR] {timeframe}: {e}")
            return None
    
    # =========================================================================
    # RUN ALL SCANS
    # =========================================================================
    
    def run_all_scans(self, 
                     timeframe: str = 'day',
                     enabled_scans: List[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Run all enabled scans for a timeframe
        
        Enabled scans:
        - pair_trading
        - mean_reversion
        - momentum
        - volatility
        - kelly_criterion
        - options_strategies
        - greeks
        - backspreads
        """
        
        if enabled_scans is None:
            enabled_scans = [
                'pair_trading',
                'mean_reversion',
                'momentum',
                'volatility',
                'kelly_criterion',
                'options_strategies',
                'greeks',
            ]
        
        logger.info(f"[SCANS] Running {len(enabled_scans)} scan types @ {timeframe}")
        
        results = {}
        
        scan_methods = {
            'pair_trading': self.scan_pair_trading,
            'mean_reversion': self.scan_mean_reversion,
            'momentum': self.scan_momentum,
            'volatility': self.scan_volatility,
            'kelly_criterion': self.scan_kelly_criterion,
            'options_strategies': self.scan_options_strategies,
            'greeks': self.scan_greeks,
            'backspreads': self.scan_backspreads,
        }
        
        for scan_name in enabled_scans:
            if scan_name in scan_methods:
                try:
                    result = scan_methods[scan_name](timeframe=timeframe)
                    results[scan_name] = result
                    self.scan_results[f"{scan_name}_{timeframe}"] = result
                except Exception as e:
                    logger.error(f"Error running {scan_name}: {e}")
                    results[scan_name] = None
        
        logger.info(f"[SCANS] Completed all scans @ {timeframe}")
        return results
    
    def get_scan_summary(self) -> Dict[str, Any]:
        """Get summary of all scan results"""
        summary = {}
        for key, result in self.scan_results.items():
            if isinstance(result, pd.DataFrame) and not result.empty:
                summary[key] = {
                    'count': len(result),
                    'last_updated': result['scan_time'].max() if 'scan_time' in result.columns else None,
                    'columns': list(result.columns),
                }
            else:
                summary[key] = {'count': 0, 'last_updated': None}
        
        return summary
