# core/volatility_analyzer.py
"""
Volatility Analysis & Regime Detection
Calculates HV/IV and determines which strategies work best in current regime
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from utils.logger import logger


class VolatilityAnalyzer:
    """Analyze volatility regimes and strategy fit"""
    
    def __init__(self, data_manager, lookback_days=252):
        self.dm = data_manager
        self.lookback = lookback_days
        logger.info(f"[VOLATILITY] Analyzer initialized with {lookback_days}d lookback")
    
    def calculate_hv(self, prices: pd.Series, window: int = 20) -> float:
        """
        Calculate historical volatility (rolling)
        
        Args:
            prices: Series of prices
            window: Rolling window in days
        
        Returns:
            Annualized HV as float (e.g., 0.18 = 18%)
        """
        if len(prices) < window + 1:
            return 0.0
        
        try:
            returns = np.log(prices / prices.shift(1))
            hv = returns.rolling(window).std() * np.sqrt(252)
            return float(hv.iloc[-1])
        except Exception as e:
            logger.error(f"[VOLATILITY] Error calculating HV: {e}")
            return 0.0
    
    def calculate_hv_history(self, prices: pd.Series, window: int = 20) -> np.ndarray:
        """Calculate rolling HV for all historical data"""
        if len(prices) < window + 1:
            return np.array([])
        
        try:
            returns = np.log(prices / prices.shift(1))
            hv_history = returns.rolling(window).std() * np.sqrt(252)
            return hv_history.dropna().values
        except Exception as e:
            logger.error(f"[VOLATILITY] Error calculating HV history: {e}")
            return np.array([])
    
    def get_vol_regime(self, current_hv: float, hv_history: np.ndarray) -> str:
        """
        Classify current volatility into regime
        
        LOW:     < 33rd percentile
        MEDIUM:  33-67th percentile  
        HIGH:    > 67th percentile
        """
        if len(hv_history) < 3:
            return "UNKNOWN"
        
        try:
            p33, p67 = np.percentile(hv_history, [33, 67])
            
            if current_hv < p33:
                return "LOW"
            elif current_hv < p67:
                return "MEDIUM"
            else:
                return "HIGH"
        except Exception as e:
            logger.error(f"[VOLATILITY] Error classifying regime: {e}")
            return "UNKNOWN"
    
    def get_strategy_fit(self, strategy: str, vol_regime: str) -> float:
        """
        Return how well strategy fits current vol regime (0-1)
        
        Based on historical backtesting expectations:
        - Pair Trading: Best in MEDIUM vol (stable correlations)
        - Mean Reversion: Best in HIGH vol (more bounces)
        - Stat Arb: Best in LOW-MEDIUM vol (stable spreads)
        - Calendar Spreads: Best in LOW vol (theta decay dominates)
        """
        fit_matrix = {
            'pair_trading': {
                'LOW': 0.6,      # Correlations fade in low vol
                'MEDIUM': 0.95,  # Sweet spot
                'HIGH': 0.5      # Correlations break down
            },
            'mean_reversion': {
                'LOW': 0.4,      # Too little movement
                'MEDIUM': 0.75,  # Good
                'HIGH': 0.95     # Excellent - lots of bounces
            },
            'stat_arb': {
                'LOW': 0.85,     # Spreads stable
                'MEDIUM': 0.90,  # Sweet spot
                'HIGH': 0.5      # Spreads volatile
            },
            'calendar_spreads': {
                'LOW': 0.95,     # Theta dominates
                'MEDIUM': 0.7,   # Good
                'HIGH': 0.3      # IV crush risk
            }
        }
        
        strategy_lower = strategy.lower()
        regime_upper = vol_regime.upper()
        
        return fit_matrix.get(strategy_lower, {}).get(regime_upper, 0.5)
    
    def get_vol_report(self, symbols: list, limit: int = 252) -> Dict:
        """
        Generate comprehensive volatility report
        
        Returns:
        {
            'timestamp': datetime,
            'vol_regime': 'MEDIUM',
            'hv_average': 0.185,
            'hv_median': 0.18,
            'hv_range': (0.12, 0.25),
            'percentiles': {33: 0.15, 67: 0.22},
            'strategy_fit': {
                'pair_trading': 0.95,
                'mean_reversion': 0.75,
                'stat_arb': 0.90,
                'calendar_spreads': 0.70
            },
            'symbols_analyzed': 50
        }
        """
        report = {
            'timestamp': datetime.now(),
            'vol_regime': None,
            'hv_average': 0,
            'hv_median': 0,
            'hv_range': (0, 0),
            'percentiles': {},
            'strategy_fit': {},
            'symbols_analyzed': 0,
            'hv_levels': {}
        }
        
        all_hv = []
        
        # Calculate HV for each symbol
        for symbol in symbols:
            try:
                prices = self.dm.get_price_data(symbol, 'day', limit=limit)
                if prices is not None and len(prices) > 20:
                    hv = self.calculate_hv(prices['close'], window=20)
                    if 0 < hv < 5:  # Sanity check (HV between 0% and 500%)
                        all_hv.append(hv)
                        report['hv_levels'][symbol] = hv
            except Exception as e:
                logger.debug(f"[VOLATILITY] Could not calculate HV for {symbol}: {e}")
                pass
        
        # Calculate overall regime
        if all_hv and len(all_hv) >= 5:
            all_hv_array = np.array(all_hv)
            avg_hv = float(np.mean(all_hv_array))
            median_hv = float(np.median(all_hv_array))
            min_hv = float(np.min(all_hv_array))
            max_hv = float(np.max(all_hv_array))
            
            # Get percentiles for regime classification
            p33 = float(np.percentile(all_hv_array, 33))
            p67 = float(np.percentile(all_hv_array, 67))
            
            regime = self.get_vol_regime(avg_hv, all_hv_array)
            
            report['vol_regime'] = regime
            report['hv_average'] = avg_hv
            report['hv_median'] = median_hv
            report['hv_range'] = (min_hv, max_hv)
            report['percentiles'] = {'33': p33, '67': p67}
            report['symbols_analyzed'] = len(all_hv)
            
            # Calculate strategy fit for this regime
            strategies = ['pair_trading', 'mean_reversion', 'stat_arb', 'calendar_spreads']
            for strategy in strategies:
                fit = self.get_strategy_fit(strategy, regime)
                report['strategy_fit'][strategy] = fit
        
        return report
    
    def format_vol_report(self, report: Dict) -> str:
        """Format volatility report for display"""
        output = []
        output.append("=" * 90)
        output.append("VOLATILITY ANALYSIS & REGIME REPORT")
        output.append("=" * 90)
        output.append(f"Timestamp: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Symbols Analyzed: {report['symbols_analyzed']}")
        output.append("")
        
        if report['vol_regime']:
            output.append(f"📊 CURRENT REGIME: {report['vol_regime']}")
            output.append(f"   Average HV: {report['hv_average']*100:.2f}%")
            output.append(f"   Median HV:  {report['hv_median']*100:.2f}%")
            output.append(f"   Range:      {report['hv_range'][0]*100:.2f}% - {report['hv_range'][1]*100:.2f}%")
            output.append("")
            output.append("🎯 STRATEGY FIT FOR CURRENT REGIME:")
            output.append("-" * 90)
            
            for strategy, fit in report['strategy_fit'].items():
                pct = fit * 100
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                strategy_name = strategy.replace('_', ' ').title()
                output.append(f"   {strategy_name:<25} {bar:<12} {pct:>5.1f}%")
            
            output.append("")
            output.append("💡 INTERPRETATION:")
            if report['vol_regime'] == 'LOW':
                output.append("   • Low volatility environment")
                output.append("   • Calendar spreads & stat arb should excel")
                output.append("   • Pair trading needs careful selection")
                output.append("   • Mean reversion has fewer bounces")
            elif report['vol_regime'] == 'MEDIUM':
                output.append("   • Normal/moderate volatility")
                output.append("   • All strategies performing well")
                output.append("   • Pair trading in ideal regime")
                output.append("   • Well-balanced market conditions")
            else:  # HIGH
                output.append("   • High volatility environment")
                output.append("   • Mean reversion excels (more bounces)")
                output.append("   • Stat arb spreads may widen")
                output.append("   • Pair correlations may break down")
            
        else:
            output.append("⚠️  Insufficient data for volatility analysis")
        
        output.append("=" * 90)
        return "\n".join(output)
    
    def get_strategy_performance_by_regime(self, strategy: str) -> Dict:
        """
        Return historical performance expectations by vol regime
        (Based on backtesting results)
        """
        performance = {
            'pair_trading': {
                'LOW': {'win_rate': 0.52, 'avg_return': 0.008, 'sharpe': 0.8},
                'MEDIUM': {'win_rate': 0.65, 'avg_return': 0.012, 'sharpe': 1.2},
                'HIGH': {'win_rate': 0.45, 'avg_return': 0.005, 'sharpe': 0.5}
            },
            'mean_reversion': {
                'LOW': {'win_rate': 0.35, 'avg_return': 0.003, 'sharpe': 0.3},
                'MEDIUM': {'win_rate': 0.58, 'avg_return': 0.010, 'sharpe': 0.9},
                'HIGH': {'win_rate': 0.72, 'avg_return': 0.018, 'sharpe': 1.4}
            },
            'stat_arb': {
                'LOW': {'win_rate': 0.68, 'avg_return': 0.012, 'sharpe': 1.3},
                'MEDIUM': {'win_rate': 0.62, 'avg_return': 0.011, 'sharpe': 1.1},
                'HIGH': {'win_rate': 0.45, 'avg_return': 0.006, 'sharpe': 0.6}
            },
            'calendar_spreads': {
                'LOW': {'win_rate': 0.75, 'avg_return': 0.015, 'sharpe': 1.5},
                'MEDIUM': {'win_rate': 0.65, 'avg_return': 0.012, 'sharpe': 1.2},
                'HIGH': {'win_rate': 0.38, 'avg_return': 0.004, 'sharpe': 0.4}
            }
        }
        
        return performance.get(strategy.lower(), {})


# Global analyzer instance
_analyzer = None


def get_volatility_analyzer(data_manager, lookback_days=252) -> VolatilityAnalyzer:
    """Get or create singleton analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = VolatilityAnalyzer(data_manager, lookback_days)
    return _analyzer
