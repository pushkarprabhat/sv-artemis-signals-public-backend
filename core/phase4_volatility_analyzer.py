#!/usr/bin/env python3
"""
Phase 4: Advanced Volatility Analysis
Provides realized volatility, GARCH modeling, and regime detection.

Features:
  - Realized volatility (various windows)
  - GARCH(1,1) volatility forecasting
  - Market regime classification (Low/Normal/High vol)
  - Volatility term structure
  - Historical vol percentiles
  - IV analysis when available
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import logger


class VolatilityAnalyzer:
    """Advanced volatility analysis and regime detection"""
    
    def __init__(self):
        self.logger = logger
        self.regime_thresholds = {
            'low': 0.15,      # < 15% = low vol regime
            'normal': 0.35,   # 15-35% = normal vol
            'high': 1.0       # > 35% = high vol regime
        }
        self.logger.info("[VOL_ANALYZER] Volatility analyzer initialized")
    
    def analyze_volatility(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Comprehensive volatility analysis
        
        Args:
            df: OHLCV DataFrame with 'close' column
            symbol: Symbol for logging
        
        Returns:
            {
                'realized_vol_10d': float,
                'realized_vol_20d': float,
                'realized_vol_30d': float,
                'realized_vol_60d': float,
                'garch_vol_forecast': float,      # 1-day ahead forecast
                'regime': str,                    # 'LOW', 'NORMAL', 'HIGH'
                'regime_percentile': float,       # Current vol percentile
                'vol_trend': str,                 # 'INCREASING', 'DECREASING', 'STABLE'
                'high_vol_periods': int,          # Days in high vol regime (last 60)
                'volatility_expansion': float,    # Recent vol / 60d vol
                'mean_reversion_score': float,    # Likelihood to mean revert (0-100)
                'skewness': float,                # Return distribution skewness
                'kurtosis': float,                # Return distribution kurtosis
                'tail_risk': float,               # Probability of extreme moves
                'implied_daily_move': float       # Expected daily move %
            }
        """
        try:
            if len(df) < 30:
                self.logger.warning(f"[VOL_ANALYZER] Insufficient data for {symbol} ({len(df)} rows)")
                return self._default_vol_analysis()
            
            # Calculate returns
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            
            # Realized volatility (annualized)
            vol_10d = returns.tail(10).std() * np.sqrt(252)
            vol_20d = returns.tail(20).std() * np.sqrt(252)
            vol_30d = returns.tail(30).std() * np.sqrt(252)
            vol_60d = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else vol_30d
            
            # GARCH(1,1) forecast
            garch_vol = self._garch_volatility_forecast(returns)
            
            # Regime classification
            current_vol = vol_10d
            regime = self._classify_regime(current_vol)
            
            # Vol percentile
            vol_percentile = self._calculate_vol_percentile(returns.tail(60), current_vol)
            
            # Vol trend
            vol_trend = self._determine_vol_trend(vol_10d, vol_20d, vol_30d)
            
            # High vol periods
            high_vol_periods = (returns.tail(60).std() > self.regime_thresholds['high']).sum()
            
            # Volatility expansion
            vol_expansion = vol_10d / vol_60d if vol_60d > 0 else 1.0
            
            # Mean reversion score (high vol = likely to revert)
            mean_reversion_score = min(100, (current_vol / self.regime_thresholds['high']) * 100)
            
            # Distribution analysis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Tail risk (probability of >2 sigma move)
            tail_risk = 2 * (1 - stats.norm.cdf(2))  # ~4.5%
            if len(returns) >= 20:
                extreme_moves = (np.abs(returns) > returns.std() * 2).sum()
                empirical_tail_risk = extreme_moves / len(returns) * 100
                tail_risk = empirical_tail_risk
            
            # Expected daily move
            daily_move_pct = current_vol / np.sqrt(252)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'realized_vol_10d': vol_10d,
                'realized_vol_20d': vol_20d,
                'realized_vol_30d': vol_30d,
                'realized_vol_60d': vol_60d,
                'garch_vol_forecast': garch_vol,
                'regime': regime,
                'regime_percentile': vol_percentile,
                'vol_trend': vol_trend,
                'high_vol_periods_60d': high_vol_periods,
                'volatility_expansion': vol_expansion,
                'mean_reversion_score': mean_reversion_score,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_risk_percent': tail_risk,
                'implied_daily_move_pct': daily_move_pct * 100,
                'data_points': len(returns),
                'quality': 'GOOD' if len(returns) >= 60 else 'LIMITED'
            }
            
        except Exception as e:
            self.logger.error(f"[VOL_ANALYZER] Error analyzing {symbol}: {e}")
            return self._default_vol_analysis()
    
    def _garch_volatility_forecast(self, returns: pd.Series, p: int = 1, q: int = 1) -> float:
        """
        Simple GARCH(1,1) model for volatility forecasting
        σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)
        
        Typical values: α=0.1, β=0.8
        """
        try:
            if len(returns) < 20:
                return returns.std() * np.sqrt(252)
            
            # Use simplified GARCH estimation
            omega = 0.00001
            alpha = 0.10
            beta = 0.80
            
            # Current variance
            current_var = returns.tail(20).var()
            
            # Latest return squared
            latest_return_squared = returns.iloc[-1] ** 2
            
            # GARCH(1,1) equation
            conditional_variance = omega + alpha * latest_return_squared + beta * current_var
            conditional_vol = np.sqrt(conditional_variance) * np.sqrt(252)
            
            return max(0.01, conditional_vol)
            
        except Exception as e:
            self.logger.debug(f"[VOL_ANALYZER] GARCH calc failed: {e}")
            return returns.std() * np.sqrt(252)
    
    def _classify_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < self.regime_thresholds['low']:
            return 'LOW'
        elif volatility < self.regime_thresholds['normal']:
            return 'NORMAL'
        else:
            return 'HIGH'
    
    def _calculate_vol_percentile(self, returns_window: pd.Series, current_vol: float) -> float:
        """Calculate volatility percentile in recent history"""
        try:
            rolling_vols = []
            for i in range(len(returns_window) - 10):
                window_vol = returns_window.iloc[i:i+10].std() * np.sqrt(252)
                rolling_vols.append(window_vol)
            
            if not rolling_vols:
                return 50
            
            percentile = stats.percentileofscore(rolling_vols, current_vol, nan_policy='omit')
            return max(0, min(100, percentile))
            
        except Exception as e:
            self.logger.debug(f"[VOL_ANALYZER] Percentile calc failed: {e}")
            return 50
    
    def _determine_vol_trend(self, vol_10d: float, vol_20d: float, vol_30d: float) -> str:
        """Determine if volatility is trending up/down"""
        # Moving average approach
        avg_recent = (vol_10d * 0.5 + vol_20d * 0.3 + vol_30d * 0.2)
        avg_older = (vol_20d * 0.5 + vol_30d * 0.5)
        
        change_pct = (avg_recent - avg_older) / avg_older if avg_older > 0 else 0
        
        if change_pct > 0.05:  # > 5% increase
            return 'INCREASING'
        elif change_pct < -0.05:  # > 5% decrease
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def _default_vol_analysis(self) -> Dict:
        """Return default/error analysis"""
        return {
            'realized_vol_10d': 0.25,
            'realized_vol_20d': 0.25,
            'realized_vol_30d': 0.25,
            'realized_vol_60d': 0.25,
            'garch_vol_forecast': 0.25,
            'regime': 'NORMAL',
            'regime_percentile': 50,
            'vol_trend': 'STABLE',
            'high_vol_periods_60d': 0,
            'volatility_expansion': 1.0,
            'mean_reversion_score': 50,
            'skewness': 0,
            'kurtosis': 0,
            'tail_risk_percent': 4.5,
            'implied_daily_move_pct': 1.58,  # ~25% annual / sqrt(252)
            'data_points': 0,
            'quality': 'INSUFFICIENT_DATA'
        }
    
    def detect_volatility_regime_change(self, df_history: Dict[str, List[float]]) -> Dict:
        """
        Detect if volatility regime has changed recently
        
        Args:
            df_history: {'dates': [...], 'volatilities': [...]}
        
        Returns:
            {
                'regime_changed': bool,
                'old_regime': str,
                'new_regime': str,
                'change_date': str,
                'impact': str  # 'MINOR', 'MODERATE', 'MAJOR'
            }
        """
        try:
            if not df_history or len(df_history.get('volatilities', [])) < 30:
                return {'regime_changed': False}
            
            vols = df_history['volatilities']
            old_vol = np.mean(vols[-30:-10])
            new_vol = np.mean(vols[-10:])
            
            old_regime = self._classify_regime(old_vol)
            new_regime = self._classify_regime(new_vol)
            
            changed = old_regime != new_regime
            vol_change = abs(new_vol - old_vol) / old_vol if old_vol > 0 else 0
            
            if vol_change < 0.1:
                impact = 'MINOR'
            elif vol_change < 0.3:
                impact = 'MODERATE'
            else:
                impact = 'MAJOR'
            
            return {
                'regime_changed': changed,
                'old_regime': old_regime,
                'new_regime': new_regime,
                'vol_change_pct': vol_change * 100,
                'impact': impact,
                'change_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.debug(f"[VOL_ANALYZER] Regime change detection failed: {e}")
            return {'regime_changed': False}
