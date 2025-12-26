"""
NIFTY-BANKNIFTY Cointegration Strategy
Pair trading between NIFTY50 and BANKNIFTY indices
Detects mean reversion in the spread
File: core/index_spread_cointegration.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import BASE_DIR
from datetime import datetime
import pytz
from utils.logger import logger
from statsmodels.tsa.stattools import coint
from scipy import stats


def load_price(symbol, tf="day"):
    """Load price data for a symbol"""
    try:
        from universe.symbols import load_universe
        
        universe = load_universe()
        instrument_token = None
        symbol_matches = universe[universe['UNDERLYING_SYMBOL'] == symbol]
        if not symbol_matches.empty:
            instrument_token = symbol_matches.iloc[0].get('instrument_token')
        
        if not instrument_token:
            symbol_to_try = symbol
        else:
            symbol_to_try = instrument_token
        
        files = list((BASE_DIR / tf).glob(f"{symbol_to_try}.parquet"))
        if not files:
            files = list((BASE_DIR / tf).glob(f"{symbol_to_try}_*.parquet"))
        
        # Fallback to 60min if daily not found
        if not files and tf == "day":
            files = list((BASE_DIR / "60minute").glob(f"{symbol_to_try}.parquet"))
            if not files:
                files = list((BASE_DIR / "60minute").glob(f"{symbol_to_try}_*.parquet"))
            
            if files:
                df = pd.read_parquet(files[0])
                if 'date' not in df.columns and 'datetime' in df.columns:
                    df['date'] = df['datetime']
                elif 'date' not in df.columns:
                    df['date'] = df.index
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                daily = df['close'].resample('D').ohlc()['close']
                return daily
        
        if not files:
            return None
        
        df = pd.read_parquet(files[0])
        if 'date' not in df.columns and 'datetime' in df.columns:
            df['date'] = df['datetime']
        elif 'date' not in df.columns:
            df['date'] = df.index
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df['close']
    except Exception as e:
        logger.debug(f"Error loading price for {symbol}: {e}")
        return None


def calculate_spread_metrics(nifty_price, banknifty_price):
    """Calculate spread and cointegration metrics"""
    try:
        # Normalize prices to base 100
        nifty_norm = nifty_price / nifty_price.iloc[0] * 100
        banknifty_norm = banknifty_price / banknifty_price.iloc[0] * 100
        
        # Calculate spread (BANKNIFTY - NIFTY)
        spread = banknifty_norm - nifty_norm
        
        # Test for cointegration
        try:
            c_stat, p_value, _ = coint(banknifty_norm, nifty_norm)
            cointegrated = p_value < 0.05
        except:
            cointegrated = False
            p_value = 1.0
        
        # Calculate spread statistics
        mean_spread = spread.mean()
        std_spread = spread.std()
        current_spread = spread.iloc[-1]
        z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0
        
        # Calculate rolling correlation
        rolling_corr = nifty_norm.rolling(20).corr(banknifty_norm)
        current_corr = rolling_corr.iloc[-1]
        
        # Calculate beta (slope of spread relationship)
        x = nifty_norm.values.reshape(-1, 1)
        y = banknifty_norm.values
        
        # Simple linear regression for beta
        slope, intercept, r_value, p_val, std_err = stats.linregress(nifty_norm.values, banknifty_norm.values)
        
        return {
            'spread': current_spread,
            'z_score': z_score,
            'mean_spread': mean_spread,
            'std_spread': std_spread,
            'cointegrated': cointegrated,
            'p_value': p_value,
            'correlation': current_corr,
            'beta': slope,
            'r_squared': r_value ** 2,
            'spread_series': spread
        }
    except Exception as e:
        logger.error(f"Error calculating spread metrics: {e}")
        return None


def scan_index_cointegration_signals(tf="day", z_threshold=1.5):
    """
    Scan for cointegration-based trading signals between NIFTY and BANKNIFTY
    
    When spread is wide (high z-score), expect mean reversion
    When cointegration is strong, pairs trading is more reliable
    
    Args:
        tf: Timeframe to use
        z_threshold: Z-score threshold for signal generation
    
    Returns:
        DataFrame with cointegration signals
    """
    try:
        # Load index prices
        nifty_price = load_price("NIFTY", tf)
        banknifty_price = load_price("BANKNIFTY", tf)
        
        if nifty_price is None or banknifty_price is None:
            logger.warning("Could not load NIFTY or BANKNIFTY prices")
            return pd.DataFrame()
        
        # Get common dates
        common_dates = nifty_price.index.intersection(banknifty_price.index)
        if len(common_dates) < 100:
            logger.warning(f"Insufficient data for cointegration (only {len(common_dates)} bars)")
            return pd.DataFrame()
        
        nifty_common = nifty_price.loc[common_dates]
        banknifty_common = banknifty_price.loc[common_dates]
        
        # Calculate metrics
        metrics = calculate_spread_metrics(nifty_common, banknifty_common)
        if metrics is None:
            return pd.DataFrame()
        
        results = []
        
        # Signal 1: Cointegration-based mean reversion
        if metrics['cointegrated']:
            # When cointegrated and spread is extreme, signal mean reversion
            if abs(metrics['z_score']) > z_threshold:
                if metrics['z_score'] > z_threshold:
                    # BANKNIFTY overperforming - expect pullback
                    recommendation = "SHORT_BANKNIFTY_LONG_NIFTY"
                    setup_desc = "BANKNIFTY overperforming (Long NIFTY, Short BANKNIFTY)"
                    confidence = min(0.95, 0.65 + abs(metrics['z_score']) * 0.08)
                else:
                    # NIFTY overperforming - expect pullback
                    recommendation = "LONG_BANKNIFTY_SHORT_NIFTY"
                    setup_desc = "NIFTY overperforming (Long BANKNIFTY, Short NIFTY)"
                    confidence = min(0.95, 0.65 + abs(metrics['z_score']) * 0.08)
                
                expected_reversion = abs(metrics['z_score']) * metrics['std_spread']
                
                results.append({
                    'pair': 'NIFTY:BANKNIFTY',
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S"),
                    'strategy': 'Index Cointegration',
                    'setup_type': 'Cointegrated Mean Reversion',
                    'setup_description': setup_desc,
                    'recommendation': recommendation,
                    'nifty_price': round(nifty_common.iloc[-1], 2),
                    'banknifty_price': round(banknifty_common.iloc[-1], 2),
                    'spread': round(metrics['spread'], 2),
                    'z_score': round(metrics['z_score'], 2),
                    'spread_mean': round(metrics['mean_spread'], 2),
                    'spread_std': round(metrics['std_spread'], 2),
                    'correlation': round(metrics['correlation'], 3),
                    'beta': round(metrics['beta'], 3),
                    'r_squared': round(metrics['r_squared'], 3),
                    'p_value_coint': round(metrics['p_value'], 4),
                    'expected_reversion': round(expected_reversion, 2),
                    'ml_score': round(confidence * 100, 1),
                    'confidence': round(confidence, 3),
                    'signal_strength': 'STRONG'
                })
        
        # Signal 2: Strong beta-based trading
        if metrics['correlation'] > 0.85:  # Very strong correlation
            # Calculate expected price move based on beta
            recent_nifty_change = ((nifty_common.iloc[-1] - nifty_common.iloc[-20]) / nifty_common.iloc[-20]) * 100
            expected_banknifty_change = recent_nifty_change * metrics['beta']
            
            # If BANKNIFTY hasn't caught up to expected move, potential mean reversion
            recent_banknifty_change = ((banknifty_common.iloc[-1] - banknifty_common.iloc[-20]) / banknifty_common.iloc[-20]) * 100
            residual = expected_banknifty_change - recent_banknifty_change
            
            if abs(residual) > 1.0:  # >1% deviation from beta expectation
                if residual > 0:
                    recommendation = "LONG_BANKNIFTY"
                    setup_desc = "BANKNIFTY lagging NIFTY (expected to catch up)"
                else:
                    recommendation = "SHORT_BANKNIFTY"
                    setup_desc = "BANKNIFTY leading NIFTY (expected to pullback)"
                
                confidence = min(0.90, 0.60 + metrics['r_squared'] * 0.20 + abs(residual) * 0.05)
                
                results.append({
                    'pair': 'NIFTY:BANKNIFTY',
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S"),
                    'strategy': 'Index Cointegration',
                    'setup_type': 'Beta-Based Divergence',
                    'setup_description': setup_desc,
                    'recommendation': recommendation,
                    'nifty_price': round(nifty_common.iloc[-1], 2),
                    'banknifty_price': round(banknifty_common.iloc[-1], 2),
                    'spread': round(metrics['spread'], 2),
                    'z_score': round(metrics['z_score'], 2),
                    'spread_mean': round(metrics['mean_spread'], 2),
                    'spread_std': round(metrics['std_spread'], 2),
                    'correlation': round(metrics['correlation'], 3),
                    'beta': round(metrics['beta'], 3),
                    'r_squared': round(metrics['r_squared'], 3),
                    'p_value_coint': round(metrics['p_value'], 4),
                    'residual_pct': round(residual, 2),
                    'ml_score': round(confidence * 100, 1),
                    'confidence': round(confidence, 3),
                    'signal_strength': 'MODERATE'
                })
        
        if not results:
            logger.debug("No cointegration signals found")
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    except Exception as e:
        logger.error(f"Error scanning index cointegration: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Scanning NIFTY-BANKNIFTY Cointegration...")
    signals = scan_index_cointegration_signals()
    
    if not signals.empty:
        print("\n" + "="*80)
        for _, signal in signals.iterrows():
            print(f"\nStrategy: {signal['strategy']}")
            print(f"Setup: {signal['setup_type']} - {signal['setup_description']}")
            print(f"NIFTY: {signal['nifty_price']} | BANKNIFTY: {signal['banknifty_price']}")
            print(f"Spread: {signal['spread']} (Z={signal['z_score']:.2f})")
            print(f"Correlation: {signal['correlation']:.3f} | Beta: {signal['beta']:.3f} | R²: {signal['r_squared']:.3f}")
            print(f"Recommendation: {signal['recommendation']} (Confidence: {signal['confidence']:.2%})")
    else:
        print("No signals generated")
