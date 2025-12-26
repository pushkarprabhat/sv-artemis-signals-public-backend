"""
Index Constituent Pairs Strategy
Pair an index (e.g., NIFTY50) against its constituent stocks
Detect mean reversion when constituent diverges from index
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import BASE_DIR, P_VALUE_THRESHOLD, HALF_LIFE_MAX
from universe.symbols import load_universe
from statsmodels.tsa.stattools import coint
from datetime import datetime
import pytz
from utils.logger import logger


# NIFTY50 constituents (as of latest update)
NIFTY50_CONSTITUENTS = [
    'RELIANCE', 'TCS', 'INFY', 'HINDUNILEVER', 'ICICIBANK',
    'HDFC', 'LT', 'SBIN', 'MARUTI', 'BAJAJFINSV',
    'BHARTIARTL', 'BAJAJHLDNG', 'ASIANPAINT', 'WIPRO', 'DRREDDY',
    'ONGC', 'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'POWERGRID',
    'NTPC', 'COALINDIA', 'JSWSTEEL', 'INDIGO', 'BPCL',
    'IOC', 'GRASIM', 'UPL', 'HINDALCO', 'BRITANNIA',
    'NESTLEIND', 'AXISBANK', 'EICHERMOT', 'TITAN', 'HCLTECH',
    'TATACONSUM', 'TECHM', 'APOLLOHOSP', 'ADANIPORTS', 'ADANIENT',
    'ADANIGREEN', 'ADANIPOWER', 'SIEMENS', 'KPITTECH', 'BAJAJFINSV',
    'ITC', 'CIPLA', 'PGHH', 'SHREECEM', 'DIVISLAB',
]

# BANKNIFTY constituents (12 major banking stocks)
BANKNIFTY_CONSTITUENTS = [
    'SBIN', 'ICICIBANK', 'HDFC', 'KOTAK', 'AXISBANK',
    'INDUSIND', 'IDBIBANK', 'BANKBARODA', 'IDFCBANK', 'KTKBANK',
    'FEDERALBNK', 'RBLBANK',
]

# NIFTY100 - NIFTY50 = NIFTY Next 50 constituents (sample)
NIFTY_NEXT50_CONSTITUENTS = [
    'AUROPHARMA', 'IDFCBANK', 'AMBUJACEMENTS', 'GODREJCP', 'MRF',
    'PIIND', 'ASHOKLEYLAND', 'TORNTTECH', 'LAXMIMACH', 'CGPOWER',
    'INDHOTEL', 'ZEEL', 'SBILIFE', 'KTKBANK', 'BANKBARODA',
    'CHANORTECH', 'DABUR', 'GICRE', 'BOSCHLTD', 'BAJAJCORP',
    'IDEA', 'VBL', 'MANAPPURAM', 'JINDALSTEL', 'MCDOWELL',
    'MAXHEALTH', 'PHASEGEN', 'SECTORALPH', 'TCSEITECH', 'VGUARD',
    'GMRINFRA', 'EXIDEIND', 'INDNIPPON', 'LAURUSLAB', 'MOIL',
    'GHCL', 'INDIANB', 'IDBIBANK', 'HUDCO', 'INDIGO_INTRA',
    'NUVOCO', 'GRINDWELL', 'POLYCAB', 'SMSPHARM', 'MOTILALOSWL',
    'NAUKRI', 'PAGEIND', 'THERMAAXE', 'CUMMINSIND', 'BEL',
]


def load_price(symbol, tf="day"):
    """Load price data for a symbol"""
    try:
        universe = load_universe()
        
        # CRITICAL: Map symbol to instrument_token for file lookup
        instrument_token = None
        symbol_matches = universe[universe['UNDERLYING_SYMBOL'] == symbol]
        if not symbol_matches.empty:
            instrument_token = symbol_matches.iloc[0].get('instrument_token')
        
        if not instrument_token:
            symbol_to_try = symbol
        else:
            symbol_to_try = instrument_token
        
        # Try both naming patterns: symbol.parquet and symbol_*.parquet
        files = list((BASE_DIR / tf).glob(f"{symbol_to_try}.parquet"))
        if not files:
            files = list((BASE_DIR / tf).glob(f"{symbol_to_try}_*.parquet"))
        
        # If timeframe not found, try fallback to intraday
        if not files and tf == "day":
            logger.debug(f"Daily data not found for {symbol}, attempting 60min aggregation")
            files = list((BASE_DIR / "60minute").glob(f"{symbol_to_try}.parquet"))
            if not files:
                files = list((BASE_DIR / "60minute").glob(f"{symbol_to_try}_*.parquet"))
            
            if files:
                df = pd.read_parquet(files[0])
                if 'date' not in df.columns:
                    if 'datetime' in df.columns:
                        df['date'] = df['datetime']
                    else:
                        df['date'] = df.index
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # Aggregate to daily
                daily = df['close'].resample('D').ohlc()['close']
                logger.debug(f"Aggregated 60min to {len(daily)} daily bars for {symbol}")
                return daily
        
        if not files:
            return None
        
        df = pd.read_parquet(files[0]).set_index('date')
        return df['close']
    except Exception as e:
        logger.debug(f"Error loading price for {symbol}: {e}")
        return None


def calculate_spread_zscore(index_price, constituent_price):
    """Calculate z-score of spread between index and constituent"""
    try:
        # Normalize prices to start at 100
        index_normalized = index_price / index_price.iloc[0] * 100
        const_normalized = constituent_price / constituent_price.iloc[0] * 100
        
        # Calculate spread
        spread = const_normalized - index_normalized
        
        # Calculate z-score
        mean_spread = spread.mean()
        std_spread = spread.std()
        
        if std_spread == 0:
            return None, None, None
        
        z_score = (spread.iloc[-1] - mean_spread) / std_spread
        
        return z_score, spread, std_spread
    except Exception as e:
        logger.debug(f"Error calculating spread z-score: {e}")
        return None, None, None


def scan_index_constituent_pairs(index_symbol, constituents=None, tf="day", z_threshold=2.0):
    """
    Scan for divergences between index and its constituent stocks
    
    Args:
        index_symbol: Index symbol (e.g., "NIFTY50INDEX" or use NIFTY as proxy)
        constituents: List of constituent symbols (e.g., NIFTY50_CONSTITUENTS)
        tf: Timeframe to use
        z_threshold: Z-score threshold for signal generation
    
    Returns:
        DataFrame with signals
    """
    if constituents is None:
        constituents = NIFTY50_CONSTITUENTS
    
    results = []
    
    try:
        # Load index price
        index_price = load_price(index_symbol, tf)
        if index_price is None:
            logger.warning(f"Could not load price for index {index_symbol}")
            return pd.DataFrame()
        
        # Scan each constituent
        for constituent in constituents:
            try:
                const_price = load_price(constituent, tf)
                if const_price is None:
                    continue
                
                # Get common dates
                common_dates = index_price.index.intersection(const_price.index)
                if len(common_dates) < 100:
                    continue
                
                idx_common = index_price.loc[common_dates]
                const_common = const_price.loc[common_dates]
                
                # Calculate spread z-score
                z_score, spread, std_spread = calculate_spread_zscore(idx_common, const_common)
                
                if z_score is None:
                    continue
                
                # Determine signal type based on z-score
                if abs(z_score) < z_threshold:
                    continue  # Not enough divergence
                
                # Signal: Constituent diverging from index
                signal_type = "MEAN_REVERSION"
                if z_score > z_threshold:
                    # Constituent overperforming (above index)
                    recommendation = "SHORT"  # Expect mean reversion down
                    confidence = min(0.95, 0.60 + abs(z_score) * 0.10)
                elif z_score < -z_threshold:
                    # Constituent underperforming (below index)
                    recommendation = "LONG"  # Expect mean reversion up
                    confidence = min(0.95, 0.60 + abs(z_score) * 0.10)
                else:
                    continue
                
                # Calculate expected move (standard deviations back to mean)
                expected_move_pct = (abs(z_score) * std_spread / const_common.iloc[-1]) * 100
                
                # Get current prices for capital calculation
                index_price_current = idx_common.iloc[-1]
                const_price_current = const_common.iloc[-1]
                
                results.append({
                    'pair': f"{index_symbol}:{constituent}",
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S"),
                    'strategy': 'Index Constituent Pair',
                    'metric': f"Z-Score={z_score:.2f} (Underperformance={expected_move_pct:.2f}%)",
                    'index': index_symbol,
                    'constituent': constituent,
                    'index_price': round(index_price_current, 2),
                    'constituent_price': round(const_price_current, 2),
                    'z_score': round(z_score, 2),
                    'spread': round(spread.iloc[-1], 4),
                    'expected_move_pct': round(expected_move_pct, 2),
                    'recommend': recommendation,
                    'ml_score': round(confidence * 100, 1),
                    'confidence': round(confidence, 3),
                    'capital_required': const_price_current * 1,  # 1 lot
                    'max_profit': round(expected_move_pct * 2, 2),
                    'max_loss': round(expected_move_pct * -2, 2),
                })
                
                logger.debug(f"Signal: {constituent} vs {index_symbol} (z={z_score:.2f})")
            
            except Exception as e:
                logger.debug(f"Error scanning {constituent}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Sort by confidence (highest first)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('confidence', ascending=False)
        
        return results_df
    
    except Exception as e:
        logger.error(f"Error scanning index constituent pairs: {e}")
        return pd.DataFrame()


def scan_all_indices(tf="day"):
    """Scan multiple indices and their constituents"""
    all_results = []
    
    # Scan NIFTY50
    nifty50_results = scan_index_constituent_pairs(
        index_symbol="NIFTY",  # Using NIFTY as proxy for NIFTY50INDEX
        constituents=NIFTY50_CONSTITUENTS,
        tf=tf,
        z_threshold=2.0
    )
    if not nifty50_results.empty:
        all_results.append(nifty50_results)
    
    # Scan BANKNIFTY
    banknifty_results = scan_index_constituent_pairs(
        index_symbol="BANKNIFTY",
        constituents=BANKNIFTY_CONSTITUENTS,
        tf=tf,
        z_threshold=2.0
    )
    if not banknifty_results.empty:
        all_results.append(banknifty_results)
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values('confidence', ascending=False)
        return combined.head(50)  # Return top 50 signals
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the strategy
    print("Scanning for index constituent divergences...")
    signals = scan_index_constituent_pairs(
        index_symbol="NIFTY",
        constituents=NIFTY50_CONSTITUENTS[:10],  # Test with first 10
        tf="day"
    )
    
    if not signals.empty:
        print(f"\nFound {len(signals)} signals:")
        print(signals[['pair', 'constituent', 'z_score', 'recommend', 'confidence']].to_string())
    else:
        print("No signals found")
