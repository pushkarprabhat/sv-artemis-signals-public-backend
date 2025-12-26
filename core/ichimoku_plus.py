"""
Ichimoku Plus Strategy - Combined Ichimoku Clouds with RSI and ADX
File: core/ichimoku_plus.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import BASE_DIR
from universe.symbols import load_universe
from datetime import datetime
import pytz
from utils.logger import logger


def load_price(symbol, tf="day"):
    """Load OHLC data for a symbol"""
    try:
        universe = load_universe()
        instrument_token = None
        symbol_matches = universe[universe['UNDERLYING_SYMBOL'] == symbol]
        if not symbol_matches.empty:
            instrument_token = symbol_matches.iloc[0].get('instrument_token')
        
        symbol_to_try = instrument_token if instrument_token else symbol
        
        files = list((BASE_DIR / tf).glob(f"{symbol_to_try}.parquet"))
        if not files:
            files = list((BASE_DIR / tf).glob(f"{symbol_to_try}_*.parquet"))
        
        if not files and tf == "day":
            files = list((BASE_DIR / "60minute").glob(f"{symbol_to_try}.parquet"))
            if not files:
                files = list((BASE_DIR / "60minute").glob(f"{symbol_to_try}_*.parquet"))
            
            if files:
                df = pd.read_parquet(files[0])
                if 'date' not in df.columns:
                    df['date'] = df.get('datetime', df.index)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df[['open', 'high', 'low', 'close']].resample('D').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                })
        
        if not files:
            return None
        
        df = pd.read_parquet(files[0])
        if 'date' not in df.columns:
            df['date'] = df.get('datetime', df.index)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df[['open', 'high', 'low', 'close']]
    except Exception as e:
        logger.debug(f"Error loading price for {symbol}: {e}")
        return None


def calculate_ichimoku(df):
    """Calculate Ichimoku Cloud indicators"""
    try:
        # Tenkan-sen (Conversion Line): 9-period high + low / 2
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line): 26-period high + low / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period high + low / 2, plotted 26 periods ahead
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Price plotted 26 periods back
        chikou_span = df['close'].shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    except Exception as e:
        logger.error(f"Error calculating Ichimoku: {e}")
        return None


def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index)"""
    try:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None


def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
    try:
        # True Range
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift(1))
        lc = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up = df['high'].diff()
        down = -df['low'].diff()
        
        pos_dm = up.where((up > down) & (up > 0), 0)
        neg_dm = down.where((down > up) & (down > 0), 0)
        
        pos_di = 100 * pos_dm.rolling(window=period).mean() / atr
        neg_di = 100 * neg_dm.rolling(window=period).mean() / atr
        
        # ADX
        di_sum = pos_di + neg_di
        di_diff = abs(pos_di - neg_di)
        dx = 100 * di_diff / di_sum
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'pos_di': pos_di,
            'neg_di': neg_di,
            'atr': atr
        }
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        return None


def scan_ichimoku_plus(tf="day"):
    """
    Scan all NIFTY500 stocks for Ichimoku Plus signals
    
    Signal Conditions:
    1. ICHIMOKU: Price above Cloud + Tenkan > Kijun + Chikou confirming uptrend
    2. RSI: 30-70 range (not overbought/oversold), >50 for bullish
    3. ADX: >25 for strong trend
    
    Returns:
        DataFrame with Ichimoku signals
    """
    try:
        from universe.symbols import load_universe
        
        universe = load_universe()
        
        # Filter to NIFTY500 + NSE
        if 'nifty500' in universe.columns:
            universe = universe[universe['nifty500'] == True]
        universe = universe[universe['segment'] == 'NSE']
        
        results = []
        
        for _, row in universe.iterrows():
            try:
                symbol = row['UNDERLYING_SYMBOL']
                
                # Load OHLC data
                ohlc = load_price(symbol, tf)
                if ohlc is None or len(ohlc) < 100:
                    continue
                
                # Calculate indicators
                ichimoku = calculate_ichimoku(ohlc)
                rsi = calculate_rsi(ohlc)
                adx_data = calculate_adx(ohlc)
                
                if ichimoku is None or rsi is None or adx_data is None:
                    continue
                
                # Get latest values
                current_price = ohlc['close'].iloc[-1]
                tenkan = ichimoku['tenkan_sen'].iloc[-1]
                kijun = ichimoku['kijun_sen'].iloc[-1]
                senkou_a = ichimoku['senkou_span_a'].iloc[-1]
                senkou_b = ichimoku['senkou_span_b'].iloc[-1]
                chikou = ichimoku['chikou_span'].iloc[-1]
                
                rsi_val = rsi.iloc[-1]
                adx_val = adx_data['adx'].iloc[-1]
                pos_di = adx_data['pos_di'].iloc[-1]
                neg_di = adx_data['neg_di'].iloc[-1]
                
                # Skip if insufficient data
                if pd.isna([tenkan, kijun, senkou_a, senkou_b, chikou, rsi_val, adx_val]).any():
                    continue
                
                # ===== BULLISH SIGNALS =====
                # Cloud top and bottom
                cloud_top = max(senkou_a, senkou_b)
                cloud_bottom = min(senkou_a, senkou_b)
                
                # Ichimoku bullish conditions
                ichimoku_bullish = (
                    current_price > cloud_top and  # Price above cloud
                    tenkan > kijun and  # Tenkan above Kijun (bullish momentum)
                    chikou > current_price  # Chikou confirming trend
                )
                
                # RSI bullish conditions
                rsi_bullish = (
                    30 < rsi_val < 70 and  # Not overbought/oversold
                    rsi_val > 50  # Bullish zone
                )
                
                # ADX trend confirmation
                adx_strong = adx_val > 25  # Strong trend
                adx_bullish = pos_di > neg_di  # Uptrend direction
                
                # ===== BEARISH SIGNALS =====
                ichimoku_bearish = (
                    current_price < cloud_bottom and
                    tenkan < kijun and
                    chikou < current_price
                )
                
                rsi_bearish = (
                    30 < rsi_val < 70 and
                    rsi_val < 50
                )
                
                adx_bearish = neg_di > pos_di
                
                # Generate signals
                if ichimoku_bullish and rsi_bullish and adx_strong and adx_bullish:
                    confidence = min(0.95, 0.70 + (adx_val - 25) * 0.01 + (rsi_val - 50) * 0.005)
                    
                    results.append({
                        'symbol': symbol,
                        'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S"),
                        'strategy': 'Ichimoku Plus',
                        'signal': 'BUY',
                        'price': round(current_price, 2),
                        'tenkan': round(tenkan, 2),
                        'kijun': round(kijun, 2),
                        'cloud_top': round(cloud_top, 2),
                        'cloud_bottom': round(cloud_bottom, 2),
                        'rsi': round(rsi_val, 1),
                        'adx': round(adx_val, 1),
                        '+DI': round(pos_di, 1),
                        '-DI': round(neg_di, 1),
                        'confidence': round(confidence, 3),
                        'ml_score': round(confidence * 100, 1),
                        'setup_description': f"Price above cloud, Tenkan > Kijun, Chikou confirming, RSI bullish, ADX strong ({adx_val:.1f})",
                        'target_1': round(current_price * 1.02, 2),
                        'target_2': round(current_price * 1.05, 2),
                        'stoploss': round(cloud_bottom, 2)
                    })
                
                elif ichimoku_bearish and rsi_bearish and adx_strong and adx_bearish:
                    confidence = min(0.95, 0.70 + (adx_val - 25) * 0.01 + (50 - rsi_val) * 0.005)
                    
                    results.append({
                        'symbol': symbol,
                        'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S"),
                        'strategy': 'Ichimoku Plus',
                        'signal': 'SELL',
                        'price': round(current_price, 2),
                        'tenkan': round(tenkan, 2),
                        'kijun': round(kijun, 2),
                        'cloud_top': round(cloud_top, 2),
                        'cloud_bottom': round(cloud_bottom, 2),
                        'rsi': round(rsi_val, 1),
                        'adx': round(adx_val, 1),
                        '+DI': round(pos_di, 1),
                        '-DI': round(neg_di, 1),
                        'confidence': round(confidence, 3),
                        'ml_score': round(confidence * 100, 1),
                        'setup_description': f"Price below cloud, Tenkan < Kijun, Chikou confirming, RSI bearish, ADX strong ({adx_val:.1f})",
                        'target_1': round(current_price * 0.98, 2),
                        'target_2': round(current_price * 0.95, 2),
                        'stoploss': round(cloud_top, 2)
                    })
            
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('ml_score', ascending=False)
        return results_df.head(30)
    
    except Exception as e:
        logger.error(f"Error in Ichimoku Plus scan: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Scanning Ichimoku Plus signals...")
    signals = scan_ichimoku_plus(tf="day")
    
    if not signals.empty:
        print(f"\nFound {len(signals)} signals")
        print(signals[['symbol', 'signal', 'price', 'rsi', 'adx', 'ml_score']].to_string())
    else:
        print("No signals found")
