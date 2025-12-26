#!/usr/bin/env python3
"""
Ichimoku Cloud Trading Strategy - Complete Implementation
==========================================================

Ichimoku Kinko Hyo consists of:
1. TENKAN-SEN (Conversion Line): (9-day high + 9-day low) / 2
2. KIJUN-SEN (Base Line): (26-day high + 26-day low) / 2
3. SENKOU SPAN A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 days ahead
4. SENKOU SPAN B (Leading Span B): (52-day high + 52-day low) / 2, plotted 26 days ahead
5. CHIKOU SPAN (Lagging Span): Close plotted 26 days back

Buy Signals:
- Price > Senkou Span A & B (Kumo)
- Tenkan > Kijun (Bullish crossover)
- Chikou > Price 26 periods ago

Sell Signals:
- Price < Senkou Span A & B (Kumo)
- Tenkan < Kijun (Bearish crossover)
- Chikou < Price 26 periods ago
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BASE_DIR
from utils.logger import logger

class IchimokuStrategy:
    """
    Complete Ichimoku Cloud Trading Strategy Implementation
    """
    
    # Ichimoku parameters (standard)
    TENKAN_PERIOD = 9
    KIJUN_PERIOD = 26
    SENKOU_B_PERIOD = 52
    SENKOU_DISPLACEMENT = 26
    CHIKOU_DISPLACEMENT = 26
    
    def __init__(self):
        self.logger = logger
    
    def calculate_tenkan_sen(self, high, low):
        """
        Tenkan-Sen (Conversion Line)
        (9-day high + 9-day low) / 2
        """
        tenkan = pd.Series(index=high.index, dtype='float64')
        for i in range(len(high)):
            if i < self.TENKAN_PERIOD - 1:
                tenkan.iloc[i] = np.nan
            else:
                period_high = high.iloc[i - self.TENKAN_PERIOD + 1:i + 1].max()
                period_low = low.iloc[i - self.TENKAN_PERIOD + 1:i + 1].min()
                tenkan.iloc[i] = (period_high + period_low) / 2
        return tenkan
    
    def calculate_kijun_sen(self, high, low):
        """
        Kijun-Sen (Base Line)
        (26-day high + 26-day low) / 2
        """
        kijun = pd.Series(index=high.index, dtype='float64')
        for i in range(len(high)):
            if i < self.KIJUN_PERIOD - 1:
                kijun.iloc[i] = np.nan
            else:
                period_high = high.iloc[i - self.KIJUN_PERIOD + 1:i + 1].max()
                period_low = low.iloc[i - self.KIJUN_PERIOD + 1:i + 1].min()
                kijun.iloc[i] = (period_high + period_low) / 2
        return kijun
    
    def calculate_senkou_span_a(self, tenkan_sen, kijun_sen):
        """
        Senkou Span A (Leading Span A)
        (Tenkan + Kijun) / 2, plotted 26 days ahead
        """
        senkou_a = (tenkan_sen + kijun_sen) / 2
        # Shift forward by displacement
        senkou_a = senkou_a.shift(self.SENKOU_DISPLACEMENT)
        return senkou_a
    
    def calculate_senkou_span_b(self, high, low):
        """
        Senkou Span B (Leading Span B)
        (52-day high + 52-day low) / 2, plotted 26 days ahead
        """
        senkou_b = pd.Series(index=high.index, dtype='float64')
        for i in range(len(high)):
            if i < self.SENKOU_B_PERIOD - 1:
                senkou_b.iloc[i] = np.nan
            else:
                period_high = high.iloc[i - self.SENKOU_B_PERIOD + 1:i + 1].max()
                period_low = low.iloc[i - self.SENKOU_B_PERIOD + 1:i + 1].min()
                senkou_b.iloc[i] = (period_high + period_low) / 2
        
        # Shift forward by displacement
        senkou_b = senkou_b.shift(self.SENKOU_DISPLACEMENT)
        return senkou_b
    
    def calculate_chikou_span(self, close):
        """
        Chikou Span (Lagging Span)
        Close plotted 26 days back (or shift backward)
        """
        # Shift backward (negative shift = move backward in time)
        chikou = close.shift(-self.CHIKOU_DISPLACEMENT)
        return chikou
    
    def get_kumo(self, senkou_a, senkou_b):
        """
        Kumo (Cloud) - the area between Senkou Span A and B
        Returns upper and lower bounds
        """
        kumo_upper = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        kumo_lower = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        return kumo_upper, kumo_lower
    
    def calculate_all_ichimoku(self, df):
        """
        Calculate all Ichimoku components
        
        Args:
            df: DataFrame with OHLC data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with all Ichimoku components added
        """
        result = df.copy()
        
        # Ensure we have required columns
        required = ['high', 'low', 'close']
        for col in required:
            if col not in result.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate components
        result['tenkan_sen'] = self.calculate_tenkan_sen(result['high'], result['low'])
        result['kijun_sen'] = self.calculate_kijun_sen(result['high'], result['low'])
        result['senkou_span_a'] = self.calculate_senkou_span_a(result['tenkan_sen'], result['kijun_sen'])
        result['senkou_span_b'] = self.calculate_senkou_span_b(result['high'], result['low'])
        result['chikou_span'] = self.calculate_chikou_span(result['close'])
        
        # Kumo bounds
        result['kumo_upper'], result['kumo_lower'] = self.get_kumo(
            result['senkou_span_a'], 
            result['senkou_span_b']
        )
        
        return result
    
    def generate_signals(self, df):
        """
        Generate buy/sell signals based on Ichimoku
        
        Returns:
            DataFrame with signal columns added
        """
        result = df.copy()
        
        # Ensure Ichimoku components are calculated
        if 'tenkan_sen' not in result.columns:
            result = self.calculate_all_ichimoku(result)
        
        # Initialize signal columns
        result['ichimoku_signal'] = 0  # 0=neutral, 1=buy, -1=sell
        result['signal_reason'] = ''
        
        # Buy/Sell conditions
        for i in range(1, len(result)):
            reasons = []
            signal = 0
            
            # Condition 1: Price > Kumo (Bullish cloud)
            if pd.notna(result['kumo_upper'].iloc[i]):
                if result['close'].iloc[i] > result['kumo_upper'].iloc[i]:
                    reasons.append("Price > Kumo")
                    signal = 1
                elif result['close'].iloc[i] < result['kumo_lower'].iloc[i]:
                    reasons.append("Price < Kumo")
                    signal = -1
            
            # Condition 2: Tenkan > Kijun (Bullish crossover)
            if pd.notna(result['tenkan_sen'].iloc[i]) and pd.notna(result['kijun_sen'].iloc[i]):
                if result['tenkan_sen'].iloc[i] > result['kijun_sen'].iloc[i]:
                    if result['tenkan_sen'].iloc[i-1] <= result['kijun_sen'].iloc[i-1]:
                        reasons.append("Tenkan > Kijun (Bullish)")
                        signal = max(signal, 1)
                elif result['tenkan_sen'].iloc[i] < result['kijun_sen'].iloc[i]:
                    if result['tenkan_sen'].iloc[i-1] >= result['kijun_sen'].iloc[i-1]:
                        reasons.append("Tenkan < Kijun (Bearish)")
                        signal = min(signal, -1)
            
            # Condition 3: Chikou > Price 26 ago (Bullish)
            if i >= self.CHIKOU_DISPLACEMENT:
                price_26_ago = result['close'].iloc[i - self.CHIKOU_DISPLACEMENT]
                chikou_current = result['chikou_span'].iloc[i]
                if pd.notna(chikou_current):
                    if chikou_current > price_26_ago:
                        reasons.append("Chikou > Price(t-26)")
                        signal = max(signal, 1)
                    elif chikou_current < price_26_ago:
                        reasons.append("Chikou < Price(t-26)")
                        signal = min(signal, -1)
            
            result['ichimoku_signal'].iloc[i] = signal
            result['signal_reason'].iloc[i] = ' | '.join(reasons) if reasons else 'No Signal'
        
        return result
    
    def find_current_signal(self, df):
        """
        Get the most recent signal and its strength
        
        Returns:
            dict with signal info
        """
        if len(df) == 0:
            return {'signal': 0, 'reason': 'No data'}
        
        # Get last valid row
        last_idx = len(df) - 1
        last_row = df.iloc[last_idx]
        
        signal = int(last_row['ichimoku_signal'])
        reason = last_row['signal_reason']
        
        # Calculate signal confidence
        confidence = self._calculate_confidence(df, last_idx)
        
        return {
            'signal': signal,  # 1=buy, -1=sell, 0=neutral
            'reason': reason,
            'confidence': confidence,
            'tenkan': last_row['tenkan_sen'],
            'kijun': last_row['kijun_sen'],
            'price': last_row['close'],
            'kumo_upper': last_row['kumo_upper'],
            'kumo_lower': last_row['kumo_lower'],
            'chikou': last_row['chikou_span']
        }
    
    def _calculate_confidence(self, df, idx):
        """
        Calculate signal confidence (0-100%)
        Based on number of bullish/bearish signals aligned
        """
        if idx < 1:
            return 0
        
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        score = 0
        max_score = 6  # Maximum possible aligned signals
        
        # 1. Price vs Kumo
        if pd.notna(row['kumo_upper']):
            if row['close'] > row['kumo_upper']:
                score += 1
            elif row['close'] < row['kumo_lower']:
                score -= 1
        
        # 2. Tenkan > Kijun
        if pd.notna(row['tenkan_sen']) and pd.notna(row['kijun_sen']):
            if row['tenkan_sen'] > row['kijun_sen']:
                score += 1
            else:
                score -= 1
        
        # 3. Tenkan crossover
        if pd.notna(row['tenkan_sen']) and pd.notna(prev_row['tenkan_sen']):
            if row['tenkan_sen'] > row['kijun_sen'] and prev_row['tenkan_sen'] <= prev_row['kijun_sen']:
                score += 2
            elif row['tenkan_sen'] < row['kijun_sen'] and prev_row['tenkan_sen'] >= prev_row['kijun_sen']:
                score -= 2
        
        # 4. Chikou confirmation
        if idx >= self.CHIKOU_DISPLACEMENT:
            price_26_ago = df['close'].iloc[idx - self.CHIKOU_DISPLACEMENT]
            chikou = row['chikou_span']
            if pd.notna(chikou):
                if chikou > price_26_ago:
                    score += 1
                else:
                    score -= 1
        
        # Normalize to 0-100
        confidence = int(50 + (score / max_score) * 50)
        confidence = max(0, min(100, confidence))
        
        return confidence
    
    def get_summary(self, symbol, df):
        """
        Generate trading summary for a symbol
        """
        if len(df) < self.SENKOU_B_PERIOD:
            return {'status': 'Insufficient data', 'minimum_required': self.SENKOU_B_PERIOD}
        
        signal_info = self.find_current_signal(df)
        
        signal_text = 'BUY' if signal_info['signal'] == 1 else ('SELL' if signal_info['signal'] == -1 else 'NEUTRAL')
        
        return {
            'symbol': symbol,
            'signal': signal_text,
            'confidence': f"{signal_info['confidence']}%",
            'current_price': f"{signal_info['price']:.2f}",
            'tenkan': f"{signal_info['tenkan']:.2f}" if pd.notna(signal_info['tenkan']) else 'N/A',
            'kijun': f"{signal_info['kijun']:.2f}" if pd.notna(signal_info['kijun']) else 'N/A',
            'kumo_upper': f"{signal_info['kumo_upper']:.2f}" if pd.notna(signal_info['kumo_upper']) else 'N/A',
            'kumo_lower': f"{signal_info['kumo_lower']:.2f}" if pd.notna(signal_info['kumo_lower']) else 'N/A',
            'chikou': f"{signal_info['chikou']:.2f}" if pd.notna(signal_info['chikou']) else 'N/A',
            'reason': signal_info['reason']
        }

def analyze_symbol_ichimoku(symbol):
    """
    Analyze single symbol with Ichimoku
    """
    try:
        # Load data
        day_file = BASE_DIR / 'day' / f"{symbol}.parquet"
        if not day_file.exists():
            return None
        
        df = pd.read_parquet(day_file)
        
        # Ensure required columns
        required = ['high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                return None
        
        # Calculate Ichimoku
        ichimoku = IchimokuStrategy()
        df_with_ichimoku = ichimoku.calculate_all_ichimoku(df)
        df_with_signals = ichimoku.generate_signals(df_with_ichimoku)
        
        # Get summary
        summary = ichimoku.get_summary(symbol, df_with_signals)
        
        return {
            'summary': summary,
            'dataframe': df_with_signals,
            'last_date': df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['date'].iloc[-1]
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

def scan_ichimoku_signals(symbols=None, min_confidence=50):
    """
    Scan multiple symbols for Ichimoku signals
    
    Args:
        symbols: List of symbols to scan (default: all in data/day)
        min_confidence: Minimum confidence threshold (0-100)
    
    Returns:
        DataFrame with all signals
    """
    if symbols is None:
        # Get all symbols from data/day directory
        day_dir = BASE_DIR / 'day'
        if not day_dir.exists():
            print(f"[ERROR] No data directory: {day_dir}")
            return pd.DataFrame()
        symbols = [f.stem for f in day_dir.glob('*.parquet')]
    
    print(f"\n[SCAN] Analyzing {len(symbols)} symbols with Ichimoku...\n")
    
    buy_signals = []
    sell_signals = []
    neutral_signals = []
    errors = []
    
    for idx, symbol in enumerate(symbols, 1):
        result = analyze_symbol_ichimoku(symbol)
        
        if result is None:
            errors.append(symbol)
            continue
        
        summary = result['summary']
        
        # Filter by confidence
        confidence = int(summary['confidence'].rstrip('%'))
        if confidence < min_confidence:
            continue
        
        # Categorize signals
        if summary['signal'] == 'BUY':
            buy_signals.append(summary)
        elif summary['signal'] == 'SELL':
            sell_signals.append(summary)
        else:
            neutral_signals.append(summary)
        
        # Progress
        if idx % 50 == 0:
            print(f"  [{idx}/{len(symbols)}] Analyzed {idx} symbols...")
    
    # Create results DataFrames
    buy_df = pd.DataFrame(buy_signals)
    sell_df = pd.DataFrame(sell_signals)
    neutral_df = pd.DataFrame(neutral_signals)
    
    print(f"\n[RESULTS] Ichimoku Signal Scan Complete:")
    print(f"  BUY Signals (Confidence > {min_confidence}%): {len(buy_df)}")
    print(f"  SELL Signals (Confidence > {min_confidence}%): {len(sell_df)}")
    print(f"  NEUTRAL (Confidence > {min_confidence}%): {len(neutral_df)}")
    print(f"  Symbols with insufficient data: {len(errors)}\n")
    
    if len(buy_df) > 0:
        print("[BUY SIGNALS] Top opportunities:")
        print(buy_df[['symbol', 'confidence', 'current_price', 'tenkan', 'kijun', 'reason']].head(10).to_string(index=False))
    
    if len(sell_df) > 0:
        print("\n[SELL SIGNALS] Top opportunities:")
        print(sell_df[['symbol', 'confidence', 'current_price', 'tenkan', 'kijun', 'reason']].head(10).to_string(index=False))
    
    return {
        'buy': buy_df,
        'sell': sell_df,
        'neutral': neutral_df,
        'errors': errors
    }

if __name__ == '__main__':
    # Example usage
    print("\n" + "="*80)
    print("ICHIMOKU CLOUD TRADING STRATEGY - SCAN")
    print("="*80)
    
    # Scan all symbols
    results = scan_ichimoku_signals(min_confidence=60)
    
    # Save results
    if len(results['buy']) > 0:
        results['buy'].to_csv('ichimoku_buy_signals.csv', index=False)
        print("\n[SAVED] Buy signals: ichimoku_buy_signals.csv")
    
    if len(results['sell']) > 0:
        results['sell'].to_csv('ichimoku_sell_signals.csv', index=False)
        print("[SAVED] Sell signals: ichimoku_sell_signals.csv")
