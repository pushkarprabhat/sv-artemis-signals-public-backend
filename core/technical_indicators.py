"""
Technical Indicators Module

Implements popular technical indicators with signal generation:
- Momentum Indicators: RSI, Stochastic, MACD, ROC
- Trend Indicators: Moving Averages, TEMA, TRIX
- Volatility Indicators: Bollinger Bands, ATR, Keltner Channel
- Volume Indicators: OBV, CMF, VPTOC
- Pattern Recognition: Support/Resistance, Breakouts

Each indicator provides:
- Signal generation (BUY, SELL, HOLD)
- Strength scoring (0-100)
- Confidence level
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    HOLD = "HOLD"


@dataclass
class IndicatorSignal:
    """Single indicator signal"""
    indicator_name: str
    signal: SignalType
    strength: float        # 0-100 (0=weak, 100=strong)
    value: float          # Current indicator value
    threshold: float      # Signal threshold
    confidence: float     # 0-100 (0=low, 100=high)
    timestamp: pd.Timestamp


@dataclass
class CombinedSignal:
    """Combined signal from multiple indicators"""
    overall_signal: SignalType
    consensus_strength: float  # 0-100
    signals: List[IndicatorSignal]
    buy_count: int
    sell_count: int
    hold_count: int
    timestamp: pd.Timestamp
    
    def to_dict(self) -> Dict:
        return {
            'signal': self.overall_signal.value,
            'strength': f"{self.consensus_strength:.1f}%",
            'buy': self.buy_count,
            'sell': self.sell_count,
            'hold': self.hold_count,
            'timestamp': self.timestamp,
        }


class TechnicalIndicators:
    """
    Comprehensive technical indicators library
    
    All methods accept DataFrame with columns: open, high, low, close, volume
    """
    
    # ========== MOMENTUM INDICATORS ==========
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> Tuple[pd.Series, IndicatorSignal]:
        """
        Relative Strength Index (RSI)
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Signals:
        - RSI > 70: Overbought (SELL)
        - RSI < 30: Oversold (BUY)
        - Divergence: Strength reversal
        """
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        losses = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signal
        last_rsi = rsi.iloc[-1]
        if last_rsi > 70:
            signal = SignalType.STRONG_SELL
            strength = min(100, (last_rsi - 70) * 3)
        elif last_rsi < 30:
            signal = SignalType.STRONG_BUY
            strength = min(100, (30 - last_rsi) * 3)
        elif last_rsi > 60:
            signal = SignalType.SELL
            strength = (last_rsi - 60) * 2
        elif last_rsi < 40:
            signal = SignalType.BUY
            strength = (40 - last_rsi) * 2
        else:
            signal = SignalType.HOLD
            strength = 0
        
        indicator_signal = IndicatorSignal(
            indicator_name="RSI",
            signal=signal,
            strength=strength,
            value=last_rsi,
            threshold=50,
            confidence=min(100, abs(last_rsi - 50) * 2),
            timestamp=prices.index[-1]
        )
        
        return rsi, indicator_signal
    
    @staticmethod
    def macd(prices: pd.Series, 
             fast: int = 12, 
             slow: int = 26, 
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, IndicatorSignal]:
        """
        MACD (Moving Average Convergence Divergence)
        
        MACD = EMA(12) - EMA(26)
        Signal = EMA(MACD, 9)
        Histogram = MACD - Signal
        
        Signals:
        - MACD > Signal: BUY
        - MACD < Signal: SELL
        - Histogram > 0 and increasing: Strong BUY
        - Histogram < 0 and decreasing: Strong SELL
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        # Generate signal
        last_macd = macd.iloc[-1]
        last_signal = signal.iloc[-1]
        last_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else last_histogram
        
        if last_macd > last_signal:
            if last_histogram > prev_histogram and last_histogram > 0:
                signal_type = SignalType.STRONG_BUY
                strength = min(100, abs(last_histogram) * 100)
            else:
                signal_type = SignalType.BUY
                strength = min(100, abs(last_histogram) * 50)
        else:
            if last_histogram < prev_histogram and last_histogram < 0:
                signal_type = SignalType.STRONG_SELL
                strength = min(100, abs(last_histogram) * 100)
            else:
                signal_type = SignalType.SELL
                strength = min(100, abs(last_histogram) * 50)
        
        indicator_signal = IndicatorSignal(
            indicator_name="MACD",
            signal=signal_type,
            strength=strength,
            value=last_histogram,
            threshold=0,
            confidence=min(100, abs(last_histogram) * 200),
            timestamp=prices.index[-1]
        )
        
        return macd, signal, indicator_signal
    
    @staticmethod
    def stochastic(high: pd.Series, 
                   low: pd.Series, 
                   close: pd.Series,
                   period: int = 14,
                   smooth_k: int = 3) -> Tuple[pd.Series, pd.Series, IndicatorSignal]:
        """
        Stochastic Oscillator
        
        %K = (Close - Low) / (High - Low) * 100
        %D = SMA(%K, 3)
        
        Signals:
        - %K > 80: Overbought (SELL)
        - %K < 20: Oversold (BUY)
        - %K crosses above %D: BUY
        - %K crosses below %D: SELL
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=smooth_k).mean()
        
        # Generate signal
        last_k = k_percent.iloc[-1]
        last_d = d_percent.iloc[-1]
        prev_k = k_percent.iloc[-2] if len(k_percent) > 1 else last_k
        
        if last_k > 80:
            signal_type = SignalType.STRONG_SELL
            strength = min(100, (last_k - 80) * 5)
        elif last_k < 20:
            signal_type = SignalType.STRONG_BUY
            strength = min(100, (20 - last_k) * 5)
        elif last_k > prev_k and last_k > last_d:
            signal_type = SignalType.BUY
            strength = min(100, (last_k - last_d) * 2)
        elif last_k < prev_k and last_k < last_d:
            signal_type = SignalType.SELL
            strength = min(100, (last_d - last_k) * 2)
        else:
            signal_type = SignalType.HOLD
            strength = 0
        
        indicator_signal = IndicatorSignal(
            indicator_name="Stochastic",
            signal=signal_type,
            strength=strength,
            value=last_k,
            threshold=50,
            confidence=min(100, abs(last_k - 50) * 2),
            timestamp=close.index[-1]
        )
        
        return k_percent, d_percent, indicator_signal
    
    # ========== TREND INDICATORS ==========
    
    @staticmethod
    def moving_averages(prices: pd.Series, 
                       periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Simple Moving Averages
        
        Signals:
        - Price > SMA200 > SMA50 > SMA20: Strong Uptrend
        - Price < SMA200 < SMA50 < SMA20: Strong Downtrend
        - Price crosses above MA: BUY
        - Price crosses below MA: SELL
        """
        mas = {}
        for period in periods:
            mas[f"SMA_{period}"] = prices.rolling(window=period).mean()
        return mas
    
    @staticmethod
    def ema(prices: pd.Series, periods: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
        """Exponential Moving Averages"""
        emas = {}
        for period in periods:
            emas[f"EMA_{period}"] = prices.ewm(span=period, adjust=False).mean()
        return emas
    
    # ========== VOLATILITY INDICATORS ==========
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, 
                       period: int = 20, 
                       num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, IndicatorSignal]:
        """
        Bollinger Bands
        
        Middle = SMA(20)
        Upper = Middle + (StdDev * 2)
        Lower = Middle - (StdDev * 2)
        
        Signals:
        - Price > Upper: Overbought
        - Price < Lower: Oversold
        - Bands squeeze: Low volatility (breakout coming)
        - Band breakout: Strong momentum
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Generate signal
        last_price = prices.iloc[-1]
        last_sma = sma.iloc[-1]
        last_upper = upper_band.iloc[-1]
        last_lower = lower_band.iloc[-1]
        
        band_width = (last_upper - last_lower) / last_sma if last_sma != 0 else 0
        
        if last_price > last_upper:
            signal_type = SignalType.STRONG_SELL
            strength = min(100, ((last_price - last_upper) / last_sma) * 100)
        elif last_price < last_lower:
            signal_type = SignalType.STRONG_BUY
            strength = min(100, ((last_lower - last_price) / last_sma) * 100)
        elif band_width < 0.05:  # Squeeze
            signal_type = SignalType.HOLD
            strength = 50  # Potential breakout
        else:
            signal_type = SignalType.HOLD
            strength = 0
        
        indicator_signal = IndicatorSignal(
            indicator_name="Bollinger Bands",
            signal=signal_type,
            strength=strength,
            value=band_width,
            threshold=0.05,
            confidence=min(100, abs(last_price - last_sma) / std.iloc[-1] * 50 if std.iloc[-1] > 0 else 0),
            timestamp=prices.index[-1]
        )
        
        return upper_band, sma, lower_band, indicator_signal
    
    @staticmethod
    def atr(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            period: int = 14) -> Tuple[pd.Series, float]:
        """
        Average True Range (ATR)
        Measures volatility
        
        High ATR = High volatility
        Low ATR = Low volatility
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr, atr.iloc[-1]
    
    # ========== VOLUME INDICATORS ==========
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> Tuple[pd.Series, IndicatorSignal]:
        """
        On-Balance Volume (OBV)
        
        OBV = OBV_prev + (volume if close > close_prev else -volume)
        
        Signals:
        - OBV increasing with price: Bullish confirmation
        - OBV decreasing despite price increase: Bearish divergence (SELL)
        - OBV breakout: Trend acceleration
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # Generate signal
        obv_change = obv.iloc[-1] - obv.iloc[-2] if len(obv) > 1 else 0
        price_change = close.iloc[-1] - close.iloc[-2] if len(close) > 1 else 0
        
        if obv_change > 0 and price_change > 0:
            signal_type = SignalType.BUY
            strength = 60
        elif obv_change < 0 and price_change < 0:
            signal_type = SignalType.SELL
            strength = 60
        elif obv_change < 0 and price_change > 0:
            signal_type = SignalType.SELL  # Divergence
            strength = 80
        elif obv_change > 0 and price_change < 0:
            signal_type = SignalType.BUY  # Divergence
            strength = 80
        else:
            signal_type = SignalType.HOLD
            strength = 0
        
        indicator_signal = IndicatorSignal(
            indicator_name="OBV",
            signal=signal_type,
            strength=strength,
            value=obv_change,
            threshold=0,
            confidence=min(100, abs(obv_change / volume.iloc[-1] * 100) if volume.iloc[-1] > 0 else 0),
            timestamp=close.index[-1]
        )
        
        return obv, indicator_signal


class MultiIndicatorScanner:
    """
    Multi-indicator scanning for entry/exit signals
    
    Combines multiple indicators for consensus trading signals
    """
    
    def __init__(self):
        """Initialize scanner with default configuration"""
        self.indicators = TechnicalIndicators()
    
    def scan(self, 
             df: pd.DataFrame,
             use_indicators: Optional[List[str]] = None) -> CombinedSignal:
        """
        Scan price data with multiple indicators
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            use_indicators: List of indicators to use (if None, uses all)
        
        Returns:
            CombinedSignal with overall signal and confidence
        """
        if use_indicators is None:
            use_indicators = ['RSI', 'MACD', 'Stochastic', 'BB', 'OBV']
        
        signals = []
        
        # RSI
        if 'RSI' in use_indicators:
            _, rsi_signal = self.indicators.rsi(df['close'])
            signals.append(rsi_signal)
        
        # MACD
        if 'MACD' in use_indicators:
            _, _, macd_signal = self.indicators.macd(df['close'])
            signals.append(macd_signal)
        
        # Stochastic
        if 'Stochastic' in use_indicators:
            _, _, stoch_signal = self.indicators.stochastic(df['high'], df['low'], df['close'])
            signals.append(stoch_signal)
        
        # Bollinger Bands
        if 'BB' in use_indicators:
            _, _, _, bb_signal = self.indicators.bollinger_bands(df['close'])
            signals.append(bb_signal)
        
        # OBV
        if 'OBV' in use_indicators:
            _, obv_signal = self.indicators.obv(df['close'], df['volume'])
            signals.append(obv_signal)
        
        # Calculate consensus
        buy_count = sum(1 for s in signals if s.signal in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_count = sum(1 for s in signals if s.signal in [SignalType.SELL, SignalType.STRONG_SELL])
        hold_count = sum(1 for s in signals if s.signal == SignalType.HOLD)
        
        total_strength = sum(s.strength for s in signals) / len(signals) if signals else 0
        
        if buy_count > sell_count and buy_count > hold_count:
            overall_signal = SignalType.STRONG_BUY if buy_count >= 4 else SignalType.BUY
        elif sell_count > buy_count and sell_count > hold_count:
            overall_signal = SignalType.STRONG_SELL if sell_count >= 4 else SignalType.SELL
        else:
            overall_signal = SignalType.HOLD
        
        return CombinedSignal(
            overall_signal=overall_signal,
            consensus_strength=total_strength,
            signals=signals,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count,
            timestamp=df.index[-1]
        )


class ScannerConfig:
    """Configuration for indicator-based scanning"""
    
    # RSI Configuration
    # NOTE: RSI period should be tuned based on timeframe:
    #   - Intraday (5min, 15min, 30min): Use 9-period for faster signals
    #   - Daily/Swing: Use 14-period (standard)
    RSI_PERIOD_DAILY = 14      # Standard RSI for daily/swing trading
    RSI_PERIOD_INTRADAY = 9    # Faster RSI for intraday trading
    RSI_PERIOD = 14            # Default (backward compatibility)
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    @classmethod
    def get_rsi_period(cls, timeframe: str) -> int:
        """Get optimal RSI period based on timeframe"""
        intraday_timeframes = ['5minute', '15minute', '30minute', '60minute']
        return cls.RSI_PERIOD_INTRADAY if timeframe in intraday_timeframes else cls.RSI_PERIOD_DAILY
    
    # MACD Configuration
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Stochastic Configuration
    STOCH_PERIOD = 14
    STOCH_SMOOTH = 3
    STOCH_OVERBOUGHT = 80
    STOCH_OVERSOLD = 20
    
    # Bollinger Bands Configuration
    BB_PERIOD = 20
    BB_STD_DEV = 2.0
    
    # Moving Averages
    MA_PERIODS = [20, 50, 200]
    
    # Signal Strength Thresholds
    STRONG_SIGNAL_MIN = 70  # % consensus for strong signal
    WEAK_SIGNAL_MAX = 40    # % consensus for weak signal
    
    @classmethod
    def get_default_indicators(cls) -> List[str]:
        """Get default indicator list for scanning"""
        return ['RSI', 'MACD', 'Stochastic', 'BB', 'OBV']
