# utils/analyzer_optimization.py
# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER OPTIMIZATION UTILITIES
# Performance enhancements for all analyzers - vectorization, caching, etc.
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# VECTORIZED TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

class VectorizedIndicators:
    """Optimized, vectorized technical indicator calculations."""
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> np.ndarray:
        """
        Vectorized RSI calculation.
        
        Args:
            close: Close prices Series
            period: RSI period
            
        Returns:
            RSI values array
        """
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).values
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return np.full(len(close), 50)
    
    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized MACD calculation.
        
        Args:
            close: Close prices Series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            (MACD line, Signal line, Histogram)
        """
        try:
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return (macd_line.values, signal_line.values, histogram.values)
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return (np.array([]), np.array([]), np.array([]))
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized Bollinger Bands calculation.
        
        Args:
            close: Close prices Series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            (Upper band, Middle band, Lower band)
        """
        try:
            middle = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return (upper.values, middle.values, lower.values)
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return (np.array([]), np.array([]), np.array([]))
    
    @staticmethod
    def calculate_moving_averages(close: pd.Series, 
                                 periods: List[int] = [20, 50, 200]) -> Dict[str, np.ndarray]:
        """
        Vectorized moving average calculation.
        
        Args:
            close: Close prices Series
            periods: List of MA periods
            
        Returns:
            Dictionary of {period: ma_values}
        """
        try:
            result = {}
            for period in periods:
                result[f'sma_{period}'] = close.rolling(window=period).mean().values
            return result
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> np.ndarray:
        """
        Vectorized ATR (Average True Range) calculation.
        
        Args:
            high: High prices Series
            low: Low prices Series
            close: Close prices Series
            period: ATR period
            
        Returns:
            ATR values array
        """
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=period).mean()
            
            return atr.fillna(method='bfill').values
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return np.array([])
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                            period: int = 14, smooth_k: int = 3, 
                            smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized Stochastic Oscillator calculation.
        
        Args:
            high: High prices Series
            low: Low prices Series
            close: Close prices Series
            period: Stochastic period
            smooth_k: K line smoothing period
            smooth_d: D line smoothing period
            
        Returns:
            (K line, D line)
        """
        try:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            
            k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            k = k.rolling(window=smooth_k).mean()
            d = k.rolling(window=smooth_d).mean()
            
            return (k.fillna(50).values, d.fillna(50).values)
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return (np.array([]), np.array([]))


# ═══════════════════════════════════════════════════════════════════════════
# CACHED CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

class CalculationCache:
    """Cache for expensive calculations."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, symbol: str, indicator: str, params: Dict) -> str:
        """Generate cache key."""
        param_str = '_'.join(f"{k}:{v}" for k, v in sorted(params.items()))
        return f"{symbol}:{indicator}:{param_str}"
    
    def get(self, symbol: str, indicator: str, params: Dict) -> Optional[Any]:
        """Get cached value."""
        key = self.get_key(symbol, indicator, params)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, symbol: str, indicator: str, params: Dict, value: Any) -> None:
        """Set cached value."""
        key = self.get_key(symbol, indicator, params)
        
        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = min(self.cache.keys(), key=lambda k: self.access_count.get(k, 0))
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.access_count.clear()


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def analyze_multiple_symbols(symbols: List[str], analyzer_class, 
                            analysis_method: str, df_dict: Dict[str, pd.DataFrame],
                            **kwargs) -> Dict[str, Dict]:
    """
    Batch analyze multiple symbols in parallel-ready format.
    
    Args:
        symbols: List of symbols to analyze
        analyzer_class: Analyzer class to use
        analysis_method: Method name to call
        df_dict: Dictionary of {symbol: dataframe}
        **kwargs: Additional arguments for analysis method
        
    Returns:
        Dictionary of {symbol: analysis_result}
    """
    results = {}
    analyzer = analyzer_class()
    
    for symbol in symbols:
        try:
            if symbol in df_dict:
                df = df_dict[symbol]
                method = getattr(analyzer, analysis_method)
                results[symbol] = method(df, **kwargs)
            else:
                logger.warning(f"No data for symbol {symbol}")
                results[symbol] = None
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            results[symbol] = None
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DATA OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def downsample_data(df: pd.DataFrame, factor: int = 5) -> pd.DataFrame:
    """
    Downsample data for faster analysis of large datasets.
    
    Args:
        df: DataFrame to downsample
        factor: Downsample factor (keep every Nth row)
        
    Returns:
        Downsampled DataFrame
    """
    return df.iloc[::factor].copy()


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    df = df.copy()
    
    # Convert float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    # Convert int64 to int32 where possible
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() < 2**31:
            df[col] = df[col].astype('int32')
    
    return df


def compress_data(df: pd.DataFrame, min_rows: int = 1000) -> pd.DataFrame:
    """
    Compress large datasets by aggregating low-variance periods.
    
    Args:
        df: DataFrame to compress
        min_rows: Minimum rows to keep
        
    Returns:
        Compressed DataFrame
    """
    if len(df) <= min_rows:
        return df
    
    compression_factor = max(1, len(df) // min_rows)
    return downsample_data(df, factor=compression_factor)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def create_feature_matrix_efficient(df: pd.DataFrame, 
                                   feature_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Efficiently create feature matrix for ML models.
    
    Args:
        df: OHLCV DataFrame
        feature_list: List of feature names to create
        
    Returns:
        DataFrame with engineered features
    """
    features_df = df.copy()
    
    if feature_list is None:
        feature_list = ['roc', 'rsi', 'volatility', 'trend', 'volume_sma']
    
    try:
        if 'roc' in feature_list:
            features_df['roc'] = df['close'].pct_change(14) * 100
        
        if 'rsi' in feature_list:
            features_df['rsi'] = VectorizedIndicators.calculate_rsi(df['close'])
        
        if 'volatility' in feature_list:
            features_df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
        
        if 'trend' in feature_list:
            sma_200 = df['close'].rolling(200).mean()
            features_df['trend'] = (df['close'] - sma_200) / sma_200 * 100
        
        if 'volume_sma' in feature_list:
            features_df['volume_sma'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Drop NaN rows
        features_df = features_df.dropna()
        
        return features_df
    
    except Exception as e:
        logger.error(f"Error creating feature matrix: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE MONITORING
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """Monitor and log analyzer performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, duration: float, data_size: int = 0) -> None:
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = {'count': 0, 'total': 0, 'avg': 0, 'max': 0, 'min': float('inf')}
        
        m = self.metrics[name]
        m['count'] += 1
        m['total'] += duration
        m['avg'] = m['total'] / m['count']
        m['max'] = max(m['max'], duration)
        m['min'] = min(m['min'], duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for name, m in self.metrics.items():
            summary[name] = {
                'avg_ms': f"{m['avg']*1000:.2f}",
                'max_ms': f"{m['max']*1000:.2f}",
                'min_ms': f"{m['min']*1000:.2f}",
                'count': m['count']
            }
        return summary
    
    def log_summary(self) -> None:
        """Log performance summary."""
        summary = self.get_summary()
        logger.info(f"Performance Summary: {summary}")


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Get DataFrame memory usage statistics."""
    return {
        'total_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'by_column': (df.memory_usage(deep=True) / 1024**2).to_dict()
    }


def suggest_optimizations(df: pd.DataFrame) -> List[str]:
    """Suggest optimizations for DataFrame."""
    suggestions = []
    
    # Check for object dtype columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        suggestions.append(f"Consider converting object columns {object_cols} to categorical")
    
    # Check for unused columns
    if len(df.columns) > 10:
        suggestions.append("Consider dropping unused columns")
    
    # Check for large integer columns
    int_cols = df.select_dtypes(include=['int64']).columns.tolist()
    if int_cols:
        suggestions.append(f"Consider converting int64 columns {int_cols} to int32")
    
    # Check data size
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > 100:
        suggestions.append(f"DataFrame is {memory_mb:.1f} MB - consider downsampling")
    
    return suggestions


if __name__ == "__main__":
    # Test optimization utilities
    print("Analyzer optimization utilities initialized")
