# core/data_aggregation_manager.py
# Aggregates tick/5-minute data to higher timeframes
# Aggregates daily data to weekly/monthly/quarterly/annual
# Calculates returns for all timeframes

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from enum import Enum

from config import BASE_DIR, TIMEFRAMES
from utils.logger import logger

logger = logging.getLogger(__name__)


class AggregationLevel(Enum):
    """Aggregation levels supported"""
    # Intraday - aggregated from 5-minute data
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    SIXTY_MIN = "60m"
    NINETY_MIN = "90m"
    ONE_TWENTY_MIN = "120m"
    ONE_FIFTY_MIN = "150m"
    ONE_EIGHTY_MIN = "180m"
    FOUR_HOUR = "240m"
    
    # Daily - aggregated from daily data
    DAILY = "day"
    WEEKLY = "week"
    MONTHLY = "month"
    QUARTERLY = "quarter"
    ANNUAL = "year"


class DataAggregationManager:
    """
    Manages aggregation of OHLCV data across multiple timeframes
    
    Intraday Aggregation:
    - Source: 5-minute data
    - Aggregates to: 15m, 30m, 60m, 90m, 120m, 150m, 180m, 240m
    
    Daily Aggregation:
    - Source: Daily data
    - Aggregates to: Weekly, Monthly, Quarterly, Annual
    
    Return Calculation:
    - Computes close-to-close % return for each bar
    - Formula: ((close - previous_close) / previous_close) * 100
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize aggregation manager
        
        Args:
            data_dir: Root data directory (default: BASE_DIR/data)
        """
        self.data_dir = data_dir or (BASE_DIR / "data")
        self.cache_dir = self.data_dir / ".aggregation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # OHLCV columns
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        self.with_return_cols = self.ohlcv_cols + ['return_%']
        
        logger.info(f"DataAggregationManager initialized at {self.data_dir}")
    
    # ========================================================================
    # INTRADAY AGGREGATION (5-minute source)
    # ========================================================================
    
    def aggregate_5m_to_15m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 15-minute bars"""
        return self._aggregate_intraday(data_5m, 3)
    
    def aggregate_5m_to_30m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 30-minute bars"""
        return self._aggregate_intraday(data_5m, 6)
    
    def aggregate_5m_to_60m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 60-minute bars"""
        return self._aggregate_intraday(data_5m, 12)
    
    def aggregate_5m_to_90m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 90-minute bars"""
        return self._aggregate_intraday(data_5m, 18)
    
    def aggregate_5m_to_120m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 120-minute bars"""
        return self._aggregate_intraday(data_5m, 24)
    
    def aggregate_5m_to_150m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 150-minute bars"""
        return self._aggregate_intraday(data_5m, 30)
    
    def aggregate_5m_to_180m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 180-minute bars"""
        return self._aggregate_intraday(data_5m, 36)
    
    def aggregate_5m_to_240m(self, data_5m: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-minute bars to 240-minute bars (4 hours)"""
        return self._aggregate_intraday(data_5m, 48)
    
    def _aggregate_intraday(self, data: pd.DataFrame, factor: int) -> pd.DataFrame:
        """
        Aggregate intraday data by grouping N bars
        
        Args:
            data: DataFrame with datetime index and OHLCV columns
            factor: Number of bars to aggregate (e.g., 3 for 15m from 5m)
        
        Returns:
            Aggregated OHLCV DataFrame
        """
        if data.empty:
            return pd.DataFrame()
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            else:
                data.index = pd.to_datetime(data.index)
        
        # Group by factor and aggregate
        grouped = data.groupby(np.arange(len(data)) // factor)
        
        agg_data = pd.DataFrame({
            'open': grouped['open'].first(),
            'high': grouped['high'].max(),
            'low': grouped['low'].min(),
            'close': grouped['close'].last(),
            'volume': grouped['volume'].sum(),
        })
        
        # Use first timestamp of each group
        agg_data.index = grouped.apply(lambda x: x.index[0]).values
        agg_data.index.name = 'datetime'
        
        # Calculate returns
        agg_data = self._calculate_returns(agg_data)
        
        return agg_data
    
    # ========================================================================
    # DAILY AGGREGATION
    # ========================================================================
    
    def aggregate_daily_to_weekly(self, data_daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily bars to weekly bars"""
        return self._aggregate_daily_timeframe(data_daily, 'W')
    
    def aggregate_daily_to_monthly(self, data_daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily bars to monthly bars"""
        return self._aggregate_daily_timeframe(data_daily, 'M')
    
    def aggregate_daily_to_quarterly(self, data_daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily bars to quarterly bars"""
        return self._aggregate_daily_timeframe(data_daily, 'Q')
    
    def aggregate_daily_to_annual(self, data_daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily bars to annual bars"""
        return self._aggregate_daily_timeframe(data_daily, 'Y')
    
    def _aggregate_daily_timeframe(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Aggregate daily data to higher timeframe
        
        Args:
            data: DataFrame with datetime index and OHLCV columns
            freq: Pandas frequency ('W'=week, 'M'=month, 'Q'=quarter, 'Y'=year)
        
        Returns:
            Aggregated OHLCV DataFrame
        """
        if data.empty:
            return pd.DataFrame()
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            else:
                data.index = pd.to_datetime(data.index)
        
        # Resample and aggregate
        resampled = data.resample(freq)
        
        agg_data = pd.DataFrame({
            'open': resampled['open'].first(),
            'high': resampled['high'].max(),
            'low': resampled['low'].min(),
            'close': resampled['close'].last(),
            'volume': resampled['volume'].sum(),
        })
        
        # Remove NaN rows (weeks/months with no data)
        agg_data = agg_data.dropna(subset=['close'])
        agg_data.index.name = 'datetime'
        
        # Calculate returns
        agg_data = self._calculate_returns(agg_data)
        
        return agg_data
    
    # ========================================================================
    # RETURN CALCULATION
    # ========================================================================
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage returns for each bar
        
        Formula: return_% = ((close - previous_close) / previous_close) * 100
        
        Args:
            data: DataFrame with 'close' column
        
        Returns:
            DataFrame with added 'return_%' column
        """
        data_copy = data.copy()
        
        # Shift close prices to get previous close
        prev_close = data_copy['close'].shift(1)
        
        # Calculate returns (avoid division by zero)
        data_copy['return_%'] = np.where(
            prev_close != 0,
            ((data_copy['close'] - prev_close) / prev_close) * 100,
            0.0
        )
        
        # First bar has no previous close, so return is 0
        data_copy.loc[data_copy.index[0], 'return_%'] = 0.0
        
        return data_copy
    
    # ========================================================================
    # BATCH AGGREGATION (Symbol + Timeframe)
    # ========================================================================
    
    def aggregate_symbol_intraday(self, symbol: str, force_recalc: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Aggregate 5-minute data to all intraday timeframes for a symbol
        
        Args:
            symbol: Stock symbol
            force_recalc: Recalculate even if cache exists
        
        Returns:
            Dictionary mapping timeframe name to aggregated DataFrame
        """
        # Load 5-minute data
        df_5m = self._load_symbol_data(symbol, '5m')
        if df_5m is None or df_5m.empty:
            logger.warning(f"No 5m data for {symbol}")
            return {}
        
        results = {
            '5m': df_5m,
            '15m': self.aggregate_5m_to_15m(df_5m),
            '30m': self.aggregate_5m_to_30m(df_5m),
            '60m': self.aggregate_5m_to_60m(df_5m),
            '90m': self.aggregate_5m_to_90m(df_5m),
            '120m': self.aggregate_5m_to_120m(df_5m),
            '150m': self.aggregate_5m_to_150m(df_5m),
            '180m': self.aggregate_5m_to_180m(df_5m),
            '240m': self.aggregate_5m_to_240m(df_5m),
        }
        
        # Cache results
        cache_key = f"{symbol}_intraday"
        self._save_to_cache(cache_key, results)
        
        return results
    
    def aggregate_symbol_daily(self, symbol: str, force_recalc: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Aggregate daily data to weekly, monthly, quarterly, annual for a symbol
        
        Args:
            symbol: Stock symbol
            force_recalc: Recalculate even if cache exists
        
        Returns:
            Dictionary mapping timeframe name to aggregated DataFrame
        """
        # Load daily data
        df_daily = self._load_symbol_data(symbol, 'day')
        if df_daily is None or df_daily.empty:
            logger.warning(f"No daily data for {symbol}")
            return {}
        
        results = {
            'day': df_daily,
            'week': self.aggregate_daily_to_weekly(df_daily),
            'month': self.aggregate_daily_to_monthly(df_daily),
            'quarter': self.aggregate_daily_to_quarterly(df_daily),
            'year': self.aggregate_daily_to_annual(df_daily),
        }
        
        # Cache results
        cache_key = f"{symbol}_daily"
        self._save_to_cache(cache_key, results)
        
        return results
    
    # ========================================================================
    # DATA LOADING & PERSISTENCE
    # ========================================================================
    
    def _load_symbol_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load symbol data from parquet file
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe directory ('5m', '15m', 'day', etc.)
        
        Returns:
            DataFrame or None if file doesn't exist
        """
        tf_dir = self.data_dir / timeframe
        file_path = tf_dir / f"{symbol}.parquet"
        
        try:
            if file_path.exists():
                df = pd.read_parquet(file_path)
                return df
        except Exception as e:
            logger.error(f"Failed to load {symbol} from {timeframe}: {e}")
        
        return None
    
    def save_aggregated_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Save aggregated data to parquet file
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe directory
            data: OHLCV DataFrame
        
        Returns:
            True if successful
        """
        tf_dir = self.data_dir / timeframe
        tf_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = tf_dir / f"{symbol}.parquet"
        
        try:
            data.to_parquet(file_path, compression='snappy')
            logger.debug(f"Saved {symbol} {timeframe} data: {len(data)} bars")
            return True
        except Exception as e:
            logger.error(f"Failed to save {symbol} {timeframe}: {e}")
            return False
    
    # ========================================================================
    # CACHING
    # ========================================================================
    
    def _save_to_cache(self, key: str, data: Dict[str, pd.DataFrame]) -> None:
        """Save aggregated data to cache"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")
    
    def _load_from_cache(self, key: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load aggregated data from cache"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
        return None
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cache entries"""
        if key:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        else:
            # Clear all cache
            import shutil
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STATISTICS & ANALYSIS
    # ========================================================================
    
    def get_return_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate return statistics for a timeframe
        
        Args:
            data: DataFrame with 'return_%' column
        
        Returns:
            Dictionary with statistics
        """
        if 'return_%' not in data.columns or data.empty:
            return {}
        
        returns = data['return_%']
        
        return {
            'mean_return_%': float(returns.mean()),
            'std_dev_%': float(returns.std()),
            'min_return_%': float(returns.min()),
            'max_return_%': float(returns.max()),
            'positive_bars': int((returns > 0).sum()),
            'negative_bars': int((returns < 0).sum()),
            'zero_bars': int((returns == 0).sum()),
            'total_bars': len(returns),
            'win_rate_%': float((returns > 0).sum() / len(returns) * 100) if len(returns) > 0 else 0,
        }
    
    def get_aggregation_metadata(self, symbol: str) -> Dict:
        """
        Get metadata about aggregations for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with aggregation metadata
        """
        metadata = {
            'symbol': symbol,
            'intraday_timeframes': [],
            'daily_timeframes': [],
            'aggregation_timestamp': datetime.now().isoformat(),
        }
        
        # Check intraday
        for tf in ['5m', '15m', '30m', '60m', '90m', '120m', '150m', '180m', '240m']:
            data = self._load_symbol_data(symbol, tf)
            if data is not None and not data.empty:
                metadata['intraday_timeframes'].append({
                    'timeframe': tf,
                    'bar_count': len(data),
                    'date_range': f"{data.index.min()} to {data.index.max()}",
                })
        
        # Check daily
        for tf in ['day', 'week', 'month', 'quarter', 'year']:
            data = self._load_symbol_data(symbol, tf)
            if data is not None and not data.empty:
                metadata['daily_timeframes'].append({
                    'timeframe': tf,
                    'bar_count': len(data),
                    'date_range': f"{data.index.min()} to {data.index.max()}",
                })
        
        return metadata


def get_data_aggregation_manager() -> DataAggregationManager:
    """Get or create singleton instance"""
    if not hasattr(get_data_aggregation_manager, '_instance'):
        get_data_aggregation_manager._instance = DataAggregationManager()
    return get_data_aggregation_manager._instance
