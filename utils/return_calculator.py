# utils/return_calculator.py
# Calculates and manages percentage returns for all timeframes

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from utils.logger import logger

logger = logging.getLogger(__name__)


class ReturnCalculator:
    """
    Calculates percentage returns for OHLCV data
    
    Features:
    - Close-to-close returns
    - Open-to-close returns (intraday)
    - High-Low range analysis
    - Return statistics (mean, std, win rate, etc.)
    - Cumulative returns
    - Multi-timeframe returns
    """
    
    @staticmethod
    def calculate_close_to_close_returns(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Calculate close-to-close percentage returns
        
        Formula: return_% = ((close[t] - close[t-1]) / close[t-1]) * 100
        
        Args:
            data: DataFrame with 'close' column and datetime index
            inplace: Modify original DataFrame
        
        Returns:
            DataFrame with 'return_%' column added
        """
        df = data if inplace else data.copy()
        
        # Ensure 'close' column exists
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Shift close prices to get previous close
        prev_close = df['close'].shift(1)
        
        # Calculate returns (avoid division by zero)
        df['return_%'] = np.where(
            prev_close != 0,
            ((df['close'] - prev_close) / prev_close) * 100,
            0.0
        )
        
        # First bar has no previous close, mark as 0
        if len(df) > 0:
            df.iloc[0, df.columns.get_loc('return_%')] = 0.0
        
        return df
    
    @staticmethod
    def calculate_open_to_close_returns(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Calculate intraday open-to-close percentage returns
        
        Formula: return_% = ((close - open) / open) * 100
        
        Args:
            data: DataFrame with 'open' and 'close' columns
            inplace: Modify original DataFrame
        
        Returns:
            DataFrame with 'intraday_return_%' column added
        """
        df = data if inplace else data.copy()
        
        # Ensure columns exist
        if 'open' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame must have 'open' and 'close' columns")
        
        # Calculate returns
        df['intraday_return_%'] = np.where(
            df['open'] != 0,
            ((df['close'] - df['open']) / df['open']) * 100,
            0.0
        )
        
        return df
    
    @staticmethod
    def calculate_high_low_range(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Calculate high-low range as percentage of close
        
        Formula: hl_range_% = ((high - low) / close) * 100
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            inplace: Modify original DataFrame
        
        Returns:
            DataFrame with 'hl_range_%' column added
        """
        df = data if inplace else data.copy()
        
        # Ensure columns exist
        required = ['high', 'low', 'close']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must have {required} columns")
        
        # Calculate range
        df['hl_range_%'] = np.where(
            df['close'] != 0,
            ((df['high'] - df['low']) / df['close']) * 100,
            0.0
        )
        
        return df
    
    @staticmethod
    def calculate_all_returns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all return types
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all return columns added
        """
        df = data.copy()
        
        # Close-to-close
        df = ReturnCalculator.calculate_close_to_close_returns(df, inplace=True)
        
        # Intraday (if open data available)
        if 'open' in df.columns:
            df = ReturnCalculator.calculate_open_to_close_returns(df, inplace=True)
        
        # High-low range
        if 'high' in df.columns and 'low' in df.columns:
            df = ReturnCalculator.calculate_high_low_range(df, inplace=True)
        
        return df
    
    @staticmethod
    def get_return_statistics(data: pd.DataFrame, return_column: str = 'return_%') -> Dict:
        """
        Calculate comprehensive return statistics
        
        Args:
            data: DataFrame with return column
            return_column: Name of return column (default: 'return_%')
        
        Returns:
            Dictionary with statistics
        """
        if return_column not in data.columns or data.empty:
            return {}
        
        returns = data[return_column].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        stats = {
            # Central tendency
            'mean_return_%': float(returns.mean()),
            'median_return_%': float(returns.median()),
            
            # Volatility
            'std_dev_%': float(returns.std()),
            'variance_%': float(returns.var()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            
            # Extremes
            'min_return_%': float(returns.min()),
            'max_return_%': float(returns.max()),
            'range_%': float(returns.max() - returns.min()),
            
            # Counts
            'total_bars': len(returns),
            'positive_bars': int((returns > 0).sum()),
            'negative_bars': int((returns < 0).sum()),
            'zero_bars': int((returns == 0).sum()),
            
            # Rates
            'win_rate_%': float((returns > 0).sum() / len(returns) * 100) if len(returns) > 0 else 0,
            'loss_rate_%': float((returns < 0).sum() / len(returns) * 100) if len(returns) > 0 else 0,
            
            # Averages
            'avg_positive_return_%': float(positive_returns.mean()) if len(positive_returns) > 0 else 0,
            'avg_negative_return_%': float(negative_returns.mean()) if len(negative_returns) > 0 else 0,
            
            # Risk metrics
            'profit_factor': float(positive_returns.sum() / abs(negative_returns.sum())) if len(negative_returns) > 0 and negative_returns.sum() != 0 else 0,
            
            # Cumulative
            'cumulative_return_%': float(returns.sum()),
            'cumulative_log_return': float(np.log1p(returns / 100).sum()),
        }
        
        return stats
    
    @staticmethod
    def calculate_cumulative_returns(data: pd.DataFrame, return_column: str = 'return_%') -> pd.DataFrame:
        """
        Calculate cumulative returns over time
        
        Args:
            data: DataFrame with return column
            return_column: Name of return column
        
        Returns:
            DataFrame with cumulative return columns added
        """
        df = data.copy()
        
        if return_column not in df.columns:
            raise ValueError(f"Column '{return_column}' not found")
        
        # Cumulative sum of returns
        df['cumulative_return_%'] = df[return_column].cumsum()
        
        # Cumulative product for compound returns
        df['cumulative_compound_return_%'] = (1 + df[return_column] / 100).cumprod() * 100 - 100
        
        # Running drawdown
        cumulative_max = df['cumulative_compound_return_%'].expanding().max()
        df['drawdown_%'] = df['cumulative_compound_return_%'] - cumulative_max
        
        return df
    
    @staticmethod
    def calculate_rolling_returns(
        data: pd.DataFrame,
        window: int,
        return_column: str = 'return_%'
    ) -> pd.DataFrame:
        """
        Calculate rolling period returns
        
        Args:
            data: DataFrame with return column
            window: Number of periods for rolling window
            return_column: Name of return column
        
        Returns:
            DataFrame with rolling return column added
        """
        df = data.copy()
        
        if return_column not in df.columns:
            raise ValueError(f"Column '{return_column}' not found")
        
        # Calculate rolling returns
        col_name = f'rolling_{window}_return_%'
        df[col_name] = df[return_column].rolling(window=window).sum()
        
        return df
    
    @staticmethod
    def compare_returns_across_timeframes(
        data_dict: Dict[str, pd.DataFrame],
        return_column: str = 'return_%'
    ) -> Dict[str, Dict]:
        """
        Compare return statistics across multiple timeframes
        
        Args:
            data_dict: Dictionary mapping timeframe name to DataFrame
            return_column: Name of return column
        
        Returns:
            Dictionary mapping timeframe to statistics
        """
        results = {}
        
        for timeframe, data in data_dict.items():
            results[timeframe] = ReturnCalculator.get_return_statistics(data, return_column)
        
        return results
    
    @staticmethod
    def identify_best_periods(
        data: pd.DataFrame,
        return_column: str = 'return_%',
        top_n: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify best and worst performing periods
        
        Args:
            data: DataFrame with return column and datetime index
            return_column: Name of return column
            top_n: Number of top periods to return
        
        Returns:
            Tuple of (best_periods_df, worst_periods_df)
        """
        if return_column not in data.columns:
            raise ValueError(f"Column '{return_column}' not found")
        
        # Sort by returns
        sorted_data = data.sort_values(return_column, ascending=False)
        
        # Get best and worst
        best = sorted_data.head(top_n)
        worst = sorted_data.tail(top_n)
        
        return best, worst
    
    @staticmethod
    def calculate_returns_by_date(
        data: pd.DataFrame,
        return_column: str = 'return_%'
    ) -> pd.Series:
        """
        Group returns by date (useful for intraday data)
        
        Args:
            data: DataFrame with datetime index and return column
            return_column: Name of return column
        
        Returns:
            Series with daily return sums
        """
        if return_column not in data.columns:
            raise ValueError(f"Column '{return_column}' not found")
        
        # Group by date
        daily_returns = data[return_column].groupby(data.index.date).sum()
        
        return daily_returns
    
    @staticmethod
    def apply_returns_to_data(
        data: pd.DataFrame,
        fill_method: str = 'close_to_close',
        include_all: bool = True
    ) -> pd.DataFrame:
        """
        Convenience method to apply returns to data
        
        Args:
            data: DataFrame with OHLCV
            fill_method: 'close_to_close', 'open_to_close', or 'all'
            include_all: Include additional metrics if True
        
        Returns:
            DataFrame with returns applied
        """
        df = data.copy()
        
        if fill_method == 'close_to_close':
            df = ReturnCalculator.calculate_close_to_close_returns(df, inplace=True)
        elif fill_method == 'open_to_close':
            df = ReturnCalculator.calculate_open_to_close_returns(df, inplace=True)
        elif fill_method == 'all':
            df = ReturnCalculator.calculate_all_returns(df)
        
        if include_all:
            if 'return_%' in df.columns:
                df = ReturnCalculator.calculate_cumulative_returns(df, inplace=True)
        
        return df


def get_return_calculator() -> ReturnCalculator:
    """Get return calculator instance (stateless, can be singleton)"""
    return ReturnCalculator()
