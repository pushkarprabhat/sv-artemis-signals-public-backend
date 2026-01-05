"""
utils/segment_data_loader.py — MULTI-SEGMENT DATA LOADER
==========================================================
Unified interface to load price data for any trading segment:
- Cash (equities)
- Futures (indices)
- Options (indices + selected stocks)
- Commodities (metals, oil, gas)
- Forex (currency pairs)
"""

import pandas as pd
import os
from pathlib import Path
from config import COMMODITY_INDICES, FOREX_INDICES, INDICES


class SegmentDataLoader:
    """Load price data for any segment"""
    
    @staticmethod
    def get_data(symbol, timeframe='day', segment=None):
        """
        Load price data for a symbol.
        
        Args:
            symbol: Symbol name (e.g., 'RELIANCE', 'GOLD', 'USDINR')
            timeframe: 'day', '15minute', '60minute', etc.
            segment: Optional segment hint ('cash', 'commodity', 'forex', 'index')
        
        Returns:
            DataFrame with OHLCV data, or None if not found
        """
        
        # Determine segment if not provided
        if segment is None:
            segment = SegmentDataLoader._determine_segment(symbol)
        
        # Load based on segment
        if segment == 'commodity':
            return SegmentDataLoader._load_commodity_data(symbol, timeframe)
        elif segment == 'forex':
            return SegmentDataLoader._load_forex_data(symbol, timeframe)
        elif segment == 'index':
            return SegmentDataLoader._load_index_data(symbol, timeframe)
        else:  # 'cash' or 'futures'
            return SegmentDataLoader._load_equity_data(symbol, timeframe)
    
    @staticmethod
    def _determine_segment(symbol):
        """Determine which segment a symbol belongs to"""
        if symbol in COMMODITY_INDICES:
            return 'commodity'
        elif symbol in FOREX_INDICES:
            return 'forex'
        elif symbol in INDICES:
            return 'index'
        else:
            return 'cash'
    
    @staticmethod
    def _load_commodity_data(symbol, timeframe):
        """Load commodity price data"""
        file_path = Path(f'marketdata/commodities/{timeframe}/{symbol}.parquet')
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            from core.data_manager import DataManager
            df = DataManager.verify_and_fix_change_pct(df)
            return df.sort_index()
        except Exception as e:
            print(f"Error loading commodity data for {symbol}: {e}")
            return None
    
    @staticmethod
    def _load_forex_data(symbol, timeframe):
        """Load forex price data"""
        file_path = Path(f'marketdata/forex/{timeframe}/{symbol}.parquet')
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            from core.data_manager import DataManager
            df = DataManager.verify_and_fix_change_pct(df)
            return df.sort_index()
        except Exception as e:
            print(f"Error loading forex data for {symbol}: {e}")
            return None
    
    @staticmethod
    def _load_index_data(symbol, timeframe):
        """Load index price data"""
        file_path = Path(f'marketdata/indices/{timeframe}/{symbol}.parquet')
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            from core.data_manager import DataManager
            df = DataManager.verify_and_fix_change_pct(df)
            return df.sort_index()
        except Exception as e:
            print(f"Error loading index data for {symbol}: {e}")
            return None
    
    @staticmethod
    def _load_equity_data(symbol, timeframe):
        """Load equity price data (cash segment)"""
        file_path = Path(f'marketdata/{timeframe}/{symbol}.parquet')
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            from core.data_manager import DataManager
            df = DataManager.verify_and_fix_change_pct(df)
            return df.sort_index()
        except Exception as e:
            print(f"Error loading equity data for {symbol}: {e}")
            return None
    
    @staticmethod
    def load_multiple(symbols, timeframe='day', segment=None):
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of symbol names
            timeframe: Timeframe for all symbols
            segment: Optional segment hint
        
        Returns:
            Dict of {symbol: DataFrame}
        """
        data_dict = {}
        
        for symbol in symbols:
            df = SegmentDataLoader.get_data(symbol, timeframe, segment)
            if df is not None:
                data_dict[symbol] = df
        
        return data_dict


class SegmentScanner:
    """Scanner interface for different segments"""
    
    @staticmethod
    def scan_commodity_pairs(lookback_days=252, min_correlation=0.5):
        """
        Scan for commodity pair trading opportunities.
        
        Args:
            lookback_days: Historical data lookback period
            min_correlation: Minimum correlation for pair consideration
        
        Returns:
            DataFrame with commodity pair signals
        """
        from core.commodity_trading import scan_commodity_pairs
        
        # Load commodity data
        commodity_data = SegmentDataLoader.load_multiple(
            COMMODITY_INDICES,
            timeframe='day',
            segment='commodity'
        )
        
        if not commodity_data:
            return pd.DataFrame()
        
        # Keep only recent data
        for symbol in commodity_data:
            commodity_data[symbol] = commodity_data[symbol].tail(lookback_days)
        
        # Run scan
        try:
            results = scan_commodity_pairs(commodity_data, min_correlation=min_correlation)
            if not results.empty:
                results['segment'] = 'commodity'
                results['strategy'] = 'Commodity Trading'
            return results
        except Exception as e:
            print(f"Error scanning commodity pairs: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def scan_forex_pairs(lookback_days=60, strategy='momentum'):
        """
        Scan for forex trading opportunities.
        
        Args:
            lookback_days: Historical data lookback period
            strategy: 'momentum' or 'mean_reversion'
        
        Returns:
            DataFrame with forex pair signals
        """
        from core.forex_trading import scan_forex_pairs
        
        # Load forex data
        forex_data = SegmentDataLoader.load_multiple(
            FOREX_INDICES,
            timeframe='day',
            segment='forex'
        )
        
        if not forex_data:
            return pd.DataFrame()
        
        # Keep only recent data
        for symbol in forex_data:
            forex_data[symbol] = forex_data[symbol].tail(lookback_days)
        
        # Run scan
        try:
            results = scan_forex_pairs(forex_data, strategy=strategy)
            if not results.empty:
                results['segment'] = 'forex'
                results['strategy'] = f'Forex {strategy.title()}'
            return results
        except Exception as e:
            print(f"Error scanning forex pairs: {e}")
            return pd.DataFrame()


class ResultFormatter:
    """Format scan results for display"""
    
    @staticmethod
    def format_commodity_results(results_df):
        """Format commodity trading results for display"""
        if results_df.empty:
            return pd.DataFrame()
        
        display_df = results_df.copy()
        
        # Rename columns for display
        column_mapping = {
            'Pair': 'Pair',
            'Commodity1': 'Leg 1',
            'Commodity2': 'Leg 2',
            'Correlation': 'Correlation',
            'Z-Score': 'Z-Score',
            'Signal': 'Signal',
            'ML_Score': 'ML Score',
            'Beta': 'Beta',
        }
        
        # Keep only relevant columns
        available_cols = [col for col in column_mapping.keys() if col in display_df.columns]
        display_df = display_df[available_cols].copy()
        
        # Rename
        for old, new in column_mapping.items():
            if old in display_df.columns:
                display_df = display_df.rename(columns={old: new})
        
        # Format numbers
        for col in ['Correlation', 'Z-Score', 'Beta']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        for col in ['ML Score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        return display_df
    
    @staticmethod
    def format_forex_results(results_df):
        """Format forex trading results for display"""
        if results_df.empty:
            return pd.DataFrame()
        
        display_df = results_df.copy()
        
        # Column mapping
        column_mapping = {
            'Pair': 'Currency Pair',
            'Strategy': 'Strategy',
            'Signal': 'Signal',
            'ML_Score': 'ML Score',
        }
        
        # Add momentum or mean reversion specific columns
        if 'momentum_strength' in display_df.columns:
            column_mapping['momentum_strength'] = 'Momentum'
        if 'z_score' in display_df.columns:
            column_mapping['z_score'] = 'Z-Score'
        
        # Keep relevant columns
        available_cols = [col for col in column_mapping.keys() if col in display_df.columns]
        display_df = display_df[available_cols].copy()
        
        # Rename
        for old, new in column_mapping.items():
            if old in display_df.columns:
                display_df = display_df.rename(columns={old: new})
        
        # Format numbers
        for col in ['Momentum', 'Z-Score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        for col in ['ML Score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        return display_df
