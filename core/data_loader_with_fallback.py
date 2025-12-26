# core/data_loader_with_fallback.py
"""
Multi-Provider Data Loader with Automatic Fallback
Tries providers in order: Zerodha → Polygon → AlphaVantage → Yahoo Finance
"""

import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
import time

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Supported data providers"""
    ZERODHA = "zerodha"
    POLYGON = "polygon"
    ALPHAVANTAGE = "alphavantage"
    YAHOO = "yahoo"


class MultiProviderDataLoader:
    """
    Load price data with automatic fallback across multiple providers
    
    Usage:
        loader = MultiProviderDataLoader()
        df = loader.load_price_data('RELIANCE', timeframe='day')
    """
    
    def __init__(self):
        """Initialize multi-provider data loader"""
        self.provider_cache: Dict[str, DataProvider] = {}  # symbol -> last successful provider
        self.provider_order = [
            DataProvider.ZERODHA,
            DataProvider.POLYGON,
            DataProvider.ALPHAVANTAGE,
            DataProvider.YAHOO
        ]
        
        # Import provider health monitor
        try:
            from core.provider_health_monitor import get_provider_monitor
            self.health_monitor = get_provider_monitor()
        except:
            self.health_monitor = None
            logger.warning("Provider health monitor not available")
    
    def load_price_data(self, 
                       symbol: str, 
                       timeframe: str = 'day',
                       from_date: datetime = None,
                       to_date: datetime = None) -> Optional[pd.DataFrame]:
        """
        Load price data with automatic provider fallback
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            timeframe: Data timeframe ('5minute', '15minute', '30minute', '60minute', 'day', 'week', 'month')
            from_date: Start date (default: 1 year ago)
            to_date: End date (default: today)
        
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume]
            None if all providers fail
        """
        # Set default dates
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=365)
        
        # Try cached provider first
        if symbol in self.provider_cache:
            cached_provider = self.provider_cache[symbol]
            df = self._try_provider(cached_provider, symbol, timeframe, from_date, to_date)
            if df is not None:
                logger.info(f"[OK] Loaded {symbol} from cached provider: {cached_provider.value}")
                return df
            else:
                logger.warning(f"Cached provider {cached_provider.value} failed for {symbol}")
        
        # Try all providers in order
        for provider in self.provider_order:
            logger.info(f"Trying provider: {provider.value} for {symbol}")
            
            start_time = time.time()
            df = self._try_provider(provider, symbol, timeframe, from_date, to_date)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if df is not None and not df.empty:
                # Success!
                logger.info(f"[OK] Loaded {symbol} from {provider.value} ({response_time:.0f}ms)")
                self.provider_cache[symbol] = provider
                
                # Record success in health monitor
                if self.health_monitor:
                    self.health_monitor.record_request(provider.value, success=True, response_time_ms=response_time)
                
                return df
            else:
                # Failure
                logger.warning(f"[WARN] Provider {provider.value} failed for {symbol}")
                if self.health_monitor:
                    self.health_monitor.record_request(provider.value, success=False)
        
        # All providers failed
        logger.error(f"[ERROR] All providers failed for {symbol}")
        return None
    
    def _try_provider(self, 
                     provider: DataProvider, 
                     symbol: str, 
                     timeframe: str,
                     from_date: datetime,
                     to_date: datetime) -> Optional[pd.DataFrame]:
        """Try loading data from a specific provider"""
        try:
            if provider == DataProvider.ZERODHA:
                return self._load_from_zerodha(symbol, timeframe, from_date, to_date)
            elif provider == DataProvider.POLYGON:
                return self._load_from_polygon(symbol, timeframe, from_date, to_date)
            elif provider == DataProvider.ALPHAVANTAGE:
                return self._load_from_alphavantage(symbol, timeframe, from_date, to_date)
            elif provider == DataProvider.YAHOO:
                return self._load_from_yahoo(symbol, timeframe, from_date, to_date)
        except Exception as e:
            logger.error(f"Error loading from {provider.value}: {e}")
            return None
    
    def _load_from_zerodha(self, symbol: str, timeframe: str, from_date: datetime, to_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from Zerodha Kite API"""
        try:
            from core.data_manager import get_data_manager
            
            data_mgr = get_data_manager()
            df = data_mgr.load_price_data(symbol, timeframe=timeframe)
            
            if df is not None and not df.empty:
                # Filter by date range
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= from_date) & (df['date'] <= to_date)]
                
                return self._standardize_dataframe(df)
            
            return None
        except Exception as e:
            logger.debug(f"Zerodha load failed: {e}")
            return None
    
    def _load_from_polygon(self, symbol: str, timeframe: str, from_date: datetime, to_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from Polygon.io API"""
        try:
            # TODO: Implement Polygon.io API integration
            # For now, return None (not implemented)
            logger.debug("Polygon.io integration not yet implemented")
            return None
        except Exception as e:
            logger.debug(f"Polygon load failed: {e}")
            return None
    
    def _load_from_alphavantage(self, symbol: str, timeframe: str, from_date: datetime, to_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from AlphaVantage API"""
        try:
            # TODO: Implement AlphaVantage API integration
            # For now, return None (not implemented)
            logger.debug("AlphaVantage integration not yet implemented")
            return None
        except Exception as e:
            logger.debug(f"AlphaVantage load failed: {e}")
            return None
    
    def _load_from_yahoo(self, symbol: str, timeframe: str, from_date: datetime, to_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from Yahoo Finance"""
        try:
            import yfinance as yf
            
            # Map symbol to Yahoo format (add .NS for NSE stocks)
            yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
            # Map timeframe to Yahoo interval
            interval_map = {
                '5minute': '5m',
                '15minute': '15m',
                '30minute': '30m',
                '60minute': '60m',
                'day': '1d',
                'week': '1wk',
                'month': '1mo'
            }
            interval = interval_map.get(timeframe, '1d')
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=from_date, end=to_date, interval=interval)
            
            if df is not None and not df.empty:
                # Rename columns to match our format
                df = df.reset_index()
                df.columns = [col.lower() for col in df.columns]
                
                return self._standardize_dataframe(df)
            
            return None
        except Exception as e:
            logger.debug(f"Yahoo Finance load failed: {e}")
            return None
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format across providers"""
        try:
            # Ensure required columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Rename common variations
            column_map = {
                'datetime': 'date',
                'timestamp': 'date',
                'time': 'date',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vol': 'volume'
            }
            
            df = df.rename(columns=column_map)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Select only required columns (if they exist)
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)
            
            return df
        except Exception as e:
            logger.error(f"Error standardizing DataFrame: {e}")
            return df
    
    def get_provider_for_symbol(self, symbol: str) -> Optional[str]:
        """Get the cached provider for a symbol"""
        if symbol in self.provider_cache:
            return self.provider_cache[symbol].value
        return None
    
    def clear_cache(self, symbol: str = None):
        """Clear provider cache for a symbol or all symbols"""
        if symbol:
            self.provider_cache.pop(symbol, None)
        else:
            self.provider_cache.clear()


# Singleton instance
_loader_instance = None

def get_multi_provider_loader() -> MultiProviderDataLoader:
    """Get singleton instance of multi-provider data loader"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = MultiProviderDataLoader()
    return _loader_instance
