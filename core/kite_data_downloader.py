"""
core/kite_data_downloader.py — KITE-FIRST DATA DOWNLOAD STRATEGY
==================================================================
Download price data from Zerodha Kite API as primary provider,
with fallback to yfinance for unavailable instruments.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from kiteconnect import KiteConnect
import time

from config import BASE_DIR, FUTURES_STOCKS, COMMODITY_INDICES, FOREX_INDICES, INDICES
from utils.logger import logger


class KiteDataDownloader:
    """Download data using Kite API with fallback to yfinance"""
    
    def __init__(self, kite=None):
        """
        Initialize with optional Kite instance.
        
        Args:
            kite: KiteConnect instance. If None, will try to use from environment.
        """
        self.kite = kite
        self.fallback_available = False
        # Load instrument mapping (tradingsymbol/display_name/kite_symbol -> instrument_token)
        self.instrument_map = self._load_instrument_map()
        
        # Try to import yfinance as fallback
        try:
            import yfinance
            self.yfinance = yfinance
            self.fallback_available = True
        except ImportError:
            logger.warning("yfinance not installed. Fallback downloads unavailable.")
    
    def download_stock_data(self, symbol, start_date, end_date, timeframe='day'):
        """
        Download stock price data from Kite (with yfinance fallback).
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            start_date: Start date
            end_date: End date
            timeframe: 'day', '15minute', '60minute', etc.
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.kite:
            return self._fallback_download(symbol, start_date, end_date, timeframe)
        
        try:
            # Try Kite first
            logger.info(f"Downloading {symbol} from Kite ({timeframe})...")
            df = self._download_from_kite(symbol, start_date, end_date, timeframe)
            
            if df is not None and not df.empty:
                logger.info(f"[OK] {symbol}: {len(df)} records from Kite")
                return df
            else:
                logger.warning(f"Kite returned empty data for {symbol}, trying fallback...")
                return self._fallback_download(symbol, start_date, end_date, timeframe)
        
        except Exception as e:
            logger.warning(f"Kite download failed for {symbol}: {e}. Using fallback...")
            return self._fallback_download(symbol, start_date, end_date, timeframe)
    
    def _download_from_kite(self, symbol, start_date, end_date, timeframe):
        """Download from Kite API"""
        try:
            # Convert symbol to NSE format if needed
            symbol_nse = f"{symbol}" if symbol.isupper() else symbol.upper()

            # Kite interval mapping
            interval_map = {
                'day': 'day',
                '15minute': '15minute',
                '60minute': '60minute',
                '30minute': '30minute',
                'week': 'week',
                'month': 'month',
            }
            
            interval = interval_map.get(timeframe, 'day')
            # Resolve instrument token: accept numeric token, tradingsymbol, kite_symbol or display_name
            instrument_token = None

            # If symbol looks like an integer token, use directly
            try:
                if str(symbol).isdigit():
                    instrument_token = int(symbol)
            except Exception:
                instrument_token = None

            # Lookup in instrument map (case-insensitive keys stored as upper())
            if instrument_token is None:
                lookup_key = str(symbol_nse).upper()
                instrument_token = self.instrument_map.get(lookup_key)

            if instrument_token is None:
                logger.warning(f"[KITE] Could not resolve instrument token for '{symbol}' — aborting Kite call")
                return None

            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
            )
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            
            # Rename columns to standard format
            column_mapping = {
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            }
            
            df = df.rename(columns=column_mapping)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        
        except Exception as e:
            logger.error(f"Kite download error for {symbol}: {e}")
            return None

    def _load_instrument_map(self):
        """Load instrument mapping from known CSV locations.

        Returns a dict mapping UPPER(tradingsymbol|kite_symbol|display_name) -> instrument_token (int)
        """
        candidates = [
            Path('marketdata') / 'metadata' / 'enriched' / 'enriched_instruments.csv',
            Path('universe') / 'app' / 'app_kite_universe.csv',
            Path('marketdata') / 'metadata' / 'enriched' / 'instruments.csv'
        ]

        mapping = {}

        for p in candidates:
            try:
                if not p.exists():
                    continue

                df = pd.read_csv(p)

                # Normalise possible column names
                token_col = None
                for c in ['InstrumentToken', 'instrument_token', 'instrumentToken']:
                    if c in df.columns:
                        token_col = c
                        break

                # possible trading symbol columns
                trad_cols = [c for c in ['tradingsymbol', 'kite_symbol', 'kiteSymbol', 'kite_symbol', 'display_name', 'tradingsymbol'.upper()] if c in df.columns]

                if token_col is None:
                    continue

                for _, row in df.iterrows():
                    try:
                        token = int(row[token_col])
                    except Exception:
                        continue

                    # Map multiple identifiers
                    for key in ['tradingsymbol', 'kite_symbol', 'display_name', 'name', 'tradingsymbol'.upper()]:
                        if key in df.columns:
                            val = row.get(key)
                            if pd.notna(val):
                                mapping[str(val).upper()] = token

                    # Also map 'name' and 'DISPLAY_NAME' variants
                    for c in df.columns:
                        if c.lower() in ('display_name', 'name', 'kite_symbol', 'tradingsymbol'):
                            v = row.get(c)
                            if pd.notna(v):
                                mapping[str(v).upper()] = token

            except Exception as e:
                logger.debug(f"_load_instrument_map: failed to read {p}: {e}")
                continue

        logger.info(f"[KITE] Loaded instrument map with {len(mapping)} entries from candidate CSVs")
        return mapping
    
    def _fallback_download(self, symbol, start_date, end_date, timeframe):
        """Fallback to yfinance if Kite fails"""
        if not self.fallback_available:
            logger.error(f"Cannot download {symbol}: Kite failed and yfinance unavailable")
            return None
        
        try:
            logger.info(f"Using yfinance fallback for {symbol}...")
            
            # Convert NSE symbol to yfinance format if needed
            yf_symbol = f"{symbol}.NS" if symbol.isupper() else f"{symbol}.NS"
            
            # Map timeframe to yfinance interval
            interval_map = {
                'day': '1d',
                '15minute': '15m',
                '60minute': '1h',
                '30minute': '30m',
                'week': '1wk',
                'month': '1mo',
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Download
            df = self.yfinance.download(
                yf_symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                return None
            
            # Standardize column names
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            return df.sort_index()
        
        except Exception as e:
            logger.error(f"yfinance fallback failed for {symbol}: {e}")
            return None
    
    def download_commodity_data(self, symbol, start_date, end_date, timeframe='day'):
        """Download commodity data (Kite doesn't have commodities, use yfinance only)"""
        if not self.fallback_available:
            logger.error(f"Cannot download commodity {symbol}: yfinance required")
            return None
        
        return self._fallback_download(symbol, start_date, end_date, timeframe)
    
    def download_forex_data(self, symbol, start_date, end_date, timeframe='day'):
        """Download forex data (Kite doesn't have forex, use yfinance only)"""
        if not self.fallback_available:
            logger.error(f"Cannot download forex {symbol}: yfinance required")
            return None
        
        return self._fallback_download(symbol, start_date, end_date, timeframe)
    
    def download_index_data(self, symbol, start_date, end_date, timeframe='day'):
        """Download index data (use yfinance for indices)"""
        if not self.fallback_available:
            logger.error(f"Cannot download index {symbol}: yfinance required")
            return None
        
        return self._fallback_download(symbol, start_date, end_date, timeframe)
    
    def download_batch_stocks(self, symbols, start_date, end_date, timeframe='day', parallel=False):
        """
        Download data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            parallel: Use parallel downloads (not yet implemented)
        
        Returns:
            Dict of {symbol: DataFrame}
        """
        data_dict = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Downloading {symbol}...")
            
            try:
                df = self.download_stock_data(symbol, start_date, end_date, timeframe)
                
                if df is not None and not df.empty:
                    data_dict[symbol] = df
                    
                    # Save to parquet
                    self._save_data(df, symbol, 'stocks', timeframe)
                
                # Rate limiting
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                continue
        
        logger.info(f"Downloaded {len(data_dict)}/{len(symbols)} stocks successfully")
        return data_dict
    
    def download_batch_futures_stocks(self, symbols=None, start_date=None, end_date=None, timeframe='day'):
        """
        Download data for futures-tradable stocks.
        
        Args:
            symbols: List of stock symbols. If None, use predefined FUTURES_STOCKS.
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: today)
            timeframe: Timeframe
        
        Returns:
            Dict of {symbol: DataFrame}
        """
        if symbols is None:
            symbols = FUTURES_STOCKS
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=1825)
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Downloading {len(symbols)} futures stocks...")
        return self.download_batch_stocks(symbols, start_date, end_date, timeframe)
    
    def _save_data(self, df, symbol, segment, timeframe):
        """Save data to parquet file"""
        try:
            # Create directory
            save_dir = BASE_DIR / segment / timeframe
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = save_dir / f"{symbol}.parquet"
            df.to_parquet(file_path, compression='snappy')
            
            logger.info(f"Saved {symbol} to {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to save {symbol}: {e}")


class FuturesDataDownloader:
    """Specialized downloader for futures contracts"""
    
    def __init__(self, kite):
        """Initialize with Kite instance"""
        self.kite = kite
        self.downloader = KiteDataDownloader(kite)
    
    def download_futures_stock(self, symbol, expiry_month=None, start_date=None, end_date=None):
        """
        Download futures contract data for a stock.
        
        Args:
            symbol: Stock symbol
            expiry_month: Expiry month (e.g., 'JAN', 'FEB'). If None, uses current month.
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with futures OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        
        if end_date is None:
            end_date = datetime.now()
        
        # For now, download regular stock data as proxy for futures
        # In production, would use Kite API for actual futures instruments
        logger.info(f"Downloading futures data for {symbol}...")
        
        try:
            df = self.downloader.download_stock_data(symbol, start_date, end_date, 'day')
            
            if df is not None:
                # Save to futures directory
                self.downloader._save_data(df, symbol, 'futures_stocks', 'day')
                return df
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to download futures for {symbol}: {e}")
            return None
    
    def download_all_futures_stocks(self, start_date=None, end_date=None):
        """Download all futures-tradable stocks"""
        logger.info(f"Downloading all {len(FUTURES_STOCKS)} futures stocks...")
        
        return self.downloader.download_batch_futures_stocks(
            symbols=FUTURES_STOCKS,
            start_date=start_date,
            end_date=end_date
        )
