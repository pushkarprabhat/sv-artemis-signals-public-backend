"""
Instrument Data Checker Utility

Scans instruments to identify which ones have no data or incomplete data.
Checks all available timeframes (1min, 5min, 15min, 30min, 60min, day).
Caches results for 1 hour to avoid repeated scanning.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class InstrumentDataChecker:
    """
    Checks data availability for instruments across all timeframes.
    
    Features:
    - Scan all timeframes for instrument
    - Find instruments with no data
    - Get data statistics per instrument
    - Cache results for 1 hour
    - Identify partial data (some timeframes missing)
    """
    
    # Standard timeframes to check
    TIMEFRAMES = ['1min', '5min', '15min', '30min', '60min', 'day']
    
    def __init__(self, data_base_path: str = "data", cache_ttl_minutes: int = 60):
        """
        Initialize data checker.
        
        Args:
            data_base_path: Base path for data files
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.data_base_path = data_base_path
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cache: Dict = {}
        self.cache_times: Dict = {}
        
        logger.info(f"InstrumentDataChecker initialized. TTL: {cache_ttl_minutes} minutes")
    
    def _get_instrument_path(self, symbol: str, timeframe: str) -> str:
        """
        Get path to instrument data file.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe (e.g., '1min', '5min', 'day')
        
        Returns:
            Path to Parquet file
        """
        # Handle different timeframe naming conventions
        tf_map = {
            '1min': '1minute',
            '5min': '5minute',
            '15min': '15minute',
            '30min': '30minute',
            '60min': '60minute',
            'day': 'day'
        }
        
        tf_folder = tf_map.get(timeframe, timeframe)
        
        # Try multiple possible paths
        possible_paths = [
            f"{self.data_base_path}/{tf_folder}/{symbol}.parquet",
            f"{self.data_base_path}/{timeframe}/{symbol}.parquet",
            f"{self.data_base_path}/{tf_folder}/{symbol}.pq",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return possible_paths[0]  # Return default even if doesn't exist
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """
        Check if cached result is still valid.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            True if cache is valid, False if expired or not cached
        """
        if symbol not in self.cache_times:
            return False
        
        if datetime.now() - self.cache_times[symbol] > self.cache_ttl:
            # Cache expired
            del self.cache[symbol]
            del self.cache_times[symbol]
            return False
        
        return True
    
    def check_instrument(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        Check data availability for single instrument across all timeframes.
        
        Args:
            symbol: Instrument symbol
            use_cache: Whether to use cached results
        
        Returns:
            Dictionary with data status:
            {
                'symbol': 'AAPL',
                'has_data': True/False,
                'data_count': 12345,
                'timeframes': {
                    '1min': {'exists': True, 'rows': 1000},
                    '5min': {'exists': False, 'rows': 0},
                    ...
                },
                'missing_timeframes': ['5min', '15min'],
                'last_checked': datetime,
                'status': 'complete' | 'partial' | 'none'
            }
        """
        # Check cache
        if use_cache and self._is_cache_valid(symbol):
            return self.cache[symbol]
        
        result = {
            'symbol': symbol,
            'has_data': False,
            'data_count': 0,
            'timeframes': {},
            'missing_timeframes': [],
            'last_checked': datetime.now().isoformat(),
            'status': 'none'  # none | partial | complete
        }
        
        total_rows = 0
        timeframes_with_data = 0
        
        for timeframe in self.TIMEFRAMES:
            tf_result = self._check_timeframe(symbol, timeframe)
            result['timeframes'][timeframe] = tf_result
            
            if tf_result['exists']:
                timeframes_with_data += 1
                total_rows += tf_result['rows']
            else:
                result['missing_timeframes'].append(timeframe)
        
        # Determine status
        if timeframes_with_data == 0:
            result['status'] = 'none'
            result['has_data'] = False
        elif timeframes_with_data < len(self.TIMEFRAMES):
            result['status'] = 'partial'
            result['has_data'] = True
        else:
            result['status'] = 'complete'
            result['has_data'] = True
        
        result['data_count'] = total_rows
        
        # Cache result
        self.cache[symbol] = result
        self.cache_times[symbol] = datetime.now()
        
        return result
    
    def _check_timeframe(self, symbol: str, timeframe: str) -> Dict:
        """
        Check if data exists for specific timeframe.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe to check
        
        Returns:
            Dictionary: {'exists': bool, 'rows': int, 'size_mb': float}
        """
        filepath = self._get_instrument_path(symbol, timeframe)
        
        if not os.path.exists(filepath):
            return {
                'exists': False,
                'rows': 0,
                'size_mb': 0.0,
                'error': 'File not found'
            }
        
        try:
            # Get file size
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / (1024 * 1024)
            
            # Try to read Parquet file and count rows
            try:
                df = pd.read_parquet(filepath)
                row_count = len(df)
                
                return {
                    'exists': True,
                    'rows': row_count,
                    'size_mb': size_mb,
                    'error': None
                }
            except Exception as e:
                # File exists but can't read (corrupted?)
                logger.warning(f"Error reading {filepath}: {e}")
                return {
                    'exists': True,
                    'rows': 0,
                    'size_mb': size_mb,
                    'error': 'File corrupted or unreadable'
                }
        except Exception as e:
            logger.error(f"Error checking {filepath}: {e}")
            return {
                'exists': False,
                'rows': 0,
                'size_mb': 0.0,
                'error': str(e)
            }
    
    def find_missing_data_instruments(self, symbols: List[str] = None) -> List[Dict]:
        """
        Find instruments with no data at all.
        
        Args:
            symbols: List of symbols to check. If None, scan all in data folder.
        
        Returns:
            List of instruments with no data:
            [
                {'symbol': 'AAPL', 'reason': 'No Data', 'checked_date': '...'},
                ...
            ]
        """
        if symbols is None:
            # Auto-discover symbols from data folder
            symbols = self._discover_symbols()
        
        missing = []
        
        for symbol in symbols:
            result = self.check_instrument(symbol)
            
            if result['status'] == 'none':
                missing.append({
                    'symbol': symbol,
                    'reason': 'No Data',
                    'checked_date': result['last_checked'],
                    'data_points': 0
                })
        
        logger.info(f"Found {len(missing)} instruments with no data out of {len(symbols)}")
        return missing
    
    def find_partial_data_instruments(self, symbols: List[str] = None) -> List[Dict]:
        """
        Find instruments with partial data (some timeframes missing).
        
        Args:
            symbols: List of symbols to check.
        
        Returns:
            List of instruments with partial data
        """
        if symbols is None:
            symbols = self._discover_symbols()
        
        partial = []
        
        for symbol in symbols:
            result = self.check_instrument(symbol)
            
            if result['status'] == 'partial':
                partial.append({
                    'symbol': symbol,
                    'reason': f"Missing timeframes: {', '.join(result['missing_timeframes'])}",
                    'checked_date': result['last_checked'],
                    'data_points': result['data_count'],
                    'missing_timeframes': result['missing_timeframes']
                })
        
        return partial
    
    def _discover_symbols(self) -> List[str]:
        """
        Discover all symbols by scanning data folder.
        
        Returns:
            List of discovered symbols
        """
        symbols = set()
        
        # Scan through all timeframe folders
        for timeframe in self.TIMEFRAMES:
            tf_map = {
                '1min': '1minute',
                '5min': '5minute',
                '15min': '15minute',
                '30min': '30minute',
                '60min': '60minute',
                'day': 'day'
            }
            
            tf_folder = tf_map.get(timeframe, timeframe)
            folder_path = f"{self.data_base_path}/{tf_folder}"
            
            if os.path.exists(folder_path):
                try:
                    files = os.listdir(folder_path)
                    for file in files:
                        if file.endswith('.parquet') or file.endswith('.pq'):
                            symbol = file.replace('.parquet', '').replace('.pq', '')
                            symbols.add(symbol)
                except Exception as e:
                    logger.warning(f"Error scanning {folder_path}: {e}")
        
        logger.info(f"Discovered {len(symbols)} unique symbols")
        return sorted(list(symbols))
    
    def get_data_statistics(self, symbol: str) -> Dict:
        """
        Get data statistics for an instrument.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Dictionary with statistics
        """
        result = self.check_instrument(symbol)
        
        stats = {
            'symbol': symbol,
            'has_data': result['has_data'],
            'status': result['status'],
            'total_data_points': result['data_count'],
            'timeframes_available': len(self.TIMEFRAMES) - len(result['missing_timeframes']),
            'timeframes_total': len(self.TIMEFRAMES),
            'data_coverage': (
                (len(self.TIMEFRAMES) - len(result['missing_timeframes'])) / len(self.TIMEFRAMES) * 100
                if len(self.TIMEFRAMES) > 0 else 0
            ),
            'timeframe_details': result['timeframes'],
            'checked': result['last_checked']
        }
        
        return stats
    
    def clear_cache(self, symbol: str = None) -> None:
        """
        Clear cache (all or specific symbol).
        
        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol is None:
            self.cache.clear()
            self.cache_times.clear()
            logger.info("Cleared entire cache")
        else:
            if symbol in self.cache:
                del self.cache[symbol]
                del self.cache_times[symbol]
                logger.info(f"Cleared cache for {symbol}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cached_symbols': len(self.cache),
            'cache_size': len(self.cache),
            'ttl_minutes': self.cache_ttl.total_seconds() / 60
        }
    
    def scan_all_instruments(self, verbose: bool = False) -> Dict:
        """
        Scan all instruments and return comprehensive report.
        
        Args:
            verbose: Print progress
        
        Returns:
            Dictionary with scan results
        """
        symbols = self._discover_symbols()
        
        report = {
            'scan_time': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'complete_data': [],
            'partial_data': [],
            'no_data': [],
            'summary': {}
        }
        
        for i, symbol in enumerate(symbols):
            if verbose and (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(symbols)}")
            
            result = self.check_instrument(symbol)
            
            if result['status'] == 'complete':
                report['complete_data'].append(symbol)
            elif result['status'] == 'partial':
                report['partial_data'].append(symbol)
            else:
                report['no_data'].append(symbol)
        
        report['summary'] = {
            'complete': len(report['complete_data']),
            'partial': len(report['partial_data']),
            'no_data': len(report['no_data']),
            'complete_percent': (len(report['complete_data']) / len(symbols) * 100) if symbols else 0,
            'partial_percent': (len(report['partial_data']) / len(symbols) * 100) if symbols else 0,
            'no_data_percent': (len(report['no_data']) / len(symbols) * 100) if symbols else 0
        }
        
        logger.info(f"Scan complete. Complete: {len(report['complete_data'])}, "
                   f"Partial: {len(report['partial_data'])}, No Data: {len(report['no_data'])}")
        
        return report


# Global data checker instance
_data_checker: Optional[InstrumentDataChecker] = None


def get_data_checker() -> InstrumentDataChecker:
    """
    Get or create global data checker instance.
    
    Returns:
        InstrumentDataChecker instance
    """
    global _data_checker
    
    if _data_checker is None:
        _data_checker = InstrumentDataChecker()
    
    return _data_checker


def initialize_data_checker(data_base_path: str = "data") -> InstrumentDataChecker:
    """
    Initialize data checker with custom path.
    
    Args:
        data_base_path: Base path for data files
    
    Returns:
        InstrumentDataChecker instance
    """
    global _data_checker
    _data_checker = InstrumentDataChecker(data_base_path=data_base_path)
    return _data_checker
