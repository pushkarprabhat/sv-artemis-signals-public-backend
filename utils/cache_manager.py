"""
Data Caching Module
Provides efficient caching for price data, scan results, and computed metrics
Reduces redundant data loads and improves performance
"""

import streamlit as st
import pandas as pd
import numpy as np
from functools import wraps
from typing import Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from config import BASE_DIR
from utils.logger import logger


class CacheConfig:
    """Cache configuration and management"""
    # TTL (Time To Live) for different cache types in seconds
    PRICE_DATA_TTL = 300  # 5 minutes for price data
    SCAN_RESULTS_TTL = 600  # 10 minutes for scan results
    GREEKS_TTL = 300  # 5 minutes for Greeks
    IV_DATA_TTL = 600  # 10 minutes for IV data
    BACKTEST_RESULTS_TTL = 3600  # 1 hour for backtest results
    
    # Cache size limits
    MAX_CACHED_DATAFRAMES = 50
    MAX_CACHED_SERIES = 100


def generate_cache_key(*args, **kwargs) -> str:
    """Generate unique cache key from function arguments"""
    key_data = {
        'args': str(args),
        'kwargs': str(sorted(kwargs.items()))
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


@st.cache_data(ttl=CacheConfig.PRICE_DATA_TTL)
def cached_load_price(symbol: str, tf: str = "day") -> Optional[pd.Series]:
    """
    Cache-enabled price data loader
    
    Args:
        symbol: Stock symbol
        tf: Timeframe (15min, 30min, 60min, day, week, month)
    
    Returns:
        pd.Series: Close prices indexed by date, or None if not found
    """
    from core.pairs import load_price
    
    try:
        price_data = load_price(symbol, tf)
        if price_data is not None:
            logger.debug(f"[CACHE] Loaded price data: {symbol} {tf} ({len(price_data)} rows)")
        return price_data
    except Exception as e:
        logger.error(f"[CACHE] Error loading price data for {symbol}: {e}")
        return None


@st.cache_data(ttl=CacheConfig.SCAN_RESULTS_TTL)
def cached_scan_all_strategies(
    tf: str = "day",
    p_value_threshold: Optional[float] = None,
    min_common: int = 100,
    capital_limit: Optional[float] = None,
    include_pairs: bool = True,
    include_strangle: bool = False,
    include_straddle: bool = False,
    include_volatility: bool = False,
    include_momentum: bool = False,
    include_mean_reversion: bool = False
) -> Optional[pd.DataFrame]:
    """
    Cache-enabled strategy scanning with all strategy options
    
    Args:
        tf: Timeframe
        p_value_threshold: P-value threshold for cointegration
        min_common: Minimum common data points
        capital_limit: Capital limit for filtering
        include_pairs: Include pair trading signals
        include_strangle: Include strangle options strategy
        include_straddle: Include straddle options strategy
        include_volatility: Include volatility-based signals
        include_momentum: Include momentum signals
        include_mean_reversion: Include mean reversion signals
    
    Returns:
        pd.DataFrame: Scan results with pairs and metrics
    """
    from core.pairs import scan_all_strategies
    
    try:
        results = scan_all_strategies(
            tf=tf,
            p_value_threshold=p_value_threshold,
            min_common=min_common,
            capital_limit=capital_limit,
            include_pairs=include_pairs,
            include_strangle=include_strangle,
            include_straddle=include_straddle,
            include_volatility=include_volatility,
            include_momentum=include_momentum,
            include_mean_reversion=include_mean_reversion
        )
        if results is not None and not results.empty:
            logger.debug(f"[CACHE] Scan results: {len(results)} pairs found")
        return results
    except Exception as e:
        logger.error(f"[CACHE] Error scanning strategies: {e}")
        return None


@st.cache_data(ttl=CacheConfig.GREEKS_TTL)
def cached_greeks(
    spot: float,
    strike: float,
    days_to_expiry: float,
    volatility: float,
    rate: float = 0.06,
    option_type: str = "CE"
) -> dict:
    """
    Cache-enabled Greeks calculation
    
    Args:
        spot: Current spot price
        strike: Strike price
        days_to_expiry: Days to expiration
        volatility: Implied volatility
        rate: Risk-free rate
        option_type: "CE" for call, "PE" for put
    
    Returns:
        dict: Greeks (delta, gamma, vega, theta, rho)
    """
    from core.greeks import black_scholes_greeks
    
    try:
        greeks = black_scholes_greeks(
            spot=spot,
            strike=strike,
            days_to_expiry=days_to_expiry,
            volatility=volatility,
            rate=rate,
            option_type=option_type
        )
        return greeks
    except Exception as e:
        logger.error(f"[CACHE] Error calculating Greeks: {e}")
        return {}


@st.cache_data(ttl=CacheConfig.IV_DATA_TTL)
def cached_iv_history(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Cache-enabled IV history loader
    
    Args:
        symbol: Stock symbol
        days: Number of days of history
    
    Returns:
        pd.DataFrame: IV history with dates and values
    """
    try:
        iv_file = BASE_DIR / "iv_history" / f"{symbol}_iv.parquet"
        if iv_file.exists():
            df = pd.read_parquet(iv_file)
            if len(df) > 0:
                # Keep only last N days
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[df['date'] >= datetime.now() - timedelta(days=days)]
            logger.debug(f"[CACHE] Loaded IV history: {symbol} ({len(df)} rows)")
            return df
    except Exception as e:
        logger.error(f"[CACHE] Error loading IV history for {symbol}: {e}")
    
    return None


@st.cache_data(ttl=300)
def cached_universe_data() -> Optional[pd.DataFrame]:
    """
    Cache-enabled universe data loader
    
    Returns:
        pd.DataFrame: Universe data with symbols and metadata
    """
    try:
        from universe.symbols import load_universe
        universe = load_universe()
        logger.debug(f"[CACHE] Loaded universe: {len(universe)} symbols")
        return universe
    except Exception as e:
        logger.error(f"[CACHE] Error loading universe: {e}")
        return None


def cache_pair_data(symbol1: str, symbol2: str, tf: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Load and cache price data for a pair
    
    Args:
        symbol1: First symbol
        symbol2: Second symbol
        tf: Timeframe
    
    Returns:
        Tuple: (price1, price2) cached series
    """
    price1 = cached_load_price(symbol1, tf)
    price2 = cached_load_price(symbol2, tf)
    return price1, price2


def clear_all_caches():
    """Clear all Streamlit caches"""
    try:
        st.cache_data.clear()
        logger.info("[CACHE] All caches cleared")
    except Exception as e:
        logger.error(f"[CACHE] Error clearing caches: {e}")


def clear_price_cache(symbol: Optional[str] = None):
    """Clear price data cache (all or specific symbol)"""
    try:
        st.cache_data.clear()  # Note: Streamlit doesn't provide selective cache clearing
        logger.info(f"[CACHE] Price cache cleared for {symbol if symbol else 'all symbols'}")
    except Exception as e:
        logger.error(f"[CACHE] Error clearing price cache: {e}")


def get_cache_stats() -> dict:
    """Get cache statistics and status"""
    return {
        'price_data_ttl': CacheConfig.PRICE_DATA_TTL,
        'scan_results_ttl': CacheConfig.SCAN_RESULTS_TTL,
        'greeks_ttl': CacheConfig.GREEKS_TTL,
        'iv_data_ttl': CacheConfig.IV_DATA_TTL,
        'backtest_results_ttl': CacheConfig.BACKTEST_RESULTS_TTL,
        'timestamp': datetime.now().isoformat()
    }


class SessionCache:
    """Session-based cache for temporary data (within single user session)"""
    
    @staticmethod
    def set(key: str, value: Any, ttl_seconds: int = 3600):
        """Store value in session cache"""
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
        
        st.session_state.cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl_seconds
        }
        logger.debug(f"[SESSION_CACHE] Stored: {key}")
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Retrieve value from session cache"""
        if 'cache' not in st.session_state:
            return default
        
        if key not in st.session_state.cache:
            return default
        
        cached_item = st.session_state.cache[key]
        elapsed = (datetime.now() - cached_item['timestamp']).total_seconds()
        
        # Check if expired
        if elapsed > cached_item['ttl']:
            del st.session_state.cache[key]
            logger.debug(f"[SESSION_CACHE] Expired: {key}")
            return default
        
        logger.debug(f"[SESSION_CACHE] Retrieved: {key} (age: {elapsed:.1f}s)")
        return cached_item['value']
    
    @staticmethod
    def clear(key: Optional[str] = None):
        """Clear session cache"""
        if 'cache' not in st.session_state:
            return
        
        if key:
            if key in st.session_state.cache:
                del st.session_state.cache[key]
                logger.debug(f"[SESSION_CACHE] Cleared: {key}")
        else:
            st.session_state.cache.clear()
            logger.debug("[SESSION_CACHE] All cleared")
    
    @staticmethod
    def stats() -> dict:
        """Get session cache statistics"""
        if 'cache' not in st.session_state:
            return {'count': 0, 'items': []}
        
        items = []
        for key, data in st.session_state.cache.items():
            age = (datetime.now() - data['timestamp']).total_seconds()
            items.append({
                'key': key,
                'age_seconds': age,
                'ttl_seconds': data['ttl'],
                'expired': age > data['ttl']
            })
        
        return {
            'count': len(items),
            'items': items,
            'timestamp': datetime.now().isoformat()
        }


# Performance monitoring
class CacheMetrics:
    """Track cache performance metrics"""
    
    @staticmethod
    def get_hit_rate() -> dict:
        """Get cache hit rate statistics"""
        if 'cache_metrics' not in st.session_state:
            st.session_state.cache_metrics = {
                'hits': 0,
                'misses': 0,
                'clears': 0
            }
        
        metrics = st.session_state.cache_metrics
        total = metrics['hits'] + metrics['misses']
        hit_rate = (metrics['hits'] / total * 100) if total > 0 else 0
        
        return {
            'total_requests': total,
            'hits': metrics['hits'],
            'misses': metrics['misses'],
            'hit_rate': hit_rate,
            'clears': metrics['clears']
        }
    
    @staticmethod
    def record_hit():
        """Record cache hit"""
        if 'cache_metrics' not in st.session_state:
            st.session_state.cache_metrics = {'hits': 0, 'misses': 0, 'clears': 0}
        st.session_state.cache_metrics['hits'] += 1
    
    @staticmethod
    def record_miss():
        """Record cache miss"""
        if 'cache_metrics' not in st.session_state:
            st.session_state.cache_metrics = {'hits': 0, 'misses': 0, 'clears': 0}
        st.session_state.cache_metrics['misses'] += 1
    
    @staticmethod
    def record_clear():
        """Record cache clear"""
        if 'cache_metrics' not in st.session_state:
            st.session_state.cache_metrics = {'hits': 0, 'misses': 0, 'clears': 0}
        st.session_state.cache_metrics['clears'] += 1
