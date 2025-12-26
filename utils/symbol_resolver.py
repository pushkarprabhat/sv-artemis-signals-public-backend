"""
utils/symbol_resolver.py — Symbol to Display Name Resolution
Provides caching mechanism to look up human-readable names for trading symbols
"""

import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
import threading
from utils.logger import logger
from core.universe_manager import get_universe_manager

# Cache for symbol -> name mappings
_SYMBOL_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_INITIALIZED = False


def initialize_cache() -> None:
    """Initialize symbol-to-name cache from universe"""
    global _SYMBOL_CACHE, _CACHE_INITIALIZED
    
    with _CACHE_LOCK:
        if _CACHE_INITIALIZED and len(_SYMBOL_CACHE) > 0:
            return  # Already initialized
        
        try:
            um = get_universe_manager()
            df_universe = um.get_universe()
            
            if df_universe is not None and len(df_universe) > 0:
                # Build symbol -> name mapping
                if 'Symbol' in df_universe.columns and 'Name' in df_universe.columns:
                    # Create mapping, handling duplicates by taking first occurrence
                    symbol_map = df_universe.drop_duplicates(subset=['Symbol'])[['Symbol', 'Name']]
                    _SYMBOL_CACHE.update(dict(zip(symbol_map['Symbol'], symbol_map['Name'])))
                    logger.debug(f"[SYMBOL-RESOLVER] Initialized cache with {len(_SYMBOL_CACHE)} symbol-name mappings")
                    _CACHE_INITIALIZED = True
                else:
                    logger.warning("[SYMBOL-RESOLVER] Universe missing Symbol or Name columns")
            else:
                logger.warning("[SYMBOL-RESOLVER] Could not load universe for cache initialization")
                
        except Exception as e:
            logger.error(f"[SYMBOL-RESOLVER] Failed to initialize cache: {e}")


def get_symbol_name(symbol: str) -> str:
    """
    Get human-readable name for a trading symbol
    
    Args:
        symbol: Trading symbol (e.g., 'RELIANCE', 'INFY')
    
    Returns:
        Display name (e.g., 'Reliance Industries Limited') or symbol itself if not found
    """
    global _SYMBOL_CACHE, _CACHE_INITIALIZED
    
    # Initialize cache if needed
    if not _CACHE_INITIALIZED:
        initialize_cache()
    
    with _CACHE_LOCK:
        # Return from cache if available
        if symbol in _SYMBOL_CACHE:
            return _SYMBOL_CACHE[symbol]
        
        # Not in cache, try to look up
        try:
            um = get_universe_manager()
            df_universe = um.get_universe()
            
            if df_universe is not None:
                matching = df_universe[df_universe['Symbol'] == symbol]
                if len(matching) > 0:
                    name = matching.iloc[0]['Name']
                    _SYMBOL_CACHE[symbol] = name
                    return name
        except Exception as e:
            logger.debug(f"[SYMBOL-RESOLVER] Could not resolve name for {symbol}: {e}")
    
    # Fallback to symbol itself
    return symbol


def get_symbol_with_name(symbol: str) -> str:
    """
    Get formatted string with symbol and name
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Formatted string like "RELIANCE (Reliance Industries Limited)"
    """
    name = get_symbol_name(symbol)
    if name == symbol:
        return symbol  # No name available
    return f"{symbol} ({name})"


def resolve_symbols_batch(symbols: List[str]) -> Dict[str, str]:
    """
    Resolve multiple symbols to names efficiently
    
    Args:
        symbols: List of trading symbols
    
    Returns:
        Dict mapping symbol -> display name
    """
    global _SYMBOL_CACHE, _CACHE_INITIALIZED
    
    # Initialize cache if needed
    if not _CACHE_INITIALIZED:
        initialize_cache()
    
    # Quick path: check what's in cache
    result = {}
    missing = []
    
    with _CACHE_LOCK:
        for symbol in symbols:
            if symbol in _SYMBOL_CACHE:
                result[symbol] = _SYMBOL_CACHE[symbol]
            else:
                missing.append(symbol)
    
    # Batch resolve missing symbols
    if missing:
        try:
            um = get_universe_manager()
            df_universe = um.get_universe()
            
            if df_universe is not None:
                # Filter for only missing symbols
                df_missing = df_universe[df_universe['Symbol'].isin(missing)]
                df_missing = df_missing.drop_duplicates(subset=['Symbol'])
                
                for _, row in df_missing.iterrows():
                    symbol = row['Symbol']
                    name = row['Name']
                    result[symbol] = name
                    _SYMBOL_CACHE[symbol] = name  # Update cache
        except Exception as e:
            logger.debug(f"[SYMBOL-RESOLVER] Batch resolve failed: {e}")
    
    # For any still missing, use the symbol itself
    for symbol in symbols:
        if symbol not in result:
            result[symbol] = symbol
    
    return result


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about symbol cache"""
    with _CACHE_LOCK:
        return {
            'cached_symbols': len(_SYMBOL_CACHE),
            'initialized': _CACHE_INITIALIZED
        }


def clear_cache() -> None:
    """Clear the symbol cache (useful for testing or refresh)"""
    global _SYMBOL_CACHE, _CACHE_INITIALIZED
    
    with _CACHE_LOCK:
        _SYMBOL_CACHE.clear()
        _CACHE_INITIALIZED = False
        logger.info("[SYMBOL-RESOLVER] Cache cleared")
