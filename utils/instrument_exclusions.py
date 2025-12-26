"""
Instrument exclusion filter for pair trading platform.
Filters out invalid, illiquid, and edge case instruments.
"""
import re
from utils.logger import logger

# Excluded suffixes (exact match)
EXCLUDED_SUFFIXES = ['-GS', '-SG', '-OLD']

# Excluded patterns (regex - suffix match)
EXCLUDED_PATTERNS = [
    r'-N[0-9A-Za-z]$',    # News/Notice issues
    r'-Y[0-9A-Za-z]$',    # Yield/Special issues  
    r'-Z[0-9A-Za-z]$',    # Special/Temporary issues
]


def is_excluded(symbol: str) -> bool:
    """
    Check if a symbol should be excluded from download.
    
    Args:
        symbol: Symbol name (e.g., 'INFY', 'NIFTY50-GS', 'ABC-N1')
        
    Returns:
        True if symbol is excluded, False otherwise
    """
    if not symbol:
        return False
    
    # Check exact suffix matches
    for suffix in EXCLUDED_SUFFIXES:
        if symbol.endswith(suffix):
            return True
    
    # Check regex patterns
    for pattern in EXCLUDED_PATTERNS:
        if re.search(pattern, symbol):
            return True
    
    return False


def get_exclusion_reason(symbol: str) -> str:
    """Get reason why a symbol is excluded."""
    if not symbol:
        return "Empty symbol"
    
    for suffix in EXCLUDED_SUFFIXES:
        if symbol.endswith(suffix):
            return f"Excluded suffix: {suffix}"
    
    for pattern in EXCLUDED_PATTERNS:
        if re.search(pattern, symbol):
            return f"Excluded pattern: {pattern}"
    
    return "Not excluded"


def filter_symbols(symbols: list) -> tuple:
    """
    Filter out excluded symbols from a list.
    
    Args:
        symbols: List of symbol names
        
    Returns:
        Tuple of (filtered_symbols, excluded_count, excluded_list)
    """
    if not symbols:
        return [], 0, []
    
    included = []
    excluded = []
    
    for symbol in symbols:
        if is_excluded(symbol):
            excluded.append(symbol)
        else:
            included.append(symbol)
    
    return included, len(excluded), excluded
