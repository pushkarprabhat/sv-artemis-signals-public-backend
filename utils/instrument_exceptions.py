#!/usr/bin/env python3
"""
Instrument Exception Management
Tracks instruments that failed to download (no data from API)
Persistent storage and checking to avoid re-attempting failed instruments
"""

import re
from pathlib import Path
from typing import Set, Tuple, List
from config import get_download_dir
from utils.logger import logger


class InstrumentExceptionManager:
    """Manages persistent list of failed instruments"""
    
    def __init__(self):
        """Initialize exception manager and load existing exceptions"""
        self.exception_file = Path(get_download_dir('NSE', 'day')).parent / 'failed_instruments.txt'
        self.exceptions: Set[str] = set()
        self.load_exceptions()
    
    def load_exceptions(self) -> Set[str]:
        """Load exceptions from file"""
        if not self.exception_file.exists():
            logger.debug("[EXCEPTIONS] No existing exception file found")
            return set()
        
        try:
            with open(self.exception_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out comments and empty lines
            exceptions = {
                line.strip() 
                for line in lines 
                if line.strip() and not line.strip().startswith('#')
            }
            
            self.exceptions = exceptions
            logger.info(f"[EXCEPTIONS] Loaded {len(self.exceptions)} failed instruments from {self.exception_file.name}")
            return exceptions
        
        except Exception as e:
            logger.error(f"[EXCEPTIONS] Error loading exceptions: {e}")
            return set()
    
    def save_exceptions(self, exceptions: Set[str] = None) -> bool:
        """Save exceptions to file"""
        if exceptions is None:
            exceptions = self.exceptions
        
        try:
            with open(self.exception_file, 'w') as f:
                f.write('# Instruments that failed to download (no data returned from API)\n')
                f.write('# These are automatically skipped in future downloads\n')
                f.write('# Format: One symbol per line\n')
                f.write('# Last updated: ' + str(Path.cwd()) + '\n')
                f.write('\n')
                f.write('\n'.join(sorted(exceptions)))
            
            logger.info(f"[EXCEPTIONS] Saved {len(exceptions)} failed instruments to {self.exception_file.name}")
            return True
        
        except Exception as e:
            logger.error(f"[EXCEPTIONS] Error saving exceptions: {e}")
            return False
    
    def is_exception(self, symbol: str) -> bool:
        """Check if symbol is in exception list"""
        return symbol in self.exceptions
    
    def add_exception(self, symbol: str) -> None:
        """Add symbol to exceptions"""
        if symbol not in self.exceptions:
            self.exceptions.add(symbol)
            logger.debug(f"[EXCEPTIONS] Added {symbol} to exception list")
    
    def add_exceptions(self, symbols: List[str]) -> int:
        """Add multiple symbols to exceptions. Returns count added."""
        added = 0
        for symbol in symbols:
            if symbol not in self.exceptions:
                self.exceptions.add(symbol)
                added += 1
        
        if added > 0:
            logger.info(f"[EXCEPTIONS] Added {added} symbols to exception list")
        
        return added
    
    def get_exceptions(self) -> Set[str]:
        """Get all exceptions"""
        return self.exceptions.copy()
    
    def filter_symbols(self, symbols: List[str]) -> Tuple[List[str], int, List[str]]:
        """
        Filter out exception symbols
        Returns: (filtered_symbols, exception_count, exception_list)
        """
        exceptions_found = []
        filtered = []
        
        for symbol in symbols:
            if self.is_exception(symbol):
                exceptions_found.append(symbol)
            else:
                filtered.append(symbol)
        
        return (filtered, len(exceptions_found), exceptions_found)
    
    def clear_exceptions(self) -> None:
        """Clear all exceptions"""
        count = len(self.exceptions)
        self.exceptions.clear()
        logger.warning(f"[EXCEPTIONS] Cleared {count} exceptions")
    
    def get_exception_reason(self, symbol: str) -> str:
        """Get reason why symbol is in exception list"""
        if symbol in self.exceptions:
            return "No data returned from Kite API (previous download failure)"
        return "Not in exception list"


# Global instance
_exception_manager = None


def get_exception_manager() -> InstrumentExceptionManager:
    """Get or create singleton exception manager"""
    global _exception_manager
    if _exception_manager is None:
        _exception_manager = InstrumentExceptionManager()
    return _exception_manager


def is_exception(symbol: str) -> bool:
    """Quick check if symbol is in exception list"""
    return get_exception_manager().is_exception(symbol)


def filter_exceptions(symbols: List[str]) -> Tuple[List[str], int, List[str]]:
    """
    Filter out exception symbols from list
    Returns: (filtered_symbols, exception_count, exception_list)
    """
    manager = get_exception_manager()
    return manager.filter_symbols(symbols)


def add_to_exceptions(symbols: List[str]) -> None:
    """Add symbols to exception list"""
    manager = get_exception_manager()
    manager.add_exceptions(symbols)
    manager.save_exceptions()


def save_exceptions() -> None:
    """Save exceptions to file"""
    get_exception_manager().save_exceptions()
