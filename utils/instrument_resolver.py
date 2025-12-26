"""
Clean Instrument Resolution Module
================================
Single source of truth for instrument lookups.
Maps: tradingsymbol → (instrument_token, display_name, underlying_symbol)

Usage:
    resolver = InstrumentResolver()
    info = resolver.get_instrument_info('RELIANCE')
    # Returns: {'symbol': 'RELIANCE', 'token': 738561, 'display_name': 'Reliance Industries Limited', 'underlying_symbol': 'RELIANCE'}

DERIVATION LOGIC (VERIFIED INTACT):
- UNDERLYING_SYMBOL: Identifies the underlying asset
  * For EQ: Copy tradingsymbol
  * For FUT/CE/PE: Copy name field
- DISPLAY_NAME: User-friendly name with expiry info
  * For EQ: Just the name
  * For FUT: Name + expiry + 'FUT' + M/W
  * For CE/PE: Name + expiry + strike + type + M/W
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache
from config import BASE_DIR
from utils.logger import logger


class InstrumentResolver:
    """Expert-level instrument resolution with caching and clean separation of concerns"""
    
    def __init__(self):
        """Initialize with lazy loading of instruments"""
        self._instruments_df: Optional[pd.DataFrame] = None
        self._instruments_path = Path(BASE_DIR).parent / "universe" / "app" / "app_kite_universe.csv"
        self._load_instruments()
    
    def _load_instruments(self) -> None:
        """Load instruments once on initialization"""
        if not self._instruments_path.exists():
            logger.error(f"[INSTRUMENT] File not found: {self._instruments_path}")
            self._instruments_df = pd.DataFrame()  # Empty dataframe to prevent crashes
            return
        
        try:
            self._instruments_df = pd.read_csv(
                self._instruments_path,
                dtype={
                    'instrument_token': 'int64',
                    'tradingsymbol': 'str',
                    'name': 'str',
                    'DISPLAY_NAME': 'str',
                    'UNDERLYING_SYMBOL': 'str',
                    'exchange': 'str',
                    'segment': 'str'
                },
                usecols=['instrument_token', 'tradingsymbol', 'name', 'DISPLAY_NAME', 'UNDERLYING_SYMBOL', 'exchange', 'segment']
            )
            logger.info(f"[INSTRUMENT] Loaded {len(self._instruments_df)} instruments from app_kite_universe.csv")
        except Exception as e:
            logger.error(f"[INSTRUMENT] Failed to load instruments: {str(e)}")
            self._instruments_df = pd.DataFrame()
    
    def get_instrument_info(self, symbol: str, exchange: str = 'NSE', segment: str = 'NSE') -> Optional[Dict]:
        """
        Get complete instrument info: token + display_name
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            exchange: Exchange code (default: 'NSE')
            segment: Segment code (default: 'NSE')
        
        Returns:
            Dict with keys: symbol, token, name, display_name
            Returns None if not found
        """
        if self._instruments_df.empty:
            return None
        
        try:
            # Find matching instrument
            match = self._instruments_df[
                (self._instruments_df['tradingsymbol'] == symbol) &
                (self._instruments_df['exchange'] == exchange) &
                (self._instruments_df['segment'] == segment)
            ]
            
            if match.empty:
                # Try NSE only without segment filter
                match = self._instruments_df[
                    (self._instruments_df['tradingsymbol'] == symbol) &
                    (self._instruments_df['exchange'] == exchange)
                ]
            
            if match.empty:
                logger.debug(f"[INSTRUMENT] Symbol '{symbol}' not found")
                return None
            
            row = match.iloc[0]
            return {
                'symbol': symbol,
                'token': int(row['instrument_token']),
                'name': str(row.get('name', '')),
                'display_name': str(row.get('DISPLAY_NAME', '')),
                'underlying_symbol': str(row.get('UNDERLYING_SYMBOL', symbol)),  # Derivation logic intact
                'exchange': str(row.get('exchange', 'NSE')),
                'segment': str(row.get('segment', 'NSE'))
            }
        
        except Exception as e:
            logger.error(f"[INSTRUMENT] Error looking up '{symbol}': {str(e)}")
            return None
    
    def get_token(self, symbol: str) -> Optional[int]:
        """Get instrument token only"""
        info = self.get_instrument_info(symbol)
        return info['token'] if info else None
    
    def get_display_name(self, symbol: str) -> Optional[str]:
        """Get display_name only"""
        info = self.get_instrument_info(symbol)
        return info['display_name'] if info else None
    
    def get_underlying_symbol(self, symbol: str) -> Optional[str]:
        """Get underlying_symbol only (derivation logic intact)"""
        info = self.get_instrument_info(symbol)
        return info['underlying_symbol'] if info else None
    
    def is_valid(self, symbol: str) -> bool:
        """Check if symbol exists in universe"""
        return self.get_instrument_info(symbol) is not None


# Global singleton instance
_resolver_instance: Optional[InstrumentResolver] = None


def get_instrument_resolver() -> InstrumentResolver:
    """Get global instrument resolver instance"""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = InstrumentResolver()
    return _resolver_instance
