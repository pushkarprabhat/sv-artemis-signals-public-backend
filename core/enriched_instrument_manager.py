#!/usr/bin/env python3
"""
Enriched Instrument Download Manager
Integrates app_kite_universe.csv (enriched instruments) with download system
Handles:
- Loading enriched instrument master (app_kite_universe.csv)
- Tracking expiry dates (monthly/weekly) by exchange
- Managing symbol lists per exchange with proper filtering
- Coordinating downloads with manual UI buttons and automatic daily jobs
- Respecting existing data accuracy mechanisms

Key Integration Points:
- Uses UNDERLYING_SYMBOL for data organization
- Uses DISPLAY_NAME for user-friendly reporting
- Uses SYMBOL for API calls
- Tracks expiry dates (monthly/weekly) per exchange
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import logging
from config import BASE_DIR

logger = logging.getLogger(__name__)

# Paths
ENRICHED_INSTRUMENTS_PATH = BASE_DIR.parent / "universe" / "app" / "app_kite_universe.csv"
EXCHANGE_EXPIRY_CACHE = BASE_DIR.parent / "cache" / "exchange_expiry_cache.json"
INSTRUMENT_DOWNLOAD_STATUS = BASE_DIR.parent / "cache" / "instrument_download_status.json"


class EnrichedInstrumentManager:
    """
    Manages enriched instrument master (app_kite_universe.csv)
    Provides filtered lists for downloads by exchange
    Tracks expiry dates (monthly/weekly) for proper scheduling
    """
    
    def __init__(self):
        """Initialize enriched instrument manager"""
        self.lock = threading.RLock()
        self._enriched_df: Optional[pd.DataFrame] = None
        self._exchange_instruments: Dict[str, List[Dict]] = {}
        self._expiry_tracker: Dict[str, Dict] = {}
        self._last_load_time: Optional[datetime] = None
        
        # Ensure cache directory exists
        EXCHANGE_EXPIRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load enriched instruments on init
        self._load_enriched_instruments()
        self._load_expiry_cache()
    
    def _load_enriched_instruments(self) -> bool:
        """
        Load enriched instrument master from CSV
        Returns True if successful, False if file not found
        """
        with self.lock:
            try:
                if not ENRICHED_INSTRUMENTS_PATH.exists():
                    logger.warning(f"Enriched instruments file not found: {ENRICHED_INSTRUMENTS_PATH}")
                    logger.info("Run: python scripts/instrument_master_pipeline.py --all")
                    return False
                
                self._enriched_df = pd.read_csv(ENRICHED_INSTRUMENTS_PATH)
                self._last_load_time = datetime.now()
                
                logger.info(f"Loaded {len(self._enriched_df)} enriched instruments")
                logger.info(f"Columns: {list(self._enriched_df.columns)}")
                
                # Build exchange mappings
                self._build_exchange_mappings()
                
                return True
                
            except Exception as e:
                logger.error(f"Error loading enriched instruments: {e}")
                return False
    
    def _build_exchange_mappings(self) -> None:
        """
        Build mappings of instruments by exchange
        Separates by instrument type (EQ, FUT, CE, PE)
        Tracks expiry information
        """
        try:
            if self._enriched_df is None:
                return
            
            # Clear existing mappings
            self._exchange_instruments = {}
            self._expiry_tracker = {}
            
            # Group by exchange
            for exchange, group_df in self._enriched_df.groupby('exchange'):
                self._exchange_instruments[exchange] = []
                
                if exchange not in self._expiry_tracker:
                    self._expiry_tracker[exchange] = {
                        'monthly': [],
                        'weekly': [],
                        'no_expiry': [],
                        'total_count': 0,
                        'last_updated': datetime.now().isoformat()
                    }
                
                # Process each instrument in exchange
                for idx, row in group_df.iterrows():
                    instrument_info = {
                        'tradingsymbol': row['tradingsymbol'],
                        'underlying_symbol': row['UNDERLYING_SYMBOL'],
                        'display_name': row['DISPLAY_NAME'],
                        'instrument_type': row.get('instrument_type', 'EQ'),
                        'expiry': row.get('expiry'),
                        'strike': row.get('strike'),
                        'instrument_token': row.get('instrument_token'),
                    }
                    
                    self._exchange_instruments[exchange].append(instrument_info)
                    
                    # Track expiry type (monthly/weekly/no expiry)
                    if pd.isna(row.get('expiry')):
                        self._expiry_tracker[exchange]['no_expiry'].append(row['tradingsymbol'])
                    else:
                        # Determine if monthly or weekly from DISPLAY_NAME
                        display_name = row['DISPLAY_NAME']
                        if ' M' in display_name:
                            self._expiry_tracker[exchange]['monthly'].append({
                                'symbol': row['tradingsymbol'],
                                'expiry': row['expiry'],
                                'display': display_name
                            })
                        elif ' W' in display_name:
                            self._expiry_tracker[exchange]['weekly'].append({
                                'symbol': row['tradingsymbol'],
                                'expiry': row['expiry'],
                                'display': display_name
                            })
                
                # Update count
                self._expiry_tracker[exchange]['total_count'] = len(self._exchange_instruments[exchange])
                
                logger.info(
                    f"[{exchange}] {len(self._exchange_instruments[exchange])} instruments | "
                    f"Monthly: {len(self._expiry_tracker[exchange]['monthly'])} | "
                    f"Weekly: {len(self._expiry_tracker[exchange]['weekly'])} | "
                    f"No Expiry: {len(self._expiry_tracker[exchange]['no_expiry'])}"
                )
            
            # Save expiry tracker
            self._save_expiry_cache()
            
        except Exception as e:
            logger.error(f"Error building exchange mappings: {e}")
    
    def _load_expiry_cache(self) -> None:
        """Load cached expiry information"""
        try:
            if EXCHANGE_EXPIRY_CACHE.exists():
                with open(EXCHANGE_EXPIRY_CACHE, 'r') as f:
                    self._expiry_tracker = json.load(f)
                logger.info(f"Loaded expiry cache from {EXCHANGE_EXPIRY_CACHE}")
        except Exception as e:
            logger.warning(f"Could not load expiry cache: {e}")
    
    def _save_expiry_cache(self) -> None:
        """Save expiry information to cache"""
        try:
            with open(EXCHANGE_EXPIRY_CACHE, 'w') as f:
                # Convert datetime to string for JSON serialization
                data_to_save = {}
                for exchange, info in self._expiry_tracker.items():
                    data_to_save[exchange] = {
                        'monthly': info['monthly'],
                        'weekly': info['weekly'],
                        'no_expiry': info['no_expiry'],
                        'total_count': info['total_count'],
                        'last_updated': info.get('last_updated', datetime.now().isoformat())
                    }
                json.dump(data_to_save, f, indent=2)
            logger.debug(f"Saved expiry cache to {EXCHANGE_EXPIRY_CACHE}")
        except Exception as e:
            logger.warning(f"Could not save expiry cache: {e}")
    
    # =========================================================================
    # PUBLIC QUERY METHODS
    # =========================================================================
    
    def get_instruments_for_exchange(self, exchange: str) -> List[Dict]:
        """
        Get all instruments for given exchange from enriched master
        
        Args:
            exchange: Exchange name (NSE, NFO, BSE, MCX, NCDEX, CDS)
        
        Returns:
            List of instrument dicts with:
            - tradingsymbol: API symbol
            - underlying_symbol: Base asset (for data organization)
            - display_name: User-friendly name
            - instrument_type: EQ/FUT/CE/PE
            - expiry: Expiry date
            - strike: Strike price (for options)
        """
        with self.lock:
            if exchange not in self._exchange_instruments:
                logger.warning(f"Exchange '{exchange}' not found in enriched instruments")
                return []
            
            return self._exchange_instruments.get(exchange, [])
    
    def get_symbols_for_exchange(self, exchange: str, instrument_type: Optional[str] = None) -> List[str]:
        """
        Get all symbols for exchange, optionally filtered by type
        
        Args:
            exchange: Exchange name
            instrument_type: Optional filter (EQ, FUT, CE, PE)
        
        Returns:
            List of trading symbols
        """
        instruments = self.get_instruments_for_exchange(exchange)
        
        if instrument_type:
            instruments = [i for i in instruments if i['instrument_type'] == instrument_type]
        
        return [i['tradingsymbol'] for i in instruments]
    
    def get_underlying_symbols(self, exchange: str) -> List[str]:
        """
        Get unique underlying symbols for exchange
        Useful for organizing downloaded data
        
        Args:
            exchange: Exchange name
        
        Returns:
            List of unique underlying symbols
        """
        instruments = self.get_instruments_for_exchange(exchange)
        underlying_set = set(i['underlying_symbol'] for i in instruments if i['underlying_symbol'])
        return sorted(list(underlying_set))
    
    def get_expiry_info(self, exchange: str) -> Dict:
        """
        Get expiry date tracking for exchange
        
        Returns:
            {
                'monthly': [{'symbol': '...', 'expiry': '...', 'display': '...'}],
                'weekly': [{'symbol': '...', 'expiry': '...', 'display': '...'}],
                'no_expiry': ['SBIN', 'RELIANCE', ...],
                'total_count': 123
            }
        """
        return self._expiry_tracker.get(exchange, {
            'monthly': [],
            'weekly': [],
            'no_expiry': [],
            'total_count': 0
        })
    
    def get_monthly_expiries(self, exchange: str) -> List[Dict]:
        """
        Get all monthly expiry contracts for exchange
        Useful for scheduling monthly contract refresh
        """
        expiry_info = self.get_expiry_info(exchange)
        return expiry_info.get('monthly', [])
    
    def get_weekly_expiries(self, exchange: str) -> List[Dict]:
        """
        Get all weekly expiry contracts for exchange
        Useful for scheduling weekly contract refresh
        """
        expiry_info = self.get_expiry_info(exchange)
        return expiry_info.get('weekly', [])
    
    def get_non_expiring_symbols(self, exchange: str) -> List[str]:
        """
        Get symbols without expiry (cash equities)
        """
        expiry_info = self.get_expiry_info(exchange)
        return expiry_info.get('no_expiry', [])
    
    def get_instrument_details(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information for a specific symbol from enriched master
        
        Returns:
            Dict with all enriched columns, or None if not found
        """
        with self.lock:
            if self._enriched_df is None:
                return None
            
            try:
                matches = self._enriched_df[self._enriched_df['tradingsymbol'] == symbol]
                if len(matches) > 0:
                    return matches.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"Error getting details for {symbol}: {e}")
            
            return None
    
    def get_all_exchanges(self) -> List[str]:
        """Get list of all exchanges with instruments"""
        return sorted(list(self._exchange_instruments.keys()))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about enriched instruments
        
        Returns:
            Dict with counts by exchange and type
        """
        stats = {
            'total_instruments': 0,
            'by_exchange': {},
            'by_type': {},
            'expiry_summary': {},
            'last_updated': self._last_load_time.isoformat() if self._last_load_time else None
        }
        
        if self._enriched_df is None:
            return stats
        
        stats['total_instruments'] = len(self._enriched_df)
        
        # By exchange
        for exchange, instruments in self._exchange_instruments.items():
            stats['by_exchange'][exchange] = len(instruments)
        
        # By type
        for itype in self._enriched_df['instrument_type'].unique():
            stats['by_type'][str(itype)] = len(self._enriched_df[self._enriched_df['instrument_type'] == itype])
        
        # Expiry summary
        for exchange, info in self._expiry_tracker.items():
            stats['expiry_summary'][exchange] = {
                'monthly': len(info.get('monthly', [])),
                'weekly': len(info.get('weekly', [])),
                'no_expiry': len(info.get('no_expiry', []))
            }
        
        return stats
    
    # =========================================================================
    # REFRESH & MANAGEMENT
    # =========================================================================
    
    def refresh_from_enriched(self) -> bool:
        """
        Refresh enriched instruments from app_kite_universe.csv
        Call this when enrichment pipeline is updated
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Refreshing enriched instruments...")
        return self._load_enriched_instruments()
    
    def export_for_download(self, exchange: str, output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Export instruments for download in a download-friendly format
        
        Args:
            exchange: Exchange to export
            output_path: Optional CSV path to save to
        
        Returns:
            DataFrame with download-relevant columns
        """
        try:
            instruments = self.get_instruments_for_exchange(exchange)
            
            if not instruments:
                logger.warning(f"No instruments found for {exchange}")
                return None
            
            df = pd.DataFrame(instruments)
            
            # Reorder columns for clarity
            column_order = [
                'tradingsymbol', 'underlying_symbol', 'display_name',
                'instrument_type', 'expiry', 'strike', 'instrument_token'
            ]
            available_cols = [c for c in column_order if c in df.columns]
            df = df[available_cols]
            
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Exported {len(df)} instruments to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error exporting instruments for {exchange}: {e}")
            return None


# Singleton instance
_manager: Optional[EnrichedInstrumentManager] = None
_manager_lock = threading.Lock()


def get_enriched_instrument_manager() -> EnrichedInstrumentManager:
    """Get singleton instance of enriched instrument manager"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = EnrichedInstrumentManager()
    return _manager


if __name__ == '__main__':
    # Test the manager
    mgr = get_enriched_instrument_manager()
    
    print("\n=== ENRICHED INSTRUMENT MANAGER TEST ===\n")
    
    # List all exchanges
    exchanges = mgr.get_all_exchanges()
    print(f"Available Exchanges: {exchanges}")
    
    # Get statistics
    stats = mgr.get_statistics()
    print(f"\nTotal Instruments: {stats['total_instruments']}")
    print(f"By Exchange: {stats['by_exchange']}")
    print(f"By Type: {stats['by_type']}")
    
    # Example: NSE statistics
    if 'NSE' in exchanges:
        print(f"\n=== NSE DETAILS ===")
        nse_symbols = mgr.get_symbols_for_exchange('NSE')
        print(f"Total NSE symbols: {len(nse_symbols)}")
        print(f"Sample: {nse_symbols[:5]}")
    
    # Example: NFO expiry tracking
    if 'NFO' in exchanges:
        print(f"\n=== NFO EXPIRY TRACKING ===")
        expiry_info = mgr.get_expiry_info('NFO')
        print(f"Monthly contracts: {len(expiry_info['monthly'])}")
        print(f"Weekly contracts: {len(expiry_info['weekly'])}")
        print(f"Sample monthly: {[m['display'] for m in expiry_info['monthly'][:3]]}")
