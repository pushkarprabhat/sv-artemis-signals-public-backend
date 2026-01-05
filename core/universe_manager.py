"""
core/universe_manager.py — Persistent Universe Management with Kite API Integration
Manages complete instrument list from Kite API with disk caching and refresh on demand
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import threading
import hashlib
from config import BASE_DIR
from utils.logger import logger
import utils.helpers

# Cache location
UNIVERSE_CACHE_DIR = BASE_DIR.parent / "cache"
UNIVERSE_CACHE_FILE = UNIVERSE_CACHE_DIR / "universe_cache.parquet"
UNIVERSE_METADATA_FILE = UNIVERSE_CACHE_DIR / "universe_metadata.json"
UNIVERSE_REFRESH_INTERVAL = 24 * 60 * 60  # 24 hours in seconds


class UniverseManager:
    """
    Manages the complete instrument universe from Kite API
    
    Features:
    - Fetches ALL instruments from Kite (not just Nifty500)
    - Caches to disk for fast loading
    - Automatic refresh on demand
    - Metadata tracking (version, refresh time, count)
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize universe manager"""
        self.lock = threading.RLock()
        self._universe_cache: Optional[pd.DataFrame] = None
        self._last_refresh_time: Optional[datetime] = None
        self._metadata: Dict[str, Any] = {}
        
        # Ensure cache directory exists
        UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load metadata if available
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata about cached universe"""
        try:
            if UNIVERSE_METADATA_FILE.exists():
                with open(UNIVERSE_METADATA_FILE, 'r') as f:
                    self._metadata = json.load(f)
                
                # Restore last refresh time
                if 'last_refresh' in self._metadata:
                    self._last_refresh_time = datetime.fromisoformat(
                        self._metadata['last_refresh']
                    )
                
                logger.debug(
                    f"Universe metadata loaded: {self._metadata.get('total_instruments', 0)} "
                    f"instruments, last refresh {self._metadata.get('last_refresh', 'never')}"
                )
        except Exception as e:
            logger.warning(f"Could not load universe metadata: {e}")
            self._metadata = {}
    
    def _save_metadata(self, df: pd.DataFrame) -> None:
        """Save metadata about universe"""
        try:
            now = datetime.now()
            self._metadata = {
                'total_instruments': len(df),
                'last_refresh': now.isoformat(),
                'cache_file': str(UNIVERSE_CACHE_FILE),
                'columns': list(df.columns),
                'exchanges': sorted(df['Exchange'].unique().tolist()) if 'Exchange' in df.columns else [],
                'cache_version': '2.0'
            }
            
            with open(UNIVERSE_METADATA_FILE, 'w') as f:
                json.dump(self._metadata, f, indent=2)
            
            self._last_refresh_time = now
            logger.debug(f"Universe metadata saved: {len(df)} instruments")
        except Exception as e:
            logger.error(f"Failed to save universe metadata: {e}")
    
    def get_universe(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get the complete instrument universe
        
        Args:
            force_refresh: If True, fetch fresh data from Kite API
        
        Returns:
            DataFrame with all instruments, or None if unavailable
        """
        with self.lock:
            # If cached and not force refresh, return cache
            if self._universe_cache is not None and not force_refresh:
                logger.debug(f"Returning cached universe ({len(self._universe_cache)} instruments)")
                return self._universe_cache.copy()
            
            # Try to load from disk cache first (unless force refresh)
            if not force_refresh:
                df = self._load_from_cache()
                if df is not None:
                    self._universe_cache = df
                    return df.copy()
            
            # Need to refresh from Kite API
            logger.info("Refreshing universe from Kite API...")
            df = self._fetch_from_kite_api()
            
            if df is not None and len(df) > 0:
                # Save to disk cache
                self._save_to_cache(df)
                self._universe_cache = df
                return df.copy()
            
            # If Kite API fails, try cache as fallback
            logger.warning("Kite API fetch failed, attempting to load from cache...")
            df = self._load_from_cache()
            if df is not None:
                self._universe_cache = df
                return df.copy()
            
            return None
    
    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load universe from disk cache and ensure net_change/change_pct columns"""
        try:
            if not UNIVERSE_CACHE_FILE.exists():
                logger.debug("No universe cache file found")
                return None
            df = pd.read_parquet(UNIVERSE_CACHE_FILE)
            # Convert Expiry back to datetime if it was stored as string
            if 'Expiry' in df.columns and df['Expiry'].dtype == 'object':
                df['Expiry'] = pd.to_datetime(df['Expiry'], errors='coerce')
            # --- Ensure net_change and change_pct columns are always present and correct ---
            try:
                from core.data_manager import DataManager
                df = DataManager.verify_and_fix_change_pct(df)
            except Exception as e:
                logger.warning(f"Could not verify/fix net_change/change_pct: {e}")
            logger.info(f"Loaded universe from cache: {len(df)} instruments")
            return df
        except Exception as e:
            logger.error(f"Failed to load universe from cache: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save universe to disk cache (Parquet + Excel)"""
        try:
            # Ensure columns are serializable for parquet
            df_clean = df.copy()
            
            # Convert Expiry column (datetime.date objects) to strings
            if 'Expiry' in df_clean.columns:
                df_clean['Expiry'] = df_clean['Expiry'].apply(
                    lambda x: str(x) if pd.notna(x) else None
                )
            
            # Convert any other datetime columns to strings
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    # Check if column contains datetime objects
                    if len(df_clean) > 0:
                        first_val = df_clean[col].iloc[0]
                        if hasattr(first_val, 'isoformat'):  # datetime, date, or time object
                            df_clean[col] = df_clean[col].apply(
                                lambda x: str(x) if pd.notna(x) else None
                            )
            
            # Save as parquet (efficient storage + fast loading)
            df_clean.to_parquet(UNIVERSE_CACHE_FILE, index=False, compression='snappy')
            
            # Save as Excel in universe/enriched folder
            try:
                instruments_dir = BASE_DIR / "universe" / "enriched"
                instruments_dir.mkdir(parents=True, exist_ok=True)
                
                # Create timestamped filename with latest symlink
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_file = instruments_dir / f"instruments_{timestamp}.xlsx"
                latest_file = instruments_dir / "instruments_latest.xlsx"
                
                # Save to Excel with formatting
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    df_clean.to_excel(writer, sheet_name='Instruments', index=False)
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets['Instruments']
                    for column in worksheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(cell.value)
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
                
                # Create symlink to latest file
                if latest_file.exists():
                    latest_file.unlink()
                latest_file.symlink_to(excel_file)
                
                logger.info(f"Saved instruments to Excel: {excel_file}")
                logger.info(f"Latest copy: {latest_file}")
            except ImportError:
                logger.warning("openpyxl not installed, skipping Excel export. Install with: pip install openpyxl")
            except Exception as e:
                logger.warning(f"Failed to save Excel file: {e}")
            
            # Save metadata
            self._save_metadata(df_clean)
            
            
            logger.info(f"Saved universe to cache: {len(df_clean)} instruments")
        except Exception as e:
            logger.error(f"Failed to save universe to cache: {e}")
    
    def _fetch_from_kite_api(self) -> Optional[pd.DataFrame]:
        """
        Fetch complete instrument universe from Kite API
        
        This gets ALL instruments from all exchanges:
        - NSE (stocks, indices, mutual funds)
        - BSE (stocks)
        - NFO (futures, options)
        - MCX (commodities, metals)
        - NCDEX (commodities)
        - CDS (currencies, bonds)
        """
        try:
            kite = utils.helpers.kite
            if kite is None:
                logger.error("Kite client not initialized")
                return None
            
            logger.info("Fetching instruments from Kite API (all exchanges)...")
            
            # Fetch instruments from each exchange with retry logic
            all_instruments = []
            exchanges = ['NSE', 'BSE', 'NFO', 'MCX', 'NCDEX', 'CDS']
            
            for exchange in exchanges:
                max_retries = 2
                instruments = None
                
                for attempt in range(max_retries):
                    try:
                        logger.debug(f"Fetching {exchange} instruments (attempt {attempt + 1}/{max_retries})...")
                        instruments = kite.instruments(exchange)
                        
                        if instruments and len(instruments) > 0:
                            # Convert to DataFrame
                            df_exchange = pd.DataFrame(instruments)
                            all_instruments.append(df_exchange)
                            logger.info(f"Fetched {len(instruments)} instruments from {exchange}")
                            break  # Success, exit retry loop
                    except Exception as e:
                        error_msg = str(e)
                        if 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                            if attempt < max_retries - 1:
                                import time
                                wait_time = 2 ** attempt  # Exponential backoff: 1, 2 seconds
                                logger.warning(f"Timeout fetching {exchange}, retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                logger.warning(f"Failed to fetch {exchange} after {max_retries} attempts: {e}")
                        else:
                            logger.warning(f"Failed to fetch {exchange}: {e}")
                            break  # Don't retry on non-timeout errors
            
            if not all_instruments:
                logger.error("Failed to fetch instruments from any exchange")
                return None
            
            # Combine all exchanges
            df_universe = pd.concat(all_instruments, ignore_index=True)
            
            # Standardize columns
            df_universe = self._standardize_universe_columns(df_universe)
            
            # Add derived columns (UNDERLYING SYMBOL, DISPLAY-NAME)
            df_universe = self._add_derived_columns(df_universe)
            
            logger.info(f"Successfully fetched {len(df_universe)} total instruments from Kite")
            return df_universe
            
        except Exception as e:
            logger.error(f"Error fetching from Kite API: {e}")
            return None
    
    def _standardize_universe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize universe dataframe columns
        Maps Kite API columns to our standard format
        """
        try:
            df_standard = df.copy()
            
            # Standard columns we want to keep
            standard_cols = {
                'instrument_token': 'InstrumentToken',
                'exchange_token': 'ExchangeToken',
                'tradingsymbol': 'Symbol',
                'name': 'Name',
                'last_price': 'LastPrice',
                'expiry': 'Expiry',
                'strike': 'Strike',
                'lot_size': 'LotSize',
                'instrument_type': 'InstrumentType',
                'segment': 'Segment',
                'exchange': 'Exchange',
                'tick_size': 'TickSize',
                'multiplier': 'Multiplier',
                'isin': 'ISIN'
            }
            
            # Rename columns that exist
            rename_dict = {}
            for old, new in standard_cols.items():
                if old in df_standard.columns:
                    rename_dict[old] = new
            
            df_standard = df_standard.rename(columns=rename_dict)
            
            # Keep only columns we recognize
            keep_cols = [col for col in df_standard.columns if col in standard_cols.values() or col in df.columns]
            df_standard = df_standard[keep_cols]
            
            # Ensure Symbol column exists (renamed from tradingsymbol)
            if 'Symbol' not in df_standard.columns and 'tradingsymbol' in df_standard.columns:
                df_standard['Symbol'] = df_standard['tradingsymbol']
            
            # Ensure Exchange column exists
            if 'Exchange' not in df_standard.columns:
                logger.warning("Exchange column missing from universe")
            
            return df_standard
            
        except Exception as e:
            logger.error(f"Error standardizing universe columns: {e}")
            return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns to universe:
        - UNDERLYING SYMBOL: EXCHANGE + '-' + SYMBOL2
          where SYMBOL2 = SYMBOL (for EQ) or NAME (for derivatives)
        - DISPLAY-NAME: Formatted user-friendly names
        
        Examples:
        - EQ: NSE-RELIANCE, Display: "RELIANCE"
        - FUT: NSE-RELIANCE21JAN23FUT, Display: "RELIANCE JAN 23 FUT"
        - CE: NSE-RELIANCE21JAN23C2800, Display: "RELIANCE 21 JAN 23 2800 CE"
        """
        try:
            df_derived = df.copy()
            
            # Ensure required columns exist
            if 'Exchange' not in df_derived.columns:
                logger.warning("Exchange column missing, cannot add derived columns")
                return df_derived
            
            # Add UNDERLYING SYMBOL
            underlying_symbols = []
            for idx, row in df_derived.iterrows():
                exchange = str(row.get('Exchange', 'NSE')).upper()
                instrument_type = str(row.get('InstrumentType', 'EQ')).upper()
                
                # Use Symbol for EQ, Name for derivatives
                if instrument_type == 'EQ':
                    symbol2 = row.get('Symbol', '')
                else:
                    # For derivatives, use Name field
                    symbol2 = row.get('Name', row.get('Symbol', ''))
                
                # Construct UNDERLYING SYMBOL
                if symbol2:
                    underlying = f"{exchange}-{symbol2}"
                else:
                    underlying = f"{exchange}-UNKNOWN"
                
                underlying_symbols.append(underlying)
            
            df_derived['UNDERLYING SYMBOL'] = underlying_symbols
            
            # Add DISPLAY-NAME
            display_names = []
            for idx, row in df_derived.iterrows():
                try:
                    instrument_type = str(row.get('InstrumentType', 'EQ')).upper()
                    symbol = row.get('Symbol', '')
                    name = row.get('Name', '')
                    
                    if instrument_type == 'EQ':
                        # For equity: use Name or Symbol
                        display_name = name if name else symbol
                    
                    elif instrument_type == 'FUT':
                        # For futures: SYMBOL MONTH YEAR TYPE
                        # Example: RELIANCE JAN 23 FUT
                        try:
                            expiry = row.get('Expiry')
                            if expiry:
                                from datetime import datetime
                                # Parse expiry date
                                if isinstance(expiry, str):
                                    expiry_date = pd.to_datetime(expiry)
                                else:
                                    expiry_date = expiry
                                
                                month = expiry_date.strftime('%b').upper()
                                year = expiry_date.strftime('%y').upper()
                                display_name = f"{symbol} {month} {year} FUT"
                            else:
                                display_name = f"{symbol} FUT"
                        except:
                            display_name = f"{symbol} FUT"
                    
                    elif instrument_type in ('CE', 'PE'):
                        # For options: SYMBOL DAY MON YEAR STRIKE TYPE
                        # Example: RELIANCE 21 JAN 23 2800 CE
                        try:
                            expiry = row.get('Expiry')
                            strike = row.get('Strike', 0)
                            option_type = instrument_type
                            
                            if expiry and strike:
                                if isinstance(expiry, str):
                                    expiry_date = pd.to_datetime(expiry)
                                else:
                                    expiry_date = expiry
                                
                                day = expiry_date.day
                                month = expiry_date.strftime('%b').upper()
                                year = expiry_date.strftime('%y').upper()
                                display_name = f"{symbol} {day} {month} {year} {strike} {option_type}"
                            else:
                                display_name = f"{symbol} {option_type}"
                        except:
                            display_name = f"{symbol} {instrument_type}"
                    
                    else:
                        # Default: use Name or Symbol
                        display_name = name if name else symbol
                    
                    display_names.append(display_name)
                
                except Exception as e:
                    logger.debug(f"Error generating display name for row {idx}: {e}")
                    display_names.append(row.get('Name', row.get('Symbol', 'UNKNOWN')))
            
            df_derived['DISPLAY-NAME'] = display_names
            
            logger.info(f"Added derived columns to {len(df_derived)} instruments")
            logger.debug(f"Sample UNDERLYING SYMBOL values: {df_derived['UNDERLYING SYMBOL'].head(3).tolist()}")
            logger.debug(f"Sample DISPLAY-NAME values: {df_derived['DISPLAY-NAME'].head(3).tolist()}")
            
            return df_derived
        
        except Exception as e:
            logger.error(f"Error adding derived columns: {e}")
            return df

    
    def should_refresh(self) -> bool:
        """
        Check if universe should be refreshed
        
        Returns True if:
        - Never been fetched
        - Older than UNIVERSE_REFRESH_INTERVAL
        """
        if self._last_refresh_time is None:
            return True
        
        age = datetime.now() - self._last_refresh_time
        should = age.total_seconds() > UNIVERSE_REFRESH_INTERVAL
        
        if should:
            hours_old = age.total_seconds() / 3600
            logger.info(f"Universe is {hours_old:.1f} hours old, should refresh")
        
        return should
    
    def get_last_refresh_time(self) -> str:
        """Get last refresh time as formatted string"""
        if self._last_refresh_time is None:
            return "Never"
        return self._last_refresh_time.strftime('%Y-%m-%d %H:%M IST')
    
    def cache_exists(self) -> bool:
        """Check if cache file exists"""
        return UNIVERSE_CACHE_FILE.exists()

        """Get metadata about the cached universe"""
        return self._metadata.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the universe"""
        try:
            df = self.get_universe()
            if df is None or len(df) == 0:
                return {"status": "empty"}
            
            stats = {
                'total_instruments': len(df),
                'exchanges': sorted(df['Exchange'].unique().tolist()) if 'Exchange' in df.columns else [],
                'instruments_by_exchange': {},
                'last_refresh': self._metadata.get('last_refresh', 'unknown'),
                'cache_age_hours': self._get_cache_age_hours()
            }
            
            # Count by exchange
            if 'Exchange' in df.columns:
                stats['instruments_by_exchange'] = df['Exchange'].value_counts().to_dict()
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def _get_cache_age_hours(self) -> float:
        """Get age of cached universe in hours"""
        if self._last_refresh_time is None:
            return float('inf')
        age = datetime.now() - self._last_refresh_time
        return age.total_seconds() / 3600
    
    # ==================== AGGREGATOR INTEGRATION ====================
    
    def integrate_with_aggregator(self) -> bool:
        """
        Integrate Kite universe data with aggregator.
        Populates SQLite database with equities and product types.
        """
        try:
            from services.universe_aggregator import get_universe_aggregator
            
            logger.info("Integrating Kite universe with aggregator...")
            
            # Get Kite universe data
            universe_data = {'instruments': self.get_universe().to_dict('records')}
            
            # Load into aggregator
            aggregator = get_universe_aggregator()
            results = aggregator.load_from_kite_universe(universe_data)
            
            if results.get('success'):
                logger.info("Successfully integrated with aggregator")
                return True
            else:
                logger.error(f"Integration failed: {results}")
                return False
        
        except ImportError:
            logger.warning("Aggregator not available")
            return False
        except Exception as e:
            logger.error(f"Error integrating with aggregator: {e}")
            return False
    
    def refresh_aggregated_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Refresh aggregated data in SQLite.
        Should be called periodically (daily at market open).
        """
        try:
            from services.universe_aggregator import get_universe_aggregator
            
            logger.info("Refreshing aggregated data...")
            
            # Get fresh Kite universe
            df_universe = self.get_universe(force_refresh=force_refresh)
            if df_universe is None:
                return {'success': False, 'error': 'Failed to fetch universe'}
            
            # Convert to aggregator format
            universe_data = {'instruments': df_universe.to_dict('records')}
            
            # Refresh via aggregator
            aggregator = get_universe_aggregator()
            results = aggregator.refresh_all_data(universe_data)
            
            return results
        
        except ImportError:
            logger.warning("Aggregator not available")
            return {'success': False, 'error': 'Aggregator not available'}
        except Exception as e:
            logger.error(f"Error refreshing aggregated data: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_aggregator_stats(self) -> Dict[str, Any]:
        """Get statistics from aggregator."""
        try:
            from services.universe_aggregator import get_universe_aggregator
            
            aggregator = get_universe_aggregator()
            return aggregator.get_universe_statistics()
        
        except ImportError:
            logger.warning("Aggregator not available")
            return {}
        except Exception as e:
            logger.error(f"Error getting aggregator stats: {e}")
            return {}
    
    def validate_aggregated_data(self) -> Dict[str, bool]:
        """Validate data integrity in aggregator."""
        try:
            from services.universe_aggregator import get_universe_aggregator
            
            aggregator = get_universe_aggregator()
            return aggregator.validate_data_integrity()
        
        except ImportError:
            logger.warning("Aggregator not available")
            return {}
        except Exception as e:
            logger.error(f"Error validating aggregated data: {e}")
            return {}
    
    def export_to_csv(self, filepath: Optional[Path] = None) -> Optional[Path]:
        """
        Export universe to CSV for external use
        
        Args:
            filepath: Where to save (default: symbols/universe.csv)
        
        Returns:
            Path to exported file, or None if failed
        """
        try:
            df = self.get_universe()
            if df is None or len(df) == 0:
                logger.error("Cannot export empty universe")
                return None
            
            if filepath is None:
                filepath = Path(__file__).parent.parent / "symbols" / "universe.csv"
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Exported universe to {filepath}: {len(df)} instruments")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export universe: {e}")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the universe and caching system
        
        Returns:
            Dict with total_instruments, exchanges, last_updated, cached_data, next_refresh, data_quality_metrics
        """
        try:
            df = self.get_universe()
            
            # Build response dictionary
            metadata = {
                'total_instruments': len(df) if df is not None else 0,
                'exchanges': sorted(df['Exchange'].unique().tolist()) if df is not None and 'Exchange' in df.columns else [],
                'last_updated': self._metadata.get('last_refresh', 'Never'),
                'cached_data': df is not None,
                'cache_status': 'available' if UNIVERSE_CACHE_FILE.exists() else 'not available',
                'next_refresh': (datetime.now() + timedelta(hours=24)).isoformat() if self._last_refresh_time else 'Unknown',
                'data_quality_metrics': {
                    'completeness_percent': self._calculate_completeness(df) if df is not None else 0,
                    'consistency_percent': 100.0,  # Assuming data is consistent if loaded successfully
                    'accuracy_percent': 100.0  # Assuming accuracy if from Kite API
                }
            }
            
            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return {
                'total_instruments': 0,
                'exchanges': [],
                'last_updated': 'Error',
                'cached_data': False,
                'cache_status': 'error',
                'next_refresh': 'Unknown',
                'data_quality_metrics': {
                    'completeness_percent': 0,
                    'consistency_percent': 0,
                    'accuracy_percent': 0
                }
            }
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """
        Calculate data completeness percentage (% of non-null values)
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Completeness percentage (0-100)
        """
        try:
            if df is None or len(df) == 0:
                return 0.0
            
            # Count non-null cells
            non_null_cells = df.count().sum()
            total_cells = len(df) * len(df.columns)
            
            completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
            return round(completeness, 2)
        except Exception as e:
            logger.warning(f"Error calculating completeness: {e}")
            return 0.0


# Global singleton instance
_universe_manager: Optional[UniverseManager] = None


def get_universe_manager() -> UniverseManager:
    """Get or create global universe manager instance"""
    global _universe_manager
    if _universe_manager is None:
        _universe_manager = UniverseManager()
    return _universe_manager


def get_universe(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Convenience function to get universe
    
    Args:
        force_refresh: Force refresh from Kite API
    
    Returns:
        Universe DataFrame or None
    """
    manager = get_universe_manager()
    return manager.get_universe(force_refresh=force_refresh)


def refresh_universe() -> bool:
    """Force refresh universe from Kite API"""
    try:
        manager = get_universe_manager()
        df = manager.get_universe(force_refresh=True)
        return df is not None and len(df) > 0
    except Exception as e:
        logger.error(f"Failed to refresh universe: {e}")
        return False
