# core/kite_instruments_manager.py — Zerodha Kite Instruments API Manager
# Fetches instrument master from Kite Connect API endpoints

import io
import csv
import gzip
import json
import pandas as pd
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import logging
from config import BASE_DIR, INSTRUMENTS_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class ExchangeSegment(Enum):
    """Exchange segments"""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"  # NSE Futures & Options
    MCX = "MCX"  # Multi Commodity Exchange
    NCDEX = "NCDEX"  # National Commodity & Derivatives
    CDS = "CDS"  # Currency Derivatives


@dataclass
class KiteInstrument:
    """Single instrument from Kite API"""
    instrument_token: str
    exchange_token: str
    tradingsymbol: str
    name: str
    last_price: Optional[float] = None
    expiry: Optional[str] = None
    strike: Optional[float] = None
    lot_size: Optional[int] = None
    instrument_type: Optional[str] = None
    segment: Optional[str] = None
    exchange: Optional[str] = None
    tick_size: Optional[float] = None
    multiplier: Optional[float] = None
    isin: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def matches_search(self, query: str) -> bool:
        """Check if instrument matches search query"""
        q = query.upper()
        return (q in (self.tradingsymbol or '').upper() or 
                q in (self.name or '').upper())


@dataclass
class FetchResult:
    """Result of fetching instruments"""
    success: bool
    exchange: Optional[str]
    total_count: int = 0
    instruments: List[KiteInstrument] = field(default_factory=list)
    timestamp: str = ""
    file_hash: str = ""
    file_size_mb: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class InstrumentsSnapshot:
    """Snapshot of instruments at a point in time"""
    timestamp: str
    exchange: Optional[str]
    total_count: int
    instrument_types: Dict[str, int] = field(default_factory=dict)
    file_hash: str = ""
    change_summary: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# KITE INSTRUMENTS MANAGER
# =============================================================================

class KiteInstrumentsManager:
    """
    Manages instrument master fetched directly from Kite Connect API.
    
    Features:
    - Fetch from /instruments and /instruments/:exchange endpoints
    - Parse gzipped CSV responses
    - Persistent storage (JSON + Parquet)
    - Searchable and filterable
    - Daily refresh with change tracking
    - Archive old versions
    - Per-exchange management
    """
    
    DATA_DIR = INSTRUMENTS_DIR
    
    # Storage structure: data/kite_instruments/{exchange}/
    # - current.json, current.parquet
    # - snapshots/, archives/, history.json
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize Kite Instruments Manager"""
        self._ensure_dirs()
        self.instruments: Dict[str, Dict[str, KiteInstrument]] = {}  # exchange -> {token -> instrument}
        self.snapshots: Dict[str, List[InstrumentsSnapshot]] = {}  # exchange -> [snapshots]
        self._load_all_instruments()
        logger.info(f"KiteInstrumentsManager initialized")
    
    @classmethod
    def get_instance(cls) -> "KiteInstrumentsManager":
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _ensure_dirs(self) -> None:
        """Create necessary directories"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for exchange in ExchangeSegment:
            exchange_dir = self.DATA_DIR / exchange.value
            exchange_dir.mkdir(exist_ok=True)
            (exchange_dir / "snapshots").mkdir(exist_ok=True)
            (exchange_dir / "archives").mkdir(exist_ok=True)
    
    def _get_exchange_dir(self, exchange: str) -> Path:
        """Get directory for an exchange (auto-creates if missing)"""
        d = self.DATA_DIR / exchange
        d.mkdir(parents=True, exist_ok=True)
        (d / "snapshots").mkdir(exist_ok=True)
        (d / "archives").mkdir(exist_ok=True)
        return d
    
    # =========================================================================
    # FETCHING FROM KITE API
    # =========================================================================
    
    def fetch_from_kite(self, kite_client, exchange: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Fetch instruments from Kite Connect API
        
        Args:
            kite_client: KiteConnect instance
            exchange: Optional exchange filter (NSE, BSE, NFO, MCX, NCDEX, CDS)
                     If None, fetches all instruments
        
        Returns:
            (success, stats)
        """
        if not kite_client:
            return False, {"error": "Kite client not initialized"}
        
        try:
            if exchange:
                logger.info(f"Fetching instruments for {exchange} from Kite API...")
                # Fetch specific exchange
                # Note: Kite API returns instruments as list, not gzipped CSV
                # We'll need to handle the response format
                instruments_data = kite_client.instruments(exchange)
                result = self._process_instruments(instruments_data, exchange)
            else:
                logger.info(f"Fetching all instruments from Kite API...")
                # Fetch all exchanges
                all_instruments = kite_client.instruments()
                result = self._process_instruments(all_instruments, None)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching from Kite: {e}")
            return False, {"error": str(e)}
    
    def fetch_from_kite_api_endpoint(self, api_url: str, headers: Dict, 
                                    exchange: Optional[str] = None) -> FetchResult:
        """
        Fetch instruments directly from Kite API HTTP endpoint
        Uses the /instruments or /instruments/:exchange endpoint that returns gzipped CSV
        
        Args:
            api_url: Base API URL (e.g., https://api.kite.trade)
            headers: Request headers including Authorization
            exchange: Optional exchange (NSE, BSE, NFO, etc.)
        
        Returns:
            FetchResult with parsed instruments
        """
        import requests
        
        result = FetchResult(
            success=False,
            exchange=exchange,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # Build endpoint
            if exchange:
                endpoint = f"{api_url}/instruments/{exchange}"
            else:
                endpoint = f"{api_url}/instruments"
            
            logger.info(f"Fetching from: {endpoint}")
            
            # Fetch gzipped CSV
            response = requests.get(
                endpoint,
                headers=headers,
                timeout=30,
                stream=True
            )
            response.raise_for_status()
            
            # Get raw bytes
            raw_content = response.content
            result.file_size_mb = len(raw_content) / (1024 * 1024)
            result.file_hash = hashlib.md5(raw_content).hexdigest()
            
            # Decompress gzip
            try:
                decompressed = gzip.decompress(raw_content)
                csv_text = decompressed.decode('utf-8')
            except:
                # Try as plain CSV if not gzipped
                csv_text = raw_content.decode('utf-8')
            
            # Parse CSV
            instruments = self._parse_instruments_csv(csv_text)
            
            if not instruments:
                result.error = "No instruments parsed from CSV"
                return result
            
            result.success = True
            result.instruments = instruments
            result.total_count = len(instruments)
            
            logger.info(f"Successfully fetched {result.total_count} instruments")
            logger.info(f"File hash: {result.file_hash}, Size: {result.file_size_mb:.2f} MB")
            
            return result
            
        except requests.exceptions.RequestException as e:
            result.error = f"HTTP error: {e}"
            logger.error(result.error)
            return result
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error: {e}")
            return result
    
    def _parse_instruments_csv(self, csv_text: str) -> List[KiteInstrument]:
        """Parse instruments CSV (from Kite API endpoint)"""
        instruments = []
        
        try:
            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(csv_text))
            
            if not csv_reader.fieldnames:
                logger.warning("CSV has no headers")
                return []
            
            logger.info(f"CSV headers: {csv_reader.fieldnames}")
            
            for row_idx, row in enumerate(csv_reader):
                try:
                    if not row or all(not v for v in row.values()):
                        continue
                    
                    token = (row.get('instrument_token') or '').strip()
                    symbol = (row.get('tradingsymbol') or '').strip()
                    
                    if not token or not symbol:
                        continue
                    
                    # Parse numeric fields safely
                    def safe_float(val):
                        try:
                            return float(val) if val else None
                        except:
                            return None
                    
                    def safe_int(val):
                        try:
                            return int(val) if val else None
                        except:
                            return None
                    
                    instr = KiteInstrument(
                        instrument_token=token,
                        exchange_token=(row.get('exchange_token') or '').strip(),
                        tradingsymbol=symbol,
                        name=(row.get('name') or '').strip(),
                        last_price=safe_float(row.get('last_price')),
                        expiry=(row.get('expiry') or '').strip() or None,
                        strike=safe_float(row.get('strike')),
                        lot_size=safe_int(row.get('lot_size')),
                        instrument_type=(row.get('instrument_type') or '').strip(),
                        segment=(row.get('segment') or '').strip(),
                        exchange=(row.get('exchange') or '').strip(),
                        tick_size=safe_float(row.get('tick_size')),
                        multiplier=safe_float(row.get('multiplier')),
                        isin=(row.get('isin') or '').strip() or None
                    )
                    
                    instruments.append(instr)
                    
                except Exception as e:
                    logger.warning(f"Error parsing row {row_idx}: {e}")
                    continue
            
            logger.info(f"Parsed {len(instruments)} instruments from CSV")
            return instruments
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return []
    
    def _process_instruments(self, instruments_data: Any, exchange: Optional[str]) -> Tuple[bool, Dict]:
        """
        Process instruments from kite.instruments() method
        
        Args:
            instruments_data: Data from kite.instruments() or kite.instruments(exchange)
            exchange: Exchange identifier
        
        Returns:
            (success, stats)
        """
        try:
            # Convert to list if needed
            if isinstance(instruments_data, dict):
                instruments_list = instruments_data.get('data', []) if isinstance(instruments_data, dict) else instruments_data
            else:
                instruments_list = list(instruments_data) if instruments_data else []
            
            if not instruments_list:
                return False, {"error": "No instruments in response"}
            
            # Convert to KiteInstrument objects
            instruments = []
            for item in instruments_list:
                try:
                    if isinstance(item, dict):
                        instr = KiteInstrument(
                            instrument_token=str(item.get('instrument_token', '')).strip(),
                            exchange_token=str(item.get('exchange_token', '')).strip(),
                            tradingsymbol=(item.get('tradingsymbol') or '').strip(),
                            name=(item.get('name') or '').strip(),
                            last_price=float(item.get('last_price', 0)) if item.get('last_price') else None,
                            expiry=str(item.get('expiry') or '').strip() or None,
                            strike=float(item.get('strike', 0)) if item.get('strike') else None,
                            lot_size=int(item.get('lot_size', 1)) if item.get('lot_size') else None,
                            instrument_type=(item.get('instrument_type') or '').strip(),
                            segment=(item.get('segment') or '').strip(),
                            exchange=(item.get('exchange') or '').strip(),
                            tick_size=float(item.get('tick_size', 0)) if item.get('tick_size') else None,
                            multiplier=float(item.get('multiplier', 1)) if item.get('multiplier') else None,
                            isin=(item.get('isin') or '').strip() or None
                        )
                        
                        if instr.instrument_token:
                            instruments.append(instr)
                except Exception as e:
                    logger.warning(f"Error processing instrument: {e}")
            
            if not instruments:
                return False, {"error": "No valid instruments parsed"}
            
            # Save instruments
            success, change_summary = self._save_instruments(instruments, exchange)
            
            stats = {
                "success": success,
                "exchange": exchange,
                "total_count": len(instruments),
                "change_summary": change_summary
            }
            
            logger.info(f"Processed {len(instruments)} instruments for {exchange or 'all exchanges'}")
            return success, stats
            
        except Exception as e:
            logger.error(f"Error processing instruments: {e}")
            return False, {"error": str(e)}
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save_instruments(self, instruments: List[KiteInstrument], 
                         exchange: Optional[str]) -> Tuple[bool, Optional[Dict]]:
        """Save instruments to persistent storage"""
        try:
            if exchange:
                # Save for specific exchange
                exchanges = [exchange]
            else:
                # Group by exchange
                by_exchange = {}
                for instr in instruments:
                    ex = instr.exchange or "UNKNOWN"
                    if ex not in by_exchange:
                        by_exchange[ex] = []
                    by_exchange[ex].append(instr)
                exchanges = list(by_exchange.keys())
            
            for ex in exchanges:
                # Filter instruments for this exchange
                ex_instruments = [i for i in instruments if (i.exchange or "UNKNOWN") == ex]
                
                if not ex_instruments:
                    continue
                
                # Detect changes
                old_instruments = self.instruments.get(ex, {})
                change_summary = self._detect_changes(ex, old_instruments, 
                                                     {i.instrument_token: i for i in ex_instruments})
                
                # Archive if changed
                if change_summary:
                    self._archive_instruments(ex, change_summary)
                
                # Update in memory
                self.instruments[ex] = {i.instrument_token: i for i in ex_instruments}
                
                # Write files
                self._write_instruments_files(ex, ex_instruments, change_summary)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error saving instruments: {e}")
            return False, str(e)
    
    def _detect_changes(self, exchange: str, old: Dict, new: Dict) -> Optional[Dict]:
        """Detect changes in instruments"""
        if not old:
            return None
        
        added = set(new.keys()) - set(old.keys())
        removed = set(old.keys()) - set(new.keys())
        
        if not added and not removed:
            return None
        
        return {
            "exchange": exchange,
            "added_count": len(added),
            "removed_count": len(removed),
            "total_change": len(added) + len(removed)
        }
    
    def _write_instruments_files(self, exchange: str, instruments: List[KiteInstrument],
                                change_summary: Optional[Dict]) -> None:
        """Write instruments to JSON and Parquet"""
        ex_dir = self._get_exchange_dir(exchange)
        
        # Write JSON
        json_file = ex_dir / "current.json"
        try:
            data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "exchange": exchange,
                    "total_count": len(instruments),
                    "change_summary": change_summary
                },
                "instruments": [i.to_dict() for i in instruments]
            }
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Wrote {len(instruments)} instruments to {json_file}")
        except Exception as e:
            logger.error(f"Failed to write JSON: {e}")
        
        # Write Parquet for efficient querying
        parquet_file = ex_dir / "current.parquet"
        try:
            records = [i.to_dict() for i in instruments]
            df = pd.DataFrame(records)
            df.to_parquet(parquet_file, index=False, engine='pyarrow')
            logger.info(f"Wrote {len(df)} instruments to {parquet_file}")
        except Exception as e:
            logger.warning(f"Failed to write Parquet: {e}")
    
    def _archive_instruments(self, exchange: str, change_summary: Dict) -> None:
        """Archive old instruments"""
        ex_dir = self._get_exchange_dir(exchange)
        json_file = ex_dir / "current.json"
        
        if not json_file.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ex_dir / "archives" / f"{exchange}_{timestamp}.json.gz"
        
        try:
            with open(json_file, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Archived to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive: {e}")
    
    # =========================================================================
    # LOADING
    # =========================================================================
    
    def _load_all_instruments(self) -> None:
        """Load all instruments from disk"""
        try:
            for exchange in ExchangeSegment:
                self._load_instruments(exchange.value)
        except Exception as e:
            logger.error(f"Error loading instruments: {e}")
    
    def _load_instruments(self, exchange: str) -> None:
        """Load instruments for specific exchange"""
        ex_dir = self._get_exchange_dir(exchange)
        json_file = ex_dir / "current.json"
        
        if not json_file.exists():
            return
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            instruments = {
                i['instrument_token']: KiteInstrument(**i)
                for i in data.get('instruments', [])
            }
            
            self.instruments[exchange] = instruments
            logger.info(f"Loaded {len(instruments)} instruments for {exchange}")
            
        except Exception as e:
            logger.error(f"Failed to load {exchange} instruments: {e}")
    
    # =========================================================================
    # QUERYING
    # =========================================================================
    
    def search(self, query: str, exchange: Optional[str] = None, 
              limit: int = 50) -> List[KiteInstrument]:
        """Search instruments by symbol or name"""
        results = []
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            ex_instruments = self.instruments.get(ex, {}).values()
            for instr in ex_instruments:
                if instr.matches_search(query):
                    results.append(instr)
        
        return results[:limit]
    
    def filter(self, exchange: Optional[str] = None, 
              instrument_type: Optional[str] = None) -> List[KiteInstrument]:
        """Filter instruments"""
        results = []
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            ex_instruments = self.instruments.get(ex, {}).values()
            for instr in ex_instruments:
                if instrument_type and instr.instrument_type != instrument_type:
                    continue
                results.append(instr)
        
        return results
    
    def get_by_symbol(self, symbol: str, exchange: Optional[str] = None) -> Optional[KiteInstrument]:
        """Get instrument by trading symbol"""
        symbol_upper = symbol.upper()
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            for instr in self.instruments.get(ex, {}).values():
                if instr.tradingsymbol.upper() == symbol_upper:
                    return instr
        
        return None
    
    def get_by_token(self, token: str, exchange: Optional[str] = None) -> Optional[KiteInstrument]:
        """Get instrument by token"""
        token_str = str(token)
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            instr = self.instruments.get(ex, {}).get(token_str)
            if instr:
                return instr
        
        return None
    
    def get_all_symbols(self, exchange: Optional[str] = None) -> List[str]:
        """Get all symbols"""
        symbols = []
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            for instr in self.instruments.get(ex, {}).values():
                symbols.append(instr.tradingsymbol)
        
        return sorted(list(set(symbols)))
    
    def to_dataframe(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """Convert to DataFrame"""
        try:
            # Try to load parquet first for performance
            if exchange:
                parquet_file = self._get_exchange_dir(exchange) / "current.parquet"
                if parquet_file.exists():
                    return pd.read_parquet(parquet_file)
            
            # Otherwise build from instruments
            records = []
            exchanges = [exchange] if exchange else self.instruments.keys()
            
            for ex in exchanges:
                for instr in self.instruments.get(ex, {}).values():
                    records.append(instr.to_dict())
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            return pd.DataFrame()
    
    def get_statistics(self, exchange: Optional[str] = None) -> Dict:
        """Get statistics"""
        stats = {}
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            ex_instrs = self.instruments.get(ex, {}).values()
            
            type_counts = {}
            for instr in ex_instrs:
                itype = instr.instrument_type or "UNKNOWN"
                type_counts[itype] = type_counts.get(itype, 0) + 1
            
            stats[ex] = {
                "total_count": len(list(ex_instrs)),
                "instrument_types": type_counts,
                "last_updated": self._get_last_update_time(ex)
            }
        
        return stats
    
    def _get_last_update_time(self, exchange: str) -> Optional[str]:
        """Get last update time for exchange"""
        json_file = self._get_exchange_dir(exchange) / "current.json"
        if json_file.exists():
            mtime = json_file.stat().st_mtime
            return datetime.fromtimestamp(mtime).isoformat()
        return None
    
    def get_archive_info(self, exchange: Optional[str] = None) -> Dict:
        """Get archive info"""
        info = {}
        
        exchanges = [exchange] if exchange else self.instruments.keys()
        
        for ex in exchanges:
            archive_dir = self._get_exchange_dir(ex) / "archives"
            if not archive_dir.exists():
                info[ex] = {"count": 0, "total_size_mb": 0}
                continue
            
            archive_files = list(archive_dir.glob("*.json.gz"))
            total_size = sum(f.stat().st_size for f in archive_files) / (1024 * 1024)
            
            info[ex] = {
                "count": len(archive_files),
                "total_size_mb": round(total_size, 2),
                "files": [f.name for f in sorted(archive_files, reverse=True)][:5]
            }
        
        return info


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_kite_instruments_manager() -> KiteInstrumentsManager:
    """Get singleton instance"""
    return KiteInstrumentsManager.get_instance()
