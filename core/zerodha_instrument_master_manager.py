# core/zerodha_instrument_master_manager.py — Zerodha Instrument Master Manager
# Manages Zerodha (Kite) instrument master with search, sort, and filter capabilities

import json
import pandas as pd
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import logging
from config import BASE_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class InstrumentSegment(Enum):
    """Zerodha instrument segments"""
    NSE_EQ = "NSE_EQ"  # NSE Equity
    NSE_FO = "NFO"     # NSE Futures & Options
    BSE_EQ = "BSE_EQ"  # BSE Equity
    BSE_FO = "BSE_FO"  # BSE Futures & Options
    MCX = "MCX"        # Multi Commodity Exchange
    NCDEX = "NCDEX"    # National Commodity & Derivatives
    CDS = "CDS"        # Currency Derivatives
    MF = "MF"          # Mutual Funds


@dataclass
class ZerodhaInstrument:
    """Single Zerodha instrument record"""
    instrument_token: str
    exchange_token: str
    tradingsymbol: str
    name: str
    segment: str
    instrument_type: str  # EQ, FUT, CE, PE, etc
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # CE/PE
    lot_size: Optional[int] = None
    tick_size: Optional[float] = None
    isin: Optional[str] = None
    multiplier: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def matches_search(self, query: str) -> bool:
        """Check if instrument matches search query"""
        q = query.upper()
        return (q in self.tradingsymbol.upper() or 
                q in self.name.upper())
    
    def get_search_score(self, query: str) -> float:
        """Get relevance score for search (0-1)"""
        q = query.upper()
        ts = self.tradingsymbol.upper()
        name = self.name.upper()
        
        # Exact match highest priority
        if ts == q or name == q:
            return 1.0
        # Starts with
        if ts.startswith(q) or name.startswith(q):
            return 0.9
        # Contains
        if q in ts or q in name:
            return 0.7
        return 0.0


@dataclass
class InstrumentMasterSnapshot:
    """Snapshot of instrument master at a point in time"""
    timestamp: str
    total_count: int
    segment_counts: Dict[str, int] = field(default_factory=dict)
    file_hash: str = ""
    file_size_mb: float = 0.0
    change_summary: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# ZERODHA INSTRUMENT MASTER MANAGER
# =============================================================================

class ZerodhaInstrumentMasterManager:
    """
    Manages Zerodha instrument master data with search/sort/filter capabilities.
    
    Features:
    - Fetch from Kite API
    - Searchable by symbol/name
    - Sortable by any field
    - Filterable by segment/type/expiry
    - Persistent storage (JSON + Parquet for performance)
    - Daily refresh with change tracking
    - Archive old versions
    """
    
    DATA_DIR = BASE_DIR / "zerodha_master"
    CURRENT_JSON = DATA_DIR / "instruments_current.json"
    CURRENT_PARQUET = DATA_DIR / "instruments_current.parquet"
    SNAPSHOT_DIR = DATA_DIR / "snapshots"
    ARCHIVE_DIR = DATA_DIR / "archives"
    HISTORY_FILE = DATA_DIR / "instruments_history.json"
    INDEX_FILE = DATA_DIR / "instruments_index.json"  # For fast lookups
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize Zerodha Instrument Master Manager"""
        self._ensure_dirs()
        self.instruments: Dict[str, ZerodhaInstrument] = {}
        self.search_index: Dict[str, List[str]] = {}  # Symbol prefix -> instrument tokens
        self.snapshots: List[InstrumentMasterSnapshot] = []
        self._load_current()
        self._load_history()
        logger.info(f"ZerodhaInstrumentMasterManager initialized with {len(self.instruments)} instruments")
    
    @classmethod
    def get_instance(cls) -> "ZerodhaInstrumentMasterManager":
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _ensure_dirs(self) -> None:
        """Create necessary directories"""
        for d in [self.DATA_DIR, self.SNAPSHOT_DIR, self.ARCHIVE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # POPULATION & PERSISTENCE
    # =========================================================================
    
    def populate_from_kite_data(self, instruments_df) -> Tuple[bool, Dict]:
        """
        Populate from Kite API instruments data
        
        Args:
            instruments_df: DataFrame from kite.instruments()
        
        Returns:
            (success, stats)
        """
        try:
            new_instruments = {}
            
            for _, row in instruments_df.iterrows():
                try:
                    token = str(row.get('instrument_token', '')).strip()
                    symbol = row.get('tradingsymbol', '').strip()
                    
                    if not token or not symbol:
                        continue
                    
                    instr = ZerodhaInstrument(
                        instrument_token=token,
                        exchange_token=str(row.get('exchange_token', '')).strip(),
                        tradingsymbol=symbol,
                        name=row.get('name', '').strip(),
                        segment=row.get('segment', '').strip(),
                        instrument_type=row.get('instrument_type', '').strip(),
                        expiry=str(row.get('expiry', '')).strip() if row.get('expiry') else None,
                        strike=float(row.get('strike', 0)) if row.get('strike') else None,
                        option_type=str(row.get('option_type', '')).strip() if row.get('option_type') else None,
                        lot_size=int(row.get('lot_size', 1)) if row.get('lot_size') else None,
                        tick_size=float(row.get('tick_size', 0)) if row.get('tick_size') else None,
                        isin=str(row.get('isin', '')).strip() if row.get('isin') else None,
                        multiplier=float(row.get('multiplier', 1)) if row.get('multiplier') else None
                    )
                    
                    new_instruments[token] = instr
                    
                except Exception as e:
                    logger.warning(f"Error parsing instrument: {e}")
                    continue
            
            if not new_instruments:
                return False, {"error": "No instruments found"}
            
            # Detect changes
            change_summary = self._detect_changes(self.instruments, new_instruments)
            
            # Archive if changed
            if change_summary:
                self._archive_current()
            
            # Update
            self.instruments = new_instruments
            self._build_search_index()
            
            # Save
            self._write_current_file(new_instruments, change_summary)
            self._write_parquet_file(new_instruments)
            self._write_search_index()
            
            stats = {
                "success": True,
                "total_count": len(new_instruments),
                "segments": self._count_by_segment(new_instruments),
                "change_summary": change_summary
            }
            
            logger.info(f"Populated {stats['total_count']} instruments from Kite")
            return True, stats
            
        except Exception as e:
            logger.error(f"Error populating instruments: {e}")
            return False, {"error": str(e)}
    
    def _detect_changes(self, old: Dict[str, ZerodhaInstrument],
                       new: Dict[str, ZerodhaInstrument]) -> Optional[Dict]:
        """Detect changes"""
        if not old:
            return None
        
        added = set(new.keys()) - set(old.keys())
        removed = set(old.keys()) - set(new.keys())
        
        if not added and not removed:
            return None
        
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "total_change": len(added) + len(removed)
        }
    
    def _archive_current(self) -> None:
        """Archive current instruments"""
        if not self.CURRENT_JSON.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"instruments_{timestamp}.json.gz"
        
        try:
            with open(self.CURRENT_JSON, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Archived instruments to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive: {e}")
    
    def _write_current_file(self, instruments: Dict[str, ZerodhaInstrument],
                           change_summary: Optional[Dict]) -> None:
        """Write current instruments to JSON"""
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_count": len(instruments),
                "segments": self._count_by_segment(instruments),
                "change_summary": change_summary
            },
            "instruments": [i.to_dict() for i in instruments.values()]
        }
        
        try:
            with open(self.CURRENT_JSON, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write JSON: {e}")
    
    def _write_parquet_file(self, instruments: Dict[str, ZerodhaInstrument]) -> None:
        """Write instruments to Parquet for efficient querying"""
        try:
            records = [i.to_dict() for i in instruments.values()]
            df = pd.DataFrame(records)
            
            # Create appropriate dtypes
            df['instrument_token'] = df['instrument_token'].astype(str)
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
            df['lot_size'] = pd.to_numeric(df['lot_size'], errors='coerce').fillna(1).astype(int)
            
            df.to_parquet(self.CURRENT_PARQUET, index=False, engine='pyarrow')
            logger.info(f"Wrote {len(df)} instruments to Parquet")
            
        except Exception as e:
            logger.warning(f"Failed to write Parquet (optional): {e}")
    
    def _build_search_index(self) -> None:
        """Build search index for fast prefix matching"""
        self.search_index.clear()
        
        for token, instr in self.instruments.items():
            # Index by symbol prefixes
            symbol = instr.tradingsymbol.upper()
            for i in range(1, len(symbol) + 1):
                prefix = symbol[:i]
                if prefix not in self.search_index:
                    self.search_index[prefix] = []
                if token not in self.search_index[prefix]:
                    self.search_index[prefix].append(token)
    
    def _write_search_index(self) -> None:
        """Persist search index"""
        try:
            with open(self.INDEX_FILE, 'w') as f:
                json.dump(self.search_index, f)
        except Exception as e:
            logger.warning(f"Failed to write search index: {e}")
    
    # =========================================================================
    # LOADING & QUERYING
    # =========================================================================
    
    def _load_current(self) -> None:
        """Load current instruments"""
        if not self.CURRENT_JSON.exists():
            logger.info("No current instruments file found")
            return
        
        try:
            with open(self.CURRENT_JSON, 'r') as f:
                data = json.load(f)
            
            self.instruments = {
                i['instrument_token']: ZerodhaInstrument(**i)
                for i in data.get('instruments', [])
            }
            self._build_search_index()
            logger.info(f"Loaded {len(self.instruments)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
    
    def _load_history(self) -> List[Dict]:
        """Load history"""
        if not self.HISTORY_FILE.exists():
            return []
        
        try:
            with open(self.HISTORY_FILE, 'r') as f:
                history = json.load(f)
            self.snapshots = [InstrumentMasterSnapshot(**s) for s in history]
            return history
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []
    
    def search(self, query: str, limit: int = 100) -> List[ZerodhaInstrument]:
        """
        Search instruments by symbol or name
        
        Args:
            query: Search query (symbol or name)
            limit: Max results
        
        Returns:
            List of matching instruments, sorted by relevance
        """
        query = query.upper()
        results = []
        
        for instr in self.instruments.values():
            if instr.matches_search(query):
                score = instr.get_search_score(query)
                results.append((score, instr))
        
        # Sort by score descending, then by symbol
        results.sort(key=lambda x: (-x[0], x[1].tradingsymbol))
        
        return [instr for _, instr in results[:limit]]
    
    def filter(self, segment: Optional[str] = None,
              instrument_type: Optional[str] = None,
              expiry: Optional[str] = None) -> List[ZerodhaInstrument]:
        """
        Filter instruments by criteria
        
        Args:
            segment: NSE_EQ, NFO, etc
            instrument_type: EQ, FUT, CE, PE, etc
            expiry: Expiry date
        
        Returns:
            Filtered instruments
        """
        results = []
        
        for instr in self.instruments.values():
            if segment and instr.segment != segment:
                continue
            if instrument_type and instr.instrument_type != instrument_type:
                continue
            if expiry and instr.expiry != expiry:
                continue
            results.append(instr)
        
        return results
    
    def get_by_symbol(self, symbol: str) -> Optional[ZerodhaInstrument]:
        """Get instrument by trading symbol"""
        symbol_upper = symbol.upper()
        for instr in self.instruments.values():
            if instr.tradingsymbol == symbol_upper:
                return instr
        return None
    
    def get_by_token(self, token: str) -> Optional[ZerodhaInstrument]:
        """Get instrument by token"""
        return self.instruments.get(token)
    
    def get_all_symbols(self, segment: Optional[str] = None) -> List[str]:
        """Get all symbols, optionally for a segment"""
        symbols = []
        for instr in self.instruments.values():
            if segment and instr.segment != segment:
                continue
            symbols.append(instr.tradingsymbol)
        return sorted(symbols)
    
    def get_segments(self) -> List[str]:
        """Get all segments with instruments"""
        return sorted(list(set(i.segment for i in self.instruments.values())))
    
    def _count_by_segment(self, instruments: Dict[str, ZerodhaInstrument]) -> Dict[str, int]:
        """Count instruments by segment"""
        counts = {}
        for instr in instruments.values():
            seg = instr.segment
            counts[seg] = counts.get(seg, 0) + 1
        return counts
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        segment_counts = self._count_by_segment(self.instruments)
        
        return {
            "total_count": len(self.instruments),
            "segment_counts": segment_counts,
            "segments": list(segment_counts.keys()),
            "last_updated": self._get_last_update_time()
        }
    
    def _get_last_update_time(self) -> Optional[str]:
        """Get last update time"""
        if self.CURRENT_JSON.exists():
            mtime = self.CURRENT_JSON.stat().st_mtime
            return datetime.fromtimestamp(mtime).isoformat()
        return None
    
    def to_dataframe(self, segment: Optional[str] = None) -> pd.DataFrame:
        """
        Convert to DataFrame for analysis
        
        Args:
            segment: Optional segment filter
        
        Returns:
            DataFrame of instruments
        """
        try:
            # Try to load parquet if exists for performance
            if self.CURRENT_PARQUET.exists() and not segment:
                return pd.read_parquet(self.CURRENT_PARQUET)
            
            # Otherwise create from current instruments
            records = [i.to_dict() for i in self.instruments.values()]
            df = pd.DataFrame(records)
            
            if segment:
                df = df[df['segment'] == segment]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            return pd.DataFrame()
    
    def get_archive_info(self) -> Dict:
        """Get archived versions info"""
        if not self.ARCHIVE_DIR.exists():
            return {"count": 0, "total_size_mb": 0}
        
        archive_files = list(self.ARCHIVE_DIR.glob("*.json.gz"))
        total_size = sum(f.stat().st_size for f in archive_files) / (1024 * 1024)
        
        return {
            "count": len(archive_files),
            "total_size_mb": round(total_size, 2),
            "files": [f.name for f in sorted(archive_files, reverse=True)][:10]
        }


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_zerodha_instrument_master_manager() -> ZerodhaInstrumentMasterManager:
    """Get singleton instance"""
    return ZerodhaInstrumentMasterManager.get_instance()
