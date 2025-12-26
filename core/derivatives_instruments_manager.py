# core/derivatives_instruments_manager.py — Derivatives (F&O) Instruments Manager
# Manages equity-based derivatives - Futures and Options available on NSE

import json
import hashlib
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import logging
from config import BASE_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class InstrumentType(Enum):
    """Type of derivative instrument"""
    FUTURES = "futures"
    OPTIONS = "options"
    BOTH = "both"


class OptionType(Enum):
    """Call or Put option"""
    CALL = "CE"
    PUT = "PE"


@dataclass
class DerivativeInstrument:
    """Represents a single derivative instrument (Future or Option)"""
    symbol: str
    underlying_symbol: str
    instrument_type: str  # FUTIDX, FUTSTK, OPTSTK, OPTIDX
    expiry_date: str  # Format: DD-MMM-YY
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # CE or PE
    token: Optional[str] = None
    lot_size: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def is_future(self) -> bool:
        return "FUT" in self.instrument_type.upper()
    
    def is_option(self) -> bool:
        return "OPT" in self.instrument_type.upper()
    
    def is_stock_derivative(self) -> bool:
        """Check if this is a stock-based derivative"""
        return "STK" in self.instrument_type.upper()
    
    def is_index_derivative(self) -> bool:
        """Check if this is an index-based derivative"""
        return "IDX" in self.instrument_type.upper()


@dataclass
class DerivativesSnapshot:
    """Snapshot of derivatives instruments at a point in time"""
    timestamp: str
    total_count: int
    futures_count: int
    options_count: int
    unique_underlyings: int
    expiry_dates: List[str] = field(default_factory=list)
    file_hash: str = ""
    file_path: str = ""
    change_summary: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# DERIVATIVES INSTRUMENTS MANAGER
# =============================================================================

class DerivativesInstrumentsManager:
    """
    Manages derivatives instruments (Futures & Options) for equity-based underlyings.
    
    Features:
    - Fetch equity F&O instruments from NSE/Kite
    - Track which equity securities have derivatives
    - Persist instruments with expiry dates
    - Archive old versions when expiry dates change
    - Daily refresh capability
    - Filter by underlying symbol, expiry, instrument type
    """
    
    DATA_DIR = BASE_DIR / "derivatives_instruments"
    CURRENT_FILE = DATA_DIR / "derivatives_current.json"
    SNAPSHOT_DIR = DATA_DIR / "snapshots"
    ARCHIVE_DIR = DATA_DIR / "archives"
    HISTORY_FILE = DATA_DIR / "derivatives_history.json"
    EQUITY_DERIVATIVES_INDEX = DATA_DIR / "equity_derivatives_index.json"
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize Derivatives Instruments Manager"""
        self._ensure_dirs()
        self.instruments: Dict[str, DerivativeInstrument] = {}
        self.equity_derivatives_map: Dict[str, List[str]] = {}  # symbol -> [derivative_symbols]
        self.snapshots: List[DerivativesSnapshot] = []
        self._load_current()
        self._load_history()
        logger.info(f"DerivativesInstrumentsManager initialized with {len(self.instruments)} instruments")
    
    @classmethod
    def get_instance(cls) -> "DerivativesInstrumentsManager":
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
        Populate derivatives from Kite API instruments data
        
        Args:
            instruments_df: DataFrame from kite.instruments()
        
        Returns:
            (success, stats)
        """
        try:
            new_instruments = {}
            equity_map = {}
            
            # Filter for derivatives only
            derivative_rows = instruments_df[
                (instruments_df['segment'].isin(['NFO', 'NSMALL', 'NCDEX', 'MCXSX']))
            ]
            
            for _, row in derivative_rows.iterrows():
                try:
                    # Skip non-equity derivatives
                    instrument_type = row.get('instrument_type', '').strip().upper()
                    if not instrument_type.startswith(('FUTSTK', 'OPTSTK', 'FUTIDX', 'OPTIDX')):
                        continue
                    
                    symbol = row.get('tradingsymbol', '').strip()
                    underlying = row.get('underlying_symbol', '').strip()
                    
                    if not symbol or not underlying:
                        continue
                    
                    instr = DerivativeInstrument(
                        symbol=symbol,
                        underlying_symbol=underlying,
                        instrument_type=instrument_type,
                        expiry_date=str(row.get('expiry', '')).strip(),
                        strike_price=float(row.get('strike', 0)) if row.get('strike') else None,
                        option_type=str(row.get('option_type', '')).strip() if row.get('option_type') else None,
                        token=str(row.get('instrument_token', '')).strip(),
                        lot_size=int(row.get('lot_size', 1)) if row.get('lot_size') else None
                    )
                    
                    new_instruments[symbol] = instr
                    
                    # Build equity map
                    if underlying not in equity_map:
                        equity_map[underlying] = []
                    equity_map[underlying].append(symbol)
                    
                except Exception as e:
                    logger.warning(f"Error parsing derivative row: {e}")
                    continue
            
            if not new_instruments:
                return False, {"error": "No derivatives found in data"}
            
            # Detect changes
            change_summary = self._detect_changes(self.instruments, new_instruments)
            
            # Archive if changed
            if change_summary:
                self._archive_current_snapshot(change_summary)
            
            # Update
            self.instruments = new_instruments
            self.equity_derivatives_map = equity_map
            
            # Save
            self._write_current_file(new_instruments, equity_map, change_summary)
            self._write_equity_index(equity_map)
            
            stats = {
                "success": True,
                "total_count": len(new_instruments),
                "futures_count": sum(1 for i in new_instruments.values() if i.is_future()),
                "options_count": sum(1 for i in new_instruments.values() if i.is_option()),
                "unique_underlyings": len(equity_map),
                "change_summary": change_summary
            }
            
            logger.info(f"Populated {stats['total_count']} derivatives from Kite")
            return True, stats
            
        except Exception as e:
            logger.error(f"Error populating derivatives from Kite: {e}")
            return False, {"error": str(e)}
    
    def _detect_changes(self, old: Dict[str, DerivativeInstrument], 
                       new: Dict[str, DerivativeInstrument]) -> Optional[Dict]:
        """Detect changes in derivatives"""
        if not old:
            return None
        
        added = set(new.keys()) - set(old.keys())
        removed = set(old.keys()) - set(new.keys())
        modified = []
        
        for symbol in set(old.keys()) & set(new.keys()):
            if asdict(old[symbol]) != asdict(new[symbol]):
                modified.append(symbol)
        
        if not added and not removed and not modified:
            return None
        
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "added_symbols": sorted(list(added))[:10],
            "removed_symbols": sorted(list(removed))[:10]
        }
    
    def _archive_current_snapshot(self, change_summary: Dict) -> None:
        """Archive current derivatives snapshot"""
        if not self.CURRENT_FILE.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"derivatives_{timestamp}.json.gz"
        
        try:
            with open(self.CURRENT_FILE, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Archived derivatives snapshot to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive: {e}")
    
    def _write_current_file(self, instruments: Dict[str, DerivativeInstrument],
                           equity_map: Dict[str, List[str]],
                           change_summary: Optional[Dict]) -> None:
        """Write current derivatives to file"""
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_count": len(instruments),
                "futures_count": sum(1 for i in instruments.values() if i.is_future()),
                "options_count": sum(1 for i in instruments.values() if i.is_option()),
                "unique_underlyings": len(equity_map),
                "change_summary": change_summary
            },
            "instruments": [i.to_dict() for i in instruments.values()]
        }
        
        try:
            with open(self.CURRENT_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write current file: {e}")
    
    def _write_equity_index(self, equity_map: Dict[str, List[str]]) -> None:
        """Write equity derivatives index for quick lookup"""
        try:
            with open(self.EQUITY_DERIVATIVES_INDEX, 'w') as f:
                json.dump(equity_map, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write equity index: {e}")
    
    # =========================================================================
    # LOADING & QUERYING
    # =========================================================================
    
    def _load_current(self) -> None:
        """Load current derivatives from file"""
        if not self.CURRENT_FILE.exists():
            logger.info("No current derivatives file found")
            return
        
        try:
            with open(self.CURRENT_FILE, 'r') as f:
                data = json.load(f)
            
            self.instruments = {
                i['symbol']: DerivativeInstrument(**i)
                for i in data.get('instruments', [])
            }
            
            # Load equity index
            if self.EQUITY_DERIVATIVES_INDEX.exists():
                with open(self.EQUITY_DERIVATIVES_INDEX, 'r') as f:
                    self.equity_derivatives_map = json.load(f)
            
            logger.info(f"Loaded {len(self.instruments)} derivatives from file")
            
        except Exception as e:
            logger.error(f"Failed to load current derivatives: {e}")
    
    def _load_history(self) -> List[Dict]:
        """Load history snapshots"""
        if not self.HISTORY_FILE.exists():
            return []
        
        try:
            with open(self.HISTORY_FILE, 'r') as f:
                history = json.load(f)
            self.snapshots = [DerivativesSnapshot(**s) for s in history]
            return history
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []
    
    def get_derivatives_for_symbol(self, symbol: str) -> List[DerivativeInstrument]:
        """Get all derivatives for an equity symbol"""
        symbols = self.equity_derivatives_map.get(symbol.upper(), [])
        return [self.instruments[s] for s in symbols if s in self.instruments]
    
    def get_futures_for_symbol(self, symbol: str) -> List[DerivativeInstrument]:
        """Get all futures for an equity symbol"""
        return [d for d in self.get_derivatives_for_symbol(symbol) if d.is_future()]
    
    def get_options_for_symbol(self, symbol: str) -> List[DerivativeInstrument]:
        """Get all options for an equity symbol"""
        return [d for d in self.get_derivatives_for_symbol(symbol) if d.is_option()]
    
    def get_equity_symbols_with_derivatives(self) -> Set[str]:
        """Get all equity symbols that have derivatives"""
        return set(self.equity_derivatives_map.keys())
    
    def get_all_instruments(self) -> List[DerivativeInstrument]:
        """Get all derivatives instruments"""
        return list(self.instruments.values())
    
    def get_statistics(self) -> Dict:
        """Get statistics about derivatives"""
        all_instrs = self.instruments.values()
        futures = [i for i in all_instrs if i.is_future()]
        options = [i for i in all_instrs if i.is_option()]
        
        # Get unique expiry dates
        expiry_dates = sorted(set(i.expiry_date for i in all_instrs if i.expiry_date))
        
        return {
            "total_count": len(self.instruments),
            "futures_count": len(futures),
            "options_count": len(options),
            "stock_derivatives_count": sum(1 for i in all_instrs if i.is_stock_derivative()),
            "index_derivatives_count": sum(1 for i in all_instrs if i.is_index_derivative()),
            "unique_underlyings": len(self.equity_derivatives_map),
            "unique_expiry_dates": len(expiry_dates),
            "expiry_dates": expiry_dates[:10],  # First 10
            "last_updated": self._get_last_update_time(),
            "snapshots_count": len(self.snapshots)
        }
    
    def _get_last_update_time(self) -> Optional[str]:
        """Get last update time"""
        if self.CURRENT_FILE.exists():
            mtime = self.CURRENT_FILE.stat().st_mtime
            return datetime.fromtimestamp(mtime).isoformat()
        return None
    
    def filter_by_expiry(self, expiry_date: str) -> List[DerivativeInstrument]:
        """Filter instruments by expiry date"""
        return [i for i in self.instruments.values() if i.expiry_date == expiry_date]
    
    def filter_by_strike(self, underlying: str, strike_price: float) -> List[DerivativeInstrument]:
        """Filter options by underlying and strike"""
        return [
            i for i in self.get_derivatives_for_symbol(underlying)
            if i.is_option() and i.strike_price == strike_price
        ]
    
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

def get_derivatives_instruments_manager() -> DerivativesInstrumentsManager:
    """Get singleton instance"""
    return DerivativesInstrumentsManager.get_instance()
