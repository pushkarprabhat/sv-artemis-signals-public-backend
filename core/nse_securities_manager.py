# core/nse_securities_manager.py — NSE Securities List Manager
# Fetches, persists, and archives NSE equity securities data
# Source: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
# Web: https://www.nseindia.com/static/market-data/securities-available-for-trading

import json
import csv
import requests
import hashlib
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import logging
from config import BASE_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class SecurityStatus(Enum):
    """Status of a security in NSE"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    UNKNOWN = "unknown"


class SecuritySeries(Enum):
    """Series classification for NSE securities"""
    EQ = "EQ"  # Equity segment
    BE = "BE"  # Batch trading
    BL = "BL"  # Block trading
    BO = "BO"  # Bulk trading
    BR = "BR"  # Bracket order
    ST = "ST"  # Short-term delivery
    MF = "MF"  # Mutual Fund
    CD = "CD"  # Corporate Debt
    IR = "IR"  # Interest Rate
    IL = "IL"  # Index Linked Notes
    MT = "MT"  # Municipal Bonds
    OTHER = "OTHER"  # Other series


@dataclass
class Security:
    """Represents a single NSE security"""
    symbol: str
    isin: str
    series: str
    name: str
    status: str = "active"
    industry: Optional[str] = None
    group: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def is_equity(self) -> bool:
        """Check if this is an equity segment security (EQ)"""
        return self.series.upper() == "EQ"


@dataclass
class SecuritySnapshot:
    """Represents a snapshot of the entire securities database at a point in time"""
    timestamp: str
    total_count: int
    eq_count: int
    file_hash: str
    file_path: str
    is_archived: bool = False
    archived_at: Optional[str] = None
    change_summary: Optional[Dict] = None  # Tracks additions/deletions/changes
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FetchResult:
    """Result of fetching securities from NSE"""
    success: bool
    securities: List[Security] = field(default_factory=list)
    total_count: int = 0
    eq_count: int = 0
    timestamp: str = ""
    file_hash: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# NSE SECURITIES MANAGER
# =============================================================================

class NSESecuritiesManager:
    """
    Manages NSE equity securities list with persistent storage, archival, and change tracking.
    
    Features:
    - Fetch latest securities from NSE EQUITY_L.csv
    - Persist to JSON with change detection
    - Archive old versions when changes detected
    - Track security status changes
    - Filter by series (EQ, BE, BL, etc.)
    - Daily refresh capability
    """
    
    # Configuration
    NSE_EQUITY_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    NSE_MARKET_DATA_PAGE = "https://www.nseindia.com/static/market-data/securities-available-for-trading"
    
    DATA_DIR = BASE_DIR / "nse_securities"
    CURRENT_FILE = DATA_DIR / "equity_securities_current.json"
    SNAPSHOT_DIR = DATA_DIR / "snapshots"
    ARCHIVE_DIR = DATA_DIR / "archives"
    HISTORY_FILE = DATA_DIR / "securities_history.json"
    
    REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/csv',
    }
    
    FETCH_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize NSE Securities Manager"""
        self._ensure_dirs()
        self.securities: Dict[str, Security] = {}
        self.snapshots: List[SecuritySnapshot] = []
        self._load_current()
        self._load_history()
        logger.info(f"NSESecuritiesManager initialized with {len(self.securities)} securities")
    
    @classmethod
    def get_instance(cls) -> "NSESecuritiesManager":
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
    # FETCHING & PARSING
    # =========================================================================
    
    def fetch_latest(self, force_refresh: bool = False) -> FetchResult:
        """
        Fetch latest securities list from NSE
        
        Args:
            force_refresh: If True, fetch even if data is recent
        
        Returns:
            FetchResult with parsed securities
        """
        # Check if we already have fresh data
        if not force_refresh and self._is_data_fresh():
            logger.info("Securities data is fresh, skipping fetch")
            return FetchResult(
                success=True,
                securities=list(self.securities.values()),
                total_count=len(self.securities),
                eq_count=self._count_by_series("EQ"),
                timestamp=datetime.now().isoformat(),
                file_hash=self._compute_securities_hash()
            )
        
        # Fetch from NSE
        result_data = {
            "success": False,
            "securities": [],
            "total_count": 0,
            "eq_count": 0,
            "timestamp": datetime.now().isoformat(),
            "file_hash": "",
            "error": None,
            "warnings": []
        }
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Fetching NSE securities (attempt {attempt + 1}/{self.MAX_RETRIES})")
                response = requests.get(
                    self.NSE_EQUITY_CSV_URL,
                    headers=self.REQUEST_HEADERS,
                    timeout=self.FETCH_TIMEOUT,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Parse CSV
                securities = self._parse_csv(response.text)
                
                if not securities:
                    result_data["error"] = "Parsed CSV but no securities found"
                    continue
                
                result_data["success"] = True
                result_data["securities"] = securities
                result_data["total_count"] = len(securities)
                result_data["eq_count"] = sum(1 for s in securities if s.is_equity())
                result_data["timestamp"] = datetime.now().isoformat()
                result_data["file_hash"] = self._compute_hash(response.content)
                
                logger.info(f"Successfully fetched {result_data['total_count']} securities ({result_data['eq_count']} EQ)")
                break
                
            except requests.exceptions.RequestException as e:
                result_data["error"] = str(e)
                result_data["warnings"].append(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Fetch attempt {attempt + 1} failed, retrying in {self.RETRY_DELAY}s")
                    import time
                    time.sleep(self.RETRY_DELAY)
            except Exception as e:
                result_data["error"] = str(e)
                result_data["warnings"].append(f"Parsing error on attempt {attempt + 1}: {e}")
                logger.error(f"Error fetching securities: {e}")
        
        return FetchResult(**result_data)
    
    def _parse_csv(self, csv_text: str) -> List[Security]:
        """Parse NSE EQUITY_L.csv format"""
        securities = []
        
        try:
            lines = csv_text.strip().split('\n')
            if not lines or len(lines) < 2:
                logger.warning("CSV appears empty or malformed")
                return []
            
            # Use CSV reader to properly handle quoted fields
            import io
            csv_reader = csv.DictReader(io.StringIO(csv_text))
            
            if not csv_reader.fieldnames:
                logger.warning("CSV has no headers")
                return []
            
            # Map potential field names (NSE format has spaces in column names)
            fieldnames_lower = [f.strip().lower() for f in csv_reader.fieldnames] if csv_reader.fieldnames else []
            
            # Create a mapping of normalized field names
            field_map = {}
            for original_field in (csv_reader.fieldnames or []):
                normalized = original_field.strip().lower()
                if 'symbol' in normalized:
                    field_map['symbol'] = original_field
                elif 'isin' in normalized:
                    field_map['isin'] = original_field
                elif 'series' in normalized:
                    field_map['series'] = original_field
                elif 'name' in normalized:
                    field_map['name'] = original_field
            
            logger.info(f"Detected fields: {field_map}")
            
            if 'symbol' not in field_map or 'isin' not in field_map:
                logger.warning(f"CSV missing required fields. Available: {csv_reader.fieldnames}")
                return []
            
            # Re-read to parse properly
            csv_reader = csv.DictReader(io.StringIO(csv_text))
            
            for row_idx, row in enumerate(csv_reader):
                try:
                    if not row or all(not v for v in row.values()):
                        continue
                    
                    symbol = (row.get(field_map['symbol']) or '').strip().upper()
                    isin = (row.get(field_map['isin']) or '').strip()
                    series = (row.get(field_map.get('series')) or '').strip().upper()
                    name = (row.get(field_map.get('name')) or '').strip()
                    
                    if not symbol or not isin:
                        continue
                    
                    security = Security(
                        symbol=symbol,
                        isin=isin,
                        series=series,
                        name=name,
                        status="active"
                    )
                    securities.append(security)
                    
                except Exception as e:
                    logger.warning(f"Error parsing row {row_idx}: {e}")
                    continue
            
            logger.info(f"Parsed {len(securities)} securities from CSV")
            return securities
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return []
    
    # =========================================================================
    # PERSISTENCE & ARCHIVAL
    # =========================================================================
    
    def save_securities(self, result: FetchResult) -> Tuple[bool, Optional[str]]:
        """
        Save fetched securities, detecting changes and archiving old versions
        
        Returns:
            (success, change_summary)
        """
        if not result.success or not result.securities:
            logger.error("Cannot save: invalid fetch result")
            return False, None
        
        # Create new security dict
        new_securities = {s.symbol: s for s in result.securities}
        
        # Detect changes
        change_summary = self._detect_changes(self.securities, new_securities)
        
        # If there are changes, archive the old version
        if change_summary:
            logger.info(f"Changes detected: {change_summary}")
            self._archive_current_snapshot(change_summary)
        
        # Update current
        self.securities = new_securities
        
        # Save to current file
        self._write_current_file(result, change_summary)
        
        # Create snapshot
        snapshot = SecuritySnapshot(
            timestamp=result.timestamp,
            total_count=result.total_count,
            eq_count=result.eq_count,
            file_hash=result.file_hash,
            file_path=str(self.CURRENT_FILE),
            is_archived=False,
            change_summary=change_summary
        )
        self.snapshots.append(snapshot)
        
        # Update history
        self._update_history(snapshot)
        
        logger.info(f"Saved {result.total_count} securities to {self.CURRENT_FILE}")
        return True, change_summary
    
    def _detect_changes(self, old_securities: Dict[str, Security], 
                       new_securities: Dict[str, Security]) -> Optional[Dict]:
        """Detect additions, deletions, and modifications"""
        if not old_securities:
            return None  # First time
        
        added = set(new_securities.keys()) - set(old_securities.keys())
        removed = set(old_securities.keys()) - set(new_securities.keys())
        modified = []
        
        for symbol in set(old_securities.keys()) & set(new_securities.keys()):
            old = old_securities[symbol]
            new = new_securities[symbol]
            if asdict(old) != asdict(new):
                modified.append(symbol)
        
        if not added and not removed and not modified:
            return None
        
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "added_symbols": sorted(list(added))[:10],  # First 10
            "removed_symbols": sorted(list(removed))[:10]
        }
    
    def _archive_current_snapshot(self, change_summary: Dict) -> None:
        """Archive the current securities snapshot with timestamp"""
        if not self.CURRENT_FILE.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"equity_securities_{timestamp}.json.gz"
        
        try:
            with open(self.CURRENT_FILE, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Archived old snapshot to {archive_file}")
            
        except Exception as e:
            logger.error(f"Failed to archive snapshot: {e}")
    
    def _write_current_file(self, result: FetchResult, change_summary: Optional[Dict]) -> None:
        """Write current securities to JSON file"""
        data = {
            "metadata": {
                "timestamp": result.timestamp,
                "fetch_url": self.NSE_EQUITY_CSV_URL,
                "total_count": result.total_count,
                "eq_count": result.eq_count,
                "file_hash": result.file_hash,
                "change_summary": change_summary
            },
            "securities": [s.to_dict() for s in result.securities]
        }
        
        try:
            with open(self.CURRENT_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write current file: {e}")
    
    def _update_history(self, snapshot: SecuritySnapshot) -> None:
        """Update history file with new snapshot"""
        history = self._load_history()
        history.append(snapshot.to_dict())
        
        try:
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update history: {e}")
    
    # =========================================================================
    # LOADING & QUERYING
    # =========================================================================
    
    def _load_current(self) -> None:
        """Load current securities from file"""
        if not self.CURRENT_FILE.exists():
            logger.info("No current securities file found")
            return
        
        try:
            with open(self.CURRENT_FILE, 'r') as f:
                data = json.load(f)
            
            self.securities = {
                s['symbol']: Security(**s)
                for s in data.get('securities', [])
            }
            logger.info(f"Loaded {len(self.securities)} securities from current file")
            
        except Exception as e:
            logger.error(f"Failed to load current securities: {e}")
    
    def _load_history(self) -> List[Dict]:
        """Load history snapshots"""
        if not self.HISTORY_FILE.exists():
            return []
        
        try:
            with open(self.HISTORY_FILE, 'r') as f:
                history = json.load(f)
            self.snapshots = [SecuritySnapshot(**s) for s in history]
            return history
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []
    
    def get_equity_symbols(self) -> List[str]:
        """Get all EQ series symbols"""
        return [
            s.symbol for s in self.securities.values()
            if s.is_equity()
        ]
    
    def get_security(self, symbol: str) -> Optional[Security]:
        """Get security by symbol"""
        return self.securities.get(symbol.upper())
    
    def get_all_securities(self) -> List[Security]:
        """Get all securities"""
        return list(self.securities.values())
    
    def filter_by_series(self, series: str) -> List[Security]:
        """Filter securities by series"""
        series_upper = series.upper()
        return [s for s in self.securities.values() if s.series == series_upper]
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded securities"""
        all_secs = self.securities.values()
        series_counts = {}
        
        for sec in all_secs:
            series = sec.series
            series_counts[series] = series_counts.get(series, 0) + 1
        
        return {
            "total_count": len(self.securities),
            "eq_count": self._count_by_series("EQ"),
            "series_distribution": series_counts,
            "last_updated": self._get_last_update_time(),
            "snapshots_count": len(self.snapshots)
        }
    
    def _count_by_series(self, series: str) -> int:
        """Count securities by series"""
        series_upper = series.upper()
        return sum(1 for s in self.securities.values() if s.series == series_upper)
    
    def _get_last_update_time(self) -> Optional[str]:
        """Get timestamp of last update"""
        if self.snapshots:
            return self.snapshots[-1].timestamp
        return None
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def _is_data_fresh(self, max_age_hours: int = 24) -> bool:
        """Check if current data is fresh enough to skip fetch"""
        if not self.CURRENT_FILE.exists():
            return False
        
        last_modified = self.CURRENT_FILE.stat().st_mtime
        age_hours = (datetime.now().timestamp() - last_modified) / 3600
        return age_hours < max_age_hours
    
    def _compute_securities_hash(self) -> str:
        """Compute hash of current securities"""
        content = json.dumps([asdict(s) for s in sorted(
            self.securities.values(), key=lambda x: x.symbol
        )], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def _compute_hash(content: bytes) -> str:
        """Compute hash of content"""
        return hashlib.md5(content).hexdigest()
    
    def get_archive_info(self) -> Dict:
        """Get information about archived versions"""
        if not self.ARCHIVE_DIR.exists():
            return {"count": 0, "total_size_mb": 0}
        
        archive_files = list(self.ARCHIVE_DIR.glob("*.json.gz"))
        total_size = sum(f.stat().st_size for f in archive_files) / (1024 * 1024)
        
        return {
            "count": len(archive_files),
            "total_size_mb": round(total_size, 2),
            "files": [f.name for f in sorted(archive_files, reverse=True)][:10]
        }
    
    def restore_snapshot(self, archive_filename: str) -> bool:
        """Restore securities from an archived snapshot"""
        archive_file = self.ARCHIVE_DIR / archive_filename
        
        if not archive_file.exists():
            logger.error(f"Archive file not found: {archive_filename}")
            return False
        
        try:
            temp_file = self.DATA_DIR / "temp_restore.json"
            with gzip.open(archive_file, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Load and verify
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            # Replace current
            shutil.copy(temp_file, self.CURRENT_FILE)
            self._load_current()
            temp_file.unlink()
            
            logger.info(f"Restored securities from {archive_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_nse_securities_manager() -> NSESecuritiesManager:
    """Get singleton instance of NSESecuritiesManager"""
    return NSESecuritiesManager.get_instance()
