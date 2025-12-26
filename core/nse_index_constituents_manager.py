# core/nse_index_constituents_manager.py — NSE Index Constituents Manager
# Manages index constituents (stocks in each index) with daily updates

import json
import requests
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
import threading
import logging
from config import BASE_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Constituent:
    """A stock constituent of an index"""
    symbol: str
    isin: Optional[str] = None
    weight: Optional[float] = None  # % weight
    open_weight: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IndexConstituentsSnapshot:
    """Snapshot of index constituents at a point in time"""
    timestamp: str
    index_code: str
    total_constituents: int
    file_hash: str = ""
    change_summary: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# NSE INDEX CONSTITUENTS MANAGER
# =============================================================================

class NSEIndexConstituentsManager:
    """
    Manages constituents of NSE indices with daily updates.
    
    Features:
    - Fetch constituents for each index from NSE
    - Track weight changes
    - Archive old versions
    - Daily refresh with change detection
    - Quick lookup of index for a symbol
    """
    
    # NSE constituents endpoints
    NSE_API_BASE = "https://www.nseindia.com"
    CONSTITUENTS_ENDPOINTS = {
        'Nifty50': '/api/equity/constituents?index=NIFTY%2050',
        'Nifty100': '/api/equity/constituents?index=NIFTY%20100',
        'Nifty200': '/api/equity/constituents?index=NIFTY%20200',
        'Nifty500': '/api/equity/constituents?index=NIFTY%20500',
        'NiftyBank': '/api/equity/constituents?index=NIFTY%20BANK',
        'NiftyAuto': '/api/equity/constituents?index=NIFTY%20AUTO',
        'NiftyIT': '/api/equity/constituents?index=NIFTY%20IT',
        'NiftyMetal': '/api/equity/constituents?index=NIFTY%20METAL',
        'NiftyPharma': '/api/equity/constituents?index=NIFTY%20PHARMA',
        'NiftyPSUBank': '/api/equity/constituents?index=NIFTY%20PSU%20BANK',
        'NiftyPrivateBank': '/api/equity/constituents?index=NIFTY%20PRIVATE%20BANK',
    }
    
    DATA_DIR = BASE_DIR / "nse_index_constituents"
    CONSTITUENTS_DIR = DATA_DIR / "constituents"
    SNAPSHOT_DIR = DATA_DIR / "snapshots"
    ARCHIVE_DIR = DATA_DIR / "archives"
    HISTORY_FILE = DATA_DIR / "constituents_history.json"
    REVERSE_INDEX = DATA_DIR / "symbol_to_indices.json"  # symbol -> [indices]
    
    REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 2
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize NSE Index Constituents Manager"""
        self._ensure_dirs()
        self.constituents: Dict[str, List[Constituent]] = {}  # index_code -> [constituents]
        self.reverse_index: Dict[str, Set[str]] = {}  # symbol -> {index_codes}
        self.snapshots: List[IndexConstituentsSnapshot] = []
        self._load_constituents()
        self._load_reverse_index()
        self._load_history()
        logger.info(f"NSEIndexConstituentsManager initialized with {len(self.constituents)} indices")
    
    @classmethod
    def get_instance(cls) -> "NSEIndexConstituentsManager":
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _ensure_dirs(self) -> None:
        """Create necessary directories"""
        for d in [self.DATA_DIR, self.CONSTITUENTS_DIR, self.SNAPSHOT_DIR, self.ARCHIVE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # FETCHING & UPDATING
    # =========================================================================
    
    def populate_from_config(self, constituents_data: Dict[str, List[str]]) -> Tuple[bool, Dict]:
        """
        Populate constituents from provided data (from config or file)
        
        Args:
            constituents_data: {index_code: [symbol1, symbol2, ...]}
        
        Returns:
            (success, stats)
        """
        try:
            new_constituents = {}
            new_reverse = {}
            
            for index_code, symbols in constituents_data.items():
                constituents_list = [
                    Constituent(symbol=s.upper()) for s in symbols
                ]
                new_constituents[index_code] = constituents_list
                
                # Build reverse index
                for const in constituents_list:
                    if const.symbol not in new_reverse:
                        new_reverse[const.symbol] = set()
                    new_reverse[const.symbol].add(index_code)
            
            # Detect changes and archive
            changed_indices = []
            for index_code, new_const in new_constituents.items():
                old_const = self.constituents.get(index_code, [])
                change_summary = self._detect_changes(index_code, old_const, new_const)
                
                if change_summary:
                    self._archive_constituents(index_code, change_summary)
                    changed_indices.append(index_code)
                
                # Write individual index file
                self._write_constituents_file(index_code, new_const)
            
            # Update current state
            self.constituents = new_constituents
            self.reverse_index = {k: list(v) for k, v in new_reverse.items()}
            
            # Write reverse index
            self._write_reverse_index()
            
            stats = {
                "success": True,
                "total_indices": len(new_constituents),
                "total_constituents": sum(len(c) for c in new_constituents.values()),
                "updated_indices": len(changed_indices),
                "changed_indices": changed_indices
            }
            
            logger.info(f"Populated constituents for {stats['total_indices']} indices")
            return True, stats
            
        except Exception as e:
            logger.error(f"Error populating constituents: {e}")
            return False, {"error": str(e)}
    
    def fetch_and_update(self, index_code: str) -> Tuple[bool, Dict]:
        """
        Fetch constituents for a specific index from NSE API
        
        Args:
            index_code: Index code (e.g., 'Nifty50')
        
        Returns:
            (success, stats)
        """
        if index_code not in self.CONSTITUENTS_ENDPOINTS:
            return False, {"error": f"Unknown index: {index_code}"}
        
        endpoint = self.CONSTITUENTS_ENDPOINTS[index_code]
        url = self.NSE_API_BASE + endpoint
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Fetching constituents for {index_code} (attempt {attempt + 1})")
                response = requests.get(
                    url,
                    headers=self.REQUEST_HEADERS,
                    timeout=self.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                constituents_data = data.get('data', [])
                
                # Parse constituents
                new_constituents = []
                for item in constituents_data:
                    try:
                        const = Constituent(
                            symbol=item.get('symbol', '').strip().upper(),
                            isin=item.get('isin', '').strip() if item.get('isin') else None,
                            weight=float(item.get('weight', 0)) if item.get('weight') else None,
                        )
                        if const.symbol:
                            new_constituents.append(const)
                    except Exception as e:
                        logger.warning(f"Error parsing constituent: {e}")
                        continue
                
                if not new_constituents:
                    logger.warning(f"No constituents found for {index_code}")
                    continue
                
                # Detect changes
                old_constituents = self.constituents.get(index_code, [])
                change_summary = self._detect_changes(index_code, old_constituents, new_constituents)
                
                if change_summary:
                    self._archive_constituents(index_code, change_summary)
                
                # Update and save
                self.constituents[index_code] = new_constituents
                self._write_constituents_file(index_code, new_constituents)
                
                # Update reverse index
                self._rebuild_reverse_index()
                self._write_reverse_index()
                
                logger.info(f"Fetched {len(new_constituents)} constituents for {index_code}")
                
                return True, {
                    "success": True,
                    "index_code": index_code,
                    "constituent_count": len(new_constituents),
                    "change_summary": change_summary
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {index_code}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    import time
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Error fetching constituents: {e}")
        
        return False, {"error": "All retries exhausted"}
    
    def _detect_changes(self, index_code: str, 
                       old_constituents: List[Constituent],
                       new_constituents: List[Constituent]) -> Optional[Dict]:
        """Detect changes in constituents"""
        if not old_constituents:
            return None
        
        old_symbols = set(c.symbol for c in old_constituents)
        new_symbols = set(c.symbol for c in new_constituents)
        
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        
        if not added and not removed:
            return None
        
        return {
            "index_code": index_code,
            "added_count": len(added),
            "removed_count": len(removed),
            "added_symbols": sorted(list(added)),
            "removed_symbols": sorted(list(removed))
        }
    
    def _archive_constituents(self, index_code: str, change_summary: Dict) -> None:
        """Archive old constituents version"""
        const_file = self.CONSTITUENTS_DIR / f"{index_code}.json"
        
        if not const_file.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"{index_code}_{timestamp}.json.gz"
        
        try:
            with open(const_file, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Archived {index_code} constituents to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive: {e}")
    
    def _write_constituents_file(self, index_code: str, 
                                constituents: List[Constituent]) -> None:
        """Write constituents for a specific index"""
        filepath = self.CONSTITUENTS_DIR / f"{index_code}.json"
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "index_code": index_code,
                "total_constituents": len(constituents)
            },
            "constituents": [c.to_dict() for c in constituents]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write constituents file: {e}")
    
    def _write_reverse_index(self) -> None:
        """Write reverse index (symbol -> indices)"""
        try:
            with open(self.REVERSE_INDEX, 'w') as f:
                json.dump(self.reverse_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write reverse index: {e}")
    
    def _rebuild_reverse_index(self) -> None:
        """Rebuild reverse index from current constituents"""
        reverse = {}
        
        for index_code, constituents in self.constituents.items():
            for const in constituents:
                if const.symbol not in reverse:
                    reverse[const.symbol] = []
                reverse[const.symbol].append(index_code)
        
        self.reverse_index = reverse
    
    # =========================================================================
    # LOADING & QUERYING
    # =========================================================================
    
    def _load_constituents(self) -> None:
        """Load constituents from files"""
        if not self.CONSTITUENTS_DIR.exists():
            return
        
        try:
            for const_file in self.CONSTITUENTS_DIR.glob("*.json"):
                try:
                    with open(const_file, 'r') as f:
                        data = json.load(f)
                    
                    index_code = data['metadata']['index_code']
                    constituents = [
                        Constituent(**c) for c in data.get('constituents', [])
                    ]
                    self.constituents[index_code] = constituents
                    
                except Exception as e:
                    logger.warning(f"Error loading {const_file}: {e}")
            
            logger.info(f"Loaded constituents for {len(self.constituents)} indices")
            
        except Exception as e:
            logger.error(f"Failed to load constituents: {e}")
    
    def _load_reverse_index(self) -> None:
        """Load reverse index"""
        if not self.REVERSE_INDEX.exists():
            self._rebuild_reverse_index()
            return
        
        try:
            with open(self.REVERSE_INDEX, 'r') as f:
                self.reverse_index = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load reverse index: {e}")
            self._rebuild_reverse_index()
    
    def _load_history(self) -> List[Dict]:
        """Load history snapshots"""
        if not self.HISTORY_FILE.exists():
            return []
        
        try:
            with open(self.HISTORY_FILE, 'r') as f:
                history = json.load(f)
            self.snapshots = [IndexConstituentsSnapshot(**s) for s in history]
            return history
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []
    
    def get_constituents(self, index_code: str) -> List[Constituent]:
        """Get constituents for an index"""
        return self.constituents.get(index_code.upper(), [])
    
    def get_constituent_symbols(self, index_code: str) -> List[str]:
        """Get constituent symbols for an index"""
        return [c.symbol for c in self.get_constituents(index_code)]
    
    def get_indices_for_symbol(self, symbol: str) -> List[str]:
        """Get indices that contain a symbol"""
        return self.reverse_index.get(symbol.upper(), [])
    
    def get_all_constituents(self) -> List[str]:
        """Get all unique constituent symbols"""
        return sorted(list(self.reverse_index.keys()))
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        total_constituents = sum(len(c) for c in self.constituents.values())
        unique_symbols = len(self.reverse_index)
        
        return {
            "total_indices": len(self.constituents),
            "total_constituents": total_constituents,
            "unique_symbols": unique_symbols,
            "last_updated": self._get_last_update_time()
        }
    
    def _get_last_update_time(self) -> Optional[str]:
        """Get last update time"""
        if self.CONSTITUENTS_DIR.exists():
            files = list(self.CONSTITUENTS_DIR.glob("*.json"))
            if files:
                mtime = max(f.stat().st_mtime for f in files)
                return datetime.fromtimestamp(mtime).isoformat()
        return None
    
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

def get_nse_index_constituents_manager() -> NSEIndexConstituentsManager:
    """Get singleton instance"""
    return NSEIndexConstituentsManager.get_instance()
