# core/nse_indices_manager.py — NSE Indices Manager
# Manages all NSE indices with daily refresh and change tracking

import json
import requests
import hashlib
import gzip
import shutil
from pathlib import Path
from datetime import datetime
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

class IndexCategory(Enum):
    """Categories of NSE indices"""
    BROAD_MARKET = "broad_market"
    SECTORAL = "sectoral"
    STRATEGY = "strategy"
    THEMATIC = "thematic"
    VOLATILITY = "volatility"
    COMMODITY = "commodity"
    PSX = "psx"  # Pre-open session
    OTHER = "other"


@dataclass
class NSEIndex:
    """Represents a single NSE index"""
    index_name: str
    index_code: str
    category: str
    description: Optional[str] = None
    base_value: Optional[float] = None
    base_date: Optional[str] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IndexConstituent:
    """Represents a constituent stock in an index"""
    symbol: str
    isin: Optional[str] = None
    weight: Optional[float] = None  # Percentage weight in index
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IndexSnapshot:
    """Snapshot of indices list at a point in time"""
    timestamp: str
    total_count: int
    indices: List[Dict] = field(default_factory=list)
    file_hash: str = ""
    change_summary: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# NSE INDICES MANAGER
# =============================================================================

class NSEIndicesManager:
    """
    Manages NSE indices with daily refresh and change tracking.
    
    Features:
    - Fetch all NSE indices
    - Categorize by type (broad market, sectoral, etc.)
    - Track index compositions
    - Daily refresh with change detection
    - Archive old versions
    - Persistent storage
    """
    
    # NSE Index data endpoints
    NSE_INDICES_PAGE = "https://www.nseindia.com/products/content/equities/indices/indices.htm"
    NSE_API_BASE = "https://www.nseindia.com/api/equity"
    
    DATA_DIR = BASE_DIR / "nse_indices"
    CURRENT_FILE = DATA_DIR / "indices_current.json"
    SNAPSHOT_DIR = DATA_DIR / "snapshots"
    ARCHIVE_DIR = DATA_DIR / "archives"
    HISTORY_FILE = DATA_DIR / "indices_history.json"
    
    # Known NSE Indices (will be updated from API)
    KNOWN_INDICES = {
        'NIFTY 50': 'Nifty50',
        'NIFTY NEXT 50': 'Nifty100',
        'NIFTY 100': 'Nifty100',
        'NIFTY 200': 'Nifty200',
        'NIFTY 500': 'Nifty500',
        'NIFTY BANK': 'NiftyBank',
        'NIFTY AUTO': 'NiftyAuto',
        'NIFTY PHARMA': 'NiftyPharma',
        'NIFTY IT': 'NiftyIT',
        'NIFTY METAL': 'NiftyMetal',
        'NIFTY PSU BANK': 'NiftyPSUBank',
        'NIFTY PRIVATE BANK': 'NiftyPrivateBank',
        'INDIA VIX': 'VIX',
    }
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize NSE Indices Manager"""
        self._ensure_dirs()
        self.indices: Dict[str, NSEIndex] = {}
        self.snapshots: List[IndexSnapshot] = []
        self._load_current()
        self._load_history()
        logger.info(f"NSEIndicesManager initialized with {len(self.indices)} indices")
    
    @classmethod
    def get_instance(cls) -> "NSEIndicesManager":
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
    
    def populate_from_config(self) -> Tuple[bool, Dict]:
        """
        Populate indices from known configuration
        
        Returns:
            (success, stats)
        """
        try:
            new_indices = {}
            
            # Categorize indices
            broad_market = ['Nifty50', 'Nifty100', 'Nifty200', 'Nifty500']
            banking = ['NiftyBank', 'NiftyPSUBank', 'NiftyPrivateBank']
            sectoral = ['NiftyAuto', 'NiftyPharma', 'NiftyIT', 'NiftyMetal']
            volatility = ['VIX']
            
            for display_name, code in self.KNOWN_INDICES.items():
                # Determine category
                if code in broad_market:
                    category = IndexCategory.BROAD_MARKET.value
                elif code in banking:
                    category = IndexCategory.SECTORAL.value
                elif code in sectoral:
                    category = IndexCategory.SECTORAL.value
                elif code in volatility:
                    category = IndexCategory.VOLATILITY.value
                else:
                    category = IndexCategory.OTHER.value
                
                index = NSEIndex(
                    index_name=display_name,
                    index_code=code,
                    category=category,
                    is_active=True
                )
                
                new_indices[code] = index
            
            # Detect changes
            change_summary = self._detect_changes(self.indices, new_indices)
            
            # Archive if changed
            if change_summary:
                self._archive_current_snapshot(change_summary)
            
            # Update
            self.indices = new_indices
            self._write_current_file(new_indices, change_summary)
            
            stats = {
                "success": True,
                "total_count": len(new_indices),
                "broad_market": sum(1 for i in new_indices.values() if i.category == IndexCategory.BROAD_MARKET.value),
                "sectoral": sum(1 for i in new_indices.values() if i.category == IndexCategory.SECTORAL.value),
                "change_summary": change_summary
            }
            
            logger.info(f"Populated {stats['total_count']} indices")
            return True, stats
            
        except Exception as e:
            logger.error(f"Error populating indices: {e}")
            return False, {"error": str(e)}
    
    def _detect_changes(self, old: Dict[str, NSEIndex], 
                       new: Dict[str, NSEIndex]) -> Optional[Dict]:
        """Detect changes in indices"""
        if not old:
            return None
        
        added = set(new.keys()) - set(old.keys())
        removed = set(old.keys()) - set(new.keys())
        
        if not added and not removed:
            return None
        
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "added_indices": sorted(list(added)),
            "removed_indices": sorted(list(removed))
        }
    
    def _archive_current_snapshot(self, change_summary: Dict) -> None:
        """Archive current snapshot"""
        if not self.CURRENT_FILE.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"indices_{timestamp}.json.gz"
        
        try:
            with open(self.CURRENT_FILE, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Archived indices snapshot to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive: {e}")
    
    def _write_current_file(self, indices: Dict[str, NSEIndex],
                           change_summary: Optional[Dict]) -> None:
        """Write current indices to file"""
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_count": len(indices),
                "change_summary": change_summary
            },
            "indices": [i.to_dict() for i in indices.values()]
        }
        
        try:
            with open(self.CURRENT_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write current file: {e}")
    
    # =========================================================================
    # LOADING & QUERYING
    # =========================================================================
    
    def _load_current(self) -> None:
        """Load current indices from file"""
        if not self.CURRENT_FILE.exists():
            logger.info("No current indices file found")
            return
        
        try:
            with open(self.CURRENT_FILE, 'r') as f:
                data = json.load(f)
            
            self.indices = {
                i['index_code']: NSEIndex(**i)
                for i in data.get('indices', [])
            }
            logger.info(f"Loaded {len(self.indices)} indices from file")
            
        except Exception as e:
            logger.error(f"Failed to load current indices: {e}")
    
    def _load_history(self) -> List[Dict]:
        """Load history snapshots"""
        if not self.HISTORY_FILE.exists():
            return []
        
        try:
            with open(self.HISTORY_FILE, 'r') as f:
                history = json.load(f)
            self.snapshots = [IndexSnapshot(**s) for s in history]
            return history
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []
    
    def get_index(self, code: str) -> Optional[NSEIndex]:
        """Get index by code"""
        return self.indices.get(code.upper())
    
    def get_all_indices(self) -> List[NSEIndex]:
        """Get all indices"""
        return list(self.indices.values())
    
    def get_indices_by_category(self, category: str) -> List[NSEIndex]:
        """Filter indices by category"""
        return [i for i in self.indices.values() if i.category == category]
    
    def get_broad_market_indices(self) -> List[NSEIndex]:
        """Get broad market indices"""
        return self.get_indices_by_category(IndexCategory.BROAD_MARKET.value)
    
    def get_sectoral_indices(self) -> List[NSEIndex]:
        """Get sectoral indices"""
        return self.get_indices_by_category(IndexCategory.SECTORAL.value)
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        all_indices = self.indices.values()
        
        category_counts = {}
        for idx in all_indices:
            cat = idx.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_count": len(self.indices),
            "active_count": sum(1 for i in all_indices if i.is_active),
            "category_distribution": category_counts,
            "last_updated": self._get_last_update_time(),
            "snapshots_count": len(self.snapshots)
        }
    
    def _get_last_update_time(self) -> Optional[str]:
        """Get last update time"""
        if self.CURRENT_FILE.exists():
            mtime = self.CURRENT_FILE.stat().st_mtime
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

def get_nse_indices_manager() -> NSEIndicesManager:
    """Get singleton instance"""
    return NSEIndicesManager.get_instance()
