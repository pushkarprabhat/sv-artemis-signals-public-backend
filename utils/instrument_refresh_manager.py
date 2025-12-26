"""
Instrument Refresh Manager
Handles daily instrument master refresh with freshness checking.
Ensures instruments are always up-to-date before downloads.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import pandas as pd
from kiteconnect import KiteConnect

from config import BASE_DIR
from utils.logger import logger

# Metadata storage location
INSTRUMENT_METADATA_DIR = BASE_DIR / "universe" / "metadata"
INSTRUMENT_METADATA_DIR.mkdir(parents=True, exist_ok=True)

INSTRUMENT_FRESHNESS_FILE = INSTRUMENT_METADATA_DIR / "instrument_freshness.json"
INSTRUMENT_REFRESH_LOCK_FILE = INSTRUMENT_METADATA_DIR / ".instrument_refresh.lock"

# Freshness threshold: instruments older than this need refresh
INSTRUMENT_FRESHNESS_THRESHOLD_HOURS = 24  # Refresh if older than 24 hours


@dataclass
class InstrumentFreshness:
    """Track instrument master freshness"""
    last_refresh_utc: Optional[str] = None  # ISO format datetime
    exchange_count: int = 0  # Number of exchanges refreshed
    total_instruments: int = 0  # Total instruments cached
    exchanges_refreshed: Dict[str, str] = None  # {exchange: timestamp}
    refresh_status: str = "unknown"  # unknown, fresh, stale, failed
    
    def to_dict(self) -> Dict:
        return {
            "last_refresh_utc": self.last_refresh_utc,
            "exchange_count": self.exchange_count,
            "total_instruments": self.total_instruments,
            "exchanges_refreshed": self.exchanges_refreshed or {},
            "refresh_status": self.refresh_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "InstrumentFreshness":
        return cls(
            last_refresh_utc=data.get("last_refresh_utc"),
            exchange_count=data.get("exchange_count", 0),
            total_instruments=data.get("total_instruments", 0),
            exchanges_refreshed=data.get("exchanges_refreshed", {}),
            refresh_status=data.get("refresh_status", "unknown")
        )
    
    def is_fresh(self, threshold_hours: int = INSTRUMENT_FRESHNESS_THRESHOLD_HOURS) -> bool:
        """Check if instruments are fresh"""
        if not self.last_refresh_utc:
            return False
        
        try:
            last_refresh = datetime.fromisoformat(self.last_refresh_utc)
            age = datetime.utcnow() - last_refresh
            return age < timedelta(hours=threshold_hours)
        except Exception as e:
            logger.warning(f"Could not parse instrument freshness timestamp: {e}")
            return False
    
    def get_age_hours(self) -> Optional[float]:
        """Get age of instruments in hours"""
        if not self.last_refresh_utc:
            return None
        
        try:
            last_refresh = datetime.fromisoformat(self.last_refresh_utc)
            age = datetime.utcnow() - last_refresh
            return age.total_seconds() / 3600
        except Exception as e:
            logger.warning(f"Could not calculate instrument age: {e}")
            return None


class InstrumentRefreshManager:
    """
    Manages instrument master data refresh with freshness checking.
    
    Features:
    - Check freshness of instrument data
    - Refresh from Kite API if stale
    - Track refresh status
    - Per-exchange refresh tracking
    - Thread-safe refresh with lock file
    """
    
    def __init__(self):
        self.metadata_dir = INSTRUMENT_METADATA_DIR
        self.freshness_file = INSTRUMENT_FRESHNESS_FILE
        self.lock_file = INSTRUMENT_REFRESH_LOCK_FILE
    
    def load_freshness(self) -> InstrumentFreshness:
        """Load current freshness status"""
        try:
            if self.freshness_file.exists():
                with open(self.freshness_file, 'r') as f:
                    data = json.load(f)
                    return InstrumentFreshness.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load instrument freshness: {e}")
        
        return InstrumentFreshness()
    
    def save_freshness(self, freshness: InstrumentFreshness) -> bool:
        """Save freshness status"""
        try:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(self.freshness_file, 'w') as f:
                json.dump(freshness.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Could not save instrument freshness: {e}")
            return False
    
    def is_fresh(self, threshold_hours: int = INSTRUMENT_FRESHNESS_THRESHOLD_HOURS) -> Tuple[bool, Optional[float]]:
        """
        Check if instruments are fresh
        
        Returns:
            (is_fresh: bool, age_hours: Optional[float])
        """
        freshness = self.load_freshness()
        is_fresh = freshness.is_fresh(threshold_hours)
        age_hours = freshness.get_age_hours()
        
        return is_fresh, age_hours
    
    def needs_refresh(self, threshold_hours: int = INSTRUMENT_FRESHNESS_THRESHOLD_HOURS) -> bool:
        """Check if refresh is needed"""
        is_fresh, _ = self.is_fresh(threshold_hours)
        return not is_fresh
    
    def get_status(self) -> Dict:
        """Get detailed freshness status"""
        freshness = self.load_freshness()
        is_fresh, age_hours = self.is_fresh()
        
        return {
            "is_fresh": is_fresh,
            "age_hours": age_hours,
            "last_refresh": freshness.last_refresh_utc,
            "total_instruments": freshness.total_instruments,
            "exchange_count": freshness.exchange_count,
            "exchanges": freshness.exchanges_refreshed or {},
            "status": freshness.refresh_status
        }
    
    def _acquire_lock(self, timeout: int = 30) -> bool:
        """Acquire refresh lock to prevent concurrent refreshes"""
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                # Try to create lock file exclusively
                self.lock_file.touch(exist_ok=False)
                return True
            except FileExistsError:
                # Check if lock is stale (older than 10 minutes)
                if self.lock_file.exists():
                    age = time.time() - self.lock_file.stat().st_mtime
                    if age > 600:  # 10 minutes
                        logger.warning("Found stale lock file, removing...")
                        try:
                            self.lock_file.unlink()
                        except:
                            pass
                time.sleep(0.1)
        
        return False
    
    def _release_lock(self) -> None:
        """Release refresh lock"""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.warning(f"Could not remove lock file: {e}")
    
    def refresh_instruments(self, kite: KiteConnect, exchanges: Optional[List[str]] = None) -> Tuple[bool, Dict]:
        """
        Refresh instrument master from Kite API
        
        Args:
            kite: KiteConnect instance
            exchanges: List of exchanges to refresh (None = all)
        
        Returns:
            (success: bool, stats: Dict)
        """
        if not self._acquire_lock():
            logger.warning("Could not acquire refresh lock - another refresh may be in progress")
            return False, {"error": "Could not acquire lock"}
        
        try:
            logger.info("\n" + "="*80)
            logger.info("INSTRUMENT MASTER REFRESH")
            logger.info("="*80)
            
            if exchanges is None:
                exchanges = ['NSE', 'BSE', 'NFO', 'MCX', 'NCDEX', 'CDS']
            
            stats = {
                "success": False,
                "exchanges_refreshed": {},
                "total_instruments": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "errors": []
            }
            
            # Refresh each exchange
            for exchange in exchanges:
                try:
                    logger.info(f"\n[REFRESH] Fetching {exchange} instruments from Kite API...")
                    
                    # Fetch from Kite
                    instruments = kite.instruments(exchange)
                    
                    if not instruments or len(instruments) == 0:
                        logger.warning(f"  ⚠️  No instruments returned for {exchange}")
                        continue
                    
                    # Convert to DataFrame for easier handling
                    df = pd.DataFrame(instruments)
                    
                    # Fix date columns that Kite API returns as date objects
                    # pyarrow/parquet can't handle datetime.date objects
                    date_columns = ['expiry']
                    for col in date_columns:
                        if col in df.columns:
                            try:
                                # Convert to string format (ISO 8601) for storage
                                df[col] = df[col].astype(str)
                            except Exception:
                                pass  # Column might not have dates, skip
                    
                    logger.info(f"  ✅ Retrieved {len(df)} instruments")
                    logger.info(f"     Columns: {list(df.columns)[:5]}...")
                    
                    # Save to local cache
                    cache_dir = self.metadata_dir / "instruments" / exchange
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as parquet and JSON
                    parquet_file = cache_dir / "instruments.parquet"
                    json_file = cache_dir / "instruments.json"
                    
                    df.to_parquet(parquet_file, index=False)
                    df.to_json(json_file, orient='records', indent=2)
                    
                    logger.info(f"  📁 Saved to {parquet_file}")
                    
                    stats["exchanges_refreshed"][exchange] = datetime.utcnow().isoformat()
                    stats["total_instruments"] += len(df)
                    
                except Exception as e:
                    error_msg = str(e)
                    # Silently skip exchanges that are not available (403 Access Denied, etc)
                    if "AccessDenied" in error_msg or "403" in error_msg:
                        logger.warning(f"  ⏭️  {exchange} not accessible (permission denied) - skipping")
                    elif "no instruments" in error_msg.lower():
                        logger.warning(f"  ⏭️  {exchange} returned no data - skipping")
                    else:
                        logger.error(f"  ❌ Failed to refresh {exchange}: {e}")
                        stats["errors"].append(f"{exchange}: {str(e)}")
                    continue
            
            if not stats["exchanges_refreshed"]:
                logger.error("❌ REFRESH FAILED - No exchanges were refreshed")
                stats["success"] = False
                return False, stats
            
            # Update freshness metadata
            freshness = InstrumentFreshness(
                last_refresh_utc=stats["timestamp"],
                exchange_count=len(stats["exchanges_refreshed"]),
                total_instruments=stats["total_instruments"],
                exchanges_refreshed=stats["exchanges_refreshed"],
                refresh_status="fresh"
            )
            
            if self.save_freshness(freshness):
                logger.info(f"\n✅ REFRESH COMPLETE")
                logger.info(f"   Exchanges: {len(stats['exchanges_refreshed'])}")
                logger.info(f"   Total instruments: {stats['total_instruments']}")
                stats["success"] = True
                return True, stats
            else:
                logger.warning("⚠️  Refresh succeeded but failed to save metadata")
                stats["success"] = False
                return False, stats
        
        finally:
            self._release_lock()
    
    def ensure_fresh(self, kite: KiteConnect, force: bool = False, 
                     threshold_hours: int = INSTRUMENT_FRESHNESS_THRESHOLD_HOURS) -> Tuple[bool, Dict]:
        """
        Ensure instruments are fresh, refresh if needed
        
        Args:
            kite: KiteConnect instance
            force: Force refresh regardless of freshness
            threshold_hours: Consider instruments stale if older than this
        
        Returns:
            (success: bool, stats: Dict)
        """
        freshness = self.load_freshness()
        is_fresh = freshness.is_fresh(threshold_hours)
        
        logger.info("\n[CHECK] Instrument freshness check:")
        
        if freshness.last_refresh_utc:
            age_hours = freshness.get_age_hours()
            logger.info(f"  Last refresh: {freshness.last_refresh_utc}")
            logger.info(f"  Age: {age_hours:.1f} hours")
        else:
            logger.info(f"  Never refreshed")
        
        if force:
            logger.info(f"  ⚠️  Force refresh requested")
            return self.refresh_instruments(kite)
        
        if is_fresh:
            logger.info(f"  ✅ Instruments are fresh (threshold: {threshold_hours}h)")
            return True, {
                "success": True,
                "action": "skipped",
                "reason": "instruments_fresh",
                "status": freshness.to_dict()
            }
        else:
            logger.info(f"  ⚠️  Instruments are stale (> {threshold_hours}h old)")
            return self.refresh_instruments(kite)


def get_instrument_refresh_manager() -> InstrumentRefreshManager:
    """Get singleton instrument refresh manager instance"""
    if not hasattr(get_instrument_refresh_manager, '_instance'):
        get_instrument_refresh_manager._instance = InstrumentRefreshManager()
    return get_instrument_refresh_manager._instance
