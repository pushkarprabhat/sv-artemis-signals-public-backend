"""
Download State Manager
Tracks data availability, ranges, and refresh history per symbol/interval
Enables smart incremental downloads with full fallback
"""

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from config import BASE_DIR
from utils.logger import logger


@dataclass
class DataRange:
    """Tracks data availability for a symbol/interval"""
    symbol: str
    interval: str
    first_date: Optional[str] = None  # YYYY-MM-DD
    last_date: Optional[str] = None   # YYYY-MM-DD
    total_rows: int = 0
    last_refresh: Optional[str] = None  # ISO timestamp
    status: str = "unknown"  # unknown, partial, complete, failed
    failure_reason: Optional[str] = None  # Reason for failure if status=failed


class DownloadStateManager:
    """Manages download state and metadata tracking"""
    
    def __init__(self):
        self.state_dir = BASE_DIR / "universe" / "metadata"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / "download_state.json"
        self.masterlist_file = self.state_dir / "masterlist_version.json"
        self.download_log_file = self.state_dir / "download_log.json"
        
        self.state = self._load_state()
        self.masterlist_info = self._load_masterlist_info()
    
    def _load_state(self) -> Dict:
        """Load state from file or create new"""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception as e:
                logger.error(f"[STATE] Error loading state: {e}")
                return {"symbols": {}, "metadata": {"created": dt.datetime.now().isoformat()}}
        return {"symbols": {}, "metadata": {"created": dt.datetime.now().isoformat()}}
    
    def _load_masterlist_info(self) -> Dict:
        """Load masterlist version info"""
        if self.masterlist_file.exists():
            try:
                return json.loads(self.masterlist_file.read_text())
            except Exception as e:
                logger.error(f"[MASTERLIST] Error loading version info: {e}")
                return {"version": None, "count": 0, "last_refresh": None}
        return {"version": None, "count": 0, "last_refresh": None}
    
    def _save_state(self):
        """Save state to file"""
        try:
            self.state_file.write_text(json.dumps(self.state, indent=2, default=str))
        except Exception as e:
            logger.error(f"[STATE] Error saving state: {e}")
    
    def update_masterlist_info(self, version: str, count: int):
        """Update instrument masterlist version and count"""
        self.masterlist_info = {
            "version": version,
            "count": count,
            "last_refresh": dt.datetime.now().isoformat(),
            "expires_at": (dt.datetime.now() + dt.timedelta(days=1)).isoformat()
        }
        
        try:
            self.masterlist_file.write_text(json.dumps(self.masterlist_info, indent=2))
            logger.info(f"[MASTERLIST] Updated: v{version}, {count} instruments, expires: {self.masterlist_info['expires_at']}")
        except Exception as e:
            logger.error(f"[MASTERLIST] Error saving version: {e}")
    
    def should_refresh_masterlist(self) -> bool:
        """Check if masterlist should be refreshed"""
        if not self.masterlist_info.get("last_refresh"):
            return True
        
        try:
            last_refresh = dt.datetime.fromisoformat(self.masterlist_info["last_refresh"])
            age = dt.datetime.now() - last_refresh
            should_refresh = age > dt.timedelta(days=1)
            
            if should_refresh:
                logger.info(f"[MASTERLIST] Needs refresh (last: {age.days}d {age.seconds//3600}h ago)")
            return should_refresh
        except Exception:
            return True
    
    def get_data_range(self, symbol: str, interval: str) -> Optional[DataRange]:
        """Get data range for symbol/interval"""
        key = f"{symbol}_{interval}"
        if key in self.state.get("symbols", {}):
            data = self.state["symbols"][key]
            return DataRange(**data)
        return None
    
    def set_data_range(self, symbol: str, interval: str, first_date: Optional[str],
                      last_date: Optional[str], total_rows: int, status: str = "partial"):
        """Record data range for symbol/interval"""
        key = f"{symbol}_{interval}"
        self.state["symbols"][key] = {
            "symbol": symbol,
            "interval": interval,
            "first_date": first_date,
            "last_date": last_date,
            "total_rows": total_rows,
            "last_refresh": dt.datetime.now().isoformat(),
            "status": status
        }
        self._save_state()
    
    def mark_failed(self, symbol: str, interval: str, reason: str = "No data"):
        """Mark download as failed"""
        key = f"{symbol}_{interval}"
        self.state["symbols"][key] = {
            "symbol": symbol,
            "interval": interval,
            "first_date": None,
            "last_date": None,
            "total_rows": 0,
            "last_refresh": dt.datetime.now().isoformat(),
            "status": "failed",
            "failure_reason": reason
        }
        self._save_state()
    
    def get_download_recommendation(self, symbol: str, interval: str) -> Dict:
        """Get recommendation for what to download"""
        range_info = self.get_data_range(symbol, interval)
        
        if range_info is None:
            return {
                "action": "FULL_DOWNLOAD",
                "reason": "No existing data",
                "from_date": (dt.date.today() - dt.timedelta(days=365)).isoformat(),
                "to_date": dt.date.today().isoformat()
            }
        
        if range_info.status == "failed":
            return {
                "action": "SKIP",
                "reason": f"Previously failed: {range_info.__dict__.get('failure_reason', 'Unknown')}",
                "skip": True
            }
        
        if range_info.total_rows < 50:  # Suspicious small count
            return {
                "action": "FULL_DOWNLOAD",
                "reason": f"Too few rows ({range_info.total_rows}), likely incomplete",
                "from_date": (dt.date.today() - dt.timedelta(days=365)).isoformat(),
                "to_date": dt.date.today().isoformat()
            }
        
        # Incremental download - FIXED: Check if data is missing for ANY day (>=1 day old)
        # This handles daily data downloads and market holidays (3-4 days gap on weekends)
        if not range_info.last_date:
            return {
                "action": "FULL_DOWNLOAD",
                "reason": "No last_date recorded",
                "from_date": (dt.date.today() - dt.timedelta(days=365)).isoformat(),
                "to_date": dt.date.today().isoformat()
            }
        
        try:
            # Handle both string and date object
            if isinstance(range_info.last_date, str):
                last_date = dt.datetime.fromisoformat(range_info.last_date).date()
            else:
                last_date = range_info.last_date
        except (ValueError, TypeError) as e:
            logger.warning(f"[STATE] Invalid last_date for {symbol}/{interval}: {range_info.last_date}, error: {e}")
            return {
                "action": "FULL_DOWNLOAD",
                "reason": f"Invalid last_date format: {range_info.last_date}",
                "from_date": (dt.date.today() - dt.timedelta(days=365)).isoformat(),
                "to_date": dt.date.today().isoformat()
            }
        
        days_old = (dt.date.today() - last_date).days
        
        # Allow up to 5 days without data (covers weekend gaps: Fri to Mon, or holidays)
        # But if gap > 5 days, something is wrong or market hasn't opened
        if days_old >= 1 and days_old <= 5:
            # Normal gap - do incremental download
            return {
                "action": "INCREMENTAL_DOWNLOAD",
                "reason": f"Data {days_old} day(s) old - downloading from {last_date + dt.timedelta(days=1)}",
                "from_date": (last_date + dt.timedelta(days=1)).isoformat(),
                "to_date": dt.date.today().isoformat()
            }
        elif days_old > 5:
            # Larger gap - might be holiday or connection issue, do full re-download to be safe
            return {
                "action": "FULL_DOWNLOAD",
                "reason": f"Data {days_old} days old (>5 days, likely holiday/issue)",
                "from_date": (dt.date.today() - dt.timedelta(days=365)).isoformat(),
                "to_date": dt.date.today().isoformat()
            }
        
        return {
            "action": "SKIP",
            "reason": f"Data is current (today's close available)",
            "skip": True
        }
    
    def log_download_attempt(self, symbol: str, interval: str, success: bool, 
                            rows: int = 0, error: str = ""):
        """Log download attempt"""
        log_entry = {
            "timestamp": dt.datetime.now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "success": success,
            "rows": rows,
            "error": error if error else None
        }
        
        logs = []
        if self.download_log_file.exists():
            try:
                logs = json.loads(self.download_log_file.read_text())
            except:
                logs = []
        
        logs.append(log_entry)
        # Keep last 10000 entries
        logs = logs[-10000:]
        
        try:
            self.download_log_file.write_text(json.dumps(logs, indent=2))
        except Exception as e:
            logger.error(f"[LOG] Error logging attempt: {e}")
    
    def get_status_summary(self) -> Dict:
        """Get summary of download status"""
        symbols = self.state.get("symbols", {})
        
        total = len(symbols)
        complete = sum(1 for s in symbols.values() if s.get("status") == "complete")
        partial = sum(1 for s in symbols.values() if s.get("status") == "partial")
        failed = sum(1 for s in symbols.values() if s.get("status") == "failed")
        
        return {
            "total_symbols": total,
            "complete": complete,
            "partial": partial,
            "failed": failed,
            "completion_rate": (complete / total * 100) if total > 0 else 0,
            "masterlist_version": self.masterlist_info.get("version"),
            "masterlist_count": self.masterlist_info.get("count"),
            "masterlist_age_hours": self._get_masterlist_age_hours()
        }
    
    def _get_masterlist_age_hours(self) -> Optional[int]:
        """Get masterlist age in hours"""
        if not self.masterlist_info.get("last_refresh"):
            return None
        
        try:
            last_refresh = dt.datetime.fromisoformat(self.masterlist_info["last_refresh"])
            age = dt.datetime.now() - last_refresh
            return int(age.total_seconds() / 3600)
        except:
            return None
    
    def print_status_report(self):
        """Print comprehensive status report"""
        summary = self.get_status_summary()
        
        logger.info("="*80)
        logger.info("DOWNLOAD STATUS REPORT")
        logger.info("="*80)
        logger.info(f"Total symbols tracked: {summary['total_symbols']}")
        logger.info(f"  ✅ Complete: {summary['complete']}")
        logger.info(f"  ⚠️  Partial: {summary['partial']}")
        logger.info(f"  ❌ Failed: {summary['failed']}")
        logger.info(f"  📊 Completion rate: {summary['completion_rate']:.1f}%")
        logger.info("")
        logger.info(f"Instrument masterlist:")
        logger.info(f"  Version: {summary['masterlist_version'] or 'NOT LOADED'}")
        logger.info(f"  Instruments: {summary['masterlist_count']}")
        
        if summary['masterlist_age_hours'] is not None:
            logger.info(f"  Last refreshed: {summary['masterlist_age_hours']} hours ago")
            if summary['masterlist_age_hours'] > 24:
                logger.warning("  ⚠️  NEEDS REFRESH - Run full download")
        else:
            logger.warning("  ⚠️  NEVER REFRESHED - Run full download")
        
        logger.info("="*80)


# Singleton instance
_state_manager = None

def get_download_state_manager() -> DownloadStateManager:
    """Get or create download state manager"""
    global _state_manager
    if _state_manager is None:
        _state_manager = DownloadStateManager()
    return _state_manager
