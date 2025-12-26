"""
EOD Maintenance Scheduler - Friday Exception List Refresh
==========================================================
Automates cleanup of exception lists every Friday at market close
Ensures exception lists remain clean and accurate
"""

import json
import os
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional
from config import BASE_DIR
from utils.logger import logger


class EODMaintenanceScheduler:
    """
    Manages EOD (End of Day) maintenance tasks.
    Primary: Refresh exception list every Friday EOD
    """
    
    EXCEPTION_FILE = "universe/metadata/exception_instruments.json"
    FAILED_FILE = "universe/metadata/failed_instruments.json"
    MAINTENANCE_LOG = "universe/metadata/maintenance_log.json"
    
    # Friday is day 4 (0=Monday, 4=Friday)
    CLEANUP_WEEKDAY = 4
    CLEANUP_HOUR = 15  # 3 PM (after market close)
    
    def __init__(self):
        self.base_dir = Path(BASE_DIR)
        self.exception_file_path = self.base_dir / self.EXCEPTION_FILE
        self.failed_file_path = self.base_dir / self.FAILED_FILE
        self.maintenance_log_path = self.base_dir / self.MAINTENANCE_LOG
        self.maintenance_log: Dict = {}
        self._load_maintenance_log()
    
    def _load_maintenance_log(self):
        """Load maintenance history"""
        if not self.maintenance_log_path.exists():
            return
        
        try:
            with open(self.maintenance_log_path, 'r') as f:
                self.maintenance_log = json.load(f)
        except Exception as e:
            logger.warning(f"[MAINTENANCE] Failed to load log: {e}")
            self.maintenance_log = {}
    
    def _save_maintenance_log(self):
        """Save maintenance history"""
        try:
            self.maintenance_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.maintenance_log_path, 'w') as f:
                json.dump(self.maintenance_log, f, indent=2)
        except Exception as e:
            logger.error(f"[MAINTENANCE] Failed to save log: {e}")
    
    def should_refresh_exceptions(self) -> bool:
        """Check if Friday EOD exception refresh should run"""
        now = dt.datetime.now()
        
        # Check if today is Friday
        if now.weekday() != self.CLEANUP_WEEKDAY:
            return False
        
        # Check if we already ran today
        today = now.strftime('%Y-%m-%d')
        if today in self.maintenance_log:
            last_run = self.maintenance_log[today]
            logger.debug(f"[MAINTENANCE] Already ran today at {last_run}")
            return False
        
        # Check if it's after market close hour
        if now.hour < self.CLEANUP_HOUR:
            logger.debug(f"[MAINTENANCE] Too early (before {self.CLEANUP_HOUR}:00)")
            return False
        
        return True
    
    def refresh_exception_list(self, keep_legacy: bool = True) -> Dict:
        """
        Refresh exception list on Friday EOD.
        
        Strategy:
        - Clear temporary exceptions (API errors)
        - Keep permanent exceptions (NO_DATA, DELISTED)
        - Optionally preserve legacy symbol tracking
        - Reset failed instruments list (temporary failures)
        
        Args:
            keep_legacy: Keep legacy symbol tracking (default: True)
        
        Returns:
            Statistics dict with cleaned counts
        """
        try:
            stats = {
                'timestamp': dt.datetime.now().isoformat(),
                'exceptions_before': 0,
                'exceptions_kept': 0,
                'exceptions_cleared': 0,
                'failed_before': 0,
                'failed_cleared': 0,
                'legacy_preserved': 0
            }
            
            # Load current exceptions
            exceptions = self._load_exceptions()
            stats['exceptions_before'] = len(exceptions)
            
            # Load current failed
            failed = self._load_failed()
            stats['failed_before'] = len(failed)
            
            # Separate permanent vs temporary exceptions
            permanent = {}
            legacy_count = 0
            
            for symbol, info in exceptions.items():
                reason = info.get('reason', 'UNKNOWN')
                
                # Keep permanent exceptions
                if reason in ('NO_DATA', 'DELISTED', 'NOT_TRADABLE', 'INVALID_TOKEN'):
                    permanent[symbol] = info
                    if info.get('legacy', False):
                        legacy_count += 1
                
                # Clear temporary exceptions (e.g., INACTIVE after 60 days)
                elif reason == 'INACTIVE':
                    timestamp = info.get('timestamp', '')
                    if timestamp:
                        try:
                            exc_date = dt.datetime.fromisoformat(timestamp).date()
                            age_days = (dt.datetime.now().date() - exc_date).days
                            if age_days > 60:
                                # Clear old inactive
                                continue
                        except:
                            pass
                    permanent[symbol] = info
            
            # Save cleaned exceptions
            self._save_exceptions(permanent)
            
            # Clear failed instruments (all temporary)
            self._save_failed({})
            
            # Update stats
            stats['exceptions_kept'] = len(permanent)
            stats['exceptions_cleared'] = stats['exceptions_before'] - len(permanent)
            stats['failed_cleared'] = stats['failed_before']
            stats['legacy_preserved'] = legacy_count
            
            # Log this maintenance run
            today = dt.datetime.now().strftime('%Y-%m-%d')
            self.maintenance_log[today] = dt.datetime.now().isoformat()
            self._save_maintenance_log()
            
            logger.info(f"[MAINTENANCE] Friday EOD refresh complete: "
                       f"Exceptions {stats['exceptions_cleared']} cleared, "
                       f"Failed {stats['failed_cleared']} cleared, "
                       f"Legacy {stats['legacy_preserved']} preserved")
            
            return stats
        
        except Exception as e:
            logger.error(f"[MAINTENANCE] Exception refresh failed: {e}")
            raise
    
    def _load_exceptions(self) -> Dict:
        """Load exception instruments"""
        if not self.exception_file_path.exists():
            return {}
        
        try:
            with open(self.exception_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[MAINTENANCE] Failed to load exceptions: {e}")
            return {}
    
    def _load_failed(self) -> Dict:
        """Load failed instruments"""
        if not self.failed_file_path.exists():
            return {}
        
        try:
            with open(self.failed_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[MAINTENANCE] Failed to load failed: {e}")
            return {}
    
    def _save_exceptions(self, data: Dict):
        """Save exception instruments"""
        try:
            self.exception_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.exception_file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[MAINTENANCE] Failed to save exceptions: {e}")
    
    def _save_failed(self, data: Dict):
        """Save failed instruments"""
        try:
            self.failed_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failed_file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[MAINTENANCE] Failed to save failed: {e}")
    
    def get_maintenance_summary(self) -> Dict:
        """Get last maintenance run summary"""
        if not self.maintenance_log:
            return {'status': 'Never run'}
        
        last_date = sorted(self.maintenance_log.keys())[-1]
        return {
            'last_run': last_date,
            'last_timestamp': self.maintenance_log[last_date],
            'total_runs': len(self.maintenance_log)
        }


# Global singleton
_scheduler_instance: Optional[EODMaintenanceScheduler] = None


def get_eod_scheduler() -> EODMaintenanceScheduler:
    """Get global EOD scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = EODMaintenanceScheduler()
    return _scheduler_instance
