# core/nse_securities_scheduler.py — Daily NSE Securities Refresh
# Schedules and manages automatic daily refresh of NSE securities list

import schedule
import time
import threading
from datetime import datetime, time as dt_time
from typing import Callable, Optional, Dict, List
import logging
from .nse_securities_manager import get_nse_securities_manager

logger = logging.getLogger(__name__)


class NSESecuritiesScheduler:
    """
    Manages automatic daily refresh of NSE securities
    
    Features:
    - Schedule daily refresh at specific time (default: market close 4 PM)
    - Manual refresh capability
    - Async background refresh
    - Status tracking
    """
    
    # Default refresh time: 4 PM (market close for NSE)
    DEFAULT_REFRESH_HOUR = 16
    DEFAULT_REFRESH_MINUTE = 0
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, refresh_hour: int = DEFAULT_REFRESH_HOUR, 
                 refresh_minute: int = DEFAULT_REFRESH_MINUTE):
        """
        Initialize scheduler
        
        Args:
            refresh_hour: Hour to refresh (0-23, default 16)
            refresh_minute: Minute to refresh (0-59, default 0)
        """
        self.refresh_hour = refresh_hour
        self.refresh_minute = refresh_minute
        self.manager = get_nse_securities_manager()
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_refresh_time: Optional[str] = None
        self.last_refresh_status: Optional[Dict] = None
        self.refresh_callbacks: list[Callable] = []
        
        logger.info(f"NSESecuritiesScheduler initialized for daily refresh at {refresh_hour:02d}:{refresh_minute:02d}")
    
    @classmethod
    def get_instance(cls) -> "NSESecuritiesScheduler":
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def start(self) -> None:
        """Start the scheduler in background"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
            name="NSESecuritiesScheduler"
        )
        self.scheduler_thread.start()
        logger.info("NSE Securities Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler"""
        self.is_running = False
        logger.info("NSE Securities Scheduler stop requested")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop"""
        # Schedule daily refresh
        schedule.every().day.at(
            f"{self.refresh_hour:02d}:{self.refresh_minute:02d}"
        ).do(self._refresh_job)
        
        logger.info("Scheduler loop started")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        
        logger.info("Scheduler loop stopped")
    
    def _refresh_job(self) -> None:
        """Actual refresh job"""
        logger.info("Starting scheduled NSE securities refresh")
        self.refresh_now()
    
    def refresh_now(self, force: bool = True) -> Dict:
        """
        Perform refresh immediately
        
        Args:
            force: Force refresh even if data is fresh
        
        Returns:
            Status dictionary
        """
        try:
            # Fetch latest
            result = self.manager.fetch_latest(force_refresh=force)
            
            if result.success:
                # Save to persistence
                success, change_summary = self.manager.save_securities(result)
                
                status = {
                    "success": success,
                    "timestamp": result.timestamp,
                    "total_count": result.total_count,
                    "eq_count": result.eq_count,
                    "change_summary": change_summary,
                    "warnings": result.warnings
                }
            else:
                status = {
                    "success": False,
                    "error": result.error,
                    "warnings": result.warnings
                }
            
            self.last_refresh_time = datetime.now().isoformat()
            self.last_refresh_status = status
            
            # Call callbacks
            for callback in self.refresh_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Error calling refresh callback: {e}")
            
            logger.info(f"Refresh completed: {status}")
            return status
            
        except Exception as e:
            logger.error(f"Error during refresh: {e}")
            status = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.last_refresh_status = status
            return status
    
    def register_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Register a callback to be called after refresh
        
        Args:
            callback: Function to call with status dict
        """
        self.refresh_callbacks.append(callback)
        logger.info(f"Registered refresh callback: {callback.__name__}")
    
    def get_status(self) -> Dict:
        """Get current scheduler status"""
        return {
            "is_running": self.is_running,
            "scheduled_time": f"{self.refresh_hour:02d}:{self.refresh_minute:02d}",
            "last_refresh_time": self.last_refresh_time,
            "last_refresh_status": self.last_refresh_status
        }
    
    def set_refresh_time(self, hour: int, minute: int) -> None:
        """
        Update refresh time (requires restart)
        
        Args:
            hour: Hour (0-23)
            minute: Minute (0-59)
        """
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError(f"Invalid time: {hour:02d}:{minute:02d}")
        
        was_running = self.is_running
        if was_running:
            self.stop()
        
        self.refresh_hour = hour
        self.refresh_minute = minute
        
        if was_running:
            # Clear schedule and restart
            schedule.clear()
            self.start()
        
        logger.info(f"Refresh time updated to {hour:02d}:{minute:02d}")


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_nse_securities_scheduler() -> NSESecuritiesScheduler:
    """Get singleton instance of NSESecuritiesScheduler"""
    return NSESecuritiesScheduler.get_instance()
