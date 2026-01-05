# core/background_downloader.py — Background Data Download Scheduler
# Manages parallel downloads with market-aware scheduling
# Offline detection and automatic resume

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, List
from pathlib import Path
import json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.symbols_manager import get_symbols_manager, ExchangeMarketHours
from utils.helpers import kite as get_kite

logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    """Download job status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    OFFLINE = "offline"


class DownloadJob:
    """Individual download job tracking"""
    
    def __init__(self, job_id: str, symbol: str, exchange: str, intervals: List[str]):
        self.job_id = job_id
        self.symbol = symbol
        self.exchange = exchange
        self.intervals = intervals
        self.status = DownloadStatus.IDLE
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.progress = 0  # 0-100
        self.intervals_completed = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'intervals': self.intervals,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error,
            'progress': self.progress,
            'intervals_completed': self.intervals_completed
        }


class BackgroundDownloader:
    """
    Manages background downloads with market-aware scheduling
    Features:
    - Parallel downloads during market hours
    - Missing data checks during off-market hours
    - Offline detection and recovery
    - Persistent state across restarts
    - Thread-safe operations
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.state_file = Path("marketdata/download_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Core attributes
        self.status = DownloadStatus.IDLE
        self.is_online = True
        self.last_online_check = datetime.now()
        self.current_exchange = None
        self.current_mode = "offline_check"  # "live_pull" or "offline_check"
        
        # Download tracking
        self.jobs: Dict[str, DownloadJob] = {}
        self.active_downloads: Dict[str, threading.Thread] = {}
        self.executor: Optional[ThreadPoolExecutor] = None
        self.max_workers = 4
        
        # Market awareness
        self.symbols_manager = get_symbols_manager()
        self.market_hours = ExchangeMarketHours()
        
        # Callbacks
        self.on_status_change: Optional[Callable] = None
        self.on_job_update: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None
        
        # Control
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.download_intervals = ["15minute", "30minute", "60minute", "day"]
        self.check_interval_seconds = 60  # Check every minute
        self.offline_check_only = False  # Only check missing data when offline
        
        self._load_state()
        self._initialized = True
    
    def _load_state(self):
        """Load persistent state from disk"""
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text())
                self.status = DownloadStatus(data.get('status', 'idle'))
                self.is_online = data.get('is_online', True)
                self.current_mode = data.get('current_mode', 'offline_check')
                logger.info(f"Loaded state: {self.status.value}, online={self.is_online}, mode={self.current_mode}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save persistent state to disk"""
        try:
            state = {
                'status': self.status.value,
                'is_online': self.is_online,
                'current_mode': self.current_mode,
                'saved_at': datetime.now().isoformat()
            }
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def check_connectivity(self) -> bool:
        """Check if internet is available"""
        try:
            # Simple check: can we reach Zerodha API
            kite = get_kite()
            if kite is None:
                return False
            # Try a simple API call
            kite.margins()
            self.is_online = True
            self.last_online_check = datetime.now()
            logger.info("Network check: ONLINE")
            return True
        except Exception as e:
            self.is_online = False
            logger.warning(f"Network check: OFFLINE - {e}")
            return False
    
    def _determine_download_mode(self) -> str:
        """
        Determine what to download based on market hours
        Returns: "live_pull" (pull live data) or "offline_check" (check missing data only)
        """
        for exchange in ['NSE', 'MCX']:
            is_open, session, _ = self.market_hours.is_market_open(exchange)
            if is_open and session == 'Regular Trading':
                return "live_pull"
        
        return "offline_check"
    
    def _update_download_mode(self):
        """Update the download mode based on current market status"""
        new_mode = self._determine_download_mode()
        
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            logger.info(f"Download mode changed to: {new_mode}")
            self._save_state()
            
            if self.on_mode_change:
                self.on_mode_change(new_mode)
    
    def _set_status(self, new_status: DownloadStatus):
        """Update status with callback"""
        if new_status != self.status:
            self.status = new_status
            logger.info(f"Status changed to: {new_status.value}")
            self._save_state()
            
            if self.on_status_change:
                self.on_status_change(new_status.value)
    
    def start(self):
        """Start the background downloader scheduler"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Downloader already running")
            return
        
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Background downloader started")
    
    def stop(self):
        """Stop the background downloader"""
        self.stop_event.set()
        
        # Wait for scheduler to stop
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=False)
        
        self._set_status(DownloadStatus.STOPPED)
        logger.info("Background downloader stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop - runs in separate thread"""
        logger.info("Scheduler loop started")
        
        # Create executor for parallel downloads
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        while not self.stop_event.is_set():
            try:
                # 1. Check connectivity every minute
                if (datetime.now() - self.last_online_check).total_seconds() > 60:
                    is_online = self.check_connectivity()
                    if is_online != self.is_online:
                        self.is_online = is_online
                        self._save_state()
                        
                        if not is_online:
                            self._set_status(DownloadStatus.OFFLINE)
                            logger.warning("Internet connection lost")
                        else:
                            logger.info("Internet connection restored")
                
                # 2. Update download mode based on market hours
                self._update_download_mode()
                
                # 3. If online, check what to download
                if self.is_online:
                    self._set_status(DownloadStatus.RUNNING)
                    
                    # Get symbols to download
                    symbols_to_download = self._get_symbols_for_download()
                    
                    if symbols_to_download:
                        logger.info(f"Starting downloads: {len(symbols_to_download)} symbols, mode={self.current_mode}")
                        self._execute_downloads(symbols_to_download)
                    
                else:
                    # Offline - still check for missing data
                    if not self.offline_check_only:
                        self._set_status(DownloadStatus.OFFLINE)
                        logger.warning("OFFLINE - Only checking for missing data")
                        self.offline_check_only = True
                
                # 4. Sleep before next check
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                self._set_status(DownloadStatus.ERROR)
                time.sleep(5)  # Wait before retrying
    
    def _get_symbols_for_download(self) -> List[str]:
        """Get list of symbols to download based on current mode"""
        all_symbols = self.symbols_manager.get_all_symbols()
        symbols = []
        
        if self.current_mode == "live_pull":
            # Download all tracked symbols
            symbols.extend(all_symbols.get('NSE', {}).keys())
            symbols.extend(all_symbols.get('MCX', {}).keys())
        else:
            # offline_check mode: only check for missing data
            # This would check data folder and look for gaps
            symbols = self._find_symbols_with_missing_data()
        
        return symbols[:10]  # Limit to 10 per cycle
    
    def _find_symbols_with_missing_data(self) -> List[str]:
        """Find symbols that have missing data"""
        # Implementation would check data files and identify gaps
        # For now, return empty (can be enhanced)
        return []
    
    def _execute_downloads(self, symbols: List[str]):
        """Execute parallel downloads for symbols and trigger signal scan after completion"""
        try:
            from core.pairs import scan_all_strategies
            import json
            import pandas as pd
            from config import SIGNALS_PATH, SIGNAL_SCAN_INTERVALS
        except ImportError as e:
            logger.error(f"Failed to import signal modules: {e}")
            return
        if not self.executor:
            return
        futures = {}
        for symbol in symbols:
            job_id = f"{symbol}_{datetime.now().timestamp()}"
            job = DownloadJob(job_id, symbol, "NSE", self.download_intervals)
            self.jobs[job_id] = job
            future = self.executor.submit(self._download_symbol, job)
            futures[future] = job_id
        completed = 0
        for future in as_completed(futures):
            job_id = futures[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                if result:
                    self.jobs[job_id].status = DownloadStatus.IDLE
                    logger.info(f"Completed: {job_id}")
            except Exception as e:
                logger.error(f"Download failed for {job_id}: {e}")
                self.jobs[job_id].error = str(e)
                self.jobs[job_id].status = DownloadStatus.ERROR
            completed += 1
            if self.on_job_update:
                self.on_job_update(self.jobs[job_id].to_dict())
        # === AUTOMATED SIGNAL SCAN & ENRICHMENT ===
        try:
            # For Shivaansh & Krishaansh — this line pays their fees
            logger.info("[AUTOMATION] Running scan_all_strategies after background download...")
            all_signals = []
            for tf in SIGNAL_SCAN_INTERVALS:
                try:
                    df = scan_all_strategies(tf=tf)
                    if df is not None and not df.empty:
                        df['timeframe'] = tf
                        all_signals.append(df)
                except Exception as e:
                    logger.error(f"Signal scan failed for {tf}: {e}")
            if all_signals:
                signals_df = pd.concat(all_signals, ignore_index=True)
                signals_df['scan_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                signals_df['source'] = 'BackgroundDownloader'
                signals_df.to_json(SIGNALS_PATH, orient='records', indent=2)
                logger.info(f"[AUTOMATION] Signals saved to {SIGNALS_PATH} ({len(signals_df)} signals)")
            else:
                logger.warning("[AUTOMATION] No signals generated after background download.")
        except Exception as e:
            logger.error(f"[AUTOMATION] Signal enrichment failed: {e}")
        # === END AUTOMATION ===
    
    def _download_symbol(self, job: DownloadJob) -> bool:
        """Download data for a single symbol"""
        try:
            job.status = DownloadStatus.RUNNING
            job.started_at = datetime.now()
            
            if self.on_job_update:
                self.on_job_update(job.to_dict())
            
            # Import here to avoid circular dependency
            from core.downloader import download_price_data
            
            # Download based on mode
            if self.current_mode == "live_pull":
                # Full download
                success = download_price_data(job.symbol, force_refresh=False)
            else:
                # Only check for missing data
                success = self._check_missing_data(job.symbol)
            
            job.completed_at = datetime.now()
            job.progress = 100
            job.intervals_completed = len(job.intervals)
            
            return success
            
        except Exception as e:
            logger.error(f"Error downloading {job.symbol}: {e}")
            job.error = str(e)
            return False
    
    def _check_missing_data(self, symbol: str) -> bool:
        """Check and fill missing data for a symbol"""
        # Implementation would identify gaps in data files
        # and download only the missing periods
        logger.info(f"Checking missing data for {symbol}")
        return True
    
    def get_status(self) -> Dict:
        """Get current downloader status"""
        return {
            'status': self.status.value,
            'is_online': self.is_online,
            'current_mode': self.current_mode,
            'active_jobs': len([j for j in self.jobs.values() if j.status == DownloadStatus.RUNNING]),
            'completed_jobs': len([j for j in self.jobs.values() if j.status == DownloadStatus.IDLE]),
            'last_online_check': self.last_online_check.isoformat()
        }
    
    def get_market_status(self) -> Dict:
        """Get market status for all exchanges"""
        return {
            'NSE': self.market_hours.get_market_status('NSE'),
            'MCX': self.market_hours.get_market_status('MCX')
        }
    
    def set_download_config(self, intervals: List[str], max_workers: int = 4):
        """Configure download parameters"""
        self.download_intervals = intervals
        self.max_workers = max_workers
        logger.info(f"Updated config: intervals={intervals}, max_workers={max_workers}")
    
    def pause(self):
        """Pause downloads"""
        self._set_status(DownloadStatus.PAUSED)
        logger.info("Downloads paused")
    
    def resume(self):
        """Resume downloads"""
        if self.is_online:
            self._set_status(DownloadStatus.RUNNING)
        else:
            self._set_status(DownloadStatus.OFFLINE)
        logger.info("Downloads resumed")


def get_background_downloader() -> BackgroundDownloader:
    """Get singleton instance of BackgroundDownloader"""
    return BackgroundDownloader()
