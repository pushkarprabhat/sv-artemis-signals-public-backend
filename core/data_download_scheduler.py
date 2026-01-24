# core/data_download_scheduler.py
# Manages scheduled downloads across all timeframes
# Runs background process to keep data fresh
# Respects market hours and implements exponential backoff for errors

import threading
import time
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Optional, List, Callable
from pathlib import Path
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BASE_DIR, TIMEFRAMES
from core.zerodha_download_status_manager import (
    get_zerodha_download_status_manager,
    DownloadStatusEnum,
    DataSourceType
)
from universe.symbols import load_universe
from utils.logger import logger
from utils.failure_logger import record_failure
from utils.instrument_exceptions import add_to_exceptions

logger = logging.getLogger(__name__)


class ScheduleFrequency(Enum):
    """Download frequency"""
    EVERY_15MIN = 15 * 60  # seconds
    EVERY_30MIN = 30 * 60
    EVERY_HOUR = 60 * 60
    EVERY_4HOURS = 4 * 60 * 60
    DAILY = 24 * 60 * 60
    WEEKLY = 7 * 24 * 60 * 60


class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"  # 6 AM - 9:15 AM
    REGULAR = "regular"  # 9:15 AM - 3:30 PM
    POST_MARKET = "post_market"  # 3:30 PM - 11:59 PM
    CLOSED = "closed"  # 12 AM - 6 AM
    WEEKEND = "weekend"


class DataDownloadScheduler:
    """
    Manages background data downloads for all symbols and timeframes
    
    Features:
    - Market-aware scheduling (respects market hours)
    - Parallel downloads with configurable workers
    - Error tracking and exponential backoff
    - Persistent state tracking
    - Real-time status reporting
    - Supports all timeframes (15m, 30m, 60m, day, week, month)
    
    Architecture:
    - Main scheduler thread monitors time and triggers downloads
    - Worker threads execute actual downloads
    - Status manager tracks progress
    - Session tracking prevents redundant downloads
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Market hours (IST timezone)
    MARKET_HOURS = {
        'pre_market': (dt_time(6, 0), dt_time(9, 15)),
        'regular': (dt_time(9, 15), dt_time(15, 30)),
        'post_market': (dt_time(15, 30), dt_time(23, 59)),
    }
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.data_dir = BASE_DIR / "data"
        self.state_file = self.data_dir / ".download_scheduler_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Status management
        self.status_manager = get_zerodha_download_status_manager()
        
        # Scheduler state
        self.is_running = False
        self.pause_requested = False
        self.stop_requested = False
        
        # Configuration
        self.max_workers = 4  # Parallel downloads
        self.batch_size = 50  # Symbols per batch
        
        # Download scheduling
        self.last_download_times: Dict[str, datetime] = {}  # symbol -> last_download_time
        self.download_frequencies: Dict[str, ScheduleFrequency] = {}  # symbol -> frequency
        
        # Error handling
        self.error_counts: Dict[str, int] = {}  # symbol -> error_count
        self.max_retries = 3
        self.backoff_factor = 2.0  # Exponential backoff
        
        # Threads
        self.scheduler_thread: Optional[threading.Thread] = None
        self.download_threads: Dict[str, threading.Thread] = {}
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Callbacks for UI updates
        self.on_download_start: Optional[Callable] = None
        self.on_download_complete: Optional[Callable] = None
        self.on_status_update: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Load persisted state
        self._load_state()
        
        self._initialized = True
    
    def _load_state(self):
        """Load persisted scheduler state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Load last download times
                for symbol, ts_str in data.get('last_download_times', {}).items():
                    self.last_download_times[symbol] = datetime.fromisoformat(ts_str)
                
                logger.info(f"Loaded scheduler state for {len(self.last_download_times)} symbols")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="scheduler_state_load_failed", details=str(e))
            except Exception:
                pass
    
    def _save_state(self):
        """Persist scheduler state"""
        try:
            data = {
                'last_download_times': {
                    k: v.isoformat() for k, v in self.last_download_times.items()
                },
                'last_save': datetime.now().isoformat(),
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="scheduler_state_save_failed", details=str(e))
            except Exception:
                pass
    
    def start(self, max_workers: int = 4):
        """Start the download scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.stop_requested = False
        self.pause_requested = False
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Download scheduler started with {max_workers} workers")
    
    def stop(self):
        """Stop the download scheduler"""
        if not self.is_running:
            return
        
        self.stop_requested = True
        self.is_running = False
        
        # Wait for scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Download scheduler stopped")
    
    def pause(self):
        """Pause downloads (can be resumed)"""
        self.pause_requested = True
        logger.info("Downloads paused")
    
    def resume(self):
        """Resume paused downloads"""
        self.pause_requested = False
        logger.info("Downloads resumed")
    
    def _scheduler_loop(self):
        """Main scheduler loop - runs in background thread"""
        while not self.stop_requested:
            try:
                if not self.pause_requested:
                    self._check_and_schedule_downloads()
                
                # Check every 30 seconds
                time.sleep(30)
            
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                try:
                    record_failure(symbol=None, exchange=None, reason="scheduler_loop_error", details=str(e))
                except Exception:
                    pass
                time.sleep(60)  # Back off on error
    
    def _check_and_schedule_downloads(self):
        """Check if downloads are needed and schedule them"""
        try:
            # Load symbols
            universe = load_universe()
            symbols = universe['Symbol'].tolist()
            
            # Determine what needs downloading
            downloads_needed = []
            
            for symbol in symbols:
                if self._should_download(symbol):
                    downloads_needed.append(symbol)
            
            if downloads_needed:
                logger.info(f"Scheduling downloads for {len(downloads_needed)} symbols")
                self._schedule_batch_downloads(downloads_needed)
        
        except Exception as e:
            logger.error(f"Error in scheduling: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="scheduling_error", details=str(e))
            except Exception:
                pass
    
    def _should_download(self, symbol: str) -> bool:
        """Check if a symbol needs downloading"""
        # Default: download if never attempted
        if symbol not in self.last_download_times:
            return True
        
        # Default frequency: intraday every 30 min during market hours
        market_session = self._get_market_session()
        
        if market_session == MarketSession.CLOSED or market_session == MarketSession.WEEKEND:
            # Daily at market close
            return (datetime.now() - self.last_download_times[symbol]) > timedelta(hours=24)
        else:
            # Every 30 minutes during market hours
            return (datetime.now() - self.last_download_times[symbol]) > timedelta(minutes=30)
    
    def _get_market_session(self) -> MarketSession:
        """Determine current market session"""
        import datetime as dt_module
        
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()
        
        # Check weekend
        if current_day >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.WEEKEND
        
        # Check market hours
        if current_time < dt_time(6, 0):
            return MarketSession.CLOSED
        elif dt_time(6, 0) <= current_time < dt_time(9, 15):
            return MarketSession.PRE_MARKET
        elif dt_time(9, 15) <= current_time < dt_time(15, 30):
            return MarketSession.REGULAR
        elif dt_time(15, 30) <= current_time < dt_time(23, 59, 59):
            return MarketSession.POST_MARKET
        else:
            return MarketSession.CLOSED
    
    def _schedule_batch_downloads(self, symbols: List[str]):
        """Schedule downloads for a batch of symbols and trigger signal scan after completion"""
        try:
            from core.downloader import download_price_data
            from core.pairs import scan_all_strategies
            import json
            import pandas as pd
            from config import SIGNALS_PATH, SIGNAL_SCAN_INTERVALS
        except ImportError as e:
            logger.error(f"Failed to import download or signal modules: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="import_error", details=str(e))
            except Exception:
                pass
            return
        if self.on_download_start:
            self.on_download_start({
                'total_symbols': len(symbols),
                'timestamp': datetime.now().isoformat(),
            })
        futures = {}
        for symbol in symbols:
            future = self.executor.submit(self._download_symbol_with_retry, symbol)
            futures[symbol] = future
        completed = 0
        for symbol, future in futures.items():
            try:
                result = future.result(timeout=300)  # 5 minutes timeout
                if result:
                    self.last_download_times[symbol] = datetime.now()
                    self.error_counts[symbol] = 0
                    completed += 1
                    if self.on_status_update:
                        self.on_status_update({
                            'symbol': symbol,
                            'status': 'complete',
                            'timestamp': datetime.now().isoformat(),
                        })
            except Exception as e:
                self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
                logger.error(f"Download failed for {symbol}: {e}")
                try:
                    record_failure(symbol=symbol, exchange=None, reason="batch_download_exception", details=str(e))
                except Exception:
                    pass
                # Add to instrument exceptions if retries exhausted
                try:
                    if self.error_counts[symbol] >= self.max_retries:
                        add_to_exceptions(symbol)
                except Exception:
                    pass
                if self.on_error:
                    self.on_error({
                        'symbol': symbol,
                        'error': str(e),
                        'error_count': self.error_counts[symbol],
                        'timestamp': datetime.now().isoformat(),
                    })
        self._save_state()
        if self.on_download_complete:
            self.on_download_complete({
                'total_symbols': len(symbols),
                'completed': completed,
                'failed': len(symbols) - completed,
                'timestamp': datetime.now().isoformat(),
            })
        # === AUTOMATED SIGNAL SCAN & ENRICHMENT ===
        try:
            # For Shivaansh & Krishaansh — this line pays their fees
            logger.info("[AUTOMATION] Running scan_all_strategies after data refresh...")
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
                signals_df['source'] = 'DataDownloadScheduler'
                signals_df.to_json(SIGNALS_PATH, orient='records', indent=2)
                logger.info(f"[AUTOMATION] Signals saved to {SIGNALS_PATH} ({len(signals_df)} signals)")
            else:
                logger.warning("[AUTOMATION] No signals generated after data refresh.")
        except Exception as e:
            logger.error(f"[AUTOMATION] Signal enrichment failed: {e}")
        # === END AUTOMATION ===
    
    def _download_symbol_with_retry(self, symbol: str) -> bool:
        """Download a symbol with retry logic"""
        from core.downloader import download_price_data
        
        max_retries = self.max_retries
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if download_price_data(symbol):
                    return True
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential backoff
                    backoff_time = self.backoff_factor ** retry_count
                    logger.warning(f"Download retry {retry_count}/{max_retries} for {symbol} after {backoff_time}s")
                    time.sleep(backoff_time)
                else:
                    logger.error(f"Download failed for {symbol} after {max_retries} retries: {e}")
                    try:
                        record_failure(symbol=symbol, exchange=None, reason="download_retries_exhausted", details=str(e))
                    except Exception:
                        pass
                    try:
                        add_to_exceptions(symbol)
                    except Exception:
                        pass
                    return False
        
        return False
    
    def trigger_immediate_download(self, symbols: Optional[List[str]] = None):
        """Trigger immediate download (not scheduled)"""
        if not symbols:
            universe = load_universe()
            symbols = universe['Symbol'].tolist()
        
        logger.info(f"Triggering immediate download for {len(symbols)} symbols")
        self._schedule_batch_downloads(symbols)
    
    def get_download_progress(self) -> Dict:
        """Get current download progress"""
        return {
            'is_running': self.is_running,
            'is_paused': self.pause_requested,
            'max_workers': self.max_workers,
            'last_download_times': {
                k: v.isoformat() for k, v in self.last_download_times.items()
            },
            'error_counts': self.error_counts,
            'timestamp': datetime.now().isoformat(),
        }
    
    def set_callback_on_download_start(self, callback: Callable):
        """Set callback when download batch starts"""
        self.on_download_start = callback
    
    def set_callback_on_download_complete(self, callback: Callable):
        """Set callback when download batch completes"""
        self.on_download_complete = callback
    
    def set_callback_on_status_update(self, callback: Callable):
        """Set callback for individual symbol completion"""
        self.on_status_update = callback
    
    def set_callback_on_error(self, callback: Callable):
        """Set callback for errors"""
        self.on_error = callback


def get_data_download_scheduler() -> DataDownloadScheduler:
    """Get or create singleton instance"""
    return DataDownloadScheduler()
