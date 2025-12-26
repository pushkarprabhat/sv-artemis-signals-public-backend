"""
Background Task Manager - Handles data refresh, missing data detection, universe updates
Integrates with enriched_downloader.py for ALL instrument downloads
Runs continuously without blocking the Streamlit UI

ARCHITECTURE:
✅ Daily 8:45 AM: Refresh enriched instruments (app_kite_universe.csv)
✅ Daily 5:45 PM: Download all market data using enriched master
✅ Friday 3:35 PM: Extra refresh of weekly expiry contracts
✅ 25th of month 3:35 PM: Extra refresh of monthly expiry contracts
✅ Preserves all existing data accuracy mechanisms (checksums, dedup)
✅ Dual-mode: Manual buttons (data_manager) + Automatic jobs

INTEGRATION:
- EnrichedDownloaderManager provides get_all_symbols_for_download()
- EnrichedInstrumentManager tracks M/W expiry by exchange
- Existing downloader.py functions unchanged (backward compatible)
"""

import threading
import time
import schedule
import datetime as dt
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
import pandas as pd
from config import BASE_DIR
from utils.logger import logger
import utils.helpers

# Global task manager instance
_task_manager: Optional['BackgroundTaskManager'] = None


class BackgroundTaskManager:
    """Manages background tasks for data refresh and universe updates"""
    
    def __init__(self):
        """Initialize background task manager"""
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.tasks: Dict[str, Dict] = {}
        self.schedule = schedule.Scheduler()
        
        # Track download progress
        self.download_queue: List[str] = []  # Queue of symbols waiting to download
        self.download_progress: Dict = {
            'current_symbol': None,
            'total_queued': 0,
            'downloaded': 0,
            'failed': 0,
            'last_run': None,
            'total_missing_5min': 0,
            'total_missing_day': 0
        }
        
        self._setup_schedules()
        logger.info("BackgroundTaskManager initialized")
    
    def _setup_schedules(self):
        """Setup all scheduled tasks"""
        # ===== ENRICHED INSTRUMENT INTEGRATION =====
        # Daily 8:45 AM: Refresh enriched instruments from Kite API
        self.schedule.every().day.at("08:45").do(self._refresh_enriched_instruments)
        
        # Daily 5:45 PM: Download all market data using enriched master
        self.schedule.every().day.at("17:45").do(self._download_all_enriched_data)
        
        # Friday 3:35 PM: Extra refresh of weekly expiry contracts
        self.schedule.every().friday.at("15:35").do(self._refresh_weekly_expiries)
        
        # 25th of month at 3:35 PM: Extra refresh of monthly expiry contracts
        # Note: schedule library doesn't natively support "day of month", using daily check
        self.schedule.every().day.at("15:35").do(self._refresh_monthly_expiries_check)
        
        # Daily 2:00 AM: Verify data integrity
        self.schedule.every().day.at("02:00").do(self._verify_data_integrity)
        
        # ===== LEGACY TASKS (preserved for backward compatibility) =====
        # Universe refresh: Once per day at 8:30 AM (5 min before enriched refresh)
        self.schedule.every().day.at("08:30").do(self._refresh_universe)
        
        # Expired contract purge: Once per day at 8:45 AM (after universe refresh)
        self.schedule.every().day.at("08:45").do(self._purge_expired_contracts)
        
        # Auto-detect missing data instruments: Once per day at 9:00 AM
        self.schedule.every().day.at("09:00").do(self._detect_missing_data_instruments)
        
        # Missing data check: Every 30 minutes
        self.schedule.every(30).minutes.do(self._auto_download_missing_data)
        
        # Timeframe data refresh: Every 5 minutes during market hours (9:15 AM - 4:00 PM)
        # Note: Live tick data comes through websockets, this is for OHLC timeframe data
        self.schedule.every(5).minutes.do(self._refresh_timeframe_data)
        
        logger.info("Scheduled tasks set up: Universe@8:30AM, ExpirePurge@8:45AM, DataDetect@9:00AM, Missing@30min, Timeframe@5min")
    
    def _refresh_universe(self):
        """Refresh the persistent universe from Kite API"""
        try:
            from core.universe_manager import get_universe_manager
            logger.info("[EARTH] [BACKGROUND] Starting universe refresh...")
            
            universe_mgr = get_universe_manager()
            universe_df = universe_mgr.get_universe(force_refresh=True)
            
            if universe_df is not None and not universe_df.empty:
                logger.info(f"[OK] [BACKGROUND] Universe refreshed: {len(universe_df)} instruments")
                self.tasks['universe_refresh'] = {
                    'status': 'success',
                    'timestamp': dt.datetime.now(),
                    'count': len(universe_df)
                }
            else:
                logger.warning("[BACKGROUND] Universe refresh returned empty")
                self.tasks['universe_refresh'] = {
                    'status': 'empty',
                    'timestamp': dt.datetime.now(),
                    'count': 0
                }
        except Exception as e:
            logger.error(f"[FAIL] [BACKGROUND] Universe refresh failed: {e}")
            self.tasks['universe_refresh'] = {
                'status': 'error',
                'timestamp': dt.datetime.now(),
                'error': str(e)
            }
    
    def _purge_expired_contracts(self):
        """Purge expired futures and options contracts"""
        try:
            from services.expired_contract_manager import get_expired_contract_manager
            logger.info("[PURGE] [BACKGROUND] Starting expired contract purge...")
            
            ecm = get_expired_contract_manager()
            stats = ecm.purge_expired_contracts(archive_before_purge=True, dry_run=False)
            
            logger.info(f"[OK] [BACKGROUND] Purge complete: Futures {stats['futures_purged']}, "
                       f"Options {stats['options_purged']}, Freed {stats['data_freed_mb']:.2f} MB")
            
            self.tasks['expired_purge'] = {
                'status': 'success',
                'timestamp': dt.datetime.now(),
                'futures_purged': stats['futures_purged'],
                'options_purged': stats['options_purged'],
                'data_freed_mb': stats['data_freed_mb'],
                'symbols_purged': len(stats['purged_symbols'])
            }
        except Exception as e:
            logger.error(f"[FAIL] [BACKGROUND] Expired contract purge failed: {e}")
            self.tasks['expired_purge'] = {
                'status': 'error',
                'timestamp': dt.datetime.now(),
                'error': str(e)
            }
    
    def _detect_missing_data_instruments(self):
        """Auto-detect instruments with missing data and flag for blocklist"""
        try:
            from utils.instrument_data_checker import get_data_checker
            from services.blocklist_manager import get_blocklist_manager
            logger.info("[DETECT] [BACKGROUND] Starting missing data detection...")
            
            dc = get_data_checker()
            bm = get_blocklist_manager()
            
            # Scan all instruments
            missing = dc.find_missing_data_instruments()
            partial = dc.find_partial_data_instruments()
            
            logger.info(f"[DETECT] [BACKGROUND] Found {len(missing)} with no data, {len(partial)} with partial data")
            
            # Auto-block instruments with no data (they've been consistent missing)
            blocked_count = 0
            for item in missing:
                symbol = item['symbol']
                if not bm.is_blocked(symbol):
                    # Check if this is a legitimate instrument or a failed download
                    # Only auto-block if it's been consistently missing (handled by Q3 logic)
                    # For now, just flag it in the system
                    logger.info(f"[FLAG] Missing data instrument: {symbol}")
            
            self.tasks['missing_data_detection'] = {
                'status': 'success',
                'timestamp': dt.datetime.now(),
                'no_data_count': len(missing),
                'partial_data_count': len(partial),
                'auto_blocked': blocked_count
            }
        except Exception as e:
            logger.error(f"[FAIL] [BACKGROUND] Missing data detection failed: {e}")
            self.tasks['missing_data_detection'] = {
                'status': 'error',
                'timestamp': dt.datetime.now(),
                'error': str(e)
            }
    
    def _auto_download_missing_data(self):
        """Automatically download missing data - respects blocklist and tracks failures"""
        try:
            from core.data_manager import get_data_manager
            from services.blocklist_manager import get_blocklist_manager
            logger.info("[DATA] [BACKGROUND] Starting missing data detection...")
            
            data_mgr = get_data_manager()
            bm = get_blocklist_manager()
            
            # Check missing 5-minute data
            missing_5min = data_mgr.detect_missing_data('5minute')
            missing_5min_symbols = missing_5min.get('missing_symbols', [])
            
            # Check missing daily data
            missing_day = data_mgr.detect_missing_data('day')
            missing_day_symbols = missing_day.get('missing_symbols', [])
            
            # Filter out blocked instruments
            missing_5min_symbols = [s for s in missing_5min_symbols if not bm.is_blocked(s)]
            missing_day_symbols = [s for s in missing_day_symbols if not bm.is_blocked(s)]
            
            # Update progress tracking
            self.download_queue = missing_5min_symbols[:15] + missing_day_symbols[:10]  # Increased limits
            self.download_progress['total_queued'] = len(self.download_queue)
            self.download_progress['total_missing_5min'] = len(missing_5min_symbols)
            self.download_progress['total_missing_day'] = len(missing_day_symbols)
            self.download_progress['last_run'] = dt.datetime.now()
            self.download_progress['downloaded'] = 0
            self.download_progress['failed'] = 0
            
            downloaded_count = 0
            failed_count = 0
            failed_symbols = {}  # Track failures for auto-blocking
            
            logger.info(f"[DATA] [BACKGROUND] Queue: {len(missing_5min_symbols)} missing 5-min, {len(missing_day_symbols)} missing daily")
            
            # Download missing 5-minute data (increased from 5 to 15)
            for i, sym in enumerate(missing_5min_symbols[:15]):
                try:
                    self.download_progress['current_symbol'] = f"{sym} (5-min) [{i+1}/15]"
                    result = data_mgr.download_price_data(sym, intervals=['5minute'], days_back=365)
                    if result.get('5minute', False):
                        downloaded_count += 1
                        self.download_progress['downloaded'] += 1
                        logger.info(f"[OK] [{i+1}/15] Downloaded 5-minute for {sym}")
                    else:
                        failed_count += 1
                        self.download_progress['failed'] += 1
                        failed_symbols[sym] = failed_symbols.get(sym, 0) + 1
                        logger.warning(f"[WARN] [{i+1}/15] Failed to download 5-minute for {sym}")
                        
                        # Q3: Auto-block after 3 failed attempts
                        if failed_symbols[sym] >= 3:
                            bm.add_to_blocklist(sym, reason="No Data", notes="Auto-blocked after 3 failed attempts")
                            logger.warning(f"[AUTO-BLOCK] {sym} blocked after 3 failed attempts")
                except Exception as e:
                    failed_count += 1
                    self.download_progress['failed'] += 1
                    failed_symbols[sym] = failed_symbols.get(sym, 0) + 1
                    logger.error(f"Error downloading {sym}: {e}")
                    
                time.sleep(0.5)  # Rate limiting
            
            # Download missing daily data (increased from 3 to 10)
            for i, sym in enumerate(missing_day_symbols[:10]):
                try:
                    self.download_progress['current_symbol'] = f"{sym} (daily) [{i+1}/10]"
                    result = data_mgr.download_price_data(sym, intervals=['day'], days_back=365)
                    if result.get('day', False):
                        downloaded_count += 1
                        self.download_progress['downloaded'] += 1
                        logger.info(f"[OK] [{i+1}/10] Downloaded daily for {sym}")
                    else:
                        failed_count += 1
                        self.download_progress['failed'] += 1
                        failed_symbols[sym] = failed_symbols.get(sym, 0) + 1
                        logger.warning(f"[WARN] [{i+1}/10] Failed to download daily for {sym}")
                        
                        # Q3: Auto-block after 3 failed attempts
                        if failed_symbols[sym] >= 3:
                            bm.add_to_blocklist(sym, reason="No Data", notes="Auto-blocked after 3 failed attempts")
                            logger.warning(f"[AUTO-BLOCK] {sym} blocked after 3 failed attempts")
                except Exception as e:
                    failed_count += 1
                    self.download_progress['failed'] += 1
                    failed_symbols[sym] = failed_symbols.get(sym, 0) + 1
                    logger.error(f"Error downloading {sym}: {e}")
                    
                time.sleep(0.5)  # Rate limiting
            
            self.download_progress['current_symbol'] = None
            logger.info(f"[DATA] [BACKGROUND] Download cycle complete: Downloaded {downloaded_count}, Failed {failed_count}")
            self.tasks['missing_data'] = {
                'status': 'complete',
                'timestamp': dt.datetime.now(),
                'downloaded': downloaded_count,
                'failed': failed_count,
                'missing_5min': len(missing_5min_symbols),
                'missing_day': len(missing_day_symbols),
                'auto_blocked': len([s for s in failed_symbols if failed_symbols[s] >= 3]),
                'progress': self.download_progress.copy()
            }
        except Exception as e:
            logger.error(f"[FAIL] [BACKGROUND] Missing data auto-download failed: {e}")
            self.tasks['missing_data'] = {
                'status': 'error',
                'timestamp': dt.datetime.now(),
                'error': str(e)
            }
    
    def _refresh_timeframe_data(self):
        """Refresh timeframe data (5-min, 15-min, etc.) during market hours
        
        Note: Live tick data comes through websockets. This refreshes OHLC timeframe data.
        """
        try:
            # Check if market is open
            from utils.market_hours import is_market_open
            if not is_market_open():
                logger.debug("[BACKGROUND] Market closed, skipping timeframe data refresh")
                return
            
            from core.data_manager import get_data_manager
            logger.info("[DATA] [BACKGROUND] Starting timeframe data refresh...")
            
            data_mgr = get_data_manager()
            
            # Get list of top symbols to refresh (most frequently used)
            top_symbols = self._get_top_symbols(limit=5)
            
            refreshed_count = 0
            for sym in top_symbols:
                try:
                    result = data_mgr.download_price_data(sym, intervals=['5minute', '15minute'], days_back=1)
                    if any(result.values()):
                        refreshed_count += 1
                        logger.debug(f"[OK] Refreshed timeframe data for {sym}")
                except Exception as e:
                    logger.warning(f"[WARN] Failed to refresh {sym}: {e}")
                
                time.sleep(0.3)  # Rate limiting
            
            logger.info(f"[DATA] [BACKGROUND] Timeframe data refresh complete: {refreshed_count}/{len(top_symbols)} symbols")
            self.tasks['timeframe_data'] = {
                'status': 'complete',
                'timestamp': dt.datetime.now(),
                'refreshed': refreshed_count
            }
        except Exception as e:
            logger.warning(f"[WARN] [BACKGROUND] Timeframe data refresh skipped: {e}")
    
    def _get_top_symbols(self, limit: int = 5) -> List[str]:
        """Get most recently accessed symbols"""
        try:
            # Check which data files were most recently modified
            top_symbols = []
            for tf_dir in ['5minute', 'day']:
                folder = BASE_DIR / tf_dir
                if folder.exists():
                    files = sorted(
                        folder.glob('*.parquet'),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )[:limit]
                    for f in files:
                        sym = f.stem
                        if sym not in top_symbols:
                            top_symbols.append(sym)
            
            return top_symbols[:limit]
        except Exception as e:
            logger.warning(f"Could not get top symbols: {e}")
            return []
    
    def _aggregate_intervals(self):
        """Automatically aggregate 5-minute data to higher intervals"""
        try:
            from pages.data_management import aggregate_all_intervals
            
            logger.info("[DATA] [BACKGROUND] Starting automatic interval aggregation...")
            
            folder_5min = BASE_DIR / '5minute'
            if not folder_5min.exists():
                logger.debug("[BACKGROUND] No 5-minute data to aggregate")
                return
            
            # Get list of recently updated files (to avoid aggregating everything every time)
            files = sorted(
                folder_5min.glob('*.parquet'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]  # Only aggregate 10 most recent files
            
            aggregated_count = 0
            for file in files:
                try:
                    symbol = file.stem
                    if aggregate_all_intervals(symbol):
                        aggregated_count += 1
                    time.sleep(0.1)  # Small delay between aggregations
                except Exception as e:
                    logger.debug(f"Aggregation skipped for {symbol}: {e}")
            
            logger.info(f"[OK] [BACKGROUND] Aggregation complete: {aggregated_count} symbols updated")
            self.tasks['aggregation'] = {
                'status': 'complete',
                'timestamp': dt.datetime.now(),
                'aggregated': aggregated_count
            }
        except Exception as e:
            logger.warning(f"[WARN] [BACKGROUND] Aggregation failed: {e}")
            self.tasks['aggregation'] = {
                'status': 'error',
                'timestamp': dt.datetime.now(),
                'error': str(e)
            }
    
    def start(self):
        """Start background task manager"""
        if self.is_running:
            logger.warning("Background task manager already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        logger.info("[OK] Background task manager started")
    
    def stop(self):
        """Stop background task manager"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("[STOP] Background task manager stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        logger.info("[BACKGROUND] Scheduler thread started")
        while self.is_running:
            try:
                self.schedule.run_pending()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"[BACKGROUND] Scheduler error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def get_status(self) -> Dict:
        """Get current status of all background tasks"""
        return {
            'is_running': self.is_running,
            'tasks': self.tasks,
            'download_progress': self.download_progress,
            'pending_jobs': len(self.schedule.jobs)
        }
    
    def get_task_status(self, task_name: str) -> Optional[Dict]:
        """Get status of a specific task"""
        return self.tasks.get(task_name)
    
    def get_download_progress(self) -> Dict:
        """Get current download progress"""
        return self.download_progress.copy()
    
    # =========================================================================
    # ENRICHED INSTRUMENT INTEGRATION TASKS
    # =========================================================================
    
    def _refresh_enriched_instruments(self):
        """
        DAILY 8:45 AM
        Refresh enriched instruments from Kite API
        Updates app_kite_universe.csv with latest contracts/expiries
        """
        try:
            logger.info("⏰ [SCHEDULED] Starting enriched instruments refresh (8:45 AM)...")
            
            from core.enriched_instrument_manager import get_enriched_instrument_manager
            
            mgr = get_enriched_instrument_manager()
            success = mgr.refresh_from_enriched()
            
            if success:
                logger.info("✅ Enriched instruments refreshed successfully")
                stats = mgr.get_statistics()
                logger.info(f"   Total instruments: {stats['total_instruments']}")
                logger.info(f"   By exchange: {stats['by_exchange']}")
            else:
                logger.error("❌ Failed to refresh enriched instruments")
                
        except Exception as e:
            logger.error(f"Error in enriched instrument refresh: {e}", exc_info=True)
    
    def _download_all_enriched_data(self):
        """
        DAILY 5:45 PM
        Download all market data using enriched instruments master
        
        Uses enriched_downloader.download_all_from_enriched():
        ✅ Incremental (only new data since last download)
        ✅ All instruments from app_kite_universe.csv
        ✅ Respects batching and API limits
        ✅ Preserves data accuracy mechanisms
        """
        try:
            logger.info("⏰ [SCHEDULED] Starting all data download (5:45 PM)...")
            logger.info("   Using enriched instruments master (app_kite_universe.csv)")
            
            from core.enriched_downloader import get_enriched_downloader
            
            downloader = get_enriched_downloader()
            result = downloader.download_all_from_enriched(
                force_refresh=False,  # Incremental only
                parallel=True,
                max_workers=4
            )
            
            logger.info(f"Download result: {result['status']}")
            if 'total_attempted' in result:
                logger.info(f"   Total: {result['total_attempted']} | Success: {result['total_success']} | Skipped: {result['total_skipped']}")
            
            # Update status
            self.tasks['enriched_download'] = {
                'last_run': dt.datetime.now(),
                'status': result.get('status'),
                'total': result.get('total_attempted'),
                'success': result.get('total_success')
            }
            
        except Exception as e:
            logger.error(f"Error in enriched data download: {e}", exc_info=True)
    
    def _refresh_weekly_expiries(self):
        """
        FRIDAY 3:35 PM
        Extra refresh of weekly expiry contracts for NFO
        Weekly options expire every Friday
        """
        try:
            logger.info("⏰ [SCHEDULED] Starting weekly expiry refresh (Friday 3:35 PM)...")
            
            from core.enriched_downloader import get_enriched_downloader
            
            downloader = get_enriched_downloader()
            result = downloader.download_weekly_expiries(exchange='NFO')
            
            logger.info(f"Weekly expiry refresh: {result['success']}/{result['total']} successful")
            
            self.tasks['weekly_expiry_refresh'] = {
                'last_run': dt.datetime.now(),
                'success': result['success'],
                'total': result['total']
            }
            
        except Exception as e:
            logger.error(f"Error in weekly expiry refresh: {e}", exc_info=True)
    
    def _refresh_monthly_expiries_check(self):
        """
        MONTHLY (25TH at 3:35 PM)
        Extra refresh of monthly expiry contracts for NFO
        Monthly contracts expire near end of month
        
        Only runs on the 25th (checked daily)
        """
        today = dt.datetime.now()
        
        # Only run on 25th of month
        if today.day != 25:
            return
        
        try:
            logger.info("⏰ [SCHEDULED] Starting monthly expiry refresh (25th 3:35 PM)...")
            
            from core.enriched_downloader import get_enriched_downloader
            
            downloader = get_enriched_downloader()
            result = downloader.download_monthly_expiries(exchange='NFO')
            
            logger.info(f"Monthly expiry refresh: {result['success']}/{result['total']} successful")
            
            self.tasks['monthly_expiry_refresh'] = {
                'last_run': dt.datetime.now(),
                'success': result['success'],
                'total': result['total']
            }
            
        except Exception as e:
            logger.error(f"Error in monthly expiry refresh: {e}", exc_info=True)
    
    def _verify_data_integrity(self):
        """
        DAILY 2:00 AM
        Verify data integrity and checksums
        Runs during off-hours to minimize impact
        """
        try:
            logger.info("⏰ [SCHEDULED] Starting data integrity verification (2:00 AM)...")
            
            from core.corporate_actions import get_corporate_actions_manager
            
            mgr = get_corporate_actions_manager()
            issues = mgr.verify_all_checksums()
            
            if issues:
                logger.warning(f"⚠️  Found {len(issues)} data integrity issues")
                for symbol, error in list(issues.items())[:5]:  # Log first 5
                    logger.warning(f"   {symbol}: {error}")
            else:
                logger.info("✅ All data integrity checks passed")
            
            self.tasks['integrity_check'] = {
                'last_run': dt.datetime.now(),
                'issues_found': len(issues) if issues else 0
            }
            
        except Exception as e:
            logger.error(f"Error in data integrity verification: {e}", exc_info=True)


def get_background_manager() -> BackgroundTaskManager:
    """Get or create background task manager"""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


def start_background_tasks():
    """Start background task manager"""
    manager = get_background_manager()
    if not manager.is_running:
        manager.start()


def stop_background_tasks():
    """Stop background task manager"""
    manager = get_background_manager()
    if manager.is_running:
        manager.stop()
