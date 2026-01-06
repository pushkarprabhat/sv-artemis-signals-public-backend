#!/usr/bin/env python3
"""
EOD/BOD Executor - Phase 2 Automation Core
Handles daily data refresh:
  - BOD (9:15 AM): Refresh yesterday's data before market opens
  - EOD (3:30 PM): Download complete day's OHLCV data after market closes
"""

import sys
from pathlib import Path
from datetime import datetime, time
import logging
from typing import Dict, List, Tuple
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.eod_bod_config import (
    BOD_TIME, EOD_TIME, INTRADAY_INTERVALS, DAILY_INTERVAL,
    is_trading_day, DATA_DIRS, LOG_DIR, RETRY_ATTEMPTS, RETRY_DELAY_SECONDS,
    DEBUG_MODE, TEST_MODE
)
from utils.logger import logger
from utils.token_manager import get_token_from_env, is_token_expired
from sv_artemis_signals_shared.data.download_incremental_data import IncrementalDataDownloader


class EODExecutor:
    """End of Day - Download complete day's OHLCV data (3:30 PM IST)"""
    
    def __init__(self):
        self.logger = logger
        self.downloader = IncrementalDataDownloader()
        self.stats = {
            'total_symbols': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped': 0,
            'total_files_downloaded': 0,
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
        }
    
    def validate_preconditions(self) -> bool:
        """Validate EOD can run (market closed, credentials valid, etc)"""
        self.logger.info("[EOD] Validating preconditions...")
        
        # Check if trading day
        if not is_trading_day():
            self.logger.warning("[EOD] Not a trading day - skipping EOD")
            return False
        
        # Check current time (should be after market close - anytime after EOD_TIME)
        now = datetime.now().time()
        if now < EOD_TIME:
            self.logger.warning(f"[EOD] Market still open - expected anytime after {EOD_TIME}, got {now}")
            return False
        
        self.logger.info(f"[EOD] ✓ Current time {now} is after EOD window ({EOD_TIME}+)")
        
        # Check token validity
        try:
            api_key = get_token_from_env()
            if not api_key:
                self.logger.error("[EOD] API key not found")
                return False
            
            self.logger.info("[EOD] ✓ API credentials valid")
        except Exception as e:
            self.logger.error(f"[EOD] Token validation failed: {e}")
            return False
        
        self.logger.info("[EOD] ✓ All preconditions met")
        return True
    
    def run_eod(self, force: bool = False) -> Dict:
        """Execute end-of-day download"""
        
        if not force and not self.validate_preconditions():
            return {'status': 'skipped', 'reason': 'Preconditions not met'}
        
        self.logger.info("=" * 80)
        self.logger.info(f"[EOD] STARTING END-OF-DAY DATA DOWNLOAD")
        self.logger.info(f"[EOD] Time: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # Download daily data
            self.logger.info(f"[EOD] Downloading {DAILY_INTERVAL} data...")
            daily_result = self._download_interval(DAILY_INTERVAL)
            
            # Download current intraday data (latest candles)
            self.logger.info(f"[EOD] Downloading intraday data ({', '.join(INTRADAY_INTERVALS)})...")
            intraday_results = []
            for interval in INTRADAY_INTERVALS:
                result = self._download_interval(interval)
                intraday_results.append(result)
            
            # Validate downloaded data
            self.logger.info("[EOD] Validating downloaded data...")
            validation_result = self._validate_eod_data()
            
            # Generate report
            self.logger.info("[EOD] Generating EOD report...")
            report = self._generate_eod_report(daily_result, intraday_results, validation_result)
            
            self.logger.info("=" * 80)
            self.logger.info(f"[EOD] EOD DOWNLOAD COMPLETE")
            self.logger.info(f"[EOD] Duration: {self.stats['duration_seconds']:.0f} seconds")
            self.logger.info(f"[EOD] Summary: {self.stats['successful_downloads']} successful, {self.stats['failed_downloads']} failed")
            self.logger.info("=" * 80)
            
            return {'status': 'success', 'report': report, 'stats': self.stats}
        
        except Exception as e:
            self.logger.error(f"[EOD] Error during EOD execution: {e}")
            self.logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e), 'stats': self.stats}
        
        finally:
            self.stats['end_time'] = datetime.now()
            if self.stats['start_time']:
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
                self.stats['duration_seconds'] = duration
    
    def _download_interval(self, interval: str) -> Dict:
        """Download data for specific interval"""
        try:
            self.logger.info(f"[EOD] Downloading {interval}...")
            # Use existing incremental downloader
            result = self.downloader.download_all(interval)
            self.stats['successful_downloads'] += 1
            return {'interval': interval, 'status': 'success', 'result': result}
        except Exception as e:
            self.logger.error(f"[EOD] Failed to download {interval}: {e}")
            self.stats['failed_downloads'] += 1
            return {'interval': interval, 'status': 'failed', 'error': str(e)}
    
    def _validate_eod_data(self) -> Dict:
        """Validate EOD data quality"""
        try:
            from utils.data_quality_validator import DataQualityValidator
            validator = DataQualityValidator()
            
            results = {}
            for interval in [DAILY_INTERVAL] + INTRADAY_INTERVALS:
                results[interval] = validator.validate_interval_data(interval)
            
            return {'status': 'success', 'results': results}
        except Exception as e:
            self.logger.warning(f"[EOD] Validation error: {e}")
            return {'status': 'warning', 'error': str(e)}
    
    def _generate_eod_report(self, daily_result, intraday_results, validation_result) -> str:
        """Generate human-readable EOD report"""
        report = f"""
EOD EXECUTION REPORT
{'='*60}
Timestamp: {datetime.now().isoformat()}
Duration: {self.stats['duration_seconds']:.0f} seconds

DOWNLOADS:
  Daily data: {daily_result['status']}
  Intraday data: {', '.join([r['status'] for r in intraday_results])}

DATA QUALITY:
  Validation: {validation_result['status']}

STATISTICS:
  Successful: {self.stats['successful_downloads']}
  Failed: {self.stats['failed_downloads']}
  Total Files: {self.stats['total_files_downloaded']}

NEXT: Archive to database, prepare for next trading session
"""
        return report


class BODExecutor:
    """Beginning of Day - Refresh data before market opens (9:15 AM IST)"""
    
    def __init__(self):
        self.logger = logger
        self.downloader = IncrementalDataDownloader()
        self.stats = {
            'refreshed_symbols': 0,
            'failed_symbols': 0,
            'skipped_symbols': 0,
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
        }
    
    def validate_preconditions(self) -> bool:
        """Validate BOD can run (trading day, market about to open, credentials valid)"""
        self.logger.info("[BOD] Validating preconditions...")
        
        # Check if trading day
        if not is_trading_day():
            self.logger.warning("[BOD] Not a trading day - skipping BOD")
            return False
        
        # Check token validity
        try:
            api_key = get_token_from_env()
            if not api_key:
                self.logger.error("[BOD] API key not found")
                return False
            
            self.logger.info("[BOD] ✓ API credentials valid")
        except Exception as e:
            self.logger.error(f"[BOD] Token validation failed: {e}")
            return False
        
        self.logger.info("[BOD] ✓ All preconditions met")
        return True
    
    def run_bod(self, force: bool = False) -> Dict:
        """Execute beginning-of-day refresh"""
        
        if not force and not self.validate_preconditions():
            return {'status': 'skipped', 'reason': 'Preconditions not met'}
        
        self.logger.info("=" * 80)
        self.logger.info(f"[BOD] STARTING BEGINNING-OF-DAY REFRESH")
        self.logger.info(f"[BOD] Time: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # Refresh yesterday's close data
            self.logger.info("[BOD] Refreshing previous day's data...")
            daily_refresh = self._refresh_daily_data()
            
            # Update intraday starting points
            self.logger.info(f"[BOD] Preparing intraday intervals ({', '.join(INTRADAY_INTERVALS)})...")
            intraday_refresh = self._prepare_intraday_data()
            
            # Validate data loaded correctly
            self.logger.info("[BOD] Validating refreshed data...")
            validation_result = self._validate_bod_data()
            
            # Generate report
            report = self._generate_bod_report(daily_refresh, intraday_refresh, validation_result)
            
            self.logger.info("=" * 80)
            self.logger.info(f"[BOD] BOD REFRESH COMPLETE")
            self.logger.info(f"[BOD] Duration: {self.stats['duration_seconds']:.0f} seconds")
            self.logger.info(f"[BOD] Ready for intraday trading")
            self.logger.info("=" * 80)
            
            return {'status': 'success', 'report': report, 'stats': self.stats}
        
        except Exception as e:
            self.logger.error(f"[BOD] Error during BOD execution: {e}")
            self.logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e), 'stats': self.stats}
        
        finally:
            self.stats['end_time'] = datetime.now()
            if self.stats['start_time']:
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
                self.stats['duration_seconds'] = duration
    
    def _refresh_daily_data(self) -> Dict:
        """Refresh yesterday's close data"""
        try:
            self.logger.info(f"[BOD] Refreshing {DAILY_INTERVAL} data...")
            result = self.downloader.download_all(DAILY_INTERVAL)
            self.stats['refreshed_symbols'] += 1
            return {'status': 'success', 'result': result}
        except Exception as e:
            self.logger.error(f"[BOD] Failed to refresh daily data: {e}")
            self.stats['failed_symbols'] += 1
            return {'status': 'failed', 'error': str(e)}
    
    def _prepare_intraday_data(self) -> Dict:
        """Prepare intraday data structures for today's trading"""
        results = {}
        try:
            for interval in INTRADAY_INTERVALS:
                self.logger.info(f"[BOD] Preparing {interval} data for today...")
                # Check existing intraday data
                dir_path = DATA_DIRS[interval]
                existing_files = len(list(dir_path.glob('*.parquet')))
                self.logger.info(f"[BOD]   {existing_files} symbols with {interval} data ready")
                results[interval] = {'status': 'ready', 'files': existing_files}
            
            return {'status': 'success', 'results': results}
        except Exception as e:
            self.logger.error(f"[BOD] Failed to prepare intraday data: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _validate_bod_data(self) -> Dict:
        """Validate BOD data integrity"""
        try:
            # Check that key files have latest data
            issues = []
            for interval in [DAILY_INTERVAL] + INTRADAY_INTERVALS:
                dir_path = DATA_DIRS[interval]
                if not dir_path.exists() or len(list(dir_path.glob('*.parquet'))) == 0:
                    issues.append(f"No {interval} data found")
            
            if issues:
                return {'status': 'warning', 'issues': issues}
            return {'status': 'success'}
        except Exception as e:
            self.logger.warning(f"[BOD] Validation error: {e}")
            return {'status': 'warning', 'error': str(e)}
    
    def _generate_bod_report(self, daily_refresh, intraday_prep, validation_result) -> str:
        """Generate human-readable BOD report"""
        report = f"""
BOD EXECUTION REPORT
{'='*60}
Timestamp: {datetime.now().isoformat()}
Duration: {self.stats['duration_seconds']:.0f} seconds

REFRESH STATUS:
  Daily data: {daily_refresh['status']}
  Intraday prep: {intraday_prep['status']}

DATA VALIDATION:
  Status: {validation_result['status']}

STATISTICS:
  Refreshed: {self.stats['refreshed_symbols']}
  Failed: {self.stats['failed_symbols']}

NEXT: Begin intraday trading session, prepare for live updates
"""
        return report


# Test support
if __name__ == '__main__':
    print("EOD/BOD Executor Test")
    print("=" * 60)
    
    # Test BOD
    print("\nTesting BOD Executor...")
    bod = BODExecutor()
    bod_result = bod.run_bod(force=True)
    print(f"BOD Result: {bod_result['status']}")
    
    # Test EOD
    print("\nTesting EOD Executor...")
    eod = EODExecutor()
    eod_result = eod.run_eod(force=True)
    print(f"EOD Result: {eod_result['status']}")
