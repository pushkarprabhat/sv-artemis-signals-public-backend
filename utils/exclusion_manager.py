"""
Unified Exclusion & Exception Management System
Distinguishes between temporary failures and permanent no-data exceptions

FAILED INSTRUMENTS: Temporary failures (bad internet, API errors)
- Should be retried periodically (every 24 hours)
- Auto-clear after 30 days
- Track reason: API_ERROR vs CONNECTION_ERROR
- Tracks retry count and last retry time
- Example: Connection timeout, API rate limit

EXCEPTION INSTRUMENTS: Permanent no-data cases  
- Should be permanently skipped
- Track reason: NO_DATA vs DELISTED vs LEGACY
- Mark legacy/obsolete symbols separately
- Example: Instrument has no data on Kite API, expired series

LEGACY SYMBOLS: Old/obsolete instruments
- Tracked within exception instruments
- Examples: Expired series (001HCCL26), old indices
- Can be cleared separately
"""

from pathlib import Path
from typing import Set, Tuple, List, Dict, Optional
from config import BASE_DIR
from utils.logger import logger
import json
from datetime import datetime, timedelta


class ExclusionManager:
    """
    Manages excluded instruments in two separate categories with reasons, timestamps, and expiration:
    - failed_instruments.json: Connection/API errors (retryable, auto-expire after 30 days)
    - exception_instruments.json: No data from API (permanent, track legacy symbols)
    
    Each entry tracks:
    - Symbol name
    - Reason category (API_ERROR, CONNECTION_ERROR, NO_DATA, DELISTED, LEGACY, etc.)
    - First failure timestamp
    - Last retry timestamp (for failed instruments)
    - Legacy status (for old/expired symbols)
    """

    FAILED_FILE = "universe/metadata/failed_instruments.json"
    EXCEPTION_FILE = "universe/metadata/exception_instruments.json"
    METADATA_FILE = "universe/metadata/exclusion_metadata.json"
    
    # Expiration settings
    FAILED_EXPIRATION_DAYS = 30  # Auto-clear failed entries after 30 days
    RETRY_INTERVAL_HOURS = 24    # Retry failed instruments every 24 hours
    
    # Reason categories with descriptions
    REASON_CATEGORIES = {
        # Failed (temporary - connection/API issues)
        'API_ERROR': 'API error (rate limit, invalid response, server error)',
        'CONNECTION_ERROR': 'Connection error (network timeout, DNS, socket error)',
        'INVALID_RESPONSE': 'Invalid API response format',
        'TIMEOUT': 'Request timeout (no response)',
        'UNKNOWN_ERROR': 'Unknown API error',
        
        # Exception (permanent - no data or delisted)
        'NO_DATA': 'No data available from API',
        'DELISTED': 'Instrument delisted or expired',
        'INVALID_TOKEN': 'Invalid instrument token',
        'NOT_TRADABLE': 'Instrument not tradable',
        'LEGACY': 'Legacy/obsolete symbol (old series)',
        'INACTIVE': 'Inactive instrument (no recent trading)',
    }

    def __init__(self):
        self.failed_instruments: Dict[str, Dict] = {}  # symbol -> {reason, timestamp, retry_count, last_retry}
        self.exception_instruments: Dict[str, Dict] = {}  # symbol -> {reason, timestamp, legacy, legacy_since}
        self.metadata = {
            'failed_cleared_count': 0,
            'exception_count': 0,
            'last_cleanup': None,
            'total_retries': 0
        }
        self._load_files()
        self._cleanup_expired_failures()
        self._log_summary()
    
    def _load_files(self):
        """Load instruments from JSON files"""
        self._load_json_file(self.FAILED_FILE, self.failed_instruments)
        self._load_json_file(self.EXCEPTION_FILE, self.exception_instruments)
        self._load_metadata()
    
    def _load_json_file(self, filepath: str, target_dict: Dict):
        """Load instruments from JSON file"""
        file_path = BASE_DIR / filepath
        
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                target_dict.update(data)
            logger.debug(f"[EXCLUSION] Loaded {len(target_dict)} instruments from {filepath}")
        except Exception as e:
            logger.warning(f"[EXCLUSION] Error loading {filepath}: {e}")
    
    def _load_metadata(self):
        """Load metadata about exclusions"""
        file_path = BASE_DIR / self.METADATA_FILE
        
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.warning(f"[EXCLUSION] Error loading metadata: {e}")
    
    def _save_files(self):
        """Save all exclusion data to JSON files"""
        self._save_json_file(self.FAILED_FILE, self.failed_instruments)
        self._save_json_file(self.EXCEPTION_FILE, self.exception_instruments)
        self._save_metadata()
    
    def _save_json_file(self, filepath: str, data: Dict):
        """Save instruments to JSON file"""
        file_path = BASE_DIR / filepath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[EXCLUSION] Saved {len(data)} instruments to {filepath}")
        except Exception as e:
            logger.error(f"[EXCLUSION] Error saving {filepath}: {e}")
    
    def _save_metadata(self):
        """Save metadata"""
        file_path = BASE_DIR / self.METADATA_FILE
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.metadata['last_update'] = datetime.now().isoformat()
            with open(file_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"[EXCLUSION] Error saving metadata: {e}")
    
    def _cleanup_expired_failures(self):
        """Remove failed instruments older than FAILED_EXPIRATION_DAYS"""
        if not self.failed_instruments:
            return
        
        now = datetime.now()
        expired = []
        
        for symbol, data in list(self.failed_instruments.items()):
            try:
                added_time = datetime.fromisoformat(data.get('timestamp', ''))
                age_days = (now - added_time).days
                
                if age_days > self.FAILED_EXPIRATION_DAYS:
                    expired.append((symbol, age_days, data.get('reason', 'UNKNOWN')))
                    del self.failed_instruments[symbol]
            except:
                pass
        
        if expired:
            logger.info(f"🧹 [CLEANUP] Removed {len(expired)} stale failures (older than {self.FAILED_EXPIRATION_DAYS} days)")
            for symbol, age, reason in expired[:3]:  # Log first 3
                logger.debug(f"  🗑️  {symbol} ({age} days old, was {reason})")
            self.metadata['failed_cleared_count'] = self.metadata.get('failed_cleared_count', 0) + len(expired)
            self._save_files()
    
    def _log_summary(self):
        """Log summary of current exclusions"""
        failed_count = len(self.failed_instruments)
        exception_count = len(self.exception_instruments)
        legacy_count = len(self.get_legacy_symbols())
        
        if failed_count > 0 or exception_count > 0:
            logger.info(f"📊 [EXCLUSION SUMMARY] Failed: {failed_count} (retryable) | "
                       f"Exception: {exception_count} (permanent, {legacy_count} legacy)")
    
    # =========================================================================
    # FAILED INSTRUMENTS (Temporary - Will be retried)
    # =========================================================================
    
    def add_failed(self, symbol: str, reason: str = "CONNECTION_ERROR"):
        """
        Add a failed instrument (temporary, will be retried).
        
        Args:
            symbol: Instrument symbol
            reason: Reason category (API_ERROR, CONNECTION_ERROR, TIMEOUT, etc.)
        """
        if symbol not in self.failed_instruments:
            self.failed_instruments[symbol] = {
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'retry_count': 0,
                'last_retry': None
            }
            logger.debug(f"🟠 [FAILED] {symbol} - {self.REASON_CATEGORIES.get(reason, reason)}")
        else:
            # Update retry count
            self.failed_instruments[symbol]['retry_count'] += 1
            self.failed_instruments[symbol]['last_retry'] = datetime.now().isoformat()
            self.failed_instruments[symbol]['reason'] = reason  # Update reason
            logger.debug(f"🟠 [FAILED-RETRY #{self.failed_instruments[symbol]['retry_count']}] {symbol}")
        
        self.metadata['total_retries'] = self.metadata.get('total_retries', 0) + 1
        self._save_files()
    
    def add_failed_batch(self, symbols: List[str], reason: str = "CONNECTION_ERROR"):
        """Add multiple failed instruments"""
        for symbol in symbols:
            if symbol not in self.failed_instruments:
                self.failed_instruments[symbol] = {
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'retry_count': 0,
                    'last_retry': None
                }
        logger.info(f"🟠 [FAILED-BATCH] Added {len(symbols)} instruments ({reason})")
        self._save_files()
    
    def is_failed(self, symbol: str) -> bool:
        """Check if instrument is in failed list"""
        return symbol in self.failed_instruments
    
    def get_failed_retry_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed retry info for a failed instrument"""
        if symbol not in self.failed_instruments:
            return None
        return self.failed_instruments[symbol].copy()
    
    # =========================================================================
    # EXCEPTION INSTRUMENTS (Permanent - Skip forever)
    # =========================================================================
    
    def add_exception(self, symbol: str, reason: str = "NO_DATA", is_legacy: bool = False):
        """
        Add an exception instrument (permanent skip, never retried).
        
        Args:
            symbol: Instrument symbol
            reason: Reason category (NO_DATA, DELISTED, INVALID_TOKEN, LEGACY, etc.)
            is_legacy: Mark as legacy/obsolete symbol
        """
        self.exception_instruments[symbol] = {
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'legacy': is_legacy,
            'legacy_since': datetime.now().isoformat() if is_legacy else None
        }
        
        # Remove from failed if it was there
        self.failed_instruments.pop(symbol, None)
        
        tag = " [LEGACY]" if is_legacy else ""
        logger.debug(f"🔴 [EXCEPTION] {symbol} - {self.REASON_CATEGORIES.get(reason, reason)}{tag}")
        self._save_files()
    
    def add_exception_batch(self, symbols: List[str], reason: str = "NO_DATA", is_legacy: bool = False):
        """Add multiple exception instruments"""
        legacy_tag = " (legacy)" if is_legacy else ""
        for symbol in symbols:
            self.exception_instruments[symbol] = {
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'legacy': is_legacy,
                'legacy_since': datetime.now().isoformat() if is_legacy else None
            }
            self.failed_instruments.pop(symbol, None)
        
        logger.info(f"🔴 [EXCEPTION-BATCH] Added {len(symbols)} instruments ({reason}{legacy_tag})")
        self._save_files()
    
    def is_exception(self, symbol: str) -> bool:
        """Check if instrument is in exception list"""
        return symbol in self.exception_instruments
    
    # =========================================================================
    # UNIFIED FILTERING & STATUS
    # =========================================================================
    
    def is_excluded(self, symbol: str) -> bool:
        """Check if instrument is excluded (failed or exception)"""
        return symbol in self.failed_instruments or symbol in self.exception_instruments
    
    def should_retry(self, symbol: str) -> bool:
        """Check if failed instrument should be retried (not too recent)"""
        if symbol not in self.failed_instruments:
            return False
        
        data = self.failed_instruments[symbol]
        last_retry = data.get('last_retry')
        
        if not last_retry:
            return True  # Never retried, should retry
        
        try:
            last_retry_time = datetime.fromisoformat(last_retry)
            hours_since = (datetime.now() - last_retry_time).total_seconds() / 3600
            return hours_since >= self.RETRY_INTERVAL_HOURS
        except:
            return True
    
    def filter_symbols(self, symbols: List[str]) -> Tuple[List[str], Dict]:
        """
        Filter symbols removing excluded ones.
        Returns: (filtered_list, stats_dict)
        
        stats_dict contains detailed breakdown with reason categories
        """
        failed = [s for s in symbols if s in self.failed_instruments]
        exceptions = [s for s in symbols if s in self.exception_instruments]
        available = [s for s in symbols if not self.is_excluded(s)]
        
        # Categorize exceptions
        legacy_count = sum(1 for s in exceptions if self.exception_instruments[s].get('legacy', False))
        no_data_count = sum(1 for s in exceptions if self.exception_instruments[s].get('reason') == 'NO_DATA')
        delisted_count = sum(1 for s in exceptions if self.exception_instruments[s].get('reason') == 'DELISTED')
        
        # Categorize failures
        api_error_count = sum(1 for s in failed if self.failed_instruments[s].get('reason') == 'API_ERROR')
        connection_error_count = sum(1 for s in failed if self.failed_instruments[s].get('reason') == 'CONNECTION_ERROR')
        timeout_count = sum(1 for s in failed if self.failed_instruments[s].get('reason') == 'TIMEOUT')
        
        stats = {
            'available_count': len(available),
            'total_excluded': len(failed) + len(exceptions),
            # Failed breakdown
            'failed_count': len(failed),
            'failed_symbols': failed,
            'failed_breakdown': {
                'API_ERROR': api_error_count,
                'CONNECTION_ERROR': connection_error_count,
                'TIMEOUT': timeout_count,
                'OTHER': len(failed) - api_error_count - connection_error_count - timeout_count
            },
            # Exception breakdown
            'exception_count': len(exceptions),
            'exception_symbols': exceptions,
            'exception_breakdown': {
                'NO_DATA': no_data_count,
                'DELISTED': delisted_count,
                'LEGACY': legacy_count,
                'OTHER': len(exceptions) - no_data_count - delisted_count - legacy_count
            }
        }
        
        return available, stats
    
    def get_failed_ready_for_retry(self) -> List[str]:
        """Get list of failed instruments ready to be retried"""
        ready = [s for s in self.failed_instruments.keys() if self.should_retry(s)]
        return ready
    
    def get_legacy_symbols(self) -> List[str]:
        """Get all legacy/obsolete symbols"""
        legacy = [s for s, data in self.exception_instruments.items() if data.get('legacy', False)]
        return legacy
    
    def get_failed_with_reason(self, reason: str) -> List[str]:
        """Get all failed instruments with specific reason"""
        return [s for s, data in self.failed_instruments.items() if data.get('reason') == reason]
    
    def get_exceptions_with_reason(self, reason: str) -> List[str]:
        """Get all exception instruments with specific reason"""
        return [s for s, data in self.exception_instruments.items() if data.get('reason') == reason]
    
    # =========================================================================
    # MANAGEMENT OPERATIONS
    # =========================================================================
    
    def clear_failed(self):
        """Clear all failed instruments (fresh retry)"""
        count = len(self.failed_instruments)
        self.failed_instruments.clear()
        logger.info(f"🟠 [CLEARED] Cleared {count} failed instruments - ready for fresh retry")
        self._save_files()
    
    def clear_exceptions(self):
        """Clear all exception instruments (WARNING: Only for manual resets)"""
        count = len(self.exception_instruments)
        self.exception_instruments.clear()
        logger.warning(f"🔴 [CLEARED] Cleared {count} exception instruments - this is permanent!")
        self._save_files()
    
    def clear_legacy_symbols(self):
        """Clear only legacy symbols from exceptions"""
        legacy = self.get_legacy_symbols()
        for symbol in legacy:
            del self.exception_instruments[symbol]
        logger.info(f"🧹 [LEGACY-CLEANUP] Removed {len(legacy)} legacy symbols")
        self._save_files()
    
    def convert_failed_to_exception(self, symbol: str, reason: str = "NO_DATA"):
        """Move a symbol from failed to exception (no data discovered after retry)"""
        if symbol in self.failed_instruments:
            self.failed_instruments.pop(symbol)
            self.add_exception(symbol, reason)
            logger.info(f"↔️  [CONVERTED] {symbol} moved from FAILED to EXCEPTION ({reason})")
    
    def move_old_failures_to_exception(self, days_threshold: int = 7, reason: str = "NO_DATA_AFTER_RETRIES"):
        """Move old failing instruments to exception after retry attempts fail"""
        now = datetime.now()
        moved = []
        
        for symbol, data in list(self.failed_instruments.items()):
            try:
                added_time = datetime.fromisoformat(data.get('timestamp', ''))
                age_days = (now - added_time).days
                
                if age_days >= days_threshold and data.get('retry_count', 0) >= 3:
                    self.convert_failed_to_exception(symbol, reason)
                    moved.append(symbol)
            except:
                pass
        
        if moved:
            logger.info(f"↔️  [AUTO-CONVERT] Moved {len(moved)} instruments from FAILED to EXCEPTION "
                       f"(aged ≥{days_threshold} days + ≥3 retries)")
    
    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get comprehensive exclusion status"""
        failed_retry_ready = len(self.get_failed_ready_for_retry())
        legacy = len(self.get_legacy_symbols())
        
        return {
            'failed_count': len(self.failed_instruments),
            'failed_ready_for_retry': failed_retry_ready,
            'failed_breakdown': {
                'API_ERROR': len(self.get_failed_with_reason('API_ERROR')),
                'CONNECTION_ERROR': len(self.get_failed_with_reason('CONNECTION_ERROR')),
                'TIMEOUT': len(self.get_failed_with_reason('TIMEOUT')),
            },
            'exception_count': len(self.exception_instruments),
            'exception_legacy_count': legacy,
            'exception_breakdown': {
                'NO_DATA': len(self.get_exceptions_with_reason('NO_DATA')),
                'DELISTED': len(self.get_exceptions_with_reason('DELISTED')),
                'LEGACY': legacy,
            },
            'total_excluded': len(self.failed_instruments) + len(self.exception_instruments),
            'metadata': self.metadata
        }
    
    def print_status(self):
        """Print detailed status report"""
        status = self.get_status()
        
        logger.info("=" * 70)
        logger.info("📊 EXCLUSION MANAGER STATUS REPORT")
        logger.info("=" * 70)
        
        # Failed instruments
        logger.info(f"\n🟠 FAILED INSTRUMENTS (Temporary - Will Retry):")
        logger.info(f"   Total: {status['failed_count']}")
        logger.info(f"   Ready for retry: {status['failed_ready_for_retry']}")
        logger.info(f"   Breakdown:")
        logger.info(f"     - API_ERROR: {status['failed_breakdown']['API_ERROR']}")
        logger.info(f"     - CONNECTION_ERROR: {status['failed_breakdown']['CONNECTION_ERROR']}")
        logger.info(f"     - TIMEOUT: {status['failed_breakdown']['TIMEOUT']}")
        
        # Exception instruments
        logger.info(f"\n🔴 EXCEPTION INSTRUMENTS (Permanent - Skip Forever):")
        logger.info(f"   Total: {status['exception_count']}")
        logger.info(f"   Breakdown:")
        logger.info(f"     - NO_DATA: {status['exception_breakdown']['NO_DATA']}")
        logger.info(f"     - DELISTED: {status['exception_breakdown']['DELISTED']}")
        logger.info(f"     - LEGACY: {status['exception_breakdown']['LEGACY']}")
        
        # Summary
        logger.info(f"\n📈 SUMMARY:")
        logger.info(f"   Total Excluded: {status['total_excluded']}")
        logger.info(f"   Total Retries Attempted: {status['metadata'].get('total_retries', 0)}")
        logger.info(f"   Total Failures Cleared: {status['metadata'].get('failed_cleared_count', 0)}")
        logger.info("=" * 70)


# Global instance
_exclusion_manager = None


def get_exclusion_manager() -> ExclusionManager:
    """Get or create global exclusion manager instance"""
    global _exclusion_manager
    if _exclusion_manager is None:
        _exclusion_manager = ExclusionManager()
    return _exclusion_manager
