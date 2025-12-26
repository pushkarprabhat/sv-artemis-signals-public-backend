"""
Download Status Bar & Progress Tracking
Provides uniform status reporting across download operations
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
import sys
from utils.logger import logger


class DownloadStatusBar:
    """
    Uniform status bar for download operations
    
    Shows real-time progress with clear distinction between:
    - Available: Ready to download
    - Completed: Successfully downloaded
    - Failed: Connection/API errors (will retry)
    - Exception: No data (won't retry)
    - Progress: Current symbol being downloaded
    """
    
    def __init__(self, total: int, operation_name: str = "Download"):
        self.total = total
        self.operation_name = operation_name
        self.completed = 0
        self.failed = 0
        self.exception = 0
        self.skipped = 0
        self.start_time = datetime.now()
        self.current_symbol: Optional[str] = None
        self.last_update = datetime.now()
    
    def update(self, 
               completed: int = 0, 
               failed: int = 0, 
               exception: int = 0,
               current_symbol: Optional[str] = None,
               force: bool = False):
        """
        Update progress (throttled to avoid spam)
        
        Args:
            completed: Increment completed count
            failed: Increment failed count
            exception: Increment exception count
            current_symbol: Current symbol being processed
            force: Force update regardless of throttle
        """
        self.completed += completed
        self.failed += failed
        self.exception += exception
        self.skipped = self.failed + self.exception
        
        if current_symbol:
            self.current_symbol = current_symbol
        
        # Throttle updates (max once per second)
        now = datetime.now()
        if force or (now - self.last_update).total_seconds() >= 1:
            self._display()
            self.last_update = now
    
    def _get_progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total == 0:
            return 0
        return (self.completed / self.total) * 100
    
    def _get_eta(self) -> str:
        """Calculate and format ETA"""
        elapsed = datetime.now() - self.start_time
        if self.completed == 0:
            return "calculating..."
        
        rate = self.completed / elapsed.total_seconds()
        remaining = self.total - self.completed
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        
        return eta_time.strftime("%H:%M:%S")
    
    def _get_elapsed(self) -> str:
        """Get formatted elapsed time"""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _display(self):
        """Display status bar"""
        percentage = self._get_progress_percentage()
        available = self.total - self.completed - self.skipped
        elapsed = self._get_elapsed()
        eta = self._get_eta() if self.completed > 0 else "calculating..."
        
        # Build status line
        status = f"\n{'='*80}\n"
        status += f"📊 {self.operation_name} Progress\n"
        status += f"{'='*80}\n"
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        status += f"Progress: [{bar}] {percentage:.1f}% ({self.completed}/{self.total})\n"
        
        # Detailed breakdown
        status += f"\n✅ Completed: {self.completed}\n"
        status += f"⏳ Remaining: {available}\n"
        status += f"🟠 Failed (will retry): {self.failed}\n"
        status += f"🔴 Exception (skip): {self.exception}\n"
        
        # Current symbol
        if self.current_symbol:
            status += f"\n🔄 Current: {self.current_symbol}\n"
        
        # Time info
        status += f"\n⏱️  Elapsed: {elapsed}\n"
        status += f"⏳ ETA: {eta}\n"
        
        status += f"{'='*80}\n"
        
        logger.info(status)
    
    def finish(self):
        """Finalize and show summary"""
        self._display()
        logger.info(f"\n✨ {self.operation_name} Complete!")
        logger.info(f"   Total: {self.total} | "
                   f"Completed: {self.completed} | "
                   f"Failed: {self.failed} | "
                   f"Exception: {self.exception}")
        logger.info(f"   Time: {self._get_elapsed()}\n")


class DownloadStatusManager:
    """
    Manages uniform status reporting for download operations
    
    Requirements from user:
    - Clear distinction between FAILED (temporary) and EXCEPTION (permanent)
    - Progress tracking with ETA
    - Current symbol being processed
    - Reasons for failures (API error vs connection error vs no data)
    """
    
    def __init__(self, total_symbols: int, operation_name: str = "Download"):
        self.total_symbols = total_symbols
        self.operation_name = operation_name
        self.status_bar = DownloadStatusBar(total_symbols, operation_name)
        
        # Tracking
        self.downloaded: Dict[str, str] = {}  # symbol -> timestamp
        self.failed: Dict[str, str] = {}  # symbol -> reason
        self.exceptions: Dict[str, str] = {}  # symbol -> reason
        
    def mark_downloading(self, symbol: str):
        """Mark symbol as currently being downloaded"""
        self.status_bar.current_symbol = symbol
    
    def mark_completed(self, symbol: str):
        """Mark symbol as successfully downloaded"""
        self.downloaded[symbol] = datetime.now().isoformat()
        self.status_bar.update(completed=1)
    
    def mark_failed(self, symbol: str, reason: str = "CONNECTION_ERROR"):
        """Mark symbol as failed (temporary, will retry)"""
        self.failed[symbol] = reason
        self.status_bar.update(failed=1)
        logger.debug(f"🟠 FAILED: {symbol} - {reason}")
    
    def mark_exception(self, symbol: str, reason: str = "NO_DATA"):
        """Mark symbol as exception (permanent skip)"""
        self.exceptions[symbol] = reason
        self.status_bar.update(exception=1)
        logger.debug(f"🔴 EXCEPTION: {symbol} - {reason}")
    
    def get_summary(self) -> Dict:
        """Get summary of download operation"""
        return {
            'total': self.total_symbols,
            'completed': len(self.downloaded),
            'failed': len(self.failed),
            'exception': len(self.exceptions),
            'remaining': self.total_symbols - len(self.downloaded) - len(self.failed) - len(self.exceptions),
            'failed_breakdown': self._get_reason_counts(self.failed),
            'exception_breakdown': self._get_reason_counts(self.exceptions),
            'elapsed': self.status_bar._get_elapsed(),
            'success_rate': (len(self.downloaded) / self.total_symbols * 100) if self.total_symbols > 0 else 0
        }
    
    def _get_reason_counts(self, items: Dict[str, str]) -> Dict[str, int]:
        """Get count of items by reason"""
        counts = {}
        for reason in items.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts
    
    def print_summary(self):
        """Print final summary"""
        summary = self.get_summary()
        
        logger.info("\n" + "="*80)
        logger.info("📈 DOWNLOAD OPERATION SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\n✅ COMPLETED: {summary['completed']} symbols")
        logger.info(f"🟠 FAILED (will retry): {summary['failed']} symbols")
        for reason, count in summary['failed_breakdown'].items():
            logger.info(f"   - {reason}: {count}")
        
        logger.info(f"\n🔴 EXCEPTIONS (permanent): {summary['exception']} symbols")
        for reason, count in summary['exception_breakdown'].items():
            logger.info(f"   - {reason}: {count}")
        
        logger.info(f"\n📊 STATISTICS:")
        logger.info(f"   Total: {summary['total']}")
        logger.info(f"   Remaining: {summary['remaining']}")
        logger.info(f"   Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"   Elapsed Time: {summary['elapsed']}")
        
        logger.info("="*80 + "\n")


def format_reason(reason: str, reason_type: str = "failed") -> str:
    """Format reason with proper description"""
    FAILURE_REASONS = {
        'API_ERROR': '❌ API error (rate limit, invalid response)',
        'CONNECTION_ERROR': '❌ Connection error (network, timeout)',
        'TIMEOUT': '❌ Request timeout',
        'UNKNOWN_ERROR': '❌ Unknown API error',
    }
    
    EXCEPTION_REASONS = {
        'NO_DATA': '⚠️  No data from API',
        'DELISTED': '⚠️  Instrument delisted',
        'INVALID_TOKEN': '⚠️  Invalid token',
        'LEGACY': '⚠️  Legacy/obsolete symbol',
        'INACTIVE': '⚠️  Inactive instrument',
    }
    
    reasons_map = FAILURE_REASONS if reason_type == "failed" else EXCEPTION_REASONS
    return reasons_map.get(reason, f"❓ {reason}")
