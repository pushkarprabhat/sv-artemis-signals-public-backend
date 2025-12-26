# core/zerodha_download_status_manager.py
# Tracks download status for all Zerodha Kite API data
# Monitors historical data, interval data (6 timeframes), live data completion
# Persistent tracking with timestamps, error logs, and progress monitoring

import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import pandas as pd
from config import BASE_DIR, TIMEFRAMES

logger = logging.getLogger(__name__)


class DownloadStatusEnum(Enum):
    """Status states for downloads"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"  # Some timeframes/symbols done
    COMPLETE = "complete"
    ERROR = "error"
    STALE = "stale"  # Data > 24 hours old


class DataSourceType(Enum):
    """Types of data available from Zerodha"""
    HISTORICAL = "historical"  # 1+ year of daily/intraday
    INTERVAL = "interval"  # Current 6 timeframes (15m, 30m, 60m, day, week, month)
    LIVE = "live"  # Real-time quotes
    OPTIONS = "options"  # Option chains
    DERIVATIVES = "derivatives"  # Futures data


@dataclass
class TimeframeStatus:
    """Status of data for a specific timeframe"""
    timeframe: str
    symbol: str
    status: DownloadStatusEnum = DownloadStatusEnum.NOT_STARTED
    row_count: int = 0
    last_candle_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    completeness_percent: float = 0.0
    expected_rows: int = 0  # Expected rows for this timeframe
    
    def is_fresh(self, hours: int = 24) -> bool:
        """Check if data is fresh (updated within X hours)"""
        if not self.last_updated:
            return False
        return datetime.now() - self.last_updated < timedelta(hours=hours)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with serialization-safe values"""
        d = asdict(self)
        d['status'] = self.status.value
        d['last_candle_time'] = self.last_candle_time.isoformat() if self.last_candle_time else None
        d['last_updated'] = self.last_updated.isoformat() if self.last_updated else None
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimeframeStatus':
        """Create from dictionary"""
        data['status'] = DownloadStatusEnum(data['status'])
        if isinstance(data['last_candle_time'], str):
            data['last_candle_time'] = datetime.fromisoformat(data['last_candle_time'])
        if isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class SymbolDownloadStatus:
    """Status of all data downloads for a symbol"""
    symbol: str
    exchange: str = "NSE"
    last_check: Optional[datetime] = None
    last_refresh: Optional[datetime] = None
    
    # Timeframe tracking
    timeframes: Dict[str, TimeframeStatus] = field(default_factory=dict)
    
    # Overall stats
    total_rows: int = 0
    average_completeness: float = 0.0
    overall_status: DownloadStatusEnum = DownloadStatusEnum.NOT_STARTED
    
    # Error tracking
    recent_errors: List[str] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def add_error(self, error: str):
        """Add error to tracking"""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.now()
        self.recent_errors.append(f"[{datetime.now().isoformat()}] {error}")
        # Keep only last 10 errors
        self.recent_errors = self.recent_errors[-10:]
    
    def update_overall_status(self):
        """Compute overall status from timeframes"""
        if not self.timeframes:
            self.overall_status = DownloadStatusEnum.NOT_STARTED
            return
        
        statuses = [tf.status for tf in self.timeframes.values()]
        
        if all(s == DownloadStatusEnum.COMPLETE for s in statuses):
            self.overall_status = DownloadStatusEnum.COMPLETE
        elif any(s == DownloadStatusEnum.ERROR for s in statuses):
            self.overall_status = DownloadStatusEnum.ERROR
        elif any(s == DownloadStatusEnum.IN_PROGRESS for s in statuses):
            self.overall_status = DownloadStatusEnum.IN_PROGRESS
        elif any(s == DownloadStatusEnum.PARTIAL for s in statuses):
            self.overall_status = DownloadStatusEnum.PARTIAL
        else:
            self.overall_status = DownloadStatusEnum.NOT_STARTED
        
        # Calculate average completeness
        self.total_rows = sum(tf.row_count for tf in self.timeframes.values())
        completeness_values = [tf.completeness_percent for tf in self.timeframes.values()]
        self.average_completeness = sum(completeness_values) / len(completeness_values) if completeness_values else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'timeframes': {k: v.to_dict() for k, v in self.timeframes.items()},
            'total_rows': self.total_rows,
            'average_completeness': self.average_completeness,
            'overall_status': self.overall_status.value,
            'recent_errors': self.recent_errors,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolDownloadStatus':
        """Create from dictionary"""
        # Handle datetime fields
        if isinstance(data.get('last_check'), str):
            data['last_check'] = datetime.fromisoformat(data['last_check'])
        if isinstance(data.get('last_refresh'), str):
            data['last_refresh'] = datetime.fromisoformat(data['last_refresh'])
        if isinstance(data.get('last_error_time'), str):
            data['last_error_time'] = datetime.fromisoformat(data['last_error_time'])
        
        # Handle timeframes
        timeframes_dict = data.pop('timeframes', {})
        obj = cls(**data)
        obj.timeframes = {k: TimeframeStatus.from_dict(v) for k, v in timeframes_dict.items()}
        return obj


class ZerodhaDownloadStatusManager:
    """
    Singleton manager for tracking Zerodha Kite API data downloads
    
    Responsibilities:
    - Track download status for all symbols/timeframes
    - Monitor data freshness and completeness
    - Log errors and track recovery
    - Provide real-time status to UI
    - Persist state across restarts
    """
    
    _instance = None
    _lock = threading.Lock()
    
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
        self.status_file = self.data_dir / ".zerodha_download_status.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self.symbols_status: Dict[str, SymbolDownloadStatus] = {}
        self.last_full_scan: Optional[datetime] = None
        self.scan_in_progress = False
        self.source_stats: Dict[DataSourceType, Dict] = {
            DataSourceType.HISTORICAL: {},
            DataSourceType.INTERVAL: {},
            DataSourceType.LIVE: {},
            DataSourceType.OPTIONS: {},
            DataSourceType.DERIVATIVES: {},
        }
        
        # Load persisted state
        self._load_state()
        
        self._initialized = True
    
    def _load_state(self):
        """Load persisted status from file"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                
                self.last_full_scan = None
                if data.get('last_full_scan'):
                    self.last_full_scan = datetime.fromisoformat(data['last_full_scan'])
                
                self.symbols_status = {
                    k: SymbolDownloadStatus.from_dict(v)
                    for k, v in data.get('symbols_status', {}).items()
                }
                
                logger.info(f"Loaded status for {len(self.symbols_status)} symbols")
        except Exception as e:
            logger.error(f"Failed to load status: {e}")
            self.symbols_status = {}
    
    def _save_state(self):
        """Persist status to file"""
        try:
            data = {
                'last_full_scan': self.last_full_scan.isoformat() if self.last_full_scan else None,
                'symbols_status': {k: v.to_dict() for k, v in self.symbols_status.items()},
                'last_save': datetime.now().isoformat(),
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def scan_downloads(self, symbols: List[str]) -> Dict[str, SymbolDownloadStatus]:
        """
        Scan actual files on disk and update status
        Returns status for all symbols
        """
        self.scan_in_progress = True
        
        try:
            results = {}
            
            for symbol in symbols:
                if symbol not in self.symbols_status:
                    self.symbols_status[symbol] = SymbolDownloadStatus(symbol=symbol)
                
                status = self.symbols_status[symbol]
                status.last_check = datetime.now()
                
                # Check each timeframe
                for timeframe in TIMEFRAMES:
                    tf_status = self._check_timeframe(symbol, timeframe)
                    status.timeframes[timeframe] = tf_status
                
                status.update_overall_status()
                results[symbol] = status
            
            self.last_full_scan = datetime.now()
            self._save_state()
            
            return results
        
        finally:
            self.scan_in_progress = False
    
    def _check_timeframe(self, symbol: str, timeframe: str) -> TimeframeStatus:
        """Check status of data file for a symbol/timeframe"""
        status = TimeframeStatus(symbol=symbol, timeframe=timeframe)
        
        try:
            timeframe_dir = self.data_dir / timeframe
            file_path = timeframe_dir / f"{symbol}.parquet"
            
            if not file_path.exists():
                status.status = DownloadStatusEnum.NOT_STARTED
                return status
            
            status.file_path = str(file_path)
            status.file_size_bytes = file_path.stat().st_size
            
            # Read parquet file for row count and date info
            try:
                df = pd.read_parquet(file_path)
                status.row_count = len(df)
                
                if 'date' in df.columns and len(df) > 0:
                    status.last_candle_time = pd.to_datetime(df['date'].max())
                    status.expected_rows = self._calculate_expected_rows(timeframe)
                    status.completeness_percent = min(100.0, (status.row_count / status.expected_rows * 100)) if status.expected_rows > 0 else 0.0
                    
                    # Determine if data is fresh
                    if status.last_candle_time > datetime.now() - timedelta(hours=24):
                        status.status = DownloadStatusEnum.COMPLETE
                    else:
                        status.status = DownloadStatusEnum.STALE
                else:
                    status.status = DownloadStatusEnum.PARTIAL
                    status.completeness_percent = 0.0
                
                status.last_updated = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            except Exception as e:
                status.status = DownloadStatusEnum.ERROR
                status.error_message = f"Failed to read file: {str(e)}"
        
        except Exception as e:
            status.status = DownloadStatusEnum.ERROR
            status.error_message = str(e)
        
        return status
    
    def _calculate_expected_rows(self, timeframe: str) -> int:
        """Calculate expected number of candles for a timeframe"""
        # Assuming 1 year of data, ~252 trading days
        trading_days = 252
        
        if timeframe == "15minute":
            return trading_days * 26  # ~6.5 hours * 4 per hour
        elif timeframe == "30minute":
            return trading_days * 13  # ~6.5 hours * 2 per hour
        elif timeframe == "60minute":
            return trading_days * 6.5  # ~6.5 hours per day
        elif timeframe == "day":
            return trading_days
        elif timeframe == "week":
            return trading_days // 5  # ~50 weeks
        elif timeframe == "month":
            return 12  # ~12 months
        
        return 0
    
    def get_summary_stats(self) -> Dict:
        """Get overall download statistics"""
        total_symbols = len(self.symbols_status)
        
        if total_symbols == 0:
            return {
                'total_symbols': 0,
                'complete': 0,
                'partial': 0,
                'not_started': 0,
                'errors': 0,
                'stale': 0,
                'average_completeness': 0.0,
                'total_data_size_gb': 0.0,
                'last_full_scan': None,
            }
        
        statuses = list(self.symbols_status.values())
        
        return {
            'total_symbols': total_symbols,
            'complete': sum(1 for s in statuses if s.overall_status == DownloadStatusEnum.COMPLETE),
            'partial': sum(1 for s in statuses if s.overall_status == DownloadStatusEnum.PARTIAL),
            'not_started': sum(1 for s in statuses if s.overall_status == DownloadStatusEnum.NOT_STARTED),
            'errors': sum(1 for s in statuses if s.overall_status == DownloadStatusEnum.ERROR),
            'stale': sum(1 for s in statuses if s.overall_status == DownloadStatusEnum.STALE),
            'average_completeness': sum(s.average_completeness for s in statuses) / total_symbols,
            'total_data_size_gb': sum(sum(tf.file_size_bytes for tf in s.timeframes.values()) for s in statuses) / (1024**3),
            'last_full_scan': self.last_full_scan.isoformat() if self.last_full_scan else None,
        }
    
    def get_symbol_status(self, symbol: str) -> Optional[SymbolDownloadStatus]:
        """Get status for a specific symbol"""
        return self.symbols_status.get(symbol)
    
    def get_symbols_by_status(self, status: DownloadStatusEnum) -> List[str]:
        """Get all symbols with a specific overall status"""
        return [
            symbol for symbol, s in self.symbols_status.items()
            if s.overall_status == status
        ]
    
    def get_incomplete_symbols(self, completeness_threshold: float = 80.0) -> List[Tuple[str, float]]:
        """Get symbols below completeness threshold with their percentages"""
        incomplete = [
            (s.symbol, s.average_completeness)
            for s in self.symbols_status.values()
            if s.average_completeness < completeness_threshold
        ]
        return sorted(incomplete, key=lambda x: x[1])
    
    def get_stale_timeframes(self, hours: int = 24) -> List[Tuple[str, str]]:
        """Get all timeframes with stale data (not updated in X hours)"""
        stale = []
        for symbol, status in self.symbols_status.items():
            for timeframe, tf_status in status.timeframes.items():
                if not tf_status.is_fresh(hours):
                    stale.append((symbol, timeframe))
        return stale
    
    def get_errors_summary(self) -> Dict:
        """Get summary of recent errors"""
        all_errors = []
        for status in self.symbols_status.values():
            for error in status.recent_errors:
                all_errors.append(f"{status.symbol}: {error}")
        
        return {
            'total_error_count': sum(s.error_count for s in self.symbols_status.values()),
            'symbols_with_errors': sum(1 for s in self.symbols_status.values() if s.error_count > 0),
            'recent_errors': all_errors[-20:] if all_errors else [],  # Last 20
        }
    
    def reset_symbol_status(self, symbol: str):
        """Reset status for a symbol (for retry after fixes)"""
        if symbol in self.symbols_status:
            self.symbols_status[symbol] = SymbolDownloadStatus(symbol=symbol)
            self._save_state()
            logger.info(f"Reset status for {symbol}")
    
    def get_download_report(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a detailed download report as DataFrame
        Useful for UI display and analysis
        """
        if not symbols:
            symbols = list(self.symbols_status.keys())
        
        rows = []
        for symbol in symbols:
            status = self.symbols_status.get(symbol)
            if not status:
                continue
            
            for timeframe, tf_status in status.timeframes.items():
                rows.append({
                    'Symbol': symbol,
                    'Timeframe': timeframe,
                    'Status': status.overall_status.value,
                    'Rows': tf_status.row_count,
                    'Completeness %': round(tf_status.completeness_percent, 1),
                    'Last Updated': tf_status.last_updated.strftime('%Y-%m-%d %H:%M') if tf_status.last_updated else 'Never',
                    'Last Candle': tf_status.last_candle_time.strftime('%Y-%m-%d %H:%M') if tf_status.last_candle_time else 'N/A',
                    'File Size MB': round(tf_status.file_size_bytes / (1024**2), 2),
                    'Errors': status.error_count,
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_zerodha_download_status_manager() -> ZerodhaDownloadStatusManager:
    """Get or create singleton instance"""
    return ZerodhaDownloadStatusManager()
