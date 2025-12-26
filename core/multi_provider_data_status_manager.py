# core/multi_provider_data_status_manager.py
# Tracks download status across multiple data providers (Zerodha, Polygon, AlphaVantage, etc.)

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
import threading

from config import BASE_DIR, TIMEFRAMES
from utils.logger import logger

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Supported data providers"""
    ZERODHA = "zerodha"
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    TWELVE_DATA = "twelve_data"
    FRED = "fred"  # Federal Reserve Economic Data
    YAHOO_FINANCE = "yahoo_finance"


class DataType(Enum):
    """Types of data downloaded from provider"""
    LIVE = "live"  # Real-time quotes
    INTRADAY = "intraday"  # 5-minute bars
    DAILY = "daily"  # Daily bars


class DownloadStatusEnum(Enum):
    """Status of download attempt"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    PARTIAL = "partial"  # Some bars missing
    ERROR = "error"  # Download failed
    STALE = "stale"  # Data > 24 hours old


@dataclass
class TimeframeDownloadStatus:
    """Status for a specific timeframe"""
    symbol: str
    timeframe: str  # '5m', '15m', 'day', 'week', etc.
    provider: str  # DataProvider name
    status: str  # DownloadStatusEnum value
    bar_count: int = 0
    completeness_percent: float = 0.0  # How much of expected data is available
    last_updated: Optional[str] = None  # ISO format
    last_candle_time: Optional[str] = None  # When the last bar was from
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    with_returns: bool = False  # Has return_% column


@dataclass
class SymbolProviderStatus:
    """Status for a symbol from a specific provider"""
    symbol: str
    provider: str
    data_types_available: List[str] = field(default_factory=list)  # ['live', 'intraday', 'daily']
    timeframes: Dict[str, TimeframeDownloadStatus] = field(default_factory=dict)
    overall_status: str = DownloadStatusEnum.NOT_STARTED.value
    average_completeness: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[str] = None
    recent_errors: List[str] = field(default_factory=list)
    total_rows: int = 0


@dataclass
class ProviderStatus:
    """Overall status for a provider"""
    provider: str
    enabled: bool = True
    symbols_attempted: int = 0
    symbols_complete: int = 0
    symbols_partial: int = 0
    symbols_with_errors: int = 0
    total_data_size_gb: float = 0.0
    last_sync: Optional[str] = None


class MultiProviderDataStatusManager:
    """
    Manages download status across multiple data providers
    
    Architecture:
    - Tracks status per: provider → symbol → data_type → timeframe
    - Supports: Zerodha, Polygon, AlphaVantage, Twelve Data, FRED, Yahoo Finance
    - Data types: Live quotes, Intraday (5m bars), Daily bars
    - Timeframes: All intraday + daily/weekly/monthly/quarterly/annual
    
    Persistence:
    - Stores state in JSON file per provider
    - Auto-loads on init
    - Auto-saves after updates
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize multi-provider status manager"""
        self.data_dir = data_dir or (BASE_DIR / "data")
        self.state_dir = self.data_dir / ".provider_status"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Status storage
        self.provider_statuses: Dict[str, ProviderStatus] = {}
        self.symbol_statuses: Dict[str, Dict[str, SymbolProviderStatus]] = {}  # provider -> symbol -> status
        
        # Load existing state
        self._load_state()
        
        logger.info(f"MultiProviderDataStatusManager initialized with {len(self.provider_statuses)} providers")
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    # ========================================================================
    # STATE PERSISTENCE
    # ========================================================================
    
    def _load_state(self) -> None:
        """Load persisted state for all providers"""
        for provider in DataProvider:
            state_file = self.state_dir / f"{provider.value}_status.json"
            
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        data = json.load(f)
                    
                    # Reconstruct provider status
                    if 'provider_status' in data:
                        ps = data['provider_status']
                        self.provider_statuses[provider.value] = ProviderStatus(
                            provider=provider.value,
                            enabled=ps.get('enabled', True),
                            symbols_attempted=ps.get('symbols_attempted', 0),
                            symbols_complete=ps.get('symbols_complete', 0),
                            symbols_partial=ps.get('symbols_partial', 0),
                            symbols_with_errors=ps.get('symbols_with_errors', 0),
                            total_data_size_gb=ps.get('total_data_size_gb', 0.0),
                            last_sync=ps.get('last_sync'),
                        )
                    
                    # Reconstruct symbol statuses
                    if 'symbol_statuses' in data:
                        for symbol, status_data in data['symbol_statuses'].items():
                            if symbol not in self.symbol_statuses:
                                self.symbol_statuses[symbol] = {}
                            
                            # Convert back to dataclass
                            sps = SymbolProviderStatus(
                                symbol=status_data['symbol'],
                                provider=status_data['provider'],
                                data_types_available=status_data.get('data_types_available', []),
                                overall_status=status_data.get('overall_status', DownloadStatusEnum.NOT_STARTED.value),
                                average_completeness=status_data.get('average_completeness', 0.0),
                                error_count=status_data.get('error_count', 0),
                                last_error=status_data.get('last_error'),
                                last_error_time=status_data.get('last_error_time'),
                                recent_errors=status_data.get('recent_errors', []),
                                total_rows=status_data.get('total_rows', 0),
                            )
                            
                            # Load timeframe statuses
                            for tf_name, tf_data in status_data.get('timeframes', {}).items():
                                sps.timeframes[tf_name] = TimeframeDownloadStatus(
                                    symbol=tf_data['symbol'],
                                    timeframe=tf_data['timeframe'],
                                    provider=tf_data['provider'],
                                    status=tf_data['status'],
                                    bar_count=tf_data.get('bar_count', 0),
                                    completeness_percent=tf_data.get('completeness_percent', 0.0),
                                    last_updated=tf_data.get('last_updated'),
                                    last_candle_time=tf_data.get('last_candle_time'),
                                    error_message=tf_data.get('error_message'),
                                    file_path=tf_data.get('file_path'),
                                    file_size_bytes=tf_data.get('file_size_bytes', 0),
                                    with_returns=tf_data.get('with_returns', False),
                                )
                            
                            self.symbol_statuses[symbol][provider.value] = sps
                    
                    logger.info(f"Loaded state for provider: {provider.value}")
                
                except Exception as e:
                    logger.error(f"Failed to load state for {provider.value}: {e}")
    
    def _save_state(self, provider: str) -> None:
        """Persist state for a specific provider"""
        try:
            data = {
                'provider_status': asdict(self.provider_statuses[provider]) if provider in self.provider_statuses else {},
                'symbol_statuses': {},
                'last_save': datetime.now().isoformat(),
            }
            
            # Save symbol statuses for this provider
            for symbol, provider_statuses in self.symbol_statuses.items():
                if provider in provider_statuses:
                    sps = provider_statuses[provider]
                    data['symbol_statuses'][symbol] = {
                        'symbol': sps.symbol,
                        'provider': sps.provider,
                        'data_types_available': sps.data_types_available,
                        'overall_status': sps.overall_status,
                        'average_completeness': sps.average_completeness,
                        'error_count': sps.error_count,
                        'last_error': sps.last_error,
                        'last_error_time': sps.last_error_time,
                        'recent_errors': sps.recent_errors,
                        'total_rows': sps.total_rows,
                        'timeframes': {
                            tf_name: asdict(tf_status)
                            for tf_name, tf_status in sps.timeframes.items()
                        },
                    }
            
            state_file = self.state_dir / f"{provider}_status.json"
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved state for provider: {provider}")
        
        except Exception as e:
            logger.error(f"Failed to save state for {provider}: {e}")
    
    # ========================================================================
    # PROVIDER MANAGEMENT
    # ========================================================================
    
    def enable_provider(self, provider: str) -> None:
        """Enable a data provider"""
        if provider not in self.provider_statuses:
            self.provider_statuses[provider] = ProviderStatus(provider=provider, enabled=True)
        else:
            self.provider_statuses[provider].enabled = True
        
        self._save_state(provider)
        logger.info(f"Enabled provider: {provider}")
    
    def disable_provider(self, provider: str) -> None:
        """Disable a data provider"""
        if provider in self.provider_statuses:
            self.provider_statuses[provider].enabled = False
            self._save_state(provider)
        
        logger.info(f"Disabled provider: {provider}")
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers"""
        return [p for p, status in self.provider_statuses.items() if status.enabled]
    
    # ========================================================================
    # SYMBOL STATUS UPDATE
    # ========================================================================
    
    def update_symbol_status(
        self,
        symbol: str,
        provider: str,
        data_type: str,  # 'live', 'intraday', 'daily'
        timeframe: str,
        status: str,
        bar_count: int = 0,
        completeness: float = 0.0,
        error: Optional[str] = None,
        file_path: Optional[str] = None,
        with_returns: bool = False,
    ) -> None:
        """
        Update status for a symbol's timeframe from a provider
        
        Args:
            symbol: Stock symbol
            provider: Data provider name
            data_type: 'live', 'intraday', or 'daily'
            timeframe: '5m', '15m', 'day', 'week', etc.
            status: DownloadStatusEnum value
            bar_count: Number of bars in file
            completeness: Percentage of expected data available (0-100)
            error: Error message if status is ERROR
            file_path: Path to data file
            with_returns: Whether return_% column exists
        """
        # Initialize symbol entry if needed
        if symbol not in self.symbol_statuses:
            self.symbol_statuses[symbol] = {}
        
        if provider not in self.symbol_statuses[symbol]:
            self.symbol_statuses[symbol][provider] = SymbolProviderStatus(
                symbol=symbol,
                provider=provider,
            )
        
        sps = self.symbol_statuses[symbol][provider]
        
        # Update timeframe status
        sps.timeframes[timeframe] = TimeframeDownloadStatus(
            symbol=symbol,
            timeframe=timeframe,
            provider=provider,
            status=status,
            bar_count=bar_count,
            completeness_percent=completeness,
            last_updated=datetime.now().isoformat(),
            last_candle_time=datetime.now().isoformat(),  # Should be from data
            error_message=error,
            file_path=str(file_path) if file_path else None,
            file_size_bytes=0,  # Should calculate from file
            with_returns=with_returns,
        )
        
        # Add data type if not already
        if data_type not in sps.data_types_available:
            sps.data_types_available.append(data_type)
        
        # Update error tracking
        if status == DownloadStatusEnum.ERROR.value:
            sps.error_count += 1
            sps.last_error = error or "Unknown error"
            sps.last_error_time = datetime.now().isoformat()
            sps.recent_errors.append(error or "Unknown error")
            sps.recent_errors = sps.recent_errors[-10:]  # Keep last 10
        
        # Update overall status
        self._update_overall_status(sps)
        
        # Save state
        self._save_state(provider)
    
    def _update_overall_status(self, sps: SymbolProviderStatus) -> None:
        """Recalculate overall status for a symbol"""
        if not sps.timeframes:
            sps.overall_status = DownloadStatusEnum.NOT_STARTED.value
            return
        
        statuses = [tf.status for tf in sps.timeframes.values()]
        completeness_values = [tf.completeness_percent for tf in sps.timeframes.values()]
        
        # If any ERROR
        if DownloadStatusEnum.ERROR.value in statuses:
            sps.overall_status = DownloadStatusEnum.ERROR.value
        # If all COMPLETE
        elif all(s == DownloadStatusEnum.COMPLETE.value for s in statuses):
            sps.overall_status = DownloadStatusEnum.COMPLETE.value
        # If any PARTIAL
        elif DownloadStatusEnum.PARTIAL.value in statuses:
            sps.overall_status = DownloadStatusEnum.PARTIAL.value
        # If any IN_PROGRESS
        elif DownloadStatusEnum.IN_PROGRESS.value in statuses:
            sps.overall_status = DownloadStatusEnum.IN_PROGRESS.value
        else:
            sps.overall_status = DownloadStatusEnum.NOT_STARTED.value
        
        # Calculate average completeness
        sps.average_completeness = np.mean(completeness_values) if completeness_values else 0.0
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    def get_provider_summary(self, provider: str) -> Dict:
        """Get summary statistics for a provider"""
        return asdict(self.provider_statuses.get(provider, ProviderStatus(provider=provider)))
    
    def get_symbol_status(self, symbol: str, provider: str) -> Optional[SymbolProviderStatus]:
        """Get status for a specific symbol from a provider"""
        return self.symbol_statuses.get(symbol, {}).get(provider)
    
    def get_symbol_all_providers(self, symbol: str) -> Dict[str, SymbolProviderStatus]:
        """Get status for a symbol from all providers"""
        return self.symbol_statuses.get(symbol, {})
    
    def get_all_statuses(self) -> Dict:
        """Get complete status tree"""
        return {
            'providers': asdict(self.provider_statuses),
            'symbols': {
                symbol: {
                    p: {
                        'data_types': sps.data_types_available,
                        'average_completeness': sps.average_completeness,
                    }
                    for p, sps in providers.items()
                }
                for symbol, providers in self.symbol_statuses.items()
            },
        }


def get_multi_provider_status_manager() -> MultiProviderDataStatusManager:
    """Get or create singleton instance"""
    return MultiProviderDataStatusManager()


# Import numpy for average calculation
import numpy as np
