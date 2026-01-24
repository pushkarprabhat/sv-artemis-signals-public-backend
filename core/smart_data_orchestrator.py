"""
Smart Data Refresh Orchestrator
Intelligently combines WebSocket live prices with periodic CSV updates
Decides when to use live streaming vs CSV refresh based on market conditions
"""

import logging
from datetime import datetime, time as dt_time, timedelta
from typing import List, Dict, Optional, Callable
import threading
import pandas as pd
from pathlib import Path
import json
from utils.failure_logger import log_failure

logger = logging.getLogger(__name__)


class MarketSession:
    """Defines market trading hours"""
    
    # IST times (India)
    MARKET_OPEN = dt_time(9, 15)
    MARKET_CLOSE = dt_time(15, 30)
    EXTENDED_CLOSE = dt_time(23, 59)
    
    @staticmethod
    def is_market_open() -> bool:
        """Check if market is currently open"""
        now = datetime.now().time()
        today = datetime.now().weekday()
        
        # Market closed on weekends
        if today >= 5:
            return False
        
        # Check if within trading hours
        return MarketSession.MARKET_OPEN <= now < MarketSession.MARKET_CLOSE
    
    @staticmethod
    def get_time_to_close() -> timedelta:
        """Get time remaining until market close"""
        now = datetime.now()
        close_time = datetime.combine(now.date(), MarketSession.MARKET_CLOSE)
        
        if now > close_time:
            # Market already closed
            return timedelta(0)
        
        return close_time - now


class DataRefreshStrategy:
    """Determines optimal data refresh strategy"""
    
    def __init__(self):
        self.last_refresh: Dict[str, datetime] = {}
        self.refresh_intervals: Dict[str, int] = {
            'day': 3600,           # 1 hour
            '5minute': 300,        # 5 minutes
            '3minute': 180,        # 3 minutes
            '1minute': 60,         # 1 minute
        }
    
    def should_refresh_csv(self, symbol: str, interval: str) -> bool:
        """
        Determine if CSV refresh is needed
        
        Args:
            symbol: Trading symbol
            interval: Data interval ('day', '5minute', etc.)
        
        Returns:
            True if CSV refresh is needed
        """
        if not MarketSession.is_market_open():
            return True  # Refresh after hours
        
        key = f"{symbol}_{interval}"
        now = datetime.now()
        
        # First refresh always happens
        if key not in self.last_refresh:
            return True
        
        # Check if enough time has passed
        elapsed = (now - self.last_refresh[key]).total_seconds()
        required_interval = self.refresh_intervals.get(interval, 300)
        
        return elapsed >= required_interval
    
    def record_refresh(self, symbol: str, interval: str):
        """Record that refresh happened"""
        key = f"{symbol}_{interval}"
        self.last_refresh[key] = datetime.now()


class SmartDataOrchestrator:
    """Orchestrates smart data refresh combining live prices and CSV updates"""
    
    def __init__(self, kite, parallel_downloader, realtime_stream):
        """
        Initialize smart orchestrator
        
        Args:
            kite: KiteConnect instance
            parallel_downloader: ParallelDownloader instance
            realtime_stream: RealtimePriceStream instance
        """
        self.kite = kite
        self.downloader = parallel_downloader
        self.realtime_stream = realtime_stream
        self.strategy = DataRefreshStrategy()
        
        self.active_symbols: set = set()
        self.signal_callbacks: List[Callable] = []
        self.refresh_lock = threading.Lock()
        self.is_running = False
    
    def initialize(self, symbols: List[str]) -> bool:
        """
        Initialize orchestrator with symbols to track
        
        Args:
            symbols: List of trading symbols to monitor
        
        Returns:
            True if initialization successful
        """
        logger.info(f"Initializing smart orchestrator for {len(symbols)} symbols")
        
        self.active_symbols = set(symbols)
        
        # Start real-time stream
        if not self.realtime_stream.start(symbols):
            logger.error("Failed to start real-time stream")
            try:
                log_failure(symbol='smart_orchestrator', exchange='KITE', reason='realtime_stream_start_failed', details='realtime_stream.start returned False')
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log realtime_stream_start_failed")
            return False
        
        # Register quote processor for hybrid data refresh
        self.realtime_stream.register_quote_processor(self._on_live_quote)
        
        # Register bar processor for signal generation
        self.realtime_stream.register_bar_processor(self._on_intraday_bar)
        
        self.is_running = True
        logger.info("Smart orchestrator initialized")
        
        return True
    
    def _on_live_quote(self, quote):
        """Handle incoming live quote"""
        symbol = quote.symbol
        
        # Decide if CSV refresh needed (for data persistence)
        for interval in ['5minute', '3minute']:
            if self.strategy.should_refresh_csv(symbol, interval):
                # Queue CSV refresh for this symbol/interval
                self._queue_csv_refresh(symbol, interval)
                self.strategy.record_refresh(symbol, interval)
    
    def _queue_csv_refresh(self, symbol: str, interval: str):
        """Queue CSV refresh in background"""
        def refresh_task():
            try:
                self.downloader.download_specific_symbols(
                    symbols=[symbol],
                    intervals=[interval],
                    days_back=5
                )
                logger.debug(f"CSV refreshed: {symbol} ({interval})")
            except Exception as e:
                logger.error(f"CSV refresh failed: {symbol} - {e}")
                try:
                    log_failure(symbol=symbol, exchange='LOCAL', reason='csv_refresh_failed', details=str(e))
                except Exception:
                    logger.debug("[FAILURE_LOG] Could not log csv_refresh_failed")
        
        # Run in thread pool
        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()
    
    def _on_intraday_bar(self, bar):
        """Handle completed intraday bar"""
        logger.info(f"New bar: {bar.symbol} {bar.interval} O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close}")
        
        # Trigger signal evaluation on completed bars
        for callback in self.signal_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
                try:
                    log_failure(symbol=bar.symbol if hasattr(bar, 'symbol') else 'signal_callback', exchange='LOCAL', reason='signal_callback_error', details=str(e))
                except Exception:
                    logger.debug("[FAILURE_LOG] Could not log signal_callback_error")
    
    def register_signal_callback(self, callback: Callable):
        """
        Register callback for signal evaluation
        
        Args:
            callback: Function(bar: IntraDayBar) -> None
        """
        self.signal_callbacks.append(callback)
    
    def get_intraday_data(self, symbol: str, interval: str = '5minute') -> Optional[pd.DataFrame]:
        """
        Get intraday data (hybrid: live prices + CSV)
        
        Args:
            symbol: Trading symbol
            interval: Data interval
        
        Returns:
            DataFrame with intraday data or None
        """
        try:
            filepath = Path(f'marketdata/NSE/{interval}/{symbol}_{interval}.csv')
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                return df
            
            return None
        except Exception as e:
            logger.error(f"Failed to load {symbol} data: {e}")
            try:
                log_failure(symbol=symbol, exchange='LOCAL', reason='intraday_load_failed', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log intraday_load_failed")
            return None
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """
        Get current live price
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Last price or None
        """
        quote = self.realtime_stream.get_latest_quote(symbol)
        return quote.last_price if quote else None
    
    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """
        Get market depth (bid/ask)
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict with bid/ask info or None
        """
        quote = self.realtime_stream.get_latest_quote(symbol)
        
        if quote:
            return {
                'bid': quote.bid,
                'ask': quote.ask,
                'bid_qty': quote.bid_qty,
                'ask_qty': quote.ask_qty,
                'spread': quote.ask - quote.bid
            }
        return None
    
    def get_all_live_quotes(self) -> Dict[str, Dict]:
        """Get all live quotes"""
        quotes = self.realtime_stream.get_all_quotes()
        
        return {
            symbol: {
                'last_price': q.last_price,
                'bid': q.bid,
                'ask': q.ask,
                'volume': q.volume,
                'timestamp': q.timestamp.isoformat()
            }
            for symbol, q in quotes.items()
        }
    
    def refresh_all_intraday(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Refresh all intraday data (happens after market close)
        
        Args:
            symbols: List of symbols to refresh (default: all active)
        
        Returns:
            Summary of refresh operation
        """
        if symbols is None:
            symbols = list(self.active_symbols)
        
        logger.info(f"Refreshing intraday data for {len(symbols)} symbols")
        
        # Do targeted download for all intraday intervals
        summary = self.downloader.download_specific_symbols(
            symbols=symbols,
            intervals=['3minute', '5minute'],
            days_back=5
        )
        
        return summary
    
    def stop(self):
        """Stop orchestrator"""
        self.is_running = False
        self.realtime_stream.stop()
        logger.info("Smart orchestrator stopped")


class InstantSignalEvaluator:
    """Evaluates signals in real-time as data updates"""
    
    def __init__(self, strategy_engine):
        """
        Initialize signal evaluator
        
        Args:
            strategy_engine: StrategyEngine instance for evaluating signals
        """
        self.strategy = strategy_engine
        self.signal_history: List[Dict] = []
        self.signal_lock = threading.Lock()
    
    def evaluate_on_bar(self, bar):
        """
        Evaluate signals when new bar completes
        
        Args:
            bar: IntraDayBar object
        """
        try:
            # Get intraday data for symbol
            filepath = Path(f'marketdata/NSE/{bar.interval}/{bar.symbol}_{bar.interval}.csv')
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                
                # Evaluate all strategies
                signals = self.strategy.evaluate(df, bar.symbol)
                
                if signals and any(signals.values()):
                    self._log_signal(bar.symbol, bar.interval, signals, bar.close)
        
        except Exception as e:
            logger.error(f"Signal evaluation error: {e}")
            try:
                log_failure(symbol=bar.symbol if hasattr(bar, 'symbol') else 'signal_eval', exchange='LOCAL', reason='signal_evaluation_error', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log signal_evaluation_error")
    
    def _log_signal(self, symbol: str, interval: str, signals: Dict, price: float):
        """Log signal detection"""
        signal_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'interval': interval,
            'price': price,
            'signals': signals
        }
        
        with self.signal_lock:
            self.signal_history.append(signal_record)
        
        logger.info(f"SIGNAL: {symbol} @ {price} - {signals}")


def create_smart_orchestrator(kite, parallel_downloader, realtime_stream) -> SmartDataOrchestrator:
    """Factory function to create smart data orchestrator"""
    return SmartDataOrchestrator(kite, parallel_downloader, realtime_stream)
