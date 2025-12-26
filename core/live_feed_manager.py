"""
Live Feed Manager
====================

Manages real-time data collection from KiteWebsocket.

Features:
- Collects live ticks from websocket
- Stores in separate location (not mixed with historical)
- Aggregates into candlesticks (5min, 15min, etc.)
- Persists to parquet files
- Bridges gaps between scheduled downloads

Usage:
    manager = LiveFeedManager(mode='ltp')
    manager.start()
    manager.add_instruments([738561, 256265])  # RELIANCE, TCS
    
    # Data automatically collected and stored
    
    manager.stop()
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, deque

from .websocket import KiteWebsocket, LTPTick, QuoteTick, FullTick

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# CONFIGURATION
# ============================================================================

LIVE_DATA_DIR = Path("data/live_feeds")
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Live tick storage (in-memory buffer before flush to disk)
MAX_TICKS_BUFFER = 10000
FLUSH_INTERVAL = 300  # Flush to disk every 5 minutes
CANDLE_AGGREGATION_INTERVALS = [5, 15, 30, 60]  # Minutes


@dataclass
class Candle:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    count: int
    """Number of ticks in this candle"""
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'count': self.count
        }


# ============================================================================
# LIVE FEED MANAGER
# ============================================================================

class LiveFeedManager:
    """
    Manages live tick collection and aggregation from KiteWebsocket.
    Stores separately from historical data for proper data lineage.
    """
    
    def __init__(
        self,
        api_key: str = None,
        access_token: str = None,
        mode: str = 'ltp',
        user_id: str = None
    ):
        """
        Initialize LiveFeedManager.
        
        Args:
            api_key: Kite API key (if None, pass websocket manually)
            access_token: Kite access token
            mode: 'ltp', 'quote', or 'full'
            user_id: Optional user ID
        """
        self.mode = mode
        self.user_id = user_id or "live_feed_manager"
        
        # Initialize websocket
        if api_key and access_token:
            self.websocket = KiteWebsocket(api_key, access_token, mode=mode, user_id=user_id)
        else:
            self.websocket = None
        
        # Tick buffers (token -> deque of ticks)
        self.tick_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_TICKS_BUFFER))
        
        # Candle builders (token -> {interval -> Candle builder state})
        self.candle_builders: Dict[int, Dict[int, dict]] = defaultdict(
            lambda: {interval: {'open': None, 'high': None, 'low': None, 'volume': 0, 'tick_count': 0, 'start_time': None}
                     for interval in CANDLE_AGGREGATION_INTERVALS}
        )
        
        # Completed candles (token -> {interval -> list of candles})
        self.completed_candles: Dict[int, Dict[int, list]] = defaultdict(
            lambda: {interval: [] for interval in CANDLE_AGGREGATION_INTERVALS}
        )
        
        # Status tracking
        self._running = False
        self._lock = threading.Lock()
        
        # Timers
        self._last_flush = time.time()
        self._flush_thread = None
        
        logger.info(f"[LIVE-FEED] Initialized with mode: {mode}")
    
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    def start(self, websocket: KiteWebsocket = None) -> bool:
        """
        Start live feed collection.
        
        Args:
            websocket: Optional external KiteWebsocket instance
        
        Returns:
            True if started successfully
        """
        try:
            if self._running:
                logger.warning("[LIVE-FEED] Already running")
                return True
            
            # Use external websocket if provided
            if websocket:
                self.websocket = websocket
            
            if not self.websocket:
                logger.error("[LIVE-FEED] No websocket available")
                return False
            
            # Set tick callback
            self.websocket.on_message(self._handle_tick)
            
            # Start websocket
            if not self.websocket.start():
                logger.error("[LIVE-FEED] Failed to start websocket")
                return False
            
            self._running = True
            
            # Start flush thread
            self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
            self._flush_thread.start()
            
            logger.info("[LIVE-FEED] Started successfully")
            return True
        
        except Exception as e:
            logger.error(f"[LIVE-FEED] Start error: {e}")
            return False
    
    
    def stop(self) -> bool:
        """
        Stop live feed collection and flush remaining data.
        
        Returns:
            True if stopped successfully
        """
        try:
            self._running = False
            
            # Stop websocket
            if self.websocket:
                self.websocket.stop()
            
            # Flush remaining data
            self.flush_all()
            
            logger.info("[LIVE-FEED] Stopped successfully")
            return True
        
        except Exception as e:
            logger.error(f"[LIVE-FEED] Stop error: {e}")
            return False
    
    
    # ========================================================================
    # INSTRUMENT MANAGEMENT
    # ========================================================================
    
    def add_instruments(self, tokens: List[int], symbols: Dict[int, str] = None) -> bool:
        """
        Add instruments to live feed collection.
        
        Args:
            tokens: List of instrument tokens
            symbols: Optional dict mapping token -> symbol
        
        Returns:
            True if added successfully
        """
        try:
            if not self.websocket:
                logger.error("[LIVE-FEED] Websocket not available")
                return False
            
            # Subscribe via websocket
            success = self.websocket.subscribe(tokens, symbols)
            
            if success:
                logger.info(f"[LIVE-FEED] Added {len(tokens)} instruments")
            
            return success
        
        except Exception as e:
            logger.error(f"[LIVE-FEED] Add instruments error: {e}")
            return False
    
    
    def remove_instruments(self, tokens: List[int]) -> bool:
        """
        Remove instruments from live feed collection.
        
        Args:
            tokens: List of instrument tokens
        
        Returns:
            True if removed successfully
        """
        try:
            if not self.websocket:
                return False
            
            # Flush data before removing
            for token in tokens:
                self._flush_token_candles(token)
            
            # Unsubscribe via websocket
            success = self.websocket.unsubscribe(tokens)
            
            if success:
                logger.info(f"[LIVE-FEED] Removed {len(tokens)} instruments")
            
            return success
        
        except Exception as e:
            logger.error(f"[LIVE-FEED] Remove instruments error: {e}")
            return False
    
    
    # ========================================================================
    # TICK HANDLING
    # ========================================================================
    
    def _handle_tick(self, tick):
        """Handle incoming tick from websocket."""
        try:
            token = tick.token
            price = tick.ltp
            timestamp = tick.timestamp
            
            # Add to buffer
            with self._lock:
                self.tick_buffers[token].append({
                    'timestamp': timestamp,
                    'price': price,
                    'tick': tick
                })
                
                # Update candles
                self._update_candles(token, price, timestamp)
            
        except Exception as e:
            logger.error(f"[LIVE-FEED] Tick handling error: {e}")
    
    
    def _update_candles(self, token: int, price: float, timestamp: datetime):
        """Update in-memory candles for all intervals."""
        try:
            builders = self.candle_builders[token]
            
            for interval in CANDLE_AGGREGATION_INTERVALS:
                builder = builders[interval]
                
                # Determine candle time window
                minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
                candle_index = minutes_since_midnight // interval
                candle_start = datetime.combine(
                    timestamp.date(),
                    pd.Timestamp.min.time()
                ) + timedelta(minutes=candle_index * interval)
                
                # Check if new candle
                if builder['start_time'] != candle_start:
                    # Save previous candle if exists
                    if builder['start_time'] is not None:
                        candle = Candle(
                            timestamp=builder['start_time'],
                            open=builder['open'],
                            high=builder['high'],
                            low=builder['low'],
                            close=price,
                            volume=builder['volume'],
                            count=builder['tick_count']
                        )
                        self.completed_candles[token][interval].append(candle)
                    
                    # Start new candle
                    builder['start_time'] = candle_start
                    builder['open'] = price
                    builder['high'] = price
                    builder['low'] = price
                    builder['volume'] = 0
                    builder['tick_count'] = 0
                
                # Update current candle
                builder['high'] = max(builder['high'], price)
                builder['low'] = min(builder['low'], price)
                builder['volume'] += 1
                builder['tick_count'] += 1
        
        except Exception as e:
            logger.debug(f"[LIVE-FEED] Candle update error: {e}")
    
    
    # ========================================================================
    # DATA PERSISTENCE
    # ========================================================================
    
    def flush_all(self) -> Dict[int, int]:
        """
        Flush all buffered data to disk.
        
        Returns:
            Dict with token -> records flushed count
        """
        result = {}
        
        with self._lock:
            for token in list(self.tick_buffers.keys()):
                result[token] = self._flush_token(token)
        
        logger.info(f"[LIVE-FEED] Flushed {sum(result.values())} total ticks")
        return result
    
    
    def _flush_token(self, token: int) -> int:
        """Flush buffered ticks and candles for single token to disk."""
        try:
            count = 0
            
            # Flush ticks
            if token in self.tick_buffers and len(self.tick_buffers[token]) > 0:
                ticks = list(self.tick_buffers[token])
                df = pd.DataFrame([{
                    'timestamp': t['timestamp'],
                    'price': t['price']
                } for t in ticks])
                
                # Save to live ticks folder
                token_dir = LIVE_DATA_DIR / "ticks"
                token_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = token_dir / f"{token}_{datetime.now().isoformat().replace(':', '-')}.parquet"
                df.to_parquet(file_path, engine='pyarrow', compression='snappy')
                
                count += len(ticks)
                logger.debug(f"[LIVE-FEED] Flushed {len(ticks)} ticks for token {token}")
            
            # Flush candles
            self._flush_token_candles(token)
            
            return count
        
        except Exception as e:
            logger.error(f"[LIVE-FEED] Flush error for token {token}: {e}")
            return 0
    
    
    def _flush_token_candles(self, token: int) -> int:
        """Flush completed candles for single token."""
        try:
            total_candles = 0
            
            for interval in CANDLE_AGGREGATION_INTERVALS:
                candles = self.completed_candles[token][interval]
                
                if candles:
                    df = pd.DataFrame([c.to_dict() for c in candles])
                    
                    # Save to candles folder
                    candles_dir = LIVE_DATA_DIR / f"candles_{interval}min"
                    candles_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_path = candles_dir / f"{token}.parquet"
                    
                    # Append or create
                    if file_path.exists():
                        existing = pd.read_parquet(file_path)
                        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=['timestamp'])
                    
                    df.to_parquet(file_path, engine='pyarrow', compression='snappy')
                    
                    total_candles += len(candles)
                    self.completed_candles[token][interval].clear()
            
            return total_candles
        
        except Exception as e:
            logger.error(f"[LIVE-FEED] Candle flush error for token {token}: {e}")
            return 0
    
    
    def _periodic_flush(self):
        """Periodically flush data to disk."""
        while self._running:
            try:
                elapsed = time.time() - self._last_flush
                
                if elapsed >= FLUSH_INTERVAL:
                    self.flush_all()
                    self._last_flush = time.time()
                
                time.sleep(10)  # Check every 10 seconds
            
            except Exception as e:
                logger.error(f"[LIVE-FEED] Periodic flush error: {e}")
    
    
    # ========================================================================
    # STATUS & UTILITIES
    # ========================================================================
    
    def get_status(self) -> dict:
        """Get live feed manager status."""
        with self._lock:
            buffer_stats = {
                token: len(buf)
                for token, buf in self.tick_buffers.items()
            }
            
            candle_stats = {
                token: {interval: len(candles)
                        for interval, candles in self.completed_candles[token].items()}
                for token in self.completed_candles
            }
        
        return {
            'running': self._running,
            'mode': self.mode,
            'websocket_connected': self.websocket.is_connected() if self.websocket else False,
            'instruments': len(self.tick_buffers),
            'buffered_ticks': buffer_stats,
            'completed_candles': candle_stats,
            'last_flush': datetime.fromtimestamp(self._last_flush).isoformat()
        }
    
    
    def get_latest_price(self, token: int) -> Optional[float]:
        """Get latest price for an instrument."""
        with self._lock:
            if token in self.tick_buffers and len(self.tick_buffers[token]) > 0:
                return self.tick_buffers[token][-1]['price']
        return None
    
    
    def get_candles(self, token: int, interval: int) -> List[Candle]:
        """Get completed candles for an instrument."""
        with self._lock:
            if token in self.completed_candles and interval in self.completed_candles[token]:
                return self.completed_candles[token][interval].copy()
        return []
