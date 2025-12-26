"""
KiteWebsocket Live Feed Implementation
========================================

Implements all 3 KiteTicker subscription modes:
1. LTP (Last Traded Price) - Only price ticks (~200 bytes/tick)
2. Quote - OHLCV data (~500 bytes/tick)
3. Full - Complete market data (~1000 bytes/tick)

Default mode: LTP (minimal bandwidth, sufficient for pair trading signals)

Usage:
    # Initialize with mode
    ws = KiteWebsocket(api_key, access_token, mode='ltp')
    
    # Subscribe to instruments
    ws.subscribe([738561, 256265])  # RELIANCE, TCS
    
    # Set callbacks
    ws.on_message(callback_function)
    ws.on_error(error_callback)
    
    # Start connection
    ws.start()
    
    # Unsubscribe when done
    ws.unsubscribe([738561])
    
    # Stop connection
    ws.stop()
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from queue import Queue, Empty

from kiteconnect import KiteTicker

# Configure logging with UTF-8 encoding
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Ensure UTF-8 encoding for file handlers
for h in logger.handlers:
    if isinstance(h, logging.FileHandler):
        h.setEncoding('utf-8')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LTPTick:
    """Last Traded Price tick data."""
    token: int
    """Instrument token"""
    timestamp: datetime
    """Tick timestamp"""
    ltp: float
    """Last traded price"""
    
    def to_dict(self) -> dict:
        return {
            'token': self.token,
            'timestamp': self.timestamp,
            'ltp': self.ltp
        }


@dataclass
class QuoteTick:
    """Quote mode tick data (OHLCV)."""
    token: int
    timestamp: datetime
    ltp: float
    open_price: float
    high: float
    low: float
    close: float
    volume: int
    oi: int
    """Open interest"""
    bid: float
    ask: float
    bid_quantity: int
    ask_quantity: int
    
    def to_dict(self) -> dict:
        return {
            'token': self.token,
            'timestamp': self.timestamp,
            'ltp': self.ltp,
            'open': self.open_price,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'oi': self.oi,
            'bid': self.bid,
            'ask': self.ask,
            'bid_qty': self.bid_quantity,
            'ask_qty': self.ask_quantity
        }


@dataclass
class FullTick:
    """Full mode tick data (complete market information)."""
    token: int
    timestamp: datetime
    ltp: float
    open_price: float
    high: float
    low: float
    close: float
    volume: int
    oi: int
    bid: float
    ask: float
    bid_quantity: int
    ask_quantity: int
    tradable: bool
    """Whether instrument is tradable"""
    mode: str
    """Subscription mode"""
    
    def to_dict(self) -> dict:
        return {
            'token': self.token,
            'timestamp': self.timestamp,
            'ltp': self.ltp,
            'open': self.open_price,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'oi': self.oi,
            'bid': self.bid,
            'ask': self.ask,
            'bid_qty': self.bid_quantity,
            'ask_qty': self.ask_quantity,
            'tradable': self.tradable,
            'mode': self.mode
        }


# ============================================================================
# KITEWEBSOCKET MAIN CLASS
# ============================================================================

class KiteWebsocket:
    """
    KiteTicker websocket wrapper with all 3 subscription modes.
    
    Modes:
    - 'ltp': Last traded price only (~200 bytes/tick)
    - 'quote': OHLCV data (~500 bytes/tick)
    - 'full': Complete market data (~1000 bytes/tick)
    
    Default: 'ltp' (recommended for pair trading)
    """
    
    def __init__(
        self,
        api_key: str,
        access_token: str,
        mode: str = 'ltp',
        user_id: str = None
    ):
        """
        Initialize KiteWebsocket.
        
        Args:
            api_key: Kite API key
            access_token: Kite access token
            mode: 'ltp', 'quote', or 'full'
            user_id: Optional user ID for analytics
        """
        if mode not in ['ltp', 'quote', 'full']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ltp', 'quote', or 'full'")
        
        self.api_key = api_key
        self.access_token = access_token
        self.mode = mode
        self.user_id = user_id or "pair_trading_bot"
        
        # Initialize KiteTicker
        self.ticker = KiteTicker(self.api_key, self.access_token)
        
        # Mode mapping to KiteTicker constants
        self.mode_map = {
            'ltp': self.ticker.MODE_LTP,
            'quote': self.ticker.MODE_QUOTE,
            'full': self.ticker.MODE_FULL
        }
        
        # Subscribed instruments
        self.subscribed_tokens: set = set()
        self.token_to_symbol: Dict[int, str] = {}
        
        # Data queues
        self.tick_queue: Queue = Queue(maxsize=1000)
        self.error_queue: Queue = Queue(maxsize=100)
        
        # Callbacks
        self._on_message_callback: Optional[Callable] = None
        self._on_error_callback: Optional[Callable] = None
        self._on_connect_callback: Optional[Callable] = None
        
        # Connection status
        self._connected = False
        self._running = False
        self._lock = threading.Lock()
        
        logger.info(f"[WEBSOCKET] Initialized with mode: {self.mode}")
    
    
    # ========================================================================
    # SUBSCRIPTION MANAGEMENT
    # ========================================================================
    
    def subscribe(self, tokens: List[int], symbols: Dict[int, str] = None) -> bool:
        """
        Subscribe to instrument tokens.
        
        Args:
            tokens: List of instrument tokens
            symbols: Optional dict mapping token -> symbol for logging
        
        Returns:
            True if subscription successful
        """
        try:
            if not self.ticker.is_connected():
                logger.warning("[WEBSOCKET] Not connected, cannot subscribe")
                return False
            
            # Update symbol mapping
            if symbols:
                self.token_to_symbol.update(symbols)
            
            # Subscribe to new tokens only
            new_tokens = [t for t in tokens if t not in self.subscribed_tokens]
            
            if new_tokens:
                self.ticker.subscribe(new_tokens)
                self.subscribed_tokens.update(new_tokens)
                logger.info(f"[WEBSOCKET] Subscribed to {len(new_tokens)} new tokens (total: {len(self.subscribed_tokens)})")
            
            return True
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Subscription error: {e}")
            return False
    
    
    def unsubscribe(self, tokens: List[int]) -> bool:
        """
        Unsubscribe from instrument tokens.
        
        Args:
            tokens: List of instrument tokens to unsubscribe
        
        Returns:
            True if unsubscription successful
        """
        try:
            if not self.ticker.is_connected():
                return False
            
            tokens_to_remove = [t for t in tokens if t in self.subscribed_tokens]
            
            if tokens_to_remove:
                self.ticker.unsubscribe(tokens_to_remove)
                self.subscribed_tokens -= set(tokens_to_remove)
                logger.info(f"[WEBSOCKET] Unsubscribed from {len(tokens_to_remove)} tokens")
            
            return True
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Unsubscription error: {e}")
            return False
    
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """
        Establish websocket connection.
        
        Returns:
            True if connection successful
        """
        try:
            if self._connected:
                logger.warning("[WEBSOCKET] Already connected")
                return True
            
            # Set up callbacks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_message = self._on_message
            self.ticker.on_error = self._on_error
            self.ticker.on_close = self._on_close
            
            # Connect
            logger.info(f"[WEBSOCKET] Connecting to Kite API (mode: {self.mode})...")
            self.ticker.connect(threaded=True)
            
            # Wait for connection (max 5 seconds)
            for _ in range(50):
                if self._connected:
                    logger.info("[WEBSOCKET] Connected successfully")
                    return True
                time.sleep(0.1)
            
            logger.error("[WEBSOCKET] Connection timeout")
            return False
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Connection error: {e}")
            return False
    
    
    def disconnect(self) -> bool:
        """
        Disconnect websocket connection.
        
        Returns:
            True if disconnection successful
        """
        try:
            self._running = False
            
            if self.ticker.is_connected():
                self.ticker.close()
                logger.info("[WEBSOCKET] Disconnected")
            
            self._connected = False
            return True
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Disconnection error: {e}")
            return False
    
    
    def start(self) -> bool:
        """
        Start websocket connection and message loop.
        
        Returns:
            True if started successfully
        """
        try:
            if self._running:
                logger.warning("[WEBSOCKET] Already running")
                return True
            
            self._running = True
            
            if not self.connect():
                self._running = False
                return False
            
            logger.info("[WEBSOCKET] Started successfully")
            return True
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Start error: {e}")
            self._running = False
            return False
    
    
    def stop(self) -> bool:
        """
        Stop websocket and cleanup.
        
        Returns:
            True if stopped successfully
        """
        try:
            self._running = False
            self.disconnect()
            logger.info("[WEBSOCKET] Stopped")
            return True
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Stop error: {e}")
            return False
    
    
    # ========================================================================
    # INTERNAL CALLBACKS (DO NOT CALL DIRECTLY)
    # ========================================================================
    
    def _on_connect(self):
        """Called when websocket connects."""
        with self._lock:
            self._connected = True
        logger.info("[WEBSOCKET] Connected to Kite API")
        
        if self._on_connect_callback:
            try:
                self._on_connect_callback()
            except Exception as e:
                logger.error(f"[WEBSOCKET] Connect callback error: {e}")
    
    
    def _on_message(self, ws, message):
        """
        Called when websocket receives a message.
        Parses tick data based on subscribed mode.
        """
        try:
            # Add to queue for processing
            self.tick_queue.put(message, block=False)
            
            # Parse and invoke user callback
            tick = self._parse_tick(message)
            
            if tick and self._on_message_callback:
                try:
                    self._on_message_callback(tick)
                except Exception as e:
                    logger.error(f"[WEBSOCKET] Message callback error: {e}")
        
        except Exception as e:
            logger.error(f"[WEBSOCKET] Message parsing error: {e}")
    
    
    def _on_error(self, ws, error):
        """Called when websocket error occurs."""
        logger.error(f"[WEBSOCKET] Error: {error}")
        
        try:
            self.error_queue.put(error, block=False)
        except:
            pass
        
        if self._on_error_callback:
            try:
                self._on_error_callback(error)
            except Exception as e:
                logger.error(f"[WEBSOCKET] Error callback failed: {e}")
    
    
    def _on_close(self, ws):
        """Called when websocket closes."""
        with self._lock:
            self._connected = False
        logger.warning("[WEBSOCKET] Disconnected from Kite API")
    
    
    # ========================================================================
    # TICK PARSING
    # ========================================================================
    
    def _parse_tick(self, message: dict):
        """
        Parse websocket message into tick object based on mode.
        
        Returns:
            LTPTick, QuoteTick, or FullTick depending on mode
        """
        try:
            if self.mode == 'ltp':
                return self._parse_ltp(message)
            elif self.mode == 'quote':
                return self._parse_quote(message)
            elif self.mode == 'full':
                return self._parse_full(message)
        except Exception as e:
            logger.error(f"[WEBSOCKET] Parse error ({self.mode}): {e}")
            return None
    
    
    def _parse_ltp(self, message: dict) -> Optional[LTPTick]:
        """
        Parse LTP mode tick.
        
        Message format (LTP):
        {
            'token': 738561,
            'ltp': 2850.50,
            'timestamp': 1701234567
        }
        """
        try:
            return LTPTick(
                token=message.get('token'),
                timestamp=datetime.fromtimestamp(message.get('timestamp', time.time())),
                ltp=float(message.get('ltp', 0))
            )
        except Exception as e:
            logger.debug(f"[WEBSOCKET] LTP parse error: {e}")
            return None
    
    
    def _parse_quote(self, message: dict) -> Optional[QuoteTick]:
        """
        Parse Quote mode tick (OHLCV).
        
        Message format (Quote):
        {
            'token': 738561,
            'ltp': 2850.50,
            'ohlc': {
                'open': 2840.0,
                'high': 2855.0,
                'low': 2835.0,
                'close': 2850.0
            },
            'depth': {'buy': [...], 'sell': [...]},
            'oi': 5000000,
            'volume': 1000000,
            'timestamp': 1701234567,
            'bid': 2850.40,
            'ask': 2850.50,
            'bid_quantity': 100,
            'ask_quantity': 200
        }
        """
        try:
            ohlc = message.get('ohlc', {})
            depth = message.get('depth', {})
            buy_depth = depth.get('buy', [{}])[0]
            sell_depth = depth.get('sell', [{}])[0]
            
            return QuoteTick(
                token=message.get('token'),
                timestamp=datetime.fromtimestamp(message.get('timestamp', time.time())),
                ltp=float(message.get('ltp', 0)),
                open_price=float(ohlc.get('open', 0)),
                high=float(ohlc.get('high', 0)),
                low=float(ohlc.get('low', 0)),
                close=float(ohlc.get('close', 0)),
                volume=int(message.get('volume', 0)),
                oi=int(message.get('oi', 0)),
                bid=float(buy_depth.get('price', 0)),
                ask=float(sell_depth.get('price', 0)),
                bid_quantity=int(buy_depth.get('quantity', 0)),
                ask_quantity=int(sell_depth.get('quantity', 0))
            )
        except Exception as e:
            logger.debug(f"[WEBSOCKET] Quote parse error: {e}")
            return None
    
    
    def _parse_full(self, message: dict) -> Optional[FullTick]:
        """
        Parse Full mode tick (complete market data).
        
        Message format (Full) - includes all Quote data plus:
        {
            'token': 738561,
            'tradable': True,
            'mode': 'full',
            ...all quote fields...
        }
        """
        try:
            ohlc = message.get('ohlc', {})
            depth = message.get('depth', {})
            buy_depth = depth.get('buy', [{}])[0]
            sell_depth = depth.get('sell', [{}])[0]
            
            return FullTick(
                token=message.get('token'),
                timestamp=datetime.fromtimestamp(message.get('timestamp', time.time())),
                ltp=float(message.get('ltp', 0)),
                open_price=float(ohlc.get('open', 0)),
                high=float(ohlc.get('high', 0)),
                low=float(ohlc.get('low', 0)),
                close=float(ohlc.get('close', 0)),
                volume=int(message.get('volume', 0)),
                oi=int(message.get('oi', 0)),
                bid=float(buy_depth.get('price', 0)),
                ask=float(sell_depth.get('price', 0)),
                bid_quantity=int(buy_depth.get('quantity', 0)),
                ask_quantity=int(sell_depth.get('quantity', 0)),
                tradable=message.get('tradable', True),
                mode=message.get('mode', self.mode)
            )
        except Exception as e:
            logger.debug(f"[WEBSOCKET] Full parse error: {e}")
            return None
    
    
    # ========================================================================
    # CALLBACKS (USER-FACING)
    # ========================================================================
    
    def on_message(self, callback: Callable):
        """
        Set callback for incoming tick messages.
        
        Callback signature: callback(tick: Union[LTPTick, QuoteTick, FullTick])
        
        Args:
            callback: Function to call for each tick
        """
        self._on_message_callback = callback
        logger.info("[WEBSOCKET] Message callback registered")
    
    
    def on_error(self, callback: Callable):
        """
        Set callback for error messages.
        
        Callback signature: callback(error: Exception)
        
        Args:
            callback: Function to call on error
        """
        self._on_error_callback = callback
        logger.info("[WEBSOCKET] Error callback registered")
    
    
    def on_connect(self, callback: Callable):
        """
        Set callback for connection established.
        
        Callback signature: callback()
        
        Args:
            callback: Function to call on connect
        """
        self._on_connect_callback = callback
        logger.info("[WEBSOCKET] Connect callback registered")
    
    
    # ========================================================================
    # STATUS & UTILITIES
    # ========================================================================
    
    def is_connected(self) -> bool:
        """Check if websocket is connected."""
        with self._lock:
            return self._connected
    
    
    def is_running(self) -> bool:
        """Check if websocket is running."""
        return self._running
    
    
    def get_subscribed_count(self) -> int:
        """Get count of subscribed instruments."""
        return len(self.subscribed_tokens)
    
    
    def get_mode(self) -> str:
        """Get current subscription mode."""
        return self.mode
    
    
    def get_status(self) -> dict:
        """Get websocket status summary."""
        return {
            'mode': self.mode,
            'connected': self.is_connected(),
            'running': self.is_running(),
            'subscribed_count': self.get_subscribed_count(),
            'queue_size': self.tick_queue.qsize(),
            'errors': self.error_queue.qsize()
        }
