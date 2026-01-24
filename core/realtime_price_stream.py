"""
Real-Time WebSocket Price Streamer
Provides instantaneous quote updates for intraday analysis
Enables sub-second signal evaluation on price changes
"""

import threading
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import queue
import time
from utils.failure_logger import log_failure
from utils.instrument_exceptions import add_to_exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LiveQuote:
    """Real-time quote data"""
    token: int
    symbol: str
    timestamp: datetime
    last_price: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    volume: int
    open_price: float
    high: float
    low: float
    close: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class IntraDayBar:
    """Aggregated bar for intraday analysis"""
    symbol: str
    timestamp: datetime
    interval: str  # '1minute', '3minute', '5minute', etc.
    open: float = 0.0
    high: float = 0.0
    low: float = float('inf')
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0
    ticks: List[float] = field(default_factory=list)
    
    def update_with_tick(self, price: float, volume: int = 1):
        """Update bar with new tick"""
        if self.tick_count == 0:
            self.open = price
            self.high = price
            self.low = price
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
        
        self.close = price
        self.volume += volume
        self.tick_count += 1
        self.ticks.append(price)


class WebSocketQuoteHandler:
    """Handles WebSocket quote streaming from KiteConnect"""
    
    def __init__(self, kite):
        """
        Initialize WebSocket handler
        
        Args:
            kite: Authenticated KiteConnect instance
        """
        self.kite = kite
        self.ws = None
        self.is_connected = False
        self.subscribed_tokens: set = set()
        self.quote_callbacks: List[Callable] = []
        self.quotes: Dict[int, LiveQuote] = {}
        self.lock = threading.Lock()
        self.receive_thread = None
        self.running = False
    
    def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = self.kite.WebSocket()
            
            def on_open():
                logger.info("WebSocket connected")
                self.is_connected = True
            
            def on_close():
                logger.warning("WebSocket disconnected")
                self.is_connected = False
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
            
            def on_message(ws, message):
                self._handle_message(message)
            
            self.ws.on_open = on_open
            self.ws.on_close = on_close
            self.ws.on_error = on_error
            self.ws.on_message = on_message
            
            # Start receiving in background thread
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            logger.info("WebSocket handler initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            try:
                log_failure(symbol='websocket_connect', exchange='KITE', reason='connect_failed', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log connect_failed")
            return False
    
    def _receive_loop(self):
        """Background loop for receiving messages"""
        try:
            self.ws.connect()
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            try:
                log_failure(symbol='websocket_receive', exchange='KITE', reason='receive_error', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log receive_error")
    
    def subscribe_quotes(self, tokens: List[int]):
        """
        Subscribe to quote updates for tokens
        
        Args:
            tokens: List of instrument tokens
        """
        try:
            self.ws.subscribe(tokens)
            self.subscribed_tokens.update(tokens)
            logger.info(f"Subscribed to {len(tokens)} instruments")
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            try:
                add_to_exceptions([str(t) for t in tokens])
            except Exception:
                logger.debug("[EXCEPTIONS] Could not add subscribe tokens to exceptions")
            try:
                log_failure(symbol='websocket_subscribe', exchange='KITE', reason='subscribe_failed', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log subscribe_failed")
    
    def subscribe_symbols(self, symbols: List[str]):
        """
        Subscribe to quote updates for symbols
        
        Args:
            symbols: List of tradingsymbols
        """
        try:
            # Get tokens from symbols
            instruments = self.kite.instruments()
            tokens = []
            
            for symbol in symbols:
                matching = [i for i in instruments if i['tradingsymbol'] == symbol]
                if matching:
                    tokens.append(matching[0]['instrument_token'])
            
            if tokens:
                self.subscribe_quotes(tokens)
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to subscribe symbols: {e}")
            try:
                log_failure(symbol='websocket_subscribe_symbols', exchange='KITE', reason='subscribe_symbols_failed', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log subscribe_symbols_failed")
            return False
    
    def _handle_message(self, message: Dict):
        """Process incoming quote message"""
        try:
            token = message.get('token')
            if token is None:
                return
            
            # Create LiveQuote object
            quote = LiveQuote(
                token=token,
                symbol=self._get_symbol_from_token(token),
                timestamp=datetime.now(),
                last_price=message.get('last_price', 0),
                bid=message.get('bid', 0),
                ask=message.get('ask', 0),
                bid_qty=message.get('bid_qty', 0),
                ask_qty=message.get('ask_qty', 0),
                volume=message.get('volume', 0),
                open_price=message.get('open', 0),
                high=message.get('high', 0),
                low=message.get('low', 0),
                close=message.get('close', 0)
            )
            
            with self.lock:
                self.quotes[token] = quote
            
            # Call registered callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    try:
                        log_failure(symbol=quote.symbol if hasattr(quote, 'symbol') else 'websocket_callback', exchange='KITE', reason='callback_error', details=str(e))
                    except Exception:
                        logger.debug("[FAILURE_LOG] Could not log callback_error")
        
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            try:
                log_failure(symbol='websocket_message', exchange='KITE', reason='message_handling_error', details=str(e))
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log message_handling_error")
    
    def register_quote_callback(self, callback: Callable):
        """
        Register callback for quote updates
        
        Args:
            callback: Function(quote: LiveQuote) -> None
        """
        self.quote_callbacks.append(callback)
    
    def get_quote(self, token: int) -> Optional[LiveQuote]:
        """Get latest quote for token"""
        with self.lock:
            return self.quotes.get(token)
    
    def get_all_quotes(self) -> Dict[int, LiveQuote]:
        """Get all current quotes"""
        with self.lock:
            return self.quotes.copy()
    
    def _get_symbol_from_token(self, token: int) -> str:
        """Get symbol from token"""
        try:
            instruments = self.kite.instruments()
            matching = [i for i in instruments if i['instrument_token'] == token]
            return matching[0]['tradingsymbol'] if matching else f"TOKEN_{token}"
        except:
            try:
                log_failure(symbol=f'TOKEN_{token}', exchange='KITE', reason='symbol_lookup_failed', details=f"token={token}")
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log symbol_lookup_failed")
            return f"TOKEN_{token}"
    
    def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        try:
            if self.ws:
                self.ws.close()
        except:
            pass
        logger.info("WebSocket disconnected")


class IntraDayBarAggregator:
    """Aggregates ticks into intraday bars"""
    
    def __init__(self):
        """Initialize bar aggregator"""
        self.bars: Dict[str, Dict[str, IntraDayBar]] = defaultdict(dict)
        self.lock = threading.Lock()
        self.bar_callbacks: List[Callable] = []
    
    def update_with_quote(self, quote: LiveQuote, intervals: List[str] = None):
        """
        Update bars with new quote
        
        Args:
            quote: LiveQuote object
            intervals: List of intervals to update (e.g., ['1minute', '3minute'])
        """
        if intervals is None:
            intervals = ['1minute', '3minute', '5minute']
        
        with self.lock:
            now = datetime.now()
            
            for interval in intervals:
                # Get or create bar for this interval
                bar_key = self._get_bar_key(now, interval)
                
                if bar_key not in self.bars[quote.symbol]:
                    self.bars[quote.symbol][bar_key] = IntraDayBar(
                        symbol=quote.symbol,
                        timestamp=now,
                        interval=interval
                    )
                
                bar = self.bars[quote.symbol][bar_key]
                bar.update_with_tick(quote.last_price)
                
                # Check if bar is complete and trigger callback
                if self._is_bar_complete(bar, now):
                    self._trigger_bar_callback(bar)
    
    def _get_bar_key(self, timestamp: datetime, interval: str) -> str:
        """Get bar key for given timestamp and interval"""
        if interval == '1minute':
            return timestamp.strftime('%Y-%m-%d %H:%M')
        elif interval == '3minute':
            minute = (timestamp.minute // 3) * 3
            return timestamp.strftime('%Y-%m-%d %H:') + f'{minute:02d}'
        elif interval == '5minute':
            minute = (timestamp.minute // 5) * 5
            return timestamp.strftime('%Y-%m-%d %H:') + f'{minute:02d}'
        else:
            return timestamp.strftime('%Y-%m-%d %H:%M')
    
    def _is_bar_complete(self, bar: IntraDayBar, current_time: datetime) -> bool:
        """Check if bar is complete"""
        if bar.interval == '1minute':
            return (current_time - bar.timestamp).seconds >= 60
        elif bar.interval == '3minute':
            return (current_time - bar.timestamp).seconds >= 180
        elif bar.interval == '5minute':
            return (current_time - bar.timestamp).seconds >= 300
        return False
    
    def _trigger_bar_callback(self, bar: IntraDayBar):
        """Trigger callbacks for completed bar"""
        for callback in self.bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
    
    def register_bar_callback(self, callback: Callable):
        """
        Register callback for completed bars
        
        Args:
            callback: Function(bar: IntraDayBar) -> None
        """
        self.bar_callbacks.append(callback)
    
    def get_latest_bar(self, symbol: str, interval: str) -> Optional[IntraDayBar]:
        """Get latest bar for symbol and interval"""
        with self.lock:
            if symbol in self.bars and self.bars[symbol]:
                # Return the most recent bar
                bars = self.bars[symbol]
                if bars:
                    return list(bars.values())[-1]
        return None


class RealtimePriceStream:
    """High-level real-time price streaming interface"""
    
    def __init__(self, kite):
        """
        Initialize real-time price stream
        
        Args:
            kite: Authenticated KiteConnect instance
        """
        self.kite = kite
        self.ws_handler = WebSocketQuoteHandler(kite)
        self.bar_aggregator = IntraDayBarAggregator()
        self.quote_processors: List[Callable] = []
        
        # Wire up aggregator to receive quotes
        self.ws_handler.register_quote_callback(self._process_quote)
    
    def start(self, symbols: List[str]) -> bool:
        """
        Start streaming prices for symbols
        
        Args:
            symbols: List of tradingsymbols to stream
        
        Returns:
            True if successful
        """
        logger.info(f"Starting real-time stream for {len(symbols)} symbols")
        
        # Connect WebSocket
        if not self.ws_handler.connect():
            logger.error("Failed to connect WebSocket")
            return False
        
        # Subscribe to symbols
        if not self.ws_handler.subscribe_symbols(symbols):
            logger.warning("Failed to subscribe to all symbols")
        
        return True
    
    def _process_quote(self, quote: LiveQuote):
        """Process incoming quote"""
        # Update intraday bars
        self.bar_aggregator.update_with_quote(quote)
        
        # Call registered quote processors
        for processor in self.quote_processors:
            try:
                processor(quote)
            except Exception as e:
                logger.error(f"Quote processor error: {e}")
    
    def register_quote_processor(self, processor: Callable):
        """
        Register processor for quote updates
        
        Args:
            processor: Function(quote: LiveQuote) -> None
        """
        self.quote_processors.append(processor)
    
    def register_bar_processor(self, processor: Callable):
        """
        Register processor for completed bars
        
        Args:
            processor: Function(bar: IntraDayBar) -> None
        """
        self.bar_aggregator.register_bar_callback(processor)
    
    def get_latest_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get latest quote for symbol"""
        instruments = self.kite.instruments()
        matching = [i for i in instruments if i['tradingsymbol'] == symbol]
        
        if matching:
            token = matching[0]['instrument_token']
            return self.ws_handler.get_quote(token)
        return None
    
    def get_all_quotes(self) -> Dict[str, LiveQuote]:
        """Get all current quotes"""
        quotes_by_symbol = {}
        for token, quote in self.ws_handler.get_all_quotes().items():
            quotes_by_symbol[quote.symbol] = quote
        return quotes_by_symbol
    
    def stop(self):
        """Stop streaming"""
        logger.info("Stopping real-time stream")
        self.ws_handler.disconnect()


def create_realtime_stream(kite) -> RealtimePriceStream:
    """Factory function to create real-time price stream"""
    return RealtimePriceStream(kite)
