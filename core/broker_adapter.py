# core/broker_adapter.py — Zerodha KiteConnect Adapter & Paper Broker
# Handles authentication, order placement, account management
# Professional broker integration and trade execution

import os
import time
from dotenv import load_dotenv
from collections import defaultdict
from kiteconnect import KiteConnect
from utils.logger import logger

load_dotenv()

class PaperBroker:
    """Lightweight paper-broker adapter for testing order flows — safe testing before live"""
    
    def __init__(self, throttle_seconds=0.1):
        """Initialize paper broker with order tracking and positions
        
        Args:
            throttle_seconds: Minimum delay between orders (prevents accidental spam)
        """
        self.positions = defaultdict(float)
        self.orders = {}
        self.throttle_seconds = throttle_seconds
        self.last_order_ts = 0
        self.killed = False

    def kill(self):
        """Activate kill-switch: no new orders allowed, cancels open orders"""
        self.killed = True
        self.orders.clear()

    def revive(self):
        """Disable kill-switch — resume order placement"""
        self.killed = False

    def place_order(self, symbol, qty, price=None, side='buy'):
        """Place a simulated market order
        
        Args:
            symbol: Stock symbol
            qty: Quantity of shares
            price: Order price (optional)
            side: 'buy' or 'sell'
        
        Returns:
            Order ID (string)
        """
        if self.killed:
            raise RuntimeError("PaperBroker is killed — no new orders allowed")

        # Apply throttle to prevent spam
        now = time.time()
        if now - self.last_order_ts < self.throttle_seconds:
            time.sleep(self.throttle_seconds - (now - self.last_order_ts))
        self.last_order_ts = time.time()

        # Create order record
        oid = f"p{int(self.last_order_ts*1000)}"
        self.orders[oid] = {
            'symbol': symbol,
            'qty': qty,
            'price': price,
            'side': side,
            'status': 'open'
        }
        # Immediate fill at price if given
        if price is not None:
            self._fill(oid, price)

        return oid

    def _fill(self, oid, fill_price):
        """Internally fill an order at a given price"""
        o = self.orders.get(oid)
        if not o:
            return
        sign = 1 if o['side'] == 'buy' else -1
        self.positions[o['symbol']] += sign * o['qty']
        o['status'] = 'filled'
        o['fill_price'] = fill_price

    def cancel_order(self, oid):
        """Cancel an open order by ID"""
        if oid in self.orders and self.orders[oid]['status'] == 'open':
            self.orders[oid]['status'] = 'cancelled'

    def cancel_all(self):
        """Cancel all open orders"""
        for oid in list(self.orders.keys()):
            self.cancel_order(oid)

    def simulate_market_fill(self, oid, market_price):
        """Externally fill an open order at a given market price for simulation"""
        if oid in self.orders and self.orders[oid]['status'] == 'open':
            self._fill(oid, market_price)

    def get_positions(self):
        """Get current positions dictionary"""
        return dict(self.positions)


class BrokerAdapter:
    """Abstraction layer for Zerodha KiteConnect API — live trading"""
    
    def __init__(self):
        """Initialize broker connection with credentials from .env"""
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        self.kite = None
        
        if not self.api_key or not self.access_token:
            logger.warning("Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN in .env")
            return
        
        try:
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("[OK] Zerodha KiteConnect authenticated successfully")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self.kite = None
    
    def is_authenticated(self) -> bool:
        """Check if broker is authenticated"""
        return self.kite is not None
    
    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """Get last traded price for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'INFY')
            exchange: Exchange code (default 'NSE')
        
        Returns:
            Last traded price or None if error
        """
        try:
            if not self.kite:
                return None
            quote = self.kite.ltp(f"{exchange}:{symbol}")
            key = f"{exchange}:{symbol}"
            if key in quote:
                return quote[key].get("last_price")
            return None
        except Exception as e:
            logger.debug(f"Could not fetch LTP for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Get full quote for a symbol (OHLC, volume, etc.)
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
        
        Returns:
            Dictionary with OHLC, volume, Greeks, etc.
        """
        try:
            if not self.kite:
                return None
            quote = self.kite.quote(f"{exchange}:{symbol}")
            key = f"{exchange}:{symbol}"
            return quote.get(key) if key in quote else None
        except Exception as e:
            logger.debug(f"Could not fetch quote for {symbol}: {e}")
            return None
    
    def get_balance(self) -> dict:
        """Get account balance and margins
        
        Returns:
            Dictionary with available balance, used margins, net value
        """
        try:
            if not self.kite:
                return None
            margins = self.kite.margins()
            return {
                'available': margins.get('equity', {}).get('available', 0),
                'used': margins.get('equity', {}).get('used', 0),
                'net': margins.get('equity', {}).get('net', 0)
            }
        except Exception as e:
            logger.warning(f"Could not fetch account balance: {e}")
            return None
    
    def place_order(self, symbol: str, transaction_type: str, quantity: int, order_type: str = "MARKET", price: float = None) -> str:
        """Place an order on Zerodha
        
        Args:
            symbol: Stock symbol (e.g., 'INFY')
            transaction_type: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: 'MARKET', 'LIMIT', etc.
            price: Price (required for LIMIT orders)
        
        Returns:
            Order ID or None if failed
        """
        try:
            if not self.kite:
                logger.error("Broker not authenticated")
                return None
            
            order_id = self.kite.place_order(
                variety="regular",
                exchange="NSE",
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            logger.info(f"[OK] Order placed: {symbol} {transaction_type} {quantity} @ Order ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Order placement failed for {symbol}: {e}")
            return None


# Singleton instances for global use
_broker = None
_paper_broker = None

def get_broker() -> BrokerAdapter:
    """Get or create live broker adapter instance"""
    global _broker
    if _broker is None:
        _broker = BrokerAdapter()
    return _broker

def get_paper_broker() -> PaperBroker:
    """Get or create paper broker instance for testing"""
    global _paper_broker
    if _paper_broker is None:
        _paper_broker = PaperBroker()
    return _paper_broker
