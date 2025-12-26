"""
Quick Entry Orchestrator
Executes positions at optimal prices using real-time signals
Coordinates signal generation → price monitoring → instant order execution
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import threading
from enum import Enum
import json

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "PENDING"
    TRIGGERED = "TRIGGERED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PriceCondition(Enum):
    """Price condition types"""
    AT_PRICE = "AT_PRICE"           # Execute at exact price
    ABOVE_PRICE = "ABOVE_PRICE"     # Execute at or above
    BELOW_PRICE = "BELOW_PRICE"     # Execute at or below
    WITHIN_SPREAD = "WITHIN_SPREAD" # Execute within bid-ask


@dataclass
class QuickEntryOrder:
    """Quick entry order definition"""
    order_id: str
    symbol: str
    signal: Dict                    # Signal that triggered order
    quantity: int
    price_target: float
    price_condition: PriceCondition
    max_slippage: float = 0.5       # Maximum price slippage %
    time_limit: int = 60             # Seconds to wait for order
    
    status: OrderStatus = OrderStatus.PENDING
    executed_price: Optional[float] = None
    executed_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)


class PriceMonitor:
    """Monitors prices for order execution"""
    
    def __init__(self, realtime_stream):
        """
        Initialize price monitor
        
        Args:
            realtime_stream: RealtimePriceStream instance
        """
        self.realtime_stream = realtime_stream
        self.watch_list: Dict[str, QuickEntryOrder] = {}
        self.execution_callbacks: List[Callable] = []
        self.lock = threading.Lock()
        self.is_running = False
    
    def watch_order(self, order: QuickEntryOrder):
        """
        Start monitoring order for execution
        
        Args:
            order: QuickEntryOrder to monitor
        """
        with self.lock:
            self.watch_list[order.order_id] = order
        
        logger.info(f"Monitoring order {order.order_id}: {order.symbol} @ {order.price_target}")
        
        # Register quote processor for this symbol
        self.realtime_stream.register_quote_processor(
            lambda quote: self._check_execution(quote, order.order_id)
        )
    
    def _check_execution(self, quote, order_id: str):
        """Check if order should be executed on this quote"""
        with self.lock:
            if order_id not in self.watch_list:
                return
            
            order = self.watch_list[order_id]
        
        # Check if symbol matches
        if quote.symbol != order.symbol:
            return
        
        # Check if order already executed
        if order.status != OrderStatus.PENDING:
            return
        
        # Check if time limit exceeded
        elapsed = (datetime.now() - order.created_at).total_seconds()
        if elapsed > order.time_limit:
            self._mark_failed(order, "Time limit exceeded")
            return
        
        # Check price condition
        if self._should_execute(quote, order):
            self._execute_order(order, quote.last_price)
    
    def _should_execute(self, quote, order: QuickEntryOrder) -> bool:
        """Determine if order should execute at current quote"""
        if order.price_condition == PriceCondition.AT_PRICE:
            # Execute if price within slippage tolerance
            slippage = abs(quote.last_price - order.price_target) / order.price_target * 100
            return slippage <= order.max_slippage
        
        elif order.price_condition == PriceCondition.ABOVE_PRICE:
            return quote.last_price >= order.price_target
        
        elif order.price_condition == PriceCondition.BELOW_PRICE:
            return quote.last_price <= order.price_target
        
        elif order.price_condition == PriceCondition.WITHIN_SPREAD:
            return order.price_target >= quote.bid and order.price_target <= quote.ask
        
        return False
    
    def _execute_order(self, order: QuickEntryOrder, price: float):
        """Execute the order"""
        with self.lock:
            order.status = OrderStatus.EXECUTED
            order.executed_price = price
            order.executed_time = datetime.now()
        
        logger.info(f"ORDER EXECUTED: {order.order_id} {order.symbol} @ {price}")
        
        # Trigger execution callbacks
        for callback in self.execution_callbacks:
            try:
                callback(order, price)
            except Exception as e:
                logger.error(f"Execution callback error: {e}")
    
    def _mark_failed(self, order: QuickEntryOrder, reason: str):
        """Mark order as failed"""
        with self.lock:
            order.status = OrderStatus.FAILED
            order.error_message = reason
        
        logger.warning(f"Order failed: {order.order_id} - {reason}")
    
    def register_execution_callback(self, callback: Callable):
        """
        Register callback for order execution
        
        Args:
            callback: Function(order: QuickEntryOrder, price: float) -> None
        """
        self.execution_callbacks.append(callback)
    
    def get_order_status(self, order_id: str) -> Optional[QuickEntryOrder]:
        """Get current order status"""
        with self.lock:
            return self.watch_list.get(order_id)


class QuickEntryOrchestrator:
    """Orchestrates rapid entry execution on signals"""
    
    def __init__(self, kite, smart_orchestrator, paper_trader):
        """
        Initialize quick entry orchestrator
        
        Args:
            kite: KiteConnect instance
            smart_orchestrator: SmartDataOrchestrator instance
            paper_trader: PaperTrader instance for execution
        """
        self.kite = kite
        self.orchestrator = smart_orchestrator
        self.paper_trader = paper_trader
        self.price_monitor = PriceMonitor(smart_orchestrator.realtime_stream)
        
        self.pending_orders: Dict[str, QuickEntryOrder] = {}
        self.execution_history: List[QuickEntryOrder] = []
        self.lock = threading.Lock()
        
        # Register signal callback
        smart_orchestrator.register_signal_callback(self._on_signal)
        
        # Register execution callback
        self.price_monitor.register_execution_callback(self._on_order_executed)
    
    def _on_signal(self, bar):
        """Handle incoming signal from smart orchestrator"""
        # Signals are already evaluated by InstantSignalEvaluator
        # This method would be called with confirmed signals only
        pass
    
    def place_quick_entry_order(
        self,
        symbol: str,
        signal: Dict,
        quantity: int,
        price_target: float,
        price_condition: PriceCondition = PriceCondition.WITHIN_SPREAD,
        max_slippage: float = 0.5,
        time_limit: int = 60
    ) -> str:
        """
        Place a quick entry order with real-time execution
        
        Args:
            symbol: Trading symbol
            signal: Signal that triggered order
            quantity: Order quantity
            price_target: Target execution price
            price_condition: How price should be checked
            max_slippage: Maximum slippage tolerance %
            time_limit: Seconds to wait for execution
        
        Returns:
            Order ID
        """
        import uuid
        order_id = str(uuid.uuid4())[:8]
        
        order = QuickEntryOrder(
            order_id=order_id,
            symbol=symbol,
            signal=signal,
            quantity=quantity,
            price_target=price_target,
            price_condition=price_condition,
            max_slippage=max_slippage,
            time_limit=time_limit
        )
        
        with self.lock:
            self.pending_orders[order_id] = order
        
        # Start monitoring order
        self.price_monitor.watch_order(order)
        
        logger.info(
            f"Quick entry order placed: {order_id} - {symbol} {quantity} @ {price_target} "
            f"({price_condition.value})"
        )
        
        return order_id
    
    def _on_order_executed(self, order: QuickEntryOrder, executed_price: float):
        """Handle order execution"""
        try:
            logger.info(f"Executing trade: {order.symbol} {order.quantity} @ {executed_price}")
            
            # Execute in paper trader
            entry_result = self.paper_trader.entry(
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=executed_price,
                signal_info=order.signal
            )
            
            if entry_result['success']:
                logger.info(f"Trade executed successfully: {entry_result}")
                
                with self.lock:
                    self.execution_history.append(order)
            else:
                logger.error(f"Trade execution failed: {entry_result}")
        
        except Exception as e:
            logger.error(f"Order execution error: {e}")
    
    def get_pending_orders(self) -> List[QuickEntryOrder]:
        """Get all pending orders"""
        with self.lock:
            return list(self.pending_orders.values())
    
    def get_execution_history(self) -> List[QuickEntryOrder]:
        """Get execution history"""
        with self.lock:
            return self.execution_history.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        with self.lock:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                logger.info(f"Order cancelled: {order_id}")
                return True
        
        return False


class OptimalPriceCalculator:
    """Calculates optimal entry prices"""
    
    @staticmethod
    def calculate_entry_price(
        current_price: float,
        bid: float,
        ask: float,
        signal_strength: float,
        volatility: float
    ) -> Dict:
        """
        Calculate optimal entry price based on market conditions
        
        Args:
            current_price: Current last price
            bid: Current bid price
            ask: Current ask price
            signal_strength: Signal strength (0-1)
            volatility: Current volatility
        
        Returns:
            Dict with entry strategies
        """
        spread = ask - bid
        
        # Strategy 1: Market order (at ask for long, at bid for short)
        market_price = ask
        
        # Strategy 2: Limit order near best price (within spread)
        limit_price = bid + (spread * 0.25)  # 25% into spread
        
        # Strategy 3: Adjusted for volatility
        # Higher volatility = more aggressive (closer to ask)
        volatility_adjustment = spread * (volatility / 100)
        volatility_aware_price = bid + volatility_adjustment
        
        # Strategy 4: Signal strength based
        # Stronger signal = more aggressive entry
        signal_price = bid + (spread * signal_strength)
        
        return {
            'market_order': market_price,
            'limit_order': limit_price,
            'volatility_aware': volatility_aware_price,
            'signal_based': signal_price,
            'recommended': signal_price  # Use signal-based by default
        }


def create_quick_entry_orchestrator(
    kite,
    smart_orchestrator,
    paper_trader
) -> QuickEntryOrchestrator:
    """Factory function to create quick entry orchestrator"""
    return QuickEntryOrchestrator(kite, smart_orchestrator, paper_trader)
