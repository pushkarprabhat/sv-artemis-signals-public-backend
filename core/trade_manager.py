"""
Trade Manager - Unified interface for Paper and Live Trading
Professional trade management and execution system
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
import logging
import sys
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import paper trading manager using importlib to avoid conflicts
paper_trading_spec = importlib.util.spec_from_file_location(
    "paper_trading_manager",
    project_root / "strategies" / "paper_trading_manager.py"
)
paper_trading_module = importlib.util.module_from_spec(paper_trading_spec)
paper_trading_spec.loader.exec_module(paper_trading_module)
PaperTradingManager = paper_trading_module.PaperTradingManager

# Import broker adapter
from core.broker_adapter import BrokerAdapter
import config

logger = logging.getLogger(config.LOGGER_NAME)


class TradeManager:
    """
    Unified Trade Manager - Routes trades to Paper or Live systems
    
    Modes:
    - PAPER: All trades simulated, no real money
    - LIVE: Real trades via KiteConnect API
    
    Features:
    - Mode toggle with safety confirmations
    - Capital management for both modes
    - Position tracking
    - P&L calculation
    - Trade history
    """
    
    def __init__(self, mode: str = "PAPER", initial_capital: float = None):
        """
        Initialize Trade Manager
        
        Args:
            mode: "PAPER" or "LIVE"
            initial_capital: Starting capital (₹)
        """
        if mode not in ["PAPER", "LIVE"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'PAPER' or 'LIVE'")
        
        self.mode = mode
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        
        # Initialize the appropriate trading system
        if self.mode == "PAPER":
            self.trader = PaperTradingManager(initial_capital=self.initial_capital)
            logger.info("📝 Trade Manager initialized in PAPER mode")
        else:
            self.trader = BrokerAdapter()
            logger.info("🔴 Trade Manager initialized in LIVE mode")
            logger.warning("⚠️  LIVE TRADING ENABLED - Real money at risk!")
        
        # Track mode switches
        self.mode_history = [{
            'timestamp': datetime.now(),
            'mode': self.mode,
            'capital': self.initial_capital
        }]
    
    def get_mode(self) -> str:
        """Get current trading mode"""
        return self.mode
    
    def switch_mode(self, new_mode: str, confirmation: bool = False) -> Tuple[bool, str]:
        """
        Switch between PAPER and LIVE modes
        
        Args:
            new_mode: "PAPER" or "LIVE"
            confirmation: Must be True to switch to LIVE mode
        
        Returns:
            (success: bool, message: str)
        """
        if new_mode not in ["PAPER", "LIVE"]:
            return False, f"Invalid mode: {new_mode}"
        
        if new_mode == self.mode:
            return True, f"Already in {new_mode} mode"
        
        # CRITICAL: Require confirmation for LIVE mode
        if new_mode == "LIVE" and not confirmation:
            return False, "⚠️ CONFIRMATION REQUIRED to switch to LIVE mode (real money at risk!)"
        
        # Check for open positions before switching
        open_positions = self.get_open_positions()
        if len(open_positions) > 0:
            return False, f"Cannot switch modes with {len(open_positions)} open positions. Close all positions first."
        
        old_mode = self.mode
        self.mode = new_mode
        
        # Re-initialize trader
        if self.mode == "PAPER":
            self.trader = PaperTradingManager(initial_capital=self.initial_capital)
            msg = f"✅ Switched from {old_mode} to PAPER mode"
        else:
            self.trader = BrokerAdapter()
            msg = f"🔴 Switched from {old_mode} to LIVE mode - REAL MONEY AT RISK"
            logger.warning(msg)
        
        # Log mode switch
        self.mode_history.append({
            'timestamp': datetime.now(),
            'mode': self.mode,
            'from_mode': old_mode,
            'capital': self.get_capital()
        })
        
        logger.info(msg)
        return True, msg
    
    def get_capital(self) -> float:
        """Get current available capital"""
        if self.mode == "PAPER":
            perf = self.trader.get_overall_performance()
            return perf.get('capital', self.initial_capital)
        else:
            # For LIVE mode, get from broker margins
            margins = self.trader.get_margins()
            return margins.get('available', {}).get('cash', 0.0)
    
    def get_total_value(self) -> float:
        """Get total portfolio value (capital + positions)"""
        if self.mode == "PAPER":
            perf = self.trader.get_overall_performance()
            return perf.get('capital', self.initial_capital)
        else:
            # LIVE mode: margins + position values
            margins = self.trader.get_margins()
            equity = margins.get('equity', {})
            return equity.get('net', 0.0)
    
    def execute_pair_trade(self, signal: Dict, comment: str = "") -> Tuple[bool, str]:
        """
        Execute a pair trade
        
        Args:
            signal: Dictionary with keys:
                - pair: "SYMBOL1-SYMBOL2"
                - symbol1, symbol2: Trading symbols
                - qty1, qty2: Quantities
                - price1, price2: Entry prices
                - recommend: "BUY" or "SELL"
                - capital_required: Total capital needed
            comment: Optional trade comment
        
        Returns:
            (success: bool, message: str)
        """
        # Validate signal
        required_fields = ['pair', 'symbol1', 'symbol2', 'qty1', 'qty2', 'recommend']
        missing = [f for f in required_fields if f not in signal]
        if missing:
            return False, f"Missing signal fields: {missing}"
        
        # Check capital availability
        capital_required = signal.get('capital_required', 0)
        available_capital = self.get_capital()
        
        if capital_required > available_capital:
            return False, f"Insufficient capital: Need ₹{capital_required:,.0f}, Have ₹{available_capital:,.0f}"
        
        # Check position limits
        open_positions = self.get_open_positions()
        if len(open_positions) >= config.MAX_POSITIONS:
            return False, f"Maximum positions ({config.MAX_POSITIONS}) reached"
        
        try:
            if self.mode == "PAPER":
                # Paper trading
                self.trader.add_pair_trade(
                    pair_name=signal['pair'],
                    symbol1=signal['symbol1'],
                    symbol2=signal['symbol2'],
                    qty1=int(signal['qty1']),
                    qty2=int(signal['qty2']),
                    entry_price1=float(signal.get('price1', 0)),
                    entry_price2=float(signal.get('price2', 0)),
                    signal_type=signal['recommend'],
                    comment=comment
                )
                msg = f"✅ PAPER TRADE: {signal['pair']} executed"
                logger.info(msg)
                return True, msg
            
            else:
                # LIVE trading
                symbol1 = signal['symbol1']
                symbol2 = signal['symbol2']
                qty1 = int(signal['qty1'])
                qty2 = int(signal['qty2'])
                
                # Determine transaction type based on recommendation
                if signal['recommend'] in ['BUY', 'STRONG BUY']:
                    trans1 = 'BUY'
                    trans2 = 'SELL'  # Pair trade: buy leg1, sell leg2
                else:
                    trans1 = 'SELL'
                    trans2 = 'BUY'
                
                # Execute leg 1
                order1 = self.trader.place_order(
                    symbol=symbol1,
                    quantity=qty1,
                    transaction_type=trans1,
                    order_type='MARKET',
                    product='MIS'  # Intraday for now
                )
                
                if not order1['success']:
                    return False, f"Leg 1 failed: {order1['message']}"
                
                # Execute leg 2
                order2 = self.trader.place_order(
                    symbol=symbol2,
                    quantity=qty2,
                    transaction_type=trans2,
                    order_type='MARKET',
                    product='MIS'
                )
                
                if not order2['success']:
                    # Try to reverse leg 1
                    logger.error(f"Leg 2 failed, attempting to reverse leg 1")
                    reverse_trans = 'SELL' if trans1 == 'BUY' else 'BUY'
                    self.trader.place_order(
                        symbol=symbol1,
                        quantity=qty1,
                        transaction_type=reverse_trans,
                        order_type='MARKET',
                        product='MIS'
                    )
                    return False, f"Leg 2 failed: {order2['message']}. Leg 1 reversed."
                
                msg = f"🔴 LIVE TRADE: {signal['pair']} executed | Order1: {order1['order_id']} | Order2: {order2['order_id']}"
                logger.info(msg)
                return True, msg
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False, f"Error: {str(e)}"
    
    def close_pair_trade(self, pair_id: str) -> Tuple[bool, str]:
        """
        Close an open pair trade
        
        Args:
            pair_id: Pair identifier (e.g., "RELIANCE-INFY")
        
        Returns:
            (success: bool, message: str)
        """
        try:
            if self.mode == "PAPER":
                success = self.trader.close_pair_trade(pair_id)
                if success:
                    msg = f"✅ PAPER TRADE CLOSED: {pair_id}"
                    logger.info(msg)
                    return True, msg
                else:
                    return False, f"Failed to close {pair_id}"
            
            else:
                # LIVE mode: Get position and reverse
                positions = self.trader.get_positions()
                
                # Find positions matching this pair
                pair_symbols = pair_id.split('-')
                if len(pair_symbols) != 2:
                    return False, f"Invalid pair format: {pair_id}"
                
                symbol1, symbol2 = pair_symbols
                
                pos1 = next((p for p in positions if p['symbol'] == symbol1), None)
                pos2 = next((p for p in positions if p['symbol'] == symbol2), None)
                
                if not pos1 or not pos2:
                    return False, f"Position not found for {pair_id}"
                
                # Close leg 1
                trans1 = 'SELL' if pos1['quantity'] > 0 else 'BUY'
                order1 = self.trader.place_order(
                    symbol=symbol1,
                    quantity=abs(pos1['quantity']),
                    transaction_type=trans1,
                    order_type='MARKET',
                    product=pos1['product']
                )
                
                # Close leg 2
                trans2 = 'SELL' if pos2['quantity'] > 0 else 'BUY'
                order2 = self.trader.place_order(
                    symbol=symbol2,
                    quantity=abs(pos2['quantity']),
                    transaction_type=trans2,
                    order_type='MARKET',
                    product=pos2['product']
                )
                
                if order1['success'] and order2['success']:
                    msg = f"🔴 LIVE TRADE CLOSED: {pair_id}"
                    logger.info(msg)
                    return True, msg
                else:
                    return False, f"Partial close: Order1={order1['success']}, Order2={order2['success']}"
                    
        except Exception as e:
            logger.error(f"Close trade failed: {e}")
            return False, f"Error: {str(e)}"
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if self.mode == "PAPER":
            all_trades = self.trader.get_all_pair_trades()
            return [t for t in all_trades if t.get('status') == 'OPEN']
        else:
            # LIVE mode: Get from broker
            positions = self.trader.get_positions()
            return positions
    
    def get_closed_trades(self) -> List[Dict]:
        """Get all closed trades"""
        if self.mode == "PAPER":
            all_trades = self.trader.get_all_pair_trades()
            return [t for t in all_trades if t.get('status') in ['CLOSED', 'EXITED']]
        else:
            # LIVE mode: Get trade history (limited by broker API)
            # This would need to be stored in local database
            logger.warning("LIVE trade history not yet implemented")
            return []
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance metrics"""
        if self.mode == "PAPER":
            return self.trader.get_overall_performance()
        else:
            # LIVE mode: Calculate from positions and margins
            margins = self.trader.get_margins()
            equity = margins.get('equity', {})
            
            return {
                'mode': 'LIVE',
                'capital': equity.get('net', 0.0),
                'initial_capital': self.initial_capital,
                'total_pnl': equity.get('net', 0.0) - self.initial_capital,
                'total_pnl_pct': ((equity.get('net', 0.0) - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0,
                'open_positions': len(self.get_open_positions()),
                'available_margin': margins.get('available', {}).get('cash', 0.0),
                'used_margin': margins.get('utilised', {}).get('debits', 0.0)
            }
    
    def get_mode_history(self) -> List[Dict]:
        """Get history of mode switches"""
        return self.mode_history
    
    def export_trade_history(self, filename: Optional[str] = None) -> str:
        """
        Export trade history to CSV
        
        Args:
            filename: Output filename (optional)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"trade_history_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = Path(config.DATA_DIR) / filename
        
        if self.mode == "PAPER":
            all_trades = self.trader.get_all_pair_trades()
            df = pd.DataFrame(all_trades)
        else:
            positions = self.get_open_positions()
            df = pd.DataFrame(positions)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Trade history exported to {output_path}")
        return str(output_path)
