# Paper Trading Engine - Zero Risk Learning
# Track entries, exits, and real-time performance
# Professional paper trading: Test strategies before live execution

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

from utils.logger import logger
from config import PAPER_CAPITAL, PAPER_CHALLENGE_DAYS, MAX_RISK_PER_TRADE_PCT, BROKERAGE_PER_TRADE


class Trade:
    """Represents a single paper trade (entry to exit)"""
    
    def __init__(self, 
                 trade_id: str,
                 symbol: str,
                 entry_price: float,
                 entry_time: datetime,
                 quantity: int = 1,
                 trade_type: str = "LONG",
                 strategy: str = "MANUAL"):
        
        self.trade_id = trade_id
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.quantity = quantity
        self.trade_type = trade_type
        self.strategy = strategy
        
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: Optional[str] = None
        
    def is_open(self) -> bool:
        """Check if trade is still active"""
        return self.exit_price is None
    
    def close(self, exit_price: float, exit_time: datetime, exit_reason: str = "MANUAL"):
        """Close the trade with exit details"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
    
    def get_pnl(self, current_price: Optional[float] = None) -> float:
        """Calculate unrealized or realized PnL"""
        price = current_price if current_price is not None else self.exit_price
        
        if price is None:
            return 0.0
        
        if self.trade_type == "LONG":
            pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            pnl = (self.entry_price - price) * self.quantity
        
        # Subtract brokerage on both legs
        pnl -= BROKERAGE_PER_TRADE * 2
        
        return pnl
    
    def get_return_pct(self, current_price: Optional[float] = None) -> float:
        """Calculate return percentage"""
        pnl = self.get_pnl(current_price)
        capital_at_risk = self.entry_price * self.quantity
        return (pnl / capital_at_risk * 100) if capital_at_risk > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/display"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'quantity': self.quantity,
            'trade_type': self.trade_type,
            'strategy': self.strategy,
            'exit_reason': self.exit_reason,
            'pnl': self.get_pnl(),
            'return_pct': self.get_return_pct()
        }


class PaperTrader:
    """
    Paper Trading Engine - Prove it works before risking real capital
    
    Features:
    - Enter trades manually or via signals
    - Track open and closed trades in real-time
    - Monitor live P&L and performance metrics
    - Calculate Sharpe ratio, max drawdown, win rate
    - 30-day challenge tracking
    """
    
    def __init__(self, 
                 initial_capital: float = PAPER_CAPITAL,
                 challenge_days: int = PAPER_CHALLENGE_DAYS):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.challenge_days = challenge_days
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(days=challenge_days)
        
        self.trades: Dict[str, Trade] = {}  # {trade_id: Trade object}
        self.trade_counter = 0
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.equity_timestamps: List[datetime] = [self.start_time]
        
        logger.info(f"Paper Trader started with {initial_capital:,.0f} capital for {challenge_days} days")
    
    def enter_trade(self,
                    symbol: str,
                    entry_price: float,
                    quantity: int = 1,
                    trade_type: str = "LONG",
                    strategy: str = "MANUAL") -> str:
        """
        Enter a new paper trade
        
        Args:
            symbol: Stock/pair symbol
            entry_price: Entry price
            quantity: Number of shares/units
            trade_type: LONG or SHORT
            strategy: Strategy name
        
        Returns:
            trade_id for tracking
        """
        
        # Calculate capital required
        capital_required = entry_price * quantity
        
        if capital_required > self.current_capital * (1 - MAX_RISK_PER_TRADE_PCT):
            logger.warning(f"Insufficient capital to enter {symbol}")
            return None
        
        # Create trade
        self.trade_counter += 1
        trade_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.trade_counter}"
        
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            trade_type=trade_type,
            strategy=strategy
        )
        
        self.trades[trade_id] = trade
        self.current_capital -= capital_required
        
        logger.info(f"Trade entered: {trade_type} {quantity}x {symbol} @ {entry_price}")
        
        return trade_id
    
    def exit_trade(self,
                   trade_id: str,
                   exit_price: float,
                   exit_reason: str = "MANUAL") -> float:
        """
        Exit an open trade
        
        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_reason: Reason for exit
        
        Returns:
            PnL from the trade
        """
        
        if trade_id not in self.trades:
            logger.error(f"Trade {trade_id} not found")
            return 0.0
        
        trade = self.trades[trade_id]
        
        if not trade.is_open():
            logger.warning(f"Trade {trade_id} is already closed")
            return trade.get_pnl()
        
        # Close the trade
        trade.close(exit_price, datetime.now(), exit_reason)
        
        # Update capital
        pnl = trade.get_pnl()
        self.current_capital += trade.entry_price * trade.quantity + pnl
        
        logger.info(f"Trade closed: {trade.symbol} @ {exit_price}, PnL: {pnl}")
        
        return pnl
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades as DataFrame"""
        
        open_trades = [t for t in self.trades.values() if t.is_open()]
        
        if not open_trades:
            return pd.DataFrame()
        
        data = []
        for trade in open_trades:
            data.append({
                'Trade ID': trade.trade_id,
                'Symbol': trade.symbol,
                'Type': trade.trade_type,
                'Entry Price': f"Rs.{trade.entry_price:,.2f}",
                'Qty': trade.quantity,
                'Strategy': trade.strategy,
                'Entry Time': trade.entry_time.strftime("%H:%M:%S"),
                'Duration': str(datetime.now() - trade.entry_time).split('.')[0]
            })
        
        return pd.DataFrame(data)
    
    def get_closed_trades(self) -> pd.DataFrame:
        """Get all closed trades as DataFrame"""
        
        closed_trades = [t for t in self.trades.values() if not t.is_open()]
        
        if not closed_trades:
            return pd.DataFrame()
        
        data = []
        for trade in closed_trades:
            pnl = trade.get_pnl()
            return_pct = trade.get_return_pct()
            
            data.append({
                'Symbol': trade.symbol,
                'Type': trade.trade_type,
                'Entry': f"Rs.{trade.entry_price:,.2f}",
                'Exit': f"Rs.{trade.exit_price:,.2f}",
                'PnL': f"Rs.{pnl:,.0f}",
                'Return': f"{return_pct:+.2f}%",
                'Exit Reason': trade.exit_reason,
                'Duration': str(trade.exit_time - trade.entry_time).split('.')[0]
            })
        
        return pd.DataFrame(data)
    
    def get_performance(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            dict with performance stats
        """
        
        closed_trades = [t for t in self.trades.values() if not t.is_open()]
        
        # Basic metrics
        total_pnl = sum(t.get_pnl() for t in closed_trades)
        total_return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital * 100)
        
        # Win rate
        winning_trades = [t for t in closed_trades if t.get_pnl() > 0]
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0
        
        # Sharpe ratio (daily returns)
        if len(self.equity_curve) > 1:
            daily_returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            daily_returns = daily_returns[~np.isnan(daily_returns)]
            
            if len(daily_returns) > 0:
                avg_daily_return = np.mean(daily_returns)
                std_daily_return = np.std(daily_returns)
                sharpe = (avg_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown_pct = np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0
        
        # Days remaining
        days_remaining = max(0, (self.end_time - datetime.now()).days)
        
        return {
            'total_capital': self.current_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'num_trades': len(closed_trades),
            'sharpe': sharpe,
            'max_drawdown_pct': max_drawdown_pct,
            'days_remaining': days_remaining
        }
