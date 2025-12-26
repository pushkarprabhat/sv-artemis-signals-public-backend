# core/portfolio.py — PORTFOLIO MANAGEMENT WITH VaR & MONTE CARLO SIMULATION
# Comprehensive portfolio management, risk metrics, position sizing
# Built for Artemis Signals platform

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from utils.logger import logger
from core.models import Trade, PositionType


@dataclass
class Position:
    """Represents an active position"""
    symbol: str
    entry_price: float
    quantity: float
    position_type: PositionType
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def current_value(self) -> float:
        """Current market value"""
        if self.position_type == PositionType.LONG:
            return self.quantity * self.current_price
        else:  # SHORT
            return self.quantity * (2 * self.entry_price - self.current_price)
    
    @property
    def pnl(self) -> float:
        """Current P&L in rupees"""
        if self.position_type == PositionType.LONG:
            return self.quantity * (self.current_price - self.entry_price)
        else:  # SHORT
            return self.quantity * (self.entry_price - self.current_price)
    
    @property
    def pnl_percent(self) -> float:
        """Current P&L as percentage"""
        if self.entry_price == 0:
            return 0
        return (self.pnl / (self.quantity * self.entry_price)) * 100


@dataclass
class VaRMetrics:
    """Value at Risk metrics"""
    var_95: float  # 95% confidence
    var_99: float  # 99% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    max_drawdown: float
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float
    expected_daily_loss: float
    worst_case_loss: float


class Portfolio:
    """
    Portfolio management with:
    - Position tracking
    - Risk metrics (VaR, CVaR, Drawdown)
    - Monte Carlo VaR simulation (10,000 scenarios)
    - Greeks aggregation
    - Position sizing
    """
    
    def __init__(self, capital: float = 100000, max_positions: int = 10):
        """
        Initialize portfolio.
        
        Args:
            capital: Initial capital
            max_positions: Maximum concurrent positions
        """
        self.capital = capital
        self.initial_capital = capital
        self.max_positions = max_positions
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        
        # History tracking
        self.equity_history: List[float] = [capital]
        self.pnl_history: List[float] = [0.0]
        self.timestamps: List[datetime] = [datetime.now()]
        
        # Risk metrics
        self.var_metrics: Optional[VaRMetrics] = None
        self.max_drawdown_ever = 0.0
        
        logger.info(f"✓ Portfolio initialized: Capital = ₹{capital:,.0f}, Max positions = {max_positions}")
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        position_type: PositionType,
        stop_loss: float,
        take_profit: float,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Add a new position to portfolio.
        
        Args:
            symbol: Symbol name
            entry_price: Entry price
            quantity: Quantity
            position_type: LONG or SHORT
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional metadata
        
        Returns:
            True if added successfully
        """
        try:
            if len(self.positions) >= self.max_positions:
                logger.error(f"❌ Max positions ({self.max_positions}) reached")
                return False
            
            if symbol in self.positions:
                logger.error(f"❌ Position '{symbol}' already exists")
                return False
            
            position = Position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                position_type=position_type,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=entry_price,
                metadata=metadata or {}
            )
            
            self.positions[symbol] = position
            
            position_value = quantity * entry_price
            logger.info(f"✓ Position added: {symbol} | {position_type.value} | Qty: {quantity} | Value: ₹{position_value:,.0f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float) -> Optional[Position]:
        """
        Update position with current price.
        
        Args:
            symbol: Symbol name
            current_price: Current market price
        
        Returns:
            Updated position or None if not found
        """
        if symbol not in self.positions:
            logger.warning(f"Position '{symbol}' not found")
            return None
        
        position = self.positions[symbol]
        position.current_price = current_price
        
        return position
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> Optional[Trade]:
        """
        Close an existing position.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            reason: Reason for closure
        
        Returns:
            Closed Trade object or None
        """
        if symbol not in self.positions:
            logger.error(f"Position '{symbol}' not found")
            return None
        
        try:
            position = self.positions[symbol]
            
            # Calculate P&L
            pnl = position.pnl if position.current_price == exit_price else (
                position.quantity * (exit_price - position.entry_price) if position.position_type == PositionType.LONG
                else position.quantity * (position.entry_price - exit_price)
            )
            
            # Create trade record
            trade = Trade(
                trade_id=f"{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                exit_time=datetime.now(),
                exit_price=exit_price,
                quantity=position.quantity,
                position_type=position.position_type,
                pnl=pnl,
                pnl_percent=(pnl / (position.quantity * position.entry_price)) * 100 if position.entry_price > 0 else 0,
                exit_reason=reason
            )
            
            # Update tracking
            self.closed_trades.append(trade)
            del self.positions[symbol]
            
            # Update capital
            self.capital += pnl
            
            logger.info(f"✓ Position closed: {symbol} | Exit: ₹{exit_price:,.2f} | P&L: ₹{pnl:,.2f} ({trade.pnl_percent:+.2f}%)")
            
            return trade
        
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None
    
    # ========================================================================
    # PORTFOLIO METRICS
    # ========================================================================
    
    def get_total_pnl(self) -> Tuple[float, float]:
        """
        Get total P&L.
        
        Returns:
            (total_pnl_rupees, total_pnl_percent)
        """
        # Open positions P&L
        open_pnl = sum([p.pnl for p in self.positions.values()])
        
        # Closed trades P&L
        closed_pnl = sum([t.pnl for t in self.closed_trades])
        
        total_pnl = open_pnl + closed_pnl
        total_pnl_percent = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        return total_pnl, total_pnl_percent
    
    def get_equity(self) -> float:
        """Get current equity (capital + open P&L)"""
        open_pnl = sum([p.pnl for p in self.positions.values()])
        return self.capital + open_pnl
    
    def get_open_pnl(self) -> float:
        """Get P&L from open positions"""
        return sum([p.pnl for p in self.positions.values()])
    
    def get_drawdown(self) -> float:
        """Get current drawdown percentage"""
        current_equity = self.get_equity()
        if self.initial_capital == 0:
            return 0
        
        drawdown = ((self.initial_capital - current_equity) / self.initial_capital) * 100
        
        # Track max drawdown
        if drawdown > self.max_drawdown_ever:
            self.max_drawdown_ever = drawdown
        
        return drawdown
    
    # ========================================================================
    # VALUE AT RISK (VaR) - MONTE CARLO SIMULATION
    # ========================================================================
    
    def calculate_var_monte_carlo(
        self,
        confidence_level: float = 0.95,
        simulations: int = 10000,
        days: int = 1
    ) -> VaRMetrics:
        """
        Calculate Value at Risk using Monte Carlo simulation.
        
        VaR = Maximum loss that could occur with X% confidence over Y days
        
        Example:
            VaR 95% = ₹5,000 means 95% chance daily loss is <= ₹5,000
            Only 5% chance of losing more than ₹5,000 in one day
        
        Args:
            confidence_level: Confidence level (0.95 = 95%)
            simulations: Number of simulations (default 10,000)
            days: Number of days to project (default 1)
        
        Returns:
            VaRMetrics object
        """
        try:
            if not self.closed_trades:
                logger.warning("Not enough historical data for VaR calculation")
                return self._create_empty_var_metrics()
            
            # Get historical returns
            returns = []
            for trade in self.closed_trades[-20:]:  # Use last 20 trades
                if trade.entry_price > 0:
                    ret = (trade.pnl / (trade.quantity * trade.entry_price))
                    returns.append(ret)
            
            if len(returns) < 5:
                logger.warning("Not enough trades for VaR")
                return self._create_empty_var_metrics()
            
            returns = np.array(returns)
            
            # Simulate portfolio returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Monte Carlo simulation
            simulated_returns = np.random.normal(mean_return, std_return, (simulations, days)).sum(axis=1)
            
            # Calculate P&L scenarios
            current_equity = self.get_equity()
            pnl_scenarios = simulated_returns * current_equity
            
            # Calculate VaR metrics
            var_95_loss = np.percentile(pnl_scenarios, (1 - confidence_level) * 100)
            var_99_loss = np.percentile(pnl_scenarios, 1)  # 99% confidence
            
            # Conditional VaR (Expected Shortfall) - average of worst 5%
            worst_5_percent = pnl_scenarios[pnl_scenarios <= var_95_loss]
            cvar_95 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_95_loss
            
            # Greeks aggregation (simplified)
            portfolio_delta = self._aggregate_greeks('delta')
            portfolio_gamma = self._aggregate_greeks('gamma')
            portfolio_theta = self._aggregate_greeks('theta')
            portfolio_vega = self._aggregate_greeks('vega')
            
            var_metrics = VaRMetrics(
                var_95=abs(var_95_loss),
                var_99=abs(var_99_loss),
                cvar_95=abs(cvar_95),
                max_drawdown=self.get_drawdown(),
                portfolio_delta=portfolio_delta,
                portfolio_gamma=portfolio_gamma,
                portfolio_theta=portfolio_theta,
                portfolio_vega=portfolio_vega,
                expected_daily_loss=abs(np.mean(pnl_scenarios[pnl_scenarios < 0])) if len(pnl_scenarios[pnl_scenarios < 0]) > 0 else 0,
                worst_case_loss=abs(np.min(pnl_scenarios))
            )
            
            self.var_metrics = var_metrics
            
            logger.info(f"✓ VaR calculated:")
            logger.info(f"  95% VaR: ₹{var_metrics.var_95:,.0f} (max daily loss with 95% confidence)")
            logger.info(f"  Expected daily loss: ₹{var_metrics.expected_daily_loss:,.0f}")
            logger.info(f"  Worst case: ₹{var_metrics.worst_case_loss:,.0f}")
            
            return var_metrics
        
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return self._create_empty_var_metrics()
    
    def _aggregate_greeks(self, greek: str) -> float:
        """
        Aggregate Greeks across all positions.
        
        Args:
            greek: 'delta', 'gamma', 'theta', or 'vega'
        
        Returns:
            Aggregated greek value
        """
        total = 0.0
        for position in self.positions.values():
            # Get greek from metadata if available
            if greek in position.metadata:
                total += position.metadata[greek]
        return total
    
    def _create_empty_var_metrics(self) -> VaRMetrics:
        """Create empty VaR metrics"""
        return VaRMetrics(
            var_95=0,
            var_99=0,
            cvar_95=0,
            max_drawdown=self.get_drawdown(),
            portfolio_delta=0,
            portfolio_gamma=0,
            portfolio_theta=0,
            portfolio_vega=0,
            expected_daily_loss=0,
            worst_case_loss=0
        )
    
    # ========================================================================
    # POSITION SIZING WITH KELLY CRITERION
    # ========================================================================
    
    def calculate_kelly_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly Criterion: f* = (bp - q) / b
        where:
            f* = fraction of capital to risk
            b = ratio of win to loss
            p = probability of win
            q = probability of loss (1-p)
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount
            kelly_fraction: Kelly fraction (0.25 = 25%, conservative)
        
        Returns:
            Position size (quantity)
        """
        try:
            if entry_price <= 0 or stop_loss == 0:
                return 0
            
            risk_per_trade = abs(entry_price - stop_loss)
            
            if avg_loss == 0:
                return 0
            
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - win_rate
            
            # Kelly formula
            kelly_fraction_calc = (b * p - q) / b
            
            # Apply conservative Kelly fraction
            kelly_fraction_calc *= kelly_fraction
            
            # Limit to reasonable range
            kelly_fraction_calc = max(0.001, min(kelly_fraction_calc, 0.05))  # 0.1% to 5%
            
            # Position size
            position_value = self.capital * kelly_fraction_calc
            position_size = position_value / entry_price
            
            logger.info(f"Kelly position size: {position_size:.2f} units")
            
            return position_size
        
        except Exception as e:
            logger.error(f"Kelly position sizing failed: {e}")
            return 0
    
    # ========================================================================
    # PORTFOLIO STATISTICS
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics"""
        total_pnl, total_pnl_pct = self.get_total_pnl()
        
        closed_trades = self.closed_trades
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        stats = {
            'capital': self.capital,
            'equity': self.get_equity(),
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_pct,
            'open_positions': len(self.positions),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'drawdown': self.get_drawdown(),
            'max_drawdown_ever': self.max_drawdown_ever,
            'var_metrics': self.var_metrics.to_dict() if self.var_metrics else None
        }
        
        return stats
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of all open positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'type': position.position_type.value,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.quantity,
                'value': position.current_value,
                'pnl': position.pnl,
                'pnl_percent': position.pnl_percent,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'risk_reward': (position.take_profit - position.entry_price) / (position.entry_price - position.stop_loss) if position.entry_price != position.stop_loss else 0
            })
        
        return pd.DataFrame(data)
    
    def get_trades_summary(self) -> pd.DataFrame:
        """Get summary of all closed trades as DataFrame"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'duration_seconds': (t.exit_time - t.entry_time).total_seconds() if t.exit_time else 0,
                'exit_reason': t.exit_reason
            }
            for t in self.closed_trades
        ])
