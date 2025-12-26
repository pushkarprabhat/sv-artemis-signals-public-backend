# core/backtester_v2.py — VECTORIZED MULTI-HORIZON BACKTESTING ENGINE
# Production-ready backtesting with performance metrics, equity curves, analytics
# Professional backtesting: Optimized for speed and accuracy

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from config import (
    Z_ENTRY, Z_EXIT, MAX_HOLDING_DAYS, CAPITAL, CAPITAL_PER_TRADE,
    SLIPPAGE_PCT, BROKERAGE_PER_TRADE, BACKTEST_HORIZONS
)
from core.kelly import position_size, half_kelly
from utils.logger import logger
import logging


@dataclass
class BacktestMetrics:
    """Complete backtest result metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # %
    total_return: float  # %
    annual_return: float  # %
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float  # %
    profit_factor: float  # Gross profit / Gross loss
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int
    best_trade: float
    worst_trade: float
    final_equity: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'total_return': round(self.total_return, 2),
            'annual_return': round(self.annual_return, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'best_trade': round(self.best_trade, 2),
            'worst_trade': round(self.worst_trade, 2),
            'final_equity': round(self.final_equity, 2),
        }


@dataclass
class Trade:
    """Individual trade record"""
    entry_date: str
    entry_price: float
    exit_date: Optional[str]
    exit_price: Optional[float]
    quantity: int
    pnl: float
    pnl_percent: float
    holding_days: int
    exit_reason: str
    
    def to_dict(self) -> dict:
        return {
            'entry_date': self.entry_date,
            'entry_price': round(self.entry_price, 2),
            'exit_date': self.exit_date,
            'exit_price': round(self.exit_price, 2) if self.exit_price else None,
            'quantity': self.quantity,
            'pnl': round(self.pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'holding_days': self.holding_days,
            'exit_reason': self.exit_reason,
        }


class VectorizedBacktester:
    """
    Vectorized backtesting engine for pair trading strategies.
    
    Features:
    - Vectorized calculations (fast)
    - Multi-horizon testing (1M/3M/6M/9M/1Y)
    - Complete trade tracking
    - Advanced metrics (Sharpe, Sortino, Profit Factor)
    - Equity curves and drawdown analysis
    """
    
    def __init__(self, capital: float = CAPITAL, slippage: float = SLIPPAGE_PCT,
                 brokerage: float = BROKERAGE_PER_TRADE):
        """Initialize backtester"""
        self.capital = capital
        self.slippage = slippage
        self.brokerage = brokerage
        self.logger = logging.getLogger(__name__)
        
    def backtest(self, pair_data: pd.DataFrame, z_entry: float = Z_ENTRY,
                 z_exit: float = Z_EXIT, window: int = 60) -> Tuple[BacktestMetrics, List[Trade], List[float]]:
        """
        Backtest a pair trading strategy.
        
        Args:
            pair_data: DataFrame with 'leg1', 'leg2' columns
            z_entry: Entry Z-score threshold
            z_exit: Exit Z-score threshold
            window: Rolling window for Z-score
        
        Returns:
            (metrics, trades, equity_curve)
        """
        try:
            df = pair_data.copy()
            if len(df) < window + 20:
                self.logger.warning(f"Insufficient data: {len(df)} rows < {window} window")
                return self._empty_result()
            
            # Calculate spread and Z-score
            df['spread'] = df['leg1'] - df['leg2']
            df['mean'] = df['spread'].rolling(window, min_periods=window).mean()
            df['std'] = df['spread'].rolling(window, min_periods=window).std()
            df['zscore'] = (df['spread'] - df['mean']) / df['std']
            
            # Generate signals: 1 (long), -1 (short), 0 (flat)
            df['signal'] = 0
            df.loc[df['zscore'] < -z_entry, 'signal'] = 1  # Long signal
            df.loc[df['zscore'] > z_entry, 'signal'] = -1  # Short signal
            
            # Track position and trades
            trades = []
            position = 0
            entry_idx = None
            entry_price = None
            equity = [self.capital]
            peak_equity = self.capital
            
            for i in range(len(df)):
                row = df.iloc[i]
                zscore = row['zscore']
                spread = row['spread']
                date = df.index[i]
                
                # Skip if Z-score not available
                if pd.isna(zscore):
                    equity.append(equity[-1])
                    continue
                
                # Entry logic
                if position == 0:
                    if zscore < -z_entry:
                        position = 1
                        entry_idx = i
                        entry_price = spread
                    elif zscore > z_entry:
                        position = -1
                        entry_idx = i
                        entry_price = spread
                    else:
                        equity.append(equity[-1])
                        continue
                
                # Holding logic
                if position != 0:
                    holding_days = (date - df.index[entry_idx]).days if entry_idx is not None else 0
                    exit_condition = False
                    exit_reason = None
                    
                    # Exit conditions
                    if position == 1 and zscore > z_exit:
                        exit_condition = True
                        exit_reason = "Mean reversion (long)"
                    elif position == -1 and zscore < -z_exit:
                        exit_condition = True
                        exit_reason = "Mean reversion (short)"
                    elif holding_days >= MAX_HOLDING_DAYS:
                        exit_condition = True
                        exit_reason = f"Time stop ({MAX_HOLDING_DAYS}d)"
                    
                    if exit_condition:
                        # Calculate P&L
                        pnl = -position * (spread - entry_price)
                        transaction_cost = (self.slippage * abs(spread) * 2) + (self.brokerage * 2)
                        pnl_net = pnl - transaction_cost
                        
                        # Update equity
                        new_equity = equity[-1] * (1 + pnl_net / equity[-1])
                        equity.append(new_equity)
                        
                        # Record trade
                        trades.append(Trade(
                            entry_date=str(df.index[entry_idx].date()),
                            entry_price=entry_price,
                            exit_date=str(date.date()),
                            exit_price=spread,
                            quantity=1,
                            pnl=pnl_net,
                            pnl_percent=(pnl_net / equity[-2]) * 100 if equity[-2] > 0 else 0,
                            holding_days=holding_days,
                            exit_reason=exit_reason
                        ))
                        
                        position = 0
                        peak_equity = max(peak_equity, new_equity)
                    else:
                        # Unrealized P&L
                        unrealized_pnl = -position * (spread - entry_price)
                        unrealized_equity = equity[-1] * (1 + unrealized_pnl / equity[-1]) if equity[-1] > 0 else equity[-1]
                        equity.append(unrealized_equity)
                        peak_equity = max(peak_equity, unrealized_equity)
            
            # Calculate metrics
            metrics = self._calculate_metrics(equity, trades)
            return metrics, trades, equity
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return self._empty_result()
    
    def multi_horizon_backtest(self, pair_data: pd.DataFrame, 
                               horizons: Optional[List[int]] = None) -> Dict[str, BacktestMetrics]:
        """
        Backtest across multiple time horizons.
        
        Args:
            pair_data: Full DataFrame
            horizons: List of horizon days (default: [30, 90, 180, 270, 365])
        
        Returns:
            Dict mapping horizon string to metrics
        """
        if horizons is None:
            horizons = BACKTEST_HORIZONS or [30, 90, 180, 270, 365]
        
        results = {}
        
        for horizon_days in horizons:
            if len(pair_data) >= horizon_days:
                horizon_data = pair_data.tail(horizon_days)
                metrics, _, _ = self.backtest(horizon_data)
                results[f'{horizon_days}d'] = metrics
                self.logger.info(f"Horizon {horizon_days}d: Return={metrics.total_return:.2f}%, Sharpe={metrics.sharpe_ratio:.2f}")
            else:
                results[f'{horizon_days}d'] = self._empty_metrics()
        
        return results
    
    def _calculate_metrics(self, equity: List[float], trades: List[Trade]) -> BacktestMetrics:
        """Calculate comprehensive metrics from equity curve and trades"""
        equity_array = np.array(equity)
        
        # Returns
        total_return = ((equity[-1] - self.capital) / self.capital) * 100
        
        # Annual return (assuming 252 trading days)
        days = len(equity_array)
        annual_return = (equity[-1] / self.capital - 1) * (252 / days) * 100 if days > 0 else 0
        
        # Sharpe Ratio
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        
        annual_vol = np.std(daily_returns) * np.sqrt(252)
        sharpe = (annual_return / 100) / annual_vol if annual_vol > 0 else 0
        
        # Sortino Ratio (downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return / 100) / downside_vol if downside_vol > 0 else 0
        
        # Max Drawdown
        cummax = np.maximum.accumulate(equity_array)
        drawdown = ((equity_array - cummax) / cummax) * 100
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        closed_trades = [t for t in trades if t.exit_date is not None]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_trades = len(closed_trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Consecutive wins/losses
        consecutive_wins = max([len(list(g)) for k, g in self._consecutive_groups(closed_trades, True)], default=0)
        consecutive_losses = max([len(list(g)) for k, g in self._consecutive_groups(closed_trades, False)], default=0)
        
        # Best/worst trades
        best_trade = max([t.pnl for t in closed_trades]) if closed_trades else 0
        worst_trade = min([t.pnl for t in closed_trades]) if closed_trades else 0
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            best_trade=best_trade,
            worst_trade=worst_trade,
            final_equity=equity[-1]
        )
    
    @staticmethod
    def _consecutive_groups(trades: List[Trade], winning: bool):
        """Group consecutive winning/losing trades"""
        from itertools import groupby
        key_func = lambda t: t.pnl > 0 if winning else t.pnl <= 0
        for k, g in groupby(trades, key_func):
            if k:
                yield k, g
    
    def _empty_result(self) -> Tuple[BacktestMetrics, List[Trade], List[float]]:
        """Return empty result"""
        return self._empty_metrics(), [], [self.capital]
    
    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics"""
        return BacktestMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_return=0, annual_return=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            profit_factor=0, avg_win=0, avg_loss=0,
            consecutive_wins=0, consecutive_losses=0,
            best_trade=0, worst_trade=0, final_equity=self.capital
        )


def backtest_pair(pair_data: pd.DataFrame) -> Dict:
    """
    Convenience function for single-horizon backtest.
    
    Args:
        pair_data: DataFrame with leg1, leg2
    
    Returns:
        Dict with metrics, trades, equity_curve
    """
    backtester = VectorizedBacktester()
    metrics, trades, equity = backtester.backtest(pair_data)
    
    return {
        'metrics': metrics.to_dict(),
        'trades': [t.to_dict() for t in trades],
        'equity_curve': equity,
        'metadata': {
            'total_trades': len(trades),
            'profitability': 'PROFITABLE' if metrics.final_equity > CAPITAL else 'UNPROFITABLE'
        }
    }


def backtest_multi_horizon(pair_data: pd.DataFrame, horizons: Optional[List[int]] = None) -> Dict:
    """
    Convenience function for multi-horizon backtest.
    
    Args:
        pair_data: DataFrame with leg1, leg2
        horizons: List of horizon days
    
    Returns:
        Dict mapping horizon to metrics
    """
    backtester = VectorizedBacktester()
    multi_results = backtester.multi_horizon_backtest(pair_data, horizons)
    
    return {horizon: metrics.to_dict() for horizon, metrics in multi_results.items()}


# Example usage
if __name__ == "__main__":
    print("✅ Vectorized Backtester v2.0 Ready")
    print("Usage: from core.backtester_v2 import VectorizedBacktester, backtest_pair, backtest_multi_horizon")
