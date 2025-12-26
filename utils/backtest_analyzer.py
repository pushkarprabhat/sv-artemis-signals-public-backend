"""
Backtest Analyzer - Calculate performance metrics and analytics
Metrics: Win rate, Sharpe ratio, drawdown, recovery factor, profit factor, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import logger


class BacktestAnalyzer:
    """Analyze backtest results and calculate performance metrics"""
    
    def __init__(self, trades_df, initial_capital=100000, risk_free_rate=0.10):
        """
        Initialize analyzer with trades dataframe
        
        Args:
            trades_df: DataFrame with trades (must include: entry_date, exit_date, entry_price, 
                       exit_price, quantity, profit_loss, strategy, symbol)
            initial_capital: Starting capital in INR
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default 10% for India)
        """
        self.trades_df = trades_df.copy() if not trades_df.empty else pd.DataFrame()
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.metrics = {}
    
    def calculate_returns(self):
        """Calculate returns series"""
        if self.trades_df.empty:
            return pd.Series(), 0
        
        # Calculate cumulative P&L
        cumulative_pnl = self.trades_df['profit_loss'].cumsum()
        
        # Calculate daily returns
        daily_returns = cumulative_pnl / self.initial_capital
        
        # Total return
        total_return = cumulative_pnl.iloc[-1] / self.initial_capital if len(cumulative_pnl) > 0 else 0
        
        return daily_returns, total_return
    
    def calculate_win_rate(self):
        """Calculate win rate and loss rate"""
        if self.trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'loss_rate': 0.0,
            }
        
        total = len(self.trades_df)
        winners = len(self.trades_df[self.trades_df['profit_loss'] > 0])
        losers = len(self.trades_df[self.trades_df['profit_loss'] < 0])
        
        return {
            'total_trades': total,
            'winning_trades': winners,
            'losing_trades': losers,
            'win_rate': winners / total if total > 0 else 0,
            'loss_rate': losers / total if total > 0 else 0,
        }
    
    def calculate_profit_metrics(self):
        """Calculate avg win, avg loss, profit factor"""
        if self.trades_df.empty:
            return {
                'total_profit': 0,
                'total_loss': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
            }
        
        winning_trades = self.trades_df[self.trades_df['profit_loss'] > 0]['profit_loss']
        losing_trades = self.trades_df[self.trades_df['profit_loss'] < 0]['profit_loss']
        
        total_profit = winning_trades.sum()
        total_loss = abs(losing_trades.sum())
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        
        # Profit factor = Total profit / Total loss
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        win_rate = len(winning_trades) / len(self.trades_df)
        loss_rate = len(losing_trades) / len(self.trades_df)
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
        }
    
    def calculate_drawdown(self):
        """Calculate maximum drawdown and recovery factor"""
        if self.trades_df.empty:
            return {
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'recovery_factor': 0,
                'drawdown_duration': 0,
            }
        
        # Calculate running maximum capital
        cumulative_pnl = self.trades_df['profit_loss'].cumsum()
        running_max = cumulative_pnl.cummax()
        
        # Drawdown = Current - Running Max
        drawdown = cumulative_pnl - running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        
        # Drawdown duration (days between max and recovery)
        max_dd_idx = drawdown.idxmin()
        recovery_idx = (drawdown[max_dd_idx:] >= 0).idxmax()
        drawdown_duration = (recovery_idx - max_dd_idx) if recovery_idx > max_dd_idx else 0
        
        # Recovery factor = Total profit / Max drawdown
        total_profit = cumulative_pnl.iloc[-1]
        recovery_factor = total_profit / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'recovery_factor': recovery_factor,
            'drawdown_duration': drawdown_duration,
        }
    
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio (annualized)"""
        if self.trades_df.empty:
            return 0
        
        # Get returns
        _, total_return = self.calculate_returns()
        
        # Calculate daily returns
        cumulative_pnl = self.trades_df['profit_loss'].cumsum()
        daily_returns = cumulative_pnl / self.initial_capital
        daily_returns_pct = daily_returns.pct_change().dropna()
        
        if len(daily_returns_pct) == 0:
            return 0
        
        # Annualize Sharpe (assuming 250 trading days per year)
        excess_return = total_return - (self.risk_free_rate / 250)  # Daily risk-free rate
        volatility = daily_returns_pct.std() * np.sqrt(250)  # Annualized volatility
        
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        return sharpe
    
    def calculate_calmar_ratio(self):
        """Calculate Calmar ratio = Annual Return / Max Drawdown"""
        if self.trades_df.empty:
            return 0
        
        _, total_return = self.calculate_returns()
        dd_metrics = self.calculate_drawdown()
        
        annual_return = total_return  # Simplified (should be annualized)
        max_dd = abs(dd_metrics['max_drawdown_pct']) / 100
        
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        return calmar
    
    def get_all_metrics(self):
        """Calculate and return all metrics"""
        self.metrics = {
            'summary': {
                'total_trades': len(self.trades_df),
                'analysis_period': f"{self.trades_df['entry_date'].min()} to {self.trades_df['entry_date'].max()}" if not self.trades_df.empty else "N/A",
                'initial_capital': self.initial_capital,
            },
            'returns': {
                'total_return_pct': (self.calculate_returns()[1] * 100) if len(self.trades_df) > 0 else 0,
                'total_profit': self.trades_df['profit_loss'].sum() if not self.trades_df.empty else 0,
            },
            'win_metrics': self.calculate_win_rate(),
            'profit_metrics': self.calculate_profit_metrics(),
            'risk_metrics': self.calculate_drawdown(),
            'ratios': {
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'calmar_ratio': self.calculate_calmar_ratio(),
            }
        }
        
        return self.metrics
    
    def print_summary(self):
        """Print summary of metrics"""
        metrics = self.get_all_metrics()
        
        print("\n" + "="*80)
        print(" BACKTEST ANALYSIS SUMMARY")
        print("="*80)
        
        # Summary
        print(f"\n📊 SUMMARY")
        print(f"  Total Trades: {metrics['summary']['total_trades']}")
        print(f"  Period: {metrics['summary']['analysis_period']}")
        print(f"  Initial Capital: ₹{metrics['summary']['initial_capital']:,.0f}")
        
        # Returns
        print(f"\n💰 RETURNS")
        print(f"  Total Return: {metrics['returns']['total_return_pct']:.2f}%")
        print(f"  Total Profit: ₹{metrics['returns']['total_profit']:,.0f}")
        
        # Win/Loss
        print(f"\n📈 WIN/LOSS METRICS")
        print(f"  Winning Trades: {metrics['win_metrics']['winning_trades']} ({metrics['win_metrics']['win_rate']*100:.1f}%)")
        print(f"  Losing Trades: {metrics['win_metrics']['losing_trades']} ({metrics['win_metrics']['loss_rate']*100:.1f}%)")
        print(f"  Average Win: ₹{metrics['profit_metrics']['avg_win']:,.0f}")
        print(f"  Average Loss: ₹{metrics['profit_metrics']['avg_loss']:,.0f}")
        
        # Risk
        print(f"\n⚠️  RISK METRICS")
        print(f"  Max Drawdown: ₹{metrics['risk_metrics']['max_drawdown']:,.0f} ({metrics['risk_metrics']['max_drawdown_pct']:.2f}%)")
        print(f"  Profit Factor: {metrics['profit_metrics']['profit_factor']:.2f}x")
        print(f"  Recovery Factor: {metrics['risk_metrics']['recovery_factor']:.2f}x")
        
        # Ratios
        print(f"\n📊 PERFORMANCE RATIOS")
        print(f"  Sharpe Ratio: {metrics['ratios']['sharpe_ratio']:.2f}")
        print(f"  Calmar Ratio: {metrics['ratios']['calmar_ratio']:.2f}")
        print(f"  Expectancy: ₹{metrics['profit_metrics']['expectancy']:,.0f} per trade")
        
        print("\n" + "="*80)
        
        return metrics
    
    def to_dataframe(self):
        """Return metrics as a formatted dataframe"""
        metrics = self.get_all_metrics()
        
        rows = [
            ['Total Trades', metrics['summary']['total_trades']],
            ['Winning Trades', metrics['win_metrics']['winning_trades']],
            ['Losing Trades', metrics['win_metrics']['losing_trades']],
            ['Win Rate (%)', f"{metrics['win_metrics']['win_rate']*100:.1f}%"],
            ['Total Return (%)', f"{metrics['returns']['total_return_pct']:.2f}%"],
            ['Total Profit (₹)', f"{metrics['returns']['total_profit']:,.0f}"],
            ['Avg Win (₹)', f"{metrics['profit_metrics']['avg_win']:,.0f}"],
            ['Avg Loss (₹)', f"{metrics['profit_metrics']['avg_loss']:,.0f}"],
            ['Profit Factor', f"{metrics['profit_metrics']['profit_factor']:.2f}x"],
            ['Max Drawdown (₹)', f"{metrics['risk_metrics']['max_drawdown']:,.0f}"],
            ['Max Drawdown (%)', f"{metrics['risk_metrics']['max_drawdown_pct']:.2f}%"],
            ['Recovery Factor', f"{metrics['risk_metrics']['recovery_factor']:.2f}x"],
            ['Sharpe Ratio', f"{metrics['ratios']['sharpe_ratio']:.2f}"],
            ['Calmar Ratio', f"{metrics['ratios']['calmar_ratio']:.2f}"],
            ['Expectancy (₹/trade)', f"{metrics['profit_metrics']['expectancy']:,.0f}"],
        ]
        
        return pd.DataFrame(rows, columns=['Metric', 'Value'])


def analyze_trades(trades_df, initial_capital=100000):
    """
    Convenience function to quickly analyze trades
    
    Args:
        trades_df: DataFrame with trade results
        initial_capital: Starting capital
    
    Returns:
        BacktestAnalyzer object with all metrics calculated
    """
    analyzer = BacktestAnalyzer(trades_df, initial_capital=initial_capital)
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("Backtest Analyzer Module")
    print("Usage: analyzer = BacktestAnalyzer(trades_df)")
    print("       metrics = analyzer.get_all_metrics()")
    print("       analyzer.print_summary()")
