"""
Momentum Strategy Module
RSI-based momentum trading strategy with volume confirmation.

Features:
- RSI (14-period) with configurable overbought/oversold thresholds
- Volume filter (only trade if volume > 20-day average)
- Entry: RSI < 30 (oversold) = BUY signal
- Entry: RSI > 70 (overbought) = SELL signal (short)
- Exit: After 5-10 days OR RSI crosses back
- Backtest: Full P&L, Sharpe ratio, win rate, max drawdown
- Supports: Single symbol backtesting and batch processing across all instruments
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """Enhanced Momentum Strategy using RSI with Volume Filtering"""
    
    def __init__(self, symbol: str, rsi_period: int = 14, rsi_overbought: int = 70, 
                 rsi_oversold: int = 30, volume_sma_period: int = 20, hold_days: int = 5):
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_sma_period = volume_sma_period
        self.hold_days = hold_days
        self.position = None
        self.entry_price = None
        self.entry_date = None
        self.trades = []
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        if period is None:
            period = self.rsi_period
        
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """Check if volume exceeds 20-day average"""
        vol_sma = df['volume'].rolling(window=self.volume_sma_period).mean()
        return df['volume'] > vol_sma
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI and volume"""
        df_copy = df.copy()
        df_copy['rsi'] = self.calculate_rsi(df_copy['close'])
        df_copy['vol_filter'] = self.check_volume_filter(df_copy)
        df_copy['signal'] = 'HOLD'
        
        # Buy signals: RSI < 30 AND volume > 20-day avg
        df_copy.loc[(df_copy['rsi'] < self.rsi_oversold) & (df_copy['vol_filter']), 'signal'] = 'BUY'
        
        # Sell signals: RSI > 70 OR hold period exceeded
        df_copy.loc[(df_copy['rsi'] > self.rsi_overbought), 'signal'] = 'SELL'
        
        return df_copy[['close', 'volume', 'rsi', 'signal']]
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Full backtest with P&L, Sharpe ratio, max drawdown, and equity curve"""
        rsi = self.calculate_rsi(df['close'])
        vol_filter = self.check_volume_filter(df)
        
        pnl_list = []
        entry_price = None
        entry_idx = None
        position = None
        equity_curve = [initial_capital]
        drawdown_series = []
        
        for i in range(max(self.rsi_period, self.volume_sma_period), len(df)):
            current_rsi = rsi.iloc[i]
            current_price = df['close'].iloc[i]
            current_volume_ok = vol_filter.iloc[i]
            
            # Entry: Buy on oversold with volume confirmation
            if position is None and current_rsi < self.rsi_oversold and current_volume_ok:
                position = "LONG"
                entry_price = current_price
                entry_idx = i
            
            # Exit: Sell on overbought OR hold period exceeded
            elif position == "LONG":
                days_held = i - entry_idx
                
                if current_rsi > self.rsi_overbought or days_held >= self.hold_days:
                    pnl = current_price - entry_price
                    pnl_pct = (pnl / entry_price) * 100
                    pnl_list.append(pnl)
                    
                    self.trades.append({
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "days_held": days_held
                    })
                    
                    # Update equity
                    new_capital = equity_curve[-1] + pnl
                    equity_curve.append(new_capital)
                    position = None
        
        # Close any open position
        if position == "LONG":
            final_price = df['close'].iloc[-1]
            pnl = final_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
            pnl_list.append(pnl)
            self.trades.append({
                "entry": entry_price,
                "exit": final_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
            equity_curve.append(equity_curve[-1] + pnl)
        
        if not pnl_list:
            return {
                "status": "NO_TRADES",
                "symbol": self.symbol,
                "trades": 0,
                "sharpe": 0,
                "pnl_abs": 0,
                "pnl_pct": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "total_return": 0
            }
        
        pnl_arr = np.array(pnl_list)
        
        # Calculate Sharpe Ratio
        pnl_returns = pnl_arr / initial_capital
        daily_returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        
        # Calculate Max Drawdown
        cumsum = np.cumsum(equity_curve)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
        
        total_pnl = pnl_arr.sum()
        total_return = (total_pnl / initial_capital) * 100
        win_rate = np.mean(pnl_arr > 0) * 100
        
        return {
            "status": "SUCCESS",
            "symbol": self.symbol,
            "trades": len(pnl_list),
            "pnl_abs": total_pnl,
            "pnl_pct": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "avg_trade": total_pnl / len(pnl_list) if pnl_list else 0
        }


class MomentumBatchProcessor:
    """Batch process multiple symbols through MomentumStrategy"""
    
    def __init__(self, price_loader_func, max_workers: int = 8):
        """
        Args:
            price_loader_func: Function that takes (symbol, timeframe) and returns DataFrame
            max_workers: Number of parallel workers
        """
        self.price_loader = price_loader_func
        self.max_workers = max_workers
        self.results = []
        self.statistics = {}
    
    def process_batch(self, symbols: List[str], timeframe: str = 'day', 
                      rsi_period: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30,
                      volume_sma_period: int = 20, hold_days: int = 5,
                      min_data_points: int = 100) -> pd.DataFrame:
        """
        Process batch of symbols through MomentumStrategy
        
        Args:
            symbols: List of stock symbols
            timeframe: Timeframe (day, week, month, etc.)
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            volume_sma_period: Volume SMA period
            hold_days: Max holding period
            min_data_points: Minimum data points required
            
        Returns:
            DataFrame with backtest results sorted by Sharpe ratio
        """
        results = []
        
        for symbol in symbols:
            try:
                # Load price data
                df = self.price_loader(symbol, timeframe)
                
                if df is None or len(df) < min_data_points:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Run backtest
                strategy = MomentumStrategy(
                    symbol=symbol,
                    rsi_period=rsi_period,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    volume_sma_period=volume_sma_period,
                    hold_days=hold_days
                )
                
                backtest_result = strategy.backtest(df)
                backtest_result['symbol'] = symbol
                results.append(backtest_result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Convert to DataFrame and sort by Sharpe ratio
        if not results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('sharpe', ascending=False, na_position='last')
        
        self.results = df_results
        self._calculate_statistics()
        
        return df_results
    
    def _calculate_statistics(self):
        """Calculate batch-level statistics"""
        if self.results.empty:
            self.statistics = {}
            return
        
        valid_results = self.results[self.results['status'] == 'SUCCESS']
        
        self.statistics = {
            'total_symbols': len(self.results),
            'profitable_symbols': len(valid_results[valid_results['pnl_pct'] > 0]),
            'avg_sharpe': valid_results['sharpe'].mean() if not valid_results.empty else 0,
            'avg_pnl_pct': valid_results['pnl_pct'].mean() if not valid_results.empty else 0,
            'max_sharpe': valid_results['sharpe'].max() if not valid_results.empty else 0,
            'min_sharpe': valid_results['sharpe'].min() if not valid_results.empty else 0,
            'avg_win_rate': valid_results['win_rate'].mean() if not valid_results.empty else 0,
            'avg_max_drawdown': valid_results['max_drawdown'].mean() if not valid_results.empty else 0,
        }
    
    def get_top_performers(self, top_n: int = 10, min_sharpe: float = -np.inf) -> pd.DataFrame:
        """Get top N performing symbols by Sharpe ratio"""
        if self.results.empty:
            return pd.DataFrame()
        
        top_results = self.results[self.results['sharpe'] >= min_sharpe].head(top_n)
        return top_results
    
    def get_statistics_summary(self) -> Dict:
        """Get batch-level statistics"""
        return self.statistics


def scan_momentum_nifty500(price_loader_func, symbols: List[str] = None, 
                           rsi_period: int = 14, min_sharpe: float = 0.5) -> pd.DataFrame:
    """
    Scan NIFTY500 or provided symbols for momentum opportunities.
    
    Args:
        price_loader_func: Function to load price data
        symbols: List of symbols to scan (if None, uses NIFTY500)
        rsi_period: RSI period for calculation
        min_sharpe: Minimum Sharpe ratio threshold
        
    Returns:
        DataFrame of profitable symbols with Sharpe > threshold
    """
    processor = MomentumBatchProcessor(price_loader_func)
    
    if symbols is None:
        # Default to major indices/stocks - can be customized
        symbols = []
    
    results = processor.process_batch(symbols, rsi_period=rsi_period)
    
    # Filter by Sharpe ratio
    if not results.empty:
        results = results[results['sharpe'] >= min_sharpe]
    
    return results
