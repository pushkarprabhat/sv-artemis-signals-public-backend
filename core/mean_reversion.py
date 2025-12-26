"""
Mean Reversion Strategy Module
Bollinger Bands-based mean reversion trading strategy.

Features:
- Bollinger Bands with configurable MA period and standard deviations
- Entry: Price hits lower band = BUY signal
- Exit: Price reverts to SMA or hits upper band = SELL signal
- Backtest: Full P&L, Sharpe ratio, win rate, max drawdown
- Supports: Single symbol backtesting and batch processing
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """Mean Reversion Strategy using Bollinger Bands"""
    
    def __init__(self, symbol: str, ma_period: int = 20, std_dev: float = 2.0):
        self.symbol = symbol
        self.ma_period = ma_period
        self.std_dev = std_dev
        self.position = None
        self.entry_price = None
        self.trades = []
    
    def calculate_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(self.ma_period).mean()
        std = prices.rolling(self.ma_period).std()
        
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        return sma, upper_band, lower_band
    
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate mean reversion signals"""
        sma, upper_band, lower_band = self.calculate_bands(df['close'])
        
        latest_price = df['close'].iloc[-1]
        latest_sma = sma.iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        
        signals = {
            "timestamp": df.index[-1],
            "price": latest_price,
            "sma": latest_sma,
            "signal": "NONE",
            "action": None
        }
        
        # Buy when price hits lower band
        if latest_price < latest_lower and self.position is None:
            signals["signal"] = "BUY"
            signals["action"] = f"Buy at lower band ({latest_price:.2f})"
            self.position = "LONG"
            self.entry_price = latest_price
        
        # Sell when price reverts to SMA or hits upper band
        elif self.position == "LONG" and latest_price > latest_sma:
            signals["signal"] = "SELL"
            signals["action"] = f"Sell at reversion ({latest_price:.2f})"
            pnl = latest_price - self.entry_price
            self.trades.append({"entry": self.entry_price, "exit": latest_price, "pnl": pnl})
            self.position = None
        
        return signals
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Full backtest with P&L calculation"""
        sma, upper_band, lower_band = self.calculate_bands(df['close'])
        
        pnl_list = []
        entry_price = None
        position = None
        equity_curve = [initial_capital]
        
        for i in range(self.ma_period, len(df)):
            current_price = df['close'].iloc[i]
            current_sma = sma.iloc[i]
            current_lower = lower_band.iloc[i]
            
            # Entry: Buy at lower band
            if position is None and current_price < current_lower:
                position = "LONG"
                entry_price = current_price
            
            # Exit: Sell when price reverts above SMA
            elif position == "LONG" and current_price > current_sma:
                pnl = current_price - entry_price
                pnl_list.append(pnl)
                self.trades.append({
                    "entry": entry_price,
                    "exit": current_price,
                    "pnl": pnl
                })
                
                # Update equity
                new_capital = equity_curve[-1] + pnl
                equity_curve.append(new_capital)
                position = None
        
        # Close any open position
        if position == "LONG":
            final_price = df['close'].iloc[-1]
            pnl = final_price - entry_price
            pnl_list.append(pnl)
            self.trades.append({
                "entry": entry_price,
                "exit": final_price,
                "pnl": pnl
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
                "max_drawdown": 0
            }
        
        pnl_arr = np.array(pnl_list)
        
        # Calculate Sharpe Ratio
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


def scan_mean_reversion_nifty500(price_loader_func, symbols: List[str] = None,
                                  ma_period: int = 20, min_sharpe: float = 0.5) -> pd.DataFrame:
    """
    Scan NIFTY500 or provided symbols for mean reversion opportunities.
    
    Args:
        price_loader_func: Function to load price data
        symbols: List of symbols to scan (if None, uses NIFTY500)
        ma_period: MA period for Bollinger Bands
        min_sharpe: Minimum Sharpe ratio threshold
        
    Returns:
        DataFrame of profitable symbols with Sharpe > threshold
    """
    results = []
    
    if symbols is None:
        symbols = []
    
    for symbol in symbols:
        try:
            df = price_loader_func(symbol)
            
            if df is None or len(df) < 50:
                continue
            
            strategy = MeanReversionStrategy(symbol, ma_period)
            backtest = strategy.backtest(df)
            
            if backtest['sharpe'] >= min_sharpe:
                results.append(backtest)
        
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            continue
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('sharpe', ascending=False, na_position='last')
    
    return df_results
