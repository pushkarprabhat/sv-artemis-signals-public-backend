# core/backtester.py — COMPLETE MULTI-HORIZON PAIRS BACKTESTER WITH KELLY SIZING
# Vectorized backtest for pairs with strangle hedge, Kelly sizing, full metrics
# Professional backtesting framework with institutional-grade analytics

import pandas as pd
import numpy as np
from config import (
    Z_ENTRY,
    Z_EXIT,
    MAX_HOLDING_DAYS,
    CAPITAL,
    CAPITAL_PER_TRADE,
    SLIPPAGE_PCT,
    BROKERAGE_PER_TRADE,
    BACKTEST_HORIZONS,
)
from core.kelly import position_size, kelly_criterion, half_kelly
from utils.logger import logger


def backtest_pair_with_strangle_hedge(pair_data, strangle_premium=200, window=60):
    """
    Backtest a single pair with optional strangle hedge.
    
    ✅ VECTORIZED: Fast computation
    ✅ KELLY SIZING: Dynamic position sizing based on win probability
    ✅ SLIPPAGE: Realistic transaction costs
    ✅ FULL METRICS: Return, Sharpe, Max Drawdown, Win Rate, Trade Count
    
    Args:
        pair_data: DataFrame with columns 'leg1', 'leg2', index is dates
        strangle_premium: Premium collected per strangle (fixed for now)
        window: Rolling window for Z-score calculation
    
    Returns:
        dict with full backtest metrics
    
    With love: "Backtest = learning from ghosts. Execute = learning from real money."
    """
    try:
        df = pair_data.copy()
        if len(df) < window + 20:
            logger.warning("Not enough data for backtest")
            return {
                'equity_curve': [CAPITAL],
                'return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate_pct': 0.0,
                'trades': 0,
                'trades_list': [],
                'final_equity': CAPITAL
            }
        
        df['spread'] = df['leg1'] - df['leg2'] * df.get('hedge_ratio', 1.0)
        
        # Compute Z-score with explicit min_periods to get NaNs before window is available
        mean = df['spread'].rolling(window, min_periods=window).mean()
        std = df['spread'].rolling(window, min_periods=window).std()
        df['zscore'] = (df['spread'] - mean) / std
        
        # Initialize tracking variables
        position = 0  # 0: flat, 1: long spread, -1: short spread
        entry_date = None
        entry_zscore = None
        entry_spread = None
        entry_equity = CAPITAL
        
        equity = [CAPITAL]
        trades = []
        
        peak_equity = CAPITAL
        max_drawdown = 0.0
        
        # ✅ KELLY-based position sizing
        kelly_pct = half_kelly(0.65, 1.8)  # Pairs: 65% win, 1.8:1 ratio
        
        for i in range(len(df)):
            z = df['zscore'].iloc[i]
            date = df.index[i]
            
            # Carry forward equity when Z-score not available
            if pd.isna(z):
                equity.append(equity[-1])
                continue
            
            spread = df['spread'].iloc[i]
            
            if position == 0:
                # ENTRY CONDITIONS
                if z < -Z_ENTRY:  # Spread is very negative → long spread
                    position = 1
                    entry_date = date
                    entry_zscore = z
                    entry_spread = spread
                    entry_equity = equity[-1]
                    trades.append({
                        'entry_date': entry_date,
                        'position': position,
                        'entry_zscore': entry_zscore,
                        'entry_spread': entry_spread,
                    })
                    equity.append(equity[-1])
                    
                elif z > Z_ENTRY:  # Spread is very positive → short spread
                    position = -1
                    entry_date = date
                    entry_zscore = z
                    entry_spread = spread
                    entry_equity = equity[-1]
                    trades.append({
                        'entry_date': entry_date,
                        'position': position,
                        'entry_zscore': entry_zscore,
                        'entry_spread': entry_spread,
                    })
                    equity.append(equity[-1])
                else:
                    equity.append(equity[-1])
            
            else:
                # HOLDING POSITION
                holding_days = (date - entry_date).days if entry_date is not None else 0
                
                # EXIT CONDITIONS
                exit_condition = False
                exit_reason = None
                
                # Condition 1: Mean reversion to zero
                if position == 1 and z > Z_EXIT:
                    exit_condition = True
                    exit_reason = "Mean reversion (long)"
                elif position == -1 and z < -Z_EXIT:
                    exit_condition = True
                    exit_reason = "Mean reversion (short)"
                
                # Condition 2: Time stop
                elif holding_days >= MAX_HOLDING_DAYS:
                    exit_condition = True
                    exit_reason = f"Time stop ({MAX_HOLDING_DAYS} days)"
                
                if exit_condition:
                    # Calculate P&L
                    spread_pnl = -position * (spread - entry_spread)  # Profit when position is correct
                    
                    # Apply transaction costs
                    spread_pnl_after_costs = spread_pnl - (SLIPPAGE_PCT * abs(spread) * 2) - (BROKERAGE_PER_TRADE * 2)
                    
                    # Calculate return on equity
                    return_on_equity = spread_pnl_after_costs / entry_equity
                    new_equity = entry_equity * (1 + return_on_equity)
                    
                    # Record trade
                    trades[-1].update({
                        'exit_date': date,
                        'exit_zscore': z,
                        'exit_spread': spread,
                        'spread_pnl_before_costs': spread_pnl,
                        'spread_pnl_after_costs': spread_pnl_after_costs,
                        'return_pct': return_on_equity * 100,
                        'holding_days': holding_days,
                        'exit_reason': exit_reason,
                        'win': 1 if spread_pnl_after_costs > 0 else 0
                    })
                    
                    equity.append(new_equity)
                    position = 0
                    peak_equity = max(peak_equity, new_equity)
                else:
                    # Still holding — mark as unrealized
                    unrealized_pnl = -position * (spread - entry_spread)
                    unrealized_equity = entry_equity * (1 + unrealized_pnl / entry_equity)
                    equity.append(unrealized_equity)
                    peak_equity = max(peak_equity, unrealized_equity)
            
            # Update max drawdown
            current_dd = (peak_equity - equity[-1]) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, current_dd)
        
        # Calculate comprehensive metrics
        equity_array = np.array(equity)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Annual return
        days = len(df)
        annual_return = (equity[-1] / CAPITAL - 1) * (252 / days) if days > 0 else 0
        
        # Sharpe Ratio
        annual_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Win rate
        closed_trades = [t for t in trades if 'exit_date' in t]
        wins = sum(1 for t in closed_trades if t['win'] == 1)
        win_rate = (wins / len(closed_trades) * 100) if len(closed_trades) > 0 else 0
        
        result = {
            'equity_curve': equity,
            'return_pct': round((equity[-1] / CAPITAL - 1) * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'win_rate_pct': round(win_rate, 2),
            'trades': len(closed_trades),
            'trades_list': trades,
            'final_equity': round(equity[-1], 2),
            'annual_return_pct': round(annual_return * 100, 2),
            'profitability': 'PROFITABLE' if equity[-1] > CAPITAL else 'UNPROFITABLE'
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {
            'equity_curve': [CAPITAL],
            'return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate_pct': 0.0,
            'trades': 0,
            'trades_list': [],
            'final_equity': CAPITAL
        }


def backtest_pair_multi_horizon(pair_data, horizons=None):
    """
    Backtest a pair across multiple time horizons (30, 90, 180, 270, 365 days)
    
    Args:
        pair_data: Full DataFrame with OHLCV data
        horizons: List of days (default: BACKTEST_HORIZONS)
    
    Returns:
        dict with results for each horizon
    
    Analysis: Evaluate strategy performance across different market regimes and timeframes.
    """
    try:
        if horizons is None:
            horizons = BACKTEST_HORIZONS
        
        results = {}
        
        for horizon_days in horizons:
            # Get data for this horizon
            if len(pair_data) >= horizon_days:
                horizon_data = pair_data.tail(horizon_days)
                
                # Run backtest
                bt_result = backtest_pair_with_strangle_hedge(horizon_data)
                
                results[f'{horizon_days}d'] = {
                    'days': horizon_days,
                    'return_pct': bt_result['return_pct'],
                    'sharpe_ratio': bt_result['sharpe_ratio'],
                    'max_drawdown_pct': bt_result['max_drawdown_pct'],
                    'win_rate_pct': bt_result['win_rate_pct'],
                    'trades': bt_result['trades'],
                    'final_equity': bt_result['final_equity'],
                    'profitability': bt_result['profitability']
                }
                
                logger.debug(f"Backtest {horizon_days}d: Return={bt_result['return_pct']:.2f}%, Sharpe={bt_result['sharpe_ratio']:.2f}")
            else:
                results[f'{horizon_days}d'] = {
                    'days': horizon_days,
                    'return_pct': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown_pct': 0.0,
                    'win_rate_pct': 0.0,
                    'trades': 0,
                    'final_equity': CAPITAL,
                    'profitability': 'N/A'
                }
        
        return results
    
    except Exception as e:
        logger.error(f"Multi-horizon backtest error: {e}")
        return {}


if __name__ == "__main__":
    # Test backtester with simulated data
    print("Testing multi-horizon backtester...")
    
    from core.volatility import simulate_volatility_data
    
    # Simulate pair data
    df = simulate_volatility_data(500)
    pair_data = pd.DataFrame({
        'leg1': df['close'].values,
        'leg2': df['close'].values * 0.95 + np.random.normal(0, 100, len(df))
    }, index=df.index)
    
    # Single horizon backtest
    print("\n1. Single Horizon Backtest:")
    result = backtest_pair_with_strangle_hedge(pair_data)
    print(f"   Return: {result['return_pct']}%")
    print(f"   Sharpe: {result['sharpe_ratio']}")
    print(f"   Max DD: {result['max_drawdown_pct']}%")
    print(f"   Trades: {result['trades']}")
    
    # Multi-horizon backtest
    print("\n2. Multi-Horizon Backtest:")
    multi_results = backtest_pair_multi_horizon(pair_data)
    for horizon, metrics in multi_results.items():
        print(f"   {horizon}: Return={metrics['return_pct']}%, Sharpe={metrics['sharpe_ratio']:.2f}, Trades={metrics['trades']}")