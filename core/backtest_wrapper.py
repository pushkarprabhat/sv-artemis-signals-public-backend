# Backtester wrapper - Connect to actual backtester logic
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from config import BASE_DIR, Z_ENTRY, Z_EXIT, MAX_HOLDING_DAYS
from core.backtester import backtest_pair_with_strangle_hedge
from core.pairs import get_all_pairs, load_price
from utils.logger import logger

def backtest_all_strategies(timeframe="day", years=2, capital=100000):
    """Run backtest on all pair strategies with real backtester
    
    Args:
        timeframe: Which timeframe to backtest (day, week, etc.)
        years: Number of years of history to use
        capital: Initial capital for backtest
    
    Returns:
        DataFrame with columns: pair, total_return, sharpe_ratio, max_drawdown, num_trades, final_capital
    """
    try:
        # Get all valid pairs
        pairs = get_all_pairs(timeframe, capital_limit=capital)
        
        if not pairs or len(pairs) == 0:
            logger.warning(f"No pairs found for backtesting on {timeframe}")
            return pd.DataFrame()
        
        results = []
        
        # Date range for backtest
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * years)
        
        for pair in pairs[:20]:  # Limit to top 20 pairs to avoid long processing
            try:
                symbol1, symbol2 = pair.split('-')
                
                # Load price data for both symbols
                price1 = load_price(symbol1, timeframe)
                price2 = load_price(symbol2, timeframe)
                
                if price1 is None or price2 is None or len(price1) < 200 or len(price2) < 200:
                    continue
                
                # Get common dates
                common_dates = price1.index.intersection(price2.index)
                if len(common_dates) < 100:
                    continue
                
                # Create pair dataframe
                pair_data = pd.DataFrame({
                    'leg1': price1.loc[common_dates].values,
                    'leg2': price2.loc[common_dates].values
                }, index=common_dates)
                
                # Run backtest
                backtest_result = backtest_pair_with_strangle_hedge(
                    pair_data,
                    strangle_premium=200,
                    window=60
                )
                
                # Calculate additional metrics
                equity_curve = np.array(backtest_result['equity_curve'])
                returns = np.diff(equity_curve) / equity_curve[:-1]
                
                # Sharpe Ratio (assuming 252 trading days)
                if len(returns) > 0:
                    annual_returns = np.mean(returns) * 252
                    annual_volatility = np.std(returns) * np.sqrt(252)
                    sharpe = annual_returns / annual_volatility if annual_volatility > 0 else 0
                else:
                    sharpe = 0
                
                # Max Drawdown
                cumulative = np.cumprod(1 + returns) if len(returns) > 0 else np.array([1])
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                results.append({
                    'pair': pair,
                    'total_return': round(backtest_result['return_pct'], 2),
                    'sharpe_ratio': round(sharpe, 2),
                    'max_drawdown': round(max_drawdown * 100, 2),
                    'num_trades': backtest_result['trades'],
                    'final_capital': round(backtest_result['final_equity'], 2),
                    'final_equity': round(backtest_result['final_equity'], 2)
                })
                
                logger.info(f"Backtested {pair}: Return={backtest_result['return_pct']:.2f}%, Trades={backtest_result['trades']}")
                
            except Exception as e:
                logger.debug(f"Backtest failed for {pair}: {e}")
                continue
        
        if not results:
            logger.warning("No successful backtests completed")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        logger.info(f"Completed backtests for {len(results_df)} pairs")
        return results_df
        
    except Exception as e:
        logger.error(f"Backtest wrapper error: {e}")
        return pd.DataFrame()
