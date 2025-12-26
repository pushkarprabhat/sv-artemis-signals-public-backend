# utils/backtester.py
"""
Unified Backtester Wrapper
Provides a consistent interface for backtesting all strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestResult:
    """Encapsulates backtest results"""
    
    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any], 
                 result_dict: Dict[str, Any], execution_time_ms: float = 0):
        """Initialize backtest result"""
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.params = params
        self.result = result_dict
        self.execution_time_ms = execution_time_ms
        self.timestamp = datetime.now()
    
    def get_status(self) -> str:
        """Get backtest status"""
        return self.result.get('status', 'UNKNOWN')
    
    def get_sharpe(self) -> float:
        """Get Sharpe ratio"""
        return self.result.get('sharpe', 0.0)
    
    def get_pnl_pct(self) -> float:
        """Get P&L percentage"""
        return self.result.get('pnl_pct', 0.0)
    
    def get_win_rate(self) -> float:
        """Get win rate percentage"""
        return self.result.get('win_rate', 0.0)
    
    def get_max_drawdown(self) -> float:
        """Get max drawdown"""
        return self.result.get('max_drawdown', 0.0)
    
    def get_trades_count(self) -> int:
        """Get number of trades"""
        return self.result.get('trades', 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy': self.strategy_name,
            'symbol': self.symbol,
            'status': self.get_status(),
            'trades': self.get_trades_count(),
            'sharpe': self.get_sharpe(),
            'pnl_pct': self.get_pnl_pct(),
            'win_rate': self.get_win_rate(),
            'max_drawdown': self.get_max_drawdown(),
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp,
        }


class UnifiedBacktester:
    """Unified backtester for all strategies"""
    
    def __init__(self):
        """Initialize backtester"""
        self.results_history = []
    
    def backtest_pairs_trading(self, symbol_a: str, symbol_b: str, df_a: pd.DataFrame, 
                               df_b: pd.DataFrame, params: Dict[str, Any],
                               initial_capital: float = 100000) -> BacktestResult:
        """Backtest pairs trading strategy"""
        import time
        from core.strategies import PairsTradingStrategy
        
        start_time = time.time()
        
        try:
            strategy = PairsTradingStrategy(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                z_entry=params.get('z_entry', 2.0),
                z_exit=params.get('z_exit', 0.5),
                lookback=params.get('lookback', 60)
            )
            
            result = strategy.backtest(df_a, df_b, initial_capital)
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol_a}-{symbol_b}: {e}")
            result = {'status': 'ERROR', 'error': str(e)}
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        backtest_result = BacktestResult(
            strategy_name='Pairs Trading',
            symbol=f"{symbol_a}-{symbol_b}",
            params=params,
            result_dict=result,
            execution_time_ms=execution_time
        )
        
        self.results_history.append(backtest_result)
        return backtest_result
    
    def backtest_momentum(self, symbol: str, df: pd.DataFrame, params: Dict[str, Any],
                         initial_capital: float = 100000) -> BacktestResult:
        """Backtest momentum strategy"""
        import time
        from core.strategies import MomentumStrategy
        
        start_time = time.time()
        
        try:
            strategy = MomentumStrategy(
                symbol=symbol,
                rsi_period=params.get('rsi_period', 14),
                rsi_overbought=params.get('rsi_overbought', 70),
                rsi_oversold=params.get('rsi_oversold', 30),
                volume_sma_period=params.get('volume_sma_period', 20),
                hold_days=params.get('hold_days', 5)
            )
            
            result = strategy.backtest(df, initial_capital)
            
        except Exception as e:
            logger.error(f"Error backtesting momentum {symbol}: {e}")
            result = {'status': 'ERROR', 'error': str(e)}
        
        execution_time = (time.time() - start_time) * 1000
        
        backtest_result = BacktestResult(
            strategy_name='Momentum',
            symbol=symbol,
            params=params,
            result_dict=result,
            execution_time_ms=execution_time
        )
        
        self.results_history.append(backtest_result)
        return backtest_result
    
    def backtest_mean_reversion(self, symbol: str, df: pd.DataFrame, params: Dict[str, Any],
                               initial_capital: float = 100000) -> BacktestResult:
        """Backtest mean reversion strategy"""
        import time
        from core.strategies import MeanReversionStrategy
        
        start_time = time.time()
        
        try:
            strategy = MeanReversionStrategy(
                symbol=symbol,
                ma_period=params.get('ma_period', 20),
                std_dev=params.get('std_dev', 2.0)
            )
            
            result = strategy.backtest(df, initial_capital)
            
        except Exception as e:
            logger.error(f"Error backtesting mean reversion {symbol}: {e}")
            result = {'status': 'ERROR', 'error': str(e)}
        
        execution_time = (time.time() - start_time) * 1000
        
        backtest_result = BacktestResult(
            strategy_name='Mean Reversion',
            symbol=symbol,
            params=params,
            result_dict=result,
            execution_time_ms=execution_time
        )
        
        self.results_history.append(backtest_result)
        return backtest_result
    
    def backtest_ma_crossover(self, symbol: str, df: pd.DataFrame, params: Dict[str, Any],
                             initial_capital: float = 100000) -> BacktestResult:
        """Backtest MA crossover strategy"""
        import time
        from core.strategies import MovingAverageCrossover
        
        start_time = time.time()
        
        try:
            strategy = MovingAverageCrossover(
                symbol=symbol,
                fast_period=params.get('fast_period', 20),
                slow_period=params.get('slow_period', 50)
            )
            
            result = strategy.backtest(df, initial_capital)
            
        except Exception as e:
            logger.error(f"Error backtesting MA crossover {symbol}: {e}")
            result = {'status': 'ERROR', 'error': str(e)}
        
        execution_time = (time.time() - start_time) * 1000
        
        backtest_result = BacktestResult(
            strategy_name='MA Crossover',
            symbol=symbol,
            params=params,
            result_dict=result,
            execution_time_ms=execution_time
        )
        
        self.results_history.append(backtest_result)
        return backtest_result
    
    def backtest_batch(self, strategy_name: str, symbols: list, df_loader: Callable,
                      params: Dict[str, Any], initial_capital: float = 100000) -> pd.DataFrame:
        """Backtest a strategy across multiple symbols"""
        results = []
        
        for symbol in symbols:
            try:
                df = df_loader(symbol)
                
                if df is None or len(df) < 100:
                    continue
                
                # Call appropriate backtest method based on strategy
                if strategy_name == 'momentum':
                    result = self.backtest_momentum(symbol, df, params, initial_capital)
                elif strategy_name == 'mean_reversion':
                    result = self.backtest_mean_reversion(symbol, df, params, initial_capital)
                elif strategy_name == 'ma_crossover':
                    result = self.backtest_ma_crossover(symbol, df, params, initial_capital)
                else:
                    continue
                
                results.append(result.to_dict())
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Convert to DataFrame and sort by Sharpe
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('sharpe', ascending=False)
            return df_results
        else:
            return pd.DataFrame()
    
    def get_results_history(self) -> list:
        """Get all backtest results"""
        return self.results_history
    
    def clear_history(self):
        """Clear results history"""
        self.results_history = []
    
    def get_last_result(self) -> Optional[BacktestResult]:
        """Get last backtest result"""
        if self.results_history:
            return self.results_history[-1]
        return None
    
    def get_results_statistics(self) -> Dict[str, Any]:
        """Get statistics from all results"""
        if not self.results_history:
            return {}
        
        sharpes = [r.get_sharpe() for r in self.results_history]
        pnl_pcts = [r.get_pnl_pct() for r in self.results_history]
        win_rates = [r.get_win_rate() for r in self.results_history]
        
        return {
            'total_backtests': len(self.results_history),
            'avg_sharpe': np.mean(sharpes) if sharpes else 0,
            'max_sharpe': max(sharpes) if sharpes else 0,
            'min_sharpe': min(sharpes) if sharpes else 0,
            'avg_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0,
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'profitable': sum(1 for pnl in pnl_pcts if pnl > 0),
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_backtester = None

def get_unified_backtester() -> UnifiedBacktester:
    """Get singleton instance of unified backtester"""
    global _backtester
    if _backtester is None:
        _backtester = UnifiedBacktester()
    return _backtester
