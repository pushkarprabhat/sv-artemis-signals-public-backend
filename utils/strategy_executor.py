"""
Parallel Strategy Execution Engine
Runs all strategies in parallel using ThreadPoolExecutor for maximum performance

Author: Trading System
Date: 2025-12-23
"""

import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class ParallelStrategyExecutor:
    """Executes all trading strategies in parallel"""
    
    def __init__(self, max_workers: int = 8, timeout: int = 300):
        """
        Initialize executor
        
        Args:
            max_workers: Number of parallel threads (default 8, recommended 8+)
            timeout: Timeout per strategy in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.execution_times = {}
        self.strategy_results = {}
    
    def _run_strategy(self, strategy_name: str, strategy_func, args: Tuple = (),
                     kwargs: Dict = None) -> Tuple[str, pd.DataFrame, float, Optional[str]]:
        """
        Run a single strategy
        
        Args:
            strategy_name: Name of strategy
            strategy_func: Callable strategy function
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            Tuple of (strategy_name, results_df, execution_time, error_message)
        """
        if kwargs is None:
            kwargs = {}
        
        start_time = datetime.now()
        try:
            # Execute strategy
            results = strategy_func(*args, **kwargs)
            
            # Ensure results is a DataFrame
            if results is None or (isinstance(results, pd.DataFrame) and results.empty):
                results = pd.DataFrame()
            elif not isinstance(results, pd.DataFrame):
                results = pd.DataFrame([results]) if isinstance(results, dict) else pd.DataFrame()
            
            # Add strategy name to results
            if not results.empty:
                results['strategy_name'] = strategy_name
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ {strategy_name}: {len(results)} signals in {execution_time:.2f}s")
            
            return strategy_name, results, execution_time, None
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ {strategy_name}: Error in {execution_time:.2f}s - {error_msg}")
            return strategy_name, pd.DataFrame(), execution_time, error_msg
    
    def run_all_strategies(self, strategies: Dict, parallel: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Run all strategies in parallel
        
        Args:
            strategies: Dict of {strategy_name: (function, args, kwargs)}
                       Example:
                       {
                           'Pair Trading': (scan_pair_trading, (), {'tf': 'day'}),
                           'Ichimoku Plus': (scan_ichimoku_plus, (), {'tf': 'day'}),
                       }
            parallel: If True, run in parallel; if False, run sequentially
        
        Returns:
            Tuple of (combined_results_df, execution_summary)
        """
        all_results = []
        execution_summary = {}
        total_start = datetime.now()
        
        if not parallel or self.max_workers == 1:
            # Sequential execution
            logger.info(f"Running {len(strategies)} strategies SEQUENTIALLY...")
            for name, (func, args, kwargs) in strategies.items():
                result = self._run_strategy(name, func, args, kwargs)
                strategy_name, df, exec_time, error = result
                all_results.append(df)
                execution_summary[strategy_name] = {
                    'status': 'SUCCESS' if error is None else 'FAILED',
                    'execution_time': exec_time,
                    'signals_count': len(df),
                    'error': error
                }
        else:
            # Parallel execution
            logger.info(f"Running {len(strategies)} strategies in PARALLEL ({self.max_workers} threads)...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_name = {}
                for strategy_name, (strategy_func, args, kwargs) in strategies.items():
                    future = executor.submit(
                        self._run_strategy,
                        strategy_name,
                        strategy_func,
                        args,
                        kwargs
                    )
                    future_to_name[future] = strategy_name
                
                # Collect results as they complete
                for future in as_completed(future_to_name, timeout=self.timeout):
                    strategy_name, df, exec_time, error = future.result()
                    all_results.append(df)
                    execution_summary[strategy_name] = {
                        'status': 'SUCCESS' if error is None else 'FAILED',
                        'execution_time': exec_time,
                        'signals_count': len(df),
                        'error': error
                    }
        
        # Combine all results
        total_time = (datetime.now() - total_start).total_seconds()
        
        if all_results and any(len(df) > 0 for df in all_results):
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        # Add summary
        execution_summary['TOTAL'] = {
            'total_strategies': len(strategies),
            'successful': sum(1 for s in execution_summary.values() if s['status'] == 'SUCCESS'),
            'failed': sum(1 for s in execution_summary.values() if s['status'] == 'FAILED'),
            'total_signals': len(combined_df),
            'total_execution_time': total_time,
            'parallel_mode': parallel,
            'workers': self.max_workers
        }
        
        return combined_df, execution_summary
    
    def print_execution_summary(self, summary: Dict):
        """Print execution summary in readable format"""
        print("\n" + "="*70)
        print("PARALLEL EXECUTION SUMMARY")
        print("="*70)
        
        total_info = summary.pop('TOTAL', {})
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Strategies: {total_info.get('total_strategies', 0)}")
        print(f"   ✅ Successful: {total_info.get('successful', 0)}")
        print(f"   ❌ Failed: {total_info.get('failed', 0)}")
        print(f"   📈 Total Signals Generated: {total_info.get('total_signals', 0)}")
        print(f"   ⏱️  Total Execution Time: {total_info.get('total_execution_time', 0):.2f}s")
        print(f"   🔄 Mode: {'PARALLEL' if total_info.get('parallel_mode') else 'SEQUENTIAL'} ({total_info.get('workers', 1)} threads)")
        
        print(f"\n📋 Strategy Details:")
        print(f"{'Strategy':<30} {'Status':<12} {'Signals':<10} {'Time':<10}")
        print("-"*62)
        
        for name, info in summary.items():
            status_emoji = "✅" if info['status'] == 'SUCCESS' else "❌"
            print(f"{name:<30} {status_emoji} {info['status']:<10} {info['signals_count']:<10} {info['execution_time']:<10.2f}s")
            if info['error']:
                print(f"  Error: {info['error'].split(chr(10))[0]}")
        
        print("\n" + "="*70)


def create_strategy_dict(timeframe: str = 'day') -> Dict:
    """
    Create dictionary of all strategies with their execution parameters
    
    Args:
        timeframe: 'day', '60min', '30min', etc.
    
    Returns:
        Dictionary suitable for run_all_strategies()
    """
    # Import all strategy functions
    from core.pairs import scan_pair_trading
    from core.strangle import scan_strangle
    from core.straddle import scan_straddle
    from core.volatility import scan_volatility_nifty500
    from core.momentum import scan_momentum_nifty500
    from core.mean_reversion import scan_mean_reversion_nifty500
    from core.index_constituent_pairs import scan_all_indices
    from core.index_spread_cointegration import scan_index_cointegration_signals
    from core.ichimoku_plus import scan_ichimoku_plus
    from core.derivatives_multi_leg import scan_multi_leg_derivatives
    
    strategies = {
        # Original Strategies
        'Pair Trading': (scan_pair_trading, (), {'tf': timeframe}),
        'Strangle (IV Crush)': (scan_strangle, (), {}),
        'Straddle (IV Expansion)': (scan_straddle, (), {}),
        'Volatility Trading (GARCH)': (scan_volatility_nifty500, (), {'tf': timeframe}),
        'Momentum (SMA)': (scan_momentum_nifty500, (), {'tf': timeframe}),
        'Mean Reversion (Z-Score)': (scan_mean_reversion_nifty500, (), {'tf': timeframe}),
        
        # Index Strategies
        'Index Constituent Pairs': (scan_all_indices, (), {'tf': timeframe}),
        'Index Cointegration (NIFTY-BANKNIFTY)': (scan_index_cointegration_signals, (), {'tf': timeframe}),
        
        # New Strategies
        'Ichimoku Plus': (scan_ichimoku_plus, (), {'tf': timeframe}),
        'Multi-Leg Derivatives': (scan_multi_leg_derivatives, (), {
            'atm_price': 50000,
            'current_iv_percentile': 50,
            'days_to_expiry': 30
        }),
    }
    
    return strategies


# ==================== STANDALONE USAGE ====================
def run_all_strategies_parallel(timeframe: str = 'day', max_workers: int = 8) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run all strategies in parallel
    
    Args:
        timeframe: 'day', '60min', '30min', etc.
        max_workers: Number of parallel threads
    
    Returns:
        Tuple of (combined_results, execution_summary)
    """
    executor = ParallelStrategyExecutor(max_workers=max_workers)
    strategies = create_strategy_dict(timeframe)
    results, summary = executor.run_all_strategies(strategies, parallel=True)
    executor.print_execution_summary(summary)
    return results, summary


if __name__ == "__main__":
    # Example usage
    print("Testing Parallel Strategy Executor...")
    
    # Run with 8 threads
    results, summary = run_all_strategies_parallel(timeframe='day', max_workers=8)
    
    if not results.empty:
        print(f"\nTop 10 Signals Generated:")
        print(results[['symbol', 'strategy_name', 'signal_type', 'price']].head(10).to_string())
