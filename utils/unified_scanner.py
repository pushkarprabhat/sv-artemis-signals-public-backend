"""
Unified Scanner - All 14 Strategies Integrated
Brings together all futures, options, and technical strategies with unified interface

Strategies:
1. Pair Trading (Delta-Neutral)
2. Index Pairs (NIFTY+BANKNIFTY correlation)
3. Cointegration (Statistical Arbitrage)
4. Strangle (Options - high IV)
5. Straddle (Options - low IV)
6. Iron Condor (Options - income)
7. Butterfly (Options - defined risk)
8. Calendar Spread (Options - theta decay)
9. Momentum (RSI + ADX)
10. Ichimoku Plus (Cloud breakout)
11. Mean Reversion (Bollinger Bands)
12. NIFTY Derivatives (Index futures)
13. BANKNIFTY Derivatives (Bank index futures)
14. Stock Derivatives (Individual stock futures)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedScanner:
    """Unified scanner for all 14 strategies"""
    
    STRATEGIES = {
        'pairs': {
            'name': 'Pair Trading (Delta-Neutral)',
            'module': 'core.pairs',
            'function': 'scan_pairs_nifty500',
            'description': 'Mean reversion pairs trading with Z-score'
        },
        'index_pairs': {
            'name': 'Index Pairs (NIFTY+BANKNIFTY)',
            'module': 'core.index_constituent_pairs',
            'function': 'scan_index_constituent_pairs',
            'description': 'Correlation-based index pairs'
        },
        'cointegration': {
            'name': 'Cointegration (Statistical Arbitrage)',
            'module': 'core.index_spread_cointegration',
            'function': 'scan_index_cointegration_signals',
            'description': 'Cointegration-based spread trading'
        },
        'strangle': {
            'name': 'Strangle (Options - High IV)',
            'module': 'core.strangle',
            'function': 'scan_strangle',
            'description': 'Short strangle when IV is high'
        },
        'straddle': {
            'name': 'Straddle (Options - Low IV)',
            'module': 'core.straddle',
            'function': 'scan_straddle',
            'description': 'Long straddle when IV is low'
        },
        'iron_condor': {
            'name': 'Iron Condor (Options - Income)',
            'module': 'core.derivatives_multi_leg',
            'function': 'scan_iron_condor',
            'description': 'Sell iron condor for income generation'
        },
        'butterfly': {
            'name': 'Butterfly (Options - Defined Risk)',
            'module': 'core.derivatives_multi_leg',
            'function': 'scan_butterfly',
            'description': 'Butterfly spread with defined risk'
        },
        'calendar': {
            'name': 'Calendar Spread (Options - Theta)',
            'module': 'core.derivatives_multi_leg',
            'function': 'scan_calendar',
            'description': 'Calendar spread to capture theta decay'
        },
        'momentum': {
            'name': 'Momentum (RSI + ADX)',
            'module': 'core.momentum',
            'function': 'scan_momentum_nifty500',
            'description': 'Trend following with RSI and volume'
        },
        'ichimoku': {
            'name': 'Ichimoku Plus (Cloud Breakout)',
            'module': 'core.ichimoku_plus',
            'function': 'scan_ichimoku_plus',
            'description': 'Cloud-based breakout strategy'
        },
        'mean_reversion': {
            'name': 'Mean Reversion (Bollinger Bands)',
            'module': 'core.mean_reversion',
            'function': 'scan_mean_reversion_nifty500',
            'description': 'Range-bound trading with Bollinger Bands'
        },
        'nifty_deriv': {
            'name': 'NIFTY Derivatives (Index Futures)',
            'module': 'core.nifty_derivatives',
            'function': 'scan_nifty_derivatives',
            'description': 'NIFTY futures with multi-leg strategies'
        },
        'banknifty_deriv': {
            'name': 'BANKNIFTY Derivatives (Bank Index)',
            'module': 'core.banknifty_derivatives',
            'function': 'scan_banknifty_derivatives',
            'description': 'BANKNIFTY futures and spreads'
        },
        'stock_deriv': {
            'name': 'Stock Derivatives (Individual Stocks)',
            'module': 'core.stock_derivatives',
            'function': 'scan_stock_derivatives',
            'description': 'Stock futures with hedging and spreads'
        },
    }
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize Unified Scanner
        
        Args:
            max_workers: Number of parallel execution threads
        """
        self.max_workers = max_workers
        self.loaded_functions = {}
        self._load_all_strategies()
    
    def _load_all_strategies(self):
        """Load all strategy functions dynamically"""
        for key, config in self.STRATEGIES.items():
            try:
                module_name = config['module']
                function_name = config['function']
                
                # Dynamic import
                module = __import__(module_name, fromlist=[function_name])
                func = getattr(module, function_name)
                
                self.loaded_functions[key] = func
                logger.info(f"✅ Loaded {config['name']}")
            
            except ImportError as e:
                logger.warning(f"⚠️  Could not load {config['name']}: {e}")
            except AttributeError as e:
                logger.warning(f"⚠️  Function not found in {config['module']}: {e}")
    
    def get_loaded_strategies(self) -> Dict[str, str]:
        """Get list of successfully loaded strategies"""
        return {
            k: self.STRATEGIES[k]['name'] 
            for k in self.loaded_functions.keys()
        }
    
    def run_single_strategy(self, strategy_key: str, **kwargs) -> Dict:
        """
        Run a single strategy
        
        Args:
            strategy_key: Strategy identifier (from STRATEGIES)
            **kwargs: Arguments to pass to strategy function
            
        Returns:
            Results dictionary with signals
        """
        if strategy_key not in self.loaded_functions:
            return {
                'strategy': strategy_key,
                'status': 'NOT_LOADED',
                'error': f'Strategy {strategy_key} not available'
            }
        
        try:
            func = self.loaded_functions[strategy_key]
            config = self.STRATEGIES[strategy_key]
            
            logger.info(f"Running {config['name']}...")
            result = func(**kwargs)
            
            # Normalize result
            if result is None:
                result = {'status': 'NO_SIGNALS'}
            elif isinstance(result, dict):
                result['strategy'] = config['name']
                result['strategy_key'] = strategy_key
            elif isinstance(result, pd.DataFrame):
                result = {
                    'strategy': config['name'],
                    'strategy_key': strategy_key,
                    'signals': result.to_dict('records') if not result.empty else [],
                    'count': len(result)
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in {strategy_key}: {e}")
            return {
                'strategy': self.STRATEGIES[strategy_key]['name'],
                'strategy_key': strategy_key,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def run_all_strategies_parallel(self, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Run all loaded strategies in parallel
        
        Args:
            **kwargs: Common arguments to pass to all strategies
            
        Returns:
            Tuple of (combined_results_df, execution_summary)
        """
        results = []
        execution_summary = {
            'total_loaded': len(self.loaded_functions),
            'timestamp': pd.Timestamp.now(),
            'strategies_executed': 0,
            'strategies_failed': 0,
            'errors': {}
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for strategy_key in self.loaded_functions.keys():
                future = executor.submit(self.run_single_strategy, strategy_key, **kwargs)
                futures[future] = strategy_key
            
            for future in as_completed(futures):
                strategy_key = futures[future]
                try:
                    result = future.result()
                    
                    if isinstance(result, dict):
                        if result.get('status') == 'ERROR':
                            execution_summary['strategies_failed'] += 1
                            execution_summary['errors'][strategy_key] = result.get('error')
                        else:
                            execution_summary['strategies_executed'] += 1
                            results.append(result)
                    
                except Exception as e:
                    execution_summary['strategies_failed'] += 1
                    execution_summary['errors'][strategy_key] = str(e)
        
        # Aggregate results
        combined_df = self._aggregate_results(results)
        
        return combined_df, execution_summary
    
    def _aggregate_results(self, results: List) -> pd.DataFrame:
        """Aggregate results from multiple strategies"""
        if not results:
            return pd.DataFrame()
        
        rows = []
        for result in results:
            if isinstance(result, dict):
                if 'signals' in result:
                    for signal in result['signals']:
                        signal['strategy'] = result.get('strategy', 'Unknown')
                        signal['strategy_key'] = result.get('strategy_key', 'Unknown')
                        rows.append(signal)
                else:
                    rows.append(result)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Sort by confidence if available
        if 'confidence' in df.columns:
            df = df.sort_values('confidence', ascending=False)
        
        return df


def run_all_strategies(**kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Quick function to run all 14 strategies
    
    Args:
        **kwargs: Arguments to pass to strategies
        
    Returns:
        Tuple of (signals_dataframe, execution_summary)
    """
    scanner = UnifiedScanner()
    return scanner.run_all_strategies_parallel(**kwargs)


def list_all_strategies() -> Dict[str, str]:
    """List all 14 available strategies"""
    return UnifiedScanner.STRATEGIES


if __name__ == "__main__":
    import json
    
    print("\n" + "="*60)
    print("🎯 UNIFIED SCANNER - 14 Strategies")
    print("="*60)
    
    scanner = UnifiedScanner(max_workers=8)
    
    print("\n📊 Available Strategies:")
    print("-" * 60)
    for i, (key, config) in enumerate(scanner.STRATEGIES.items(), 1):
        print(f"{i:2d}. {config['name']:<40} ({key})")
        print(f"    └─ {config['description']}")
    
    print(f"\n✅ Loaded: {len(scanner.loaded_functions)}/{len(scanner.STRATEGIES)}")
    
    print("\n" + "="*60)
    print("Running all strategies in parallel...")
    print("="*60)
    
    results, summary = scanner.run_all_strategies_parallel()
    
    print(f"\n📈 Execution Summary:")
    print(f"   Total Loaded: {summary['total_loaded']}")
    print(f"   Executed: {summary['strategies_executed']}")
    print(f"   Failed: {summary['strategies_failed']}")
    
    if summary['errors']:
        print(f"\n⚠️  Errors:")
        for strategy, error in summary['errors'].items():
            print(f"   - {strategy}: {error}")
    
    if not results.empty:
        print(f"\n🎁 Signals Generated: {len(results)}")
        print("\nTop 10 Signals:")
        print(results.head(10).to_string())
    else:
        print("\n⚠️  No signals generated")
