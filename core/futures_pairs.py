# core/futures_pairs.py — Futures Pair Trading Analyzer
# Finds: Correlated futures pairs for spread trading
# Strategies: Statistical arbitrage, pairs trading, mean reversion
# Features: Correlation analysis, spread signals, backtesting support

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import pearsonr, spearmanr
from config import BASE_DIR
from utils.logger import logger


class FuturesPairsAnalyzer:
    """Analyzes futures pairs for trading opportunities"""
    
    def __init__(self):
        self.base_dir = BASE_DIR / 'data' / 'futures'
        self.pairs_cache_dir = BASE_DIR / 'data' / 'futures_pairs'
        self.pairs_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_correlation(self, symbol1, symbol2, lookback_days=252, category='index'):
        """Calculate correlation between two futures symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            lookback_days: Period for correlation calculation
            category: 'index', 'stocks', 'currency', 'commodities'
        
        Returns:
            dict: {pearson_corr, spearman_corr, lookback_days}
        """
        # Load data for both symbols
        file1 = self.base_dir / category / symbol1 / 'day.parquet'
        file2 = self.base_dir / category / symbol2 / 'day.parquet'
        
        if not file1.exists() or not file2.exists():
            logger.warning(f"[PAIRS] Data not found for {symbol1}/{symbol2}")
            return {'error': 'missing_data', 'symbol1': symbol1, 'symbol2': symbol2}
        
        try:
            df1 = pd.read_parquet(file1)
            df2 = pd.read_parquet(file2)
            
            # Ensure data is sorted by date
            df1 = df1.sort_values('date')
            df2 = df2.sort_values('date')
            
            # Merge on date
            merged = df1[['date', 'close']].merge(
                df2[['date', 'close']],
                on='date',
                suffixes=('_1', '_2')
            )
            
            # Filter to lookback period
            cutoff_date = dt.datetime.now().date() - dt.timedelta(days=lookback_days)
            merged = merged[merged['date'] >= pd.Timestamp(cutoff_date)]
            
            if len(merged) < 30:
                logger.warning(f"[PAIRS] Insufficient data for {symbol1}/{symbol2}")
                return {'error': 'insufficient_data'}
            
            # Calculate returns
            returns1 = np.log(merged['close_1'] / merged['close_1'].shift(1)).dropna()
            returns2 = np.log(merged['close_2'] / merged['close_2'].shift(1)).dropna()
            
            if len(returns1) < 30 or len(returns2) < 30:
                return {'error': 'insufficient_returns'}
            
            # Calculate correlations
            pearson_corr, pearson_pval = pearsonr(returns1, returns2)
            spearman_corr, spearman_pval = spearmanr(returns1, returns2)
            
            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'pearson_correlation': pearson_corr,
                'pearson_pvalue': pearson_pval,
                'spearman_correlation': spearman_corr,
                'spearman_pvalue': spearman_pval,
                'lookback_days': lookback_days,
                'data_points': len(returns1),
                'correlation_strength': 'very_strong' if abs(pearson_corr) > 0.9 else
                                       'strong' if abs(pearson_corr) > 0.7 else
                                       'moderate' if abs(pearson_corr) > 0.5 else
                                       'weak'
            }
        
        except Exception as e:
            logger.error(f"[PAIRS] Error calculating correlation: {e}")
            return {'error': str(e)}
    
    def find_correlated_pairs(self, symbols, min_correlation=0.7, category='index',
                             lookback_days=252):
        """Find all correlated pairs from a list of symbols
        
        Args:
            symbols: List of symbols to analyze
            min_correlation: Minimum correlation threshold
            category: Data category
            lookback_days: Lookback period
        
        Returns:
            DataFrame: All pairs above threshold [symbol1, symbol2, correlation, strength]
        """
        logger.info(f"[PAIRS] Finding correlated pairs from {len(symbols)} symbols (min_corr={min_correlation})")
        
        pairs = []
        
        # Use parallel execution for correlation calculations
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures_map = {}
            
            # Submit all pair calculations
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    future = executor.submit(
                        self.calculate_correlation, sym1, sym2,
                        lookback_days, category
                    )
                    futures_map[future] = (sym1, sym2)
            
            # Collect results
            for future in as_completed(futures_map):
                sym1, sym2 = futures_map[future]
                try:
                    result = future.result()
                    
                    if 'error' not in result:
                        corr = abs(result['pearson_correlation'])
                        if corr >= min_correlation:
                            pairs.append({
                                'symbol1': result['symbol1'],
                                'symbol2': result['symbol2'],
                                'correlation': result['pearson_correlation'],
                                'abs_correlation': corr,
                                'strength': result['correlation_strength'],
                                'pvalue': result['pearson_pvalue'],
                                'lookback_days': lookback_days
                            })
                
                except Exception as e:
                    logger.warning(f"[PAIRS] Error for {sym1}/{sym2}: {e}")
        
        if pairs:
            df_pairs = pd.DataFrame(pairs)
            df_pairs = df_pairs.sort_values('abs_correlation', ascending=False)
            logger.info(f"[PAIRS] Found {len(df_pairs)} correlated pairs")
            return df_pairs
        
        logger.warning(f"[PAIRS] No correlated pairs found (min_correlation={min_correlation})")
        return pd.DataFrame()
    
    def calculate_spread(self, symbol1, symbol2, category='index', interval='day'):
        """Calculate spread (ratio) between two futures
        
        Args:
            symbol1: Primary symbol
            symbol2: Secondary symbol
            category: Data category
            interval: Time interval ('day', '5minute', etc.)
        
        Returns:
            DataFrame: [date, price1, price2, spread, spread_zscore]
        """
        file1 = self.base_dir / category / symbol1 / f"{interval}.parquet"
        file2 = self.base_dir / category / symbol2 / f"{interval}.parquet"
        
        if not file1.exists() or not file2.exists():
            logger.warning(f"[PAIRS] Data not found for {symbol1}/{symbol2}/{interval}")
            return None
        
        try:
            df1 = pd.read_parquet(file1)[['date', 'close']].rename(columns={'close': 'price1'})
            df2 = pd.read_parquet(file2)[['date', 'close']].rename(columns={'close': 'price2'})
            
            # Merge
            df_spread = df1.merge(df2, on='date')
            
            # Calculate spread (ratio)
            df_spread['spread'] = df_spread['price1'] / df_spread['price2']
            
            # Calculate z-score for spread (mean reversion signal)
            df_spread['spread_mean'] = df_spread['spread'].rolling(window=20).mean()
            df_spread['spread_std'] = df_spread['spread'].rolling(window=20).std()
            df_spread['spread_zscore'] = (df_spread['spread'] - df_spread['spread_mean']) / df_spread['spread_std']
            
            return df_spread
        
        except Exception as e:
            logger.error(f"[PAIRS] Error calculating spread: {e}")
            return None
    
    def generate_pair_signals(self, symbol1, symbol2, category='index',
                             z_score_threshold=2.0, interval='day'):
        """Generate trading signals from spread analysis
        
        Args:
            symbol1: Primary symbol
            symbol2: Secondary symbol
            category: Data category
            z_score_threshold: Z-score threshold for signal generation
            interval: Time interval
        
        Returns:
            DataFrame: Signals [date, spread, zscore, signal, confidence]
        """
        df_spread = self.calculate_spread(symbol1, symbol2, category, interval)
        
        if df_spread is None:
            return None
        
        df_spread = df_spread.copy()
        
        # Generate signals based on z-score
        df_spread['signal'] = 'HOLD'
        df_spread['confidence'] = 0
        
        # Buy signal: spread too low (will revert upward)
        buy_mask = df_spread['spread_zscore'] < -z_score_threshold
        df_spread.loc[buy_mask, 'signal'] = 'BUY'
        df_spread.loc[buy_mask, 'confidence'] = abs(df_spread.loc[buy_mask, 'spread_zscore']) / z_score_threshold * 100
        
        # Sell signal: spread too high (will revert downward)
        sell_mask = df_spread['spread_zscore'] > z_score_threshold
        df_spread.loc[sell_mask, 'signal'] = 'SELL'
        df_spread.loc[sell_mask, 'confidence'] = abs(df_spread.loc[sell_mask, 'spread_zscore']) / z_score_threshold * 100
        
        # Cap confidence at 100
        df_spread['confidence'] = df_spread['confidence'].clip(upper=100)
        
        return df_spread[['date', 'price1', 'price2', 'spread', 'spread_zscore', 'signal', 'confidence']]
    
    def save_pair_analysis(self, symbol1, symbol2, analysis_df):
        """Save pair analysis results
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            analysis_df: Analysis DataFrame
        
        Returns:
            Path: Where analysis was saved
        """
        pair_name = f"{symbol1}_{symbol2}"
        file_path = self.pairs_cache_dir / f"{pair_name}.parquet"
        
        analysis_df.to_parquet(file_path, index=False)
        logger.info(f"[PAIRS] Saved analysis for {pair_name}")
        
        return file_path
    
    def backtest_pair_strategy(self, symbol1, symbol2, category='index',
                              z_score_threshold=2.0, transaction_cost=0.001):
        """Backtest a simple mean reversion pair trading strategy
        
        Args:
            symbol1: Primary symbol
            symbol2: Secondary symbol
            category: Data category
            z_score_threshold: Entry signal threshold
            transaction_cost: Fraction of trade value (e.g., 0.001 = 0.1%)
        
        Returns:
            dict: Backtest results [total_return, win_rate, sharpe, drawdown]
        """
        signals_df = self.generate_pair_signals(symbol1, symbol2, category, z_score_threshold)
        
        if signals_df is None or signals_df.empty:
            return {'error': 'no_signals'}
        
        signals_df = signals_df.copy()
        signals_df = signals_df.dropna()
        
        # Initialize position tracking
        position = 0  # 0=flat, 1=long pair, -1=short pair
        entry_price = 0
        pnl_list = []
        
        for idx, row in signals_df.iterrows():
            signal = row['signal']
            spread = row['spread']
            
            # Entry logic
            if position == 0:
                if signal == 'BUY':
                    position = 1
                    entry_price = spread
                elif signal == 'SELL':
                    position = -1
                    entry_price = spread
            
            # Exit logic (close at mean or opposite signal)
            elif position != 0:
                if signal == 'HOLD':
                    # Continue holding
                    pnl = (spread - entry_price) * position
                    pnl_list.append(pnl)
                else:
                    # Close position
                    pnl = (spread - entry_price) * position - transaction_cost * spread
                    pnl_list.append(pnl)
                    position = 0
        
        # Calculate metrics
        if len(pnl_list) == 0:
            return {'error': 'no_trades'}
        
        pnl_array = np.array(pnl_list)
        total_pnl = pnl_array.sum()
        winning_trades = (pnl_array > 0).sum()
        total_trades = len(pnl_array)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Sharpe ratio (assuming daily returns)
        daily_returns = np.diff(np.cumsum(pnl_array)) / np.mean(np.abs(pnl_array))
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        
        # Drawdown
        cumulative_pnl = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-6)
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_pnl': float(total_pnl),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_win': float(pnl_array[pnl_array > 0].mean()) if (pnl_array > 0).any() else 0,
            'avg_loss': float(pnl_array[pnl_array < 0].mean()) if (pnl_array < 0).any() else 0,
            'sharpe_ratio': float(sharpe),
            'max_drawdown_pct': float(max_drawdown),
            'symbol1': symbol1,
            'symbol2': symbol2,
            'z_score_threshold': z_score_threshold
        }
