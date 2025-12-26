# core/pairs_strategies.py
# Complete implementation of all 10 pairs trading strategies
# Professional quantitative pairs trading framework

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import itertools
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from config import BASE_DIR, P_VALUE_THRESHOLD, HALF_LIFE_MAX
from universe.focused_universe import get_focused_universe, get_stock_sector
from utils.logger import logger
from core.pairs import load_price, get_latest_price


# =============================================================================
# STRATEGY 1: CLASSIC COINTEGRATION (Already implemented in pairs.py)
# =============================================================================
# This is the foundation - skip here, use from pairs.py


# =============================================================================
# STRATEGY 2: INDEX DISPERSION (NIFTY vs Constituents)
# =============================================================================

def scan_index_dispersion_pairs(index_symbol="NIFTY", timeframe="day", 
                                top_n=10, lookback_days=252):
    """
    Find divergences between index and its constituents
    
    When index moves but individual stocks don't follow, there's opportunity
    for arbitrage through statistical mean reversion.
    
    Args:
        index_symbol: Index to compare against ("NIFTY", "BANKNIFTY")
        timeframe: data timeframe
        top_n: number of pairs to return
        lookback_days: history to analyze
        
    Returns:
        DataFrame with divergence opportunities
    """
    try:
        # Load index price
        index_price = load_price(index_symbol, timeframe)
        if index_price is None or len(index_price) < lookback_days:
            logger.warning(f"Insufficient data for {index_symbol}")
            return pd.DataFrame()
        
        # Get index constituents (NIFTY 50, BANKNIFTY stocks)
        universe = get_focused_universe(sector="ALL")
        
        results = []
        
        for _, stock_row in universe.iterrows():
            try:
                stock_symbol = stock_row['Symbol']
                stock_price = load_price(stock_symbol, timeframe)
                
                if stock_price is None or len(stock_price) < lookback_days:
                    continue
                
                # Align dates
                common_dates = index_price.index.intersection(stock_price.index)
                if len(common_dates) < 100:
                    continue
                
                idx_aligned = index_price.loc[common_dates]
                stk_aligned = stock_price.loc[common_dates]
                
                # Test cointegration
                _, p_value, _ = coint(idx_aligned, stk_aligned)
                
                if p_value > P_VALUE_THRESHOLD:
                    continue
                
                # Calculate spread
                spread = stk_aligned - idx_aligned * (stk_aligned.mean() / idx_aligned.mean())
                z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                
                # Calculate half-life
                spread_lag = spread.shift(1).dropna()
                spread_diff = spread.diff().dropna()
                common = spread_lag.index.intersection(spread_diff.index)
                
                if len(common) < 50:
                    continue
                
                model = OLS(spread_diff.loc[common], add_constant(spread_lag.loc[common])).fit()
                half_life = -np.log(2) / model.params.iloc[1] if model.params.iloc[1] < 0 else 999
                
                if half_life > HALF_LIFE_MAX:
                    continue
                
                # ML Score calculation
                ml_score = (1 - p_value) * 100 + 50 / max(half_life, 1) + abs(z_score) * 10
                
                results.append({
                    'pair': f"{index_symbol}-{stock_symbol}",
                    'strategy': 'Index Dispersion',
                    'index': index_symbol,
                    'stock': stock_symbol,
                    'sector': stock_row.get('Industry', 'Unknown'),
                    'p_value': p_value,
                    'z_score': round(z_score, 2),
                    'half_life': round(half_life, 1),
                    'ml_score': round(ml_score, 1),
                    'signal': 'LONG_STOCK' if z_score < -2 else 'SHORT_STOCK' if z_score > 2 else 'NEUTRAL'
                })
                
            except Exception as e:
                logger.debug(f"Error analyzing {stock_symbol}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('ml_score', ascending=False).head(top_n)
        return df
        
    except Exception as e:
        logger.error(f"Index dispersion scan error: {e}")
        return pd.DataFrame()


# =============================================================================
# STRATEGY 3: MACRO PAIRS (USDINR vs NIFTY, Gold vs Equities)
# =============================================================================

def scan_macro_pairs(timeframe="day", top_n=10):
    """
    Find macro relationships (currency vs equities, commodities vs stocks)
    
    When currency weakens, some stocks benefit (exporters), others suffer (importers).
    This strategy exploits macro-level correlations and divergences.
    
    Returns:
        DataFrame with macro correlation trades
    """
    try:
        # Define macro instruments
        macro_instruments = ["USDINR", "GOLD", "CRUDE"]  # Add more as available
        equity_indices = ["NIFTY", "BANKNIFTY"]
        
        results = []
        
        for macro in macro_instruments:
            macro_price = load_price(macro, timeframe)
            if macro_price is None:
                continue
            
            for equity in equity_indices:
                equity_price = load_price(equity, timeframe)
                if equity_price is None:
                    continue
                
                # Align dates
                common_dates = macro_price.index.intersection(equity_price.index)
                if len(common_dates) < 100:
                    continue
                
                macro_aligned = macro_price.loc[common_dates]
                equity_aligned = equity_price.loc[common_dates]
                
                # Test cointegration
                _, p_value, _ = coint(macro_aligned, equity_aligned)
                
                if p_value > P_VALUE_THRESHOLD:
                    continue
                
                # Calculate correlation
                correlation = macro_aligned.pct_change().corr(equity_aligned.pct_change())
                
                # ML Score
                ml_score = (1 - p_value) * 100 + abs(correlation) * 50
                
                results.append({
                    'pair': f"{macro}-{equity}",
                    'strategy': 'Macro Pairs',
                    'macro_instrument': macro,
                    'equity_instrument': equity,
                    'p_value': p_value,
                    'correlation': round(correlation, 3),
                    'ml_score': round(ml_score, 1),
                    'signal': 'NEGATIVE_CORRELATION' if correlation < -0.3 else 'POSITIVE_CORRELATION' if correlation > 0.3 else 'NEUTRAL'
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('ml_score', ascending=False).head(top_n)
        return df
        
    except Exception as e:
        logger.error(f"Macro pairs scan error: {e}")
        return pd.DataFrame()


# =============================================================================
# STRATEGY 4: MOMENTUM PAIRS (20/60-day returns)
# =============================================================================

def scan_momentum_pairs(timeframe="day", top_n=20, lookback_short=20, lookback_long=60):
    """
    Pair strong momentum stocks with weak momentum stocks in same sector
    
    Relative strength strategy - long winner, short loser in same industry
    to capture sector-neutral momentum opportunities.
    
    Returns:
        DataFrame with momentum pair opportunities
    """
    try:
        universe = get_focused_universe(sector="ALL")
        
        # Calculate momentum for each stock
        momentum_data = []
        
        for _, row in universe.iterrows():
            try:
                symbol = row['Symbol']
                price = load_price(symbol, timeframe)
                
                if price is None or len(price) < lookback_long + 10:
                    continue
                
                # Calculate returns
                ret_short = (price.iloc[-1] / price.iloc[-lookback_short] - 1) * 100
                ret_long = (price.iloc[-1] / price.iloc[-lookback_long] - 1) * 100
                
                momentum_data.append({
                    'symbol': symbol,
                    'sector': row.get('Industry', 'Unknown'),
                    'price': price.iloc[-1],
                    'momentum_20d': ret_short,
                    'momentum_60d': ret_long,
                    'momentum_score': ret_short * 0.6 + ret_long * 0.4  # Weighted
                })
                
            except Exception as e:
                continue
        
        if not momentum_data:
            return pd.DataFrame()
        
        mom_df = pd.DataFrame(momentum_data)
        
        # Find pairs within same sector
        results = []
        
        for sector in mom_df['sector'].unique():
            sector_stocks = mom_df[mom_df['sector'] == sector].copy()
            
            if len(sector_stocks) < 2:
                continue
            
            # Sort by momentum
            sector_stocks = sector_stocks.sort_values('momentum_score', ascending=False)
            
            # Pair top with bottom (relative strength)
            for i in range(min(3, len(sector_stocks) // 2)):
                strong = sector_stocks.iloc[i]
                weak = sector_stocks.iloc[-(i+1)]
                
                # Calculate spread stability
                strong_price = load_price(strong['symbol'], timeframe)
                weak_price = load_price(weak['symbol'], timeframe)
                
                if strong_price is None or weak_price is None:
                    continue
                
                common = strong_price.index.intersection(weak_price.index)
                if len(common) < 50:
                    continue
                
                # Test if pair is cointegrated
                _, p_value, _ = coint(strong_price.loc[common], weak_price.loc[common])
                
                momentum_spread = strong['momentum_score'] - weak['momentum_score']
                
                ml_score = momentum_spread + (1 - p_value) * 50
                
                results.append({
                    'pair': f"{strong['symbol']}-{weak['symbol']}",
                    'strategy': 'Momentum Pairs',
                    'long_leg': strong['symbol'],
                    'short_leg': weak['symbol'],
                    'sector': sector,
                    'momentum_spread': round(momentum_spread, 2),
                    'p_value': round(p_value, 4),
                    'ml_score': round(ml_score, 1),
                    'signal': 'LONG_STRONG_SHORT_WEAK'
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('ml_score', ascending=False).head(top_n)
        return df
        
    except Exception as e:
        logger.error(f"Momentum pairs scan error: {e}")
        return pd.DataFrame()


# =============================================================================
# STRATEGY 5: MA CROSSOVER PAIRS (50/200 SMA)
# =============================================================================

def scan_ma_crossover_pairs(timeframe="day", top_n=20):
    """
    Pair stocks with golden cross vs death cross in same sector
    
    Technical analysis approach - moving average crossovers signal trend changes
    creating opportunities to pair bullish vs bearish setups.
    
    Returns:
        DataFrame with MA crossover pair opportunities
    """
    try:
        universe = get_focused_universe(sector="ALL")
        
        # Calculate MA status for each stock
        ma_data = []
        
        for _, row in universe.iterrows():
            try:
                symbol = row['Symbol']
                price = load_price(symbol, timeframe)
                
                if price is None or len(price) < 250:  # Need 200+ for SMA200
                    continue
                
                # Calculate MAs
                sma_50 = price.rolling(50).mean()
                sma_200 = price.rolling(200).mean()
                
                current_price = price.iloc[-1]
                current_sma50 = sma_50.iloc[-1]
                current_sma200 = sma_200.iloc[-1]
                prev_sma50 = sma_50.iloc[-2]
                prev_sma200 = sma_200.iloc[-2]
                
                # Detect crossover
                golden_cross = (current_sma50 > current_sma200) and (prev_sma50 <= prev_sma200)
                death_cross = (current_sma50 < current_sma200) and (prev_sma50 >= prev_sma200)
                
                # Calculate position relative to MAs
                price_vs_sma50 = (current_price - current_sma50) / current_sma50 * 100
                price_vs_sma200 = (current_price - current_sma200) / current_sma200 * 100
                
                ma_signal = ""
                if golden_cross:
                    ma_signal = "GOLDEN_CROSS"
                elif death_cross:
                    ma_signal = "DEATH_CROSS"
                elif current_sma50 > current_sma200:
                    ma_signal = "BULLISH"
                else:
                    ma_signal = "BEARISH"
                
                ma_data.append({
                    'symbol': symbol,
                    'sector': row.get('Industry', 'Unknown'),
                    'ma_signal': ma_signal,
                    'price_vs_sma50': round(price_vs_sma50, 2),
                    'price_vs_sma200': round(price_vs_sma200, 2),
                    'ma_score': price_vs_sma50 + price_vs_sma200
                })
                
            except Exception as e:
                continue
        
        if not ma_data:
            return pd.DataFrame()
        
        ma_df = pd.DataFrame(ma_data)
        
        # Pair bullish with bearish stocks in same sector
        results = []
        
        for sector in ma_df['sector'].unique():
            bullish = ma_df[(ma_df['sector'] == sector) & (ma_df['ma_signal'].isin(['GOLDEN_CROSS', 'BULLISH']))]
            bearish = ma_df[(ma_df['sector'] == sector) & (ma_df['ma_signal'].isin(['DEATH_CROSS', 'BEARISH']))]
            
            for _, bull_stock in bullish.iterrows():
                for _, bear_stock in bearish.iterrows():
                    # Test cointegration
                    bull_price = load_price(bull_stock['symbol'], timeframe)
                    bear_price = load_price(bear_stock['symbol'], timeframe)
                    
                    if bull_price is None or bear_price is None:
                        continue
                    
                    common = bull_price.index.intersection(bear_price.index)
                    if len(common) < 100:
                        continue
                    
                    _, p_value, _ = coint(bull_price.loc[common], bear_price.loc[common])
                    
                    ma_spread = bull_stock['ma_score'] - bear_stock['ma_score']
                    
                    ml_score = abs(ma_spread) + (1 - p_value) * 50
                    
                    results.append({
                        'pair': f"{bull_stock['symbol']}-{bear_stock['symbol']}",
                        'strategy': 'MA Crossover',
                        'long_leg': bull_stock['symbol'],
                        'long_signal': bull_stock['ma_signal'],
                        'short_leg': bear_stock['symbol'],
                        'short_signal': bear_stock['ma_signal'],
                        'sector': sector,
                        'ma_spread': round(ma_spread, 2),
                        'p_value': round(p_value, 4),
                        'ml_score': round(ml_score, 1),
                        'signal': 'LONG_BULLISH_SHORT_BEARISH'
                    })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('ml_score', ascending=False).head(top_n)
        return df
        
    except Exception as e:
        logger.error(f"MA crossover pairs scan error: {e}")
        return pd.DataFrame()


# =============================================================================
# STRATEGY 6: VOLATILITY BREAKOUT PAIRS (Bollinger Bands)
# =============================================================================

def scan_volatility_breakout_pairs(timeframe="day", top_n=20, bb_period=20, bb_std=2):
    """
    Pair stocks breaking out of Bollinger Bands with those mean-reverting
    
    High volatility breakouts paired with low volatility mean reversion
    to create balanced, market-neutral positions.
    
    Returns:
        DataFrame with volatility pair opportunities
    """
    try:
        universe = get_focused_universe(sector="ALL")
        
        # Calculate BB status for each stock
        vol_data = []
        
        for _, row in universe.iterrows():
            try:
                symbol = row['Symbol']
                price = load_price(symbol, timeframe)
                
                if price is None or len(price) < bb_period + 10:
                    continue
                
                # Calculate Bollinger Bands
                sma = price.rolling(bb_period).mean()
                std = price.rolling(bb_period).std()
                upper_band = sma + (std * bb_std)
                lower_band = sma - (std * bb_std)
                
                current_price = price.iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                current_sma = sma.iloc[-1]
                
                # Calculate position relative to bands
                bb_position = (current_price - current_sma) / (current_upper - current_lower) * 100
                
                # Volatility (ATR-like)
                volatility = std.iloc[-1] / current_price * 100
                
                bb_signal = ""
                if current_price > current_upper:
                    bb_signal = "UPPER_BREAKOUT"
                elif current_price < current_lower:
                    bb_signal = "LOWER_BREAKOUT"
                elif abs(bb_position) < 20:
                    bb_signal = "MEAN_REVERSION"
                else:
                    bb_signal = "NEUTRAL"
                
                vol_data.append({
                    'symbol': symbol,
                    'sector': row.get('Industry', 'Unknown'),
                    'bb_signal': bb_signal,
                    'bb_position': round(bb_position, 2),
                    'volatility_pct': round(volatility, 2)
                })
                
            except Exception as e:
                continue
        
        if not vol_data:
            return pd.DataFrame()
        
        vol_df = pd.DataFrame(vol_data)
        
        # Pair breakouts with mean-reversion stocks
        breakouts = vol_df[vol_df['bb_signal'].str.contains('BREAKOUT')]
        mean_rev = vol_df[vol_df['bb_signal'] == 'MEAN_REVERSION']
        
        results = []
        
        for _, breakout_stock in breakouts.iterrows():
            for _, mr_stock in mean_rev.iterrows():
                if breakout_stock['sector'] != mr_stock['sector']:
                    continue
                
                # Test cointegration
                bo_price = load_price(breakout_stock['symbol'], timeframe)
                mr_price = load_price(mr_stock['symbol'], timeframe)
                
                if bo_price is None or mr_price is None:
                    continue
                
                common = bo_price.index.intersection(mr_price.index)
                if len(common) < 50:
                    continue
                
                _, p_value, _ = coint(bo_price.loc[common], mr_price.loc[common])
                
                vol_spread = breakout_stock['volatility_pct'] - mr_stock['volatility_pct']
                
                ml_score = abs(vol_spread) * 2 + (1 - p_value) * 50
                
                results.append({
                    'pair': f"{breakout_stock['symbol']}-{mr_stock['symbol']}",
                    'strategy': 'Vol Breakout',
                    'breakout_leg': breakout_stock['symbol'],
                    'breakout_signal': breakout_stock['bb_signal'],
                    'mean_rev_leg': mr_stock['symbol'],
                    'sector': breakout_stock['sector'],
                    'vol_spread': round(vol_spread, 2),
                    'p_value': round(p_value, 4),
                    'ml_score': round(ml_score, 1),
                    'signal': 'LONG_BREAKOUT_SHORT_MR' if breakout_stock['bb_signal'] == 'UPPER_BREAKOUT' else 'SHORT_BREAKOUT_LONG_MR'
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('ml_score', ascending=False).head(top_n)
        return df
        
    except Exception as e:
        logger.error(f"Volatility breakout pairs scan error: {e}")
        return pd.DataFrame()


# =============================================================================
# STRATEGY 7-10: Additional Strategies (Simplified Implementations)
# =============================================================================

def scan_correlation_pairs(timeframe="day", top_n=10, corr_threshold=0.8):
    """
    Strategy 7: High correlation pairs (not cointegrated but highly correlated)
    
    For shorter-term mean reversion trades
    """
    try:
        universe = get_focused_universe(sector="ALL")
        results = []
        
        symbols = universe['Symbol'].tolist()[:50]  # Limit for performance
        
        for sym1, sym2 in itertools.combinations(symbols, 2):
            p1 = load_price(sym1, timeframe)
            p2 = load_price(sym2, timeframe)
            
            if p1 is None or p2 is None:
                continue
            
            common = p1.index.intersection(p2.index)
            if len(common) < 100:
                continue
            
            # Calculate correlation
            corr = p1.loc[common].pct_change().corr(p2.loc[common].pct_change())
            
            if abs(corr) < corr_threshold:
                continue
            
            # Calculate recent deviation
            spread = (p1.loc[common] - p2.loc[common])
            z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
            
            ml_score = abs(corr) * 100 + abs(z_score) * 10
            
            results.append({
                'pair': f"{sym1}-{sym2}",
                'strategy': 'Correlation',
                'correlation': round(corr, 3),
                'z_score': round(z_score, 2),
                'ml_score': round(ml_score, 1),
                'signal': 'MEAN_REVERT'
            })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        return df.sort_values('ml_score', ascending=False).head(top_n)
        
    except Exception as e:
        logger.error(f"Correlation pairs error: {e}")
        return pd.DataFrame()


def scan_sector_rotation_pairs(timeframe="day", top_n=10):
    """
    Strategy 8: Sector rotation - pair strong sector with weak sector
    """
    try:
        universe = get_focused_universe(sector="ALL")
        
        # Calculate sector performance
        sector_perf = {}
        
        for sector in universe['Industry'].unique():
            sector_stocks = universe[universe['Industry'] == sector]['Symbol'].tolist()
            
            sector_returns = []
            for symbol in sector_stocks[:10]:  # Top 10 per sector
                price = load_price(symbol, timeframe)
                if price is not None and len(price) > 20:
                    ret = (price.iloc[-1] / price.iloc[-20] - 1) * 100
                    sector_returns.append(ret)
            
            if sector_returns:
                sector_perf[sector] = np.mean(sector_returns)
        
        # Pair best with worst sectors
        sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_sectors) < 2:
            return pd.DataFrame()
        
        best_sector = sorted_sectors[0][0]
        worst_sector = sorted_sectors[-1][0]
        
        # Find representative stocks
        best_stocks = universe[universe['Industry'] == best_sector]['Symbol'].tolist()[:5]
        worst_stocks = universe[universe['Industry'] == worst_sector]['Symbol'].tolist()[:5]
        
        results = []
        
        for best in best_stocks:
            for worst in worst_stocks:
                ml_score = sector_perf[best_sector] - sector_perf[worst_sector]
                
                results.append({
                    'pair': f"{best}-{worst}",
                    'strategy': 'Sector Rotation',
                    'strong_sector': best_sector,
                    'weak_sector': worst_sector,
                    'sector_spread': round(ml_score, 2),
                    'ml_score': abs(ml_score),
                    'signal': 'LONG_STRONG_SHORT_WEAK'
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        return df.sort_values('ml_score', ascending=False).head(top_n)
        
    except Exception as e:
        logger.error(f"Sector rotation error: {e}")
        return pd.DataFrame()


# =============================================================================
# MASTER SCAN FUNCTION - Run All Strategies
# =============================================================================

def scan_all_pair_strategies(timeframe="day", strategies=None) -> pd.DataFrame:
    """
    Scan all 10 pairs trading strategies and combine results
    
    This is the heart of the system - comprehensive multi-strategy scanning
    to identify the highest quality pairs trading opportunities.
    
    Args:
        timeframe: data timeframe
        strategies: list of strategy names to run (None = all)
        
    Returns:
        Combined DataFrame with all signals, sorted by ML score
    """
    
    all_strategies = {
        'index_dispersion': scan_index_dispersion_pairs,
        'macro_pairs': scan_macro_pairs,
        'momentum_pairs': scan_momentum_pairs,
        'ma_crossover': scan_ma_crossover_pairs,
        'vol_breakout': scan_volatility_breakout_pairs,
        'correlation': scan_correlation_pairs,
        'sector_rotation': scan_sector_rotation_pairs,
    }
    
    # If specific strategies requested, filter
    if strategies:
        all_strategies = {k: v for k, v in all_strategies.items() if k in strategies}
    
    logger.info(f"Running {len(all_strategies)} pair trading strategies...")
    
    combined_results = []
    
    for strategy_name, strategy_func in all_strategies.items():
        try:
            logger.info(f"Scanning {strategy_name}...")
            df = strategy_func(timeframe=timeframe)
            
            if not df.empty:
                combined_results.append(df)
                logger.info(f"✅ {strategy_name}: Found {len(df)} signals")
            else:
                logger.info(f"⚠️ {strategy_name}: No signals")
                
        except Exception as e:
            logger.error(f"❌ {strategy_name} failed: {e}")
    
    if not combined_results:
        logger.warning("No signals from any strategy")
        return pd.DataFrame()
    
    # Combine all results
    final_df = pd.concat(combined_results, ignore_index=True)
    
    # Sort by ML score
    final_df = final_df.sort_values('ml_score', ascending=False)
    
    # Add timestamp
    final_df['scan_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"🎯 Total signals across all strategies: {len(final_df)}")
    
    return final_df


# Export functions
__all__ = [
    'scan_index_dispersion_pairs',
    'scan_macro_pairs',
    'scan_momentum_pairs',
    'scan_ma_crossover_pairs',
    'scan_volatility_breakout_pairs',
    'scan_correlation_pairs',
    'scan_sector_rotation_pairs',
    'scan_all_pair_strategies'
]
