# core/strategies.py
import pandas as pd
from config import Z_ENTRY, Z_EXIT
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
import numpy as np
from typing import Dict, Tuple, Optional, List
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

def get_hedge_ratio(price_x, price_y):
    model = OLS(price_x, price_y).fit()
    return model.params[0]

def get_zscore(spread, window=60):
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

def should_enter_with_options(iv_rank, side="long", sell_threshold=70, buy_threshold=30):
    """Decide action based on IV rank and configurable thresholds.

    - `sell_threshold`: minimum IV rank to consider selling premium
    - `buy_threshold`: maximum IV rank to consider buying premium
    """
    if side == "short" and iv_rank > sell_threshold:
        return "sell_strangle"
    elif side == "long" and iv_rank < buy_threshold:
        return "buy_straddle"
    else:
        return "futures"  # default

def is_liquid(df, min_volume=50000):
    return df['volume'].mean() > min_volume


class PairsTradingStrategy:
    """Pairs Trading Strategy using cointegration and z-score"""
    
    def __init__(self, symbol_a: str, symbol_b: str, z_entry: float = 2.0, z_exit: float = 0.5, lookback: int = 60):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback
        self.hedge_ratio = 1.0
        self.positions = {"long": 0, "short": 0}
        self.trades = []
    
    def test_cointegration(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[float, bool]:
        """Test if two price series are cointegrated (p-value < 0.05)"""
        try:
            _, p_value, _ = coint(df_a['close'].values, df_b['close'].values)
            is_cointegrated = p_value < 0.05
            return p_value, is_cointegrated
        except:
            return 1.0, False
    
    def calculate_spread(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.Series:
        """Calculate spread as: price_a - hedge_ratio * price_b"""
        self.hedge_ratio = get_hedge_ratio(df_a['close'].values, df_b['close'].values)
        spread = df_a['close'] - self.hedge_ratio * df_b['close']
        return spread
    
    def generate_signals(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict:
        """Generate trading signals based on z-score"""
        spread = self.calculate_spread(df_a, df_b)
        zscore = get_zscore(spread, window=self.lookback)
        
        latest_z = zscore.iloc[-1]
        
        signals = {
            "timestamp": df_a.index[-1],
            "spread": spread.iloc[-1],
            "zscore": latest_z,
            "signal": "NONE",
            "action": None
        }
        
        # Entry signals
        if latest_z > self.z_entry and self.positions["long"] == 0:
            signals["signal"] = "SHORT_ENTRY"
            signals["action"] = "Sell A, Buy B"
            self.positions["short"] += 1
        elif latest_z < -self.z_entry and self.positions["long"] == 0:
            signals["signal"] = "LONG_ENTRY"
            signals["action"] = "Buy A, Sell B"
            self.positions["long"] += 1
        
        # Exit signals
        if abs(latest_z) < self.z_exit:
            if self.positions["long"] > 0:
                signals["signal"] = "LONG_EXIT"
                signals["action"] = "Close A-B"
                self.positions["long"] = 0
            elif self.positions["short"] > 0:
                signals["signal"] = "SHORT_EXIT"
                signals["action"] = "Close B-A"
                self.positions["short"] = 0
        
        return signals
    
    
    def calculate_rolling_ols(self, df_a: pd.DataFrame, df_b: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate rolling Hedge Ratio and Spread to avoid look-ahead bias.
        Uses Rolling OLS to determine beta for each day based on past 'window' days.
        """
        hedge_ratios = []
        spreads = []
        
        # We need at least 'window' data points to start
        # Fill initial part with NaNs
        hedge_ratios = [np.nan] * window
        spreads = [np.nan] * window
        
        y = df_a['close'].values
        x = df_b['close'].values
        
        for i in range(window, len(df_a)):
            # Slice the lookback window (i-window to i)
            y_window = y[i-window:i]
            x_window = x[i-window:i]
            
            # Simple OLS: y = beta * x + alpha (we ignore alpha for pure hedge ratio typically, or keep it. 
            # Ideally: price_A = beta * price_B + spread
            # beta = cov(x,y) / var(x)
            
            # Using numpy for speed
            beta = np.cov(x_window, y_window)[0, 1] / np.var(x_window)
            
            # Current spread using TODAY's price and YESTERDAY's beta (simulating real trading)
            # OR Today's beta. Using Today's beta is fine as it uses past window.
            
            current_spread = y[i] - beta * x[i]
            
            hedge_ratios.append(beta)
            spreads.append(current_spread)
            
        return pd.Series(hedge_ratios, index=df_a.index), pd.Series(spreads, index=df_a.index)

    def backtest(self, df_a: pd.DataFrame, df_b: pd.DataFrame, initial_capital: float = 100000, use_rolling_ols: bool = True) -> Dict:
        """Full backtest with P&L calculation (Rolling OLS supported)"""
        
        # 1. Calculate Spread & Z-Score
        if use_rolling_ols:
            # ROLLING OLS (Honest Backtest)
            hedge_ratio_series, spread_series = self.calculate_rolling_ols(df_a, df_b, self.lookback)
            # Z-score of the rolling spread (dynamic mean/std of the dynamic spread)
            # Note: We usually z-score the rolling spread over the same lookback or a shorter one.
            # Standard practice: Z = (Spread - RollingMean(Spread)) / RollingStd(Spread)
            rolling_mean = spread_series.rolling(window=self.lookback).mean()
            rolling_std = spread_series.rolling(window=self.lookback).std()
            zscore = (spread_series - rolling_mean) / rolling_std
        else:
            # STATIC OLS (Traditional/Optimistic)
            # Tests cointegration over entire period first
            p_value, is_cointegrated = self.test_cointegration(df_a, df_b)
            if not is_cointegrated:
                return {
                    "status": "FAILED",
                    "reason": f"Not cointegrated (p={p_value:.4f})",
                    "trades": 0, "sharpe": 0, "pnl_pct": 0
                }
            hedge_ratio_static = get_hedge_ratio(df_a['close'].values, df_b['close'].values)
            spread_series = df_a['close'] - hedge_ratio_static * df_b['close']
            zscore = get_zscore(spread_series, window=self.lookback)
            hedge_ratio_series = pd.Series([hedge_ratio_static] * len(df_a), index=df_a.index)

        pnl_list = []
        entry_price_a, entry_price_b = None, None
        position = None
        entry_hedge_ratio = 1.0
        
        # Start loop after lookback (and if rolling, we need lookback*2 presumably to get z-score of rolling spread)
        start_idx = self.lookback * 2 if use_rolling_ols else self.lookback
        
        for i in range(start_idx, len(df_a)):
            if pd.isna(zscore.iloc[i]): continue
            
            z = zscore.iloc[i]
            price_a = df_a['close'].iloc[i]
            price_b = df_b['close'].iloc[i]
            current_hedge_ratio = hedge_ratio_series.iloc[i]
            
            # Entry
            if position is None:
                if z > self.z_entry:
                    position = "SHORT" # Sell A, Buy B
                    entry_price_a = price_a
                    entry_price_b = price_b
                    entry_hedge_ratio = current_hedge_ratio
                elif z < -self.z_entry:
                    position = "LONG" # Buy A, Sell B
                    entry_price_a = price_a
                    entry_price_b = price_b
                    entry_hedge_ratio = current_hedge_ratio
            
            # Exit
            elif abs(z) < self.z_exit:
                # We use the Hedge Ratio captured AT ENTRY to calculate PnL 
                # (You traded X shares of B based on entry beta, you hold that specific quantity til exit)
                
                if position == "SHORT":
                    # Sold A (P_entry - P_exit), Bought B * Beta (P_exit - P_entry)
                    pnl_a = entry_price_a - price_a
                    pnl_b = (price_b - entry_price_b) * entry_hedge_ratio
                else: # LONG
                    # Bought A (P_exit - P_entry), Sold B * Beta (P_entry - P_exit)
                    pnl_a = price_a - entry_price_a
                    pnl_b = (entry_price_b - price_b) * entry_hedge_ratio
                
                total_pnl = pnl_a + pnl_b
                # We might want to scale this to invested capital, but for raw PnL:
                
                pnl_list.append(total_pnl)
                self.trades.append({
                    "entry_a": entry_price_a, "exit_a": price_a,
                    "entry_b": entry_price_b, "exit_b": price_b,
                    "beta": entry_hedge_ratio,
                    "pnl": total_pnl,
                    "position": position
                })
                position = None
        
        if not pnl_list:
            return {"status": "NO_TRADES", "trades": 0, "sharpe": 0, "pnl_pct": 0}
        
        pnl_arr = np.array(pnl_list)
        total_pnl = pnl_arr.sum()
        # ROI Approximation: Assume we use ~20% of capital for margin per trade or full cash
        total_return = (total_pnl / initial_capital) * 100
        sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-6) * np.sqrt(252)
        
        return {
            "status": "SUCCESS",
            "trades": len(pnl_list),
            "total_pnl": total_pnl,
            "pnl_pct": total_return,
            "sharpe": sharpe,
            "win_rate": np.mean(pnl_arr > 0) * 100
        }


class PairsTradingBatchScanner:
    """Scan all NIFTY 500 pairs for cointegration"""
    
    def __init__(self, symbols: List[str], z_entry: float = 2.0, z_exit: float = 0.5, 
                 lookback: int = 60, min_cointegration_p: float = 0.05):
        self.symbols = symbols  # All NIFTY 500 symbols
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback
        self.min_cointegration_p = min_cointegration_p
        self.cointegrated_pairs = []
        self.symbols_with_options = set()  # Track which have options
    
    def set_options_availability(self, symbols_with_options: set):
        """Set which symbols have options trading available"""
        self.symbols_with_options = symbols_with_options
    
    def scan_all_pairs(self, price_data_func) -> List[Dict]:
        """
        Test all pairs for cointegration
        
        Args:
            price_data_func: Callable that takes (symbol_a, symbol_b) and returns (df_a, df_b)
        
        Returns:
            List of cointegrated pairs sorted by p-value (strongest first)
        """
        self.cointegrated_pairs = []
        pair_count = 0
        cointegrated_count = 0
        
        logger.info(f"Starting pair scan for {len(self.symbols)} symbols...")
        
        # Generate all unique pairs
        for symbol_a, symbol_b in combinations(self.symbols, 2):
            pair_count += 1
            
            try:
                # Get price data for both symbols
                df_a, df_b = price_data_func(symbol_a, symbol_b)
                
                if df_a is None or df_b is None or len(df_a) < self.lookback:
                    continue
                
                # Test cointegration
                _, p_value, _ = coint(df_a['close'].values, df_b['close'].values)
                
                # If cointegrated, store result
                if p_value < self.min_cointegration_p:
                    cointegrated_count += 1
                    
                    # Calculate hedge ratio
                    hedge_ratio = get_hedge_ratio(df_a['close'].values, df_b['close'].values)
                    
                    # Quick backtest
                    strategy = PairsTradingStrategy(symbol_a, symbol_b, self.z_entry, 
                                                   self.z_exit, self.lookback)
                    backtest_result = strategy.backtest(df_a, df_b)
                    
                    # Check options availability
                    options_available = (symbol_a in self.symbols_with_options and 
                                       symbol_b in self.symbols_with_options)
                    
                    pair_info = {
                        "symbol_a": symbol_a,
                        "symbol_b": symbol_b,
                        "p_value": p_value,
                        "hedge_ratio": hedge_ratio,
                        "options_available": options_available,
                        "backtest_result": backtest_result,
                        "sharpe": backtest_result.get("sharpe", 0),
                        "pnl_pct": backtest_result.get("pnl_pct", 0),
                        "trades": backtest_result.get("trades", 0),
                        "win_rate": backtest_result.get("win_rate", 0)
                    }
                    
                    self.cointegrated_pairs.append(pair_info)
                
                if pair_count % 1000 == 0:
                    logger.info(f"Scanned {pair_count} pairs, found {cointegrated_count} cointegrated")
            
            except Exception as e:
                logger.debug(f"Error scanning pair {symbol_a}-{symbol_b}: {e}")
                continue
        
        # Sort by p-value (strongest cointegration first)
        self.cointegrated_pairs.sort(key=lambda x: x['p_value'])
        
        logger.info(f"Pair scan complete: {cointegrated_count} cointegrated pairs from {pair_count} tested")
        
        return self.cointegrated_pairs
    
    def get_top_pairs(self, limit: int = 50, filter_options: bool = False) -> List[Dict]:
        """
        Get top N cointegrated pairs
        
        Args:
            limit: Number of top pairs to return
            filter_options: If True, only return pairs where both have options
        
        Returns:
            Ranked list of pairs
        """
        pairs = self.cointegrated_pairs
        
        if filter_options:
            pairs = [p for p in pairs if p['options_available']]
        
        return pairs[:limit]
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return all cointegrated pairs as DataFrame for easy analysis"""
        if not self.cointegrated_pairs:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'Symbol A': p['symbol_a'],
            'Symbol B': p['symbol_b'],
            'P-Value': f"{p['p_value']:.6f}",
            'Hedge Ratio': f"{p['hedge_ratio']:.4f}",
            'Options': 'Yes' if p['options_available'] else 'No',
            'Sharpe': f"{p['sharpe']:.2f}",
            'PnL %': f"{p['pnl_pct']:.2f}",
            'Trades': p['trades'],
            'Win Rate': f"{p['win_rate']:.1f}%"
        } for p in self.cointegrated_pairs])


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
                position = None
        
        if not pnl_list:
            return {"status": "NO_TRADES", "trades": 0, "sharpe": 0, "pnl_pct": 0}
        
        pnl_arr = np.array(pnl_list)
        total_pnl = pnl_arr.sum()
        total_return = (total_pnl / initial_capital) * 100
        sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-6) * np.sqrt(252)
        
        return {
            "status": "SUCCESS",
            "trades": len(pnl_list),
            "total_pnl": total_pnl,
            "pnl_pct": total_return,
            "sharpe": sharpe,
            "win_rate": np.mean(pnl_arr > 0) * 100
        }


class MovingAverageCrossover:
    """Moving Average Crossover Strategy"""
    
    def __init__(self, symbol: str, fast_period: int = 20, slow_period: int = 50, use_ema: bool = False):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        self.position = None
        self.entry_price = None
        self.trades = []
    
    def calculate_mas(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate moving averages (SMA or EMA)"""
        if self.use_ema:
            fast_ma = prices.ewm(span=self.fast_period).mean()
            slow_ma = prices.ewm(span=self.slow_period).mean()
        else:
            fast_ma = prices.rolling(self.fast_period).mean()
            slow_ma = prices.rolling(self.slow_period).mean()
        
        return fast_ma, slow_ma
    
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate MA crossover signals"""
        fast_ma, slow_ma = self.calculate_mas(df['close'])
        
        latest_price = df['close'].iloc[-1]
        latest_fast = fast_ma.iloc[-1]
        latest_slow = slow_ma.iloc[-1]
        
        signals = {
            "timestamp": df.index[-1],
            "price": latest_price,
            "fast_ma": latest_fast,
            "slow_ma": latest_slow,
            "signal": "NONE",
            "action": None
        }
        
        # Golden cross: Fast MA > Slow MA
        if latest_fast > latest_slow and self.position is None:
            signals["signal"] = "BUY"
            signals["action"] = "Golden Cross"
            self.position = "LONG"
            self.entry_price = latest_price
        
        # Death cross: Fast MA < Slow MA
        elif latest_fast < latest_slow and self.position == "LONG":
            signals["signal"] = "SELL"
            signals["action"] = "Death Cross"
            pnl = latest_price - self.entry_price
            self.trades.append({"entry": self.entry_price, "exit": latest_price, "pnl": pnl})
            self.position = None
        
        return signals
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Full backtest with P&L calculation"""
        fast_ma, slow_ma = self.calculate_mas(df['close'])
        
        pnl_list = []
        entry_price = None
        position = None
        
        for i in range(self.slow_period, len(df)):
            current_price = df['close'].iloc[i]
            current_fast = fast_ma.iloc[i]
            current_slow = slow_ma.iloc[i]
            
            # Entry: Golden cross
            if position is None and current_fast > current_slow:
                position = "LONG"
                entry_price = current_price
            
            # Exit: Death cross
            elif position == "LONG" and current_fast < current_slow:
                pnl = current_price - entry_price
                pnl_list.append(pnl)
                self.trades.append({
                    "entry": entry_price,
                    "exit": current_price,
                    "pnl": pnl
                })
                position = None
        
        if not pnl_list:
            return {"status": "NO_TRADES", "trades": 0, "sharpe": 0, "pnl_pct": 0}
        
        pnl_arr = np.array(pnl_list)
        total_pnl = pnl_arr.sum()
        total_return = (total_pnl / initial_capital) * 100
        sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-6) * np.sqrt(252)
        
        return {
            "status": "SUCCESS",
            "trades": len(pnl_list),
            "total_pnl": total_pnl,
            "pnl_pct": total_return,
            "sharpe": sharpe,
            "win_rate": np.mean(pnl_arr > 0) * 100
        }


class StrategyBatchProcessor:
    """Process strategies across all symbols in batches"""
    
    def __init__(self, symbols: List[str], strategy_type: str):
        self.symbols = symbols
        self.strategy_type = strategy_type
        self.results = []
    
    def process_momentum_batch(self, price_data_func, rsi_period: int = 14) -> pd.DataFrame:
        """Backtest momentum strategy for all symbols"""
        results = []
        
        logger.info(f"Processing Momentum for {len(self.symbols)} symbols...")
        
        for i, symbol in enumerate(self.symbols):
            try:
                df = price_data_func(symbol)
                
                if df is None or len(df) < 50:
                    continue
                
                strategy = MomentumStrategy(symbol, rsi_period)
                backtest = strategy.backtest(df)
                
                result = {
                    "symbol": symbol,
                    "status": backtest.get("status", "UNKNOWN"),
                    "trades": backtest.get("trades", 0),
                    "pnl_pct": backtest.get("pnl_pct", 0),
                    "sharpe": backtest.get("sharpe", 0),
                    "win_rate": backtest.get("win_rate", 0)
                }
                
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.symbols)} symbols")
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        self.results = results
        
        # Sort by Sharpe ratio
        results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        
        return pd.DataFrame(results_sorted)
    
    def process_mean_reversion_batch(self, price_data_func, ma_period: int = 20) -> pd.DataFrame:
        """Backtest mean reversion for all symbols"""
        results = []
        
        logger.info(f"Processing Mean Reversion for {len(self.symbols)} symbols...")
        
        for i, symbol in enumerate(self.symbols):
            try:
                df = price_data_func(symbol)
                
                if df is None or len(df) < 50:
                    continue
                
                strategy = MeanReversionStrategy(symbol, ma_period)
                backtest = strategy.backtest(df)
                
                result = {
                    "symbol": symbol,
                    "status": backtest.get("status", "UNKNOWN"),
                    "trades": backtest.get("trades", 0),
                    "pnl_pct": backtest.get("pnl_pct", 0),
                    "sharpe": backtest.get("sharpe", 0),
                    "win_rate": backtest.get("win_rate", 0)
                }
                
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.symbols)} symbols")
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        self.results = results
        
        # Sort by Sharpe ratio
        results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        
        return pd.DataFrame(results_sorted)
    
    def process_ma_crossover_batch(self, price_data_func, fast_period: int = 20, 
                                   slow_period: int = 50) -> pd.DataFrame:
        """Backtest MA crossover for all symbols"""
        results = []
        
        logger.info(f"Processing MA Crossover for {len(self.symbols)} symbols...")
        
        for i, symbol in enumerate(self.symbols):
            try:
                df = price_data_func(symbol)
                
                if df is None or len(df) < 50:
                    continue
                
                strategy = MovingAverageCrossover(symbol, fast_period, slow_period)
                backtest = strategy.backtest(df)
                
                result = {
                    "symbol": symbol,
                    "status": backtest.get("status", "UNKNOWN"),
                    "trades": backtest.get("trades", 0),
                    "pnl_pct": backtest.get("pnl_pct", 0),
                    "sharpe": backtest.get("sharpe", 0),
                    "win_rate": backtest.get("win_rate", 0)
                }
                
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.symbols)} symbols")
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        self.results = results
        
        # Sort by Sharpe ratio
        results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        
        return pd.DataFrame(results_sorted)

class MomentumStrategy:
    """Enhanced Momentum Strategy using RSI with Volume Filtering"""
    """
    RSI-based momentum trading strategy with volume confirmation.
    
    Features:
    - RSI (14-period) with configurable overbought/oversold thresholds
    - Volume filter (only trade if volume > 20-day average)
    - Entry: RSI < 30 (oversold) = BUY signal
    - Entry: RSI > 70 (overbought) = SELL signal (short)
    - Exit: After 5-10 days OR RSI crosses back
    - Backtest: Full P&L, Sharpe ratio, win rate, max drawdown
    
    Supports:
    - Single symbol backtesting
    - Batch processing across all Zerodha instruments
    - NSE equities, derivatives, indices
    """
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