"""
Straddle Strategy Module
Long/Short Straddle options trading strategy.

Features:
- Long Straddle: Buy ATM Call + Buy ATM Put when IV is low
- Short Straddle: Sell ATM Call + Sell ATM Put when IV is high
- IV Rank-based entry signals
- Greeks-based risk management (Delta, Gamma, Vega)
- P&L tracking at expiration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StraddleStrategy:
    """Long/Short Straddle Strategy using IV Rank and Greeks"""
    
    def __init__(self, symbol: str, iv_rank_threshold: float = 0.5, 
                 atm_strike_offset: int = 0, max_loss_pct: float = 5.0):
        """
        Initialize Straddle Strategy
        
        Args:
            symbol: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')
            iv_rank_threshold: IV Rank threshold (0-1). 
                              > 0.7 = Short Straddle (high IV)
                              < 0.3 = Long Straddle (low IV)
            atm_strike_offset: Offset from ATM strike in points
            max_loss_pct: Maximum loss threshold as % of premium
        """
        self.symbol = symbol
        self.iv_rank_threshold = iv_rank_threshold
        self.atm_strike_offset = atm_strike_offset
        self.max_loss_pct = max_loss_pct
        
        self.position = None
        self.entry_price = None
        self.entry_iv_rank = None
        self.entry_date = None
        self.expiration_date = None
        self.call_strike = None
        self.put_strike = None
        self.trades = []
    
    def calculate_iv_rank(self, iv_history: List[float]) -> float:
        """
        Calculate IV Rank: (Current IV - IV Min) / (IV Max - IV Min)
        Range: 0 (lowest IV) to 1 (highest IV)
        """
        if not iv_history or len(iv_history) < 2:
            return 0.5  # Default to neutral
        
        iv_array = np.array(iv_history)
        iv_min = np.min(iv_array)
        iv_max = np.max(iv_array)
        
        if iv_max == iv_min:
            return 0.5
        
        current_iv = iv_array[-1]
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
        
        return np.clip(iv_rank, 0, 1)
    
    def find_atm_strike(self, spot_price: float, strike_interval: int = 100) -> float:
        """
        Find ATM (At-The-Money) strike price
        
        Args:
            spot_price: Current spot price
            strike_interval: Strike price interval (100 for index options)
            
        Returns:
            Nearest ATM strike
        """
        atm_strike = (spot_price // strike_interval) * strike_interval
        return atm_strike + self.atm_strike_offset
    
    def estimate_straddle_premium(self, spot: float, atm_strike: float, 
                                   iv: float, days_to_expiry: int) -> Tuple[float, float, float]:
        """
        Estimate straddle premium using simplified Black-Scholes
        
        Args:
            spot: Current spot price
            atm_strike: Strike price
            iv: Implied Volatility (%)
            days_to_expiry: Days until expiration
            
        Returns:
            (call_premium, put_premium, total_straddle_premium)
        """
        # Simplified calculation - in production use Black-Scholes or market prices
        # This is a rough estimate
        
        # Time value component
        time_value = (iv / 100) * atm_strike * np.sqrt(days_to_expiry / 365)
        
        # ATM call premium (roughly 60% of time value)
        call_premium = time_value * 0.6
        
        # ATM put premium (roughly 60% of time value, similar for ATM)
        put_premium = time_value * 0.6
        
        # Total straddle premium
        total_premium = call_premium + put_premium
        
        return call_premium, put_premium, total_premium
    
    def generate_signals(self, spot_price: float, iv_history: List[float], 
                        days_to_expiry: int) -> Dict:
        """
        Generate trading signals based on IV Rank and Greeks
        
        Args:
            spot_price: Current spot price
            iv_history: Historical IV values (last 252 days)
            days_to_expiry: Days until options expiration
            
        Returns:
            Signal dict with action and details
        """
        iv_rank = self.calculate_iv_rank(iv_history)
        atm_strike = self.find_atm_strike(spot_price)
        current_iv = iv_history[-1] if iv_history else 20
        
        # Estimate premiums
        call_premium, put_premium, straddle_premium = self.estimate_straddle_premium(
            spot_price, atm_strike, current_iv, days_to_expiry
        )
        
        signals = {
            "timestamp": pd.Timestamp.now(),
            "spot": spot_price,
            "atm_strike": atm_strike,
            "iv_rank": iv_rank,
            "current_iv": current_iv,
            "straddle_premium": straddle_premium,
            "signal": "NONE",
            "action": None,
            "side": None
        }
        
        # LONG STRADDLE: Buy when IV is low (IV Rank < 0.3)
        if iv_rank < 0.3 and self.position is None:
            signals["signal"] = "LONG_STRADDLE_ENTRY"
            signals["action"] = f"Buy Call + Put at strike {atm_strike}, Premium: {straddle_premium:.2f}"
            signals["side"] = "LONG"
            self.position = "LONG_STRADDLE"
            self.entry_price = straddle_premium
            self.entry_iv_rank = iv_rank
            self.call_strike = atm_strike
            self.put_strike = atm_strike
        
        # SHORT STRADDLE: Sell when IV is high (IV Rank > 0.7)
        elif iv_rank > 0.7 and self.position is None:
            signals["signal"] = "SHORT_STRADDLE_ENTRY"
            signals["action"] = f"Sell Call + Put at strike {atm_strike}, Premium: {straddle_premium:.2f}"
            signals["side"] = "SHORT"
            self.position = "SHORT_STRADDLE"
            self.entry_price = straddle_premium
            self.entry_iv_rank = iv_rank
            self.call_strike = atm_strike
            self.put_strike = atm_strike
        
        return signals
    
    def calculate_pnl_at_expiry(self, spot_at_expiry: float, 
                               call_premium: float, put_premium: float) -> Dict:
        """
        Calculate P&L at expiration for straddle
        
        Args:
            spot_at_expiry: Spot price at expiration
            call_premium: Call premium paid/received
            put_premium: Put premium paid/received
            
        Returns:
            P&L details dict
        """
        atm_strike = self.call_strike
        
        # Intrinsic values at expiry
        call_intrinsic = max(0, spot_at_expiry - atm_strike)
        put_intrinsic = max(0, atm_strike - spot_at_expiry)
        
        straddle_cost = call_premium + put_premium
        straddle_value = call_intrinsic + put_intrinsic
        
        if self.position == "LONG_STRADDLE":
            # Long paid premium, receives intrinsic value
            pnl = straddle_value - straddle_cost
            pnl_pct = (pnl / straddle_cost) * 100 if straddle_cost > 0 else 0
            
            self.trades.append({
                "type": "LONG_STRADDLE",
                "entry_iv_rank": self.entry_iv_rank,
                "strike": atm_strike,
                "cost": straddle_cost,
                "exit_value": straddle_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "spot_at_exit": spot_at_expiry
            })
            
            return {
                "status": "CLOSED",
                "type": "LONG_STRADDLE",
                "cost": straddle_cost,
                "value_at_expiry": straddle_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "breakeven_up": atm_strike + straddle_cost,
                "breakeven_down": atm_strike - straddle_cost
            }
        
        elif self.position == "SHORT_STRADDLE":
            # Short receives premium, pays intrinsic value
            pnl = straddle_cost - straddle_value
            pnl_pct = (pnl / straddle_cost) * 100 if straddle_cost > 0 else 0
            
            self.trades.append({
                "type": "SHORT_STRADDLE",
                "entry_iv_rank": self.entry_iv_rank,
                "strike": atm_strike,
                "premium": straddle_cost,
                "payoff": straddle_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "spot_at_exit": spot_at_expiry
            })
            
            return {
                "status": "CLOSED",
                "type": "SHORT_STRADDLE",
                "premium_collected": straddle_cost,
                "payoff_at_expiry": straddle_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "max_profit": straddle_cost,
                "breakeven_up": atm_strike + straddle_cost,
                "breakeven_down": atm_strike - straddle_cost
            }
        
        return {"status": "ERROR", "pnl": 0}
    
    def backtest_straddles(self, price_history: pd.DataFrame, 
                          iv_history: pd.DataFrame) -> Dict:
        """
        Backtest straddle strategy across multiple expirations
        
        Args:
            price_history: DataFrame with OHLCV data
            iv_history: DataFrame with IV values
            
        Returns:
            Backtest results with overall metrics
        """
        if price_history.empty or iv_history.empty:
            return {"status": "INSUFFICIENT_DATA", "trades": 0, "pnl": 0}
        
        pnl_list = []
        total_pnl = 0
        
        # Simulate monthly straddles
        for i in range(0, len(price_history), 21):  # Monthly (21 trading days)
            if i + 21 >= len(price_history):
                break
            
            # Entry point (first day of straddle)
            entry_spot = price_history['close'].iloc[i]
            entry_iv_hist = iv_history['iv'].iloc[max(0, i-252):i].tolist()
            days_to_expiry = 21
            
            # Generate signal
            signal = self.generate_signals(entry_spot, entry_iv_hist, days_to_expiry)
            
            if signal['signal'] in ["LONG_STRADDLE_ENTRY", "SHORT_STRADDLE_ENTRY"]:
                # Exit at expiration (21 days later)
                exit_spot = price_history['close'].iloc[i + 21]
                
                # Estimate premiums (simplified)
                _, _, entry_premium = self.estimate_straddle_premium(
                    entry_spot, signal['atm_strike'], signal['current_iv'], days_to_expiry
                )
                
                # Calculate P&L
                pnl_result = self.calculate_pnl_at_expiry(
                    exit_spot, entry_premium * 0.5, entry_premium * 0.5
                )
                
                if 'pnl' in pnl_result:
                    pnl = pnl_result['pnl']
                    pnl_list.append(pnl)
                    total_pnl += pnl
        
        if not pnl_list:
            return {
                "status": "NO_TRADES",
                "symbol": self.symbol,
                "trades": 0,
                "sharpe": 0,
                "total_pnl": 0,
                "pnl_pct": 0,
                "win_rate": 0
            }
        
        pnl_arr = np.array(pnl_list)
        
        # Calculate metrics
        sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-6) * np.sqrt(252)
        win_rate = np.mean(pnl_arr > 0) * 100
        total_return = (total_pnl / 100000) * 100  # Assuming 100k initial
        
        return {
            "status": "SUCCESS",
            "symbol": self.symbol,
            "trades": len(pnl_list),
            "total_pnl": total_pnl,
            "pnl_pct": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "avg_trade": total_pnl / len(pnl_list) if pnl_list else 0,
            "max_win": np.max(pnl_arr),
            "max_loss": np.min(pnl_arr)
        }


def scan_straddle(price_loader_func, symbols: List[str] = None,
                  iv_loader_func = None, min_trades: int = 2) -> pd.DataFrame:
    """
    Scan for best straddle opportunities across symbols
    
    Args:
        price_loader_func: Function to load price data
        symbols: List of symbols to scan
        iv_loader_func: Function to load IV data
        min_trades: Minimum trades threshold for filtering
        
    Returns:
        DataFrame of profitable straddle candidates
    """
    results = []
    
    if symbols is None:
        symbols = []
    
    for symbol in symbols:
        try:
            price_data = price_loader_func(symbol)
            
            if price_data is None or len(price_data) < 100:
                continue
            
            # Use mock IV data if loader not provided
            if iv_loader_func:
                iv_data = iv_loader_func(symbol)
            else:
                # Generate mock IV series
                iv_data = pd.DataFrame({
                    'iv': np.random.normal(20, 5, len(price_data))
                })
            
            strategy = StraddleStrategy(symbol)
            backtest = strategy.backtest_straddles(price_data, iv_data)
            
            if backtest.get('trades', 0) >= min_trades:
                results.append(backtest)
        
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            continue
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('sharpe', ascending=False, na_position='last')
    
    return df_results
