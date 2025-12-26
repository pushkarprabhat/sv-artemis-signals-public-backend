"""
Stock Derivatives Trading Strategies
Individual stock futures, covered calls, protective puts, collars, and spreads.
Designed for retail traders with conservative leverage (1-2x max).

Classes:
- StockDerivativesScanner: Main scanner for all stock derivative strategies
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class StockDerivativesScanner:
    """
    Scanner for individual stock derivative strategies.
    Handles futures, options spreads, and hedging strategies.
    """
    
    def __init__(self, max_leverage: float = 2.0):
        """
        Initialize stock derivatives scanner.
        
        Args:
            max_leverage: Maximum leverage (1-2x recommended for stocks)
        """
        self.max_leverage = max_leverage
        self.typical_leverage = 1.5
        self.min_leverage = 1.0
        self.typical_beta = 0.85  # Average stock beta vs NIFTY
        
    def scan_stock_futures_momentum(
        self,
        current_price: float,
        rsi: float,
        adx: float,
        atr: float,
        volume: float,
        avg_volume: float,
        capital: float = 500000
    ) -> Dict:
        """
        Scan individual stock futures using momentum indicators.
        
        Uses RSI + ADX confirmation with ATR-based position sizing.
        Conservative: 1-1.5x leverage.
        
        Args:
            current_price: Current stock price
            rsi: RSI(14) value
            adx: ADX(14) value
            atr: Average True Range
            volume: Current volume
            avg_volume: Average volume (20-day)
            capital: Available capital (default 500K)
            
        Returns:
            Dict with signal, quantity, stop loss, targets, confidence
        """
        signal = "HOLD"
        confidence = 0
        quantity = 0
        stop_loss = 0
        target_1 = 0
        target_2 = 0
        
        # Trend confirmation: ADX > 20
        if adx > 20:
            # Volume confirmation
            vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            has_volume = vol_ratio > 1.2
            
            if rsi > 55 and has_volume:
                # BUY signal
                signal = "BUY"
                stop_loss = current_price - (2.0 * atr)
                profit_range = 3.0 * atr
                target_1 = current_price + (profit_range * 0.4)
                target_2 = current_price + (profit_range * 1.0)
                
                # Position sizing: capital * leverage / price
                leverage = min(self.typical_leverage, self.max_leverage)
                position_size = (capital * leverage) / current_price
                quantity = int(position_size)
                
                # Confidence: ADX + RSI strength + volume
                adx_score = min(100, (adx - 20) * 5)  # 0-100
                rsi_score = min(100, (rsi - 55) * 5)  # 0-100
                vol_score = min(100, vol_ratio * 50)
                confidence = int((adx_score + rsi_score + vol_score) / 3)
                
            elif rsi < 45 and has_volume:
                # SELL signal
                signal = "SELL"
                stop_loss = current_price + (2.0 * atr)
                profit_range = 3.0 * atr
                target_1 = current_price - (profit_range * 0.4)
                target_2 = current_price - (profit_range * 1.0)
                
                leverage = min(self.typical_leverage, self.max_leverage)
                position_size = (capital * leverage) / current_price
                quantity = int(position_size)
                
                adx_score = min(100, (adx - 20) * 5)
                rsi_score = min(100, (100 - rsi) * 5)
                vol_score = min(100, vol_ratio * 50)
                confidence = int((adx_score + rsi_score + vol_score) / 3)
        
        return {
            "strategy": "Stock_Futures_Momentum",
            "signal": signal,
            "quantity": quantity,
            "stop_loss": round(stop_loss, 2),
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "confidence": confidence,
            "context": f"ADX={adx:.1f}, RSI={rsi:.1f}, ATR={atr:.2f}",
            "holding_days": 3 if signal != "HOLD" else 0
        }
    
    def scan_covered_calls(
        self,
        stock_name: str,
        current_price: float,
        own_shares: int,
        iv_percentile: float,
        days_to_expiry: int = 30,
        capital: float = 500000
    ) -> Dict:
        """
        Covered call strategy: own stock + sell call.
        Income generation when stock is range-bound or mildly bullish.
        
        Args:
            stock_name: Stock symbol (e.g., 'INFY')
            current_price: Current stock price
            own_shares: Shares already owned (from stock portfolio)
            iv_percentile: IV percentile (0-100)
            days_to_expiry: Days until option expiry (default 30)
            capital: Available capital
            
        Returns:
            Dict with covered call recommendation
        """
        signal = "HOLD"
        recommendation = ""
        shares_to_own = 0
        call_strike = 0
        premium_income = 0
        confidence = 0
        
        # Covered calls work best in high IV environment (>60)
        if iv_percentile > 60 and days_to_expiry > 20:
            # Determine call strike: ATM or slightly OTM
            if iv_percentile > 75:
                # Very high IV: sell OTM call (more upside, less premium)
                call_strike = round(current_price * 1.03, 2)  # 3% OTM
                premium_estimate = current_price * 0.02  # 2% of stock price
                confidence = 70
            else:
                # High IV: sell ATM call (max premium, limited upside)
                call_strike = current_price
                premium_estimate = current_price * 0.025  # 2.5% of stock price
                confidence = 80
            
            # Calculate shares to own if not already owning
            if own_shares == 0:
                position_capital = capital * 0.5  # Use 50% of capital for covered calls
                shares_to_own = int(position_capital / current_price)
            else:
                shares_to_own = own_shares
            
            premium_income = round(premium_estimate * shares_to_own, 2)
            
            # Annualized return
            days_held = days_to_expiry
            annual_return = (premium_income / (shares_to_own * current_price)) * (365 / days_held)
            
            signal = "SELL_CALL"
            recommendation = (
                f"Own {shares_to_own} shares @ ₹{current_price}, "
                f"Sell {shares_to_own} calls @ ₹{call_strike}. "
                f"Income: ₹{premium_income:.0f}, "
                f"Annualized: {annual_return*100:.1f}%"
            )
        
        return {
            "strategy": "Covered_Calls",
            "stock": stock_name,
            "signal": signal,
            "shares_to_own": shares_to_own,
            "call_strike": call_strike,
            "shares_to_sell_call": shares_to_own,
            "premium_income": premium_income,
            "confidence": confidence,
            "recommendation": recommendation,
            "days_to_expiry": days_to_expiry,
            "max_profit": premium_income + (shares_to_own * (call_strike - current_price))
        }
    
    def scan_protective_puts(
        self,
        stock_name: str,
        current_price: float,
        own_shares: int,
        iv_percentile: float,
        portfolio_value: float,
        max_loss_pct: float = 0.10,
        days_to_expiry: int = 30
    ) -> Dict:
        """
        Protective put strategy: own stock + buy put.
        Downside protection when concerned about fall.
        
        Args:
            stock_name: Stock symbol
            current_price: Current stock price
            own_shares: Shares already owned
            iv_percentile: IV percentile (0-100)
            portfolio_value: Total portfolio value (for risk calculation)
            max_loss_pct: Maximum acceptable loss (default 10%)
            days_to_expiry: Days until expiry
            
        Returns:
            Dict with protective put recommendation
        """
        signal = "HOLD"
        recommendation = ""
        put_strike = 0
        put_cost = 0
        confidence = 0
        
        if own_shares > 0:
            # Calculate acceptable loss amount
            max_loss_amount = portfolio_value * max_loss_pct
            max_loss_per_share = max_loss_amount / own_shares
            put_strike = round(current_price - max_loss_per_share, 2)
            
            # Put cost estimation based on IV
            if iv_percentile > 60:
                put_cost_pct = 0.03  # 3% of stock price (high IV)
                confidence = 60
            elif iv_percentile > 40:
                put_cost_pct = 0.025  # 2.5% of stock price (normal IV)
                confidence = 70
            else:
                put_cost_pct = 0.015  # 1.5% of stock price (low IV)
                confidence = 50
            
            put_cost = round(put_cost_pct * current_price * own_shares, 2)
            total_cost = put_cost
            
            # Net protected value
            protected_value = (put_strike * own_shares)
            current_value = (current_price * own_shares)
            insurance_cost_pct = (put_cost / current_value) * 100
            
            signal = "BUY_PUT"
            recommendation = (
                f"Own {own_shares} shares @ ₹{current_price}. "
                f"Buy {own_shares} puts @ ₹{put_strike}. "
                f"Insurance cost: ₹{total_cost:.0f} ({insurance_cost_pct:.2f}%). "
                f"Protected floor: ₹{put_strike} per share."
            )
        
        return {
            "strategy": "Protective_Puts",
            "stock": stock_name,
            "signal": signal,
            "shares_owned": own_shares,
            "put_strike": put_strike,
            "shares_to_buy_put": own_shares,
            "put_cost": put_cost,
            "confidence": confidence,
            "recommendation": recommendation,
            "days_to_expiry": days_to_expiry,
            "protected_downside": own_shares * put_strike
        }
    
    def scan_collars(
        self,
        stock_name: str,
        current_price: float,
        own_shares: int,
        iv_percentile: float,
        max_loss_pct: float = 0.15,
        days_to_expiry: int = 30
    ) -> Dict:
        """
        Collar strategy: own stock + buy put + sell call.
        Cost-reduced downside protection.
        
        Args:
            stock_name: Stock symbol
            current_price: Current stock price
            own_shares: Shares already owned
            iv_percentile: IV percentile (0-100)
            max_loss_pct: Maximum acceptable loss (default 15%)
            days_to_expiry: Days until expiry
            
        Returns:
            Dict with collar recommendation
        """
        signal = "HOLD"
        recommendation = ""
        put_strike = 0
        call_strike = 0
        net_cost = 0
        confidence = 0
        
        if own_shares > 0:
            # Put strike: downside protection
            acceptable_loss = current_price * max_loss_pct
            put_strike = round(current_price - acceptable_loss, 2)
            
            # Call strike: capped upside
            upside_cap = 0.05  # 5% upside cap (typical)
            call_strike = round(current_price * (1 + upside_cap), 2)
            
            # Cost estimation
            if iv_percentile > 60:
                put_cost_pct = 0.025  # 2.5% of price
                call_premium_pct = 0.020  # 2.0% of price
            else:
                put_cost_pct = 0.015
                call_premium_pct = 0.010
            
            put_cost = put_cost_pct * current_price * own_shares
            call_premium = call_premium_pct * current_price * own_shares
            net_cost = round(put_cost - call_premium, 2)
            
            # Calculate collar range
            loss_at_put = own_shares * (put_strike - current_price)
            gain_at_call = own_shares * (call_strike - current_price)
            
            if net_cost > 0:
                confidence = 75  # Paying for collar
            else:
                confidence = 85  # Zero-cost or credit collar
            
            signal = "SETUP_COLLAR"
            recommendation = (
                f"Own {own_shares} shares @ ₹{current_price}. "
                f"Buy {own_shares} puts @ ₹{put_strike}, "
                f"Sell {own_shares} calls @ ₹{call_strike}. "
                f"Net cost: ₹{abs(net_cost):.0f} ({'cost' if net_cost > 0 else 'credit'}). "
                f"Downside floor: ₹{put_strike}, Upside cap: ₹{call_strike}."
            )
        
        return {
            "strategy": "Collar",
            "stock": stock_name,
            "signal": signal,
            "shares_owned": own_shares,
            "put_strike": put_strike,
            "call_strike": call_strike,
            "net_cost": net_cost,
            "confidence": confidence,
            "recommendation": recommendation,
            "days_to_expiry": days_to_expiry,
            "max_profit": own_shares * (call_strike - current_price) - abs(net_cost),
            "max_loss": own_shares * (current_price - put_strike) + abs(net_cost)
        }
    
    def scan_vertical_spreads(
        self,
        stock_name: str,
        current_price: float,
        iv_percentile: float,
        strategy_type: str = "BULL_CALL",
        days_to_expiry: int = 30
    ) -> Dict:
        """
        Vertical spreads: Bull Call, Bear Call, Bull Put, Bear Put.
        Limited risk/reward with defined strikes.
        
        Args:
            stock_name: Stock symbol
            current_price: Current stock price
            iv_percentile: IV percentile
            strategy_type: BULL_CALL, BEAR_CALL, BULL_PUT, BEAR_PUT
            days_to_expiry: Days until expiry
            
        Returns:
            Dict with vertical spread details
        """
        signal = "HOLD"
        recommendation = ""
        long_strike = 0
        short_strike = 0
        max_profit = 0
        max_loss = 0
        confidence = 0
        
        # Strike width (typically 100 rupees for mid-cap, 200 for large-cap)
        strike_width = 100
        
        if strategy_type == "BULL_CALL" and iv_percentile > 50:
            # Bullish: buy ATM call, sell OTM call
            long_strike = round(current_price / 100) * 100  # Round to nearest 100
            short_strike = long_strike + strike_width
            max_profit = round((short_strike - long_strike) * 100, 2)  # 100 shares
            max_loss = 0  # Limited to premium paid (not calculated here)
            signal = "SETUP_BULL_CALL"
            confidence = 75
            recommendation = (
                f"Buy {stock_name} {long_strike} Call @ ₹X, "
                f"Sell {short_strike} Call @ ₹Y. "
                f"Max profit: ₹{max_profit}, Risk: Premium paid."
            )
            
        elif strategy_type == "BEAR_CALL" and iv_percentile > 60:
            # Bearish: sell ATM call, buy OTM call
            short_strike = round(current_price / 100) * 100
            long_strike = short_strike + strike_width
            max_profit = round((short_strike - long_strike + current_price) * 100, 2)
            max_loss = round(strike_width * 100, 2)
            signal = "SETUP_BEAR_CALL"
            confidence = 70
            recommendation = (
                f"Sell {stock_name} {short_strike} Call @ ₹X, "
                f"Buy {long_strike} Call @ ₹Y. "
                f"Max profit: ₹{max_profit}, Max loss: ₹{max_loss}."
            )
            
        elif strategy_type == "BULL_PUT":
            # Bullish bias: sell OTM put, buy deeper OTM put
            short_strike = round(current_price / 100) * 100 - strike_width
            long_strike = short_strike - strike_width
            max_profit = round((short_strike - long_strike) * 100, 2)
            max_loss = round(strike_width * 100, 2)
            signal = "SETUP_BULL_PUT"
            confidence = 70 if iv_percentile > 60 else 60
            recommendation = (
                f"Sell {stock_name} {short_strike} Put @ ₹X, "
                f"Buy {long_strike} Put @ ₹Y. "
                f"Max profit: ₹{max_profit}, Max loss: ₹{max_loss}."
            )
            
        elif strategy_type == "BEAR_PUT":
            # Bearish: sell put, buy deeper put
            short_strike = round(current_price / 100) * 100
            long_strike = short_strike - strike_width
            max_profit = round((short_strike - long_strike) * 100, 2)
            max_loss = round(strike_width * 100, 2)
            signal = "SETUP_BEAR_PUT"
            confidence = 65 if iv_percentile > 50 else 55
            recommendation = (
                f"Sell {stock_name} {short_strike} Put @ ₹X, "
                f"Buy {long_strike} Put @ ₹Y. "
                f"Max profit: ₹{max_profit}, Max loss: ₹{max_loss}."
            )
        
        return {
            "strategy": f"Vertical_Spread_{strategy_type}",
            "stock": stock_name,
            "signal": signal,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "confidence": confidence,
            "recommendation": recommendation,
            "days_to_expiry": days_to_expiry,
            "risk_reward_ratio": max_profit / max_loss if max_loss > 0 else float('inf')
        }
    
    def get_stock_position_sizing(
        self,
        capital: float,
        risk_per_trade_pct: float = 0.02,
        atr: float = 10.0,
        stop_multiple: float = 2.0,
        current_price: float = 100.0
    ) -> Dict:
        """
        Calculate position sizes for different leverage levels.
        
        Args:
            capital: Available capital
            risk_per_trade_pct: Risk per trade as % of capital (default 2%)
            atr: Average True Range
            stop_multiple: ATR multiple for stop loss (default 2.0)
            current_price: Current price
            
        Returns:
            Dict with position sizes for 1x, 1.5x, 2x leverage
        """
        max_loss = capital * risk_per_trade_pct
        stop_distance = atr * stop_multiple
        
        # Calculate shares for each leverage
        shares_1x = int((capital * 1.0) / current_price)
        shares_1_5x = int((capital * 1.5) / current_price)
        shares_2x = int((capital * 2.0) / current_price)
        
        return {
            "risk_per_trade": max_loss,
            "stop_distance": round(stop_distance, 2),
            "position_1x": {
                "shares": shares_1x,
                "capital_used": shares_1x * current_price,
                "max_loss": shares_1x * stop_distance,
                "leverage": 1.0
            },
            "position_1_5x": {
                "shares": shares_1_5x,
                "capital_used": shares_1_5x * current_price,
                "max_loss": shares_1_5x * stop_distance,
                "leverage": 1.5
            },
            "position_2x": {
                "shares": shares_2x,
                "capital_used": shares_2x * current_price,
                "max_loss": shares_2x * stop_distance,
                "leverage": 2.0
            }
        }
    
    def scan_all_stock_strategies(
        self,
        stock_name: str,
        current_price: float,
        rsi: float,
        adx: float,
        atr: float,
        volume: float,
        avg_volume: float,
        iv_percentile: float,
        own_shares: int = 0,
        capital: float = 500000,
        days_to_expiry: int = 30
    ) -> pd.DataFrame:
        """
        Run all stock strategies and return ranked signals.
        
        Args:
            stock_name: Stock symbol
            current_price: Current stock price
            rsi: RSI(14)
            adx: ADX(14)
            atr: Average True Range
            volume: Current volume
            avg_volume: Average volume (20-day)
            iv_percentile: IV percentile (0-100)
            own_shares: Shares already owned (for hedging strategies)
            capital: Available capital
            days_to_expiry: Days to option expiry
            
        Returns:
            DataFrame with all strategies ranked by confidence
        """
        strategies = []
        
        # 1. Futures momentum
        futures = self.scan_stock_futures_momentum(
            current_price, rsi, adx, atr, volume, avg_volume, capital
        )
        if futures["signal"] != "HOLD":
            strategies.append(futures)
        
        # 2. Covered calls (income generation)
        covered = self.scan_covered_calls(
            stock_name, current_price, own_shares, iv_percentile, days_to_expiry, capital
        )
        if covered["signal"] != "HOLD":
            strategies.append(covered)
        
        # 3. Protective puts (downside protection)
        protective = self.scan_protective_puts(
            stock_name, current_price, own_shares, iv_percentile, capital * 2, days_to_expiry=days_to_expiry
        )
        if protective["signal"] != "HOLD":
            strategies.append(protective)
        
        # 4. Collars (cost-reduced protection)
        collar = self.scan_collars(
            stock_name, current_price, own_shares, iv_percentile, days_to_expiry=days_to_expiry
        )
        if collar["signal"] != "HOLD":
            strategies.append(collar)
        
        # 5. Vertical spreads
        for spread_type in ["BULL_CALL", "BEAR_PUT", "BULL_PUT"]:
            spread = self.scan_vertical_spreads(
                stock_name, current_price, iv_percentile, spread_type, days_to_expiry
            )
            if spread["signal"] != "HOLD":
                strategies.append(spread)
        
        # Convert to DataFrame
        if strategies:
            df = pd.DataFrame(strategies)
            return df.sort_values("confidence", ascending=False)
        
        return pd.DataFrame()


def scan_stock_derivatives(
    stock_name: str,
    current_price: float,
    rsi: float,
    adx: float,
    atr: float,
    volume: float = 1000000,
    avg_volume: float = 1000000,
    iv_percentile: float = 50.0,
    own_shares: int = 0,
    capital: float = 500000,
    days_to_expiry: int = 30
) -> pd.DataFrame:
    """
    Standalone function to scan all stock derivative strategies.
    
    Example:
        signals = scan_stock_derivatives(
            stock_name='INFY',
            current_price=2800,
            rsi=65,
            adx=28,
            atr=45,
            iv_percentile=75
        )
    """
    scanner = StockDerivativesScanner(max_leverage=2.0)
    return scanner.scan_all_stock_strategies(
        stock_name, current_price, rsi, adx, atr, volume, avg_volume,
        iv_percentile, own_shares, capital, days_to_expiry
    )


if __name__ == "__main__":
    # Example: Scan a stock
    example_signals = scan_stock_derivatives(
        stock_name="INFY",
        current_price=2800,
        rsi=65,
        adx=28,
        atr=45,
        iv_percentile=75,
        own_shares=100,
        capital=500000
    )
    
    if not example_signals.empty:
        print("Stock Derivative Signals (INFY):")
        print(example_signals[["strategy", "signal", "confidence", "recommendation"]].to_string())
    else:
        print("No signals generated")
