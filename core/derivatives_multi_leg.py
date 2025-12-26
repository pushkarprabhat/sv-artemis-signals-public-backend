"""
Multi-Leg Derivative Strategies
Implements: Iron Condor, Butterfly, Calendar, Diagonal, Ratio Spreads, etc.

Author: Trading System
Date: 2025-12-23
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiLegDerivativesScanner:
    """Scanner for multi-leg derivative strategies"""
    
    def __init__(self, symbols_list: List[str] = None):
        """Initialize with list of symbols to scan"""
        # Focus on NIFTY, BANKNIFTY, and top 50 stocks
        self.symbols = symbols_list or [
            'NIFTY', 'BANKNIFTY',
            'INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'SBIN',
            'WIPRO', 'BAJAJFINSV', 'MARUTI', 'NESTLEIND', 'HINDUNILVR',
            'ITC', 'LT', 'AXISBANK', 'HDFC', 'KOTAK', 'SUNPHARMA',
            'ASIANPAINT', 'BAJAJHOLDING', 'BHARTIARTL', 'BPCL', 'CIPLA',
            'DIVISLAB', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HEROMOTOCO',
            'HINDALCO', 'INDIGO', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LTIM',
            'M&M', 'NTPC', 'ONGC', 'POWERGRID', 'SBICARD', 'SBILIFE',
            'SIEMENS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'ULTRACEMCO',
            'YESBANK', 'ZEEL'
        ]
    
    # ==================== IRON CONDOR ====================
    def scan_iron_condor(self, symbol: str, atm_price: float, 
                         current_iv_percentile: float = 50,
                         days_to_expiry: int = 30) -> Optional[Dict]:
        """
        Iron Condor: Sell call spread + sell put spread
        
        Setup:
        - Sell call at ATM + Buy call at +2% OTM (4 point spread assumed)
        - Sell put at ATM + Buy put at -2% OTM (4 point spread assumed)
        
        Profit: Premium collected from both spreads
        Max Loss: Width of spreads minus premium
        Suitable: Neutral to slightly bullish market, low IV
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Iron Condor best sold when IV is high (above 60th percentile)
        if current_iv_percentile < 40:
            return None
        
        # Setup details (assuming 2% OTM strikes)
        call_sell_strike = atm_price * 1.00  # At money
        call_buy_strike = atm_price * 1.02   # 2% OTM
        put_sell_strike = atm_price * 1.00   # At money
        put_buy_strike = atm_price * 0.98    # 2% OTM
        
        # Expected premium (rough estimates based on IV)
        call_sell_premium = (atm_price * 0.02 * (current_iv_percentile / 100))  # ~2% of price
        call_buy_premium = (atm_price * 0.01 * (current_iv_percentile / 100))   # ~1% of price
        put_sell_premium = (atm_price * 0.02 * (current_iv_percentile / 100))   # ~2% of price
        put_buy_premium = (atm_price * 0.01 * (current_iv_percentile / 100))    # ~1% of price
        
        # Net credit
        net_credit = (call_sell_premium - call_buy_premium) + (put_sell_premium - put_buy_premium)
        max_loss = (call_buy_strike - call_sell_strike) - net_credit
        
        return {
            'strategy': 'Iron Condor',
            'symbol': symbol,
            'signal_type': 'SELL',
            'position_type': 'SHORT_CALL_SPREAD + SHORT_PUT_SPREAD',
            'atm_price': atm_price,
            'call_sell_strike': round(call_sell_strike, 2),
            'call_buy_strike': round(call_buy_strike, 2),
            'put_sell_strike': round(put_sell_strike, 2),
            'put_buy_strike': round(put_buy_strike, 2),
            'net_credit': round(net_credit, 2),
            'max_profit': round(net_credit, 2),
            'max_loss': round(max(0, max_loss), 2),
            'breakeven_call': round(call_sell_strike + net_credit, 2),
            'breakeven_put': round(put_sell_strike - net_credit, 2),
            'probability_profit': 65,  # Typically 60-70%
            'capital_required': round(max(0, max_loss), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': min(100, (current_iv_percentile / 40))  # Higher IV = better setup
        }
    
    # ==================== BUTTERFLY SPREAD ====================
    def scan_butterfly_spread(self, symbol: str, atm_price: float,
                              current_iv_percentile: float = 50,
                              days_to_expiry: int = 30,
                              spread_type: str = 'call') -> Optional[Dict]:
        """
        Butterfly Spread: Long 1 ATM + Short 2 OTM + Long 1 farther OTM
        
        Setup (Call Butterfly):
        - Buy 1 call at 50-0 strike (ATM)
        - Sell 2 calls at 100-0 strike (+1% OTM)
        - Buy 1 call at 100-0 strike (+2% OTM)
        
        Profit: Spread width - premium paid (max when price stays at middle strike)
        Max Loss: Spread premium
        Suitable: Low volatility, small profit moves expected
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            spread_type: 'call' or 'put'
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Butterfly best when IV is low (below 50th percentile)
        if current_iv_percentile > 60:
            return None
        
        # Strike setup
        inner_strike = atm_price * 1.01  # Inner spread strike (1% OTM)
        outer_strike = atm_price * 1.02  # Outer spread strike (2% OTM)
        
        # Premium estimates
        long_outer_premium = atm_price * 0.01 * (current_iv_percentile / 100)
        short_inner_premium = atm_price * 0.015 * (current_iv_percentile / 100)
        long_atm_premium = atm_price * 0.025 * (current_iv_percentile / 100)
        
        # Net debit (paid upfront)
        net_debit = long_atm_premium + long_outer_premium - (2 * short_inner_premium)
        spread_width = inner_strike - atm_price
        max_profit = spread_width - net_debit
        
        return {
            'strategy': f'Butterfly Spread ({spread_type.upper()})',
            'symbol': symbol,
            'signal_type': 'BUY',
            'position_type': f'LONG {spread_type.upper()} + SHORT 2x {spread_type.upper()} + LONG {spread_type.upper()}',
            'atm_price': atm_price,
            'long_strike': round(atm_price, 2),
            'short_strike': round(inner_strike, 2),
            'outer_strike': round(outer_strike, 2),
            'net_debit': round(max(0, net_debit), 2),
            'max_profit': round(max(0, max_profit), 2),
            'max_loss': round(max(0, net_debit), 2),
            'breakeven_lower': round(atm_price + net_debit, 2),
            'breakeven_upper': round(inner_strike - (net_debit / 2), 2),
            'profit_zone': f'{round(atm_price, 2)} to {round(inner_strike, 2)}',
            'probability_profit': 50,  # 50/50 directional bet
            'capital_required': round(max(0, net_debit), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': min(100, ((100 - current_iv_percentile) / 40))  # Lower IV = better
        }
    
    # ==================== DIAGONAL SPREAD ====================
    def scan_diagonal_spread(self, symbol: str, atm_price: float,
                            current_iv_percentile: float = 50,
                            days_to_expiry: int = 30,
                            spread_type: str = 'call') -> Optional[Dict]:
        """
        Diagonal Spread: Sell near-term ATM, Buy far-term OTM
        
        Combines time decay (near-term) + directional bias (far-term OTM)
        Profit from: Theta decay + limited downside from OTM long option
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            spread_type: 'call' or 'put'
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Sell near-term ATM, Buy far-term OTM
        near_strike = atm_price
        far_strike = atm_price * (1.02 if spread_type == 'call' else 0.98)
        
        # Premium estimates (diagonal = different expirations)
        near_sell = atm_price * 0.025 * (current_iv_percentile / 100)
        far_buy = atm_price * 0.015 * (current_iv_percentile / 100)
        
        net_debit = far_buy - near_sell
        max_profit = near_sell if net_debit < 0 else near_sell - net_debit
        
        return {
            'strategy': f'Diagonal Spread ({spread_type.upper()})',
            'symbol': symbol,
            'signal_type': 'SELL',
            'position_type': f'SELL near {spread_type.upper()} + BUY far {spread_type.upper()}',
            'atm_price': atm_price,
            'near_strike': round(near_strike, 2),
            'far_strike': round(far_strike, 2),
            'near_days': days_to_expiry,
            'far_days': days_to_expiry + 30,
            'sell_premium': round(near_sell, 2),
            'buy_premium': round(far_buy, 2),
            'net_debit': round(abs(net_debit), 2),
            'max_profit': round(max(0, max_profit), 2),
            'max_loss': round(far_strike - near_strike if net_debit > 0 else abs(net_debit), 2),
            'direction_bias': 'Bullish' if spread_type == 'call' else 'Bearish',
            'theta_benefit': 'Daily decay from short near-term',
            'capital_required': round(far_strike - near_strike if spread_type == 'call' else abs(net_debit), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': 70
        }
    
    # ==================== CALENDAR SPREAD ====================
    def scan_calendar_spread(self, symbol: str, atm_price: float,
                            current_iv_percentile: float = 50,
                            days_to_expiry: int = 30,
                            spread_type: str = 'call') -> Optional[Dict]:
        """
        Calendar Spread: Sell near-term option, Buy far-term same strike
        
        Profit from: Time decay of near-term option
        Risk: Opposite side if price moves sharply
        Suitable: Expect sideways movement, collect theta decay
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry (for near-term)
            spread_type: 'call' or 'put'
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Both expirations at ATM
        strike = atm_price
        
        # Theta decay highest for 20-40 DTE
        if days_to_expiry < 15 or days_to_expiry > 50:
            return None
        
        # Premium estimates
        near_term_premium = atm_price * 0.025 * (current_iv_percentile / 100) * (days_to_expiry / 30)
        far_term_premium = atm_price * 0.035 * (current_iv_percentile / 100) * ((days_to_expiry + 30) / 60)
        
        # Net debit (pay more for longer term)
        net_debit = far_term_premium - near_term_premium
        
        return {
            'strategy': f'Calendar Spread ({spread_type.upper()})',
            'symbol': symbol,
            'signal_type': 'BUY',
            'position_type': f'SELL near {spread_type.upper()} + BUY far {spread_type.upper()}',
            'atm_price': atm_price,
            'strike': round(strike, 2),
            'near_term_days': days_to_expiry,
            'far_term_days': days_to_expiry + 30,
            'sell_premium': round(near_term_premium, 2),
            'buy_premium': round(far_term_premium, 2),
            'net_debit': round(max(0, net_debit), 2),
            'max_profit': round(near_term_premium, 2),  # Premium collected from near-term
            'max_loss': round(max(0, net_debit), 2),     # Premium paid for far-term
            'breakeven': round(strike, 2),  # Price at strike = max profit
            'profit_window': f'±5% of {atm_price}',
            'theta_decay_daily': round(near_term_premium / days_to_expiry, 4),
            'capital_required': round(max(0, net_debit), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': 70 + (current_iv_percentile / 5)  # Higher IV = better theta
        }
    
    # ==================== RATIO SPREAD ====================
    def scan_ratio_spread(self, symbol: str, atm_price: float,
                         current_iv_percentile: float = 50,
                         days_to_expiry: int = 30,
                         spread_type: str = 'call') -> Optional[Dict]:
        """
        Ratio Spread: Buy 1 ATM option, Sell 2 OTM options (2:1 ratio)
        
        Profit: Premium from selling 2 contracts vs 1 bought
        Risk: Unlimited on uncovered short (use stops)
        Suitable: Expect stock to stay within range, moderate IV
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            spread_type: 'call' or 'put'
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Best with moderate IV
        if current_iv_percentile < 30 or current_iv_percentile > 80:
            return None
        
        # Strikes
        atm_strike = atm_price
        short_strike = atm_price * 1.015  # 1.5% OTM
        
        # Premium estimates
        long_premium = atm_price * 0.025 * (current_iv_percentile / 100)
        short_premium = atm_price * 0.010 * (current_iv_percentile / 100)
        
        # Net credit (sell 2, buy 1)
        net_credit = (2 * short_premium) - long_premium
        
        return {
            'strategy': f'Call Ratio Spread {spread_type.upper()}',
            'symbol': symbol,
            'signal_type': 'SELL',
            'position_type': f'LONG 1x {spread_type.upper()} + SHORT 2x {spread_type.upper()}',
            'atm_price': atm_price,
            'long_strike': round(atm_strike, 2),
            'short_strike': round(short_strike, 2),
            'ratio': '1:2',
            'net_credit': round(max(0, net_credit), 2),
            'max_profit': round(max(0, net_credit), 2),
            'max_loss': 'UNLIMITED (requires stops)',
            'stop_loss_price': round(short_strike + (short_strike - atm_strike) * 1.5, 2),
            'breakeven_upper': round(short_strike + net_credit, 2),
            'profit_zone': f'{round(atm_strike, 2)} to {round(short_strike, 2)}',
            'risk_level': 'MEDIUM-HIGH',
            'capital_required': round(max(0, net_credit * 2), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'warning': 'Short calls exposed to unlimited loss - REQUIRES CLOSE MONITORING',
            'confidence': 60 if 40 <= current_iv_percentile <= 70 else 40
        }
    
    # ==================== VERTICAL SPREAD ====================
    def scan_vertical_spread(self, symbol: str, atm_price: float,
                            current_iv_percentile: float = 50,
                            days_to_expiry: int = 30,
                            direction: str = 'bullish',
                            spread_type: str = 'call') -> Optional[Dict]:
        """
        Vertical Spread: Same expiration, different strikes (Debit or Credit spread)
        
        Bull Call Spread: Buy ATM call, Sell OTM call (Debit spread)
        Bear Call Spread: Sell ATM call, Buy OTM call (Credit spread)
        Bull Put Spread: Sell OTM put, Buy farther OTM put (Credit spread)
        Bear Put Spread: Buy OTM put, Sell farther OTM put (Debit spread)
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            direction: 'bullish' or 'bearish'
            spread_type: 'call' or 'put'
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Strikes based on direction
        if spread_type == 'call':
            if direction == 'bullish':
                long_strike = atm_price
                short_strike = atm_price * 1.015
                signal_type = 'BUY'
                position = f'BULL CALL SPREAD'
            else:
                long_strike = atm_price * 1.015
                short_strike = atm_price
                signal_type = 'SELL'
                position = f'BEAR CALL SPREAD'
        else:  # put
            if direction == 'bullish':
                long_strike = atm_price * 0.985
                short_strike = atm_price
                signal_type = 'SELL'
                position = f'BULL PUT SPREAD'
            else:
                long_strike = atm_price
                short_strike = atm_price * 0.985
                signal_type = 'BUY'
                position = f'BEAR PUT SPREAD'
        
        # Premium estimates
        long_premium = atm_price * 0.020 * (current_iv_percentile / 100)
        short_premium = atm_price * 0.010 * (current_iv_percentile / 100)
        
        # Net debit/credit
        if signal_type == 'BUY':
            net_amount = long_premium - short_premium
            max_profit = abs(long_strike - short_strike) - net_amount
        else:
            net_amount = short_premium - long_premium
            max_profit = net_amount
        
        spread_width = abs(long_strike - short_strike)
        
        return {
            'strategy': f'Vertical Spread ({direction.upper()})',
            'symbol': symbol,
            'signal_type': signal_type,
            'position_type': position,
            'atm_price': atm_price,
            'long_strike': round(long_strike, 2),
            'short_strike': round(short_strike, 2),
            'spread_width': round(spread_width, 2),
            'net_amount': round(abs(net_amount), 2),
            'max_profit': round(max(0, max_profit), 2),
            'max_loss': round(abs(net_amount) if signal_type == 'BUY' else spread_width - abs(net_amount), 2),
            'probability_profit': 60 if direction == 'bullish' else 55,
            'risk_reward_ratio': round(max(0, max_profit) / abs(net_amount) if net_amount != 0 else 1, 2),
            'capital_required': round(spread_width if signal_type == 'SELL' else abs(net_amount), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': 75
        }
    
    # ==================== STRADDLE ADJUSTMENTS ====================
    def scan_straddle_adjustments(self, symbol: str, atm_price: float,
                                 current_iv_percentile: float = 50,
                                 days_to_expiry: int = 30,
                                 entry_price: Optional[float] = None) -> Optional[Dict]:
        """
        Straddle Adjustments: Buy ATM call + Put (or short for income)
        
        Long Straddle: Buy ATM call + put (bet on move in either direction)
        Short Straddle: Sell ATM call + put (collect premium, bet on no move)
        Adjustment: Close losing side when profitable, let winner run
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            entry_price: Original entry price for adjustment logic
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Call + Put at ATM
        strike = atm_price
        
        # Premium estimates
        call_premium = atm_price * 0.025 * (current_iv_percentile / 100)
        put_premium = atm_price * 0.025 * (current_iv_percentile / 100)
        
        if entry_price is None:
            # New entry
            long_straddle_cost = call_premium + put_premium
            short_straddle_credit = call_premium + put_premium
            
            return {
                'strategy': 'Long Straddle Setup',
                'symbol': symbol,
                'signal_type': 'BUY',
                'position_type': 'LONG CALL + LONG PUT (at ATM)',
                'atm_price': atm_price,
                'strike': round(strike, 2),
                'call_premium': round(call_premium, 2),
                'put_premium': round(put_premium, 2),
                'total_cost': round(long_straddle_cost, 2),
                'max_profit': 'UNLIMITED',
                'max_loss': round(long_straddle_cost, 2),
                'breakeven_upper': round(strike + long_straddle_cost, 2),
                'breakeven_lower': round(strike - long_straddle_cost, 2),
                'expected_move': round(long_straddle_cost * 1.2, 2),  # Need 20% move
                'suitable_for': 'High volatility, earnings, economic events',
                'capital_required': round(long_straddle_cost, 2),
                'days_to_expiry': days_to_expiry,
                'iv_percentile': current_iv_percentile,
                'confidence': min(100, (current_iv_percentile / 30))  # High IV = better
            }
        else:
            # Adjustment scenario
            current_profit_loss = atm_price - entry_price
            
            return {
                'strategy': 'Straddle Adjustment',
                'symbol': symbol,
                'signal_type': 'ADJUST',
                'position_type': 'CLOSE LOSING SIDE, LET WINNER RUN',
                'entry_price': entry_price,
                'current_price': atm_price,
                'current_pnl': round(current_profit_loss, 2),
                'adjustment_suggestion': 'Close if up 50-75% of max profit',
                'trailing_stop': round(strike + (long_straddle_cost / 2), 2),
                'expected_move_remaining': round(long_straddle_cost * 0.8, 2),
                'days_to_expiry': days_to_expiry,
                'confidence': 65
            }
    
    # ==================== IRON BUTTERFLY ====================
    def scan_iron_butterfly(self, symbol: str, atm_price: float,
                           current_iv_percentile: float = 50,
                           days_to_expiry: int = 30) -> Optional[Dict]:
        """
        Iron Butterfly: Sell call spread + sell put spread (same strike for both)
        Similar to Iron Condor but tighter (profits in smaller range)
        
        Max Profit: Premium collected
        Max Loss: Spread width minus premium
        Suitable: Very neutral market, lower capital requirement
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        if current_iv_percentile < 50:
            return None
        
        atm_strike = atm_price
        otm_strike = atm_price * 1.01  # 1% OTM
        
        # Premium estimates
        atm_sell = atm_price * 0.03 * (current_iv_percentile / 100)
        otm_buy = atm_price * 0.015 * (current_iv_percentile / 100)
        
        net_credit = (2 * atm_sell) - (2 * otm_buy)
        spread_width = otm_strike - atm_strike
        
        return {
            'strategy': 'Iron Butterfly',
            'symbol': symbol,
            'signal_type': 'SELL',
            'position_type': 'SELL ATM CALL/PUT + BUY OTM CALL/PUT',
            'atm_price': atm_price,
            'atm_strike': round(atm_strike, 2),
            'otm_strike': round(otm_strike, 2),
            'spread_width': round(spread_width, 2),
            'net_credit': round(net_credit, 2),
            'max_profit': round(net_credit, 2),
            'max_loss': round(spread_width - net_credit, 2),
            'profit_zone': f'{round(atm_strike - spread_width, 2)} to {round(atm_strike + spread_width, 2)}',
            'probability_profit': 70,
            'capital_required': round(spread_width - net_credit, 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': 80 if current_iv_percentile > 60 else 65
        }
        sell_strike = atm_price
        buy_strike = atm_price * 1.02  # 2% OTM
        
        # Premium estimates
        sell_premium = atm_price * 0.025 * (current_iv_percentile / 100) * (days_to_expiry / 30)
        buy_premium = atm_price * 0.015 * (current_iv_percentile / 100) * ((days_to_expiry + 30) / 60)
        
        # Net credit or small debit
        net_credit = sell_premium - buy_premium
        
        return {
            'strategy': f'Diagonal Spread ({spread_type.upper()})',
            'symbol': symbol,
            'signal_type': 'SELL' if net_credit > 0 else 'BUY',
            'position_type': f'SELL {spread_type.upper()} at {sell_strike:.2f} + BUY {spread_type.upper()} at {buy_strike:.2f}',
            'atm_price': atm_price,
            'sell_strike': round(sell_strike, 2),
            'buy_strike': round(buy_strike, 2),
            'sell_premium': round(sell_premium, 2),
            'buy_premium': round(buy_premium, 2),
            'net_credit': round(max(0, net_credit), 2),
            'max_profit': round(sell_premium, 2),
            'max_loss': round(abs(buy_strike - sell_strike) - net_credit, 2),
            'breakeven': round(sell_strike + net_credit, 2),
            'directional_bias': 'NEUTRAL_TO_BULLISH' if spread_type == 'call' else 'NEUTRAL_TO_BEARISH',
            'capital_required': round(abs(buy_strike - sell_strike) - net_credit, 2),
            'near_term_days': days_to_expiry,
            'far_term_days': days_to_expiry + 30,
            'iv_percentile': current_iv_percentile,
            'confidence': 65 + (current_iv_percentile / 5)
        }
    
    # ==================== RATIO SPREAD ====================
    def scan_ratio_spread(self, symbol: str, atm_price: float,
                         current_iv_percentile: float = 50,
                         days_to_expiry: int = 30,
                         spread_type: str = 'call') -> Optional[Dict]:
        """
        Ratio Spread: Buy 1 option, Sell 2-3 options at different strike
        
        Example: Buy 1 call at 2000, Sell 2 calls at 2100
        
        Profit: Max when price expires between strikes
        Risk: UNLIMITED if price exceeds short strikes
        Suitable: High probability range-bound trades
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            spread_type: 'call' or 'put'
        
        Returns:
            Dictionary with setup details or None if not suitable
        """
        # Buy 1 ATM, Sell 2 OTM
        buy_strike = atm_price
        sell_strike = atm_price * 1.015  # 1.5% OTM
        
        # Premium estimates
        buy_premium = atm_price * 0.025 * (current_iv_percentile / 100) * (days_to_expiry / 30)
        sell_premium = atm_price * 0.015 * (current_iv_percentile / 100) * (days_to_expiry / 30)
        
        # Net credit (collect more from short than pay for long)
        net_credit = (2 * sell_premium) - buy_premium
        
        return {
            'strategy': f'Ratio Spread ({spread_type.upper()}) 1x2',
            'symbol': symbol,
            'signal_type': 'SELL' if net_credit > 0 else 'BUY',
            'position_type': f'LONG 1x {spread_type.upper()} at {buy_strike:.2f} + SHORT 2x {spread_type.upper()} at {sell_strike:.2f}',
            'atm_price': atm_price,
            'long_strike': round(buy_strike, 2),
            'short_strike': round(sell_strike, 2),
            'long_premium': round(buy_premium, 2),
            'short_premium': round(sell_premium, 2),
            'net_credit': round(max(0, net_credit), 2),
            'max_profit': round(max(0, net_credit), 2),
            'max_loss': 'UNLIMITED',  # If price exceeds short strike
            'safe_zone': f'{round(buy_strike - net_credit, 2)} to {round(sell_strike + net_credit, 2)}',
            'WARNING': 'HIGH RISK - Unlimited loss potential above short strike',
            'capital_required': round(abs(sell_strike - buy_strike), 2),
            'days_to_expiry': days_to_expiry,
            'iv_percentile': current_iv_percentile,
            'confidence': 50  # Risky, lower confidence
        }
    
    # ==================== STRADDLE ADJUSTMENT ====================
    def scan_straddle_adjustment(self, symbol: str, atm_price: float,
                                current_iv_percentile: float = 50,
                                days_to_expiry: int = 30,
                                adjustment_type: str = 'iron_butterfly') -> Optional[Dict]:
        """
        Straddle Adjustment: Dynamic management of long/short straddles
        
        Adjust straddles with Iron Butterfly when delta exceeds thresholds
        Convert to strangle with defined risk
        
        Args:
            symbol: Trading symbol
            atm_price: At-the-money option price
            current_iv_percentile: Current IV rank (0-100)
            days_to_expiry: Days remaining to expiry
            adjustment_type: 'iron_butterfly' or 'strangle_conversion'
        
        Returns:
            Dictionary with adjustment details
        """
        if adjustment_type == 'iron_butterfly':
            # Sell OTM calls/puts to define risk
            adjustment_strike_call = atm_price * 1.02  # 2% OTM
            adjustment_strike_put = atm_price * 0.98   # 2% OTM
            
            return {
                'strategy': 'Straddle Adjustment - Iron Butterfly',
                'symbol': symbol,
                'signal_type': 'SELL',
                'position_type': 'SELL_CALL_SPREAD + SELL_PUT_SPREAD',
                'original_position': 'LONG_STRADDLE',
                'adjustment_action': 'Convert to Iron Butterfly by selling OTM calls/puts',
                'atm_price': atm_price,
                'call_short_strike': round(adjustment_strike_call, 2),
                'put_short_strike': round(adjustment_strike_put, 2),
                'max_loss_defined': True,
                'new_max_loss': round(abs(adjustment_strike_call - atm_price) * 2, 2),
                'breakeven_call': round(adjustment_strike_call * 1.02, 2),
                'breakeven_put': round(adjustment_strike_put * 0.98, 2),
                'days_to_expiry': days_to_expiry,
                'iv_percentile': current_iv_percentile,
                'confidence': 75
            }
        
        else:  # strangle_conversion
            # Convert to strangle (wider strike distance)
            strangle_strike_call = atm_price * 1.03
            strangle_strike_put = atm_price * 0.97
            
            return {
                'strategy': 'Straddle Adjustment - Strangle Conversion',
                'symbol': symbol,
                'signal_type': 'ADJUST',
                'position_type': 'CONVERT_STRADDLE_TO_STRANGLE',
                'original_position': 'LONG_STRADDLE',
                'adjustment_action': 'Widen strikes to reduce losses from small moves',
                'atm_price': atm_price,
                'call_strike': round(strangle_strike_call, 2),
                'put_strike': round(strangle_strike_put, 2),
                'profit_zone': f'{round(strangle_strike_put, 2)} to {round(strangle_strike_call, 2)}',
                'reduced_max_loss': True,
                'days_to_expiry': days_to_expiry,
                'iv_percentile': current_iv_percentile,
                'confidence': 70
            }
    
    # ==================== SCAN ALL MULTI-LEG STRATEGIES ====================
    def scan_all_strategies(self, atm_price: float = 50000, 
                           current_iv_percentile: float = 50,
                           days_to_expiry: int = 30) -> pd.DataFrame:
        """
        Scan all multi-leg strategies for all symbols
        
        Returns highest confidence setups
        """
        results = []
        
        for symbol in self.symbols[:10]:  # Limit to top 10 for speed
            try:
                # Iron Condor
                ic = self.scan_iron_condor(symbol, atm_price, current_iv_percentile, days_to_expiry)
                if ic:
                    results.append(ic)
                
                # Butterfly Spread (Call)
                bf_call = self.scan_butterfly_spread(symbol, atm_price, current_iv_percentile, days_to_expiry, 'call')
                if bf_call:
                    results.append(bf_call)
                
                # Butterfly Spread (Put)
                bf_put = self.scan_butterfly_spread(symbol, atm_price, current_iv_percentile, days_to_expiry, 'put')
                if bf_put:
                    results.append(bf_put)
                
                # Calendar Spread (Call)
                cal_call = self.scan_calendar_spread(symbol, atm_price, current_iv_percentile, days_to_expiry, 'call')
                if cal_call:
                    results.append(cal_call)
                
                # Diagonal Spread (Call)
                diag_call = self.scan_diagonal_spread(symbol, atm_price, current_iv_percentile, days_to_expiry, 'call')
                if diag_call:
                    results.append(diag_call)
                
                # Ratio Spread (Call)
                ratio_call = self.scan_ratio_spread(symbol, atm_price, current_iv_percentile, days_to_expiry, 'call')
                if ratio_call:
                    results.append(ratio_call)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('confidence', ascending=False).head(30)
        
        return df


# ==================== STANDALONE USAGE ====================
def scan_multi_leg_derivatives(atm_price: float = 50000,
                               current_iv_percentile: float = 50,
                               days_to_expiry: int = 30) -> pd.DataFrame:
    """
    Standalone function to scan multi-leg derivatives
    
    Returns top 30 setups sorted by confidence
    """
    scanner = MultiLegDerivativesScanner()
    return scanner.scan_all_strategies(atm_price, current_iv_percentile, days_to_expiry)


def scan_iron_condor(atm_price: float = 50000, current_iv_percentile: float = 50,
                    days_to_expiry: int = 30, **kwargs) -> Dict:
    """Scan for iron condor opportunities"""
    scanner = MultiLegDerivativesScanner()
    result = scanner.scan_iron_condor('NIFTY', atm_price, current_iv_percentile, days_to_expiry)
    return result if result else {'signal': 'NO_SETUP', 'confidence': 0.0}


def scan_butterfly(atm_price: float = 50000, current_iv_percentile: float = 50,
                  days_to_expiry: int = 30, **kwargs) -> Dict:
    """Scan for butterfly spread opportunities"""
    scanner = MultiLegDerivativesScanner()
    result = scanner.scan_butterfly_spread('NIFTY', atm_price, current_iv_percentile, days_to_expiry)
    return result if result else {'signal': 'NO_SETUP', 'confidence': 0.0}


def scan_calendar(atm_price: float = 50000, current_iv_percentile: float = 50,
                 days_to_expiry: int = 30, **kwargs) -> Dict:
    """Scan for calendar spread opportunities"""
    scanner = MultiLegDerivativesScanner()
    result = scanner.scan_calendar_spread('NIFTY', atm_price, current_iv_percentile, days_to_expiry)
    return result if result else {'signal': 'NO_SETUP', 'confidence': 0.0}


if __name__ == "__main__":
    # Example usage
    results = scan_multi_leg_derivatives(atm_price=50000, current_iv_percentile=60, days_to_expiry=30)
    print("\nMulti-Leg Derivative Strategies:")
    print(results[['symbol', 'strategy', 'position_type', 'signal_type', 'max_profit', 'confidence']].to_string())
