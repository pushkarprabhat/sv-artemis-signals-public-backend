# core/options_spreads.py — Options Spread Strategy Builder
# Builds: Complex options strategies (spreads, straddles, strangles, condors)
# Features: Risk/reward calculation, Greeks aggregation, probability analysis
# Supports: Bull/bear spreads, iron condors, covered calls, protective puts

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from config import BASE_DIR
from utils.logger import logger
from .greeks_calculator import GreeksCalculator, calculate_option_price_bs


class OptionsSpreadsBuilder:
    """Builds and analyzes options spreads"""
    
    def __init__(self, risk_free_rate=0.06):
        self.greeks_calc = GreeksCalculator(risk_free_rate=risk_free_rate)
        self.base_dir = BASE_DIR / 'data' / 'options_spreads'
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_leg_price(self, spot, strike, time_to_expiry, iv, option_type='call'):
        """Calculate theoretical option price for a leg
        
        Args:
            spot: Spot price
            strike: Strike price
            time_to_expiry: Time in years
            iv: Implied volatility
            option_type: 'call' or 'put'
        
        Returns:
            float: Option price
        """
        return calculate_option_price_bs(spot, strike, time_to_expiry, 0.06, iv, option_type)
    
    def bull_call_spread(self, spot, buy_strike, sell_strike, expiry_date, buy_iv=0.20, sell_iv=0.20):
        """Build a bull call spread
        
        Strategy: Buy lower strike call, sell higher strike call
        Max Profit: sell_strike - buy_strike - net_debit
        Max Loss: net_debit
        Breakeven: buy_strike + net_debit
        
        Args:
            spot: Current spot price
            buy_strike: Lower strike (buy call)
            sell_strike: Higher strike (sell call)
            expiry_date: Expiry date
            buy_iv: IV for buy leg
            sell_iv: IV for sell leg
        
        Returns:
            dict: Spread details and risk metrics
        """
        time_to_expiry = (expiry_date - dt.datetime.now()).days / 365
        
        # Calculate prices
        buy_price = self._calculate_leg_price(spot, buy_strike, time_to_expiry, buy_iv, 'call')
        sell_price = self._calculate_leg_price(spot, sell_strike, time_to_expiry, sell_iv, 'call')
        
        # Net debit/credit
        net_cost = buy_price - sell_price
        
        # Risk/reward
        max_profit = (sell_strike - buy_strike) - net_cost
        max_loss = net_cost
        breakeven = buy_strike + net_cost
        
        # Probability of profit (simplified: at-the-money assumption)
        prob_profit = 50 if spot <= breakeven else 100  # Simplified
        
        # Greeks aggregation
        greeks_long = self.greeks_calc.calculate_all_greeks(spot, buy_strike, time_to_expiry, buy_iv, 'call')
        greeks_short = self.greeks_calc.calculate_all_greeks(spot, sell_strike, time_to_expiry, sell_iv, 'call')
        
        return {
            'strategy': 'BULL_CALL_SPREAD',
            'spot_price': spot,
            'buy_strike': buy_strike,
            'sell_strike': sell_strike,
            'expiry': expiry_date,
            'legs': [
                {'type': 'CALL', 'side': 'LONG', 'strike': buy_strike, 'price': buy_price, 'iv': buy_iv},
                {'type': 'CALL', 'side': 'SHORT', 'strike': sell_strike, 'price': sell_price, 'iv': sell_iv}
            ],
            'net_debit': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_ratio': max_profit / abs(max_loss) if max_loss != 0 else 0,
            'breakeven': breakeven,
            'probability_of_profit': prob_profit,
            'spread_delta': greeks_long['delta'] - greeks_short['delta'],
            'spread_gamma': greeks_long['gamma'] - greeks_short['gamma'],
            'spread_theta': greeks_long['theta'] - greeks_short['theta'],
            'spread_vega': greeks_long['vega'] - greeks_short['vega']
        }
    
    def bear_call_spread(self, spot, buy_strike, sell_strike, expiry_date, buy_iv=0.20, sell_iv=0.20):
        """Build a bear call spread
        
        Strategy: Sell lower strike call, buy higher strike call
        Max Profit: credit - (sell_strike - buy_strike)
        Max Loss: (sell_strike - buy_strike) - credit
        
        Args:
            spot: Current spot price
            buy_strike: Higher strike (buy call for protection)
            sell_strike: Lower strike (sell call for credit)
            expiry_date: Expiry date
            buy_iv: IV for buy leg
            sell_iv: IV for sell leg
        
        Returns:
            dict: Spread details
        """
        time_to_expiry = (expiry_date - dt.datetime.now()).days / 365
        
        buy_price = self._calculate_leg_price(spot, buy_strike, time_to_expiry, buy_iv, 'call')
        sell_price = self._calculate_leg_price(spot, sell_strike, time_to_expiry, sell_iv, 'call')
        
        net_credit = sell_price - buy_price
        
        max_profit = net_credit
        max_loss = (buy_strike - sell_strike) - net_credit
        breakeven = sell_strike + net_credit
        
        greeks_long = self.greeks_calc.calculate_all_greeks(spot, buy_strike, time_to_expiry, buy_iv, 'call')
        greeks_short = self.greeks_calc.calculate_all_greeks(spot, sell_strike, time_to_expiry, sell_iv, 'call')
        
        return {
            'strategy': 'BEAR_CALL_SPREAD',
            'spot_price': spot,
            'sell_strike': sell_strike,
            'buy_strike': buy_strike,
            'expiry': expiry_date,
            'legs': [
                {'type': 'CALL', 'side': 'SHORT', 'strike': sell_strike, 'price': sell_price, 'iv': sell_iv},
                {'type': 'CALL', 'side': 'LONG', 'strike': buy_strike, 'price': buy_price, 'iv': buy_iv}
            ],
            'net_credit': net_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_ratio': max_profit / abs(max_loss) if max_loss != 0 else 0,
            'breakeven': breakeven,
            'spread_delta': -greeks_short['delta'] + greeks_long['delta'],
            'spread_gamma': -greeks_short['gamma'] + greeks_long['gamma'],
            'spread_theta': -greeks_short['theta'] + greeks_long['theta'],
            'spread_vega': -greeks_short['vega'] + greeks_long['vega']
        }
    
    def iron_condor(self, spot, lower_put_strike, upper_put_strike,
                    lower_call_strike, upper_call_strike, expiry_date, iv=0.20):
        """Build an iron condor
        
        Strategy: Bull put spread + bear call spread
        - Sell put at lower_put_strike
        - Buy put at upper_put_strike
        - Sell call at lower_call_strike
        - Buy call at upper_call_strike
        
        Args:
            spot: Current spot price
            lower_put_strike: Lower put strike
            upper_put_strike: Upper put strike
            lower_call_strike: Lower call strike
            upper_call_strike: Upper call strike
            expiry_date: Expiry date
            iv: Implied volatility (assumed same for all)
        
        Returns:
            dict: Spread details
        """
        time_to_expiry = (expiry_date - dt.datetime.now()).days / 365
        
        # Put spread (bull put)
        sell_put_price = self._calculate_leg_price(spot, lower_put_strike, time_to_expiry, iv, 'put')
        buy_put_price = self._calculate_leg_price(spot, upper_put_strike, time_to_expiry, iv, 'put')
        put_credit = sell_put_price - buy_put_price
        
        # Call spread (bear call)
        sell_call_price = self._calculate_leg_price(spot, lower_call_strike, time_to_expiry, iv, 'call')
        buy_call_price = self._calculate_leg_price(spot, upper_call_strike, time_to_expiry, iv, 'call')
        call_credit = sell_call_price - buy_call_price
        
        # Total
        net_credit = put_credit + call_credit
        max_width = (upper_call_strike - lower_call_strike)
        
        max_profit = net_credit
        max_loss = max_width - net_credit
        
        # Breakeven points
        lower_be = lower_put_strike - net_credit
        upper_be = upper_call_strike + net_credit
        
        return {
            'strategy': 'IRON_CONDOR',
            'spot_price': spot,
            'put_strikes': [lower_put_strike, upper_put_strike],
            'call_strikes': [lower_call_strike, upper_call_strike],
            'expiry': expiry_date,
            'legs': [
                {'type': 'PUT', 'side': 'SHORT', 'strike': lower_put_strike, 'price': sell_put_price},
                {'type': 'PUT', 'side': 'LONG', 'strike': upper_put_strike, 'price': buy_put_price},
                {'type': 'CALL', 'side': 'SHORT', 'strike': lower_call_strike, 'price': sell_call_price},
                {'type': 'CALL', 'side': 'LONG', 'strike': upper_call_strike, 'price': buy_call_price}
            ],
            'net_credit': net_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_ratio': max_profit / abs(max_loss) if max_loss != 0 else 0,
            'lower_breakeven': lower_be,
            'upper_breakeven': upper_be,
            'profit_zone': (lower_be, upper_be),
            'max_width': max_width
        }
    
    def strangle(self, spot, put_strike, call_strike, expiry_date, put_iv=0.20, call_iv=0.20):
        """Build a long strangle (buy out-of-the-money call and put)
        
        Strategy: Buy OTM call and OTM put
        Max Profit: Unlimited (upside) and limited (downside)
        Max Loss: call_price + put_price
        
        Args:
            spot: Current spot price
            put_strike: Put strike (below spot)
            call_strike: Call strike (above spot)
            expiry_date: Expiry date
            put_iv: IV for put
            call_iv: IV for call
        
        Returns:
            dict: Spread details
        """
        time_to_expiry = (expiry_date - dt.datetime.now()).days / 365
        
        put_price = self._calculate_leg_price(spot, put_strike, time_to_expiry, put_iv, 'put')
        call_price = self._calculate_leg_price(spot, call_strike, time_to_expiry, call_iv, 'call')
        
        total_debit = put_price + call_price
        
        # Breakeven points
        lower_be = put_strike - put_price
        upper_be = call_strike + call_price
        
        return {
            'strategy': 'STRANGLE',
            'spot_price': spot,
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expiry': expiry_date,
            'legs': [
                {'type': 'PUT', 'side': 'LONG', 'strike': put_strike, 'price': put_price},
                {'type': 'CALL', 'side': 'LONG', 'strike': call_strike, 'price': call_price}
            ],
            'total_debit': total_debit,
            'max_profit_upside': 'unlimited',
            'max_profit_downside': spot - put_strike - total_debit,
            'max_loss': total_debit,
            'lower_breakeven': lower_be,
            'upper_breakeven': upper_be,
            'profit_zones': [(0, lower_be), (upper_be, float('inf'))]
        }
    
    def straddle(self, spot, strike, expiry_date, iv=0.20):
        """Build a long straddle (buy ATM call and put)
        
        Strategy: Buy ATM call and ATM put
        Used for: High volatility expected
        
        Args:
            spot: Current spot price
            strike: Strike price (at-the-money)
            expiry_date: Expiry date
            iv: Implied volatility
        
        Returns:
            dict: Spread details
        """
        time_to_expiry = (expiry_date - dt.datetime.now()).days / 365
        
        put_price = self._calculate_leg_price(spot, strike, time_to_expiry, iv, 'put')
        call_price = self._calculate_leg_price(spot, strike, time_to_expiry, iv, 'call')
        
        total_debit = put_price + call_price
        
        return {
            'strategy': 'STRADDLE',
            'spot_price': spot,
            'strike': strike,
            'expiry': expiry_date,
            'legs': [
                {'type': 'PUT', 'side': 'LONG', 'strike': strike, 'price': put_price},
                {'type': 'CALL', 'side': 'LONG', 'strike': strike, 'price': call_price}
            ],
            'total_debit': total_debit,
            'max_loss': total_debit,
            'lower_breakeven': strike - total_debit,
            'upper_breakeven': strike + total_debit,
            'max_profit_upside': 'unlimited',
            'max_profit_downside': 'unlimited'
        }
    
    def save_spread_analysis(self, spread_data, symbol):
        """Save spread analysis to file
        
        Args:
            spread_data: Spread dictionary
            symbol: Symbol name
        
        Returns:
            Path: Where analysis was saved
        """
        strategy = spread_data.get('strategy', 'UNKNOWN')
        file_path = self.base_dir / f"{symbol}_{strategy}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        df = pd.DataFrame([spread_data])
        df.to_parquet(file_path, index=False)
        
        logger.info(f"[SPREADS] Saved {strategy} analysis for {symbol}")
        return file_path
    
    def compare_spreads(self, spreads_list):
        """Compare multiple spread strategies
        
        Args:
            spreads_list: List of spread dictionaries
        
        Returns:
            DataFrame: Comparison with columns [strategy, max_profit, max_loss, ratio, cost/credit]
        """
        comparison_data = []
        
        for spread in spreads_list:
            comparison_data.append({
                'strategy': spread.get('strategy'),
                'max_profit': spread.get('max_profit'),
                'max_loss': spread.get('max_loss'),
                'profit_ratio': spread.get('profit_ratio'),
                'net_debit': spread.get('net_debit', 0),
                'net_credit': spread.get('net_credit', 0),
                'delta': spread.get('spread_delta', 0),
                'theta': spread.get('spread_theta', 0),
                'vega': spread.get('spread_vega', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('profit_ratio', ascending=False)
