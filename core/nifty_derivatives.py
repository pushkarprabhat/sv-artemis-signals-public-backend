"""
NIFTY Derivatives Strategy
Index futures + options spreads + calendar spreads on NIFTY50
Implements delta-neutral and directional strategies with momentum confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NIFTYDerivativesScanner:
    """Scan and implement NIFTY derivatives strategies (futures + options)"""
    
    def __init__(self, initial_leverage: float = 2.0, max_leverage: float = 4.0):
        """
        Initialize NIFTY Derivatives Scanner
        
        Args:
            initial_leverage: Starting leverage (1-4x, default 2x)
            max_leverage: Maximum allowed leverage
        """
        self.leverage_range = (1.0, max_leverage)
        self.initial_leverage = min(initial_leverage, max_leverage)
        self.symbol = 'NIFTY'
        self.min_adx = 20  # Minimum ADX for trend confirmation
    
    def scan_nifty_futures_momentum(
        self,
        current_price: float,
        rsi: float,
        adx: float,
        atr: float,
        daily_volume: float
    ) -> Dict:
        """
        Scan NIFTY futures for momentum-based trading
        
        Args:
            current_price: Current NIFTY level
            rsi: RSI(14) value
            adx: ADX(14) value
            atr: Average True Range for position sizing
            daily_volume: Daily trading volume (contracts)
        
        Returns:
            Dict with signal, position size, stop loss, profit target
        """
        signal = {
            'symbol': 'NIFTY',
            'instrument': 'NIFTY_FUTURES',
            'signal': 'HOLD',
            'confidence': 0,
            'entry_price': current_price,
            'leverage': self.initial_leverage,
            'position_type': None,
            'quantity': 0,
            'stop_loss': 0,
            'profit_target_1': 0,  # 30% of expected move
            'profit_target_2': 0,  # 70% of expected move
            'holding_days': 5,
            'atr': atr
        }
        
        # Trend confirmation
        if adx < self.min_adx:
            signal['confidence'] = 0
            return signal
        
        # Uptrend signal
        if rsi > 55 and adx >= self.min_adx:
            signal['signal'] = 'BUY'
            signal['position_type'] = 'LONG'
            signal['confidence'] = min(100, (rsi - 50) + (adx - 20) * 2)
            
            # Position sizing based on volatility (ATR)
            # Higher ATR = larger stop loss = smaller position
            position_value = 500000 * self.initial_leverage  # ₹5 lakh base capital
            signal['quantity'] = int(position_value / current_price)
            
            # Stop loss: 2 ATR below entry
            signal['stop_loss'] = current_price - (2 * atr)
            
            # Profit targets
            expected_move = atr * 3  # 3 ATR expected move
            signal['profit_target_1'] = current_price + (expected_move * 0.4)
            signal['profit_target_2'] = current_price + (expected_move * 0.8)
        
        # Downtrend signal
        elif rsi < 45 and adx >= self.min_adx:
            signal['signal'] = 'SELL'
            signal['position_type'] = 'SHORT'
            signal['confidence'] = min(100, (50 - rsi) + (adx - 20) * 2)
            
            position_value = 500000 * self.initial_leverage
            signal['quantity'] = int(position_value / current_price)
            
            # Stop loss: 2 ATR above entry
            signal['stop_loss'] = current_price + (2 * atr)
            
            # Profit targets (downside)
            expected_move = atr * 3
            signal['profit_target_1'] = current_price - (expected_move * 0.4)
            signal['profit_target_2'] = current_price - (expected_move * 0.8)
        
        return signal
    
    def scan_nifty_options_spreads(
        self,
        atm_price: float,
        iv_percentile: float,
        days_to_expiry: int,
        bid_ask_spread: Optional[float] = None
    ) -> Dict:
        """
        Scan NIFTY options for spread strategies based on IV
        
        Args:
            atm_price: Current ATM price
            iv_percentile: IV percentile (0-100)
            days_to_expiry: Days until expiry
            bid_ask_spread: Bid-ask spread (optional)
        
        Returns:
            Dict with strategy recommendation and setup
        """
        strategy = {
            'symbol': 'NIFTY',
            'instrument': 'NIFTY_OPTIONS',
            'recommendation': 'SKIP',
            'confidence': 0,
            'strategy_type': None,
            'long_leg': None,
            'short_leg': None,
            'max_profit': 0,
            'max_loss': 0,
            'breakeven': [],
            'days_to_expiry': days_to_expiry
        }
        
        # Iron Condor: High IV (>65th percentile)
        if iv_percentile > 65:
            strategy['strategy_type'] = 'IRON_CONDOR'
            strategy['recommendation'] = 'SELL'
            strategy['confidence'] = min(100, (iv_percentile - 50) * 2)
            
            # Setup: Sell Call Spread + Sell Put Spread
            atm_strike = round(atm_price / 100) * 100
            call_short = atm_strike + 200  # OTM call
            call_long = atm_strike + 300   # Further OTM
            put_short = atm_strike - 200   # OTM put
            put_long = atm_strike - 300    # Further OTM
            
            strategy['long_leg'] = f"BUY {call_long}CE + BUY {put_long}PE"
            strategy['short_leg'] = f"SELL {call_short}CE + SELL {put_short}PE"
            strategy['max_profit'] = 15000  # Simulated net credit
            strategy['max_loss'] = 35000   # Width - credit
            strategy['breakeven'] = [call_short - (strategy['max_profit']/100),
                                     put_short + (strategy['max_profit']/100)]
        
        # Butterfly: Medium IV (40-60th percentile)
        elif 40 <= iv_percentile <= 60:
            strategy['strategy_type'] = 'BUTTERFLY'
            strategy['recommendation'] = 'BUY'
            strategy['confidence'] = 50 - abs(iv_percentile - 50)  # More confident at middle
            
            atm_strike = round(atm_price / 100) * 100
            long_strike = atm_strike - 100
            middle_strike = atm_strike
            short_strike = atm_strike + 100
            
            strategy['long_leg'] = f"BUY {long_strike}PE + BUY {short_strike}CE"
            strategy['short_leg'] = f"SELL 2x {middle_strike}PE + SELL 2x {middle_strike}CE"
            strategy['max_profit'] = 8000   # At middle strike
            strategy['max_loss'] = 2000    # Net debit
            strategy['breakeven'] = [middle_strike - 2, middle_strike + 2]
        
        # Calendar Spread: Low IV (<40th percentile)
        elif iv_percentile < 40:
            strategy['strategy_type'] = 'CALENDAR_SPREAD'
            strategy['recommendation'] = 'BUY'
            strategy['confidence'] = min(100, (40 - iv_percentile) * 2)
            
            atm_strike = round(atm_price / 100) * 100
            
            strategy['long_leg'] = f"BUY 1-Month {atm_strike}CE"
            strategy['short_leg'] = f"SELL 1-Week {atm_strike}CE"
            strategy['max_profit'] = 5000   # Theta decay capture
            strategy['max_loss'] = 10000  # If move is large
            strategy['breakeven'] = [atm_strike - 3, atm_strike + 3]
        
        return strategy
    
    def scan_calendar_spreads_multi_week(
        self,
        atm_price: float,
        iv_level: float,
        current_theta: float
    ) -> List[Dict]:
        """
        Scan for multi-week calendar spreads (exploit theta decay across weeks)
        
        Args:
            atm_price: Current ATM price
            iv_level: Current IV level
            current_theta: Weekly theta value
        
        Returns:
            List of calendar spread opportunities
        """
        spreads = []
        
        atm_strike = round(atm_price / 100) * 100
        
        # Multi-week calendar spreads
        weeks = [1, 2, 3]  # 1-week, 2-week, 3-week
        
        for i in range(len(weeks) - 1):
            near_week = weeks[i]
            far_week = weeks[i + 1]
            
            spread = {
                'symbol': 'NIFTY',
                'strategy': 'MULTI_WEEK_CALENDAR',
                'strike': atm_strike,
                'near_expiry_days': 7 * near_week,
                'far_expiry_days': 7 * far_week,
                'setup': f"SELL {atm_strike}CE ({near_week}W), BUY {atm_strike}CE ({far_week}W)",
                'theta_daily': current_theta / near_week,
                'theta_advantage': 'Positive (profit from time decay)',
                'profit_on_collapse': current_theta * (far_week - near_week) * 7,
                'break_even_move': atm_price * 0.02,  # 2% move
                'confidence': 65 + (iv_level * 0.2)  # Higher IV = better setup
            }
            spreads.append(spread)
        
        return spreads
    
    def scan_delta_neutral_nifty_pair(
        self,
        nifty_price: float,
        banknifty_price: float,
        nifty_iv: float,
        bank_iv: float,
        correlation: float
    ) -> Dict:
        """
        Scan for delta-neutral NIFTY-BANKNIFTY spread
        
        Args:
            nifty_price: Current NIFTY level
            banknifty_price: Current BANKNIFTY level
            nifty_iv: NIFTY IV percentile
            bank_iv: BANKNIFTY IV percentile
            correlation: Correlation between indices
        
        Returns:
            Dict with delta-neutral setup
        """
        setup = {
            'symbol_long': 'NIFTY',
            'symbol_short': 'BANKNIFTY',
            'correlation': correlation,
            'recommendation': 'SKIP',
            'confidence': 0,
            'long_instrument': None,
            'short_instrument': None,
            'equal_exposure': 500000  # ₹5 lakh each
        }
        
        if correlation < 0.75:
            setup['recommendation'] = 'SKIP'
            setup['reason'] = 'Correlation too low'
            return setup
        
        # IV comparison: Which is relatively more expensive?
        iv_difference = nifty_iv - bank_iv
        
        if abs(iv_difference) < 5:
            setup['recommendation'] = 'SKIP'
            setup['reason'] = 'IV difference too small'
            return setup
        
        # NIFTY relatively more expensive (higher IV) → SELL NIFTY, BUY BANKNIFTY
        if nifty_iv > bank_iv + 5:
            setup['recommendation'] = 'EXECUTE'
            setup['position'] = 'NIFTY SHORT, BANKNIFTY LONG'
            setup['confidence'] = min(100, iv_difference * 3)
            
            # Choose instruments based on IV and expiry
            if nifty_iv > 70:
                setup['long_instrument'] = 'BANKNIFTY_FUTURES'
                setup['short_instrument'] = 'NIFTY_CALL_SPREAD (Sell High IV)'
            else:
                setup['long_instrument'] = 'BANKNIFTY_FUTURES'
                setup['short_instrument'] = 'NIFTY_FUTURES'
        
        # BANKNIFTY relatively more expensive → SELL BANKNIFTY, BUY NIFTY
        elif bank_iv > nifty_iv + 5:
            setup['recommendation'] = 'EXECUTE'
            setup['position'] = 'BANKNIFTY SHORT, NIFTY LONG'
            setup['confidence'] = min(100, iv_difference * 3)
            
            if bank_iv > 70:
                setup['long_instrument'] = 'NIFTY_FUTURES'
                setup['short_instrument'] = 'BANKNIFTY_CALL_SPREAD (Sell High IV)'
            else:
                setup['long_instrument'] = 'NIFTY_FUTURES'
                setup['short_instrument'] = 'BANKNIFTY_FUTURES'
        
        return setup
    
    def get_nifty_position_sizing(
        self,
        capital: float = 500000,
        risk_per_trade: float = 0.02,
        atr: float = 0,
        current_price: float = 0
    ) -> Dict:
        """
        Calculate position sizing for NIFTY futures
        
        Args:
            capital: Available capital
            risk_per_trade: Risk per trade (% of capital)
            atr: ATR value for stop loss calculation
            current_price: Current price for quantity calculation
        
        Returns:
            Dict with position details
        """
        max_loss = capital * risk_per_trade
        
        sizing = {
            'total_capital': capital,
            'risk_per_trade_rupees': max_loss,
            'current_price': current_price,
            'stop_loss_atr_multiple': 2,
            'stop_loss_points': atr * 2 if atr > 0 else 0,
            'qty_for_leverage_1x': int(capital / current_price) if current_price > 0 else 0,
            'qty_for_leverage_2x': int((capital * 2) / current_price) if current_price > 0 else 0,
            'qty_for_leverage_3x': int((capital * 3) / current_price) if current_price > 0 else 0,
            'max_loss_rupees': max_loss,
            'recommended_leverage': self.initial_leverage
        }
        
        # Calculate actual position based on risk
        if atr > 0 and current_price > 0:
            stop_loss_rupees_per_unit = atr * 2 * 1  # 2 ATR per unit
            qty = int(max_loss / stop_loss_rupees_per_unit)
            sizing['risk_based_quantity'] = qty
            sizing['actual_leverage'] = (qty * current_price) / capital
        
        return sizing
    
    def scan_all_nifty_strategies(
        self,
        current_price: float = 50000,
        rsi: float = 50,
        adx: float = 30,
        atr: float = 200,
        iv_percentile: float = 50,
        days_to_expiry: int = 7,
        banknifty_price: Optional[float] = None,
        banknifty_iv: Optional[float] = None,
        correlation: float = 0.85
    ) -> pd.DataFrame:
        """
        Scan all NIFTY strategies in one call
        
        Args:
            current_price: Current NIFTY level
            rsi: RSI value
            adx: ADX value
            atr: Average True Range
            iv_percentile: IV percentile
            days_to_expiry: Days to expiry
            banknifty_price: BANKNIFTY price (optional)
            banknifty_iv: BANKNIFTY IV (optional)
            correlation: NIFTY-BANKNIFTY correlation
        
        Returns:
            DataFrame with all scan results
        """
        results = []
        
        # 1. Futures momentum
        futures = self.scan_nifty_futures_momentum(current_price, rsi, adx, atr, 500000)
        if futures['signal'] != 'HOLD':
            results.append({
                'strategy': 'NIFTY_FUTURES_MOMENTUM',
                'signal': futures['signal'],
                'confidence': futures['confidence'],
                'instrument': 'NIFTY Futures',
                'qty': futures['quantity'],
                'stop_loss': futures['stop_loss'],
                'target': futures['profit_target_2']
            })
        
        # 2. Options spreads
        spreads = self.scan_nifty_options_spreads(current_price, iv_percentile, days_to_expiry)
        if spreads['recommendation'] != 'SKIP':
            results.append({
                'strategy': spreads['strategy_type'],
                'signal': spreads['recommendation'],
                'confidence': spreads['confidence'],
                'instrument': 'NIFTY Options',
                'setup': f"{spreads['long_leg']} / {spreads['short_leg']}",
                'max_profit': spreads['max_profit'],
                'max_loss': spreads['max_loss']
            })
        
        # 3. Calendar spreads
        calendars = self.scan_calendar_spreads_multi_week(current_price, iv_percentile, 50)
        for cal in calendars:
            results.append({
                'strategy': 'CALENDAR_SPREAD',
                'signal': 'BUY',
                'confidence': cal['confidence'],
                'instrument': 'NIFTY Options Calendar',
                'setup': cal['setup'],
                'daily_theta': cal['theta_daily']
            })
        
        # 4. Delta-neutral pair (if BANKNIFTY data available)
        if banknifty_price and banknifty_iv:
            pair = self.scan_delta_neutral_nifty_pair(
                current_price, banknifty_price, iv_percentile, banknifty_iv, correlation
            )
            if pair['recommendation'] != 'SKIP':
                results.append({
                    'strategy': 'DELTA_NEUTRAL_PAIR',
                    'signal': pair['recommendation'],
                    'confidence': pair['confidence'],
                    'position': pair['position'],
                    'instruments': f"{pair['long_instrument']} / {pair['short_instrument']}"
                })
        
        df = pd.DataFrame(results) if results else pd.DataFrame()
        return df.sort_values('confidence', ascending=False) if not df.empty else df


def scan_nifty_derivatives(
    current_price: float = 50000,
    rsi: float = 50,
    adx: float = 30,
    atr: float = 200,
    iv_percentile: float = 50,
    days_to_expiry: int = 7,
    include_banknifty: bool = True
) -> pd.DataFrame:
    """
    Standalone function to scan NIFTY derivatives strategies
    
    Args:
        current_price: Current NIFTY level
        rsi: RSI(14) value
        adx: ADX(14) value
        atr: Average True Range
        iv_percentile: IV percentile (0-100)
        days_to_expiry: Days until expiry
        include_banknifty: Include NIFTY-BANKNIFTY pair analysis
    
    Returns:
        DataFrame with all strategies and signals
    """
    scanner = NIFTYDerivativesScanner(initial_leverage=2.0)
    
    return scanner.scan_all_nifty_strategies(
        current_price=current_price,
        rsi=rsi,
        adx=adx,
        atr=atr,
        iv_percentile=iv_percentile,
        days_to_expiry=days_to_expiry,
        correlation=0.85 if include_banknifty else None
    )


# Example usage
if __name__ == "__main__":
    scanner = NIFTYDerivativesScanner()
    
    # Scan with sample data
    results = scanner.scan_all_nifty_strategies(
        current_price=50000,
        rsi=58,
        adx=32,
        atr=200,
        iv_percentile=65,
        days_to_expiry=7,
        banknifty_price=42000,
        banknifty_iv=45,
        correlation=0.87
    )
    
    print("NIFTY Derivatives Scan Results:")
    print(results)
    
    # Get position sizing
    sizing = scanner.get_nifty_position_sizing(
        capital=500000,
        risk_per_trade=0.02,
        atr=200,
        current_price=50000
    )
    print("\nPosition Sizing:")
    for key, value in sizing.items():
        print(f"  {key}: {value}")
