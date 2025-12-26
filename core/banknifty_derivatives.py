"""
BANKNIFTY Derivatives Strategy
Banking sector-specific index futures + options + constituent analysis
Leverages banking sector dynamics and correlation patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BANKNIFTYDerivativesScanner:
    """Scan and implement BANKNIFTY derivatives strategies (futures + options)"""
    
    def __init__(self, max_leverage: float = 3.0):
        """
        Initialize BANKNIFTY Derivatives Scanner
        
        Args:
            max_leverage: Maximum allowed leverage (banking more volatile than NIFTY)
        """
        self.leverage_range = (1.0, max_leverage)
        self.symbol = 'BANKNIFTY'
        
        # Banking sector constituents
        self.bank_constituents = [
            'SBIN', 'ICICIBANK', 'HDFC', 'KOTAK', 'AXISBANK', 
            'INDUSIND', 'IDBIBANK', 'BANKBARODA', 'IDFCBANK',
            'KTKBANK', 'FEDERALBNK', 'RBLBANK'
        ]
        
        # Typical beta to NIFTY
        self.typical_beta = 0.95  # BANKNIFTY slightly less volatile than NIFTY
    
    def scan_banknifty_futures_momentum(
        self,
        current_price: float,
        rsi: float,
        adx: float,
        atr: float,
        volume: float,
        nifty_price: Optional[float] = None,
        nifty_rsi: Optional[float] = None
    ) -> Dict:
        """
        Scan BANKNIFTY futures with banking sector context
        
        Args:
            current_price: Current BANKNIFTY level
            rsi: RSI(14) value
            adx: ADX(14) value
            atr: Average True Range
            volume: Daily volume
            nifty_price: NIFTY price (for relative strength)
            nifty_rsi: NIFTY RSI (for comparison)
        
        Returns:
            Dict with signal and position details
        """
        signal = {
            'symbol': 'BANKNIFTY',
            'instrument': 'BANKNIFTY_FUTURES',
            'signal': 'HOLD',
            'confidence': 0,
            'entry_price': current_price,
            'position_type': None,
            'quantity': 0,
            'stop_loss': 0,
            'profit_target_1': 0,
            'profit_target_2': 0,
            'relative_strength': None,
            'sector_context': None,
            'holding_days': 3,  # Shorter holding on BANKNIFTY (more volatile)
            'atr': atr,
            'adx': adx
        }
        
        # Trend confirmation
        if adx < 20:
            signal['confidence'] = 0
            return signal
        
        # Relative strength to NIFTY
        if nifty_price and nifty_rsi:
            relative_return = (current_price - nifty_price) / nifty_price * 100
            rsi_relative = rsi - nifty_rsi
            
            signal['relative_strength'] = {
                'outperformance_pct': relative_return,
                'rsi_lead': rsi_relative
            }
        
        # Banking sector uptrend
        if rsi > 55 and adx >= 20:
            signal['signal'] = 'BUY'
            signal['position_type'] = 'LONG'
            signal['confidence'] = min(100, (rsi - 50) * 1.5 + (adx - 20) * 1.5)
            signal['sector_context'] = 'BULLISH_BANKING'
            
            # Conservative leverage for banking sector (more volatile)
            leverage = 1.5
            position_value = 400000 * leverage  # Smaller position base
            signal['quantity'] = int(position_value / current_price)
            
            # Tighter stops on banking (higher volatility)
            signal['stop_loss'] = current_price - (1.8 * atr)
            
            expected_move = atr * 2.5
            signal['profit_target_1'] = current_price + (expected_move * 0.5)
            signal['profit_target_2'] = current_price + (expected_move * 0.9)
        
        # Banking sector downtrend
        elif rsi < 45 and adx >= 20:
            signal['signal'] = 'SELL'
            signal['position_type'] = 'SHORT'
            signal['confidence'] = min(100, (50 - rsi) * 1.5 + (adx - 20) * 1.5)
            signal['sector_context'] = 'BEARISH_BANKING'
            
            leverage = 1.5
            position_value = 400000 * leverage
            signal['quantity'] = int(position_value / current_price)
            
            signal['stop_loss'] = current_price + (1.8 * atr)
            
            expected_move = atr * 2.5
            signal['profit_target_1'] = current_price - (expected_move * 0.5)
            signal['profit_target_2'] = current_price - (expected_move * 0.9)
        
        return signal
    
    def scan_banknifty_constituent_correlation(
        self,
        banknifty_price: float,
        constituent_prices: Dict[str, float],
        constituent_strengths: Dict[str, float]
    ) -> Dict:
        """
        Analyze correlation between BANKNIFTY and its constituents
        
        Args:
            banknifty_price: Current BANKNIFTY level
            constituent_prices: Dict of {symbol: current_price}
            constituent_strengths: Dict of {symbol: rsi_or_momentum}
        
        Returns:
            Dict with constituent divergence signals
        """
        analysis = {
            'symbol': 'BANKNIFTY',
            'analysis_type': 'CONSTITUENT_DIVERGENCE',
            'signal': 'HOLD',
            'confidence': 0,
            'divergence_explanation': None,
            'key_movers': [],
            'recommendation': None
        }
        
        if not constituent_prices or not constituent_strengths:
            return analysis
        
        # Calculate average constituent strength
        strengths = list(constituent_strengths.values())
        avg_strength = np.mean(strengths)
        
        # Identify strong and weak constituents
        strong_stocks = [s for s, str_val in constituent_strengths.items() if str_val > avg_strength + 5]
        weak_stocks = [s for s, str_val in constituent_strengths.items() if str_val < avg_strength - 5]
        
        analysis['key_movers'] = {
            'strong': strong_stocks,
            'weak': weak_stocks,
            'avg_strength': avg_strength
        }
        
        # Divergence signal: Strong constituents but weak index (or vice versa)
        if len(strong_stocks) >= 5 and len(weak_stocks) <= 2:
            analysis['signal'] = 'BUY'
            analysis['confidence'] = 60 + len(strong_stocks) * 5
            analysis['divergence_explanation'] = f"Majority of banking stocks strong ({len(strong_stocks)}/12), index may catch up"
            analysis['recommendation'] = "LONG BANKNIFTY (Catch-up play)"
        
        elif len(weak_stocks) >= 5 and len(strong_stocks) <= 2:
            analysis['signal'] = 'SELL'
            analysis['confidence'] = 60 + len(weak_stocks) * 5
            analysis['divergence_explanation'] = f"Majority of banking stocks weak ({len(weak_stocks)}/12), index may decline"
            analysis['recommendation'] = "SHORT BANKNIFTY (Weakness spread)"
        
        return analysis
    
    def scan_banknifty_options_oi_based(
        self,
        atm_price: float,
        call_max_oi_strike: float,
        put_max_oi_strike: float,
        call_buildup: float,
        put_buildup: float,
        iv_percentile: float
    ) -> Dict:
        """
        Scan BANKNIFTY options using Open Interest (OI) analysis
        
        Args:
            atm_price: Current ATM price
            call_max_oi_strike: Strike with max call OI
            put_max_oi_strike: Strike with max put OI
            call_buildup: % change in call OI (positive = fresh shorts)
            put_buildup: % change in put OI (positive = fresh longs)
            iv_percentile: IV percentile
        
        Returns:
            Dict with OI-based signal
        """
        signal = {
            'symbol': 'BANKNIFTY',
            'analysis_type': 'OI_ANALYSIS',
            'signal': 'HOLD',
            'confidence': 0,
            'max_pain': None,
            'oi_structure': None,
            'recommendation': None,
            'strategy': None
        }
        
        # Calculate max pain (likely closing price)
        # Simplified: Average of max call OI and max put OI strikes
        max_pain = (call_max_oi_strike + put_max_oi_strike) / 2
        signal['max_pain'] = max_pain
        
        # OI structure analysis
        distance_from_atm = abs(max_pain - atm_price) / atm_price * 100
        
        # If max pain is 2%+ above current, bullish bias
        if max_pain > atm_price * 1.02:
            signal['signal'] = 'BUY'
            signal['oi_structure'] = f"Bullish (Max pain {distance_from_atm:.2f}% above current)"
            signal['confidence'] = min(100, distance_from_atm * 3)
            
            # Choose strategy based on buildup
            if call_buildup > 10 and iv_percentile > 60:
                signal['strategy'] = 'SELL_CALL_SPREADS (Fresh shorts getting trapped)'
            else:
                signal['strategy'] = 'BUY_CALL_SPREADS'
        
        # If max pain is 2%+ below current, bearish bias
        elif max_pain < atm_price * 0.98:
            signal['signal'] = 'SELL'
            signal['oi_structure'] = f"Bearish (Max pain {distance_from_atm:.2f}% below current)"
            signal['confidence'] = min(100, distance_from_atm * 3)
            
            if put_buildup > 10 and iv_percentile > 60:
                signal['strategy'] = 'SELL_PUT_SPREADS (Fresh longs getting trapped)'
            else:
                signal['strategy'] = 'BUY_PUT_SPREADS'
        
        else:
            signal['oi_structure'] = 'Neutral (Max pain near current price)'
            signal['strategy'] = 'CALENDAR_SPREADS or VOLATILITY_PLAYS'
        
        return signal
    
    def scan_banknifty_options_spreads(
        self,
        atm_price: float,
        iv_percentile: float,
        days_to_expiry: int
    ) -> List[Dict]:
        """
        Scan for BANKNIFTY options spreads based on IV level
        
        Args:
            atm_price: Current ATM price
            iv_percentile: IV percentile (0-100)
            days_to_expiry: Days until expiry
        
        Returns:
            List of recommended spreads
        """
        spreads = []
        
        atm_strike = round(atm_price / 100) * 100
        
        # High IV: Sell spreads (reduce risk with defined loss)
        if iv_percentile > 70:
            # Iron Condor
            spreads.append({
                'type': 'IRON_CONDOR',
                'recommendation': 'SELL',
                'confidence': min(100, (iv_percentile - 50) * 2),
                'setup': f"Sell {atm_strike+100}CE, Buy {atm_strike+200}CE / Sell {atm_strike-100}PE, Buy {atm_strike-200}PE",
                'max_profit': 12000,
                'max_loss': 28000,
                'breakeven': [atm_strike - 88, atm_strike + 88],
                'rationale': 'High IV = premium collection opportunity'
            })
            
            # Bull Call Spread (directional bias)
            spreads.append({
                'type': 'BULL_CALL_SPREAD',
                'recommendation': 'SELL_SPREADS',
                'confidence': 70,
                'setup': f"Sell {atm_strike}CE, Buy {atm_strike+100}CE",
                'max_profit': 8000,
                'max_loss': 2000,
                'breakeven': [atm_strike - 8, atm_strike + 92],
                'rationale': 'Premium decay on short call dominates'
            })
        
        # Low IV: Buy spreads (capture volatility expansion)
        elif iv_percentile < 40:
            # Long Straddle
            spreads.append({
                'type': 'LONG_STRADDLE',
                'recommendation': 'BUY',
                'confidence': min(100, (40 - iv_percentile) * 2),
                'setup': f"Buy {atm_strike}CE, Buy {atm_strike}PE",
                'max_loss': 6000,
                'max_profit': 'Unlimited',
                'breakeven': [atm_strike - 6, atm_strike + 6],
                'rationale': 'Low IV = option prices cheap, expect expansion'
            })
            
            # Butterfly (low capital requirement)
            spreads.append({
                'type': 'BUTTERFLY',
                'recommendation': 'BUY',
                'confidence': 55,
                'setup': f"Buy {atm_strike-100}CE, Sell 2x {atm_strike}CE, Buy {atm_strike+100}CE",
                'max_loss': 3000,
                'max_profit': 7000,
                'breakeven': [atm_strike - 97, atm_strike + 97],
                'rationale': 'Defined risk, low capital for range trade'
            })
        
        # Medium IV: Calendar spreads
        else:
            spreads.append({
                'type': 'CALENDAR_SPREAD',
                'recommendation': 'NEUTRAL',
                'confidence': 60,
                'setup': f"Sell 1-week {atm_strike}CE, Buy 1-month {atm_strike}CE",
                'theta_daily': 300,
                'max_loss': 5000,
                'max_profit': 6000,
                'breakeven': [atm_strike - 6, atm_strike + 6],
                'rationale': 'Theta decay on near-term contract'
            })
        
        return spreads
    
    def scan_banking_earnings_impact(
        self,
        days_to_earnings: int,
        earnings_symbols: List[str],
        historical_volatility_increase: float = 0.15
    ) -> Dict:
        """
        Analyze potential impact of banking earnings on BANKNIFTY
        
        Args:
            days_to_earnings: Days until next major earnings
            earnings_symbols: Banks announcing earnings
            historical_volatility_increase: Historical IV increase % (usually 15-20%)
        
        Returns:
            Dict with earnings impact analysis
        """
        analysis = {
            'symbol': 'BANKNIFTY',
            'analysis_type': 'EARNINGS_IMPACT',
            'days_to_earnings': days_to_earnings,
            'major_earnings': earnings_symbols,
            'expected_volatility_increase': historical_volatility_increase,
            'strategy_recommendation': None,
            'timing': None
        }
        
        if days_to_earnings > 10:
            analysis['timing'] = 'PRE_EARNINGS'
            analysis['strategy_recommendation'] = 'BUY STRADDLES/STRANGLES (Buy volatility before earnings)'
        elif days_to_earnings <= 3:
            analysis['timing'] = 'POST_EARNINGS'
            analysis['strategy_recommendation'] = 'SELL SPREADS (Sell inflated volatility after earnings)'
        else:
            analysis['timing'] = 'EARNINGS_WEEK'
            analysis['strategy_recommendation'] = 'HOLD or REDUCE LEVERAGE'
        
        analysis['expected_iv_increase'] = historical_volatility_increase * 100
        
        return analysis
    
    def get_banknifty_position_sizing(
        self,
        capital: float = 400000,
        risk_per_trade: float = 0.015,
        atr: float = 150,
        current_price: float = 42000
    ) -> Dict:
        """
        Conservative position sizing for BANKNIFTY (higher leverage risk)
        
        Args:
            capital: Available capital (lower than NIFTY due to volatility)
            risk_per_trade: Risk per trade (% of capital, lower for BANKNIFTY)
            atr: ATR value
            current_price: Current BANKNIFTY price
        
        Returns:
            Dict with position sizing
        """
        max_loss = capital * risk_per_trade
        
        sizing = {
            'total_capital': capital,
            'risk_per_trade_rupees': max_loss,
            'current_price': current_price,
            'stop_loss_atr_multiple': 1.8,
            'stop_loss_points': atr * 1.8,
            'qty_leverage_1x': int(capital / current_price),
            'qty_leverage_1_5x': int((capital * 1.5) / current_price),
            'recommended_leverage': 1.5,
            'max_loss_rupees': max_loss,
            'warning': 'BANKNIFTY more volatile - conservative sizing recommended'
        }
        
        if atr > 0 and current_price > 0:
            risk_based_qty = int(max_loss / (atr * 1.8))
            sizing['risk_based_quantity'] = risk_based_qty
            sizing['actual_leverage'] = (risk_based_qty * current_price) / capital
        
        return sizing
    
    def scan_all_banknifty_strategies(
        self,
        current_price: float = 42000,
        rsi: float = 50,
        adx: float = 25,
        atr: float = 150,
        iv_percentile: float = 50,
        days_to_expiry: int = 7,
        constituent_data: Optional[Dict] = None,
        nifty_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Scan all BANKNIFTY strategies in single call
        
        Args:
            current_price: Current BANKNIFTY level
            rsi: RSI value
            adx: ADX value
            atr: Average True Range
            iv_percentile: IV percentile
            days_to_expiry: Days to expiry
            constituent_data: Dict with constituent prices/strengths
            nifty_data: Dict with NIFTY price/RSI
        
        Returns:
            DataFrame with all signals
        """
        results = []
        
        # 1. Futures momentum
        futures = self.scan_banknifty_futures_momentum(
            current_price, rsi, adx, atr, 300000,
            nifty_price=nifty_data.get('price') if nifty_data else None,
            nifty_rsi=nifty_data.get('rsi') if nifty_data else None
        )
        if futures['signal'] != 'HOLD':
            results.append({
                'strategy': 'BANKNIFTY_FUTURES_MOMENTUM',
                'signal': futures['signal'],
                'confidence': futures['confidence'],
                'sector_context': futures['sector_context'],
                'instrument': 'BANKNIFTY Futures'
            })
        
        # 2. Constituent correlation
        if constituent_data:
            correlation = self.scan_banknifty_constituent_correlation(
                current_price,
                constituent_data.get('prices', {}),
                constituent_data.get('strengths', {})
            )
            if correlation['signal'] != 'HOLD':
                results.append({
                    'strategy': 'CONSTITUENT_DIVERGENCE',
                    'signal': correlation['signal'],
                    'confidence': correlation['confidence'],
                    'key_movers': str(correlation['key_movers']),
                    'recommendation': correlation['recommendation']
                })
        
        # 3. Options spreads
        spreads = self.scan_banknifty_options_spreads(current_price, iv_percentile, days_to_expiry)
        for spread in spreads:
            results.append({
                'strategy': spread['type'],
                'signal': spread['recommendation'],
                'confidence': spread['confidence'],
                'setup': spread['setup'],
                'max_profit': spread.get('max_profit'),
                'max_loss': spread.get('max_loss')
            })
        
        df = pd.DataFrame(results) if results else pd.DataFrame()
        return df.sort_values('confidence', ascending=False) if not df.empty else df


def scan_banknifty_derivatives(
    current_price: float = 42000,
    rsi: float = 50,
    adx: float = 25,
    atr: float = 150,
    iv_percentile: float = 50,
    days_to_expiry: int = 7
) -> pd.DataFrame:
    """
    Standalone function to scan BANKNIFTY derivatives
    
    Args:
        current_price: Current BANKNIFTY level
        rsi: RSI(14) value
        adx: ADX(14) value
        atr: Average True Range
        iv_percentile: IV percentile
        days_to_expiry: Days until expiry
    
    Returns:
        DataFrame with all signals
    """
    scanner = BANKNIFTYDerivativesScanner()
    return scanner.scan_all_banknifty_strategies(
        current_price, rsi, adx, atr, iv_percentile, days_to_expiry
    )


# Example usage
if __name__ == "__main__":
    scanner = BANKNIFTYDerivativesScanner()
    
    # Sample constituent data
    constituent_prices = {
        'SBIN': 550, 'ICICIBANK': 850, 'HDFC': 2800, 'KOTAK': 1750,
        'AXISBANK': 920, 'INDUSIND': 1300, 'IDBIBANK': 58,
        'BANKBARODA': 190, 'IDFCBANK': 98, 'KTKBANK': 150,
        'FEDERALBNK': 180, 'RBLBANK': 210
    }
    
    constituent_strengths = {
        s: np.random.randint(30, 70) for s in constituent_prices.keys()
    }
    
    results = scanner.scan_all_banknifty_strategies(
        current_price=42000,
        rsi=58,
        adx=28,
        atr=150,
        iv_percentile=60,
        days_to_expiry=7,
        constituent_data={
            'prices': constituent_prices,
            'strengths': constituent_strengths
        },
        nifty_data={'price': 50000, 'rsi': 55}
    )
    
    print("BANKNIFTY Derivatives Scan Results:")
    print(results)
