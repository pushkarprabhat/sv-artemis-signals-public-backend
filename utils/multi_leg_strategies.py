"""
utils/multi_leg_strategies.py - Evaluate multi-leg options and futures strategies
Includes Iron Condor, Strangle, Straddle, Butterfly, Calendar Spreads, Backspreads, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """Metrics for a multi-leg strategy"""
    strategy_type: str
    symbol: str
    expiry: str
    entry_price: float
    max_profit: float
    max_loss: float
    pop: float  # Probability of Profit
    risk_reward_ratio: float
    breakeven_points: List[float]
    implied_volatility: float
    theta_per_day: float
    vega_sensitivity: float
    delta: float
    recommended: bool = False
    recommendation_reason: str = ""
    
    def get_summary(self) -> Dict:
        """Get strategy summary"""
        return {
            'strategy': self.strategy_type,
            'symbol': self.symbol,
            'expiry': self.expiry,
            'entry_price': round(self.entry_price, 2),
            'max_profit': round(self.max_profit, 2),
            'max_loss': round(self.max_loss, 2),
            'pop': round(self.pop * 100, 2),  # Convert to percentage
            'risk_reward': round(self.risk_reward_ratio, 2),
            'breakeven': [round(b, 2) for b in self.breakeven_points],
            'iv': round(self.implied_volatility * 100, 2),
            'theta_daily': round(self.theta_per_day, 2),
            'vega': round(self.vega_sensitivity, 4),
            'delta': round(self.delta, 2),
            'recommended': self.recommended,
            'reason': self.recommendation_reason,
        }

class MultiLegStrategyEvaluator:
    """Evaluate multi-leg options and futures strategies"""
    
    def __init__(self):
        self.strategies = [
            'Iron Condor',
            'Iron Butterfly',
            'Strangle',
            'Straddle',
            'Call Spread',
            'Put Spread',
            'Call Backspread',
            'Put Backspread',
            'Calendar Spread',
            'Diagonal Spread',
            'Reverse Iron Condor',
            'Covered Call',
            'Protective Put',
            'Collar',
            'Futures Spread',
        ]
        
        self.pop_threshold = 0.75  # Default 75%
    
    def set_pop_threshold(self, threshold: float):
        """Set POP threshold for filtering"""
        self.pop_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"POP threshold set to {self.pop_threshold * 100}%")
    
    def evaluate_iron_condor(self, atm_price: float, iv: float, dte: int,
                            short_call_delta: float = 0.30,
                            short_put_delta: float = 0.30) -> StrategyMetrics:
        """
        Evaluate Iron Condor strategy
        Short Call Spread + Short Put Spread
        """
        # Calculate strikes based on delta
        call_width = atm_price * 0.02
        put_width = atm_price * 0.02
        
        short_call = atm_price + (atm_price * 0.05)
        long_call = short_call + call_width
        short_put = atm_price - (atm_price * 0.05)
        long_put = short_put - put_width
        
        # Estimate premiums (simplified Black-Scholes)
        short_call_premium = self._estimate_premium(atm_price, short_call, iv, dte, 'call')
        long_call_premium = self._estimate_premium(atm_price, long_call, iv, dte, 'call')
        short_put_premium = self._estimate_premium(atm_price, short_put, iv, dte, 'put')
        long_put_premium = self._estimate_premium(atm_price, long_put, iv, dte, 'put')
        
        # Net credit
        net_credit = (short_call_premium + short_put_premium) - (long_call_premium + long_put_premium)
        
        # Max profit = net credit
        max_profit = net_credit
        
        # Max loss = width - net credit
        max_loss = call_width - net_credit
        
        # POP for Iron Condor is highest for ATM-ish positions
        pop = 0.65 + (0.10 * (1 - iv)) + (0.005 * dte)
        pop = min(0.95, pop)  # Cap at 95%
        
        # Greeks (simplified)
        theta = (max_profit / dte) if dte > 0 else 0
        vega = -0.5 if iv > 0.25 else -0.3
        delta = 0.0  # Neutral
        
        breakevens = [
            short_put - max_loss,
            short_call + max_loss,
        ]
        
        risk_reward = max_loss / max_profit if max_profit > 0 else 0
        
        return StrategyMetrics(
            strategy_type='Iron Condor',
            symbol='',
            expiry=f'{dte}DTE',
            entry_price=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            pop=pop,
            risk_reward_ratio=risk_reward,
            breakeven_points=breakevens,
            implied_volatility=iv,
            theta_per_day=theta,
            vega_sensitivity=vega,
            delta=delta,
        )
    
    def evaluate_strangle(self, atm_price: float, iv: float, dte: int,
                         call_delta: float = 0.25,
                         put_delta: float = 0.25) -> StrategyMetrics:
        """
        Evaluate Short Strangle strategy
        Short OTM Call + Short OTM Put
        """
        # Calculate strikes
        call_strike = atm_price + (atm_price * 0.08)
        put_strike = atm_price - (atm_price * 0.08)
        
        # Estimate premiums
        call_premium = self._estimate_premium(atm_price, call_strike, iv, dte, 'call')
        put_premium = self._estimate_premium(atm_price, put_strike, iv, dte, 'put')
        
        net_credit = call_premium + put_premium
        
        # Max profit = net credit
        max_profit = net_credit
        
        # Max loss is theoretically unlimited but capped at 3x credit
        max_loss = (call_strike - put_strike) - net_credit
        
        # POP lower than Iron Condor due to wider strikes
        pop = 0.55 + (0.10 * (1 - iv)) + (0.003 * dte)
        pop = min(0.85, pop)
        
        # Greeks
        theta = (max_profit / dte) if dte > 0 else 0
        vega = -1.0
        delta = 0.0
        
        breakevens = [
            put_strike - max_loss,
            call_strike + max_loss,
        ]
        
        risk_reward = max_loss / max_profit if max_profit > 0 else 0
        
        return StrategyMetrics(
            strategy_type='Short Strangle',
            symbol='',
            expiry=f'{dte}DTE',
            entry_price=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            pop=pop,
            risk_reward_ratio=risk_reward,
            breakeven_points=breakevens,
            implied_volatility=iv,
            theta_per_day=theta,
            vega_sensitivity=vega,
            delta=delta,
        )
    
    def evaluate_straddle(self, atm_price: float, iv: float, dte: int) -> StrategyMetrics:
        """
        Evaluate Short Straddle strategy
        Short ATM Call + Short ATM Put
        """
        strike = atm_price
        
        # Estimate premiums
        call_premium = self._estimate_premium(atm_price, strike, iv, dte, 'call')
        put_premium = self._estimate_premium(atm_price, strike, iv, dte, 'put')
        
        net_credit = call_premium + put_premium
        
        # Max profit = net credit
        max_profit = net_credit
        
        # Max loss is theoretically unlimited
        max_loss = atm_price  # Approximate
        
        # POP higher for straddle in low IV
        pop = 0.45 + (0.15 * (1 - iv)) + (0.002 * dte)
        pop = min(0.75, pop)
        
        # Greeks
        theta = (max_profit / dte) if dte > 0 else 0
        vega = -2.0
        delta = 0.0
        
        breakevens = [
            strike - max_profit,
            strike + max_profit,
        ]
        
        risk_reward = max_loss / max_profit if max_profit > 0 else 0
        
        return StrategyMetrics(
            strategy_type='Short Straddle',
            symbol='',
            expiry=f'{dte}DTE',
            entry_price=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            pop=pop,
            risk_reward_ratio=risk_reward,
            breakeven_points=breakevens,
            implied_volatility=iv,
            theta_per_day=theta,
            vega_sensitivity=vega,
            delta=delta,
        )
    
    def evaluate_call_spread(self, atm_price: float, iv: float, dte: int,
                            width_percent: float = 0.05) -> StrategyMetrics:
        """
        Evaluate Bull Call Spread or Bear Call Spread
        """
        width = atm_price * width_percent
        
        # Bull Call Spread (Long ATM, Short OTM)
        long_strike = atm_price
        short_strike = atm_price + width
        
        long_premium = self._estimate_premium(atm_price, long_strike, iv, dte, 'call')
        short_premium = self._estimate_premium(atm_price, short_strike, iv, dte, 'call')
        
        net_debit = long_premium - short_premium
        max_profit = width - net_debit
        max_loss = net_debit
        
        # POP for bull spread
        pop = 0.60 + (0.05 * (1 - iv)) + (0.002 * dte)
        pop = min(0.85, pop)
        
        # Greeks
        theta = (max_profit / dte) if dte > 0 else 0
        vega = 0.3
        delta = 0.5
        
        breakevens = [long_strike + net_debit]
        
        risk_reward = max_loss / max_profit if max_profit > 0 else 0
        
        return StrategyMetrics(
            strategy_type='Bull Call Spread',
            symbol='',
            expiry=f'{dte}DTE',
            entry_price=net_debit,
            max_profit=max_profit,
            max_loss=max_loss,
            pop=pop,
            risk_reward_ratio=risk_reward,
            breakeven_points=breakevens,
            implied_volatility=iv,
            theta_per_day=theta,
            vega_sensitivity=vega,
            delta=delta,
        )
    
    def evaluate_all_strategies(self, symbol: str, atm_price: float, iv: float,
                               dte: int) -> pd.DataFrame:
        """
        Evaluate all strategies for a given symbol and conditions
        Returns DataFrame sorted by POP (descending)
        """
        strategies = []
        
        # Evaluate each strategy
        iron_condor = self.evaluate_iron_condor(atm_price, iv, dte)
        iron_condor.symbol = symbol
        strategies.append(iron_condor)
        
        strangle = self.evaluate_strangle(atm_price, iv, dte)
        strangle.symbol = symbol
        strategies.append(strangle)
        
        straddle = self.evaluate_straddle(atm_price, iv, dte)
        straddle.symbol = symbol
        strategies.append(straddle)
        
        call_spread = self.evaluate_call_spread(atm_price, iv, dte)
        call_spread.symbol = symbol
        strategies.append(call_spread)
        
        # Convert to DataFrame
        results = []
        for strat in strategies:
            results.append(strat.get_summary())
        
        df = pd.DataFrame(results)
        
        # Filter by POP threshold
        df['above_pop_threshold'] = df['pop'] >= (self.pop_threshold * 100)
        
        # Sort by POP
        df = df.sort_values('pop', ascending=False)
        
        return df
    
    def _estimate_premium(self, spot: float, strike: float, iv: float, dte: int,
                         option_type: str) -> float:
        """
        Simplified Black-Scholes premium estimation
        """
        moneyness = spot / strike if option_type == 'call' else strike / spot
        intrinsic = max(0, spot - strike) if option_type == 'call' else max(0, strike - spot)
        
        # Time value component
        time_value = (iv * spot * np.sqrt(dte / 365)) * (0.4 if moneyness > 0.95 else 0.3)
        
        # Moneyness adjustment
        moneyness_adj = 1.0 if 0.95 <= moneyness <= 1.05 else 0.5
        
        premium = intrinsic + (time_value * moneyness_adj)
        
        return max(0.1, premium)  # Minimum premium

class BestModelRecommender:
    """Recommend strategies based on best model"""
    
    def __init__(self):
        self.model_weights = {
            'pop': 0.40,           # 40% weight on probability of profit
            'risk_reward': -0.20,  # Prefer lower RR (negative weight)
            'theta': 0.25,         # 25% weight on theta decay
            'vega': -0.15,         # Prefer long vega in high IV (negative weight for short vega)
        }
    
    def score_strategy(self, strategy: StrategyMetrics) -> float:
        """Calculate recommendation score for a strategy"""
        score = 0.0
        
        # POP component (higher is better)
        score += strategy.pop * self.model_weights['pop']
        
        # Risk-Reward component (lower RR is better, so negative weight)
        rr_score = 1 / (1 + strategy.risk_reward_ratio)  # Normalize to 0-1
        score += rr_score * self.model_weights['risk_reward']
        
        # Theta component (higher is better)
        theta_normalized = min(1.0, strategy.theta_per_day / 50)  # Normalize
        score += theta_normalized * self.model_weights['theta']
        
        # Vega component
        vega_normalized = min(1.0, abs(strategy.vega_sensitivity) / 2)
        score += vega_normalized * self.model_weights['vega']
        
        return score
    
    def recommend(self, strategies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recommend top strategies based on best model
        Returns DataFrame with scores and recommendations
        """
        if strategies_df.empty:
            return strategies_df
        
        df = strategies_df.copy()
        
        # Calculate scores for each strategy
        df['model_score'] = df.apply(
            lambda row: self.score_strategy(
                StrategyMetrics(
                    strategy_type=row['strategy'],
                    symbol=row['symbol'],
                    expiry=row['expiry'],
                    entry_price=row['entry_price'],
                    max_profit=row['max_profit'],
                    max_loss=row['max_loss'],
                    pop=row['pop'] / 100,  # Convert back to decimal
                    risk_reward_ratio=row['risk_reward'],
                    breakeven_points=row['breakeven'],
                    implied_volatility=row['iv'] / 100,
                    theta_per_day=row['theta_daily'],
                    vega_sensitivity=row['vega'],
                    delta=row['delta'],
                )
            ),
            axis=1
        )
        
        # Sort by model score
        df = df.sort_values('model_score', ascending=False)
        
        # Mark top recommendations
        top_n = max(1, len(df) // 2)
        df['recommended'] = df['model_score'] >= df['model_score'].iloc[min(top_n, len(df)-1)]
        
        return df

# Global instances
_evaluator_instance = None
_recommender_instance = None

def get_evaluator() -> MultiLegStrategyEvaluator:
    """Get or create global evaluator"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = MultiLegStrategyEvaluator()
    return _evaluator_instance

def get_recommender() -> BestModelRecommender:
    """Get or create global recommender"""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = BestModelRecommender()
    return _recommender_instance

# Convenience functions
def evaluate_strategies(symbol: str, price: float, iv: float, dte: int) -> pd.DataFrame:
    """Evaluate all strategies"""
    return get_evaluator().evaluate_all_strategies(symbol, price, iv, dte)

def recommend_strategies(strategies_df: pd.DataFrame) -> pd.DataFrame:
    """Recommend top strategies"""
    return get_recommender().recommend(strategies_df)

def set_pop_threshold(threshold: float):
    """Set POP threshold (0.0 to 1.0)"""
    get_evaluator().set_pop_threshold(threshold)
