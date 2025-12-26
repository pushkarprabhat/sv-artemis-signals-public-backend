#!/usr/bin/env python3
"""
Phase 4: Position Recommender
Determines optimal position sizing, entry/exit levels, and risk management.

Features:
  - Kelly Criterion based position sizing
  - Risk-adjusted position limits
  - Dynamic stop loss based on volatility
  - Profit target calculation
  - Risk/Reward ratio optimization
  - Drawdown protection
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import logger
from core.kelly import kelly_criterion, half_kelly, position_size as calc_position_size


class PositionRecommender:
    """Recommends optimal position sizing and risk management parameters"""
    
    def __init__(self, account_size: float = 100000, max_risk_per_trade: float = 0.02):
        """
        Args:
            account_size: Total account equity
            max_risk_per_trade: Maximum % of account to risk per trade (default 2%)
        """
        self.logger = logger
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_single_position = account_size * 0.15  # Max 15% per position
        self.max_total_open_positions = 5  # Max 5 concurrent positions
        self.daily_loss_limit = account_size * 0.05  # Stop trading if lost 5% in a day
        
        # Track positions
        self.open_positions = []
        self.daily_pnl = 0
        self.session_start_equity = account_size
        
        self.logger.info(f"[RECOMMENDER] Position recommender initialized (Account: ${account_size:,.2f})")
    
    def get_recommendation(self, signal_data: Dict, symbol: str, 
                         historical_data: pd.DataFrame) -> Dict:
        """
        Get position sizing recommendation for a signal
        
        Args:
            signal_data: From SignalConsolidator (signal, confidence, entry_level, etc)
            symbol: Symbol to trade
            historical_data: OHLCV data for risk calculations
        
        Returns:
            {
                'symbol': str,
                'signal': str,                    # BUY, SELL, or HOLD
                'confidence': float,              # 0-100%
                'position_size': int,             # Shares to buy/sell
                'position_value': float,          # $ value
                'entry_price': float,             # Recommended entry
                'stop_loss': float,               # Hard stop loss
                'target': float,                  # Profit target
                'risk_amount': float,             # $ at risk
                'reward_amount': float,           # $ potential reward
                'risk_reward_ratio': float,       # reward/risk ratio
                'kelly_percentage': float,        # Kelly Criterion %
                'volatility': float,              # Annualized volatility
                'position_duration_hours': int,   # Expected hold time
                'status': str,                    # OK, REJECTED, WARNING
                'rejection_reason': str,          # Why rejected (if any)
                'drawdown_limit': float,          # Remaining daily loss budget
                'max_concurrent_check': bool      # True if can open more positions
            }
        """
        try:
            signal = signal_data.get('signal', 'NEUTRAL')
            
            # Validate signal
            if signal == 'NEUTRAL':
                return {
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'status': 'REJECTED',
                    'rejection_reason': 'No clear signal',
                    'position_size': 0
                }
            
            # Check daily loss limit
            remaining_loss_budget = self.daily_loss_limit - abs(self.daily_pnl)
            if remaining_loss_budget <= 0:
                return {
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'status': 'REJECTED',
                    'rejection_reason': f'Daily loss limit reached (${self.daily_loss_limit:,.2f})',
                    'position_size': 0,
                    'drawdown_limit': remaining_loss_budget
                }
            
            # Check max concurrent positions
            can_open = len(self.open_positions) < self.max_total_open_positions
            
            # Calculate volatility
            volatility = self._calculate_volatility(historical_data)
            
            # Determine entry, stop, and target
            entry_price = signal_data.get('entry_level_avg', historical_data.iloc[-1]['close'])
            base_stop = signal_data.get('stop_loss', entry_price * 0.98)
            base_target = signal_data.get('target', entry_price * 1.02)
            
            # Adjust stops/targets for volatility
            stop_loss, target = self._adjust_stops_for_volatility(
                entry_price, base_stop, base_target, volatility, signal
            )
            
            # Calculate position size
            risk_amount = entry_price - stop_loss if signal == 'BUY' else stop_loss - entry_price
            risk_percent = risk_amount / entry_price
            
            # Kelly Criterion calculation
            kelly_pct = self._calculate_kelly_sizing(
                signal_data.get('confidence', 50) / 100,
                risk_reward_ratio=(target - entry_price) / abs(risk_amount) if risk_amount > 0 else 1
            )
            
            # Position size calculation
            position_size = self._calculate_position_size(
                signal,
                risk_amount,
                entry_price,
                kelly_pct,
                remaining_loss_budget
            )
            
            if position_size == 0:
                return {
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'status': 'REJECTED',
                    'rejection_reason': 'Position size would exceed limits',
                    'position_size': 0
                }
            
            position_value = position_size * entry_price
            actual_risk = position_size * abs(risk_amount)
            expected_reward = position_size * (target - entry_price)
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': signal_data.get('confidence', 50),
                'position_size': int(position_size),
                'position_value': position_value,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'risk_amount': actual_risk,
                'reward_amount': expected_reward,
                'risk_reward_ratio': expected_reward / actual_risk if actual_risk > 0 else 1,
                'kelly_percentage': kelly_pct * 100,
                'volatility': volatility,
                'position_duration_hours': signal_data.get('duration_hours', 24),
                'status': 'OK' if can_open else 'WARNING',
                'rejection_reason': '' if can_open else 'Max concurrent positions reached',
                'drawdown_limit': remaining_loss_budget,
                'max_concurrent_check': can_open,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[RECOMMENDER] Error getting recommendation for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'status': 'REJECTED',
                'rejection_reason': f'Error: {str(e)}',
                'position_size': 0
            }
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate annualized volatility from price data"""
        try:
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            return annual_vol
        except Exception as e:
            self.logger.debug(f"[RECOMMENDER] Volatility calc failed: {e}")
            return 0.25  # Default 25% volatility
    
    def _adjust_stops_for_volatility(self, entry: float, base_stop: float, 
                                     base_target: float, volatility: float,
                                     signal: str) -> Tuple[float, float]:
        """Adjust stop loss and target based on volatility"""
        # Higher volatility = wider stops
        vol_adjustment = 1 + (volatility * 0.5)  # Volatility scaled adjustment
        
        if signal == 'BUY':
            adjusted_stop = entry - (entry - base_stop) * vol_adjustment
            adjusted_target = entry + (base_target - entry) * vol_adjustment
        else:  # SELL
            adjusted_stop = entry + (base_stop - entry) * vol_adjustment
            adjusted_target = entry - (entry - base_target) * vol_adjustment
        
        return adjusted_stop, adjusted_target
    
    def _calculate_kelly_sizing(self, win_probability: float, 
                                risk_reward_ratio: float) -> float:
        """
        Calculate Kelly Criterion position sizing
        Kelly % = (bp - q) / b
        where:
          b = risk/reward ratio
          p = win probability
          q = 1 - p
        """
        try:
            q = 1 - win_probability
            b = risk_reward_ratio
            
            # Calculate Kelly
            kelly = (b * win_probability - q) / b if b > 0 else 0
            
            # Apply fractional Kelly for safety (use 25% of Kelly)
            fractional_kelly = kelly * 0.25
            
            # Clamp between 0% and 5% of account
            return max(0, min(0.05, fractional_kelly))
            
        except Exception as e:
            self.logger.debug(f"[RECOMMENDER] Kelly calc failed: {e}")
            return 0.01  # Default 1% sizing
    
    def _calculate_position_size(self, signal: str, risk_amount: float, 
                                entry_price: float, kelly_pct: float,
                                remaining_loss_budget: float) -> float:
        """
        Calculate optimal position size with constraints
        """
        # Kelly-based sizing
        kelly_position = (remaining_loss_budget / risk_amount) * kelly_pct if risk_amount > 0 else 0
        
        # Risk-based sizing (max 2% of account at risk)
        max_risk_amount = self.account_size * self.max_risk_per_trade
        risk_position = max_risk_amount / risk_amount if risk_amount > 0 else 0
        
        # Value-based constraint (max 15% of account)
        value_position = self.max_single_position / entry_price
        
        # Take minimum of all constraints
        position_size = min(kelly_position, risk_position, value_position)
        
        # Ensure odd lot (NSE odd lot size variations)
        position_size = max(0, int(position_size))
        
        return position_size
    
    def register_position(self, symbol: str, signal: str, size: int, 
                         entry: float, stop: float, target: float):
        """Register an opened position for tracking"""
        self.open_positions.append({
            'symbol': symbol,
            'signal': signal,
            'size': size,
            'entry': entry,
            'stop': stop,
            'target': target,
            'timestamp': datetime.now(),
            'status': 'OPEN'
        })
        self.logger.info(f"[RECOMMENDER] Registered position: {symbol} {signal} {size}@{entry}")
    
    def close_position(self, symbol: str, close_price: float) -> Dict:
        """Close a position and record P&L"""
        pos = next((p for p in self.open_positions if p['symbol'] == symbol and p['status'] == 'OPEN'), None)
        if not pos:
            return {'status': 'NOT_FOUND'}
        
        pnl = (close_price - pos['entry']) * pos['size'] if pos['signal'] == 'BUY' else (pos['entry'] - close_price) * pos['size']
        self.daily_pnl += pnl
        pos['status'] = 'CLOSED'
        pos['close_price'] = close_price
        pos['pnl'] = pnl
        
        self.logger.info(f"[RECOMMENDER] Closed {symbol}: PnL ${pnl:,.2f}")
        return pos
    
    def get_open_positions_summary(self) -> Dict:
        """Get summary of all open positions"""
        open = [p for p in self.open_positions if p['status'] == 'OPEN']
        return {
            'num_open': len(open),
            'positions': open,
            'daily_pnl': self.daily_pnl,
            'remaining_loss_budget': self.daily_loss_limit - abs(self.daily_pnl),
            'can_open_more': len(open) < self.max_total_open_positions
        }
