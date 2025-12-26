#!/usr/bin/env python3
"""
Phase 4: Signal Consolidation Engine
Consolidates multi-strategy pair signals with dynamic weighting and confidence scoring.

Features:
  - Combines signals from multiple strategies (pairs, momentum, mean reversion)
  - Dynamic weighting based on recent performance
  - Confidence scoring (0-100%)
  - Signal quality filtering
  - Timestamp tracking
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import logger
from core.strategies import PairsTradingStrategy, MeanReversionStrategy, MomentumStrategy


class SignalConsolidator:
    """Consolidates signals from multiple sources into unified trading signals"""
    
    def __init__(self, lookback_days: int = 30):
        """
        Args:
            lookback_days: History to track for performance weighting
        """
        self.logger = logger
        self.lookback_days = lookback_days
        self.lookback_date = datetime.now() - timedelta(days=lookback_days)
        
        # Strategy instances
        self.pairs_strategy = PairsTradingStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.momentum = MomentumStrategy()
        
        # Performance tracking
        self.strategy_performance = {
            'pairs': {'wins': 0, 'losses': 0, 'hit_rate': 0.5},
            'mean_reversion': {'wins': 0, 'losses': 0, 'hit_rate': 0.5},
            'momentum': {'wins': 0, 'losses': 0, 'hit_rate': 0.5},
        }
        
        self.logger.info("[CONSOLIDATION] Signal consolidation engine initialized")
    
    def consolidate_signals(self, pair_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Consolidate signals from all strategies for a pair
        
        Args:
            pair_data: {
                'primary': pd.DataFrame,      # Main leg OHLCV
                'secondary': pd.DataFrame,    # Pair leg OHLCV
                'pair_name': str              # e.g., 'BHARTIARTL-IDEA'
            }
        
        Returns:
            {
                'pair': str,
                'timestamp': datetime,
                'signal': str,                # BUY, SELL, NEUTRAL
                'confidence': float,          # 0-100%
                'source_signals': {
                    'pairs': {...},
                    'mean_reversion': {...},
                    'momentum': {...}
                },
                'entry_levels': [float],
                'stop_loss': float,
                'target': float,
                'duration_hours': int,
                'quality_score': float        # Overall signal quality
            }
        """
        try:
            pair_name = pair_data.get('pair_name', 'UNKNOWN')
            timestamp = datetime.now()
            
            # Generate individual signals
            signals = {}
            
            # Pairs trading signal
            if 'primary' in pair_data and 'secondary' in pair_data:
                signals['pairs'] = self._generate_pairs_signal(
                    pair_data['primary'],
                    pair_data['secondary'],
                    pair_name
                )
            
            # Mean reversion signal
            if 'primary' in pair_data:
                signals['mean_reversion'] = self._generate_mean_reversion_signal(
                    pair_data['primary'],
                    pair_name
                )
            
            # Momentum signal
            if 'primary' in pair_data:
                signals['momentum'] = self._generate_momentum_signal(
                    pair_data['primary'],
                    pair_name
                )
            
            # Consolidate with weighting
            consolidated = self._weight_and_consolidate(signals, pair_name)
            consolidated['pair'] = pair_name
            consolidated['timestamp'] = timestamp
            consolidated['source_signals'] = signals
            
            return consolidated
            
        except Exception as e:
            self.logger.error(f"[CONSOLIDATION] Error consolidating signals for {pair_name}: {e}")
            return {
                'pair': pair_name,
                'signal': 'NEUTRAL',
                'confidence': 0,
                'error': str(e)
            }
    
    def _generate_pairs_signal(self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame, 
                               pair_name: str) -> Dict:
        """Generate signal from pairs trading strategy"""
        try:
            signal_dict = self.pairs_strategy.generate_signals(df_primary, df_secondary)
            
            return {
                'signal': signal_dict.get('signal', 'NEUTRAL'),
                'confidence': signal_dict.get('confidence', 0),
                'z_score': signal_dict.get('zscore', 0),
                'entry_level': signal_dict.get('entry_level', df_primary.iloc[-1]['close']),
                'stop_loss': signal_dict.get('stop_loss', df_primary.iloc[-1]['close'] * 0.99),
                'target': signal_dict.get('target', df_primary.iloc[-1]['close'] * 1.02),
                'duration_hours': 24,
                'weight': self.strategy_performance['pairs']['hit_rate']
            }
        except Exception as e:
            self.logger.debug(f"[CONSOLIDATION] Pairs signal generation failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'weight': 0.5}
    
    def _generate_mean_reversion_signal(self, df: pd.DataFrame, pair_name: str) -> Dict:
        """Generate signal from mean reversion strategy"""
        try:
            signal_df = self.mean_reversion.generate_signals(df.copy())
            latest = signal_df.iloc[-1]
            
            signal = 'BUY' if latest.get('signal') == 1 else ('SELL' if latest.get('signal') == -1 else 'NEUTRAL')
            confidence = abs(latest.get('momentum', 0)) * 100
            
            return {
                'signal': signal,
                'confidence': min(confidence, 100),
                'momentum': latest.get('momentum', 0),
                'entry_level': latest['close'],
                'stop_loss': latest['close'] * 0.985,
                'target': latest['close'] * 1.015,
                'duration_hours': 4,
                'weight': self.strategy_performance['mean_reversion']['hit_rate']
            }
        except Exception as e:
            self.logger.debug(f"[CONSOLIDATION] Mean reversion signal generation failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'weight': 0.5}
    
    def _generate_momentum_signal(self, df: pd.DataFrame, pair_name: str) -> Dict:
        """Generate signal from momentum strategy"""
        try:
            signal_df = self.momentum.generate_signals(df.copy())
            latest = signal_df.iloc[-1]
            
            signal = 'BUY' if latest.get('signal') == 1 else ('SELL' if latest.get('signal') == -1 else 'NEUTRAL')
            confidence = abs(latest.get('strength', 0)) * 100
            
            return {
                'signal': signal,
                'confidence': min(confidence, 100),
                'strength': latest.get('strength', 0),
                'entry_level': latest['close'],
                'stop_loss': latest['close'] * 0.98,
                'target': latest['close'] * 1.025,
                'duration_hours': 2,
                'weight': self.strategy_performance['momentum']['hit_rate']
            }
        except Exception as e:
            self.logger.debug(f"[CONSOLIDATION] Momentum signal generation failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'weight': 0.5}
    
    def _weight_and_consolidate(self, signals: Dict, pair_name: str) -> Dict:
        """Consolidate signals with weighted averaging"""
        if not signals:
            return {'signal': 'NEUTRAL', 'confidence': 0}
        
        # Filter valid signals
        valid_signals = {k: v for k, v in signals.items() if v.get('signal') != 'NEUTRAL'}
        
        if not valid_signals:
            return {'signal': 'NEUTRAL', 'confidence': 0}
        
        # Calculate weighted vote
        signal_votes = {'BUY': 0, 'SELL': 0}
        total_weight = 0
        confidence_values = []
        entry_levels = []
        stop_losses = []
        targets = []
        durations = []
        
        for strategy_name, signal_data in valid_signals.items():
            weight = signal_data.get('weight', 0.5)
            total_weight += weight
            
            sig = signal_data.get('signal', 'NEUTRAL')
            if sig in signal_votes:
                signal_votes[sig] += weight
            
            confidence_values.append(signal_data.get('confidence', 0) * weight)
            entry_levels.append(signal_data.get('entry_level', 0))
            stop_losses.append(signal_data.get('stop_loss', 0))
            targets.append(signal_data.get('target', 0))
            durations.append(signal_data.get('duration_hours', 24))
        
        # Determine final signal
        if signal_votes['BUY'] > signal_votes['SELL']:
            final_signal = 'BUY'
        elif signal_votes['SELL'] > signal_votes['BUY']:
            final_signal = 'SELL'
        else:
            final_signal = 'NEUTRAL'
        
        # Calculate confidence and quality
        confidence = sum(confidence_values) / total_weight if total_weight > 0 else 0
        agreement_score = abs(signal_votes['BUY'] - signal_votes['SELL']) / total_weight * 100 if total_weight > 0 else 0
        
        return {
            'signal': final_signal,
            'confidence': min(confidence, 100),
            'agreement_score': agreement_score,
            'entry_levels': entry_levels,
            'entry_level_avg': np.mean(entry_levels) if entry_levels else 0,
            'stop_loss': np.mean(stop_losses) if stop_losses else 0,
            'target': np.mean(targets) if targets else 0,
            'duration_hours': int(np.mean(durations)) if durations else 24,
            'quality_score': (confidence * agreement_score) / 100 if confidence > 0 else 0,
            'num_strategies': len(valid_signals)
        }
    
    def update_strategy_performance(self, strategy_name: str, outcome: str):
        """
        Update performance metrics for a strategy
        
        Args:
            strategy_name: 'pairs', 'mean_reversion', or 'momentum'
            outcome: 'win' or 'loss'
        """
        if strategy_name not in self.strategy_performance:
            return
        
        if outcome.lower() == 'win':
            self.strategy_performance[strategy_name]['wins'] += 1
        elif outcome.lower() == 'loss':
            self.strategy_performance[strategy_name]['losses'] += 1
        
        # Recalculate hit rate
        total = self.strategy_performance[strategy_name]['wins'] + self.strategy_performance[strategy_name]['losses']
        if total > 0:
            self.strategy_performance[strategy_name]['hit_rate'] = (
                self.strategy_performance[strategy_name]['wins'] / total
            )
        
        self.logger.debug(f"[CONSOLIDATION] Updated {strategy_name}: {self.strategy_performance[strategy_name]}")
