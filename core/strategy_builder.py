# core/strategy_builder.py — STRATEGY BUILDER FRAMEWORK
# Easy builder pattern for creating custom trading strategies
# Professional strategy construction using builder pattern

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
from core.models import StrategyBase, Signal, Trade, OrderType, PositionType
from utils.logger import logger
import logging


# ============================================================================
# SIGNAL BUILDER - Easy signal creation DSL
# ============================================================================

class SignalBuilder:
    """Builder for creating trading signals with fluent API"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.signal_type = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.confidence = 0.5
        self.kelly_fraction = 0.02
        self.z_score = 0.0
        self.metadata = {}
    
    def buy(self, entry_price: float, stop: float, target: float) -> 'SignalBuilder':
        """Configure buy signal"""
        self.signal_type = 'BUY'
        self.entry_price = entry_price
        self.stop_loss = stop
        self.take_profit = target
        return self
    
    def sell(self, entry_price: float, stop: float, target: float) -> 'SignalBuilder':
        """Configure sell signal"""
        self.signal_type = 'SELL'
        self.entry_price = entry_price
        self.stop_loss = stop
        self.take_profit = target
        return self
    
    def with_confidence(self, conf: float) -> 'SignalBuilder':
        """Set signal confidence (0-1)"""
        self.confidence = max(0, min(1, conf))
        return self
    
    def with_kelly(self, kelly_frac: float) -> 'SignalBuilder':
        """Set Kelly fraction for position sizing"""
        self.kelly_fraction = kelly_frac
        return self
    
    def with_zscore(self, z: float) -> 'SignalBuilder':
        """Set Z-score (for mean reversion)"""
        self.z_score = z
        return self
    
    def with_metadata(self, **kwargs) -> 'SignalBuilder':
        """Add metadata"""
        self.metadata.update(kwargs)
        return self
    
    def build(self) -> Signal:
        """Build the signal"""
        return Signal(
            symbol=self.symbol,
            timestamp=datetime.now(),
            strategy_name='CustomStrategy',
            signal_type=self.signal_type or 'HOLD',
            entry_price=self.entry_price or 0,
            confidence=self.confidence,
            stop_loss=self.stop_loss or 0,
            take_profit=self.take_profit or 0,
            position_size=100,
            kelly_fraction=self.kelly_fraction,
            z_score=self.z_score,
            metadata=self.metadata
        )


# ============================================================================
# STRATEGY TEMPLATES - Base classes for common patterns
# ============================================================================

class StrategyTemplate(StrategyBase):
    """Template for easy strategy creation"""
    
    def __init__(self, name: str, capital: float = 100000, max_positions: int = 5):
        super().__init__(name=name, capital=capital, max_position_size=capital/max_positions)
        self.max_positions = max_positions
        self.signals = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicator(self, data: pd.DataFrame, name: str, func: Callable) -> pd.Series:
        """Helper to calculate custom indicators"""
        return func(data)
    
    def generate_signal(self, symbol: str, signal_type: str, entry: float, 
                       stop: float, target: float, confidence: float = 0.5) -> Signal:
        """Helper to generate signals"""
        builder = SignalBuilder(symbol)
        
        if signal_type == 'BUY':
            builder.buy(entry, stop, target)
        elif signal_type == 'SELL':
            builder.sell(entry, stop, target)
        
        return builder.with_confidence(confidence).build()
    
    def _log(self, message: str, level: str = 'info'):
        """Logging helper"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)


# ============================================================================
# CONCRETE STRATEGY EXAMPLES
# ============================================================================

class MeanReversionStrategy(StrategyTemplate):
    """
    Mean Reversion Strategy Template
    
    Trades when pair Z-score exceeds entry threshold
    Exits when spread reverts to zero
    """
    
    def __init__(self, z_entry: float = 2.0, z_exit: float = 0.5, 
                 window: int = 60, capital: float = 100000):
        super().__init__("MeanReversion", capital)
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.window = window
        self.position = 0
        self.entry_price = None
    
    def scan(self, data: pd.DataFrame, metadata: Dict = None) -> List[Signal]:
        """Generate mean reversion signals"""
        signals = []
        
        try:
            if len(data) < self.window:
                return signals
            
            # Calculate spread (simple: leg1 - leg2)
            spread = data['leg1'] - data['leg2']
            mean = spread.rolling(self.window).mean()
            std = spread.rolling(self.window).std()
            zscore = (spread - mean) / std
            
            latest_z = zscore.iloc[-1]
            
            # Entry signals
            if self.position == 0:
                if latest_z < -self.z_entry:
                    # Buy signal
                    signals.append(
                        self.generate_signal(
                            symbol='PAIR',
                            signal_type='BUY',
                            entry=spread.iloc[-1],
                            stop=spread.iloc[-1] - (2 * std.iloc[-1]),
                            target=spread.iloc[-1] + (2 * std.iloc[-1]),
                            confidence=min(0.95, 0.5 + abs(latest_z) / 5)
                        )
                    )
                    self.position = 1
                    self.entry_price = spread.iloc[-1]
                
                elif latest_z > self.z_entry:
                    # Sell signal
                    signals.append(
                        self.generate_signal(
                            symbol='PAIR',
                            signal_type='SELL',
                            entry=spread.iloc[-1],
                            stop=spread.iloc[-1] + (2 * std.iloc[-1]),
                            target=spread.iloc[-1] - (2 * std.iloc[-1]),
                            confidence=min(0.95, 0.5 + abs(latest_z) / 5)
                        )
                    )
                    self.position = -1
                    self.entry_price = spread.iloc[-1]
            
            # Exit signals
            elif self.position != 0:
                if self.position == 1 and latest_z > self.z_exit:
                    signals.append(
                        self.generate_signal(
                            symbol='PAIR',
                            signal_type='SELL',
                            entry=spread.iloc[-1],
                            stop=spread.iloc[-1] + 0.01,
                            target=spread.iloc[-1] - 0.01,
                            confidence=0.8
                        )
                    )
                    self.position = 0
                
                elif self.position == -1 and latest_z < -self.z_exit:
                    signals.append(
                        self.generate_signal(
                            symbol='PAIR',
                            signal_type='BUY',
                            entry=spread.iloc[-1],
                            stop=spread.iloc[-1] - 0.01,
                            target=spread.iloc[-1] + 0.01,
                            confidence=0.8
                        )
                    )
                    self.position = 0
            
            self._log(f"Scanned {len(data)} bars, generated {len(signals)} signals")
            return signals
        
        except Exception as e:
            self._log(f"Error in scan: {e}", 'error')
            return signals
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Backtest strategy"""
        from core.backtester_v2 import VectorizedBacktester
        
        try:
            backtester = VectorizedBacktester()
            metrics, trades, equity = backtester.backtest(data, self.z_entry, self.z_exit, self.window)
            
            return {
                'metrics': metrics.to_dict(),
                'trades': [t.to_dict() for t in trades],
                'equity_curve': equity
            }
        except Exception as e:
            self._log(f"Backtest error: {e}", 'error')
            return {}
    
    def execute(self, latest_data: pd.DataFrame) -> List[Signal]:
        """Execute strategy live"""
        return self.scan(latest_data)


class MomentumStrategy(StrategyTemplate):
    """
    Momentum Strategy Template
    
    Trades when price breaks above/below moving average
    with volume confirmation
    """
    
    def __init__(self, fast_ma: int = 10, slow_ma: int = 30, 
                 volume_threshold: float = 1.5, capital: float = 100000):
        super().__init__("Momentum", capital)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.volume_threshold = volume_threshold
    
    def scan(self, data: pd.DataFrame, metadata: Dict = None) -> List[Signal]:
        """Generate momentum signals"""
        signals = []
        
        try:
            if len(data) < self.slow_ma:
                return signals
            
            # Calculate moving averages
            fast = data['close'].rolling(self.fast_ma).mean()
            slow = data['close'].rolling(self.slow_ma).mean()
            
            # Calculate volume
            avg_volume = data['volume'].rolling(self.fast_ma).mean()
            vol_ratio = data['volume'].iloc[-1] / avg_volume.iloc[-1]
            
            # Signals
            if fast.iloc[-1] > slow.iloc[-1] and vol_ratio > self.volume_threshold:
                signals.append(
                    self.generate_signal(
                        symbol=data.get('symbol', 'UNKNOWN'),
                        signal_type='BUY',
                        entry=data['close'].iloc[-1],
                        stop=data['close'].iloc[-1] * 0.98,
                        target=data['close'].iloc[-1] * 1.05,
                        confidence=0.7
                    )
                )
            
            elif fast.iloc[-1] < slow.iloc[-1] and vol_ratio > self.volume_threshold:
                signals.append(
                    self.generate_signal(
                        symbol=data.get('symbol', 'UNKNOWN'),
                        signal_type='SELL',
                        entry=data['close'].iloc[-1],
                        stop=data['close'].iloc[-1] * 1.02,
                        target=data['close'].iloc[-1] * 0.95,
                        confidence=0.7
                    )
                )
            
            self._log(f"Momentum strategy: {len(signals)} signals generated")
            return signals
        
        except Exception as e:
            self._log(f"Error in momentum scan: {e}", 'error')
            return signals
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Backtest momentum strategy"""
        return {'status': 'Backtest implementation pending'}
    
    def execute(self, latest_data: pd.DataFrame) -> List[Signal]:
        """Execute strategy live"""
        return self.scan(latest_data)


class VolatilityStrategy(StrategyTemplate):
    """
    Volatility Strategy Template
    
    Trades based on volatility regimes
    Buy when IV is low, Sell when IV is high
    """
    
    def __init__(self, lookback: int = 20, volatility_threshold: float = 0.02, 
                 capital: float = 100000):
        super().__init__("Volatility", capital)
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold
    
    def calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate historical volatility"""
        returns = np.log(data['close'] / data['close'].shift(1))
        return returns.std()
    
    def scan(self, data: pd.DataFrame, metadata: Dict = None) -> List[Signal]:
        """Generate volatility signals"""
        signals = []
        
        try:
            if len(data) < self.lookback:
                return signals
            
            # Calculate volatility
            vol = self.calculate_volatility(data[-self.lookback:])
            
            # Low volatility: buy signal
            if vol < self.volatility_threshold:
                signals.append(
                    self.generate_signal(
                        symbol=data.get('symbol', 'UNKNOWN'),
                        signal_type='BUY',
                        entry=data['close'].iloc[-1],
                        stop=data['close'].iloc[-1] * 0.97,
                        target=data['close'].iloc[-1] * 1.05,
                        confidence=0.6
                    )
                )
            
            # High volatility: sell signal
            elif vol > self.volatility_threshold * 2:
                signals.append(
                    self.generate_signal(
                        symbol=data.get('symbol', 'UNKNOWN'),
                        signal_type='SELL',
                        entry=data['close'].iloc[-1],
                        stop=data['close'].iloc[-1] * 1.03,
                        target=data['close'].iloc[-1] * 0.95,
                        confidence=0.6
                    )
                )
            
            return signals
        
        except Exception as e:
            self._log(f"Error in volatility scan: {e}", 'error')
            return signals
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Backtest volatility strategy"""
        return {'status': 'Backtest implementation pending'}
    
    def execute(self, latest_data: pd.DataFrame) -> List[Signal]:
        """Execute strategy live"""
        return self.scan(latest_data)


# ============================================================================
# STRATEGY REGISTRY - Easy strategy management
# ============================================================================

class StrategyRegistry:
    """Registry for managing available strategies"""
    
    _strategies = {
        'mean_reversion': MeanReversionStrategy,
        'momentum': MomentumStrategy,
        'volatility': VolatilityStrategy,
    }
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new strategy"""
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    @classmethod
    def get(cls, name: str, **kwargs) -> StrategyBase:
        """Get strategy instance"""
        strategy_class = cls._strategies.get(name.lower())
        if not strategy_class:
            raise ValueError(f"Strategy not found: {name}")
        return strategy_class(**kwargs)
    
    @classmethod
    def list(cls) -> List[str]:
        """List available strategies"""
        return list(cls._strategies.keys())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    ✅ Strategy Builder Framework Ready!
    
    Available Strategies:
    - MeanReversionStrategy
    - MomentumStrategy  
    - VolatilityStrategy
    
    Usage:
    
    1. Using Registry:
    ───────────────────
    from core.strategy_builder import StrategyRegistry
    
    strategy = StrategyRegistry.get('mean_reversion', z_entry=2.0, z_exit=0.5)
    signals = strategy.scan(data)
    
    2. Direct Instantiation:
    ────────────────────────
    from core.strategy_builder import MeanReversionStrategy
    
    strategy = MeanReversionStrategy(z_entry=2.0)
    signals = strategy.scan(pair_data)
    backtest_result = strategy.backtest(pair_data)
    
    3. Custom Strategy:
    ───────────────────
    from core.strategy_builder import StrategyTemplate
    
    class MyStrategy(StrategyTemplate):
        def scan(self, data, metadata=None):
            # Your logic here
            return signals
        
        def backtest(self, data):
            # Backtest logic
            return results
        
        def execute(self, latest_data):
            # Live execution
            return signals
    
    strategy = MyStrategy("MyCustom")
    """)
