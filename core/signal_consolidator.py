# core/signal_consolidator.py
"""
Unified Signal Consolidation
Normalizes signals from all 4 strategies into single format
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
from utils.logger import logger


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    PAIR_LONG_SHORT = "PAIR_LONG_SHORT"


@dataclass
class UnifiedSignal:
    """Normalized signal format across all strategies"""
    timestamp: datetime
    symbol: str
    strategy: str  # 'pair_trading', 'mean_reversion', 'stat_arb', 'calendar_spreads'
    signal_type: str  # BUY, SELL, SHORT, PAIR_LONG_SHORT, etc.
    entry_price: float
    stop_loss: float
    target_price: float
    strength: float  # 0-1 confidence
    additional_symbol: Optional[str] = None  # For pair/spread strategies
    additional_entry: Optional[float] = None  # Secondary entry for pairs
    additional_stop: Optional[float] = None  # Secondary stop for pairs
    additional_target: Optional[float] = None  # Secondary target for pairs
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalConsolidator:
    """Consolidates signals from all 4 strategy engines"""
    
    def __init__(self):
        self.signals: List[UnifiedSignal] = []
        self.strategy_count = {'pair_trading': 0, 'mean_reversion': 0, 
                              'stat_arb': 0, 'calendar_spreads': 0}
        logger.info("[CONSOLIDATOR] Signal consolidator initialized")
    
    def add_pair_trading_signals(self, signals: List) -> None:
        """Convert pair trading signals to unified format"""
        if not signals:
            return
            
        for sig in signals:
            try:
                unified = UnifiedSignal(
                    timestamp=datetime.now(),
                    symbol=sig.symbol,
                    strategy='pair_trading',
                    signal_type=SignalType.PAIR_LONG_SHORT.value,
                    entry_price=float(sig.entry_price),
                    stop_loss=float(sig.stop_loss),
                    target_price=float(sig.target_price),
                    strength=float(sig.strength),
                    additional_symbol=sig.secondary_symbol if hasattr(sig, 'secondary_symbol') else None,
                    additional_entry=float(sig.secondary_entry) if hasattr(sig, 'secondary_entry') else None,
                    additional_stop=float(sig.secondary_stop) if hasattr(sig, 'secondary_stop') else None,
                    additional_target=float(sig.secondary_target) if hasattr(sig, 'secondary_target') else None,
                    metadata={
                        'correlation': getattr(sig, 'correlation', None),
                        'zscore': getattr(sig, 'zscore', None),
                        'beta': getattr(sig, 'beta', None)
                    }
                )
                self.signals.append(unified)
                self.strategy_count['pair_trading'] += 1
            except Exception as e:
                logger.error(f"[CONSOLIDATOR] Error converting pair trading signal: {e}")
    
    def add_mean_reversion_signals(self, signals: List) -> None:
        """Convert mean reversion signals to unified format"""
        if not signals:
            return
            
        for sig in signals:
            try:
                # Determine signal type
                if hasattr(sig, 'signal_type'):
                    signal_type = sig.signal_type
                else:
                    signal_type = SignalType.BUY.value if sig.strength > 0.5 else SignalType.SELL.value
                
                unified = UnifiedSignal(
                    timestamp=datetime.now(),
                    symbol=sig.symbol,
                    strategy='mean_reversion',
                    signal_type=signal_type,
                    entry_price=float(sig.entry_price),
                    stop_loss=float(sig.stop_loss),
                    target_price=float(sig.target_price),
                    strength=float(sig.strength),
                    metadata={
                        'zscore': getattr(sig, 'zscore', None),
                        'volatility': getattr(sig, 'volatility', None),
                        'half_life': getattr(sig, 'half_life', None),
                        'vol_regime': getattr(sig, 'vol_regime', None)
                    }
                )
                self.signals.append(unified)
                self.strategy_count['mean_reversion'] += 1
            except Exception as e:
                logger.error(f"[CONSOLIDATOR] Error converting mean reversion signal: {e}")
    
    def add_stat_arb_signals(self, signals: List) -> None:
        """Convert statistical arbitrage signals to unified format"""
        if not signals:
            return
            
        for sig in signals:
            try:
                unified = UnifiedSignal(
                    timestamp=datetime.now(),
                    symbol=sig.symbol,
                    strategy='stat_arb',
                    signal_type=SignalType.PAIR_LONG_SHORT.value,
                    entry_price=float(sig.entry_price),
                    stop_loss=float(sig.stop_loss),
                    target_price=float(sig.target_price),
                    strength=float(sig.strength),
                    additional_symbol=sig.secondary_symbol if hasattr(sig, 'secondary_symbol') else None,
                    additional_entry=float(sig.secondary_entry) if hasattr(sig, 'secondary_entry') else None,
                    additional_stop=float(sig.secondary_stop) if hasattr(sig, 'secondary_stop') else None,
                    additional_target=float(sig.secondary_target) if hasattr(sig, 'secondary_target') else None,
                    metadata={
                        'adf_pvalue': getattr(sig, 'adf_pvalue', None),
                        'half_life': getattr(sig, 'half_life', None),
                        'spread': getattr(sig, 'spread', None),
                        'zscore': getattr(sig, 'zscore', None)
                    }
                )
                self.signals.append(unified)
                self.strategy_count['stat_arb'] += 1
            except Exception as e:
                logger.error(f"[CONSOLIDATOR] Error converting stat arb signal: {e}")
    
    def add_calendar_spreads_signals(self, signals: List) -> None:
        """Convert calendar spread signals to unified format"""
        if not signals:
            return
            
        for sig in signals:
            try:
                # Calendar spreads work with options, extract underlying
                underlying = sig.underlying if hasattr(sig, 'underlying') else sig.symbol
                
                unified = UnifiedSignal(
                    timestamp=datetime.now(),
                    symbol=underlying,
                    strategy='calendar_spreads',
                    signal_type=getattr(sig, 'signal_type', SignalType.BUY.value),
                    entry_price=float(sig.entry_price),
                    stop_loss=float(sig.stop_loss),
                    target_price=float(sig.target_price),
                    strength=float(sig.strength),
                    metadata={
                        'short_expiry': getattr(sig, 'short_expiry', None),
                        'long_expiry': getattr(sig, 'long_expiry', None),
                        'theta': getattr(sig, 'theta', None),
                        'iv_percentile': getattr(sig, 'iv_percentile', None),
                        'strike': getattr(sig, 'strike', None)
                    }
                )
                self.signals.append(unified)
                self.strategy_count['calendar_spreads'] += 1
            except Exception as e:
                logger.error(f"[CONSOLIDATOR] Error converting calendar spreads signal: {e}")
    
    def get_all_signals(self) -> List[UnifiedSignal]:
        """Return all consolidated signals"""
        return self.signals.copy()
    
    def get_signals_by_symbol(self, symbol: str) -> List[UnifiedSignal]:
        """Return all signals for a specific symbol"""
        return [s for s in self.signals if s.symbol == symbol]
    
    def get_signals_by_strategy(self, strategy: str) -> List[UnifiedSignal]:
        """Return all signals from a specific strategy"""
        return [s for s in self.signals if s.strategy == strategy]
    
    def get_signals_by_confidence(self, min_confidence: float) -> List[UnifiedSignal]:
        """Return signals with minimum confidence level"""
        return [s for s in self.signals if s.strength >= min_confidence]
    
    def get_summary(self) -> Dict:
        """Get summary of consolidated signals"""
        return {
            'total_signals': len(self.signals),
            'pair_trading': self.strategy_count['pair_trading'],
            'mean_reversion': self.strategy_count['mean_reversion'],
            'stat_arb': self.strategy_count['stat_arb'],
            'calendar_spreads': self.strategy_count['calendar_spreads'],
            'unique_symbols': len(set(s.symbol for s in self.signals))
        }
    
    def clear(self) -> None:
        """Clear all signals for next batch"""
        self.signals = []
        self.strategy_count = {'pair_trading': 0, 'mean_reversion': 0, 
                              'stat_arb': 0, 'calendar_spreads': 0}
        logger.debug("[CONSOLIDATOR] Cleared all signals")


# Global consolidator instance
_consolidator = None


def get_consolidator() -> SignalConsolidator:
    """Get or create singleton consolidator"""
    global _consolidator
    if _consolidator is None:
        _consolidator = SignalConsolidator()
    return _consolidator
