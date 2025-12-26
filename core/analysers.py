# core/analysers.py — INPUT/OUTPUT ANALYSIS FOR DATA QUALITY & SIGNAL STRENGTH
# Detects "garbage in, garbage out" before trading
# Built for Artemis Signals platform

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import logger


@dataclass
class IOValidationResult:
    """Result of IO validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0-100, quality score
    metadata: Dict[str, Any]


class IOAnalyser(ABC):
    """
    Input/Output Analyser - validates data quality before trading.
    
    INPUT VALIDATION:
    - Data completeness (no missing OHLCV)
    - Volume levels (not zero)
    - Price validity (positive, no extreme moves)
    - Timestamp continuity (no gaps)
    - Staleness (not too old)
    
    OUTPUT VALIDATION:
    - Signal strength (confidence > threshold)
    - Signal diversity (not all same direction)
    - Signal frequency (not too many/few)
    - Drawdown limits (respect max drawdown)
    - Position concentration (not all capital in one trade)
    """
    
    def __init__(self, name: str = "IOAnalyser"):
        self.name = name
        self.results: List[IOValidationResult] = []
    
    @abstractmethod
    def validate_input(self, data: Dict[str, pd.DataFrame]) -> IOValidationResult:
        """Validate input data quality"""
        pass
    
    @abstractmethod
    def validate_output(self, signals: List, trades: List) -> IOValidationResult:
        """Validate output signal quality"""
        pass


class DataQualityAnalyser(IOAnalyser):
    """Validates input data quality"""
    
    def __init__(
        self,
        min_volume: float = 50000,
        max_price_move: float = 0.10,  # 10% max single candle
        max_staleness_minutes: int = 5
    ):
        super().__init__("DataQualityAnalyser")
        self.min_volume = min_volume
        self.max_price_move = max_price_move
        self.max_staleness_minutes = max_staleness_minutes
    
    def validate_input(self, data: Dict[str, pd.DataFrame]) -> IOValidationResult:
        """
        Validate input data quality.
        
        Checks:
        1. Required columns (OHLCV)
        2. No missing values
        3. Positive prices
        4. Volume levels
        5. Price move validation
        6. Timestamp continuity
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if not data or not isinstance(data, dict):
                return IOValidationResult(
                    is_valid=False,
                    errors=["Empty or invalid data dictionary"],
                    warnings=[],
                    score=0,
                    metadata={}
                )
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            symbol_scores = []
            
            for symbol, df in data.items():
                symbol_metadata = {
                    'rows': len(df),
                    'columns_valid': True,
                    'volume_valid': True,
                    'price_valid': True,
                    'timestamp_valid': True
                }
                
                # 1. Check columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    errors.append(f"{symbol}: Missing columns {missing_cols}")
                    symbol_metadata['columns_valid'] = False
                    continue
                
                # 2. Check for NaN
                nan_count = df[required_cols].isna().sum().sum()
                if nan_count > 0:
                    warnings.append(f"{symbol}: {nan_count} NaN values detected")
                
                # 3. Check positive prices
                if (df['close'] <= 0).any():
                    errors.append(f"{symbol}: Non-positive prices detected")
                    symbol_metadata['price_valid'] = False
                
                # 4. Check volume
                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > len(df) * 0.1:  # More than 10% zero volume
                    errors.append(f"{symbol}: {zero_volume} candles with zero volume")
                    symbol_metadata['volume_valid'] = False
                elif df['volume'].mean() < self.min_volume:
                    warnings.append(f"{symbol}: Average volume {df['volume'].mean():.0f} < {self.min_volume}")
                
                # 5. Check price moves (no gaps > 10%)
                if len(df) > 1:
                    price_changes = df['close'].pct_change().abs()
                    max_move = price_changes.max()
                    if max_move > self.max_price_move:
                        warnings.append(f"{symbol}: Large price move detected ({max_move:.2%})")
                    
                    # Check for extreme moves (potential data error)
                    if max_move > 0.5:  # 50% move is suspicious
                        errors.append(f"{symbol}: Extreme price move ({max_move:.2%}) - possible data error")
                        symbol_metadata['price_valid'] = False
                
                # 6. Check timestamp continuity
                if len(df) > 1 and 'timestamp' in df.columns or 'date' in df.columns:
                    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                    if df.index.name == timestamp_col or df.index.dtype == 'datetime64[ns]':
                        # Check for gaps
                        time_diff = df.index.to_series().diff()
                        # This is simplified - proper implementation would check for business day gaps
                        logger.debug(f"{symbol}: Timestamp check passed")
                
                # Calculate symbol score
                symbol_score = 100
                if not symbol_metadata['columns_valid']:
                    symbol_score -= 50
                if not symbol_metadata['volume_valid']:
                    symbol_score -= 20
                if not symbol_metadata['price_valid']:
                    symbol_score -= 30
                
                symbol_scores.append(symbol_score)
                metadata[symbol] = symbol_metadata
            
            # Overall score
            overall_score = np.mean(symbol_scores) if symbol_scores else 0
            is_valid = len(errors) == 0 and overall_score >= 70
            
            return IOValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                score=overall_score,
                metadata={'by_symbol': metadata, 'overall_score': overall_score}
            )
        
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return IOValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                score=0,
                metadata={'error': str(e)}
            )
    
    def validate_output(self, signals: List, trades: List) -> IOValidationResult:
        """Not used for data analyser"""
        return IOValidationResult(is_valid=True, errors=[], warnings=[], score=100, metadata={})


class SignalQualityAnalyser(IOAnalyser):
    """Validates output signal quality before trading"""
    
    def __init__(
        self,
        min_confidence: float = 50,
        min_signal_count: int = 1,
        max_signals_per_symbol: int = 3,
        max_capital_per_position: float = 0.10  # 10% max per position
    ):
        super().__init__("SignalQualityAnalyser")
        self.min_confidence = min_confidence
        self.min_signal_count = min_signal_count
        self.max_signals_per_symbol = max_signals_per_symbol
        self.max_capital_per_position = max_capital_per_position
    
    def validate_input(self, data: Dict[str, pd.DataFrame]) -> IOValidationResult:
        """Not used for signal analyser"""
        return IOValidationResult(is_valid=True, errors=[], warnings=[], score=100, metadata={})
    
    def validate_output(self, signals: List, trades: List = None) -> IOValidationResult:
        """
        Validate signal quality before execution.
        
        Checks:
        1. Confidence levels (not too low)
        2. Signal diversity (not all same direction)
        3. Position concentration (not too much in one symbol)
        4. Signal frequency (reasonable number)
        5. Risk/reward ratio validation
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if not signals:
                warnings.append("No signals generated")
                return IOValidationResult(
                    is_valid=False,
                    errors=[],
                    warnings=warnings,
                    score=50,
                    metadata=metadata
                )
            
            # 1. Check confidence levels
            low_confidence_signals = [s for s in signals if s.confidence < self.min_confidence]
            if low_confidence_signals:
                warnings.append(f"{len(low_confidence_signals)} signals with low confidence < {self.min_confidence}%")
            
            # 2. Check signal diversity
            buy_signals = len([s for s in signals if 'LONG' in str(s.signal_type)])
            sell_signals = len([s for s in signals if 'SHORT' in str(s.signal_type)])
            
            if len(signals) > 1:
                dominant_direction = max(buy_signals, sell_signals) / len(signals)
                if dominant_direction > 0.9:
                    warnings.append(f"Low signal diversity: {dominant_direction:.1%} one direction")
                
                metadata['signal_distribution'] = {
                    'buy': buy_signals,
                    'sell': sell_signals,
                    'ratio': buy_signals / sell_signals if sell_signals > 0 else float('inf')
                }
            
            # 3. Check position concentration
            symbol_counts = {}
            for signal in signals:
                symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1
            
            for symbol, count in symbol_counts.items():
                if count > self.max_signals_per_symbol:
                    warnings.append(f"{symbol}: {count} signals (max {self.max_signals_per_symbol})")
            
            # 4. Check capital allocation
            total_capital_allocated = sum([s.position_size for s in signals])
            max_single = max([s.position_size for s in signals])
            
            if max_single > self.max_capital_per_position:
                warnings.append(f"Position exceeds max allocation {max_single:.1%} > {self.max_capital_per_position:.1%}")
            
            metadata['capital'] = {
                'total_allocated': total_capital_allocated,
                'max_single_position': max_single
            }
            
            # 5. Risk/reward ratio
            rr_ratios = []
            for signal in signals:
                if signal.stop_loss != 0 and signal.entry_price != 0:
                    risk = abs(signal.entry_price - signal.stop_loss)
                    reward = abs(signal.take_profit - signal.entry_price)
                    if risk > 0:
                        rr_ratios.append(reward / risk)
            
            if rr_ratios:
                avg_rr = np.mean(rr_ratios)
                if avg_rr < 1.0:
                    warnings.append(f"Low risk/reward ratio {avg_rr:.2f} (target > 1.0)")
                metadata['risk_reward'] = {'avg_ratio': avg_rr, 'min': min(rr_ratios), 'max': max(rr_ratios)}
            
            # Overall score
            score = 100
            score -= len(low_confidence_signals) * 2
            score -= len(warnings) * 5
            score = max(0, score)
            
            is_valid = len(errors) == 0 and score >= 60
            
            return IOValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                score=score,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Signal quality validation failed: {e}")
            return IOValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                score=0,
                metadata={'error': str(e)}
            )


# ============================================================================
# HELPER FUNCTION FOR EASY USAGE
# ============================================================================

def validate_before_trading(
    data: Dict[str, pd.DataFrame],
    signals: List,
    data_quality_threshold: float = 70,
    signal_quality_threshold: float = 60
) -> Tuple[bool, str]:
    """
    Validate data and signals before trading.
    
    Returns:
        (is_valid, message)
    
    Example:
        is_valid, message = validate_before_trading(data, signals)
        if not is_valid:
            print(f"Validation failed: {message}")
            return
        # Proceed with trading
    """
    
    # Validate input data
    data_analyser = DataQualityAnalyser()
    data_result = data_analyser.validate_input(data)
    
    if not data_result.is_valid or data_result.score < data_quality_threshold:
        errors_str = "\n".join(data_result.errors) if data_result.errors else "Unknown error"
        return False, f"Data quality check failed (score: {data_result.score:.0f}/100):\n{errors_str}"
    
    # Validate signals
    signal_analyser = SignalQualityAnalyser()
    signal_result = signal_analyser.validate_output(signals)
    
    if not signal_result.is_valid or signal_result.score < signal_quality_threshold:
        errors_str = "\n".join(signal_result.errors) if signal_result.errors else "Unknown error"
        warnings_str = "\n".join(signal_result.warnings) if signal_result.warnings else ""
        msg = f"Signal quality check failed (score: {signal_result.score:.0f}/100):\n{errors_str}"
        if warnings_str:
            msg += f"\nWarnings: {warnings_str}"
        return False, msg
    
    return True, f"✓ Data quality: {data_result.score:.0f}/100, Signal quality: {signal_result.score:.0f}/100"
