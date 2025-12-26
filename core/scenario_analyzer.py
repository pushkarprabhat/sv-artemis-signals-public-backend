"""
Scenario Analyzer - Strategic Trading Decision Support

Analyzes market conditions and recommends optimal trading strategies based on:
- Market volatility regimes
- Trend direction and strength
- Correlation dynamics
- Relative valuation metrics
- Risk-reward characteristics

Helps traders decide when to use:
- Pair trading vs trend following
- Index hedging vs long-only strategies
- Mean reversion vs momentum strategies
- Technical signals vs fundamental factors
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market condition regimes"""
    HIGH_VOL_TREND = "high_volatility_trending"      # High vol + strong trend
    HIGH_VOL_CHOP = "high_volatility_choppy"         # High vol + no direction
    LOW_VOL_TREND = "low_volatility_trending"        # Low vol + strong trend
    LOW_VOL_RANGE = "low_volatility_range_bound"     # Low vol + range-bound
    CRASH_REGIME = "crash_regime"                     # Sharp decline


class CorrelationRegime(Enum):
    """Index/stock correlation regimes"""
    HIGH_CORRELATION = "high_correlation"             # Normal hedging poor
    LOW_CORRELATION = "low_correlation"               # Good hedging
    NEGATIVE_CORRELATION = "negative_correlation"    # Perfect hedge
    BREAKING_DOWN = "correlation_breaking_down"      # Unusual behavior


class RecommendedStrategy(Enum):
    """Recommended trading strategies for each regime"""
    TREND_FOLLOWING = "trend_following"              # Follow momentum
    MEAN_REVERSION = "mean_reversion"                # Revert to mean
    PAIR_TRADING = "pair_trading"                    # Long-short neutral
    HEDGED_LONG = "hedged_long"                      # Long with hedge
    CASH_DEFENSIVE = "cash_defensive"                # Reduce exposure


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics"""
    current_vol: float                # Current annualized volatility
    vol_ma20: float                   # 20-day MA of volatility
    vol_ma60: float                   # 60-day MA of volatility
    vol_regime: str                   # High/Low relative to MA60
    vol_percentile: float             # Current vol percentile (0-100)
    vol_trend: str                    # Increasing/Stable/Decreasing


@dataclass
class TrendMetrics:
    """Trend analysis metrics"""
    price: float
    sma50: float
    sma200: float
    trend_strength: float             # 0-100, higher = stronger
    trend_direction: str              # UP/DOWN/SIDEWAYS
    distance_from_ma200: float        # % above/below MA200


@dataclass
class CorrelationMetrics:
    """Correlation analysis for pair trading"""
    current_correlation: float
    correlation_ma30: float           # 30-day MA
    correlation_deviation: float      # Current vs MA
    pair_spread_zscore: float        # How far from mean
    hedge_quality: str                # Excellent/Good/Fair/Poor


@dataclass
class ScenarioRecommendation:
    """Complete trading recommendation for current market conditions"""
    market_regime: MarketRegime
    correlation_regime: CorrelationRegime
    primary_strategy: RecommendedStrategy
    secondary_strategy: Optional[RecommendedStrategy]
    confidence: float                 # 0-100
    risk_level: str                   # Low/Medium/High/Very High
    position_size: str                # Micro/Small/Medium/Large
    
    # Detailed metrics
    volatility: VolatilityMetrics
    trend: TrendMetrics
    correlation: CorrelationMetrics
    
    # Reasoning
    reasoning: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display/logging"""
        return {
            'market_regime': self.market_regime.value,
            'correlation_regime': self.correlation_regime.value,
            'primary_strategy': self.primary_strategy.value,
            'secondary_strategy': self.secondary_strategy.value if self.secondary_strategy else None,
            'confidence': f"{self.confidence:.0f}%",
            'risk_level': self.risk_level,
            'position_size': self.position_size,
            'vol_current': f"{self.volatility.current_vol:.1%}",
            'vol_percentile': f"{self.volatility.vol_percentile:.0f}",
            'trend_direction': self.trend.trend_direction,
            'trend_strength': f"{self.trend.trend_strength:.0f}%",
            'correlation': f"{self.correlation.current_correlation:.2f}",
            'pair_spread_zscore': f"{self.correlation.pair_spread_zscore:.2f}",
            'reasoning': self.reasoning,
            'warnings': self.warnings,
        }


class ScenarioAnalyzer:
    """
    Analyzes market conditions and recommends trading strategies
    
    Usage:
        analyzer = ScenarioAnalyzer()
        recommendation = analyzer.analyze(nifty_df, banknifty_df)
        print(recommendation.primary_strategy)
    """
    
    # Volatility thresholds (annualized)
    VOL_LOW = 0.10      # < 10% annual
    VOL_HIGH = 0.25     # > 25% annual
    
    # Correlation thresholds
    CORR_HIGH = 0.75    # Strong positive
    CORR_LOW = 0.40     # Weak/moderate
    CORR_NEG = 0.00     # Negative
    
    # Z-score thresholds for pair spread
    ZSCORE_EXTREME = 2.5
    ZSCORE_HIGH = 2.0
    ZSCORE_NORMAL = 1.0
    
    def analyze(self,
                index1_df: pd.DataFrame,
                index2_df: pd.DataFrame,
                index1_name: str = "NIFTY50",
                index2_name: str = "BANKNIFTY") -> ScenarioRecommendation:
        """
        Analyze market conditions and recommend trading strategy
        
        Args:
            index1_df: DataFrame with OHLCV for first index
            index2_df: DataFrame with OHLCV for second index
            index1_name: Name of first index (for display)
            index2_name: Name of second index (for display)
        
        Returns:
            ScenarioRecommendation with detailed analysis
        """
        # Calculate all metrics
        vol_metrics = self._analyze_volatility(index1_df)
        trend_metrics = self._analyze_trend(index1_df)
        corr_metrics = self._analyze_correlation(index1_df, index2_df)
        
        # Determine regimes
        market_regime = self._get_market_regime(vol_metrics, trend_metrics)
        corr_regime = self._get_correlation_regime(corr_metrics)
        
        # Recommend strategy
        primary_strategy, secondary_strategy = self._recommend_strategy(
            market_regime, corr_regime, vol_metrics, trend_metrics
        )
        
        # Calculate confidence and position sizing
        confidence = self._calculate_confidence(
            vol_metrics, trend_metrics, corr_metrics
        )
        risk_level = self._assess_risk_level(vol_metrics, trend_metrics)
        position_size = self._recommend_position_size(vol_metrics, risk_level)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            market_regime, corr_regime, vol_metrics, trend_metrics, corr_metrics
        )
        warnings = self._generate_warnings(vol_metrics, trend_metrics, corr_metrics)
        
        return ScenarioRecommendation(
            market_regime=market_regime,
            correlation_regime=corr_regime,
            primary_strategy=primary_strategy,
            secondary_strategy=secondary_strategy,
            confidence=confidence,
            risk_level=risk_level,
            position_size=position_size,
            volatility=vol_metrics,
            trend=trend_metrics,
            correlation=corr_metrics,
            reasoning=reasoning,
            warnings=warnings,
        )
    
    def _analyze_volatility(self, df: pd.DataFrame) -> VolatilityMetrics:
        """Calculate volatility metrics"""
        # Annualized volatility
        returns = df['close'].pct_change()
        current_vol = returns.std() * np.sqrt(252)
        
        # Volatility moving averages
        vol_series = returns.rolling(window=20).std() * np.sqrt(252)
        vol_ma20 = vol_series.rolling(window=20).mean().iloc[-1]
        vol_ma60 = vol_series.rolling(window=60).mean().iloc[-1]
        
        # Volatility percentile
        vol_percentile = (vol_series <= current_vol).sum() / len(vol_series) * 100
        
        # Regime and trend
        vol_regime = "High" if current_vol > self.VOL_HIGH else "Low" if current_vol < self.VOL_LOW else "Medium"
        vol_trend = self._get_vol_trend(vol_series)
        
        return VolatilityMetrics(
            current_vol=current_vol,
            vol_ma20=vol_ma20,
            vol_ma60=vol_ma60,
            vol_regime=vol_regime,
            vol_percentile=vol_percentile,
            vol_trend=vol_trend,
        )
    
    def _analyze_trend(self, df: pd.DataFrame) -> TrendMetrics:
        """Calculate trend metrics"""
        price = df['close'].iloc[-1]
        sma50 = df['close'].rolling(window=50).mean().iloc[-1]
        sma200 = df['close'].rolling(window=200).mean().iloc[-1]
        
        # Trend strength using position relative to bands
        highest_50 = df['close'].rolling(window=50).max().iloc[-1]
        lowest_50 = df['close'].rolling(window=50).min().iloc[-1]
        band_width = highest_50 - lowest_50
        trend_strength = abs(price - sma50) / band_width * 100 if band_width > 0 else 0
        trend_strength = min(100, trend_strength)
        
        # Trend direction
        if price > sma50 > sma200:
            trend_direction = "UP"
        elif price < sma50 < sma200:
            trend_direction = "DOWN"
        else:
            trend_direction = "SIDEWAYS"
        
        # Distance from MA200
        distance = (price - sma200) / sma200 * 100
        
        return TrendMetrics(
            price=price,
            sma50=sma50,
            sma200=sma200,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            distance_from_ma200=distance,
        )
    
    def _analyze_correlation(self,
                            index1_df: pd.DataFrame,
                            index2_df: pd.DataFrame) -> CorrelationMetrics:
        """Calculate correlation metrics for pair trading"""
        # Calculate correlation
        returns1 = index1_df['close'].pct_change()
        returns2 = index2_df['close'].pct_change()
        
        current_corr = returns1.corr(returns2)
        corr_ma30 = returns1.rolling(window=30).corr(returns2).rolling(window=10).mean().iloc[-1]
        
        deviation = current_corr - corr_ma30
        
        # Pair spread Z-score
        spread = index1_df['close'] / index2_df['close']
        spread_mean = spread.rolling(window=30).mean().iloc[-1]
        spread_std = spread.rolling(window=30).std().iloc[-1]
        zscore = (spread.iloc[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Hedge quality assessment
        if abs(current_corr) > 0.85:
            hedge_quality = "Excellent"
        elif abs(current_corr) > 0.70:
            hedge_quality = "Good"
        elif abs(current_corr) > 0.50:
            hedge_quality = "Fair"
        else:
            hedge_quality = "Poor"
        
        return CorrelationMetrics(
            current_correlation=current_corr,
            correlation_ma30=corr_ma30,
            correlation_deviation=deviation,
            pair_spread_zscore=zscore,
            hedge_quality=hedge_quality,
        )
    
    def _get_market_regime(self,
                          vol_metrics: VolatilityMetrics,
                          trend_metrics: TrendMetrics) -> MarketRegime:
        """Determine market regime"""
        # Check if trending or choppy
        is_trending = trend_metrics.trend_strength > 40
        is_high_vol = vol_metrics.current_vol > self.VOL_HIGH
        
        if is_high_vol:
            if is_trending:
                return MarketRegime.HIGH_VOL_TREND
            else:
                return MarketRegime.HIGH_VOL_CHOP
        else:
            if is_trending:
                return MarketRegime.LOW_VOL_TREND
            else:
                return MarketRegime.LOW_VOL_RANGE
    
    def _get_correlation_regime(self, corr_metrics: CorrelationMetrics) -> CorrelationRegime:
        """Determine correlation regime"""
        corr = corr_metrics.current_correlation
        
        if corr < 0:
            return CorrelationRegime.NEGATIVE_CORRELATION
        elif corr < self.CORR_LOW:
            return CorrelationRegime.LOW_CORRELATION
        elif corr < self.CORR_HIGH:
            return CorrelationRegime.BREAKING_DOWN  # Weakening
        else:
            return CorrelationRegime.HIGH_CORRELATION
    
    def _recommend_strategy(self,
                           market_regime: MarketRegime,
                           corr_regime: CorrelationRegime,
                           vol_metrics: VolatilityMetrics,
                           trend_metrics: TrendMetrics) -> Tuple[RecommendedStrategy, Optional[RecommendedStrategy]]:
        """Recommend primary and secondary strategies"""
        
        if market_regime == MarketRegime.HIGH_VOL_TREND:
            # Use trend following
            primary = RecommendedStrategy.TREND_FOLLOWING
            secondary = RecommendedStrategy.HEDGED_LONG
            
        elif market_regime == MarketRegime.HIGH_VOL_CHOP:
            # Use mean reversion or hedging
            primary = RecommendedStrategy.MEAN_REVERSION
            secondary = RecommendedStrategy.HEDGED_LONG
            
        elif market_regime == MarketRegime.LOW_VOL_TREND:
            # Trend following with larger positions
            primary = RecommendedStrategy.TREND_FOLLOWING
            secondary = RecommendedStrategy.PAIR_TRADING if corr_regime in [
                CorrelationRegime.LOW_CORRELATION,
                CorrelationRegime.NEGATIVE_CORRELATION
            ] else None
            
        elif market_regime == MarketRegime.LOW_VOL_RANGE:
            # Mean reversion ideal
            primary = RecommendedStrategy.MEAN_REVERSION
            secondary = RecommendedStrategy.PAIR_TRADING
            
        else:  # CRASH_REGIME
            # Defensive
            primary = RecommendedStrategy.CASH_DEFENSIVE
            secondary = RecommendedStrategy.HEDGED_LONG
        
        return primary, secondary
    
    def _calculate_confidence(self,
                             vol_metrics: VolatilityMetrics,
                             trend_metrics: TrendMetrics,
                             corr_metrics: CorrelationMetrics) -> float:
        """Calculate confidence in recommendation"""
        # Higher confidence when:
        # 1. Strong trend
        # 2. Stable volatility
        # 3. Clear correlation regime
        
        trend_score = trend_metrics.trend_strength
        vol_stability = 100 - abs(vol_metrics.vol_ma20 - vol_metrics.vol_ma60) / vol_metrics.vol_ma60 * 100
        corr_stability = 100 - abs(corr_metrics.correlation_deviation) * 100
        
        confidence = (trend_score + vol_stability + corr_stability) / 3
        return min(100, max(0, confidence))
    
    def _assess_risk_level(self,
                          vol_metrics: VolatilityMetrics,
                          trend_metrics: TrendMetrics) -> str:
        """Assess current risk level"""
        vol = vol_metrics.current_vol
        
        if vol > 0.35:
            return "Very High"
        elif vol > 0.25:
            return "High"
        elif vol > 0.15:
            return "Medium"
        else:
            return "Low"
    
    def _recommend_position_size(self, vol_metrics: VolatilityMetrics, risk_level: str) -> str:
        """Recommend position sizing based on volatility"""
        if risk_level == "Very High":
            return "Micro (0.5% account)"
        elif risk_level == "High":
            return "Small (1-2% account)"
        elif risk_level == "Medium":
            return "Medium (2-5% account)"
        else:
            return "Large (5-10% account)"
    
    def _get_vol_trend(self, vol_series: pd.Series) -> str:
        """Determine if volatility is trending up, down, or stable"""
        recent_vol = vol_series.iloc[-20:].mean()
        older_vol = vol_series.iloc[-60:-20].mean()
        
        change = (recent_vol - older_vol) / older_vol
        
        if change > 0.10:
            return "Increasing"
        elif change < -0.10:
            return "Decreasing"
        else:
            return "Stable"
    
    def _generate_reasoning(self,
                           market_regime: MarketRegime,
                           corr_regime: CorrelationRegime,
                           vol_metrics: VolatilityMetrics,
                           trend_metrics: TrendMetrics,
                           corr_metrics: CorrelationMetrics) -> List[str]:
        """Generate human-readable reasoning for the recommendation"""
        reasons = []
        
        # Market regime reasoning
        if market_regime == MarketRegime.HIGH_VOL_TREND:
            reasons.append(f"High volatility ({vol_metrics.current_vol:.1%}) with strong {trend_metrics.trend_direction} trend")
            reasons.append("Trend-following strategies well-suited for capturing momentum")
        elif market_regime == MarketRegime.HIGH_VOL_CHOP:
            reasons.append(f"High volatility ({vol_metrics.current_vol:.1%}) without clear direction")
            reasons.append("Mean-reversion opportunities likely")
        elif market_regime == MarketRegime.LOW_VOL_TREND:
            reasons.append(f"Low volatility ({vol_metrics.current_vol:.1%}) with {trend_metrics.trend_direction} trend")
            reasons.append("Stable conditions allow larger positions")
        elif market_regime == MarketRegime.LOW_VOL_RANGE:
            reasons.append(f"Low volatility ({vol_metrics.current_vol:.1%}), range-bound movement")
            reasons.append("Pair trading and mean-reversion effective")
        
        # Correlation reasoning
        if corr_metrics.current_correlation > 0.85:
            reasons.append(f"Indices highly correlated ({corr_metrics.current_correlation:.2f}) - hedging less effective")
        elif corr_metrics.current_correlation < 0.40:
            reasons.append(f"Indices weakly correlated ({corr_metrics.current_correlation:.2f}) - excellent hedging opportunity")
        
        # Distance from MA reasoning
        if trend_metrics.distance_from_ma200 > 10:
            reasons.append(f"Price {trend_metrics.distance_from_ma200:.1f}% above 200-day MA - extended move")
        elif trend_metrics.distance_from_ma200 < -10:
            reasons.append(f"Price {trend_metrics.distance_from_ma200:.1f}% below 200-day MA - potential reversal")
        
        return reasons
    
    def _generate_warnings(self,
                          vol_metrics: VolatilityMetrics,
                          trend_metrics: TrendMetrics,
                          corr_metrics: CorrelationMetrics) -> List[str]:
        """Generate warnings about current market conditions"""
        warnings = []
        
        # Volatility warnings
        if vol_metrics.vol_percentile > 90:
            warnings.append(f"⚠️ Volatility at 90th percentile - extreme conditions")
        
        if vol_metrics.vol_trend == "Increasing":
            warnings.append("⚠️ Volatility trending up - expect larger moves")
        
        # Distance warnings
        if abs(trend_metrics.distance_from_ma200) > 20:
            warnings.append("⚠️ Price far from 200-day MA - high reversal risk")
        
        # Correlation warnings
        if abs(corr_metrics.pair_spread_zscore) > self.ZSCORE_EXTREME:
            warnings.append(f"⚠️ Pair spread at extreme ({corr_metrics.pair_spread_zscore:.2f} SD) - mean reversion risk")
        
        # Trend confidence warnings
        if trend_metrics.trend_strength < 20:
            warnings.append("⚠️ Weak trend - signals less reliable")
        
        return warnings
