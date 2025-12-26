# core/position_recommender.py
"""
Position Recommendation Engine
Scores and ranks all signals for optimal position selection
Parameters: 75% confidence threshold, 15+ positions per day, confidence-weighted sizing
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
from utils.logger import logger


@dataclass
class RecommendedPosition:
    """A recommended trading position"""
    rank: int
    symbol: str
    strategy: str
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    sizing: float  # 0-1 position size multiplier (confidence-weighted)
    reasoning: str
    agreement_bonus: float = 0.0
    vol_fit: float = 0.0


class PositionRecommender:
    """Score, rank, and recommend trading positions"""
    
    def __init__(self, volatility_analyzer=None):
        self.vol_analyzer = volatility_analyzer
        self.min_confidence = 75.0  # 75% confidence threshold
        self.min_rr = 1.0  # Minimum 1:1 risk/reward
        logger.info(f"[RECOMMENDER] Initialized with {self.min_confidence}% confidence threshold")
    
    def score_signal(self, signal, vol_regime=None, all_signals=None):
        """
        Score a signal from 0-100
        
        Components:
        - Strategy strength: 40 points
        - Multi-strategy agreement: 20 points
        - Risk/reward: 20 points
        - Vol regime fit: 15 points
        - Position quality: 5 points
        """
        score = 0
        details = {}
        
        # 1. Strategy strength (40 points)
        strength_pts = signal.strength * 40
        score += strength_pts
        details['strength'] = strength_pts
        
        # 2. Risk/reward ratio (20 points)
        risk = signal.entry_price - signal.stop_loss
        reward = signal.target_price - signal.entry_price
        rr = 0
        rr_pts = 0
        
        if risk > 0 and reward > 0:
            rr = reward / risk
            if rr >= 3.0:
                rr_pts = 20
            elif rr >= 2.5:
                rr_pts = 18
            elif rr >= 2.0:
                rr_pts = 16
            elif rr >= 1.5:
                rr_pts = 12
            elif rr >= 1.0:
                rr_pts = 8
            else:
                rr_pts = 2
        
        score += rr_pts
        details['rr'] = (rr_pts, rr)
        
        # 3. Volatility regime fit (15 points)
        vol_fit = 0
        if vol_regime and self.vol_analyzer:
            vol_fit = self.vol_analyzer.get_strategy_fit(signal.strategy, vol_regime)
            score += vol_fit * 15
        details['vol_fit'] = vol_fit
        
        # 4. Agreement bonus (20 points) - multiple strategies for same symbol
        agreement = 0
        if all_signals:
            symbol_signals = [s for s in all_signals if s.symbol == signal.symbol]
            strategies = set(s.strategy for s in symbol_signals)
            
            if len(strategies) >= 4:
                agreement = 20  # All 4 agree!
            elif len(strategies) == 3:
                agreement = 15
            elif len(strategies) == 2:
                agreement = 8
            else:
                agreement = 0
        
        score += agreement
        details['agreement'] = agreement
        
        # 5. Position quality (5 points)
        # Bonus for good technical setup (multiple data points favor this)
        quality = 5 if len(details) >= 4 else 3
        score += quality
        details['quality'] = quality
        
        final_score = min(100, score)
        
        return {
            'score': final_score,
            'rr': rr,
            'vol_fit': vol_fit,
            'agreement': agreement,
            'details': details
        }
    
    def calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence (confidence-weighted)
        
        Scales from 0.5x to 2.0x based on confidence:
        - 75%: 0.5x (minimum)
        - 80%: 1.0x (base)
        - 90%: 1.5x (high confidence)
        - 95%+: 2.0x (very high confidence)
        """
        if confidence < self.min_confidence:
            return 0  # Don't trade below threshold
        
        # Scale from 75% (0.5x) to 100% (2.0x)
        # At 75%: size = 0.5
        # At 80%: size = 1.0
        # At 95%: size = 1.5
        # At 100%: size = 2.0
        
        if confidence < 80:
            size = 0.5 + (confidence - 75) * 0.1  # 75-80 range
        elif confidence < 90:
            size = 1.0 + (confidence - 80) * 0.05  # 80-90 range
        else:
            size = 1.5 + (confidence - 90) * 0.05  # 90-100 range
        
        return min(2.0, max(0.5, size))
    
    def rank_positions(self, 
                      signals: List, 
                      vol_regime: str = None,
                      portfolio_positions: Dict = None,
                      top_n: int = 15,
                      max_correlation: float = 0.7) -> List[RecommendedPosition]:
        """
        Rank all signals and return top N recommendations
        
        Args:
            signals: List of UnifiedSignal objects
            vol_regime: Current volatility regime (LOW/MEDIUM/HIGH)
            portfolio_positions: Current holdings (optional)
            top_n: Return top N recommendations (15+ positions)
            max_correlation: Max correlation between positions
        
        Returns:
            List of RecommendedPosition objects, ranked by confidence
        """
        if not signals:
            logger.warning("[RECOMMENDER] No signals to rank")
            return []
        
        recommendations = []
        
        # Score each signal
        for signal in signals:
            score_result = self.score_signal(signal, vol_regime, signals)
            confidence = score_result['score']
            rr = score_result['rr']
            vol_fit = score_result['vol_fit']
            agreement = score_result['agreement']
            
            # Calculate position size (confidence-weighted)
            size = self.calculate_position_size(confidence)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(signal, confidence, agreement, vol_regime, rr)
            
            # Create recommendation
            if confidence >= self.min_confidence:  # Only include if meets threshold
                rec = RecommendedPosition(
                    rank=0,  # Set after sorting
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    confidence=confidence,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    target_price=signal.target_price,
                    risk_reward=rr,
                    sizing=size,
                    reasoning=reasoning,
                    agreement_bonus=agreement,
                    vol_fit=vol_fit * 100  # Convert to percentage
                )
                recommendations.append(rec)
        
        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        # Set ranks and limit to top_n
        for i, rec in enumerate(recommendations[:top_n]):
            rec.rank = i + 1
        
        logger.info(f"[RECOMMENDER] Generated {len(recommendations[:top_n])} recommendations "
                   f"(min confidence: {self.min_confidence}%)")
        
        return recommendations[:top_n]
    
    def _generate_reasoning(self, signal, confidence, agreement, vol_regime, rr):
        """Generate human-readable reason for position"""
        parts = []
        
        # Strategy
        parts.append(f"{signal.strategy.replace('_', ' ').title()}")
        
        # Agreement bonus
        if agreement >= 15:
            parts.append(f"4-strategy agreement")
        elif agreement >= 8:
            parts.append(f"multi-strategy confirmed")
        
        # Vol regime
        if vol_regime:
            fit = "ideal" if vol_regime == "MEDIUM" else "good"
            parts.append(f"{fit} for {vol_regime} vol")
        
        # Risk/reward
        if rr >= 2.0:
            parts.append(f"excellent R:R {rr:.1f}:1")
        elif rr >= 1.5:
            parts.append(f"good R:R {rr:.1f}:1")
        
        return " | ".join(parts)
    
    def format_recommendations(self, 
                              recommendations: List[RecommendedPosition],
                              vol_report: Dict = None) -> str:
        """Format recommendations for display/reporting"""
        output = []
        output.append("=" * 110)
        output.append("ARTEMIS SIGNALS - DAILY POSITION RECOMMENDATIONS")
        output.append("=" * 110)
        output.append("")
        
        # Vol regime info
        if vol_report:
            output.append(f"📊 Market Conditions:")
            output.append(f"   Volatility Regime: {vol_report.get('vol_regime', 'UNKNOWN')}")
            output.append(f"   HV Level: {vol_report.get('hv_average', 0):.1f}%")
            output.append(f"   Data: {vol_report.get('symbols_analyzed', 0)} symbols analyzed")
            output.append("")
        
        # Recommendations table
        output.append(f"{'Rank':<5} {'Symbol':<10} {'Strategy':<20} {'Confidence':<15} "
                     f"{'Size':<6} {'R:R':<7} {'Entry':<10} {'Target':<10} {'Risk':<10}")
        output.append("-" * 110)
        
        total_sizing = 0
        for rec in recommendations:
            sizing_pct = rec.sizing * 100
            total_sizing += sizing_pct
            
            output.append(
                f"{rec.rank:<5} {rec.symbol:<10} {rec.strategy[:19]:<20} "
                f"{rec.confidence:>6.1f}% ({sizing_pct:>5.1f}%) {sizing_pct:>5.1f}x "
                f"{rec.risk_reward:>6.2f}:1 ${rec.entry_price:>8.2f} ${rec.target_price:>8.2f} "
                f"${abs(rec.entry_price - rec.stop_loss):>8.2f}"
            )
        
        output.append("")
        output.append("=" * 110)
        output.append(f"SUMMARY: {len(recommendations)} positions recommended")
        output.append(f"Total Portfolio Sizing: {total_sizing:.1f}% (cumulative position sizes)")
        output.append(f"Confidence Threshold: {self.min_confidence:.0f}%")
        output.append("")
        output.append("POSITION DETAILS:")
        output.append("-" * 110)
        
        for rec in recommendations:
            output.append(f"\n{rec.rank}. {rec.symbol} - {rec.strategy.title()}")
            output.append(f"   Confidence: {rec.confidence:.1f}% | Size: {rec.sizing:.2f}x | R:R: {rec.risk_reward:.2f}:1")
            output.append(f"   Entry: ${rec.entry_price:.2f} | Stop: ${rec.stop_loss:.2f} | Target: ${rec.target_price:.2f}")
            output.append(f"   Risk: ${abs(rec.entry_price - rec.stop_loss):.2f} per share")
            output.append(f"   {rec.reasoning}")
        
        output.append("")
        output.append("=" * 110)
        
        return "\n".join(output)
    
    def export_csv(self, recommendations: List[RecommendedPosition], filename: str) -> bool:
        """Export recommendations to CSV"""
        try:
            import csv
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Rank', 'Symbol', 'Strategy', 'Confidence %', 'Position Size',
                    'Entry Price', 'Stop Loss', 'Target Price', 'Risk/Reward',
                    'Risk per Share', 'Reasoning'
                ])
                
                for rec in recommendations:
                    writer.writerow([
                        rec.rank,
                        rec.symbol,
                        rec.strategy,
                        f"{rec.confidence:.2f}",
                        f"{rec.sizing:.2f}",
                        f"{rec.entry_price:.2f}",
                        f"{rec.stop_loss:.2f}",
                        f"{rec.target_price:.2f}",
                        f"{rec.risk_reward:.2f}:1",
                        f"{abs(rec.entry_price - rec.stop_loss):.2f}",
                        rec.reasoning
                    ])
            
            logger.info(f"[RECOMMENDER] Exported {len(recommendations)} recommendations to {filename}")
            return True
        except Exception as e:
            logger.error(f"[RECOMMENDER] Error exporting recommendations: {e}")
            return False


# Global recommender instance
_recommender = None


def get_recommender(vol_analyzer=None) -> PositionRecommender:
    """Get or create singleton recommender"""
    global _recommender
    if _recommender is None:
        _recommender = PositionRecommender(vol_analyzer)
    return _recommender
