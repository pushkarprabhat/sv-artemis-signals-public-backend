"""
utils/scan_recommendation_classifier.py - Classify scan recommendations with market segment info
Tags recommendations with derivative status and strategy type
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MarketSegment(Enum):
    """Market segments"""
    STOCK_WITH_OPTIONS_AND_FUTURES = "Stock+Options+Futures"
    STOCK_WITH_OPTIONS = "Stock+Options"
    STOCK_WITH_FUTURES = "Stock+Futures"
    STOCK_ONLY = "Stock-Only"
    INDEX = "Index"
    UNKNOWN = "Unknown"

class StrategyType(Enum):
    """Strategy types for classification"""
    PAIR_TRADING = "Pair Trading (Stock-Pairs)"
    INDEX_CONSTITUENT = "Strategy (Index-Constituent Pairs)"
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "Mean Reversion"
    VOLATILITY = "Volatility"
    KELLY = "Kelly Criterion"
    OPTIONS = "Options Strategies"
    GREEKS = "Greeks-Based"
    BACKSPREAD = "Backspread"

@dataclass
class RecommendationInfo:
    """Enhanced recommendation information"""
    symbol: str
    strategy: str
    signal: str  # BUY, SELL, HOLD
    score: float
    derivative_status: str
    has_options: bool
    has_futures: bool
    is_index: bool
    market_segment: str
    is_pair_trading: bool
    is_index_pair: bool  # Is this part of index-constituent pair?
    pair_partner: Optional[str] = None
    entry_note: Optional[str] = None
    
    def get_trading_vehicles(self) -> List[str]:
        """Get available trading vehicles for this recommendation"""
        vehicles = ['Stock']
        if self.has_options:
            vehicles.append('Options')
        if self.has_futures:
            vehicles.append('Futures')
        return vehicles
    
    def get_recommendation_note(self) -> str:
        """Get formatted recommendation note with market segment info"""
        vehicles = ", ".join(self.get_trading_vehicles())
        
        if self.is_pair_trading:
            if self.is_index_pair:
                return (f"{self.strategy} | {self.signal} {self.symbol} "
                       f"(Index-Constituent) | Vehicles: {vehicles}")
            else:
                return (f"{self.strategy} | {self.signal} {self.symbol} "
                       f"(Stock-Pair) | Vehicles: {vehicles}")
        else:
            return (f"{self.strategy} | {self.signal} {self.symbol} "
                   f"({self.market_segment}) | Vehicles: {vehicles}")

class ScanRecommendationClassifier:
    """Classify scan results with derivative status and strategy information"""
    
    def __init__(self, derivatives_manager=None):
        """
        Initialize classifier
        
        Args:
            derivatives_manager: NSEDerivativesManager instance or None to create new
        """
        if derivatives_manager is None:
            from utils.nse_derivatives_manager import get_derivatives_manager
            self.derivatives_manager = get_derivatives_manager()
        else:
            self.derivatives_manager = derivatives_manager
        
        self.nifty500_df = None
        self._load_nifty500()
    
    def _load_nifty500(self):
        """Load NIFTY500 constituents"""
        try:
            self.nifty500_df = self.derivatives_manager.get_nifty500_with_derivatives()
        except Exception as e:
            logger.warning(f"Could not load NIFTY500: {e}")
            self.nifty500_df = pd.DataFrame()
    
    def classify_scan_result(self, symbol: str, strategy: str, signal: str,
                            score: float = 0.0, pair_partner: str = None) -> RecommendationInfo:
        """
        Classify a single scan result
        
        Args:
            symbol: Stock/index symbol
            strategy: Strategy name (e.g., 'Pair Trading', 'Momentum')
            signal: BUY, SELL, HOLD
            score: Signal strength (0-1)
            pair_partner: If pair trading, the partner symbol
        
        Returns:
            RecommendationInfo with full classification
        """
        symbol = symbol.strip().upper()
        
        # Get derivative classification
        classification = self.derivatives_manager.classify_recommendation(symbol, strategy)
        
        # Determine if index-constituent pair
        is_index_pair = False
        is_pair = False
        
        if 'Pair' in strategy:
            is_pair = True
            if pair_partner:
                # Check if one is index and other is constituent
                is_partner_index = self.derivatives_manager.symbol_classification.get(
                    pair_partner.upper(), {}
                ).get('is_index', False)
                
                is_symbol_index = classification.get('is_index', False)
                
                if (is_partner_index and not is_symbol_index) or \
                   (is_symbol_index and not is_partner_index):
                    is_index_pair = True
        
        # Get market segment
        market_segment = classification.get('market_segment', 'Unknown')
        
        return RecommendationInfo(
            symbol=symbol,
            strategy=strategy,
            signal=signal,
            score=score,
            derivative_status=classification.get('derivative_status', 'Unknown'),
            has_options=classification.get('has_options', False),
            has_futures=classification.get('has_futures', False),
            is_index=classification.get('is_index', False),
            market_segment=market_segment,
            is_pair_trading=is_pair,
            is_index_pair=is_index_pair,
            pair_partner=pair_partner,
        )
    
    def classify_scan_results(self, results_df: pd.DataFrame, strategy: str,
                             pair_partners: Dict[str, str] = None) -> pd.DataFrame:
        """
        Classify scan results DataFrame
        
        Args:
            results_df: Scan results with columns: symbol, signal, score (optional)
            strategy: Strategy name
            pair_partners: Dict mapping symbol to pair partner
        
        Returns:
            Enhanced DataFrame with classification columns
        """
        if results_df.empty:
            return results_df
        
        pair_partners = pair_partners or {}
        
        # Classify each result
        classified = []
        for _, row in results_df.iterrows():
            symbol = row['symbol'] if 'symbol' in row else row.get(0)
            signal = row.get('signal', row.get('signal_type', 'HOLD'))
            score = row.get('score', row.get('strength', 0.0))
            pair_partner = pair_partners.get(symbol)
            
            rec_info = self.classify_scan_result(symbol, strategy, signal, score, pair_partner)
            
            classified.append({
                'symbol': rec_info.symbol,
                'strategy': rec_info.strategy,
                'signal': rec_info.signal,
                'score': rec_info.score,
                'derivative_status': rec_info.derivative_status,
                'has_options': rec_info.has_options,
                'has_futures': rec_info.has_futures,
                'is_index': rec_info.is_index,
                'market_segment': rec_info.market_segment,
                'is_pair_trading': rec_info.is_pair_trading,
                'is_index_pair': rec_info.is_index_pair,
                'trading_vehicles': ', '.join(rec_info.get_trading_vehicles()),
                'recommendation_note': rec_info.get_recommendation_note(),
                **{k: v for k, v in row.items() if k != 'symbol'}  # Keep other columns
            })
        
        return pd.DataFrame(classified)
    
    def filter_by_derivative_status(self, results_df: pd.DataFrame,
                                   statuses: List[str] = None) -> pd.DataFrame:
        """
        Filter results by derivative status
        
        Args:
            results_df: Classified results DataFrame
            statuses: List of statuses to include (e.g., ['Stock+Options', 'Stock+Futures'])
                     If None, includes all statuses
        
        Returns:
            Filtered DataFrame
        """
        if statuses is None or 'derivative_status' not in results_df.columns:
            return results_df
        
        return results_df[results_df['derivative_status'].isin(statuses)]
    
    def filter_pair_trading(self, results_df: pd.DataFrame,
                           include_index_pairs: bool = True) -> pd.DataFrame:
        """
        Filter for pair trading recommendations
        
        Args:
            results_df: Classified results DataFrame
            include_index_pairs: Include index-constituent pairs
        
        Returns:
            Filtered pair trading results
        """
        if 'is_pair_trading' not in results_df.columns:
            return pd.DataFrame()
        
        df = results_df[results_df['is_pair_trading'] == True]
        
        if not include_index_pairs:
            df = df[df['is_index_pair'] == False]
        
        return df
    
    def filter_index_constituent_pairs(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Get only index-constituent pair trading results"""
        if 'is_index_pair' not in results_df.columns:
            return pd.DataFrame()
        
        return results_df[results_df['is_index_pair'] == True]
    
    def get_summary_stats(self, results_df: pd.DataFrame) -> Dict:
        """Get classification summary statistics"""
        if results_df.empty:
            return {}
        
        stats = {
            'total_recommendations': len(results_df),
        }
        
        # By derivative status
        if 'derivative_status' in results_df.columns:
            stats['by_derivative_status'] = results_df['derivative_status'].value_counts().to_dict()
        
        # By market segment
        if 'market_segment' in results_df.columns:
            stats['by_market_segment'] = results_df['market_segment'].value_counts().to_dict()
        
        # Pair trading breakdown
        if 'is_pair_trading' in results_df.columns:
            pairs = results_df[results_df['is_pair_trading'] == True]
            stats['total_pair_trading'] = len(pairs)
            
            if 'is_index_pair' in results_df.columns:
                stats['index_constituent_pairs'] = len(pairs[pairs['is_index_pair'] == True])
                stats['stock_pair_trading'] = len(pairs[pairs['is_index_pair'] == False])
        
        # By signal
        if 'signal' in results_df.columns:
            stats['by_signal'] = results_df['signal'].value_counts().to_dict()
        
        return stats

# Global instance
_classifier_instance = None

def get_classifier() -> ScanRecommendationClassifier:
    """Get or create global classifier"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ScanRecommendationClassifier()
    return _classifier_instance

# Convenience functions
def classify_result(symbol: str, strategy: str, signal: str,
                   score: float = 0.0, pair_partner: str = None) -> RecommendationInfo:
    """Classify a single scan result"""
    return get_classifier().classify_scan_result(symbol, strategy, signal, score, pair_partner)

def classify_results(results_df: pd.DataFrame, strategy: str,
                    pair_partners: Dict[str, str] = None) -> pd.DataFrame:
    """Classify scan results"""
    return get_classifier().classify_scan_results(results_df, strategy, pair_partners)

def get_nifty500_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
    """Get NIFTY500 stock pair trading results"""
    classifier = get_classifier()
    classified = classifier.classify_scan_results(results_df, 'Pair Trading')
    return classifier.filter_pair_trading(classified, include_index_pairs=False)

def get_index_constituent_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
    """Get index-constituent pair trading results"""
    classifier = get_classifier()
    classified = classifier.classify_scan_results(results_df, 'Pair Trading')
    return classifier.filter_index_constituent_pairs(classified)
