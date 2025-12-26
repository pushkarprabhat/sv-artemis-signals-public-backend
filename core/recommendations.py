"""
Strategy Recommendations with Derivative Support Indicators

This module enhances strategy recommendations by indicating:
1. Instrument type (Stock vs Index)
2. Derivative availability (F&O, F only, no options)
3. Universe eligibility (NIFTY50, NIFTY500, only futures, etc.)
4. Pair trading capability (between indices and constituents)

Usage:
    from core.recommendations import StrategyRecommender
    
    recommender = StrategyRecommender()
    recommendations = recommender.recommend_pair_trades(strategy_results)
    # Returns list with derivative indicators for each position
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from core.derivatives_universe import DerivativesUniverse, DerivativeType, InstrumentType


class StrategyRecommender:
    """
    Generates strategy recommendations with derivative support labels
    
    Indicates for each position:
    - STOCK | F&O     : Stock with both futures and options trading
    - STOCK | F only  : Stock with futures only, no options available
    - INDEX | F&O     : Index with derivatives trading
    - NIFTY50 | F&O   : NIFTY50 constituent
    - NIFTY500 | F    : NIFTY500 constituent (futures only)
    - PAIR | INDEX ↔ STOCK : Pair trading between index and constituent
    """
    
    def __init__(self):
        """Initialize recommender with derivatives universe"""
        self.universe = DerivativesUniverse()
    
    def format_instrument_label(self, symbol: str, pair_partner: Optional[str] = None) -> str:
        """
        Format instrument label with type and derivative info
        
        Args:
            symbol: Stock or index symbol
            pair_partner: Optional partner symbol for pair trading
        
        Returns:
            Formatted label like "[STOCK | F&O]" or "[NIFTY50 ↔ RELIANCE]"
        """
        # Handle pair trading display
        if pair_partner:
            return f"[{symbol} ↔ {pair_partner}]"
        
        # Get derivative support label
        return self.universe.get_recommendation_label(symbol)
    
    def recommend_pair_trades(
        self, 
        strategy_results: pd.DataFrame,
        show_derivative_support: bool = True
    ) -> pd.DataFrame:
        """
        Enhance pair trading recommendations with derivative indicators
        
        Args:
            strategy_results: DataFrame with columns like 'symbol1', 'symbol2', 'z_score', etc.
            show_derivative_support: Add derivative support columns
        
        Returns:
            Enhanced DataFrame with derivative indicators and trading feasibility
        """
        if strategy_results.empty:
            return strategy_results
        
        df = strategy_results.copy()
        
        if show_derivative_support:
            # Add instrument type labels
            df['symbol1_type'] = df['symbol1'].apply(self.universe.get_recommendation_label)
            df['symbol2_type'] = df['symbol2'].apply(self.universe.get_recommendation_label)
            
            # Check if both symbols have futures (minimum for pair trading)
            df['both_have_futures'] = df.apply(
                lambda row: (self.universe.has_futures(row['symbol1']) and 
                           self.universe.has_futures(row['symbol2'])),
                axis=1
            )
            
            # Check if both have options (for options-based pair trading)
            df['both_have_options'] = df.apply(
                lambda row: (self.universe.has_options(row['symbol1']) and 
                           self.universe.has_options(row['symbol2'])),
                axis=1
            )
            
            # Check if it's index-constituent pair trading
            df['index_vs_constituent'] = df.apply(
                lambda row: self._is_index_constituent_pair(row['symbol1'], row['symbol2']),
                axis=1
            )
            
            # Trading feasibility tier (1-5 stars)
            df['trading_feasibility'] = df.apply(
                lambda row: self._calculate_trading_feasibility(row),
                axis=1
            )
            
            # Add recommendation note
            df['recommendation_note'] = df.apply(
                lambda row: self._get_recommendation_note(row),
                axis=1
            )
        
        return df
    
    def _is_index_constituent_pair(self, symbol1: str, symbol2: str) -> str:
        """Check if pair is index-constituent trading opportunity"""
        # Check if one is index and other is constituent
        pairs = []
        
        if self.universe.is_index(symbol1) and self.universe.is_stock(symbol2):
            if self.universe.is_nifty500_constituent(symbol2):
                pairs.append(f"{symbol1} (INDEX) ↔ {symbol2} (CONSTITUENT)")
        
        if self.universe.is_index(symbol2) and self.universe.is_stock(symbol1):
            if self.universe.is_nifty500_constituent(symbol1):
                pairs.append(f"{symbol1} (CONSTITUENT) ↔ {symbol2} (INDEX)")
        
        return " | ".join(pairs) if pairs else ""
    
    def _calculate_trading_feasibility(self, row: pd.Series) -> str:
        """Calculate trading feasibility tier (1-5 stars)"""
        score = 0
        
        # Both have futures: +2
        if row.get('both_have_futures', False):
            score += 2
        
        # Both have options: +1
        if row.get('both_have_options', False):
            score += 1
        
        # NIFTY50/NIFTY500 constituents: +1
        if (self.universe.is_nifty50_constituent(row['symbol1']) or 
            self.universe.is_nifty50_constituent(row['symbol2'])):
            score += 1
        
        # Index-constituent pair: +1
        if row.get('index_vs_constituent'):
            score += 1
        
        # Convert to stars (0-5)
        stars = min(5, max(1, score))
        return "★" * stars + "☆" * (5 - stars)
    
    def _get_recommendation_note(self, row: pd.Series) -> str:
        """Generate recommendation note based on derivative support"""
        notes = []
        
        # Derivative support
        if row.get('both_have_futures') and row.get('both_have_options'):
            notes.append("✓ F&O tradeable")
        elif row.get('both_have_futures'):
            notes.append("✓ Futures tradeable")
        else:
            notes.append("✗ No derivative support")
        
        # Universe tier
        if self.universe.is_nifty50_constituent(row['symbol1']) and \
           self.universe.is_nifty50_constituent(row['symbol2']):
            notes.append("NIFTY50 pair")
        elif self.universe.is_nifty500_constituent(row['symbol1']) and \
             self.universe.is_nifty500_constituent(row['symbol2']):
            notes.append("NIFTY500 pair")
        
        # Index-constituent
        if row.get('index_vs_constituent'):
            notes.append("Index-constituent pair")
        
        return " | ".join(notes)
    
    def get_recommended_positions(
        self,
        strategy_results: pd.DataFrame,
        min_feasibility_stars: int = 3
    ) -> pd.DataFrame:
        """
        Get only recommended positions with sufficient derivative support
        
        Args:
            strategy_results: Strategy results DataFrame
            min_feasibility_stars: Minimum feasibility rating (1-5)
        
        Returns:
            Filtered DataFrame with high-feasibility positions only
        """
        recommendations = self.recommend_pair_trades(strategy_results)
        
        # Count stars
        def count_stars(s):
            return s.count('★')
        
        recommendations['stars'] = recommendations['trading_feasibility'].apply(count_stars)
        
        # Filter
        recommended = recommendations[
            recommendations['stars'] >= min_feasibility_stars
        ].copy()
        
        # Sort by feasibility and z-score
        if 'z_score' in recommended.columns:
            recommended = recommended.sort_values(
                ['stars', 'z_score'],
                ascending=[False, True]
            )
        
        return recommended.drop('stars', axis=1)
    
    def get_derivative_summary(self, symbols: List[str]) -> pd.DataFrame:
        """Get derivative summary for list of symbols"""
        summaries = []
        
        for symbol in symbols:
            summaries.append({
                'symbol': symbol,
                'type': self.universe.get_instrument_type(symbol).value if self.universe.get_instrument_type(symbol) else 'UNKNOWN',
                'derivatives': self.universe.get_derivative_type(symbol).value if self.universe.get_derivative_type(symbol) else 'NONE',
                'has_futures': self.universe.has_futures(symbol),
                'has_options': self.universe.has_options(symbol),
                'nifty50': self.universe.is_nifty50_constituent(symbol),
                'nifty500': self.universe.is_nifty500_constituent(symbol),
                'label': self.universe.get_recommendation_label(symbol),
            })
        
        return pd.DataFrame(summaries)
    
    def print_recommendation_report(self, recommendations: pd.DataFrame):
        """Print formatted recommendation report"""
        print("\n" + "=" * 120)
        print("STRATEGY RECOMMENDATIONS WITH DERIVATIVE SUPPORT")
        print("=" * 120)
        
        if recommendations.empty:
            print("No recommendations available")
            return
        
        # Define columns to display
        display_cols = [
            'symbol1', 'symbol2', 'symbol1_type', 'symbol2_type',
            'trading_feasibility', 'recommendation_note'
        ]
        
        # Add optional columns if they exist
        if 'z_score' in recommendations.columns:
            display_cols.insert(3, 'z_score')
        if 'p_value' in recommendations.columns:
            display_cols.insert(4, 'p_value')
        
        # Filter to available columns
        display_cols = [c for c in display_cols if c in recommendations.columns]
        
        # Display
        display_df = recommendations[display_cols].head(20)
        print(display_df.to_string(index=False))
        
        print("\n" + "=" * 120)
        print(f"Total recommendations: {len(recommendations)}")
        if 'trading_feasibility' in recommendations.columns:
            feasible = (recommendations['trading_feasibility'].str.count('★') >= 3).sum()
            print(f"Highly feasible (3+ stars): {feasible}")


# Convenience instance
recommender = StrategyRecommender()


if __name__ == "__main__":
    # Test the recommender
    print("=" * 80)
    print("Strategy Recommender - Derivative Support")
    print("=" * 80)
    
    # Test symbols
    test_symbols = [
        'RELIANCE', 'TCS', 'INFY', 'PAYTM', 'ADANIPOWER',
        'NIFTY50', 'BANKNIFTY', 'FINNIFTY'
    ]
    
    print("\n📊 Derivative Summary:")
    summary = recommender.get_derivative_summary(test_symbols)
    print(summary)
    
    # Test pair recommendation
    print("\n\n📈 Sample Pair Recommendations:")
    sample_results = pd.DataFrame({
        'symbol1': ['RELIANCE', 'TCS', 'INFY'],
        'symbol2': ['JSWSTEEL', 'INFOSYS', 'WIPRO'],
        'z_score': [2.5, 1.8, 3.2],
        'p_value': [0.01, 0.05, 0.005],
    })
    
    recommendations = recommender.recommend_pair_trades(sample_results)
    print(recommendations[['symbol1', 'symbol2', 'symbol1_type', 'symbol2_type', 'trading_feasibility']].to_string(index=False))
    
    # Get recommended only
    print("\n\n⭐ Highly Feasible Recommendations (3+ stars):")
    recommended = recommender.get_recommended_positions(sample_results, min_feasibility_stars=3)
    if not recommended.empty:
        print(recommended[['symbol1', 'symbol2', 'trading_feasibility', 'recommendation_note']].to_string(index=False))
    else:
        print("No recommendations with 3+ star feasibility")
