"""
utils/nifty500_pair_trading.py - Pair trading specialized for NIFTY500 constituents
Uses NIFTY500 universe with derivative status indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from utils.nse_derivatives_manager import get_derivatives_manager
    from utils.scan_recommendation_classifier import get_classifier
except ImportError:
    logger.warning("Could not import derivative managers")

class NIFTY500PairTrading:
    """Pair trading specialized for NIFTY500 with derivative tracking"""
    
    def __init__(self):
        self.derivatives_manager = get_derivatives_manager() if 'get_derivatives_manager' in dir() else None
        self.classifier = get_classifier() if 'get_classifier' in dir() else None
        self.nifty500_df = None
        self._load_nifty500()
    
    def _load_nifty500(self):
        """Load NIFTY500 constituents with derivative info"""
        try:
            if self.derivatives_manager:
                self.nifty500_df = self.derivatives_manager.get_nifty500_with_derivatives()
                logger.info(f"Loaded {len(self.nifty500_df)} NIFTY500 constituents")
        except Exception as e:
            logger.error(f"Error loading NIFTY500: {e}")
    
    def get_tradeable_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all tradeable NIFTY500 pairs
        Returns list of (stock1, stock2) tuples
        """
        if self.nifty500_df is None or self.nifty500_df.empty:
            logger.warning("NIFTY500 data not available")
            return []
        
        symbols = self.nifty500_df['symbol'].unique()
        
        # Generate all pairs
        pairs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pairs.append((symbols[i], symbols[j]))
        
        return pairs
    
    def get_pairs_with_derivatives(self, derivative_requirement: str = "any") -> List[Tuple[str, str]]:
        """
        Get NIFTY500 pairs with specific derivative requirements
        
        Args:
            derivative_requirement: "any", "options", "futures", "both", "stock_only"
        
        Returns:
            List of (symbol1, symbol2) tuples matching criteria
        """
        if self.nifty500_df is None or self.nifty500_df.empty:
            return []
        
        df = self.nifty500_df.copy()
        
        if derivative_requirement == "options":
            df = df[df['has_options'] == True]
        elif derivative_requirement == "futures":
            df = df[df['has_futures'] == True]
        elif derivative_requirement == "both":
            df = df[(df['has_options'] == True) & (df['has_futures'] == True)]
        elif derivative_requirement == "stock_only":
            df = df[(df['has_options'] == False) & (df['has_futures'] == False)]
        
        symbols = df['symbol'].unique()
        
        # Generate pairs
        pairs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pairs.append((symbols[i], symbols[j]))
        
        return pairs
    
    def classify_pair_recommendation(self, symbol1: str, symbol2: str,
                                   signal: str, strength: float = 0.0,
                                   strategy_name: str = "Pair Trading") -> Dict:
        """
        Classify a pair trading recommendation with full details
        
        Args:
            symbol1: First symbol in pair
            symbol2: Second symbol in pair
            signal: BUY/SELL/HOLD
            strength: Signal strength (0-1)
            strategy_name: Name of strategy (default: Pair Trading)
        
        Returns:
            Dict with classification information for both symbols
        """
        symbol1 = symbol1.strip().upper()
        symbol2 = symbol2.strip().upper()
        
        if not self.classifier:
            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'signal': signal,
                'strategy': strategy_name,
                'error': 'Classifier not available'
            }
        
        # Get classification for both symbols
        rec1 = self.classifier.classify_scan_result(symbol1, strategy_name, signal, strength, symbol2)
        rec2 = self.classifier.classify_scan_result(symbol2, strategy_name, signal, strength, symbol1)
        
        return {
            'pair': f"{symbol1}-{symbol2}",
            'symbol1': symbol1,
            'symbol2': symbol2,
            'signal': signal,
            'strength': strength,
            'strategy': strategy_name,
            'symbol1_derivative_status': rec1.derivative_status,
            'symbol2_derivative_status': rec2.derivative_status,
            'symbol1_vehicles': rec1.get_trading_vehicles(),
            'symbol2_vehicles': rec2.get_trading_vehicles(),
            'symbol1_has_options': rec1.has_options,
            'symbol1_has_futures': rec1.has_futures,
            'symbol2_has_options': rec2.has_options,
            'symbol2_has_futures': rec2.has_futures,
            'both_have_options': rec1.has_options and rec2.has_options,
            'both_have_futures': rec1.has_futures and rec2.has_futures,
            'symbol1_note': rec1.get_recommendation_note(),
            'symbol2_note': rec2.get_recommendation_note(),
            'pair_trading_note': self._generate_pair_note(rec1, rec2, signal),
        }
    
    def _generate_pair_note(self, rec1, rec2, signal: str) -> str:
        """Generate pair trading recommendation note"""
        vehicles1 = ", ".join(rec1.get_trading_vehicles())
        vehicles2 = ", ".join(rec2.get_trading_vehicles())
        
        note = f"PAIR TRADING ({signal}): {rec1.symbol} ({vehicles1}) vs {rec2.symbol} ({vehicles2})"
        
        # Add special notes
        if rec1.has_options and rec2.has_options:
            note += " [Both have Options] "
        if rec1.has_futures and rec2.has_futures:
            note += " [Both have Futures] "
        
        return note
    
    def scan_pair_signals(self, price_data: Dict[str, pd.DataFrame],
                         timeframe: str = "15minute") -> pd.DataFrame:
        """
        Scan for pair trading signals in NIFTY500
        
        Args:
            price_data: Dict mapping symbol to price DataFrame
            timeframe: Timeframe to analyze
        
        Returns:
            DataFrame with pair trading signals, classified with derivative info
        """
        signals = []
        
        try:
            # Get all tradeable pairs
            pairs = self.get_tradeable_pairs()
            
            logger.info(f"Scanning {len(pairs)} NIFTY500 pairs for {timeframe}...")
            
            for symbol1, symbol2 in pairs:
                if symbol1 not in price_data or symbol2 not in price_data:
                    continue
                
                # Get price data
                df1 = price_data[symbol1]
                df2 = price_data[symbol2]
                
                # Calculate pair signal
                signal = self._calculate_pair_signal(df1, df2)
                
                if signal:
                    # Classify with derivative info
                    rec = self.classify_pair_recommendation(
                        symbol1, symbol2,
                        signal.get('signal', 'HOLD'),
                        signal.get('strength', 0.0)
                    )
                    
                    signals.append(rec)
            
            logger.info(f"Found {len(signals)} pair signals in {timeframe}")
            
            return pd.DataFrame(signals) if signals else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error scanning pair signals: {e}")
            return pd.DataFrame()
    
    def _calculate_pair_signal(self, df1: pd.DataFrame, df2: pd.DataFrame,
                              lookback: int = 20) -> Optional[Dict]:
        """
        Calculate pair trading signal between two price series
        
        Args:
            df1: Price DataFrame for symbol 1
            df2: Price DataFrame for symbol 2
            lookback: Lookback period
        
        Returns:
            Dict with signal, strength, etc. or None
        """
        try:
            if len(df1) < lookback or len(df2) < lookback:
                return None
            
            # Get closing prices
            close1 = df1['close'].tail(lookback).values
            close2 = df2['close'].tail(lookback).values
            
            # Normalize prices
            norm1 = (close1 - close1.mean()) / close1.std()
            norm2 = (close2 - close2.mean()) / close2.std()
            
            # Calculate spread
            spread = norm1 - norm2
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            if spread_std == 0:
                return None
            
            # Z-score of current spread
            z_score = (spread[-1] - spread_mean) / spread_std
            
            # Generate signal
            if z_score > 1.5:
                signal = "SELL"
                strength = min(abs(z_score) / 3, 1.0)
            elif z_score < -1.5:
                signal = "BUY"
                strength = min(abs(z_score) / 3, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0
            
            return {
                'signal': signal,
                'strength': strength,
                'z_score': z_score,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
            }
            
        except Exception as e:
            logger.debug(f"Error calculating pair signal: {e}")
            return None
    
    def get_recommendations_for_strategy(self, strategy_name: str) -> Dict:
        """
        Get all recommendations for a strategy with derivative status
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            Dict with recommendations organized by derivative status
        """
        if not self.nifty500_df is not None or self.nifty500_df.empty:
            return {'error': 'NIFTY500 data not available'}
        
        by_status = {}
        
        for status in self.nifty500_df['derivative_status'].unique():
            symbols = self.nifty500_df[
                self.nifty500_df['derivative_status'] == status
            ]['symbol'].tolist()
            
            by_status[status] = {
                'count': len(symbols),
                'symbols': symbols,
            }
        
        return {
            'strategy': strategy_name,
            'universe': 'NIFTY500',
            'total_constituents': len(self.nifty500_df),
            'by_derivative_status': by_status,
        }

# Global instance
_nifty500_pair_trading_instance = None

def get_nifty500_pair_trader() -> NIFTY500PairTrading:
    """Get or create global NIFTY500 pair trader"""
    global _nifty500_pair_trading_instance
    if _nifty500_pair_trading_instance is None:
        _nifty500_pair_trading_instance = NIFTY500PairTrading()
    return _nifty500_pair_trading_instance

# Convenience functions
def scan_nifty500_pairs(price_data: Dict[str, pd.DataFrame],
                       timeframe: str = "15minute") -> pd.DataFrame:
    """Scan NIFTY500 pairs for trading signals"""
    return get_nifty500_pair_trader().scan_pair_signals(price_data, timeframe)

def get_nifty500_stats() -> Dict:
    """Get NIFTY500 derivative statistics"""
    trader = get_nifty500_pair_trader()
    if trader.nifty500_df is None or trader.nifty500_df.empty:
        return {}
    
    df = trader.nifty500_df
    return {
        'total': len(df),
        'with_options_and_futures': len(df[(df['has_options']) & (df['has_futures'])]),
        'with_options_only': len(df[(df['has_options']) & (~df['has_futures'])]),
        'with_futures_only': len(df[(~df['has_options']) & (df['has_futures'])]),
        'stock_only': len(df[(~df['has_options']) & (~df['has_futures'])]),
    }
