"""
OI Position Analysis Engine
Analyzes Open Interest changes to detect market positioning:
- Long Buildup: OI increase + price increase
- Long Unwinding: OI decrease + price decrease
- Short Buildup: OI increase + price decrease
- Short Unwinding: OI decrease + price increase
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

@dataclass
class OIAnalysisResult:
    """Result of OI analysis for a symbol"""
    symbol: str
    underlying: str
    exchange: str
    
    # Daily analysis (comparing to previous day)
    daily_oi_change: float          # Absolute OI change
    daily_oi_change_pct: float      # Percentage OI change
    daily_price_change_pct: float   # Percentage price change
    daily_position: str             # long_buildup|long_unwinding|short_buildup|short_unwinding|neutral
    
    # Intraday analysis (comparing to previous day close)
    intraday_oi_change: float       # Absolute OI change from prev close
    intraday_oi_change_pct: float   # Percentage OI change from prev close
    intraday_price_change_pct: float # Percentage price change from prev close
    intraday_position: str          # Same as daily_position
    
    # Additional metrics
    oi_strength: float              # OI change strength (0-100)
    price_strength: float           # Price change strength (0-100)
    conviction: float               # Conviction score (0-100) - how confident the signal is
    
    last_updated: datetime
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'exchange': self.exchange,
            'daily_oi_change': round(self.daily_oi_change, 2),
            'daily_oi_change_pct': round(self.daily_oi_change_pct, 2),
            'daily_price_change_pct': round(self.daily_price_change_pct, 4),
            'daily_position': self.daily_position,
            'intraday_oi_change': round(self.intraday_oi_change, 2),
            'intraday_oi_change_pct': round(self.intraday_oi_change_pct, 2),
            'intraday_price_change_pct': round(self.intraday_price_change_pct, 4),
            'intraday_position': self.intraday_position,
            'oi_strength': round(self.oi_strength, 2),
            'price_strength': round(self.price_strength, 2),
            'conviction': round(self.conviction, 2),
            'last_updated': self.last_updated.isoformat(),
        }


class OIAnalyzer:
    """Analyze Open Interest changes to determine market positioning"""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize OI Analyzer
        
        Args:
            data_dir: Root directory for market data (default: marketdata/)
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "marketdata"
        self.results: Dict[str, OIAnalysisResult] = {}
        self.last_analysis: Optional[datetime] = None
        
    def analyze_daily(self, symbol: str, current_data: Dict, previous_data: Dict) -> OIAnalysisResult:
        """
        Analyze daily OI changes
        
        Args:
            symbol: Trading symbol
            current_data: Current day OHLCV + OI (Dict with 'close', 'open', 'oi', etc)
            previous_data: Previous day OHLCV + OI (Dict)
        
        Returns:
            OIAnalysisResult with daily positioning
        """
        # Extract values
        curr_oi = float(current_data.get('oi', 0))
        prev_oi = float(previous_data.get('oi', 0))
        curr_close = float(current_data.get('close', 0))
        prev_close = float(previous_data.get('close', 0))
        
        # Calculate changes
        oi_change = curr_oi - prev_oi
        oi_change_pct = (oi_change / prev_oi * 100) if prev_oi != 0 else 0
        price_change_pct = ((curr_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
        
        # Determine position
        position = self._determine_position(oi_change_pct, price_change_pct)
        
        # Calculate strength scores
        oi_strength = min(100, abs(oi_change_pct) * 10)  # 1% OI change = 10 strength
        price_strength = min(100, abs(price_change_pct) * 50)  # 1% price change = 50 strength
        
        # Conviction: how aligned are OI and price changes?
        conviction = self._calculate_conviction(oi_change_pct, price_change_pct, position)
        
        return OIAnalysisResult(
            symbol=symbol,
            underlying=current_data.get('underlying', symbol),
            exchange=current_data.get('exchange', ''),
            daily_oi_change=oi_change,
            daily_oi_change_pct=oi_change_pct,
            daily_price_change_pct=price_change_pct,
            daily_position=position,
            intraday_oi_change=0.0,
            intraday_oi_change_pct=0.0,
            intraday_price_change_pct=0.0,
            intraday_position='neutral',
            oi_strength=oi_strength,
            price_strength=price_strength,
            conviction=conviction,
            last_updated=datetime.now(),
        )
    
    def analyze_intraday(self, symbol: str, current_data: Dict, prev_close_data: Dict) -> OIAnalysisResult:
        """
        Analyze intraday OI changes (compared to previous day close)
        
        Args:
            symbol: Trading symbol
            current_data: Current live OHLCV + OI
            prev_close_data: Previous day close OHLCV + OI
        
        Returns:
            OIAnalysisResult with intraday positioning
        """
        # Extract values
        curr_oi = float(current_data.get('oi', 0))
        prev_oi = float(prev_close_data.get('oi', 0))
        curr_price = float(current_data.get('ltp', current_data.get('close', 0)))
        prev_price = float(prev_close_data.get('close', 0))
        
        # Calculate changes
        oi_change = curr_oi - prev_oi
        oi_change_pct = (oi_change / prev_oi * 100) if prev_oi != 0 else 0
        price_change_pct = ((curr_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        
        # Determine position
        position = self._determine_position(oi_change_pct, price_change_pct)
        
        # Calculate strength scores
        oi_strength = min(100, abs(oi_change_pct) * 10)
        price_strength = min(100, abs(price_change_pct) * 50)
        conviction = self._calculate_conviction(oi_change_pct, price_change_pct, position)
        
        return OIAnalysisResult(
            symbol=symbol,
            underlying=current_data.get('underlying', symbol),
            exchange=current_data.get('exchange', ''),
            daily_oi_change=0.0,
            daily_oi_change_pct=0.0,
            daily_price_change_pct=0.0,
            daily_position='neutral',
            intraday_oi_change=oi_change,
            intraday_oi_change_pct=oi_change_pct,
            intraday_price_change_pct=price_change_pct,
            intraday_position=position,
            oi_strength=oi_strength,
            price_strength=price_strength,
            conviction=conviction,
            last_updated=datetime.now(),
        )
    
    @staticmethod
    def _determine_position(oi_change_pct: float, price_change_pct: float) -> str:
        """
        Determine market position based on OI and price changes
        
        Logic:
        - Long Buildup: OI↑ + Price↑ (traders adding bullish positions)
        - Long Unwinding: OI↓ + Price↓ (bulls exiting)
        - Short Buildup: OI↑ + Price↓ (traders adding bearish positions)
        - Short Unwinding: OI↓ + Price↑ (shorts exiting)
        """
        # Thresholds (in percentage)
        OI_THRESHOLD = 0.5  # 0.5% OI change
        PRICE_THRESHOLD = 0.05  # 0.05% price change
        
        oi_up = oi_change_pct > OI_THRESHOLD
        oi_down = oi_change_pct < -OI_THRESHOLD
        price_up = price_change_pct > PRICE_THRESHOLD
        price_down = price_change_pct < -PRICE_THRESHOLD
        
        if oi_up and price_up:
            return 'long_buildup'
        elif oi_down and price_down:
            return 'long_unwinding'
        elif oi_up and price_down:
            return 'short_buildup'
        elif oi_down and price_up:
            return 'short_unwinding'
        else:
            return 'neutral'
    
    @staticmethod
    def _calculate_conviction(oi_change_pct: float, price_change_pct: float, position: str) -> float:
        """
        Calculate conviction score (0-100)
        Higher conviction when OI and price changes are strongly aligned
        """
        if position == 'neutral':
            return 0.0
        
        # Conviction based on alignment strength
        oi_strength = abs(oi_change_pct)
        price_strength = abs(price_change_pct)
        
        # Both moving strongly in same direction
        if position in ['long_buildup', 'short_unwinding']:
            alignment = min(oi_strength / 10, price_strength / 0.5)  # Favor price changes
        else:  # long_unwinding, short_buildup
            alignment = min(oi_strength / 10, price_strength / 0.5)
        
        return min(100, alignment * 25)
    
    def get_summary(self) -> Dict:
        """Get summary of current OI analysis results"""
        if not self.results:
            return {}
        
        # Group by position
        by_position = {}
        for result in self.results.values():
            position = result.daily_position
            if position not in by_position:
                by_position[position] = []
            by_position[position].append(result)
        
        return {
            'total_analyzed': len(self.results),
            'last_analysis': self.last_analysis.isoformat() if self.last_analysis else None,
            'by_position': {
                pos: len(symbols) for pos, symbols in by_position.items()
            },
            'summary': {
                pos: [s.to_dict() for s in symbols]
                for pos, symbols in by_position.items()
            }
        }
    
    def save_results(self, output_file: Path = None):
        """Save analysis results to JSON file"""
        output_file = output_file or self.data_dir / '.oi_analysis_results.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': {sym: result.to_dict() for sym, result in self.results.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_file
    
    def load_results(self, input_file: Path = None) -> Dict:
        """Load analysis results from JSON file"""
        input_file = input_file or self.data_dir / '.oi_analysis_results.json'
        
        if not input_file.exists():
            return {}
        
        with open(input_file, 'r') as f:
            return json.load(f)


# Example usage and testing
if __name__ == '__main__':
    analyzer = OIAnalyzer()
    
    # Test daily analysis
    print("=" * 80)
    print("OI POSITION ANALYSIS - DAILY EXAMPLE")
    print("=" * 80)
    
    # Example: Long buildup
    prev_day = {'close': 100, 'oi': 1000000, 'exchange': 'NSE', 'underlying': 'SBIN'}
    curr_day = {'close': 102, 'oi': 1020000, 'exchange': 'NSE', 'underlying': 'SBIN'}
    
    result = analyzer.analyze_daily('SBIN', curr_day, prev_day)
    print(f"\nLong Buildup Example:")
    print(f"  OI Change: {result.daily_oi_change_pct:.2f}% (↑)")
    print(f"  Price Change: {result.daily_price_change_pct:.2f}% (↑)")
    print(f"  Position: {result.daily_position}")
    print(f"  Conviction: {result.conviction:.1f}/100")
    
    # Example: Short buildup
    prev_day = {'close': 100, 'oi': 1000000, 'exchange': 'NSE', 'underlying': 'INFY'}
    curr_day = {'close': 98, 'oi': 1020000, 'exchange': 'NSE', 'underlying': 'INFY'}
    
    result = analyzer.analyze_daily('INFY', curr_day, prev_day)
    print(f"\nShort Buildup Example:")
    print(f"  OI Change: {result.daily_oi_change_pct:.2f}% (↑)")
    print(f"  Price Change: {result.daily_price_change_pct:.2f}% (↓)")
    print(f"  Position: {result.daily_position}")
    print(f"  Conviction: {result.conviction:.1f}/100")
    
    # Example: Intraday analysis
    print("\n" + "=" * 80)
    print("OI POSITION ANALYSIS - INTRADAY EXAMPLE")
    print("=" * 80)
    
    prev_close = {'close': 100, 'oi': 1000000, 'exchange': 'NSE', 'underlying': 'TCS'}
    curr_live = {'ltp': 101.5, 'oi': 1010000, 'close': 100, 'exchange': 'NSE', 'underlying': 'TCS'}
    
    result = analyzer.analyze_intraday('TCS', curr_live, prev_close)
    print(f"\nIntraday Analysis:")
    print(f"  OI Change: {result.intraday_oi_change_pct:.2f}% (↑)")
    print(f"  Price Change: {result.intraday_price_change_pct:.2f}% (↑)")
    print(f"  Position: {result.intraday_position}")
    print(f"  Conviction: {result.conviction:.1f}/100")
