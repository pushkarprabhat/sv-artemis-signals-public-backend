"""
Data Loader - Load instruments and configuration from JSON/CSV files

Purpose:
- Load instrument metadata from JSON files
- Load industry classification from JSON
- Load stock-industry mapping from CSV
- Load segment configuration from JSON
- Decouple hardcoded data from code

Files Used:
- universe/metadata/instruments_metadata.json: Complete instrument details
- universe/metadata/industry_classification.json: Hierarchical industry structure
- universe/metadata/stock_industry_mapping.csv: Stock -> Industry mapping
- universe/metadata/segments_config.json: Segment configuration and strategies
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class DataLoader:
    """Load configuration and instrument data from files"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader with optional custom data directory"""
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        
        # Cache loaded data to avoid reloading
        self._instruments_cache = None
        self._industry_cache = None
        self._stock_industry_cache = None
        self._segments_cache = None
        
    def load_instruments_metadata(self) -> Dict[str, Any]:
        """Load instruments metadata from JSON"""
        if self._instruments_cache is not None:
            return self._instruments_cache
            
        file_path = self.data_dir / "instruments_metadata.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self._instruments_cache = data
        return data
    
    def load_industry_classification(self) -> Dict[str, Any]:
        """Load industry classification from JSON"""
        if self._industry_cache is not None:
            return self._industry_cache
            
        file_path = self.data_dir / "industry_classification.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self._industry_cache = data
        return data
    
    def load_stock_industry_mapping(self) -> pd.DataFrame:
        """Load stock-industry mapping CSV as DataFrame"""
        if self._stock_industry_cache is not None:
            return self._stock_industry_cache
            
        file_path = self.data_dir / "stock_industry_mapping.csv"
        df = pd.read_csv(file_path)
        
        self._stock_industry_cache = df
        return df
    
    def load_segments_config(self) -> Dict[str, Any]:
        """Load segment configuration from JSON"""
        if self._segments_cache is not None:
            return self._segments_cache
            
        file_path = self.data_dir / "segments_config.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self._segments_cache = data
        return data
    
    # =========================================================================
    # ACCESSOR METHODS - Get specific data
    # =========================================================================
    
    def get_indices(self) -> Dict[str, Dict[str, Any]]:
        """Get all indices metadata"""
        metadata = self.load_instruments_metadata()
        return metadata.get("indices", {})
    
    def get_commodities(self) -> Dict[str, Dict[str, Any]]:
        """Get all commodities metadata"""
        metadata = self.load_instruments_metadata()
        return metadata.get("commodities", {})
    
    def get_forex_pairs(self) -> Dict[str, Dict[str, Any]]:
        """Get all forex pairs metadata"""
        metadata = self.load_instruments_metadata()
        return metadata.get("forex", {})
    
    def get_segments(self) -> Dict[str, Dict[str, Any]]:
        """Get segment configuration"""
        config = self.load_segments_config()
        return config.get("segments", {})
    
    def get_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get strategy compatibility matrix"""
        config = self.load_segments_config()
        return config.get("strategy_compatibility", {})
    
    def get_stock_industries(self) -> pd.DataFrame:
        """Get stock to industry mapping"""
        return self.load_stock_industry_mapping()
    
    def get_stocks_by_industry(self, industry: str) -> List[str]:
        """Get all stocks in a specific industry"""
        df = self.load_stock_industry_mapping()
        matching = df[df['industry_sector'] == industry]['symbol'].tolist()
        return matching
    
    def get_stock_industry(self, symbol: str) -> Optional[str]:
        """Get industry for a specific stock"""
        df = self.load_stock_industry_mapping()
        matches = df[df['symbol'] == symbol]
        if matches.empty:
            return None
        return matches.iloc[0]['industry_sector']
    
    def get_stock_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get complete details for a specific stock"""
        df = self.load_stock_industry_mapping()
        matches = df[df['symbol'] == symbol]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()
    
    def get_industries(self) -> Set[str]:
        """Get all unique industries"""
        df = self.load_stock_industry_mapping()
        return set(df['industry_sector'].unique())
    
    def get_industry_stocks(self, industry: str) -> Dict[str, Any]:
        """Get structured view of an industry from classification"""
        classification = self.load_industry_classification()
        industry_data = classification.get("industry_classification", {})
        return industry_data.get(industry, {})
    
    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================
    
    def validate_pair(self, symbol1: str, symbol2: str) -> tuple[bool, str]:
        """
        Validate if two stocks can form a pair.
        Returns: (is_valid, reason_if_invalid)
        
        Constraints:
        - Both symbols must exist in mapping
        - Both must be from SAME industry_sector
        """
        industry1 = self.get_stock_industry(symbol1)
        industry2 = self.get_stock_industry(symbol2)
        
        if industry1 is None:
            return False, f"Symbol {symbol1} not found in mapping"
        if industry2 is None:
            return False, f"Symbol {symbol2} not found in mapping"
        if industry1 != industry2:
            return False, f"Different industries: {symbol1} in {industry1}, {symbol2} in {industry2}"
        
        return True, ""
    
    def get_valid_pairs_for_stocks(self, symbols: List[str]) -> List[tuple[str, str]]:
        """
        Get all valid pairs from a list of stocks.
        A valid pair: same industry, both in list
        
        Returns list of (symbol1, symbol2) tuples where symbol1 < symbol2
        """
        valid_pairs = []
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                is_valid, _ = self.validate_pair(sym1, sym2)
                if is_valid:
                    valid_pairs.append((sym1, sym2))
        
        return valid_pairs
    
    def get_stocks_for_pair_trading(self, min_per_industry: int = 3) -> Dict[str, List[str]]:
        """
        Get stocks grouped by industry for pair trading.
        Only includes industries with at least min_per_industry stocks.
        
        Returns dict: {industry: [stocks]}
        """
        df = self.load_stock_industry_mapping()
        
        # Filter for stocks with futures (required for pair trading)
        # Note: CSV uses 'Yes'/'No' strings, not booleans
        df_tradeable = df[df['futures_available'].astype(str) == 'Yes']
        
        # Group by industry
        industry_groups = {}
        for industry, group in df_tradeable.groupby('industry_sector'):
            stocks = group['symbol'].tolist()
            if len(stocks) >= min_per_industry:
                industry_groups[industry] = stocks
        
        return industry_groups
    
    # =========================================================================
    # SUMMARY METHODS
    # =========================================================================
    
    def print_data_summary(self):
        """Print summary of loaded data"""
        print("\n" + "="*80)
        print("DATA LOADER SUMMARY")
        print("="*80)
        
        # Instruments
        metadata = self.load_instruments_metadata()
        print(f"\n📊 INSTRUMENTS:")
        print(f"   Indices: {len(metadata.get('indices', {}))}")
        print(f"   Commodities: {len(metadata.get('commodities', {}))}")
        print(f"   Forex Pairs: {len(metadata.get('forex', {}))}")
        
        # Industries
        industries = self.get_industries()
        print(f"\n🏢 INDUSTRIES:")
        print(f"   Total unique industries: {len(industries)}")
        
        df = self.load_stock_industry_mapping()
        print(f"   Total stocks in mapping: {len(df)}")
        
        # Segments
        segments = self.get_segments()
        print(f"\n📍 SEGMENTS:")
        for segment_name, details in segments.items():
            print(f"   - {segment_name}: {details.get('description', 'N/A')}")
        
        # Pair trading groups
        pair_groups = self.get_stocks_for_pair_trading()
        print(f"\n🔗 PAIR TRADING GROUPS:")
        print(f"   Industries with 3+ stocks: {len(pair_groups)}")
        for industry, stocks in sorted(pair_groups.items()):
            print(f"   - {industry}: {len(stocks)} stocks")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    
    # Print summary
    loader.print_data_summary()
    
    # Test pair validation
    print("\n" + "="*80)
    print("PAIR VALIDATION TESTS")
    print("="*80)
    
    # Valid pairs
    test_pairs = [
        ("RELIANCE", "BPCL"),  # Both energy
        ("TCS", "INFY"),  # Both IT
        ("HDFC", "ICICIBANK"),  # Both banking
        ("TCS", "RELIANCE"),  # Different industries - invalid
    ]
    
    for sym1, sym2 in test_pairs:
        is_valid, reason = loader.validate_pair(sym1, sym2)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{sym1} + {sym2}: {status}")
        if not is_valid:
            print(f"   Reason: {reason}")
    
    # Show IT stocks for pair trading
    print("\n" + "="*80)
    print("IT STOCKS FOR PAIR TRADING")
    print("="*80)
    it_stocks = loader.get_stocks_by_industry("IT")
    print(f"Total IT stocks: {len(it_stocks)}")
    print(f"Available for trading: {it_stocks[:10]}...")  # Show first 10
    
    # Show all pair trading groups
    print("\n" + "="*80)
    print("ALL PAIR TRADING GROUPS (3+ stocks per industry)")
    print("="*80)
    pair_groups = loader.get_stocks_for_pair_trading()
    for industry, stocks in sorted(pair_groups.items()):
        print(f"\n{industry}:")
        print(f"  Count: {len(stocks)}")
        print(f"  Stocks: {', '.join(stocks[:5])}{'...' if len(stocks) > 5 else ''}")
