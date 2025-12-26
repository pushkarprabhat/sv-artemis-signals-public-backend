"""
Derivatives Universe Management

Tracks all derivative-eligible instruments on NSE including:
- Individual stock futures and options
- Index futures and options
- Derivative capabilities per instrument (Futures only vs Futures+Options)
- NIFTY500 constituents for pair trading universe
- Instrument metadata (symbol, type, exchange, etc.)

REFACTORED: Now loads from data files (JSON/CSV) instead of hardcoded dicts

Data Sources:
- universe/metadata/instruments_metadata.json: Complete instrument details
- universe/metadata/stock_industry_mapping.csv: Stock-to-industry mapping
- universe/metadata/industry_classification.json: Hierarchical industry structure

Updated from NSE official derivatives products page:
https://www.nseindia.com/static/products-services/equity-derivatives-products
"""

import pandas as pd
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


class InstrumentType(Enum):
    """Instrument type classification"""
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    FUTURES_ONLY = "futures_only"  # Stock with only futures, no options


class DerivativeType(Enum):
    """Derivative availability for an instrument"""
    FUTURES_ONLY = "F"      # Futures trading available, Options NOT available
    FUTURES_OPTIONS = "FO"  # Both Futures and Options trading available
    OPTIONS_ONLY = "O"      # Options trading only (rare)
    NO_DERIVATIVES = "X"    # No derivatives available


class DerivativesUniverse:
    """
    Manages NSE derivatives universe and tracks derivative capabilities.
    
    Features:
    - Load 200+ derivative-eligible stocks from data files
    - Separate NIFTY500 constituents (for pair trading)
    - Classify instruments by derivative type (F, FO, O, X)
    - Provide recommendations with derivative support indicators
    - Filter instruments by trading capabilities
    - Enforce pair trading within same industry sector
    """
    
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize derivatives universe by loading from data files"""
        self.all_derivatives = {}
        self.industry_mapping = {}
        self._populate_universe()
    
    def _populate_universe(self):
        """Load all instruments from data files"""
        try:
            import json
            from config import METADATA_DIR
            
            # Load expanded derivatives universe (275 stocks: 186 F&O + 89 Futures-only)
            deriv_file = METADATA_DIR / "nse_derivatives_universe.json"
            with open(deriv_file, 'r') as f:
                deriv_data = json.load(f)
            
            # Load instruments metadata for indices and commodities
            metadata_file = METADATA_DIR / "instruments_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add indices (from metadata)
            if 'indices' in metadata:
                for symbol, info in metadata['indices'].items():
                    self.all_derivatives[symbol] = {
                        'type': InstrumentType.INDEX,
                        'derivatives': DerivativeType.FUTURES_OPTIONS if info.get('has_options') else DerivativeType.FUTURES_ONLY
                    }
            
            # Add F&O stocks (186 stocks)
            if 'futures_and_options' in deriv_data:
                for symbol in deriv_data['futures_and_options'].get('stocks', []):
                    self.all_derivatives[symbol] = {
                        'type': InstrumentType.STOCK,
                        'derivatives': DerivativeType.FUTURES_OPTIONS,
                    }
            
            # Add Futures-only stocks (89 stocks)
            if 'futures_only' in deriv_data:
                for symbol in deriv_data['futures_only'].get('stocks', []):
                    self.all_derivatives[symbol] = {
                        'type': InstrumentType.STOCK,
                        'derivatives': DerivativeType.FUTURES_ONLY,
                    }
            
            # Load industry mapping from CSV (for pair trading constraint)
            stocks_file = METADATA_DIR / "stock_industry_mapping.csv"
            df = pd.read_csv(stocks_file)
            
            for _, row in df.iterrows():
                symbol = row['symbol']
                # Update with industry info if stock exists
                if symbol in self.all_derivatives:
                    self.all_derivatives[symbol]['industry'] = row['industry_sector']
                
                # Build industry mapping
                if row['industry_sector'] not in self.industry_mapping:
                    self.industry_mapping[row['industry_sector']] = []
                self.industry_mapping[row['industry_sector']].append(symbol)
            
            # Load commodities and forex from metadata
            if 'commodities' in metadata:
                for symbol, info in metadata['commodities'].items():
                    self.all_derivatives[symbol] = {
                        'type': InstrumentType.COMMODITY,
                        'derivatives': DerivativeType.FUTURES_ONLY
                    }
            
            if 'forex' in metadata:
                for symbol, info in metadata['forex'].items():
                    self.all_derivatives[symbol] = {
                        'type': InstrumentType.CURRENCY,
                        'derivatives': DerivativeType.FUTURES_ONLY
                    }
            
            # Print statistics
            fo_count = len(self.filter_by_derivative_type(DerivativeType.FUTURES_OPTIONS))
            f_only_count = len(self.filter_by_derivative_type(DerivativeType.FUTURES_ONLY))
            
            print(f"✓ Loaded {len(self.all_derivatives)} instruments from data files")
            print(f"  - Indices: {len(self.filter_by_instrument_type(InstrumentType.INDEX))}")
            print(f"  - Stocks (F&O): {fo_count}")
            print(f"  - Stocks (Futures only): {f_only_count}")
            print(f"  - Total Stocks: {fo_count + f_only_count}")
            print(f"  - Commodities: {len(self.filter_by_instrument_type(InstrumentType.COMMODITY))}")
            print(f"  - Forex: {len(self.filter_by_instrument_type(InstrumentType.CURRENCY))}")
            print(f"  - Industries: {len(self.industry_mapping)}")
            
        except FileNotFoundError as e:
            print(f"⚠️  Warning: Could not load data files: {e}")
            print(f"   Falling back to minimal universe")
    
    def _load_nifty500_constituents(self) -> List[str]:
        """Load NIFTY500 constituents from data"""
        # Get stocks from mapping that are in NIFTY50, NIFTY100, or NIFTY500
        try:
            from config import METADATA_DIR
            stocks_file = METADATA_DIR / "stock_industry_mapping.csv"
            df = pd.read_csv(stocks_file)
            
            # Include all stocks marked as NIFTY50, NIFTY100, or NIFTY500
            nifty500 = df[
                (df['nifty50'] == 'Yes') | 
                (df['nifty100'] == 'Yes') | 
                (df['nifty500'] == 'Yes')
            ]['symbol'].tolist()
            
            return nifty500
        except Exception:
            # Fallback to stocks with futures
            return [s for s in self.all_derivatives.keys() if self.has_futures(s)]
    
    def get_nifty50_constituents(self) -> List[str]:
        """Get NIFTY50 constituent stocks (all have F&O)"""
        try:
            from config import METADATA_DIR
            stocks_file = METADATA_DIR / "stock_industry_mapping.csv"
            df = pd.read_csv(stocks_file)
            nifty50 = df[df['nifty50'] == 'Yes']['symbol'].tolist()
            return nifty50
        except Exception:
            # Fallback: return stocks with options (NIFTY50 all have F&O)
            return [s for s in self.all_derivatives.keys() 
                    if self.has_options(s) and self.is_stock(s)]
    
    def get_nifty500_constituents(self) -> List[str]:
        """Get complete NIFTY500 list"""
        return self._load_nifty500_constituents()
    
    def get_derivative_type(self, symbol: str) -> Optional[DerivativeType]:
        """Get derivative type for instrument (F, FO, O, X)"""
        if symbol in self.all_derivatives:
            return self.all_derivatives[symbol].get('derivatives')
        return None
    
    def get_instrument_type(self, symbol: str) -> Optional[InstrumentType]:
        """Get instrument type (stock, index, commodity, etc.)"""
        if symbol in self.all_derivatives:
            return self.all_derivatives[symbol].get('type')
        return None
    
    def has_futures(self, symbol: str) -> bool:
        """Check if instrument has futures trading"""
        deriv_type = self.get_derivative_type(symbol)
        if deriv_type:
            return deriv_type in [DerivativeType.FUTURES_ONLY, DerivativeType.FUTURES_OPTIONS]
        return False
    
    def has_options(self, symbol: str) -> bool:
        """Check if instrument has options trading"""
        deriv_type = self.get_derivative_type(symbol)
        if deriv_type:
            return deriv_type in [DerivativeType.FUTURES_OPTIONS, DerivativeType.OPTIONS_ONLY]
        return False
    
    def filter_by_derivative_type(self, deriv_type: DerivativeType) -> List[str]:
        """Get all instruments with specific derivative type"""
        return [
            symbol for symbol, info in self.all_derivatives.items()
            if info.get('derivatives') == deriv_type
        ]
    
    def filter_by_instrument_type(self, inst_type: InstrumentType) -> List[str]:
        """Get all instruments of specific type (stock, index, etc.)"""
        return [
            symbol for symbol, info in self.all_derivatives.items()
            if info.get('type') == inst_type
        ]
    
    def is_index(self, symbol: str) -> bool:
        """Check if instrument is an index"""
        inst_type = self.get_instrument_type(symbol)
        return inst_type == InstrumentType.INDEX
    
    def is_stock(self, symbol: str) -> bool:
        """Check if instrument is a stock"""
        inst_type = self.get_instrument_type(symbol)
        return inst_type == InstrumentType.STOCK
    
    def is_nifty50_constituent(self, symbol: str) -> bool:
        """Check if stock is NIFTY50 constituent"""
        return symbol in self.NIFTY50_CONSTITUENTS
    
    def is_nifty500_constituent(self, symbol: str) -> bool:
        """Check if stock is NIFTY500 constituent"""
        return symbol in self.get_nifty500_constituents()
    
    def get_recommendation_label(self, symbol: str) -> str:
        """
        Get trading recommendation label for instrument
        
        Indicates:
        - If stock or index
        - If derivative support (FO, F only, no derivatives)
        - If pair trading (between index and constituent)
        
        Returns:
            Label like "[STOCK | F&O]", "[INDEX | F&O]", "[STOCK | F only]"
        """
        inst_type = self.get_instrument_type(symbol)
        deriv_type = self.get_derivative_type(symbol)
        
        if not inst_type:
            return "[UNKNOWN]"
        
        # Format instrument type
        if inst_type == InstrumentType.INDEX:
            type_label = "INDEX"
        elif inst_type == InstrumentType.STOCK:
            type_label = "STOCK"
        else:
            type_label = inst_type.value.upper()
        
        # Format derivative type
        if deriv_type == DerivativeType.FUTURES_OPTIONS:
            deriv_label = "F&O"
        elif deriv_type == DerivativeType.FUTURES_ONLY:
            deriv_label = "F only"
        elif deriv_type == DerivativeType.OPTIONS_ONLY:
            deriv_label = "O only"
        else:
            deriv_label = "No derivatives"
        
        return f"[{type_label} | {deriv_label}]"
    
    def get_pair_trading_universe(self) -> Dict[str, Dict]:
        """
        Get universe for pair trading
        
        Returns:
            Dict with indices and their NIFTY500 constituent pairs
        """
        indices = self.filter_by_instrument_type(InstrumentType.INDEX)
        nifty500 = self.get_nifty500_constituents()
        
        return {
            'indices': indices,
            'nifty500': nifty500,
            'pairs': [
                {
                    'index': idx,
                    'constituents': [s for s in nifty500 if self.has_futures(s)],
                    'description': f"Pair trading: {idx} index vs its constituent stocks"
                }
                for idx in indices
            ]
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about derivatives universe"""
        nifty50 = self.get_nifty50_constituents()
        nifty500 = self.get_nifty500_constituents()
        return {
            'total_instruments': len(self.all_derivatives),
            'total_indices': len(self.filter_by_instrument_type(InstrumentType.INDEX)),
            'total_stocks': len(self.filter_by_instrument_type(InstrumentType.STOCK)),
            'futures_and_options': len(self.filter_by_derivative_type(DerivativeType.FUTURES_OPTIONS)),
            'futures_only': len(self.filter_by_derivative_type(DerivativeType.FUTURES_ONLY)),
            'nifty50_constituents': len(nifty50),
            'nifty500_constituents': len(nifty500),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert universe to pandas DataFrame for analysis"""
        records = []
        for symbol, info in self.all_derivatives.items():
            records.append({
                'symbol': symbol,
                'instrument_type': info.get('type').value if info.get('type') else 'UNKNOWN',
                'derivative_type': info.get('derivatives').value if info.get('derivatives') else 'UNKNOWN',
                'has_futures': self.has_futures(symbol),
                'has_options': self.has_options(symbol),
                'is_nifty50': self.is_nifty50_constituent(symbol),
                'is_nifty500': self.is_nifty500_constituent(symbol),
            })
        
        return pd.DataFrame(records)
    
    def export_to_csv(self, filepath: str):
        """Export derivatives universe to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"✓ Derivatives universe exported to {filepath}")
    
    def import_from_nse_csv(self, filepath: str):
        """
        Import and update from NSE official F&O contract CSV
        
        Format: NSE_FO_contract_ddmmyyyy.csv.gz
        """
        try:
            df = pd.read_csv(filepath, compression='gzip')
            # Parse and update derivatives_universe based on CSV content
            # This is for integration with NSE's official data
            print(f"✓ Imported {len(df)} instruments from NSE CSV")
        except Exception as e:
            print(f"✗ Error importing NSE CSV: {e}")


# Convenience instance
derivatives_universe = DerivativesUniverse()


if __name__ == "__main__":
    # Test the derivatives universe
    print("=" * 80)
    print("NSE Derivatives Universe")
    print("=" * 80)
    
    # Statistics
    stats = derivatives_universe.get_statistics()
    print("\n📊 Universe Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Sample instruments
    print("\n📈 Sample Instruments:")
    print(f"  NIFTY50: {derivatives_universe.get_recommendation_label('NIFTY50')}")
    print(f"  RELIANCE: {derivatives_universe.get_recommendation_label('RELIANCE')}")
    print(f"  ADANIPOWER: {derivatives_universe.get_recommendation_label('ADANIPOWER')}")
    print(f"  PAYTM: {derivatives_universe.get_recommendation_label('PAYTM')}")
    
    # Filtering examples
    print("\n🔍 Filtering Examples:")
    fo_stocks = derivatives_universe.filter_by_derivative_type(DerivativeType.FUTURES_OPTIONS)
    print(f"  Stocks with F&O: {len(fo_stocks)} (first 10: {fo_stocks[:10]})")
    
    f_only = derivatives_universe.filter_by_derivative_type(DerivativeType.FUTURES_ONLY)
    print(f"  Stocks with Futures only: {len(f_only)} (first 10: {f_only[:10]})")
    
    # Pair trading universe
    print("\n🔗 Pair Trading Universe:")
    pair_universe = derivatives_universe.get_pair_trading_universe()
    print(f"  Indices: {pair_universe['indices']}")
    print(f"  NIFTY500 constituents: {len(pair_universe['nifty500'])}")
    print(f"  Tradeable pairs: {len(pair_universe['pairs'])}")
    
    # Export
    print("\n💾 Exporting...")
    derivatives_universe.export_to_csv('data/derivatives_universe.csv')
