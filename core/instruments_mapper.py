# core/instruments_mapper.py
# Handles Zerodha Kite CSV instruments dump parsing and JSON conversion
# Also manages cross-provider instrument mapping

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from config import BASE_DIR
from utils.logger import logger

logger = logging.getLogger(__name__)


@dataclass
class Instrument:
    """Represents a single instrument"""
    instrument_token: int
    exchange_token: int
    tradingsymbol: str
    name: str
    exchange: str
    segment: str
    lot_size: int
    expiry: Optional[str]
    instrument_type: str
    strike: float
    tick_size: float
    last_price: float


class InstrumentsMapper:
    """
    Manages instrument data from Zerodha Kite Connect API
    
    Features:
    - Parse Kite CSV dump to JSON
    - Store and retrieve instruments
    - Cross-provider instrument mapping
    - Segment filtering (NSE EQ, BSE EQ, NFO-FUT, etc.)
    
    Kite CSV Columns:
    - instrument_token (int)
    - exchange_token (int)
    - tradingsymbol (str)
    - name (str)
    - exchange (str): NSE, BSE, NFO, CDS, MCX, NCDEX
    - segment (str): NSE EQ, BSE EQ, NFO-FUT, NFO-OPT, etc.
    - lot_size (int)
    - expiry (date or null)
    - instrument_type (str): EQ, FUT, OPT, IND
    - strike (float)
    - tick_size (float)
    - last_price (float)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize instruments mapper"""
        self.data_dir = data_dir or (BASE_DIR / "data")
        self.instruments_dir = self.data_dir / "instruments"
        self.instruments_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.instruments: Dict[int, Instrument] = {}  # token -> instrument
        self.symbol_to_instruments: Dict[str, List[int]] = {}  # symbol -> [tokens]
        self.segment_instruments: Dict[str, List[int]] = {}  # segment -> [tokens]
        
        # Load existing
        self._load_instruments()
        
        logger.info(f"InstrumentsMapper initialized with {len(self.instruments)} instruments")
    
    # ========================================================================
    # CSV PARSING & CONVERSION
    # ========================================================================
    
    def parse_kite_csv(self, csv_path: Path) -> List[Dict]:
        """
        Parse Zerodha Kite CSV dump and convert to list of dictionaries
        
        The CSV file from Kite Connect API (/instruments endpoint) contains
        all available instruments in gzipped CSV format.
        
        Args:
            csv_path: Path to CSV file (can be gzipped)
        
        Returns:
            List of instrument dictionaries
        """
        try:
            # Handle gzipped files
            if str(csv_path).endswith('.gz'):
                df = pd.read_csv(csv_path, compression='gzip')
            else:
                df = pd.read_csv(csv_path)
            
            logger.info(f"Loaded {len(df)} instruments from {csv_path}")
            
            # Convert to list of dicts
            instruments_list = df.to_dict('records')
            
            return instruments_list
        
        except Exception as e:
            logger.error(f"Failed to parse Kite CSV: {e}")
            return []
    
    def csv_to_json(self, csv_path: Path, output_path: Optional[Path] = None) -> Dict:
        """
        Convert Kite CSV dump to JSON format
        
        Args:
            csv_path: Path to CSV file
            output_path: Where to save JSON (default: instruments_dir/kite.json)
        
        Returns:
            Dictionary of instruments grouped by segment
        """
        # Parse CSV
        instruments_list = self.parse_kite_csv(csv_path)
        
        if not instruments_list:
            return {}
        
        # Group by segment
        grouped = {}
        for instr in instruments_list:
            segment = instr.get('segment', 'UNKNOWN')
            if segment not in grouped:
                grouped[segment] = []
            grouped[segment].append(instr)
        
        # Save to JSON
        output_file = output_path or (self.instruments_dir / "kite_instruments.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(grouped, f, indent=2)
            logger.info(f"Saved {len(instruments_list)} instruments to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
        
        return grouped
    
    # ========================================================================
    # LOADING & STORAGE
    # ========================================================================
    
    def _load_instruments(self) -> None:
        """Load instruments from JSON"""
        json_file = self.instruments_dir / "kite_instruments.json"
        
        if not json_file.exists():
            logger.warning("No instruments JSON found")
            return
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Flatten and load
            for segment, instr_list in data.items():
                for instr_data in instr_list:
                    instr = Instrument(
                        instrument_token=instr_data.get('instrument_token', 0),
                        exchange_token=instr_data.get('exchange_token', 0),
                        tradingsymbol=instr_data.get('tradingsymbol', ''),
                        name=instr_data.get('name', ''),
                        exchange=instr_data.get('exchange', ''),
                        segment=instr_data.get('segment', ''),
                        lot_size=instr_data.get('lot_size', 1),
                        expiry=instr_data.get('expiry'),
                        instrument_type=instr_data.get('instrument_type', ''),
                        strike=instr_data.get('strike', 0.0),
                        tick_size=instr_data.get('tick_size', 0.05),
                        last_price=instr_data.get('last_price', 0.0),
                    )
                    
                    self.instruments[instr.instrument_token] = instr
                    
                    # Update indices
                    if instr.tradingsymbol not in self.symbol_to_instruments:
                        self.symbol_to_instruments[instr.tradingsymbol] = []
                    self.symbol_to_instruments[instr.tradingsymbol].append(instr.instrument_token)
                    
                    if segment not in self.segment_instruments:
                        self.segment_instruments[segment] = []
                    self.segment_instruments[segment].append(instr.instrument_token)
            
            logger.info(f"Loaded {len(self.instruments)} instruments from JSON")
        
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
    
    def save_instruments_json(self, output_path: Path) -> bool:
        """Save current instruments to JSON"""
        try:
            data = {}
            for instr in self.instruments.values():
                segment = instr.segment
                if segment not in data:
                    data[segment] = []
                data[segment].append(asdict(instr))
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.instruments)} instruments")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save instruments: {e}")
            return False
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    def get_instrument(self, token: int) -> Optional[Instrument]:
        """Get instrument by token"""
        return self.instruments.get(token)
    
    def get_instruments_by_symbol(self, symbol: str) -> List[Instrument]:
        """Get all instruments with a given symbol (e.g., equity + options)"""
        tokens = self.symbol_to_instruments.get(symbol, [])
        return [self.instruments[token] for token in tokens]
    
    def get_instruments_by_segment(self, segment: str) -> List[Instrument]:
        """Get all instruments in a segment"""
        tokens = self.segment_instruments.get(segment, [])
        return [self.instruments[token] for token in tokens]
    
    def get_equity_symbols(self) -> List[str]:
        """Get all equity symbols (NSE & BSE)"""
        symbols = set()
        for segment in ['NSE EQ', 'BSE EQ']:
            for token in self.segment_instruments.get(segment, []):
                symbols.add(self.instruments[token].tradingsymbol)
        return sorted(list(symbols))
    
    def get_nfo_symbols(self) -> List[str]:
        """Get all NFO (F&O) symbols"""
        symbols = set()
        for segment in ['NFO-FUT', 'NFO-OPT']:
            for token in self.segment_instruments.get(segment, []):
                symbols.add(self.instruments[token].tradingsymbol)
        return sorted(list(symbols))
    
    def get_future_contracts(self, base_symbol: str) -> List[Instrument]:
        """Get all future contracts for a symbol"""
        contracts = []
        symbols = self.get_instruments_by_symbol(base_symbol)
        for instr in symbols:
            if instr.instrument_type == 'FUT':
                contracts.append(instr)
        return contracts
    
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> List[Instrument]:
        """
        Get option chain for a symbol
        
        Args:
            symbol: Base symbol (e.g., 'NIFTY50')
            expiry: Optional specific expiry date
        
        Returns:
            List of option instruments
        """
        options = []
        symbols = self.get_instruments_by_symbol(symbol)
        for instr in symbols:
            if instr.instrument_type == 'OPT':
                if expiry is None or instr.expiry == expiry:
                    options.append(instr)
        return options
    
    # ========================================================================
    # CROSS-PROVIDER MAPPING
    # ========================================================================
    
    def create_provider_mapping(
        self,
        provider_name: str,
        provider_symbols: List[str],
    ) -> Dict[str, int]:
        """
        Create mapping from provider symbols to Kite tokens
        
        Args:
            provider_name: Name of provider (e.g., 'polygon')
            provider_symbols: List of symbols from provider
        
        Returns:
            Dictionary mapping provider_symbol -> kite_token
        """
        mapping = {}
        
        for provider_symbol in provider_symbols:
            # Try exact match
            eq_symbols = self.get_instruments_by_symbol(provider_symbol)
            
            if eq_symbols:
                # Get equity variant if available
                for instr in eq_symbols:
                    if instr.segment in ['NSE EQ', 'BSE EQ']:
                        mapping[provider_symbol] = instr.instrument_token
                        break
                
                # If no equity, use first available
                if provider_symbol not in mapping:
                    mapping[provider_symbol] = eq_symbols[0].instrument_token
        
        return mapping
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded instruments"""
        stats = {
            'total_instruments': len(self.instruments),
            'total_symbols': len(self.symbol_to_instruments),
            'segments': {},
        }
        
        for segment, tokens in self.segment_instruments.items():
            stats['segments'][segment] = len(tokens)
        
        return stats


def get_instruments_mapper() -> InstrumentsMapper:
    """Get or create singleton instance"""
    if not hasattr(get_instruments_mapper, '_instance'):
        get_instruments_mapper._instance = InstrumentsMapper()
    return get_instruments_mapper._instance
