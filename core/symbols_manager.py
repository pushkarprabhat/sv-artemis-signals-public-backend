# core/symbols_manager.py — Persistent Symbol & Exchange Metadata Management
# Maintains NSE, MCX symbols with instrument types and exchange information
# Provides singleton access throughout the application

import json
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
from utils.failure_logger import record_failure


class ExchangeMarketHours:
    """Market hours configuration for each exchange"""
    
    MARKET_HOURS = {
        'NSE': {
            'name': 'National Stock Exchange',
            'trading_days': ['MON', 'TUE', 'WED', 'THU', 'FRI'],
            'sessions': [
                {
                    'name': 'Pre-Market',
                    'start': time(9, 0),
                    'end': time(9, 15),
                    'type': 'auction'
                },
                {
                    'name': 'Regular Trading',
                    'start': time(9, 15),
                    'end': time(15, 30),
                    'type': 'live'
                },
                {
                    'name': 'Post-Market',
                    'start': time(15, 30),
                    'end': time(16, 0),
                    'type': 'auction'
                }
            ],
            'holidays': [
                '2025-01-26',  # Republic Day
                '2025-03-08',  # Maha Shivaratri
                '2025-03-10',  # Holi
                '2025-03-29',  # Good Friday
                '2025-04-17',  # Ram Navami
                '2025-04-21',  # Mahavir Jayanti
                '2025-08-15',  # Independence Day
                '2025-08-27',  # Janmashtami
                '2025-09-16',  # Milad un-Nabi
                '2025-10-02',  # Gandhi Jayanti
                '2025-10-20',  # Dussehra
                '2025-11-01',  # Diwali
                '2025-11-15',  # Guru Nanak Jayanti
                '2025-12-25',  # Christmas
            ]
        },
        'MCX': {
            'name': 'Multi Commodity Exchange',
            'trading_days': ['MON', 'TUE', 'WED', 'THU', 'FRI'],
            'sessions': [
                {
                    'name': 'Morning',
                    'start': time(9, 0),
                    'end': time(17, 30),
                    'type': 'live'
                },
                {
                    'name': 'Evening',
                    'start': time(17, 45),
                    'end': time(23, 30),
                    'type': 'live'
                }
            ],
            'holidays': [
                '2025-01-26',  # Republic Day
                '2025-03-08',  # Maha Shivaratri
                '2025-03-10',  # Holi
                '2025-04-17',  # Ram Navami
                '2025-08-15',  # Independence Day
                '2025-08-27',  # Janmashtami
                '2025-10-02',  # Gandhi Jayanti
                '2025-10-20',  # Dussehra
                '2025-11-01',  # Diwali
                '2025-11-15',  # Guru Nanak Jayanti
                '2025-12-25',  # Christmas
            ]
        }
    }
    
    @classmethod
    def is_market_open(cls, exchange: str, check_time: datetime = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if market is open for an exchange
        
        Args:
            exchange: Exchange name (NSE, MCX)
            check_time: DateTime to check (defaults to now)
        
        Returns:
            Tuple: (is_open: bool, session_name: str or None, next_session: str or None)
        """
        if check_time is None:
            check_time = datetime.now()
        
        if exchange not in cls.MARKET_HOURS:
            return False, None, "Unknown exchange"
        
        config = cls.MARKET_HOURS[exchange]
        
        # Check if holiday
        check_date = check_time.strftime('%Y-%m-%d')
        if check_date in config['holidays']:
            return False, None, "Holiday"
        
        # Check trading day
        day_name = check_time.strftime('%a').upper()
        if day_name not in config['trading_days']:
            return False, None, f"Market closed on {day_name}"
        
        # Check session
        current_time = check_time.time()
        for session in config['sessions']:
            if session['start'] <= current_time < session['end']:
                return session['type'] == 'live', session['name'], None
        
        # Market closed, find next session
        next_session = None
        for session in config['sessions']:
            if session['start'] > current_time:
                next_session = f"{session['name']} ({session['start'].strftime('%H:%M')})"
                break
        
        return False, None, next_session
    
    @classmethod
    def get_market_status(cls, exchange: str) -> Dict:
        """Get complete market status for an exchange"""
        is_open, session, next_info = cls.is_market_open(exchange)
        
        config = cls.MARKET_HOURS[exchange]
        sessions = config['sessions']
        
        return {
            'exchange': exchange,
            'is_open': is_open,
            'current_session': session,
            'next_session': next_info,
            'name': config['name'],
            'sessions': [
                {
                    'name': s['name'],
                    'start': s['start'].strftime('%H:%M'),
                    'end': s['end'].strftime('%H:%M'),
                    'type': s['type']
                }
                for s in sessions
            ]
        }


class SymbolsManager:
    """
    Persistent management of symbols across exchanges
    Singleton pattern for app-wide access
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.data_dir = Path("data/symbols_metadata")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.nse_file = self.data_dir / "nse_symbols.json"
        self.mcx_file = self.data_dir / "mcx_symbols.json"
        self.metadata_file = self.data_dir / "metadata.json"
        
        self.nse_symbols = {}
        self.mcx_symbols = {}
        self.metadata = {}
        
        self._load()
        self._initialized = True
    
    def _load(self):
        """Load symbol databases from disk"""
        try:
            if self.nse_file.exists():
                self.nse_symbols = json.loads(self.nse_file.read_text())
            if self.mcx_file.exists():
                self.mcx_symbols = json.loads(self.mcx_file.read_text())
            if self.metadata_file.exists():
                self.metadata = json.loads(self.metadata_file.read_text())
            logger.info(f"Loaded {len(self.nse_symbols)} NSE symbols, {len(self.mcx_symbols)} MCX symbols")
        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="symbols_load_failed", details=str(e))
            except Exception:
                pass
    
    def _save(self):
        """Save symbol databases to disk"""
        try:
            self.nse_file.write_text(json.dumps(self.nse_symbols, indent=2))
            self.mcx_file.write_text(json.dumps(self.mcx_symbols, indent=2))
            self.metadata_file.write_text(json.dumps(self.metadata, indent=2))
        except Exception as e:
            logger.error(f"Error saving symbols: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="symbols_save_failed", details=str(e))
            except Exception:
                pass
    
    def add_nse_symbol(self, symbol: str, instrument_type: str, sector: Optional[str] = None, 
                      market_cap: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add NSE symbol"""
        self.nse_symbols[symbol] = {
            'exchange': 'NSE',
            'instrument_type': instrument_type,
            'sector': sector,
            'market_cap': market_cap,
            'added_date': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save()
    
    def add_mcx_symbol(self, symbol: str, commodity_type: str, contract_month: Optional[str] = None,
                      lot_size: Optional[int] = None, metadata: Optional[Dict] = None):
        """Add MCX symbol"""
        self.mcx_symbols[symbol] = {
            'exchange': 'MCX',
            'commodity_type': commodity_type,
            'contract_month': contract_month,
            'lot_size': lot_size,
            'added_date': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save()
    
    def get_nse_symbols(self, instrument_type: Optional[str] = None) -> Dict:
        """Get NSE symbols, optionally filtered by type"""
        if instrument_type:
            return {k: v for k, v in self.nse_symbols.items() if v.get('instrument_type') == instrument_type}
        return self.nse_symbols
    
    def get_mcx_symbols(self, commodity_type: Optional[str] = None) -> Dict:
        """Get MCX symbols, optionally filtered by commodity type"""
        if commodity_type:
            return {k: v for k, v in self.mcx_symbols.items() if v.get('commodity_type') == commodity_type}
        return self.mcx_symbols
    
    def get_all_symbols(self) -> Dict[str, Dict]:
        """Get all symbols organized by exchange"""
        return {
            'NSE': self.nse_symbols,
            'MCX': self.mcx_symbols
        }
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get info for a specific symbol"""
        if symbol in self.nse_symbols:
            return self.nse_symbols[symbol]
        if symbol in self.mcx_symbols:
            return self.mcx_symbols[symbol]
        return None
    
    def get_instrument_types(self) -> Dict[str, List[str]]:
        """Get all instrument types by exchange"""
        nse_types = list(set(v.get('instrument_type') for v in self.nse_symbols.values() if v.get('instrument_type')))
        mcx_types = list(set(v.get('commodity_type') for v in self.mcx_symbols.values() if v.get('commodity_type')))
        
        return {
            'NSE': sorted(nse_types),
            'MCX': sorted(mcx_types)
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about symbol database"""
        nse_types = {}
        for sym in self.nse_symbols.values():
            t = sym.get('instrument_type', 'unknown')
            nse_types[t] = nse_types.get(t, 0) + 1
        
        mcx_types = {}
        for sym in self.mcx_symbols.values():
            t = sym.get('commodity_type', 'unknown')
            mcx_types[t] = mcx_types.get(t, 0) + 1
        
        return {
            'nse_total': len(self.nse_symbols),
            'nse_by_type': nse_types,
            'mcx_total': len(self.mcx_symbols),
            'mcx_by_type': mcx_types,
            'total_symbols': len(self.nse_symbols) + len(self.mcx_symbols),
            'last_updated': self.metadata.get('last_updated', 'unknown')
        }
    
    def initialize_with_defaults(self):
        """Initialize with default NSE and MCX symbols"""
        from config import INDICES
        
        # NSE Equities (from universe.csv, populated by InstrumentsManager)
        # NSE Indices
        for index_name, index_info in INDICES.items():
            self.add_nse_symbol(
                index_name,
                instrument_type='index',
                sector=None,
                metadata={'description': index_info.get('description', '')}
            )
        
        # MCX Commodities
        commodities = {
            'GOLDM': {'name': 'Gold Micro', 'lot_size': 1},
            'GOLD': {'name': 'Gold', 'lot_size': 100},
            'SILVERM': {'name': 'Silver Micro', 'lot_size': 1},
            'SILVER': {'name': 'Silver', 'lot_size': 30},
            'CRUDEOIL': {'name': 'Crude Oil', 'lot_size': 100},
            'NATURALGAS': {'name': 'Natural Gas', 'lot_size': 1},
            'COPPER': {'name': 'Copper', 'lot_size': 250},
            'ALUMINUM': {'name': 'Aluminum', 'lot_size': 5},
            'ZINC': {'name': 'Zinc', 'lot_size': 250},
            'NICKEL': {'name': 'Nickel', 'lot_size': 100},
        }
        
        for symbol, info in commodities.items():
            self.add_mcx_symbol(
                symbol,
                commodity_type='metal' if 'GOLD' in symbol or 'SILVER' in symbol else 'energy',
                lot_size=info.get('lot_size'),
                metadata={'name': info.get('name')}
            )
        
        self.metadata['last_updated'] = datetime.now().isoformat()
        self._save()
        logger.info("Initialized SymbolsManager with default symbols")


def get_symbols_manager() -> SymbolsManager:
    """Get singleton instance of SymbolsManager"""
    return SymbolsManager()
