# core/instruments_manager.py — NSE Instruments Management System
# Downloads and maintains complete list of all instruments traded on NSE
# Includes: Equities, Indices, Currencies, Commodities, Futures, Options, Derivatives Metadata

import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BASE_DIR
from utils.logger import logger
from utils.failure_logger import record_failure

# Instruments data directory
INSTRUMENTS_DIR = BASE_DIR.parent / 'instruments'
INSTRUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
INSTRUMENTS_FILE = INSTRUMENTS_DIR / 'all_instruments.csv'
INSTRUMENTS_JSON = INSTRUMENTS_DIR / 'all_instruments.json'
DERIVATIVES_METADATA = INSTRUMENTS_DIR / 'derivatives_metadata.json'
DERIVATIVES_EXPIRIES = INSTRUMENTS_DIR / 'expiries.json'
LAST_UPDATE_FILE = INSTRUMENTS_DIR / '.last_update'

# API endpoints
NSE_INSTRUMENTS_URL = "https://www.nseindia.com/api/market-data-csv-files"
NSE_HOLIDAYS_URL = "https://www.nseindia.com/api/holidays-master"

class InstrumentsManager:
    """Manage all NSE instruments and derivatives metadata"""
    
    def __init__(self):
        self.instruments = {}
        self.derivatives_metadata = {}
        self.expiries = {}
        self.load_instruments()
    
    def load_instruments(self):
        """Load instruments from local cache"""
        if INSTRUMENTS_JSON.exists():
            try:
                self.instruments = json.loads(INSTRUMENTS_JSON.read_text())
                logger.info(f"Loaded {len(self.instruments)} instruments from cache")
            except Exception as e:
                logger.warning(f"Failed to load instruments cache: {e}")
                try:
                    record_failure(symbol=None, exchange=None, reason="instruments_cache_load_failed", details=str(e))
                except Exception:
                    pass
        
        if DERIVATIVES_METADATA.exists():
            try:
                self.derivatives_metadata = json.loads(DERIVATIVES_METADATA.read_text())
                logger.info(f"Loaded derivatives metadata for {len(self.derivatives_metadata)} symbols")
            except Exception as e:
                logger.warning(f"Failed to load derivatives metadata: {e}")
                try:
                    record_failure(symbol=None, exchange=None, reason="derivatives_metadata_load_failed", details=str(e))
                except Exception:
                    pass
        
        if DERIVATIVES_EXPIRIES.exists():
            try:
                self.expiries = json.loads(DERIVATIVES_EXPIRIES.read_text())
                logger.info(f"Loaded expiries for {len(self.expiries)} symbols")
            except Exception as e:
                logger.warning(f"Failed to load expiries: {e}")
                try:
                    record_failure(symbol=None, exchange=None, reason="derivatives_expiries_load_failed", details=str(e))
                except Exception:
                    pass
    
    def check_for_updates(self) -> bool:
        """Check if instruments list needs updating (daily)
        
        Returns:
            bool: True if update needed, False otherwise
        """
        if not LAST_UPDATE_FILE.exists():
            return True
        
        try:
            last_update = datetime.fromisoformat(LAST_UPDATE_FILE.read_text().strip())
            if datetime.now() - last_update > timedelta(days=1):
                return True
        except:
            pass
        
        return False
    
    def download_all_instruments(self) -> Dict:
        """Download complete list of all NSE instruments
        
        Returns:
            dict: Downloaded instruments organized by type
        """
        logger.info("Downloading NSE instruments list...")
        
        instruments = {
            'equities': {},
            'indices': {},
            'currencies': {},
            'commodities': {},
            'futures': {},
            'options': {},
            'derivatives': {}
        }
        
        try:
            # Download equity instruments
            equities = self._download_equities()
            instruments['equities'] = equities
            logger.info(f"Downloaded {len(equities)} equities")
            
            # Download indices
            indices = self._download_indices()
            instruments['indices'] = indices
            logger.info(f"Downloaded {len(indices)} indices")
            
            # Download currencies
            currencies = self._download_currencies()
            instruments['currencies'] = currencies
            logger.info(f"Downloaded {len(currencies)} currencies")
            
            # Download commodities
            commodities = self._download_commodities()
            instruments['commodities'] = commodities
            logger.info(f"Downloaded {len(commodities)} commodities")
            
            # Download derivatives info
            futures, options = self._download_derivatives()
            instruments['futures'] = futures
            instruments['options'] = options
            logger.info(f"Downloaded {len(futures)} futures, {len(options)} options")
            
            # Save to files
            self._save_instruments(instruments)
            
            # Update last update time
            LAST_UPDATE_FILE.write_text(datetime.now().isoformat())
            
            self.instruments = instruments
            return instruments
        
        except Exception as e:
            logger.error(f"Failed to download instruments: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="download_all_instruments_failed", details=str(e))
            except Exception:
                pass
            return {}
    
    def _download_equities(self) -> Dict:
        """Download equity instruments from NSE"""
        equities = {}
        
        try:
            # Try to load from csv if available
            csv_path = Path(__file__).parent.parent / 'data' / 'instruments_NFO.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    symbol = row.get('Symbol', row.get('symbol', '')).strip().upper()
                    if symbol:
                        equities[symbol] = {
                            'name': row.get('Name', row.get('name', symbol)),
                            'exchange': 'NSE',
                            'type': 'equity',
                            'isin': row.get('ISIN', row.get('isin', '')),
                            'sector': row.get('Sector', row.get('sector', '')),
                            'industry': row.get('Industry', row.get('industry', '')),
                            'has_futures': False,
                            'has_options': False
                        }
            
            logger.info(f"Loaded {len(equities)} equities from local data")
        
        except Exception as e:
            logger.warning(f"Failed to download equities: {e}")
            try:
                record_failure(symbol=None, exchange="NSE", reason="equities_download_failed", details=str(e))
            except Exception:
                pass
        
        return equities
    
    def _download_indices(self) -> Dict:
        """Download indices"""
        indices = {
            'NIFTY50': {'name': 'NIFTY 50', 'exchange': 'NSE', 'type': 'index'},
            'NIFTY100': {'name': 'NIFTY 100', 'exchange': 'NSE', 'type': 'index'},
            'NIFTY200': {'name': 'NIFTY 200', 'exchange': 'NSE', 'type': 'index'},
            'NIFTY500': {'name': 'NIFTY 500', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYAUTO': {'name': 'NIFTY Auto', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYBANK': {'name': 'NIFTY Bank', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYINFRA': {'name': 'NIFTY Infrastructure', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYIT': {'name': 'NIFTY IT', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYPHARMA': {'name': 'NIFTY Pharma', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYPVTBANK': {'name': 'NIFTY Private Bank', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYCPSE': {'name': 'NIFTY CPSE', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYPSE': {'name': 'NIFTY PSE', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYLOWVOL50': {'name': 'NIFTY Low Volatility 50', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYSMALLCAP50': {'name': 'NIFTY Smallcap 50', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYSMALLCAP250': {'name': 'NIFTY Smallcap 250', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYMIDCAP50': {'name': 'NIFTY Midcap 50', 'exchange': 'NSE', 'type': 'index'},
            'NIFTYMIDCAP150': {'name': 'NIFTY Midcap 150', 'exchange': 'NSE', 'type': 'index'},
            'INDIA_VIX': {'name': 'India VIX', 'exchange': 'NSE', 'type': 'index'},
        }
        return indices
    
    def _download_currencies(self) -> Dict:
        """Download currency pairs"""
        currencies = {
            'EURINR': {'name': 'EUR/INR', 'exchange': 'NSE', 'type': 'currency'},
            'GBPINR': {'name': 'GBP/INR', 'exchange': 'NSE', 'type': 'currency'},
            'JPYINR': {'name': 'JPY/INR', 'exchange': 'NSE', 'type': 'currency'},
            'USDINR': {'name': 'USD/INR', 'exchange': 'NSE', 'type': 'currency'},
        }
        return currencies
    
    def _download_commodities(self) -> Dict:
        """Download commodities"""
        commodities = {
            'GOLD': {'name': 'Gold', 'exchange': 'MCX', 'type': 'commodity'},
            'SILVER': {'name': 'Silver', 'exchange': 'MCX', 'type': 'commodity'},
            'CRUDEOIL': {'name': 'Crude Oil', 'exchange': 'MCX', 'type': 'commodity'},
            'NATURALGAS': {'name': 'Natural Gas', 'exchange': 'MCX', 'type': 'commodity'},
            'COPPER': {'name': 'Copper', 'exchange': 'MCX', 'type': 'commodity'},
            'ALUMINUM': {'name': 'Aluminum', 'exchange': 'MCX', 'type': 'commodity'},
            'ZINC': {'name': 'Zinc', 'exchange': 'MCX', 'type': 'commodity'},
            'NICKEL': {'name': 'Nickel', 'exchange': 'MCX', 'type': 'commodity'},
        }
        return commodities
    
    def _download_derivatives(self) -> Tuple[Dict, Dict]:
        """Download derivatives (futures and options) metadata"""
        futures = {}
        options = {}
        
        try:
            # Futures stocks - Major NSE stocks with futures
            futures_stocks = [
                'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HDFC', 'ICICIBANK',
                'AXISBANK', 'SBIN', 'BAJAJFINSV', 'MARUTI', 'HINDUNILVR',
                'ITC', 'LT', 'WIPRO', 'ASIANPAINT', 'SUNPHARMA',
                'KOTAKBANK', 'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA',
                'JSWSTEEL', 'TATASTEEL', 'HEROMOTOCORP', 'BAJAJ-AUTO',
                'BHARTIARTL', 'APOLLOHSP', 'INFRATEL', 'INDIGO', 'TITAN',
                'ADANIPORTS', 'ADANIPOWER', 'ADANIENT', 'ADANIGREEN',
                'BANKINDIA', 'BPCL', 'GAIL', 'HINDPETRO', 'IBREALEST',
                'IBULHSGFIN', 'INDHOTEL', 'IOC', 'JSWSTEEL', 'LTTS',
                'MARICO', 'NESTLEIND', 'PGHISERIES', 'PIDILITIND', 'RBLBANK',
                'SBICARD', 'SBILIFE', 'SHREECEM', 'SIEMENS', 'SUZLON',
                'TECHM', 'UPL', 'ULTRACEMCO', 'VEDL', 'YESBANK',
                'NIFTY', 'BANKNIFTY'
            ]
            
            for symbol in futures_stocks:
                futures[symbol] = {
                    'name': symbol,
                    'type': 'futures',
                    'exchange': 'NSE',
                    'lot_size': self._get_lot_size(symbol),
                    'has_options': symbol in self._get_options_stocks()
                }
            
            # Options stocks - Stocks with options
            options_stocks = self._get_options_stocks()
            for symbol in options_stocks:
                options[symbol] = {
                    'name': symbol,
                    'type': 'options',
                    'exchange': 'NSE',
                    'lot_size': self._get_lot_size(symbol),
                    'has_futures': symbol in futures_stocks
                }
            
            logger.info(f"Loaded {len(futures)} futures, {len(options)} options")
        
        except Exception as e:
            logger.warning(f"Failed to download derivatives: {e}")
            try:
                record_failure(symbol=None, exchange="NSE", reason="derivatives_download_failed", details=str(e))
            except Exception:
                pass
        
        return futures, options
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size for a symbol"""
        # Common lot sizes
        lot_sizes = {
            'NIFTY': 50,
            'BANKNIFTY': 25,
            'FINNIFTY': 25,
            'MIDCPNIFTY': 75,
            'RELIANCE': 1,
            'TCS': 1,
            'INFY': 1,
            'HDFCBANK': 1,
            'HDFC': 1,
            'ICICIBANK': 1,
            'AXISBANK': 1,
            'SBIN': 1,
            'HINDUNILVR': 1,
            'MARUTI': 1,
            'LT': 1,
            'WIPRO': 1,
            'ASIANPAINT': 1,
            'SUNPHARMA': 1,
            'KOTAKBANK': 1,
            'POWERGRID': 1,
            'NTPC': 1,
            'ONGC': 1,
        }
        return lot_sizes.get(symbol, 1)
    
    def _get_options_stocks(self) -> List[str]:
        """Get list of stocks with options"""
        return [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HDFC', 'ICICIBANK',
            'AXISBANK', 'SBIN', 'BAJAJFINSV', 'MARUTI', 'HINDUNILVR',
            'ITC', 'LT', 'WIPRO', 'ASIANPAINT', 'SUNPHARMA',
            'KOTAKBANK', 'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA',
            'JSWSTEEL', 'TATASTEEL', 'HEROMOTOCORP', 'BAJAJ-AUTO',
            'BHARTIARTL', 'APOLLOHSP', 'INDIGO', 'TITAN',
            'ADANIPORTS', 'ADANIGREEN',
            'NESTLEIND', 'RBLBANK', 'SBICARD', 'SBILIFE',
            'TECHM', 'UPL', 'ULTRACEMCO', 'VEDL',
            'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'
        ]
    
    def _save_instruments(self, instruments: Dict):
        """Save instruments to files"""
        try:
            # Save as JSON
            INSTRUMENTS_JSON.write_text(json.dumps(instruments, indent=2, default=str))
            
            # Save as CSV (flat format)
            rows = []
            for category, items in instruments.items():
                for symbol, info in items.items():
                    info['category'] = category
                    info['symbol'] = symbol
                    rows.append(info)
            
            df = pd.DataFrame(rows)
            df.to_csv(INSTRUMENTS_FILE, index=False)
            
            logger.info(f"Saved instruments to {INSTRUMENTS_FILE} and {INSTRUMENTS_JSON}")
        
        except Exception as e:
            logger.error(f"Failed to save instruments: {e}")
            try:
                record_failure(symbol=None, exchange=None, reason="instruments_save_failed", details=str(e))
            except Exception:
                pass
    
    def get_instruments_by_type(self) -> Dict:
        """Get instruments organized by type
        
        Returns:
            dict: {type: {symbol: info}}
        """
        return self.instruments
    
    def get_instruments_by_exchange(self) -> Dict:
        """Get instruments organized by exchange
        
        Returns:
            dict: {exchange: {type: {symbol: info}}}
        """
        by_exchange = {}
        
        for category, items in self.instruments.items():
            for symbol, info in items.items():
                exchange = info.get('exchange', 'NSE')
                if exchange not in by_exchange:
                    by_exchange[exchange] = {}
                if category not in by_exchange[exchange]:
                    by_exchange[exchange][category] = {}
                
                by_exchange[exchange][category][symbol] = info
        
        return by_exchange
    
    def get_derivatives_metadata(self, symbol: str = None) -> Dict:
        """Get derivatives metadata (expiries, lot sizes, options availability)
        
        Args:
            symbol: Specific symbol or None for all
        
        Returns:
            dict: Derivatives metadata
        """
        if symbol:
            return self.derivatives_metadata.get(symbol, {})
        return self.derivatives_metadata
    
    def get_expiries(self, symbol: str = None) -> Dict:
        """Get expiry dates for derivatives
        
        Args:
            symbol: Specific symbol or None for all
        
        Returns:
            dict: Expiry information
        """
        if symbol:
            return self.expiries.get(symbol, {})
        return self.expiries
    
    def get_coverage_stats(self) -> Dict:
        """Get statistics about instruments coverage by segment and exchange
        
        Returns:
            dict: Coverage statistics
        """
        stats = {
            'by_type': {},
            'by_exchange': {},
            'by_exchange_and_type': {}
        }
        
        # By type
        for category, items in self.instruments.items():
            stats['by_type'][category] = len(items)
        
        # By exchange
        by_exchange = self.get_instruments_by_exchange()
        for exchange, types in by_exchange.items():
            total = sum(len(symbols) for symbols in types.values())
            stats['by_exchange'][exchange] = {
                'total': total,
                'by_type': {t: len(s) for t, s in types.items()}
            }
        
        # By exchange and type
        for exchange, types in by_exchange.items():
            stats['by_exchange_and_type'][exchange] = {}
            for category, items in types.items():
                stats['by_exchange_and_type'][exchange][category] = len(items)
        
        return stats


# Convenience function
_instruments_manager = None

def get_instruments_manager() -> InstrumentsManager:
    """Get or create InstrumentsManager singleton"""
    global _instruments_manager
    if _instruments_manager is None:
        _instruments_manager = InstrumentsManager()
    return _instruments_manager
