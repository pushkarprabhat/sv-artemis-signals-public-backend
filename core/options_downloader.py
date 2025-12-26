# core/options_downloader.py — Options Chain Downloader
# Downloads: Options chains for all expiries and strikes
# Supports: Index and Stock options
# Data storage: Per-expiry, per-symbol structure

import pandas as pd
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import BASE_DIR
from utils.logger import logger
import utils.helpers


class OptionsDownloader:
    """Downloads options chain data"""
    
    def __init__(self):
        self.kite = utils.helpers.kite
        self.base_dir = BASE_DIR / 'data' / 'options'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._instrument_cache = {}
        self._expiry_cache = {}
    
    def _get_instruments(self):
        """Fetch and cache NFO instruments"""
        if self._instrument_cache:
            return self._instrument_cache
        
        try:
            instruments = self.kite.instruments('NFO')
            self._instrument_cache = {instr['tradingsymbol']: instr for instr in instruments}
            logger.info(f"[OPTIONS] Cached {len(self._instrument_cache)} NFO instruments")
            return self._instrument_cache
        except Exception as e:
            logger.error(f"[OPTIONS] Error fetching instruments: {e}")
            return {}
    
    def fetch_expiries(self, symbol):
        """Fetch available expiry dates for a symbol
        
        Args:
            symbol: 'NIFTY50', 'BANKNIFTY', 'TCS', etc.
        
        Returns:
            List of expiry dates as strings (YYYY-MM-DD)
        """
        if symbol in self._expiry_cache:
            return self._expiry_cache[symbol]
        
        try:
            instruments = self._get_instruments()
            
            # Find all expirations for this symbol
            expiries = set()
            for trading_symbol, instr in instruments.items():
                if instr.get('name') == symbol and instr.get('segment') == 'NFO-OPT':
                    if 'expiry' in instr:
                        expiry_date = instr['expiry']
                        if isinstance(expiry_date, str):
                            expiries.add(expiry_date)
                        else:
                            expiries.add(expiry_date.strftime('%Y-%m-%d'))
            
            expiry_list = sorted(list(expiries))
            self._expiry_cache[symbol] = expiry_list
            logger.info(f"[OPTIONS] Found {len(expiry_list)} expiries for {symbol}")
            
            return expiry_list
        
        except Exception as e:
            logger.error(f"[OPTIONS] Error fetching expiries for {symbol}: {e}")
            return []
    
    def download_options_chain(self, symbol, expiry, category='index'):
        """Download complete options chain for a symbol/expiry
        
        Args:
            symbol: 'NIFTY50', 'BANKNIFTY', 'TCS', etc.
            expiry: Expiry date string (YYYY-MM-DD)
            category: 'index' or 'stocks'
        
        Returns:
            dict: {'calls': DataFrame, 'puts': DataFrame} or {'error': msg}
        """
        try:
            instruments = self._get_instruments()
            
            # Find all option contracts for this symbol/expiry
            calls_data = []
            puts_data = []
            
            for trading_symbol, instr in instruments.items():
                if (instr.get('name') != symbol or 
                    instr.get('segment') != 'NFO-OPT'):
                    continue
                
                # Check if this is the right expiry
                instr_expiry = instr.get('expiry')
                if isinstance(instr_expiry, str):
                    instr_expiry_str = instr_expiry
                else:
                    instr_expiry_str = instr_expiry.strftime('%Y-%m-%d')
                
                if instr_expiry_str != expiry:
                    continue
                
                # Get quote for this contract
                try:
                    quote = self.kite.quote(instr['instrument_token'])
                    
                    if quote and instr['instrument_token'] in quote:
                        quote_data = quote[instr['instrument_token']]
                        
                        # Extract option data
                        option_data = {
                            'trading_symbol': trading_symbol,
                            'instrument_token': instr['instrument_token'],
                            'strike_price': instr.get('strike'),
                            'instrument_type': instr.get('instrument_type'),
                            'ltp': quote_data.get('last_price', 0),
                            'bid': quote_data.get('bid', 0),
                            'ask': quote_data.get('ask', 0),
                            'volume': quote_data.get('volume', 0),
                            'oi': quote_data.get('oi', 0),
                            'bid_qty': quote_data.get('bid_qty', 0),
                            'ask_qty': quote_data.get('ask_qty', 0),
                            'timestamp': dt.datetime.now()
                        }
                        
                        if instr.get('instrument_type') == 'CE':
                            calls_data.append(option_data)
                        else:
                            puts_data.append(option_data)
                
                except Exception as e:
                    logger.warning(f"[OPTIONS] Error fetching quote for {trading_symbol}: {e}")
                
                # Rate limiting
                time.sleep(0.1)
            
            # Convert to DataFrames
            result = {}
            if calls_data:
                df_calls = pd.DataFrame(calls_data)
                result['calls'] = df_calls
                logger.info(f"[OPTIONS] Downloaded {len(df_calls)} calls for {symbol}/{expiry}")
            else:
                logger.warning(f"[OPTIONS] No calls found for {symbol}/{expiry}")
            
            if puts_data:
                df_puts = pd.DataFrame(puts_data)
                result['puts'] = df_puts
                logger.info(f"[OPTIONS] Downloaded {len(df_puts)} puts for {symbol}/{expiry}")
            else:
                logger.warning(f"[OPTIONS] No puts found for {symbol}/{expiry}")
            
            # Save to parquet
            symbol_dir = self.base_dir / category / symbol / expiry
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            if 'calls' in result:
                result['calls'].to_parquet(symbol_dir / 'calls.parquet', index=False)
            if 'puts' in result:
                result['puts'].to_parquet(symbol_dir / 'puts.parquet', index=False)
            
            logger.info(f"[OPTIONS] ✓ Saved chain for {symbol}/{expiry}")
            return result
        
        except Exception as e:
            logger.error(f"[OPTIONS] Error downloading chain for {symbol}/{expiry}: {e}")
            return {'error': str(e)}
    
    def download_all_expiries(self, symbol, category='index', max_expiries=3):
        """Download all or recent expiries for a symbol
        
        Args:
            symbol: Symbol name
            category: 'index' or 'stocks'
            max_expiries: Maximum number of expiries to download (0 = all)
        
        Returns:
            dict: Results for each expiry
        """
        logger.info(f"[OPTIONS] Downloading options for {symbol}")
        
        expiries = self.fetch_expiries(symbol)
        if not expiries:
            logger.error(f"[OPTIONS] No expiries found for {symbol}")
            return {}
        
        # Limit expiries if requested
        if max_expiries > 0:
            expiries = expiries[:max_expiries]
        
        results = {}
        for expiry in expiries:
            results[expiry] = self.download_options_chain(symbol, expiry, category)
        
        return results
    
    def download_index_options(self, symbols=['NIFTY50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
                               max_expiries=3):
        """Download index options
        
        Args:
            symbols: Index symbols
            max_expiries: Max expiries per symbol
        
        Returns:
            dict: Results for all symbols
        """
        logger.info(f"[OPTIONS] Downloading index options: {symbols}")
        all_results = {}
        
        for symbol in symbols:
            all_results[symbol] = self.download_all_expiries(symbol, 'index', max_expiries)
        
        return all_results
    
    def download_stock_options(self, symbols=None, max_expiries=3):
        """Download stock options
        
        Args:
            symbols: Stock symbols (None = sample)
            max_expiries: Max expiries per symbol
        
        Returns:
            dict: Results for all symbols
        """
        if symbols is None:
            symbols = ['TCS', 'INFY', 'RELIANCE', 'HDFC', 'ICICIBANK']
        
        logger.info(f"[OPTIONS] Downloading stock options: {len(symbols)} symbols")
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures_map = {
                executor.submit(self.download_all_expiries, symbol, 'stocks', max_expiries): symbol
                for symbol in symbols
            }
            
            for future in as_completed(futures_map):
                symbol = futures_map[future]
                try:
                    all_results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"[OPTIONS] Error downloading {symbol}: {e}")
                    all_results[symbol] = {}
        
        return all_results
    
    def get_options_chain(self, symbol, expiry, option_type='call', category='index'):
        """Retrieve downloaded options chain
        
        Args:
            symbol: Symbol name
            expiry: Expiry date (YYYY-MM-DD)
            option_type: 'call' or 'put'
            category: 'index' or 'stocks'
        
        Returns:
            DataFrame or None
        """
        file_path = self.base_dir / category / symbol / expiry
        
        if option_type.lower() == 'call':
            file_path = file_path / 'calls.parquet'
        else:
            file_path = file_path / 'puts.parquet'
        
        if not file_path.exists():
            logger.warning(f"[OPTIONS] Data not found: {symbol}/{expiry}/{option_type}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            logger.error(f"[OPTIONS] Error reading {symbol}/{expiry}/{option_type}: {e}")
            return None
    
    def verify_options_data(self, symbol, category='index'):
        """Verify options data integrity for a symbol
        
        Args:
            symbol: Symbol name
            category: 'index' or 'stocks'
        
        Returns:
            dict: Verification results
        """
        symbol_dir = self.base_dir / category / symbol
        
        if not symbol_dir.exists():
            return {'status': 'not_found'}
        
        results = {'status': 'ok', 'expiries': {}}
        
        for expiry_dir in symbol_dir.iterdir():
            if not expiry_dir.is_dir():
                continue
            
            expiry = expiry_dir.name
            results['expiries'][expiry] = {}
            
            # Check calls
            calls_file = expiry_dir / 'calls.parquet'
            if calls_file.exists():
                try:
                    df_calls = pd.read_parquet(calls_file)
                    results['expiries'][expiry]['calls'] = {
                        'rows': len(df_calls),
                        'columns': list(df_calls.columns)
                    }
                except Exception as e:
                    results['expiries'][expiry]['calls'] = {'error': str(e)}
            
            # Check puts
            puts_file = expiry_dir / 'puts.parquet'
            if puts_file.exists():
                try:
                    df_puts = pd.read_parquet(puts_file)
                    results['expiries'][expiry]['puts'] = {
                        'rows': len(df_puts),
                        'columns': list(df_puts.columns)
                    }
                except Exception as e:
                    results['expiries'][expiry]['puts'] = {'error': str(e)}
        
        return results
    
    def list_downloaded_options(self, category=None):
        """List all downloaded options
        
        Args:
            category: 'index', 'stocks', or None for all
        
        Returns:
            dict: Structure of downloaded data
        """
        result = {}
        
        if category:
            path = self.base_dir / category
            if path.exists():
                for symbol_dir in path.iterdir():
                    if symbol_dir.is_dir():
                        expiries = [d.name for d in symbol_dir.iterdir() if d.is_dir()]
                        result[symbol_dir.name] = expiries
        else:
            for cat_dir in self.base_dir.iterdir():
                if cat_dir.is_dir():
                    cat_result = {}
                    for symbol_dir in cat_dir.iterdir():
                        if symbol_dir.is_dir():
                            expiries = [d.name for d in symbol_dir.iterdir() if d.is_dir()]
                            cat_result[symbol_dir.name] = expiries
                    if cat_result:
                        result[cat_dir.name] = cat_result
        
        return result


# Convenience function
def download_all_options(index_symbols=['NIFTY50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
                         stock_symbols=None, max_expiries=3):
    """Download all options"""
    downloader = OptionsDownloader()
    
    results = {}
    results['index'] = downloader.download_index_options(index_symbols, max_expiries)
    
    if stock_symbols:
        results['stocks'] = downloader.download_stock_options(stock_symbols, max_expiries)
    
    return results
