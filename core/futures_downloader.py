# core/futures_downloader.py — Futures Data Downloader
# Downloads: Index, Stock, Currency, Commodity Futures
# Supports: Daily and intraday data
# Integration: Uses Kite API + NSE backfill strategy

import pandas as pd
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import BASE_DIR
from utils.logger import logger
import utils.helpers
from .data_manager import DataManager

# Futures instrument token mappings (will be fetched from Kite)
INDEX_FUTURES = {
    'NIFTY50': None,
    'BANKNIFTY': None,
    'FINNIFTY': None,
    'MIDCPNIFTY': None,
}

CURRENCY_FUTURES = {
    'USDINR': None,
    'EURINR': None,
}

COMMODITY_FUTURES = {
    'GOLD': None,
    'SILVER': None,
    'CRUDE': None,
}

STOCK_FUTURES_SAMPLE = [
    'TCS', 'INFY', 'RELIANCE', 'HDFC', 'ICICIBANK', 'SBIN', 'BAJAJFINSV',
    'MARUTI', 'WIPRO', 'HDFCBANK', 'LTIMINDTREE', 'HINDUNILVR', 'ITC',
    'BHARTIARTL', 'SUNPHARMA', 'KOTAKBANK', 'ASIANPAINT', 'AXISBANK'
]


class FuturesDownloader:
    """Downloads futures data for multiple categories"""
    
    def __init__(self):
        self.kite = utils.helpers.kite
        self.data_manager = DataManager()
        self.base_dir = BASE_DIR / 'data' / 'futures'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._instrument_cache = {}
    
    def _get_instruments(self):
        """Fetch all instruments and cache them"""
        if self._instrument_cache:
            return self._instrument_cache
        
        try:
            instruments = self.kite.instruments('NFO')
            self._instrument_cache = {instr['tradingsymbol']: instr for instr in instruments}
            logger.info(f"[FUTURES] Cached {len(self._instrument_cache)} NFO instruments")
            return self._instrument_cache
        except Exception as e:
            logger.error(f"[FUTURES] Error fetching instruments: {e}")
            return {}
    
    def _find_active_contract(self, symbol, contract_type='FUT'):
        """Find the active future contract for a symbol
        
        Args:
            symbol: 'NIFTY50', 'BANKNIFTY', 'TCS', etc.
            contract_type: 'FUT' for futures, 'CE'/'PE' for options
        
        Returns:
            tradingsymbol if found, None otherwise
        """
        instruments = self._get_instruments()
        
        # Look for contracts that match the symbol and type
        candidates = [
            instr['tradingsymbol']
            for instr in instruments.values()
            if instr.get('name') == symbol and contract_type in instr['tradingsymbol']
        ]
        
        if not candidates:
            logger.warning(f"[FUTURES] No {contract_type} contracts found for {symbol}")
            return None
        
        # Return the contract with nearest expiry (usually the first)
        candidates.sort()
        return candidates[0]
    
    def download_index_futures(self, symbols=['NIFTY50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
                               intervals=['day', '5minute', '15minute'],
                               lookback_days=365):
        """Download index futures data
        
        Args:
            symbols: List of index futures symbols
            intervals: Time intervals to download
            lookback_days: How many days of history to fetch
        
        Returns:
            dict: Download results {symbol: {interval: success}}
        """
        logger.info(f"[FUTURES] Downloading index futures: {symbols}")
        results = {}
        
        for symbol in symbols:
            results[symbol] = self._download_futures_symbol(symbol, 'index', intervals, lookback_days)
        
        return results
    
    def download_stock_futures(self, symbols=None, intervals=['day'],
                               lookback_days=365):
        """Download stock futures data
        
        Args:
            symbols: List of stock symbols (None = use sample)
            intervals: Time intervals to download
            lookback_days: How many days of history
        
        Returns:
            dict: Download results
        """
        if symbols is None:
            symbols = STOCK_FUTURES_SAMPLE
        
        logger.info(f"[FUTURES] Downloading stock futures: {len(symbols)} symbols")
        results = {}
        
        # Batch process
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures_map = {
                executor.submit(self._download_futures_symbol, symbol, 'stocks', 
                              intervals, lookback_days): symbol
                for symbol in symbols
            }
            
            for future in as_completed(futures_map):
                symbol = futures_map[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"[FUTURES] Error downloading {symbol}: {e}")
                    results[symbol] = {interval: False for interval in intervals}
        
        return results
    
    def download_currency_futures(self, symbols=['USDINR', 'EURINR'],
                                  intervals=['day'],
                                  lookback_days=365):
        """Download currency futures data
        
        Args:
            symbols: Currency pairs
            intervals: Time intervals
            lookback_days: Historical days
        
        Returns:
            dict: Download results
        """
        logger.info(f"[FUTURES] Downloading currency futures: {symbols}")
        results = {}
        
        for symbol in symbols:
            results[symbol] = self._download_futures_symbol(symbol, 'currency', 
                                                           intervals, lookback_days)
        
        return results
    
    def download_commodity_futures(self, symbols=['GOLD', 'SILVER', 'CRUDE'],
                                   intervals=['day'],
                                   lookback_days=365):
        """Download commodity futures data (MCX)
        
        Args:
            symbols: Commodity names
            intervals: Time intervals
            lookback_days: Historical days
        
        Returns:
            dict: Download results
        """
        logger.info(f"[FUTURES] Downloading commodity futures: {symbols}")
        results = {}
        
        for symbol in symbols:
            results[symbol] = self._download_futures_symbol(symbol, 'commodities',
                                                           intervals, lookback_days)
        
        return results
    
    def _download_futures_symbol(self, symbol, category, intervals, lookback_days):
        """Download a single futures symbol across multiple intervals
        
        Args:
            symbol: Trading symbol
            category: 'index', 'stocks', 'currency', 'commodities'
            intervals: Time intervals to download
            lookback_days: Historical period
        
        Returns:
            dict: {interval: success}
        """
        trading_symbol = self._find_active_contract(symbol)
        if not trading_symbol:
            logger.warning(f"[FUTURES] Skipping {symbol}: no active contract found")
            return {interval: False for interval in intervals}
        
        # Get instrument token
        instruments = self._get_instruments()
        instr = instruments.get(trading_symbol)
        if not instr:
            logger.warning(f"[FUTURES] No instrument token for {trading_symbol}")
            return {interval: False for interval in intervals}
        
        instrument_token = instr['instrument_token']
        results = {}
        
        # Create directory for this symbol
        symbol_dir = self.base_dir / category / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Download each interval
        for interval in intervals:
            try:
                # Calculate date range
                end_date = dt.datetime.now()
                start_date = end_date - dt.timedelta(days=lookback_days)
                
                logger.info(f"[FUTURES] {symbol}/{interval}: Downloading {start_date.date()} to {end_date.date()}")
                
                # Fetch data from Kite API
                data = self.kite.historical_data(
                    instrument_token,
                    interval,
                    start_date,
                    end_date
                )
                
                if not data:
                    logger.warning(f"[FUTURES] No data returned for {symbol}/{interval}")
                    results[interval] = False
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df['date'] = pd.to_datetime(df['date'])
                
                # Remove timezone
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                
                # Save to parquet
                file_path = symbol_dir / f"{interval}.parquet"
                df.to_parquet(file_path, index=False)
                
                logger.info(f"[FUTURES] ✓ {symbol}/{interval}: {len(df)} rows saved")
                results[interval] = True
                
            except Exception as e:
                logger.error(f"[FUTURES] Error downloading {symbol}/{interval}: {e}")
                results[interval] = False
            
            # Rate limiting
            time.sleep(0.5)
        
        return results
    
    def get_futures_data(self, symbol, category, interval='day'):
        """Retrieve downloaded futures data
        
        Args:
            symbol: Symbol name
            category: 'index', 'stocks', 'currency', 'commodities'
            interval: 'day', '5minute', '15minute', etc.
        
        Returns:
            DataFrame or None
        """
        file_path = self.base_dir / category / symbol / f"{interval}.parquet"
        
        if not file_path.exists():
            logger.warning(f"[FUTURES] Data not found: {symbol}/{interval}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            logger.error(f"[FUTURES] Error reading {symbol}/{interval}: {e}")
            return None
    
    def list_downloaded_futures(self, category=None):
        """List all downloaded futures symbols
        
        Args:
            category: Filter by category ('index', 'stocks', etc.) or None for all
        
        Returns:
            dict: {category: [symbols]}
        """
        result = {}
        
        if category:
            path = self.base_dir / category
            if path.exists():
                result[category] = [d.name for d in path.iterdir() if d.is_dir()]
        else:
            for cat_path in self.base_dir.iterdir():
                if cat_path.is_dir():
                    result[cat_path.name] = [d.name for d in cat_path.iterdir() if d.is_dir()]
        
        return result
    
    def verify_futures_data(self, symbol, category):
        """Verify data integrity for a futures symbol
        
        Args:
            symbol: Symbol name
            category: Category name
        
        Returns:
            dict: Verification results
        """
        symbol_dir = self.base_dir / category / symbol
        
        if not symbol_dir.exists():
            return {'status': 'not_found'}
        
        results = {'status': 'ok', 'intervals': {}}
        
        for interval_file in symbol_dir.glob("*.parquet"):
            interval_name = interval_file.stem
            try:
                df = pd.read_parquet(interval_file)
                results['intervals'][interval_name] = {
                    'rows': len(df),
                    'date_range': f"{df['date'].min()} to {df['date'].max()}",
                    'columns': list(df.columns),
                    'missing_values': df.isnull().sum().to_dict()
                }
            except Exception as e:
                results['intervals'][interval_name] = {'error': str(e)}
        
        return results


# Convenience functions
def download_all_futures(categories=['index', 'stocks', 'currency', 'commodities']):
    """Download all futures across all categories"""
    downloader = FuturesDownloader()
    all_results = {}
    
    for category in categories:
        if category == 'index':
            all_results['index'] = downloader.download_index_futures()
        elif category == 'stocks':
            all_results['stocks'] = downloader.download_stock_futures()
        elif category == 'currency':
            all_results['currency'] = downloader.download_currency_futures()
        elif category == 'commodities':
            all_results['commodities'] = downloader.download_commodity_futures()
    
    return all_results
