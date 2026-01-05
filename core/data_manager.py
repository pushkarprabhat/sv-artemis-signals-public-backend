# core/data_manager.py — Enhanced Data Management System
# Downloads: Nifty500, indices, currencies, commodities, metals
# Supports: 5-minute and daily intervals
# Features: Download history tracking, missing data detection, automatic aggregation

import pandas as pd
import datetime as dt
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import BASE_DIR, INDICES
from universe.symbols import load_universe
from utils.logger import logger
import utils.helpers
from utils.market_hours import NSE_HOLIDAYS_2024_2025

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Will be imported lazily to avoid circular imports
_universe_manager = None

def get_universe_manager():
    """Lazy import to avoid circular imports"""
    global _universe_manager
    if _universe_manager is None:
        from .universe_manager import get_universe_manager as get_um
        _universe_manager = get_um()
    return _universe_manager


class DataDownloadHistory:
    """Track download history for auditing and resuming"""
    
    def __init__(self, history_file="data/download_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load()
    
    def _load(self):
        """Load existing history"""
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text())
            except:
                return {'runs': []}
        return {'runs': []}
    
    def _save(self):
        """Save history"""
        self.history_file.write_text(json.dumps(self.history, indent=2, default=str))
    
    def add_run(self, run_info):
        """Add a new download run"""
        self.history['runs'].append({
            'timestamp': dt.datetime.now().isoformat(),
            'status': run_info.get('status'),
            'symbols_attempted': run_info.get('symbols_attempted', 0),
            'symbols_successful': run_info.get('symbols_successful', 0),
            'symbols_failed': run_info.get('symbols_failed', 0),
            'duration_seconds': run_info.get('duration_seconds'),
            'intervals': run_info.get('intervals', []),
            'notes': run_info.get('notes', '')
        })
        self._save()
    
    def get_last_run(self):
        """Get the last download run info"""
        if self.history['runs']:
            return self.history['runs'][-1]
        return None
    
    def get_recent_runs(self, limit=10):
        """Get recent download runs"""
        return self.history['runs'][-limit:]


class DataManager:
    """Unified data management for all instrument types"""

    @staticmethod
    def verify_and_fix_change_pct(df):
        """
        Verify 'change_pct' and 'net_change' columns are present and correct. If missing or incorrect, recompute.
        For Shivaansh & Krishaansh — this line pays their fees
        """
        if 'close' not in df.columns:
            return df
        needs_recalc = False
        # Check for both columns
        if 'change_pct' not in df.columns or 'net_change' not in df.columns:
            needs_recalc = True
        else:
            prev_close = df['close'].shift(1)
            mask = prev_close > 0
            expected_pct = (df['close'] - prev_close) / prev_close * 100
            expected_pct = expected_pct.fillna(0.0)
            expected_net = (df['close'] - prev_close).fillna(0.0)
            # Allow small float error
            if not (abs(df['change_pct'].fillna(0.0) - expected_pct) < 1e-6).all() or not (abs(df['net_change'].fillna(0.0) - expected_net) < 1e-6).all():
                needs_recalc = True
        if needs_recalc:
            import warnings
            warnings.warn("change_pct or net_change column missing or incorrect, auto-recalculating.")
            df = DataManager.compute_change_pct(df)
        return df

    @staticmethod
    def compute_change_pct(df):
        """
        Compute and add 'change_pct' and 'net_change' columns:
        - change_pct: (current close - previous close)/previous close * 100
        - net_change: (current close - previous close)
        Handles missing previous close gracefully (sets to 0 for first row or missing prev).
        For Shivaansh & Krishaansh — this line pays their fees
        """
        if 'close' not in df.columns:
            return df
        df = df.copy()
        df['prev_close'] = df['close'].shift(1)
        mask = df['prev_close'] > 0
        df['change_pct'] = 0.0
        df['net_change'] = 0.0
        df.loc[mask, 'change_pct'] = (df.loc[mask, 'close'] - df.loc[mask, 'prev_close']) / df.loc[mask, 'prev_close'] * 100
        df.loc[mask, 'net_change'] = (df.loc[mask, 'close'] - df.loc[mask, 'prev_close'])
        df['change_pct'] = df['change_pct'].fillna(0.0)
        df['net_change'] = df['net_change'].fillna(0.0)
        df.drop(columns=['prev_close'], inplace=True)
        return df

    def __init__(self):
        self.history = DataDownloadHistory()
        self.kite = utils.helpers.kite
        self._download_universe_cache = None  # Cache to avoid reloading
    
    def rebuild_download_universe(self):
        """Rebuild the download universe from data sources
        
        Returns:
            dict: Status of rebuild operation
        """
        try:
            # Simply reload the universe
            universe = self.get_download_universe()
            total_symbols = sum(len(v) for v in universe.values())
            return {'status': 'success', 'total_symbols': total_symbols}
        except Exception as e:
            logger.error(f"Failed to rebuild universe: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_download_universe(self):
        """Get all symbols to download as a flat list
        
        Uses:
        1. UniverseManager for complete Kite instruments list (persistent + cached)
        2. Falls back to app_kite_universe.csv if needed
        3. CACHED - loads only once per DataManager instance
        
        Returns:
            List[str]: Flat list of trading symbols to download
        """
        # Return cached version if available
        if self._download_universe_cache is not None:
            return self._download_universe_cache
        
        symbols = []
        
        # Try to use persistent universe manager first
        try:
            um = get_universe_manager()
            df_complete = um.get_universe()
            
            if df_complete is not None and len(df_complete) > 0:
                # Extract stocks from complete universe
                try:
                    # Filter for stocks if column exists
                    if 'InNifty500' in df_complete.columns:
                        stocks = df_complete[df_complete['InNifty500'] == 'Y']['Symbol'].tolist()
                    else:
                        # Fallback: take NSE equity instruments
                        stocks = df_complete[
                            (df_complete['Exchange'] == 'NSE') & 
                            (df_complete['InstrumentType'] == 'EQ')
                        ]['Symbol'].tolist()
                    
                    symbols.extend(stocks)
                    logger.info(f"[OK] Loaded {len(symbols)} stocks from persistent universe")
                except Exception as e:
                    logger.warning(f"Could not extract stocks from persistent universe: {e}")
                    # Fall through to CSV fallback
            else:
                logger.info("Persistent universe empty, falling back to CSV")
        except Exception as e:
            logger.warning(f"Could not load persistent universe: {e}")
        
        # Fallback: load from CSV if needed
        if len(symbols) == 0:
            try:
                df_universe = load_universe()
                stocks = df_universe[
                    (df_universe['instrument_type'].isin(['EQ', 'FUT'])) &
                    (df_universe['exchange'] == 'NSE')
                ]['tradingsymbol'].tolist()
                symbols.extend(stocks)
                logger.info(f"[OK] Loaded {len(symbols)} stocks from CSV fallback (EQ+FUT)")
            except Exception as e:
                logger.error(f"Failed to load stocks from CSV: {e}")
        
        # Add major indices
        indices = ['NIFTY50', 'NIFTY100', 'NIFTY500', 'NIFTYBANK', 'NIFTY_IT', 'NIFTY_PHARMA', 'INDIA_VIX']
        symbols.extend(indices)
        logger.info(f"[OK] Added {len(indices)} major indices")
        
        # Add currency pairs
        currency_pairs = ['EURINR', 'GBPINR', 'JPYINR', 'USDINR']
        symbols.extend(currency_pairs)
        logger.info(f"[OK] Added {len(currency_pairs)} currency pairs")
        
        # Add commodities and metals
        commodities = ['GOLDPETAL', 'SILVERPETAL', 'CRUDEOIL', 'NATURALGAS', 'COPPER', 'ALUMINUM', 'ZINC', 'NICKEL']
        symbols.extend(commodities)
        logger.info(f"[OK] Added {len(commodities)} commodities")
        
        logger.info(f"[DOWNLOAD-UNIVERSE] Total symbols to download: {len(symbols)}")
        
        # Cache the result to avoid reloading
        self._download_universe_cache = symbols
        return symbols
    
    def download_price_data_batch(self, symbols, exchange='NSE', intervals=None, days_back=365, parallel_workers=4):
        """Batch download price data for multiple symbols
        
        Args:
            symbols: List of trading symbols
            exchange: Exchange (NSE/MCX/etc)
            intervals: List of intervals (e.g., ['5minute', 'day'])
            days_back: Days to download (default 365 = 1 year)
            parallel_workers: Number of parallel workers per batch
        
        Returns:
            dict: {
                'total_symbols': total count,
                'successful': count of successful downloads,
                'failed': count of failed downloads,
                'by_interval': {interval: {successful, failed}},
                'duration_seconds': elapsed time
            }
        """
        if intervals is None:
            intervals = ['5minute', 'day']
        
        start_time = dt.datetime.now()
        BATCH_SIZE = 250
        
        # Organize symbols into batches
        batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
        
        total = len(symbols)
        successful = 0
        failed = 0
        by_interval = {iv: {'successful': 0, 'failed': 0} for iv in intervals}
        
        logger.info(f"[BATCH-DOWNLOAD] Starting download of {total} symbols in {len(batches)} batch(es) of {BATCH_SIZE}")
        
        # Process each batch
        for batch_num, batch_symbols in enumerate(batches, 1):
            logger.info(f"[BATCH {batch_num}/{len(batches)}] Processing {len(batch_symbols)} symbols...")
            
            # Download batch using parallel workers
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(self.download_price_data, sym, exchange=exchange, intervals=intervals, days_back=days_back): sym 
                    for sym in batch_symbols
                }
                
                processed = 0
                for future in as_completed(futures):
                    sym = futures[future]
                    processed += 1
                    try:
                        results = future.result()
                        # results is a dict like {'day': True/False}
                        # Count per interval directly
                        for iv in intervals:
                            if results.get(iv, False):
                                by_interval[iv]['successful'] += 1
                            else:
                                by_interval[iv]['failed'] += 1
                        
                        # Count overall based on any interval success
                        if any(results.values()):
                            successful += 1
                        else:
                            failed += 1
                            logger.debug(f"      [NO-DATA] [{processed}/{len(batch_symbols)}] {sym}: No data returned")
                    except Exception as e:
                        failed += 1
                        logger.error(f"[ERROR] [{processed}/{len(batch_symbols)}] {sym}: {str(e)}")
                        for iv in intervals:
                            by_interval[iv]['failed'] += 1
            
            # Rate limiting between batches to avoid API throttling
            if batch_num < len(batches):
                time.sleep(1)
        
        duration = (dt.datetime.now() - start_time).total_seconds()
        
        summary = {
            'total_symbols': total,
            'successful': successful,
            'failed': failed,
            'by_interval': by_interval,
            'duration_seconds': duration,
            'batches': len(batches)
        }
        
        logger.info(f"[BATCH-DOWNLOAD] Complete: {successful}/{total} successful, {failed}/{total} failed in {duration:.1f}s")
        
        return summary
    
    def download_price_data(self, symbol, exchange='NSE', intervals=None, days_back=365):
        """Download price data for one symbol
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE/MCX/etc)
            intervals: List of intervals (e.g., ['5minute', 'day', 'week', 'month', 'quarter', 'year'])
            days_back: Days to download (default 365 = 1 year)
        
        Returns:
            dict: Download results {'interval': success_bool}
            
        STRATEGY: If daily data is unavailable from Kite, skip 3-min/5-min intraday downloads
                 to avoid downloading intraday candles without foundation (daily baseline)
        """
        if intervals is None:
            intervals = ['5minute', 'day', 'week', 'month', 'quarter', 'year']
        
        results = {}

        # Determine asset details for logging
        segment = "EQUITY" # Default
        asset_type = "STOCK" # Default
        if exchange == 'NFO':
            segment = "DERIVATIVE"
            asset_type = "FUT/OPT"
        elif exchange == 'MCX':
            segment = "COMMODITY"
            asset_type = "FUT"
        elif exchange == 'CDS':
            segment = "CURRENCY"
            asset_type = "FUT"
            
        log_identifier = f"[{exchange}/{segment}/{symbol} - {asset_type}]"
        logger.info(f"{log_identifier} Downloading price data...")
        
        # Get token
        try:
            kite = utils.helpers.kite
            if kite is None:
                logger.error(f"KiteConnect not initialized")
                return {iv: False for iv in intervals}
            
            quote = kite.ltp(f"{exchange}:{symbol}")
            
            # Safe token extraction with better error handling
            if not quote or len(quote) == 0:
                logger.error(f"No quote data for {symbol} ({exchange}) - may be an index or illiquid security")
                return {iv: False for iv in intervals}
            
            quote_data = list(quote.values())[0]
            if "instrument_token" not in quote_data:
                logger.error(f"No instrument_token in quote for {symbol} ({exchange})")
                return {iv: False for iv in intervals}
            
            token = quote_data["instrument_token"]
        except IndexError as e:
            logger.error(f"Quote parsing failed for {symbol} ({exchange}): {e} - symbol may not be tradeable")
            return {iv: False for iv in intervals}
        except Exception as e:
            logger.error(f"Token fetch failed for {symbol}: {e}")
            return {iv: False for iv in intervals}
        
        # Download for each interval
        to_date = dt.datetime.now().date()
        from_date = to_date - dt.timedelta(days=days_back)
        
        logger.debug(f"[DOWNLOAD] {symbol}: Requested intervals: {intervals}, Date range: {from_date} to {to_date}")
        
        for interval in intervals:
            try:
                folder = BASE_DIR / interval
                folder.mkdir(parents=True, exist_ok=True)
                file_path = folder / f"{symbol}.parquet"
                
                # Check if file exists and is recent (skip if up-to-date)
                if file_path.exists():
                    try:
                        existing_df = pd.read_parquet(file_path)
                        if len(existing_df) > 0:
                            last_date = pd.to_datetime(existing_df['date'].max()).date()
                            if last_date >= to_date:
                                logger.info(f"[SKIP] {symbol} {interval}: Already up-to-date")
                                results[interval] = True
                                continue
                            # Incremental: start from last date + 1
                            from_date = last_date + dt.timedelta(days=1)
                    except:
                        pass
                
        # Download data using batching to avoid "interval exceeds max limit" errors
                from core.downloader import download_with_batching, BATCH_SIZES
                batch_days = BATCH_SIZES.get(interval, 100)  # Get batch size from config
                
                # STRATEGY: Skip intraday downloads (3min, 5min) if daily data is unavailable
                intraday_intervals = ['3minute', '5minute']
                if interval in intraday_intervals:
                    # Check if daily data is available
                    daily_file = BASE_DIR / 'day' / f"{symbol}.parquet"
                    if not daily_file.exists():
                        logger.warning(f"[SKIP] {symbol} {interval}: Daily data unavailable (foundation required for intraday)")
                        results[interval] = False
                        continue
                
                data = download_with_batching(
                    kite=kite,
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval,
                    batch_days=batch_days
                )
                
                if not data:
                    logger.warning(f"[NO_DATA] {symbol} {interval}: No data returned from Kite API")
                    results[interval] = False
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(data)
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df['date'] = pd.to_datetime(df['date'])
                
                # Remove timezone
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                
                # Merge with existing if present
                if file_path.exists():
                    try:
                        old_df = pd.read_parquet(file_path)
                        if old_df['date'].dt.tz is not None:
                            old_df['date'] = old_df['date'].dt.tz_localize(None)
                        df = pd.concat([old_df, df]).drop_duplicates('date').sort_values('date').reset_index(drop=True)
                    except:
                        pass
                
                df = df.sort_values('date').reset_index(drop=True)
                df.to_parquet(file_path, index=False, engine='pyarrow')
                
                logger.info(f"[OK] {symbol} {interval}: {len(df)} rows")
                results[interval] = True
                
                # Auto-aggregate 5-minute data to 15/30/60 min intervals immediately after 5min download
                if interval == '5minute':
                    try:
                        from core.data_aggregator import aggregate_all_intervals
                        if aggregate_all_intervals(symbol):
                            logger.debug(f"[AUTO-AGG] {symbol}: Aggregated 5-min → 15/30/60 min after download")
                    except Exception as e:
                        logger.debug(f"[AUTO-AGG-SKIP] {symbol}: {e}")
                
                time.sleep(0.3)  # Rate limit
                
            except Exception as e:
                logger.error(f"[FAIL] {symbol} {interval}: {e}")
                results[interval] = False
        
        return results
    
    def download_all_data(self, segments=None, intervals=None, days_back=365, parallel=True, max_workers=4, check_updates=True):
        """Download data for all symbols across all segments using optimized 250-symbol batching
        
        Args:
            segments: List of segments to download ('stocks', 'indices', 'currencies', 'commodities', 'metals')
            intervals: List of intervals to download (['5minute', 'day'])
            days_back: Days of history to download (default 365)
            parallel: Use parallel downloads (always uses batching for efficiency)
            max_workers: Number of parallel workers per batch
            check_updates: Check for instrument updates before downloading
        
        Returns:
            dict: Summary of download results
            
        STRATEGY: Always ensures 'day' (daily) interval is downloaded first.
                 Other intraday intervals (3min, 5min) will only download if daily data exists.
        """
        # Check for instrument updates before downloading
        if check_updates:
            self._check_and_update_instruments()
        
        if segments is None:
            segments = ['stocks', 'indices', 'currencies', 'commodities', 'metals']
        
        if intervals is None:
            intervals = ['5minute', 'day']
        
        # STRATEGY: Ensure 'day' is always first
        intervals = ['day'] + [iv for iv in intervals if iv != 'day']
        
        logger.info(f"[DOWNLOAD-STRATEGY] Download order: {intervals} (daily first for foundation)")
        
        start_time = dt.datetime.now()
        universe = self.get_download_universe()
        
        # Flatten universe for all requested segments
        all_symbols = []
        for segment in segments:
            if segment in universe:
                all_symbols.extend(universe[segment].keys())
        
        total = len(all_symbols)
        
        if HAS_STREAMLIT:
            st.write(f"Downloading {total} symbols in {len(all_symbols)//250 + 1} batches of 250...")
            st.info(f"Download strategy: Daily data FIRST (foundation), then intraday (if daily available)")
            progress_bar = st.progress(0)
            status_text = st.empty()
        else:
            logger.info(f"Downloading {total} symbols using optimized batch download (250 per batch)...")
            logger.info(f"Download strategy: Daily data FIRST (foundation), then intraday (only if daily available)")
        
        # Use optimized batch download
        results = self.download_price_data_batch(
            symbols=all_symbols,
            exchange='NSE',
            intervals=intervals,
            days_back=days_back,
            parallel_workers=max_workers
        )
        
        # Trigger automatic aggregation after download
        try:
            self._auto_aggregate_data()
        except Exception as e:
            logger.warning(f"Auto-aggregation failed: {e}")
        
        # Create comprehensive summary
        duration = (dt.datetime.now() - start_time).total_seconds()
        summary = {
            'status': 'complete',
            'symbols_attempted': total,
            'symbols_successful': results['successful'],
            'symbols_failed': results['failed'],
            'duration_seconds': duration,
            'intervals': intervals,
            'segments': segments,
            'batches_used': results['batches'],
            'by_interval': results['by_interval']
        }
        
        self.history.add_run(summary)
        
        if HAS_STREAMLIT:
            st.success(f"Download complete! Success: {results['successful']}/{total} | Failed: {results['failed']}")
            progress_bar.progress(1.0)
        else:
            logger.info(f"Download complete! Success: {results['successful']}/{total} | Failed: {results['failed']} | Duration: {duration:.1f}s")
        
        return summary
    
    def detect_missing_data(self, interval='day', symbols=None):
        """Detect which symbols are missing data
        
        Args:
            interval: Interval to check (e.g., 'day', '5minute')
            symbols: List of symbols to check (default: all symbols with data files in this interval)
        
        Returns:
            dict: {
                'missing_symbols': [...], 
                'complete_symbols': [...], 
                'missing_details': {symbol: {date_from, date_to, count}},
                'stats': {...}
            }
        """
        folder = BASE_DIR / interval
        
        # If symbols not provided, get all symbols that have data files in this interval
        if symbols is None:
            if not folder.exists():
                return {
                    'missing_symbols': [],
                    'complete_symbols': [],
                    'missing_details': {},
                    'stats': {'total': 0, 'complete': 0, 'missing': 0, 'coverage_pct': 0}
                }
            # Count distinct symbols from actual parquet files
            symbols = [f.stem for f in folder.glob('*.parquet')]
        
        all_symbols = symbols if symbols else []
        
        missing = []
        complete = []
        missing_details = {}  # Track date ranges for missing data
        
        today = dt.datetime.now().date()
        year_ago = today - dt.timedelta(days=365)
        
        for symbol in all_symbols:
            file_path = folder / f"{symbol}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    if len(df) > 10:  # At least 10 candles
                        # Check if data is complete (reaches back 1 year)
                        oldest_date = pd.to_datetime(df['date'].min()).date()
                        newest_date = pd.to_datetime(df['date'].max()).date()
                        
                        # If data doesn't go back 1 year, mark as incomplete
                        if oldest_date > year_ago:
                            missing.append(symbol)
                            missing_details[symbol] = {
                                'from_date': oldest_date.isoformat(),
                                'to_date': newest_date.isoformat(),
                                'rows': len(df),
                                'reason': 'insufficient_history'
                            }
                        else:
                            complete.append(symbol)
                    else:
                        missing.append(symbol)
                        missing_details[symbol] = {
                            'rows': len(df),
                            'reason': 'too_few_candles'
                        }
                except Exception as e:
                    missing.append(symbol)
                    missing_details[symbol] = {'reason': f'parse_error: {str(e)}'}
            else:
                missing.append(symbol)
                missing_details[symbol] = {'reason': 'no_data_file'}
        
        return {
            'missing_symbols': missing,
            'complete_symbols': complete,
            'missing_details': missing_details,  # NEW: Detailed per-symbol info
            'stats': {
                'total': len(all_symbols),
                'complete': len(complete),
                'missing': len(missing),
                'coverage_pct': 100 * len(complete) / len(all_symbols) if all_symbols else 0
            }
        }
    
    def count_distinct_symbols_with_data(self, interval='day'):
        """Count distinct symbols that actually have data files (for accurate metrics)
        
        Args:
            interval: Interval to check (e.g., 'day', '5minute')
        
        Returns:
            tuple: (total_symbols_with_files, symbols_list)
        """
        folder = BASE_DIR / interval
        if not folder.exists():
            return 0, []
        
        # Count actual parquet files in the folder
        symbols_with_data = [f.stem for f in folder.glob('*.parquet')]
        return len(symbols_with_data), symbols_with_data
    
    def calculate_coverage_percentage(self, interval='day'):
        """Calculate data coverage percentage based on expected vs actual data points
        
        Args:
            interval: Interval to check (e.g., 'day', '5minute')
        
        Returns:
            dict: {
                'total_symbols': total unique symbols,
                'expected_points_per_symbol': expected bars for 1 year,
                'coverage_details': {symbol: {'actual': X, 'expected': Y, 'coverage_pct': Z}},
                'overall_coverage_pct': overall coverage percentage,
                'summary': {
                    'total_expected': sum of all expected points,
                    'total_actual': sum of all actual points,
                    'coverage_pct': overall percentage
                }
            }
        """
        folder = BASE_DIR / interval
        
        # Get all symbols in universe (every possible instrument)
        all_symbols = self.get_download_universe()
        
        # Calculate expected data points for 1 year based on interval
        expected_points_per_symbol = self._calculate_expected_bars(interval)
        
        coverage_details = {}
        total_expected = 0
        total_actual = 0
        
        for symbol in all_symbols:
            file_path = folder / f"{symbol}.parquet"
            expected = expected_points_per_symbol
            actual = 0
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    actual = len(df)
                except Exception as e:
                    logger.debug(f"Error reading {symbol} {interval}: {e}")
            
            coverage_pct = (actual / expected * 100) if expected > 0 else 0
            
            coverage_details[symbol] = {
                'actual': actual,
                'expected': expected,
                'coverage_pct': coverage_pct
            }
            
            total_expected += expected
            total_actual += actual
        
        overall_coverage = (total_actual / total_expected * 100) if total_expected > 0 else 0
        
        return {
            'total_symbols': len(all_symbols),
            'expected_points_per_symbol': expected_points_per_symbol,
            'coverage_details': coverage_details,
            'summary': {
                'total_expected': total_expected,
                'total_actual': total_actual,
                'coverage_pct': overall_coverage
            }
        }

    def get_detailed_data_audit(self, interval='day', symbols=None, limit=None):
        """
        Audit data files to get explicit start/end dates and counts.
        Returns a DataFrame with detailed stats.
        
        Args:
            interval (str): Timeframe interval
            symbols (list, optional): Filter by specific symbols
            limit (int, optional): Limit number of files to process
            
        Returns:
            pd.DataFrame: Audit results
        """
        try:
            folder = BASE_DIR / interval
            if not folder.exists():
                return pd.DataFrame()

            audit_data = []
            
            # Get all parquet files
            files = list(folder.glob('*.parquet'))
            
            # Filter by symbols if provided
            if symbols:
                files = [f for f in files if f.stem in symbols]
                
            # Apply limit
            if limit:
                files = files[:limit]
            
            today = dt.datetime.now().date()
            one_year_ago = dt.datetime.now() - dt.timedelta(days=365)
            
            for file_path in files:
                try:
                    symbol = file_path.stem
                    # Read minimal columns
                    df = pd.read_parquet(file_path, columns=['date'])
                    
                    if not df.empty:
                        start_date = df['date'].min()
                        end_date = df['date'].max()
                        count = len(df)
                        
                        # Convert to pydatetime if needed
                        if hasattr(start_date, 'to_pydatetime'):
                            start_date = start_date.to_pydatetime()
                        if hasattr(end_date, 'to_pydatetime'):
                            end_date = end_date.to_pydatetime()
                            
                        # Check 1 year health (approximate)
                        has_history = start_date <= one_year_ago
                        
                        # Check freshness (today/yesterday/last-trading-day logic simplified)
                        last_date = end_date.date() if hasattr(end_date, 'date') else end_date
                        days_lag = (today - last_date).days
                        is_fresh = days_lag <= 3  # Allow weekend gap
                        
                        audit_data.append({
                            'Symbol': symbol,
                            'Interval': interval,
                            'Start Date': start_date,
                            'End Date': end_date,
                            'Rows': count,
                            'Has 1Yr+': has_history,
                            'Is Fresh': is_fresh,
                            'Days Lag': days_lag
                        })
                except Exception as e:
                    logger.warning(f"Error auditing {file_path.name}: {e}")
                    
            if not audit_data:
                return pd.DataFrame()
                
            return pd.DataFrame(audit_data).sort_values('Symbol')
            
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return pd.DataFrame()
    
    def _calculate_expected_bars(self, interval):
        """Calculate expected number of bars for 1 year based on interval
        
        Trading days per year: Calculated from NSE_HOLIDAYS_2024_2025
        - Total days: 365
        - Minus weekends: 104 (52 weeks x 2)
        - Minus NSE holidays: ~35-40
        - Result: ~220-230 trading days per year
        
        Market hours: 9:15 AM to 3:30 PM (6.25 hours = 375 minutes)
        
        Args:
            interval: Timeframe (e.g., '5minute', 'day', 'week', 'month')
        
        Returns:
            Expected number of bars in 1 year
        """
        trading_days_per_year = self._get_trading_days_in_year()
        
        # Map intervals to bars per trading day
        bars_per_day_map = {
            '5minute': 375 / 5,      # 6.25 hours * 60 / 5 = 75 bars
            '15minute': 375 / 15,    # 25 bars
            '30minute': 375 / 30,    # 12.5 ≈ 12 bars
            '60minute': 375 / 60,    # 6.25 ≈ 6 bars
            '120minute': 375 / 120,  # 3.125 ≈ 3 bars
            '150minute': 375 / 150,  # 2.5 ≈ 2 bars
            '180minute': 375 / 180,  # 2.08 ≈ 2 bars
            '240minute': 375 / 240,  # 1.56 ≈ 1 bar
            'day': 1,                # 1 bar per day
            'week': 1/5,             # 1 bar per 5 trading days
            'month': 1/21,           # 1 bar per ~21 trading days
            'quarter': 1/63,         # 1 bar per ~63 trading days
            'year': 1/252,           # 1 bar per year
        }
        
        bars_per_day = bars_per_day_map.get(interval, 1)
        expected_bars = int(trading_days_per_year * bars_per_day)
        
        return max(1, expected_bars)  # At least 1 bar
    
    def _get_trading_days_in_year(self, year=None):
        """Calculate actual trading days in a year using NSE holiday list
        
        Args:
            year: Year to calculate for (default: current year)
        
        Returns:
            Number of trading days in the year
        """
        if year is None:
            year = dt.date.today().year
            
        # Get all holidays for the year
        year_holidays = [h.date() for h in NSE_HOLIDAYS_2024_2025 if h.year == year]
        
        # Count trading days: exclude weekends and holidays
        trading_days = 0
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            # Check if weekday (Monday=0 to Friday=4)
            if current_date.weekday() < 5:  # Not Saturday (5) or Sunday (6)
                # Check if not a holiday
                if current_date not in year_holidays:
                    trading_days += 1
            current_date += dt.timedelta(days=1)
        
        return trading_days
    
    def _get_trading_days_year_to_date(self):
        """Calculate trading days from start of current year to today
        
        Useful for calculating expected data for partial year (Jan 1 to today)
        
        Returns:
            Number of trading days from Jan 1 to today
        """
        today = dt.date.today()
        year = today.year
        
        # Get all holidays for the year
        year_holidays = [h.date() for h in NSE_HOLIDAYS_2024_2025 if h.year == year]
        
        # Count trading days: exclude weekends and holidays
        trading_days = 0
        start_date = dt.date(year, 1, 1)
        end_date = today
        
        current_date = start_date
        while current_date <= end_date:
            # Check if weekday (Monday=0 to Friday=4)
            if current_date.weekday() < 5:  # Not Saturday (5) or Sunday (6)
                # Check if not a holiday
                if current_date not in year_holidays:
                    trading_days += 1
            current_date += dt.timedelta(days=1)
        
        return trading_days
    
    def get_download_logs(self, log_level='WARNING', limit=100):
        """Retrieve download-related error and warning logs
        
        Args:
            log_level: 'WARNING', 'ERROR', or 'ALL' for both
            limit: Maximum number of log entries to return
        
        Returns:
            list: Log entries sorted by timestamp (newest first)
        """
        log_file = BASE_DIR.parent / 'logs' / 'bot.log'
        
        if not log_file.exists():
            return []
        
        entries = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Parse log lines: "2025-12-11 03:40:31 WARNING  [svpair] message"
                    if not line.strip():
                        continue
                    
                    # Check if line contains download-related keywords
                    download_keywords = ['download', 'fetch', 'fail', 'error', 'missing', 'incomplete', 'data', 'interval', 'symbol', 'parquet']
                    if not any(keyword.lower() in line.lower() for keyword in download_keywords):
                        continue
                    
                    # Check log level
                    if log_level == 'WARNING' and 'WARNING' not in line and 'ERROR' not in line:
                        continue
                    elif log_level == 'ERROR' and 'ERROR' not in line:
                        continue
                    
                    entries.append(line.strip())
        except Exception as e:
            logger.debug(f"Error reading log file: {e}")
        
        # Return newest entries first
        return entries[-limit:][::-1] if entries else []
    
    def validate_candles(self, interval='day', symbols=None):
        """Validate candle data for consistency and completeness
        
        Args:
            interval: Data interval to validate
            symbols: Symbols to validate (default: all)
        
        Returns:
            dict: Validation results with counts and issues
        """
        universe = self.get_download_universe()
        all_symbols = list(set(
            list(universe['stocks'].keys()) +
            list(universe['indices'].keys()) +
            list(universe['currencies'].keys())
        ))
        
        if symbols:
            all_symbols = [s for s in all_symbols if s in symbols]
        
        folder = BASE_DIR / interval
        valid_count = 0
        duplicate_count = 0
        missing_values_count = 0
        issues = {}
        
        for symbol in all_symbols:
            file_path = folder / f"{symbol}.parquet"
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_parquet(file_path)
                
                # Check for duplicates
                dups = df[df.duplicated(subset=['date'], keep=False)]
                if len(dups) > 0:
                    duplicate_count += len(dups)
                    if symbol not in issues:
                        issues[symbol] = []
                    issues[symbol].append(f"Found {len(dups)} duplicate candles")
                
                # Check for missing values
                missing = df.isnull().sum().sum()
                if missing > 0:
                    missing_values_count += missing
                    if symbol not in issues:
                        issues[symbol] = []
                    issues[symbol].append(f"Found {missing} missing values")
                
                valid_count += len(df)
            except Exception as e:
                if symbol not in issues:
                    issues[symbol] = []
                issues[symbol].append(f"Error reading file: {str(e)}")
        
        return {
            'valid_count': valid_count,
            'duplicate_count': duplicate_count,
            'missing_values_count': missing_values_count,
            'issues_count': len(issues),
            'issues': issues
        }
    
    def validate_data_integrity(self, interval='day', symbols=None):
        """Check overall data integrity and completeness
        
        Args:
            interval: Data interval to validate
            symbols: Symbols to check (default: all)
        
        Returns:
            dict: Integrity check results with health score
        """
        results = self.detect_missing_data(interval, symbols)
        
        by_segment = {}
        universe = self.get_download_universe()
        
        for segment, syms in universe.items():
            segment_total = len(syms)
            segment_complete = len([s for s in syms.keys() if s in results['complete_symbols']])
            if segment_total > 0:
                by_segment[segment] = 100 * segment_complete / segment_total
        
        stats = results['stats']
        health_score = stats['coverage_pct']
        
        return {
            'total_symbols': stats['total'],
            'complete_symbols': stats['complete'],
            'incomplete_symbols': stats['missing'],
            'health_score': health_score,
            'missing_symbols': results['missing_symbols'],
            'complete_symbols': results['complete_symbols'],
            'by_segment': by_segment
        }
    
    def get_live_aggregation(self, symbol, from_interval, to_interval):
        """Convert candles between different timeframes on-the-fly
        
        Args:
            symbol: Trading symbol
            from_interval: Source interval (e.g., 'day')
            to_interval: Target interval (e.g., 'week')
        
        Returns:
            DataFrame: Aggregated candles
        """
        folder = BASE_DIR / from_interval
        file_path = folder / f"{symbol}.parquet"
        
        if not file_path.exists():
            logger.warning(f"Data not found for {symbol} {from_interval}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Map interval names
            interval_map = {
                '15min': '15T', '30min': '30T', '60min': '60T',
                'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q', 'year': 'Y'
            }
            
            target_freq = interval_map.get(to_interval, to_interval)
            
            df.set_index('date', inplace=True)
            
            # Aggregate OHLCV
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            df_agg = df.resample(target_freq).agg(agg_dict).dropna()
            df_agg = df_agg.reset_index()
            
            return df_agg
        
        except Exception as e:
            logger.error(f"Aggregation failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def check_aggregation_completeness(self, interval='day', symbols=None):
        """Check if weekly/monthly/yearly data exists for symbols
        
        Args:
            interval: Base interval to check aggregations for
            symbols: Symbols to check
        
        Returns:
            dict: Aggregation completeness status
        """
        universe = self.get_download_universe()
        all_symbols = list(set(
            list(universe['stocks'].keys()) +
            list(universe['indices'].keys()) +
            list(universe['currencies'].keys()) +
            list(universe['commodities'].keys()) +
            list(universe['metals'].keys())
        ))
        
        if symbols:
            all_symbols = [s for s in all_symbols if s in symbols]
        
        folder = BASE_DIR / interval
        missing = []
        complete = []
        
        for symbol in all_symbols:
            file_path = folder / f"{symbol}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    if len(df) > 10:  # At least 10 candles
                        complete.append(symbol)
                    else:
                        missing.append(symbol)
                except:
                    missing.append(symbol)
            else:
                missing.append(symbol)
        
        return {
            'missing_symbols': missing,
            'complete_symbols': complete,
            'stats': {
                'total': len(all_symbols),
                'complete': len(complete),
                'missing': len(missing),
                'coverage_pct': 100 * len(complete) / len(all_symbols) if all_symbols else 0
            }
        }
    
    def sync_data(self, interval='day', schedule='manual', symbols=None, parallel=True):
        """Automatically sync missing or stale data
        
        Args:
            interval: Interval to sync
            schedule: Sync schedule ('manual', 'hourly', 'daily', 'weekly')
            symbols: Symbols to sync (default: all)
            parallel: Use parallel processing
        
        Returns:
            dict: Sync results
        """
        # Check for missing data
        missing_results = self.detect_missing_data(interval, symbols)
        missing_symbols = missing_results['missing_symbols']
        
        if not missing_symbols:
            # Even if price data is synced, we must refresh Metadata/Master/Option Chains for completeness
            self._refresh_auxiliary_data()
            return {'status': 'uptodate', 'symbols_synced': 0}
        
        # 1. Refresh Auxiliary Data (Masters, Indices, Option Chains)
        self._refresh_auxiliary_data()

        # 2. Download Missing Price Data
        start_time = dt.datetime.now()
        results = self.download_all_data(
            segments=None,
            intervals=[interval],
            parallel=parallel,
            max_workers=4,
            symbols_subset=missing_symbols # Only download what is missing
        )
        
        return {
            'status': 'synced',
            'symbols_synced': len(missing_symbols),
            'duration_seconds': (dt.datetime.now() - start_time).total_seconds()
        }

    def _refresh_auxiliary_data(self):
        """Helper to refresh non-price datasets for total completeness"""
        try:
            logger.info("[SYNC] Refreshing Instrument Masters & Indices...")
            # these lazy imports avoid circular deps at top level
            from core.kite_instruments_manager import get_kite_instruments_manager
            from core.nse_indices_manager import get_nse_indices_manager
            
            # 1. Refresh Instrument Masters (All segments)
            kim = get_kite_instruments_manager()
            kim.refresh_master_instruments() 
            
            # 2. Refresh Index Constituents (NIFTY50, etc.)
            nim = get_nse_indices_manager()
            nim.refresh_all_indices()
            
            logger.info("[SYNC] Auxiliary data refresh complete.")
        except Exception as e:
            logger.error(f"[SYNC] Auxiliary data refresh failed: {e}")
    
    def get_download_history(self):
        """Get complete download history
        
        Returns:
            dict: Download history indexed by symbol
        """
        try:
            # Try to load from stored history
            history_data = self.history.history.get('runs', [])
            
            # Convert to symbol-indexed format
            symbol_history = {}
            for run in history_data:
                if run.get('symbols_successful', 0) > 0:
                    symbol_history[f"run_{run['timestamp']}"] = {
                        'last_download': run.get('timestamp'),
                        'status': run.get('status', 'unknown'),
                        'record_count': run.get('symbols_successful', 0)
                    }
            
            return symbol_history
        except:
            return {}
    
    def get_download_stats(self):
        """Get summary statistics of all downloads
        
        Returns:
            dict: Download statistics
        """
        try:
            runs = self.history.history.get('runs', [])
            
            total_downloads = len(runs)
            successful = sum(1 for r in runs if r.get('status') == 'complete')
            failed = sum(1 for r in runs if r.get('status') != 'complete')
            avg_duration = sum(r.get('duration_seconds', 0) for r in runs) / max(total_downloads, 1)
            
            return {
                'total_downloads': total_downloads,
                'successful': successful,
                'failed': failed,
                'avg_duration': avg_duration
            }
        except:
            return {'total_downloads': 0, 'successful': 0, 'failed': 0, 'avg_duration': 0}
    
    def _check_and_update_instruments(self):
        """Check for instrument list updates before download"""
        try:
            from core.instruments_manager import get_instruments_manager
            
            im = get_instruments_manager()
            if im.check_for_updates():
                if HAS_STREAMLIT:
                    with st.spinner("Updating instruments list..."):
                        im.download_all_instruments()
                        st.info("Instruments list updated")
                else:
                    im.download_all_instruments()
                    logger.info("Instruments list updated")
        except Exception as e:
            logger.warning(f"Failed to check/update instruments: {e}")
    
    def _auto_aggregate_data(self):
        """Automatically aggregate daily data to longer intervals after download"""
        try:
            from core.data_aggregator import aggregate_daily_to_all_longer_intervals
            
            universe = self.get_download_universe()
            all_symbols = {}
            for segment in universe.values():
                all_symbols.update(segment)
            
            if HAS_STREAMLIT:
                st.info("Starting automatic data aggregation...")
                progress_bar = st.progress(0)
                status_text = st.empty()
            else:
                logger.info(f"Starting automatic aggregation for {len(all_symbols)} symbols...")
            
            success = 0
            for idx, symbol in enumerate(all_symbols.keys()):
                try:
                    aggregate_daily_to_all_longer_intervals(symbol)
                    success += 1
                except Exception as e:
                    logger.debug(f"Aggregation failed for {symbol}: {e}")
                
                if HAS_STREAMLIT:
                    status_text.text(f"Aggregating {symbol}... ({idx+1}/{len(all_symbols)})")
                    progress_bar.progress((idx + 1) / len(all_symbols))
            
            if HAS_STREAMLIT:
                st.success(f"Aggregation complete: {success}/{len(all_symbols)} symbols")
            else:
                logger.info(f"Aggregation complete: {success}/{len(all_symbols)} symbols")
        
        except Exception as e:
            logger.warning(f"Auto-aggregation error: {e}")
    
    def aggregate_daily_data(self, parallel=True, max_workers=4):
        """Aggregate all daily data to weekly, monthly, quarterly, yearly
        
        Args:
            parallel: Use parallel processing
            max_workers: Number of parallel workers
        
        Returns:
            dict: Summary of aggregation results
        """
        from core.data_aggregator import aggregate_daily_to_all_longer_intervals
        
        start_time = dt.datetime.now()
        universe = self.get_download_universe()
        
        # Flatten universe for all symbols
        all_symbols = {}
        for segment in universe.values():
            all_symbols.update(segment)
        
        total = len(all_symbols)
        success = 0
        failed = 0
        
        if HAS_STREAMLIT:
            st.write(f"Aggregating daily data to weekly/monthly/quarterly/yearly for {total} symbols...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        else:
            logger.info(f"Aggregating daily data for {total} symbols...")
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for idx, symbol in enumerate(all_symbols.keys()):
                    future = executor.submit(aggregate_daily_to_all_longer_intervals, symbol)
                    futures[future] = (idx, symbol)
                
                completed = 0
                for future in as_completed(futures):
                    idx, symbol = futures[future]
                    try:
                        results = future.result()
                        if results:
                            success += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Aggregation failed for {symbol}: {e}")
                        failed += 1
                    
                    completed += 1
                    if HAS_STREAMLIT:
                        status_text.text(f"Aggregating {symbol}... ({completed}/{total})")
                        progress_bar.progress(completed / total)
        else:
            # Sequential
            for idx, symbol in enumerate(all_symbols.keys()):
                results = aggregate_daily_to_all_longer_intervals(symbol)
                if results:
                    success += 1
                else:
                    failed += 1
                
                if HAS_STREAMLIT:
                    status_text.text(f"Aggregating {symbol}... ({idx+1}/{total})")
                    progress_bar.progress((idx + 1) / total)
        
        duration = (dt.datetime.now() - start_time).total_seconds()
        summary = {
            'status': 'complete',
            'symbols_processed': total,
            'symbols_successful': success,
            'symbols_failed': failed,
            'duration_seconds': duration,
            'intervals': ['week', 'month', 'quarter', 'year']
        }
        
        if HAS_STREAMLIT:
            st.success(f"Aggregation complete! Success: {success}/{total}")
        else:
            logger.info(f"Aggregation complete! Success: {success}/{total} | Duration: {duration:.1f}s")
        
        return summary

    def load_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a symbol from local storage.
        
        Args:
            symbol (str): Symbol name (e.g., 'INFY')
            interval (str): Timeframe (e.g., 'day', '5minute')
            
        Returns:
            pd.DataFrame: OHLCV data or None if not found
        """
        try:
            # Handle timeframe mapping if needed
            if interval not in ['day', '5minute', 'week', 'month']:
                # Map '1d' -> 'day', etc. if you have such mapping
                pass

            file_path = BASE_DIR / interval / f"{symbol}.parquet"
            
            if not file_path.exists():
                logger.debug(f"[DATA] No local file for {symbol} ({interval})")
                return None
                
            df = pd.read_parquet(file_path)
            
            # Ensure index is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"[DATA] Error loading {symbol}: {e}")
            return None


# Convenience functions
_manager = None

def get_data_manager():
    """Get or create DataManager instance"""
    global _manager
    if _manager is None:
        _manager = DataManager()
    return _manager

# Import here to avoid circular imports
from core.data_aggregator import aggregate_daily_to_all_longer_intervals
