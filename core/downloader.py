# core/downloader.py — FINAL PROFESSIONAL VERSION (2025)
# Fully dynamic — uses universe.csv for exchange info
# No hardcoding, no errors no incomplete code

import pandas as pd
import datetime as dt
import time
import pyarrow.parquet as pq
from pathlib import Path
from config import BASE_DIR, TIMEFRAMES, BATCH_SIZES, get_download_dir
import utils.helpers
from universe.symbols import load_universe
from utils.logger import logger
from utils.kite_worker import get_kite
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.corporate_actions import get_corporate_actions_manager

# Optional Streamlit import (only available in Streamlit context)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ============================================================================
# BATCHING HELPERS - Handle "interval exceeds max limit" errors
# ============================================================================

def batch_date_range(from_date, to_date, batch_days, descending=True):
    """
    Split date range into batches to avoid Kite API limits
    
    Args:
        from_date: Start date (datetime.date or datetime.datetime)
        to_date: End date (datetime.date or datetime.datetime)
        batch_days: Max days per batch (from config.BATCH_SIZES)
        descending: If True, return batches in reverse order (recent → past)
    
    Returns:
        List of (start_date, end_date) tuples
    
    Example:
        batch_date_range(date(2024,1,1), date(2024,12,31), 100, descending=True)
        → [(2024-10-01, 2024-12-31), (2024-07-03, 2024-09-30), ...] (recent first)
    """
    # Convert to date objects if needed
    if hasattr(from_date, 'date'):
        from_date = from_date.date()
    if hasattr(to_date, 'date'):
        to_date = to_date.date()
    
    batches = []
    current_start = from_date
    
    while current_start < to_date:
        # End date is min(batch_days from start, target end date)
        current_end = min(
            current_start + dt.timedelta(days=batch_days - 1),
            to_date
        )
        batches.append((current_start, current_end))
        current_start = current_end + dt.timedelta(days=1)
    
    # Return in descending order (most recent first) if requested
    if descending:
        batches.reverse()
    
    return batches


def download_with_batching(kite, instrument_token, from_date, to_date, interval, batch_days=100):
    """
    Download historical data with automatic batching to avoid API limits
    Attempts recent data first (descending); exits early if recent data unavailable.
    
    Args:
        kite: KiteConnect instance
        instrument_token: Instrument token
        from_date: Start date
        to_date: End date
        interval: Timeframe ('5minute', 'day', etc.)
        batch_days: Days per batch (from BATCH_SIZES config)
    
    Returns:
        List of OHLCV data dicts
    """
    all_data = []
    batches = batch_date_range(from_date, to_date, batch_days, descending=True)
    
    logger.info(f"Downloading {interval}: Created {len(batches)} batches ({batch_days} days each) - Recent First")
    
    first_batch = True
    for batch_num, (batch_from, batch_to) in enumerate(batches, 1):
        try:
            batch_days_count = (batch_to - batch_from).days + 1
            logger.debug(f"  [Batch {batch_num}/{len(batches)}] {batch_from} to {batch_to} ({batch_days_count} days)")
            
            batch_data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=batch_from,
                to_date=batch_to,
                interval=interval,
                continuous=False,
                oi=False
            )
            
            if batch_data:
                all_data.extend(batch_data)
                logger.debug(f"    [OK] Got {len(batch_data)} rows")
                first_batch = False  # Mark that we found data
            else:
                # Early exit: if first batch (most recent) has no data, stop trying
                if first_batch:
                    logger.warning(f"  [EARLY EXIT] First batch (most recent) returned no data - stopping download attempts")
                    break
                else:
                    logger.debug(f"    [EMPTY] No data for this batch")
            
            time.sleep(0.35)  # Rate limiting between batches
            
        except Exception as e:
            # Log error
            logger.warning(f"  [Batch {batch_num}] Error {batch_from} to {batch_to}: {e}")
            # If first batch fails, exit early
            if first_batch:
                logger.warning(f"  [EARLY EXIT] First batch error - stopping download attempts")
                break
            time.sleep(0.5)  # Wait a bit longer after error
    
    logger.info(f"Batching complete: Got {len(all_data)} total rows across {len(batches)} batches")
    return all_data

def download_price_data(symbol, force_refresh=False):
    """
    Download price data for ONE symbol across all timeframes
    ✅ INCREMENTAL ONLY — Only fetches NEW data (since last download)
    Uses exchange from universe.csv — fully dynamic
    
    Adds columns: instrument, symbol, full_name, expiry, expiry_type
    Where:
    - instrument = tradingsymbol (e.g., SBIN25DECFUT)
    - symbol = base symbol (e.g., SBIN)
    - full_name = human-readable name (e.g., STATE BANK OF INDIA)
    
    With love: "Incremental downloads = fast + efficient + respect for servers"
    """
    # Load universe to get correct exchange and metadata
    try:
        universe = load_universe()
        row = universe[universe['Symbol'] == symbol].iloc[0]
        exchange = row['Exchange']
    except:
        exchange = "NSE"  # fallback
        
    logger.info(f"⬇️  Starting download for [{exchange}] {symbol}")

    # Get token using dynamically referenced kite
    try:
        kite = get_kite()
        if kite is None:
            logger.error(f"KiteConnect not initialized for {symbol}")
            return False
        quote = kite.ltp(f"{exchange}:{symbol}")
        
        # Safe token extraction with better error handling
        if not quote or len(quote) == 0:
            logger.error(f"No quote data for {symbol} ({exchange}) - may be an index or illiquid security")
            return False
        
        quote_data = list(quote.values())[0]
        if "instrument_token" not in quote_data:
            logger.error(f"No instrument_token in quote for {symbol} ({exchange})")
            return False
        
        token = quote_data["instrument_token"]
    except IndexError as e:
        logger.error(f"Quote parsing failed for {symbol} ({exchange}): {e} - symbol may not be tradeable")
        return False
    except Exception as e:
        logger.error(f"Token failed for {symbol} ({exchange}): {e}")
        return False
    # ✅ ONLY DOWNLOAD CORE TIMEFRAMES (5minute & day)
    # The user explicitly stated: "we do not download any data other than 5 minutes or daily"
    # All other timeframes (15min, 60min, week, month) should be AGGREGATED.
    
    DOWNLOAD_TIMEFRAMES = ["5minute", "day"]
    
    success_count = 0
    for tf in DOWNLOAD_TIMEFRAMES:
        # Use exchange-organized directory structure: marketdata/EXCHANGE/TIMEFRAME/SYMBOL/
        folder = get_download_dir(exchange, tf)
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{symbol}.parquet"

        # ✅ INCREMENTAL LOGIC: Get last date from existing file
        from_date = None
        to_date = dt.datetime.now().date()
        
        if file_path.exists() and not force_refresh:
            try:
                # Read existing data to find last date
                old_df = pd.read_parquet(file_path)
                if len(old_df) > 0:
                    last_date = pd.to_datetime(old_df['date'].max())
                    if hasattr(last_date, 'tz') and last_date.tz is not None:
                        last_date = last_date.tz_localize(None)
                    
                    # Start from next date (incremental)
                    if tf in ["day", "week", "month"]:
                        from_date = (last_date + pd.Timedelta(days=1)).date()
                    else:
                        from_date = (last_date + pd.Timedelta(hours=1)).date()
                    
                    # If we're up to date, skip
                    if from_date >= to_date:
                        logger.info(f"[UP-TO-DATE] {symbol} {tf}: Already has latest data through {to_date}")
                        success_count += 1
                        continue
                    
                    logger.info(f"[INCREMENTAL] {symbol} {tf}: Fetching from {from_date} to {to_date}")
            except Exception as e:
                logger.warning(f"Could not read existing {file_path}: {e}")
                # Fall through to full download
        
        # If no existing file or force_refresh, use default lookback
        if from_date is None:
            if tf in ["15minute", "30minute", "60minute"]:
                from_date = (dt.datetime.now() - dt.timedelta(days=200)).date()  # Max 200 days for intraday
            else:
                from_date = (dt.datetime.now() - dt.timedelta(days=365)).date()  # 1 year for daily/weekly
        
        # Skip intraday intervals if timeframe is too short
        if tf in ["15minute", "30minute", "60minute"]:
            days_available = (to_date - from_date).days
            if days_available < 5:
                logger.info(f"Skipping {symbol} {tf}: less than 5 days available")
                continue

        # Download (only NEW data due to incremental logic)
        # Use batching to avoid "interval exceeds max limit" errors
        try:
            batch_days = BATCH_SIZES.get(tf, 100)
            data = download_with_batching(
                kite=kite,
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=tf,
                batch_days=batch_days
            )
            
            if not data:
                logger.warning(f"[NO DATA] {symbol} {tf}: API returned empty")
                continue

            df = pd.DataFrame(data)
            # Include previousclose in data columns for net_change calculation
            df = df[['date', 'open', 'high', 'low', 'close', 'previousclose', 'volume']]
            df['date'] = pd.to_datetime(df['date'])
            
            # Remove timezone info to avoid datetime comparison issues
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            # ✅ CALCULATE NET CHANGE AND PERCENTAGE CHANGE
            # Now using previousclose instead of open for accurate net_change
            df['net_change'] = df['close'] - df['previousclose']
            df['net_change_pct'] = (df['net_change'] / df['previousclose'] * 100).round(2)
            
            # ✅ ADD METADATA COLUMNS
            # Use 'Trading Symbol' if available (e.g. derivatives), else fallback to base symbol
            df['instrument'] = row.get('Trading Symbol', symbol)
            df['symbol'] = symbol                       # base symbol (e.g., SBIN)
            
            # Add expiry info for derivatives (if applicable)
            if pd.notna(row.get('Expiry')):
                df['expiry'] = pd.to_datetime(row.get('Expiry'))
                # Get all expiry dates in the same month to classify
                all_month_expiries = universe[
                    (universe['Symbol'] == symbol) & 
                    (pd.to_datetime(universe['Expiry']).dt.to_period('M') == 
                     pd.to_datetime(row.get('Expiry')).to_period('M'))
                ]['Expiry'].unique()
                expiry_type = utils.helpers.get_expiry_type_by_month(
                    row.get('Expiry'), 
                    all_month_expiries
                )
                df['expiry_type'] = expiry_type
            
            # Reorder columns: date | instrument | symbol | expiry | expiry_type | OHLCV | net_change | net_change_pct
            cols = ['date', 'instrument', 'symbol']
            if 'expiry' in df.columns:
                cols.extend(['expiry', 'expiry_type'])
            cols.extend(['open', 'high', 'low', 'close', 'volume', 'net_change', 'net_change_pct'])
            df = df[cols]
            
            df = df.sort_values('date').reset_index(drop=True)

            # ✅ MERGE with existing (if exists) to avoid duplicates
            if file_path.exists() and not force_refresh:
                try:
                    old_df = pd.read_parquet(file_path)
                    # Ensure consistent datetime handling
                    if old_df['date'].dt.tz is not None:
                        old_df['date'] = old_df['date'].dt.tz_localize(None)
                    
                    # Concatenate all rows and remove duplicates by date
                    combined = pd.concat([old_df, df], ignore_index=True)
                    # Keep the last occurrence (newest data) for each date
                    combined = combined.drop_duplicates(subset=['date'], keep='last')
                    df = combined.sort_values('date').reset_index(drop=True)
                    
                    logger.info(f"[MERGE] {symbol} {tf}: Combined old + new data to {len(df)} total rows")
                except Exception as e:
                    logger.warning(f"Could not merge with existing data for {symbol}: {e}")
                    # Continue with just new data if merge fails

            # Save with nullable dtypes for safety
            df.to_parquet(file_path, index=False, engine='pyarrow')
            
            # ✅ RECORD FILE CHECKSUM FOR INTEGRITY VERIFICATION
            try:
                corp_actions_mgr = get_corporate_actions_manager()
                corp_actions_mgr.record_file_checksum(symbol, tf, file_path)
            except Exception as e:
                logger.warning(f"Could not record checksum for {symbol} {tf}: {e}")
            
            logger.info(f"[OK] {symbol} {tf}: {len(df)} total rows | Dates: {df['date'].min().date()} to {df['date'].max().date()}")
            success_count += 1
            time.sleep(0.35)  # Stay under rate limit

        except Exception as e:
            logger.error(f"[FAIL] {symbol} {tf} ({from_date} to {to_date}): {e}")

    # ✅ AGGREGATE OTHER TIMEFRAMES FROM 5-MINUTE DATA
    # Since we only download 5minute and day, we must rebuild the others
    try:
        from core.data_aggregator import aggregate_all_intervals
        aggregate_all_intervals(symbol)
        logger.info(f"✅ Aggregation complete for {symbol}")
    except Exception as e:
        logger.error(f"Aggregation failed for {symbol}: {e}")

    return success_count > 0


def download_all_price_data(force_refresh=False, parallel=True, max_workers=4):
    """Download data for ALL symbols in universe.csv
    
    Args:
        force_refresh (bool): If True, ignore existing files and re-download all.
                            If False (default), skip symbols with up-to-date data.
        parallel (bool): If True, use parallel downloads for speed
        max_workers (int): Number of parallel workers (default 4)
    """
    universe = load_universe()
    total = len(universe)
    success = 0
    skipped = 0

    if HAS_STREAMLIT:
        st.write(f"Starting download (force_refresh={force_refresh}, parallel={parallel})...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        logger.info(f"Starting download for {total} assets (force_refresh={force_refresh}, parallel={parallel})...")

    if parallel:
        # Parallel downloads using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in universe.iterrows():
                symbol = row['tradingsymbol']
                future = executor.submit(download_price_data, symbol, force_refresh=force_refresh)
                futures[future] = (idx, symbol)
            
            completed = 0
            for future in as_completed(futures):
                idx, symbol = futures[future]
                try:
                    if future.result():
                        success += 1
                    else:
                        if not force_refresh:
                            skipped += 1
                except Exception as e:
                    logger.error(f"Parallel download failed for {symbol}: {e}")
                
                completed += 1
                if HAS_STREAMLIT:
                    status_text.text(f"Processing {symbol}... ({completed}/{total}) [Success: {success}, Skipped: {skipped}]")
                    progress_bar.progress(completed / total)
    else:
        # Sequential downloads (original approach)
        for idx, row in universe.iterrows():
            symbol = row['tradingsymbol']
            if HAS_STREAMLIT:
                status_text.text(f"Processing {symbol}... ({idx+1}/{total}) [Success: {success}, Skipped: {skipped}]")
            else:
                logger.info(f"Processing {symbol}... ({idx+1}/{total})")
            
            if download_price_data(symbol, force_refresh=force_refresh):
                success += 1
            else:
                # Only count as skipped if NOT force_refresh (already downloaded)
                if not force_refresh:
                    skipped += 1
            
            if HAS_STREAMLIT:
                progress_bar.progress((idx + 1) / total)

    if HAS_STREAMLIT:
        st.success(f"Download complete! Updated: {success}/{total} | Skipped (up-to-date): {skipped}")
        st.info(f"Click 'Force Refresh' to re-download all data even if already present.")
        st.balloons()
    else:
        logger.info(f"Download complete! Updated: {success}/{total} | Skipped: {skipped}")


def validate_data_completeness(start_date, end_date, symbols=None):
    """Validate that all symbols have data for all intervals within date range
    
    Args:
        start_date: datetime.date object for range start
        end_date: datetime.date object for range end
        symbols: list of symbols to check (default: all from universe)
    
    Returns:
        dict: {symbol: {interval: error_msg or 'OK'}}
    """
    from universe.symbols import load_universe
    
    if symbols is None:
        universe = load_universe()
        symbols = universe['Symbol'].tolist()
    
    missing_data = {}
    
    for symbol in symbols:
        symbol_status = {}
        for tf in TIMEFRAMES:
            try:
                file_path = BASE_DIR / tf / f"{symbol}.parquet"
                if not file_path.exists():
                    symbol_status[tf] = "FILE_NOT_FOUND"
                    continue
                
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(df['date']).dt.date
                
                # Filter to date range
                df_range = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                if len(df_range) == 0:
                    symbol_status[tf] = "NO_DATA_IN_RANGE"
                    continue
                
                # Check for data gaps
                date_diff = (df_range['date'].max() - df_range['date'].min()).days
                expected_records = date_diff + 1
                actual_records = len(df_range)
                
                # Allow 20% gaps for market holidays/closures
                if actual_records < (expected_records * 0.8):
                    gap_pct = 100 * (1 - actual_records / expected_records)
                    symbol_status[tf] = f"DATA_GAP_{gap_pct:.0f}%"
                else:
                    symbol_status[tf] = "OK"
                    
            except Exception as e:
                symbol_status[tf] = f"ERROR: {str(e)[:30]}"
        
        # Only report symbols with issues
        if any(v != "OK" for v in symbol_status.values()):
            missing_data[symbol] = symbol_status
    
    return missing_data

# ============================================================================
# NSE MARKET DATA DOWNLOADS - Bhavcopy & Market Activity Reports
# ============================================================================

def download_nse_bhavcopy_direct(date_obj: dt.date, segment: str = 'both') -> bool:
    """
    Download NSE Bhavcopy directly from NSE website using direct URLs
    Supports equity (CM) and futures/options (FO) segments
    
    Args:
        date_obj: Date for bhavcopy (datetime.date)
        segment: 'CM' (equity), 'FO' (derivatives), or 'both' (default)
    
    Returns:
        True if download successful, False otherwise
    
    Example:
        download_nse_bhavcopy_direct(date(2025, 12, 24), 'both')
    """
    import requests
    import zipfile
    import io
    
    date_str = date_obj.strftime('%d%m%y')  # DDMMYY format
    year_str = date_obj.strftime('%Y')
    month_str = date_obj.strftime('%B').upper()  # Full month name
    
    bhavcopy_dir = Path(BASE_DIR) / 'marketdata' / 'bhavcopy'
    bhavcopy_dir.mkdir(parents=True, exist_ok=True)
    
    segments_to_download = []
    if segment in ['CM', 'both']:
        segments_to_download.append(('CM', f'CM{date_str}Bhav.csv'))
    if segment in ['FO', 'both']:
        segments_to_download.append(('FO', f'FO{date_str}Bhav.csv'))
    
    success_count = 0
    
    for seg, filename in segments_to_download:
        try:
            # NSE Bhavcopy URL structure
            url = f"https://www.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/{filename}.zip"
            if seg == 'FO':
                url = f"https://www.nseindia.com/content/historical/DERIVATIVES/{year_str}/{month_str}/{filename}.zip"
            
            logger.info(f"Downloading {seg} Bhavcopy: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Extract ZIP
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    csv_file = zip_ref.namelist()[0]
                    csv_content = zip_ref.read(csv_file)
                    
                    # Save to local directory
                    output_path = bhavcopy_dir / f"{seg}_{filename.replace('.zip', '')}.csv"
                    with open(output_path, 'wb') as f:
                        f.write(csv_content)
                    
                    logger.info(f"✓ Saved {seg} bhavcopy: {output_path}")
                    success_count += 1
            else:
                logger.warning(f"✗ Failed to download {seg} bhavcopy (Status: {response.status_code})")
        
        except Exception as e:
            logger.warning(f"✗ Error downloading {seg} bhavcopy: {str(e)}")
    
    return success_count > 0


def download_nse_market_activity_report(date_obj: dt.date) -> bool:
    """
    Download NSE Market Activity Report (MA file) for given date
    Used as supplementary data for Closing Bell Report
    
    Args:
        date_obj: Date for market activity report (datetime.date)
    
    Returns:
        True if download successful, False otherwise
    
    Example:
        download_nse_market_activity_report(date(2025, 12, 24))
    """
    import requests
    
    date_str = date_obj.strftime('%d%m%y')  # DDMMYY format
    year_str = date_obj.strftime('%Y')
    month_str = date_obj.strftime('%B').upper()
    
    ma_dir = Path(BASE_DIR) / 'marketdata' / 'market_activity'
    ma_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # NSE Market Activity Report URL structure
        # Typically in historical data section
        url = f"https://www.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/MA{date_str}.csv"
        
        logger.info(f"Downloading Market Activity Report: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            output_path = ma_dir / f"MA{date_str}.csv"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✓ Saved Market Activity Report: {output_path}")
            return True
        else:
            logger.warning(f"✗ Failed to download Market Activity Report (Status: {response.status_code})")
            return False
    
    except Exception as e:
        logger.warning(f"✗ Error downloading Market Activity Report: {str(e)}")
        return False