# utils/helpers.py — Shared utilities & KiteConnect helpers
# Global utilities for price fetching, formatting, and system operations
# Professional trading infrastructure

from kiteconnect import KiteConnect
from utils.logger import logger
import pandas as pd
from datetime import datetime

# Global KiteConnect instance
kite = None

def set_kite(instance: KiteConnect):
    """Set global KiteConnect instance
    
    Args:
        instance: Initialized KiteConnect object or None
    """
    global kite
    kite = instance
    if instance is not None:
        logger.info("[OK] KiteConnect instance set globally")
    else:
        logger.warning("[WARN] KiteConnect instance set to None")

def get_kite_instance() -> KiteConnect:
    """Get the global KiteConnect instance
    
    Returns:
        KiteConnect instance or None if not initialized
    """
    global kite
    return kite

def get_ltp(symbol: str, exchange: str = "NSE") -> float:
    """Get last traded price for a symbol with error handling
    
    Args:
        symbol: Stock symbol (e.g., 'INFY') or index name
        exchange: Exchange code (default 'NSE')
    
    Returns:
        Last traded price or None if error
    
    Note:
        Indices require full names in Kite API:
        - Use "NIFTY 50" not "NIFTY"
        - Use "NIFTY BANK" not "BANKNIFTY"
        Function auto-maps common abbreviations
    """
    try:
        if kite is None:
            logger.debug(f"KiteConnect not initialized, cannot fetch LTP for {symbol}")
            return None
        
        # Map common symbols to correct Kite format
        # Indices require full names, commodities need different handling
        symbol_map = {
            # NSE Indices
            "NIFTY": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY": "NIFTY FIN SERVICE",
            "MIDCPNIFTY": "NIFTY MID SELECT",
            "INDIAVIX": "INDIA VIX",
            # BSE
            "SENSEX": "SENSEX",  # Note: May not support live quotes
            # MCX Commodities - require futures contracts (skipped for now)
            # CRUDEOIL, GOLD, SILVER, USDINR need contract months
        }
        
        # Use mapped symbol if available
        mapped_symbol = symbol_map.get(symbol, symbol)
        
        key = f"{exchange}:{mapped_symbol}"
        data = kite.ltp(key)
        
        if key in data and 'last_price' in data[key]:
            logger.debug(f"✅ LTP fetched for {symbol}: ₹{data[key]['last_price']:,.2f}")
            return data[key]['last_price']
        else:
            logger.debug(f"No valid price data for {symbol} (tried {mapped_symbol})")
            return None
    except Exception as e:
        logger.debug(f"Could not fetch LTP for {symbol}: {e}")
        return None

def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as Indian Rupees with comma separator
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
    
    Returns:
        Formatted string like '₹1,23,456.00'
    """
    try:
        if value is None or pd.isna(value):
            return "₹0.00"
        return f"₹{value:,.{decimals}f}"
    except:
        return "₹0.00"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage
    
    Args:
        value: Numeric value (as decimal, e.g., 0.15 for 15%)
        decimals: Number of decimal places
    
    Returns:
        Formatted string like '15.00%'
    """
    try:
        if value is None or pd.isna(value):
            return "0.00%"
        return f"{value*100:.{decimals}f}%"
    except:
        return "0.00%"

def format_date(dt_obj: datetime, fmt: str = "%Y-%m-%d") -> str:
    """Format datetime object to string
    
    Args:
        dt_obj: datetime object
        fmt: Format string
    
    Returns:
        Formatted date string
    """
    try:
        if dt_obj is None:
            return "N/A"
        if isinstance(dt_obj, str):
            return dt_obj
        return dt_obj.strftime(fmt)
    except:
        return "N/A"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if division not possible
    
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def get_live_indices_data() -> dict:
    """Get live data for major indices, VIX, commodities and forex
    
    Returns:
        Dictionary with live prices and changes for all instruments
    """
    try:
        if kite is None:
            logger.debug("KiteConnect not initialized, cannot fetch indices data")
            return {}
        
        # Major indices, VIX, commodities, and forex
        instruments = [
            # Equity Indices
            "NSE:NIFTY 50",           # NIFTY 50
            "NSE:NIFTY 100",          # NIFTY 100
            "NSE:NIFTY 500",          # NIFTY 500
            "NSE:BANKNIFTY",          # BANKNIFTY
            "NSE:SENSEX",             # SENSEX (BSE)
            
            # Volatility Index
            "NSE:INDIAVIX",           # INDIA VIX
            
            # Commodities (MCX format)
            "MCX:CRUDEOIL",           # Crude Oil
            "MCX:NATURALGAS",         # Natural Gas
            "MCX:GOLD",               # Gold
            "MCX:SILVER",             # Silver
            
            # Currency Pairs (FOREX)
            "NSE:USDINR",             # USD/INR
            "NSE:EURINR",             # EUR/INR
            "NSE:GBPINR",             # GBP/INR
            "NSE:JPYINR",             # JPY/INR
        ]
        
        try:
            data = kite.ltp(instruments)
        except Exception as e:
            logger.debug(f"Partial data fetch error: {e}")
            # Try fetching indices only if full list fails
            instruments = [
                "NSE:NIFTY 50",
                "NSE:NIFTY 500",
                "NSE:SENSEX",
                "NSE:INDIAVIX",
                "NSE:USDINR",
            ]
            try:
                data = kite.ltp(instruments)
            except:
                return {}
        
        if not data:
            return {}
        
        indices_data = {}
        for key, value in data.items():
            try:
                if not isinstance(value, dict):
                    continue
                
                symbol = key.split(':')[1] if ':' in key else key
                ltp = value.get('last_price', 0)
                ohlc = value.get('ohlc', {}) if isinstance(value.get('ohlc'), dict) else {}
                open_price = ohlc.get('open', 0) if ohlc else 0
                high = ohlc.get('high', 0)
                low = ohlc.get('low', 0)
                
                change = ltp - open_price if open_price else 0
                change_pct = (change / open_price * 100) if open_price != 0 else 0
                
                indices_data[symbol] = {
                    'ltp': float(ltp),
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'change': float(change),
                    'change_pct': float(change_pct),
                }
            except Exception as row_err:
                logger.debug(f"Error processing {key}: {row_err}")
                continue
        
        if indices_data:
            logger.info(f"[SUCCESS] Fetched live data for {len(indices_data)} instruments")
        
        return indices_data
        
    except Exception as e:
        logger.error(f"Could not fetch indices data: {e}")
        return {}

def get_nifty500_heatmap_data() -> pd.DataFrame:
    """Get current prices for NIFTY500 stocks for heatmap
    
    Returns:
        DataFrame with symbol, price, change, change_pct
    """
    try:
        if kite is None:
            logger.debug("KiteConnect not initialized")
            return pd.DataFrame()
        
        from universe.symbols import load_universe
        
        # Load universe
        universe = load_universe()
        
        # Get NIFTY50 stocks (In_NIFTY50 column)
        if 'In_NIFTY50' in universe.columns:
            nifty_stocks = universe[universe['In_NIFTY50'] == 'Y'].copy()
        else:
            # Fallback to first 50 stocks
            nifty_stocks = universe.head(50).copy()
        
        if len(nifty_stocks) == 0:
            logger.debug("No NIFTY50 stocks found in universe")
            return pd.DataFrame()
        
        # Build instrument list - use Symbol column if available
        if 'Symbol' in nifty_stocks.columns:
            symbols = nifty_stocks['Symbol'].unique()[:50]  # Limit to 50 for speed
        elif 'tradingsymbol' in nifty_stocks.columns:
            symbols = nifty_stocks['tradingsymbol'].unique()[:50]
        else:
            logger.debug("No Symbol column found in universe")
            return pd.DataFrame()
        
        instruments = [f"NSE:{sym}" for sym in symbols]
        
        if not instruments:
            return pd.DataFrame()
        
        # Fetch data
        try:
            data = kite.ltp(instruments)
        except Exception as ltp_err:
            logger.debug(f"Failed to fetch LTP data: {ltp_err}")
            return pd.DataFrame()
        
        if not data:
            return pd.DataFrame()
        
        rows = []
        for key, value in data.items():
            try:
                symbol = key.split(':')[1] if ':' in key else key
                ltp = value.get('last_price', 0) if isinstance(value, dict) else 0
                ohlc = value.get('ohlc', {}) if isinstance(value, dict) else {}
                open_price = ohlc.get('open', 0) if isinstance(ohlc, dict) else 0
                
                if open_price != 0:
                    change = ltp - open_price
                    change_pct = (change / open_price) * 100
                else:
                    change = 0
                    change_pct = 0
                
                rows.append({
                    'symbol': symbol,
                    'ltp': float(ltp),
                    'open': float(open_price),
                    'change': float(change),
                    'change_pct': float(change_pct),
                })
            except Exception as row_err:
                logger.debug(f"Error processing row {key}: {row_err}")
                continue
        
        if not rows:
            logger.debug("No valid data rows extracted from heatmap")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(rows)
        logger.info(f"[OK] Fetched heatmap data for {len(result_df)} stocks")
        return result_df
        
    except Exception as e:
        logger.error(f"Could not fetch NIFTY500 heatmap data: {e}")
        return pd.DataFrame()


# ============================================================================
# SYMBOL EXTRACTION HELPERS
# ============================================================================

import re

def get_symbol_from_instrument(instrument):
    """
    Extract trading symbol from instrument object/dict
    
    Args:
        instrument: Dict with instrument data from Kite API
                   Keys: 'tradingsymbol', 'symbol', 'instrument_type', etc.
    
    Returns:
        str: Trading symbol (e.g., 'RELIANCE', 'INFY')
    
    Raises:
        ValueError: If no symbol found in instrument
    
    Example:
        >>> instr = {'tradingsymbol': 'RELIANCE', 'instrument_type': 'EQ'}
        >>> get_symbol_from_instrument(instr)
        'RELIANCE'
    """
    # Try different possible keys
    for key in ['tradingsymbol', 'symbol', 'name']:
        if key in instrument and instrument[key]:
            return instrument[key]
    
    raise ValueError(f"No symbol found in instrument: {instrument}")


def get_base_symbol(trading_symbol, instrument_type='EQ'):
    """
    Extract base symbol from trading symbol, handling derivatives
    
    Args:
        trading_symbol: Full trading symbol (e.g., 'RELIANCE24DEC2500CE')
        instrument_type: Type hint - 'EQ', 'FUT', 'OPT', etc.
    
    Returns:
        str: Base symbol (e.g., 'RELIANCE')
    
    Examples:
        Equity: 'RELIANCE' → 'RELIANCE'
        Futures: 'RELIANCE24DEC' → 'RELIANCE'
        Options: 'RELIANCE24DEC2500CE' → 'RELIANCE'
    """
    if instrument_type == 'EQ':
        # Equity: symbol is as-is
        return trading_symbol
    
    elif instrument_type in ['FUT', 'OPT']:
        # Futures/Options: Extract alphabetic prefix
        # Patterns: RELIANCE24DEC, RELIANCE24DEC2500CE, etc.
        match = re.match(r'^([A-Z&\-]+)', trading_symbol)
        if match:
            return match.group(1)
        return trading_symbol
    
    else:
        # For unknown types, return full symbol
        return trading_symbol


def extract_symbol_by_type(instrument):
    """
    Extract base symbol from instrument based on its type
    
    Args:
        instrument: Dict with 'tradingsymbol' and 'instrument_type' keys
    
    Returns:
        str: Base symbol extracted per instrument type
    
    Example:
        >>> opt = {'tradingsymbol': 'RELIANCE24DEC2500CE', 'instrument_type': 'OPT'}
        >>> extract_symbol_by_type(opt)
        'RELIANCE'
    """
    trading_symbol = get_symbol_from_instrument(instrument)
    instrument_type = instrument.get('instrument_type', 'EQ')
    
    return get_base_symbol(trading_symbol, instrument_type)


# Mapping table for efficient symbol extraction
SYMBOL_EXTRACTORS = {
    'EQ': lambda ts: ts,  # Equity: return as-is
    'FUT': lambda ts: re.match(r'^([A-Z&\-]+)', ts).group(1) if re.match(r'^([A-Z&\-]+)', ts) else ts,
    'OPT': lambda ts: re.match(r'^([A-Z&\-]+)', ts).group(1) if re.match(r'^([A-Z&\-]+)', ts) else ts,
}

def get_symbol_mapped(instrument):
    """
    Fast symbol extraction using mapping table
    
    Args:
        instrument: Dict with instrument data
    
    Returns:
        str: Base symbol
    """
    trading_symbol = get_symbol_from_instrument(instrument)
    instrument_type = instrument.get('instrument_type', 'EQ')
    
    extractor = SYMBOL_EXTRACTORS.get(instrument_type, lambda ts: ts)
    return extractor(trading_symbol)


def get_expiry_type_by_month(expiry_date, all_expiry_dates_in_month):
    """
    Classify expiry as MONTHLY or WEEKLY
    
    NEW LOGIC (Updated Dec 2025):
    - MONTHLY: Last expiry date in the month
    - WEEKLY: All other expiry dates in the month
    
    ADVANTAGE: Exchange/segment-agnostic, not tied to any specific day of week
    Simply sorts all expiry dates in a month and marks the last one as MONTHLY.
    
    Args:
        expiry_date: The expiry date to classify (datetime.date or pd.Timestamp)
        all_expiry_dates_in_month: List of ALL expiry dates in that month
    
    Returns:
        str: 'MONTHLY' or 'WEEKLY'
    
    Example:
        >>> import pandas as pd
        >>> from datetime import date
        >>> all_dates = [
        ...     date(2024, 12, 5),   # Weekly
        ...     date(2024, 12, 12),  # Weekly
        ...     date(2024, 12, 19),  # Weekly
        ...     date(2024, 12, 26)   # MONTHLY (last)
        ... ]
        >>> get_expiry_type_by_month(date(2024, 12, 26), all_dates)
        'MONTHLY'
        >>> get_expiry_type_by_month(date(2024, 12, 19), all_dates)
        'WEEKLY'
    """
    from datetime import date
    
    # Convert to date if needed
    if hasattr(expiry_date, 'date'):
        expiry_date = expiry_date.date()
    
    # Convert all dates to date objects
    dates_as_dates = []
    for d in all_expiry_dates_in_month:
        if hasattr(d, 'date'):
            dates_as_dates.append(d.date())
        else:
            dates_as_dates.append(d)
    
    # Sort and find max
    sorted_dates = sorted(dates_as_dates)
    
    if not sorted_dates:
        return 'UNKNOWN'
    
    last_date = sorted_dates[-1]
    
    # If this date is the last one in the month
    if expiry_date == last_date:
        return 'MONTHLY'
    else:
        return 'WEEKLY'
