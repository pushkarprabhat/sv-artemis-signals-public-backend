# utils/index_data.py
# Major indices data fetcher with OHLCV
# Professional market index data management

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from utils.helpers import get_ltp
from utils.logger import logger

# Index constituents mapping
INDEX_CONSTITUENTS = {
    "NIFTY 50": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HDFC", "ITC", "SBIN", 
        "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "HCL", "MARUTI",
        "BAJFINANCE", "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND", "ONGC", "NTPC",
        "POWERGRID", "SUNPHARMA", "INDUSINDBK", "M&M", "TECHM", "JSWSTEEL", "TATASTEEL",
        "ADANIENT", "BAJAJFINSV", "TATAMOTORS", "HINDALCO", "COALINDIA", "CIPLA",
        "GRASIM", "EICHERMOT", "BRITANNIA", "HEROMOTOCO", "BPCL", "DIVISLAB", "DRREDDY",
        "SHREECEM", "APOLLOHOSP", "SBILIFE", "HDFCLIFE", "TATACONSUM", "ADANIPORTS",
        "HINDUNILVR", "UPL"
    ],
    "NIFTY BANK": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK",
        "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "AUBANK", "BANKBARODA"
    ],
    "NIFTY FIN SERVICE": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "BAJFINANCE",
        "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULIFE", "HDFCAMC", "MUTHOOTFIN",
        "CHOLAFIN", "LICHSGFIN", "PNBHOUSING", "SHRIRAMFIN", "RECLTD", "PFC",
        "INDUSINDBK", "POONAWALLA"
    ],
    "NIFTY MID SELECT": [
        "CONCOR", "SUPREMEIND", "ASTRAZEN", "BOSCHLTD", "CUMMINSIND", "HAL",
        "MAXHEALTH", "BANKBARODA", "PERSISTENT", "CANFINHOME", "FEDERALBNK",
        "CROMPTON", "MPHASIS", "LAURUSLABS", "BATAINDIA", "DEEPAKNTR",
        "MFSL", "LTTS", "INDUSTOWER", "TORNTPHARM", "PEL", "PFIZER",
        "DALBHARAT", "LALPATHLAB", "PAGEIND"
    ]
}

def get_index_metadata() -> Dict:
    """Get metadata for major indices"""
    return {
        "NIFTY": {
            "full_name": "NIFTY 50",
            "description": "India's benchmark stock index",
            "constituent_count": 50,
            "sector": "Broad Market",
            "color": "#4CAF50"
        },
        "BANKNIFTY": {
            "full_name": "NIFTY BANK",
            "description": "Banking sector index",
            "constituent_count": 12,
            "sector": "Banking",
            "color": "#2196F3"
        },
        "FINNIFTY": {
            "full_name": "NIFTY FIN SERVICE",
            "description": "Financial services index",
            "constituent_count": 20,
            "sector": "Finance",
            "color": "#FF9800"
        },
        "MIDCPNIFTY": {
            "full_name": "NIFTY MID SELECT",
            "description": "Mid-cap stocks index",
            "constituent_count": 25,
            "sector": "Mid Cap",
            "color": "#9C27B0"
        },
        "INDIAVIX": {
            "full_name": "INDIA VIX",
            "description": "Volatility index",
            "constituent_count": 0,
            "sector": "Volatility",
            "color": "#F44336"
        }
    }

def get_index_ohlcv(symbol: str, timeframe: str = "day") -> Optional[Dict]:
    """Get OHLCV data for an index from historical files
    
    Args:
        symbol: Index symbol (e.g., "NIFTY", "BANKNIFTY")
        timeframe: Data timeframe (default: "day")
    
    Returns:
        Dict with open, high, low, close, volume, prev_close
    """
    try:
        # Map short names to full names for file lookup
        symbol_map = {
            "NIFTY": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY": "NIFTY FIN SERVICE",
            "MIDCPNIFTY": "NIFTY MID SELECT",
            "INDIAVIX": "INDIA VIX"
        }
        
        file_symbol = symbol_map.get(symbol, symbol)
        
        # Try to load from marketdata directory
        data_path = Path(f"marketdata/NSE/{timeframe}")
        if not data_path.exists():
            logger.debug(f"Data path not found: {data_path}")
            return None
        
        # Look for parquet or csv file
        file_pattern = f"{file_symbol}*.parquet"
        files = list(data_path.glob(file_pattern))
        
        if not files:
            # Try alternative naming
            file_pattern = f"*{symbol}*.parquet"
            files = list(data_path.glob(file_pattern))
        
        if not files:
            logger.debug(f"No data file found for {symbol}")
            return None
        
        # Load the most recent file
        df = pd.read_parquet(files[0])
        
        if df.empty or len(df) < 2:
            return None
        
        # Get latest and previous day
        df = df.sort_index()
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        return {
            "open": float(latest.get('open', 0)),
            "high": float(latest.get('high', 0)),
            "low": float(latest.get('low', 0)),
            "close": float(latest.get('close', 0)),
            "volume": int(latest.get('volume', 0)),
            "prev_close": float(previous.get('close', 0)),
            "timestamp": latest.name if hasattr(latest, 'name') else datetime.now()
        }
        
    except Exception as e:
        logger.debug(f"Error getting OHLCV for {symbol}: {e}")
        return None

def get_constituent_data(index_name: str) -> pd.DataFrame:
    """Get live data for all constituents of an index
    
    Args:
        index_name: Full index name (e.g., "NIFTY 50")
    
    Returns:
        DataFrame with constituent data
    """
    constituents = INDEX_CONSTITUENTS.get(index_name, [])
    
    if not constituents:
        return pd.DataFrame()
    
    data = []
    for symbol in constituents:
        try:
            # Get live price
            ltp = get_ltp(symbol)
            
            # Get OHLCV from historical
            ohlcv = get_index_ohlcv(symbol)
            
            if ohlcv and ltp:
                change = ltp - ohlcv['prev_close']
                change_pct = (change / ohlcv['prev_close']) * 100 if ohlcv['prev_close'] > 0 else 0
                
                data.append({
                    'symbol': symbol,
                    'ltp': ltp,
                    'open': ohlcv['open'],
                    'high': ohlcv['high'],
                    'low': ohlcv['low'],
                    'close': ohlcv['close'],
                    'volume': ohlcv['volume'],
                    'prev_close': ohlcv['prev_close'],
                    'change': change,
                    'change_pct': change_pct
                })
        except Exception as e:
            logger.debug(f"Error getting data for {symbol}: {e}")
            continue
    
    return pd.DataFrame(data)

def format_large_number(num: float) -> str:
    """Format large numbers with K/M/B suffix"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"
