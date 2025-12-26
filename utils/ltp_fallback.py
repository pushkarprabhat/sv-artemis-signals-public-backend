# utils/ltp_fallback.py
# Robust LTP fetching with multiple fallback sources
# Enterprise-grade price data reliability

from typing import Optional
from utils.helpers import get_ltp
from utils.index_data import get_index_ohlcv
from utils.logger import logger

def get_ltp_with_fallback(symbol: str, exchange: str = "NSE") -> Optional[float]:
    """
    Get Last Traded Price with multiple fallback sources.
    
    Fallback chain:
    1. Live Kite API (real-time during market hours)
    2. LTP Database (cached from last successful fetch)
    3. Historical OHLCV (latest close price)
    4. Return None only if all sources fail
    
    Args:
        symbol: Stock symbol or index name
        exchange: Exchange code (default 'NSE')
    
    Returns:
        Last traded price or None if all sources fail
    
    Fallback strategy:
    - During market hours: Uses live API → cached → yesterday's close
    - After hours: Uses yesterday's close (most reliable)
    """
    
    # SOURCE 1: Try live Kite API (fastest during market hours)
    try:
        ltp = get_ltp(symbol, exchange)
        if ltp and ltp > 0:
            logger.debug(f"✅ LTP from Kite API: {symbol} = ₹{ltp:,.2f}")
            return ltp
    except Exception as e:
        logger.debug(f"Kite API unavailable for {symbol}: {e}")
    
    # SOURCE 2: Try LTP Database (cached from last successful fetch)
    try:
        from core.ltp_database import get_ltp_database
        ltp_db = get_ltp_database()
        cached = ltp_db.get_ltp(symbol)
        
        if cached and cached.get('last_price'):
            price = cached['last_price']
            timestamp = cached.get('timestamp', 'unknown')
            logger.debug(f"📦 LTP from cache: {symbol} = ₹{price:,.2f} (as of {timestamp})")
            return price
    except Exception as e:
        logger.debug(f"LTP Database unavailable for {symbol}: {e}")
    
    # SOURCE 3: Try historical OHLCV (latest close price)
    try:
        ohlcv = get_index_ohlcv(symbol)
        if ohlcv and ohlcv.get('close'):
            price = ohlcv['close']
            logger.debug(f"📊 LTP from historical: {symbol} = ₹{price:,.2f} (last close)")
            return price
    except Exception as e:
        logger.debug(f"Historical OHLCV unavailable for {symbol}: {e}")
    
    # SOURCE 4: All sources failed
    logger.warning(f"⚠️ No LTP available for {symbol} from any source")
    return None


def get_ohlcv_with_fallback(symbol: str, timeframe: str = "day") -> Optional[dict]:
    """
    Get OHLCV data with fallback to previous day if today unavailable.
    
    Args:
        symbol: Stock symbol or index name
        timeframe: Data timeframe (default 'day')
    
    Returns:
        Dict with OHLCV data or None if unavailable
    """
    try:
        ohlcv = get_index_ohlcv(symbol, timeframe)
        if ohlcv:
            return ohlcv
        
        # If today's data not available, try to construct from cache
        from core.ltp_database import get_ltp_database
        ltp_db = get_ltp_database()
        cached = ltp_db.get_ltp(symbol)
        
        if cached:
            # Use cached LTP as approximation
            price = cached['last_price']
            return {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': cached.get('volume', 0),
                'prev_close': price * 0.99,  # Approximate -1% previous close
                'timestamp': cached.get('timestamp')
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting OHLCV with fallback for {symbol}: {e}")
        return None
