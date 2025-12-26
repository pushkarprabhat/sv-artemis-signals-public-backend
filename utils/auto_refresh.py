# utils/auto_refresh.py — AUTOMATIC DATA REFRESH SCHEDULING
# Downloads new data at market intervals (15min, 30min, 60min after market open)

import datetime as dt
from pathlib import Path
from config import BASE_DIR, TIMEFRAMES
import pandas as pd
from utils.logger import logger
from core.downloader import download_price_data, download_all_price_data

# Market hours: 9:15 AM - 3:30 PM (IST)
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)

# Intervals (in minutes) after market open when data becomes available
DATA_AVAILABLE_AFTER = {
    "15minute": 15,    # 15-min candle available at 9:30 AM
    "30minute": 30,    # 30-min candle available at 9:45 AM
    "60minute": 60,    # 60-min candle available at 10:15 AM
    "day": 0,          # Daily data after market close
    "week": 0,         # Weekly data after market close
    "month": 0         # Monthly data after market close
}

def get_next_refresh_time(timeframe):
    """Get the next time when data for this timeframe will be available
    
    Args:
        timeframe: "15minute", "30minute", "60minute", "day", "week", "month"
    
    Returns:
        datetime: Next expected refresh time
    """
    now = dt.datetime.now()
    minutes_after_open = DATA_AVAILABLE_AFTER[timeframe]
    
    if timeframe in ["15minute", "30minute", "60minute"]:
        # Intraday data available during market hours
        next_refresh = now.replace(
            hour=MARKET_OPEN.hour,
            minute=MARKET_OPEN.minute,
            second=0,
            microsecond=0
        ) + dt.timedelta(minutes=minutes_after_open)
        
        # If that time has passed today, schedule for tomorrow
        if next_refresh <= now:
            next_refresh += dt.timedelta(days=1)
        
        return next_refresh
    else:
        # Daily/weekly/monthly data after market close
        next_refresh = now.replace(
            hour=MARKET_CLOSE.hour,
            minute=MARKET_CLOSE.minute,
            second=0,
            microsecond=0
        )
        
        # If market already closed today, schedule for next day
        if next_refresh <= now:
            next_refresh += dt.timedelta(days=1)
        
        return next_refresh

def should_refresh(timeframe, last_refresh_time=None):
    """Check if data should be refreshed for this timeframe
    
    Args:
        timeframe: "15minute", "30minute", "60minute", "day", "week", "month"
        last_refresh_time: datetime of last refresh (check file mtime if None)
    
    Returns:
        bool: True if refresh is due
    """
    now = dt.datetime.now()
    
    # Check if market is open
    if not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
        return False
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    next_refresh = get_next_refresh_time(timeframe)
    return now >= next_refresh

def get_minutes_until_refresh(timeframe):
    """Get minutes until next refresh is available
    
    Args:
        timeframe: "15minute", "30minute", "60minute", "day", "week", "month"
    
    Returns:
        int: Minutes until data is available (-1 if market closed)
    """
    now = dt.datetime.now()
    
    # Check if market is open
    if not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
        return -1
    
    next_refresh = get_next_refresh_time(timeframe)
    minutes_remaining = (next_refresh - now).total_seconds() / 60
    
    return max(0, int(minutes_remaining))

def display_refresh_status():
    """Display when data will be refreshed for each timeframe
    
    Returns:
        dict: {timeframe: "X minutes" or "Available now" or "Market closed"}
    """
    status = {}
    for tf in TIMEFRAMES:
        minutes = get_minutes_until_refresh(tf)
        if minutes == -1:
            status[tf] = "⏸️ Market closed"
        elif minutes == 0:
            status[tf] = "✅ Available now"
        else:
            status[tf] = f"⏳ {minutes} min"
    
    return status

def auto_refresh_check(symbols):
    """Check each timeframe and auto-refresh if needed
    
    Args:
        symbols: list of symbols to refresh
    
    Returns:
        dict: {timeframe: (success_count, total_count)}
    """
    results = {}
    
    for tf in TIMEFRAMES:
        if should_refresh(tf):
            success = 0
            for symbol in symbols:
                if download_price_data(symbol, force_refresh=False):
                    success += 1
            results[tf] = (success, len(symbols))
        else:
            results[tf] = (0, len(symbols))  # Not due for refresh
    
    return results
