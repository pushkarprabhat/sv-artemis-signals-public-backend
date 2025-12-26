"""
utils/market_aware_refresh.py - Intelligent market-aware data refresh logic

This module implements intelligent refresh strategies based on market status:
- During market hours: Refresh with live data (15-min, 30-min, 60-min intervals)
- After market close: Refresh with daily closing data only
- Before market open: Skip live refresh, only update historical data if needed
- Weekends/Holidays: Skip live refresh entirely
"""

from datetime import datetime, time, timedelta
import pytz
from typing import Dict, List, Optional
from utils.logger import logger
from utils.market_hours import (
    MARKET_OPEN, MARKET_CLOSE, IST, is_market_open, NSE_HOLIDAYS_2024_2025
)

# ============================================================================
# MARKET STATUS CLASSIFICATION
# ============================================================================

class MarketSession:
    """Enum-like class for market session states"""
    LIVE = "LIVE"           # Market is currently open (9:15 AM - 3:30 PM)
    AFTER_CLOSE = "AFTER_CLOSE"  # Market closed, pre-settlement (3:30 PM - 6:00 PM)
    EVENING = "EVENING"     # Evening session (6:00 PM - 9:00 PM)
    NIGHT = "NIGHT"         # Night session (9:00 PM - 9:15 AM)
    WEEKEND = "WEEKEND"     # Saturday or Sunday
    HOLIDAY = "HOLIDAY"     # NSE Holiday
    
    @classmethod
    def all_values(cls):
        return [cls.LIVE, cls.AFTER_CLOSE, cls.EVENING, cls.NIGHT, cls.WEEKEND, cls.HOLIDAY]


def get_market_session(check_time: datetime = None) -> str:
    """
    Determine current market session/status
    
    Args:
        check_time: datetime to check (default: current time in IST)
    
    Returns:
        MarketSession value indicating current market state
    """
    if check_time is None:
        check_time = datetime.now(IST)
    else:
        # Ensure timezone aware
        if check_time.tzinfo is None:
            check_time = IST.localize(check_time)
        else:
            check_time = check_time.astimezone(IST)
    
    # Check weekday (0=Monday, 6=Sunday)
    weekday = check_time.weekday()
    if weekday >= 5:  # Saturday=5, Sunday=6
        return MarketSession.WEEKEND
    
    # Check if holiday
    check_date = check_time.date()
    if check_date in [h.date() for h in NSE_HOLIDAYS_2024_2025]:
        return MarketSession.HOLIDAY
    
    # Check time within session
    current_time = check_time.time()
    
    if MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return MarketSession.LIVE
    elif MARKET_CLOSE < current_time < time(18, 0):  # 3:30 PM to 6:00 PM
        return MarketSession.AFTER_CLOSE
    elif time(18, 0) <= current_time < time(21, 0):  # 6:00 PM to 9:00 PM
        return MarketSession.EVENING
    else:  # 9:00 PM to 9:15 AM
        return MarketSession.NIGHT


def get_refresh_strategy(check_time: datetime = None) -> Dict:
    """
    Get intelligent refresh strategy based on market status
    
    Returns dictionary with:
    {
        'session': Current market session
        'fetch_live': Whether to fetch live intraday data (15/30/60 min)
        'fetch_daily': Whether to fetch daily closing data
        'fetch_historical': Whether to backfill historical data
        'reason': Explanation of strategy
    }
    """
    session = get_market_session(check_time)
    
    strategies = {
        MarketSession.LIVE: {
            'session': MarketSession.LIVE,
            'fetch_live': True,
            'fetch_daily': False,  # Daily not ready yet
            'fetch_historical': False,
            'reason': '📊 Market OPEN: Fetching live intraday data (15/30/60 min)',
            'recommended_intervals': ['15minute', '30minute', '60minute']
        },
        
        MarketSession.AFTER_CLOSE: {
            'session': MarketSession.AFTER_CLOSE,
            'fetch_live': False,  # No new live data after close
            'fetch_daily': True,   # Daily data now available
            'fetch_historical': False,
            'reason': '📈 Market CLOSED (3:30-6:00 PM): Fetching daily closing data',
            'recommended_intervals': ['day']
        },
        
        MarketSession.EVENING: {
            'session': MarketSession.EVENING,
            'fetch_live': False,
            'fetch_daily': True,   # Daily settlement data available
            'fetch_historical': False,
            'reason': '📊 Evening session (6:00-9:00 PM): Fetching/validating daily data',
            'recommended_intervals': ['day', 'week', 'month']
        },
        
        MarketSession.NIGHT: {
            'session': MarketSession.NIGHT,
            'fetch_live': False,
            'fetch_daily': False,  # Daily already downloaded
            'fetch_historical': True,  # Can backfill historical data
            'reason': '🌙 Night session (9:00 PM-9:15 AM): Backfilling historical data only',
            'recommended_intervals': []  # Don't fetch live intraday
        },
        
        MarketSession.WEEKEND: {
            'session': MarketSession.WEEKEND,
            'fetch_live': False,
            'fetch_daily': False,
            'fetch_historical': True,
            'reason': '🏖️ WEEKEND: Backfilling historical data, no live fetch',
            'recommended_intervals': []
        },
        
        MarketSession.HOLIDAY: {
            'session': MarketSession.HOLIDAY,
            'fetch_live': False,
            'fetch_daily': False,
            'fetch_historical': True,
            'reason': '🎉 NSE HOLIDAY: Backfilling historical data only',
            'recommended_intervals': []
        }
    }
    
    return strategies[session]


# ============================================================================
# INTELLIGENT REFRESH FILTERING
# ============================================================================

def filter_intervals_for_session(
    requested_intervals: List[str],
    check_time: datetime = None
) -> Dict:
    """
    Filter requested intervals based on market session
    
    Args:
        requested_intervals: List of intervals user wants to download
        check_time: Time to check market status for
    
    Returns:
        Dictionary with:
        {
            'approved': Intervals safe to fetch now,
            'deferred': Intervals to fetch later,
            'reason': Explanation
        }
    """
    strategy = get_refresh_strategy(check_time)
    session = strategy['session']
    
    # Categorize intervals
    live_intervals = ['15minute', '30minute', '60minute']
    daily_intervals = ['day']
    weekly_monthly_intervals = ['week', 'month']
    
    approved = []
    deferred = []
    
    for interval in requested_intervals:
        if interval in live_intervals and strategy['fetch_live']:
            approved.append(interval)
        elif interval in daily_intervals and strategy['fetch_daily']:
            approved.append(interval)
        elif interval in weekly_monthly_intervals and strategy['fetch_historical']:
            approved.append(interval)
        else:
            deferred.append(interval)
    
    deferral_reason = ""
    if deferred:
        if session == MarketSession.LIVE:
            deferral_reason = f"⏳ Deferred {deferred}: Will be available after market close (3:30 PM)"
        elif session == MarketSession.WEEKEND:
            deferral_reason = f"⏳ Deferred {deferred}: Weekend - resume Monday 9:15 AM"
        elif session == MarketSession.HOLIDAY:
            deferral_reason = f"⏳ Deferred {deferred}: NSE Holiday - resume trading day"
        elif session == MarketSession.NIGHT:
            deferral_reason = f"⏳ Deferred {deferred}: Night session - intraday data only during market hours"
    
    return {
        'approved': approved,
        'deferred': deferred,
        'reason': deferral_reason,
        'session': session
    }


def get_last_market_close_time(check_time: datetime = None) -> datetime:
    """
    Get the datetime of the last NSE market close
    
    Returns IST datetime of last 3:30 PM market close
    """
    if check_time is None:
        check_time = datetime.now(IST)
    else:
        if check_time.tzinfo is None:
            check_time = IST.localize(check_time)
        else:
            check_time = check_time.astimezone(IST)
    
    # Start with today at 3:30 PM
    last_close = check_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # If we haven't reached 3:30 PM today yet, use yesterday's close
    if check_time < last_close:
        last_close -= timedelta(days=1)
    
    # If yesterday was weekend or holiday, go back further
    while last_close.weekday() >= 5 or last_close.date() in [h.date() for h in NSE_HOLIDAYS_2024_2025]:
        last_close -= timedelta(days=1)
    
    return last_close


def should_skip_intraday_fetch(check_time: datetime = None) -> bool:
    """
    Check if intraday fetch should be skipped
    
    Returns True if we're outside market hours or on weekend/holiday
    """
    session = get_market_session(check_time)
    return session != MarketSession.LIVE


def get_next_market_open_time(check_time: datetime = None) -> datetime:
    """
    Get datetime of next NSE market open (9:15 AM)
    
    Returns IST datetime of next 9:15 AM market open
    """
    if check_time is None:
        check_time = datetime.now(IST)
    else:
        if check_time.tzinfo is None:
            check_time = IST.localize(check_time)
        else:
            check_time = check_time.astimezone(IST)
    
    # Start with today at 9:15 AM
    next_open = check_time.replace(hour=9, minute=15, second=0, microsecond=0)
    
    # If we've already passed today's open, try tomorrow
    if check_time >= next_open:
        next_open += timedelta(days=1)
    
    # Skip weekends and holidays
    while next_open.weekday() >= 5 or next_open.date() in [h.date() for h in NSE_HOLIDAYS_2024_2025]:
        next_open += timedelta(days=1)
    
    return next_open


# ============================================================================
# REFRESH RECOMMENDATIONS
# ============================================================================

def get_refresh_recommendation(
    last_refresh_time: datetime = None,
    requested_intervals: List[str] = None
) -> Dict:
    """
    Get comprehensive refresh recommendation based on market status and last refresh
    
    Args:
        last_refresh_time: When data was last refreshed (IST datetime)
        requested_intervals: Intervals user wants to download
    
    Returns:
        Dictionary with refresh recommendation and timing
    """
    now = datetime.now(IST)
    strategy = get_refresh_strategy(now)
    
    if requested_intervals is None:
        requested_intervals = ['15minute', '30minute', '60minute', 'day', 'week', 'month']
    
    interval_filter = filter_intervals_for_session(requested_intervals, now)
    
    # Calculate time since last refresh
    time_since_refresh = "Never"
    if last_refresh_time:
        if last_refresh_time.tzinfo is None:
            last_refresh_time = IST.localize(last_refresh_time)
        else:
            last_refresh_time = last_refresh_time.astimezone(IST)
        
        delta = now - last_refresh_time
        if delta.total_seconds() < 60:
            time_since_refresh = f"{int(delta.total_seconds())} seconds ago"
        elif delta.total_seconds() < 3600:
            time_since_refresh = f"{int(delta.total_seconds() // 60)} minutes ago"
        elif delta.total_seconds() < 86400:
            time_since_refresh = f"{int(delta.total_seconds() // 3600)} hours ago"
        else:
            time_since_refresh = f"{int(delta.total_seconds() // 86400)} days ago"
    
    # Determine if refresh is needed
    should_refresh = len(interval_filter['approved']) > 0
    
    return {
        'session': strategy['session'],
        'should_refresh': should_refresh,
        'approved_intervals': interval_filter['approved'],
        'deferred_intervals': interval_filter['deferred'],
        'last_refresh': time_since_refresh,
        'strategy_reason': strategy['reason'],
        'deferral_reason': interval_filter['reason'],
        'next_market_open': get_next_market_open_time(now).isoformat() if strategy['session'] != MarketSession.LIVE else None,
        'last_market_close': get_last_market_close_time(now).isoformat()
    }
