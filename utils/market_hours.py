"""
utils/market_hours.py - NSE market hours and holiday utilities
"""
from datetime import datetime, time, timedelta
import pytz

# India timezone
IST = pytz.timezone('Asia/Kolkata')

# NSE Market Hours (Monday to Friday)
MARKET_OPEN = time(9, 15)  # 09:15 AM
MARKET_CLOSE = time(15, 30)  # 03:30 PM

# NSE Market Holidays 2025-2026 (Comprehensive List)
NSE_HOLIDAYS_2024_2025 = [
    # 2025 Holidays
    datetime(2025, 1, 26),   # Republic Day (Sunday)
    datetime(2025, 3, 8),    # Maha Shivaratri
    datetime(2025, 3, 10),   # Holi
    datetime(2025, 3, 25),   # Ramzan Id
    datetime(2025, 4, 11),   # Good Friday
    datetime(2025, 4, 17),   # Ram Navami
    datetime(2025, 4, 21),   # Mahavir Jayanti
    datetime(2025, 5, 1),    # May Day
    datetime(2025, 8, 15),   # Independence Day (Friday)
    datetime(2025, 8, 27),   # Janmashtami
    datetime(2025, 9, 16),   # Milad-un-Nabi (Eid-ul-Fitr)
    datetime(2025, 10, 2),   # Gandhi Jayanti (Thursday)
    datetime(2025, 10, 20),  # Dussehra
    datetime(2025, 10, 30),  # Diwali (Thursday)
    datetime(2025, 10, 31),  # Diwali - Day After (Friday)
    datetime(2025, 11, 5),   # Diwali - Govardhan Puja
    datetime(2025, 11, 15),  # Guru Nanak Jayanti (Saturday)
    datetime(2025, 12, 25),  # Christmas (Thursday)
    
    # 2026 Holidays
    datetime(2026, 1, 26),   # Republic Day (Monday)
    datetime(2026, 2, 28),   # Maha Shivaratri (Saturday)
    datetime(2026, 3, 25),   # Holi
    datetime(2026, 4, 2),    # Good Friday
    datetime(2026, 4, 14),   # Ramzan Id (Eid-ul-Fitr)
    datetime(2026, 4, 17),   # Ram Navami (Friday)
    datetime(2026, 4, 21),   # Mahavir Jayanti (Tuesday)
    datetime(2026, 5, 1),    # May Day (Friday)
    datetime(2026, 8, 15),   # Independence Day (Saturday)
    datetime(2026, 8, 31),   # Janmashtami (Monday)
    datetime(2026, 9, 4),    # Milad-un-Nabi (Friday)
    datetime(2026, 10, 2),   # Gandhi Jayanti (Friday)
    datetime(2026, 10, 12),  # Dussehra (Monday)
    datetime(2026, 10, 19),  # Diwali (Monday)
    datetime(2026, 10, 20),  # Diwali - Day After (Tuesday)
    datetime(2026, 10, 25),  # Diwali - Govardhan Puja (Sunday)
    datetime(2026, 11, 7),   # Guru Nanak Jayanti (Saturday)
    datetime(2026, 11, 30),  # Ramzan Id (Monday)
    datetime(2026, 12, 25),  # Christmas (Friday)
]


def is_market_open(check_time: datetime = None) -> bool:
    """
    Check if NSE market is currently open
    
    Args:
        check_time: datetime object to check. Default is current time in IST
    
    Returns:
        Boolean indicating if market is open
    """
    if check_time is None:
        check_time = datetime.now(IST)
    else:
        # Convert to IST if needed
        if check_time.tzinfo is None:
            check_time = IST.localize(check_time)
        else:
            check_time = check_time.astimezone(IST)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if check_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's a holiday
    check_date = check_time.date()
    if check_date in [h.date() for h in NSE_HOLIDAYS_2024_2025]:
        return False
    
    # Check if time is within market hours
    current_time = check_time.time()
    if MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return True
    
    return False


def get_market_status() -> dict:
    """
    Get detailed market status
    
    Returns:
        Dictionary with market info:
        {
            'is_open': bool,
            'opens_at': time,
            'closes_at': time,
            'time_to_open': timedelta or None,
            'time_to_close': timedelta or None,
            'day_name': str,
            'is_holiday': bool,
            'is_weekend': bool
        }
    """
    now = datetime.now(IST)
    current_time = now.time()
    current_date = now.date()
    
    day_name = now.strftime("%A")
    is_weekend = now.weekday() >= 5
    is_holiday = current_date in [h.date() for h in NSE_HOLIDAYS_2024_2025]
    is_open = is_market_open(now)
    
    # Calculate time to open/close
    time_to_open = None
    time_to_close = None
    
    if is_open:
        # Market is open, calculate time to close
        close_dt = datetime.combine(current_date, MARKET_CLOSE, tzinfo=IST)
        time_to_close = close_dt - now
    elif current_time < MARKET_OPEN:
        # Before market open
        open_dt = datetime.combine(current_date, MARKET_OPEN, tzinfo=IST)
        time_to_open = open_dt - now
    else:
        # After market close, calculate time to next day open
        next_date = current_date + timedelta(days=1)
        # Skip weekends and holidays
        while next_date.weekday() >= 5 or next_date in [h.date() for h in NSE_HOLIDAYS_2024_2025]:
            next_date += timedelta(days=1)
        open_dt = datetime.combine(next_date, MARKET_OPEN, tzinfo=IST)
        time_to_open = open_dt - now
    
    return {
        'is_open': is_open,
        'opens_at': MARKET_OPEN,
        'closes_at': MARKET_CLOSE,
        'time_to_open': time_to_open,
        'time_to_close': time_to_close,
        'day_name': day_name,
        'is_holiday': is_holiday,
        'is_weekend': is_weekend,
        'current_time': now.time(),
        'current_date': current_date
    }


def get_holidays_list(year: int = None) -> list:
    """
    Get list of market holidays
    
    Args:
        year: Specific year to filter (optional)
    
    Returns:
        List of datetime objects
    """
    if year:
        return [h for h in NSE_HOLIDAYS_2024_2025 if h.year == year]
    return NSE_HOLIDAYS_2024_2025


def format_time_delta(td: timedelta) -> str:
    """
    Format timedelta to readable string
    
    Args:
        td: timedelta object
    
    Returns:
        Formatted string like "2h 30m" or "45m"
    """
    if td is None:
        return "N/A"
    
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def display_market_status_widget():
    """
    Display market status widget in Streamlit
    Requires: import streamlit as st
    """
    import streamlit as st
    
    status = get_market_status()
    
    if status['is_open']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Market", "OPEN", delta=format_time_delta(status['time_to_close']) + " to close")
        with col2:
            st.metric("⏰ Current Time", status['current_time'].strftime("%H:%M:%S"))
        with col3:
            st.metric("📅 Day", status['day_name'])
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if status['is_holiday']:
                st.metric("🔴 Market", "HOLIDAY")
            elif status['is_weekend']:
                st.metric("🔴 Market", "WEEKEND")
            else:
                st.metric("🔴 Market", "CLOSED")
        with col2:
            if status['time_to_open']:
                st.metric("⏰ Opens In", format_time_delta(status['time_to_open']))
            else:
                st.metric("⏰ Opens At", status['opens_at'].strftime("%H:%M"))
        with col3:
            st.metric("📅 Day", status['day_name'])
