from datetime import time, datetime
from pathlib import Path
import os

# Import common constants from parent config
from . import BASE_DIR, LOG_DIR

# Time constants (IST)
BOD_TIME = time(7, 0)      # 07:00 AM IST
EOD_TIME = time(17, 30)    # 05:30 PM IST (Market close + buffer)

# Intervals
DAILY_INTERVAL = 'day'
INTRADAY_INTERVALS = ['5minute', '15minute', '30minute', '60minute']

# Data Directories mapping
DATA_DIRS = {
    'day': BASE_DIR / 'NSE' / 'day',
    '5minute': BASE_DIR / 'NSE' / '5minute',
    '15minute': BASE_DIR / 'NSE' / '15minute',
    '30minute': BASE_DIR / 'NSE' / '30minute',
    '60minute': BASE_DIR / 'NSE' / '60minute',
}

# Ensure directories exist
for d in DATA_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Performance & Reliability
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

# Debug & Test Modes
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
TEST_MODE = os.getenv('TEST_MODE', 'False').lower() == 'true'

def is_trading_day(dt: datetime = None) -> bool:
    """Check if given date is a trading day (not weekend or holiday)"""
    if dt is None:
        dt = datetime.now()
    
    # Weekends (Saturday=5, Sunday=6)
    if dt.weekday() >= 5:
        return False
        
    # Holidays
    try:
        from utils.market_hours import NSE_HOLIDAYS_2024_2025
        if dt.date() in [h.date() for h in NSE_HOLIDAYS_2024_2025]:
            return False
    except ImportError:
        pass
        
    return True
