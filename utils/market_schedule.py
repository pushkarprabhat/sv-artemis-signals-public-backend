"""
Market Schedule Management Module
File: utils/market_schedule.py

Manages:
1. NSE Market hours (9:15 AM - 3:30 PM IST)
2. BOD (Beginning of Day) scheduling (7:00 AM - before market open)
3. EOD (End of Day) scheduling (5:50 PM - after market close)
4. Background data refresh scheduling
5. Smart sleep logic during market closed

Key Features:
- Market hours checking with timezone support
- BOD/EOD "due now" detection
- Smart background refresh that sleeps during market closed if data complete
- Time calculations for next market events

Note: All time constants imported from config.py (single source of truth)
"""

from datetime import datetime, time, timedelta
import pytz
from typing import Dict, Optional
import logging

# Import time constants from config (SINGLE SOURCE OF TRUTH)
from config import (
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    BOD_IDEAL_HOUR, BOD_IDEAL_MINUTE,
    BOD_WINDOW_START_HOUR, BOD_WINDOW_END_HOUR,
    EOD_IDEAL_HOUR, EOD_IDEAL_MINUTE,
    EOD_WINDOW_START_HOUR, EOD_WINDOW_START_MINUTE,
    EOD_WINDOW_END_HOUR, EOD_WINDOW_END_MINUTE,
    TIMEZONE
)

logger = logging.getLogger(__name__)

# IST timezone
IST = pytz.timezone(TIMEZONE)

# Convert config integers to time objects (for internal use)
MARKET_OPEN_TIME = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
MARKET_CLOSE_TIME = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)

# BOD (Beginning of Day) Process
BOD_IDEAL_TIME = time(BOD_IDEAL_HOUR, BOD_IDEAL_MINUTE)
BOD_WINDOW_START = time(BOD_WINDOW_START_HOUR, 0)
BOD_WINDOW_END = time(BOD_WINDOW_END_HOUR, 0)

# EOD (End of Day) Process
EOD_IDEAL_TIME = time(EOD_IDEAL_HOUR, EOD_IDEAL_MINUTE)
EOD_WINDOW_START = time(EOD_WINDOW_START_HOUR, EOD_WINDOW_START_MINUTE)
EOD_WINDOW_END = time(EOD_WINDOW_END_HOUR, EOD_WINDOW_END_MINUTE)


class MarketSchedule:
    """Manage market hours and BOD/EOD scheduling"""
    
    @staticmethod
    def get_ist_now() -> datetime:
        """Get current time in IST"""
        return datetime.now(IST)
    
    @staticmethod
    def is_market_open(check_time: Optional[datetime] = None) -> bool:
        """
        Check if NSE market is currently open.
        
        Args:
            check_time: DateTime to check (defaults to now)
        
        Returns:
            True if market is open, False otherwise
        """
        if check_time is None:
            check_time = MarketSchedule.get_ist_now()
        
        # Convert to IST if needed
        if check_time.tzinfo is None:
            check_time = IST.localize(check_time)
        else:
            check_time = check_time.astimezone(IST)
        
        # Check if weekday (Monday=0, Sunday=6)
        if check_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check time range
        check_time_only = check_time.time()
        return MARKET_OPEN_TIME <= check_time_only <= MARKET_CLOSE_TIME
    
    @staticmethod
    def get_market_status() -> Dict:
        """
        Get current market status.
        
        Returns:
            Dict with market status information
        """
        now = MarketSchedule.get_ist_now()
        is_open = MarketSchedule.is_market_open(now)
        
        return {
            'is_open': is_open,
            'status': 'OPEN' if is_open else 'CLOSED',
            'current_time': now,
            'market_open_time': MARKET_OPEN_TIME,
            'market_close_time': MARKET_CLOSE_TIME,
            'next_market_open': MarketSchedule.get_next_market_open(now),
            'next_market_close': MarketSchedule.get_next_market_close(now) if is_open else None,
            'is_weekday': now.weekday() < 5
        }
    
    @staticmethod
    def get_next_market_open(from_time: Optional[datetime] = None) -> datetime:
        """Get next market opening time"""
        if from_time is None:
            from_time = MarketSchedule.get_ist_now()
        
        if from_time.tzinfo is None:
            from_time = IST.localize(from_time)
        else:
            from_time = from_time.astimezone(IST)
        
        current_time_only = from_time.time()
        current_date = from_time.date()
        
        # If before market open today, market opens today
        if current_time_only < MARKET_OPEN_TIME and from_time.weekday() < 5:
            return IST.localize(datetime.combine(current_date, MARKET_OPEN_TIME))
        
        # Find next weekday
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # Skip weekends
            next_date += timedelta(days=1)
        
        return IST.localize(datetime.combine(next_date, MARKET_OPEN_TIME))
    
    @staticmethod
    def get_next_market_close(from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Get next market closing time (today if market still open)"""
        if from_time is None:
            from_time = MarketSchedule.get_ist_now()
        
        if from_time.tzinfo is None:
            from_time = IST.localize(from_time)
        else:
            from_time = from_time.astimezone(IST)
        
        current_time_only = from_time.time()
        current_date = from_time.date()
        
        # If before or during market hours, market closes today
        if current_time_only <= MARKET_CLOSE_TIME and from_time.weekday() < 5:
            return IST.localize(datetime.combine(current_date, MARKET_CLOSE_TIME))
        
        # Otherwise, market closes next trading day
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        
        return IST.localize(datetime.combine(next_date, MARKET_CLOSE_TIME))
    
    @staticmethod
    def time_to_market_open() -> int:
        """Calculate seconds until market opens"""
        now = MarketSchedule.get_ist_now()
        next_open = MarketSchedule.get_next_market_open(now)
        return max(0, int((next_open - now).total_seconds()))
    
    @staticmethod
    def time_to_market_close() -> int:
        """Calculate seconds until market closes (0 if market closed)"""
        now = MarketSchedule.get_ist_now()
        if not MarketSchedule.is_market_open(now):
            return 0
        next_close = MarketSchedule.get_next_market_close(now)
        return max(0, int((next_close - now).total_seconds()))

    
    @staticmethod
    def get_bod_status() -> Dict:
        """Get BOD process status and schedule"""
        now = MarketSchedule.get_ist_now()
        now_time = now.time()
        
        # Check if currently in BOD window
        is_in_window = BOD_WINDOW_START <= now_time <= BOD_WINDOW_END
        
        # Check if BOD is due (if in window)
        is_due = is_in_window
        
        # Calculate time until due
        if is_due:
            time_until_due = 0
        else:
            today_bod = datetime.combine(now.date(), BOD_WINDOW_START)
            if now > IST.localize(today_bod):
                # Window passed, calculate for tomorrow
                tomorrow_bod = today_bod + timedelta(days=1)
                time_until_due = int((IST.localize(tomorrow_bod) - now).total_seconds())
            else:
                time_until_due = int((IST.localize(today_bod) - now).total_seconds())
        
        return {
            'is_due': is_due,
            'ideal_time': BOD_IDEAL_TIME.strftime('%H:%M'),
            'window_start': BOD_WINDOW_START.strftime('%H:%M'),
            'window_end': BOD_WINDOW_END.strftime('%H:%M'),
            'description': 'Prepare daily data, check instruments, download historical data',
            'time_until_due': time_until_due,
            'in_window': is_in_window
        }
    
    @staticmethod
    def get_eod_status() -> Dict:
        """Get EOD process status and schedule"""
        now = MarketSchedule.get_ist_now()
        now_time = now.time()
        
        # Check if currently in EOD window
        is_in_window = EOD_WINDOW_START <= now_time <= EOD_WINDOW_END
        
        # Check if EOD is due (if in window)
        is_due = is_in_window
        
        # Calculate time until due
        if is_due:
            time_until_due = 0
        else:
            today_eod = datetime.combine(now.date(), EOD_WINDOW_START)
            if now > IST.localize(today_eod):
                # Window passed, calculate for tomorrow
                tomorrow_eod = today_eod + timedelta(days=1)
                time_until_due = int((IST.localize(tomorrow_eod) - now).total_seconds())
            else:
                time_until_due = int((IST.localize(today_eod) - now).total_seconds())
        
        return {
            'is_due': is_due,
            'ideal_time': EOD_IDEAL_TIME.strftime('%H:%M'),
            'window_start': EOD_WINDOW_START.strftime('%H:%M'),
            'window_end': EOD_WINDOW_END.strftime('%H:%M'),
            'description': 'Compile end-of-day data, calculate metrics, generate reports',
            'time_until_due': time_until_due,
            'in_window': is_in_window
        }

    
    @staticmethod
    def should_refresh_market_data(data_complete: bool = False) -> bool:
        """
        Determine if background should refresh market data.
        
        Smart logic:
        - If market OPEN: Always refresh (need live data)
        - If market CLOSED + data_complete=True: Don't refresh (sleep)
        - If market CLOSED + data_complete=False: Refresh (catch up)
        
        Args:
            data_complete: Whether all required data has been downloaded
        
        Returns:
            True if refresh should run, False if should sleep
        """
        if MarketSchedule.is_market_open():
            # Market open - always refresh
            return True
        
        if data_complete:
            # Market closed AND data complete - can sleep
            return False
        
        # Market closed BUT data incomplete - keep refreshing
        return True
    
    @staticmethod
    def get_background_refresh_status(data_complete: bool = False) -> Dict:
        """
        Get background refresh status.
        
        Returns:
            Dict with refresh status and recommendation
        """
        should_run = MarketSchedule.should_refresh_market_data(data_complete)
        is_open = MarketSchedule.is_market_open()
        now = MarketSchedule.get_ist_now()
        
        if is_open:
            next_event = MarketSchedule.get_next_market_close(now)
            time_to_event = int((next_event - now).total_seconds()) if next_event else 0
            reason = "📈 Market is OPEN - Refreshing live data"
            next_event_name = "MARKET_CLOSE"
        else:
            next_event = MarketSchedule.get_next_market_open(now)
            time_to_event = int((next_event - now).total_seconds())
            if data_complete:
                reason = "😴 Market CLOSED & data complete - Background refresh SLEEPING"
                next_event_name = "MARKET_OPEN"
            else:
                reason = "🔄 Market CLOSED but data incomplete - Refreshing to catch up"
                next_event_name = "MARKET_OPEN"
        
        return {
            'should_run': should_run,
            'is_sleeping': not should_run and not is_open,
            'reason': reason,
            'market_status': 'OPEN' if is_open else 'CLOSED',
            'data_complete': data_complete,
            'time_to_next_market_event': time_to_event,
            'next_event': next_event_name,
            'refresh_interval': 60 if is_open else (300 if not data_complete else 0)
        }



# Convenience functions for UI integration

def is_market_open() -> bool:
    """Check if market is open (convenience function)"""
    return MarketSchedule.is_market_open()


def get_bod_status() -> Dict:
    """Get BOD status (convenience function)"""
    return MarketSchedule.get_bod_status()


def get_eod_status() -> Dict:
    """Get EOD status (convenience function)"""
    return MarketSchedule.get_eod_status()


def should_refresh_market_data(data_complete: bool = False) -> bool:
    """Check if background refresh should run"""
    return MarketSchedule.should_refresh_market_data(data_complete)


def get_background_refresh_status(data_complete: bool = False) -> Dict:
    """Get background refresh status (convenience function)"""
    return MarketSchedule.get_background_refresh_status(data_complete)


def get_market_status() -> Dict:
    """Get market status (convenience function)"""
    return MarketSchedule.get_market_status()


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("MARKET SCHEDULE STATUS")
    print("="*80)
    
    # Get market status
    market = get_market_status()
    print(f"\nMarket Status: {market['status']}")
    print(f"Current Time: {market['current_time'].strftime('%H:%M:%S')}")
    print(f"Market Hours: {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')}")
    
    # Get BOD/EOD status
    bod = get_bod_status()
    eod = get_eod_status()
    
    print(f"\nBOD Status:")
    print(f"  Due Now: {'✅ YES' if bod['is_due'] else '❌ No'}")
    print(f"  Window: {bod['window_start']} - {bod['window_end']}")
    print(f"  Description: {bod['description']}")
    
    print(f"\nEOD Status:")
    print(f"  Due Now: {'✅ YES' if eod['is_due'] else '❌ No'}")
    print(f"  Window: {eod['window_start']} - {eod['window_end']}")
    print(f"  Description: {eod['description']}")
    
    # Get refresh status
    refresh_complete = get_background_refresh_status(data_complete=True)
    refresh_incomplete = get_background_refresh_status(data_complete=False)
    
    print(f"\nBackground Refresh (Data Complete):")
    print(f"  Should Run: {'✅ Yes' if refresh_complete['should_run'] else '❌ No'}")
    print(f"  Status: {refresh_complete['reason']}")
    
    print(f"\nBackground Refresh (Data Incomplete):")
    print(f"  Should Run: {'✅ Yes' if refresh_incomplete['should_run'] else '❌ No'}")
    print(f"  Status: {refresh_incomplete['reason']}")
    
    print("\n" + "="*80)
