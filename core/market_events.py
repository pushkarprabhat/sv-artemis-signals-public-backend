"""
core/market_events.py — MARKET EVENTS & ECONOMIC CALENDAR
==========================================================
Economic calendar and market events for informed trading decisions.
Includes RBI, Fed, ECB, NSE events and stock-specific earnings.
"""

import pandas as pd
from datetime import datetime, timedelta
from enum import Enum


class EventSeverity(Enum):
    """Event impact severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MarketEvents:
    """Market events and economic calendar"""
    
    # Central Bank Events
    CENTRAL_BANK_EVENTS = {
        'RBI': {
            'MPC_MONETARY_POLICY': {
                'frequency': 'Quarterly',
                'impact': EventSeverity.CRITICAL,
                'months': [2, 4, 6, 8, 10, 12],  # Bi-monthly
                'description': 'RBI Monetary Policy Committee decision',
                'affected_instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR', 'NIFTY50'],
            },
            'RBI_POLICY_RATE': {
                'frequency': 'Quarterly',
                'impact': EventSeverity.CRITICAL,
                'description': 'RBI Policy Rate announcement',
                'affected_instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR', 'NIFTYBANK'],
            },
            'CRR_SLR_CHANGE': {
                'frequency': 'Quarterly',
                'impact': EventSeverity.HIGH,
                'description': 'Cash Reserve Ratio and Statutory Liquidity Ratio changes',
                'affected_instruments': ['NIFTYBANK', 'SBIN', 'ICICIBANK'],
            },
        },
        'FED': {
            'FOMC_MEETING': {
                'frequency': 'Bi-monthly',
                'impact': EventSeverity.CRITICAL,
                'description': 'US Federal Reserve FOMC meeting',
                'affected_instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR', 'NIFTY50'],
            },
            'INTEREST_RATE_DECISION': {
                'frequency': 'Bi-monthly',
                'impact': EventSeverity.CRITICAL,
                'description': 'US Federal Funds Rate decision',
                'affected_instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR'],
            },
            'QE_DECISION': {
                'frequency': 'Quarterly',
                'impact': EventSeverity.CRITICAL,
                'description': 'Quantitative Easing announcement',
                'affected_instruments': ['USDINR', 'GOLD', 'NIFTY50'],
            },
        },
        'ECB': {
            'ECB_RATE_DECISION': {
                'frequency': 'Monthly',
                'impact': EventSeverity.CRITICAL,
                'description': 'European Central Bank interest rate decision',
                'affected_instruments': ['EURINR', 'USDINR', 'GBPINR'],
            },
            'ECB_MONETARY_POLICY': {
                'frequency': 'Monthly',
                'impact': EventSeverity.HIGH,
                'description': 'ECB Monetary Policy statement',
                'affected_instruments': ['EURINR', 'USDINR'],
            },
        },
    }
    
    # India Economic Events
    INDIA_ECONOMIC_EVENTS = {
        'GDP_DATA': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.HIGH,
            'months': [2, 5, 8, 11],
            'description': 'India GDP growth data release',
            'affected_instruments': ['NIFTY50', 'NIFTY100', 'USDINR', 'EURINR'],
        },
        'INFLATION_CPI': {
            'frequency': 'Monthly',
            'impact': EventSeverity.HIGH,
            'description': 'Consumer Price Index inflation data',
            'affected_instruments': ['NIFTY50', 'NIFTYBANK', 'USDINR'],
        },
        'INFLATION_WPI': {
            'frequency': 'Monthly',
            'impact': EventSeverity.MEDIUM,
            'description': 'Wholesale Price Index data',
            'affected_instruments': ['NIFTY50', 'NIFTY_METAL', 'CRUDEOIL'],
        },
        'INDUSTRIAL_PRODUCTION': {
            'frequency': 'Monthly',
            'impact': EventSeverity.MEDIUM,
            'description': 'India Industrial Production data',
            'affected_instruments': ['NIFTY50', 'NIFTY_IT', 'NIFTY_METAL'],
        },
        'CURRENT_ACCOUNT': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.HIGH,
            'description': 'Current Account Balance data',
            'affected_instruments': ['USDINR', 'EURINR', 'GBPINR'],
        },
        'FOREX_RESERVES': {
            'frequency': 'Weekly',
            'impact': EventSeverity.MEDIUM,
            'description': 'Foreign Exchange Reserves data',
            'affected_instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR'],
        },
    }
    
    # NSE Market Events
    NSE_MARKET_EVENTS = {
        'BUDGET_ANNOUNCEMENT': {
            'frequency': 'Annual',
            'impact': EventSeverity.CRITICAL,
            'month': 2,
            'day': 1,
            'description': 'Union Budget announcement',
            'affected_instruments': ['NIFTY50', 'NIFTY100', 'NIFTY500'],
        },
        'INTERIM_BUDGET': {
            'frequency': 'Annual',
            'impact': EventSeverity.HIGH,
            'month': 2,
            'description': 'Interim Budget (if applicable)',
            'affected_instruments': ['NIFTY50', 'NIFTY100'],
        },
        'BUDGET_SESSION_START': {
            'frequency': 'Annual',
            'impact': EventSeverity.MEDIUM,
            'month': 1,
            'description': 'Parliament Budget Session starts',
            'affected_instruments': ['NIFTY50'],
        },
        'RESULTS_SEASON_Q1': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.MEDIUM,
            'months': [4, 5, 6],
            'description': 'Q1 earnings announcement season',
            'affected_instruments': ['NIFTY50', 'NIFTY100', 'NIFTY500'],
        },
        'RESULTS_SEASON_Q2': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.MEDIUM,
            'months': [7, 8, 9],
            'description': 'Q2 earnings announcement season',
            'affected_instruments': ['NIFTY50', 'NIFTY100', 'NIFTY500'],
        },
        'RESULTS_SEASON_Q3': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.MEDIUM,
            'months': [10, 11, 12],
            'description': 'Q3 earnings announcement season',
            'affected_instruments': ['NIFTY50', 'NIFTY100', 'NIFTY500'],
        },
        'RESULTS_SEASON_Q4': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.MEDIUM,
            'months': [1, 2, 3],
            'description': 'Q4 earnings announcement season',
            'affected_instruments': ['NIFTY50', 'NIFTY100', 'NIFTY500'],
        },
        'INDEX_REBALANCE': {
            'frequency': 'Quarterly',
            'impact': EventSeverity.MEDIUM,
            'months': [3, 6, 9, 12],
            'day': 15,
            'description': 'Index rebalancing (NIFTY50, NIFTY500)',
            'affected_instruments': ['NIFTY50', 'NIFTY500'],
        },
        'DIVIDEND_SEASON': {
            'frequency': 'Annual',
            'impact': EventSeverity.LOW,
            'months': [3, 4, 5],
            'description': 'Dividend payment season',
            'affected_instruments': ['NIFTY50', 'NIFTY100'],
        },
    }
    
    # Stock-specific earnings events
    STOCK_EARNINGS_CALENDAR = {
        'RELIANCE': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
        'TCS': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
        'INFY': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
        'WIPRO': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.MEDIUM},
        'HCL': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.MEDIUM},
        'HDFC': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
        'ICICIBANK': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
        'AXISBANK': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.MEDIUM},
        'KOTAK': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
        'SBIN': {'quarter': 'Q4', 'typical_month': 1, 'impact': EventSeverity.HIGH},
    }
    
    @staticmethod
    def get_upcoming_events(days_ahead=30):
        """Get upcoming events for next N days, sorted by date"""
        events = []
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Add central bank events with realistic dates
        for bank, event_dict in MarketEvents.CENTRAL_BANK_EVENTS.items():
            for event_name, event_details in event_dict.items():
                # Calculate realistic future event dates
                if bank == 'RBI':
                    # RBI MPC meets on specific dates - next meeting typically 40-50 days out
                    event_date = today + timedelta(days=45)
                elif bank == 'FED':
                    # FOMC meets roughly every 6 weeks
                    event_date = today + timedelta(days=35)
                else:  # ECB
                    # ECB usually meets mid-month
                    event_date = today + timedelta(days=20)
                
                # Only add if within lookhead period
                if event_date <= end_date:
                    event_obj = {
                        'date': event_date,
                        'date_str': event_date.strftime('%a, %d %b %Y'),
                        'name': f"{bank} - {event_name.replace('_', ' ')}",
                        'type': 'Central Bank',
                        'impact': event_details['impact'].name,
                        'affected': event_details['affected_instruments'],
                        'description': event_details['description'],
                    }
                    events.append(event_obj)
        
        # Add India economic events with realistic dates
        for event_name, event_details in MarketEvents.INDIA_ECONOMIC_EVENTS.items():
            # GDP typically released on specific months
            if 'GDP' in event_name:
                event_date = today + timedelta(days=28)  # Usually month-end
            elif 'INFLATION' in event_name or 'CPI' in event_name:
                # CPI usually released mid-month
                event_date = today + timedelta(days=12)
            else:
                event_date = today + timedelta(days=15)
            
            if event_date <= end_date:
                event_obj = {
                    'date': event_date,
                    'date_str': event_date.strftime('%a, %d %b %Y'),
                    'name': event_name.replace('_', ' '),
                    'type': 'India Economy',
                    'impact': event_details['impact'].name,
                    'affected': event_details.get('affected_instruments', []),
                    'description': event_details['description'],
                }
                events.append(event_obj)
        
        # Add NSE events
        for event_name, event_details in MarketEvents.NSE_MARKET_EVENTS.items():
            # NSE holidays/events scattered throughout
            event_date = today + timedelta(days=22)
            
            if event_date <= end_date:
                event_obj = {
                    'date': event_date,
                    'date_str': event_date.strftime('%a, %d %b %Y'),
                    'name': event_name.replace('_', ' '),
                    'type': 'NSE Market',
                    'impact': event_details['impact'].name,
                    'affected': event_details.get('affected_instruments', []),
                    'description': event_details['description'],
                }
                events.append(event_obj)
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(events)
        if not df.empty and 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def get_events_for_instrument(instrument, days_ahead=30):
        """Get events affecting a specific instrument"""
        events = []
        all_events = MarketEvents.get_upcoming_events(days_ahead)
        
        # Filter events that affect this instrument
        for idx, event in all_events.iterrows():
            affected = event.get('affected', [])
            if instrument in affected or len(affected) == 0:
                events.append(event)
        
        return pd.DataFrame(events) if events else pd.DataFrame()
    
    @staticmethod
    def get_high_impact_events(days_ahead=7):
        """Get only high-impact events for next N days"""
        all_events = MarketEvents.get_upcoming_events(days_ahead)
        
        high_impact = all_events[all_events['impact'].isin(['HIGH', 'CRITICAL'])]
        return high_impact.sort_values('impact', ascending=False)
    
    @staticmethod
    def is_event_date(date, buffer_days=1):
        """Check if a date is near an event (within buffer_days)"""
        upcoming = MarketEvents.get_upcoming_events(buffer_days * 2)
        
        if upcoming.empty:
            return False, None
        
        # Check if any events are within buffer_days of the given date
        for idx, event in upcoming.iterrows():
            if event['date'] and abs((event['date'] - date).days) <= buffer_days:
                return True, event['name']
        
        return False, None


class StockEarningsCalendar:
    """Stock-specific earnings and events calendar"""
    
    @staticmethod
    def get_stock_earnings(stock_symbol):
        """Get earnings details for a stock"""
        calendar = MarketEvents.STOCK_EARNINGS_CALENDAR
        
        if stock_symbol in calendar:
            return calendar[stock_symbol]
        return None
    
    @staticmethod
    def get_stocks_with_earnings(month):
        """Get stocks with earnings in a specific month"""
        stocks_this_month = []
        
        for stock, details in MarketEvents.STOCK_EARNINGS_CALENDAR.items():
            if details['typical_month'] == month:
                stocks_this_month.append({
                    'stock': stock,
                    'quarter': details['quarter'],
                    'impact': details['impact'].name,
                })
        
        return pd.DataFrame(stocks_this_month) if stocks_this_month else pd.DataFrame()
    
    @staticmethod
    def is_earnings_period(stock_symbol, date, days_before=3, days_after=3):
        """
        Check if a date is within earnings announcement period.
        
        Args:
            stock_symbol: Stock symbol
            date: Date to check
            days_before: Days before typical earnings to consider
            days_after: Days after typical earnings to consider
        """
        earnings = StockEarningsCalendar.get_stock_earnings(stock_symbol)
        
        if not earnings:
            return False
        
        typical_month = earnings['typical_month']
        typical_day = 15  # Approximate mid-month
        
        # Check if date is in earnings month with buffer
        current_month = date.month
        days_until_earnings = 30 * (typical_month - current_month)
        
        if abs(days_until_earnings) <= (days_before + days_after):
            return True
        
        return False


class VolatilityCalendar:
    """Calculate expected volatility around events"""
    
    # Expected volatility spike multipliers
    VOLATILITY_MULTIPLIERS = {
        EventSeverity.CRITICAL: 2.5,  # 2.5x normal volatility
        EventSeverity.HIGH: 1.8,       # 1.8x normal volatility
        EventSeverity.MEDIUM: 1.3,     # 1.3x normal volatility
        EventSeverity.LOW: 1.1,        # 1.1x normal volatility
    }
    
    @staticmethod
    def get_expected_volatility(instrument, base_volatility, days_ahead=7):
        """
        Calculate expected volatility considering upcoming events.
        
        Args:
            instrument: Symbol to check
            base_volatility: Normal volatility (e.g., from GARCH model)
            days_ahead: Days to look ahead for events
        
        Returns:
            Adjusted volatility multiplier
        """
        events = MarketEvents.get_events_for_instrument(instrument, days_ahead)
        
        if events.empty:
            return 1.0  # No events, normal volatility
        
        # Get the highest severity event
        severity_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        max_severity_value = max([severity_map.get(e, 0) for e in events['impact']])
        
        # Map back to EventSeverity
        for severity in EventSeverity:
            if severity.value == max_severity_value:
                return VolatilityCalendar.VOLATILITY_MULTIPLIERS[severity]
        
        return 1.0
