# config.py — FINAL COMPLETE VERSION (2025) — ALL VARIABLES INCLUDED
from pathlib import Path
import os

# =============================================================================
# PROJECT BRANDING & MODE
# =============================================================================
PROJECT_NAME = "Artemis Signals by ManekBaba"
COMMERCIAL_MODE = os.getenv('COMMERCIAL_MODE', 'True').lower() == 'true'

# Conditional content based on mode
# Hide family content in commercial mode
if not COMMERCIAL_MODE:
    # 💙 For my sons Shivaansh & Krishaansh:
    # This system will generate ₹40L/year for your education
    # Every algorithm, every optimization, every line of code is my love for you
    PROJECT_TAGLINE = "Built with Papa's love — Every trade is for your dreams"
    PROJECT_TAGLINE2 = "For Shivaansh & Krishaansh — We Will Pay Every Rupee"
else:
    PROJECT_TAGLINE = "Institutional-Grade Algorithmic Trading Platform"
    PROJECT_TAGLINE2 = "Institutional-Grade Algorithmic Trading Platform"

# Hide family/personal data in commercial mode
SHOW_FAMILY_DASHBOARD = not COMMERCIAL_MODE
SHOW_DEBT_TRACKER = not COMMERCIAL_MODE
SHOW_ADMIN_MODE = not COMMERCIAL_MODE

# =============================================================================
# PATHS (Must be defined BEFORE any core imports to avoid circular deps)
# =============================================================================
BASE_DIR = Path("marketdata")  # Stock ticker OHLCV data
UNIVERSE_DIR = Path("universe")
SYMBOLS_DIR = UNIVERSE_DIR / "symbols"
INSTRUMENTS_DIR = UNIVERSE_DIR / "instruments"
METADATA_DIR = UNIVERSE_DIR / "metadata"
ENRICHED_DIR = UNIVERSE_DIR / "enriched"  # Enriched instruments data

OPTION_CHAINS_DIR = BASE_DIR / "option_chains"
OPTION_IV_HISTORY_DIR = BASE_DIR / "iv_history"
LOG_DIR = Path("logs")

# Bhavcopy paths - Organized by market type
BHAVCOPY_DIR = BASE_DIR / "bhavcopy"
BHAVCOPY_EQUITY_DIR = BHAVCOPY_DIR / "equity"
BHAVCOPY_FO_DIR = BHAVCOPY_DIR / "fo"

for p in [BASE_DIR, UNIVERSE_DIR, SYMBOLS_DIR, INSTRUMENTS_DIR, METADATA_DIR, ENRICHED_DIR, OPTION_CHAINS_DIR, OPTION_IV_HISTORY_DIR, LOG_DIR, BHAVCOPY_DIR, BHAVCOPY_EQUITY_DIR, BHAVCOPY_FO_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MARKET SETTINGS
# =============================================================================
# All supported timeframes (5min is source, others are aggregated or downloaded from API)
# STRATEGY: Download 5-minute and daily directly from API
# Aggregate from 5-minute: 15min, 30min, 60min
# Aggregate from daily: weekly, monthly, quarterly, annual
# OPTIONS DATA: Live refresh during trading hours (no aggregation)
TIMEFRAMES = ["5minute", "15minute", "30minute", "60minute", "day", "week", "month", "quarter", "year"]
UNIFIED_TIMEFRAMES = ["15minute", "30minute", "60minute", "day"]  # All scans use these timeframes
AGGREGATED_INTERVALS = [15, 30, 60]  # Aggregate these from 5min data
DAILY_AGGREGATED_INTERVALS = ["week", "month", "quarter", "year"]  # Aggregate these from daily data

# =============================================================================
# DOWNLOAD STRATEGY (NEW: Instrument Type Based)
# =============================================================================
# Strategy Tiers:
#   1. INTERVAL DATA (5min + Daily): FUT, EQ, Indices
#      - Downloaded in batches
#      - Aggregated to higher timeframes (15min, 30min, 60min, week, month, year)
#      - Live-refreshed every 5 minutes
#      - Max staleness: 5 minutes
#      - Use case: Scanner, position-taking, backtesting
#
#   2. EOD DERIVATIVES (CE, PE): 1 candle/day
#      - Downloaded daily at market close (3:30 PM IST)
#      - For strategy reference, Greeks calculation
#      - No aggregation
#      - Use case: Portfolio Greeks, IV analysis, strategy validation
#
#   3. LIVE FEED (Everything): WebSocket ticks
#      - Real-time LTP, bid/ask, volume for all 119k+ instruments
#      - Used for current price in position-taking
#      - Greeks calculation from live ticks
#
# Instrument Type Classification:
#   FUT  = Futures (index futures, stock futures)
#   EQ   = Equities (stocks from NSE/BSE)
#   CE   = Call Options (live-only, no historical download)
#   PE   = Put Options (live-only, no historical download)
#   
# Indices: Treated as EQ type, downloaded from nse_indices_list.csv

# Instruments to download interval data for
DOWNLOADABLE_INSTRUMENT_TYPES = ['FUT', 'EQ']  # Futures + Equities + Indices (EQ includes indices)

# Derivatives downloaded EOD only
DERIVATIVES_EOD_TYPES = ['CE', 'PE']  # Call & Put Options - EOD only

# Live-only instruments (excluded from interval downloads)
EXCLUDED_INSTRUMENT_TYPES = ['CE', 'PE']  # Call Options, Put Options (live-only, EOD reference only)

# DEPRECATED: Use DOWNLOADABLE_INSTRUMENT_TYPES instead
# LIVE_ONLY_SEGMENTS = ['BFO-FUT', 'BFO-OPT', 'MCX', 'NCDEX']  # OLD - replaced by type-based filtering

# ============================================================================
# Batch size limits for Kite API historical data requests
# ============================================================================
# Kite API has different maximum date ranges per interval to avoid
# "interval exceeds max limit" errors. These batch sizes split large
# date ranges into multiple API calls automatically.
#
# DOWNLOAD & AGGREGATION STRATEGY:
#   DIRECT API DOWNLOADS (2 intervals only):
#     - 5-minute data (100-day batches) → stored as-is
#     - Daily data (1000-day batches) → stored as-is
#   
#   AGGREGATED FROM 5-MINUTE (3 intervals from inline aggregation):
#     - 15-minute, 30-minute, 60-minute → auto-aggregated after 5-min download
#   
#   AGGREGATED FROM DAILY (4 intervals from daily OHLC):
#     - Weekly, monthly, quarterly, annual → aggregated from daily bars
#   
# TOTAL TIMEFRAME COVERAGE: 9 intervals
#   Downloaded: 2 (5min, day)
#   Aggregated from 5min: 3 (15min, 30min, 60min)
#   Aggregated from daily: 4 (week, month, quarter, year)
#
# OPTIONS DATA STRATEGY:
#   - Stored separately in NFO/ directory
#   - NO historical aggregation (options are live instruments)
#   - Refreshed during trading hours using live ticks
#   - Greeks calculated using live market prices
#   - Entry/exit decisions based on current market conditions

BATCH_SIZES = {
    '5minute': 100,      # Intraday: Token size 100 days per batch (Kite API limit)
    '15minute': 365,     # Aggregated from 5min data (no direct download)
    '30minute': 365,     # Aggregated from 5min data (no direct download)
    '60minute': 365,     # Direct from API if available, else aggregated
    'day': 1000,         # Daily bars: Token size 1000 days per batch
    'week': 2000,        # Weekly bars: Max 2000 days per batch (~5.5 years)
    'month': 2000,       # Monthly bars: Max 2000 days per batch (~5.5 years)
}

# ============================================================================
# DOWNLOAD API BATCH CONFIGURATION
# ============================================================================
# Configurable batch sizes for API requests to optimize download speed
# while respecting API rate limits and connection constraints.

# Number of instruments to request OHLCV data for per API request
# Kite API can handle multiple instruments in a single request
# Larger batches = fewer requests but larger payload
# Default: 500 instruments per request (configurable)
OHLCV_INSTRUMENTS_PER_REQUEST = 500

# Number of LTP (Last Traded Price) values to fetch per API request
# Kite API can return LTP for multiple instruments in a single request
# Default: 1000 instruments per request (configurable)
LTP_INSTRUMENTS_PER_REQUEST = 1000

# Rate limiting between requests (in seconds)
# Prevents API throttling and connection issues
REQUEST_DELAY_SECONDS = 0.1

# Maximum retries for failed API requests
MAX_RETRIES = 3

# Timeout for each API request (in seconds)
API_REQUEST_TIMEOUT = 30

# All Indices with their types and segments
# NSE Indices (from nse_indices_list.csv) - Major F&O Indices
NSE_INDICES = {
    # F&O Indices (Broad Market)
    'NIFTY': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty 50', 'index_type': 'F&O'},
    'BANKNIFTY': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Bank', 'index_type': 'F&O'},
    'FINNIFTY': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Financial Services', 'index_type': 'F&O'},
    'MIDCPNIFTY': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Midcap Select', 'index_type': 'F&O'},
    'NIFTYNXT50': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Next 50', 'index_type': 'F&O'},
    
    # Sectoral Indices
    'AUTO': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Auto', 'index_type': 'Sectoral'},
    'CONSUMPTION': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Consumer Durables', 'index_type': 'Sectoral'},
    'ENERGY': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Energy', 'index_type': 'Sectoral'},
    'FMCG': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty FMCG', 'index_type': 'Sectoral'},
    'HEALTHCARE': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Healthcare', 'index_type': 'Sectoral'},
    'IT': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty IT', 'index_type': 'Sectoral'},
    'MEDIA': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Media', 'index_type': 'Sectoral'},
    'METAL': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Metal', 'index_type': 'Sectoral'},
    'PHARMA': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Pharma', 'index_type': 'Sectoral'},
    'PRIVATEBANK': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Private Bank', 'index_type': 'Sectoral'},
    'PSU': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty PSU Bank', 'index_type': 'Sectoral'},
    'REALTY': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Realty', 'index_type': 'Sectoral'},
    'FINANCIAL': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Financial Services Ex Bank', 'index_type': 'Sectoral'},
    'PSE': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty PSE', 'index_type': 'Sectoral'},
    'CPSE': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty CPSE', 'index_type': 'Sectoral'},
    'INFRA': {'type': 'index', 'segment': 'nse', 'full_name': 'Nifty Infrastructure', 'index_type': 'Sectoral'},
}

# Additional Indices (Broad Market, Volatility, Commodities, Forex)
INDICES = {
    # Broad Market Indices
    'NIFTY50': {'type': 'index', 'segment': 'cash', 'symbol': '^NSEI'},
    'NIFTY100': {'type': 'index', 'segment': 'cash', 'symbol': '^NSEINDEX'},
    'NIFTY500': {'type': 'index', 'segment': 'cash', 'symbol': '^NIFTYINDEX'},
    
    # Volatility Index
    'INDIA_VIX': {'type': 'index', 'segment': 'cash', 'symbol': '^INDIAVIX'},
    
    # Commodity Indices (Precious Metals, Oil & Gas)
    'GOLD': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'GOLDPETAL', 'description': 'Gold Futures'},
    'SILVER': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'SILVERPETAL', 'description': 'Silver Futures'},
    'CRUDEOIL': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'CRUDEOIL', 'description': 'Crude Oil Futures'},
    'NATURALGAS': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'NATURALGAS', 'description': 'Natural Gas Futures'},
    'COPPER': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'COPPER', 'description': 'Copper Futures'},
    'ALUMINUM': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'ALUMINUM', 'description': 'Aluminum Futures'},
    'ZINC': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'ZINC', 'description': 'Zinc Futures'},
    'NICKEL': {'type': 'commodity', 'segment': 'commodity', 'symbol': 'NICKEL', 'description': 'Nickel Futures'},
    
    # Forex Indices (Currency Pairs)
    'USDINR': {'type': 'forex', 'segment': 'forex', 'symbol': 'USDINR', 'description': 'US Dollar vs Indian Rupee'},
    'EURINR': {'type': 'forex', 'segment': 'forex', 'symbol': 'EURINR', 'description': 'Euro vs Indian Rupee'},
    'GBPINR': {'type': 'forex', 'segment': 'forex', 'symbol': 'GBPINR', 'description': 'British Pound vs Indian Rupee'},
    'JPYINR': {'type': 'forex', 'segment': 'forex', 'symbol': 'JPYINR', 'description': 'Japanese Yen vs Indian Rupee'},
}

# Option Indices (for Strangle/Straddle strategies)
OPTION_INDICES = ["NIFTY50", "NIFTYBANK", "NIFTY_IT", "INDIA_VIX"]

# Commodity Indices (for commodity trading)
COMMODITY_INDICES = ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER", "ALUMINUM", "ZINC", "NICKEL"]

# Forex Indices (for currency trading)
FOREX_INDICES = ["USDINR", "EURINR", "GBPINR", "JPYINR"]

# Futures Stocks (stocks available for futures trading)
# These are major liquid stocks with futures contracts on NSE
FUTURES_STOCKS = [
    'RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HCL',
    'HDFC', 'ICICIBANK', 'AXISBANK', 'KOTAK', 'SBIN',
    'MARUTI', 'BAJAJFINSV', 'LT', 'TATAMOTORS', 'TATASTEEL',
    'ASIANPAINT', 'DRREDDY', 'SUNPHARMA', 'BHARTIARTL', 'JSWSTEEL'
]

# =============================================================================
# STRATEGY-SEGMENT COMPATIBILITY MATRIX
# =============================================================================
STRATEGY_COMPATIBILITY = {
    'Pair Trading': {
        'cash': True,
        'futures': True,
        'options': False,
        'description': 'Statistical arbitrage between correlated pairs',
        'min_spread': 0.05,
        'applicable_to': ['stocks', 'index_futures'],
    },
    'Mean Reversion': {
        'cash': True,
        'futures': True,
        'options': False,
        'description': 'Trade mean-reversion moves in equities',
        'applicable_to': ['stocks', 'index_futures'],
    },
    'Momentum': {
        'cash': True,
        'futures': True,
        'options': False,
        'description': 'Trend-following strategy for trending markets',
        'applicable_to': ['stocks', 'index_futures'],
    },
    'Volatility Trading': {
        'cash': False,
        'futures': True,
        'options': True,
        'description': 'GARCH-based volatility trading',
        'applicable_to': ['indices', 'options'],
    },
    'Strangle': {
        'cash': False,
        'futures': False,
        'options': True,
        'description': 'Short strangles during high IV',
        'applicable_to': ['index_options'],
        'iv_rank_threshold': 80,
    },
    'Straddle': {
        'cash': False,
        'futures': False,
        'options': True,
        'description': 'Long straddles during low IV',
        'applicable_to': ['index_options'],
        'iv_rank_threshold': 30,
    },
    'Commodity Trading': {
        'commodity': True,
        'description': 'Pair trading and trend following in commodities (metals, oil, gas)',
        'applicable_to': ['commodities'],
        'instruments': ['GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS', 'COPPER', 'ALUMINUM', 'ZINC', 'NICKEL'],
    },
    'Forex Trading': {
        'forex': True,
        'description': 'Currency pair trading with mean reversion and momentum strategies',
        'applicable_to': ['forex'],
        'instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR'],
    },
}

# =============================================================================
# CAPITAL & RISK
# =============================================================================
CAPITAL = 200_000  # CONFIGURABLE: Total trading capital (default: ₹2 Lakh)
CAPITAL_PER_TRADE = 50_000
CAPITAL_PER_LEG = 100_000  # Maximum capital per leg in pair trading strategy
MAX_RISK_PER_TRADE_PCT = 0.02
MAX_CONCURRENT_TRADES = 10
RISK_FREE_RATE = 0.10

# =============================================================================
# RAZORPAY SETTINGS
# =============================================================================
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_placeholder")  # Sandbox/Test Key
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "placeholder_secret")
RAZORPAY_PLAN_MONTHLY = "999"  # Amount in INR
RAZORPAY_PLAN_ANNUAL = "9999"

# =============================================================================
# KITE API SETTINGS
# =============================================================================
KITE_BATCH_SIZE = 250  # Max instruments per quote request (Kite API limit)

# =============================================================================
# PAIRS TRADING PARAMETERS 
# =============================================================================
Z_ENTRY = 2.0
Z_EXIT = 0.5
MAX_HOLDING_DAYS = 10
P_VALUE_THRESHOLD = 0.05
HALF_LIFE_MAX = 25
MIN_STOCKS_PER_INDUSTRY = 3
VOLUME_MIN_DAILY = 50_000

# =============================================================================
# ML & SCORING — THE EDGE
# =============================================================================
ML_SCORE_MIN = 140          # Only trade top 10% pairs
MIN_SIGNAL_CONFIDENCE = 0.60   # Minimum confidence (60%) to display signal
MIN_AUTO_ENTRY_CONFIDENCE = 0.75  # Minimum confidence (75%) for auto-entry in paper trading

# =============================================================================
# BACKTESTING — MULTI-HORIZON ANALYSIS
# =============================================================================
BACKTEST_YEARS = 1
SLIPPAGE_PCT = 0.0005
BROKERAGE_PER_TRADE = 20
BACKTEST_HORIZONS = [30, 90, 180, 270, 365]  # Days: 1mo, 3mo, 6mo, 9mo, 1yr

# =============================================================================
# MARKET HOURS & SCHEDULING — SINGLE SOURCE OF TRUTH
# =============================================================================
# Note: This is the ONLY place where market times should be defined
# All other modules should import from here

# Market Hours (NSE)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
MARKET_HOURS_START = "09:15"      # String format (for backward compatibility)
MARKET_HOURS_END = "15:30"        # String format (for backward compatibility)

# BOD (Beginning of Day) Schedule - Pre-market Analysis
BOD_IDEAL_HOUR = 7                # 7:00 AM - Ideal start time
BOD_IDEAL_MINUTE = 49             # 7:49 AM - Iteal start time !! 
BOD_WINDOW_START_HOUR = 7         # 7:00 AM - Earliest start
BOD_WINDOW_END_HOUR = 9           # 9:00 AM - Latest start (before market open)
BOD_WINDOW_END_MINUTE = 0

# EOD (End of Day) Schedule - Post-market Processing
EOD_IDEAL_HOUR = 17               # 5:00 PM (17:00) - Ideal start time
EOD_IDEAL_MINUTE = 50             # 5:50 PM - 20 min buffer after market close
EOD_WINDOW_START_HOUR = 17        # 3:30 PM - Can start after market close
EOD_WINDOW_START_MINUTE = 30
EOD_WINDOW_END_HOUR = 21          # 6:00 PM - Latest end
EOD_WINDOW_END_MINUTE = 0

# Data Download Schedules
DERIVATIVES_DOWNLOAD_HOUR = 18    # 6:45 PM
DERIVATIVES_DOWNLOAD_MINUTE = 45
IV_DOWNLOAD_HOUR = 18             # 6:45 PM (after market close)
IV_DOWNLOAD_MINUTE = 45

# Timezone
TIMEZONE = "Asia/Kolkata"         # IST (Indian Standard Time)

# Live scanning settings
ENABLE_LIVE_SCANNING = True
SCAN_INTERVAL_SECONDS = 300       # Scan every 5 minutes during market hours
LIVE_SCAN_TIMEFRAMES = ["15minute", "30minute", "60minute", "day"]  # Real-time scanning

# =============================================================================
# DATA SEGMENTS CONFIGURATION
# =============================================================================
DATA_SEGMENTS = {
    'cash': {
        'description': 'Spot/Cash market for stocks and indices',
        'strategies': ['Pair Trading', 'Mean Reversion', 'Momentum'],
        'folder': 'day',  # Base folder for daily data
        'has_intraday': True,
    },
    'futures': {
        'description': 'Index and Stock Futures',
        'strategies': ['Pair Trading', 'Mean Reversion', 'Momentum', 'Volatility Trading'],
        'folder': 'futures',
        'has_intraday': True,
    },
    'options': {
        'description': 'Index and Stock Options',
        'strategies': ['Strangle', 'Straddle', 'Volatility Trading'],
        'folder': 'options',
        'has_intraday': True,
    },
    'commodity': {
        'description': 'Commodity Futures (Metals, Oil & Gas)',
        'strategies': ['Commodity Trading', 'Pair Trading', 'Mean Reversion', 'Momentum'],
        'folder': 'commodities',
        'has_intraday': True,
        'instruments': ['GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS', 'COPPER', 'ALUMINUM', 'ZINC', 'NICKEL'],
    },
    'forex': {
        'description': 'Foreign Exchange Currency Pairs',
        'strategies': ['Forex Trading', 'Pair Trading', 'Mean Reversion', 'Momentum'],
        'folder': 'forex',
        'has_intraday': True,
        'instruments': ['USDINR', 'EURINR', 'GBPINR', 'JPYINR'],
    },
}

# =============================================================================
# PAPER TRADING — ZERO RISK LEARNING
# =============================================================================
ENABLE_PAPER_TRADING = True       # Paper trading with virtual capital
PAPER_CAPITAL = 100_000           # ₹1 Lakh for 30-day challenge
PAPER_CHALLENGE_DAYS = 30         # Prove the strategy works in 30 days

# =============================================================================
# NOTIFICATIONS
# =============================================================================
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "pushkar.prabhat@gmail.com")
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "true").lower() == "true"

# Telegram Configuration - Supports both single and multiple chat IDs
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

# Load chat IDs - supports both single ID and multiple IDs (comma-separated)
_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
_chat_ids = os.getenv("TELEGRAM_CHAT_IDS", "")

if _chat_ids:
    # Parse multiple chat IDs (comma-separated or space-separated)
    try:
        TELEGRAM_CHAT_IDS = [int(cid.strip()) for cid in _chat_ids.replace('[', '').replace(']', '').split(',') if cid.strip().isdigit()]
    except:
        TELEGRAM_CHAT_IDS = []
else:
    # Single chat ID
    try:
        TELEGRAM_CHAT_ID = int(_chat_id) if _chat_id.isdigit() else None
        TELEGRAM_CHAT_IDS = [TELEGRAM_CHAT_ID] if TELEGRAM_CHAT_ID else []
    except:
        TELEGRAM_CHAT_IDS = []   
# =============================================================================
# DATA FLOW CONFIGURATION — PRODUCTION DOWNLOAD/SYNC/REFRESH
# =============================================================================

# Download Strategy - SEPARATE ENDPOINTS FOR LTP & OHLCV
# 
# KITE API LIMITS (as of 2025):
#   /quote/ltp (last traded price):       1000 instruments per request  ← LTP DOWNLOADS
#   /quote/ohlc (open/high/low/close):    1000 instruments per request  ← OHLCV DOWNLOADS
#   Rate limit: ~3 requests/second (333ms minimum between requests)
#
# OPTIMIZATION:
#   LTP: 1000-instrument batch size
#   OHLCV: 1000-instrument batch size (with net change & percentage computations)
#   Total batches: ~120 each endpoint = ~240 total API calls
#   Time saved: 90 minutes → ~5-10 minutes for full download
#
DOWNLOAD_BATCH_SIZE_LTP = 1000                  # Symbols per batch for /quote/ltp endpoint
DOWNLOAD_BATCH_SIZE_OHLCV = 500                 # Symbols per batch for /quote/ohlc endpoint (with net change, %)
DOWNLOAD_RATE_LIMIT_DELAY = 0.33                # Seconds between API requests (3 req/sec = 0.33s)
DOWNLOAD_RATE_LIMIT_DELAY_MS = 330             # Rate limit delay in milliseconds (default 330ms = 0.33s)
DOWNLOAD_PARALLEL_WORKERS = 8                  # Parallel workers for downloads (reduced from 16 to avoid connection resets)
DOWNLOAD_MAX_RETRIES = 3                        # Retry failed requests 3 times
DOWNLOAD_RETRY_DELAY = 2.0                      # Seconds between retries (exponential backoff applied)
DOWNLOAD_HISTORY_DAYS = 365                     # One year of historical data

# Exchange Configuration - SELECT WHICH EXCHANGES TO DOWNLOAD FROM
# Available exchanges: NSE (equity), NFO (derivatives), BSE (equity), MCX (commodities), NCDEX (commodities)
DOWNLOAD_ALL_EXCHANGES = False                  # True = download from all exchanges, False = selected only
DOWNLOAD_SELECTED_EXCHANGES = ['NSE', 'NFO']    # Exchanges to download when DOWNLOAD_ALL_EXCHANGES = False
                                                # Options: 'NSE', 'NFO', 'BSE', 'MCX', 'NCDEX'

# =============================================================================
# INDICES DOWNLOAD CONFIGURATION (NEW)
# =============================================================================
# NSE Indices to download interval data for
# These are downloaded alongside equities and treated as instrument_type='EQ'
# Data structure: data/NSE/5minute/NIFTY/..., data/NSE/day/BANKNIFTY/..., etc.
#
# Benefits:
#   - Historical index data for backtesting
#   - Support for index pair trading strategies
#   - Index-tracking portfolio construction
#
# Note: These are the tradeable index symbols with F&O contracts
INDICES_TO_DOWNLOAD = list(NSE_INDICES.keys())  # All 21 NSE indices from nse_indices_list.csv

# =============================================================================
# DERIVATIVES EOD CONFIGURATION (NEW)
# =============================================================================
# Derivatives to download EOD only (1 candle per day)
# Purpose: Strategy reference, Greeks calculation, IV analysis
# 
# Strategy:
#   - Download at market close (3:30 PM IST)
#   - 1 daily candle per derivative instrument
#   - No aggregation (options are short-lived)
#   - Stored in: data/derivatives/eod/
#
# Update Frequency: Daily (once at close)
# Use Case: Portfolio Greeks, IV rank tracking, strategy validation

DERIVATIVES_DOWNLOAD_EOD = True                 # Enable EOD derivatives download
DERIVATIVES_EOD_TIME = (15, 45)                 # Download at 3:45 PM IST (after close)
DERIVATIVES_LOOKBACK_DAYS = 30                  # Keep last 30 days of EOD data

# Download Directories - ORGANIZED BY EXCHANGE TO AVOID SYMBOL CONFLICTS
# Structure: data/EXCHANGE/TIMEFRAME/SYMBOL/files.parquet
# Example: data/NSE/5minute/RELIANCE/2024-01-01.parquet
#          data/NFO/5minute/NIFTY50-I/2024-01-01.parquet
#          data/MCX/day/GOLD/2024-01-01.parquet
#
# Benefits:
# - Symbols like "RELIANCE" exist on NSE (cash), NFO (futures/options) - organized separately
# - Clear separation between cash, derivatives, commodities, forex
# - Prevents data overwriting when same symbol exists in multiple exchanges
#
def get_download_dir(exchange, timeframe):
    """Get data directory for specific exchange and timeframe."""
    return BASE_DIR / exchange / timeframe

# Directory helpers for each timeframe (defaults to NSE for backward compatibility)
DOWNLOAD_DIR_5MIN = get_download_dir('NSE', '5minute')
DOWNLOAD_DIR_15MIN = get_download_dir('NSE', '15minute')
DOWNLOAD_DIR_30MIN = get_download_dir('NSE', '30minute')
DOWNLOAD_DIR_60MIN = get_download_dir('NSE', '60minute')
DOWNLOAD_DIR_120MIN = get_download_dir('NSE', '120minute')
DOWNLOAD_DIR_150MIN = get_download_dir('NSE', '150minute')
DOWNLOAD_DIR_180MIN = get_download_dir('NSE', '180minute')
DOWNLOAD_DIR_240MIN = get_download_dir('NSE', '240minute')
DOWNLOAD_DIR_DAY = get_download_dir('NSE', 'day')
DOWNLOAD_DIR_WEEK = get_download_dir('NSE', 'week')
DOWNLOAD_DIR_MONTH = get_download_dir('NSE', 'month')

# Aggregation Configuration
MARKET_OPEN_TIME = 9 * 60 + 15                  # 09:15 IST in minutes from midnight
MARKET_CLOSE_TIME = 15 * 60 + 30                # 15:30 IST in minutes from midnight
AGGREGATION_INTERVALS = [15, 30, 60, 120, 150, 180, 240]  # Minutes to aggregate from 5min data
MARKET_TIMEZONE = 'Asia/Kolkata'                # IST timezone

# Data Verification Configuration
VERIFY_MIN_DATA_COMPLETENESS_PCT = 99           # Strict: require 99% of expected candles (max 1% missing)
VERIFY_MISSING_VALUE_THRESHOLD_PCT = 1          # Allow max 1% missing OHLCV values
VERIFY_ZERO_VALUE_THRESHOLD_PCT = 2             # Allow max 2% zero values in volume
VERIFY_DUPLICATE_CHECK = True                   # Check for duplicate timestamps
VERIFY_TIMESTAMP_ORDER = True                   # Verify timestamps are in order

# Data Accuracy Configuration
ACCURACY_FRESHNESS_CHECK_INTERVAL_HOURS = 6     # Check data freshness every 6 hours
ACCURACY_KITE_SOURCE_VERIFICATION = True        # Verify data came from Kite API
ACCURACY_CHECKSUM_VALIDATION = True             # Validate file checksums
ACCURACY_SYNC_STATUS_TRACKING = True            # Track PRIMARY vs CACHE sync status

# Live Feed Configuration
LIVE_FEED_ENABLED = True                        # Real-time updates during market
LIVE_FEED_BUFFER_SIZE = 1000                    # Keep last 1000 updates in memory
LIVE_FEED_SYNC_INTERVAL = 60                    # Sync live data to disk every 60 seconds
LIVE_FEED_INDICES = ['NIFTY50', 'NIFTYBANK', 'INDIA_VIX']  # Monitor these indices
LIVE_FEED_VERIFY_AGAINST_LAST_CLOSE = True     # Verify live quotes match last close

# Cache/Sync Configuration
# The enriched instruments file from Kite API contains all 119,387 instruments
ENRICHED_INSTRUMENTS_PATH = Path('universe/app/app_kite_universe.csv')  # 119,387 instruments from Kite
UNIVERSE_APP_PATH = Path('universe/app/app_kite_universe.csv')  # Alias for clarity
AUTO_SYNC_ON_STARTUP = True                     # Sync instruments on app start
AUTO_SYNC_INTERVAL_MINUTES = 1440               # Auto-sync every 24 hours (in production)

# Data Purge Configuration
PURGE_DATA_OLDER_THAN_DAYS = 730                # Keep 2 years of history (2×365)
PURGE_ENABLED = False                           # Don't auto-purge (manual only)
PURGE_DRY_RUN = True                            # Dry-run purge (show what would delete)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "DEBUG"     # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True
# =============================================================================
# DERIVATIVES UNIVERSE (Lazy-loaded only when needed)
# =============================================================================
# NOTE: The old DerivativesUniverse (266 instruments) is deprecated.
# We now use the enriched Kite API file (119,387 instruments) for all downloads.
# DerivativesUniverse is only loaded on-demand if needed by other parts of the app.

derivatives_universe = None

def get_derivatives_universe():
    """Lazy-load derivatives universe only when needed (not during downloads)"""
    global derivatives_universe
    if derivatives_universe is None:
        try:
            from core.derivatives_universe import DerivativesUniverse, DerivativeType, InstrumentType
            derivatives_universe = DerivativesUniverse()
        except Exception as e:
            import logging
            logging.warning(f"Failed to initialize DerivativesUniverse: {e}")
            return None
    return derivatives_universe

# =============================================================================
# END OF CONFIGURATION
# =============================================================================
