# core/__init__.py — Module initialization
# Makes core a proper Python package

from .downloader import download_price_data, download_all_price_data, validate_data_completeness
from .pairs import scan_all_strategies
from .volatility import garch_volatility, historical_volatility, india_vix_signal
from .strangle import get_strangle_setup
from .kelly import kelly_criterion, half_kelly, position_size
from .backtester import backtest_pair_with_strangle_hedge
from .greeks import black_scholes_greeks, strangle_greeks
from .options_chain import download_and_save_atm_iv, get_latest_iv_rank
from .papertrader import PaperTrader, Trade
from .data_manager import DataManager, get_data_manager
from .instruments_manager import InstrumentsManager, get_instruments_manager
from .symbols_manager import SymbolsManager, get_symbols_manager, ExchangeMarketHours
from .background_downloader import BackgroundDownloader, get_background_downloader, DownloadStatus, DownloadJob
from .data_sources_manager import DataSourcesManager, get_data_sources_manager, ProviderStatus, TokenStatus as ProviderTokenStatus
from .token_manager import TokenManager, get_token_manager, TokenStatus as TokenValidationStatus
from .data_archiver import DataArchiver, get_data_archiver, ArchivalStatus
from .websocket import KiteWebsocket, LTPTick, QuoteTick, FullTick
from .live_feed_manager import LiveFeedManager, Candle
from .nse_securities_manager import (
    NSESecuritiesManager, 
    get_nse_securities_manager, 
    Security, 
    SecuritySnapshot, 
    SecurityStatus,
    SecuritySeries,
    FetchResult
)
from .strategies import (
    PairsTradingStrategy, 
    MomentumStrategy, 
    MomentumBatchProcessor,
    PairsTradingBatchScanner,
    MeanReversionStrategy,
    MovingAverageCrossover,
    StrategyBatchProcessor,
    get_hedge_ratio, 
    get_zscore, 
    should_enter_with_options, 
    is_liquid
)

# Optional scheduler import (requires 'schedule' package)
try:
    from .nse_securities_scheduler import NSESecuritiesScheduler, get_nse_securities_scheduler
except ImportError:
    NSESecuritiesScheduler = None
    get_nse_securities_scheduler = None

from .derivatives_instruments_manager import (
    DerivativesInstrumentsManager,
    get_derivatives_instruments_manager,
    DerivativeInstrument,
    InstrumentType,
    OptionType
)
from .nse_indices_manager import (
    NSEIndicesManager,
    get_nse_indices_manager,
    NSEIndex,
    IndexCategory
)
from .nse_index_constituents_manager import (
    NSEIndexConstituentsManager,
    get_nse_index_constituents_manager,
    Constituent
)
from .zerodha_instrument_master_manager import (
    ZerodhaInstrumentMasterManager,
    get_zerodha_instrument_master_manager,
    ZerodhaInstrument,
    InstrumentSegment
)
from .kite_instruments_manager import (
    KiteInstrumentsManager,
    get_kite_instruments_manager,
    KiteInstrument,
    ExchangeSegment as KiteExchangeSegment,
    FetchResult as KiteFetchResult
)
from .zerodha_download_status_manager import (
    ZerodhaDownloadStatusManager,
    get_zerodha_download_status_manager,
    DownloadStatusEnum,
    DataSourceType,
    TimeframeStatus,
    SymbolDownloadStatus
)
from .data_download_scheduler import (
    DataDownloadScheduler,
    get_data_download_scheduler,
    ScheduleFrequency,
    MarketSession
)
from .data_aggregation_manager import (
    DataAggregationManager,
    AggregationLevel,
    get_data_aggregation_manager
)
from .multi_provider_data_status_manager import (
    MultiProviderDataStatusManager,
    get_multi_provider_status_manager,
    DataProvider,
    DataType,
    DownloadStatusEnum as MultiProviderDownloadStatusEnum
)
from .instruments_mapper import (
    InstrumentsMapper,
    get_instruments_mapper,
    Instrument
)
from .universe_manager import (
    UniverseManager,
    get_universe_manager,
    get_universe,
    refresh_universe
)

# Database and Service Layer (NEW)
from .database import SessionLocal, engine, get_db
from .auth_manager import AuthenticationManager as AuthService

# Create alias for HistoricalDataService (uses download functions from downloader module)
class HistoricalDataService:
    """Service class for historical data download operations"""
    @staticmethod
    def download(symbol, timeframe='5min', days=365):
        """Download historical data"""
        from .downloader import download_price_data
        return download_price_data(symbol)

# Import DataQualityService from services

# Import DataQualityService from services
import sys
from pathlib import Path
_services_path = str(Path(__file__).parent.parent / "services")
if _services_path not in sys.path:
    sys.path.insert(0, _services_path)

try:
    from data_quality import DataQualityService
except ImportError:
    # If direct import fails, create a fallback
    DataQualityService = None

__all__ = [
    # Data Management
    'DataManager',
    'get_data_manager',
    'InstrumentsManager',
    'get_instruments_manager',
    'SymbolsManager',
    'get_symbols_manager',
    'ExchangeMarketHours',
    'BackgroundDownloader',
    'get_background_downloader',
    'DownloadStatus',
    'DownloadJob',
    # Live Feed Management (NEW)
    'KiteWebsocket',
    'LTPTick',
    'QuoteTick',
    'FullTick',
    'LiveFeedManager',
    'Candle',
    # Data Sources Management (NEW)
    'DataSourcesManager',
    'get_data_sources_manager',
    'ProviderStatus',
    'ProviderTokenStatus',
    # Token Management (NEW)
    'TokenManager',
    'get_token_manager',
    'TokenValidationStatus',
    # Data Archival (NEW)
    'DataArchiver',
    'get_data_archiver',
    'ArchivalStatus',
    # NSE Securities Management (NEW)
    'NSESecuritiesManager',
    'get_nse_securities_manager',
    'Security',
    'SecuritySnapshot',
    'SecurityStatus',
    'SecuritySeries',
    'FetchResult',
    # NSE Securities Scheduler (NEW)
    'NSESecuritiesScheduler',
    'get_nse_securities_scheduler',
    # Derivatives Instruments Management (NEW)
    'DerivativesInstrumentsManager',
    'get_derivatives_instruments_manager',
    'DerivativeInstrument',
    'InstrumentType',
    'OptionType',
    # NSE Indices Management (NEW)
    'NSEIndicesManager',
    'get_nse_indices_manager',
    'NSEIndex',
    'IndexCategory',
    # NSE Index Constituents Management (NEW)
    'NSEIndexConstituentsManager',
    'get_nse_index_constituents_manager',
    'Constituent',
    # Zerodha Instrument Master Management (NEW)
    'ZerodhaInstrumentMasterManager',
    'get_zerodha_instrument_master_manager',
    'ZerodhaInstrument',
    'InstrumentSegment',
    # Kite Instruments Management (NEW)
    'KiteInstrumentsManager',
    'get_kite_instruments_manager',
    'KiteInstrument',
    'KiteExchangeSegment',
    'KiteFetchResult',
    # Zerodha Download Status Management (NEW)
    'ZerodhaDownloadStatusManager',
    'get_zerodha_download_status_manager',
    'DownloadStatusEnum',
    'DataSourceType',
    'TimeframeStatus',
    'SymbolDownloadStatus',
    # Data Download Scheduler (NEW)
    'DataDownloadScheduler',
    'get_data_download_scheduler',
    'ScheduleFrequency',
    'MarketSession',
    # Data Aggregation Manager (NEW)
    'DataAggregationManager',
    'AggregationLevel',
    'get_data_aggregation_manager',
    # Multi-Provider Data Status Manager (NEW)
    'MultiProviderDataStatusManager',
    'get_multi_provider_status_manager',
    'DataProvider',
    'DataType',
    'MultiProviderDownloadStatusEnum',
    # Instruments Mapper (NEW)
    'InstrumentsMapper',
    'get_instruments_mapper',
    'Instrument',
    # Universe Management (NEW)
    'UniverseManager',
    'get_universe_manager',
    'get_universe',
    'refresh_universe',
    # Database and Service Layer (NEW)
    'SessionLocal',
    'engine',
    'get_db',
    'AuthService',
    'HistoricalDataService',
    'DataQualityService',
    # Downloads
    'download_price_data',
    'download_all_price_data',
    'validate_data_completeness',
    # Strategy Analysis
    'scan_all_strategies',
    # Volatility
    'garch_volatility',
    'historical_volatility',
    'india_vix_signal',
    # Options
    'get_strangle_setup',
    'download_and_save_atm_iv',
    'get_latest_iv_rank',
    # Kelly Criterion
    'kelly_criterion',
    'half_kelly',
    'position_size',
    # Backtesting
    'backtest_pair_with_strangle_hedge',
    # Greeks
    'black_scholes_greeks',
    'strangle_greeks',
    # Paper Trading
    'PaperTrader',
    'Trade',
]
