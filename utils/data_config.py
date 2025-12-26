"""
ARTEMIS DATA CONFIGURATION SYSTEM
Central, parameterized data location management
All downloads → All calculations use THIS system
"""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataConfig:
    """
    Central data configuration system
    Single source of truth for all data locations
    """
    
    # =========================================================================
    # CORE DIRECTORY STRUCTURE
    # =========================================================================
    
    # Root directory - can be overridden via env var
    DATA_ROOT = Path(os.getenv('ARTEMIS_DATA_ROOT', 'marketdata'))
    
    # Ensure root exists
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # DOWNLOADED DATA (NSE Market Data)
    # =========================================================================
    
    NSE_DIR = DATA_ROOT / "NSE"
    
    # Interval data directories
    INTERVAL_DATA = {
        'day': NSE_DIR / 'day',
        '3minute': NSE_DIR / '3minute',
        '5minute': NSE_DIR / '5minute',
        '15minute': NSE_DIR / '15minute',
        '30minute': NSE_DIR / '30minute',
        '60minute': NSE_DIR / '60minute',
        'week': NSE_DIR / 'week',
        'month': NSE_DIR / 'month',
    }
    
    # Options data
    OPTIONS_DIR = NSE_DIR / 'options'
    IV_HISTORY_DIR = NSE_DIR / 'iv_history'
    OPTION_CHAINS_DIR = NSE_DIR / 'option_chains'
    
    # Bhavcopy (EOD data)
    BHAVCOPY_DIR = NSE_DIR / 'bhavcopy'
    BHAVCOPY_EQUITY = BHAVCOPY_DIR / 'equity'
    BHAVCOPY_FO = BHAVCOPY_DIR / 'fo'
    
    # =========================================================================
    # METADATA & TRACKING
    # =========================================================================
    
    METADATA_DIR = DATA_ROOT / 'metadata'
    
    # Download state tracking
    DOWNLOAD_STATE_FILE = METADATA_DIR / 'download_state.json'
    MASTERLIST_VERSION_FILE = METADATA_DIR / 'masterlist_version.json'
    DOWNLOAD_LOG_FILE = METADATA_DIR / 'download_log.json'
    FAILED_INSTRUMENTS_FILE = Path('failed_instruments.txt')  # Root level
    
    # =========================================================================
    # UNIVERSE DATA
    # =========================================================================
    
    UNIVERSE_DIR = DATA_ROOT.parent / 'universe'  # Separate from market data
    SYMBOLS_DIR = UNIVERSE_DIR / 'symbols'
    INSTRUMENTS_DIR = UNIVERSE_DIR / 'instruments'
    ENRICHED_DIR = UNIVERSE_DIR / 'enriched'
    
    # =========================================================================
    # LOGS
    # =========================================================================
    
    LOG_DIR = Path('logs')
    EOD_BOD_LOG_FILE = LOG_DIR / 'eod_bod.log'
    DOWNLOAD_LOG_FILE_PATH = LOG_DIR / 'download.log'
    STRATEGY_LOG_FILE = LOG_DIR / 'strategies.log'
    
    # =========================================================================
    # STRATEGY DATA (Calculated/Derived)
    # =========================================================================
    
    STRATEGY_DIR = DATA_ROOT / 'strategies'
    PAIR_TRADES_DIR = STRATEGY_DIR / 'pair_trades'
    MEAN_REVERSION_DIR = STRATEGY_DIR / 'mean_reversion'
    STAT_ARB_DIR = STRATEGY_DIR / 'stat_arb'
    CALENDAR_SPREADS_DIR = STRATEGY_DIR / 'calendar_spreads'
    
    # =========================================================================
    # BACKTEST DATA
    # =========================================================================
    
    BACKTEST_DIR = DATA_ROOT / 'backtest'
    BACKTEST_RESULTS_DIR = BACKTEST_DIR / 'results'
    BACKTEST_LOGS_DIR = BACKTEST_DIR / 'logs'
    
    @classmethod
    def initialize_all_directories(cls):
        """Create all required directories"""
        dirs_to_create = [
            # Interval data
            *cls.INTERVAL_DATA.values(),
            # Options
            cls.OPTIONS_DIR,
            cls.IV_HISTORY_DIR,
            cls.OPTION_CHAINS_DIR,
            # Bhavcopy
            cls.BHAVCOPY_EQUITY,
            cls.BHAVCOPY_FO,
            # Metadata
            cls.METADATA_DIR,
            # Universe
            cls.SYMBOLS_DIR,
            cls.INSTRUMENTS_DIR,
            cls.ENRICHED_DIR,
            # Logs
            cls.LOG_DIR,
            # Strategies
            cls.PAIR_TRADES_DIR,
            cls.MEAN_REVERSION_DIR,
            cls.STAT_ARB_DIR,
            cls.CALENDAR_SPREADS_DIR,
            # Backtest
            cls.BACKTEST_RESULTS_DIR,
            cls.BACKTEST_LOGS_DIR,
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_interval_dir(cls, interval: str) -> Path:
        """Get directory for specific interval"""
        if interval not in cls.INTERVAL_DATA:
            raise ValueError(f"Unknown interval: {interval}. Valid: {list(cls.INTERVAL_DATA.keys())}")
        return cls.INTERVAL_DATA[interval]
    
    @classmethod
    def get_symbol_file(cls, symbol: str, interval: str) -> Path:
        """Get parquet file path for symbol/interval"""
        interval_dir = cls.get_interval_dir(interval)
        return interval_dir / f"{symbol}.parquet"
    
    @classmethod
    def get_strategy_output_dir(cls, strategy: str) -> Path:
        """Get output directory for strategy results"""
        strategies = {
            'pair_trading': cls.PAIR_TRADES_DIR,
            'mean_reversion': cls.MEAN_REVERSION_DIR,
            'stat_arb': cls.STAT_ARB_DIR,
            'calendar_spreads': cls.CALENDAR_SPREADS_DIR,
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Valid: {list(strategies.keys())}")
        
        return strategies[strategy]
    
    @classmethod
    def get_status_dict(cls) -> Dict:
        """Get current configuration status"""
        return {
            'data_root': str(cls.DATA_ROOT),
            'data_root_exists': cls.DATA_ROOT.exists(),
            'intervals_available': {
                interval: dir_path.exists()
                for interval, dir_path in cls.INTERVAL_DATA.items()
            },
            'metadata_available': cls.METADATA_DIR.exists(),
            'universe_available': cls.UNIVERSE_DIR.exists(),
            'logs_available': cls.LOG_DIR.exists(),
            'total_tracked_dirs': len(cls.INTERVAL_DATA) + 8,
        }
    
    @classmethod
    def print_status(cls):
        """Print configuration status"""
        print("\n" + "="*70)
        print("ARTEMIS DATA CONFIGURATION")
        print("="*70)
        
        status = cls.get_status_dict()
        print(f"\nDATA ROOT: {status['data_root']}")
        print(f"Status: {'✅ READY' if status['data_root_exists'] else '❌ MISSING'}")
        
        print(f"\nINTERVAL DATA:")
        for interval, exists in status['intervals_available'].items():
            symbol = '✅' if exists else '❌'
            print(f"  {symbol} {interval:12s} -> {cls.get_interval_dir(interval)}")
        
        print(f"\nMETADATA: {'✅' if status['metadata_available'] else '❌'} {cls.METADATA_DIR}")
        print(f"UNIVERSE: {'✅' if status['universe_available'] else '❌'} {cls.UNIVERSE_DIR}")
        print(f"LOGS:     {'✅' if status['logs_available'] else '❌'} {cls.LOG_DIR}")
        
        print("\n" + "="*70 + "\n")


# Initialize all directories on import
DataConfig.initialize_all_directories()

# Singleton instance
_config_instance = DataConfig()

def get_data_config() -> DataConfig:
    """Get singleton DataConfig instance"""
    return _config_instance
