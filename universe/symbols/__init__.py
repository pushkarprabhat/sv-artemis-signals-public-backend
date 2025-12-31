import pandas as pd
from pathlib import Path
import logging

# Use logger from utils if available
try:
    from utils.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

def load_universe():
    """
    Load the global instrument universe.
    Tries to load from UniverseManager cache, falls back to CSV files.
    """
    try:
        # Lazy import of UniverseManager to avoid circular dependencies
        from core.universe_manager import get_universe_manager
        
        manager = get_universe_manager()
        df = manager.get_universe()
        if df is not None and len(df) > 0:
            # Map common column name variations for different consumers
            if 'Symbol' in df.columns and 'tradingsymbol' not in df.columns:
                df['tradingsymbol'] = df['Symbol']
            if 'Exchange' in df.columns and 'exchange' not in df.columns:
                df['exchange'] = df['Exchange']
            if 'InstrumentType' in df.columns and 'instrument_type' not in df.columns:
                df['instrument_type'] = df['InstrumentType']
            return df
    except Exception as e:
        logger.warning(f"Failed to load universe via UniverseManager: {e}")
    
    # Fallback to loading any available CSV in universe directory
    # These are legacy paths that might still exist in some environments
    project_root = Path(__file__).parent.parent.parent
    fallback_paths = [
        project_root / "universe/app/app_kite_universe.csv",
        project_root / "universe/symbols/universe.csv",
        project_root / "universe/instruments_NFO.csv",
        project_root / "instruments_NFO.csv"
    ]
    
    for path in fallback_paths:
        if path.exists():
            try:
                logger.info(f"Loading fallback universe from {path}")
                df = pd.read_csv(path)
                # Map common column name variations
                if 'Symbol' not in df.columns and 'tradingsymbol' in df.columns:
                    df['Symbol'] = df['tradingsymbol']
                if 'Exchange' not in df.columns and 'exchange' in df.columns:
                    df['Exchange'] = df['exchange']
                return df
            except Exception as e:
                logger.warning(f"Failed to load fallback universe from {path}: {e}")
                
    # Final fallback: return empty DataFrame with expected columns to prevent crashes
    logger.error("CRITICAL: No universe data found. Returning empty DataFrame.")
    return pd.DataFrame(columns=['Symbol', 'Exchange', 'InstrumentType', 'tradingsymbol', 'exchange', 'instrument_type'])
