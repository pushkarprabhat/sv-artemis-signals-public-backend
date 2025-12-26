# core/iv_tracker.py — Implied Volatility Tracker
# Tracks: Historical IV per symbol/expiry/strike
# Features: IV percentile, IV surface, volatility smile analysis

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import json
from config import BASE_DIR
from utils.logger import logger


class IVTracker:
    """Tracks and analyzes implied volatility"""
    
    def __init__(self):
        self.base_dir = BASE_DIR / 'data' / 'iv_history'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.surface_dir = self.base_dir / 'surfaces'
        self.surface_dir.mkdir(parents=True, exist_ok=True)
    
    def store_iv_snapshot(self, symbol, iv_data, reference_date=None):
        """Store IV snapshot for a symbol and its options
        
        Args:
            symbol: Stock/index symbol
            iv_data: DataFrame with columns [expiry, strike, option_type, iv, bid_iv, ask_iv]
            reference_date: Date of snapshot (default=today)
        
        Returns:
            Path: Where snapshot was saved
        """
        if reference_date is None:
            reference_date = dt.datetime.now().date()
        
        # Create directory for symbol
        symbol_dir = self.base_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to data
        iv_data = iv_data.copy()
        iv_data['snapshot_date'] = reference_date
        iv_data['snapshot_time'] = dt.datetime.now()
        
        # Save snapshot
        snapshot_file = symbol_dir / f"{reference_date}.parquet"
        iv_data.to_parquet(snapshot_file, index=False)
        
        logger.info(f"[IV] Stored IV snapshot for {symbol}: {len(iv_data)} strikes")
        return snapshot_file
    
    def get_historical_iv(self, symbol, expiry=None, lookback_days=30):
        """Retrieve historical IV data for a symbol
        
        Args:
            symbol: Symbol name
            expiry: Specific expiry (None = all expiries)
            lookback_days: Days of history to retrieve
        
        Returns:
            DataFrame: Historical IV data
        """
        symbol_dir = self.base_dir / symbol
        if not symbol_dir.exists():
            logger.warning(f"[IV] No data found for {symbol}")
            return None
        
        # Collect data from recent days
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=lookback_days)
        
        data_frames = []
        for snapshot_file in sorted(symbol_dir.glob("*.parquet")):
            try:
                file_date = dt.datetime.strptime(snapshot_file.stem, '%Y-%m-%d').date()
                if start_date <= file_date <= end_date:
                    df = pd.read_parquet(snapshot_file)
                    if expiry and 'expiry' in df.columns:
                        df = df[df['expiry'] == expiry]
                    if not df.empty:
                        data_frames.append(df)
            except Exception as e:
                logger.warning(f"[IV] Error reading {snapshot_file}: {e}")
        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        
        return None
    
    def calculate_iv_percentile(self, symbol, lookback_days=365):
        """Calculate IV percentile (how high current IV is historically)
        
        Args:
            symbol: Symbol name
            lookback_days: Historical period for percentile calculation
        
        Returns:
            dict: {iv_metric: percentile}
        """
        historical_data = self.get_historical_iv(symbol, lookback_days=lookback_days)
        
        if historical_data is None or historical_data.empty:
            return {}
        
        if 'iv' not in historical_data.columns:
            return {}
        
        # Calculate percentile for current vs historical
        current_date = dt.datetime.now().date()
        current_data = self.get_historical_iv(symbol, lookback_days=1)
        
        if current_data is None or current_data.empty:
            return {}
        
        # Average current IV
        current_iv = current_data['iv'].mean()
        
        # Historical IV distribution
        historical_iv = historical_data['iv'].dropna()
        
        if len(historical_iv) == 0:
            return {'current_iv': current_iv}
        
        # Calculate percentile
        percentile = (historical_iv < current_iv).sum() / len(historical_iv) * 100
        
        return {
            'current_iv': current_iv,
            'percentile': percentile,
            'historical_min': historical_iv.min(),
            'historical_max': historical_iv.max(),
            'historical_mean': historical_iv.mean(),
            'historical_std': historical_iv.std(),
            'iv_rank': percentile  # Same as percentile, different name
        }
    
    def get_iv_surface(self, symbol, expiry, reference_date=None):
        """Get IV surface for a symbol/expiry (IV across strikes)
        
        Args:
            symbol: Symbol name
            expiry: Expiry date
            reference_date: Date for surface (default=today)
        
        Returns:
            DataFrame: IV surface data [strike, call_iv, put_iv, volatility_smile]
        """
        if reference_date is None:
            reference_date = dt.datetime.now().date()
        
        symbol_dir = self.base_dir / symbol
        snapshot_file = symbol_dir / f"{reference_date}.parquet"
        
        if not snapshot_file.exists():
            logger.warning(f"[IV] Snapshot not found for {symbol}/{reference_date}")
            return None
        
        try:
            df = pd.read_parquet(snapshot_file)
            
            # Filter by expiry
            if 'expiry' in df.columns:
                df = df[df['expiry'] == expiry]
            
            if df.empty:
                return None
            
            # Pivot to get call/put IV by strike
            surface = df.pivot_table(
                index='strike',
                columns='option_type',
                values='iv',
                aggfunc='mean'
            )
            
            # Calculate volatility smile (ATM-normalized)
            if 'call' in surface.columns and 'put' in surface.columns:
                # Volatility smile: IV skew across strikes
                surface['avg_iv'] = (surface.get('call', 0) + surface.get('put', 0)) / 2
                surface['iv_skew'] = surface.get('call', 0) - surface.get('put', 0)
            
            return surface
        
        except Exception as e:
            logger.error(f"[IV] Error creating surface for {symbol}/{expiry}: {e}")
            return None
    
    def track_iv_change(self, symbol, lookback_days=30):
        """Track IV changes over time
        
        Args:
            symbol: Symbol name
            lookback_days: Period to analyze
        
        Returns:
            dict: IV change metrics
        """
        historical_data = self.get_historical_iv(symbol, lookback_days=lookback_days)
        
        if historical_data is None or historical_data.empty:
            return {}
        
        if 'iv' not in historical_data.columns:
            return {}
        
        # Group by date and calculate average IV
        daily_iv = historical_data.groupby('snapshot_date')['iv'].agg(['mean', 'std', 'min', 'max'])
        
        if daily_iv.empty:
            return {}
        
        # Calculate changes
        first_iv = daily_iv['mean'].iloc[0]
        last_iv = daily_iv['mean'].iloc[-1]
        iv_change = last_iv - first_iv
        iv_change_pct = (iv_change / first_iv * 100) if first_iv != 0 else 0
        
        return {
            'current_iv': last_iv,
            'iv_change': iv_change,
            'iv_change_pct': iv_change_pct,
            'iv_trend': 'up' if iv_change > 0 else 'down',
            'iv_volatility': daily_iv['mean'].std(),
            'avg_iv': daily_iv['mean'].mean(),
            'min_iv': daily_iv['mean'].min(),
            'max_iv': daily_iv['mean'].max()
        }
    
    def save_iv_surface_json(self, symbol, expiry, surface_df, reference_date=None):
        """Save IV surface as JSON for visualization
        
        Args:
            symbol: Symbol name
            expiry: Expiry date
            surface_df: Surface DataFrame
            reference_date: Date for surface
        
        Returns:
            Path: Where JSON was saved
        """
        if reference_date is None:
            reference_date = dt.datetime.now().date()
        
        if surface_df is None or surface_df.empty:
            return None
        
        surface_data = {
            'symbol': symbol,
            'expiry': str(expiry),
            'reference_date': str(reference_date),
            'strikes': surface_df.index.tolist(),
            'call_iv': surface_df.get('call', [None]*len(surface_df)).tolist(),
            'put_iv': surface_df.get('put', [None]*len(surface_df)).tolist(),
            'volatility_smile': surface_df.get('iv_skew', [None]*len(surface_df)).tolist()
        }
        
        # Save JSON
        json_file = self.surface_dir / f"{symbol}_{expiry}_{reference_date}.json"
        with open(json_file, 'w') as f:
            json.dump(surface_data, f, indent=2, default=str)
        
        logger.info(f"[IV] Saved surface JSON: {json_file}")
        return json_file
    
    def list_tracked_symbols(self):
        """List all symbols with IV data
        
        Returns:
            List: Symbol names
        """
        if not self.base_dir.exists():
            return []
        
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def cleanup_old_data(self, days_to_keep=365):
        """Delete IV snapshots older than specified days
        
        Args:
            days_to_keep: Number of days of history to keep
        
        Returns:
            int: Number of files deleted
        """
        cutoff_date = dt.datetime.now().date() - dt.timedelta(days=days_to_keep)
        deleted_count = 0
        
        for symbol_dir in self.base_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            for snapshot_file in symbol_dir.glob("*.parquet"):
                try:
                    file_date = dt.datetime.strptime(snapshot_file.stem, '%Y-%m-%d').date()
                    if file_date < cutoff_date:
                        snapshot_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"[IV] Error cleaning {snapshot_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"[IV] Cleaned {deleted_count} old snapshots")
        
        return deleted_count
