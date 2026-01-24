"""
Websocket Live Feed Aggregator
Aggregates tick data from Zerodha websocket into OHLC bars
Builds 5-min, 15-min, 30-min, 60-min bars from live ticks
"""

import pandas as pd
import datetime as dt
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from utils.failure_logger import log_failure

logger = logging.getLogger('websocket_aggregator')


class BarAggregator:
    """Aggregates tick data into OHLC bars for different timeframes"""
    
    # Supported timeframes (in seconds)
    TIMEFRAMES = {
        '5minute': 5 * 60,
        '15minute': 15 * 60,
        '30minute': 30 * 60,
        '60minute': 60 * 60,
        '120minute': 120 * 60,
        '150minute': 150 * 60,
        '180minute': 180 * 60,
        '240minute': 240 * 60,
    }
    
    def __init__(self):
        """Initialize bar aggregator"""
        # Store tick data per symbol: {symbol: [(timestamp, price, volume), ...]}
        self.tick_buffer = defaultdict(list)
        
        # Store completed bars per symbol: {symbol: {timeframe: [bar, bar, ...]}}
        self.completed_bars = defaultdict(lambda: defaultdict(list))
        
        # Track current incomplete bar per symbol: {symbol: {timeframe: bar_data}}
        self.current_bars = defaultdict(lambda: defaultdict(dict))
    
    def add_tick(self, symbol: str, timestamp: dt.datetime, price: float, volume: int = 1):
        """
        Add a tick to the aggregator
        
        Args:
            symbol: Trading symbol (e.g., 'BANKNIFTY25D2400CE')
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume (default 1)
        """
        self.tick_buffer[symbol].append({
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        })
    
    def get_bars(self, symbol: str, timeframe: str) -> List[Dict]:
        """
        Get OHLC bars for a symbol and timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe key (e.g., '5minute')
        
        Returns:
            List of OHLC bar dicts
        """
        if timeframe not in self.TIMEFRAMES:
            logger.error(f"[ERROR] Unsupported timeframe: {timeframe}")
            try:
                log_failure(symbol=symbol or 'unknown', exchange='LOCAL', reason='unsupported_timeframe', details=f"timeframe={timeframe}")
            except Exception:
                logger.debug("[FAILURE_LOG] Could not log unsupported_timeframe")
            return []
        
        # Process pending ticks first
        self._process_ticks(symbol, timeframe)
        
        return self.completed_bars[symbol][timeframe]
    
    def _process_ticks(self, symbol: str, timeframe: str):
        """
        Process buffered ticks and aggregate into bars
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to aggregate to
        """
        if not self.tick_buffer[symbol]:
            return
        
        interval_seconds = self.TIMEFRAMES[timeframe]
        ticks = self.tick_buffer[symbol]
        
        # Process each tick
        for tick in ticks:
            timestamp = tick['timestamp']
            price = tick['price']
            volume = tick['volume']
            
            # Calculate bar start time
            bar_start = self._get_bar_start_time(timestamp, interval_seconds)
            
            # Get or create current bar
            bar_key = bar_start.isoformat()
            current_bar = self.current_bars[symbol][timeframe]
            
            if bar_key not in current_bar:
                # New bar
                current_bar[bar_key] = {
                    'timestamp': bar_start,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
            else:
                # Update existing bar
                bar = current_bar[bar_key]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
                bar['volume'] += volume
        
        # Move completed bars to final storage
        self._finalize_bars(symbol, timeframe)
        
        # Clear processed ticks
        self.tick_buffer[symbol] = []
    
    def _get_bar_start_time(self, timestamp: dt.datetime, interval_seconds: int) -> dt.datetime:
        """
        Calculate bar start time for a given timestamp
        
        Args:
            timestamp: Tick timestamp
            interval_seconds: Bar interval in seconds
        
        Returns:
            Bar start time (rounded down)
        """
        # Convert to seconds since epoch
        epoch_seconds = int(timestamp.timestamp())
        
        # Round down to bar interval
        bar_epoch = (epoch_seconds // interval_seconds) * interval_seconds
        
        return dt.datetime.fromtimestamp(bar_epoch)
    
    def _finalize_bars(self, symbol: str, timeframe: str):
        """
        Move completed bars to final storage
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        current_bar = self.current_bars[symbol][timeframe]
        
        if not current_bar:
            return
        
        # Get the earliest and latest bar timestamps
        bar_keys = sorted(current_bar.keys())
        now = dt.datetime.now()
        
        # Finalize bars that are no longer current (older than 1 bar interval)
        interval_seconds = self.TIMEFRAMES[timeframe]
        
        bars_to_finalize = []
        for bar_key in bar_keys:
            bar = current_bar[bar_key]
            bar_timestamp = bar['timestamp']
            
            # Check if bar is complete (its interval has passed)
            next_bar_time = bar_timestamp + dt.timedelta(seconds=interval_seconds)
            if now >= next_bar_time:
                bars_to_finalize.append(bar_key)
        
        # Move finalized bars to storage
        for bar_key in bars_to_finalize:
            bar = current_bar.pop(bar_key)
            self.completed_bars[symbol][timeframe].append(bar)
    
    def to_dataframe(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Convert aggregated bars to DataFrame
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        bars = self.get_bars(symbol, timeframe)
        
        if not bars:
            return None
        
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df


# Global aggregator instance
_aggregator = None


def get_aggregator() -> BarAggregator:
    """Get or create global bar aggregator instance"""
    global _aggregator
    if _aggregator is None:
        _aggregator = BarAggregator()
        logger.info("[OK] Bar aggregator initialized")
    return _aggregator


def add_tick(symbol: str, timestamp: dt.datetime, price: float, volume: int = 1):
    """Add a tick to the global aggregator"""
    agg = get_aggregator()
    agg.add_tick(symbol, timestamp, price, volume)


def get_bars(symbol: str, timeframe: str) -> List[Dict]:
    """Get bars from global aggregator"""
    agg = get_aggregator()
    return agg.get_bars(symbol, timeframe)


def get_bars_df(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Get bars as DataFrame from global aggregator"""
    agg = get_aggregator()
    return agg.to_dataframe(symbol, timeframe)
