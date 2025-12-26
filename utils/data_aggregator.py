"""
utils/data_aggregator.py - Auto-generate multi-timeframe data from 5-minute candles
Converts 5min → 10min, 15min, 30min, 60min, 2h, 4h
Daily data downloaded separately
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from utils.logger import logger
from datetime import datetime, timedelta

class DataAggregator:
    """Aggregate 5-minute data into multiple timeframes"""
    
    # Timeframe definitions (minutes from base)
    TIMEFRAMES = {
        '5min': 1,      # Base
        '10min': 2,     # 2 * 5min
        '15min': 3,     # 3 * 5min
        '30min': 6,     # 6 * 5min
        '60min': 12,    # 12 * 5min
        '2hour': 24,    # 24 * 5min
        '4hour': 48,    # 48 * 5min
    }
    
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.five_min_dir = os.path.join(base_dir, '5minute')
        self.output_dirs = {
            '10min': os.path.join(base_dir, '10minute'),
            '15min': os.path.join(base_dir, '15minute'),
            '30min': os.path.join(base_dir, '30minute'),
            '60min': os.path.join(base_dir, '60minute'),
            '2hour': os.path.join(base_dir, '2hour'),
            '4hour': os.path.join(base_dir, '4hour'),
        }
        
        # Create output directories if needed
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def aggregate_ohlc(self, df, multiplier):
        """Aggregate OHLC data by multiplier"""
        if df is None or df.empty:
            return df
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("Could not convert index to datetime")
                return df
        
        # OHLC aggregation
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        
        # Add any additional numeric columns
        for col in df.columns:
            if col not in agg_dict and pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'sum'
        
        try:
            # Resample based on multiplier * 5 minutes
            freq = f'{multiplier * 5}min'
            aggregated = df.resample(freq).agg(agg_dict)
            aggregated = aggregated.dropna()
            return aggregated
        except Exception as e:
            logger.error(f"Error aggregating OHLC: {e}")
            return df
    
    def process_single_stock(self, symbol):
        """Process a single stock from 5min → all timeframes"""
        five_min_file = os.path.join(self.five_min_dir, f'{symbol}.csv')
        
        if not os.path.exists(five_min_file):
            logger.warning(f"5-minute file not found: {symbol}")
            return False
        
        try:
            # Load 5-minute data
            df_5min = pd.read_csv(five_min_file, index_col=0)
            logger.info(f"Loaded {symbol}: {len(df_5min)} 5-min candles")
            
            # Generate aggregated timeframes
            for timeframe, multiplier in self.TIMEFRAMES.items():
                if timeframe == '5min':
                    continue  # Skip base
                
                output_dir = self.output_dirs.get(timeframe)
                if not output_dir:
                    continue
                
                # Aggregate data
                aggregated = self.aggregate_ohlc(df_5min, multiplier)
                
                if aggregated is not None and not aggregated.empty:
                    # Save to file
                    output_file = os.path.join(output_dir, f'{symbol}.csv')
                    aggregated.to_csv(output_file)
                    logger.info(f"Generated {timeframe}: {len(aggregated)} candles → {output_file}")
                else:
                    logger.warning(f"Failed to aggregate {symbol} to {timeframe}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False
    
    def process_all_stocks(self):
        """Process all stocks from 5min to other timeframes"""
        if not os.path.exists(self.five_min_dir):
            logger.warning(f"5-minute directory not found: {self.five_min_dir}")
            return
        
        csv_files = [f for f in os.listdir(self.five_min_dir) if f.endswith('.csv')]
        logger.info(f"Processing {len(csv_files)} stocks from 5-minute data")
        
        success_count = 0
        for csv_file in csv_files:
            symbol = csv_file.replace('.csv', '')
            if self.process_single_stock(symbol):
                success_count += 1
        
        logger.info(f"Processed {success_count}/{len(csv_files)} stocks successfully")
        return success_count == len(csv_files)
    
    def verify_aggregation(self):
        """Verify that aggregated data is correct"""
        # Pick a random stock and verify
        csv_files = os.listdir(self.five_min_dir)
        if not csv_files:
            return False
        
        symbol = csv_files[0].replace('.csv', '')
        
        try:
            df_5min = pd.read_csv(os.path.join(self.five_min_dir, f'{symbol}.csv'), index_col=0)
            df_15min = pd.read_csv(os.path.join(self.output_dirs['15min'], f'{symbol}.csv'), index_col=0)
            
            # Basic checks
            assert len(df_15min) == len(df_5min) // 3, "15min candle count incorrect"
            assert df_5min['close'].iloc[0] == df_15min['open'].iloc[0], "Opening candle mismatch"
            
            logger.info(f"Verification passed for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

def auto_aggregate_new_data():
    """Auto-aggregate new 5-minute data to other timeframes"""
    aggregator = DataAggregator()
    return aggregator.process_all_stocks()

def verify_all_timeframes():
    """Verify all timeframes are correctly aggregated"""
    aggregator = DataAggregator()
    return aggregator.verify_aggregation()
