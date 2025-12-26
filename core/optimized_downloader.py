"""
core/optimized_downloader.py - Fast data download with smart strategy
Downloads 5-minute base data quickly, then auto-aggregates to other timeframes
Daily data downloaded separately
"""

import pandas as pd
from datetime import datetime, timedelta
import os
from utils.logger import logger
from utils.data_aggregator import DataAggregator
from kiteconnect import KiteConnect
import time

class OptimizedDownloader:
    """Optimized data downloader for fast 5-minute → multi-timeframe strategy"""
    
    def __init__(self, kite_instance):
        self.kite = kite_instance
        self.aggregator = DataAggregator()
        self.base_dir = 'data'
        self.five_min_dir = os.path.join(self.base_dir, '5minute')
        self.daily_dir = os.path.join(self.base_dir, 'day')
        
        # Create directories
        os.makedirs(self.five_min_dir, exist_ok=True)
        os.makedirs(self.daily_dir, exist_ok=True)
    
    def download_five_minute_data(self, symbol, days_back=365):
        """Download 5-minute data for a symbol (fast baseline)"""
        try:
            logger.info(f"Downloading 5-minute data for {symbol} ({days_back} days)")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_data = []
            current_date = start_date
            
            # Download in chunks (1 day at a time for 5-minute data)
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=1), end_date)
                
                try:
                    # Get historical data
                    data = self.kite.historical_data(
                        instrument_token=symbol,
                        from_date=current_date,
                        to_date=chunk_end,
                        interval='5minute'
                    )
                    
                    if data:
                        all_data.extend(data)
                    
                    current_date = chunk_end
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error downloading chunk for {symbol}: {e}")
                    current_date = chunk_end
                    continue
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data)
                df['timestamp'] = pd.to_datetime(df['date'])
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                # Save to file
                output_file = os.path.join(self.five_min_dir, f'{symbol}.csv')
                df.to_csv(output_file)
                
                logger.info(f"Downloaded {len(df)} 5-minute candles for {symbol}")
                return df
            
        except Exception as e:
            logger.error(f"Failed to download 5-minute data for {symbol}: {e}")
        
        return None
    
    def download_daily_data(self, symbol, days_back=1095):
        """Download daily data separately (longer history)"""
        try:
            logger.info(f"Downloading daily data for {symbol} ({days_back} days)")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Daily data in one request
            data = self.kite.historical_data(
                instrument_token=symbol,
                from_date=start_date,
                to_date=end_date,
                interval='day'
            )
            
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['date'])
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                # Save to file
                output_file = os.path.join(self.daily_dir, f'{symbol}.csv')
                df.to_csv(output_file)
                
                logger.info(f"Downloaded {len(df)} daily candles for {symbol}")
                return df
            
        except Exception as e:
            logger.error(f"Failed to download daily data for {symbol}: {e}")
        
        return None
    
    def download_all_nifty50(self, days_back=365):
        """Download 5-minute data for all NIFTY50 stocks (optimized)"""
        from universe.symbols import load_universe
        
        universe = load_universe()
        nifty50 = universe[universe['In_NIFTY50'] == 'Y']
        
        logger.info(f"Downloading 5-minute data for {len(nifty50)} NIFTY50 stocks")
        
        success_count = 0
        for idx, row in nifty50.iterrows():
            symbol = row['symbol']
            try:
                self.download_five_minute_data(symbol, days_back)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
        
        logger.info(f"Downloaded 5-minute data for {success_count}/{len(nifty50)} stocks")
        
        # Auto-aggregate to other timeframes
        logger.info("Auto-aggregating 5-minute data to other timeframes...")
        self.aggregator.process_all_stocks()
        
        # Download daily data
        logger.info("Downloading daily data...")
        daily_success = 0
        for idx, row in nifty50.iterrows():
            symbol = row['symbol']
            try:
                self.download_daily_data(symbol, 1095)  # 3 years
                daily_success += 1
            except Exception as e:
                logger.error(f"Failed to download daily data for {symbol}: {e}")
        
        logger.info(f"Download complete: {success_count} 5-min stocks, {daily_success} daily stocks")
        return success_count == len(nifty50)
    
    def update_intraday_data(self, symbols=None):
        """Quick update of intraday 5-minute data (incremental)"""
        if symbols is None:
            # Get list of existing symbols
            if os.path.exists(self.five_min_dir):
                symbols = [f.replace('.csv', '') for f in os.listdir(self.five_min_dir) if f.endswith('.csv')]
            else:
                logger.warning("No existing data to update")
                return False
        
        logger.info(f"Updating 5-minute data for {len(symbols)} symbols")
        
        success_count = 0
        for symbol in symbols:
            try:
                # Download just today's data
                self.download_five_minute_data(symbol, days_back=1)
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to update {symbol}: {e}")
        
        # Re-aggregate for consistency
        logger.info("Re-aggregating updated data...")
        self.aggregator.process_all_stocks()
        
        logger.info(f"Updated {success_count}/{len(symbols)} symbols")
        return success_count > 0
    
    def get_data_status(self):
        """Get status of downloaded data"""
        status = {
            '5minute': 0,
            '10minute': 0,
            '15minute': 0,
            '30minute': 0,
            '60minute': 0,
            '2hour': 0,
            '4hour': 0,
            'daily': 0,
        }
        
        # Count files
        for tf, dir_path in [
            ('5minute', self.five_min_dir),
            ('10minute', os.path.join(self.base_dir, '10minute')),
            ('15minute', os.path.join(self.base_dir, '15minute')),
            ('30minute', os.path.join(self.base_dir, '30minute')),
            ('60minute', os.path.join(self.base_dir, '60minute')),
            ('2hour', os.path.join(self.base_dir, '2hour')),
            ('4hour', os.path.join(self.base_dir, '4hour')),
            ('daily', self.daily_dir),
        ]:
            if os.path.exists(dir_path):
                status[tf] = len([f for f in os.listdir(dir_path) if f.endswith('.csv')])
        
        return status
