# core/data_aggregator.py — OHLCV Aggregation Engine
# Aggregates 5-minute data into 15min, 30min, 60min, 120min, 180min, 240min
# Market hours: 09:15 - 15:30 IST

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from config import BASE_DIR
from utils.logger import logger

# Market hours (IST)
MARKET_OPEN = 9 * 60 + 15  # 09:15 in minutes
MARKET_CLOSE = 15 * 60 + 30  # 15:30 in minutes

def get_5min_candles_per_day():
    """Calculate number of 5-min candles in a trading day"""
    return (MARKET_CLOSE - MARKET_OPEN) // 5  # 78 candles per day

def aggregate_5min_to_interval(df_5min, interval_minutes):
    """
    Aggregate 5-minute OHLCV data to target interval
    
    Args:
        df_5min: DataFrame with 5-min candles (columns: date, open, high, low, close, volume)
        interval_minutes: Target interval (15, 30, 60, 120, 180, 240)
    
    Returns:
        DataFrame with aggregated candles
    """
    if df_5min.empty:
        return pd.DataFrame()
    
    # Ensure date is datetime
    df = df_5min.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract date and time components
    df['trading_date'] = df['date'].dt.date
    df['time'] = df['date'].dt.time
    df['minutes_from_open'] = (df['date'].dt.hour * 60 + df['date'].dt.minute - MARKET_OPEN)
    
    # For each date, group candles into intervals starting from market open
    aggregated = []
    
    for trading_date, group in df.groupby('trading_date'):
        # Sort by time to ensure correct aggregation
        group = group.sort_values('date').reset_index(drop=True)
        
        # Group by interval bucket: bucket = minutes_from_open // interval_minutes
        group['interval_bucket'] = group['minutes_from_open'] // interval_minutes
        
        # Aggregate
        for bucket, candles in group.groupby('interval_bucket'):
            if len(candles) == 0:
                continue
            
            agg_candle = {
                'date': candles.iloc[0]['date'],  # First candle's time
                'open': candles.iloc[0]['open'],
                'high': candles['high'].max(),
                'low': candles['low'].min(),
                'close': candles.iloc[-1]['close'],
                'volume': candles['volume'].sum()
            }
            aggregated.append(agg_candle)
    
    result = pd.DataFrame(aggregated)
    
    if not result.empty:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    
    return result


def aggregate_all_intervals(symbol):
    """
    Load 5-min data and aggregate to all required intervals
    
    Args:
        symbol: Trading symbol (e.g., 'TCS')
    
    Returns:
        dict: {interval: DataFrame} for all aggregated intervals
    """
    result = {}
    
    # Load 5-minute data
    file_5min = BASE_DIR / "5minute" / f"{symbol}.parquet"
    
    if not file_5min.exists():
        logger.warning(f"5-minute data not found for {symbol}")
        return result
    
    try:
        df_5min = pd.read_parquet(file_5min)
        logger.info(f"Loaded {len(df_5min)} 5-min candles for {symbol}")
        
        # Aggregate to each target interval
        target_intervals = [15, 30, 60, 120, 180, 240]
        
        for interval in target_intervals:
            try:
                df_agg = aggregate_5min_to_interval(df_5min, interval)
                
                # Save to parquet
                folder = BASE_DIR / f"{interval}minute"
                folder.mkdir(parents=True, exist_ok=True)
                output_file = folder / f"{symbol}.parquet"
                
                df_agg.to_parquet(output_file, index=False, engine='pyarrow')
                result[f"{interval}minute"] = df_agg
                
                logger.info(f"Aggregated {symbol} to {interval}min: {len(df_agg)} candles")
                
            except Exception as e:
                logger.error(f"Failed to aggregate {symbol} to {interval}min: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to load 5-min data for {symbol}: {e}")
        return result


def validate_aggregation(symbol, interval_minutes):
    """
    Validate that aggregated data is correct
    
    Args:
        symbol: Trading symbol
        interval_minutes: Aggregated interval (e.g., 15, 30, 60)
    
    Returns:
        dict: Validation report
    """
    report = {'valid': True, 'errors': []}
    
    try:
        # Load 5-min and aggregated data
        df_5min = pd.read_parquet(BASE_DIR / "5minute" / f"{symbol}.parquet")
        df_agg = pd.read_parquet(BASE_DIR / f"{interval_minutes}minute" / f"{symbol}.parquet")
        
        if df_agg.empty:
            report['valid'] = False
            report['errors'].append(f"No aggregated data found")
            return report
        
        # Validate OHLC ordering
        if (df_agg['open'] > df_agg['high']).any() or (df_agg['low'] > df_agg['close']).any():
            report['valid'] = False
            report['errors'].append("Invalid OHLC values (open > high or low > close)")
        
        # Validate volume is sum of 5-min volumes
        df_5min['date'] = pd.to_datetime(df_5min['date'])
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        
        # Check first aggregated candle's volume
        first_agg_time = df_agg.iloc[0]['date']
        time_window_end = first_agg_time + timedelta(minutes=interval_minutes)
        
        vol_5min = df_5min[
            (df_5min['date'] >= first_agg_time) & 
            (df_5min['date'] < time_window_end)
        ]['volume'].sum()
        
        vol_agg = df_agg.iloc[0]['volume']
        
        if vol_5min > 0 and abs(vol_agg - vol_5min) > vol_5min * 0.01:  # Allow 1% variance
            report['valid'] = False
            report['errors'].append(
                f"Volume mismatch: 5min sum={vol_5min}, agg={vol_agg}"
            )
        
        # Validate no data gaps
        date_diffs = df_agg['date'].diff()
        expected_diff = pd.Timedelta(minutes=interval_minutes)
        gaps = date_diffs[date_diffs > expected_diff * 1.5]  # Allow 50% overage for gaps
        
        if len(gaps) > 0:
            report['valid'] = False
            report['errors'].append(f"Data gaps detected: {len(gaps)} gaps")
        
        report['candles'] = len(df_agg)
        report['date_range'] = f"{df_agg['date'].min().date()} to {df_agg['date'].max().date()}"
        
    except Exception as e:
        report['valid'] = False
        report['errors'].append(f"Validation error: {str(e)}")
    
    return report


def aggregate_daily_to_weekly(df_daily):
    """
    Aggregate daily OHLCV data to weekly candles
    
    Args:
        df_daily: DataFrame with daily candles (columns: date, open, high, low, close, volume)
    
    Returns:
        DataFrame with weekly aggregated candles (candles on Fridays)
    """
    if df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Group by week (starting Monday, ending Sunday)
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.isocalendar().year
    
    aggregated = []
    
    for (year, week), group in df.groupby(['year', 'week']):
        if len(group) == 0:
            continue
        
        group = group.sort_values('date')
        agg_candle = {
            'date': group.iloc[-1]['date'],  # Last day of week (usually Friday)
            'open': group.iloc[0]['open'],
            'high': group['high'].max(),
            'low': group['low'].min(),
            'close': group.iloc[-1]['close'],
            'volume': group['volume'].sum()
        }
        aggregated.append(agg_candle)
    
    result = pd.DataFrame(aggregated)
    if not result.empty:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    
    return result


def aggregate_daily_to_monthly(df_daily):
    """
    Aggregate daily OHLCV data to monthly candles
    
    Args:
        df_daily: DataFrame with daily candles
    
    Returns:
        DataFrame with monthly aggregated candles (candles on last trading day of month)
    """
    if df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Group by year-month
    df['year_month'] = df['date'].dt.to_period('M')
    
    aggregated = []
    
    for period, group in df.groupby('year_month'):
        if len(group) == 0:
            continue
        
        group = group.sort_values('date')
        agg_candle = {
            'date': group.iloc[-1]['date'],  # Last day of month
            'open': group.iloc[0]['open'],
            'high': group['high'].max(),
            'low': group['low'].min(),
            'close': group.iloc[-1]['close'],
            'volume': group['volume'].sum()
        }
        aggregated.append(agg_candle)
    
    result = pd.DataFrame(aggregated)
    if not result.empty:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    
    return result


def aggregate_daily_to_quarterly(df_daily):
    """
    Aggregate daily OHLCV data to quarterly candles
    
    Args:
        df_daily: DataFrame with daily candles
    
    Returns:
        DataFrame with quarterly aggregated candles (candles on last trading day of quarter)
    """
    if df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Group by year-quarter
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    aggregated = []
    
    for (year, quarter), group in df.groupby(['year', 'quarter']):
        if len(group) == 0:
            continue
        
        group = group.sort_values('date')
        agg_candle = {
            'date': group.iloc[-1]['date'],  # Last day of quarter
            'open': group.iloc[0]['open'],
            'high': group['high'].max(),
            'low': group['low'].min(),
            'close': group.iloc[-1]['close'],
            'volume': group['volume'].sum()
        }
        aggregated.append(agg_candle)
    
    result = pd.DataFrame(aggregated)
    if not result.empty:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    
    return result


def aggregate_daily_to_yearly(df_daily):
    """
    Aggregate daily OHLCV data to yearly candles
    
    Args:
        df_daily: DataFrame with daily candles
    
    Returns:
        DataFrame with yearly aggregated candles (candles on last trading day of year)
    """
    if df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Group by year
    df['year'] = df['date'].dt.year
    
    aggregated = []
    
    for year, group in df.groupby('year'):
        if len(group) == 0:
            continue
        
        group = group.sort_values('date')
        agg_candle = {
            'date': group.iloc[-1]['date'],  # Last day of year
            'open': group.iloc[0]['open'],
            'high': group['high'].max(),
            'low': group['low'].min(),
            'close': group.iloc[-1]['close'],
            'volume': group['volume'].sum()
        }
        aggregated.append(agg_candle)
    
    result = pd.DataFrame(aggregated)
    if not result.empty:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    
    return result


def aggregate_daily_to_all_longer_intervals(symbol):
    """
    Load daily data and aggregate to weekly, monthly, quarterly, yearly
    
    Args:
        symbol: Trading symbol
    
    Returns:
        dict: {interval: DataFrame} for weekly, monthly, quarterly, yearly
    """
    result = {}
    
    # Load daily data
    file_daily = BASE_DIR / "day" / f"{symbol}.parquet"
    
    if not file_daily.exists():
        logger.warning(f"Daily data not found for {symbol}")
        return result
    
    try:
        df_daily = pd.read_parquet(file_daily)
        logger.info(f"Loaded {len(df_daily)} daily candles for {symbol}")
        
        # Define aggregation functions and output folders
        aggregations = {
            'week': (aggregate_daily_to_weekly, BASE_DIR / 'week'),
            'month': (aggregate_daily_to_monthly, BASE_DIR / 'month'),
            'quarter': (aggregate_daily_to_quarterly, BASE_DIR / 'quarter'),
            'year': (aggregate_daily_to_yearly, BASE_DIR / 'year')
        }
        
        for interval_name, (agg_func, folder) in aggregations.items():
            try:
                folder.mkdir(parents=True, exist_ok=True)
                output_file = folder / f"{symbol}.parquet"
                
                df_agg = agg_func(df_daily)
                
                if not df_agg.empty:
                    df_agg.to_parquet(output_file, index=False, engine='pyarrow')
                    result[interval_name] = df_agg
                    logger.info(f"Aggregated {symbol} to {interval_name}: {len(df_agg)} candles")
                else:
                    logger.warning(f"No {interval_name} candles generated for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to aggregate {symbol} to {interval_name}: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to load daily data for {symbol}: {e}")
        return result
