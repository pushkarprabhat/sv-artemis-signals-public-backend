#!/usr/bin/env python3
"""
Instrument Metadata Aggregation
Maintains: segment, exchange, symbol, expiry_date, expiry_type, lot_size, instrument_type, tick_size
This is SEGMENT-EXCHANGE-SYMBOL-EXPIRY-DATA table
Key Feature: Lot size can differ by expiry for same symbol
"""

import pandas as pd
import os
from pathlib import Path
from universe.symbols import load_universe
from utils.helpers import get_expiry_type_by_month
from utils.logger import logger

def build_segment_exchange_symbol_expiry_data():
    """
    Build aggregated metadata table from universe
    
    Returns aggregation with columns:
    - segment: NFO-FUT, NFO-OPT, NSE
    - exchange: NSE, NFO
    - symbol: SBIN (base symbol)
    - instrument: SBIN25DECFUT (full trading symbol)
    - instrument_type: EQ, FUT, CE, PE
    - expiry_date: 2025-12-30 (NULL for EQ)
    - expiry_type: MONTHLY or WEEKLY (NULL for EQ)
    - lot_size: Contract size (can vary by expiry!)
    - tick_size: Minimum price movement
    - instrument_token: API token
    - exchange_token: Exchange-specific token
    """
    
    print("=" * 80)
    print("BUILDING SEGMENT-EXCHANGE-SYMBOL-EXPIRY-DATA TABLE")
    print("=" * 80)
    
    # Load full universe
    universe = load_universe()
    print(f"\nLoaded {len(universe)} instruments from universe")
    print(f"Available columns: {universe.columns.tolist()}")
    
    # Prepare aggregation
    agg_data = []
    
    # Ensure we have consistent column names
    required_cols = {
        'symbol': ['Symbol'],
        'instrument': ['Trading Symbol', 'tradingsymbol'],
        'segment': ['Segment', 'segment'],
        'exchange': ['Exchange', 'exchange'],
        'lot_size': ['Lot Size', 'lot_size'],
        'tick_size': ['Tick Size', 'tick_size'],
        'instrument_type': ['Instrument Type', 'instrument_type'],
        'expiry': ['Expiry', 'expiry'],
        'instrument_token': ['Instrument Token', 'instrument_token'],
        'exchange_token': ['Exchange Token', 'exchange_token'],
    }
    
    # Get first available column for each field
    col_map = {}
    for field, possible_cols in required_cols.items():
        for col in possible_cols:
            if col in universe.columns:
                col_map[field] = col
                break
    
    print(f"\nColumn mapping: {col_map}")
    
    for idx, row in universe.iterrows():
        try:
            # Extract values using mapped columns
            symbol = row[col_map.get('symbol')] if 'symbol' in col_map else 'UNKNOWN'
            instrument = row[col_map.get('instrument')] if 'instrument' in col_map else symbol
            segment = row[col_map.get('segment')] if 'segment' in col_map else 'NSE'
            exchange = row[col_map.get('exchange')] if 'exchange' in col_map else 'NSE'
            lot_size = row[col_map.get('lot_size')] if 'lot_size' in col_map else 1
            tick_size = row[col_map.get('tick_size')] if 'tick_size' in col_map else 0.05
            inst_type = row[col_map.get('instrument_type')] if 'instrument_type' in col_map else 'EQ'
            instrument_token = row[col_map.get('instrument_token')] if 'instrument_token' in col_map else None
            exchange_token = row[col_map.get('exchange_token')] if 'exchange_token' in col_map else None
            
            # Build record
            record = {
                'segment': segment,
                'exchange': exchange,
                'symbol': symbol,
                'instrument': instrument,
                'instrument_type': inst_type,
                'lot_size': lot_size,
                'tick_size': tick_size,
                'instrument_token': instrument_token,
                'exchange_token': exchange_token,
            }
            
            # Add expiry info if present
            if 'expiry' in col_map and pd.notna(row[col_map['expiry']]):
                expiry_date = pd.to_datetime(row[col_map['expiry']])
                record['expiry_date'] = expiry_date
                
                # Get all expiry dates in same month for classification
                expiry_col = col_map['expiry']
                same_month_mask = (
                    (universe[col_map['symbol']] == symbol) &
                    (universe[col_map['segment']] == segment) &
                    (universe[col_map['exchange']] == exchange) &
                    (pd.to_datetime(universe[expiry_col]).dt.to_period('M') == expiry_date.to_period('M'))
                )
                same_month_expiries = pd.to_datetime(universe[same_month_mask][expiry_col]).unique()
                
                # Classify expiry
                record['expiry_type'] = get_expiry_type_by_month(expiry_date, same_month_expiries)
            else:
                record['expiry_date'] = pd.NaT
                record['expiry_type'] = None
            
            agg_data.append(record)
            
        except Exception as e:
            logger.warning(f"Could not process row: {e}")
            continue
    
    # Create DataFrame
    agg_df = pd.DataFrame(agg_data)
    
    # Remove duplicates (same segment-exchange-symbol-expiry can have multiple strikes)
    # Keep first occurrence
    group_cols = ['segment', 'exchange', 'symbol', 'expiry_date']
    agg_df = agg_df.drop_duplicates(subset=group_cols, keep='first')
    
    # Sort for readability
    agg_df = agg_df.sort_values(
        by=['segment', 'exchange', 'symbol', 'expiry_date'],
        na_position='last'
    ).reset_index(drop=True)
    
    print(f"\nAggregated to {len(agg_df)} unique segment-exchange-symbol-expiry combinations")
    print(f"\nColumns in aggregated table:")
    print(agg_df.columns.tolist())
    
    # Save to parquet
    output_dir = Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'segment_exchange_symbol_expiry_data.parquet'
    
    agg_df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"\nSaved to: {output_file}")
    
    return agg_df


def get_lot_size(symbol, segment, exchange, expiry_date=None):
    """
    Get lot size for a specific symbol-segment-exchange-expiry combination
    
    Args:
        symbol: Base symbol (SBIN)
        segment: Segment (NFO-FUT, NFO-OPT, NSE)
        exchange: Exchange (NSE, NFO)
        expiry_date: Optional expiry date for derivatives
    
    Returns:
        int: Lot size
    """
    try:
        agg_file = Path('data/segment_exchange_symbol_expiry_data.parquet')
        if not agg_file.exists():
            print("Warning: Aggregation table not found, rebuilding...")
            agg_df = build_segment_exchange_symbol_expiry_data()
        else:
            agg_df = pd.read_parquet(agg_file)
        
        # Filter by symbol, segment, exchange
        mask = (agg_df['symbol'] == symbol) & \
               (agg_df['segment'] == segment) & \
               (agg_df['exchange'] == exchange)
        
        # If expiry specified, filter further
        if expiry_date is not None:
            expiry_date = pd.to_datetime(expiry_date)
            mask = mask & (agg_df['expiry_date'] == expiry_date)
        
        matches = agg_df[mask]
        
        if len(matches) > 0:
            return int(matches.iloc[0]['lot_size'])
        else:
            logger.warning(f"No lot size found for {symbol} {segment} {exchange}")
            return 1  # Default
    
    except Exception as e:
        logger.error(f"Error getting lot size: {e}")
        return 1  # Default


def get_tick_size(symbol, segment, exchange, expiry_date=None):
    """
    Get tick size for a specific symbol-segment-exchange-expiry combination
    
    Args:
        symbol: Base symbol (SBIN)
        segment: Segment (NFO-FUT, NFO-OPT, NSE)
        exchange: Exchange (NSE, NFO)
        expiry_date: Optional expiry date for derivatives
    
    Returns:
        float: Tick size
    """
    try:
        agg_file = Path('data/segment_exchange_symbol_expiry_data.parquet')
        if not agg_file.exists():
            agg_df = build_segment_exchange_symbol_expiry_data()
        else:
            agg_df = pd.read_parquet(agg_file)
        
        mask = (agg_df['symbol'] == symbol) & \
               (agg_df['segment'] == segment) & \
               (agg_df['exchange'] == exchange)
        
        if expiry_date is not None:
            expiry_date = pd.to_datetime(expiry_date)
            mask = mask & (agg_df['expiry_date'] == expiry_date)
        
        matches = agg_df[mask]
        
        if len(matches) > 0:
            return float(matches.iloc[0]['tick_size'])
        else:
            logger.warning(f"No tick size found for {symbol} {segment} {exchange}")
            return 0.05  # Default
    
    except Exception as e:
        logger.error(f"Error getting tick size: {e}")
        return 0.05  # Default


if __name__ == '__main__':
    # Build the aggregation table
    agg_df = build_segment_exchange_symbol_expiry_data()
    
    # Show samples
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    
    print("\nFirst 10 rows:")
    print(agg_df.head(10).to_string())
    
    print("\n\nSBIN-related records:")
    sbin_data = agg_df[agg_df['symbol'] == 'SBIN']
    print(f"Found {len(sbin_data)} SBIN records")
    if len(sbin_data) > 0:
        print(sbin_data[['symbol', 'segment', 'exchange', 'instrument_type', 'expiry_date', 'expiry_type', 'lot_size']].to_string())
    
    print("\n" + "=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print(agg_df.dtypes)
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total records: {len(agg_df)}")
    print(f"\nBy segment:")
    print(agg_df['segment'].value_counts())
    print(f"\nBy instrument type:")
    print(agg_df['instrument_type'].value_counts())
    print(f"\nLot size range: {agg_df['lot_size'].min()} to {agg_df['lot_size'].max()}")
    print(f"Tick size range: {agg_df['tick_size'].min()} to {agg_df['tick_size'].max()}")
