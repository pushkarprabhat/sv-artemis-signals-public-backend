# utils/data_checker.py — DATA AVAILABILITY VERIFICATION
# Checks if historical data completeness for all stocks and timeframes
# Provides guidance on whether data download is needed

import pandas as pd
from pathlib import Path
from datetime import datetime
import os
from config import BASE_DIR
from typing import Dict, List, Tuple

class DataChecker:
    """Verify data availability in different segments"""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.timeframes = ['day', '15minute', '30minute', '60minute', 'week', 'month']
        
    def check_segment_data(self, segment='cash', timeframe='day'):
        """
        Check data availability in a specific segment
        
        Args:
            segment: 'cash', 'futures', or 'options'
            timeframe: 'day', '15minute', '30minute', '60minute'
        
        Returns:
            Dictionary with availability info
        """
        segment_folders = {
            'cash': ['day', '15minute', '30minute', '60minute', 'week', 'month'],
            'futures': ['futures'],
            'options': ['options', 'option_chains'],
        }
        
        folders_to_check = segment_folders.get(segment, [])
        results = {}
        
        for folder in folders_to_check:
            folder_path = self.base_dir / folder
            if folder_path.exists():
                files = list(folder_path.glob('*.parquet'))
                
                if files:
                    # Check file dates
                    recent_files = []
                    for f in files:
                        try:
                            mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d")
                            if mtime == self.today_str:
                                recent_files.append(f.name)
                        except:
                            pass
                    
                    results[folder] = {
                        'total_files': len(files),
                        'today_files': len(recent_files),
                        'latest_update': datetime.fromtimestamp(
                            os.path.getmtime(files[-1])
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        'sample_files': [f.name for f in files[-3:]],
                    }
                else:
                    results[folder] = {
                        'total_files': 0,
                        'today_files': 0,
                        'status': 'EMPTY'
                    }
            else:
                results[folder] = {'status': 'NOT_FOUND'}
        
        return results
    
    def check_index_data(self, index_symbol, segment='cash'):
        """
        Check if data exists for a specific index
        
        Args:
            index_symbol: e.g., 'NIFTY50', 'NIFTYBANK'
            segment: 'cash', 'futures', or 'options'
        
        Returns:
            Dictionary with index data status
        """
        segment_folders = {
            'cash': ['day', '15minute', '30minute', '60minute'],
            'futures': ['futures'],
            'options': ['options', 'option_chains'],
        }
        
        status = {
            'index': index_symbol,
            'segment': segment,
            'timeframes': {}
        }
        
        for folder in segment_folders.get(segment, []):
            folder_path = self.base_dir / folder
            
            # Try different name variations
            possible_names = [
                f"{index_symbol}.parquet",
                f"{index_symbol}_FUT.parquet",
                f"{index_symbol}_OPT.parquet",
            ]
            
            file_exists = False
            for name in possible_names:
                file_path = folder_path / name
                if file_path.exists():
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        is_today = mtime.strftime("%Y-%m-%d") == self.today_str
                        
                        status['timeframes'][folder] = {
                            'exists': True,
                            'filename': name,
                            'last_update': mtime.strftime("%Y-%m-%d %H:%M:%S"),
                            'is_today': is_today,
                        }
                        file_exists = True
                        break
                    except:
                        pass
            
            if not file_exists:
                status['timeframes'][folder] = {'exists': False}
        
        return status
    
    def get_available_indices(self, segment='cash'):
        """
        Get list of indices with available data
        
        Args:
            segment: 'cash', 'futures', or 'options'
        
        Returns:
            List of available indices
        """
        from universe.symbols.indices_universe import INDICES_METADATA
        
        available = []
        
        for index_name, metadata in INDICES_METADATA.items():
            status = self.check_index_data(index_name, segment)
            
            # Check if any timeframe has data
            has_data = any(
                tf_status.get('exists', False) 
                for tf_status in status['timeframes'].values()
            )
            
            if has_data:
                available.append({
                    'name': index_name,
                    'full_name': metadata['full_name'],
                    'segment': metadata['segment'],
                    'status': status
                })
        
        return available
    
    def get_summary(self):
        """Get comprehensive data availability summary"""
        print("=" * 80)
        print("DATA AVAILABILITY SUMMARY")
        print("=" * 80)
        print(f"\nCurrent Date: {self.today_str}")
        
        # Cash Segment
        print("\n1. CASH SEGMENT (Stocks & Indices):")
        print("-" * 80)
        cash_data = self.check_segment_data('cash', 'day')
        for tf, info in cash_data.items():
            if info.get('status') == 'EMPTY':
                print(f"  {tf:15} : No data")
            elif info.get('status') == 'NOT_FOUND':
                print(f"  {tf:15} : Folder not found")
            else:
                today_indicator = "TODAY" if info['today_files'] > 0 else "OLD"
                print(f"  {tf:15} : {info['total_files']:4} files | Today: {info['today_files']:3} | Updated: {info['latest_update']} [{today_indicator}]")
        
        # Futures Segment
        print("\n2. FUTURES SEGMENT (Index & Stock Futures):")
        print("-" * 80)
        futures_data = self.check_segment_data('futures')
        for tf, info in futures_data.items():
            if info.get('status') == 'EMPTY':
                print(f"  {tf:15} : No data")
            elif info.get('status') == 'NOT_FOUND':
                print(f"  {tf:15} : Folder not found")
            else:
                today_indicator = "TODAY" if info['today_files'] > 0 else "OLD"
                print(f"  {tf:15} : {info['total_files']:4} files | Today: {info['today_files']:3} | Updated: {info['latest_update']} [{today_indicator}]")
        
        # Options Segment
        print("\n3. OPTIONS SEGMENT (Index Options):")
        print("-" * 80)
        options_data = self.check_segment_data('options')
        for tf, info in options_data.items():
            if info.get('status') == 'EMPTY':
                print(f"  {tf:15} : No data")
            elif info.get('status') == 'NOT_FOUND':
                print(f"  {tf:15} : Folder not found")
            else:
                today_indicator = "TODAY" if info['today_files'] > 0 else "OLD"
                print(f"  {tf:15} : {info['total_files']:4} files | Today: {info['today_files']:3} | Updated: {info['latest_update']} [{today_indicator}]")
    
    def check_nifty50_completeness(self) -> Dict:
        """Check if NIFTY50 data is complete across all timeframes
        
        Returns:
            {
                'total_stocks': int,
                'stocks_with_all_data': int,
                'stocks_missing_data': List[str],
                'timeframes': {tf: {'files': int, 'missing': int}},
                'download_required': bool,
                'data_size_mb': float
            }
        """
        try:
            from universe.symbols import load_universe
            
            universe = load_universe()
            nifty50_stocks = universe[universe.get('In_NIFTY50', 'N') == 'Y']['Symbol'].tolist()
            
            result = {
                'total_stocks': len(nifty50_stocks),
                'stocks_with_all_data': 0,
                'stocks_missing_data': [],
                'timeframes': {},
                'total_files': 0,
                'data_size_mb': 0.0,
                'download_required': False
            }
            
            # Check each timeframe
            for tf in self.timeframes:
                # Check both naming conventions
                tf_dir = self.base_dir / tf
                tf_dir_alt = self.base_dir / tf.replace('minute', 'min')
                
                if tf_dir.exists():
                    use_dir = tf_dir
                elif tf_dir_alt.exists():
                    use_dir = tf_dir_alt
                else:
                    result['timeframes'][tf] = {
                        'exists': False,
                        'files': 0,
                        'missing': len(nifty50_stocks),
                        'status': '❌ Directory missing'
                    }
                    result['download_required'] = True
                    continue
                
                # Count files in this timeframe
                files = list(use_dir.glob('*.parquet')) + list(use_dir.glob('*.csv'))
                result['timeframes'][tf] = {
                    'exists': True,
                    'files': len(files),
                    'missing': max(0, len(nifty50_stocks) - len(files)),
                    'status': '✅ Complete' if len(files) >= len(nifty50_stocks) else '⚠️  Partial'
                }
                result['total_files'] += len(files)
                
                # Calculate size
                for f in files:
                    if f.exists():
                        result['data_size_mb'] += f.stat().st_size / (1024 * 1024)
            
            # Check which stocks have data across ALL timeframes
            for stock in nifty50_stocks:
                has_all = True
                for tf in self.timeframes:
                    tf_dir = self.base_dir / tf
                    tf_dir_alt = self.base_dir / tf.replace('minute', 'min')
                    
                    if tf_dir.exists():
                        check_dir = tf_dir
                    elif tf_dir_alt.exists():
                        check_dir = tf_dir_alt
                    else:
                        has_all = False
                        break
                    
                    if not list(check_dir.glob(f"{stock}.*")):
                        has_all = False
                        break
                
                if has_all:
                    result['stocks_with_all_data'] += 1
                else:
                    result['stocks_missing_data'].append(stock)
            
            result['download_required'] = len(result['stocks_missing_data']) > 0
            
            return result
        except Exception as e:
            return {
                'total_stocks': 0,
                'stocks_with_all_data': 0,
                'stocks_missing_data': [],
                'timeframes': {},
                'total_files': 0,
                'data_size_mb': 0.0,
                'download_required': True,
                'error': str(e)
            }
    
    def get_completeness_summary(self) -> str:
        """Get human-readable summary of data completeness"""
        status = self.check_nifty50_completeness()
        
        if 'error' in status:
            return f"❌ Could not check completeness: {status['error']}"
        
        lines = [
            f"📊 DATA COMPLETENESS REPORT",
            f"{'='*60}",
            f"",
            f"Total NIFTY50 Stocks: {status['total_stocks']}",
            f"Stocks with ALL data: {status['stocks_with_all_data']} ({status['stocks_with_all_data']/max(1,status['total_stocks'])*100:.1f}%)",
            f"Stocks missing data: {len(status['stocks_missing_data'])}",
            f"Total files: {status['total_files']:,}",
            f"Data size: {status['data_size_mb']:.1f} MB",
            f"",
            f"TIMEFRAME STATUS:",
            f"{'─'*60}",
        ]
        
        for tf in self.timeframes:
            tf_status = status['timeframes'].get(tf, {})
            if tf_status.get('exists'):
                lines.append(
                    f"  {tf:8} {tf_status['status']:20} {tf_status['files']:3} files, "
                    f"{tf_status['missing']:3} missing"
                )
            else:
                lines.append(f"  {tf:8} {tf_status['status']:20} - no data folder")
        
        lines.extend([
            f"",
            f"DOWNLOAD STATUS:",
            f"{'─'*60}",
        ])
        
        if status['download_required']:
            lines.append(f"❌ DOWNLOAD REQUIRED")
            lines.append(f"   Missing {len(status['stocks_missing_data'])} stocks")
            if status['stocks_missing_data']:
                sample = status['stocks_missing_data'][:5]
                lines.append(f"   Example: {', '.join(sample)}")
                if len(status['stocks_missing_data']) > 5:
                    lines.append(f"   ... and {len(status['stocks_missing_data'])-5} more")
        else:
            lines.append(f"✅ ALL DATA AVAILABLE - Ready for live scanning!")
        
        return "\n".join(lines)
    
    def needs_download(self) -> bool:
        """Simple check - is download needed?"""
        status = self.check_nifty50_completeness()
        return status.get('download_required', True)


if __name__ == "__main__":
    checker = DataChecker()
    checker.get_summary()
