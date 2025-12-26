#!/usr/bin/env python3
"""
Data Quality Validator
Validates downloaded OHLCV data for:
- Consistency (high >= low, open/close within range)
- Gaps and missing dates
- Anomalies (extreme moves, volume spikes)
- Statistical outliers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from config import get_download_dir
from utils.logger import logger
import datetime as dt


class DataQualityValidator:
    """Validates OHLCV data quality"""
    
    def __init__(self, interval: str = 'day'):
        """
        Initialize validator
        Args:
            interval: 'day', '3minute', '5minute'
        """
        self.interval = interval
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def validate_file(self, file_path: Path) -> Dict:
        """Validate a single OHLCV file"""
        self.issues = []
        self.warnings = []
        self.stats = {}
        
        try:
            df = pd.read_parquet(file_path)
            symbol = file_path.stem
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'file': file_path.name,
                    'valid': False,
                    'reason': 'Empty file',
                    'issues': ['No data rows']
                }
            
            # Run validations
            self._check_columns(df, symbol)
            self._check_ohlcv_consistency(df, symbol)
            self._check_date_gaps(df, symbol)
            self._check_anomalies(df, symbol)
            
            valid = len(self.issues) == 0
            
            return {
                'symbol': symbol,
                'file': file_path.name,
                'valid': valid,
                'rows': len(df),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'issues': self.issues,
                'warnings': self.warnings,
                'stats': self.stats
            }
        
        except Exception as e:
            return {
                'symbol': file_path.stem,
                'file': file_path.name,
                'valid': False,
                'reason': f'Error reading file: {e}',
                'issues': [str(e)]
            }
    
    def _check_columns(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for required columns"""
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            self.issues.append(f"Missing columns: {', '.join(missing)}")
            return
        
        self.stats['columns_present'] = len(df.columns)
    
    def _check_ohlcv_consistency(self, df: pd.DataFrame, symbol: str) -> None:
        """Check OHLCV relationships"""
        issues = []
        
        # Check high >= low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            issues.append(f"{len(invalid_hl)} rows: high < low")
        
        # Check high >= open/close
        invalid_ho = df[df['high'] < df['open']]
        if len(invalid_ho) > 0:
            issues.append(f"{len(invalid_ho)} rows: high < open")
        
        invalid_hc = df[df['high'] < df['close']]
        if len(invalid_hc) > 0:
            issues.append(f"{len(invalid_hc)} rows: high < close")
        
        # Check low <= open/close
        invalid_lo = df[df['low'] > df['open']]
        if len(invalid_lo) > 0:
            issues.append(f"{len(invalid_lo)} rows: low > open")
        
        invalid_lc = df[df['low'] > df['close']]
        if len(invalid_lc) > 0:
            issues.append(f"{len(invalid_lc)} rows: low > close")
        
        # Check for NaN values
        nan_counts = df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
        if nan_counts.sum() > 0:
            for col, count in nan_counts.items():
                if count > 0:
                    issues.append(f"{count} NaN values in {col}")
        
        if issues:
            self.issues.extend(issues)
        else:
            self.stats['ohlcv_consistency'] = 'OK'
    
    def _check_date_gaps(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for date gaps"""
        df_sorted = df.sort_values('date').copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
        # Check for duplicate dates
        duplicates = df_sorted['date'].duplicated().sum()
        if duplicates > 0:
            self.issues.append(f"{duplicates} duplicate dates")
        
        # Check for gaps
        date_diff = df_sorted['date'].diff()
        
        if self.interval == 'day':
            # For daily data, expect 1 day gaps (or weekends)
            expected_diff = pd.Timedelta(days=1)
            tolerance = pd.Timedelta(days=3)  # Allow for weekends/holidays
        elif self.interval == '3minute':
            expected_diff = pd.Timedelta(minutes=3)
            tolerance = pd.Timedelta(minutes=15)  # Allow some gaps
        else:  # 5minute
            expected_diff = pd.Timedelta(minutes=5)
            tolerance = pd.Timedelta(minutes=20)
        
        gaps = date_diff[date_diff > tolerance]
        if len(gaps) > 0:
            gap_count = len(gaps)
            max_gap = gaps.max()
            self.warnings.append(f"{gap_count} gaps detected, max: {max_gap}")
            self.stats['gap_count'] = gap_count
        else:
            self.stats['date_continuity'] = 'OK'
    
    def _check_anomalies(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for statistical anomalies"""
        warnings = []
        
        # Calculate returns
        df_temp = df.copy()
        df_temp['return'] = df_temp['close'].pct_change() * 100
        
        # Check for extreme moves (>20% single candle)
        extreme_moves = df_temp[abs(df_temp['return']) > 20]
        if len(extreme_moves) > 0:
            warnings.append(f"{len(extreme_moves)} extreme price moves (>20%)")
        
        # Check for zero prices
        zero_prices = df[(df['open'] == 0) | (df['close'] == 0)]
        if len(zero_prices) > 0:
            warnings.append(f"{len(zero_prices)} rows with zero prices")
        
        # Check for zero volume (some stocks might have this legitimately)
        zero_volume = df[df['volume'] == 0]
        if len(zero_volume) > len(df) * 0.1:  # More than 10% zero volume
            warnings.append(f"{len(zero_volume)} rows with zero volume ({len(zero_volume)/len(df)*100:.1f}%)")
        
        # Statistical outliers in volume
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        if volume_std > 0:
            volume_z = (df['volume'] - volume_mean) / volume_std
            outliers = (abs(volume_z) > 5).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                self.warnings.append(f"{outliers} volume outliers detected")
        
        self.warnings.extend(warnings)
    
    def validate_directory(self, directory: Path, pattern: str = '*.parquet') -> Dict:
        """Validate all files in directory"""
        files = list(directory.glob(pattern))
        
        if not files:
            logger.warning(f"[VALIDATOR] No files found in {directory}")
            return {'files': 0, 'valid': 0, 'invalid': 0, 'results': []}
        
        results = []
        valid_count = 0
        invalid_count = 0
        
        for file_path in sorted(files):
            result = self.validate_file(file_path)
            results.append(result)
            
            if result.get('valid', False):
                valid_count += 1
            else:
                invalid_count += 1
        
        return {
            'interval': self.interval,
            'directory': str(directory),
            'files': len(files),
            'valid': valid_count,
            'invalid': invalid_count,
            'validity_rate': f"{valid_count/len(files)*100:.1f}%",
            'results': results
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate validation report"""
        report = []
        report.append("\n" + "="*70)
        report.append(f"DATA QUALITY VALIDATION REPORT - {self.interval.upper()}")
        report.append("="*70)
        
        if 'files' in results:
            # Directory validation report
            report.append(f"\nTotal files: {results['files']}")
            report.append(f"Valid: {results['valid']} ({results['validity_rate']})")
            report.append(f"Invalid: {results['invalid']}")
            
            if results['invalid'] > 0:
                report.append("\nInvalid files:")
                for r in results['results']:
                    if not r.get('valid', True):
                        report.append(f"  {r['symbol']}: {', '.join(r.get('issues', ['Unknown']))}")
            
            if any(r.get('warnings') for r in results['results']):
                report.append("\nWarnings:")
                for r in results['results']:
                    if r.get('warnings'):
                        report.append(f"  {r['symbol']}: {', '.join(r['warnings'])}")
        else:
            # Single file validation report
            report.append(f"\nSymbol: {results.get('symbol', 'Unknown')}")
            report.append(f"File: {results.get('file', 'Unknown')}")
            report.append(f"Valid: {results.get('valid', False)}")
            report.append(f"Rows: {results.get('rows', 0)}")
            
            if results.get('issues'):
                report.append(f"Issues: {', '.join(results['issues'])}")
            if results.get('warnings'):
                report.append(f"Warnings: {', '.join(results['warnings'])}")
        
        report.append("="*70 + "\n")
        return "\n".join(report)


def validate_downloaded_data(interval: str = 'day') -> Dict:
    """Validate all downloaded data for an interval"""
    validator = DataQualityValidator(interval)
    data_dir = get_download_dir('NSE', interval)
    
    logger.info(f"[VALIDATOR] Validating {interval} data in {data_dir}")
    
    results = validator.validate_directory(data_dir)
    report = validator.generate_report(results)
    
    logger.info(report)
    
    return results


def validate_all_intervals() -> Dict:
    """Validate all downloaded intervals"""
    all_results = {}
    
    for interval in ['day', '3minute', '5minute']:
        results = validate_downloaded_data(interval)
        all_results[interval] = results
    
    return all_results
