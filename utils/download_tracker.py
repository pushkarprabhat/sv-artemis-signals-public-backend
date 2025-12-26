"""
Download Tracker - Keeps log of last download date for each instrument
Allows skipping validation if data was recently downloaded
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class DownloadTracker:
    """Track when instruments were last downloaded"""
    
    def __init__(self, tracker_file: Path = None):
        """
        Initialize download tracker
        
        Args:
            tracker_file: Path to JSON file storing download history
                         Default: marketdata/.download_tracker.json
        """
        self.tracker_file = tracker_file or Path(__file__).parent.parent / 'marketdata' / '.download_tracker.json'
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load tracker data from disk"""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load tracker: {e}")
        
        return {
            'schema_version': '1.0',
            'created': datetime.now().isoformat(),
            'instruments': {}
        }
    
    def save(self):
        """Save tracker data to disk"""
        try:
            self.data['last_updated'] = datetime.now().isoformat()
            with open(self.tracker_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save tracker: {e}")
    
    def mark_downloaded(self, symbol: str, timeframe: str = None, status: str = 'success', 
                       candles_count: int = 0, file_path: str = None):
        """
        Mark a symbol as downloaded
        
        Args:
            symbol: Trading symbol (e.g., 'SBIN')
            timeframe: Timeframe (e.g., '5', 'day', '60')
            status: Download status ('success', 'failed', 'partial')
            candles_count: Number of candles downloaded
            file_path: Path to stored data file
        """
        if symbol not in self.data['instruments']:
            self.data['instruments'][symbol] = {
                'symbol': symbol,
                'timeframes': {}
            }
        
        if timeframe is None:
            timeframe = 'all'
        
        self.data['instruments'][symbol]['timeframes'][timeframe] = {
            'status': status,
            'last_downloaded': datetime.now().isoformat(),
            'candles_count': candles_count,
            'file_path': file_path,
        }
        
        self.save()
    
    def needs_download(self, symbol: str, timeframe: str = None, 
                      max_age_hours: int = 24) -> bool:
        """
        Check if symbol needs download (not downloaded recently)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (if None, checks if ANY timeframe was downloaded)
            max_age_hours: Maximum age of download before re-downloading (default 24 hours)
        
        Returns:
            True if symbol needs download, False otherwise
        """
        if symbol not in self.data['instruments']:
            return True
        
        if timeframe is None:
            timeframe = 'all'
        
        sym_data = self.data['instruments'][symbol]
        
        if timeframe not in sym_data.get('timeframes', {}):
            return True
        
        tf_data = sym_data['timeframes'][timeframe]
        
        # Check status
        if tf_data.get('status') != 'success':
            return True
        
        # Check age
        last_download = datetime.fromisoformat(tf_data.get('last_downloaded', '2000-01-01'))
        age_hours = (datetime.now() - last_download).total_seconds() / 3600
        
        return age_hours > max_age_hours
    
    def get_symbols_needing_download(self, symbols: List[str], timeframe: str = None,
                                    max_age_hours: int = 24) -> List[str]:
        """
        Filter symbols that need download
        
        Args:
            symbols: List of symbols to check
            timeframe: Timeframe (if None, checks if any timeframe was downloaded)
            max_age_hours: Maximum age before re-download
        
        Returns:
            List of symbols that need download
        """
        return [s for s in symbols if self.needs_download(s, timeframe, max_age_hours)]
    
    def get_last_download_info(self, symbol: str, timeframe: str = None) -> Optional[Dict]:
        """
        Get info about last download for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (if None, gets latest)
        
        Returns:
            Dict with download info or None
        """
        if symbol not in self.data['instruments']:
            return None
        
        if timeframe is None:
            timeframe = 'all'
        
        sym_data = self.data['instruments'][symbol]
        return sym_data.get('timeframes', {}).get(timeframe)
    
    def get_download_stats(self) -> Dict:
        """Get overall download statistics"""
        total_symbols = len(self.data['instruments'])
        successful = 0
        failed = 0
        partial = 0
        
        for sym_data in self.data['instruments'].values():
            for tf_data in sym_data.get('timeframes', {}).values():
                status = tf_data.get('status')
                if status == 'success':
                    successful += 1
                elif status == 'failed':
                    failed += 1
                elif status == 'partial':
                    partial += 1
        
        return {
            'total_symbols': total_symbols,
            'total_timeframes': successful + failed + partial,
            'successful': successful,
            'failed': failed,
            'partial': partial,
            'success_rate': (successful / (successful + failed + partial) * 100) if (successful + failed + partial) > 0 else 0,
        }
    
    def get_symbols_with_status(self, status: str) -> List[str]:
        """Get all symbols with specific download status"""
        symbols = []
        for sym, sym_data in self.data['instruments'].items():
            for tf_data in sym_data.get('timeframes', {}).values():
                if tf_data.get('status') == status:
                    symbols.append(sym)
                    break
        return list(set(symbols))  # Remove duplicates


# Example usage
if __name__ == '__main__':
    tracker = DownloadTracker()
    
    # Mark some downloads
    print("Marking downloads...")
    tracker.mark_downloaded('SBIN', '5', 'success', 1440, 'marketdata/5min/SBIN.parquet')
    tracker.mark_downloaded('SBIN', 'day', 'success', 250, 'marketdata/day/SBIN.parquet')
    tracker.mark_downloaded('INFY', '5', 'partial', 720, 'marketdata/5min/INFY.parquet')
    tracker.mark_downloaded('TCS', '5', 'failed', 0, None)
    
    # Check what needs download
    print("\nNeed download?")
    print(f"  SBIN: {tracker.needs_download('SBIN', '5')}")  # False (just downloaded)
    print(f"  GAIL: {tracker.needs_download('GAIL', '5')}")  # True (never downloaded)
    
    # Filter symbols
    symbols = ['SBIN', 'INFY', 'TCS', 'GAIL', 'ONGC']
    need_download = tracker.get_symbols_needing_download(symbols)
    print(f"\nSymbols needing download: {need_download}")
    
    # Get stats
    print(f"\nDownload stats: {tracker.get_download_stats()}")
    print(f"\nSuccessful downloads: {tracker.get_symbols_with_status('success')}")
    print(f"Failed downloads: {tracker.get_symbols_with_status('failed')}")
