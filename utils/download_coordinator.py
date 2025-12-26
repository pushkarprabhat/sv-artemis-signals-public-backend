"""
Download Coordinator - Provides detailed pre-download summaries and real-time progress tracking.

Uses the enhanced instrument list from app_kite_universe.csv which contains:
- All 119k+ instruments with proper exchange/segment/type/underlying_symbol info
- Exchange: NSE, BSE, NFO, BFO, CDS, MCX, NSEIX, GLOBAL, NCO
- Segment: NSE, BSE, NFO-FUT, NFO-OPT, NFO, BFO-FUT, BFO-OPT, CDS-FUT, CDS-OPT, MCX-FUT, MCX-OPT, NCO, NCO-FUT, NCO-OPT, INDICES, GLOBAL, NSEIX
- Type: EQ (Equities), FUT (Futures), CE (Call Options), PE (Put Options)
- UNDERLYING_SYMBOL: For derivatives, shows the underlying (e.g., SBIN, BANKNIFTY, NIFTY50, etc.)

Features:
- Exchange/Segment/InstrumentType breakdowns before download
- Real-time progress with percentage completion
- Time-to-completion estimates after each symbol
- Organized download batches by exchange/segment/underlying
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from utils.logger import logger


class DownloadStats:
    """Track download statistics in real-time"""
    
    def __init__(self, total_symbols: int):
        self.total_symbols = total_symbols
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = datetime.now()
        self.last_symbol_time = self.start_time
        self.times_by_exchange = defaultdict(list)
        self.times_by_segment = defaultdict(list)
        self.times_by_type = defaultdict(list)
    
    def record_symbol(self, duration_seconds: float, exchange: str = None, 
                     segment: str = None, instrument_type: str = None, 
                     success: bool = True):
        """Record download completion for a symbol"""
        if success:
            self.downloaded += 1
        else:
            self.failed += 1
        
        self.last_symbol_time = datetime.now()
        
        if exchange:
            self.times_by_exchange[exchange].append(duration_seconds)
        if segment:
            self.times_by_segment[segment].append(duration_seconds)
        if instrument_type:
            self.times_by_type[instrument_type].append(duration_seconds)
    
    def get_avg_time(self, category: str = None):
        """Get average download time"""
        if category is None:
            # Overall average
            total_time = (self.last_symbol_time - self.start_time).total_seconds()
            downloaded = self.downloaded + self.failed
            return total_time / downloaded if downloaded > 0 else 0
        elif category in self.times_by_exchange:
            times = self.times_by_exchange[category]
            return sum(times) / len(times) if times else 0
        elif category in self.times_by_segment:
            times = self.times_by_segment[category]
            return sum(times) / len(times) if times else 0
        elif category in self.times_by_type:
            times = self.times_by_type[category]
            return sum(times) / len(times) if times else 0
        return 0
    
    def get_eta(self):
        """Get estimated time to completion"""
        remaining = self.total_symbols - self.downloaded - self.failed - self.skipped
        avg_time = self.get_avg_time()
        if avg_time > 0:
            eta_seconds = remaining * avg_time
            return eta_seconds
        return 0
    
    def get_progress_pct(self):
        """Get overall progress percentage"""
        completed = self.downloaded + self.failed + self.skipped
        return (completed / self.total_symbols * 100) if self.total_symbols > 0 else 0


class DownloadPreSummary:
    """Pre-download summary showing what will be downloaded"""
    
    def __init__(self):
        self.by_exchange = defaultdict(int)
        self.by_segment = defaultdict(int)
        self.by_type = defaultdict(int)
        self.by_exchange_segment = defaultdict(int)
        self.by_exchange_type = defaultdict(int)
        self.by_segment_type = defaultdict(int)
        self.by_exchange_segment_type = defaultdict(int)
        self.total = 0
    
    def add_instrument(self, exchange: str, segment: str, instrument_type: str):
        """Add an instrument to the summary"""
        self.by_exchange[exchange] += 1
        self.by_segment[segment] += 1
        self.by_type[instrument_type] += 1
        self.by_exchange_segment[f"{exchange}:{segment}"] += 1
        self.by_exchange_type[f"{exchange}:{instrument_type}"] += 1
        self.by_segment_type[f"{segment}:{instrument_type}"] += 1
        self.by_exchange_segment_type[f"{exchange}:{segment}:{instrument_type}"] += 1
        self.total += 1
    
    def print_summary(self):
        """Print detailed pre-download summary"""
        print("\n" + "="*100)
        print("PRE-DOWNLOAD SUMMARY")
        print("="*100)
        
        print(f"\n[TOTAL] {self.total:,} instruments to download\n")
        
        # Exchange-wise breakdown
        print("[EXCHANGE-WISE BREAKDOWN]")
        for exchange in sorted(self.by_exchange.keys()):
            count = self.by_exchange[exchange]
            pct = (count / self.total * 100) if self.total > 0 else 0
            print(f"  {exchange:15} {count:8,} instruments  ({pct:6.2f}%)")
        
        # Segment-wise breakdown
        print("\n[SEGMENT-WISE BREAKDOWN]")
        for segment in sorted(self.by_segment.keys()):
            count = self.by_segment[segment]
            pct = (count / self.total * 100) if self.total > 0 else 0
            print(f"  {segment:15} {count:8,} instruments  ({pct:6.2f}%)")
        
        # Instrument Type breakdown
        print("\n[INSTRUMENT-TYPE BREAKDOWN]")
        for itype in sorted(self.by_type.keys()):
            count = self.by_type[itype]
            pct = (count / self.total * 100) if self.total > 0 else 0
            print(f"  {itype:15} {count:8,} instruments  ({pct:6.2f}%)")
        
        # Exchange + Segment breakdown
        print("\n[EXCHANGE + SEGMENT BREAKDOWN]")
        for key in sorted(self.by_exchange_segment.keys()):
            count = self.by_exchange_segment[key]
            pct = (count / self.total * 100) if self.total > 0 else 0
            print(f"  {key:30} {count:8,} instruments  ({pct:6.2f}%)")
        
        # Exchange + Type breakdown
        print("\n[EXCHANGE + INSTRUMENT-TYPE BREAKDOWN]")
        for key in sorted(self.by_exchange_type.keys()):
            count = self.by_exchange_type[key]
            pct = (count / self.total * 100) if self.total > 0 else 0
            print(f"  {key:30} {count:8,} instruments  ({pct:6.2f}%)")
        
        print("\n" + "="*100)


class DownloadProgressTracker:
    """Real-time progress tracking with periodic reports"""
    
    def __init__(self, stats: DownloadStats, summary: DownloadPreSummary, report_interval: int = 50):
        self.stats = stats
        self.summary = summary
        self.report_interval = report_interval
        self.last_report_count = 0
    
    def update(self, symbol: str, exchange: str, segment: str, instrument_type: str, 
               duration_seconds: float, success: bool = True) -> bool:
        """
        Update progress and return True if a progress report should be printed
        
        Returns True when report_interval is reached
        """
        self.stats.record_symbol(duration_seconds, exchange, segment, instrument_type, success)
        
        # Check if we should print a report
        should_report = (self.stats.downloaded + self.stats.failed) % self.report_interval == 0
        
        if should_report:
            self.print_progress_report(symbol, exchange, segment, instrument_type)
            return True
        
        return False
    
    def print_progress_report(self, last_symbol: str, last_exchange: str, 
                             last_segment: str, last_type: str):
        """Print detailed progress report"""
        progress = self.stats.get_progress_pct()
        completed = self.stats.downloaded + self.stats.failed
        
        eta_seconds = self.stats.get_eta()
        eta_str = self._format_time(eta_seconds)
        
        print(f"\n[PROGRESS] {completed:,}/{self.stats.total_symbols:,} ({progress:6.2f}%)")
        print(f"  Last: {last_symbol:20} | {last_exchange}:{last_segment}:{last_type}")
        print(f"  Success: {self.stats.downloaded:,} | Failed: {self.stats.failed:,} | Skipped: {self.stats.skipped:,}")
        print(f"  Avg Time/Symbol: {self.stats.get_avg_time():.2f}s")
        print(f"  ETA: {eta_str}")
        
        # Exchange-wise progress
        print(f"\n  [EXCHANGE-WISE PROGRESS]")
        for exchange in sorted(self.summary.by_exchange.keys()):
            total = self.summary.by_exchange[exchange]
            # Estimate downloads from this exchange
            pct = (total / self.stats.total_symbols) * 100
            est_completed = int((completed / self.stats.total_symbols) * total)
            est_pct = (est_completed / total * 100) if total > 0 else 0
            print(f"    {exchange:15} {est_completed:8,}/{total:8,} ({est_pct:6.2f}%)  ETA: {self._format_time(self.stats.get_eta() * (total / self.stats.total_symbols))}")
        
        # Segment-wise progress
        print(f"\n  [SEGMENT-WISE PROGRESS]")
        for segment in sorted(self.summary.by_segment.keys()):
            total = self.summary.by_segment[segment]
            est_completed = int((completed / self.stats.total_symbols) * total)
            est_pct = (est_completed / total * 100) if total > 0 else 0
            print(f"    {segment:15} {est_completed:8,}/{total:8,} ({est_pct:6.2f}%)  ETA: {self._format_time(self.stats.get_eta() * (total / self.stats.total_symbols))}")
        
        # Type-wise progress
        print(f"\n  [TYPE-WISE PROGRESS]")
        for itype in sorted(self.summary.by_type.keys()):
            total = self.summary.by_type[itype]
            est_completed = int((completed / self.stats.total_symbols) * total)
            est_pct = (est_completed / total * 100) if total > 0 else 0
            print(f"    {itype:15} {est_completed:8,}/{total:8,} ({est_pct:6.2f}%)  ETA: {self._format_time(self.stats.get_eta() * (total / self.stats.total_symbols))}")
    
    def print_final_report(self):
        """Print final download report"""
        elapsed = (datetime.now() - self.stats.start_time).total_seconds()
        
        print("\n" + "="*100)
        print("FINAL DOWNLOAD REPORT")
        print("="*100)
        print(f"\n[SUMMARY]")
        print(f"  Total Time: {self._format_time(elapsed)}")
        print(f"  Downloaded: {self.stats.downloaded:,}")
        print(f"  Failed: {self.stats.failed:,}")
        print(f"  Skipped: {self.stats.skipped:,}")
        print(f"  Avg Time/Symbol: {self.stats.get_avg_time():.2f}s")
        
        print(f"\n[AVG TIME BY EXCHANGE]")
        for exchange in sorted(self.summary.by_exchange.keys()):
            avg_time = self.stats.get_avg_time(exchange)
            if avg_time > 0:
                print(f"  {exchange:15} {avg_time:8.2f}s")
        
        print(f"\n[AVG TIME BY SEGMENT]")
        for segment in sorted(self.summary.by_segment.keys()):
            avg_time = self.stats.get_avg_time(segment)
            if avg_time > 0:
                print(f"  {segment:15} {avg_time:8.2f}s")
        
        print(f"\n[AVG TIME BY TYPE]")
        for itype in sorted(self.summary.by_type.keys()):
            avg_time = self.stats.get_avg_time(itype)
            if avg_time > 0:
                print(f"  {itype:15} {avg_time:8.2f}s")
        
        print("\n" + "="*100)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class DownloadBatchOrganizer:
    """Organize downloads by exchange/segment/underlying"""
    
    def __init__(self, instruments_df: pd.DataFrame):
        self.instruments_df = instruments_df
        self.batches = {}
    
    def organize_by_exchange(self) -> Dict[str, List[str]]:
        """Organize symbols by exchange"""
        batches = defaultdict(list)
        
        for _, row in self.instruments_df.iterrows():
            exchange = row.get('exchange', row.get('EXCHANGE', 'NSE'))
            symbol = row.get('tradingsymbol', row.get('TRADINGSYMBOL', row.get('Symbol')))
            if symbol:
                batches[exchange].append(symbol)
        
        return dict(batches)
    
    def organize_by_segment(self) -> Dict[str, List[str]]:
        """Organize symbols by segment"""
        batches = defaultdict(list)
        
        for _, row in self.instruments_df.iterrows():
            segment = row.get('segment', row.get('SEGMENT', row.get('instrument_type', 'EQ')))
            symbol = row.get('tradingsymbol', row.get('TRADINGSYMBOL', row.get('Symbol')))
            if symbol:
                batches[segment].append(symbol)
        
        return dict(batches)
    
    def organize_by_underlying(self) -> Dict[str, List[str]]:
        """Organize derivatives by underlying symbol"""
        batches = defaultdict(list)
        
        for _, row in self.instruments_df.iterrows():
            underlying = row.get('UNDERLYING_SYMBOL', row.get('underlying_symbol', None))
            if underlying:
                symbol = row.get('tradingsymbol', row.get('TRADINGSYMBOL', row.get('Symbol')))
                if symbol:
                    batches[underlying].append(symbol)
        
        return dict(batches)
    
    def organize_by_exchange_segment(self) -> Dict[Tuple[str, str], List[str]]:
        """Organize symbols by exchange and segment"""
        batches = defaultdict(list)
        
        for _, row in self.instruments_df.iterrows():
            exchange = row.get('exchange', row.get('EXCHANGE', 'NSE'))
            segment = row.get('segment', row.get('SEGMENT', row.get('instrument_type', 'EQ')))
            symbol = row.get('tradingsymbol', row.get('TRADINGSYMBOL', row.get('Symbol')))
            if symbol:
                batches[(exchange, segment)].append(symbol)
        
        return dict(batches)


class DownloadCoordinator:
    """Main coordinator for download planning and execution"""
    
    def __init__(self, data_dir: Path = Path("universe/app")):
        self.data_dir = Path(data_dir)
        self.instruments_df = None
        self.load_instruments()
    
    def load_instruments(self):
        """Load instruments from app_kite_universe.csv"""
        # Try multiple locations
        possible_paths = [
            self.data_dir / "app_kite_universe.csv",
            Path(__file__).parent.parent / "universe" / "app" / "app_kite_universe.csv",
        ]
        
        universe_file = None
        for path in possible_paths:
            if path.exists():
                universe_file = path
                break
        
        if not universe_file:
            logger.warning(f"App Kite Universe not found. Checked: {[str(p) for p in possible_paths]}")
            self.instruments_df = pd.DataFrame()
            return
        
        try:
            self.instruments_df = pd.read_csv(universe_file)
            logger.info(f"Loaded {len(self.instruments_df):,} instruments from app_kite_universe.csv")
        except Exception as e:
            logger.error(f"Error loading app_kite_universe.csv: {e}")
            self.instruments_df = pd.DataFrame()
    
    def get_symbols_by_exchange(self, exchange: Optional[str] = None) -> List[str]:
        """Get all trading symbols, optionally filtered by exchange"""
        if self.instruments_df.empty:
            return []
        
        df = self.instruments_df
        
        if exchange:
            df = df[df['exchange'].str.upper() == exchange.upper()]
        
        # Get symbol column - try tradingsymbol first, then others
        for col in ['tradingsymbol', 'TRADINGSYMBOL', 'Symbol', 'symbol']:
            if col in df.columns:
                return df[col].dropna().tolist()
        
        return []
    
    def get_symbols_by_segment(self, segment: Optional[str] = None) -> List[str]:
        """Get all trading symbols, optionally filtered by segment"""
        if self.instruments_df.empty:
            return []
        
        df = self.instruments_df
        
        if segment:
            df = df[df['segment'].str.upper() == segment.upper()]
        
        for col in ['tradingsymbol', 'TRADINGSYMBOL', 'Symbol', 'symbol']:
            if col in df.columns:
                return df[col].dropna().tolist()
        
        return []
    
    def get_derivatives_by_underlying(self, underlying: str) -> List[str]:
        """Get all derivatives for a specific underlying symbol"""
        if self.instruments_df.empty:
            return []
        
        df = self.instruments_df[self.instruments_df['UNDERLYING_SYMBOL'].str.upper() == underlying.upper()]
        
        for col in ['tradingsymbol', 'TRADINGSYMBOL', 'Symbol', 'symbol']:
            if col in df.columns:
                return df[col].dropna().tolist()
        
        return []
    
    def get_pre_summary(self, symbols: List[str]) -> DownloadPreSummary:
        """Get pre-download summary for a list of symbols"""
        if self.instruments_df.empty:
            return DownloadPreSummary()
        
        summary = DownloadPreSummary()
        symbol_col = None
        
        # Find symbol column
        for col in ['tradingsymbol', 'TRADINGSYMBOL', 'Symbol', 'symbol']:
            if col in self.instruments_df.columns:
                symbol_col = col
                break
        
        if not symbol_col:
            return summary
        
        for symbol in symbols:
            row = self.instruments_df[self.instruments_df[symbol_col] == symbol]
            if not row.empty:
                row = row.iloc[0]
                exchange = row.get('exchange', 'NSE')
                segment = row.get('segment', 'NSE')
                itype = row.get('instrument_type', 'EQ')
                summary.add_instrument(exchange, segment, itype)
        
        return summary
    
    def create_stats_tracker(self, symbols: List[str]) -> DownloadStats:
        """Create a stats tracker for a list of symbols"""
        return DownloadStats(len(symbols))
    
    def create_progress_tracker(self, symbols: List[str], report_interval: int = 50) -> DownloadProgressTracker:
        """Create a progress tracker for a list of symbols"""
        stats = self.create_stats_tracker(symbols)
        summary = self.get_pre_summary(symbols)
        return DownloadProgressTracker(stats, summary, report_interval)
    
    def organize_batch(self, symbols: List[str]) -> DownloadBatchOrganizer:
        """Create a batch organizer for a list of symbols"""
        # Filter dataframe to only these symbols
        symbol_col = None
        for col in ['tradingsymbol', 'TRADINGSYMBOL', 'Symbol', 'symbol']:
            if col in self.instruments_df.columns:
                symbol_col = col
                break
        
        if symbol_col:
            filtered_df = self.instruments_df[self.instruments_df[symbol_col].isin(symbols)]
        else:
            filtered_df = self.instruments_df
        
        return DownloadBatchOrganizer(filtered_df)


def build_pre_download_summary(instruments_df: pd.DataFrame) -> DownloadPreSummary:
    """Build pre-download summary from instrument dataframe"""
    summary = DownloadPreSummary()
    
    for _, row in instruments_df.iterrows():
        exchange = row.get('exchange', row.get('EXCHANGE', 'NSE'))
        segment = row.get('segment', row.get('SEGMENT', row.get('instrument_type', 'EQ')))
        itype = row.get('instrument_type', row.get('INSTRUMENT_TYPE', 'EQ'))
        
        summary.add_instrument(exchange, segment, itype)
    
    return summary
