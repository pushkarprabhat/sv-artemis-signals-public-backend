"""
parallel_executor.py - Execute all scans, downloads, analysers, and backtests in parallel
Stores results with portfolio tracking and live configuration
"""

import asyncio
import pandas as pd
import threading
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Results storage configuration
RESULTS_DIR = Path("results")
SCAN_RESULTS_DIR = RESULTS_DIR / "scans"
BACKTEST_RESULTS_DIR = RESULTS_DIR / "backtests"
DOWNLOAD_RESULTS_DIR = RESULTS_DIR / "downloads"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"
PORTFOLIO_RESULTS_DIR = RESULTS_DIR / "portfolios"

# Create directories
for d in [RESULTS_DIR, SCAN_RESULTS_DIR, BACKTEST_RESULTS_DIR, DOWNLOAD_RESULTS_DIR, ANALYSIS_RESULTS_DIR, PORTFOLIO_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class LiveConfig:
    """Live configuration that can be updated while tasks run"""
    
    def __init__(self):
        self._config = {
            # Scan settings
            'pair_trading_enabled': True,
            'pair_trading_timeframes': ['15minute', '30minute', '60minute', 'day'],
            'momentum_enabled': True,
            'momentum_timeframes': ['15minute', '30minute', '60minute', 'day'],
            'mean_reversion_enabled': True,
            'mean_reversion_timeframes': ['15minute', '30minute', '60minute', 'day'],
            'volatility_enabled': True,
            'volatility_timeframes': ['15minute', '30minute', '60minute', 'day'],
            'kelly_enabled': True,
            'kelly_timeframes': ['15minute', '30minute', '60minute', 'day'],
            'options_enabled': True,
            'options_timeframes': ['day'],
            
            # Download settings
            'download_intraday': True,
            'download_daily': True,
            'download_interval_seconds': 300,
            
            # Analysis settings
            'analysis_enabled': True,
            'analysis_symbols': [],
            
            # Backtest settings
            'backtest_enabled': True,
            'backtest_initial_capital': 100000,
            'backtest_risk_per_trade': 0.02,
            
            # General settings
            'market_hours_only': True,
            'max_workers': 4,
            'result_persistence': True,
        }
        self._lock = threading.RLock()
    
    def get(self, key: str, default=None) -> Any:
        """Thread-safe get"""
        with self._lock:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Thread-safe set"""
        with self._lock:
            self._config[key] = value
            logger.info(f"Config updated: {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """Thread-safe get all"""
        with self._lock:
            return dict(self._config)
    
    def update_multiple(self, updates: Dict[str, Any]) -> None:
        """Thread-safe batch update"""
        with self._lock:
            self._config.update(updates)
            logger.info(f"Config batch updated: {len(updates)} items")


class ResultsPersistence:
    """Store and retrieve scan/backtest results with portfolio tracking"""
    
    @staticmethod
    def save_scan_result(scan_type: str, timeframe: str, results: pd.DataFrame, metadata: Dict = None) -> Path:
        """Save scan results to timestamped file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scan_type}_{timeframe}_{timestamp}.parquet"
        filepath = SCAN_RESULTS_DIR / filename
        
        try:
            results.to_parquet(filepath)
            logger.info(f"Scan results saved: {filepath}")
            
            # Also save metadata
            if metadata:
                metadata_file = filepath.with_suffix('.json')
                metadata['timestamp'] = timestamp
                metadata['scan_type'] = scan_type
                metadata['timeframe'] = timeframe
                metadata['result_count'] = len(results)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            return filepath
        except Exception as e:
            logger.error(f"Error saving scan result: {e}")
            return None
    
    @staticmethod
    def save_backtest_result(strategy: str, results: Dict[str, Any]) -> Path:
        """Save backtest results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy}_{timestamp}.json"
        filepath = BACKTEST_RESULTS_DIR / filename
        
        try:
            results['timestamp'] = timestamp
            results['strategy'] = strategy
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Backtest results saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving backtest result: {e}")
            return None
    
    @staticmethod
    def save_portfolio_state(portfolio_data: Dict[str, Any]) -> Path:
        """Save portfolio state with all positions and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_{timestamp}.json"
        filepath = PORTFOLIO_RESULTS_DIR / filename
        
        try:
            portfolio_data['timestamp'] = timestamp
            with open(filepath, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            logger.info(f"Portfolio saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return None
    
    @staticmethod
    def load_latest_scan_results(scan_type: str, limit: int = 5) -> List[Dict]:
        """Load latest scan results"""
        try:
            files = sorted(SCAN_RESULTS_DIR.glob(f"{scan_type}_*.parquet"), reverse=True)[:limit]
            results = []
            for f in files:
                metadata_file = f.with_suffix('.json')
                df = pd.read_parquet(f)
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as mf:
                        metadata = json.load(mf)
                results.append({
                    'file': f.name,
                    'data': df,
                    'metadata': metadata
                })
            return results
        except Exception as e:
            logger.error(f"Error loading scan results: {e}")
            return []


class ParallelScanExecutor:
    """Execute multiple scans in parallel"""
    
    def __init__(self, config: LiveConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results = {}
        self._lock = threading.RLock()
    
    def add_scan_task(self, 
                     scan_type: str, 
                     timeframe: str, 
                     scan_func: Callable,
                     scan_kwargs: Dict = None) -> None:
        """Add a scan task to the executor"""
        
        if scan_kwargs is None:
            scan_kwargs = {}
        
        def wrapped_scan():
            try:
                logger.info(f"[SCAN START] {scan_type} @ {timeframe}")
                result = scan_func(timeframe=timeframe, **scan_kwargs)
                logger.info(f"[SCAN DONE] {scan_type} @ {timeframe}: {len(result) if isinstance(result, pd.DataFrame) else 'N/A'} signals")
                
                # Save results
                if isinstance(result, pd.DataFrame) and not result.empty:
                    ResultsPersistence.save_scan_result(scan_type, timeframe, result)
                
                return result
            except Exception as e:
                logger.error(f"[SCAN ERROR] {scan_type} @ {timeframe}: {e}")
                return None
        
        future = self.executor.submit(wrapped_scan)
        
        with self._lock:
            key = f"{scan_type}_{timeframe}"
            self.results[key] = {'status': 'pending', 'future': future}
    
    def add_multiple_timeframe_scans(self, 
                                    scan_type: str,
                                    scan_func: Callable,
                                    timeframes: List[str] = None,
                                    scan_kwargs: Dict = None) -> None:
        """Add the same scan across multiple timeframes"""
        
        if timeframes is None:
            timeframes = self.config.get(f"{scan_type}_timeframes", ['15minute', '30minute', '60minute', 'day'])
        
        for tf in timeframes:
            self.add_scan_task(scan_type, tf, scan_func, scan_kwargs)
    
    def get_result(self, scan_type: str, timeframe: str, timeout: float = 30) -> Optional[pd.DataFrame]:
        """Get result of a specific scan (blocking)"""
        key = f"{scan_type}_{timeframe}"
        
        with self._lock:
            if key not in self.results:
                return None
            result_info = self.results[key]
        
        try:
            future = result_info['future']
            result = future.result(timeout=timeout)
            
            with self._lock:
                self.results[key]['status'] = 'completed'
                self.results[key]['result'] = result
            
            return result
        except Exception as e:
            logger.error(f"Error getting result for {key}: {e}")
            with self._lock:
                self.results[key]['status'] = 'failed'
            return None
    
    def get_all_results(self, timeout: float = 60) -> Dict[str, Any]:
        """Get all results (blocking)"""
        all_results = {}
        
        with self._lock:
            pending_keys = [k for k, v in self.results.items() if v['status'] == 'pending']
        
        for key in pending_keys:
            with self._lock:
                result_info = self.results[key]
            
            try:
                future = result_info['future']
                result = future.result(timeout=timeout)
                all_results[key] = result
                
                with self._lock:
                    self.results[key]['status'] = 'completed'
                    self.results[key]['result'] = result
            except Exception as e:
                logger.error(f"Error getting result for {key}: {e}")
                with self._lock:
                    self.results[key]['status'] = 'failed'
                all_results[key] = None
        
        return all_results
    
    def wait_all(self, timeout: float = None) -> bool:
        """Wait for all tasks to complete"""
        with self._lock:
            futures = [v['future'] for v in self.results.values()]
        
        if not futures:
            return True
        
        try:
            for future in as_completed(futures, timeout=timeout):
                future.result()
            return True
        except Exception as e:
            logger.error(f"Error waiting for tasks: {e}")
            return False


class ParallelDownloadExecutor:
    """Execute data downloads in parallel"""
    
    def __init__(self, config: LiveConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results = {}
        self._lock = threading.RLock()
    
    def add_download_task(self, 
                         symbol: str,
                         download_func: Callable,
                         download_kwargs: Dict = None) -> None:
        """Add a download task"""
        
        if download_kwargs is None:
            download_kwargs = {}
        
        def wrapped_download():
            try:
                logger.info(f"[DOWNLOAD START] {symbol}")
                result = download_func(symbol, **download_kwargs)
                logger.info(f"[DOWNLOAD DONE] {symbol}")
                
                # Save download result
                if result:
                    ResultsPersistence.save_scan_result('download', symbol, result if isinstance(result, pd.DataFrame) else pd.DataFrame())
                
                return result
            except Exception as e:
                logger.error(f"[DOWNLOAD ERROR] {symbol}: {e}")
                return None
        
        future = self.executor.submit(wrapped_download)
        
        with self._lock:
            self.results[symbol] = {'status': 'pending', 'future': future}
    
    def add_multiple_downloads(self,
                              symbols: List[str],
                              download_func: Callable,
                              download_kwargs: Dict = None) -> None:
        """Add downloads for multiple symbols"""
        
        for symbol in symbols:
            self.add_download_task(symbol, download_func, download_kwargs)
    
    def wait_all(self, timeout: float = None) -> bool:
        """Wait for all downloads to complete"""
        with self._lock:
            futures = [v['future'] for v in self.results.values()]
        
        if not futures:
            return True
        
        try:
            for future in as_completed(futures, timeout=timeout):
                future.result()
            return True
        except Exception as e:
            logger.error(f"Error waiting for downloads: {e}")
            return False


class ParallelBacktestExecutor:
    """Execute backtests in parallel"""
    
    def __init__(self, config: LiveConfig, max_workers: int = 2):
        self.config = config
        self.max_workers = max_workers
        # Use ProcessPoolExecutor for CPU-intensive backtests
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.results = {}
        self._lock = threading.RLock()
    
    def add_backtest_task(self,
                         strategy: str,
                         symbol: str,
                         backtest_func: Callable,
                         backtest_kwargs: Dict = None) -> None:
        """Add a backtest task"""
        
        if backtest_kwargs is None:
            backtest_kwargs = {}
        
        backtest_kwargs['initial_capital'] = self.config.get('backtest_initial_capital', 100000)
        backtest_kwargs['risk_per_trade'] = self.config.get('backtest_risk_per_trade', 0.02)
        
        def wrapped_backtest():
            try:
                logger.info(f"[BACKTEST START] {strategy} on {symbol}")
                result = backtest_func(symbol, **backtest_kwargs)
                logger.info(f"[BACKTEST DONE] {strategy} on {symbol}")
                
                # Save backtest result
                if result:
                    ResultsPersistence.save_backtest_result(f"{strategy}_{symbol}", result)
                
                return result
            except Exception as e:
                logger.error(f"[BACKTEST ERROR] {strategy} on {symbol}: {e}")
                return None
        
        future = self.executor.submit(wrapped_backtest)
        
        with self._lock:
            key = f"{strategy}_{symbol}"
            self.results[key] = {'status': 'pending', 'future': future}
    
    def wait_all(self, timeout: float = None) -> bool:
        """Wait for all backtests to complete"""
        with self._lock:
            futures = [v['future'] for v in self.results.values()]
        
        if not futures:
            return True
        
        try:
            for future in as_completed(futures, timeout=timeout):
                future.result()
            return True
        except Exception as e:
            logger.error(f"Error waiting for backtests: {e}")
            return False


class ParallelAnalysisExecutor:
    """Execute analyses in parallel"""
    
    def __init__(self, config: LiveConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results = {}
        self._lock = threading.RLock()
    
    def add_analysis_task(self,
                         symbol: str,
                         analysis_func: Callable,
                         analysis_kwargs: Dict = None) -> None:
        """Add an analysis task"""
        
        if analysis_kwargs is None:
            analysis_kwargs = {}
        
        def wrapped_analysis():
            try:
                logger.info(f"[ANALYSIS START] {symbol}")
                result = analysis_func(symbol, **analysis_kwargs)
                logger.info(f"[ANALYSIS DONE] {symbol}")
                
                # Save analysis result
                if result:
                    ResultsPersistence.save_scan_result('analysis', symbol, result if isinstance(result, pd.DataFrame) else pd.DataFrame())
                
                return result
            except Exception as e:
                logger.error(f"[ANALYSIS ERROR] {symbol}: {e}")
                return None
        
        future = self.executor.submit(wrapped_analysis)
        
        with self._lock:
            self.results[symbol] = {'status': 'pending', 'future': future}
    
    def wait_all(self, timeout: float = None) -> bool:
        """Wait for all analyses to complete"""
        with self._lock:
            futures = [v['future'] for v in self.results.values()]
        
        if not futures:
            return True
        
        try:
            for future in as_completed(futures, timeout=timeout):
                future.result()
            return True
        except Exception as e:
            logger.error(f"Error waiting for analyses: {e}")
            return False


class ParallelExecutionCoordinator:
    """Coordinate all parallel execution: scans, downloads, analysers, backtests"""
    
    def __init__(self):
        self.config = LiveConfig()
        self.scan_executor = ParallelScanExecutor(self.config)
        self.download_executor = ParallelDownloadExecutor(self.config)
        self.backtest_executor = ParallelBacktestExecutor(self.config)
        self.analysis_executor = ParallelAnalysisExecutor(self.config)
    
    def run_all(self, timeout: float = 300) -> Dict[str, Any]:
        """Run all executors in parallel and wait for completion"""
        logger.info("Starting parallel execution")
        
        # Wait for all to complete
        scan_ok = self.scan_executor.wait_all(timeout=timeout)
        download_ok = self.download_executor.wait_all(timeout=timeout)
        backtest_ok = self.backtest_executor.wait_all(timeout=timeout)
        analysis_ok = self.analysis_executor.wait_all(timeout=timeout)
        
        logger.info(f"Execution complete - Scans: {scan_ok}, Downloads: {download_ok}, Backtests: {backtest_ok}, Analysis: {analysis_ok}")
        
        return {
            'scans_ok': scan_ok,
            'downloads_ok': download_ok,
            'backtests_ok': backtest_ok,
            'analysis_ok': analysis_ok,
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all results with portfolio state"""
        portfolio = {
            'timestamp': datetime.now().isoformat(),
            'scans': self.scan_executor.results,
            'downloads': self.download_executor.results,
            'backtests': self.backtest_executor.results,
            'analysis': self.analysis_executor.results,
            'config': self.config.get_all(),
        }
        
        if self.config.get('result_persistence', True):
            ResultsPersistence.save_portfolio_state(portfolio)
        
        return portfolio
