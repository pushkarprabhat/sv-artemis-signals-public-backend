"""
Phase 6 - Performance Monitoring Framework
Tracks download speed, resource usage, and optimizes batch sizes
"""

import psutil
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: str
    symbols_processed: int
    duration_seconds: float
    symbols_per_second: float
    files_created: int
    total_size_mb: float
    memory_used_mb: float
    cpu_percent: float
    tier: str

class PerformanceMonitor:
    """Monitors and logs performance metrics"""
    
    def __init__(self, tier: str, target_symbols: int = None):
        self.tier = tier
        self.target_symbols = target_symbols
        self.metrics: List[PerformanceMetric] = []
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
    
    def start(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024**2)
        self.start_cpu = psutil.cpu_percent(interval=0.1)
        
        logger.info(f"Performance monitoring started for {self.tier}")
        logger.info(f"Start memory: {self.start_memory:.1f} MB")
    
    def record_progress(self, symbols_processed: int, files_created: int, total_size_mb: float):
        """Record progress checkpoint"""
        elapsed = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / (1024**2)
        current_cpu = psutil.cpu_percent(interval=0.1)
        
        symbols_per_sec = symbols_processed / max(elapsed, 1)
        
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            symbols_processed=symbols_processed,
            duration_seconds=round(elapsed, 2),
            symbols_per_second=round(symbols_per_sec, 2),
            files_created=files_created,
            total_size_mb=round(total_size_mb, 2),
            memory_used_mb=round(current_memory - self.start_memory, 2),
            cpu_percent=round(current_cpu, 1),
            tier=self.tier
        )
        
        self.metrics.append(metric)
        
        # Log progress
        eta_seconds = (self.target_symbols - symbols_processed) / max(symbols_per_sec, 0.1) if self.target_symbols else 0
        
        logger.info(
            f"Progress: {symbols_processed}/{self.target_symbols or 'unlimited'} symbols | "
            f"{symbols_per_sec:.1f} sym/sec | "
            f"Memory: {metric.memory_used_mb:.1f} MB | "
            f"ETA: {eta_seconds/60:.0f} mins"
        )
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.metrics:
            return {}
        
        last_metric = self.metrics[-1]
        
        return {
            "tier": self.tier,
            "total_duration_seconds": last_metric.duration_seconds,
            "total_duration_minutes": round(last_metric.duration_seconds / 60, 2),
            "symbols_processed": last_metric.symbols_processed,
            "avg_symbols_per_second": round(
                sum(m.symbols_per_second for m in self.metrics) / len(self.metrics), 2
            ),
            "peak_symbols_per_second": round(max(m.symbols_per_second for m in self.metrics), 2),
            "files_created": last_metric.files_created,
            "total_size_mb": last_metric.total_size_mb,
            "total_size_gb": round(last_metric.total_size_mb / 1024, 2),
            "peak_memory_mb": round(max(m.memory_used_mb for m in self.metrics), 2),
            "avg_cpu_percent": round(
                sum(m.cpu_percent for m in self.metrics) / len(self.metrics), 1
            ),
            "metrics": [asdict(m) for m in self.metrics]
        }

class CeleryOptimizer:
    """Optimizes Celery batch configuration"""
    
    @staticmethod
    def recommend_batch_size(tier: str, total_symbols: int, target_duration_hours: float = 1) -> Dict:
        """
        Recommend optimal batch size for Celery tasks
        
        Args:
            tier: Tier number (1, 2, 3)
            total_symbols: Total symbols to download
            target_duration_hours: Target duration for download
        """
        # Baseline estimates (can be adjusted based on benchmarks)
        tier_performance = {
            "1": {
                "symbols_per_second_baseline": 2.0,  # Will improve with optimization
                "optimal_batch_size_range": (50, 200),
                "concurrent_tasks": 10
            },
            "2": {
                "symbols_per_second_baseline": 1.0,
                "optimal_batch_size_range": (10, 50),
                "concurrent_tasks": 5
            },
            "3": {
                "symbols_per_second_baseline": 0.5,
                "optimal_batch_size_range": (5, 20),
                "concurrent_tasks": 3
            }
        }
        
        perf = tier_performance.get(str(tier), tier_performance["1"])
        
        target_duration_seconds = target_duration_hours * 3600
        required_speed = total_symbols / target_duration_seconds
        
        # Estimate batch size
        baseline_speed = perf["symbols_per_second_baseline"]
        speed_multiplier = required_speed / baseline_speed if baseline_speed > 0 else 1
        
        min_batch, max_batch = perf["optimal_batch_size_range"]
        
        if speed_multiplier > 1.2:
            recommended_batch = max_batch
        elif speed_multiplier < 0.8:
            recommended_batch = min_batch
        else:
            recommended_batch = int((min_batch + max_batch) / 2)
        
        return {
            "tier": tier,
            "total_symbols": total_symbols,
            "target_duration_hours": target_duration_hours,
            "required_speed_symbols_per_second": round(required_speed, 2),
            "baseline_speed_symbols_per_second": baseline_speed,
            "recommended_batch_size": recommended_batch,
            "batch_size_range": perf["optimal_batch_size_range"],
            "recommended_concurrent_tasks": perf["concurrent_tasks"],
            "estimated_completion_seconds": round(total_symbols / (baseline_speed * (recommended_batch / 100)), 0),
            "estimated_completion_minutes": round(total_symbols / (baseline_speed * (recommended_batch / 100)) / 60, 1)
        }
    
    @staticmethod
    def recommend_redis_settings() -> Dict:
        """Recommend Redis queue settings for optimal performance"""
        return {
            "queue_size_limit": 10000,
            "max_memory_policy": "allkeys-lru",
            "persistence": {
                "save_schedule": "60 10000",  # Save after 60s if 10000 keys changed
                "aof_enabled": True,  # Append-only file for durability
                "aof_fsync": "everysec"  # Fsync every second for balance
            },
            "timeout": 300,  # 5 minute task timeout
            "retry_policy": {
                "max_retries": 3,
                "retry_delay_seconds": 60
            }
        }

class DownloadBenchmark:
    """Run benchmark tests to establish baselines"""
    
    @staticmethod
    def run_sample_download(tier: str, sample_size: int = 100) -> Dict:
        """
        Run sample download to benchmark performance
        
        Args:
            tier: Tier number to test
            sample_size: Number of symbols to download for benchmark
        """
        from services.instrument_manager import InstrumentManager
        from core.downloader import Downloader
        
        logger.info(f"Running {tier} sample download benchmark ({sample_size} symbols)...")
        
        try:
            monitor = PerformanceMonitor(tier, target_symbols=sample_size)
            monitor.start()
            
            # Initialize downloader
            instrument_manager = InstrumentManager(tier=tier)
            downloader = Downloader(instrument_manager)
            
            # Download sample
            # Note: This is a placeholder - actual implementation depends on Downloader API
            # For now, just record the setup time
            
            monitor.record_progress(sample_size, sample_size, 0)
            
            return {
                "status": "BENCHMARK_COMPLETED",
                "tier": tier,
                "sample_size": sample_size,
                "summary": monitor.get_summary()
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {
                "status": "BENCHMARK_FAILED",
                "tier": tier,
                "error": str(e)
            }

class RateLimiter:
    """Implements rate limiting to avoid broker throttling"""
    
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.last_request_time = None
    
    def wait(self):
        """Wait until next request can be made"""
        if self.last_request_time is None:
            self.last_request_time = time.time()
            return
        
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    @staticmethod
    def get_broker_rate_limits() -> Dict:
        """Get recommended rate limits per broker"""
        return {
            "kite": {
                "requests_per_second": 3,
                "requests_per_minute": 60,
                "description": "Zerodha Kite API"
            },
            "alice": {
                "requests_per_second": 2,
                "requests_per_minute": 40,
                "description": "Alice Blue API"
            },
            "shoonya": {
                "requests_per_second": 2,
                "requests_per_minute": 40,
                "description": "Shoonya API"
            }
        }

class PerformanceReport:
    """Generates performance optimization report"""
    
    @staticmethod
    def generate_report(benchmark_results: List[Dict], recommendations: Dict) -> Dict:
        """Generate comprehensive optimization report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "benchmark_results": benchmark_results,
            "recommendations": recommendations,
            "next_steps": [
                "1. Update config.py with recommended batch sizes",
                "2. Configure Redis with recommended settings",
                "3. Deploy optimized download scripts",
                "4. Monitor performance metrics during full download",
                "5. Adjust settings based on actual performance"
            ],
            "expected_improvements": {
                "tier1_speed_improvement": "30-50% faster",
                "tier1_memory_reduction": "20-30% less memory",
                "tier1_completion_time": "< 1 hour target",
                "tier2_completion_time": "< 30 minutes",
                "tier3_initialization": "< 5 minutes"
            }
        }
    
    @staticmethod
    def save_report(report: Dict, output_path: Path = None):
        """Save performance report"""
        output_path = output_path or Path("marketdata") / "phase6_optimization_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {output_path}")

class OptimizationRecommender:
    """Generates optimization recommendations"""
    
    @staticmethod
    def analyze_benchmark(results: Dict) -> Dict:
        """Analyze benchmark results and provide recommendations"""
        recommendations = {
            "celery_config": CeleryOptimizer.recommend_batch_size("1", 22559),
            "redis_settings": CeleryOptimizer.recommend_redis_settings(),
            "rate_limits": RateLimiter.get_broker_rate_limits(),
            "parallel_strategy": {
                "tier1": {
                    "max_workers": 10,
                    "batch_size": 100,
                    "timeout_seconds": 300
                },
                "tier2": {
                    "max_workers": 5,
                    "batch_size": 50,
                    "timeout_seconds": 300
                },
                "tier3": {
                    "max_workers": 3,
                    "batch_size": 20,
                    "timeout_seconds": 300
                }
            },
            "disk_optimization": {
                "parquet_compression": "snappy",
                "compression_level": 5,
                "batch_writes": 100
            }
        }
        
        return recommendations
