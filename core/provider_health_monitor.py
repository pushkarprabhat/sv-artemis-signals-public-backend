# core/provider_health_monitor.py
"""
Provider Health Monitoring System
Tracks success rates, response times, and availability of data providers
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Metrics for a data provider"""
    provider_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time_ms: float
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    is_healthy: bool


class ProviderHealthMonitor:
    """Monitor and track health of data providers"""
    
    def __init__(self, db_path: str = "data/provider_health.db"):
        """Initialize provider health monitor"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS provider_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL,
                response_time_ms REAL,
                error_message TEXT
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_provider_timestamp 
            ON provider_requests(provider, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def record_request(self, 
                      provider: str, 
                      success: bool, 
                      response_time_ms: float = None,
                      error_message: str = None):
        """Record a data provider request"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO provider_requests (provider, success, response_time_ms, error_message)
                VALUES (?, ?, ?, ?)
            ''', (provider, success, response_time_ms, error_message))
            
            conn.commit()
            conn.close()
            
            # Check if we should alert
            if not success:
                self._check_alert_threshold(provider)
        
        except Exception as e:
            logger.error(f"Error recording provider request: {e}")
    
    def get_provider_metrics(self, provider: str, lookback_hours: int = 24) -> ProviderMetrics:
        """Get metrics for a specific provider"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            # Get total requests
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN success = 1 THEN response_time_ms ELSE NULL END) as avg_response_time
                FROM provider_requests
                WHERE provider = ? AND timestamp >= ?
            ''', (provider, cutoff_time))
            
            row = cursor.fetchone()
            total, successful, failed, avg_response_time = row
            
            total = total or 0
            successful = successful or 0
            failed = failed or 0
            avg_response_time = avg_response_time or 0
            
            success_rate = (successful / total * 100) if total > 0 else 0
            
            # Get last success/failure timestamps
            cursor.execute('''
                SELECT MAX(timestamp) FROM provider_requests
                WHERE provider = ? AND success = 1
            ''', (provider,))
            last_success = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT MAX(timestamp) FROM provider_requests
                WHERE provider = ? AND success = 0
            ''', (provider,))
            last_failure = cursor.fetchone()[0]
            
            conn.close()
            
            # Determine health status
            is_healthy = success_rate >= 80 and (last_success is not None)
            
            return ProviderMetrics(
                provider_name=provider,
                total_requests=total,
                successful_requests=successful,
                failed_requests=failed,
                success_rate=success_rate,
                avg_response_time_ms=avg_response_time,
                last_success=datetime.fromisoformat(last_success) if last_success else None,
                last_failure=datetime.fromisoformat(last_failure) if last_failure else None,
                is_healthy=is_healthy
            )
        
        except Exception as e:
            logger.error(f"Error getting provider metrics: {e}")
            return ProviderMetrics(
                provider_name=provider,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0,
                avg_response_time_ms=0,
                last_success=None,
                last_failure=None,
                is_healthy=False
            )
    
    def get_all_providers_status(self, lookback_hours: int = 24) -> Dict[str, ProviderMetrics]:
        """Get status of all providers"""
        providers = ['zerodha', 'polygon', 'alphavantage', 'yahoo']
        return {
            provider: self.get_provider_metrics(provider, lookback_hours)
            for provider in providers
        }
    
    def _check_alert_threshold(self, provider: str):
        """Check if provider failures exceed alert threshold"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get last 10 requests
            cursor.execute('''
                SELECT success FROM provider_requests
                WHERE provider = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (provider,))
            
            recent_requests = cursor.fetchall()
            conn.close()
            
            if len(recent_requests) >= 10:
                failures = sum(1 for (success,) in recent_requests if not success)
                
                if failures >= 7:  # 70% failure rate
                    logger.warning(f"[ALERT] Provider {provider} has {failures}/10 failures!")
                    # TODO: Send email/Telegram alert
        
        except Exception as e:
            logger.error(f"Error checking alert threshold: {e}")
    
    def get_fastest_provider(self, lookback_hours: int = 24) -> Optional[str]:
        """Get the fastest responding provider"""
        all_status = self.get_all_providers_status(lookback_hours)
        
        # Filter healthy providers
        healthy_providers = {
            name: metrics 
            for name, metrics in all_status.items() 
            if metrics.is_healthy and metrics.avg_response_time_ms > 0
        }
        
        if not healthy_providers:
            return None
        
        # Return provider with lowest response time
        fastest = min(healthy_providers.items(), key=lambda x: x[1].avg_response_time_ms)
        return fastest[0]
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old request data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute('''
                DELETE FROM provider_requests
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted} old provider request records")
        
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")


# Singleton instance
_monitor_instance = None

def get_provider_monitor() -> ProviderHealthMonitor:
    """Get singleton instance of provider health monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ProviderHealthMonitor()
    return _monitor_instance
