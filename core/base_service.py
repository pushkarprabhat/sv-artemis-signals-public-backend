"""
core/base_service.py — Abstract base class for all services
Provides consistent interface, lifecycle management, logging, error handling
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# ============================================================================
# ABSTRACT BASE SERVICE
# ============================================================================

class BaseService(ABC):
    """
    Abstract base class for all services in the trading system
    
    Enforces:
    - Singleton pattern with thread-safe initialization
    - Consistent interface (initialize, shutdown, get_status)
    - Logging on all operations
    - Error handling and recovery
    
    Usage:
        class MyService(BaseService):
            def __init__(self):
                super().__init__("MyService")
            
            def initialize(self) -> bool:
                # Set up resources
                return True
            
            def shutdown(self) -> bool:
                # Clean up resources
                return True
    """
    
    _instances: Dict[str, 'BaseService'] = {}
    _lock = threading.Lock()
    
    def __init__(self, service_name: str):
        """
        Initialize service
        
        Args:
            service_name: Human-readable name (e.g., "DataDownloader")
        """
        self.service_name = service_name
        self.logger = logging.getLogger(f"[{service_name}]")
        self.initialized_at = datetime.utcnow()
        self.is_running = False
        self.error_count = 0
        self.last_error = None
        
        self.logger.info(f"Service instantiated")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize service resources (database, API connections, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Clean up and shutdown service
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def start(self) -> bool:
        """
        Start the service
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                self.logger.warning("Service already running")
                return True
            
            self.logger.info("Starting service...")
            
            if not self.initialize():
                self.logger.error("Service initialization failed")
                return False
            
            self.is_running = True
            self.logger.info("Service started successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting service: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    def stop(self) -> bool:
        """
        Stop the service
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self.is_running:
                self.logger.info("Service not running")
                return True
            
            self.logger.info("Stopping service...")
            
            if not self.shutdown():
                self.logger.warning("Service shutdown had issues")
            
            self.is_running = False
            self.logger.info("Service stopped")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping service: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed service status
        
        Returns:
            {
                'service': service_name,
                'running': bool,
                'initialized_at': ISO timestamp,
                'uptime_seconds': float,
                'error_count': int,
                'last_error': str or None
            }
        """
        uptime = (datetime.utcnow() - self.initialized_at).total_seconds()
        
        return {
            'service': self.service_name,
            'running': self.is_running,
            'initialized_at': self.initialized_at.isoformat(),
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'last_error': self.last_error
        }
    
    def log_error(self, message: str, exc: Optional[Exception] = None):
        """
        Log error and track count
        
        Args:
            message: Error message
            exc: Optional exception object
        """
        self.error_count += 1
        self.last_error = message
        
        if exc:
            self.logger.error(f"{message}: {exc}")
        else:
            self.logger.error(message)
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    @classmethod
    def get_instance(cls, service_name: Optional[str] = None) -> 'BaseService':
        """
        Get singleton instance of service (thread-safe)
        
        Args:
            service_name: Optional name override
        
        Returns:
            Service instance
        """
        key = service_name or cls.__name__
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = cls()
        
        return cls._instances[key]


# ============================================================================
# CONCRETE SERVICE IMPLEMENTATIONS
# ============================================================================

class DataDownloaderService(BaseService):
    """Service for downloading market data"""
    
    def __init__(self):
        super().__init__("DataDownloader")
    
    def initialize(self) -> bool:
        """Initialize downloader resources"""
        try:
            from core.downloader import load_price
            self.log_info("Downloader initialized")
            return True
        except Exception as e:
            self.log_error("Failed to initialize downloader", e)
            return False
    
    def shutdown(self) -> bool:
        """Shutdown downloader"""
        self.log_info("Downloader shutdown")
        return True


class DataAggregationService(BaseService):
    """Service for aggregating OHLCV data across timeframes"""
    
    def __init__(self):
        super().__init__("DataAggregation")
        self.manager = None
    
    def initialize(self) -> bool:
        """Initialize aggregation manager"""
        try:
            from core.data_aggregation_manager import DataAggregationManager
            self.manager = DataAggregationManager()
            self.log_info("Aggregation manager initialized")
            return True
        except Exception as e:
            self.log_error("Failed to initialize aggregation", e)
            return False
    
    def shutdown(self) -> bool:
        """Shutdown aggregator"""
        self.log_info("Aggregator shutdown")
        return True


class CorporateActionsService(BaseService):
    """Service for handling corporate actions (dividends, splits, etc.)"""
    
    def __init__(self):
        super().__init__("CorporateActions")
        self.engine = None
    
    def initialize(self) -> bool:
        """Initialize corporate actions engine"""
        try:
            from core.corporate_actions import get_adjustment_engine
            self.engine = get_adjustment_engine()
            self.log_info("Corporate actions engine initialized")
            return True
        except Exception as e:
            self.log_error("Failed to initialize corporate actions", e)
            return False
    
    def shutdown(self) -> bool:
        """Shutdown corporate actions service"""
        self.log_info("Corporate actions shutdown")
        return True


class DataValidationService(BaseService):
    """Service for validating data quality"""
    
    def __init__(self):
        super().__init__("DataValidation")
    
    def initialize(self) -> bool:
        """Initialize validation service"""
        try:
            self.log_info("Data validation service initialized")
            return True
        except Exception as e:
            self.log_error("Failed to initialize validation", e)
            return False
    
    def shutdown(self) -> bool:
        """Shutdown validation service"""
        self.log_info("Data validation shutdown")
        return True


class LiveFeedService(BaseService):
    """Service for live market data feeds"""
    
    def __init__(self):
        super().__init__("LiveFeed")
        self.manager = None
    
    def initialize(self) -> bool:
        """Initialize live feed manager"""
        try:
            from core.live_feed_manager import get_live_feed_manager
            self.manager = get_live_feed_manager()
            self.log_info("Live feed manager initialized")
            return True
        except Exception as e:
            self.log_error("Failed to initialize live feed", e)
            return False
    
    def shutdown(self) -> bool:
        """Shutdown live feed"""
        if self.manager:
            try:
                self.manager.stop()
            except:
                pass
        self.log_info("Live feed shutdown")
        return True
