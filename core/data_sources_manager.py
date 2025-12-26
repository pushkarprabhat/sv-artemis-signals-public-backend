# core/data_sources_manager.py — Data Source Provider Management System
# Tracks connection status, health, token validity, and provider metadata
# Thread-safe singleton with persistent state

import json
import os
import requests
import threading
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

class ProviderStatus(Enum):
    """Provider connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    ERROR = "error"

class TokenStatus(Enum):
    """Token validity status"""
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    NOT_SET = "not_set"
    INVALID = "invalid"

class DataSourceMetadata:
    """Metadata for a data source provider"""
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("name")
        self.provider_id = None  # Set by manager
        self.type = config.get("type")  # primary, alternative, cache
        self.enabled = config.get("enabled", True)
        self.priority = config.get("priority", 999)
        self.description = config.get("description")
        self.exchanges = config.get("exchanges", [])
        self.segments = config.get("segments", [])
        self.timeframes = config.get("timeframes", [])
        self.data_type = config.get("data_type")
        self.authentication = config.get("authentication", {})
        self.endpoints = config.get("endpoints", {})
        self.rate_limits = config.get("rate_limits", {})
        self.status_check = config.get("status_check", {})

class ProviderHealthStatus:
    """Track health metrics for a provider"""
    def __init__(self):
        self.status = ProviderStatus.UNKNOWN
        self.last_check_time = None
        self.response_time_ms = None
        self.error_message = None
        self.consecutive_failures = 0
        self.last_successful_request = None
        self.request_count_today = 0
        self.uptime_percentage = 100.0
        
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "last_check_time": self.last_check_time,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_request": self.last_successful_request,
            "request_count_today": self.request_count_today,
            "uptime_percentage": self.uptime_percentage
        }

class DataSourcesManager:
    """
    Manages data source providers, their health, tokens, and availability
    
    Features:
    - Load provider configuration from JSON
    - Track connection status for each provider
    - Monitor token validity and expiration
    - Health checks and diagnostics
    - Data availability tracking by exchange/segment/timeframe
    - Thread-safe operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = True
        self._lock = threading.Lock()
        self.config_path = Path("config/data_sources.json")
        self.state_file = Path("data/.data_sources_state.json")
        
        self.providers: Dict[str, DataSourceMetadata] = {}
        self.health_status: Dict[str, ProviderHealthStatus] = {}
        self.token_status: Dict[str, TokenStatus] = {}
        self.data_availability: Dict[str, Dict] = {}
        
        self._load_configuration()
        self._load_state()
        self._initialize_health_tracking()
        
        logger.info(f"[DataSourcesManager] Initialized with {len(self.providers)} providers")
    
    def _load_configuration(self):
        """Load provider configuration from data_sources.json"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for provider_id, provider_config in config.get("data_sources", {}).items():
                metadata = DataSourceMetadata(provider_config)
                metadata.provider_id = provider_id
                self.providers[provider_id] = metadata
                
                logger.debug(f"Loaded provider: {provider_id} ({metadata.name})")
        
        except Exception as e:
            logger.error(f"Error loading data sources config: {e}")
    
    def _load_state(self):
        """Load persistent state (health status, tokens) from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore health status
                for provider_id, health_data in state.get("health_status", {}).items():
                    health = ProviderHealthStatus()
                    health.status = ProviderStatus(health_data.get("status", "unknown"))
                    health.last_check_time = health_data.get("last_check_time")
                    health.response_time_ms = health_data.get("response_time_ms")
                    health.uptime_percentage = health_data.get("uptime_percentage", 100.0)
                    self.health_status[provider_id] = health
                
                # Restore token status
                for provider_id, token_status_str in state.get("token_status", {}).items():
                    self.token_status[provider_id] = TokenStatus(token_status_str)
                
                logger.debug("Loaded persistent state for providers")
        
        except Exception as e:
            logger.error(f"Error loading provider state: {e}")
    
    def _save_state(self):
        """Persist state (health status, tokens) to disk"""
        try:
            state = {
                "health_status": {
                    provider_id: health.to_dict()
                    for provider_id, health in self.health_status.items()
                },
                "token_status": {
                    provider_id: status.value
                    for provider_id, status in self.token_status.items()
                },
                "last_saved": datetime.now().isoformat()
            }
            
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving provider state: {e}")
    
    def _initialize_health_tracking(self):
        """Initialize health status for all providers"""
        for provider_id in self.providers.keys():
            if provider_id not in self.health_status:
                self.health_status[provider_id] = ProviderHealthStatus()
            if provider_id not in self.token_status:
                self.token_status[provider_id] = TokenStatus.NOT_SET
    
    def check_provider_health(self, provider_id: str) -> Tuple[ProviderStatus, Optional[str]]:
        """
        Check if a provider is online and healthy
        
        Args:
            provider_id: Provider identifier
        
        Returns:
            Tuple of (status, error_message)
        """
        if provider_id not in self.providers:
            return ProviderStatus.UNKNOWN, f"Provider not found: {provider_id}"
        
        provider = self.providers[provider_id]
        if not provider.enabled:
            return ProviderStatus.OFFLINE, "Provider disabled"
        
        health = self.health_status.get(provider_id, ProviderHealthStatus())
        
        # For local cache, always online
        if provider.type == "cache":
            health.status = ProviderStatus.ONLINE
            self.health_status[provider_id] = health
            return ProviderStatus.ONLINE, None
        
        # Check endpoint connectivity
        check_config = provider.status_check
        if not check_config:
            logger.warning(f"No health check config for {provider_id}")
            return ProviderStatus.UNKNOWN, "No health check configured"
        
        try:
            endpoint = check_config.get("endpoint")
            method = check_config.get("method", "GET").upper()
            timeout = check_config.get("timeout_seconds", 10)
            
            if not endpoint:
                return ProviderStatus.UNKNOWN, "No endpoint configured"
            
            start_time = datetime.now()
            
            # Special handling for different providers
            if provider_id == "kite_zerodha":
                # Check if access token is valid
                token_status = self.check_token_validity("kite_zerodha")
                if token_status != TokenStatus.VALID:
                    return ProviderStatus.OFFLINE, f"Invalid token: {token_status.value}"
                
                # Make a simple API call to verify connection
                try:
                    from utils.helpers import kite as get_kite
                    kite = get_kite()
                    if kite is None:
                        return ProviderStatus.OFFLINE, "KiteConnect not initialized"
                    kite.margins()  # Simple API call
                except Exception as e:
                    health.consecutive_failures += 1
                    return ProviderStatus.OFFLINE, str(e)
            
            elif provider_id in ["yahoo_finance", "nse_api"]:
                # HTTP endpoint check
                response = requests.get(endpoint, timeout=timeout)
                response.raise_for_status()
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update health status
            health.status = ProviderStatus.ONLINE
            health.response_time_ms = response_time
            health.consecutive_failures = 0
            health.last_successful_request = datetime.now().isoformat()
            health.error_message = None
            
            self.health_status[provider_id] = health
            return ProviderStatus.ONLINE, None
        
        except requests.Timeout:
            error_msg = "Connection timeout"
            health.consecutive_failures += 1
            health.status = ProviderStatus.OFFLINE
            health.error_message = error_msg
            self.health_status[provider_id] = health
            return ProviderStatus.OFFLINE, error_msg
        
        except Exception as e:
            error_msg = str(e)
            health.consecutive_failures += 1
            health.status = ProviderStatus.DEGRADED if health.consecutive_failures < 3 else ProviderStatus.OFFLINE
            health.error_message = error_msg
            self.health_status[provider_id] = health
            return health.status, error_msg
    
    def check_token_validity(self, provider_id: str) -> TokenStatus:
        """
        Check if provider's authentication token is valid and not expired
        
        Args:
            provider_id: Provider identifier
        
        Returns:
            TokenStatus
        """
        provider = self.providers.get(provider_id)
        if not provider:
            return TokenStatus.INVALID
        
        auth_config = provider.authentication
        if not auth_config:
            return TokenStatus.NOT_SET
        
        config_keys = auth_config.get("config_keys", [])
        
        # Check if all required credentials are set in environment
        missing_keys = [key for key in config_keys if not os.getenv(key)]
        
        if missing_keys:
            logger.debug(f"Missing credentials for {provider_id}: {missing_keys}")
            return TokenStatus.NOT_SET
        
        # For Zerodha, check token expiration
        if provider_id == "kite_zerodha":
            try:
                from utils.helpers import kite as get_kite
                kite = get_kite()
                if kite is None:
                    return TokenStatus.NOT_SET
                
                # Try a simple API call
                kite.margins()
                
                # Token is valid
                access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
                if access_token:
                    # Tokens typically last 180 days
                    # For simplicity, we assume it's valid if API call succeeds
                    self.token_status[provider_id] = TokenStatus.VALID
                    return TokenStatus.VALID
                
            except Exception as e:
                logger.debug(f"Token validation failed for {provider_id}: {e}")
                self.token_status[provider_id] = TokenStatus.EXPIRED
                return TokenStatus.EXPIRED
        
        # For public APIs, always valid if keys are set
        if provider.type == "alternative" and not auth_config.get("required"):
            self.token_status[provider_id] = TokenStatus.VALID
            return TokenStatus.VALID
        
        # Default: assume valid if credentials exist
        self.token_status[provider_id] = TokenStatus.VALID
        return TokenStatus.VALID
    
    def get_login_url(self, provider_id: str) -> Optional[str]:
        """Get login URL for a provider to obtain/refresh token"""
        provider = self.providers.get(provider_id)
        if not provider:
            return None
        
        auth_config = provider.authentication
        if not auth_config:
            return None
        
        return auth_config.get("login_url")
    
    def get_available_providers(self, exchange: str = None, segment: str = None) -> List[str]:
        """
        Get list of available providers filtered by exchange/segment
        
        Args:
            exchange: Filter by exchange (e.g., 'NSE', 'MCX')
            segment: Filter by segment (e.g., 'cash', 'futures', 'options')
        
        Returns:
            List of provider IDs sorted by priority
        """
        available = []
        
        for provider_id, provider in self.providers.items():
            # Must be enabled
            if not provider.enabled:
                continue
            
            # Check exchange match
            if exchange and exchange not in provider.exchanges:
                continue
            
            # Check segment match
            if segment and segment not in provider.segments:
                continue
            
            # Check if online
            status, _ = self.check_provider_health(provider_id)
            if status in [ProviderStatus.ONLINE, ProviderStatus.CACHE]:
                available.append((provider_id, provider.priority))
        
        # Sort by priority
        available.sort(key=lambda x: x[1])
        return [provider_id for provider_id, _ in available]
    
    def get_provider_info(self, provider_id: str) -> Optional[Dict]:
        """Get detailed information about a provider"""
        provider = self.providers.get(provider_id)
        if not provider:
            return None
        
        health = self.health_status.get(provider_id, ProviderHealthStatus())
        token_status = self.token_status.get(provider_id, TokenStatus.NOT_SET)
        
        return {
            "provider_id": provider_id,
            "name": provider.name,
            "type": provider.type,
            "enabled": provider.enabled,
            "priority": provider.priority,
            "description": provider.description,
            "exchanges": provider.exchanges,
            "segments": provider.segments,
            "timeframes": provider.timeframes,
            "data_type": provider.data_type,
            "health_status": health.status.value,
            "token_status": token_status.value,
            "last_check_time": health.last_check_time,
            "response_time_ms": health.response_time_ms,
            "uptime_percentage": health.uptime_percentage,
            "consecutive_failures": health.consecutive_failures,
            "login_url": self.get_login_url(provider_id)
        }
    
    def get_all_providers_status(self) -> Dict[str, Dict]:
        """Get status summary for all providers"""
        status_dict = {}
        
        for provider_id in self.providers.keys():
            info = self.get_provider_info(provider_id)
            if info:
                status_dict[provider_id] = info
        
        return status_dict
    
    def refresh_all_health_checks(self):
        """Refresh health status for all providers"""
        results = {}
        
        for provider_id in self.providers.keys():
            status, error = self.check_provider_health(provider_id)
            results[provider_id] = {
                "status": status.value,
                "error": error
            }
        
        self._save_state()
        return results
    
    def get_data_availability_matrix(self) -> Dict[str, Dict]:
        """
        Get data availability across all dimensions
        Returns matrix: exchange -> segment -> timeframe -> providers
        """
        matrix = {}
        
        for provider_id, provider in self.providers.items():
            if not provider.enabled:
                continue
            
            status, _ = self.check_provider_health(provider_id)
            if status == ProviderStatus.OFFLINE:
                continue
            
            for exchange in provider.exchanges:
                if exchange not in matrix:
                    matrix[exchange] = {}
                
                for segment in provider.segments:
                    if segment == "all":
                        # Skip generic 'all' - use actual segments
                        continue
                    
                    if segment not in matrix[exchange]:
                        matrix[exchange][segment] = {}
                    
                    timeframes = provider.timeframes if provider.timeframes != ["all"] else ["day", "week"]
                    for timeframe in timeframes:
                        if timeframe not in matrix[exchange][segment]:
                            matrix[exchange][segment][timeframe] = []
                        
                        matrix[exchange][segment][timeframe].append(provider_id)
        
        return matrix


def get_data_sources_manager() -> DataSourcesManager:
    """Get singleton instance of DataSourcesManager"""
    return DataSourcesManager()
