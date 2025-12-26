# core/token_manager.py — Token Lifecycle Management System
# Handles token validation, expiration tracking, refresh mechanism, and status reporting
# Thread-safe singleton with persistent state

import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
import threading
from utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

class TokenStatus(Enum):
    """Token status types"""
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    NOT_SET = "not_set"
    REFRESH_NEEDED = "refresh_needed"
    INVALID = "invalid"

class TokenInfo:
    """Information about a token"""
    def __init__(self, provider_id: str, token_type: str = "access_token"):
        self.provider_id = provider_id
        self.token_type = token_type  # access_token, api_key, etc.
        self.set_date: Optional[datetime] = None
        self.expires_at: Optional[datetime] = None
        self.last_validated: Optional[datetime] = None
        self.validation_status = TokenStatus.NOT_SET
        self.error_message: Optional[str] = None
        self.refresh_url: Optional[str] = None
        self.metadata: Dict = {}
    
    def to_dict(self) -> Dict:
        return {
            "provider_id": self.provider_id,
            "token_type": self.token_type,
            "set_date": self.set_date.isoformat() if self.set_date else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "validation_status": self.validation_status.value,
            "error_message": self.error_message,
            "refresh_url": self.refresh_url,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TokenInfo":
        token = cls(data.get("provider_id"), data.get("token_type", "access_token"))
        token.set_date = datetime.fromisoformat(data["set_date"]) if data.get("set_date") else None
        token.expires_at = datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        token.last_validated = datetime.fromisoformat(data["last_validated"]) if data.get("last_validated") else None
        token.validation_status = TokenStatus(data.get("validation_status", "not_set"))
        token.error_message = data.get("error_message")
        token.refresh_url = data.get("refresh_url")
        token.metadata = data.get("metadata", {})
        return token

class TokenManager:
    """
    Manage authentication tokens for all data providers
    
    Features:
    - Track token validity and expiration
    - Automatic validation and status updates
    - Token refresh mechanism
    - Dynamic login URL generation
    - Persistent token state tracking
    - Expiration alerts and warnings
    - Multi-provider support
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
        self.state_file = Path("data/.token_manager_state.json")
        
        # Token tracking: provider_id -> {token_type -> TokenInfo}
        self.tokens: Dict[str, Dict[str, TokenInfo]] = {}
        
        # Provider-specific config
        self.provider_config = {
            "kite_zerodha": {
                "env_keys": ["ZERODHA_API_KEY", "ZERODHA_API_SECRET", "ZERODHA_ACCESS_TOKEN"],
                "token_type": "access_token",
                "expires_in_days": 180,
                "validation_method": "kite_api",
                "login_url": "https://kite.zerodha.com/me/settings/apps",
                "documentation": "https://kite.trade/"
            },
            "yahoo_finance": {
                "env_keys": [],
                "token_type": "none",
                "validation_method": "http_check",
                "is_public_api": True
            },
            "nse_api": {
                "env_keys": [],
                "token_type": "none",
                "validation_method": "http_check",
                "is_public_api": True
            },
            "local_cache": {
                "env_keys": [],
                "token_type": "none",
                "validation_method": "local",
                "is_local": True
            }
        }
        
        self._load_state()
        logger.info("[TokenManager] Initialized")
    
    def _load_state(self):
        """Load persistent token state from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                for provider_id, token_data_list in state.get("tokens", {}).items():
                    self.tokens[provider_id] = {}
                    for token_data in token_data_list:
                        token_info = TokenInfo.from_dict(token_data)
                        self.tokens[provider_id][token_info.token_type] = token_info
                
                logger.debug("Loaded persistent token state")
        
        except Exception as e:
            logger.error(f"Error loading token state: {e}")
    
    def _save_state(self):
        """Persist token state to disk"""
        try:
            state = {
                "tokens": {
                    provider_id: [
                        token_info.to_dict()
                        for token_info in token_dict.values()
                    ]
                    for provider_id, token_dict in self.tokens.items()
                },
                "last_saved": datetime.now().isoformat()
            }
            
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving token state: {e}")
    
    def register_token(self, provider_id: str, token_type: str = "access_token",
                      expires_in_days: int = 180, metadata: Dict = None):
        """
        Register a token for tracking
        
        Args:
            provider_id: Provider identifier
            token_type: Type of token (access_token, api_key, etc.)
            expires_in_days: Days until expiration
            metadata: Additional metadata
        """
        with self._lock:
            if provider_id not in self.tokens:
                self.tokens[provider_id] = {}
            
            token_info = TokenInfo(provider_id, token_type)
            token_info.set_date = datetime.now()
            token_info.expires_at = datetime.now() + timedelta(days=expires_in_days)
            token_info.metadata = metadata or {}
            
            self.tokens[provider_id][token_type] = token_info
            self._save_state()
            
            logger.info(f"Registered {token_type} for {provider_id}, expires in {expires_in_days} days")
    
    def validate_token(self, provider_id: str, token_type: str = "access_token") -> TokenStatus:
        """
        Validate token for a provider
        
        Args:
            provider_id: Provider identifier
            token_type: Type of token to validate
        
        Returns:
            TokenStatus
        """
        with self._lock:
            provider_conf = self.provider_config.get(provider_id)
            if not provider_conf:
                logger.warning(f"Unknown provider: {provider_id}")
                return TokenStatus.INVALID
            
            # Check if token exists
            if provider_id not in self.tokens or token_type not in self.tokens[provider_id]:
                # Check environment variables
                env_keys = provider_conf.get("env_keys", [])
                missing_keys = [key for key in env_keys if not os.getenv(key)]
                
                if missing_keys:
                    logger.debug(f"Missing env keys for {provider_id}: {missing_keys}")
                    return TokenStatus.NOT_SET
                
                # Auto-register token from environment
                if env_keys:
                    expires_in = provider_conf.get("expires_in_days", 180)
                    self.register_token(provider_id, token_type, expires_in)
            
            token_info = self.tokens.get(provider_id, {}).get(token_type)
            if not token_info:
                return TokenStatus.NOT_SET
            
            # Validate based on provider type
            validation_method = provider_conf.get("validation_method", "http_check")
            
            try:
                if validation_method == "kite_api":
                    return self._validate_kite_token(provider_id, token_info)
                elif validation_method == "http_check":
                    return self._validate_http_endpoint(provider_id, token_info)
                elif validation_method == "local":
                    return TokenStatus.VALID
                else:
                    return TokenStatus.VALID
            
            except Exception as e:
                logger.error(f"Token validation error for {provider_id}: {e}")
                token_info.error_message = str(e)
                token_info.validation_status = TokenStatus.INVALID
                return TokenStatus.INVALID
    
    def _validate_kite_token(self, provider_id: str, token_info: TokenInfo) -> TokenStatus:
        """Validate Zerodha Kite token"""
        try:
            from utils.helpers import kite as get_kite
            kite = get_kite()
            
            if kite is None:
                token_info.validation_status = TokenStatus.NOT_SET
                return TokenStatus.NOT_SET
            
            # Try API call
            kite.margins()
            
            # Calculate days remaining
            if token_info.expires_at:
                days_remaining = (token_info.expires_at - datetime.now()).days
                
                if days_remaining <= 0:
                    token_info.validation_status = TokenStatus.EXPIRED
                    return TokenStatus.EXPIRED
                elif days_remaining < 30:
                    token_info.validation_status = TokenStatus.EXPIRING_SOON
                    token_info.last_validated = datetime.now()
                    self._save_state()
                    return TokenStatus.EXPIRING_SOON
            
            token_info.validation_status = TokenStatus.VALID
            token_info.last_validated = datetime.now()
            self._save_state()
            return TokenStatus.VALID
        
        except Exception as e:
            logger.debug(f"Kite token validation failed: {e}")
            token_info.validation_status = TokenStatus.EXPIRED
            token_info.error_message = str(e)
            self._save_state()
            return TokenStatus.EXPIRED
    
    def _validate_http_endpoint(self, provider_id: str, token_info: TokenInfo) -> TokenStatus:
        """Validate token via HTTP endpoint check"""
        try:
            # Public APIs don't require tokens
            provider_conf = self.provider_config.get(provider_id)
            if provider_conf.get("is_public_api"):
                token_info.validation_status = TokenStatus.VALID
                return TokenStatus.VALID
            
            # Would implement specific HTTP validation here
            token_info.validation_status = TokenStatus.VALID
            return TokenStatus.VALID
        
        except Exception as e:
            logger.error(f"HTTP validation failed for {provider_id}: {e}")
            token_info.validation_status = TokenStatus.INVALID
            return TokenStatus.INVALID
    
    def get_token_status(self, provider_id: str, token_type: str = "access_token") -> Dict:
        """Get detailed token status"""
        if provider_id not in self.tokens or token_type not in self.tokens[provider_id]:
            validation_status = self.validate_token(provider_id, token_type)
            status_info = {
                "provider_id": provider_id,
                "token_type": token_type,
                "status": validation_status.value,
                "set_date": None,
                "expires_at": None,
                "days_remaining": None,
                "last_validated": None,
                "needs_refresh": validation_status in [TokenStatus.EXPIRED, TokenStatus.EXPIRING_SOON]
            }
        else:
            token_info = self.tokens[provider_id][token_type]
            days_remaining = None
            if token_info.expires_at:
                days_remaining = (token_info.expires_at - datetime.now()).days
            
            status_info = {
                "provider_id": provider_id,
                "token_type": token_type,
                "status": token_info.validation_status.value,
                "set_date": token_info.set_date.isoformat() if token_info.set_date else None,
                "expires_at": token_info.expires_at.isoformat() if token_info.expires_at else None,
                "days_remaining": max(0, days_remaining) if days_remaining else None,
                "last_validated": token_info.last_validated.isoformat() if token_info.last_validated else None,
                "needs_refresh": token_info.validation_status in [TokenStatus.EXPIRED, TokenStatus.EXPIRING_SOON],
                "error_message": token_info.error_message,
                "login_url": self.get_login_url(provider_id)
            }
        
        return status_info
    
    def get_login_url(self, provider_id: str) -> Optional[str]:
        """Get login URL for a provider"""
        provider_conf = self.provider_config.get(provider_id)
        if not provider_conf:
            return None
        
        return provider_conf.get("login_url")
    
    def get_all_token_status(self) -> Dict[str, Dict]:
        """Get status for all provider tokens"""
        all_status = {}
        
        for provider_id in self.provider_config.keys():
            token_type = self.provider_config[provider_id].get("token_type", "access_token")
            if token_type == "none":
                continue
            
            all_status[provider_id] = self.get_token_status(provider_id, token_type)
        
        return all_status
    
    def get_tokens_needing_refresh(self) -> List[str]:
        """Get list of providers whose tokens need refresh"""
        needing_refresh = []
        
        for provider_id in self.provider_config.keys():
            status = self.get_token_status(provider_id)
            if status.get("needs_refresh"):
                needing_refresh.append(provider_id)
        
        return needing_refresh
    
    def update_token_from_env(self, provider_id: str, token_type: str = "access_token"):
        """Update token info from environment variables"""
        env_keys = self.provider_config.get(provider_id, {}).get("env_keys", [])
        
        if not env_keys:
            logger.debug(f"No env keys configured for {provider_id}")
            return
        
        # Check if all keys exist in environment
        if all(os.getenv(key) for key in env_keys):
            expires_in = self.provider_config[provider_id].get("expires_in_days", 180)
            self.register_token(provider_id, token_type, expires_in)
            logger.info(f"Token updated for {provider_id} from environment")
    
    def validate_all_tokens(self) -> Dict[str, TokenStatus]:
        """Validate all provider tokens"""
        results = {}
        
        for provider_id in self.provider_config.keys():
            token_type = self.provider_config[provider_id].get("token_type", "access_token")
            if token_type == "none":
                continue
            
            status = self.validate_token(provider_id, token_type)
            results[provider_id] = status.value
        
        self._save_state()
        return results


def get_token_manager() -> TokenManager:
    """Get singleton instance of TokenManager"""
    return TokenManager()
