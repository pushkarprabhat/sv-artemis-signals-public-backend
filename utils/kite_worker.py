# utils/kite_worker.py - Initialize KiteConnect for Celery workers
# Workers run in separate processes and need their own KiteConnect instance

import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from utils.logger import logger

from dotenv import load_dotenv

# Always reload .env when asked for a kite instance so interactive token
# refreshes are picked up by long-running processes.
load_dotenv(override=True)

_kite_instance = None
_kite_access_token = None

def get_kite():
    """
    Get or initialize a KiteConnect instance for the current worker.
    Uses access_token from .env (saved after login).
    
    Returns:
        KiteConnect instance if authenticated, None otherwise
    """
    global _kite_instance
    global _kite_access_token

    # Reload .env to pick up any changes made at runtime
    try:
        load_dotenv(override=True)
    except Exception:
        pass

    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")

    # If we already have an instance and the token hasn't changed, return it
    if _kite_instance is not None and _kite_access_token == access_token:
        return _kite_instance
    
    if not api_key or not access_token:
        logger.error("ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN not found in .env")
        return None
    
    try:
        _kite_instance = KiteConnect(api_key=api_key)
        _kite_instance.set_access_token(access_token)
        _kite_access_token = access_token
        logger.info(f"✅ KiteConnect initialized for worker (token: {access_token[:20]}...)")
        return _kite_instance
    except Exception as e:
        logger.error(f"Failed to initialize KiteConnect: {e}")
        return None

def reset_kite():
    """Reset the KiteConnect instance (e.g., if token expires)"""
    global _kite_instance
    _kite_instance = None
