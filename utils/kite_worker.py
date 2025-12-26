# utils/kite_worker.py - Initialize KiteConnect for Celery workers
# Workers run in separate processes and need their own KiteConnect instance

import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

_kite_instance = None

def get_kite():
    """
    Get or initialize a KiteConnect instance for the current worker.
    Uses access_token from .env (saved after login).
    
    Returns:
        KiteConnect instance if authenticated, None otherwise
    """
    global _kite_instance
    
    if _kite_instance is not None:
        return _kite_instance
    
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    if not api_key or not access_token:
        logger.error("ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN not found in .env")
        return None
    
    try:
        _kite_instance = KiteConnect(api_key=api_key)
        _kite_instance.set_access_token(access_token)
        logger.info(f"✅ KiteConnect initialized for worker (token: {access_token[:20]}...)")
        return _kite_instance
    except Exception as e:
        logger.error(f"Failed to initialize KiteConnect: {e}")
        return None

def reset_kite():
    """Reset the KiteConnect instance (e.g., if token expires)"""
    global _kite_instance
    _kite_instance = None
