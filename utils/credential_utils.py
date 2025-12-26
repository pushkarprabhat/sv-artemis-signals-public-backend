"""
utils/credential_utils.py - Credential status display utilities for dashboard
Shows token status, expiry time, and credential validation in Streamlit UI
"""

import os
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

load_dotenv()

def get_credential_status():
    """Get status of all Zerodha credentials"""
    
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    status = {
        "api_key_set": bool(api_key),
        "api_secret_set": bool(api_secret),
        "access_token_set": bool(access_token),
        "all_set": bool(api_key and api_secret and access_token),
        "api_key_display": _mask_token(api_key, 6) if api_key else "NOT SET",
        "api_secret_display": _mask_token(api_secret, 6) if api_secret else "NOT SET",
        "access_token_display": _mask_token(access_token, 8) if access_token else "NOT SET",
    }
    
    return status

def _mask_token(token, show_chars=8):
    """Mask a token showing only first and last chars"""
    if not token or len(token) < show_chars:
        return "***" if token else "NOT SET"
    return f"{token[:show_chars]}...{token[-4:]}"

def display_credential_status():
    """Display credential status in Streamlit sidebar"""
    
    status = get_credential_status()
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔐 Credentials Status")
        
        # Overall status
        if status["all_set"]:
            st.success("✅ All credentials configured")
        else:
            st.error("[ERROR] Missing credentials")
        
        # Individual credentials
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.text("API Key:")
            st.caption(status["api_key_display"])
        with col2:
            if status["api_key_set"]:
                st.success("✅")
            else:
                st.error("[ERROR]")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text("API Secret:")
            st.caption(status["api_secret_display"])
        with col2:
            if status["api_secret_set"]:
                st.success("✅")
            else:
                st.error("[ERROR]")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text("Access Token:")
            st.caption(status["access_token_display"])
        with col2:
            if status["access_token_set"]:
                st.success("✅")
            else:
                st.error("[ERROR]")
        
        # Action buttons
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Token"):
                st.info("Run: `python login.py`")
        with col2:
            if st.button("✓ Check Connection"):
                st.info("Run: `python scripts/check_kite_credentials.py --test`")

def display_credential_warning():
    """Display warning if credentials are missing or invalid"""
    
    status = get_credential_status()
    
    if not status["all_set"]:
        st.warning(
            """
            ⚠️ **Missing Credentials**
            
            To use the trading dashboard, add your Zerodha credentials:
            
            1. Get API Key & Secret from: https://kite.zerodha.com/me/settings/apps
            2. Add to `.env` file:
               ```
               ZERODHA_API_KEY=your_key
               ZERODHA_API_SECRET=your_secret
               ```
            3. Generate access token:
               ```
               python login.py
               ```
            4. The app will auto-save token to `.env`
            5. Restart this app
            
            See `.env.example` for complete setup guide.
            """,
            icon="⚠️"
        )
        return False
    
    return True

if __name__ == "__main__":
    # Test the utilities
    print("Credential Status:")
    status = get_credential_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
