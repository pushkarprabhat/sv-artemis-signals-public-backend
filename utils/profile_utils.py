"""
utils/profile_utils.py - Extract user profile info from Kite and display in dashboard
"""
import os
from kiteconnect import KiteConnect
from typing import Dict, Optional
import streamlit as st
from utils.logger import logger

def get_kite_user_profile(kite: Optional[KiteConnect] = None) -> Dict:
    """Get user profile information from Kite
    
    Args:
        kite: KiteConnect instance. If None, tries to initialize from env credentials.
    
    Returns:
        {
            'user': str,
            'user_id': str,
            'email': str,
            'broker': str,
            'api_key': str,
            'status': 'success' or 'error',
            'message': str
        }
    """
    try:
        # If no kite instance provided, create one
        if kite is None:
            api_key = os.getenv("ZERODHA_API_KEY")
            access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
            
            if not api_key or not access_token:
                return {
                    'status': 'error',
                    'message': 'Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN in .env'
                }
            
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
        
        # Fetch profile
        profile = kite.profile()
        
        # Extract key fields
        result = {
            'user': profile.get('user_name', 'N/A'),
            'user_id': profile.get('user_id', 'N/A'),
            'email': profile.get('email', 'N/A'),
            'broker': profile.get('broker', 'Zerodha'),
            'api_key': os.getenv("ZERODHA_API_KEY", "").replace(os.getenv("ZERODHA_API_KEY", "")[8:], "..."),
            'status': 'success',
            'message': 'Profile loaded successfully'
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Could not fetch user profile: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'user': 'Unknown',
            'user_id': 'N/A',
            'email': 'N/A',
            'broker': 'N/A',
            'api_key': 'N/A'
        }

def display_user_profile_card(kite: Optional[KiteConnect] = None):
    """Display user profile information in Streamlit sidebar
    
    Args:
        kite: KiteConnect instance for fetching profile
    """
    profile = get_kite_user_profile(kite)
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("👤 User Profile")
        
        if profile['status'] == 'success':
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**User:** {profile['user']}")
                st.caption(f"ID: {profile['user_id']}")
            with col2:
                st.success("✅")
            
            st.caption(f"📧 {profile['email']}")
            st.caption(f"🏢 {profile['broker']}")
            st.caption(f"🔑 API: {profile['api_key']}")
        else:
            st.error(f"[ERROR] {profile['message']}")

def get_profile_summary_html(kite: Optional[KiteConnect] = None) -> str:
    """Get user profile as HTML for display
    
    Returns:
        HTML string with profile information
    """
    profile = get_kite_user_profile(kite)
    
    if profile['status'] == 'error':
        return f"<div style='color: red;'>❌ {profile['message']}</div>"
    
    html = f"""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 15px; border-radius: 10px; border-left: 4px solid #FFD700;'>
        <div style='color: #FFD700; font-weight: bold; font-size: 16px;'>
            👤 {profile['user']}
        </div>
        <div style='color: #C9D1D9; margin-top: 8px; font-size: 12px;'>
            <div>📧 {profile['email']}</div>
            <div>🆔 {profile['user_id']}</div>
            <div>🏢 {profile['broker']}</div>
        </div>
    </div>
    """
    
    return html
