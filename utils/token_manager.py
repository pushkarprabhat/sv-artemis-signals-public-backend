"""
utils/token_manager.py - Handle token expiry and renewal
"""
import os
import webbrowser
import streamlit as st
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
from utils.logger import logger
from pathlib import Path

load_dotenv()


def get_token_from_env():
    """Get access token from .env file"""
    token = os.getenv("ZERODHA_ACCESS_TOKEN")
    return token if token else None


def save_token_to_env(access_token: str, env_file: str = ".env"):
    """Save access token to .env file"""
    try:
        set_key(env_file, "ZERODHA_ACCESS_TOKEN", access_token)
        logger.info(f"[OK] Saved new token to {env_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save token: {e}")
        return False


def is_token_expired(api_key: str, access_token: str) -> bool:
    """Check if token is expired by calling profile API
    
    Returns:
        True if token is expired/invalid, False if still valid
    """
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        profile = kite.profile()
        
        if profile and 'user_id' in profile:
            return False  # Token is valid
        else:
            return True  # Token is invalid
    except Exception as e:
        error_msg = str(e).lower()
        # Token is definitely expired if we get these errors
        if "invalid" in error_msg or "token" in error_msg or "unauthorized" in error_msg:
            return True
        # For other errors, assume token might still be valid
        return False


def validate_token(api_key: str, access_token: str) -> dict:
    """Validate if token is still valid by calling profile API
    
    Returns:
        {
            'valid': bool,
            'user_id': str or None,
            'message': str
        }
    """
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        profile = kite.profile()
        
        if profile and 'user_id' in profile:
            return {
                'valid': True,
                'user_id': profile.get('user_id'),
                'message': f"✅ Token valid for {profile.get('user_id')}"
            }
        else:
            return {
                'valid': False,
                'user_id': None,
                'message': "❌ Token validation failed"
            }
    except Exception as e:
        error_msg = str(e)
        if "invalid" in error_msg.lower() or "token" in error_msg.lower():
            return {
                'valid': False,
                'user_id': None,
                'message': f"❌ Invalid or expired token: {error_msg}"
            }
        return {
            'valid': False,
            'user_id': None,
            'message': f"❌ Token validation error: {error_msg}"
        }


def request_token_via_login(api_key: str, api_secret: str) -> str:
    """Request token via Zerodha login page
    
    Returns:
        Access token or None if failed
    """
    try:
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        
        # Fix the URL bug
        fixed_url = login_url.replace("api.kite.trade", "kite.zerodha.com")
        
        st.info(f"[GLOBE] Opening Zerodha login page...")
        st.write(f"[Click here to login]({fixed_url})")
        
        st.markdown("---")
        st.write("""
        **After login:**
        1. You'll see a URL with `request_token` parameter
        2. Copy the value after `request_token=` (long string)
        3. Paste it in the box below
        """)
        
        request_token = st.text_input(
            "🔐 Paste REQUEST_TOKEN from browser URL:",
            key="token_input",
            type="password"
        )
        
        if request_token:
            with st.spinner("🔄 Exchanging token..."):
                try:
                    data = kite.generate_session(request_token, api_secret=api_secret)
                    access_token = data.get("access_token")
                    
                    if access_token:
                        st.success(f"✅ Got token!")
                        
                        # Save to .env
                        if save_token_to_env(access_token):
                            st.success("✅ Token saved to .env file")
                            st.balloons()
                            return access_token
                        else:
                            st.warning("[WARN] Could not save token automatically")
                            st.write(f"**Manual save:** Add this to .env:")
                            st.code(f"ZERODHA_ACCESS_TOKEN={access_token}")
                            return access_token
                except Exception as e:
                    st.error(f"[ERROR] Token exchange failed: {e}")
                    return None
        
        return None
        
    except Exception as e:
        st.error(f"[ERROR] Login failed: {e}")
        return None


def show_token_renewal_popup(api_key: str, api_secret: str):
    """Show modal popup for token renewal when token expires
    
    This function shows a Streamlit modal with:
    - Clear message that token has expired
    - Button to open Kite login URL
    - Input field for REQUEST_TOKEN
    - Verification and auto-save
    """
    st.markdown("""
    <style>
        .token-popup-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .token-popup-modal {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 500px;
            color: white;
            animation: slideUp 0.3s ease-out;
        }
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .token-popup-modal h2 {
            margin-top: 0;
            color: #FFD700;
            font-size: 24px;
        }
        .token-popup-modal p {
            margin: 15px 0;
            line-height: 1.6;
        }
        .token-popup-modal .warning {
            background: rgba(255, 193, 7, 0.2);
            padding: 15px;
            border-left: 4px solid #FFD700;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for popup
    if "token_renewal_step" not in st.session_state:
        st.session_state.token_renewal_step = 1
    if "token_renewal_active" not in st.session_state:
        st.session_state.token_renewal_active = True
    
    if st.session_state.token_renewal_active:
        # Create modal using Streamlit's columns to center it
        col1, col2, col3 = st.columns([0.5, 2, 0.5])
        
        with col2:
            with st.container(border=True):
                st.markdown("## ⏱️ Token Expired")
                st.markdown("Your Zerodha access token has expired and needs to be renewed.")
                
                st.markdown("---")
                
                if st.session_state.token_renewal_step == 1:
                    # Step 1: Login
                    st.markdown("### Step 1: Login to Zerodha")
                    st.markdown("""
                    Click the button below to open Zerodha login page in your browser.
                    """)
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("🌐 Open Login", width="stretch", type="primary"):
                            try:
                                kite = KiteConnect(api_key=api_key)
                                login_url = kite.login_url()
                                
                                # Store URL in session
                                st.session_state.current_login_url = login_url
                                
                                # Open browser
                                try:
                                    webbrowser.open(login_url)
                                    st.success("✅ Browser opened! Complete login in your browser.")
                                except:
                                    st.info("[CLIPBOARD] Browser couldn't open. Use the link below:")
                                    st.markdown(f"[LINK Click here to login]({login_url})")
                                
                                st.session_state.token_renewal_step = 2
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    
                    with col_btn2:
                        if st.button("❌ Close", width="stretch"):
                            st.session_state.token_renewal_active = False
                            st.rerun()
                
                elif st.session_state.token_renewal_step == 2:
                    # Step 2: Get REQUEST_TOKEN
                    st.markdown("### Step 2: Enter REQUEST_TOKEN")
                    st.markdown("""
                    After login, you'll see a URL with `request_token` parameter.
                    
                    **Example URL:**
                    ```
                    https://kite.zerodha.com/?request_token=abc123xyz...&status=success
                    ```
                    
                    Copy the token value (long string) and paste it below.
                    """)
                    
                    request_token = st.text_input(
                        "🔐 REQUEST_TOKEN",
                        placeholder="Paste the token here...",
                        type="password",
                        key="renewal_request_token"
                    )
                    
                    col_submit1, col_submit2 = st.columns(2)
                    
                    with col_submit1:
                        if st.button("✅ Verify & Renew", width="stretch", type="primary"):
                            if not request_token or len(request_token) < 10:
                                st.error("[ERROR] Invalid token. Please paste the complete REQUEST_TOKEN.")
                            else:
                                with st.spinner("🔄 Exchanging token..."):
                                    try:
                                        kite = KiteConnect(api_key=api_key)
                                        session_data = kite.generate_session(
                                            request_token,
                                            api_secret=api_secret
                                        )
                                        
                                        new_access_token = session_data.get("access_token")
                                        
                                        if new_access_token:
                                            # Save token
                                            if save_token_to_env(new_access_token):
                                                os.environ["ZERODHA_ACCESS_TOKEN"] = new_access_token
                                                st.success("✅ Token renewed successfully!")
                                                st.balloons()
                                                
                                                st.session_state.token_renewal_active = False
                                                st.session_state.token_renewal_step = 1
                                                
                                                # Small delay then rerun
                                                import time
                                                time.sleep(1)
                                                st.rerun()
                                            else:
                                                st.error("[ERROR] Could not save token. Please try again.")
                                        else:
                                            st.error("[ERROR] Token exchange failed. Please try again.")
                                    except Exception as e:
                                        st.error(f"[ERROR] Error: {str(e)[:200]}")
                    
                    with col_submit2:
                        if st.button("🔙 Back", width="stretch"):
                            st.session_state.token_renewal_step = 1
                            st.rerun()


def check_and_validate_token():
    """Check token validity at app startup
    
    Returns:
        True if token is valid, False if needs input
    """
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    
    if not api_key or not access_token or not api_secret:
        st.error("[ERROR] Missing credentials in .env file")
        st.write("Please add to .env:")
        st.code("""
ZERODHA_API_KEY=your_key
ZERODHA_API_SECRET=your_secret
ZERODHA_ACCESS_TOKEN=your_token
""")
        return False
    
    validation = validate_token(api_key, access_token)
    
    if validation['valid']:
        return True
    else:
        # Show input dialog
        show_token_input_dialog(api_key, api_secret)
        return False
