# core/auth_manager.py — Zerodha Authentication Manager
# Handles token validation, expiration checks, and automatic refresh
# Professional authentication and session management

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
from config.config import TELEGRAM_TOKEN
from utils.logger import logger

from typing import Dict, Any, Optional

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None
    # For Shivaansh & Krishaansh — mock KiteConnect for local/dev
    class KiteConnect:
        def __init__(self, api_key=None):
            pass
        def set_access_token(self, token):
            pass
        def profile(self):
            return {"user_name": "mock_user"}


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        if not username or not tenant_id:
            return None
        return {"username": username, "tenant_id": tenant_id}
    except JWTError:
        return None


class AuthenticationManager:
    """Manages Zerodha KiteConnect authentication and token lifecycle"""
    
    # Token lifetime (Zerodha tokens expire after 24 hours or logout)
    TOKEN_EXPIRY_HOURS = 24
    
    def __init__(self):
        """Initialize authentication manager"""
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        self.kite = None
        self.token_obtained_time = None
        self.is_valid = False
        
    def validate_token(self) -> bool:
        """Validate if current access token is valid by making a test API call
        
        Returns:
            True if token is valid, False otherwise
        """
        if not self.access_token:
            logger.error("[ERROR] No access token available")
            return False
        
        try:
            # Create temporary Kite instance
            kite_test = KiteConnect(api_key=self.api_key)
            kite_test.set_access_token(self.access_token)
            
            # Try to fetch account profile (lightweight API call)
            profile = kite_test.profile()
            
            if profile:
                self.is_valid = True
                self.kite = kite_test
                user_name = profile.get("user_name", "Unknown") if isinstance(profile, dict) else "mock_user"
                logger.info(f"[OK] Token is valid. User: {user_name}")
                return True
        except Exception as e:
            logger.error(f"[FAILED] Token validation failed: {e}")
            self.is_valid = False
            return False
    
    def check_token_expiry(self) -> Dict[str, Any]:
        """Check if token is likely expired based on age and validity
        
        Returns:
            Dictionary with:
                - 'is_expired': bool - whether token is expired
                - 'is_valid': bool - whether token is currently working
                - 'hours_remaining': int - estimated hours until expiry
                - 'status': str - human readable status
                - 'action': str - recommended action
        """
        result = {
            'is_expired': False,
            'is_valid': False,
            'hours_remaining': 0,
            'status': 'Unknown',
            'action': 'Unknown'
        }
        
        # Test current token validity
        is_currently_valid = self.validate_token()
        result['is_valid'] = is_currently_valid
        
        if not is_currently_valid:
            result['is_expired'] = True
            result['status'] = '❌ Token has expired or is invalid'
            result['action'] = 'Generate a new token from Zerodha login'
            return result
        
        # If valid, estimate time remaining
        if self.token_obtained_time:
            elapsed = datetime.now() - self.token_obtained_time
            hours_elapsed = elapsed.total_seconds() / 3600
            hours_remaining = max(0, self.TOKEN_EXPIRY_HOURS - hours_elapsed)
            result['hours_remaining'] = int(hours_remaining)
            
            if hours_remaining > 12:
                result['status'] = f"✅ Token is valid ({int(hours_remaining)}h remaining)"
                result['action'] = 'No action needed'
            elif hours_remaining > 0:
                result['status'] = f"⚠️ Token expiring soon ({int(hours_remaining)}h remaining)"
                result['action'] = 'Consider refreshing token before expiry'
            else:
                result['is_expired'] = True
                result['status'] = '❌ Token has expired'
                result['action'] = 'Generate a new token immediately'
        else:
            result['status'] = '✅ Token appears valid'
            result['action'] = 'Monitor for expiry'
            result['hours_remaining'] = self.TOKEN_EXPIRY_HOURS
        
        return result
    
    def get_kite_instance(self):
        """Get authenticated Kite instance
        
        Returns:
            KiteConnect instance if valid, None otherwise
        """
        if not self.is_valid:
            self.validate_token()
        return self.kite
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Kite connection and retrieve account info
        
        Returns:
            Dictionary with connection status and account details
        """
        try:
            if not self.kite:
                if not self.validate_token():
                    return {
                        'connected': False,
                        'message': '❌ Failed to validate token',
                        'details': None
                    }
            
            # Fetch profile
            profile = self.kite.profile()
            
            # Fetch margins
            margins = self.kite.margins()
            
            return {
                'connected': True,
                'message': '✅ Connected to Zerodha',
                'details': {
                    'user': profile.get('user_name'),
                    'email': profile.get('email'),
                    'broker': profile.get('broker'),
                    'available_balance': margins.get('equity', {}).get('available', 0),
                    'used_balance': margins.get('equity', {}).get('used', 0),
                    'net_value': margins.get('equity', {}).get('net', 0),
                }
            }
        except Exception as e:
            return {
                'connected': False,
                'message': f'❌ Connection test failed: {e}',
                'details': None
            }
    
    @staticmethod
    def generate_login_url() -> str:
        """Generate Zerodha login URL for manual token generation
        
        Returns:
            URL to login and generate new access token
        """
        api_key = os.getenv("ZERODHA_API_KEY")
        if not api_key:
            return "Error: ZERODHA_API_KEY not configured"
        
        base_url = "https://kite.zerodha.com/connect/login"
        login_url = f"{base_url}?api_key={api_key}"
        return login_url
    
    def exchange_request_token(self, request_token: str) -> bool:
        """Exchange request token for access token and update .env
        
        Args:
            request_token: The request token from Zerodha login
            
        Returns:
            True if successful, False otherwise
        """
        try:
            api_secret = os.getenv("ZERODHA_API_SECRET")
            if not api_secret:
                logger.error("[ERROR] ZERODHA_API_SECRET not found in .env")
                return False
                
            # Initialize Kite
            kite = KiteConnect(api_key=self.api_key)
            
            # Generate session
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            if not access_token:
                logger.error("[ERROR] Failed to generate access token")
                return False
            
            # Update instance state
            self.access_token = access_token
            self.kite = kite
            self.kite.set_access_token(access_token)
            self.is_valid = True
            self.token_obtained_time = datetime.now()
            
            # Update .env file
            try:
                env_path = os.path.join(os.getcwd(), ".env")
                lines = []
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        lines = f.readlines()
                
                new_lines = []
                found = False
                for line in lines:
                    if line.strip().startswith("ZERODHA_ACCESS_TOKEN="):
                        new_lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}\n")
                        found = True
                    else:
                        new_lines.append(line)
                
                if not found:
                    new_lines.append(f"\nZERODHA_ACCESS_TOKEN={access_token}\n")
                
                with open(env_path, "w") as f:
                    f.writelines(new_lines)
                    
                logger.info("[OK] Access token updated in .env")
                
                # Update global helpers
                from utils.helpers import set_kite
                set_kite(self.kite)
                
                return True
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to update .env: {e}")
                # Even if .env update fails, we have the token in memory
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] Token exchange failed: {e}")
            return False

    @staticmethod
    def generate_token_instructions() -> str:
        """Generate step-by-step instructions for manual token generation
        
        Returns:
            Formatted instructions
        """
        api_key = os.getenv("ZERODHA_API_KEY")
        
        instructions = f"""
╔════════════════════════════════════════════════════════════════╗
║         ZERODHA ACCESS TOKEN GENERATION INSTRUCTIONS            ║
╚════════════════════════════════════════════════════════════════╝

📋 STEP-BY-STEP GUIDE:

1️⃣  OPEN LOGIN PAGE:
   Visit: https://kite.zerodha.com/connect/login?api_key={api_key}
   
2️⃣  LOGIN TO ZERODHA:
   - Enter your username (email or client ID)
   - Enter your password
   - Complete 2FA/OTP verification
   
3️⃣  AUTHORIZE YOUR APP:
   - You'll be redirected to authorize "SV Pair Trading"
   - Click "Approve" or "Authorize"
   
4️⃣  COPY REQUEST TOKEN:
   - You'll see a page with "Request Token" (looks like: abc123def456)
   - Copy the entire token
   
5️⃣  GENERATE ACCESS TOKEN:
   - Use the script: python scripts/generate_access_token.py
   - Paste the request token when prompted
   
6️⃣  UPDATE .env FILE:
   - Add to your .env file:
   
   ZERODHA_API_KEY={api_key}
   ZERODHA_ACCESS_TOKEN=<paste_here>
   ZERODHA_REQUEST_TOKEN=<if_needed>
   
7️⃣  RESTART THE APPLICATION:
   - Save the .env file
   - Restart Streamlit app: streamlit run main.py

⏰ TOKEN VALIDITY:
   ✓ Access tokens are valid for 24 hours
   ✓ After 24 hours or logout, you need a new token
   ✓ Request tokens expire after 30 minutes if not used

🔄 AUTOMATIC REFRESH (Coming Soon):
   The app will automatically warn you before token expiry
   and provide an easy refresh mechanism.

❓ NEED HELP?
   - Check .env file exists in project root
   - Verify API credentials are correct
   - Ensure 2FA is not blocking the connection
   - Check Zerodha API status at: https://status.zerodha.com/

╚════════════════════════════════════════════════════════════════╝
"""
        return instructions


# Global authentication manager instance
_auth_manager = None


def get_auth_manager() -> AuthenticationManager:
    """Get or create global authentication manager
    
    Returns:
        AuthenticationManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


def validate_zerodha_token() -> bool:
    """Quick validation of Zerodha token
    
    Returns:
        True if token is valid, False otherwise
    """
    manager = get_auth_manager()
    return manager.validate_token()


def check_token_status() -> dict:
    """Check current token status
    
    Returns:
        Dictionary with token status information
    """
    manager = get_auth_manager()
    return manager.check_token_expiry()


def test_zerodha_connection() -> dict:
    """Test connection to Zerodha
    
    Returns:
        Dictionary with connection test results
    """
    manager = get_auth_manager()
    return manager.test_connection()
