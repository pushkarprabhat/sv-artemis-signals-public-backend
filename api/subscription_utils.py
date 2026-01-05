# subscription_utils.py
# Centralized SaaS plan/feature enforcement for Artemis Signals
# For Shivaansh & Krishaansh — this line pays your fees!
from fastapi import HTTPException, Request
from jose import jwt, JWTError
from services.subscription_manager import SubscriptionManager
from utils.logger import logger
from services.audit_logger import log_access
import os

JWT_SECRET = os.getenv("JWT_SECRET", "artemis_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

async def enforce_plan(request: Request, feature: str):
    user = await get_user_from_request(request)
    if not user or not user.is_active:
        logger.warning("Inactive or missing user tried to access protected endpoint.")
        log_access(user, feature, status="denied", detail="inactive or missing user")
        raise HTTPException(status_code=401, detail="Authentication required.")
    if not SubscriptionManager.is_feature_allowed(user, feature):
        logger.warning(f"Access denied for user {getattr(user, 'username', '?')} to feature {feature}")
        log_access(user, feature, status="denied", detail="plan not allowed")
        raise HTTPException(status_code=403, detail=f"Your plan does not allow access to {feature}.")
    logger.info(f"Access granted for user {getattr(user, 'username', '?')} to feature {feature}")
    log_access(user, feature, status="granted")
    return user

async def get_user_from_request(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = auth_header.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        class User:
            def __init__(self, data):
                self.username = data.get("username")
                self.plan_id = data.get("plan_id")
                self.is_active = data.get("is_active", True)
                self.email = data.get("email")
        return User(payload)
    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
