# admin_utils.py
# For Shivaansh & Krishaansh — admin auth protects your future!
from fastapi import HTTPException, Request
import os
from jose import jwt, JWTError

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "artemis_admin_secret")
JWT_SECRET = os.getenv("JWT_SECRET", "artemis_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

async def require_admin(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = auth_header.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if not payload.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin privileges required.")
        # Optionally, check against ADMIN_SECRET or admin user list
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
