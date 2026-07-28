"""
JWT Handler — Token creation, verification, and FastAPI dependency
"""
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings

security = HTTPBearer(auto_error=False)


def create_token(email: str, is_2fa_complete: bool = False) -> str:
    """Create a JWT token for the given user."""
    s = get_settings()
    payload = {
        "sub": email,
        "2fa_complete": is_2fa_complete,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises on failure."""
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    FastAPI dependency that requires a valid JWT with 2FA completed.
    Returns the decoded JWT payload.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_token(credentials.credentials)

    if not payload.get("2fa_complete"):
        raise HTTPException(status_code=403, detail="2FA verification required")

    return payload


def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    FastAPI dependency that extracts auth if present, returns None for guests.
    """
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        if not payload.get("2fa_complete"):
            return None
        return payload
    except Exception:
        return None
