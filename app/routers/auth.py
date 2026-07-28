"""
Auth Router — /auth/register, /auth/login, /auth/verify-2fa, /auth/me
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.auth.user_store import create_user, find_user_by_email, verify_password, delete_user
from app.auth.jwt_handler import create_token, require_auth
from app.sessions.store import get_session_store
from app.auth.totp_handler import generate_totp_secret, get_provisioning_uri, generate_qr_base64, verify_totp

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    emirates_id: str
    account_number: str
    account_type: str
    balance: str


class RegisterResponse(BaseModel):
    message: str
    qr_code_base64: str
    totp_secret: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    token: str
    requires_2fa: bool


class Verify2FARequest(BaseModel):
    token: str
    code: str


class Verify2FAResponse(BaseModel):
    token: str


class UserProfile(BaseModel):
    email: str
    name: str
    emirates_id: str
    account_number: str
    account_type: str
    balance: str


@router.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    # Check if user already exists
    if find_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Generate TOTP secret
    totp_secret = generate_totp_secret()

    # Create user in Qdrant
    create_user(
        email=req.email,
        password=req.password,
        name=req.name,
        emirates_id=req.emirates_id,
        account_number=req.account_number,
        account_type=req.account_type,
        balance=req.balance,
        totp_secret=totp_secret,
    )

    # Generate QR Code
    uri = get_provisioning_uri(req.email, totp_secret)
    qr_b64 = generate_qr_base64(uri)

    return RegisterResponse(
        message="User registered successfully. Please configure 2FA.",
        qr_code_base64=qr_b64,
        totp_secret=totp_secret
    )


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    user = find_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Issue a token with 2fa_complete = False
    token = create_token(user["email"], is_2fa_complete=False)

    return LoginResponse(
        token=token,
        requires_2fa=True
    )


@router.post("/verify-2fa", response_model=Verify2FAResponse)
async def verify_2fa(req: Verify2FARequest):
    from app.auth.jwt_handler import decode_token
    
    # Decode the pending token
    try:
        payload = decode_token(req.token)
    except HTTPException as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
        
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = find_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_totp(user["totp_secret"], req.code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")

    # Issue full token
    full_token = create_token(user["email"], is_2fa_complete=True)

    return Verify2FAResponse(token=full_token)


@router.get("/me", response_model=UserProfile)
async def get_me(token_payload: dict = Depends(require_auth)):
    email = token_payload.get("sub")
    user = find_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserProfile(
        email=user["email"],
        name=user["name"],
        emirates_id=user["emirates_id"],
        account_number=user["account_number"],
        account_type=user["account_type"],
        balance=user["balance"]
    )


@router.get("/export")
async def export_data(token_payload: dict = Depends(require_auth)):
    """Export all user data (profile and chats) for data portability compliance."""
    email = token_payload.get("sub")
    user = find_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    store = get_session_store()
    chats = store.get_user_chats(email)
    
    # Exclude sensitive internal data
    user.pop("password_hash", None)
    user.pop("totp_secret", None)
    user.pop("_point_id", None)
    
    return {
        "profile": user,
        "chats": chats
    }


@router.delete("/account")
async def delete_account(token_payload: dict = Depends(require_auth)):
    """Permanently delete user account and all associated data (Right to be Forgotten)."""
    email = token_payload.get("sub")
    
    # 1. Delete all chat sessions from Redis
    store = get_session_store()
    store.delete_user_data(email)
    
    # 2. Delete user profile from Qdrant
    success = delete_user(email)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete user profile")
        
    return {"message": "Account and all associated data have been permanently deleted"}
