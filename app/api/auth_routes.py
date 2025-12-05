"""
Authentication API Routes

Handles user signup, verification, login, and Google Sign-In.
Supports both email/OTP and Firebase Google authentication.
"""
import random
import string
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from app.services.database import (
    create_user, 
    verify_user, 
    get_user_by_id, 
    get_user_by_email,
    get_user_credits,
    get_or_create_user_by_firebase,
    get_user_by_email_password,
    add_to_waitlist
)
from app.services.password_auth import verify_password
from app.services.jwt_auth import (
    create_access_token, 
    create_refresh_token,
    validate_authorization_header,
    validate_refresh_token
)
from app.services.firebase_auth import verify_firebase_token
from app.config import SENDGRID_API_KEY, DEFAULT_ASK_DOUBT_CREDITS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SignupRequest(BaseModel):
    email: EmailStr


class WaitlistRequest(BaseModel):
    """Request body for simple waitlist signup"""
    email: EmailStr


class VerifyRequest(BaseModel):
    email: EmailStr
    code: str


class RefreshRequest(BaseModel):
    refresh_token: str


class GoogleSignInRequest(BaseModel):
    """Request body for Google Sign-In via Firebase"""
    id_token: str  # Firebase ID token from frontend


class PasswordLoginRequest(BaseModel):
    """Request body for email/password login"""
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user: Optional[dict] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_verification_code() -> str:
    """Generate a 6-digit verification code"""
    return ''.join(random.choices(string.digits, k=6))


async def send_verification_email(email: str, code: str) -> bool:
    """
    Send verification code to user's email
    
    For now, we just log the code (dev mode).
    In production, integrate with SendGrid, Mailgun, etc.
    """
    if SENDGRID_API_KEY:
        # TODO: Implement SendGrid email sending
        # For now, fall through to logging
        pass
    
    # Development mode - only log that a code was sent (do not expose the code itself)
    logger.info("=" * 60)
    logger.info(f"VERIFICATION CODE sent to {email}")
    logger.info("=" * 60)
    
    # In a real app, you'd send an actual email here
    # For demo purposes, the code is logged and also returned in response
    return True


# ============================================================================
# ROUTES
# ============================================================================

@router.post("/waitlist")
async def join_waitlist(request: WaitlistRequest):
    """
    Simple waitlist signup - just stores email, no verification required
    
    This is a simple lead capture for users who want early access.
    No authentication tokens are returned - just confirmation of signup.
    """
    try:
        logger.info(f"Processing waitlist signup for: {request.email}")
        
        # Add user to waitlist
        user = await add_to_waitlist(request.email)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Successfully joined the waitlist!",
                "user_id": user['id'],
                "email": user['email']
            }
        )
        
    except Exception as e:
        logger.error(f"Waitlist signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """
    Register a new user or request a new verification code
    
    Sends a 6-digit code to the user's email (or logs it in dev mode)
    """
    try:
        # Generate verification code
        code = generate_verification_code()
        
        # Create or update user in database
        user = await create_user(request.email, code)
        
        # Send verification email
        email_sent = await send_verification_email(request.email, code)
        
        if not email_sent:
            raise HTTPException(status_code=500, detail="Failed to send verification email")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Verification code sent to your email"
            }
        )
        
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify", response_model=AuthResponse)
async def verify(request: VerifyRequest):
    """
    Verify email with the 6-digit code and return access token
    """
    try:
        # Verify the code
        user = await verify_user(request.email, request.code)
        
        if not user:
            raise HTTPException(
                status_code=401, 
                detail="Invalid or expired verification code"
            )
        
        # Generate tokens
        access_token = create_access_token(
            user_id=user['id'],
            email=user['email'],
            is_waitlist=user.get('is_waitlist', True)
        )
        refresh_token = create_refresh_token(user['id'])
        
        # Return user info and tokens
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Email verified successfully",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "is_waitlist": user.get('is_waitlist', True),
                    "ask_doubt_credits": user.get('ask_doubt_credits', DEFAULT_ASK_DOUBT_CREDITS)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh", response_model=AuthResponse)
async def refresh(request: RefreshRequest):
    """
    Get a new access token using a refresh token
    """
    try:
        # Validate refresh token
        user_id = validate_refresh_token(request.refresh_token)
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
        
        # Get user data
        user = await get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate new access token
        access_token = create_access_token(
            user_id=user['id'],
            email=user['email'],
            is_waitlist=user.get('is_waitlist', True)
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Token refreshed",
                "access_token": access_token,
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "is_waitlist": user.get('is_waitlist', True),
                    "ask_doubt_credits": user.get('ask_doubt_credits', DEFAULT_ASK_DOUBT_CREDITS)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user(authorization: str = Header(...)):
    """
    Get current user info from access token
    """
    try:
        # Validate token
        user_info = validate_authorization_header(authorization)
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        # Get full user data from database
        user = await get_user_by_id(user_info['id'])
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return JSONResponse(
            status_code=200,
            content={
                "id": user['id'],
                "email": user['email'],
                "is_waitlist": user.get('is_waitlist', True),
                "ask_doubt_credits": user.get('ask_doubt_credits', DEFAULT_ASK_DOUBT_CREDITS),
                "is_verified": user.get('is_verified', False)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/credits")
async def get_credits(authorization: str = Header(...)):
    """
    Get current credit balance for authenticated user
    """
    try:
        # Validate token
        user_info = validate_authorization_header(authorization)
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        # Get credits
        credits = await get_user_credits(user_info['id'])
        
        return JSONResponse(
            status_code=200,
            content={
                "credits_remaining": credits,
                "is_waitlist": user_info.get('is_waitlist', True),
                "user_id": user_info['id'],
                "email": user_info.get('email')
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get credits error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GOOGLE SIGN-IN (Firebase Auth)
# ============================================================================

@router.post("/google")
async def google_sign_in(request: GoogleSignInRequest):
    """
    Authenticate via Firebase/Google Sign-In
    
    Flow:
    1. Frontend signs in with Google via Firebase Auth
    2. Frontend sends Firebase ID token to this endpoint
    3. We verify the token with Firebase Admin SDK
    4. We get or create the user in our Firestore database
    5. We return our own JWT tokens + user data
    
    This allows seamless Google Sign-In while keeping our own session management.
    """
    try:
        logger.info("Processing Google Sign-In request...")
        
        # Step 1: Verify Firebase ID token
        firebase_user = verify_firebase_token(request.id_token)
        
        if not firebase_user:
            logger.warning("Invalid Firebase token received")
            raise HTTPException(
                status_code=401, 
                detail="Invalid or expired Google sign-in token"
            )
        
        logger.info(f"Firebase token verified for: {firebase_user.get('email')}")
        
        # Step 2: Get or create user in our database
        user = await get_or_create_user_by_firebase(
            uid=firebase_user['uid'],
            email=firebase_user['email'],
            name=firebase_user.get('name'),
            picture=firebase_user.get('picture')
        )
        
        logger.info(f"User retrieved/created: {user.get('email')} (ID: {user.get('id')})")
        
        # Step 3: Generate our JWT tokens
        access_token = create_access_token(
            user_id=user['id'],
            email=user['email'],
            is_waitlist=user.get('is_waitlisted', True) or user.get('is_waitlist', True)
        )
        refresh_token = create_refresh_token(user['id'])
        
        # Step 4: Return response with tokens and user data
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Signed in with Google successfully",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "name": user.get('name'),
                    "picture": user.get('picture'),
                    "is_waitlist": user.get('is_waitlisted', True) or user.get('is_waitlist', True),
                    "ask_doubt_credits": user.get('ask_doubt_credits', DEFAULT_ASK_DOUBT_CREDITS),
                    "voice_tutor_usage_seconds": user.get('voice_tutor_usage_seconds', 0),
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google Sign-In error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Google sign-in failed: {str(e)}"
        )


# ============================================================================
# EMAIL/PASSWORD LOGIN
# ============================================================================

@router.post("/login")
async def password_login(request: PasswordLoginRequest):
    """
    Authenticate via email/password
    
    Used for:
    - Admin login
    - Custom users created by admin
    
    Flow:
    1. Find user by email
    2. Verify password against stored hash
    3. Return JWT tokens + user data
    """
    try:
        logger.info(f"Processing password login for: {request.email}")
        
        # Step 1: Find user by email with password auth
        user = await get_user_by_email_password(request.email)
        
        if not user:
            logger.warning(f"User not found or no password set: {request.email}")
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Step 2: Verify password
        if not verify_password(request.password, user['password_hash']):
            logger.warning(f"Invalid password for: {request.email}")
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        logger.info(f"Password verified for: {request.email}")
        
        # Step 3: Generate JWT tokens
        access_token = create_access_token(
            user_id=user['id'],
            email=user['email'],
            is_waitlist=user.get('is_waitlisted', False) or user.get('is_waitlist', False)
        )
        refresh_token = create_refresh_token(user['id'])
        
        # Step 4: Return response with tokens and user data
        # Include role and limits for admin/custom users
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Logged in successfully",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "name": user.get('name'),
                    "role": user.get('role', 'user'),
                    "is_waitlist": user.get('is_waitlisted', False) or user.get('is_waitlist', False),
                    "ask_doubt_credits": user.get('ask_doubt_credits', DEFAULT_ASK_DOUBT_CREDITS),
                    "ask_doubt_limit": user.get('ask_doubt_limit', DEFAULT_ASK_DOUBT_CREDITS),
                    "voice_tutor_limit_seconds": user.get('voice_tutor_limit_seconds', 180),
                    "voice_tutor_usage_seconds": user.get('voice_tutor_usage_seconds', 0),
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )
