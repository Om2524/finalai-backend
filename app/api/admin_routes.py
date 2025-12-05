"""
Admin API Routes

Handles admin operations like user management, creating custom users,
and viewing statistics. All routes require admin authentication.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from app.services.database import (
    get_user_by_id,
    create_custom_user,
    update_user_limits,
    delete_user,
    list_all_users,
    get_admin_stats
)
from app.services.password_auth import hash_password, is_strong_password
from app.services.jwt_auth import validate_authorization_header

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateUserRequest(BaseModel):
    """Request body for creating a new user"""
    email: EmailStr
    password: str
    name: Optional[str] = None
    voice_tutor_limit_seconds: int = 180  # Default 3 minutes
    ask_doubt_limit: int = 3  # Default 3 credits


class UpdateUserRequest(BaseModel):
    """Request body for updating a user"""
    voice_tutor_limit_seconds: Optional[int] = None
    ask_doubt_limit: Optional[int] = None
    name: Optional[str] = None
    reset_usage: bool = False


class SetPasswordRequest(BaseModel):
    """Request body for setting user password"""
    password: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def require_admin(authorization: str = Header(...)) -> dict:
    """
    Dependency that validates admin access
    
    Args:
        authorization: Bearer token from Authorization header
        
    Returns:
        Admin user info
        
    Raises:
        HTTPException 401 if not authenticated
        HTTPException 403 if not admin
    """
    user_info = validate_authorization_header(authorization)
    
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Get full user data to check role
    user = await get_user_by_id(user_info['id'])
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return user


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/stats")
async def get_stats(admin: dict = Depends(require_admin)):
    """
    Get dashboard statistics (admin only)
    """
    try:
        stats = await get_admin_stats()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats
            }
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_users(
    limit: int = 100,
    offset: int = 0,
    role: Optional[str] = None,
    admin: dict = Depends(require_admin)
):
    """
    List all users with pagination (admin only)
    
    Query params:
    - limit: Max users to return (default 100)
    - offset: Number to skip (default 0)
    - role: Filter by role ('user', 'admin', or None for all)
    """
    try:
        users, total = await list_all_users(limit=limit, offset=offset, role_filter=role)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "users": users,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        )
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}")
async def get_user(user_id: str, admin: dict = Depends(require_admin)):
    """
    Get a specific user's details (admin only)
    """
    try:
        user = await get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Remove sensitive data
        user.pop('password_hash', None)
        user.pop('verification_code', None)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user": user
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users")
async def create_user(request: CreateUserRequest, admin: dict = Depends(require_admin)):
    """
    Create a new user with custom limits (admin only)
    
    Body:
    - email: User's email
    - password: Initial password
    - name: User's name (optional)
    - voice_tutor_limit_seconds: Voice tutor time limit (0 = unlimited)
    - ask_doubt_limit: AskDoubt credit limit (0 = unlimited)
    """
    try:
        # Validate password
        is_valid, error_msg = is_strong_password(request.password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Hash password
        password_hash = hash_password(request.password)
        
        # Create user
        user = await create_custom_user(
            email=request.email,
            password_hash=password_hash,
            name=request.name,
            voice_tutor_limit_seconds=request.voice_tutor_limit_seconds,
            ask_doubt_limit=request.ask_doubt_limit,
            created_by=admin['id']
        )
        
        # Remove sensitive data
        user.pop('password_hash', None)
        
        logger.info(f"Admin {admin['email']} created user: {request.email}")
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": "User created successfully",
                "user": user
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    admin: dict = Depends(require_admin)
):
    """
    Update a user's limits (admin only)
    
    Body:
    - voice_tutor_limit_seconds: New voice limit (0 = unlimited)
    - ask_doubt_limit: New credit limit (0 = unlimited)
    - name: Update name
    - reset_usage: If true, reset usage counters to 0
    """
    try:
        user = await update_user_limits(
            user_id=user_id,
            voice_tutor_limit_seconds=request.voice_tutor_limit_seconds,
            ask_doubt_limit=request.ask_doubt_limit,
            name=request.name,
            reset_usage=request.reset_usage
        )
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Remove sensitive data
        user.pop('password_hash', None)
        user.pop('verification_code', None)
        
        logger.info(f"Admin {admin['email']} updated user: {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "User updated successfully",
                "user": user
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}")
async def remove_user(user_id: str, admin: dict = Depends(require_admin)):
    """
    Delete a user (admin only)
    
    Note: Admins cannot be deleted via this endpoint
    """
    try:
        # Check if trying to delete self
        if user_id == admin['id']:
            raise HTTPException(status_code=400, detail="Cannot delete yourself")
        
        success = await delete_user(user_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="User not found or is an admin (admins cannot be deleted)"
            )
        
        logger.info(f"Admin {admin['email']} deleted user: {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "User deleted successfully"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    request: SetPasswordRequest,
    admin: dict = Depends(require_admin)
):
    """
    Reset a user's password (admin only)
    """
    try:
        from app.services.database import set_user_password
        
        # Validate password
        is_valid, error_msg = is_strong_password(request.password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Hash password
        password_hash = hash_password(request.password)
        
        success = await set_user_password(user_id, password_hash)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"Admin {admin['email']} reset password for user: {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Password reset successfully"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        raise HTTPException(status_code=500, detail=str(e))
