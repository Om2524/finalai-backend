"""
JWT Authentication Service

Handles token generation and validation for our custom auth system.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from app.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION_HOURS

logger = logging.getLogger(__name__)


def create_access_token(user_id: str, email: str, is_waitlist: bool = True) -> str:
    """
    Create a JWT access token for a user
    
    Args:
        user_id: User's unique ID
        email: User's email
        is_waitlist: Whether user is on waitlist
        
    Returns:
        Encoded JWT token string
    """
    now = datetime.utcnow()
    expires = now + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        'sub': user_id,  # Subject (user ID)
        'email': email,
        'is_waitlist': is_waitlist,
        'iat': now,      # Issued at
        'exp': expires,  # Expiration
        'type': 'access'
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.info(f"Created access token for user: {email}")
    
    return token


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None


def validate_authorization_header(authorization: str) -> Optional[Dict[str, Any]]:
    """
    Validate an Authorization header and extract user info
    
    Args:
        authorization: Authorization header value (Bearer <token>)
        
    Returns:
        User info dict with 'id', 'email', 'is_waitlist' if valid, None otherwise
    """
    if not authorization:
        return None
    
    if not authorization.startswith('Bearer '):
        logger.warning("Invalid authorization header format")
        return None
    
    token = authorization[7:]  # Remove 'Bearer ' prefix
    
    payload = decode_token(token)
    
    if not payload:
        return None
    
    if payload.get('type') != 'access':
        logger.warning("Token is not an access token")
        return None
    
    return {
        'id': payload.get('sub'),
        'email': payload.get('email'),
        'is_waitlist': payload.get('is_waitlist', True)
    }


def create_refresh_token(user_id: str) -> str:
    """
    Create a refresh token (longer lived, for getting new access tokens)
    
    Args:
        user_id: User's unique ID
        
    Returns:
        Encoded JWT refresh token
    """
    now = datetime.utcnow()
    expires = now + timedelta(days=30)  # 30 days
    
    payload = {
        'sub': user_id,
        'iat': now,
        'exp': expires,
        'type': 'refresh'
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def validate_refresh_token(token: str) -> Optional[str]:
    """
    Validate a refresh token and return the user ID
    
    Args:
        token: Refresh token string
        
    Returns:
        User ID if valid, None otherwise
    """
    payload = decode_token(token)
    
    if not payload:
        return None
    
    if payload.get('type') != 'refresh':
        logger.warning("Token is not a refresh token")
        return None
    
    return payload.get('sub')
