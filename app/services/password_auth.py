"""
Password Authentication Service

Handles password hashing and verification using bcrypt.
Used for admin and custom user credentials.
"""
import bcrypt
import logging

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    # Generate salt and hash
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        password: Plain text password to verify
        hashed: Stored hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Check if password meets minimum requirements
    
    Requirements:
    - At least 8 characters
    - Contains lowercase, uppercase, and a number
    
    Args:
        password: Password to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not any(c.islower() for c in password):
        return False, "Password must include a lowercase letter"
    
    if not any(c.isupper() for c in password):
        return False, "Password must include an uppercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must include a number"
    
    return True, ""
