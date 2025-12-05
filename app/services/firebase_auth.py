"""
Firebase Auth Token Verification

Verifies Firebase ID tokens from Google Sign-In on the frontend.
Uses Firebase Admin SDK for secure server-side token validation.
"""
import logging
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import auth, credentials

from app.config import GCP_PROJECT_ID

logger = logging.getLogger(__name__)

# Firebase Admin SDK initialization state
_firebase_initialized = False


def init_firebase():
    """
    Initialize Firebase Admin SDK
    
    In Cloud Run, this uses Application Default Credentials automatically.
    The service account must have Firebase Admin SDK permissions.
    """
    global _firebase_initialized
    
    if _firebase_initialized:
        return
    
    try:
        # Check if already initialized
        try:
            firebase_admin.get_app()
            _firebase_initialized = True
            return
        except ValueError:
            # Not initialized yet, proceed
            pass
        
        # Initialize with Application Default Credentials
        # In Cloud Run, this uses the service account automatically
        firebase_admin.initialize_app(options={
            'projectId': GCP_PROJECT_ID
        })
        
        _firebase_initialized = True
        logger.info(f"Firebase Admin SDK initialized for project: {GCP_PROJECT_ID}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        raise


def verify_firebase_token(id_token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Firebase ID token and return the decoded claims
    
    Args:
        id_token: The Firebase ID token from the frontend (from Google Sign-In)
        
    Returns:
        Dict with user info (uid, email, name, picture, etc.) or None if invalid
        
    The returned dict contains:
        - uid: Firebase user ID (unique identifier)
        - email: User's email address
        - name: User's display name (from Google)
        - picture: URL to user's profile picture
        - email_verified: Whether email is verified
    """
    # Ensure Firebase is initialized
    init_firebase()
    
    try:
        # Verify the token
        decoded_token = auth.verify_id_token(id_token)
        
        # Extract user information
        user_info = {
            'uid': decoded_token['uid'],
            'email': decoded_token.get('email'),
            'name': decoded_token.get('name'),
            'picture': decoded_token.get('picture'),
            'email_verified': decoded_token.get('email_verified', False),
        }
        
        logger.info(f"Firebase token verified for user: {user_info.get('email')}")
        return user_info
        
    except auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid Firebase ID token: {e}")
        return None
        
    except auth.ExpiredIdTokenError as e:
        logger.warning(f"Expired Firebase ID token: {e}")
        return None
        
    except auth.RevokedIdTokenError as e:
        logger.warning(f"Revoked Firebase ID token: {e}")
        return None
        
    except auth.CertificateFetchError as e:
        logger.error(f"Failed to fetch Firebase certificates: {e}")
        return None
        
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        return None


def get_firebase_user(uid: str) -> Optional[Dict[str, Any]]:
    """
    Get a Firebase user by UID
    
    Args:
        uid: Firebase user ID
        
    Returns:
        Dict with user info or None if not found
    """
    init_firebase()
    
    try:
        user = auth.get_user(uid)
        return {
            'uid': user.uid,
            'email': user.email,
            'name': user.display_name,
            'picture': user.photo_url,
            'email_verified': user.email_verified,
            'disabled': user.disabled,
        }
    except auth.UserNotFoundError:
        logger.warning(f"Firebase user not found: {uid}")
        return None
    except Exception as e:
        logger.error(f"Failed to get Firebase user: {e}")
        return None
