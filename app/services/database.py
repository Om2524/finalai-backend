"""
Firestore Database Service

Handles all database operations for users and credits.
Uses Google Cloud Firestore for serverless NoSQL storage.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter

import bcrypt

from app.config import GCP_PROJECT_ID, DEFAULT_ASK_DOUBT_CREDITS

logger = logging.getLogger(__name__)


def serialize_firestore_doc(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Firestore document data to JSON-serializable format.
    
    Converts DatetimeWithNanoseconds and other Firestore types to strings.
    
    Args:
        data: Firestore document data
        
    Returns:
        JSON-serializable dict
    """
    if data is None:
        return None
    
    result = {}
    for key, value in data.items():
        if value is None:
            result[key] = None
        elif hasattr(value, 'isoformat'):
            # datetime or DatetimeWithNanoseconds
            result[key] = value.isoformat()
        elif hasattr(value, 'timestamp'):
            # Firestore Timestamp
            result[key] = datetime.fromtimestamp(value.timestamp()).isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_firestore_doc(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_firestore_doc(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result


def _hash_verification_code(code: str) -> str:
    """Hash a verification code before storing."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(code.encode("utf-8"), salt).decode("utf-8")


def _code_matches(provided: str, stored: Optional[str]) -> bool:
    """
    Check a provided code against stored value (supports legacy plain-text codes).
    """
    if not stored:
        return False
    # Support legacy plain-text codes
    if not stored.startswith("$2"):
        return stored == provided
    try:
        return bcrypt.checkpw(provided.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        return False

# Firestore client (singleton)
_db: Optional[firestore.Client] = None


def get_db() -> firestore.Client:
    """Get or create Firestore client"""
    global _db
    if _db is None:
        try:
            # In Cloud Run, this automatically uses the service account
            _db = firestore.Client(project=GCP_PROJECT_ID)
            logger.info(f"Firestore client initialized for project: {GCP_PROJECT_ID}")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            raise
    return _db


# ============================================================================
# USER OPERATIONS
# ============================================================================

async def create_user(email: str, verification_code: str) -> Dict[str, Any]:
    """
    Create a new user or update existing unverified user
    
    Args:
        email: User's email address
        verification_code: 6-digit verification code
        
    Returns:
        User document data
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Check if user already exists
    existing = users_ref.where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    now = datetime.utcnow()
    
    hashed_code = _hash_verification_code(verification_code)
    if existing:
        # User exists - update verification code
        user_doc = existing[0]
        user_data = user_doc.to_dict()
        
        if user_data.get('is_verified', False):
            # Already verified - just update the code for re-login
            user_doc.reference.update({
                'verification_code': hashed_code,
                'code_created_at': now,
                'updated_at': now
            })
            user_data.update({
                'verification_code': hashed_code,
                'code_created_at': now
            })
        else:
            # Not verified yet - update code
            user_doc.reference.update({
                'verification_code': hashed_code,
                'code_created_at': now,
                'updated_at': now
            })
            user_data.update({
                'verification_code': hashed_code,
                'code_created_at': now
            })
        
        user_data['id'] = user_doc.id
        return user_data
    else:
        # Create new user
        user_data = {
            'email': email,
            'verification_code': hashed_code,
            'code_created_at': now,
            'is_verified': False,
            'is_waitlist': True,
            'ask_doubt_credits': DEFAULT_ASK_DOUBT_CREDITS,
            'created_at': now,
            'updated_at': now
        }
        
        doc_ref = users_ref.document()
        doc_ref.set(user_data)
        user_data['id'] = doc_ref.id
        
        logger.info(f"Created new user: {email}")
        return user_data


async def verify_user(email: str, code: str) -> Optional[Dict[str, Any]]:
    """
    Verify a user's email with the provided code
    
    Args:
        email: User's email
        code: 6-digit verification code
        
    Returns:
        User data if verification successful, None otherwise
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Find user by email
    docs = users_ref.where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    if not docs:
        logger.warning(f"User not found: {email}")
        return None
    
    user_doc = docs[0]
    user_data = user_doc.to_dict()
    
    # Check verification code
    stored_code = user_data.get('verification_code', '')
    
    if not _code_matches(code, stored_code):
        logger.warning(f"Invalid verification code for: {email}")
        return None
    
    # Check if code is expired (10 minutes)
    code_created = user_data.get('code_created_at')
    if code_created:
        try:
            # Handle both datetime and Firestore DatetimeWithNanoseconds
            if hasattr(code_created, 'timestamp'):
                # Firestore timestamp or datetime with timezone
                code_timestamp = code_created.timestamp()
            else:
                # Naive datetime
                code_timestamp = code_created.replace(tzinfo=None).timestamp() if hasattr(code_created, 'replace') else 0
            
            current_timestamp = datetime.utcnow().timestamp()
            age_seconds = current_timestamp - code_timestamp
            
            if age_seconds > 600:  # 10 minutes
                logger.warning(f"Expired verification code for: {email}")
                return None
        except Exception as e:
            logger.warning(f"Could not check code expiration: {e}")
    
    # Mark as verified
    user_doc.reference.update({
        'is_verified': True,
        'verification_code': None,  # Clear the code
        'updated_at': datetime.utcnow()
    })
    
    user_data['id'] = user_doc.id
    user_data['is_verified'] = True
    
    logger.info(f"User verified: {email}")
    return user_data


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by document ID"""
    db = get_db()
    doc = db.collection('users').document(user_id).get()
    
    if not doc.exists:
        return None
    
    user_data = doc.to_dict()
    user_data['id'] = doc.id
    return serialize_firestore_doc(user_data)


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    db = get_db()
    docs = db.collection('users').where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    if not docs:
        return None
    
    user_doc = docs[0]
    user_data = user_doc.to_dict()
    user_data['id'] = user_doc.id
    return serialize_firestore_doc(user_data)


# ============================================================================
# CREDIT OPERATIONS
# ============================================================================

async def get_user_credits(user_id: str) -> int:
    """Get current credit count for a user"""
    user = await get_user_by_id(user_id)
    if not user:
        return 0
    return user.get('ask_doubt_credits', 0)


async def use_credit(user_id: str) -> tuple[bool, int]:
    """
    Decrement a user's credits by 1 (atomic operation)
    
    Args:
        user_id: User's document ID
        
    Returns:
        Tuple of (success, remaining_credits)
    """
    db = get_db()
    user_ref = db.collection('users').document(user_id)
    
    @firestore.transactional
    def update_credits(transaction, doc_ref):
        snapshot = doc_ref.get(transaction=transaction)
        if not snapshot.exists:
            return False, 0
        
        current_credits = snapshot.get('ask_doubt_credits') or 0
        
        if current_credits <= 0:
            return False, 0
        
        new_credits = current_credits - 1
        transaction.update(doc_ref, {
            'ask_doubt_credits': new_credits,
            'updated_at': datetime.utcnow()
        })
        
        return True, new_credits
    
    try:
        transaction = db.transaction()
        success, remaining = update_credits(transaction, user_ref)
        
        if success:
            logger.info(f"Credit used for user {user_id}. Remaining: {remaining}")
        
        return success, remaining
    except Exception as e:
        logger.error(f"Error using credit: {e}")
        return False, 0


async def add_credits(user_id: str, amount: int) -> int:
    """Add credits to a user's account"""
    db = get_db()
    user_ref = db.collection('users').document(user_id)
    
    doc = user_ref.get()
    if not doc.exists:
        return 0
    
    current = doc.to_dict().get('ask_doubt_credits', 0)
    new_credits = current + amount
    
    user_ref.update({
        'ask_doubt_credits': new_credits,
        'updated_at': datetime.utcnow()
    })
    
    logger.info(f"Added {amount} credits to user {user_id}. New total: {new_credits}")
    return new_credits


# ============================================================================
# FIREBASE AUTH USER OPERATIONS
# ============================================================================

async def get_or_create_user_by_firebase(
    uid: str, 
    email: str, 
    name: Optional[str] = None,
    picture: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get existing user by Firebase UID or create a new one
    
    This is used when a user signs in with Google via Firebase Auth.
    
    Args:
        uid: Firebase user ID (unique identifier from Google Sign-In)
        email: User's email address
        name: User's display name (from Google profile)
        picture: URL to user's profile picture
        
    Returns:
        User document data with 'id' field added
        
    Schema:
    {
        "uid": "firebase_uid",
        "email": "user@example.com", 
        "name": "User Name",
        "picture": "https://...",
        "voice_tutor_usage_seconds": 0,
        "is_waitlisted": true,
        "ask_doubt_credits": 3,
        "ask_doubt_used": 0,
        "created_at": timestamp,
        "updated_at": timestamp
    }
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Check if user already exists by Firebase UID
    existing_docs = users_ref.where(filter=FieldFilter('uid', '==', uid)).limit(1).get()
    
    now = datetime.utcnow()
    
    if existing_docs:
        # User exists - update last login and return
        user_doc = existing_docs[0]
        user_data = user_doc.to_dict()
        
        # Update name/picture if changed
        updates = {'updated_at': now}
        if name and name != user_data.get('name'):
            updates['name'] = name
        if picture and picture != user_data.get('picture'):
            updates['picture'] = picture
        
        user_doc.reference.update(updates)
        
        user_data['id'] = user_doc.id
        user_data.update(updates)
        
        logger.info(f"Existing Firebase user logged in: {email}")
        return user_data
    
    # Also check if user exists by email (migrating from email/OTP auth)
    email_docs = users_ref.where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    if email_docs:
        # User exists with email but not Firebase UID - link accounts
        user_doc = email_docs[0]
        user_data = user_doc.to_dict()
        
        # Add Firebase UID to existing user
        updates = {
            'uid': uid,
            'name': name or user_data.get('name'),
            'picture': picture,
            'updated_at': now
        }
        
        user_doc.reference.update(updates)
        user_data.update(updates)
        user_data['id'] = user_doc.id
        
        logger.info(f"Linked Firebase UID to existing user: {email}")
        return user_data
    
    # Create new user
    user_data = {
        'uid': uid,
        'email': email,
        'name': name,
        'picture': picture,
        'voice_tutor_usage_seconds': 0,
        'is_waitlisted': True,
        'is_waitlist': True,  # Keep both for compatibility
        'ask_doubt_credits': DEFAULT_ASK_DOUBT_CREDITS,
        'ask_doubt_used': 0,
        'is_verified': True,  # Firebase auth = verified
        'created_at': now,
        'updated_at': now,
    }
    
    doc_ref = users_ref.document()
    doc_ref.set(user_data)
    user_data['id'] = doc_ref.id
    
    logger.info(f"Created new user via Firebase: {email}")
    return user_data


async def get_user_by_firebase_uid(uid: str) -> Optional[Dict[str, Any]]:
    """Get user by Firebase UID"""
    db = get_db()
    docs = db.collection('users').where(filter=FieldFilter('uid', '==', uid)).limit(1).get()
    
    if not docs:
        return None
    
    user_doc = docs[0]
    user_data = user_doc.to_dict()
    user_data['id'] = user_doc.id
    return user_data


async def update_voice_tutor_usage(user_id: str, seconds: int) -> bool:
    """
    Update user's voice tutor usage time
    
    Args:
        user_id: User's document ID
        seconds: Number of seconds to add to usage
        
    Returns:
        True if updated successfully
    """
    db = get_db()
    user_ref = db.collection('users').document(user_id)
    
    try:
        doc = user_ref.get()
        if not doc.exists:
            return False
        
        current_usage = doc.to_dict().get('voice_tutor_usage_seconds', 0)
        new_usage = current_usage + seconds
        
        user_ref.update({
            'voice_tutor_usage_seconds': new_usage,
            'updated_at': datetime.utcnow()
        })
        
        logger.info(f"Updated voice tutor usage for user {user_id}: {new_usage}s total")
        return True
        
    except Exception as e:
        logger.error(f"Error updating voice tutor usage: {e}")
        return False


# ============================================================================
# ADMIN & CUSTOM USER OPERATIONS
# ============================================================================

async def create_admin(
    email: str,
    password_hash: str,
    name: str = "Admin"
) -> Dict[str, Any]:
    """
    Create an admin user
    
    Args:
        email: Admin email
        password_hash: Bcrypt hashed password
        name: Admin name
        
    Returns:
        Admin user document data
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Check if admin already exists
    existing = users_ref.where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    now = datetime.utcnow()
    
    if existing:
        # Admin exists - update password if needed
        admin_doc = existing[0]
        admin_data = admin_doc.to_dict()
        
        admin_doc.reference.update({
            'password_hash': password_hash,
            'role': 'admin',
            'voice_tutor_limit_seconds': 0,  # 0 = unlimited
            'ask_doubt_limit': 0,  # 0 = unlimited
            'updated_at': now
        })
        
        admin_data['id'] = admin_doc.id
        admin_data['role'] = 'admin'
        logger.info(f"Updated existing admin: {email}")
        return admin_data
    
    # Create new admin
    admin_data = {
        'email': email,
        'password_hash': password_hash,
        'name': name,
        'role': 'admin',
        'voice_tutor_limit_seconds': 0,  # 0 = unlimited
        'voice_tutor_usage_seconds': 0,
        'ask_doubt_limit': 0,  # 0 = unlimited
        'ask_doubt_used': 0,
        'ask_doubt_credits': 0,  # 0 = unlimited for admin
        'is_verified': True,
        'is_waitlisted': False,
        'is_waitlist': False,
        'auth_method': 'password',
        'created_at': now,
        'updated_at': now
    }
    
    doc_ref = users_ref.document()
    doc_ref.set(admin_data)
    admin_data['id'] = doc_ref.id
    
    logger.info(f"Created admin user: {email}")
    return admin_data


async def get_user_by_email_password(email: str) -> Optional[Dict[str, Any]]:
    """
    Get user by email for password-based authentication
    Returns user with password_hash for verification
    
    Args:
        email: User's email
        
    Returns:
        User data including password_hash, or None if not found
    """
    db = get_db()
    docs = db.collection('users').where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    if not docs:
        return None
    
    user_doc = docs[0]
    user_data = user_doc.to_dict()
    user_data['id'] = user_doc.id
    
    # Only return if user has password_hash (password-based auth enabled)
    if not user_data.get('password_hash'):
        return None
    
    return user_data


async def create_custom_user(
    email: str,
    password_hash: str,
    name: Optional[str] = None,
    voice_tutor_limit_seconds: int = 180,  # Default 3 minutes
    ask_doubt_limit: int = 3,  # Default 3 credits
    created_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a custom user with specified limits (created by admin)
    
    Args:
        email: User's email
        password_hash: Bcrypt hashed password
        name: User's name
        voice_tutor_limit_seconds: Voice tutor time limit in seconds (0 = unlimited)
        ask_doubt_limit: AskDoubt credit limit (0 = unlimited)
        created_by: Admin user ID who created this user
        
    Returns:
        User document data
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Check if user already exists
    existing = users_ref.where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    now = datetime.utcnow()
    
    if existing:
        # Update existing user with password auth
        user_doc = existing[0]
        user_data = user_doc.to_dict()
        
        updates = {
            'password_hash': password_hash,
            'voice_tutor_limit_seconds': voice_tutor_limit_seconds,
            'ask_doubt_limit': ask_doubt_limit,
            'ask_doubt_credits': ask_doubt_limit,  # Set credits to limit
            'is_waitlisted': False,
            'is_waitlist': False,
            'auth_method': 'both' if user_data.get('uid') else 'password',
            'updated_at': now
        }
        
        if name:
            updates['name'] = name
        if created_by:
            updates['created_by'] = created_by
        
        user_doc.reference.update(updates)
        user_data.update(updates)
        user_data['id'] = user_doc.id
        
        logger.info(f"Updated existing user with custom limits: {email}")
        return serialize_firestore_doc(user_data)
    
    # Create new custom user
    user_data = {
        'email': email,
        'password_hash': password_hash,
        'name': name,
        'role': 'user',
        'voice_tutor_limit_seconds': voice_tutor_limit_seconds,
        'voice_tutor_usage_seconds': 0,
        'ask_doubt_limit': ask_doubt_limit,
        'ask_doubt_credits': ask_doubt_limit,
        'ask_doubt_used': 0,
        'is_verified': True,
        'is_waitlisted': False,
        'is_waitlist': False,
        'auth_method': 'password',
        'created_by': created_by,
        'created_at': now,
        'updated_at': now
    }
    
    doc_ref = users_ref.document()
    doc_ref.set(user_data)
    user_data['id'] = doc_ref.id
    
    logger.info(f"Created custom user: {email} with {voice_tutor_limit_seconds}s voice limit, {ask_doubt_limit} credits")
    return serialize_firestore_doc(user_data)


async def update_user_limits(
    user_id: str,
    voice_tutor_limit_seconds: Optional[int] = None,
    ask_doubt_limit: Optional[int] = None,
    name: Optional[str] = None,
    reset_usage: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Update a user's limits (admin function)
    
    Args:
        user_id: User document ID
        voice_tutor_limit_seconds: New voice tutor limit (0 = unlimited)
        ask_doubt_limit: New AskDoubt limit (0 = unlimited)
        name: Update user name
        reset_usage: If True, reset usage counters to 0
        
    Returns:
        Updated user data, or None if user not found
    """
    db = get_db()
    user_ref = db.collection('users').document(user_id)
    
    doc = user_ref.get()
    if not doc.exists:
        return None
    
    updates = {'updated_at': datetime.utcnow()}
    
    if voice_tutor_limit_seconds is not None:
        updates['voice_tutor_limit_seconds'] = voice_tutor_limit_seconds
    
    if ask_doubt_limit is not None:
        updates['ask_doubt_limit'] = ask_doubt_limit
        updates['ask_doubt_credits'] = ask_doubt_limit  # Reset credits to new limit
    
    if name is not None:
        updates['name'] = name
    
    if reset_usage:
        updates['voice_tutor_usage_seconds'] = 0
        updates['ask_doubt_used'] = 0
        if ask_doubt_limit is not None:
            updates['ask_doubt_credits'] = ask_doubt_limit
        else:
            # Reset credits to current limit
            current_limit = doc.to_dict().get('ask_doubt_limit', DEFAULT_ASK_DOUBT_CREDITS)
            updates['ask_doubt_credits'] = current_limit
    
    user_ref.update(updates)
    
    user_data = doc.to_dict()
    user_data.update(updates)
    user_data['id'] = doc.id
    
    logger.info(f"Updated user limits for {user_id}: {updates}")
    return serialize_firestore_doc(user_data)


async def delete_user(user_id: str) -> bool:
    """
    Delete a user (admin function)
    
    Args:
        user_id: User document ID
        
    Returns:
        True if deleted successfully
    """
    db = get_db()
    user_ref = db.collection('users').document(user_id)
    
    doc = user_ref.get()
    if not doc.exists:
        return False
    
    # Don't allow deleting admins
    user_data = doc.to_dict()
    if user_data.get('role') == 'admin':
        logger.warning(f"Cannot delete admin user: {user_id}")
        return False
    
    user_ref.delete()
    logger.info(f"Deleted user: {user_id}")
    return True


async def list_all_users(
    limit: int = 100,
    offset: int = 0,
    role_filter: Optional[str] = None
) -> tuple[list[Dict[str, Any]], int]:
    """
    List all users (admin function)
    
    Args:
        limit: Maximum number of users to return
        offset: Number of users to skip
        role_filter: Filter by role ('user', 'admin', or None for all)
        
    Returns:
        Tuple of (list of users, total count)
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Build query
    query = users_ref
    
    if role_filter:
        query = query.where(filter=FieldFilter('role', '==', role_filter))
    
    # Get total count (approximation for now)
    all_docs = list(query.stream())
    total_count = len(all_docs)
    
    # Apply pagination
    users = []
    for i, doc in enumerate(all_docs):
        if i < offset:
            continue
        if len(users) >= limit:
            break
        
        user_data = doc.to_dict()
        user_data['id'] = doc.id
        # Remove sensitive data
        user_data.pop('password_hash', None)
        user_data.pop('verification_code', None)
        users.append(serialize_firestore_doc(user_data))
    
    return users, total_count


async def get_admin_stats() -> Dict[str, Any]:
    """
    Get statistics for admin dashboard
    
    Returns:
        Dict with various stats
    """
    db = get_db()
    users_ref = db.collection('users')
    
    all_users = list(users_ref.stream())
    
    total_users = 0
    admin_count = 0
    waitlist_count = 0
    custom_users = 0
    total_voice_usage = 0
    total_doubt_usage = 0
    
    for doc in all_users:
        data = doc.to_dict()
        total_users += 1
        
        if data.get('role') == 'admin':
            admin_count += 1
        elif data.get('created_by'):
            custom_users += 1
        elif data.get('is_waitlisted') or data.get('is_waitlist'):
            waitlist_count += 1
        
        total_voice_usage += data.get('voice_tutor_usage_seconds', 0)
        total_doubt_usage += data.get('ask_doubt_used', 0)
    
    return {
        'total_users': total_users,
        'admin_count': admin_count,
        'waitlist_count': waitlist_count,
        'custom_users': custom_users,
        'regular_users': total_users - admin_count - custom_users - waitlist_count,
        'total_voice_usage_seconds': total_voice_usage,
        'total_voice_usage_hours': round(total_voice_usage / 3600, 2),
        'total_doubt_requests': total_doubt_usage
    }


async def add_to_waitlist(email: str) -> Dict[str, Any]:
    """
    Add a user to the waitlist (simple email signup, no verification needed)
    
    Args:
        email: User's email address
        
    Returns:
        User document data with 'id' field
    """
    db = get_db()
    users_ref = db.collection('users')
    
    # Check if user already exists
    existing = users_ref.where(filter=FieldFilter('email', '==', email)).limit(1).get()
    
    now = datetime.utcnow()
    
    if existing:
        # User already exists - just return their data
        user_doc = existing[0]
        user_data = user_doc.to_dict()
        user_data['id'] = user_doc.id
        
        # If not already on waitlist, update the status
        if not user_data.get('is_waitlist') and not user_data.get('is_waitlisted'):
            user_doc.reference.update({
                'is_waitlist': True,
                'is_waitlisted': True,
                'waitlist_joined_at': now,
                'updated_at': now
            })
            user_data['is_waitlist'] = True
            user_data['is_waitlisted'] = True
            user_data['waitlist_joined_at'] = now
            logger.info(f"Added existing user to waitlist: {email}")
        else:
            logger.info(f"User already on waitlist: {email}")
        
        return serialize_firestore_doc(user_data)
    
    # Create new waitlist user
    user_data = {
        'email': email,
        'is_waitlist': True,
        'is_waitlisted': True,
        'is_verified': False,
        'ask_doubt_credits': DEFAULT_ASK_DOUBT_CREDITS,
        'voice_tutor_usage_seconds': 0,
        'waitlist_joined_at': now,
        'created_at': now,
        'updated_at': now
    }
    
    doc_ref = users_ref.document()
    doc_ref.set(user_data)
    user_data['id'] = doc_ref.id
    
    logger.info(f"Added new user to waitlist: {email}")
    return serialize_firestore_doc(user_data)


async def set_user_password(user_id: str, password_hash: str) -> bool:
    """
    Set or update a user's password
    
    Args:
        user_id: User document ID
        password_hash: Bcrypt hashed password
        
    Returns:
        True if updated successfully
    """
    db = get_db()
    user_ref = db.collection('users').document(user_id)
    
    doc = user_ref.get()
    if not doc.exists:
        return False
    
    user_data = doc.to_dict()
    auth_method = user_data.get('auth_method', 'password')
    
    # If user already has Firebase auth, mark as 'both'
    if user_data.get('uid'):
        auth_method = 'both'
    else:
        auth_method = 'password'
    
    user_ref.update({
        'password_hash': password_hash,
        'auth_method': auth_method,
        'updated_at': datetime.utcnow()
    })
    
    logger.info(f"Set password for user: {user_id}")
    return True
