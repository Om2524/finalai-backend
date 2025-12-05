"""
Configuration settings for Ask Doubt Backend
"""
import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Admin credentials (must be provided via environment in production)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8000))

# JWT Configuration (for our own auth system)
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))  # Auto-generate if not set
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", 24 * 7))  # 1 week default

# Google Cloud Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gen-lang-client-0839252651")

# Email Configuration (using a simple approach - can be upgraded to SendGrid/Mailgun later)
# For now, we'll use verification codes that are shown in logs (dev mode)
# In production, integrate with SendGrid, Mailgun, or Gmail API
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@beorigin.app")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# Storage Configuration
BASE_DIR = Path(__file__).parent.parent
VIDEO_STORAGE_PATH = BASE_DIR / os.getenv("VIDEO_STORAGE_PATH", "videos")
TEMP_CODE_PATH = BASE_DIR / os.getenv("TEMP_CODE_PATH", "temp")

# Ensure directories exist
VIDEO_STORAGE_PATH.mkdir(exist_ok=True)
TEMP_CODE_PATH.mkdir(exist_ok=True)

# Limits
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 10))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Manim Configuration - Optimized for Starter Tier
MANIM_QUALITY = os.getenv("MANIM_QUALITY", "qh")  # qh = high quality (1080p60fps) for Starter tier
MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", 180))  # 3 minutes - Starter CPU is 2x faster

# User Configuration
DEFAULT_ASK_DOUBT_CREDITS = int(os.getenv("DEFAULT_ASK_DOUBT_CREDITS", 3))

# Validate required settings
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")
