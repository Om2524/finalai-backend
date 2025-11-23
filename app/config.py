"""
Configuration settings for Ask Doubt Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8000))

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

# Validate required settings
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")
