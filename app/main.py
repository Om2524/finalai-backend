"""
Main FastAPI Application for Ask Doubt Backend

Features:
- Ask Doubt: Generate Manim animated solutions
- Authentication: JWT-based auth with Firestore
- Credits: Track and manage user credits
- Admin: User management and custom limits
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import CORS_ORIGINS, VIDEO_STORAGE_PATH, ADMIN_EMAIL, ADMIN_PASSWORD
from app.api.routes import router
from app.api.auth_routes import router as auth_router
from app.api.admin_routes import router as admin_router
from app.services.database import create_admin, get_user_by_email
from app.services.password_auth import hash_password

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Ask Doubt Backend",
    description="Backend service for generating Manim animation solutions with auth",
    version="2.0.0"
)

# Configure CORS - use configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=str(VIDEO_STORAGE_PATH)), name="videos")

# Include API routes
app.include_router(router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(admin_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Ask Doubt Backend",
        "version": "2.1.0",
        "status": "running",
        "api_docs": "/docs",
        "endpoints": {
            "ask_doubt": "/api/ask-doubt",
            "auth_signup": "/api/auth/signup",
            "auth_verify": "/api/auth/verify",
            "auth_login": "/api/auth/login",
            "auth_google": "/api/auth/google",
            "auth_me": "/api/auth/me",
            "auth_credits": "/api/auth/credits",
            "admin_users": "/api/admin/users",
            "admin_stats": "/api/admin/stats",
            "health": "/api/health"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("Ask Doubt Backend v2.1 Started")
    print("=" * 60)
    print("Features: Manim Solutions + JWT Auth + Firestore + Admin")
    print(f"API Docs: http://localhost:8000/docs")
    print(f"Video Storage: {VIDEO_STORAGE_PATH}")
    print("=" * 60)
    
    # Seed admin account
    await seed_admin()


async def seed_admin():
    """
    Create or update admin account on startup
    
    This ensures the admin account always exists with correct credentials.
    """
    try:
        if not ADMIN_EMAIL or not ADMIN_PASSWORD:
            logger.warning("ADMIN_EMAIL/ADMIN_PASSWORD not set; admin seeding skipped")
            return

        logger.info("Checking/creating admin account...")
        
        # Check if admin exists
        existing = await get_user_by_email(ADMIN_EMAIL)
        
        if existing and existing.get('role') == 'admin':
            logger.info(f"Admin account already exists: {ADMIN_EMAIL}")
            return
        
        # Create admin with hashed password
        password_hash = hash_password(ADMIN_PASSWORD)
        admin = await create_admin(
            email=ADMIN_EMAIL,
            password_hash=password_hash,
            name="Admin"
        )
        
        logger.info(f"Admin account created/updated: {ADMIN_EMAIL}")
        print(f"Admin account ready: {ADMIN_EMAIL}")
        
    except Exception as e:
        logger.error(f"Failed to seed admin account: {e}")
        print(f"WARNING: Failed to create admin account: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
