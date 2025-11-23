"""
Main FastAPI Application for Ask Doubt Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import CORS_ORIGINS, VIDEO_STORAGE_PATH
from app.api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Ask Doubt Backend",
    description="Backend service for generating Manim animation solutions",
    version="1.0.0"
)

# Configure CORS - Allow all origins for Lovable deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Lovable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=str(VIDEO_STORAGE_PATH)), name="videos")

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Ask Doubt Backend",
        "status": "running",
        "api_docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("Ask Doubt Backend Started")
    print("=" * 60)
    print(f"API Docs: http://localhost:8000/docs")
    print(f"Video Storage: {VIDEO_STORAGE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
