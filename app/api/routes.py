"""
API Routes for Ask Doubt Backend
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from app.services.gemini_client import get_gemini_client
from app.services.manim_renderer import get_manim_renderer
from app.config import MAX_IMAGE_SIZE_BYTES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask-doubt")
async def ask_doubt(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Main endpoint for Ask Doubt feature
    
    Flow:
    1. Receive image + question
    2. Validate inputs
    3. Send to Gemini for code generation
    4. Render video with Manim
    5. Return video URL
    """
    try:
        # Validate inputs
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not question or len(question.strip()) < 3:
            raise HTTPException(status_code=400, detail="Question must be at least 3 characters")
        
        # Read image data
        image_data = await image.read()
        
        # Check file size
        if len(image_data) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image size exceeds maximum allowed ({MAX_IMAGE_SIZE_BYTES / 1024 / 1024}MB)"
            )
        
        logger.info(f"Processing doubt request: question length={len(question)}, image size={len(image_data)} bytes")
        
        # Step 1: Generate Manim code using Gemini
        gemini_client = get_gemini_client()
        logger.info("Generating Manim code with Gemini...")
        
        manim_code = await gemini_client.generate_manim_code(
            image_bytes=image_data,
            question=question,
            image_mime_type=image.content_type
        )
        
        logger.info(f"Manim code generated successfully ({len(manim_code)} characters)")
        
        # Step 2: Render video with Manim
        manim_renderer = get_manim_renderer()
        logger.info("Rendering video with Manim...")
        
        video_info = await manim_renderer.render(manim_code)
        
        logger.info(f"Video rendered successfully: {video_info['filename']}")
        
        # Return response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "video_url": video_info["video_url"],
                "filename": video_info["filename"],
                "duration": video_info["duration"],
                "generated_at": video_info["generated_at"],
                "message": "Solution generated successfully"
            }
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing doubt: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate solution: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ask-doubt-backend"}
