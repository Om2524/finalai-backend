"""
API Routes for Ask Doubt Backend

Includes:
- ask-doubt: Main endpoint with optional auth + credit checking
- health: Health check
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from app.services.gemini_client import get_gemini_client
from app.services.manim_renderer import get_manim_renderer
from app.services.jwt_auth import validate_authorization_header
from app.services.database import get_user_credits, use_credit
from app.config import MAX_IMAGE_SIZE_BYTES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask-doubt")
async def ask_doubt(
    image: UploadFile = File(...),
    question: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    """
    Main endpoint for Ask Doubt feature
    
    Authentication:
    - If Authorization header is provided, validates JWT and checks credits
    - If no auth, allows request (anonymous usage tracked on frontend)
    
    Flow:
    1. Validate auth (if provided) and check credits
    2. Receive image + question
    3. Validate inputs
    4. Send to Gemini for code generation
    5. Render video with Manim
    6. Decrement credit (if authenticated)
    7. Return video URL + remaining credits
    """
    user = None
    credits_remaining = None
    
    # Check authentication if header provided
    if authorization:
        user = validate_authorization_header(authorization)
        
        if user:
            logger.info(f"Authenticated request from user: {user.get('email')}")
            
            # Check credits
            credits = await get_user_credits(user['id'])
            
            if credits <= 0:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "no_credits",
                        "message": "No credits remaining. Join the waitlist for more!",
                        "credits_remaining": 0
                    }
                )
            
            logger.info(f"User has {credits} credits remaining")
        else:
            logger.warning("Invalid authorization token provided")
    else:
        logger.info("Anonymous request (no auth header)")
    
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
        
        # Step 3: Decrement credit if authenticated
        if user:
            success, credits_remaining = await use_credit(user['id'])
            if success:
                logger.info(f"Credit used. Remaining: {credits_remaining}")
            else:
                logger.warning("Failed to decrement credit (but video was generated)")
        
        # Return response
        response_content = {
            "status": "success",
            "video_url": video_info["video_url"],
            "filename": video_info["filename"],
            "duration": video_info["duration"],
            "generated_at": video_info["generated_at"],
            "message": "Solution generated successfully"
        }
        
        # Add credits info if authenticated
        if user and credits_remaining is not None:
            response_content["credits_remaining"] = credits_remaining
        
        return JSONResponse(status_code=200, content=response_content)
        
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
    return {
        "status": "healthy", 
        "service": "ask-doubt-backend",
        "auth_type": "jwt-firestore"
    }
