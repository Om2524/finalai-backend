"""
Manim Renderer Service - Executes generated code and produces video
"""
import subprocess
import uuid
import asyncio
import shutil
import re
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from app.config import TEMP_CODE_PATH, VIDEO_STORAGE_PATH, MANIM_QUALITY, MANIM_TIMEOUT

# Setup logging
logger = logging.getLogger(__name__)


class ManimRenderer:
    """Service for rendering Manim animations"""
    
    def __init__(self):
        self.temp_dir = TEMP_CODE_PATH
        self.output_dir = VIDEO_STORAGE_PATH
        
        # Ensure directories exist
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def _extract_scene_class_name(self, code: str) -> str:
        """Extract the Scene class name from generated code"""
        
        # Strategy 1: Look for class that inherits from Scene
        match = re.search(r'class\s+(\w+)\s*\([^)]*Scene[^)]*\):', code)
        if match:
            class_name = match.group(1)
            logger.info(f"Found Scene class: {class_name}")
            return class_name
        
        # Strategy 2: Look for PhysicsSolution specifically
        if "class PhysicsSolution" in code:
            logger.info("Found PhysicsSolution class")
            return "PhysicsSolution"
        
        # Strategy 3: Look for SolutionScene
        if "class SolutionScene" in code:
            logger.info("Found SolutionScene class")
            return "SolutionScene"
        
        # Strategy 4: Find any class with Scene in the name
        match = re.search(r'class\s+(\w*Scene\w*)', code)
        if match:
            class_name = match.group(1)
            logger.info(f"Found Scene class by name pattern: {class_name}")
            return class_name
        
        # Fallback to PhysicsSolution
        logger.warning("Could not detect Scene class, using PhysicsSolution as fallback")
        return "PhysicsSolution"
    
    async def render(self, code: str) -> dict:
        """
        Render Manim code to video
        
        Args:
            code: Complete Manim Python code
            
        Returns:
            dict with video_path, filename, duration, etc.
        """
        # Generate unique identifier
        file_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create temp code file
        code_filename = f"scene_{file_id}_{timestamp}.py"
        code_file = self.temp_dir / code_filename
        
        try:
            # Write code to file
            code_file.write_text(code)
            
            # Extract scene class name dynamically
            scene_class = self._extract_scene_class_name(code)
            logger.info(f"Extracted scene class name: {scene_class}")
            
            # Create output subdirectory for this render
            render_output_dir = self.temp_dir / f"render_{file_id}"
            render_output_dir.mkdir(exist_ok=True)
            
            # Execute manim with dynamic class name
            result = await self._execute_manim(code_file, render_output_dir, scene_class)
            
            # Find generated video
            video_path = self._find_generated_video(render_output_dir)
            
            if not video_path:
                raise Exception("Video file not found after rendering")
            
            # Move video to final location
            final_filename = f"solution_{file_id}_{timestamp}.mp4"
            final_path = self.output_dir / final_filename
            shutil.move(str(video_path), str(final_path))
            
            # Get video info
            duration = await self._get_video_duration(final_path)
            
            return {
                "video_path": str(final_path),
                "filename": final_filename,
                "video_url": f"/videos/{final_filename}",
                "duration": duration,
                "generated_at": datetime.now().isoformat(),
                "file_id": file_id
            }
            
        except subprocess.TimeoutExpired:
            raise Exception(f"Manim rendering timed out after {MANIM_TIMEOUT} seconds")
        except Exception as e:
            raise Exception(f"Manim rendering failed: {str(e)}")
        finally:
            # Cleanup temp files
            self._cleanup_temp_files(code_file, render_output_dir if 'render_output_dir' in locals() else None)
    
    async def _execute_manim(self, code_file: Path, output_dir: Path, scene_class: str) -> subprocess.CompletedProcess:
        """Execute manim command"""
        cmd = [
            "manim",
            f"-{MANIM_QUALITY}",  # Remove 'p' flag - quality only, no preview
            "--format", "mp4",
            "--disable_caching",  # Disable caching for cleaner execution
            "--media_dir", str(output_dir),
            str(code_file),
            scene_class  # Dynamic class name
        ]
        
        logger.info(f"Executing Manim with class: {scene_class}")
        
        # Run manim asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=MANIM_TIMEOUT
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise Exception(f"Manim command failed: {error_msg}")
            
            return process
            
        except asyncio.TimeoutError:
            process.kill()
            raise subprocess.TimeoutExpired(cmd, MANIM_TIMEOUT)
    
    def _find_generated_video(self, output_dir: Path) -> Optional[Path]:
        """Find the generated MP4 video in output directory"""
        # Manim creates videos in media/videos/scene_*/quality/
        video_files = list(output_dir.rglob("*.mp4"))
        
        if not video_files:
            return None
        
        # Return the most recent video file
        return max(video_files, key=lambda p: p.stat().st_mtime)
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            duration_str = stdout.decode('utf-8').strip()
            
            return float(duration_str) if duration_str else 0.0
            
        except Exception:
            return 0.0
    
    def _cleanup_temp_files(self, code_file: Optional[Path], render_dir: Optional[Path]):
        """Clean up temporary files"""
        try:
            if code_file and code_file.exists():
                code_file.unlink()
            
            if render_dir and render_dir.exists():
                shutil.rmtree(render_dir, ignore_errors=True)
        except Exception:
            pass  # Best effort cleanup


# Singleton instance
_manim_renderer: Optional[ManimRenderer] = None

def get_manim_renderer() -> ManimRenderer:
    """Get or create Manim renderer singleton"""
    global _manim_renderer
    if _manim_renderer is None:
        _manim_renderer = ManimRenderer()
    return _manim_renderer
