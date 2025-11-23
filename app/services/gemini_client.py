"""
Gemini API Client for generating Manim code from image + text
"""
import re
from typing import Optional
from pathlib import Path
from PIL import Image
import io
import google.generativeai as genai
from app.config import GEMINI_API_KEY


class GeminiClient:
    """Client for interacting with Gemini 3 Pro Image Preview API (Nano Banana Pro)"""
    
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Configure generation parameters for consistent code output
        generation_config = {
            "temperature": 0.4,  # Lower temperature for more consistent, focused code
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 32768,  # Maximum for gemini-3-pro-image-preview (increased from 8192)
        }
        
        # Configure safety settings to reduce blocking (for educational content)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        # Using Gemini 3 Pro Image Preview (vision-optimized model)
        # Model: models/gemini-3-pro-image-preview (Display: Nano Banana Pro)
        # Input limit: 131K tokens, Output: 32K tokens
        self.model = genai.GenerativeModel(
            'models/gemini-3-pro-image-preview',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
    async def generate_manim_code(
        self, 
        image_bytes: bytes, 
        question: str,
        image_mime_type: str = "image/png"
    ) -> str:
        """
        Generate Manim code from image and question text
        
        Args:
            image_bytes: Image data as bytes
            question: User's question text
            image_mime_type: MIME type of the image (unused, kept for compatibility)
            
        Returns:
            Complete Manim Python code as string
        """
        prompt = self._build_prompt(question)
        
        try:
            # Convert bytes to PIL Image (Gemini 3 Pro Image Preview prefers this format)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate content with simpler API format
            response = self.model.generate_content([prompt, image])
            
            # Log response for debugging
            import logging
            logger = logging.getLogger(__name__)
            
            # Check if response has valid parts
            if not response.parts:
                logger.error("Gemini returned empty response!")
                logger.error(f"Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                logger.error(f"Safety ratings: {response.candidates[0].safety_ratings if response.candidates else 'None'}")
                raise ValueError("Gemini returned empty response - possible safety filter block or API issue")
            
            # Check if response has text
            response_text = ""
            try:
                response_text = response.text
            except Exception as e:
                logger.error(f"Could not access response.text: {e}")
                # Try to get text from parts manually
                if response.parts:
                    response_text = "".join([part.text for part in response.parts if hasattr(part, 'text')])
                
                if not response_text:
                    raise ValueError(f"Could not extract text from Gemini response: {str(e)}")
            
            logger.info(f"Gemini response received: {len(response_text)} characters")
            logger.debug(f"Response preview (first 200 chars): {response_text[:200]}")
            
            # Extract code from response
            try:
                code = self._extract_code(response_text)
            except ValueError as extraction_error:
                # If extraction fails, retry with more explicit instructions
                logger.warning(f"Code extraction failed: {extraction_error}")
                logger.warning("Retrying with explicit code-only request...")
                
                retry_prompt = f"""CRITICAL: Your previous response did not contain valid Python code.

You MUST respond with Manim Python code ONLY.

Problem: {question}

Your response must be executable Python code that starts with:
from manim import *

class SolutionScene(Scene):
    def construct(self):
        # Your animation code here

DO NOT write explanations. DO NOT write text.
ONLY Python code. Start with: from manim import *

Generate the code now:"""
                
                retry_response = self.model.generate_content([retry_prompt, image])
                retry_text = ""
                if retry_response.parts:
                    try:
                        retry_text = retry_response.text
                    except:
                        retry_text = "".join([p.text for p in retry_response.parts if hasattr(p, 'text')])
                
                logger.info(f"Retry response: {len(retry_text)} characters")
                code = self._extract_code(retry_text)  # Try extraction again
            
            # Check completeness (balanced parentheses, etc.)
            is_complete, completeness_error = self._check_completeness(code)
            if not is_complete:
                logger.error(f"Code appears incomplete: {completeness_error}")
                from pathlib import Path
                debug_file = Path("/tmp/incomplete_code.py")
                debug_file.write_text(code)
                logger.error(f"Incomplete code saved to: {debug_file}")
                raise ValueError(f"Generated code is incomplete: {completeness_error}")
            
            # Check Python syntax
            is_valid_syntax, syntax_error = self._check_syntax(code)
            if not is_valid_syntax:
                logger.error(f"Generated code has syntax errors: {syntax_error}")
                from pathlib import Path
                debug_file = Path("/tmp/syntax_error_code.py")
                debug_file.write_text(code)
                logger.error(f"Code with syntax error saved to: {debug_file}")
                raise ValueError(f"Generated code has syntax errors: {syntax_error}")
            
            # Auto-fix common issues (missing imports, etc.)
            original_code = code  # Save original before auto-fix
            code_with_fixes = self._fix_common_issues(code)
            
            # CRITICAL: Re-check syntax AFTER auto-fix (auto-fix might introduce errors)
            is_valid_after_fix, syntax_error_after = self._check_syntax(code_with_fixes)
            
            if is_valid_after_fix:
                # Auto-fix succeeded, use the fixed code
                code = code_with_fixes
                logger.info("✓ Auto-fix applied successfully without breaking syntax")
            else:
                # Auto-fix broke the code - fallback to original
                logger.warning(f"Auto-fix broke syntax: {syntax_error_after}")
                logger.warning("Falling back to original code (may have runtime warnings)")
                from pathlib import Path
                debug_file = Path("/tmp/autofix_broke_code.py")
                debug_file.write_text(code_with_fixes)
                logger.error(f"Broken code saved to: {debug_file}")
                
                # Use original code (might have warnings but at least valid syntax)
                code = original_code
                logger.info("Using original code without auto-fix")
            
            # Validate basic code structure
            if not self._validate_code(code):
                # Save failed code to file for debugging
                from pathlib import Path
                
                debug_file = Path("/tmp/failed_validation_code.py")
                debug_file.write_text(code)
                logger.error(f"Failed validation code saved to: {debug_file}")
                
                raise ValueError("Generated code failed validation")
            
            return code
            
        except Exception as e:
            raise Exception(f"Gemini API error (model: {self.model.model_name}): {str(e)}")
    
    def _build_prompt(self, question: str) -> str:
        """Build the prompt for Gemini"""
        return f"""You are an expert Manim animator and STEM educator specializing in:
• Physics (mechanics, forces, motion, energy, waves)
• Mathematics (algebra, geometry, calculus, trigonometry)
• Probability & Statistics (combinatorics, arrangements, distributions)
• Problem-Solving (visual explanations for ANY STEM topic)

TASK: Analyze the provided image and create a Manim animation that VISUALIZES the solution step-by-step.

PROBLEM TYPES YOU MUST HANDLE WITH VISUAL ANIMATIONS:

1. PHYSICS PROBLEMS (mechanics, forces, motion):
   - Use Circle() for particles/masses
   - Use Rectangle() for blocks/surfaces
   - Use Arrow() for force/velocity vectors
   - Animate motion, collisions, rotation

2. MATHEMATICS PROBLEMS (algebra, geometry, calculus):
   - Show equation transformations with MathTex and Transform
   - Draw geometric shapes (Circle, Polygon, Line)
   - Highlight key steps with color changes
   - Animate algebraic manipulations

3. PROBABILITY & COMBINATORICS:
   - Show arrangements visually (e.g., dots on a circle for circular table)
   - Use color coding for different categories (RED=Indian, BLUE=American)
   - Highlight favorable outcomes
   - Show counting process with animations
   - Display fractions and probability calculations

4. GEOMETRY PROBLEMS:
   - Draw accurate shapes (Triangle, Circle, Rectangle, Polygon)
   - Show measurements and angles
   - Animate constructions and proofs
   - Use labels for points and segments

CRITICAL: You MUST generate Manim Python code for ANY type of problem!
Even pure math/probability needs visual animation, not just text explanation.

ADVANCED MANIM FEATURES (USE FOR PROFESSIONAL QUALITY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For Equation Transformations (Mathematics):
✓ Use TransformMatchingTex for morphing equations:
  eq1 = MathTex(r"x^2 + 2x + 1 = 0")
  eq2 = MathTex(r"(x+1)^2 = 0")
  eq3 = MathTex(r"x = -1")
  self.play(TransformMatchingTex(eq1, eq2))
  self.play(TransformMatchingTex(eq2, eq3))
  CRITICAL: Each equation must be a SEPARATE MathTex object!

For Smooth Animations (Physics/Math):
✓ Import and use rate functions:
  from manim.utils.rate_functions import smooth
  self.play(obj.animate.move_to(target), rate_func=smooth, run_time=2)

For Highlighting (Emphasis):
✓ Color individual parts:
  equation = MathTex(r"E", r"=", r"mc^2")
  self.play(equation[0].animate.set_color(YELLOW))
  
For Advanced Geometry:
✓ Use CurvedArrow for curved vectors:
  arrow = CurvedArrow(start, end, angle=TAU/4, color=RED)
  
✓ Use ParametricFunction for curves:
  curve = ParametricFunction(lambda t: [t, t**2, 0], t_range=[-2, 2])

Quality Guidelines:
• Use Transform for object-to-object changes
• Use TransformMatchingTex ONLY for equation morphing (separate MathTex objects)
• Use rate_func=smooth for fluid motion
• Use lag_ratio in animations for sequential effects
• Add strategic self.wait() for pacing

USER'S QUESTION: {question}

CODE LENGTH AND COMPLETENESS (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Keep code CONCISE but COMPLETE - aim for 50-80 lines maximum
• For complex problems: Focus on KEY steps, combine similar equations
• ENSURE ALL PARENTHESES ARE CLOSED: Count your ( and ) - must be equal!
• ENSURE ALL BRACKETS ARE CLOSED: Count your [ and ] - must be equal!
• Double-check the last line is complete (not cut off mid-statement)
• If creating VGroup, ensure closing parenthesis: VGroup(item1, item2)
• Test: Your code must be valid Python with no syntax errors

CRITICAL VISUALIZATION REQUIREMENTS:
1. RECREATE THE SCENE: Use Manim geometric shapes to draw the diagram.
   - Use `Circle()` for particles/masses.
   - Use `Line()` or `Rectangle()` for rods/surfaces.
   - Use `Arrow()` or `CurvedArrow()` for velocity vectors.
2. ANIMATE THE PHYSICS:
   - If objects move, use `.animate.shift()` or `MoveAlongPath()`.
   - If objects rotate, use `.animate.rotate()` or `Rotate()`.
   - If a collision happens, visually show the objects touching and then reacting (bouncing/stopping).
3. SYNC MATH WITH ACTION: Show the relevant equation *after* or *during* the visual event. (e.g., Show "Conservation of Momentum" text right after the collision animation).

ADVANCED ANIMATION TECHNIQUES (FOR PROFESSIONAL QUALITY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Smooth Motion (Use rate functions):
✓ from manim.utils.rate_functions import smooth, rush_into, rush_from
✓ self.play(obj.animate.shift(RIGHT*3), rate_func=smooth, run_time=2)

Curved Arrows (For physics vectors):
✓ Use CurvedArrow instead of Arrow with path_arc:
  arrow = CurvedArrow(start_point, end_point, angle=TAU/6, color=RED)
✓ For short distances, use straight Arrow (no curve)

Equation Morphing (TransformMatchingTex):
✓ Create SEPARATE MathTex for each step:
  step1 = MathTex(r"a^2 + b^2")
  step2 = MathTex(r"c^2")
  self.play(TransformMatchingTex(step1, step2))
✗ DON'T use indexing: TransformMatchingTex(eq[0], eq[1])
✗ DON'T use on .copy() - use ReplacementTransform instead

Sequential Animations (lag_ratio):
✓ self.play(Create(group), lag_ratio=0.1, run_time=2)
  This creates each element with a slight delay - professional effect!

Method Compatibility:
✓ Use Create (not ShowCreation - deprecated)
✓ Use point_from_proportion(t) for arc positions (t from 0 to 1)
✗ DON'T use point_at_angle() - doesn't exist
✗ DON'T use angle_from_proportion() - doesn't exist
✗ DON'T use .midpoint() - use .next_to() instead

FRAME BOUNDARIES AND SIZING (ABSOLUTELY CRITICAL - NO OVERFLOW ALLOWED):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manim Frame: 14.22 units wide × 8 units tall
Safe Content Area: -5.5 to +5.5 (x-axis), -2.8 to +2.8 (y-axis)

FONT SIZE LIMITS (STRICTLY ENFORCED - SMALLER FOR SAFETY):
• Main titles: font_size=26-28 (NEVER larger than 30)
• Long titles: font_size=24 OR split into 2-3 lines
• Equations: font_size=20-24 (smaller for dense content)
• Explanatory text: font_size=18-20
• Small labels: font_size=16-18
• If >6 text elements: Reduce ALL sizes by 20% with .scale(0.8)

SPACING RULES (PREVENT OVERLAP):
• VGroup items: buff=0.5 MINIMUM (never less!)
• Between major sections: buff=0.8
• Text blocks: Ensure 0.6 unit space minimum
• Dense content: buff=0.6 for all VGroup.arrange()

POSITIONING WITH GENEROUS MARGINS:
• Titles: .to_edge(UP, buff=1.0) - larger margin!
• Left content: .to_edge(LEFT, buff=1.2) - extra space!
• Right content: .to_edge(RIGHT, buff=1.2) - extra space!
• Bottom content: .to_edge(DOWN, buff=1.2) - prevent cutoff!
• Center content: .shift(UP*0.5) - keep away from bottom

CRITICAL RULES:
✗ NEVER buff < 0.5 in VGroup.arrange()
✗ NEVER buff < 1.0 in .to_edge()
✗ NEVER font_size > 30
✗ NEVER position beyond ±5.5 (x) or ±2.8 (y)

CONTENT DENSITY MANAGEMENT:
• Count your text elements before creating
• If >6 items → font_size=20, buff=0.6, .scale(0.8)
• If >8 items → font_size=18, buff=0.7, .scale(0.75)
• Always prioritize: READABILITY over quantity
• Better to show fewer steps clearly than many steps cramped

SCALING FOR FIT:
• If content might overflow: Use .scale(0.7) or .scale(0.8)
• For diagrams: .scale_to_fit_width(5) for left side
• For equation groups: equations.to_edge(RIGHT, buff=1.0)

CODE STRUCTURE REQUIREMENTS:
- Use the `Scene` class.
- Grouping: Use `VGroup` to manage equations and objects separately.
- Layout: Keep visual diagram on LEFT, equations on RIGHT, always use .to_edge()

COLORS (ONLY USE THESE - STRICTLY ENFORCED):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID Manim colors (from manim import *):
• Primary: BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE
• Grayscale: WHITE, BLACK, GRAY
• Additional: PINK, TEAL, GOLD, MAROON

DO NOT use these (will cause NameError):
✗ BROWN (not available - use ORANGE or "#8B4513" hex instead)
✗ BRONZE (use GOLD)
✗ SILVER (use GRAY)
✗ TAN, BEIGE, CREAM (use YELLOW or WHITE)

For custom colors, use hex strings:
• Brown: "#8B4513"
• Silver: "#C0C0C0"
• Any color: "#RRGGBB" format

Example CORRECT:
✓ Circle(color=BLUE)
✓ Rectangle(color=ORANGE)
✓ Line(color="#8B4513")  # Custom brown

Example INCORRECT (will crash):
✗ Circle(color=BROWN)  # NameError!

STRICT CODE TEMPLATE WITH PROPER LAYOUT:
```python
from manim import *

class PhysicsSolution(Scene):
    def construct(self):
        # 1. TITLE (Short, with margins, split if long)
        title = Text("Problem Title", font_size=32).to_edge(UP, buff=0.5)
        # For long titles:
        # title = Text("Long Title\\nSplit Into Lines", font_size=30).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        
        # 2. VISUAL DIAGRAM (Left side, within frame)
        particle = Circle(radius=0.3, color=BLUE)
        particle.shift(LEFT*4)  # Keep within -6 to 6
        self.play(Create(particle))
        
        # 3. EQUATIONS (Right side, arranged properly)
        eq1 = MathTex(r"F = ma", font_size=28)
        eq2 = MathTex(r"v = u + at", font_size=28)
        equations = VGroup(eq1, eq2).arrange(DOWN, buff=0.4)
        equations.to_edge(RIGHT, buff=1.0)  # Critical: keeps within frame
        self.play(Write(equations))
        
        # 4. ANIMATE PHYSICS (smooth motion)
        self.play(particle.animate.shift(RIGHT*4), run_time=2)
        self.wait(1)
        
        # 5. FINAL ANSWER (center, clear, reasonable size)
        answer = Text("Final Answer", font_size=32, color=GREEN)
        answer.move_to(ORIGIN)
        self.play(Write(answer))
        self.wait(2)
```

CRITICAL LAYOUT EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example 1: Long Title (CORRECT)
    title = Text("Radiation Heat Transfer\\nbetween Parallel Plates", font_size=30)
    title.to_edge(UP, buff=0.5)

Example 2: Multiple Equations (CORRECT)
    eq1 = MathTex(r"W_0 = \\sigma T_P^4 - \\sigma T_Q^4", font_size=26)
    eq2 = MathTex(r"W_S = \\sigma T_P^4 - \\sigma T_A^4", font_size=26)
    eq_group = VGroup(eq1, eq2).arrange(DOWN, buff=0.3)
    eq_group.to_edge(RIGHT, buff=1.0)

Example 3: Probability/Combinatorics (CORRECT for circular arrangements)
    # Circular table with people
    table = Circle(radius=2, color=WHITE)
    # Positions around circle (10 people = 10 positions)
    positions = VGroup(*[Dot(table.point_from_proportion(i/10), color=BLUE) for i in range(10)])
    # Labels for people
    labels = VGroup(*[Text("P", font_size=18).next_to(pos, OUT) for pos in positions])
    # Probability calculation
    prob_eq = MathTex("P(A|B) = P(A and B) / P(B)", font_size=28)
    prob_eq.to_edge(RIGHT, buff=1.0)
    
Example 4: Geometry Problem (CORRECT)
    # Triangle with measurements
    triangle = Polygon(ORIGIN, RIGHT*3, UP*2, color=YELLOW)
    side_label = Text("a = 5", font_size=24).next_to(triangle, DOWN)
    self.play(Create(triangle), Write(side_label))

Example 3: Diagram + Math Layout (CORRECT)
    # Left side - diagram
    diagram = VGroup(plate1, plate2, arrows).scale(0.8)
    diagram.to_edge(LEFT, buff=1.0)
    
    # Right side - equations
    equations = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.3)
    equations.to_edge(RIGHT, buff=1.0)
```

ANIMATION QUALITY GUIDELINES (FOR BEST RESULTS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timing and Pacing:
✓ Use run_time parameter for control: run_time=2 (seconds)
✓ Use rate_func for smooth motion (import first!):
  from manim.utils.rate_functions import smooth
  self.play(animation, rate_func=smooth)
✓ Use lag_ratio for sequential effects:
  self.play(Create(group), lag_ratio=0.1)

Animation Selection:
✓ Write() - For text and equations appearing
✓ Create() - For drawing shapes (not ShowCreation)
✓ FadeIn/FadeOut - For gentle appearance/disappearance
✓ Transform() - For general object transformations
✓ TransformMatchingTex() - ONLY for equation morphing (separate objects!)
✓ ReplacementTransform() - When replacing one object with another

Strategic Wait Times:
✓ self.wait(1) - After major steps
✓ self.wait(0.5) - Between quick transitions
✓ self.wait(2) - For final answer display

GOAL: Create 3Blue1Brown quality animations - smooth, clear, professional!

OUTPUT FORMAT (ABSOLUTELY CRITICAL - FOLLOW EXACTLY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR RESPONSE MUST START WITH THESE EXACT CHARACTERS:
f
r
o
m

That is: "from manim import *" must be your FIRST LINE.

DO NOT START WITH:
✗ "Here's the solution:"
✗ "```python"
✗ "Let me create..."
✗ ANY explanatory text

YOUR ENTIRE RESPONSE FORMAT:
from manim import *

class SolutionScene(Scene):
    def construct(self):
        [your actual animation code here]

Use class names like: PhysicsSolution, ProbabilitySolution, GeometrySolution, or SolutionScene

CRITICAL RULES FOR ALL PROBLEM TYPES:
- Physics problems → Animate physical objects and motion
- Math problems → Animate equations and transformations  
- Probability → Visualize arrangements, outcomes, counting
- Geometry → Draw shapes, show measurements, prove visually
- Return ONLY the raw Python code - nothing else
- ALWAYS include visual animations, not just text explanations
- DO NOT use ImageMobject or external image files - draw everything with code
- DO NOT reference external files - create all visuals with Manim shapes
- Do not use external image assets; draw everything with code
- Ensure all LaTeX is properly escaped (e.g., MathTex(r"\\\\frac{{m}}{{M}}"))
- Class name MUST be exactly: class PhysicsSolution(Scene):
- NO markdown wrappers (no ```)
- NO explanations before or after
- DO NOT include 'pass' statements - write actual animation code
- Your response should be valid Python that can be saved to a .py file and executed immediately

REMEMBER: First character = 'f', First line = "from manim import *"
"""
    
    def _extract_code(self, response_text: str) -> str:
        """Extract Python code from Gemini's response with multiple robust strategies"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Clean up response text
        response_text = response_text.strip()
        
        # Strategy 1: Try markdown code blocks (all variations)
        patterns = [
            r"```python\s*(.*?)\s*```",
            r"```py\s*(.*?)\s*```",
            r"```Python\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Check if this block contains Manim code
                if ("from manim import" in match or "import manim" in match) and "class" in match and "Scene" in match:
                    logger.info("Code extracted via markdown block pattern")
                    return match.strip()
        
        # Strategy 2: Look for code starting with "from manim" or "import manim"
        if "from manim import" in response_text or "import manim" in response_text:
            lines = response_text.split('\n')
            start_idx = None
            
            # Find first line with import
            for i, line in enumerate(lines):
                if 'from manim import' in line or 'import manim' in line:
                    start_idx = i
                    break
            
            if start_idx is not None:
                # Extract from import to end, filtering out trailing explanations
                code_lines = lines[start_idx:]
                # Find where code likely ends (empty lines followed by explanatory text)
                # For now, take everything
                code = '\n'.join(code_lines).strip()
                
                # Verify it has class definition
                if "class" in code and "Scene" in code:
                    logger.info("Code extracted via import statement search")
                    return code
        
        # Strategy 3: Look for class definition and work backwards
        if "class " in response_text and "Scene" in response_text and "def construct" in response_text:
            # Find the class
            class_match = re.search(r'(class\s+\w+\s*\([^)]*Scene[^)]*\):.*?def\s+construct.*)', response_text, re.DOTALL)
            if class_match:
                code = class_match.group(1).strip()
                # Prepend import if missing
                if "from manim import" not in code and "import manim" not in code:
                    code = "from manim import *\n\n" + code
                logger.info("Code extracted via class definition search")
                return code
        
        # Strategy 4: Aggressive extraction - find anything that looks like Python code
        if "def construct" in response_text:
            # Find the earliest import or class statement
            start_markers = ["from manim", "import manim", "class "]
            earliest_pos = len(response_text)
            
            for marker in start_markers:
                pos = response_text.find(marker)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            
            if earliest_pos < len(response_text):
                code = response_text[earliest_pos:].strip()
                # Clean up any trailing explanation text (heuristic)
                # If we see non-code patterns after the class, cut there
                if '\n\n\n' in code:
                    # Multiple blank lines often indicate end of code
                    code = code.split('\n\n\n')[0]
                
                logger.info("Code extracted via aggressive pattern matching")
                return code
        
        # All strategies failed - log details
        logger.error("=" * 60)
        logger.error("CODE EXTRACTION FAILED - Response Analysis:")
        logger.error(f"Response length: {len(response_text)} characters")
        logger.error(f"Contains 'from manim': {'from manim' in response_text}")
        logger.error(f"Contains 'import manim': {'import manim' in response_text}")
        logger.error(f"Contains 'class': {'class' in response_text}")
        logger.error(f"Contains 'Scene': {'Scene' in response_text}")
        logger.error(f"Contains 'def construct': {'def construct' in response_text}")
        logger.error(f"First 300 chars of response:")
        logger.error(response_text[:300])
        logger.error(f"Last 300 chars of response:")
        logger.error(response_text[-300:])
        logger.error("=" * 60)
        
        raise ValueError("No valid Python code found in response")
    
    def _fix_common_issues(self, code: str) -> str:
        """Automatically fix common issues in generated code including layout problems"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Fix 0: Replace invalid Manim color names with valid alternatives
        color_replacements = {
            'BROWN': '"#8B4513"',      # Brown hex color (QUOTED)
            'BRONZE': 'GOLD',          # Close alternative
            'SILVER': 'GRAY',          # Close alternative
            'GREY_BROWN': 'GRAY',
            'TAN': 'YELLOW',
            'BEIGE': '"#F5F5DC"',      # Beige hex (QUOTED)
            'CREAM': 'WHITE',
            'OLIVE': 'GREEN',
            'NAVY': 'BLUE',
        }
        
        for invalid_color, replacement in color_replacements.items():
            # Check if this invalid color is used
            pattern = rf'\bcolor\s*=\s*{invalid_color}\b'
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Replacing invalid color {invalid_color} with {replacement}")
                code = re.sub(pattern, f'color={replacement}', code, flags=re.IGNORECASE)
        
        # Fix 0b: Smart handling of path_arc (keep for CurvedArrow, remove from regular Arrow if problematic)
        # path_arc on very short arrows can cause geometry errors
        # But it's great for curved arrows - we want to keep quality!
        # For now, suggest using CurvedArrow instead of Arrow with path_arc
        if 'Arrow(' in code and 'path_arc' in code:
            logger.warning("Converting Arrow with path_arc to CurvedArrow")
            # Replace Arrow with path_arc → CurvedArrow with angle parameter
            # CurvedArrow uses start_point and end_point (not start/end)
            code = re.sub(
                r'Arrow\(start\s*=\s*([^,]+),\s*end\s*=\s*([^,]+),\s*path_arc\s*=\s*([^,)]+)([^)]*)\)',
                r'CurvedArrow(start_point=\1, end_point=\2, angle=\3\4)',
                code
            )
            # Also handle without parameter names
            code = re.sub(
                r'Arrow\(([^,]+),\s*([^,]+),\s*path_arc\s*=\s*([^,)]+)([^)]*)\)',
                r'CurvedArrow(start_point=\1, end_point=\2, angle=\3\4)',
                code
            )
        
        # Fix: Fix CurvedArrow parameter names (start/end → start_point/end_point)
        if 'CurvedArrow' in code:
            logger.warning("Fixing CurvedArrow parameter names")
            # Replace start= with start_point=
            code = re.sub(r'CurvedArrow\(([^)]*)start\s*=', r'CurvedArrow(\1start_point=', code)
            # Replace end= with end_point=
            code = re.sub(r'CurvedArrow\(([^)]*)end\s*=', r'CurvedArrow(\1end_point=', code)
        
        # Remove arc_center from arrows (also problematic)
        if 'arc_center' in code and 'Arrow' in code:
            logger.warning("Removing arc_center parameter to prevent errors")
            code = re.sub(r',\s*arc_center\s*=\s*[^,)]+', '', code)
        
        # Fix: Replace invalid linestyle parameters (COMPREHENSIVE)
        # DASHED, DOTTED, SOLID are not valid - Manim uses DashedVMobject or DashedLine
        if 'linestyle' in code:
            logger.warning("Converting/removing linestyle parameter")
            
            # Strategy 1: Convert Circle/Arc with linestyle to DashedVMobject (keeps quality!)
            if 'Circle' in code or 'Arc' in code:
                # Wrap Circle/Arc in DashedVMobject for dashed effect
                code = re.sub(
                    r'(Circle|Arc)\(([^)]+),\s*linestyle\s*=\s*\w+([^)]*)\)',
                    r'DashedVMobject(\1(\2\3))',
                    code
                )
            
            # Strategy 2: Remove linestyle from any remaining objects
            code = re.sub(r',\s*linestyle\s*=\s*\w+', '', code)
            code = re.sub(r'linestyle\s*=\s*\w+\s*,', '', code)
            code = re.sub(r'linestyle\s*=\s*\w+', '', code)
        
        # Fix: Remove non-existent Arc methods
        if 'angle_from_proportion' in code:
            logger.warning("Removing angle_from_proportion (doesn't exist on Arc)")
            # Remove the method call and replace rotate with simple angle
            code = re.sub(
                r'\.rotate\([^.]*\.angle_from_proportion\([^)]*\)[^)]*\)',
                '.rotate(0)',
                code
            )
        
        # Fix: Remove ImageMobject usage (external files don't exist)
        if 'ImageMobject' in code:
            logger.warning("Removing ImageMobject - external images not available")
            # Remove entire ImageMobject lines
            code = re.sub(r'.*ImageMobject\([^)]+\).*\n', '', code)
            # Remove any variable assignments using ImageMobject
            code = re.sub(r'\w+\s*=\s*ImageMobject\([^)]+\)[^\n]*\n', '', code)
            # Remove references to the removed variable in self.play/add
            code = re.sub(r'self\.play\([^)]*problem_image[^)]*\)\s*\n', '', code)
            code = re.sub(r'self\.add\([^)]*problem_image[^)]*\)\s*\n', '', code)
        
        # Fix: Replace deprecated/non-existent Manim names
        # ShowCreation is deprecated → use Create
        if 'ShowCreation' in code:
            logger.warning("Replacing deprecated ShowCreation with Create")
            code = code.replace('ShowCreation', 'Create')
        
        # Fix: Handle TransformMatchingTex usage issues
        # This is a POWERFUL feature - we want to keep it but fix incorrect usage
        if 'TransformMatchingTex' in code:
            # Check for problematic pattern: TransformMatchingTex(obj[0], obj[1])
            # where obj is a single MathTex (not VGroup)
            problematic_pattern = r'TransformMatchingTex\((\w+)\[(\d+)\](?:\.copy\(\))?,\s*\1\[(\d+)\]'
            if re.search(problematic_pattern, code):
                logger.warning("Fixing TransformMatchingTex indexing issue")
                # Replace with ReplacementTransform which handles this better
                code = re.sub(problematic_pattern, r'ReplacementTransform(\1[\2], \1[\3])', code)
            
            # Also handle cross-object transforms that might fail
            # TransformMatchingTex works best with full MathTex objects
            # If it's being used on .copy(), just use Transform instead
            code = re.sub(
                r'TransformMatchingTex\(([^,]+\.copy\(\))',
                r'Transform(\1',
                code
            )
        
        # Fix: Remove other non-existent methods
        non_existent_methods = ['get_arc_center', 'set_arc_center', 'get_tangent_vector', 'get_unit_vector', 'midpoint', 'point_at_angle']
        for method in non_existent_methods:
            if method in code:
                logger.warning(f"Removing/replacing non-existent method: {method}")
                if method == 'midpoint':
                    # Replace .midpoint() with .next_to() which is the correct method
                    code = re.sub(r'\.midpoint\(', '.next_to(', code)
                elif method == 'point_at_angle':
                    # Replace with point_from_proportion
                    code = re.sub(r'\.point_at_angle\(([^)]+)\)', r'.point_from_proportion(\1/(2*PI))', code)
                else:
                    code = re.sub(rf'\.{method}\([^)]*\)', '', code)
        
        # Fix: Fix LaTeX escape sequences (SAFE VERSION - only simple single-line cases)
        def safe_latex_to_raw(match):
            """Safely convert MathTex to raw string"""
            content = match.group(1)
            # Only convert simple cases (single line, reasonable length, no nested quotes)
            if '\n' not in content and len(content) < 100 and '\\' in content:
                return f'MathTex(r"{content}"'
            return match.group(0)  # Return unchanged if complex
        
        # Apply to simple MathTex only
        code = re.sub(r'MathTex\("([^"\n]+)"(?=[,)])', safe_latex_to_raw, code)
        code = re.sub(r"MathTex\('([^'\n]+)'(?=[,)])", lambda m: f"MathTex(r'{m.group(1)}'", code)
        
        # Fix 1: Add missing rate function imports
        # Check if code uses ease_* rate functions
        needs_rate_funcs = bool(re.search(r'rate_func\s*=\s*ease_\w+', code))
        
        if needs_rate_funcs and 'from manim.utils.rate_functions import' not in code:
            # Extract which rate functions are used
            rate_funcs = set(re.findall(r'rate_func\s*=\s*(ease_\w+)', code))
            
            if rate_funcs:
                # Add import statement after main manim import
                import_statement = f"from manim.utils.rate_functions import {', '.join(sorted(rate_funcs))}"
                code = code.replace(
                    'from manim import *',
                    f'from manim import *\n{import_statement}'
                )
        
        # Fix 2: Cap font sizes to prevent overflow (MORE CONSERVATIVE)
        # Replace any font_size > 30 with 28 (stricter for better layout)
        code = re.sub(r'font_size\s*=\s*([3-9]\d|[1-9]\d{2,})', 'font_size=28', code)
        
        # Fix 2b: If many text elements, scale down ALL fonts by 20%
        text_count = code.count('Text(') + code.count('MathTex(')
        if text_count > 8:
            logger.warning(f"Detected {text_count} text elements - reducing all font sizes by 20%")
            # Reduce all fonts by 20% for dense content
            def reduce_font(match):
                size = int(match.group(1))
                new_size = max(16, int(size * 0.8))  # Min 16, reduce by 20%
                return f'font_size={new_size}'
            code = re.sub(r'font_size\s*=\s*(\d+)', reduce_font, code)
        
        # Fix 3: Replace extreme positioning with .to_edge()
        # Extreme left: LEFT*7, LEFT*8, etc. → .to_edge(LEFT, buff=1.0)
        code = re.sub(r'\.shift\(LEFT\s*\*\s*([7-9]|1\d)\)', '.to_edge(LEFT, buff=1.0)', code)
        code = re.sub(r'\.move_to\(LEFT\s*\*\s*([7-9]|1\d)\)', '.to_edge(LEFT, buff=1.0).shift(ORIGIN)', code)
        
        # Extreme right
        code = re.sub(r'\.shift\(RIGHT\s*\*\s*([7-9]|1\d)\)', '.to_edge(RIGHT, buff=1.0)', code)
        code = re.sub(r'\.move_to\(RIGHT\s*\*\s*([7-9]|1\d)\)', '.to_edge(RIGHT, buff=1.0).shift(ORIGIN)', code)
        
        # Extreme up
        code = re.sub(r'\.shift\(UP\s*\*\s*([4-9]|1\d)\)', '.to_edge(UP, buff=0.5)', code)
        code = re.sub(r'\.move_to\(UP\s*\*\s*([4-9]|1\d)\)', '.to_edge(UP, buff=0.5).shift(ORIGIN)', code)
        
        # Extreme down
        code = re.sub(r'\.shift\(DOWN\s*\*\s*([4-9]|1\d)\)', '.to_edge(DOWN, buff=1.0)', code)
        code = re.sub(r'\.move_to\(DOWN\s*\*\s*([4-9]|1\d)\)', '.to_edge(DOWN, buff=1.0).shift(ORIGIN)', code)
        
        # Fix 4: Remove standalone 'pass' statements if there's real code
        if 'self.play' in code or 'self.wait' in code:
            code = re.sub(r'^\s*pass\s*$', '', code, flags=re.MULTILINE)
        
        # Fix 5: Clean up excessive whitespace
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        # Fix 5b: Enforce minimum buff values for spacing (PREVENT OVERLAP)
        # Replace small buff with safer minimums
        code = re.sub(r'buff\s*=\s*0\.[0-3](?!\d)', 'buff=0.5', code)  # 0.0-0.3 → 0.5
        # For .to_edge(), enforce buff >= 1.0
        code = re.sub(r'\.to_edge\(([^,]+),\s*buff\s*=\s*0\.\d+', r'.to_edge(\1, buff=1.2', code)
        
        # Fix 5c: Add auto-scaling for dense VGroups (>6 items)
        # Detect VGroup with many items and add .scale()
        def add_scale_for_dense_vgroup(match):
            full_match = match.group(0)
            # Count commas to estimate items
            comma_count = full_match.count(',')
            if comma_count > 5:
                # Add scale(0.8) if not already present
                if '.scale(' not in full_match:
                    return full_match + '.scale(0.8)'
            return full_match
        
        code = re.sub(
            r'VGroup\([^)]+\)\.arrange\([^)]+\)',
            add_scale_for_dense_vgroup,
            code
        )
        
        # Fix 6: Clean up malformed comma/parenthesis patterns (CRITICAL)
        # These can be introduced by parameter removal
        code = re.sub(r',\s*,', ',', code)  # Double comma → single comma
        code = re.sub(r',\s*\)', ')', code)  # Trailing comma before ) → remove
        code = re.sub(r'\(\s*,', '(', code)  # Comma after ( → remove
        code = re.sub(r',\s*\]', ']', code)  # Trailing comma before ] → remove
        code = re.sub(r'\[\s*,', '[', code)  # Comma after [ → remove
        
        # Fix 7: Fix malformed constructor calls (MORE SPECIFIC)
        # Pattern: Arc(...), color=...) → Arc(..., color=...)
        # Only match specific Manim constructors to avoid false positives
        manim_objects = r'(?:Arc|Circle|Rectangle|Line|Arrow|Dot|Text|MathTex|VGroup|Ellipse|Polygon|Square|Triangle)'
        code = re.sub(
            rf'({manim_objects}\([^)]+)\)\s*,\s*(\w+\s*=)',
            r'\1, \2',
            code
        )
        
        return code
    
    def _check_syntax(self, code: str) -> tuple[bool, str]:
        """
        Check if code has valid Python syntax
        Returns: (is_valid, error_message)
        """
        import ast
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Try to parse as Python AST
            ast.parse(code)
            logger.info("✓ Python syntax check PASSED")
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.error(f"✗ Python syntax check FAILED")
            logger.error(f"  Line {e.lineno}: {e.text}")
            logger.error(f"  Error: {e.msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            logger.error(f"✗ Python syntax check FAILED: {error_msg}")
            return False, error_msg
    
    def _check_completeness(self, code: str) -> tuple[bool, str]:
        """Check if code appears complete (not truncated)"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Check 1: Balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            msg = f"Unbalanced parentheses: {open_parens} open, {close_parens} close"
            logger.error(f"✗ Completeness check FAILED: {msg}")
            return False, msg
        
        # Check 2: Balanced brackets
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        if open_brackets != close_brackets:
            msg = f"Unbalanced brackets: {open_brackets} open, {close_brackets} close"
            logger.error(f"✗ Completeness check FAILED: {msg}")
            return False, msg
        
        # Check 3: Balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            msg = f"Unbalanced braces: {open_braces} open, {close_braces} close"
            logger.error(f"✗ Completeness check FAILED: {msg}")
            return False, msg
        
        logger.info("✓ Completeness check PASSED")
        return True, ""
    
    def _validate_code(self, code: str) -> bool:
        """Flexible validation of generated code with detailed logging"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Check 1: Has manim import (flexible)
        has_import = (
            "from manim import" in code or 
            "import manim" in code
        )
        
        # Check 2: Has a Scene class (VERY FLEXIBLE - accepts any class inheriting from Scene)
        # Match: class ClassName(Scene): or class ClassName (Scene): or any variation
        has_scene_class = bool(re.search(r'class\s+\w+\s*\([^)]*Scene[^)]*\)', code))
        
        # Check 3: Has construct method (FLEXIBLE - accepts variations with kwargs, whitespace)
        has_construct = bool(re.search(r'def\s+construct\s*\(\s*self\s*[,)]', code))
        
        # Check 4: Has some actual animation code (FLEXIBLE - accepts any Scene method)
        has_animation = (
            "self.play" in code or 
            "self.wait" in code or 
            "self.add" in code or
            "self.remove" in code or
            bool(re.search(r'self\.\w+\(', code))  # Any self.method() call
        )
        
        # Check 5: Has basic Python structure
        has_class_keyword = "class " in code
        has_def_keyword = "def " in code
        
        # Log validation results
        logger.info("=" * 60)
        logger.info("VALIDATION CHECKS:")
        logger.info(f"  ✓ Has import: {has_import}")
        logger.info(f"  ✓ Has Scene class: {has_scene_class}")
        logger.info(f"  ✓ Has construct(): {has_construct}")
        logger.info(f"  ✓ Has animation code: {has_animation}")
        logger.info(f"  ✓ Has 'class' keyword: {has_class_keyword}")
        logger.info(f"  ✓ Has 'def' keyword: {has_def_keyword}")
        
        # Determine if valid
        is_valid = has_import and has_scene_class and has_construct and has_animation
        
        if not is_valid:
            logger.error("VALIDATION FAILED!")
            logger.error("Generated code (first 1500 characters):")
            logger.error("-" * 60)
            logger.error(code[:1500])
            logger.error("-" * 60)
            logger.error("Missing requirements:")
            if not has_import:
                logger.error("  ✗ Missing manim import")
            if not has_scene_class:
                logger.error("  ✗ Missing Scene class")
            if not has_construct:
                logger.error("  ✗ Missing construct() method")
            if not has_animation:
                logger.error("  ✗ Missing animation code (self.play/wait/add)")
        else:
            logger.info("✓ VALIDATION PASSED!")
        
        logger.info("=" * 60)
        
        return is_valid


# Singleton instance
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
