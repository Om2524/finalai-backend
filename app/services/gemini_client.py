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

PHYSICS LOGIC & ACCURACY RULES (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ROTATIONAL DYNAMICS (Hinged Rods/Beams):
   - You MUST determine rotation direction based on the impact vector.
   - RULE: If a particle travels RIGHT and hits the BOTTOM of a hanging rod, the rod MUST rotate COUNTER-CLOCKWISE (Positive Angle).
   - RULE: If a particle travels LEFT and hits the BOTTOM of a hanging rod, the rod MUST rotate CLOCKWISE (Negative Angle).
   - ALWAYS use the 'about_point' parameter in Rotate().
     Example: self.play(Rotate(rod, angle=PI/4, about_point=pivot_location))

2. COLLISION MOMENTUM:
   - After collision, the object hitting the rod usually stops or rebounds.
   - The rod MUST swing in the SAME direction as the incoming particle's velocity vector.

3. COORDINATE SYSTEMS:
   - Define a fixed pivot point (e.g., UP*2).
   - Ensure the Rod is defined relative to that pivot.

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
✓ Use .get_center() to get object position
✗ DON'T use point_at_angle() - doesn't exist
✗ DON'T use angle_from_proportion() - doesn't exist
✗ DON'T use .midpoint() - use .next_to() instead
✗ DON'T use .get_position() - use .get_center() instead
✗ DON'T use .aligned_edge() as a method - it's a PARAMETER for next_to()/arrange()
   Example WRONG: obj.move_to(target).aligned_edge(LEFT)
   Example CORRECT: obj.next_to(target, DOWN, aligned_edge=LEFT)

FRAME BOUNDARIES AND SIZING (ABSOLUTELY CRITICAL - NO OVERFLOW ALLOWED):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manim Frame: 14.22 units wide × 8 units tall
Safe Content Area: -5.5 to +5.5 (x-axis), -2.5 to +2.5 (y-axis)

⚠️⚠️⚠️ MANDATORY SCREEN MANAGEMENT (VIOLATION = BROKEN VIDEO) ⚠️⚠️⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE #1 CAUSE OF BROKEN VIDEOS IS TEXT OVERLAP! FOLLOW THESE RULES EXACTLY!

RULE 1: MAXIMUM 3-4 TEXT ELEMENTS ON SCREEN AT ANY TIME
- Before adding ANY new text, count visible text elements
- If you have 3+ text elements, you MUST FadeOut some before adding more
- Diagrams/shapes don't count, but their labels DO count

RULE 2: NUCLEAR CLEAR BETWEEN MAJOR SECTIONS
Every solution has sections (Problem → Setup → Solution → Answer).
Between EACH major section, use the nuclear clear:
```python
self.play(*[FadeOut(mob) for mob in self.mobjects])  # ← CLEARS EVERYTHING
self.wait(0.3)
```

RULE 3: NEVER STACK TEXT AT SAME Y-POSITION
- If text1 is at to_edge(UP), text2 CANNOT use to_edge(UP) until text1 is gone
- Use Transform() to REPLACE text, not Write() to ADD more:
```python
# WRONG (causes overlap):
title1 = Text("Step 1").to_edge(UP)
self.play(Write(title1))
title2 = Text("Step 2").to_edge(UP)
self.play(Write(title2))  # ← OVERLAPS WITH title1!

# CORRECT (clean replacement):
title = Text("Step 1").to_edge(UP)
self.play(Write(title))
new_title = Text("Step 2").to_edge(UP)
self.play(Transform(title, new_title))  # ← REPLACES, no overlap!
```

RULE 4: ZONE-BASED LAYOUT (ONE ELEMENT PER ZONE)
Divide the screen into zones. Only ONE text element per zone at a time!
```
┌─────────────────────────────────────────┐
│         TOP ZONE (y > 2.5)              │ ← Titles ONLY
├───────────────────┬─────────────────────┤
│   LEFT ZONE       │    RIGHT ZONE       │
│   (x < -2)        │    (x > 2)          │ ← Equations OR Diagram
│   Equations       │    Diagram          │
├───────────────────┴─────────────────────┤
│        BOTTOM ZONE (y < -2.5)           │ ← Final answers, notes
└─────────────────────────────────────────┘
```

RULE 5: THE SACRED PATTERN (COPY THIS EXACTLY):
```python
def construct(self):
    # ══════════ SECTION 1: TITLE (5 sec) ══════════
    title = Text("Problem Title", font_size=26).to_edge(UP)
    self.play(Write(title))
    self.wait(1)
    self.play(FadeOut(title))  # ← CLEAR!
    
    # ══════════ SECTION 2: SETUP (10 sec) ══════════
    # Maximum: 1 title + 1 diagram + 2 labels = 4 elements
    setup_title = Text("Given:", font_size=22).to_edge(UP)
    diagram = VGroup(...).move_to(ORIGIN)
    self.play(Write(setup_title), Create(diagram))
    self.wait(2)
    self.play(*[FadeOut(mob) for mob in self.mobjects])  # ← NUCLEAR CLEAR!
    
    # ══════════ SECTION 3: SOLUTION (15 sec) ══════════
    # Show ONE equation at a time, then clear or transform
    eq1 = MathTex(r"Step 1...").move_to(ORIGIN)
    self.play(Write(eq1))
    self.wait(1)
    
    eq2 = MathTex(r"Step 2...").move_to(ORIGIN)
    self.play(Transform(eq1, eq2))  # ← REPLACE, don't add!
    self.wait(1)
    
    self.play(FadeOut(eq1))  # ← CLEAR before answer!
    
    # ══════════ SECTION 4: ANSWER (5 sec) ══════════
    answer = MathTex(r"Final Answer", font_size=28, color=GREEN)
    answer.move_to(ORIGIN)
    box = SurroundingRectangle(answer, color=GREEN, buff=0.3)
    self.play(Write(answer), Create(box))
    self.wait(3)
```

RULE 6: FORBIDDEN PATTERNS (NEVER DO THESE):
```python
# FORBIDDEN 1: Multiple to_edge(UP) without clear
text1.to_edge(UP)
text2.to_edge(UP)  # ← FORBIDDEN! Clear text1 first!

# FORBIDDEN 2: Adding text in a loop without clearing
for step in steps:
    eq = MathTex(step)
    self.play(Write(eq))  # ← FORBIDDEN! Each iteration adds more!

# FORBIDDEN 3: More than 4 Write() calls without FadeOut
self.play(Write(a))
self.play(Write(b))
self.play(Write(c))
self.play(Write(d))
self.play(Write(e))  # ← FORBIDDEN! Too many elements!
```

FONT SIZE LIMITS (STRICTLY ENFORCED):
• Main titles: font_size=24-26 (NEVER larger than 28)
• Equations: font_size=22-24 (scale down if long)
• Explanatory text: font_size=18-20
• Small labels: font_size=14-16

LONG EQUATION HANDLING (CRITICAL - PREVENTS CUTOFF):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For ANY equation longer than 30 characters:
1. ALWAYS use .scale_to_fit_width(12) after creating
2. OR break into multiple lines with aligned MathTex

Example - Long equation (CORRECT):
```python
long_eq = MathTex(r"3W_S = \sigma(T_P^4 - T_1^4) + \sigma(T_1^4 - T_2^4) + \sigma(T_2^4 - T_Q^4)")
long_eq.scale_to_fit_width(12)  # ← MANDATORY for long equations
long_eq.move_to(ORIGIN)
```

Example - Multi-line equations (CORRECT):
```python
eq_line1 = MathTex(r"3W_S = \sigma(T_P^4 - T_1^4)", font_size=22)
eq_line2 = MathTex(r"+ \sigma(T_1^4 - T_2^4)", font_size=22)
eq_line3 = MathTex(r"+ \sigma(T_2^4 - T_Q^4)", font_size=22)
eq_group = VGroup(eq_line1, eq_line2, eq_line3).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
eq_group.move_to(ORIGIN)
```

SPACING RULES (PREVENT OVERLAP):
• VGroup items: buff=0.4 to 0.6 (consistent spacing)
• Between major sections: Use FadeOut/FadeIn transitions
• Never place two text objects without explicit positioning

POSITIONING WITH GENEROUS MARGINS:
• Titles: .to_edge(UP, buff=0.5)
• Left content: .to_edge(LEFT, buff=1.0)
• Right content: .to_edge(RIGHT, buff=1.0)
• Bottom content: .to_edge(DOWN, buff=1.0)
• Center: .move_to(ORIGIN)

CRITICAL RULES:
✗ NEVER show multiple text blocks simultaneously without clearing previous
✗ NEVER skip FadeOut between major content changes
✗ NEVER let equations extend beyond frame (use scale_to_fit_width)
✗ NEVER position text at same location as existing text
✗ NEVER use font_size > 28

CONTENT DENSITY MANAGEMENT:
• Maximum 4-5 text elements visible at any time
• Always clear previous section before new section
• If showing multiple equations: use VGroup.arrange() with buff=0.4
• For step-by-step: show one step, explain, FadeOut, show next step

SCALING FOR FIT (MANDATORY FOR LONG CONTENT):
• All equations: Check width, use .scale_to_fit_width(12) if needed
• Equation groups: .scale_to_fit_width(11) for the VGroup
• Single long equation: .scale_to_fit_width(10)
• Very long equation: Break into 2-3 separate MathTex objects

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

⚠️ BANNED COLORS (WILL CRASH - NEVER USE THESE WORDS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ BROWN - DOES NOT EXIST! Use "#8B4513" instead
✗ BRONZE - DOES NOT EXIST! Use GOLD instead
✗ SILVER - DOES NOT EXIST! Use GRAY instead
✗ TAN, BEIGE, CREAM - DO NOT EXIST! Use "#F5DEB3" or WHITE

CRITICAL: The word "BROWN" must NEVER appear in your code!
- NOT in color=BROWN
- NOT in interpolate_color(BROWN, ...)
- NOT anywhere else!

For wood/brown elements, ALWAYS use: "#8B4513"

Example CORRECT:
✓ Rectangle(color="#8B4513")  # Wood/brown color
✓ interpolate_color("#8B4513", WHITE, 0.2)  # Brown blend
✓ Circle(color=BLUE)

Example INCORRECT (will crash):
✗ color=BROWN  # NameError!
✗ interpolate_color(BROWN, WHITE, 0.2)  # NameError!

GEOMETRY & STYLING RULES (CRITICAL - PREVENTS CRASHES):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DASHED LINES:
   ✗ NEVER use 'dash_length' parameter inside Line() - IT DOES NOT EXIST!
   ✗ NEVER use 'linestyle' parameter - IT DOES NOT EXIST!
   ✓ USE the 'DashedLine' class for dashed lines:
     Example: DashedLine(start=A, end=B, dashed_ratio=0.5)
   ✓ Or wrap in DashedVMobject: DashedVMobject(Line(A, B))

2. ARROWS:
   ✗ NEVER use 'Arrow(..., path_arc=...)' - will crash!
   ✓ USE 'CurvedArrow(start_point, end_point, angle=...)' for curved paths.
   ✗ NEVER use 'arrow_size' parameter - use 'tip_length' instead.

3. INVALID PARAMETERS THAT WILL CRASH (NEVER USE):
   ✗ dash_length - DOES NOT EXIST in Manim
   ✗ linestyle - DOES NOT EXIST in Manim (this is Matplotlib!)
   ✗ arrow_size - DOES NOT EXIST (use tip_length)
   ✗ dashes - DOES NOT EXIST
   ✗ linewidth - use 'stroke_width' instead

4. CORRECT EXAMPLES:
   ✓ Line(A, B, color=GRAY)  # Solid line
   ✓ DashedLine(A, B, dashed_ratio=0.5)  # Dashed line
   ✓ Arrow(A, B, tip_length=0.2)  # Arrow with custom tip
   ✓ CurvedArrow(A, B, angle=PI/4)  # Curved arrow

STRICT CODE TEMPLATE WITH NUCLEAR CLEARS (MUST FOLLOW THIS PATTERN):
```python
from manim import *

class PhysicsSolution(Scene):
    def construct(self):
        # ═══════════════ SECTION 1: TITLE (5 seconds) ═══════════════
        title = Text("Problem Title", font_size=26).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(2)
        
        # *** NUCLEAR CLEAR before Section 2 ***
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # ═══════════════ SECTION 2: SETUP/DIAGRAM (10 seconds) ═══════════════
        # Maximum 4 elements: section_title + diagram + 2 labels
        section_title = Text("Setup", font_size=22).to_edge(UP)
        diagram = Circle(radius=1, color=BLUE).move_to(ORIGIN)
        label = MathTex("r=1", font_size=18).next_to(diagram, DOWN)
        
        self.play(Write(section_title))
        self.play(Create(diagram), Write(label))
        self.wait(2)
        
        # *** NUCLEAR CLEAR before Section 3 ***
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # ═══════════════ SECTION 3: SOLUTION STEPS (15 seconds) ═══════════════
        # Show ONE step at a time using Transform (not Write!)
        step_title = Text("Solution", font_size=22).to_edge(UP)
        self.play(Write(step_title))
        
        # Step 1
        eq1 = MathTex(r"A = \pi r^2", font_size=24).move_to(ORIGIN)
        self.play(Write(eq1))
        self.wait(1)
        
        # Step 2 - TRANSFORM to replace, not Write to add!
        eq2 = MathTex(r"A = \pi (1)^2", font_size=24).move_to(ORIGIN)
        self.play(Transform(eq1, eq2))
        self.wait(1)
        
        # Step 3 - Continue transforming
        eq3 = MathTex(r"A = \pi", font_size=24).move_to(ORIGIN)
        self.play(Transform(eq1, eq3))
        self.wait(1)
        
        # *** NUCLEAR CLEAR before Final Answer ***
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # ═══════════════ SECTION 4: FINAL ANSWER (5 seconds) ═══════════════
        answer_title = Text("Answer", font_size=24, color=GREEN).to_edge(UP)
        final_answer = MathTex(r"A = \pi \approx 3.14", font_size=28, color=GREEN)
        final_answer.move_to(ORIGIN)
        box = SurroundingRectangle(final_answer, color=GREEN, buff=0.3)
        
        self.play(Write(answer_title))
        self.play(Write(final_answer), Create(box))
        self.wait(3)
```

CRITICAL LAYOUT EXAMPLES (FOLLOW THESE PATTERNS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example 1: PROPER SECTION FLOW (ALWAYS DO THIS)
```python
# Title section - ALONE
title = Text("Heat Transfer", font_size=26).to_edge(UP, buff=0.5)
self.play(Write(title))
self.wait(1)
self.play(FadeOut(title))  # ← Clear before content

# Content section
diagram = VGroup(plate1, plate2).to_edge(LEFT, buff=1.0)
self.play(Create(diagram))
# ... more animations ...
self.play(FadeOut(diagram))  # ← Clear before final answer

# Final answer section
answer = MathTex(r"Answer").move_to(ORIGIN)
self.play(Write(answer))
```

Example 2: LONG EQUATIONS (MUST SCALE)
```python
# For equations longer than ~30 chars, ALWAYS scale
long_eq = MathTex(r"3W_S = \sigma(T_P^4 - T_1^4) + \sigma(T_1^4 - T_2^4) + \sigma(T_2^4 - T_Q^4)")
long_eq.scale_to_fit_width(12)  # ← MANDATORY
long_eq.move_to(ORIGIN)
self.play(Write(long_eq))
```

Example 3: MULTIPLE EQUATIONS (VERTICAL GROUP)
```python
eq1 = MathTex(r"W_0 = \sigma(T_P^4 - T_Q^4)", font_size=22)
eq2 = MathTex(r"W_S = \sigma(T_P^4 - T_1^4)", font_size=22)
eq3 = MathTex(r"W_S = \sigma(T_1^4 - T_2^4)", font_size=22)
eq_group = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
eq_group.scale_to_fit_width(11)  # ← Scale group if needed
eq_group.move_to(ORIGIN)
self.play(Write(eq_group))
```

Example 4: DIAGRAM + EQUATIONS LAYOUT
```python
# Clear title first, then show diagram and equations
self.play(FadeOut(title))

# Left side - diagram (scaled to fit)
diagram = VGroup(plate1, plate2, arrows)
diagram.scale_to_fit_width(5)
diagram.to_edge(LEFT, buff=1.0)

# Right side - equations (scaled to fit)
equations = VGroup(eq1, eq2).arrange(DOWN, buff=0.4)
equations.scale_to_fit_width(5)
equations.to_edge(RIGHT, buff=1.0)

self.play(Create(diagram), Write(equations))
```

Example 5: AVOID OVERLAP (WRONG vs CORRECT)
```python
# WRONG - will overlap:
title = Text("Title").to_edge(UP)
eq = MathTex("E=mc^2").to_edge(UP)  # ← Same position as title!

# CORRECT - clear first:
title = Text("Title").to_edge(UP)
self.play(Write(title))
self.play(FadeOut(title))  # ← Clear title
eq = MathTex("E=mc^2").to_edge(UP)  # ← Now safe to use same position
self.play(Write(eq))
```

3D SCENE TEXT RULES (CRITICAL FOR READABLE TEXT IN 3D):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When using ThreeDScene, ThreeDAxes, or any 3D coordinates:

⚠️ PROBLEM: Text placed directly in 3D space becomes rotated and unreadable!

1. USE FIXED FRAME TEXT FOR ALL LABELS AND EQUATIONS:
   ✓ ALWAYS use self.add_fixed_in_frame_mobjects() for text in 3D scenes
   This keeps text as a 2D overlay that doesn't rotate with the camera.

2. 3D SCENE TEMPLATE (MUST FOLLOW):
```python
from manim import *

class Solution3D(ThreeDScene):
    def construct(self):
        # Setup camera angle
        self.set_camera_orientation(phi=75*DEGREES, theta=45*DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes()
        self.play(Create(axes))
        
        # CORRECT: Title as fixed 2D overlay
        title = Text("3D Geometry Problem", font_size=26)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)  # ← CRITICAL!
        self.play(Write(title))
        
        # Create 3D objects
        point = Dot3D(point=[1, 2, 3], color=RED)
        self.play(Create(point))
        
        # CORRECT: Point label as fixed overlay
        label = MathTex("P(1, 2, 3)", font_size=20)
        label.to_corner(UR)  # Position in 2D screen space
        self.add_fixed_in_frame_mobjects(label)  # ← CRITICAL!
        self.play(Write(label))
        
        # CORRECT: Equations as fixed overlay
        equation = MathTex(r"d = \\sqrt{{x^2 + y^2 + z^2}}", font_size=24)
        equation.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(equation)  # ← CRITICAL!
        self.play(Write(equation))
        
        # Camera rotation (text stays fixed!)
        self.move_camera(phi=60*DEGREES, theta=120*DEGREES, run_time=2)
        self.wait(2)
```

3. RULES FOR 3D TEXT:
   ✗ NEVER place Text/MathTex directly at 3D coordinates without fixed_in_frame
   ✗ NEVER use text.move_to([x, y, z]) for 3D points - it will rotate!
   ✓ ALWAYS call self.add_fixed_in_frame_mobjects(text) BEFORE animating
   ✓ Position fixed text using to_corner(), to_edge(), or 2D coordinates
   ✓ Use smaller font_size (18-22) for labels in 3D scenes

4. LABELING 3D POINTS (CORRECT APPROACH):
```python
# Create the 3D point
point_P = Dot3D(point=[2, 3, 1], color=BLUE)
self.play(Create(point_P))

# Create label as FIXED 2D text
label_P = MathTex("P(2, 3, 1)", font_size=18, color=BLUE)
label_P.to_corner(UR).shift(DOWN*0.5)  # 2D screen position
self.add_fixed_in_frame_mobjects(label_P)
self.play(Write(label_P))

# For multiple point labels, stack them:
labels = VGroup(
    MathTex("P(2, 3, 1)", font_size=18),
    MathTex("Q(1, 0, 2)", font_size=18),
    MathTex("R(0, 1, 3)", font_size=18)
).arrange(DOWN, buff=0.3).to_corner(UR)
self.add_fixed_in_frame_mobjects(labels)
self.play(Write(labels))
```

5. COMMON 3D MISTAKES TO AVOID:
```python
# WRONG - text will be tilted/unreadable:
label = Text("Point A").move_to([1, 2, 3])
self.play(Write(label))

# WRONG - text rotates with camera:
label = MathTex("P").next_to(point_3d, UP)
self.play(Write(label))

# CORRECT - text stays readable:
label = MathTex("P", font_size=20).to_corner(UR)
self.add_fixed_in_frame_mobjects(label)
self.play(Write(label))
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
- CRITICAL: When using Greek letters (\\lambda, \\alpha, etc.) in Title or Text, ALWAYS use raw strings:
  ✓ Title(r"Solve for \\lambda")  # CORRECT - raw string with r prefix
  ✗ Title("Solve for \\lambda")   # WRONG - will cause LaTeX error!
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
        
        # CRITICAL FIX: Global replacement of BROWN (catches all usages including interpolate_color)
        if 'BROWN' in code:
            logger.warning("Found BROWN in code - replacing with hex color globally")
            code = code.replace('BROWN', '"#8B4513"')
        
        # CRITICAL FIX: Remove invalid 'dash_length' parameter (Gemini hallucinates this from Matplotlib)
        # Line() does not accept dash_length - use DashedLine instead
        if 'dash_length' in code:
            logger.warning("Found invalid dash_length parameter - removing it")
            code = re.sub(r',\s*dash_length\s*=\s*[\d\.]+', '', code)
            code = re.sub(r'dash_length\s*=\s*[\d\.]+\s*,?', '', code)
        
        # CRITICAL FIX: Remove invalid 'dashed_ratio' if used incorrectly on Line
        # Only DashedLine accepts dashed_ratio
        if 'Line(' in code and 'dashed_ratio' in code:
            logger.warning("Found invalid dashed_ratio on Line - removing it")
            code = re.sub(r',\s*dashed_ratio\s*=\s*[\d\.]+', '', code)
        
        # CRITICAL FIX: Remove 'arrow_size' parameter (use tip_length instead)
        if 'arrow_size' in code:
            logger.warning("Found invalid arrow_size parameter - removing it")
            code = re.sub(r',\s*arrow_size\s*=\s*[\d\.]+', '', code)
            code = re.sub(r'arrow_size\s*=\s*[\d\.]+\s*,?', '', code)
        
        # CRITICAL FIX: Remove 'dashes' parameter (doesn't exist in Manim)
        if 'dashes' in code:
            logger.warning("Found invalid dashes parameter - removing it")
            code = re.sub(r',\s*dashes\s*=\s*\[[^\]]*\]', '', code)
            code = re.sub(r'dashes\s*=\s*\[[^\]]*\]\s*,?', '', code)
        
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
        
        # Fix: 3D Scene Text - Add fixed_in_frame_mobjects for text in ThreeDScene
        # This prevents text from rotating with the camera and becoming unreadable
        if 'ThreeDScene' in code or 'ThreeDAxes' in code or 'Dot3D' in code:
            logger.warning("3D Scene detected - ensuring text is fixed in frame")
            
            # Find all Text and MathTex variable assignments
            text_vars = re.findall(r'(\w+)\s*=\s*(?:Text|MathTex)\s*\(', code)
            
            # For each text variable, check if it's added with fixed_in_frame
            for var in text_vars:
                # Check if this variable already has add_fixed_in_frame_mobjects
                if f'add_fixed_in_frame_mobjects({var})' not in code and f'add_fixed_in_frame_mobjects( {var})' not in code:
                    # Find where this text is animated (self.play(Write(var)))
                    write_pattern = rf'(self\.play\s*\(\s*Write\s*\(\s*{var}\s*\))'
                    if re.search(write_pattern, code):
                        # Insert add_fixed_in_frame_mobjects before the self.play
                        code = re.sub(
                            write_pattern,
                            rf'self.add_fixed_in_frame_mobjects({var})\n        \1',
                            code
                        )
                        logger.info(f"Added fixed_in_frame for text variable: {var}")
                    
                    # Also check for FadeIn
                    fadein_pattern = rf'(self\.play\s*\(\s*FadeIn\s*\(\s*{var}\s*\))'
                    if re.search(fadein_pattern, code):
                        code = re.sub(
                            fadein_pattern,
                            rf'self.add_fixed_in_frame_mobjects({var})\n        \1',
                            code
                        )
                        logger.info(f"Added fixed_in_frame for text variable: {var}")
        
        # Fix: Replace deprecated/non-existent Manim names
        # ShowCreation is deprecated → use Create
        if 'ShowCreation' in code:
            logger.warning("Replacing deprecated ShowCreation with Create")
            code = code.replace('ShowCreation', 'Create')
        
        # Fix: Title/Text with LaTeX content (like \lambda, \alpha, etc.)
        # Title("Solve for \lambda") fails because \l is invalid escape
        # Convert to raw string or use proper escaping
        latex_commands = [r'\\lambda', r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\theta', 
                         r'\\sigma', r'\\pi', r'\\omega', r'\\mu', r'\\epsilon', r'\\phi', r'\\psi',
                         r'\\frac', r'\\sqrt', r'\\sum', r'\\int', r'\\infty', r'\\partial']
        
        # Find Title or Text with unescaped LaTeX
        for latex_cmd in latex_commands:
            # Pattern: Title("...\\lambda...") where \\ is not doubled (raw string issue)
            # We need to find Title("...\lambda...") and convert to Title(r"...\lambda...")
            simple_pattern = latex_cmd.replace('\\\\', '\\')  # Convert \\lambda to \lambda for matching
            
            # Check for Title with this LaTeX command (non-raw string)
            if f'Title("{simple_pattern[1:]}' in code or f"Title('{simple_pattern[1:]}" in code:
                logger.warning(f"Found Title with LaTeX {simple_pattern} - converting to raw string")
                # Convert Title("...\lambda...") to Title(r"...\lambda...")
                code = re.sub(
                    rf'Title\("([^"]*{re.escape(simple_pattern)}[^"]*)"\)',
                    lambda m: f'Title(r"{m.group(1)}")',
                    code
                )
                code = re.sub(
                    rf"Title\('([^']*{re.escape(simple_pattern)}[^']*)'\)",
                    lambda m: f"Title(r'{m.group(1)}')",
                    code
                )
            
            # Same for Text
            if f'Text("{simple_pattern[1:]}' in code or f"Text('{simple_pattern[1:]}" in code:
                logger.warning(f"Found Text with LaTeX {simple_pattern} - converting to raw string")
                code = re.sub(
                    rf'Text\("([^"]*{re.escape(simple_pattern)}[^"]*)"\)',
                    lambda m: f'Text(r"{m.group(1)}")',
                    code
                )
                code = re.sub(
                    rf"Text\('([^']*{re.escape(simple_pattern)}[^']*)'\)",
                    lambda m: f"Text(r'{m.group(1)}')",
                    code
                )
        
        # Also fix the common case: Title("...\l...") which is an invalid escape
        # This happens when \lambda is written but Python sees \l as invalid
        if '\\l' in code and 'Title(' in code:
            logger.warning("Found potential invalid escape in Title - adding raw string prefix")
            # Find Title with potential escape issues and make them raw
            code = re.sub(
                r'Title\("([^"]*\\[a-zA-Z][^"]*)"\)',
                lambda m: f'Title(r"{m.group(1)}")',
                code
            )
        
        if '\\l' in code and 'Text(' in code:
            logger.warning("Found potential invalid escape in Text - adding raw string prefix")
            code = re.sub(
                r'Text\("([^"]*\\[a-zA-Z][^"]*)"\)',
                lambda m: f'Text(r"{m.group(1)}")',
                code
            )
        
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
        non_existent_methods = ['get_arc_center', 'set_arc_center', 'get_tangent_vector', 'get_unit_vector', 'midpoint', 'point_at_angle', 'get_position']
        for method in non_existent_methods:
            if method in code:
                logger.warning(f"Removing/replacing non-existent method: {method}")
                if method == 'midpoint':
                    # Replace .midpoint() with .next_to() which is the correct method
                    code = re.sub(r'\.midpoint\(', '.next_to(', code)
                elif method == 'point_at_angle':
                    # Replace with point_from_proportion
                    code = re.sub(r'\.point_at_angle\(([^)]+)\)', r'.point_from_proportion(\1/(2*PI))', code)
                elif method == 'get_position':
                    # Replace .get_position() with .get_center() - the correct Manim method
                    logger.warning("Replacing .get_position() with .get_center()")
                    code = re.sub(r'\.get_position\(\)', '.get_center()', code)
                else:
                    code = re.sub(rf'\.{method}\([^)]*\)', '', code)
        
        # CRITICAL FIX: Remove invalid .aligned_edge() method chain
        # Manim doesn't have .aligned_edge() method - it's a parameter for .next_to() or .arrange()
        # Pattern: .move_to(...).aligned_edge(LEFT) → .move_to(...)
        if '.aligned_edge(' in code:
            logger.warning("Removing invalid .aligned_edge() method chain")
            code = re.sub(r'\.aligned_edge\s*\(\s*\w+\s*\)', '', code)
        
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
        
        # Fix 5d: Auto-scale long MathTex equations to prevent overflow
        # Find MathTex with long content (>50 chars in the LaTeX string)
        def check_and_scale_long_mathtex(match):
            full_line = match.group(0)
            latex_content = match.group(1) if match.lastindex >= 1 else ""
            
            # If LaTeX content is long and no scale_to_fit_width already present
            if len(latex_content) > 50 and 'scale_to_fit_width' not in full_line:
                # Check if next line already has scale
                return full_line  # Return as-is, we'll add a post-processing step
            return full_line
        
        # Add scale_to_fit_width for equations that are positioned at ORIGIN or center
        # and have long content
        lines = code.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            
            # Check if this line creates a MathTex with long content
            mathtex_match = re.search(r'(\w+)\s*=\s*MathTex\(r?["\'](.{50,})["\']', line)
            if mathtex_match:
                var_name = mathtex_match.group(1)
                # Check if next lines already have scale_to_fit_width for this var
                has_scale = False
                for j in range(i+1, min(i+5, len(lines))):
                    if f'{var_name}.scale_to_fit_width' in lines[j] or f'{var_name}.scale(' in lines[j]:
                        has_scale = True
                        break
                
                # If no scale and the var is used with move_to(ORIGIN) or to_edge
                if not has_scale:
                    for j in range(i+1, min(i+5, len(lines))):
                        if f'{var_name}.move_to' in lines[j] or f'{var_name}.to_edge' in lines[j]:
                            # Insert scale_to_fit_width before the positioning
                            indent = len(lines[j]) - len(lines[j].lstrip())
                            scale_line = ' ' * indent + f'{var_name}.scale_to_fit_width(11)'
                            new_lines.append(scale_line)
                            logger.warning(f"Auto-added scale_to_fit_width for long equation: {var_name}")
                            break
            
            i += 1
        
        code = '\n'.join(new_lines)
        
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
        
        # Fix 8: OVERLAP PREVENTION - Inject nuclear clears if too many Write() without FadeOut
        # Count Write/FadeIn vs FadeOut calls
        write_count = len(re.findall(r'self\.play\s*\(\s*(?:Write|FadeIn|Create)\s*\(', code))
        fadeout_count = len(re.findall(r'(?:FadeOut|self\.play\s*\(\s*\*\s*\[)', code))
        
        if write_count > fadeout_count + 5:
            logger.warning(f"Detected potential overlap: {write_count} writes vs {fadeout_count} fadeouts")
            
            # Find long sections without any FadeOut and inject nuclear clears
            lines = code.split('\n')
            new_lines = []
            write_streak = 0
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # Count writes in this line
                if re.search(r'self\.play\s*\(\s*(?:Write|FadeIn|Create)\s*\(', line):
                    write_streak += 1
                
                # Reset streak on FadeOut
                if 'FadeOut' in line or 'FadeOut(mob)' in line:
                    write_streak = 0
                
                # If we've had 5+ writes without a fadeout, inject a nuclear clear
                if write_streak >= 5:
                    # Check if next line is already a FadeOut
                    next_line = lines[i+1] if i+1 < len(lines) else ""
                    if 'FadeOut' not in next_line and 'self.wait' in line:
                        indent = len(line) - len(line.lstrip())
                        nuclear_clear = ' ' * indent + "self.play(*[FadeOut(mob) for mob in self.mobjects])  # Auto-injected clear"
                        new_lines.append(nuclear_clear)
                        logger.warning(f"Injected nuclear clear after line {i+1}")
                        write_streak = 0
            
            code = '\n'.join(new_lines)
        
        # Fix 9: Detect multiple to_edge(UP) without intermediate FadeOut
        lines = code.split('\n')
        to_edge_up_lines = []
        last_fadeout_line = -1
        
        for i, line in enumerate(lines):
            if 'FadeOut' in line:
                last_fadeout_line = i
            if '.to_edge(UP' in line or 'to_edge(UP' in line:
                # Check if there's been a FadeOut since last to_edge(UP)
                if to_edge_up_lines and to_edge_up_lines[-1] > last_fadeout_line:
                    # Potential overlap - log warning
                    logger.warning(f"Potential overlap: Multiple to_edge(UP) at lines {to_edge_up_lines[-1]+1} and {i+1} without FadeOut")
                to_edge_up_lines.append(i)
        
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
