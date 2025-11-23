#!/bin/bash

# ============================================
# Verify Render Deployment Setup
# ============================================

echo "🔍 Verifying Render deployment setup..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

checks_passed=0
checks_failed=0

# Check 1: Dockerfile exists
echo -n "Checking Dockerfile... "
if [ -f "Dockerfile" ]; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗${NC}"
    checks_failed=$((checks_failed + 1))
fi

# Check 2: requirements.txt doesn't contain manim
echo -n "Checking requirements.txt (should NOT contain 'manim')... "
if ! grep -q "^manim$" requirements.txt; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗ (manim found - should be removed)${NC}"
    checks_failed=$((checks_failed + 1))
fi

# Check 3: CORS allows all origins
echo -n "Checking CORS configuration... "
if grep -q 'allow_origins=\["\*"\]' app/main.py; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${YELLOW}⚠ (should be allow_origins=[\"*\"])${NC}"
    checks_failed=$((checks_failed + 1))
fi

# Check 4: Git initialized
echo -n "Checking git repository... "
if [ -d ".git" ]; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗${NC}"
    checks_failed=$((checks_failed + 1))
fi

# Check 5: Git has commits
echo -n "Checking git commits... "
if git log --oneline -1 &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗${NC}"
    checks_failed=$((checks_failed + 1))
fi

# Check 6: All required files present
echo -n "Checking required files... "
required_files=("app/main.py" "app/config.py" "app/api/routes.py" "app/services/gemini_client.py" "app/services/manim_renderer.py")
all_present=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        all_present=false
        break
    fi
done
if $all_present; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗${NC}"
    checks_failed=$((checks_failed + 1))
fi

# Check 7: .gitignore exists
echo -n "Checking .gitignore... "
if [ -f ".gitignore" ]; then
    echo -e "${GREEN}✓${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗${NC}"
    checks_failed=$((checks_failed + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VERIFICATION RESULTS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "✓ Passed: ${GREEN}$checks_passed${NC}"
echo -e "✗ Failed: ${RED}$checks_failed${NC}"
echo ""

if [ $checks_failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! Ready to deploy.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./deploy-to-github.sh"
    echo "  2. Go to: https://dashboard.render.com"
    echo "  3. Follow the prompts to deploy"
    echo ""
else
    echo -e "${RED}⚠️  Some checks failed. Please fix the issues above.${NC}"
    echo ""
fi

# Show file structure
echo "📁 Project Structure:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tree -L 2 -I 'venv|videos|temp|__pycache__|*.pyc' . 2>/dev/null || find . -maxdepth 2 -not -path '*/\.*' -not -path '*/venv/*' -not -path '*/videos/*' -not -path '*/temp/*' | head -20
echo ""
