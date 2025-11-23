#!/bin/bash

# ============================================
# Deploy Backend to GitHub for Render
# ============================================

echo "🚀 Deploying Ask Doubt Backend to GitHub..."
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found!"
    echo "📦 Install it with: brew install gh"
    echo ""
    echo "Alternatively, create repo manually at:"
    echo "   https://github.com/new"
    echo "   Repository name: finalai-backend"
    echo "   Visibility: Public"
    echo ""
    echo "Then run:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/finalai-backend.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    exit 1
fi

# Check if user is logged in
if ! gh auth status &> /dev/null; then
    echo "🔐 Please login to GitHub first..."
    gh auth login
fi

# Get GitHub username
GITHUB_USER=$(gh api user --jq .login)
echo "👤 GitHub user: $GITHUB_USER"
echo ""

# Check if repo already exists
REPO_NAME="finalai-backend"
if gh repo view $GITHUB_USER/$REPO_NAME &> /dev/null; then
    echo "⚠️  Repository '$REPO_NAME' already exists!"
    echo ""
    read -p "Do you want to push to existing repo? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Deployment cancelled."
        exit 1
    fi
    
    # Add remote if not exists
    if ! git remote get-url origin &> /dev/null; then
        git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git
    fi
else
    # Create new repository
    echo "📦 Creating GitHub repository '$REPO_NAME'..."
    gh repo create $REPO_NAME \
        --public \
        --source=. \
        --remote=origin \
        --description="FastAPI + Manim backend for Ask a Doubt feature - AI-powered video solutions"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create repository!"
        exit 1
    fi
    echo "✅ Repository created!"
fi

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🔗 Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Go to: https://dashboard.render.com"
    echo "2. Click: New + → Web Service"
    echo "3. Connect: $GITHUB_USER/$REPO_NAME"
    echo "4. Runtime: Docker (auto-detected)"
    echo "5. Instance: Free or Starter (\$7/mo recommended)"
    echo ""
    echo "6. Add Environment Variables:"
    echo "   ┌─────────────────────────────────────────────────┐"
    echo "   │ GEMINI_API_KEY = AIzaSyB1ajPDIk8ujdQpfL...      │"
    echo "   │ VIDEO_STORAGE_PATH = /app/videos                │"
    echo "   │ TEMP_CODE_PATH = /app/temp                      │"
    echo "   │ MAX_IMAGE_SIZE_MB = 10                          │"
    echo "   │ MANIM_QUALITY = ql                              │"
    echo "   │ PORT = 8000                                     │"
    echo "   └─────────────────────────────────────────────────┘"
    echo ""
    echo "7. Click: Create Web Service"
    echo "8. Wait: 8-12 minutes for Docker build"
    echo "9. Copy: Your Render URL (e.g., https://finalai-backend.onrender.com)"
    echo ""
    echo "10. Update Lovable:"
    echo "    Add environment variable in Lovable settings:"
    echo "    VITE_ASK_DOUBT_API_URL = https://your-render-url.onrender.com/api"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📖 Full guide: See RENDER_DEPLOYMENT.md"
    echo ""
else
    echo "❌ Failed to push to GitHub!"
    echo "Please check your git configuration and try again."
    exit 1
fi
