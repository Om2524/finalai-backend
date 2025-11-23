# Render.com Deployment Guide

This backend is configured to deploy on Render.com using Docker.

## Prerequisites
- GitHub account
- Render.com account (free tier available)

## Environment Variables to Set on Render

Add these in Render Dashboard → Environment Variables:

```
GEMINI_API_KEY = AIzaSyB1ajPDIk8ujdQpfLjU38JoASbEvGGYhLw
VIDEO_STORAGE_PATH = /app/videos
TEMP_CODE_PATH = /app/temp
MAX_IMAGE_SIZE_MB = 10
MANIM_QUALITY = ql
PORT = 8000
```

## Deployment Steps

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial backend setup for Render"
   git remote add origin https://github.com/YOUR_USERNAME/finalai-backend.git
   git push -u origin main
   ```

2. **Create Web Service on Render:**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Runtime: **Docker** (auto-detected)
   - Instance Type: **Free** or **Starter** ($7/mo recommended)
   - Add environment variables listed above
   - Click "Create Web Service"

3. **Wait for Build:**
   - First build takes 8-12 minutes
   - Watch logs for any errors
   - Once live, copy your service URL

4. **Update Frontend:**
   - Add to Lovable environment variables:
     ```
     VITE_ASK_DOUBT_API_URL = https://your-render-url.onrender.com/api
     ```

## Testing

Test your deployment:

```bash
# Health check
curl https://your-render-url.onrender.com/api/health

# Expected: {"status":"healthy","service":"ask-doubt-backend"}
```

## Notes

- Docker image is based on `manimcommunity/manim:v0.18.0`
- Includes FFmpeg, LaTeX, and all Manim dependencies
- Free tier spins down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- Videos stored in ephemeral `/app/videos` directory

## Troubleshooting

**Build fails:**
- Check Render logs for errors
- Ensure all files are committed to GitHub

**CORS errors:**
- Verify `allow_origins=["*"]` in `app/main.py`

**Out of memory:**
- Upgrade to Starter plan ($7/mo) for 1GB RAM
- Free tier (512MB) may crash during rendering

**504 Timeout:**
- Render Free tier may be too slow for heavy rendering
- Consider Starter plan for better performance
