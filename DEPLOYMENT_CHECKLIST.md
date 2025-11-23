# ✅ Deployment Checklist

Use this checklist to track your deployment progress.

## Pre-Deployment Setup

- [x] Dockerfile created
- [x] requirements.txt updated (manim removed)
- [x] CORS configured (allow_origins=["*"])
- [x] Git repository initialized
- [x] First commit created
- [x] Deployment scripts created
- [x] Documentation prepared

**Status: COMPLETE ✅**

---

## Step 1: Push to GitHub

- [ ] GitHub CLI installed (`brew install gh`)
- [ ] Logged into GitHub (`gh auth login`)
- [ ] Run deployment script: `./deploy-to-github.sh`
- [ ] Repository created: `finalai-backend`
- [ ] Code pushed successfully
- [ ] Repository URL copied

**GitHub Repository URL:**
```
https://github.com/YOUR_USERNAME/finalai-backend
```

---

## Step 2: Deploy on Render

### 2.1 Create Account
- [ ] Go to https://dashboard.render.com
- [ ] Sign up / Log in (use GitHub for easy connection)

### 2.2 Create Web Service
- [ ] Click "New +" → "Web Service"
- [ ] Click "Connect a repository"
- [ ] Find and select `finalai-backend`
- [ ] Click "Connect"

### 2.3 Configure Service
- [ ] Name: `finalai-backend`
- [ ] Region: Oregon (US West) or closest
- [ ] Branch: `main`
- [ ] Runtime: Docker (auto-detected) ✅
- [ ] Instance Type: **Starter** ($7/mo) selected

### 2.4 Environment Variables

Add these variables (copy/paste):

- [ ] `GEMINI_API_KEY` = `AIzaSyB1ajPDIk8ujdQpfLjU38JoASbEvGGYhLw`
- [ ] `VIDEO_STORAGE_PATH` = `/app/videos`
- [ ] `TEMP_CODE_PATH` = `/app/temp`
- [ ] `MAX_IMAGE_SIZE_MB` = `10`
- [ ] `MANIM_QUALITY` = `ql`
- [ ] `PORT` = `8000`

### 2.5 Deploy
- [ ] Click "Create Web Service"
- [ ] Build started (watch logs)
- [ ] Build completed (8-12 minutes)
- [ ] Service shows "Live" with green dot
- [ ] Service URL copied

**Render Service URL:**
```
https://_____________________.onrender.com
```

---

## Step 3: Test Backend

Run these tests after deployment:

### Health Check
```bash
curl https://YOUR-RENDER-URL.onrender.com/api/health
```

Expected Response:
```json
{"status":"healthy","service":"ask-doubt-backend"}
```

- [ ] Health check returns 200 OK
- [ ] Response contains `"status":"healthy"`

### Root Endpoint
```bash
curl https://YOUR-RENDER-URL.onrender.com/
```

Expected Response:
```json
{
  "service":"Ask Doubt Backend",
  "status":"running",
  "api_docs":"/docs"
}
```

- [ ] Root endpoint returns 200 OK
- [ ] Response shows service info

### API Documentation
- [ ] Visit: `https://YOUR-RENDER-URL.onrender.com/docs`
- [ ] Swagger UI loads correctly
- [ ] All endpoints visible

---

## Step 4: Update Lovable

### 4.1 Access Lovable Dashboard
- [ ] Go to your Lovable project
- [ ] Navigate to Settings
- [ ] Find Environment Variables section

### 4.2 Add Environment Variable
- [ ] Click "Add Environment Variable"
- [ ] Key: `VITE_ASK_DOUBT_API_URL`
- [ ] Value: `https://YOUR-RENDER-URL.onrender.com/api`
- [ ] Save changes

### 4.3 Redeploy
- [ ] Click "Deploy" or "Publish"
- [ ] Wait for deployment to complete
- [ ] Note new deployment URL

**Lovable Site URL:**
```
https://_____________________.lovable.app
```

---

## Step 5: End-to-End Testing

Test the complete feature on your Lovable site:

### 5.1 Access Feature
- [ ] Visit your Lovable site
- [ ] "Ask a Doubt" button appears in top-right
- [ ] Button is clickable

### 5.2 Upload Image
- [ ] Click "Ask a Doubt" button
- [ ] Dialog opens
- [ ] Can drag & drop image OR click to upload
- [ ] Image preview shows after upload

### 5.3 Enter Question
- [ ] Can type in text box
- [ ] Microphone icon visible
- [ ] Voice recording works (if tested)
- [ ] Question text updates

### 5.4 Generate Solution
- [ ] Click "Generate Solution" button
- [ ] Loading spinner appears
- [ ] Shows "Generating your solution..." message
- [ ] No errors in browser console (F12)

### 5.5 Video Playback
- [ ] Video generates successfully (60-90 seconds)
- [ ] Video player appears
- [ ] Video plays automatically
- [ ] Video controls work (pause, play, seek)
- [ ] Can fullscreen video
- [ ] "Download Video" button works

### 5.6 Error Handling
- [ ] Try without image → shows error
- [ ] Try without question → shows error
- [ ] "Ask Another" button resets form

---

## Browser Console Check

Open DevTools (F12) → Console tab:

**Should SEE:**
- [ ] `POST https://YOUR-RENDER-URL.onrender.com/api/ask-doubt 200 OK`
- [ ] No CORS errors
- [ ] No 404 errors

**Should NOT see:**
- [ ] `ERR_CONNECTION_REFUSED`
- [ ] `504 Timeout`
- [ ] CORS policy errors
- [ ] `localhost:8000` requests

---

## Performance Check

- [ ] First request after deploy: < 90 seconds
- [ ] Subsequent requests: 60-90 seconds
- [ ] Video file size: < 5MB
- [ ] No memory errors on Render logs

---

## Optional: Production Optimizations

After successful deployment, consider:

- [ ] Add custom domain to Render
- [ ] Enable auto-deploy on git push
- [ ] Set up monitoring (Render dashboard)
- [ ] Configure video CDN (Cloudflare/Cloudinary)
- [ ] Add request logging
- [ ] Set up error alerts
- [ ] Implement video caching
- [ ] Add rate limiting
- [ ] Secure API with authentication

---

## Troubleshooting Reference

If issues occur, check:

1. **Render Logs:**
   - Dashboard → Your Service → Logs tab
   - Look for errors during build/runtime

2. **Browser Console:**
   - F12 → Console tab
   - Check for CORS errors, 404s, network issues

3. **Environment Variables:**
   - Verify all 6 variables are set correctly
   - Check for typos in URLs

4. **Git Repository:**
   - Ensure latest code is pushed
   - Check Dockerfile is present

5. **Lovable Environment:**
   - Verify `VITE_ASK_DOUBT_API_URL` is correct
   - Ensure redeployment happened

---

## Success Criteria

Deployment is successful when ALL these are true:

- ✅ Backend responds at Render URL
- ✅ Health check returns `{"status":"healthy"}`
- ✅ Lovable site loads without errors
- ✅ "Ask a Doubt" button appears
- ✅ Can upload image
- ✅ Can enter question
- ✅ Video generates in 60-90 seconds
- ✅ Video plays in browser
- ✅ No CORS errors
- ✅ No console errors
- ✅ Download button works

---

## Completion

Date Completed: ________________

Total Time: ________________

Notes:
_______________________________________________________
_______________________________________________________
_______________________________________________________

---

**🎉 Congratulations! Your "Ask a Doubt" feature is live!**
