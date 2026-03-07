# Deployment Guide - Guitar Tab Extractor

This guide will help you deploy the Guitar Tab Extractor to free cloud hosting services.

## Overview

- **Backend**: Deploy to Render.com (free tier)
- **Frontend**: Deploy to Netlify or Vercel (free tier)

## Prerequisites

1. GitHub account (free)
2. Render.com account (free)
3. Netlify or Vercel account (free)

## Step 1: Prepare Your Repository

### Files to Upload

**Essential files only:**
```
guitar-tab-app/
鈹溾攢鈹€ backend/
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ app.py
鈹?  鈹溾攢鈹€ detector.py
鈹?  鈹溾攢鈹€ pipeline.py
鈹?  鈹斺攢鈹€ models/
鈹?      鈹斺攢鈹€ best.pt          # YOLO model (required)
鈹溾攢鈹€ frontend/
鈹?  鈹斺攢鈹€ index.html
鈹溾攢鈹€ requirements-deploy.txt  # Minimal dependencies
鈹溾攢鈹€ render.yaml              # Render deployment config
鈹溾攢鈹€ Procfile                 # Process file for Render
鈹溾攢鈹€ runtime.txt              # Python version
鈹斺攢鈹€ .renderignore            # Files to exclude
```

**Files to EXCLUDE (don't upload):**
- `data/` - Training data
- `dataset/` - Training dataset
- `runs/` - Training runs
- `prev version/` - Old code
- `backend/outputs/temp_crops/` - Temporary files
- `backend/models/siamese_cnn.pt` - Optional if duplicate filtering is disabled
- `backend/training/` - Training scripts
- `backend/tools/` - Utility scripts
- Installation scripts
- Documentation files (optional)

## Step 2: Deploy Backend to Render.com

### 2.1 Create GitHub Repository

1. Create a new repository on GitHub
2. Upload only the essential files listed above
3. Make sure `.renderignore` is included

### 2.2 Deploy to Render

1. Go to [Render.com](https://render.com) and sign up/login
2. Click "New +" 鈫?"Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `guitar-tab-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

5. Add Environment Variables (optional):
   - `FRONTEND_URL`: Your frontend URL (e.g., `https://your-app.netlify.app`)
   - `PYTHON_VERSION`: `3.9.18`

6. Click "Create Web Service"
7. Wait for deployment (first time takes ~5-10 minutes)
8. Copy your backend URL (e.g., `https://guitar-tab-backend.onrender.com`)

### 2.3 Update Frontend API URL

1. Open `frontend/index.html`
2. Find the line with `return 'https://guitar-tab-backend.onrender.com';`
3. Replace with your actual Render backend URL

## Step 3: Deploy Frontend to Netlify

### Option A: Netlify (Recommended)

1. Go to [Netlify](https://www.netlify.com) and sign up/login
2. Click "Add new site" 鈫?"Import an existing project"
3. Connect to GitHub and select your `guitar-tab-website` repository
4. **IMPORTANT**: Configure build settings:
   - **Base directory**: Leave empty (root)
   - **Publish directory**: `frontend`
   - **Build command**: Leave empty (or use: `echo "No build needed"`)
5. **CRITICAL - Disable Python Detection**:
   - Go to **Site settings** 鈫?**Build & deploy** 鈫?**Environment**
   - Look for any `PYTHON_VERSION` or `RUNTIME_VERSION` variables
   - **Delete them** if they exist
   - Add a new variable: `PYTHON_VERSION` = `""` (empty string) to explicitly disable Python
6. Click "Deploy site"
7. Copy your frontend URL (e.g., `https://your-app.netlify.app`)

**If you still get Python errors:**
1. Go to **Site settings** 鈫?**Build & deploy** 鈫?**Build settings**
2. Under "Build command", make sure it's empty or just: `echo "Static site"`
3. Under "Environment variables", ensure `PYTHON_VERSION` is set to empty string `""`
4. Save and trigger a new deploy

### Option B: Vercel

1. Go to [Vercel](https://vercel.com) and sign up/login
2. Click "Add New Project"
3. Import your GitHub repository
4. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Other
5. Click "Deploy"
6. Copy your frontend URL

## Step 4: Update CORS Settings

1. Go back to Render.com dashboard
2. Open your backend service
3. Go to "Environment" tab
4. Add environment variable:
   - **Key**: `FRONTEND_URL`
   - **Value**: Your frontend URL (e.g., `https://your-app.netlify.app`)
5. Save and redeploy

## Step 5: Test Your Deployment

1. Visit your frontend URL
2. Upload a test video
3. Verify screenshots are extracted
4. Select screenshots and create PDF
5. Download should work

## Troubleshooting

### Backend Issues

**"Module not found" errors:**
- Check `requirements-deploy.txt` includes all dependencies
- Verify build logs in Render dashboard

**"Port not found" errors:**
- Make sure start command uses `$PORT` variable
- Render automatically assigns port

**CORS errors:**
- Set `FRONTEND_URL` environment variable in Render
- Update frontend `API_URL` to match backend URL

### Frontend Issues

**"Network error":**
- Check backend URL in `frontend/index.html`
- Verify backend is running (check Render dashboard)
- Check browser console for CORS errors

**API not responding:**
- Verify backend URL is correct
- Check Render service is "Live" (not sleeping)
- Free tier services sleep after 15 minutes of inactivity

## Free Tier Limitations

### Render.com
- Services sleep after 15 minutes of inactivity
- First request after sleep takes ~30 seconds (cold start)
- 750 hours/month free (enough for personal use)

### Netlify/Vercel
- 100GB bandwidth/month
- Unlimited requests
- No sleep time

## Cost

**Total: $0/month** (completely free!)

## Alternative: Railway.app

If Render doesn't work, try Railway:
1. Sign up at [Railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy with same settings
4. Free tier: $5 credit/month

## Maintenance

- Monitor Render dashboard for errors
- Check logs if issues occur
- Update dependencies periodically
- Keep model file (`best.pt`) in repository

## Support

If you encounter issues:
1. Check Render/Netlify logs
2. Verify all files are uploaded correctly
3. Ensure API URLs match between frontend and backend
4. Check CORS settings


