# Pre-Deployment Checklist

Before deploying, make sure you have these files ready:

## 鉁?Files to Include in Repository

### Backend Files
- [x] `backend/__init__.py`
- [x] `backend/app.py`
- [x] `backend/detector.py`
- [x] `backend/pipeline.py`
- [x] `backend/models/best.pt` (YOLO model - **REQUIRED**)

### Frontend Files
- [x] `frontend/index.html`

### Configuration Files
- [x] `requirements-deploy.txt`
- [x] `render.yaml`
- [x] `Procfile`
- [x] `runtime.txt`
- [x] `.renderignore`
- [x] `.gitignore`
- [x] `netlify.toml` (for Netlify)
- [x] `vercel.json` (for Vercel)

## 鉂?Files to EXCLUDE (Don't Upload)

- [ ] `data/` folder
- [ ] `dataset/` folder
- [ ] `runs/` folder
- [ ] `prev version/` folder
- [ ] `backend/outputs/temp_crops/` folder
- [ ] `backend/outputs/*.pdf` files
- [ ] `backend/models/siamese_cnn.pt` (optional if duplicate filtering is disabled)
- [ ] `backend/training/` folder
- [ ] `backend/tools/` folder
- [ ] `install_dependencies.*` files
- [ ] `quick_install.*` files
- [ ] `start_server.py`
- [ ] `README_SERVER.md`
- [ ] `QUICK_START.md`
- [ ] `guitar_tabs.pdf`
- [ ] `yolo11n.pt` (if not needed)
- [ ] `*.iml` files

## 馃摑 Before Deploying

1. **Update API URL in frontend/index.html**
   - Line 383: Replace `'https://guitar-tab-backend.onrender.com'` with your actual Render backend URL after deployment

2. **Test Locally First**
   - Make sure everything works locally
   - Test with a sample video

3. **Check File Sizes**
   - `backend/models/best.pt` might be large (10-50MB)
   - Make sure it's included in repository
   - GitHub allows files up to 100MB

4. **Verify Dependencies**
   - Check `requirements-deploy.txt` has all needed packages
   - Test installation: `pip install -r requirements-deploy.txt`

## 馃殌 Quick Start Deployment

1. Create GitHub repository
2. Upload only checked files above
3. Deploy backend to Render.com
4. Deploy frontend to Netlify/Vercel
5. Update API URL in frontend
6. Test!

See `DEPLOYMENT.md` for detailed instructions.


