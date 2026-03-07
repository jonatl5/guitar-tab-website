# Guitar Tab Extractor

Extract and compose guitar tabs from video screenshots. Upload a video, select which screenshots to keep, and generate a PDF of your guitar tabs.

## Features

- 🎸 Extract guitar tab screenshots from videos
- ⏱️ Time-based screenshot extraction (every 2 seconds)
- ✅ Manual selection of screenshots to include
- 📄 Generate PDF from selected screenshots
- 🎨 Modern, tech-focused UI with guitar-themed design

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements-deploy.txt
```

Or use the quick install script:
- Windows: `quick_install.bat`
- Linux/Mac: `pip install uvicorn[standard] fastapi python-multipart opencv-python numpy pillow ultralytics`

### 2. Start Backend Server

```bash
python start_server.py
```

Or manually:
```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 3. Open Frontend

Open `frontend/index.html` in your web browser.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying to free cloud hosting (Render.com + Netlify/Vercel).

## Project Structure

```
guitar-tab-app/
├── backend/           # FastAPI backend
│   ├── app.py        # Main API endpoints
│   ├── pipeline.py   # Video processing & PDF generation
│   ├── detector.py   # YOLO tab detection
│   └── models/       # ML models
│       └── best.pt   # YOLO model (required)
├── frontend/         # Static HTML frontend
│   └── index.html
└── requirements-deploy.txt  # Production dependencies
```

## How It Works

1. **Upload Video**: User uploads a video file
2. **Extract Screenshots**: Backend extracts screenshots every 2 seconds using YOLO to detect guitar tab regions
3. **User Selection**: User reviews and selects which screenshots to include
4. **Generate PDF**: Selected screenshots are combined into a PDF with proper layout

## Requirements

- Python 3.9+
- YOLO model file (`backend/models/best.pt`)
- See `requirements-deploy.txt` for Python dependencies

## License

Free to use and modify.

## Support

For deployment issues, see [DEPLOYMENT.md](DEPLOYMENT.md).
For local development issues, see [README_SERVER.md](README_SERVER.md).

