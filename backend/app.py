# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from pathlib import Path
from uuid import uuid4
import tempfile
import os
from typing import List, Dict

from backend.pipeline import extract_screenshots, create_pdf_from_selected
from backend.downloader import download_video

app = FastAPI(
    title="Guitar Tab Extractor API",
    description="Extract and compose guitar tabs from video screenshots",
    version="1.0.0"
)

# CORS configuration - allow frontend origin from environment variable
frontend_url = os.getenv("FRONTEND_URL", "*")
allowed_origins = [origin.strip() for origin in frontend_url.split(",") if origin.strip()]

if frontend_url == "*" or not allowed_origins:
    # When allowing all origins, credentials must be False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # When specifying a specific origin, we can use credentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# In-memory session store (in production, use Redis or database)
sessions: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return JSONResponse({
        "message": "Guitar Tab Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "process": "/process (POST)",
            "process_url": "/process-url (POST)",
            "create_pdf": "/create-pdf (POST)"
        },
        "status": "running"
    })


class CreatePDFRequest(BaseModel):
    session_id: str
    selected_indices: List[int]


class ProcessURLRequest(BaseModel):
    url: HttpUrl
    output_dir: str = "data/videos/"


def _process_video_path(video_path: str, session_id: str) -> Dict:
    screenshots, crops_dir = extract_screenshots(video_path, session_id)

    response_screenshots = [
        {
            "index": s["index"],
            "image": s["image"],
            "timestamp": s["timestamp"],
        }
        for s in screenshots
    ]

    sessions[session_id] = {
        "screenshots": screenshots,  # Full data with crop_path
        "crops_dir": crops_dir,
    }

    return {
        "session_id": session_id,
        "screenshots": response_screenshots,
    }


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """Extract screenshots from video at regular intervals."""
    # 1) cross-platform temp dir
    tmpdir = Path(tempfile.gettempdir()) / "guitartab"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # 2) pick a unique filename, keep original suffix
    suffix = Path(file.filename).suffix or ".mp4"
    dest = tmpdir / f"{uuid4().hex}{suffix}"

    # 3) stream upload to disk in chunks
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)   # 1MB
            if not chunk:
                break
            out.write(chunk)
    await file.close()

    # 4) Generate session ID
    session_id = uuid4().hex

    # 5) Extract screenshots
    try:
        result = _process_video_path(str(dest), session_id)

        # Clean up video file
        dest.unlink(missing_ok=True)

        return result
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-url")
async def process_url(request: ProcessURLRequest):
    """
    Download a video from URL, then extract screenshots from it.
    """
    session_id = uuid4().hex
    try:
        downloaded_video = download_video(str(request.url), request.output_dir)
        result = _process_video_path(downloaded_video, session_id)
        result["video_path"] = downloaded_video
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-pdf")
async def create_pdf(request: CreatePDFRequest):
    """Create PDF from selected screenshots."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    screenshots = session_data['screenshots']
    crops_dir = session_data['crops_dir']
    
    try:
        pdf_path = create_pdf_from_selected(screenshots, request.selected_indices, crops_dir)
        
        if not Path(pdf_path).exists():
            raise HTTPException(status_code=500, detail="PDF generation failed")
        
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="guitar_tabs.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
