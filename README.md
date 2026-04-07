# Guitar Tab Extractor

Extract guitar tab screenshots from videos, review the detected tab crops, and generate a PDF of the selections you want to keep.

## Stack

- `backend/`: FastAPI + OpenCV + YOLO-based tab detection
- `frontend/`: Next.js + Tailwind CSS + shadcn/ui web app

## Local Development

### 1. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend

```bash
python start_server.py
```

The API runs at `http://127.0.0.1:8000`.

### 3. Start the frontend

```bash
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

The web app runs at `http://localhost:3000`.

By default, the frontend uses same-origin `/api` requests and rewrites them to the backend URL from `frontend/.env.local`, so the browser-facing API shape already matches production routing.

## API Flow

1. `POST /process-url`: download and process a YouTube video
2. `POST /process`: upload and process a local video file
3. `POST /create-pdf`: generate a PDF from selected screenshot indices

The backend returns extracted screenshots as base64 PNG data and uses an in-memory `session_id` to connect extraction and PDF generation.

## Deployment

- DigitalOcean App Platform: see `DIGITALOCEAN.md`
- Legacy Render/Netlify notes: see `DEPLOYMENT.md`
