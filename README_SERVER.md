# Backend Setup

## Install

```bash
pip install -r requirements.txt
```

## Run

Recommended:

```bash
python start_server.py
```

Or directly with Uvicorn:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

## Frontend Pairing

Run the frontend separately from `frontend/`:

```bash
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

The Next.js app proxies `/api/*` to the backend URL in `.env.local`, so you do not need to hardcode backend URLs in the browser UI.
