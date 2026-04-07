# DigitalOcean Deployment

This repo is set up to deploy on DigitalOcean App Platform with:

- `frontend/` as a Next.js web service
- `backend/` as a FastAPI web service
- a shared app ingress rule that sends `/api` traffic to FastAPI and everything else to Next.js

That means the browser can use same-origin requests to `/api`, which keeps production deployment simpler and avoids CORS problems.

## Before You Deploy

1. Push this repo to GitHub or GitLab.
2. Update the placeholder repo URL in `.do/deploy.template.yaml`.

Current placeholders:

- `https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git`

## App Platform Setup

You can either:

1. Use the template in `.do/deploy.template.yaml`, or
2. Recreate the same settings manually in the DigitalOcean UI.

### Frontend Service

- Name: `frontend`
- Source directory: `frontend`
- Environment: `node-js`
- Build command: `npm ci && npm run build`
- Run command: `npm start`

### Backend Service

- Name: `backend`
- Source directory: `.`
- Environment: `python`
- Build command: `pip install -r requirements-deploy.txt`
- Run command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
- Runtime env:
  - `FRONTEND_URL=${APP_URL}`

## Ingress Rules

Configure routing so the public app domain works like this:

- `/api` -> `backend`
- `/` -> `frontend`

The frontend already defaults to `/api`, so no extra production API URL variable is required.

## Local Development

### Backend

```powershell
python start_server.py
```

### Frontend

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Default local behavior:

- Next.js runs on `http://localhost:3000`
- FastAPI runs on `http://127.0.0.1:8000`
- Next.js rewrites `/api/*` to the backend using `BACKEND_URL` from `frontend/.env.example`

## Notes

- The frontend is now a Next.js app, not the old static `index.html` page.
- The backend still supports both `POST /process-url` and `POST /process`.
- Session state is still in-memory on the backend, so restarting the backend clears any active extraction session.
