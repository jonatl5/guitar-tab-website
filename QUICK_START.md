# Quick Start

## Backend

```bash
pip install -r requirements.txt
python start_server.py
```

The backend will be available at `http://127.0.0.1:8000`.

## Frontend

```bash
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## Notes

- The frontend now uses Next.js instead of the old static `frontend/index.html` page.
- Local frontend requests go to `/api` and are rewritten to the backend URL in `.env.local`.
- If you change the backend port, update `BACKEND_URL` in `frontend/.env.local`.
