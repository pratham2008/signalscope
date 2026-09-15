# SignalScope web workspace

The web interface is a local Next.js client for the SignalScope FastAPI service. It lets a user upload an image, view the calibrated AI-generation likelihood, and compare the original against the model-evidence overlay.

## Run locally

From the repository root, activate the Python environment and start the API:

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

In another terminal, run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The client expects the API at `http://127.0.0.1:8000` by default. To use another endpoint, create `frontend/.env.local` with:

```bash
NEXT_PUBLIC_API_URL=http://your-host:8000
```

The backend expects the checkpoint configured in `api/main.py` to be present. It returns an evidence overlay only when the selected model supports that explanation mode.

## Verification

```bash
npm run lint
```

The page uses local browser object URLs for uploaded images and API data URLs for evidence overlays; these deliberately remain plain `<img>` elements instead of Next image optimization.
