# SignalScope web workspace

The web interface is a local Next.js client for the SignalScope FastAPI service. It lets a user upload an image, view the calibrated AI-generation likelihood, and compare the original against the model-evidence overlay.

## Run locally

### Windows PowerShell

From the repository root, open PowerShell and activate the Python environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the API from the repository root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

If PowerShell reports that `npm.ps1` cannot be loaded because script execution is disabled, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then retry the npm command.

As a fallback:

```powershell
npm.cmd install
npm.cmd run dev
```

### Linux / macOS

From the repository root, activate the Python environment and start the API:

```bash
source .venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The client expects the API at:

```text
http://127.0.0.1:8000
```

To use another API endpoint, create `frontend/.env.local`:

```text
NEXT_PUBLIC_API_URL=http://your-host:8000
```

The backend expects the production checkpoint configured in `api/main.py` to be present. The production checkpoint is:

```text
model/dual/clip_fft_resolution.pt
```

The first analysis after starting the backend may take longer while the model and dependencies initialize. Subsequent analyses should normally respond faster.

## Verification

Run:

```bash
npm run lint
```

The page uses local browser object URLs for uploaded images and API data URLs for evidence overlays; these deliberately remain plain `<img>` elements instead of Next image optimization.
