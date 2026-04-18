# LiveArt Implementation Plan

> **For agentic workers:** Execute task-by-task. Code belongs in source files, not this plan. Reference the spec at [2026-04-18-liveart-design.md](../specs/2026-04-18-liveart-design.md) for all API shapes, schemas, architecture, and ML details.

**Goal:** Full-stack web application for real-time Neural Style Transfer with 10 pre-trained styles, webcam/video inference over WebSockets, and a live training UI for custom styles.

**Architecture:** FastAPI backend hosts PyTorch Fast Style Transfer models with VGG-19 perceptual loss. Inference runs in a `ThreadPoolExecutor` so WebSockets remain responsive. React + Vite + TypeScript frontend communicates via WebSockets (streaming) and REST (discrete operations).

**Tech Stack:** Python 3.10+, PyTorch 2.2+, FastAPI, OpenCV, React 18, Vite, TypeScript, Tailwind CSS, Recharts.

**Reference Spec:** [docs/superpowers/specs/2026-04-18-liveart-design.md](../specs/2026-04-18-liveart-design.md) — single source of truth for all API shapes, component contracts, ML hyperparameters, and file structure.

---

## Phase 1: Project Setup

### Task 1 — Root configuration and scaffolding

**Scope:** Root-level files and directory structure.

**Deliverables:**
- `requirements.txt` with pinned minimum versions (fastapi, uvicorn, torch, torchvision, opencv-python, Pillow, numpy, python-multipart, aiofiles, pydantic, requests, tqdm)
- `.gitignore` excluding `__pycache__/`, `*.pth`, `node_modules/`, `dist/`, `backend/uploads/*`, `backend/models/*.pth`, `backend/trained_models/*.pth`
- `README.md` with quickstart (backend start command, frontend start command, download script usage)
- Directory structure per spec §6: `backend/{routers,core,schemas,models,thumbnails,uploads,trained_models}`, `frontend/src/{pages,components,hooks,api,types}`, `scripts/`
- Python package `__init__.py` files for all backend sub-packages

---

## Phase 2: Backend

### Task 2 — ML model architecture (`backend/core/transform_net.py`)

Implements the Fast Style Transfer network per Johnson et al. (2016) as specified in spec §3.1: encoder (3 downsampling convs) → **9 residual blocks** with **Instance Normalization** → decoder (2 upsampling convs + output conv) → `tanh` activation. Use reflection padding in all convs. Output is in `[-1, 1]`.

### Task 3 — VGG-19 feature extractor (`backend/core/vgg.py`)

Frozen VGG-19 loaded from torchvision with ImageNet weights. Exposes features at indices `[3, 8, 15, 24]` (relu1_2, relu2_2, relu3_3, relu4_3). Register ImageNet mean/std buffers. Internally converts `[-1, 1]` input to `[0, 1]` then normalizes. Also provide `gram_matrix(features)` helper normalized by `C*H*W`.

### Task 4 — Model manager (`backend/core/model_manager.py`)

Thread-safe singleton that loads all `.pth` files from `backend/models/` (pre-trained, immutable) and `backend/trained_models/` (user-trained, deletable) at startup. Loads optional `<style_id>.json` metadata files (name, artist). Provides `get_model`, `has`, `list_styles`, `register_trained`, `delete`. Sets `is_pretrained` flag based on source directory. Uses `torch.load(..., weights_only=True)` for safety.

### Task 5 — Inference engine (`backend/core/inference_engine.py`)

Async wrapper around the model manager. Single-worker `ThreadPoolExecutor` for deterministic GPU use. `stylize_b64(style_id, b64_jpeg) → (b64_jpeg, elapsed_ms)` decodes JPEG, converts to tensor in `[-1, 1]`, runs `inference_mode()` forward pass, encodes JPEG at quality 85. Also provides `stylize_tensor` for the video pipeline. Raises `KeyError` for unknown styles.

### Task 6 — Training engine (`backend/core/training_engine.py`)

Implements the full training loop per spec §3.2–3.3. Dataset: `ImageFolderDataset` recursively scans a directory for images (COCO 2014 target). Transforms: resize → center crop to 256 → ToTensor → normalize to `[-1, 1]`. Loss: `content_weight * MSE(relu3_3) + style_weight * Σ MSE(gram(layer_i)) + tv_weight * TV(output)`. Optimizer: Adam with lr=1e-3. Scheduler: StepLR(step=1, gamma=0.1). Emit events every 50 batches via a callback (shape per spec §4.1 training WS). Supports stop via `threading.Event`. Saves `.pth` + metadata JSON on completion. Single job at a time — `is_running()` + `get_status()` state machine.

### Task 7 — Video processor (`backend/core/video_processor.py`)

OpenCV-based video pipeline running in a daemon thread. `VideoJobRegistry` tracks jobs in a thread-safe dict. `VideoProcessor.submit(input_path, output_path, style_id) → job_id` queues a job. Worker decodes with `cv2.VideoCapture`, runs inference per frame at native resolution via `engine.stylize_tensor`, re-encodes with `cv2.VideoWriter` using `mp4v` fourcc. Updates progress every 10 frames. Preserves source FPS and dimensions.

### Task 8 — Pydantic schemas (`backend/schemas/__init__.py`)

Single module with: `StyleInfo`, `TrainingConfig` (with spec §4.2 defaults), `TrainingStatus`, `VideoJobResponse`, `VideoJobStatus`, `HealthStatus`. Exact field shapes per spec §4.

### Task 9 — Inference WebSocket (`backend/routers/inference_ws.py`)

`WS /ws/inference`. Per-connection frame-dropping: if a previous frame is still processing, discard incoming frames (don't queue them). After each successful inference, rolling 1-second window to compute FPS. Send error messages for `style_not_found` without closing the socket. Import `engine` lazily from `main` to avoid circular imports.

### Task 10 — Training WebSocket + REST (`backend/routers/training_ws.py`, `backend/routers/training.py`)

**`training_ws.py`:** `WS /ws/training`. `TrainingEventBroadcaster` keeps a set of connected clients and an `asyncio.AbstractEventLoop` reference captured at startup. `emit_threadsafe(event)` uses `run_coroutine_threadsafe` to push events from the training thread to all clients.

**`training.py`:** `POST /api/training/start` (multipart: `style_image` file + `config` JSON string; validates `TrainingConfig`; slugifies `style_name` to `style_id`; persists the reference image as the new style's thumbnail; spawns a training thread with `on_event` = broadcaster emit; returns `{job_id: style_id}`; returns 409 if already running). `POST /api/training/stop` calls `training_engine.request_stop()`. `GET /api/training/status` returns `TrainingStatus`.

### Task 11 — Video + Models routers (`backend/routers/video.py`, `backend/routers/models.py`)

**`video.py`:** `POST /api/video/stylize` (multipart, 500MB cap, validates style exists, writes to `uploads/`, submits to `VideoProcessor`, returns `{job_id}`). `GET /api/video/status/{job_id}`. `GET /api/video/download/{job_id}` (FileResponse, 409 if not done). `DELETE /api/video/{job_id}` (removes files + registry entry).

**`models.py`:** `GET /api/styles` and `GET /api/models` both return `model_manager.list_styles()`. `GET /api/styles/{style_id}/thumbnail` serves the thumbnail file (jpg or png). `DELETE /api/models/{model_id}` rejects pretrained, deletes trained ones.

### Task 12 — FastAPI app (`backend/main.py`)

App factory with `@asynccontextmanager` lifespan. Captures the running event loop and sets it on the training broadcaster. Auto-detects CUDA, logs device. Creates directories. Instantiates singletons in lifespan: `ModelManager`, `InferenceEngine`, `TrainingEngine`, `VideoJobRegistry`, `VideoProcessor`. CORS middleware allowing all origins (local dev). Registers all routers. `GET /health` returns `HealthStatus`. Exports singletons as module-level attributes so routers can import them lazily. Entry point: `uvicorn backend.main:app --reload --port 8000`.

---

## Phase 3: Scripts

### Task 13 — Model download script (`scripts/download_models.py`)

Downloads the `pytorch/examples` fast_neural_style `saved_models.zip` (contains `mosaic`, `candy`, `rain_princess`, `udnie`) and extracts matching `.pth` files to `backend/models/`. For the 6 styles not in that bundle (`starry_night`, `the_scream`, `la_muse`, `feathers`, `great_wave`, `composition_viii`), write metadata JSON stubs and log a message instructing the user to train them via the UI. Generate placeholder thumbnail images (colored gradient + style name text) for all 10 styles using PIL. Idempotent — skips existing files.

---

## Phase 4: Frontend

### Task 14 — Frontend scaffolding

**Files:** `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`, `frontend/tailwind.config.ts`, `frontend/postcss.config.js`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/index.css`

Vite + React 18 + TypeScript + Tailwind + React Router v6 + Recharts. Vite proxy forwards `/api`, `/ws`, `/health` to `http://localhost:8000` (with `ws: true` for WS). Dark theme by default via Tailwind. `index.css` has `@tailwind base/components/utilities` and a global body `bg-zinc-950 text-zinc-100`.

### Task 15 — Types and API client (`frontend/src/types/index.ts`, `frontend/src/api/client.ts`)

**Types:** `StyleInfo`, `TrainingConfig`, `TrainingStatus`, `TrainingEvent`, `VideoJobStatus`, `InferenceStats`, `HealthStatus` — mirror backend schemas exactly.

**Client:** typed fetch wrappers — `fetchStyles()`, `fetchModels()`, `deleteModel(id)`, `uploadVideo(file, styleId)`, `videoStatus(id)`, `videoDownloadUrl(id)`, `startTraining(imageFile, config)`, `stopTraining()`, `trainingStatus()`, `health()`.

### Task 16 — Custom hooks (`frontend/src/hooks/useWebcam.ts`, `useInferenceSocket.ts`, `useTrainingSocket.ts`)

**`useWebcam(width, height)`:** `getUserMedia`, manages `videoRef`/`canvasRef`, provides `captureFrame()` returning base64-encoded JPEG (no data URL prefix). Handles permission errors.

**`useInferenceSocket(url)`:** WS lifecycle with exponential-backoff reconnect (max 5 retries, 1s→30s). Exposes `sendFrame(style, base64)`, `lastFrame`, `stats` (`fps`, `inferenceMs`, `style`), `connected`, `error`.

**`useTrainingSocket(url)`:** Accumulates `TrainingEvent[]`. Exposes `events`, `latestEvent`, `connected`, `reset()` helper.

### Task 17 — Inference components

- `StyleGrid` — horizontal scrollable strip of cards (thumbnail + name + selected ring); keyboard-navigable
- `StylizedCanvas` — receives base64 JPEG, draws to `<canvas>` via the `Image` API with `requestAnimationFrame` for smooth paint
- `StatsBar` — shows active style name, FPS, inference latency, connection indicator dot
- `WebcamFeed` — consumes `useWebcam`; on parent-controlled interval (throttled to ~20 fps) invokes `captureFrame` and calls `onFrame` prop
- `VideoUpload` — drag-drop file input, uploads via `uploadVideo`, polls `videoStatus` every 500ms, shows progress bar, reveals download link when done

### Task 18 — Training components

- `LossChart` — Recharts `LineChart` with three series (`content_loss`, `style_loss`, `total_loss`); X-axis: batch; Y-axis: log scale
- `HyperparamForm` — controlled form matching `TrainingConfig`; number inputs with good defaults per spec; disabled while training
- `TrainingLog` — scrollable list of raw events (epoch, batch, losses); auto-scroll to bottom
- `ModelExport` — appears when training status = `done`; shows "Saved as: `<style_id>`" and a "Back to Library" link

### Task 19 — Pages + App

- `InferencePage` (`/`) — source toggle (Webcam / Upload), active style state, `StyleGrid`, conditional `WebcamFeed` or `VideoUpload`, `StylizedCanvas` for output, `StatsBar`. On mount, fetches `/api/styles`. Uses `useInferenceSocket` and throttles webcam frame sending.
- `TrainingPage` (`/training`) — style-image upload, `HyperparamForm`, Start/Stop buttons, `LossChart`, `TrainingLog`, conditional `ModelExport`. Uses `useTrainingSocket` and polls `/api/training/status` when needed.
- `App.tsx` — `BrowserRouter`, top navbar with links to both pages, dark theme wrapper.

---

## Self-Review

- **Spec coverage:** All 10 REST endpoints (Task 10–11), 2 WS endpoints (Task 9, 10), 9 React components (Task 17–18), 3 hooks (Task 16), 2 pages (Task 19), all 6 core modules (Task 2–7), download script (Task 13). ✓
- **Type consistency:** `style_id` (snake_case) used in all backend boundaries. Frontend types mirror Pydantic exactly. `TrainingConfig`, `StyleInfo`, `TrainingStatus` shapes consistent across tasks.
- **Interface contracts:** Inference engine raises `KeyError` on unknown style → WS router translates to `{error: "style_not_found"}`. Training events flow: engine thread → callback → broadcaster.emit_threadsafe → event loop → all WS clients.
