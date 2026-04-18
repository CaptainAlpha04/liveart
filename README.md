# LiveArt

Real-time Neural Style Transfer for webcam and video — apply the aesthetic of famous artworks to live video with near-zero perceived latency. Pick a style from a curated library of 10 pre-trained models, or train your own from a single reference image and watch the loss curves converge live.

## Features

- 10 pre-trained styles (Starry Night, The Scream, The Great Wave, Composition VIII, and more)
- Live webcam stylization over WebSockets with target end-to-end latency of 15–40 ms
- Video file processing — upload any MP4/MOV, stylize every frame at native resolution, download the result
- Custom style training with live loss graphs (content, style, total) streamed over WebSocket
- CUDA auto-detection with CPU fallback; frame-dropping keeps the UI smooth under load

## Tech Stack

| Layer     | Tool                                                   |
| --------- | ------------------------------------------------------ |
| Backend   | FastAPI + Uvicorn (Python 3.10+)                       |
| ML        | PyTorch 2.2+, torchvision, VGG-19 perceptual loss      |
| Video I/O | OpenCV, Pillow                                         |
| Transport | WebSockets (streaming) + REST (discrete operations)    |
| Frontend  | React 18 + Vite + TypeScript                           |
| Styling   | Tailwind CSS                                           |
| Charts    | Recharts                                               |
| Routing   | React Router v6                                        |

## Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) CUDA-capable GPU — CPU works but is ~10x slower
- (Optional, for training) COCO 2014 Train images (~13 GB) — required only if you want to train new custom styles

## Quickstart (one command)

```bash
python run.py
```

`run.py` is a dev launcher that:

1. Ensures backend Python deps (`fastapi`, `uvicorn`, `websockets`, `python-multipart`, `aiofiles`) are installed in the target env.
2. Runs `npm install` in `frontend/` if `node_modules` is missing.
3. Runs `scripts/download_models.py` if no `.pth` files are present yet.
4. Starts the FastAPI backend on `:8000` and the Vite frontend on `:5173`, streaming both outputs with colored prefixes.
5. Shuts both processes down cleanly on Ctrl+C.

By default it targets `C:\Users\Ali\miniconda3\envs\study\python.exe` (the `study` conda env). Override with:

```bash
LIVEART_PY=/path/to/python.exe python run.py
```

## Manual Setup (alternative)

**1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**2. Download pre-trained models and generate thumbnails**

```bash
python scripts/download_models.py
```

This downloads the 4 styles bundled by the official `pytorch/examples` release (`mosaic`, `candy`, `rain_princess`, `udnie`) into `backend/models/`, writes metadata JSON for all 10 styles, and generates placeholder thumbnails in `backend/thumbnails/`. The remaining 6 styles (`starry_night`, `the_scream`, `la_muse`, `feathers`, `great_wave`, `composition_viii`) can be trained via the UI.

**3. Install frontend dependencies**

```bash
cd frontend && npm install
```

**4. Run backend and frontend**

Backend (from the repo root):

```bash
uvicorn backend.main:app --reload --port 8000
```

Frontend (in another terminal):

```bash
cd frontend && npm run dev
```

The app opens at http://localhost:5173 and proxies `/api`, `/ws`, and `/health` to the backend on port 8000.

## Training Custom Styles

Training requires the COCO 2014 Train image set (~82,000 images, ~13 GB) as the content corpus.

1. Download COCO 2014 Train images from https://cocodataset.org/#download (the `2014 Train images` zip).
2. Extract them to `data/coco_train/` (default location), or configure a different path via the `COCO_TRAIN_DIR` environment variable:

   ```bash
   export COCO_TRAIN_DIR=/path/to/coco/train2014
   ```

3. Go to the Training page in the UI, upload a reference style image, tune hyperparameters, and press Start. Training runs for 2 epochs (~41,000 iterations) and streams loss metrics live. The resulting model is saved to `backend/trained_models/` and appears in the style library on the Inference page.

## API Documentation

FastAPI auto-generates interactive API docs at http://localhost:8000/docs (Swagger UI) and http://localhost:8000/redoc (ReDoc) once the backend is running.

## Project Structure

```
liveart/
├── backend/              FastAPI server + PyTorch inference & training engines
│   ├── core/             Transform net, VGG-19, model manager, inference/training/video engines
│   ├── routers/          REST endpoints + WebSocket routes
│   ├── schemas/          Pydantic models
│   ├── models/           Pre-trained .pth files (gitignored)
│   ├── trained_models/   User-trained .pth files (gitignored)
│   ├── thumbnails/       Style preview images
│   └── uploads/          Temporary video storage (gitignored)
├── frontend/             React 18 + Vite + TypeScript SPA
│   └── src/              Pages, components, hooks, API client, types
├── scripts/              Utility scripts (model downloader, etc.)
├── docs/                 Design spec & implementation plan
├── requirements.txt      Python dependencies
└── README.md
```

## Documentation

- [Design Specification](docs/superpowers/specs/2026-04-18-liveart-design.md) — full system design, API shapes, ML details
- [Implementation Plan](docs/superpowers/plans/2026-04-18-liveart-implementation.md) — task-by-task build order
