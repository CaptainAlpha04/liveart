# LiveArt — System Design Specification

**Date:** 2026-04-18  
**Status:** Approved  
**Scope:** Full-stack web application for real-time Neural Style Transfer via webcam and video file

---

## 1. Overview

LiveArt is a web-based deep learning application that applies the aesthetic style of famous artworks to live webcam feeds and pre-recorded video files in near-real-time. Users select a style from a library of pre-trained models (analogous to Snapchat filters) and the stylization is applied immediately with no perceptible delay attributable to transport. A secondary Training page allows users to train new custom styles by uploading their own reference artwork and monitoring loss curves live.

**Target environment:** Local machine or shared LAN server. No authentication required.  
**Primary users:** A single developer or a small group (semester project demo).

---

## 2. Technical Architecture

### 2.1 System Components

```
Browser (React 18 + Vite + TypeScript)
│
├── Inference Page
│   ├── Webcam source → WebSocket /ws/inference → stylized frame response
│   └── Video file source → POST /api/video/stylize → poll → download
│
└── Training Page
    ├── POST /api/training/start (style image + hyperparams)
    └── WebSocket /ws/training → live loss events → Recharts graph

FastAPI Backend (Python)
├── model_manager.py     — loads all .pth models at startup, holds in RAM
├── inference_engine.py  — async frame queue, background worker thread
├── training_engine.py   — full training loop with VGG-19 perceptual loss
└── video_processor.py   — OpenCV decode → inference → re-encode
```

### 2.2 Transport Layer

**WebSockets** are used for all streaming data (inference frames and training logs). HTTP REST is used for discrete operations (video upload, training control, model management).

Rationale: WebSockets provide bidirectional, low-overhead streaming without the complexity of WebRTC (which saves <2ms on local transport — negligible vs. inference time). Perceived end-to-end latency target is 15–40ms depending on hardware.

### 2.3 Latency Optimization Strategy

| Technique | Effect |
|-----------|--------|
| All 10 models loaded in RAM at startup | Zero model-load latency on style switch |
| Async frame queue with background worker thread | UI never blocks waiting for inference |
| Frame dropping when queue depth > 2 | Eliminates stale-frame accumulation |
| Inference runs on 480p frames | ~4x faster than 1080p with minimal quality loss |
| CUDA auto-detection at startup | GPU used when available, CPU fallback |
| Client-side immediate canvas render | Browser draws each received frame without buffering |

---

## 3. Machine Learning Design

### 3.1 Model Architecture — Fast Neural Style Transfer

Based on Johnson et al. (2016), "Perceptual Losses for Real-Time Style Transfer and Super-Resolution."

**Transform Network (one per style):**
- Encoder: 3 convolutional layers with stride-2 downsampling
- Residual core: **9 residual blocks** with Instance Normalization
- Decoder: 2 fractionally-strided (transposed) convolutional layers + output conv
- Activation: ReLU throughout; Tanh on output layer
- Normalization: **Instance Normalization** (Ulyanov et al., 2017) — produces sharper, more consistent stylization than Batch Normalization
- Parameter count: ~1.7M per model; file size ~6–7MB per `.pth`

**Loss Network:**
- **VGG-19** (pretrained on ImageNet, frozen throughout training)
- Content layer: `relu3_3`
- Style layers: `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` (multi-scale style capture)

### 3.2 Loss Function

```
L_total = content_weight * L_content
        + style_weight   * L_style
        + tv_weight      * L_tv
```

| Loss Term | Formula | Default Weight |
|-----------|---------|----------------|
| Content loss | MSE between VGG feature maps of output and content image | `1e5` |
| Style loss | MSE between Gram matrices of VGG feature maps | `1e10` |
| Total Variation loss | Sum of squared differences between adjacent pixels | `1e-6` |

Gram matrix: `G[i,j] = (1 / C*H*W) * sum(F[i,:] * F[j,:])`  
TV loss suppresses high-frequency noise and produces visually smooth output.

### 3.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training dataset | COCO 2014 Train (~82,000 images) | Standard benchmark, diverse content |
| Input resolution | 256×256 | Balances training speed and quality |
| Batch size | 4 | Fits within typical GPU VRAM (4–8GB) |
| Optimizer | Adam | `lr=1e-3`, `betas=(0.9, 0.999)` |
| LR schedule | Step decay: `1e-3 → 1e-4` after epoch 1 | Stabilizes convergence in epoch 2 |
| Epochs | 2 | ~41,000 iterations per style (82k images / batch 4 × 2 epochs) |
| Checkpoint interval | Every 500 batches | Resumes interrupted training |

### 3.4 Pre-trained Style Library (10 Styles)

| ID | Name | Artwork | Artist |
|----|------|---------|--------|
| `starry_night` | Starry Night | The Starry Night | Van Gogh |
| `the_scream` | The Scream | The Scream | Edvard Munch |
| `candy` | Candy | Abstract | — |
| `mosaic` | Mosaic | Byzantine tile art | — |
| `udnie` | Udnie | Udnie | Francis Picabia |
| `rain_princess` | Rain Princess | Rain Princess | Leonid Afremov |
| `la_muse` | La Muse | La Muse | Pablo Picasso |
| `feathers` | Feathers | Abstract feather | — |
| `great_wave` | The Great Wave | The Great Wave | Hokusai |
| `composition_viii` | Composition VIII | Composition VIII | Wassily Kandinsky |

Weights are sourced from the official `pytorch/examples` repository and community-verified releases. All 10 are downloaded via `scripts/download_models.py` before first run.

---

## 4. API Specification

### 4.1 WebSocket Endpoints

#### `WS /ws/inference`

Streams webcam frames from client to server and returns stylized frames.

**Client → Server message (per frame):**
```json
{
  "style": "starry_night",
  "frame": "<base64-encoded JPEG string>",
  "width": 480,
  "height": 360
}
```

**Server → Client message (per frame):**
```json
{
  "frame": "<base64-encoded JPEG string>",
  "inference_ms": 18,
  "fps": 24.3,
  "style": "starry_night"
}
```

**Error message:**
```json
{
  "error": "style_not_found",
  "detail": "No model loaded for style: unknown_style"
}
```

#### `WS /ws/training`

Pushes training progress events from server to client during an active training job. Read-only from client perspective. Events are emitted every 50 batches.

**Server → Client message (every 50 batches):**
```json
{
  "epoch": 1,
  "batch": 1200,
  "total_batches": 41000,
  "content_loss": 3.21,
  "style_loss": 8.74,
  "tv_loss": 0.04,
  "total_loss": 11.99,
  "elapsed_s": 142,
  "eta_s": 3200,
  "status": "running"
}
```

**Completion message:**
```json
{
  "status": "done",
  "model_id": "my_custom_style",
  "elapsed_s": 18400
}
```

---

### 4.2 REST Endpoints

#### Styles & Models

| Method | Path | Request Body | Response | Description |
|--------|------|-------------|----------|-------------|
| `GET` | `/api/styles` | — | `StyleInfo[]` | List all styles (pre-trained + custom) |
| `GET` | `/api/styles/{style_id}/thumbnail` | — | `image/jpeg` | Sample thumbnail for style |
| `GET` | `/api/models` | — | `ModelInfo[]` | List all loaded model metadata |
| `DELETE` | `/api/models/{model_id}` | — | `204 No Content` | Delete custom model (pre-trained immutable) |

**`StyleInfo` schema:**
```json
{
  "id": "starry_night",
  "name": "Starry Night",
  "artist": "Van Gogh",
  "is_pretrained": true,
  "thumbnail_url": "/api/styles/starry_night/thumbnail"
}
```

#### Video Processing

| Method | Path | Request Body | Response | Description |
|--------|------|-------------|----------|-------------|
| `POST` | `/api/video/stylize` | `multipart/form-data: {file, style_id}` | `{job_id}` | Submit video for processing |
| `GET` | `/api/video/status/{job_id}` | — | `VideoJobStatus` | Poll processing progress |
| `GET` | `/api/video/download/{job_id}` | — | `video/mp4` | Download completed video |
| `DELETE` | `/api/video/{job_id}` | — | `204 No Content` | Clean up job and files from disk |

**`VideoJobStatus` schema:**
```json
{
  "job_id": "abc123",
  "status": "processing",
  "progress": 0.42,
  "total_frames": 1800,
  "processed_frames": 756,
  "elapsed_s": 34
}
```

Status values: `queued` | `processing` | `done` | `error`

#### Training

| Method | Path | Request Body | Response | Description |
|--------|------|-------------|----------|-------------|
| `POST` | `/api/training/start` | `multipart/form-data: {style_image, config}` | `{job_id}` | Start training job |
| `POST` | `/api/training/stop` | — | `{status: "stopped"}` | Gracefully stop training |
| `GET` | `/api/training/status` | — | `TrainingStatus` | Current training job state |

**`TrainingConfig` schema (sent as JSON field in multipart):**
```json
{
  "style_name": "my_style",
  "style_weight": 1e10,
  "content_weight": 1e5,
  "tv_weight": 1e-6,
  "learning_rate": 1e-3,
  "epochs": 2,
  "batch_size": 4
}
```

**`TrainingStatus` schema:**
```json
{
  "state": "running",
  "style_name": "my_style",
  "epoch": 1,
  "batch": 4200,
  "total_batches": 41000,
  "progress": 0.102
}
```

State values: `idle` | `running` | `done` | `error`

#### Health

| Method | Path | Response | Description |
|--------|------|----------|-------------|
| `GET` | `/health` | `HealthStatus` | Server health + GPU availability |

**`HealthStatus` schema:**
```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "models_loaded": 10,
  "uptime_s": 342
}
```

---

## 5. Frontend Component Specification

### 5.1 Pages

#### `InferencePage` (`/`)
Primary user-facing page. Contains source toggle, style selector, and live/processed output.

**State managed:**
- `activeStyle: string` — currently selected style ID
- `sourceMode: "webcam" | "video"` — active input source
- `isConnected: boolean` — WS connection state
- `stats: {fps, inference_ms}` — updated per frame

#### `TrainingPage` (`/training`)
Secondary page for custom style training.

**State managed:**
- `trainingStatus: TrainingStatus` — polled from server
- `lossHistory: TrainingEvent[]` — accumulated for chart
- `config: TrainingConfig` — form-controlled hyperparameters

---

### 5.2 Component Contracts

#### `WebcamFeed`
- Accesses `getUserMedia({video: true})`
- Captures frames at 30fps via `requestAnimationFrame` + `canvas.toDataURL("image/jpeg", 0.8)`
- Emits frames via callback prop `onFrame(base64: string)`
- Props: `onFrame`, `width`, `height`, `active`

#### `StyleGrid`
- Renders scrollable horizontal strip of style cards
- Each card: thumbnail image + style name + selected state ring
- Props: `styles: StyleInfo[]`, `selected: string`, `onSelect: (id: string) => void`

#### `StylizedCanvas`
- Receives base64 JPEG strings, draws to `<canvas>` via `Image` API
- Props: `frame: string | null`, `width`, `height`

#### `StatsBar`
- Displays: active style name, current FPS, inference latency (ms), WS connection state
- Props: `style`, `fps`, `inferenceMs`, `connected`

#### `LossChart`
- Recharts `LineChart` with three series: `content_loss`, `style_loss`, `total_loss`
- X-axis: batch number; Y-axis: loss value (log scale)
- Props: `data: TrainingEvent[]`

#### `HyperparamForm`
- Controlled form for `TrainingConfig`
- Fields pre-filled with production-quality defaults
- Includes tooltip explanations for each hyperparameter
- Props: `value: TrainingConfig`, `onChange`, `disabled`

---

### 5.3 Custom Hooks

#### `useInferenceSocket(url: string)`
- Manages WebSocket lifecycle: connect, disconnect, reconnect on error
- Exposes: `sendFrame(style, base64)`, `lastFrame: string | null`, `stats`, `connected`

#### `useTrainingSocket(url: string)`
- Connects to `/ws/training`, accumulates `TrainingEvent[]`
- Exposes: `events: TrainingEvent[]`, `latestEvent`, `connected`

#### `useWebcam(width, height)`
- Wraps `getUserMedia`, provides `videoRef`, `captureFrame(): string`
- Handles permission errors gracefully

---

## 6. File Structure

```
liveart/
├── backend/
│   ├── main.py                   # FastAPI app entry point, lifespan startup/shutdown
│   ├── routers/
│   │   ├── inference_ws.py       # WS /ws/inference
│   │   ├── training_ws.py        # WS /ws/training
│   │   ├── video.py              # /api/video/* endpoints
│   │   ├── training.py           # /api/training/* endpoints
│   │   └── models.py             # /api/styles, /api/models endpoints
│   ├── core/
│   │   ├── model_manager.py      # Singleton: loads + caches all .pth models
│   │   ├── inference_engine.py   # Frame queue, worker thread, dispatch
│   │   ├── training_engine.py    # Training loop: VGG-19, gram matrices, TV loss
│   │   ├── video_processor.py    # OpenCV video decode → inference → re-encode
│   │   └── vgg.py                # VGG-19 feature extractor (frozen)
│   ├── schemas/
│   │   ├── inference.py          # InferenceRequest, InferenceResponse
│   │   ├── training.py           # TrainingConfig, TrainingEvent, TrainingStatus
│   │   └── video.py              # VideoJobStatus, VideoJobResponse
│   ├── models/                   # .pth files (gitignored)
│   ├── thumbnails/               # Style sample images
│   └── uploads/                  # Temporary video storage (auto-cleaned)
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── InferencePage.tsx
│   │   │   └── TrainingPage.tsx
│   │   ├── components/
│   │   │   ├── WebcamFeed.tsx
│   │   │   ├── VideoUpload.tsx
│   │   │   ├── StyleGrid.tsx
│   │   │   ├── StylizedCanvas.tsx
│   │   │   ├── StatsBar.tsx
│   │   │   ├── LossChart.tsx
│   │   │   ├── TrainingLog.tsx
│   │   │   ├── HyperparamForm.tsx
│   │   │   └── ModelExport.tsx
│   │   ├── hooks/
│   │   │   ├── useInferenceSocket.ts
│   │   │   ├── useTrainingSocket.ts
│   │   │   └── useWebcam.ts
│   │   ├── api/
│   │   │   └── client.ts          # Typed fetch wrappers for all REST endpoints
│   │   └── types/
│   │       └── index.ts           # Shared TypeScript interfaces
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
│
├── scripts/
│   └── download_models.py         # Downloads all 10 .pth files, verifies checksums
├── requirements.txt
└── docs/
    ├── Implementation_plan.md
    └── superpowers/specs/
        └── 2026-04-18-liveart-design.md
```

---

## 7. Dependencies

### Backend (`requirements.txt`)

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
websockets>=12.0
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
Pillow>=10.3.0
numpy>=1.26.0
python-multipart>=0.0.9
aiofiles>=23.2.1
pydantic>=2.7.0
requests>=2.31.0
tqdm>=4.66.0
```

### Frontend (`package.json` key dependencies)

```json
{
  "react": "^18.3.0",
  "react-dom": "^18.3.0",
  "react-router-dom": "^6.23.0",
  "recharts": "^2.12.0",
  "tailwindcss": "^3.4.0",
  "typescript": "^5.4.0",
  "vite": "^5.2.0"
}
```

---

## 8. Error Handling

| Scenario | Backend Behavior | Frontend Behavior |
|----------|-----------------|-------------------|
| Style model not found | WS error message `{error: "style_not_found"}` | Toast notification, revert to last valid style |
| WebSocket disconnect | Server cleans up frame queue | Auto-reconnect with exponential backoff (max 5 retries) |
| Video upload > 500MB | HTTP 413 before processing | Upload rejected with size error message |
| Training already running | HTTP 409 Conflict | "Training in progress" badge, Start button disabled |
| CUDA OOM during training | Caught, job marked `error`, WS notifies client | Error state shown on Training page with message |
| Webcam permission denied | N/A | Graceful fallback to video-file-only mode |

---

## 9. Non-Goals (Explicitly Out of Scope)

- User authentication or session management
- Multi-user concurrent training jobs (one training job at a time)
- Model quantization or ONNX export
- Mobile browser support (desktop Chrome/Firefox only)
- Deployment to cloud infrastructure
