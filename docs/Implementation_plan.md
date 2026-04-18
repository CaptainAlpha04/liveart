# LiveArt: Real-Time Video Style Transfer

## Project Overview
LiveArt is a deep learning application that applies the aesthetic style of famous artworks—such as Van Gogh’s *The Starry Night* or Edvard Munch’s *The Scream*—to live video feeds and pre-recorded files. Unlike traditional Neural Style Transfer (NST), which requires hundreds of iterations per image, this project utilizes a **Pre-trained Feed-forward Transformation Network** to achieve real-time inference speeds.

## Technical Architecture
The system utilizes a dual-component architecture:
1.  **The Loss Network:** A pre-trained VGG-16 used during the original training phase to define content and style representations.
2.  **The Transformation Network:** A deep residual convolutional neural network that takes a raw image as input and outputs the stylized version in a single forward pass.



## Features
* **Live Webcam Inference:** Real-time stylization of camera feeds using OpenCV.
* **Video Post-Processing:** Batch processing of existing `.mp4` files.
* **Multi-Model Support:** Toggle between different artistic styles (Mosaic, Starry Night, Candy, etc.).
* **Optimized Performance:** Optimized for CPU/GPU switching using the `cv2.dnn` module.

---

## Implementation Plan (The Weekend Sprint)

### Phase 1: Environment & Asset Acquisition (Saturday AM)
* **Task 1:** Set up a virtual environment and install `opencv-python`, `torch`, `torchvision`, and `numpy`.
* **Task 2:** Download pre-trained `.t7` (Torch) or `.pb` (TensorFlow) weights. These weights are widely available in the Torchvision samples or the official "Fast Style Transfer" repositories.
* **Task 3:** Write a "Model Manager" script to load these weights into OpenCV’s Deep Neural Network (DNN) module.

### Phase 2: Core Logic & Image Inference (Saturday PM)
* **Task 4:** Implement the preprocessing pipeline (mean subtraction and blob scaling) required for VGG-based models.
* **Task 5:** Create a script that stylizes a single static image. This serves as the "Proof of Concept" for the semester project lab component.

### Phase 3: Video Integration & UI (Sunday AM)
* **Task 6:** Develop the video processing loop. Use Claude to optimize the `while True` loop for webcam capture, ensuring frame resizing maintains a high FPS.
* **Task 7:** (Optional/Cool Factor) Add a "Side-by-Side" mode where the original and stylized frames are displayed together.

### Phase 4: Documentation & Testing (Sunday PM)
* **Task 8:** Run benchmarks (Inference time per frame) to include in the project report.
* **Task 9:** Finalize the README with "How to Run" instructions.

---

## Setup and Usage

### Prerequisites
* Python 3.10+
* OpenCV 4.x
* Pre-trained models (place in `/models` directory)

### Running the Application
To start the live webcam stylization, execute:
```bash
python main.py --source webcam --style starry_night
```
To process a video file:
```bash
python main.py --source input_video.mp4 --style mosaic --output result.mp4
```

## Datasets
* **Training (Reference):** This project uses weights pre-trained on the **COCO 2014 Training Dataset**.
* **Validation:** Standard test videos (e.g., Big Buck Bunny) or live user-captured data.

---

### Instructions for Claude Code Max
To begin, run this command in your terminal with Claude Code:

> "Claude, initialize a project named LiveArt. Use the provided README to create a folder structure. Start by writing `utils.py` for image preprocessing and a `download_models.sh` script to fetch pre-trained weights for Starry Night and Mosaic styles from public URLs."