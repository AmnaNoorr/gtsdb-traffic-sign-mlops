# GTSDB Traffic Sign Recognition: MLOps Pipeline

An end-to-end pipeline for detecting and classifying German traffic signs in full-resolution road scenes. A **YOLOv8s** detector trained on the **German Traffic Sign Detection Benchmark (GTSDB)** is served through a **FastAPI** REST API with strict **Pydantic** input validation, packaged with **Docker**, explored through a **Streamlit** web UI, and tested automatically with **Pytest** on every push through **GitHub Actions**.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Model Training](#model-training)
4. [Results](#results)
5. [Limitations](#limitations)
6. [Project Structure](#project-structure)
7. [Quickstart](#quickstart)
8. [Docker](#docker)
9. [API Reference](#api-reference)
10. [Testing and CI/CD](#testing-and-cicd)
11. [Future Work](#future-work)

---

## Overview

Real driving footage does not contain neatly cropped signs. A detector has to find small signs inside a large, cluttered frame and classify them at the same time, fast enough to be useful. This project covers the full path from model to service:

- **Detection**: bounding boxes and class labels for 43 GTSDB sign classes, trained on full 1360x800 scenes (not cropped signs).
- **Speed-aware evaluation**: accuracy (mAP) is reported together with latency and FPS.
- **Serving**: a FastAPI backend that returns detections as JSON and rejects malformed requests with `400 Bad Request`.
- **Packaging**: a Docker image so the service runs the same way on any machine.
- **Frontend**: a Streamlit app to upload an image, adjust the confidence threshold, and see the boxes drawn.
- **Automation**: GitHub Actions runs the test suite and a Docker build check on every push.

**Tech stack:** Python, PyTorch, Ultralytics YOLOv8, OpenCV, FastAPI, Uvicorn, Pydantic, Streamlit, Pytest, Docker, GitHub Actions.

---

## System Architecture

```text
   [ Image upload ]
          |
          v
 +----------------------+
 |  Streamlit Web UI    |--- HTTP POST /predict --+
 +----------------------+                         |
                                                  v
 +--------------------------------------------------------------+
 |                   FastAPI REST Server                        |
 |  +--------------------------------------------------------+  |
 |  | Input validation (file type, decodable image, params)  |  |
 |  | Invalid input -> HTTP 400 Bad Request                  |  |
 |  +--------------------------------------------------------+  |
 |                            |                                 |
 |                            v                                 |
 |              YOLOv8s inference (best.pt)                     |
 |              confidence threshold filter                     |
 +--------------------------------------------------------------+
                              |
                              v
        [ JSON: class, confidence, bounding box per sign ]
```

---

## Model Training

| Setting | Value |
| :--- | :--- |
| Architecture | YOLOv8s (about 11.1M parameters, 28.5 GFLOPs) |
| Initialization | COCO-pretrained `yolov8s.pt` (transfer learning) |
| Classes | 43 GTSDB sign classes |
| Input size | 960 px |
| Epochs | 100 |
| Batch size | 8 |
| Augmentation | Mosaic on, scale 0.5, horizontal flip **off** |
| Hardware | Tesla T4 GPU (Google Colab) |

**Design choices**

- **Full scenes, not crops.** Signs occupy only a few pixels at low resolution, so training uses a larger input (960 px) than the 640 px default.
- **No horizontal flipping.** Flipping would turn a "turn left ahead" sign into a "turn right ahead" sign and corrupt the labels.
- **Transfer learning.** With under 1,000 images, starting from COCO weights is essential.

The full training and evaluation workflow is in `notebooks/GTSDB_YOLOv8_Training.ipynb`.

---

## Results

Evaluated on a held-out validation split of **149 images**.

### Overall metrics

| Metric | Score |
| :--- | :--- |
| mAP@0.5 | 0.308 |
| mAP@0.5:0.95 | 0.283 |
| Precision | 0.281 |
| Recall | 0.324 |
<img width="450" height="224" alt="image" src="https://github.com/user-attachments/assets/bc5390ae-adbd-438e-ad90-fe6dd636169f" />
<img width="558" height="422" alt="image" src="https://github.com/user-attachments/assets/95164fd5-65e1-41fc-b65c-94c4eb19d1d1" />


### Speed (Tesla T4, per image)

| Stage | Time |
| :--- | :--- |
| Preprocessing | 6.5 ms |
| Model inference | 15.7 ms |
| Postprocessing (NMS) | 4.3 ms |
| **End to end** | **26.5 ms (about 38 FPS)** |

Inference alone is about 64 FPS. End-to-end is the more realistic figure because a deployed system also has to prepare each frame and filter overlapping boxes. Both are above the usual 25-30 FPS real-time bar. These timings come from the validation run, which processes images in batches; a strict batch-size-1 benchmark may differ slightly.

### Per-class behaviour

Performance depends heavily on how many training examples each class has.

| Group | Classes (validation instances) | mAP@0.5 |
| :--- | :--- | :--- |
| Detected well | speed_limit_100 (5), speed_limit_30 (9), keep_right (14), road_work (4), yield (8) | 0.59 to 0.89 |
| Detected partially | priority_road (8), no_overtaking_trucks (11), general_caution (4) | 0.32 to 0.45 |
| Not detected | turn_left_ahead (4), double_curve (2), slippery_road (2), children_crossing (1), bicycles_crossing (1), pedestrians (2), end_of_no_overtaking_trucks (1) | 0.00 |

Classes such as `stop`, `no_entry`, and `ice_snow` reach 0.995 mAP, but they have only 1 to 3 validation instances, so those scores are not statistically reliable and should not be read as perfect detection.


## Limitations

- **Small dataset.** GTSDB has about 900 full-scene images, and many classes have fewer than 10 examples. Rare classes are the main source of the low overall mAP.
- **Class imbalance.** Common classes (speed limits, keep right) dominate; rare warning signs are mostly missed.
- **Noisy per-class numbers.** With 1 to 5 validation instances per class, per-class scores swing widely.
- **Label conversion.** The validation split for this model contains exactly one annotated sign per image (149 images, 149 instances), which suggests multi-sign scenes were not fully converted from the original annotations. A later retraining with corrected labels (180 validation images, 170 instances) reached 0.319 mAP@0.5, but on a different validation set, so the two numbers are not directly comparable.
- **Domain gap.** The model is trained on German roads from a single dataset. A photo of a phone screen, for example, produced a correct but low-confidence detection.

---

## Project Structure

```text
gtsdb-traffic-sign-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions: tests + Docker build check
├── app/
│   └── main.py                          # FastAPI app, Pydantic schemas, endpoints
├── assets/
│   ├── results.png                      # Training curves
│   └── confusion_matrix.png             # Confusion matrix
├── notebooks/
│   └── GTSDB_YOLOv8_Training.ipynb      # Training and evaluation notebook
├── tests/
│   └── test_api.py                      # Pytest: health, valid input, 400 cases
├── weights/
│   └── best.pt                          # Trained YOLOv8s weights
├── Dockerfile                           # Container build
├── frontend.py                          # Streamlit app
├── requirements.txt
└── README.md
```

---


## Quickstart

### 1. Install

Requires Python 3.10+.

```bash
git clone https://github.com/AmnaNoorr/gtsdb-traffic-sign-mlops.git
cd gtsdb-traffic-sign-mlops

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

- Interactive docs (Swagger UI): http://127.0.0.1:8000/docs
- Health check: `GET /health`
- Prediction: `POST /predict`

### 3. Start the web interface

In a second terminal:

```bash
streamlit run frontend.py
```

Open http://localhost:8501, upload an image, adjust the confidence threshold, and view the detections.

---

## Docker

Build and run the API in a container:

```bash
docker build -t gtsdb-api .
docker run -p 8000:8000 gtsdb-api
```

The API is then available at http://localhost:8000, with docs at `/docs`.

---

## API Reference

### `GET /health`

Returns a simple status response confirming the service and model are loaded.

### `POST /predict`

| Part | Details |
| :--- | :--- |
| Body | `multipart/form-data` with an image file (`jpg`, `jpeg`, `png`, `ppm`) |
| Query parameter | `conf_threshold` (float, default `0.25`): detections below this confidence are discarded |

**Example request**

```bash
curl -X POST "http://localhost:8000/predict?conf_threshold=0.25" \
  -F "file=@sample_sign.png"
```

**Success response (200 OK)**

```json
{
  "filename": "sample_sign.png",
  "total_detections": 1,
  "predictions": [
    {
      "class_id": 14,
      "class_name": "stop",
      "confidence": 0.9421,
      "box": { "x1": 120.45, "y1": 85.12, "x2": 240.80, "y2": 205.30 }
    }
  ]
}
```

**Validation error (400 Bad Request)**

Invalid input is rejected before it reaches the model. Examples include an unsupported file type, a file that cannot be decoded as an image, or a non-numeric `conf_threshold`.

```json
{
  "detail": "Invalid file format 'txt'. Allowed image formats: ['jpg', 'jpeg', 'png', 'ppm']"
}
```

---

## Testing and CI/CD

### Run tests locally

```bash
python -m pytest tests/
```

The suite covers the health endpoint, a valid image returning detections, and invalid inputs returning `400`.

### GitHub Actions

On every push and pull request to `main`, the workflow in `.github/workflows/ci.yml`:

1. Sets up a clean Linux runner with Python.
2. Installs dependencies.
3. Runs the Pytest suite.
4. Builds the Docker image to confirm the container builds.

---

## Future Work

- **Group rare classes** into the four official GTSDB categories (prohibitory, danger, mandatory, other) for a stronger baseline.
- **Add data** (for example Mapillary Traffic Sign Dataset) and oversample rare classes.
- **Fix and re-evaluate labels** so multi-sign scenes are fully annotated, and compare all models on one common validation set.
- **Export to ONNX or TensorRT** and benchmark latency at batch size 1 with warm-up.
- **Track signs across video frames** to stabilize detections.
- **Push the Docker image** to a container registry from CI.

---

## License

Released under the MIT License.
