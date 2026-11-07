# GTSDB Traffic Sign Recognition & MLOps Pipeline

An end-to-end computer vision serving pipeline utilizing **YOLOv8s**, **FastAPI**, **Pydantic**, **Streamlit**, and **Docker**.

## Project Architecture
- `app/main.py`: FastAPI REST endpoint with Pydantic schema validation (HTTP 400 rejection).
- `frontend.py`: Interactive Streamlit web interface.
- `tests/`: Automated unit test suite run via Pytest & GitHub Actions.
- `Dockerfile`: Container configuration for reproducible deployment.


## 📊 Model Training & Performance Evaluation

The **YOLOv8s** model was trained for 100 epochs on the German Traffic Sign Detection Benchmark (GTSDB) dataset at $960 \times 960$ resolution.

### 1. Training Convergence & Validation Metrics
Bounding box losses (`train/box_loss`), classification losses (`train/cls_loss`), and distribution focal losses (`train/dfl_loss`) showed smooth convergence without significant overfitting across 100 epochs.

<img width="450" height="224" alt="image" src="https://github.com/user-attachments/assets/bc5390ae-adbd-438e-ad90-fe6dd636169f" />

| Evaluation Metric | Score | Description |
| :--- | :--- | :--- |
| **mAP_50** | **0.3082** | Mean Average Precision at IoU threshold 0.50 |
| **mAP_50-95** | **0.2827** | Mean Average Precision across IoU thresholds 0.50 to 0.95 |
| **Precision** | **28.09%** | Correct positive predictions out of total predicted positives |
| **Recall** | **32.35%** | Correct positive predictions out of total ground truth signs |
| **Inference Latency** | **15.7 ms** | Per-image processing speed on Tesla T4 GPU |

---

### 2. Class-Level Confusion Matrix
The confusion matrix highlights strong classification accuracy on high-frequency signs (`stop`, `no_entry`, `keep_right`) while identifying class imbalance challenges on lower-frequency warning indicators.

<img width="558" height="422" alt="image" src="https://github.com/user-attachments/assets/95164fd5-65e1-41fc-b65c-94c4eb19d1d1" />



## Quickstart

### 1. Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI Backend
uvicorn app.main:app --reload --port 8000

# Start Streamlit Frontend (in separate terminal)
streamlit run frontend.py
