# GTSDB Traffic Sign Recognition & MLOps Pipeline

An end-to-end computer vision serving pipeline utilizing **YOLOv8s**, **FastAPI**, **Pydantic**, **Streamlit**, and **Docker**.

## Project Architecture
- `app/main.py`: FastAPI REST endpoint with Pydantic schema validation (HTTP 400 rejection).
- `frontend.py`: Interactive Streamlit web interface.
- `tests/`: Automated unit test suite run via Pytest & GitHub Actions.
- `Dockerfile`: Container configuration for reproducible deployment.

## Quickstart

### 1. Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI Backend
uvicorn app.main:app --reload --port 8000

# Start Streamlit Frontend (in separate terminal)
streamlit run frontend.py