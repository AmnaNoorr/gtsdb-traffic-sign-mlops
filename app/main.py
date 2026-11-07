import io
from typing import List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field
from ultralytics import YOLO

app = FastAPI(
    title="GTSDB Traffic Sign Recognition API",
    description="Production-grade FastAPI serving pipeline for YOLOv8 Traffic Sign Detection",
    version="1.0.0"
)

# Load trained YOLOv8 model weights
MODEL_PATH = "weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception:
    model = None


# --- Pydantic Schemas for Strict Input/Output Validation ---
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class DetectionResult(BaseModel):
    class_id: int = Field(..., description="GTSDB detected class ID")
    class_name: str = Field(..., description="Human-readable sign category")
    confidence: float = Field(..., description="Model confidence score (0.0 to 1.0)")
    box: BoundingBox


class PredictionResponse(BaseModel):
    filename: str
    total_detections: int
    predictions: List[DetectionResult]


# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running"}


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model weights failed to load."
        )
    return {"status": "healthy", "model_weights": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_traffic_sign(
    file: UploadFile = File(...), 
    conf_threshold: float = 0.25
):
    # Industry Constraint: Input Schema Validation (HTTP 400 Bad Request on invalid format)
    allowed_extensions = ["jpg", "jpeg", "png", "ppm"]
    extension = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format '{extension}'. Allowed image formats: {allowed_extensions}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be decoded as a valid image."
        )

    # Perform YOLOv8 inference directly on PIL RGB image
    results = model.predict(source=image, imgsz=960, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        
        detections.append(DetectionResult(
            class_id=cls_id,
            class_name=str(cls_name),
            confidence=round(conf, 4),
            box=BoundingBox(
                x1=round(coords[0], 2),
                y1=round(coords[1], 2),
                x2=round(coords[2], 2),
                y2=round(coords[3], 2)
            )
        ))

    return PredictionResponse(
        filename=file.filename,
        total_detections=len(detections),
        predictions=detections
    )