import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

st.set_page_config(page_title="GTSDB Traffic Sign Recognition", layout="wide")

st.title("🚥 GTSDB Traffic Sign Detection")
st.write("Upload an image to detect traffic signs.")

# Sidebar Controls
st.sidebar.header("Inference Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

# File Uploader Input Widget
uploaded_file = st.file_uploader(
    "Choose a traffic sign image...", 
    type=["jpg", "jpeg", "png", "ppm"]
)

if uploaded_file is not None:
    # Display split layout
    col1, col2 = st.columns(2)
    
    # Read PIL Image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    # Trigger Prediction
    if st.button("Run Traffic Sign Detection", type="primary"):
        with st.spinner("Connecting to FastAPI backend..."):
            try:
                # Prepare payload for FastAPI /predict endpoint
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                params = {"conf_threshold": conf_threshold}
                
                response = requests.post(
                    "http://127.0.0.1:8000/predict", 
                    files=files, 
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    detections = data.get("predictions", [])
                    
                    # Draw bounding boxes on image
                    annotated_image = image.copy()
                    draw = ImageDraw.Draw(annotated_image)
                    
                    for det in detections:
                        box = det["box"]
                        draw.rectangle(
                            [box["x1"], box["y1"], box["x2"], box["y2"]], 
                            outline="red", 
                            width=4
                        )
                        label = f"{det['class_name']} ({det['confidence']:.2f})"
                        draw.text((box["x1"], max(0, box["y1"] - 10)), label, fill="red")
                    
                    with col2:
                        st.subheader("Detections Output")
                        st.image(annotated_image, use_column_width=True)
                        st.success(f"Detected {data['total_detections']} traffic sign(s)")
                        st.json(detections)
                else:
                    st.error(f"API Error {response.status_code}: {response.json().get('detail', 'Prediction failed')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Ensure Uvicorn is running on http://127.0.0.1:8000")