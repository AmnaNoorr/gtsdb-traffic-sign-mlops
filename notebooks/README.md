# GTSDB Model Training & Evaluation Notebooks

This folder contains the complete Google Colab training workflow for the German Traffic Sign Detection Benchmark (GTSDB) dataset using YOLOv8.

## Key Experiments & Benchmarks

| Model Variant | Resolution | Dataset Split | mAP@0.5 | mAP@0.5:0.95 | Latency (T4 GPU) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | 960x960 | 43 Classes | ~0.299 | ~0.255 | 8.2 ms | Benchmarked |
| **YOLOv8s** | 960x960 | 43 Classes | ~0.319 | ~0.281 | 14.7 ms | **Selected for Serving** |
| **YOLOv8m** | 960x960 | 43 Classes | Evaluated | Evaluated | ~22.5 ms | Benchmarked |

## Multi-Sign Label Fix
* Corrected the label parsing pipeline to ensure multi-sign bounding boxes per image are preserved across training and validation splits (`720` train images / `696` boxes, `180` val images / `170` boxes).