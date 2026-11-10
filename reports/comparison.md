| Modele                                       | mAP50-95 test | mAP50 test | Taille (Mo) | Latence CPU (ms) |
| -------------------------------------------- | ------------- | ---------- | ----------- | ---------------- |
| Faster R-CNN MobileNetV3 (baseline, PyTorch) | 0.771         | 0.978      | -           | 24.9 (cuda)      |
| YOLO11n (PyTorch)                            | 0.849         | 0.991      | -           | 2.2 (GPU)        |
| YOLO11n ONNX fp16                            | 0.843         | 0.991      | 5.4         | 80.5 (p95 104.8) |
| YOLO11n ONNX fp32                            | 0.842         | 0.991      | 10.6        | 64.2 (p95 71.9)  |
| YOLO11n ONNX int8                            | 0.696         | 0.867      | 3.2         | 76.0 (p95 98.6)  |
