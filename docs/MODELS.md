# Optional model setup

No pretrained weights are bundled. The default demo uses synthetic scene geometry and annotations, and does not claim neural inference. The application never downloads weights automatically when it starts. `scripts/download_models.py` is the explicit download command.

## YOLO detector

- Default helper checkpoint: `yolov8n.pt`, downloaded through Ultralytics.
- Adapter calls `YOLO(path).predict` with RGB converted into OpenCV BGR order, CPU device, image size 480 and configurable confidence.
- Class names, confidence and XYXY boxes come from the model's returned results. Categories depend on the checkpoint's training data; do not assume drone or arbitrary custom classes are present.
- Reference: https://docs.ultralytics.com/modes/predict/
- License information: https://www.ultralytics.com/license
- Review Ultralytics' AGPL-3.0 / Enterprise options before redistribution or commercial deployment. The MIT license of this project's original code does not change upstream requirements.

## Depth Anything Small

- Helper model: `LiheYoung/depth-anything-small-hf`.
- Model card: https://huggingface.co/LiheYoung/depth-anything-small-hf
- Transformers API: https://huggingface.co/docs/transformers/model_doc/depth_anything
- Loads through the `depth-estimation` pipeline from a local directory.
- Uses the raw predicted tensor, resizing to RGB resolution. Normalizes inverse depth between the 2nd and 98th percentiles and maps it into a 1–5 relative display range.
- Those values are intentionally labeled relative units. They do not represent measured depth, metric motion, calibrated reconstruction or stable scene scale between frames.
- Consult the model card and repository license for the particular downloaded revision. Model weights, framework and project licenses are separate.

## Installing and debugging

Use Python 3.12 x64. Keep one OpenCV distribution installed. PyTorch/Torchvision must be mutually compatible. `requirements-ai.txt` uses supported major-version ranges; unlike a frozen environment, those ranges permit future patch releases. After a successful installation, capture your exact environment using:

```bash
python -m pip freeze > installed-environment.txt
```

Use official PyTorch CPU installation instructions if your pip resolver selects a large CUDA distribution. Stop the server before installing/uninstalling packages. Restart after dependency changes.

The model adapters follow the documented APIs but were not run with actual neural weights in the delivery environment. Hardware-specific camera behavior and neural model performance must be verified on your machine. No accuracy or FPS benchmark is promised.
