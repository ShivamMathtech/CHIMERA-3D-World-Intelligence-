# Sample inputs

`room.npz` and `room_rgb.png` are generated from the synthetic ray-cast room. The NPZ contains registered RGB, depth in synthetic metres and exact camera intrinsics. Upload it through **+ Source**. A static RGB-D snapshot processes once and pauses. Without YOLO, it reconstructs geometry but produces no semantic detections.

`synthetic-room.mp4`, when present, tests video decoding, pause/resume and end-of-file replay. The coloured geometric objects are not realistic YOLO test images and do not establish detection accuracy.

Regenerate with `python scripts/make_sample.py`. Add `--video` to also create the 8-second test clip; this needs OpenCV.
