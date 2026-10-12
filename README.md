# CHIMERA — 3D World Intelligence Workbench

A runnable, local Python computer-vision project with a browser dashboard inspired by the supplied CHIMERA reference. It includes backend processing, an interactive 3D renderer, camera/video ingestion, optional real AI inference, RGB-D reconstruction, tracking, persistence, exports, sample data and tests.

**Start here:** use **Python 3.12 (64-bit)**. The synthetic demo runs without a GPU, model weights, API keys, Node.js, or a frontend build. Python 3.11 is also supported. Python 3.13/3.14 are not the recommended target for the optional ML stack.

## Windows: easiest start

1. Extract the ZIP completely into a writable folder such as `G:\major_projects\Chimera`.
2. Install Python 3.12 from https://www.python.org/downloads/ if needed. Enable the Python launcher during installation.
3. Double-click `start_windows.bat`.
4. The launcher creates `.venv`, installs the small base requirements if absent, and opens **http://127.0.0.1:8765**.
5. Keep the terminal running. Close the browser freely; stop the backend with Ctrl+C.

The first dependency installation requires internet. Later demo/RGB-D launches work offline.

### Manual Windows PowerShell setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

No PowerShell activation-policy changes are necessary: these commands invoke the virtual environment directly.

### Linux / macOS

```bash
python3 --version  # use 3.11 or 3.12
bash start_linux.sh
```

Alternatively create a venv, install `requirements.txt`, and run `python run.py`. Some Linux distributions require their `python3-venv` package. Use `python run.py --port 8766` if port 8765 is occupied. `--no-browser` disables auto-opening.

## What is working

| Capability | Implementation / operating requirement |
|---|---|
| Reference-inspired dashboard | Dark blue/cyan layout; central 3D view; source previews; object counts; map; trajectories; events; settings; navigation |
| Offline synthetic room | Deterministic ray-cast RGB-D scene with moving objects, known depth and explicitly labeled ground-truth annotations |
| Interactive 3D | Drag to orbit, wheel to zoom, reset view, auto rotation, labels, screenshot download |
| Point cloud | Calibrated pinhole back-projection of aligned RGB and depth |
| Surface mesh / wireframe | Triangles between neighboring depth samples, with depth-discontinuity rejection |
| Webcam / uploaded video | OpenCV input; real decoded RGB frames; honest no-model state |
| Real object detection | Optional local Ultralytics YOLO `.pt` checkpoint; CPU inference |
| Persistent IDs | Class-aware IoU association, lost-track expiry and bounded histories |
| Monocular depth | Optional local Depth Anything Small Hugging Face checkpoint; relative depth visualization |
| Spatial localization | Median depth around bounding-box center, camera-space XYZ |
| Spatial map / trajectories | Derived X–Z map and per-track X-position histories |
| Storage | SQLite sessions, events and one-second track samples, persisted in `data/` |
| Export | Current scene PLY (vertices + faces), session track CSV, current state JSON, scene screenshot |
| Source control | Demo, indexed webcam, video upload, RGB-D NPZ upload, pause/resume, reset/replay |
| Validation | Standard-library unit/integration tests; no test framework dependency |

**Data honesty:** the demo does not run a neural detector: its known annotations are marked as synthetic. Without a detector, real video shows no detections. Without a depth source, no 3D geometry is fabricated. Valid-depth coverage is not reconstruction accuracy. Accuracy is shown as “Not evaluated.”

## Use the dashboard

- **Live Feed:** inspect the source and reconstructed scene.
- **3D Reconstruction:** enlarges the reconstruction workspace. Choose Point Cloud, Mesh or Wireframe.
- **Tracking:** displays current IDs, class, confidence, camera-space position and history length.
- **Spatial Map:** enlarged top-down view of the visible scene in camera coordinates.
- **Analytics:** measured processing throughput, point count, active tracks and downloadable session history.
- **+ Source:** open the synthetic room, a webcam, a local video or aligned RGB-D snapshot.
- **Auto Scan:** rotates the visualization; it does not move a physical camera or run SLAM.
- **Pause / Resume:** stops/restarts frame processing. Orbit and zoom remain available.
- **Reset:** starts a new session, clears current track history, and rewinds uploaded video.
- **Settings:** configure model paths, stride, confidence and target FPS.
- **PLY / CSV / JSON:** exports scene geometry, persisted trajectory samples, or current state.

The four source panes are **one input plus derived depth, detections, and spatial views**, not four connected cameras. Only the latest source is processed. The demo has eight synthetic objects; counts reflect actual visible annotations, not the illustrative numbers in the reference.

## Enable webcam and video

With the application stopped, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt
.\.venv\Scripts\python.exe run.py
```

Open **+ Source → Open webcam**. Try index 0, then 1 if necessary. Grant Windows camera access to desktop apps and close competing camera applications. Video supports whatever codecs your OpenCV installation decodes; MP4/H.264 or the included synthetic MP4 are reasonable starting points.

Uploads are limited to **256 MB**. They are stored under `data/uploads/`. Upload a larger source only after reducing/compressing it externally. Decoding is sequential at the configured processing target: this is an analysis workbench, not a guaranteed real-time, synchronized video player. At end of file, processing pauses; Reset replays it. A slow AI model lowers the achievable processing rate.

## Enable real AI detection and relative depth

Model files and large ML packages are **not bundled** in this ZIP. Download them explicitly on your computer. See [docs/MODELS.md](docs/MODELS.md) for details and licenses.

If you previously installed the camera-only requirements, remove the headless OpenCV distribution before installing the AI requirements, since Ultralytics installs the desktop distribution:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python-headless
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
.\.venv\Scripts\python.exe scripts\download_models.py --detector --depth
.\.venv\Scripts\python.exe run.py
```

Open Settings, then paste the absolute paths printed by the download script, for example:

```text
YOLO checkpoint: G:\major_projects\Chimera\models\yolov8n.pt
Depth folder:    G:\major_projects\Chimera\models\depth-anything-small
```

Apply Settings, then open a webcam/video. Models are loaded on CPU. For detection without depth, leave the depth path blank. To disable all inference, leave both paths blank. Model loading is transactional: a failed load preserves the previously loaded models. Blank paths on a later Apply disable those models, so keep configured paths in the form when adjusting processing settings. Settings and model paths reset when the server restarts.

For a smaller CPU-only PyTorch install, follow PyTorch's official installer at https://pytorch.org/get-started/locally/ before installing `requirements-ai.txt`. Allow several GB of disk space for optional ML dependencies and weights. Use trusted `.pt` files only: checkpoint loading is not a sandbox.

## Calibrated RGB-D input

Upload `samples/room.npz` to test this pipeline without a camera. It is a **synthetic** sample with exact scene calibration. An NPZ filename does not itself prove sensor calibration; CHIMERA trusts the supplied scale/intrinsics.

Required NPZ arrays:

```python
import numpy as np
np.savez_compressed(
    'my_frame.npz',
    rgb=rgb_uint8,       # shape (H, W, 3), RGB order, dtype uint8
    depth=depth_meters,  # shape (H, W), numeric, metres; zero/NaN = invalid
    intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
)
```

Colour and depth must already be registered/aligned at identical resolution. If your sensor returns millimetres, convert to metres first. The point/mesh pipeline rejects depths ≤0.05 m or ≥20 m. Dimensions are limited to 3 megapixels and expanded NPZ content to 64 MB. Snapshot imports process once and then pause. Resume reprocesses the snapshot once, which is useful after loading a detector.

Coordinates: **X right, Y up, Z forward**, with the camera at the origin. Pinhole projection is `X=(u-cx)Z/fx`, `Y=-(v-cy)Z/fy`. Position is the visible surface near the box center, not the true 3D object centroid. Displayed 3D markers have heuristic dimensions; they are not estimated oriented 3D bounding boxes.

## Scope and limits

This is a working **local research/education workbench**, not a validated production perception system. The reference is a conceptual dashboard with future-model suggestions; this implementation makes a defined subset operational end to end.

- Reconstructs the visible depth surface. It does **not** infer unseen walls or create a complete watertight room model.
- No SLAM, camera pose estimation, multi-view registration, ORB-SLAM/COLMAP integration, calibrated multi-camera fusion or simultaneous multi-camera acquisition.
- No SAM segmentation or instance masks; semantic detection is bounding-box based.
- No direct RealSense/Kinect device SDK adapter. Supply aligned RGB-D via NPZ for metric processing.
- Monocular inverse-depth output is normalized per frame into relative display units. Distances and motions from that mode are **not metric** and can vary with scene composition. RGB-camera intrinsics are assumed, not calibrated.
- Simple IoU association can switch IDs after occlusion or fast motion. It is not ByteTrack, re-identification or identity recognition.
- No GPU acceleration toggle, training pipeline, RTSP source control, mobile client, cloud deployment, multi-user authentication or packaged Windows executable. Source code and Windows launcher are included.
- Single processing worker; inference, mesh generation and PNG encoding share that worker. Slow inference lowers frame rate. Point count is capped via adaptive sampling near 14,000 points.
- The loopback server is for one local user. It is deliberately not exposed on your network. Database and uploaded videos remain until you remove `data/` with the app stopped.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The camera/video integration test automatically skips if OpenCV is not installed. See [docs/VALIDATION.md](docs/VALIDATION.md) for tests actually run for this release and remaining verification limits.

## Project structure

```text
run.py                     Local launcher and loopback server startup
start_windows.bat          Windows environment/setup launcher
start_linux.sh             Linux/macOS launcher
requirements*.txt          Base, camera, and optional AI dependencies
chimera/
  server.py                HTTP API, upload validation, static frontend
  engine.py                Processing worker, source lifecycle, exports
  geometry.py              Back-projection, mesh and PLY serialization
  tracker.py               Class-aware IoU tracking
  models.py                Optional YOLO / Depth Anything adapters
  demo.py                  Synthetic RGB-D ray-cast room
  store.py                 SQLite session/event/sample persistence
web/                       Complete offline HTML/CSS/JavaScript frontend
scripts/                   Model downloader and sample generator
samples/                   Synthetic aligned RGB-D and sample video
tests/                     Unit and integration tests
docs/                      API, architecture, model and validation notes
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Python not found | Install Python 3.12, enable the launcher; run `py -3.12 --version`. |
| NumPy build / Rust / Visual Studio errors | Use Python 3.12 x64 in a fresh venv rather than Python 3.14. Upgrade pip. |
| Empty objects on real video | Install AI dependencies, download YOLO weights and configure the checkpoint. |
| Empty 3D view on real video | Load a depth model or upload calibrated RGB-D. RGB alone provides no measured depth. |
| Camera opens then stops | Check the visible error banner, Windows permissions, camera index and other camera users. Switch to Demo then reopen the camera. |
| Slow processing | Use detector-only mode, increase point stride, and reduce target FPS. Depth inference can dominate CPU time. |
| Model download fails | Check network/proxy settings; follow the official model links in docs/MODELS.md. |
| `cv2` import error | Use only one OpenCV distribution; uninstall both then reinstall the one needed by your chosen requirements file. |
| Page disconnected | Keep the Python terminal running; refresh the local URL. |
| Source file does not decode | Re-encode to a supported video codec; check the visible backend error. |
| Port already in use | Run `python run.py --port 8766`. |

App code is provided under the MIT license. Optional pretrained models and third-party packages retain their own licenses.
