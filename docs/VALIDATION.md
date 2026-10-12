# Delivery validation

## Tests executed

Execution environment: Linux, Python 3.12, NumPy 2.3.5, Pillow, OpenCV 4.11.0.86. These checks do not constitute a benchmark on the user's Windows hardware.

`python -m unittest discover -s tests -v`: **8 tests passed**, no skips after installing OpenCV.

1. Pinhole projection matches known 3D coordinates; flat-depth grid produces expected triangle count; PLY vertex/face declarations agree.
2. Invalid/non-finite depth is excluded; discontinuity-spanning triangles are rejected; invalid focal length fails validation.
3. Track IDs persist under small motion, class mismatch creates a separate ID, stale tracks expire.
4. Association remains one-to-one and history is capped at 160 samples.
5. Demo produces geometry and tracks; pause stops frame advancement; reset starts a new session; PNG/PLY/CSV endpoints return data.
6. Unsupported source/settings/control requests are rejected; cross-origin writes and static path traversal are blocked.
7. An aligned RGB-D NPZ uploads, reconstructs in metric mode and pauses as a snapshot; invalid shape is rejected.
8. A real encoded MJPG video is decoded through OpenCV, reaches EOF, pauses, and rewinds/replays. With no AI model, it correctly reports no semantic tracks or depth geometry.

`node --check web/app.js`: passed.

A DOM/Canvas integration harness also passed: frontend initialized from live backend state, rendered point/mesh/wire modes, navigated all five views, opened/closed Settings, and paused/resumed processing through the HTTP API. The scene canvas was rendered to a PNG and visually inspected. This does not replace a full browser/CSS layout test.

`python -m compileall -q .`: passed.

## Verification limits

- Chromium installation was attempted for browser smoke/visual testing. The available runtime blocked Chromium launch through a socket permission restriction. No successful visual browser QA is claimed.
- Physical webcam access, Windows launch behavior, GPU use and hardware FPS were not verified in this environment.
- Neural inference adapters were implemented against official Ultralytics and Transformers API documentation, but actual YOLO/Depth Anything weights were not loaded or evaluated here.
- There are no claims of measured reconstruction accuracy, detection accuracy, identity accuracy or real-time throughput on the user's machine.

## Suggested on-machine acceptance check

1. Launch with `start_windows.bat`, verify the synthetic badge, point cloud, map and event list.
2. Orbit/zoom, change rendering modes, open each navigation view and inspect the layout at your display resolution.
3. Pause and verify source frames stop; resume; reset and check the session ID changes.
4. Upload `samples/room.npz`: check point geometry appears, units are metres and the snapshot pauses.
5. Open the included synthetic MP4 with camera dependencies installed: verify frames update and EOF pauses.
6. Download PLY and open it with a viewer such as MeshLab or CloudCompare; confirm it is a visible-surface mesh in camera coordinates.
7. Install optional AI dependencies and trusted weights, then test a real image/video containing categories that the detector was trained to recognize.
8. Test the physical webcam, camera permissions, CPU load and expected processing rate on the target machine.
