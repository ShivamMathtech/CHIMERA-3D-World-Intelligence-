# Architecture and data flow

The backend uses Python's threaded HTTP server on IPv4 loopback. The frontend uses native Canvas 2D with a JavaScript perspective-projection renderer. It requires no WebGL driver, Three.js download, external fonts, CDN, Node build, or web framework. This deliberately keeps the source distribution small and offline-capable.

## Processing

1. Source: deterministic synthetic RGB-D, OpenCV webcam/video, or aligned RGB-D NPZ.
2. Optional detection: local YOLO `.pt` checkpoint on CPU. Demo uses known synthetic labels.
3. Depth: provided metres for RGB-D; optional normalized inverse-depth estimates for RGB video.
4. Class-aware greedy one-to-one IoU assignment (threshold 0.12); tracks expire after 2.5 elapsed seconds without detections.
5. Visible object localization using 7×7 median valid depth near the detection center.
6. Sampled pinhole back-projection; adaptive sampling limits point count to approximately 14,000.
7. Two triangles per valid depth-grid cell; reject triangles spanning a relative discontinuity greater than 8% of mean Z, with a 0.1 unit floor.
8. Publish PNG previews, JSON state/cloud and exports. Browser state polls every 500 ms, rendered views run at at most 20 Hz.
9. Save observed events and track positions once per second to SQLite. Histories in live state are bounded to 160 observations.

## Concurrency

One engine worker owns frame processing. Source changes, settings, snapshot construction and exports acquire a re-entrant lock. Model loading holds this lock; the UI waits for the load to finish. HTTP requests run on daemon threads. Large model loading is therefore visible as a temporary pause, not asynchronous queued progress. Camera reads may be device-dependent and cannot always be interrupted immediately.

Pause does not disconnect/release the camera; it stops reading frames. End-of-file pauses a video. Reset starts a new session and rewinds the existing video capture. Webcam failures produce an error rather than reverting silently to simulated data.

## Data tables

- `sessions(id, started, source)`
- `events(id, session, time, kind, message)`
- `samples(id, session, time, track, label, x, y, z, confidence)`

SQLite WAL mode and a store-level lock serialize writes. Session IDs disambiguate reused track IDs. Samples without depth retain null coordinates. CSV rows contain elapsed session seconds. Events use UTC ISO timestamps; the frontend displays local time. Historical exports are trajectory samples, not full video replay archives.

## Local boundary

The server binds to `127.0.0.1`. Host validation prevents DNS rebinding via arbitrary Host values. Mutating browser requests enforce same-origin policy; there are no permissive CORS headers. Static file resolution rejects path traversal. Upload names are replaced by random UUIDs. NPZ loading disables pickled arrays, validates shapes/types and checks expanded archive size. This is local single-user software, not an authenticated internet service.

## Extension points

A future registered mapping implementation should retain intrinsics, synchronized depth and camera extrinsics, transform each frame into a shared world frame, reject dynamic pixels and fuse into a voxel/TSDF map. Such a system requires explicit calibration and drift evaluation; the current camera-frame cloud must not be described as SLAM.

For stronger tracking, substitute an independently tested tracker behind the `Tracker.update` interface. For sensor integration, add a source adapter that yields registered RGB, metric depth, intrinsics and timestamps. Sensor-specific alignment, synchronization and calibration remain responsibilities of that adapter.
