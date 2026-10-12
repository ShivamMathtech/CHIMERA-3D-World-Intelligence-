# Local HTTP API

Default origin: `http://127.0.0.1:8765`. JSON responses use standard JSON; non-finite coordinates are excluded before serialization. Error responses use `{"error":"message"}` and a non-2xx HTTP status.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | Source, models, tracks, events, metrics, session ID and settings |
| GET | `/api/cloud` | `points` rows `[x,y,z,r,g,b]`, indexed triangular `faces`, units and frame counter |
| GET | `/api/frame/rgb` | Most recent PNG source image |
| GET | `/api/frame/depth` | Most recent PNG depth visualization; black if unavailable |
| GET | `/api/frame/tracks` | Most recent annotated PNG |
| GET | `/api/sessions` | Latest 100 persisted sessions |
| GET | `/api/export/scene.ply` | Current point cloud + surface mesh; 400 if no depth geometry |
| GET | `/api/export/tracks.csv?session=ID` | Persisted track samples; current session by default |
| GET | `/api/export/session.json` | Current state snapshot |
| POST | `/api/source` | `{"kind":"demo"}` or `{"kind":"camera","index":0}` |
| POST | `/api/control` | `{"action":"pause"}`, `resume`, or `reset` |
| POST | `/api/settings` | Numeric processing settings and optional local model paths |
| POST | `/api/upload` | Raw binary body and `X-Filename` header, not multipart form data |

Settings example:

```json
{"stride":6,"confidence":0.35,"fps":8,"detector_path":"C:/models/yolov8n.pt","depth_path":"C:/models/depth-anything-small"}
```

Omit both model-path keys to change processing settings without reloading models. Include both keys when configuring models; an empty string disables that model. Source/control changes reset different state as described in README. The source API deliberately does not accept arbitrary server file paths or RTSP URLs.

Upload suffixes: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.npz`. Limit 256 MiB. General JSON request limit: 16 KiB. An NPZ archive must expand to at most 64 MiB. Image dimensions cannot exceed 3 megapixels.

PLY comments describe coordinate orientation. Unit interpretation comes from the current state; export session JSON alongside the PLY if provenance is important. RGB-D uses supplied metres; synthetic demo units are synthetic metres; monocular estimates use relative units.
