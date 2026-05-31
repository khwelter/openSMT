# Frontend <-> Backend Communication (Qt Control + Runtime)

This document describes how the native Qt operator UI communicates with the runtime backend, including request/response patterns, asynchronous job execution, camera streaming, and common workflow sequences.

## 1. Components and Roles

- Frontend: Qt desktop client in `src/opensmt/monitor/qt_control.py`
- Backend: HTTP API module in `src/opensmt/modules/camera_vision.py`
- Runtime command queue: `src/opensmt/runtime/command_runner.py`
- Hardware abstraction: `src/opensmt/hardware/driver.py` and `src/opensmt/hardware/board.py`

High-level flow:

1. User clicks in Qt UI
2. Qt sends HTTP request to runtime
3. Runtime validates request and usually enqueues an async command
4. Runtime returns `status: accepted` and `job_id`
5. Qt polls status and/or job state until completion
6. Hardware driver sends G-code over serial and updates position/state stores

## 2. Transport and Encoding

Frontend transport is plain HTTP over local network.

- JSON API requests: `GET`, `POST`, `PUT`
- JSON API responses: object payloads
- Camera thumbnails: JPEG bytes via `GET /thumb/{name}`

Frontend API helper (`ControlApiClient`) behavior:

- Base URL configured from host/port in GUI
- `get_json(path, callback)`
- `post_json(path, payload, callback)`
- `put_json(path, payload, callback)`
- Callback signature: `(ok: bool, status: int, data: dict)`
- If response body is not valid JSON, frontend sets `{"error": "invalid_json_response"}`
- Non-2xx responses are treated as failure and surfaced to Diagnostics Log

## 3. Backend Routing Surface

Routes are registered in `_start_web()` of `CameraVisionModule`.

Main groups:

- Camera image:
  - `GET /thumb/{name}`

- Coordinate/motion:
  - `POST /api/coord/jog`
  - `POST /api/coord/home`
  - `POST /api/coord/home-xy`
  - `POST /api/coord/park`
  - `POST /api/coord/dispose`
  - `POST /api/coord/homing-fiducial-main`
  - `POST /api/coord/secondary-fiducial`
  - `POST /api/coord/nozzle-change`
  - `POST /api/coord/calibration-spot`
  - `POST /api/coord/set-home-here`
  - `POST /api/coord/set-calibration-spot-here`
  - `POST /api/coord/move-xy`
  - `GET /api/coord/positions`
  - `GET /api/coord/m114`

- Nozzle/head:
  - `POST /api/head/nozzle/{name}/move`
  - `POST /api/head/nozzle/{name}/move-absolute`
  - `POST /api/head/nozzle/{name}/move-standard-down`
  - `POST /api/head/nozzle/{name}/rotate`
  - `POST /api/head/nozzle/{name}/home`
  - `POST /api/head/nozzle/{name}/park`
  - `POST /api/head/nozzle/{name}/vacuum`

- Nozzle/camera alignment utilities:
  - `POST /api/nozzle/{name}/move-to-camera`
  - `POST /api/nozzle/{name}/move-to-bottom-camera`
  - `POST /api/nozzle/{name}/move-camera-here`
  - `POST /api/nozzle/{name}/calculate-offset-top`

- Camera settings:
  - `POST /api/camera/{name}/light`
  - `POST /api/camera/{name}/settings`
  - `POST /api/camera/{name}/calibrate-resolution`

- Feeders:
  - `GET /api/feeders`
  - `POST /api/feeders`
  - `GET /api/feeders/{feeder_id}`
  - `PUT /api/feeders/{feeder_id}`
  - `POST /api/feeders/{feeder_id}/reset`
  - `POST /api/feeders/{feeder_id}/advance-pick`

- Runtime/config/status:
  - `POST /api/config/location/{name}`
  - `POST /api/config/nozzle/{name}`
  - `GET /api/status`
  - `GET /api/jobs/{job_id}`

## 4. Asynchronous Job Model

Most movement and actuator requests are asynchronous.

### 4.1 Job submission pattern

Backend handlers call `_submit_domain_command(domain, name, command)`.

Return payload usually includes:

- `status: "accepted"`
- `job_id: "..."`
- `previous_job_canceled: <job id or null>`

### 4.2 Domain cancellation behavior

Before submitting a new command in a domain (for example `coord`, `head`, or `nozzle`), backend tries to cancel the latest active job in that same domain.

Implication:

- Commands within one domain are "latest-wins"
- Rapid repeated UI clicks can cancel prior in-flight job in that domain

### 4.3 Job lifecycle

`CommandRunner` states:

- `queued`
- `running`
- `succeeded`
- `failed`
- `canceled`

Each job tracks:

- `created_at`
- `started_at`
- `finished_at`
- `error`

Frontend can poll `GET /api/jobs/{job_id}` to get exact execution state.

## 5. Status Polling and UI Refresh

Qt starts a timer that polls `GET /api/status` every 800 ms.

Status response drives:

- X/Y coordinate display
- camera list/online state/resolution/flip/lights
- nozzle cards and valve state
- feeder table and selected feeder editor

Important:

- `/api/status` is operational state snapshot
- `/api/jobs/{job_id}` is execution-state truth for a specific command

## 6. Camera Rendering Path

Selected active camera is shown in the large camera panel.

Process:

1. Active camera name chosen in UI
2. Qt requests `GET /thumb/{name}`
3. Backend encodes latest processed frame to JPEG
4. Qt decodes and displays frame

Startup camera preference:

- Qt prefers `TOP` camera by default when available
- Falls back to first available camera otherwise

## 7. Homing and Motion Preconditions

Many XY routes enforce homing gate:

- If X or Y are not homed, backend returns `409` with `xy_not_homed`

Recent homing hardening in board/driver flow:

- Homing completion now uses explicit completion barriers and fallback querying
- Driver marks homed axes robustly even if firmware does not emit coordinate line in expected format

## 8. XY-then-Z Safety Sequencing

Frontend safety-sensitive flows (for example pick sequence) use a two-stage gate:

1. Wait for XY command job to reach completion (`/api/jobs/{job_id}`)
2. Confirm XY with live firmware query via `GET /api/coord/m114`

Only then frontend triggers Z movement.

This avoids relying on stale/cached position snapshots for inter-axis safety.

## 9. Typical End-to-End Flows

### 9.1 XY jog

1. Frontend `POST /api/coord/jog` with `{dx, dy}`
2. Backend validates homing and enqueues `driver.jog_xy`
3. Backend returns accepted + job_id
4. Driver sends board motion G-code and waits for completion
5. Position store updates
6. Frontend status poll updates coordinates

### 9.2 Move camera to XY

1. Frontend `POST /api/coord/move-xy` with `{x, y}`
2. Backend enqueues motion job and returns job_id
3. Frontend starts XY motion gate on that job
4. Frontend blocks dependent Z actions while XY gate active

### 9.3 Pick step (simplified)

1. Advance feeder (`/api/feeders/{id}/advance-pick`)
2. Move XY (`/api/coord/move-xy`) and get job_id
3. Wait for job completion + M114 target confirmation
4. Move nozzle Z down (`/api/head/nozzle/{name}/move-absolute`)
5. Vacuum on (`/api/head/nozzle/{name}/vacuum`)
6. Dwell
7. Raise Z

## 10. Error Propagation and Diagnostics

Frontend logs operation outcomes in Diagnostics Log tab.

Common signatures:

- `ERR 409 ... xy_not_homed`
- `ERR 0 ... Connection refused` (backend not reachable)
- API-level error payloads from backend validation

Troubleshooting rule:

- If Diagnostics shows connection errors, also inspect runtime terminal output because process-level exceptions are emitted there.

## 11. Persistence vs Runtime State

Two persistence channels are involved:

- Runtime machine state: in-memory stores (`PositionStore`, `LocationStore`, nozzle/valve runtime state)
- Catalog/business data: SQLite catalog DB used by GUI editors (packages, parts, feeders, plus newer production entities)

HTTP API primarily controls runtime state and motion, while Qt also persists editor entities through catalog interfaces.

## 12. Practical Integration Notes

- Treat `accepted` as "queued", not "completed"
- Use `job_id` for reliable completion checks
- Use `m114` for authoritative live XY when safety-critical
- Expect domain-level cancellation of prior jobs on rapid repeated command issue
- Keep homing status valid before sending XY routes
