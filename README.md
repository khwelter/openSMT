[Behind the scenes](behindthescenes.md)

# openSMT

openSMT is a Python-based control stack for an SMT pick-and-place system.
The current production path is:

- backend runtime: `opensmt run`
- operator UI: native Qt control client (`opensmt control-gui`)
- transport: HTTP API served by the runtime

The legacy SCPI broker/monitor tools are still available for development,
but they are not required for the main machine workflow.

## Current Feature Set

- Native Qt control GUI for machine operation (no browser required)
- Camera thumbnails, XY jog, nozzle actions, feeder workflows
- Setup editors for cameras and special positions
- Package, part, nozzle-tip and nozzle editors via popup dialogs
- Dynamic nozzle-tip compatibility matrix in package editor
- Tray feeder part field with type-ahead selection list
- JSON include-based system configuration (`$include`)
- SQLite catalog persistence for packages, parts, feeders
- Runtime status indicator in GUI footer for active catalog DB and row counts
- Absolute motion enforcement on board move commands (`G90` before motion)

## Runtime Architecture

`python3 -m opensmt run --config config/examples/system.json` starts:

- serial board connections
- `HardwareDriver`
- `PositionStore`, `LocationStore`
- `CameraVisionModule` (HTTP API + camera handling)

`python3 -m opensmt control-gui --host 127.0.0.1 --port 8080` starts
the operator client that polls and commands the runtime.

## Quick Start

1. Set Python path in a source checkout

```bash
export PYTHONPATH=src
```

2. Review config and hardware ports

- main config: `config/examples/system.json`
- board ports: `boards.*.device`
- API bind: `camera.web_host`, `camera.web_port`

3. Run backend

```bash
python3 -m opensmt run --config config/examples/system.json
```

4. Run Qt client

```bash
python3 -m opensmt control-gui --host 127.0.0.1 --port 8080
```

## Configuration Layout

The example config is split into modular chunks and merged via `$include`.
Important files:

- `config/examples/system.json` (entrypoint)
- `config/examples/system.boards.json`
- `config/examples/system.driver.json`
- `config/examples/system.locations.json`
- `config/examples/nozzles.json`
- `config/examples/camera/camera.core.json`
- `config/examples/camera/camera.cameras.json`
- `config/examples/camera/camera.pipelines.json`
- `config/examples/feeders.json`
- `config/examples/parts.json`

## SQLite Catalog (Packages / Parts / Feeders)

Current primary persistence for catalog data is SQLite:

- default DB: `config/examples/catalog.sqlite`
- tables: `packages`, `parts`, `feeders`

Runtime behavior:

- packages bootstrap from `config/examples/packages/*.json` when DB is empty
- parts bootstrap from `config/examples/parts.json` when DB is empty
- feeders bootstrap from merged config/feeders JSON when DB is empty

The Qt control footer shows the active DB and row counts.

## Qt Control GUI Overview

- Cameras pane
  - camera selector, light control, resolution calibration, vector moves
- XY pane
  - jog, homing, park/dispose/fiducial/calibration shortcuts
- Nozzles pane
  - home/move/rotate/park/standard-down/vacuum/camera alignment actions
- General Purpose tabs
  - Setup
    - Cameras
    - Special Positions
  - Production (reserved)
  - Parts and Packages
    - Packages (popup editor)
    - Parts (popup editor)
    - Nozzle Tips (popup editor)
    - Nozzles (popup editor)
  - Feeders
    - feeder survey
    - tray feeder editor with part type-ahead selection
  - Diagnostics Log

## HTTP API (Current Operator Path)

Primary routes used by the Qt control client:

- `GET /api/status`
- `GET /thumb/{name}`
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
- `POST /api/config/location/{name}`
- `POST /api/config/nozzle/{name}`
- `GET /api/feeders`
- `POST /api/feeders`
- `GET /api/feeders/{feeder_id}`
- `PUT /api/feeders/{feeder_id}`
- `POST /api/feeders/{feeder_id}/reset`
- `POST /api/feeders/{feeder_id}/advance-pick`
- `POST /api/head/nozzle/{name}/move`
- `POST /api/head/nozzle/{name}/move-absolute`
- `POST /api/head/nozzle/{name}/move-standard-down`
- `POST /api/head/nozzle/{name}/rotate`
- `POST /api/head/nozzle/{name}/home`
- `POST /api/head/nozzle/{name}/park`
- `POST /api/head/nozzle/{name}/vacuum`
- `POST /api/nozzle/{name}/move-to-camera`
- `POST /api/nozzle/{name}/move-to-bottom-camera`
- `POST /api/nozzle/{name}/move-camera-here`
- `POST /api/nozzle/{name}/calculate-offset-top`
- `POST /api/camera/{name}/light`
- `POST /api/camera/{name}/settings`
- `POST /api/camera/{name}/calibrate-resolution`

## Motion Mode Policy

The system is intended to run in absolute positioning mode only.
Board motion now explicitly sets absolute mode before move commands.

## Legacy CLI Tools (Optional)

These remain available for development and migration:

- `python3 -m opensmt broker --host 127.0.0.1 --port 8765`
- `python3 -m opensmt monitor --host 127.0.0.1 --port 8765 --name MONITOR`
- `python3 -m opensmt monitor-gui --host 127.0.0.1 --port 8765 --name MONITOR_QT`

## License

Public Domain (Unlicense), see `LICENSE`.
