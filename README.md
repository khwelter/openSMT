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

The runtime loads one entry file and resolves `$include` recursively.

Primary entrypoint:

- `config/examples/system.json`
  - Contains the include list that assembles the full machine config:
    - `feeders.json`
    - `system.boards.json`
    - `system.driver.json`
    - `system.locations.json`
    - `parts.json`
    - `nozzles.json`
    - `camera/camera.core.json`
    - `camera/camera.cameras.json`
    - `camera/camera.pipelines.json`

### Core Machine Config Files

- `config/examples/system.boards.json`
  - Top-level key: `boards`
  - One object per board (`XY`, `AB`, `CD` in the example)
  - Per-board serial settings:
    - `device`, `baudrate`, `bytesize`, `parity`, `stopbits`
    - `xonxoff`, `rtscts`, `dsrdtr`

- `config/examples/system.driver.json`
  - Top-level key: `driver`
  - Global motion settings:
    - `speed_factor`, `xy_slack_compensation_mm`
    - `default_velocity`, `default_homing_velocity`
  - Axis mapping:
    - `axes[]` entries with `axis`, `board`, `gcode_letter`
  - Per-axis limits/speeds:
    - `axis_velocity`, `homing_velocity`
  - Group homing:
    - `home_groups` (for example `XY`, `Z1Z2`, `Z3Z4`)

- `config/examples/system.locations.json`
  - Top-level key: `locations`
  - Named XY presets (`park`, `dispose`, `nozzle_change`, `fiducial_main`, `fiducial_second`, `calibration_spot`)
  - Each location contains `X` and `Y`

- `config/examples/nozzles.json`
  - Top-level key: `camera.nozzle_tips` and `camera.nozzles`
  - `camera.nozzle_tips[]` entries:
    - `id`, `suction_hole_diameter_mm`, `component_min_mm`, `component_max_mm`
  - `camera.nozzles[]` entries:
    - identity/mechanics: `name`, `z_axis`, `min_z`, `max_z`, `safe_zone_z`
    - setup values: `tip_id`, `standard_down_z`, `offset_x`, `offset_y`
    - valve wiring: `vacuum_valve` and optional `air_valve` (`board`, `io_type`, `pin`)

- `config/examples/feeders.json`
  - Top-level key: `feeders`
  - Each feeder entry includes:
    - identity/type: `feeder_id`, `feeder_type`, `manufacturer_part_number`
    - pick data: `pick_location` (`x`, `y`), `pick_height`
    - optional tray geometry (`type_data`) and runtime tray state (`actual_data`)

- `config/examples/parts.json`
  - Top-level key: `parts`
  - Seed list for parts catalog (can be empty)

### Camera Config Files

- `config/examples/camera/camera.core.json`
  - Top-level key: `camera`
  - API server binding for runtime camera module:
    - `web_host`, `web_port`

- `config/examples/camera/camera.cameras.json`
  - Top-level key: `camera.cameras`
  - Per-camera entries include:
    - identity and source: `name`, `device`, `fps`
    - calibration: `resolution_dpcm_x`, `resolution_dpcm_y`
    - orientation: `flip_horizontal`, `flip_vertical`, `rotation_deg`
    - optional camera XY mount location: `x`, `y`
    - light mapping: `lights.<key>.board`, `index`, `on_value`
    - allowed pipelines: `pipelines[]`

- `config/examples/camera/camera.pipelines.json`
  - Top-level key: `camera.pipelines`
  - Named pipeline definitions:
    - each entry has `name`, `type`

### Catalog and Runtime Persistence Files

- `config/examples/catalog.sqlite`
  - Backend catalog database (authoritative for runtime catalog entities)
  - Tables include:
    - `packages`, `parts`, `feeders`, `pcbs`, `panels`, `jobs`

- `config/examples/packages/*.json`
  - Package seed files used to bootstrap backend catalog when empty
  - Per file fields (example):
    - `name`, `footprint`, `length_mm`, `width_mm`, `height_mm`, `pin_count`

- `config/examples/feeders/*.json`
  - Per-feeder persistence files (one file per feeder id)
  - Same feeder schema as in `feeders.json`
  - Loaded on backend startup and used for feeder state persistence

### Legacy / Support Config Files

- `config/examples/broker.json`
  - Legacy SCPI broker host/port (`broker.host`, `broker.port`)

- `config/examples/playback.txt`
  - Legacy monitor playback script with command lines and `SLEEP` statements

## Persistence Behavior Summary

- Location edits persist on backend to locations store path (defaults to `config/examples/system.locations.json`)
- Camera settings and calibration persist on backend to camera settings path (defaults to `config/examples/camera/camera.cameras.json`)
- Nozzle and nozzle-tip edits persist on backend to nozzle config path (defaults to `config/examples/nozzles.json`)
- Feeders persist on backend to feeder files directory (defaults to `config/examples/feeders/`)
- Catalog entities (packages, parts, pcbs, panels, jobs, feeders) persist on backend in `config/examples/catalog.sqlite`

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
