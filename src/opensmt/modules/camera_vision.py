from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from aiohttp import web

from opensmt.messaging import BusNode, SCPIMessage
from opensmt.vision import PassthroughPipeline, VisionPipelineBase

from .base import ModuleBase

log = logging.getLogger(__name__)

# Allowlist for names used in URL path segments
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Registry: pipeline "type" string  →  class
_PIPELINE_REGISTRY: dict[str, type[VisionPipelineBase]] = {
    "passthrough": PassthroughPipeline,
}

_COORD_AXES = ["X", "Y", "Z1", "R1", "Z2", "R2", "Z3", "R3", "Z4", "R4"]


def register_pipeline(type_name: str):
    """Class decorator that registers a VisionPipelineBase subclass by type name."""

    def decorator(cls: type[VisionPipelineBase]) -> type[VisionPipelineBase]:
        _PIPELINE_REGISTRY[type_name.lower()] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Config data-classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LightConfig:
    bus_command: str  # SCPI SET path, e.g. ":GCODE:ANOUT:2"
    on_value: int     # brightness used for the "ON" shorthand


@dataclass(slots=True)
class CameraConfig:
    name: str
    device: str
    fps: float
    resolution_dpcm_x: float
    resolution_dpcm_y: float
    flip_horizontal: bool
    flip_vertical: bool
    rotation_deg: int
    lights: dict[str, LightConfig]   # key: "standard" | "spec1" | "spec2"
    pipeline_names: list[str]


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    config: CameraConfig
    cap: cv2.VideoCapture | None = None
    frame: np.ndarray | None = None
    frame_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    capture_task: asyncio.Task | None = None
    light_values: dict[str, int] = field(default_factory=dict)
    active_pipeline: str | None = None
    pipeline_params: dict[str, Any] = field(default_factory=dict)
    last_pipeline_result: dict[str, Any] = field(default_factory=dict)
    current_rotation_deg: int = field(default_factory=lambda: 0)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class CameraVisionModule(ModuleBase):
    """OpenCV camera module with per-camera lights, vision pipelines and
    an embedded aiohttp web dashboard (Bootstrap 5)."""

    def __init__(self, name: str, config: dict[str, Any], node: BusNode) -> None:
        super().__init__(name, config, node)
        self._cameras: dict[str, CameraState] = {}
        self._pipelines: dict[str, VisionPipelineBase] = {
            "PASSTHROUGH": PassthroughPipeline("PASSTHROUGH", {}),
        }
        self._web_host = str(config.get("web_host", "0.0.0.0"))
        self._web_port = int(config.get("web_port", 8080))
        self._coord_target = str(config.get("coord_target", "COORD")).upper()
        self._coord_positions: dict[str, float | None] = {axis: None for axis in _COORD_AXES}
        self._coord_waiters: dict[str, list[asyncio.Future[float]]] = {axis: [] for axis in _COORD_AXES}
        self._coord_ws_clients: set[web.WebSocketResponse] = set()
        self._coord_ws_task: asyncio.Task[None] | None = None
        self._last_coord_broadcast: dict[str, float | None] | None = None
        self._runner: web.AppRunner | None = None

        # Build named pipeline pool
        for pipe_cfg in config.get("pipelines", []):
            pipe_name = str(pipe_cfg["name"]).upper()
            pipe_type = str(pipe_cfg.get("type", "passthrough")).lower()
            cls = _PIPELINE_REGISTRY.get(pipe_type, PassthroughPipeline)
            self._pipelines[pipe_name] = cls(pipe_name, pipe_cfg)

        # Build per-camera state
        for cam_cfg in config.get("cameras", []):
            cam_name = str(cam_cfg["name"]).upper()
            device = str(cam_cfg.get("device", "")).strip()
            if not device.startswith("/dev/"):
                raise ValueError(
                    f"Camera {cam_name} must use a /dev device path, got: {device or '<empty>'}"
                )

            lights: dict[str, LightConfig] = {}
            for light_key, light_val in cam_cfg.get("lights", {}).items():
                key = light_key.lower()
                if isinstance(light_val, str):
                    # "  :GCODE:ANOUT:2 64  " → command + on_value
                    parts = light_val.strip().split()
                    bus_cmd = parts[0]
                    on_val = int(parts[1]) if len(parts) > 1 else 1
                elif isinstance(light_val, dict):
                    bus_cmd = str(light_val["command"])
                    on_val = int(light_val.get("on_value", 1))
                else:
                    continue
                lights[key] = LightConfig(bus_command=bus_cmd, on_value=on_val)

            pipe_names = [str(p).upper() for p in cam_cfg.get("pipelines", [])]

            cfg = CameraConfig(
                name=cam_name,
                device=device,
                fps=float(cam_cfg.get("fps", 10.0)),
                resolution_dpcm_x=float(cam_cfg.get("resolution_dpcm_x", 0.0)),
                resolution_dpcm_y=float(cam_cfg.get("resolution_dpcm_y", 0.0)),
                flip_horizontal=bool(cam_cfg.get("flip_horizontal", False)),
                flip_vertical=bool(cam_cfg.get("flip_vertical", False)),
                rotation_deg=int(cam_cfg.get("rotation_deg", 0)),
                lights=lights,
                pipeline_names=pipe_names,
            )
            state = CameraState(config=cfg)
            state.current_rotation_deg = cfg.rotation_deg
            self._cameras[cam_name] = state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self.node.on_query("*", self._handle_query)
        self.node.on_set("*", self._handle_set)
        self.node.on_response("*", self._handle_response)

        for state in self._cameras.values():
            await self._open_camera(state)

        await self._start_web()
        self._coord_ws_task = asyncio.create_task(
            self._coord_ws_loop(),
            name=f"coord-ws-loop-{self.name}",
        )

    async def stop(self) -> None:
        if self._coord_ws_task:
            self._coord_ws_task.cancel()
            try:
                await self._coord_ws_task
            except asyncio.CancelledError:
                pass
            self._coord_ws_task = None

        for ws in list(self._coord_ws_clients):
            await ws.close()
        self._coord_ws_clients.clear()

        for state in self._cameras.values():
            if state.capture_task:
                state.capture_task.cancel()
                try:
                    await state.capture_task
                except asyncio.CancelledError:
                    pass
            if state.cap:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, state.cap.release)
                state.cap = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------------
    # Camera capture
    # ------------------------------------------------------------------

    async def _open_camera(self, state: CameraState) -> None:
        loop = asyncio.get_running_loop()

        def _open() -> cv2.VideoCapture | None:
            cap = cv2.VideoCapture(state.config.device)
            return cap if cap.isOpened() else None

        cap = await loop.run_in_executor(None, _open)
        if cap is not None:
            state.cap = cap
            state.capture_task = asyncio.create_task(
                self._capture_loop(state),
                name=f"cam-capture-{state.config.name}",
            )
            log.info("Opened camera %s (device %s)", state.config.name, state.config.device)
        else:
            log.warning(
                "Could not open camera %s (device %s)", state.config.name, state.config.device
            )

    async def _capture_loop(self, state: CameraState) -> None:
        assert state.cap is not None
        loop = asyncio.get_running_loop()
        interval = 1.0 / max(state.config.fps, 0.1)

        while True:
            t0 = loop.time()
            ok, frame = await loop.run_in_executor(None, state.cap.read)
            if ok:
                frame = self._transform_frame(frame, state)
                async with state.frame_lock:
                    state.frame = frame
            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.001, interval - elapsed))

    def _transform_frame(self, frame: np.ndarray, state: CameraState) -> np.ndarray:
        cfg = state.config
        if cfg.flip_horizontal and cfg.flip_vertical:
            frame = cv2.flip(frame, -1)
        elif cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)
        elif cfg.flip_vertical:
            frame = cv2.flip(frame, 0)

        rotation = state.current_rotation_deg % 360
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation != 0:
            h, w = frame.shape[:2]
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, -float(rotation), 1.0)
            frame = cv2.warpAffine(frame, matrix, (w, h))

        return frame

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _no_signal_frame(self) -> np.ndarray:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "No Signal", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (70, 70, 70), 2)
        return img

    async def _apply_pipeline(
        self, state: CameraState
    ) -> tuple[np.ndarray, dict[str, Any]]:
        async with state.frame_lock:
            frame = state.frame

        if frame is None:
            return self._no_signal_frame(), {}

        pipe_name = state.active_pipeline
        if pipe_name and pipe_name in self._pipelines:
            pipeline = self._pipelines[pipe_name]
            params = dict(state.pipeline_params)
            loop = asyncio.get_running_loop()
            processed, results = await loop.run_in_executor(
                None, pipeline.process, frame.copy(), params
            )
            state.last_pipeline_result = results
            return self._draw_coordinate_overlay(processed, state.config), results

        return self._draw_coordinate_overlay(frame, state.config), {}

    def _draw_coordinate_overlay(self, frame: np.ndarray, cfg: CameraConfig) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2

        # Main axes, with the coordinate origin at the center of the image.
        cv2.line(out, (0, cy), (w - 1, cy), (0, 220, 0), 1, cv2.LINE_AA)
        cv2.line(out, (cx, 0), (cx, h - 1), (0, 220, 0), 1, cv2.LINE_AA)

        px_per_10mm_x = int(round(cfg.resolution_dpcm_x))
        px_per_10mm_y = int(round(cfg.resolution_dpcm_y))

        if px_per_10mm_x > 0:
            step = px_per_10mm_x
            x = cx + step
            while x < w:
                cv2.line(out, (x, 0), (x, h - 1), (90, 90, 90), 1, cv2.LINE_AA)
                x += step
            x = cx - step
            while x >= 0:
                cv2.line(out, (x, 0), (x, h - 1), (90, 90, 90), 1, cv2.LINE_AA)
                x -= step

        if px_per_10mm_y > 0:
            step = px_per_10mm_y
            y = cy + step
            while y < h:
                cv2.line(out, (0, y), (w - 1, y), (90, 90, 90), 1, cv2.LINE_AA)
                y += step
            y = cy - step
            while y >= 0:
                cv2.line(out, (0, y), (w - 1, y), (90, 90, 90), 1, cv2.LINE_AA)
                y -= step

        # Short ticks along the main axes at -1, 0, +1 mm.
        px_per_mm_x = cfg.resolution_dpcm_x / 10.0 if cfg.resolution_dpcm_x > 0 else 0.0
        px_per_mm_y = cfg.resolution_dpcm_y / 10.0 if cfg.resolution_dpcm_y > 0 else 0.0
        tick_len = 8

        for mm in (-1.0, 0.0, 1.0):
            if px_per_mm_x > 0:
                tx = int(round(cx + mm * px_per_mm_x))
                if 0 <= tx < w:
                    cv2.line(out, (tx, cy - tick_len), (tx, cy + tick_len), (0, 220, 0), 1, cv2.LINE_AA)
            if px_per_mm_y > 0:
                ty = int(round(cy - mm * px_per_mm_y))
                if 0 <= ty < h:
                    cv2.line(out, (cx - tick_len, ty), (cx + tick_len, ty), (0, 220, 0), 1, cv2.LINE_AA)

        cv2.putText(out, "0/0", (cx + 6, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Bus handlers
    # ------------------------------------------------------------------

    async def _handle_query(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return
        parts = msg.command.strip(":").split(":")
        if not parts or parts[0] != self.name:
            return

        target = packet.get("source")

        # :CAMERA:STATUS?  →  dict of all camera statuses
        if len(parts) == 2 and parts[1] == "STATUS":
            statuses = {
                n: ("online" if s.cap is not None else "offline")
                for n, s in self._cameras.items()
            }
            await self.node.send_response(msg.command, json.dumps(statuses), target=target)
            return

        if len(parts) < 3:
            return
        cam_name = parts[1]
        state = self._cameras.get(cam_name)
        if not state:
            return

        verb = parts[2]

        if verb == "STATUS" and len(parts) == 3:
            status = "online" if state.cap is not None else "offline"
            await self.node.send_response(msg.command, status, target=target)

        elif verb == "LIGHT" and len(parts) == 4:
            light = parts[3].lower()
            val = state.light_values.get(light, 0)
            await self.node.send_response(msg.command, val, target=target)

        elif verb == "PIPELINE" and len(parts) == 3:
            await self.node.send_response(
                msg.command, state.active_pipeline or "", target=target
            )

    async def _handle_set(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return
        parts = msg.command.strip(":").split(":")
        if len(parts) < 4 or parts[0] != self.name:
            return

        target = packet.get("source")
        cam_name = parts[1]
        state = self._cameras.get(cam_name)
        if not state:
            return

        verb = parts[2]

        # :CAMERA:<name>:LIGHT:<light> <value|ON|OFF>
        if verb == "LIGHT" and len(parts) == 4:
            light = parts[3].lower()
            light_cfg = state.config.lights.get(light)
            if not light_cfg:
                await self.node.send_response(
                    msg.command, f"UNKNOWN_LIGHT:{light}", target=target
                )
                return

            raw = str(msg.value).strip().upper()
            if raw == "ON":
                value = light_cfg.on_value
            elif raw == "OFF":
                value = 0
            else:
                try:
                    value = int(float(raw))
                except ValueError:
                    await self.node.send_response(msg.command, "INVALID_VALUE", target=target)
                    return

            if not (0 <= value <= 65535):
                await self.node.send_response(msg.command, "VALUE_OUT_OF_RANGE", target=target)
                return

            await self.node.send_set(light_cfg.bus_command, value)
            state.light_values[light] = value
            await self.node.send_response(msg.command, value, target=target)

        # :CAMERA:<name>:PIPELINE:<pipe_name> [json_params]
        elif verb == "PIPELINE" and len(parts) == 4:
            pipe_name = parts[3].upper()
            if pipe_name not in self._pipelines:
                await self.node.send_response(
                    msg.command, f"UNKNOWN_PIPELINE:{pipe_name}", target=target
                )
                return

            raw_params = str(msg.value).strip() if msg.value else ""
            if raw_params:
                try:
                    params: dict[str, Any] = json.loads(raw_params)
                    if not isinstance(params, dict):
                        raise ValueError
                except (json.JSONDecodeError, ValueError):
                    await self.node.send_response(msg.command, "INVALID_PARAMS_JSON", target=target)
                    return
            else:
                params = {}

            state.active_pipeline = pipe_name
            state.pipeline_params = params
            await self.node.send_response(msg.command, pipe_name, target=target)

    async def _handle_response(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        parts = msg.command.strip(":").split(":")
        if len(parts) != 3:
            return

        scope = parts[1].upper()
        axis = parts[2].upper()
        if scope not in {"ABS", "POS"} or axis not in self._coord_positions:
            return

        value = self._parse_numeric(msg.value)
        if value is None:
            return

        self._coord_positions[axis] = value
        for fut in self._coord_waiters[axis]:
            if not fut.done():
                fut.set_result(value)
        self._coord_waiters[axis].clear()
        await self._broadcast_coord_positions(force=False)

    # ------------------------------------------------------------------
    # Web server
    # ------------------------------------------------------------------

    async def _start_web(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._web_main)
        app.router.add_get("/ws/coord", self._ws_coord)
        app.router.add_get("/camera/{name}", self._web_camera)
        app.router.add_get("/stream/{name}", self._web_stream)
        app.router.add_get("/thumb/{name}", self._web_thumb)
        app.router.add_post("/api/camera/{name}/light/{light}", self._api_light)
        app.router.add_post("/api/camera/{name}/pipeline/{pipe}", self._api_pipeline)
        app.router.add_get("/api/camera/{name}/pipeline/result", self._api_pipeline_result)
        app.router.add_post("/api/camera/{name}/rotation", self._api_rotation)
        app.router.add_post("/api/coord/jog", self._api_coord_jog)
        app.router.add_post("/api/coord/home", self._api_coord_home)
        app.router.add_post("/api/coord/home-xy", self._api_coord_home_xy)
        app.router.add_post("/api/coord/park", self._api_coord_park)
        app.router.add_post("/api/coord/dispose", self._api_coord_dispose)
        app.router.add_post("/api/coord/homing-fiducial-main", self._api_coord_homing_fiducial_main)
        app.router.add_post("/api/coord/secondary-fiducial", self._api_coord_secondary_fiducial)
        app.router.add_post("/api/coord/nozzle-change", self._api_coord_nozzle_change)
        app.router.add_post("/api/coord/calibration-spot", self._api_coord_calibration_spot)
        app.router.add_get("/api/coord/positions", self._api_coord_positions)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._web_host, self._web_port)
        await site.start()
        log.info("Camera web UI on http://%s:%d", self._web_host, self._web_port)

    # --- Page handlers ---

    async def _web_main(self, request: web.Request) -> web.Response:
        cards = ""
        for cam_name, state in self._cameras.items():
            badge_cls = "success" if state.cap is not None else "secondary"
            badge_txt = "Online" if state.cap is not None else "Offline"
            pipe_txt = state.active_pipeline or "—"
            cards += f"""
            <div class="col-sm-6 col-xl-4">
              <div class="card h-100 shadow-sm">
                <div class="card-img-top overflow-hidden bg-dark text-center" style="height:180px">
                  <img id="thumb-{cam_name}" src="/thumb/{cam_name}"
                       class="h-100" style="object-fit:cover;width:100%"
                       onerror="this.style.opacity='0'" />
                </div>
                <div class="card-body d-flex flex-column">
                  <h5 class="card-title d-flex justify-content-between align-items-center">
                    {cam_name}
                    <span class="badge bg-{badge_cls}">{badge_txt}</span>
                  </h5>
                  <p class="card-text text-muted small mb-1">
                    Device: <code>{state.config.device}</code> &nbsp;|&nbsp; FPS: {state.config.fps}
                  </p>
                  <p class="card-text text-muted small">
                                        Resolution: <code>{state.config.resolution_dpcm_x:g}</code> dpcm X &nbsp;|&nbsp;
                                        <code>{state.config.resolution_dpcm_y:g}</code> dpcm Y
                                    </p>
                                    <p class="card-text text-muted small">
                    Pipeline: <code>{pipe_txt}</code>
                  </p>
                  <a href="/camera/{cam_name}" class="btn btn-primary mt-auto">
                    &#9654; View Stream
                  </a>
                </div>
              </div>
            </div>"""

        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>openSMT Vision</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
</head>
<body class="bg-light">
  <nav class="navbar navbar-dark bg-dark shadow-sm">
    <div class="container-fluid">
      <span class="navbar-brand fw-bold">&#128247; openSMT Vision</span>
    </div>
  </nav>
  <div class="container py-4">
    <h4 class="mb-3 text-secondary">Camera Dashboard</h4>
        <div class="row g-4">{cards}</div>

        <div class="row g-4 mt-1">
            <div class="col-12 col-xl-6">
                <div class="card shadow-sm">
                    <div class="card-header fw-semibold">XY Positioning</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="step-range" class="form-label small text-muted mb-1">Step Size (mm)</label>
                            <input type="range" class="form-range" min="0" max="8" step="1" id="step-range" value="3" oninput="updateStepLabel()">
                            <div class="d-flex justify-content-between small text-muted">
                                <span>0.01</span><span>0.1</span><span>0.5</span><span>1.0</span><span>5.0</span><span>10.0</span><span>25.0</span><span>50.0</span><span>100</span>
                            </div>
                            <div class="small mt-2">Selected: <span class="badge bg-primary" id="step-label">1.0 mm</span></div>
                        </div>

                        <div class="mb-2 d-flex justify-content-center gap-2">
                            <button class="btn btn-outline-dark" onclick="goHome()" title="Home All">&#127968;</button>
                            <button class="btn btn-outline-success" onclick="goHomeXY()" title="Home X & Y (simultaneous)">XY↻</button>
                            <button class="btn btn-outline-info" onclick="goCalibrationSpot()" title="Calibration Spot">CAL</button>
                        </div>

                        <div class="d-grid gap-2 justify-content-center" style="grid-template-columns: 64px 64px 64px;">
                            <button class="btn btn-outline-secondary" onclick="goHomingFiducialMain()" title="Homing Fiducial Main">HM</button>
                            <button class="btn btn-outline-primary" onclick="jog(0,1)">&#8593;</button>
                            <button class="btn btn-outline-secondary" onclick="goSecondaryFiducial()" title="Secondary Fiducial">HS</button>

                            <button class="btn btn-outline-primary" onclick="jog(-1,0)">&#8592;</button>
                            <button class="btn btn-success" onclick="goPark()">P</button>
                            <button class="btn btn-outline-primary" onclick="jog(1,0)">&#8594;</button>

                            <button class="btn btn-outline-warning" onclick="goNozzleChange()" title="Nozzle Change">N</button>
                            <button class="btn btn-outline-primary" onclick="jog(0,-1)">&#8595;</button>
                            <button class="btn btn-danger" onclick="goDispose()">Dispose</button>
                        </div>

                        <div class="small text-muted mt-3" id="coord-status">Ready</div>
                    </div>
                </div>
            </div>

            <div class="col-12 col-xl-6">
                <div class="card shadow-sm h-100">
                    <div class="card-header fw-semibold">Current Coordinates</div>
                    <div class="card-body">
                        <div class="row g-2 mb-2">
                            <div class="col-6">
                                <div class="border rounded p-2 bg-light-subtle">
                                    <div class="small text-muted">X</div>
                                    <div class="fw-semibold" id="coord-x">--</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-2 bg-light-subtle">
                                    <div class="small text-muted">Y</div>
                                    <div class="fw-semibold" id="coord-y">--</div>
                                </div>
                            </div>
                        </div>

                        <div class="row g-2">
                            <div class="col-6">
                                <div class="border rounded p-2">
                                    <div class="small text-muted">Z1 / R1</div>
                                    <div class="fw-semibold"><span id="coord-z1">--</span> / <span id="coord-r1">--</span></div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-2">
                                    <div class="small text-muted">Z2 / R2</div>
                                    <div class="fw-semibold"><span id="coord-z2">--</span> / <span id="coord-r2">--</span></div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-2">
                                    <div class="small text-muted">Z3 / R3</div>
                                    <div class="fw-semibold"><span id="coord-z3">--</span> / <span id="coord-r3">--</span></div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-2">
                                    <div class="small text-muted">Z4 / R4</div>
                                    <div class="fw-semibold"><span id="coord-z4">--</span> / <span id="coord-r4">--</span></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
  </div>
  <script>
        var coordSocket = null;

        function selectedStep() {{
            var values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0];
            var idx = parseInt(document.getElementById('step-range').value, 10);
            return values[idx] || 1.0;
        }}

        function updateStepLabel() {{
            var step = selectedStep();
            document.getElementById('step-label').textContent = step.toFixed(2).replace(/[.]00$/, '.0') + ' mm';
        }}

        function fmtCoord(v) {{
            if (v === null || v === undefined) return '--';
            var num = Number(v);
            if (Number.isNaN(num)) return '--';
            return num.toFixed(2);
        }}

        function refreshCoords() {{
            if (coordSocket && coordSocket.readyState === WebSocket.OPEN) {{
                coordSocket.send('refresh');
            }}
        }}

        function applyCoords(pos) {{
            ['x','y','z1','r1','z2','r2','z3','r3','z4','r4'].forEach(function(k) {{
                var el = document.getElementById('coord-' + k);
                if (el) el.textContent = fmtCoord(pos[k.toUpperCase()]);
            }});
        }}

        function connectCoordSocket() {{
            var proto = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            coordSocket = new WebSocket(proto + window.location.host + '/ws/coord');
            coordSocket.onmessage = function(ev) {{
                try {{
                    var data = JSON.parse(ev.data);
                    if (data && data.type === 'coord' && data.positions) {{
                        applyCoords(data.positions);
                    }}
                }} catch (e) {{
                    // Ignore malformed frame
                }}
            }};
            coordSocket.onclose = function() {{
                setTimeout(connectCoordSocket, 1000);
            }};
            coordSocket.onerror = function() {{
                coordSocket.close();
            }};
        }}

        function setCoordStatus(text, isError) {{
            var el = document.getElementById('coord-status');
            el.textContent = text;
            el.className = 'small mt-3 ' + (isError ? 'text-danger' : 'text-muted');
        }}

        function jog(dxSign, dySign) {{
            var step = selectedStep();
            fetch('/api/coord/jog', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{dx: dxSign * step, dy: dySign * step}})
            }})
            .then(function(r) {{ return r.json(); }})
            .then(function(d) {{
                if (d.error) {{ setCoordStatus('Jog failed: ' + d.error, true); return; }}
                setCoordStatus('Jogged X=' + d.dx + ' mm, Y=' + d.dy + ' mm', false);
            }})
            .catch(function(e) {{ setCoordStatus('Jog failed: ' + e.message, true); }});
        }}

        function goPark() {{
            fetch('/api/coord/park', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Park failed: ' + d.error, true); return; }}
                    setCoordStatus('Park command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Park failed: ' + e.message, true); }});
        }}

        function goHome() {{
            fetch('/api/coord/home', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Home failed: ' + d.error, true); return; }}
                    setCoordStatus('Home command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Home failed: ' + e.message, true); }});
        }}

        function goHomeXY() {{
            fetch('/api/coord/home-xy', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Home XY failed: ' + d.error, true); return; }}
                    setCoordStatus('Home X & Y (simultaneous) command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Home XY failed: ' + e.message, true); }});
        }}

        function goDispose() {{
            fetch('/api/coord/dispose', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Dispose failed: ' + d.error, true); return; }}
                    setCoordStatus('Dispose command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Dispose failed: ' + e.message, true); }});
        }}

        function goHomingFiducialMain() {{
            fetch('/api/coord/homing-fiducial-main', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Homing Fiducial Main failed: ' + d.error, true); return; }}
                    setCoordStatus('Homing Fiducial Main command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Homing Fiducial Main failed: ' + e.message, true); }});
        }}

        function goSecondaryFiducial() {{
            fetch('/api/coord/secondary-fiducial', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Secondary Fiducial failed: ' + d.error, true); return; }}
                    setCoordStatus('Secondary Fiducial command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Secondary Fiducial failed: ' + e.message, true); }});
        }}

        function goNozzleChange() {{
            fetch('/api/coord/nozzle-change', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Nozzle Change failed: ' + d.error, true); return; }}
                    setCoordStatus('Nozzle Change command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Nozzle Change failed: ' + e.message, true); }});
        }}

        function goCalibrationSpot() {{
            fetch('/api/coord/calibration-spot', {{method: 'POST'}})
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (d.error) {{ setCoordStatus('Calibration Spot failed: ' + d.error, true); return; }}
                    setCoordStatus('Calibration Spot command sent', false);
                }})
                .catch(function(e) {{ setCoordStatus('Calibration Spot failed: ' + e.message, true); }});
        }}

        updateStepLabel();
        connectCoordSocket();
        setInterval(refreshCoords, 1000);

    setInterval(function() {{
      document.querySelectorAll('[id^="thumb-"]').forEach(function(img) {{
        var base = img.src.split('?')[0];
        img.src = base + '?t=' + Date.now();
        img.style.opacity = '1';
      }});
    }}, 2000);
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def _web_camera(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            raise web.HTTPNotFound()
        name = raw_name.upper()
        state = self._cameras.get(name)
        if not state:
            raise web.HTTPNotFound()

        # ---- lights panel ----
        lights_html = ""
        for light_key, light_cfg in state.config.lights.items():
            current = state.light_values.get(light_key, 0)
            lights_html += f"""
            <div class="mb-3">
              <label class="form-label fw-semibold text-capitalize">{light_key}</label>
              <div class="d-flex gap-2 align-items-center">
                <button class="btn btn-sm btn-outline-secondary"
                        onclick="setLight('{name}','{light_key}',0)">Off</button>
                <button class="btn btn-sm btn-outline-warning"
                        onclick="setLight('{name}','{light_key}',{light_cfg.on_value})">On</button>
                <input type="range" class="form-range flex-grow-1" min="0" max="65535"
                       value="{current}" id="rng-{light_key}"
                       oninput="setLight('{name}','{light_key}',this.value)">
                <span class="badge bg-secondary ms-1" id="lval-{light_key}">{current}</span>
              </div>
            </div>"""

        # ---- pipeline panel ----
        cam_pipes: list[str] = ["PASSTHROUGH"]
        for p in state.config.pipeline_names:
            if p != "PASSTHROUGH":
                cam_pipes.append(p)

        pipe_options = "".join(
            f'<option value="{p}" {"selected" if p == state.active_pipeline else ""}>{p}</option>'
            for p in cam_pipes
        )

        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera: {name}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
</head>
<body class="bg-light">
  <nav class="navbar navbar-dark bg-dark shadow-sm">
    <div class="container-fluid">
      <a href="/" class="btn btn-outline-light btn-sm me-3">&larr; Back</a>
      <span class="navbar-brand">&#128247; {name}</span>
    </div>
  </nav>
  <div class="container-fluid py-3">
    <div class="row g-3">

      <!-- Stream column -->
      <div class="col-lg-8">
        <div class="card shadow-sm">
          <div class="card-header d-flex justify-content-between align-items-center">
            <span>Live Stream</span>
                        <span class="badge bg-info text-dark">&#9654; {state.config.fps} fps</span>
          </div>
          <div class="card-body p-1 bg-dark text-center" style="min-height:200px">
            <img src="/stream/{name}" class="img-fluid" style="max-height:72vh"
                 onerror="this.alt='Stream unavailable'" />
          </div>
                    <div class="card-footer text-muted small">
                        Resolution: <code>{state.config.resolution_dpcm_x:g}</code> dpcm X &nbsp;|&nbsp;
                        <code>{state.config.resolution_dpcm_y:g}</code> dpcm Y
                    </div>
        </div>
      </div>

      <!-- Controls column -->
      <div class="col-lg-4">

        <!-- Rotation -->
        <div class="card shadow-sm mb-3">
          <div class="card-header fw-semibold">🔄 Rotation</div>
          <div class="card-body">
            <div class="d-flex align-items-center gap-2 mb-3">
              <span class="text-muted small">Angle:</span>
              <span class="badge bg-primary" id="rot-display">{state.config.rotation_deg}°</span>
            </div>
            <div class="mb-2 d-grid gap-2" style="grid-template-columns: 1fr 1fr;">
              <button class="btn btn-outline-secondary btn-sm" onclick="adjustRotation('{name}', -1)">◀ -1°</button>
              <button class="btn btn-outline-secondary btn-sm" onclick="adjustRotation('{name}', 1)">+1° ▶</button>
            </div>
            <div class="d-grid gap-2" style="grid-template-columns: 1fr 1fr;">
              <button class="btn btn-outline-secondary btn-sm" onclick="adjustRotation('{name}', -45)">-45°</button>
              <button class="btn btn-outline-secondary btn-sm" onclick="adjustRotation('{name}', 45)">+45°</button>
            </div>
            <div class="d-grid gap-2 mt-2">
              <button class="btn btn-outline-warning btn-sm" onclick="setRotation('{name}', 0)">Reset (0°)</button>
            </div>
          </div>
        </div>

        <!-- Lights -->
        <div class="card shadow-sm mb-3">
          <div class="card-header fw-semibold">&#128161; Lights</div>
          <div class="card-body">{lights_html or '<p class="text-muted small mb-0">No lights configured.</p>'}</div>
        </div>

        <!-- Pipeline selector -->
        <div class="card shadow-sm mb-3">
          <div class="card-header fw-semibold">&#9881; Vision Pipeline</div>
          <div class="card-body">
            <div class="mb-2">
              <label class="form-label small">Active Pipeline</label>
              <select class="form-select form-select-sm" id="pipe-select">{pipe_options}</select>
            </div>
            <div class="mb-2">
              <label class="form-label small">Parameters (JSON)</label>
              <textarea class="form-control form-control-sm font-monospace" id="pipe-params"
                        rows="4" placeholder='{{"threshold": 128}}'></textarea>
            </div>
            <button class="btn btn-primary btn-sm w-100" onclick="applyPipeline('{name}')">
              &#9654; Apply Pipeline
            </button>
          </div>
        </div>

        <!-- Last result -->
        <div class="card shadow-sm">
          <div class="card-header fw-semibold">&#128202; Last Result</div>
          <div class="card-body p-2">
            <pre class="small mb-0 text-break" id="pipe-result"
                 style="max-height:160px;overflow:auto;white-space:pre-wrap">—</pre>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    function setLight(cam, light, value) {{
      value = parseInt(value);
      var badge = document.getElementById('lval-' + light);
      if (badge) badge.textContent = value;
      fetch('/api/camera/' + cam + '/light/' + light, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{value: value}})
      }});
    }}

    function adjustRotation(cam, delta) {{
      fetch('/api/camera/' + cam + '/rotation', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{delta: delta}})
      }})
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (d.error) {{ console.error('Rotation error:', d.error); return; }}
        document.getElementById('rot-display').textContent = d.rotation + '°';
      }})
      .catch(function(e) {{ console.error('Rotation failed:', e); }});
    }}

    function setRotation(cam, value) {{
      fetch('/api/camera/' + cam + '/rotation', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{set: value}})
      }})
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (d.error) {{ console.error('Rotation error:', d.error); return; }}
        document.getElementById('rot-display').textContent = d.rotation + '°';
      }})
      .catch(function(e) {{ console.error('Rotation failed:', e); }});
    }}

    function applyPipeline(cam) {{
      var pipe = document.getElementById('pipe-select').value;
      var raw = document.getElementById('pipe-params').value.trim();
      var params = {{}};
      if (raw) {{
        try {{ params = JSON.parse(raw); }}
        catch (e) {{ alert('Invalid JSON: ' + e.message); return; }}
      }}
      fetch('/api/camera/' + cam + '/pipeline/' + pipe, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{params: params}})
      }}).then(function() {{ fetchResult(cam); }});
    }}

    function fetchResult(cam) {{
      fetch('/api/camera/' + cam + '/pipeline/result')
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
          document.getElementById('pipe-result').textContent = JSON.stringify(d, null, 2) || '—';
        }});
    }}

    // Auto-refresh result every 2 s when a non-passthrough pipeline is active
    setInterval(function() {{
      var pipe = document.getElementById('pipe-select');
      if (pipe && pipe.value !== 'PASSTHROUGH') fetchResult('{name}');
    }}, 2000);
  </script>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def _web_stream(self, request: web.Request) -> web.StreamResponse:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            raise web.HTTPNotFound()
        name = raw_name.upper()
        state = self._cameras.get(name)
        if not state:
            raise web.HTTPNotFound()

        response = web.StreamResponse()
        response.headers["Content-Type"] = "multipart/x-mixed-replace; boundary=frame"
        response.headers["Cache-Control"] = "no-cache"
        await response.prepare(request)

        interval = 1.0 / max(state.config.fps, 0.1)
        loop = asyncio.get_running_loop()

        try:
            while True:
                t0 = loop.time()
                frame, _ = await self._apply_pipeline(state)
                ok, jpg = await loop.run_in_executor(
                    None,
                    lambda f=frame: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75]),
                )
                if ok:
                    data: bytes = jpg.tobytes()  # type: ignore[union-attr]
                    await response.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                    )
                elapsed = loop.time() - t0
                await asyncio.sleep(max(0.001, interval - elapsed))
        except (ConnectionResetError, asyncio.CancelledError):
            pass

        return response

    async def _web_thumb(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            raise web.HTTPNotFound()
        name = raw_name.upper()
        state = self._cameras.get(name)
        if not state:
            raise web.HTTPNotFound()

        frame, _ = await self._apply_pipeline(state)
        h, w = frame.shape[:2]
        if w > 320:
            thumb_w, thumb_h = 320, int(h * 320 / w)
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(
                None, lambda f=frame: cv2.resize(f, (thumb_w, thumb_h))
            )

        loop = asyncio.get_running_loop()
        ok, jpg = await loop.run_in_executor(
            None,
            lambda f=frame: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 65]),
        )
        if not ok:
            raise web.HTTPInternalServerError()

        return web.Response(
            body=jpg.tobytes(),  # type: ignore[union-attr]
            content_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store"},
        )

    # --- REST API handlers ---

    async def _api_light(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        raw_light = request.match_info["light"]
        if not _NAME_RE.match(raw_name) or not _NAME_RE.match(raw_light):
            return web.json_response({"error": "invalid_name"}, status=400)

        state = self._cameras.get(raw_name.upper())
        if not state:
            return web.json_response({"error": "unknown_camera"}, status=404)

        light = raw_light.lower()
        light_cfg = state.config.lights.get(light)
        if not light_cfg:
            return web.json_response({"error": "unknown_light"}, status=404)

        try:
            body = await request.json()
            value = int(body["value"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        if not (0 <= value <= 65535):
            return web.json_response({"error": "value_out_of_range"}, status=422)

        await self.node.send_set(light_cfg.bus_command, value)
        state.light_values[light] = value
        return web.json_response({"light": light, "value": value})

    async def _api_pipeline(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        raw_pipe = request.match_info["pipe"]
        if not _NAME_RE.match(raw_name) or not _NAME_RE.match(raw_pipe):
            return web.json_response({"error": "invalid_name"}, status=400)

        state = self._cameras.get(raw_name.upper())
        if not state:
            return web.json_response({"error": "unknown_camera"}, status=404)

        pipe_name = raw_pipe.upper()
        if pipe_name not in self._pipelines:
            return web.json_response({"error": "unknown_pipeline"}, status=404)

        try:
            body = await request.json()
            params: dict[str, Any] = body.get("params", {})
            if not isinstance(params, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        state.active_pipeline = pipe_name
        state.pipeline_params = params
        return web.json_response({"pipeline": pipe_name, "params": params})

    async def _api_pipeline_result(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        state = self._cameras.get(raw_name.upper())
        if not state:
            return web.json_response({"error": "unknown_camera"}, status=404)

        return web.json_response(state.last_pipeline_result)

    async def _api_rotation(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        state = self._cameras.get(raw_name.upper())
        if not state:
            return web.json_response({"error": "unknown_camera"}, status=404)

        try:
            body = await request.json()
            # Support both "delta" (increment) and "set" (absolute) operations
            if "delta" in body:
                delta = int(body["delta"])
                state.current_rotation_deg = (state.current_rotation_deg + delta) % 360
            elif "set" in body:
                rotation = int(body["set"])
                state.current_rotation_deg = rotation % 360
            else:
                return web.json_response({"error": "missing_delta_or_set"}, status=400)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        return web.json_response({"rotation": state.current_rotation_deg})

    async def _api_coord_jog(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            dx = float(body.get("dx", 0.0))
            dy = float(body.get("dy", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        if dx != 0.0:
            await self.node.send_set(f":{self._coord_target}:REL:X", dx, target=self._coord_target)
        if dy != 0.0:
            await self.node.send_set(f":{self._coord_target}:REL:Y", dy, target=self._coord_target)

        return web.json_response({"dx": dx, "dy": dy})

    async def _api_coord_home(self, request: web.Request) -> web.Response:
        await self.node.send_action(f":{self._coord_target}:HOME", target=self._coord_target)
        return web.json_response({"status": "ok"})

    async def _api_coord_home_xy(self, request: web.Request) -> web.Response:
        await self.node.send_action(f":{self._coord_target}:HOME:XY", target=self._coord_target)
        return web.json_response({"status": "ok"})

    async def _api_coord_park(self, request: web.Request) -> web.Response:
        await self.node.send_action(f":{self._coord_target}:PARK", target=self._coord_target)
        return web.json_response({"status": "ok"})

    async def _api_coord_dispose(self, request: web.Request) -> web.Response:
        await self.node.send_action(f":{self._coord_target}:DISPOSE", target=self._coord_target)
        return web.json_response({"status": "ok"})

    async def _api_coord_homing_fiducial_main(self, request: web.Request) -> web.Response:
        await self.node.send_action(
            f":{self._coord_target}:HOMINGFIDUCIALMAIN",
            target=self._coord_target,
        )
        return web.json_response({"status": "ok"})

    async def _api_coord_secondary_fiducial(self, request: web.Request) -> web.Response:
        await self.node.send_action(
            f":{self._coord_target}:SECONDARYFIDUCIAL",
            target=self._coord_target,
        )
        return web.json_response({"status": "ok"})

    async def _api_coord_nozzle_change(self, request: web.Request) -> web.Response:
        await self.node.send_action(
            f":{self._coord_target}:NOZZLECHANGE",
            target=self._coord_target,
        )
        return web.json_response({"status": "ok"})

    async def _api_coord_calibration_spot(self, request: web.Request) -> web.Response:
        await self.node.send_action(
            f":{self._coord_target}:CALIBRATIONSPOT",
            target=self._coord_target,
        )
        return web.json_response({"status": "ok"})

    async def _api_coord_positions(self, request: web.Request) -> web.Response:
        await self._refresh_coord_positions(timeout=0.25)
        return web.json_response(self._coord_payload())

    async def _ws_coord(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self._coord_ws_clients.add(ws)

        await self._refresh_coord_positions(timeout=0.25)
        await ws.send_json({"type": "coord", "positions": self._coord_payload()})

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT and msg.data == "refresh":
                    await self._refresh_coord_positions(timeout=0.25)
                    await ws.send_json({"type": "coord", "positions": self._coord_payload()})
                elif msg.type in {web.WSMsgType.CLOSE, web.WSMsgType.ERROR}:
                    break
        finally:
            self._coord_ws_clients.discard(ws)

        return ws

    async def _coord_ws_loop(self) -> None:
        while True:
            await self._refresh_coord_positions(timeout=0.25)
            await self._broadcast_coord_positions(force=True)
            await asyncio.sleep(1.0)

    async def _refresh_coord_positions(self, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[float]] = {}
        for axis in _COORD_AXES:
            fut: asyncio.Future[float] = loop.create_future()
            self._coord_waiters[axis].append(fut)
            futures[axis] = fut
            await self.node.send_query(f":{self._coord_target}:ABS:{axis}", target=self._coord_target)

        for axis, fut in futures.items():
            try:
                await asyncio.wait_for(fut, timeout=timeout)
            except asyncio.TimeoutError:
                if fut in self._coord_waiters[axis]:
                    self._coord_waiters[axis].remove(fut)

    def _coord_payload(self) -> dict[str, float | None]:
        return {axis: self._coord_positions[axis] for axis in _COORD_AXES}

    async def _broadcast_coord_positions(self, force: bool) -> None:
        if not self._coord_ws_clients:
            return

        payload = self._coord_payload()
        if not force and payload == self._last_coord_broadcast:
            return
        self._last_coord_broadcast = dict(payload)

        stale: list[web.WebSocketResponse] = []
        for ws in self._coord_ws_clients:
            if ws.closed:
                stale.append(ws)
                continue
            try:
                await ws.send_json({"type": "coord", "positions": payload})
            except ConnectionError:
                stale.append(ws)

        for ws in stale:
            self._coord_ws_clients.discard(ws)

    @staticmethod
    def _parse_numeric(value: Any) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None
