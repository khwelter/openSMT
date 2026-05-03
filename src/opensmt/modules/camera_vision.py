from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aiohttp import web

from opensmt.vision import PassthroughPipeline, VisionPipelineBase

from opensmt.hardware.driver import HardwareDriver
from opensmt.runtime.command_runner import CommandRunner
from opensmt.store.location_store import LocationStore
from opensmt.store.nozzle_config import NozzleConfigStore
from opensmt.store.position_store import PositionStore
from opensmt.store.valve_store import ValveStore

log = logging.getLogger(__name__)

# Allowlist for names used in URL path segments
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Registry: pipeline "type" string  →  class
_PIPELINE_REGISTRY: dict[str, type[VisionPipelineBase]] = {
    "passthrough": PassthroughPipeline,
}

_COORD_AXES = ["X", "Y", "Z1", "R1", "Z2", "R2", "Z3", "R3", "Z4", "R4"]
_LIGHT_MIN = 0
_LIGHT_MAX = 3


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
    board_id: str     # target board, e.g. "XY"
    index: int        # board-local analog output index
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


@dataclass(slots=True)
class RuntimeNozzleConfig:
    """Runtime nozzle state: Z-axis limits and rotation axis."""
    name: str
    z_axis: str
    r_axis: str
    min_z: float     # Minimum Z position (default -50.0)
    max_z: float     # Maximum Z position (default 0.0)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class CameraVisionModule:
    """OpenCV camera module with per-camera lights, vision pipelines and
    an embedded aiohttp web dashboard (Bootstrap 5)."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        driver: HardwareDriver,
        position_store: PositionStore,
        location_store: LocationStore,
        nozzle_config_store: NozzleConfigStore | None = None,
        valve_store: ValveStore | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self._driver = driver
        self._position_store = position_store
        self._location_store = location_store
        self._nozzle_config_store = nozzle_config_store
        self._valve_store = valve_store
        self._cameras: dict[str, CameraState] = {}
        self._pipelines: dict[str, VisionPipelineBase] = {
            "PASSTHROUGH": PassthroughPipeline("PASSTHROUGH", {}),
        }
        self._web_host = str(config.get("web_host", "0.0.0.0"))
        self._web_port = int(config.get("web_port", 8080))
        self._icons_dir = Path(__file__).resolve().parents[3] / "assets" / "icons" / "opensmt-ui" / "128"
        offsets_persist_path_raw = config.get("_nozzle_offsets_persist_path")
        self._nozzle_offsets_persist_path = Path(str(offsets_persist_path_raw)).expanduser() if offsets_persist_path_raw else None

        # Build nozzle runtime config table (limits only; current position lives in PositionStore)
        self._nozzles: dict[str, RuntimeNozzleConfig] = {}
        for item in config.get("nozzles", []):
            n_name = str(item["name"]).upper()
            z_axis = str(item["z_axis"]).upper()
            if z_axis.startswith("Z") and len(z_axis) > 1:
                r_axis = f"R{z_axis[1:]}"
            else:
                r_axis = "R1"
            min_z = float(item.get("min_z", -50.0))
            max_z = float(item.get("max_z", 0.0))
            self._nozzles[n_name] = RuntimeNozzleConfig(
                name=n_name,
                z_axis=z_axis,
                r_axis=r_axis,
                min_z=min_z,
                max_z=max_z,
            )
        log.debug("Nozzle configs: %s", {n: (c.z_axis, c.r_axis, c.min_z, c.max_z) for n, c in self._nozzles.items()})

        self._coord_ws_clients: set[web.WebSocketResponse] = set()
        self._last_coord_broadcast: dict[str, float | None] | None = None
        self._runner: web.AppRunner | None = None
        self._commands = CommandRunner(max_history=500)

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
                if not isinstance(light_val, dict):
                    log.warning("Skipping light %s/%s: expected dict with board/index/on_value", cam_cfg["name"], light_key)
                    continue
                try:
                    board_id = str(light_val["board"]).upper()
                    index = int(light_val["index"])
                    on_val = int(light_val.get("on_value", 1))
                    on_val = min(max(on_val, _LIGHT_MIN), _LIGHT_MAX)
                except (KeyError, ValueError, TypeError):
                    log.warning("Invalid light config for %s/%s: %s", cam_cfg["name"], light_key, light_val)
                    continue
                lights[key] = LightConfig(board_id=board_id, index=index, on_value=on_val)

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

    def _cancel_latest_active_domain_job(self, domain: str) -> str | None:
        prefix = f"{domain}_"
        for job in self._commands.recent(limit=200):
            name = str(job.get("name", ""))
            state = str(job.get("state", ""))
            if not name.startswith(prefix):
                continue
            if state in {"succeeded", "failed", "canceled"}:
                continue
            job_id = str(job.get("job_id", ""))
            result = self._commands.cancel(job_id)
            if result == "cancel_requested":
                return job_id
            return None
        return None

    def _submit_domain_command(
        self,
        domain: str,
        name: str,
        command: callable,
    ) -> tuple[str, str | None]:
        canceled_job_id = self._cancel_latest_active_domain_job(domain)
        return self._commands.submit(name, command), canceled_job_id

    async def start(self) -> None:
        self._position_store.subscribe(self._on_position_update)
        for state in self._cameras.values():
            await self._open_camera(state)
        await self._start_web()

    async def stop(self) -> None:
        self._position_store.unsubscribe(self._on_position_update)
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
    # ------------------------------------------------------------------
    # Web server
    # ------------------------------------------------------------------

    async def _start_web(self) -> None:
        app = web.Application()
        if self._icons_dir.is_dir():
            app.router.add_static("/assets/icons", path=str(self._icons_dir), show_index=False)
        else:
            log.warning("Icon directory not found: %s", self._icons_dir)
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
        app.router.add_post("/api/head/nozzle/{name}/move", self._api_head_move)
        app.router.add_post("/api/head/nozzle/{name}/rotate", self._api_head_rotate)
        app.router.add_post("/api/head/nozzle/{name}/home", self._api_head_home)
        app.router.add_post("/api/head/nozzle/{name}/park", self._api_head_park)
        app.router.add_post("/api/coord/park", self._api_coord_park)
        app.router.add_post("/api/coord/dispose", self._api_coord_dispose)
        app.router.add_post("/api/coord/homing-fiducial-main", self._api_coord_homing_fiducial_main)
        app.router.add_post("/api/coord/secondary-fiducial", self._api_coord_secondary_fiducial)
        app.router.add_post("/api/coord/nozzle-change", self._api_coord_nozzle_change)
        app.router.add_post("/api/coord/calibration-spot", self._api_coord_calibration_spot)
        app.router.add_post("/api/coord/set-home-here", self._api_coord_set_home_here)
        app.router.add_post("/api/coord/set-calibration-spot-here", self._api_coord_set_calibration_spot_here)
        app.router.add_get("/api/coord/positions", self._api_coord_positions)
        app.router.add_post("/api/nozzle/{name}/move-to-camera", self._api_nozzle_move_to_camera)
        app.router.add_post("/api/nozzle/{name}/move-to-bottom-camera", self._api_nozzle_move_to_bottom_camera)
        app.router.add_post("/api/nozzle/{name}/move-camera-here", self._api_nozzle_move_camera_here)
        app.router.add_post("/api/nozzle/{name}/calculate-offset-top", self._api_nozzle_calculate_offset_top)
        app.router.add_post("/api/nozzle/{name}/vacuum", self._api_nozzle_vacuum)
        app.router.add_post("/api/nozzle/{name}/air", self._api_nozzle_air)
        app.router.add_get("/api/status", self._api_status)
        app.router.add_get("/api/jobs", self._api_jobs)
        app.router.add_get("/api/jobs/{job_id}", self._api_job)
        app.router.add_post("/api/jobs/{job_id}/cancel", self._api_job_cancel)
        app.router.add_post("/api/jobs/cancel-latest/{domain}", self._api_jobs_cancel_latest)

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
                        <div class="col-12 col-md-4 col-xl-3">
              <div class="card h-100 shadow-sm">
                                <div class="card-img-top overflow-hidden bg-dark text-center camera-thumb-wrap">
                  <img src="/stream/{cam_name}"
                                             class="camera-thumb"
                       onerror="this.style.opacity='0.2'" />
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

        coord_card = """
            <div class="col-12 col-md-4 col-xl-3">
                <div class="card shadow-sm h-100">
                    <div class="card-header fw-semibold">Current Coordinates</div>
                    <div class="card-body">
                        <div class="row g-2 mb-1">
                            <div class="col-6">
                                <div class="border rounded p-1 bg-light-subtle">
                                    <div class="small text-muted">X</div>
                                    <div class="fw-semibold" id="coord-x">--</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-1 bg-light-subtle">
                                    <div class="small text-muted">Y</div>
                                    <div class="fw-semibold" id="coord-y">--</div>
                                </div>
                            </div>
                        </div>

                        <div class="row g-2">
                            <div class="col-6">
                                <div class="border rounded p-1">
                                    <div class="small text-muted">Z1 / R1</div>
                                    <div class="fw-semibold"><span id="coord-z1">--</span> / <span id="coord-r1">--</span></div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-1">
                                    <div class="small text-muted">Z2 / R2</div>
                                    <div class="fw-semibold"><span id="coord-z2">--</span> / <span id="coord-r2">--</span></div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-1">
                                    <div class="small text-muted">Z3 / R3</div>
                                    <div class="fw-semibold"><span id="coord-z3">--</span> / <span id="coord-r3">--</span></div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="border rounded p-1">
                                    <div class="small text-muted">Z4 / R4</div>
                                    <div class="fw-semibold"><span id="coord-z4">--</span> / <span id="coord-r4">--</span></div>
                                </div>
                            </div>
                        </div>

                        <div class="d-grid gap-1 mt-2" style="grid-template-columns: 1fr 1fr;">
                            <button class="btn btn-sm btn-outline-secondary" onclick="setHomeHere()" title="Set current XY as the new home/park location">
                                Set Home Here
                            </button>
                            <button class="btn btn-sm btn-outline-secondary" onclick="setCalibrationSpotHere()" title="Set current XY as calibration spot">
                                Set Cal Here
                            </button>
                        </div>
                    </div>
                </div>
            </div>"""

        nozzle_controls = ""
        for nozzle in self._nozzles:
            nozzle_controls += f"""
            <div class="col-12 col-sm-6 col-lg-3">
                <div class="border rounded p-1 h-100 bg-light-subtle nozzle-card">
                    <div class="fw-semibold mb-1 d-flex justify-content-between align-items-center">
                        <span>{nozzle}</span>
                        <span class="badge bg-secondary small" id="nozzle-status-{nozzle}">--</span>
                    </div>
                    
                    <!-- Position Display -->
                    <div class="small mb-1 border-bottom pb-1">
                        <div class="text-muted">Position (offset from camera):</div>
                        <div class="font-monospace small">
                            X: <span id="nozzle-pos-x-{nozzle}">--</span> mm
                            Y: <span id="nozzle-pos-y-{nozzle}">--</span> mm
                        </div>
                    </div>
                    
                    <!-- Valve Controls -->
                    <div class="small mb-1 border-bottom pb-1">
                        <div class="text-muted mb-1">Valves:</div>
                        <div class="d-flex gap-1">
                            <button class="btn btn-sm btn-outline-info flex-grow-1" id="vacuum-btn-{nozzle}" onclick="toggleNozzleVacuum('{nozzle}')">
                                🔸 Vacuum
                            </button>
                            <button class="btn btn-sm btn-outline-info flex-grow-1" id="air-btn-{nozzle}" onclick="toggleNozzleAir('{nozzle}')" style="display:none;">
                                💨 Air
                            </button>
                        </div>
                    </div>
                    
                    <!-- Motion Controls -->
                    <div class="d-grid gap-1 justify-content-center" style="grid-template-columns: 1fr 1fr 1fr; max-width: 210px;">
                        <button class="icon-btn" style="grid-column:1;grid-row:1" onclick="headHome('{nozzle}')" title="Home axis">
                            <img src="/assets/icons/home_axis.png" alt="" class="btn-icon"><span class="visually-hidden">Home</span>
                        </button>
                        <button class="icon-btn" style="grid-column:2;grid-row:1" onclick="headMove('{nozzle}', 1)" title="Move up">
                            <img src="/assets/icons/z_up.png" alt="" class="btn-icon"><span class="visually-hidden">Up</span>
                        </button>
                        <div style="grid-column:3;grid-row:1"></div>
                        <div style="grid-column:1;grid-row:2"></div>
                        <button class="icon-btn" style="grid-column:2;grid-row:2" onclick="headPark('{nozzle}')" title="Park to zero">
                            <img src="/assets/icons/park_zero.png" alt="" class="btn-icon"><span class="visually-hidden">Park</span>
                        </button>
                        <div style="grid-column:3;grid-row:2"></div>
                        <button class="icon-btn" style="grid-column:1;grid-row:3" onclick="headRotate('{nozzle}', -1)" title="Rotate CCW">
                            <img src="/assets/icons/rotate_ccw.png" alt="" class="btn-icon"><span class="visually-hidden">CCW</span>
                        </button>
                        <button class="icon-btn" style="grid-column:2;grid-row:3" onclick="headMove('{nozzle}', -1)" title="Move down">
                            <img src="/assets/icons/z_down.png" alt="" class="btn-icon"><span class="visually-hidden">Down</span>
                        </button>
                        <button class="icon-btn" style="grid-column:3;grid-row:3" onclick="headRotate('{nozzle}', 1)" title="Rotate CW">
                            <img src="/assets/icons/rotate_cw.png" alt="" class="btn-icon"><span class="visually-hidden">CW</span>
                        </button>
                    </div>
                    
                    <!-- Nozzle Sync Buttons -->
                    <div class="d-grid gap-1 mt-1 nozzle-sync" style="grid-template-columns: 1fr 1fr;">
                        <button class="btn btn-sm btn-outline-success" onclick="moveNozzleToCamera('{nozzle}')" title="Align nozzle to camera position">
                            ↓ Align to Cam
                        </button>
                        <button class="btn btn-sm btn-outline-success" onclick="moveCameraToNozzle('{nozzle}')" title="Align camera to nozzle position">
                            ↑ Cam to Nozzle
                        </button>
                        <button class="btn btn-sm btn-outline-warning" style="grid-column: 1 / span 2;" onclick="calculateNozzleOffsetTop('{nozzle}')" title="Click when this nozzle is centered above Homing Fiducial Main">
                            Cal Offset @ Fiducial
                        </button>
                        <button class="btn btn-sm btn-outline-primary" style="grid-column: 1 / span 2;" onclick="moveNozzleToBottomCamera('{nozzle}')" title="Move nozzle above configured BOTTOM camera position">
                            ◎ Above Bottom Cam
                        </button>
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
    <style>
        /* ── density variables ─────────────────────────────── */
        :root {{
            --icon-btn-size: 32px;
            --icon-btn-pad:  3px;
            --icon-grid-col: 40px;
        }}
        /* compact */
        .compact-dashboard {{
            --bs-body-font-size: 0.86rem;
        }}
        .compact-dashboard .container {{
            max-width: 1680px;
        }}
        .compact-dashboard .card-header {{
            padding: 0.3rem 0.5rem;
            font-size: 0.84rem;
        }}
        .compact-dashboard .card-body {{
            padding: 0.4rem;
        }}
        .compact-dashboard .btn-sm {{
            --bs-btn-padding-y: 0.08rem;
            --bs-btn-padding-x: 0.35rem;
            --bs-btn-font-size: 0.7rem;
        }}
        .compact-dashboard .row.g-2 {{
            --bs-gutter-x: 0.3rem;
            --bs-gutter-y: 0.3rem;
        }}
        .compact-dashboard .form-label {{
            margin-bottom: 0.15rem;
            font-size: 0.75rem;
        }}
        /* ultra-compact — overrides compact */
        .ultra-compact .card-header {{
            padding: 0.15rem 0.35rem;
            font-size: 0.76rem;
        }}
        .ultra-compact .card-body {{
            padding: 0.25rem;
        }}
        .ultra-compact .btn-sm {{
            --bs-btn-padding-y: 0.06rem;
            --bs-btn-padding-x: 0.24rem;
            --bs-btn-font-size: 0.66rem;
        }}
        .ultra-compact .row.g-2, .ultra-compact .row.g-1 {{
            --bs-gutter-x: 0.2rem;
            --bs-gutter-y: 0.2rem;
        }}
        .ultra-compact .small {{ font-size: 0.72em; }}
        .ultra-compact {{ --bs-body-font-size: 0.78rem; }}
        .ultra-compact h4 {{ font-size: 0.92rem; margin-bottom: 0.15rem !important; }}
        .ultra-compact .container {{ padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; }}
        .ultra-compact {{
            --icon-btn-size: 26px;
            --icon-btn-pad:  2px;
            --icon-grid-col: 32px;
        }}
        .camera-thumb-wrap {{
            aspect-ratio: 4 / 3;
            min-height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .camera-thumb {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        /* shared nozzle helpers */
        .nozzle-card .font-monospace {{
            line-height: 1.2;
        }}
        .nozzle-card {{
            padding: 0.2rem !important;
        }}
        .nozzle-sync .btn {{
            white-space: nowrap;
            line-height: 1.1;
        }}
        /* icon buttons — size driven by CSS vars */
        .btn-icon {{
            width: var(--icon-btn-size);
            height: var(--icon-btn-size);
            object-fit: contain;
            display: inline-block;
            vertical-align: middle;
            background: #e9ecef;
            border-radius: 5px;
            padding: var(--icon-btn-pad);
        }}
        .icon-btn {{
            border: 0;
            background: transparent;
            padding: 0;
            line-height: 1;
        }}
        .icon-btn:hover .btn-icon {{ background: #dde2e7; }}
        .icon-btn:focus-visible {{
            outline: 2px solid #198754;
            outline-offset: 2px;
            border-radius: 4px;
        }}
    </style>
</head>
<body class="bg-light compact-dashboard ultra-compact" id="app-body">
  <nav class="navbar navbar-dark bg-dark shadow-sm py-1">
    <div class="container-fluid">
      <span class="navbar-brand fw-bold">&#128247; openSMT Vision</span>
    <button class="btn btn-outline-light btn-sm ms-auto" id="density-btn" onclick="toggleDensity()" title="Toggle ultra-compact mode">&#9638; Normal</button>
    </div>
  </nav>
    <div class="container py-1">
        <h4 class="mb-1 text-secondary">Camera Dashboard</h4>
                <div class="row g-1">{cards}{coord_card}</div>

                <div class="row g-1 mt-1">
            <div class="col-12 col-xl-6">
                <div class="card shadow-sm">
                    <div class="card-header fw-semibold">XY Positioning</div>
                    <div class="card-body">
                                                <div class="mb-2">
                            <label for="step-range" class="form-label small text-muted mb-1">Step Size (mm)</label>
                            <input type="range" class="form-range" min="0" max="8" step="1" id="step-range" value="3" oninput="updateStepLabel()">
                            <div class="d-flex justify-content-between small text-muted">
                                <span>0.01</span><span>0.1</span><span>0.5</span><span>1.0</span><span>5.0</span><span>10.0</span><span>25.0</span><span>50.0</span><span>100</span>
                            </div>
                            <div class="small mt-2">Selected: <span class="badge bg-primary" id="step-label">1.0 mm</span></div>
                        </div>

                        <div class="mb-1 d-flex justify-content-center gap-1">
                            <button class="icon-btn" onclick="goHome()" title="Home All">
                                <img src="/assets/icons/home_all.png" alt="" class="btn-icon"><span class="visually-hidden">Home All</span>
                            </button>
                            <button class="icon-btn" onclick="goHomeXY()" title="Home X & Y (simultaneous)">
                                <img src="/assets/icons/home_xy.png" alt="" class="btn-icon"><span class="visually-hidden">Home XY</span>
                            </button>
                            <button class="icon-btn" onclick="goCalibrationSpot()" title="Calibration Spot">
                                <img src="/assets/icons/calibration_spot.png" alt="" class="btn-icon"><span class="visually-hidden">Calibration Spot</span>
                            </button>
                        </div>

                        <div class="d-grid gap-1 justify-content-center" style="grid-template-columns: var(--icon-grid-col) var(--icon-grid-col) var(--icon-grid-col);">
                            <button class="icon-btn" onclick="goHomingFiducialMain()" title="Homing Fiducial Main">
                                <img src="/assets/icons/fiducial_main.png" alt="" class="btn-icon"><span class="visually-hidden">Homing Fiducial Main</span>
                            </button>
                            <button class="icon-btn" onclick="jog(0,1)" title="Jog up">
                                <img src="/assets/icons/move_up.png" alt="" class="btn-icon"><span class="visually-hidden">Up</span>
                            </button>
                            <button class="icon-btn" onclick="goSecondaryFiducial()" title="Secondary Fiducial">
                                <img src="/assets/icons/fiducial_secondary.png" alt="" class="btn-icon"><span class="visually-hidden">Secondary Fiducial</span>
                            </button>

                            <button class="icon-btn" onclick="jog(-1,0)" title="Jog left">
                                <img src="/assets/icons/move_left.png" alt="" class="btn-icon"><span class="visually-hidden">Left</span>
                            </button>
                            <button class="icon-btn" onclick="goPark()" title="Park">
                                <img src="/assets/icons/park_zero.png" alt="" class="btn-icon"><span class="visually-hidden">Park</span>
                            </button>
                            <button class="icon-btn" onclick="jog(1,0)" title="Jog right">
                                <img src="/assets/icons/move_right.png" alt="" class="btn-icon"><span class="visually-hidden">Right</span>
                            </button>

                            <button class="icon-btn" onclick="goNozzleChange()" title="Nozzle Change">
                                <img src="/assets/icons/nozzle_change.png" alt="" class="btn-icon"><span class="visually-hidden">Nozzle Change</span>
                            </button>
                            <button class="icon-btn" onclick="jog(0,-1)" title="Jog down">
                                <img src="/assets/icons/move_down.png" alt="" class="btn-icon"><span class="visually-hidden">Down</span>
                            </button>
                            <button class="icon-btn" onclick="goDispose()" title="Dispose">
                                <img src="/assets/icons/dispose.png" alt="" class="btn-icon"><span class="visually-hidden">Dispose</span>
                            </button>
                        </div>

                        <div class="d-flex align-items-center justify-content-between mt-1">
                            <div class="small text-muted" id="coord-status">Ready</div>
                            <button class="btn btn-outline-danger btn-sm" onclick="cancelCoordJob()">Stop</button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-xl-6">
                <div class="card shadow-sm">
                    <div class="card-header fw-semibold">Head Nozzles</div>
                    <div class="card-body">
                        <div class="mb-2">
                            <label for="nozzle-step-range" class="form-label small text-muted mb-1">Nozzle Step Size (Z mm / Rotation deg)</label>
                            <input type="range" class="form-range" min="0" max="4" step="1" id="nozzle-step-range" value="2" oninput="updateNozzleStepLabel()">
                            <div class="d-flex justify-content-between small text-muted">
                                <span>0.1</span><span>0.5</span><span>1.0</span><span>5.0</span><span>10.0</span>
                            </div>
                            <div class="small mt-2">Selected: <span class="badge bg-primary" id="nozzle-step-label">1.0 mm</span></div>
                        </div>

                        <div class="row g-1">{nozzle_controls}</div>
                        <div class="d-flex align-items-center justify-content-between mt-1">
                            <div class="small text-muted" id="head-status">Ready</div>
                            <button class="btn btn-outline-danger btn-sm" onclick="cancelHeadJob()">Stop</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
  </div>
  <script>
        var coordSocket = null;
      var activeCoordJobId = null;
      var activeHeadJobId = null;

        function selectedStep() {{
            var values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0];
            var idx = parseInt(document.getElementById('step-range').value, 10);
            return values[idx] || 1.0;
        }}

        function updateStepLabel() {{
            var step = selectedStep();
            document.getElementById('step-label').textContent = step.toFixed(2).replace(/[.]00$/, '.0') + ' mm';
        }}

        function selectedNozzleStep() {{
            var values = [0.1, 0.5, 1.0, 5.0, 10.0];
            var idx = parseInt(document.getElementById('nozzle-step-range').value, 10);
            return values[idx] || 1.0;
        }}

        function updateNozzleStepLabel() {{
            var step = selectedNozzleStep();
            document.getElementById('nozzle-step-label').textContent = step.toFixed(1) + ' mm';
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

        function setCoordStatus(text, level) {{
            var el = document.getElementById('coord-status');
            el.textContent = text;
            var cls = 'text-muted';
            if (level === 'error') cls = 'text-danger';
            if (level === 'warning') cls = 'text-warning';
            el.className = 'small mt-1 ' + cls;
        }}

        function setHeadStatus(text, level) {{
            var el = document.getElementById('head-status');
            el.textContent = text;
            var cls = 'text-muted';
            if (level === 'error') cls = 'text-danger';
            if (level === 'warning') cls = 'text-warning';
            el.className = 'small mt-1 ' + cls;
        }}

        function postJson(url, bodyObj) {{
            var opts = {{ method: 'POST', headers: {{'Content-Type': 'application/json'}} }};
            if (bodyObj !== undefined) opts.body = JSON.stringify(bodyObj);
            return fetch(url, opts)
                .then(function(r) {{
                    return r.json()
                        .then(function(data) {{ return {{ ok: r.ok, status: r.status, data: data }}; }})
                        .catch(function() {{ return {{ ok: r.ok, status: r.status, data: {{ error: 'invalid_response' }} }}; }});
                }});
        }}

        function waitForJob(jobId, setStatusFn, successText, failPrefix) {{
            var tries = 0;
            function poll() {{
                fetch('/api/jobs/' + encodeURIComponent(jobId))
                    .then(function(r) {{
                        return r.json()
                            .then(function(data) {{ return {{ ok: r.ok, data: data }}; }})
                            .catch(function() {{ return {{ ok: false, data: {{ error: 'invalid_response' }} }}; }});
                    }})
                    .then(function(res) {{
                        var d = res.data || {{}};
                        if (!res.ok || d.error) {{
                            setStatusFn(failPrefix + ': ' + (d.error || 'job_lookup_failed'), 'error');
                            return;
                        }}

                        if (d.state === 'queued' || d.state === 'running') {{
                            tries += 1;
                            if (tries > 300) {{
                                setStatusFn(failPrefix + ': timeout waiting for job', 'error');
                                return;
                            }}
                            setTimeout(poll, 200);
                            return;
                        }}

                        if (d.state === 'succeeded') {{
                            setStatusFn(successText, 'info');
                            return;
                        }}

                        if (d.state === 'canceled') {{
                            setStatusFn(failPrefix.replace('failed', 'canceled'), 'warning');
                            return;
                        }}

                        if (d.state === 'failed') {{
                            setStatusFn(failPrefix + ': ' + (d.error || 'unknown_error'), 'error');
                            return;
                        }}

                        setStatusFn(failPrefix + ': unexpected_state ' + String(d.state), 'error');
                    }})
                    .catch(function(e) {{
                        setStatusFn(failPrefix + ': ' + e.message, 'error');
                    }});
            }}

            poll();
        }}

        function submitTrackedCommand(url, bodyObj, setStatusFn, pendingText, successText, failPrefix) {{
            setStatusFn(pendingText, 'info');
            postJson(url, bodyObj)
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{
                        setStatusFn(failPrefix + ': ' + (d.error || ('HTTP ' + res.status)), 'error');
                        return;
                    }}
                    if (!d.job_id) {{
                        setStatusFn(failPrefix + ': missing_job_id', 'error');
                        return;
                    }}

                    if (d.previous_job_canceled) {{
                        setStatusFn('Previous command canceled (job ' + d.previous_job_canceled.slice(0, 8) + '), executing latest...', 'warning');
                    }}

                    if (setStatusFn === setCoordStatus) activeCoordJobId = d.job_id;
                    if (setStatusFn === setHeadStatus) activeHeadJobId = d.job_id;

                    waitForJob(d.job_id, setStatusFn, successText(d), failPrefix);
                }})
                .catch(function(e) {{
                    setStatusFn(failPrefix + ': ' + e.message, 'error');
                }});
        }}

        function cancelCoordJob() {{
            var url = activeCoordJobId
                ? '/api/jobs/' + encodeURIComponent(activeCoordJobId) + '/cancel'
                : '/api/jobs/cancel-latest/coord';
            postJson(url)
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{ setCoordStatus('Stop failed: ' + (d.error || ('HTTP ' + res.status)), 'error'); return; }}
                    setCoordStatus('Stop requested', 'warning');
                }})
                .catch(function(e) {{ setCoordStatus('Stop failed: ' + e.message, 'error'); }});
        }}

        function cancelHeadJob() {{
            var url = activeHeadJobId
                ? '/api/jobs/' + encodeURIComponent(activeHeadJobId) + '/cancel'
                : '/api/jobs/cancel-latest/head';
            postJson(url)
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{ setHeadStatus('Stop failed: ' + (d.error || ('HTTP ' + res.status)), 'error'); return; }}
                    setHeadStatus('Stop requested', 'warning');
                }})
                .catch(function(e) {{ setHeadStatus('Stop failed: ' + e.message, 'error'); }});
        }}

        function headMove(nozzle, directionSign) {{
            var step = selectedNozzleStep();
            var delta = directionSign > 0 ? step : -step;
            submitTrackedCommand(
                '/api/head/nozzle/' + encodeURIComponent(nozzle) + '/move',
                {{delta: delta}},
                setHeadStatus,
                'Moving ' + nozzle + '...',
                function(d) {{ return 'Moved ' + nozzle + ' by ' + d.applied_delta + ' mm (Z=' + d.new_z + ')'; }},
                'Nozzle move failed'
            );
        }}

        function headPark(nozzle) {{
            submitTrackedCommand(
                '/api/head/nozzle/' + encodeURIComponent(nozzle) + '/park',
                undefined,
                setHeadStatus,
                'Parking ' + nozzle + '...',
                function(d) {{ return 'Parked ' + nozzle + ' to Z=' + d.parked_z; }},
                'Nozzle park failed'
            );
        }}

        function headHome(nozzle) {{
            submitTrackedCommand(
                '/api/head/nozzle/' + encodeURIComponent(nozzle) + '/home',
                undefined,
                setHeadStatus,
                'Homing ' + nozzle + '...',
                function(d) {{ return 'Homed ' + nozzle + ' (' + d.z_axis + ')'; }},
                'Nozzle home failed'
            );
        }}

        function headRotate(nozzle, directionSign) {{
            var step = selectedNozzleStep();
            var delta = directionSign > 0 ? step : -step;
            submitTrackedCommand(
                '/api/head/nozzle/' + encodeURIComponent(nozzle) + '/rotate',
                {{delta: delta}},
                setHeadStatus,
                'Rotating ' + nozzle + '...',
                function(d) {{ return 'Rotated ' + nozzle + ' by ' + d.applied_delta + ' deg (R=' + d.new_r + ')'; }},
                'Nozzle rotation failed'
            );
        }}

        function jog(dxSign, dySign) {{
            var step = selectedStep();
            var dx = dxSign * step;
            var dy = dySign * step;
            submitTrackedCommand(
                '/api/coord/jog',
                {{dx: dx, dy: dy}},
                setCoordStatus,
                'Jogging XY...',
                function(d) {{ return 'Jogged X=' + d.dx + ' mm, Y=' + d.dy + ' mm'; }},
                'Jog failed'
            );
        }}

        function goPark() {{
            submitTrackedCommand('/api/coord/park', undefined, setCoordStatus, 'Moving to Park...', function() {{ return 'Reached Park'; }}, 'Park failed');
        }}

        function goHome() {{
            submitTrackedCommand('/api/coord/home', undefined, setCoordStatus, 'Homing all axes...', function() {{ return 'Home all completed'; }}, 'Home failed');
        }}

        function goHomeXY() {{
            submitTrackedCommand('/api/coord/home-xy', undefined, setCoordStatus, 'Homing X and Y...', function() {{ return 'Home X and Y completed'; }}, 'Home XY failed');
        }}

        function goDispose() {{
            submitTrackedCommand('/api/coord/dispose', undefined, setCoordStatus, 'Moving to Dispose...', function() {{ return 'Reached Dispose'; }}, 'Dispose failed');
        }}

        function goHomingFiducialMain() {{
            submitTrackedCommand('/api/coord/homing-fiducial-main', undefined, setCoordStatus, 'Moving to Homing Fiducial Main...', function() {{ return 'Reached Homing Fiducial Main'; }}, 'Homing Fiducial Main failed');
        }}

        function goSecondaryFiducial() {{
            submitTrackedCommand('/api/coord/secondary-fiducial', undefined, setCoordStatus, 'Moving to Secondary Fiducial...', function() {{ return 'Reached Secondary Fiducial'; }}, 'Secondary Fiducial failed');
        }}

        function goNozzleChange() {{
            submitTrackedCommand('/api/coord/nozzle-change', undefined, setCoordStatus, 'Moving to Nozzle Change...', function() {{ return 'Reached Nozzle Change'; }}, 'Nozzle Change failed');
        }}

        function goCalibrationSpot() {{
            submitTrackedCommand('/api/coord/calibration-spot', undefined, setCoordStatus, 'Moving to Calibration Spot...', function() {{ return 'Reached Calibration Spot'; }}, 'Calibration Spot failed');
        }}

        function setHomeHere() {{
            postJson('/api/coord/set-home-here')
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{
                        setCoordStatus('Set Home failed: ' + (d.error || ('HTTP ' + res.status)), 'error');
                        return;
                    }}
                    var msg = 'Home/Park set to X=' + Number(d.x).toFixed(2) + ', Y=' + Number(d.y).toFixed(2);
                    if (d.persist_path) msg += ' (saved)';
                    setCoordStatus(msg, 'info');
                }})
                .catch(function(e) {{ setCoordStatus('Set Home failed: ' + e.message, 'error'); }});
        }}

        function setCalibrationSpotHere() {{
            postJson('/api/coord/set-calibration-spot-here')
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{
                        setCoordStatus('Set Cal failed: ' + (d.error || ('HTTP ' + res.status)), 'error');
                        return;
                    }}
                    var msg = 'Calibration spot set to X=' + Number(d.x).toFixed(2) + ', Y=' + Number(d.y).toFixed(2);
                    if (d.persist_path) msg += ' (saved)';
                    setCoordStatus(msg, 'info');
                }})
                .catch(function(e) {{ setCoordStatus('Set Cal failed: ' + e.message, 'error'); }});
        }}

        // ===================================================================
        // NOZZLE CONTROL FUNCTIONS
        // ===================================================================

        function moveNozzleToCamera(nozzle) {{
            submitTrackedCommand(
                '/api/nozzle/' + encodeURIComponent(nozzle) + '/move-to-camera',
                undefined,
                setNozzleStatus,
                'Aligning ' + nozzle + ' to camera...',
                function(d) {{ return 'Aligned ' + nozzle + ' to camera position'; }},
                'Nozzle alignment failed'
            );
        }}

        function moveNozzleToBottomCamera(nozzle) {{
            submitTrackedCommand(
                '/api/nozzle/' + encodeURIComponent(nozzle) + '/move-to-bottom-camera',
                undefined,
                setNozzleStatus,
                'Moving ' + nozzle + ' above BOTTOM camera...',
                function() {{ return 'Moved ' + nozzle + ' above BOTTOM camera'; }},
                'Move to BOTTOM camera failed'
            );
        }}

        function moveCameraToNozzle(nozzle) {{
            submitTrackedCommand(
                '/api/nozzle/' + encodeURIComponent(nozzle) + '/move-camera-here',
                undefined,
                setNozzleStatus,
                'Moving camera to ' + nozzle + '...',
                function(d) {{ return 'Camera aligned to ' + nozzle + ' position'; }},
                'Camera alignment failed'
            );
        }}

        function calculateNozzleOffsetTop(nozzle) {{
            postJson('/api/nozzle/' + encodeURIComponent(nozzle) + '/calculate-offset-top')
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{
                        setHeadStatus('Offset calibration failed for ' + nozzle + ': ' + (d.error || ('HTTP ' + res.status)), 'error');
                        return;
                    }}
                    var msg = nozzle + ' offset set to X=' + Number(d.new_offset_x).toFixed(3) + ' mm, Y=' + Number(d.new_offset_y).toFixed(3) + ' mm';
                    if (d.persisted === false) {{
                        msg += ' (runtime only';
                        if (d.persist_error) msg += ': ' + String(d.persist_error);
                        msg += ')';
                    }}
                    setHeadStatus(msg, 'info');
                    refreshNozzleStatus();
                }})
                .catch(function(e) {{
                    setHeadStatus('Offset calibration failed for ' + nozzle + ': ' + e.message, 'error');
                }});
        }}

        function toggleNozzleVacuum(nozzle) {{
            var btn = document.getElementById('vacuum-btn-' + nozzle);
            var isOn = btn && btn.classList.contains('active');
            var newState = !isOn;
            
            var url = '/api/nozzle/' + encodeURIComponent(nozzle) + '/vacuum?on=' + (newState ? 'true' : 'false');
            postJson(url)
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{
                        console.error('Vacuum toggle failed:', d.error || res.status);
                        return;
                    }}
                    if (btn) {{
                        if (newState) {{
                            btn.classList.add('active');
                            btn.classList.remove('btn-outline-info');
                            btn.classList.add('btn-info');
                        }} else {{
                            btn.classList.remove('active');
                            btn.classList.add('btn-outline-info');
                            btn.classList.remove('btn-info');
                        }}
                    }}
                }})
                .catch(function(e) {{ console.error('Vacuum toggle error:', e.message); }});
        }}

        function toggleNozzleAir(nozzle) {{
            var btn = document.getElementById('air-btn-' + nozzle);
            var isOn = btn && btn.classList.contains('active');
            var newState = !isOn;
            
            var url = '/api/nozzle/' + encodeURIComponent(nozzle) + '/air?on=' + (newState ? 'true' : 'false');
            postJson(url)
                .then(function(res) {{
                    var d = res.data || {{}};
                    if (!res.ok || d.error) {{
                        console.error('Air toggle failed:', d.error || res.status);
                        return;
                    }}
                    if (btn) {{
                        if (newState) {{
                            btn.classList.add('active');
                            btn.classList.remove('btn-outline-info');
                            btn.classList.add('btn-info');
                        }} else {{
                            btn.classList.remove('active');
                            btn.classList.add('btn-outline-info');
                            btn.classList.remove('btn-info');
                        }}
                    }}
                }})
                .catch(function(e) {{ console.error('Air toggle error:', e.message); }});
        }}

        function setNozzleStatus(text, level) {{
            // Aggregate nozzle status message (optional: display in a modal or log)
            console.log('[Nozzle] ' + level.toUpperCase() + ': ' + text);
        }}

        function refreshNozzleStatus() {{
            fetch('/api/status')
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    if (!d.nozzles) return;
                    
                    d.nozzles.forEach(function(nozzle) {{
                        var nameId = nozzle.name;
                        
                        // Update position displays (showing offset from camera)
                        var xElem = document.getElementById('nozzle-pos-x-' + nameId);
                        var yElem = document.getElementById('nozzle-pos-y-' + nameId);
                        if (xElem) xElem.textContent = (nozzle.offset_x || 0).toFixed(1);
                        if (yElem) yElem.textContent = (nozzle.offset_y || 0).toFixed(1);
                        
                        // Update valve button states
                        var vacuumBtn = document.getElementById('vacuum-btn-' + nameId);
                        if (vacuumBtn) {{
                            if (nozzle.vacuum_on) {{
                                vacuumBtn.classList.add('active', 'btn-info');
                                vacuumBtn.classList.remove('btn-outline-info');
                            }} else {{
                                vacuumBtn.classList.remove('active', 'btn-info');
                                vacuumBtn.classList.add('btn-outline-info');
                            }}
                        }}
                        
                        var airBtn = document.getElementById('air-btn-' + nameId);
                        if (airBtn) {{
                            if (nozzle.has_air_valve) {{
                                airBtn.style.display = 'block';
                                if (nozzle.air_on) {{
                                    airBtn.classList.add('active', 'btn-info');
                                    airBtn.classList.remove('btn-outline-info');
                                }} else {{
                                    airBtn.classList.remove('active', 'btn-info');
                                    airBtn.classList.add('btn-outline-info');
                                }}
                            }} else {{
                                airBtn.style.display = 'none';
                            }}
                        }}
                        
                        // Update status badge
                        var statusBadge = document.getElementById('nozzle-status-' + nameId);
                        if (statusBadge) {{
                            var statusText = 'Z=' + (nozzle.z_position !== null ? nozzle.z_position.toFixed(1) : '--');
                            statusBadge.textContent = statusText;
                        }}
                    }});
                }})
                .catch(function(e) {{ console.error('Failed to refresh nozzle status:', e); }});
        }}

        function toggleDensity() {{
            var body = document.getElementById('app-body');
            var btn  = document.getElementById('density-btn');
            var ultra = body.classList.toggle('ultra-compact');
            btn.textContent = ultra ? '\u25a3 Normal' : '\u25a3 Ultra';
            try {{ localStorage.setItem('smt-density', ultra ? 'ultra' : 'compact'); }} catch(e) {{}}
        }}
        (function() {{
            try {{
                if (localStorage.getItem('smt-density') === 'ultra') {{
                    document.getElementById('app-body').classList.add('ultra-compact');
                    document.getElementById('density-btn').textContent = '\u25a3 Normal';
                }}
            }} catch(e) {{}}
        }})();

        updateStepLabel();
        updateNozzleStepLabel();
        connectCoordSocket();
        setInterval(refreshCoords, 1000);
        setInterval(refreshNozzleStatus, 1500);

    // Dashboard uses MJPEG streams directly — no polling required.
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
                                <input type="range" class="form-range flex-grow-1" min="0" max="3" step="1"
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
            value = parseInt(value, 10);
            if (Number.isNaN(value)) value = 0;
            value = Math.max(0, Math.min(3, value));
            var slider = document.getElementById('rng-' + light);
            if (slider) slider.value = String(value);
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

        if not (_LIGHT_MIN <= value <= _LIGHT_MAX):
            return web.json_response({"error": "value_out_of_range"}, status=422)

        try:
            await self._driver.set_analog_out(light_cfg.board_id, light_cfg.index, value)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
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

    async def _api_head_move(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        try:
            body = await request.json()
            delta = float(body.get("delta", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        nozzle = raw_name.upper()
        nozzle_cfg = self._nozzles.get(nozzle)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        current_z = self._position_store.get(nozzle_cfg.z_axis)
        if current_z is None:
            current_z = nozzle_cfg.max_z  # HOME position until first move
        requested_z = current_z + delta
        new_z = min(max(requested_z, nozzle_cfg.min_z), nozzle_cfg.max_z)
        applied_delta = new_z - current_z

        job_id, canceled_prev = self._submit_domain_command(
            "head",
            f"head_move:{nozzle}:{nozzle_cfg.z_axis}",
            lambda axis=nozzle_cfg.z_axis, position=new_z: self._driver.move_axis(axis, position),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle,
            "requested_delta": delta,
            "applied_delta": applied_delta,
            "clamped": new_z != requested_z,
            "new_z": new_z,
        })

    async def _api_head_rotate(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle = raw_name.upper()
        nozzle_cfg = self._nozzles.get(nozzle)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        try:
            body = await request.json()
            delta = float(body.get("delta", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        current_raw = self._position_store.get(nozzle_cfg.r_axis)
        current = float(current_raw if current_raw is not None else 0.0)

        current_norm = current % 360.0
        target_norm = (current_norm + delta) % 360.0

        # Move via the shortest equivalent path on a 360-degree circle.
        shortest_delta = ((target_norm - current_norm + 540.0) % 360.0) - 180.0
        new_r = current + shortest_delta

        job_id, canceled_prev = self._submit_domain_command(
            "head",
            f"head_rotate:{nozzle}:{nozzle_cfg.r_axis}",
            lambda axis=nozzle_cfg.r_axis, position=new_r: self._driver.move_axis(axis, position),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle,
            "r_axis": nozzle_cfg.r_axis,
            "requested_delta": delta,
            "applied_delta": shortest_delta,
            "new_r": new_r,
            "new_r_norm": target_norm,
        })

    async def _api_head_park(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle = raw_name.upper()
        nozzle_cfg = self._nozzles.get(nozzle)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        park_target = min(max(0.0, nozzle_cfg.min_z), nozzle_cfg.max_z)

        job_id, canceled_prev = self._submit_domain_command(
            "head",
            f"head_park:{nozzle}:{nozzle_cfg.z_axis}",
            lambda axis=nozzle_cfg.z_axis, position=park_target: self._driver.move_axis(axis, position),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle,
            "parked_z": park_target,
        })

    async def _api_head_home(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle = raw_name.upper()
        nozzle_cfg = self._nozzles.get(nozzle)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        job_id, canceled_prev = self._submit_domain_command(
            "head",
            f"head_home:{nozzle}:{nozzle_cfg.z_axis}",
            lambda axis=nozzle_cfg.z_axis: self._driver.home_axes([axis]),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle,
            "z_axis": nozzle_cfg.z_axis,
        })

    async def _api_coord_jog(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response(
                {"error": "xy_not_homed", "message": "Home X and Y before XY movement"},
                status=409,
            )

        try:
            body = await request.json()
            dx = float(body.get("dx", 0.0))
            dy = float(body.get("dy", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        job_id, canceled_prev = self._submit_domain_command(
            "coord",
            "coord_jog_xy",
            lambda x=dx, y=dy: self._driver.jog_xy(x, y),
        )
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev, "dx": dx, "dy": dy})

    async def _api_coord_home(self, request: web.Request) -> web.Response:
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_home_all", self._driver.home_all)
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_home_xy(self, request: web.Request) -> web.Response:
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_home_xy", lambda: self._driver.home_group("XY"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_park(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_move:park", lambda: self._driver.move_to_location("park"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_dispose(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_move:dispose", lambda: self._driver.move_to_location("dispose"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_homing_fiducial_main(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_move:fiducial_main", lambda: self._driver.move_to_location("fiducial_main"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_secondary_fiducial(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_move:fiducial_second", lambda: self._driver.move_to_location("fiducial_second"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_nozzle_change(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_move:nozzle_change", lambda: self._driver.move_to_location("nozzle_change"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_calibration_spot(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)
        job_id, canceled_prev = self._submit_domain_command("coord", "coord_move:calibration_spot", lambda: self._driver.move_to_location("calibration_spot"))
        return web.json_response({"status": "accepted", "job_id": job_id, "previous_job_canceled": canceled_prev})

    async def _api_coord_set_home_here(self, request: web.Request) -> web.Response:
        x = self._position_store.get("X")
        y = self._position_store.get("Y")
        if x is None or y is None:
            return web.json_response({"error": "xy_position_unknown", "message": "Home XY first"}, status=409)

        # "Home location" in dashboard context maps to the Park preset location.
        self._location_store.set("park", {"X": x, "Y": y})
        return web.json_response({
            "status": "ok",
            "location": "park",
            "x": x,
            "y": y,
            "persist_path": self._location_store.persist_path(),
        })

    async def _api_coord_set_calibration_spot_here(self, request: web.Request) -> web.Response:
        x = self._position_store.get("X")
        y = self._position_store.get("Y")
        if x is None or y is None:
            return web.json_response({"error": "xy_position_unknown", "message": "Home XY first"}, status=409)

        self._location_store.set("calibration_spot", {"X": x, "Y": y})
        return web.json_response({
            "status": "ok",
            "location": "calibration_spot",
            "x": x,
            "y": y,
            "persist_path": self._location_store.persist_path(),
        })

    async def _api_coord_positions(self, request: web.Request) -> web.Response:
        return web.json_response(self._position_store.all())

    async def _api_jobs(self, request: web.Request) -> web.Response:
        limit_raw = request.query.get("limit", "50")
        try:
            limit = int(limit_raw)
        except ValueError:
            return web.json_response({"error": "invalid_limit"}, status=400)
        return web.json_response({"jobs": self._commands.recent(limit=limit)})

    async def _api_job(self, request: web.Request) -> web.Response:
        job_id = request.match_info.get("job_id", "")
        job = self._commands.get(job_id)
        if job is None:
            return web.json_response({"error": "unknown_job"}, status=404)
        return web.json_response(job)

    async def _api_job_cancel(self, request: web.Request) -> web.Response:
        job_id = request.match_info.get("job_id", "")
        result = self._commands.cancel(job_id)
        if result == "unknown":
            return web.json_response({"error": "unknown_job"}, status=404)
        return web.json_response({"job_id": job_id, "result": result})

    async def _api_jobs_cancel_latest(self, request: web.Request) -> web.Response:
        domain = str(request.match_info.get("domain", "")).lower()
        if domain not in {"coord", "head", "nozzle"}:
            return web.json_response({"error": "invalid_domain"}, status=400)

        recent = self._commands.recent(limit=200)
        for job in recent:
            name = str(job.get("name", ""))
            state = str(job.get("state", ""))
            if state in {"succeeded", "failed", "canceled"}:
                continue

            if domain == "coord" and name.startswith("coord_"):
                result = self._commands.cancel(str(job["job_id"]))
                return web.json_response({"domain": domain, "job_id": job["job_id"], "result": result})

            if domain == "head" and name.startswith("head_"):
                result = self._commands.cancel(str(job["job_id"]))
                return web.json_response({"domain": domain, "job_id": job["job_id"], "result": result})

            if domain == "nozzle" and name.startswith("nozzle_"):
                result = self._commands.cancel(str(job["job_id"]))
                return web.json_response({"domain": domain, "job_id": job["job_id"], "result": result})

        return web.json_response({"domain": domain, "result": "no_active_job"})

    # =========================================================================
    # NOZZLE CONTROL (move-to-camera, offset calibration, valve control)
    # =========================================================================

    def _get_bottom_camera_xy(self) -> tuple[float, float] | None:
        for cam_cfg in self.config.get("cameras", []):
            if str(cam_cfg.get("name", "")).upper() != "BOTTOM":
                continue
            x = cam_cfg.get("x")
            y = cam_cfg.get("y")
            if x is None or y is None:
                return None
            try:
                return float(x), float(y)
            except (TypeError, ValueError):
                return None
        return None

    def _persist_nozzle_offsets(self) -> str | None:
        """Persist current nozzle offsets to runtime sidecar file.

        Returns None on success, or an error message.
        """
        if self._nozzle_offsets_persist_path is None or self._nozzle_config_store is None:
            return "persistence_not_configured"

        payload: dict[str, dict[str, float]] = {}
        for nozzle_name in self._nozzle_config_store.names():
            cfg = self._nozzle_config_store.get(nozzle_name)
            if cfg is None:
                continue
            payload[nozzle_name.upper()] = {
                "offset_x": float(cfg.offset_x),
                "offset_y": float(cfg.offset_y),
            }

        try:
            self._nozzle_offsets_persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._nozzle_offsets_persist_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            return str(exc)

        return None

    async def _api_nozzle_move_to_camera(self, request: web.Request) -> web.Response:
        """Move nozzle XY to align with camera's current position.
        
        Path: POST /api/nozzle/{name}/move-to-camera
        """
        raw_name = request.match_info.get("name", "")
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle_name = raw_name.upper()

        # Verify nozzle exists in hardware config
        if not self._nozzle_config_store:
            return web.json_response({"error": "nozzle_config_not_available"}, status=500)

        nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        # Get current camera position
        cam_x = self._position_store.get("X")
        cam_y = self._position_store.get("Y")
        if cam_x is None or cam_y is None:
            return web.json_response(
                {"error": "camera_position_unknown", "message": "Home XY first"},
                status=409,
            )

        # Submit movement command through job system with nozzle domain
        job_id, canceled_prev = self._submit_domain_command(
            "nozzle",
            f"nozzle_move_to_camera:{nozzle_name}",
            lambda: self._driver.jog_nozzle_to_camera_position(
                nozzle_cfg, (cam_x, cam_y)
            ),
        )

        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle_name,
            "camera_position": {"x": cam_x, "y": cam_y},
            "nozzle_offset": {"dx": nozzle_cfg.offset_x, "dy": nozzle_cfg.offset_y},
            "machine_target": {
                "x": cam_x - nozzle_cfg.offset_x,
                "y": cam_y - nozzle_cfg.offset_y,
            },
        })

    async def _api_nozzle_move_to_bottom_camera(self, request: web.Request) -> web.Response:
        """Move nozzle XY to align with configured BOTTOM camera position.

        Path: POST /api/nozzle/{name}/move-to-bottom-camera
        """
        raw_name = request.match_info.get("name", "")
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle_name = raw_name.upper()

        # Verify nozzle exists in hardware config
        if not self._nozzle_config_store:
            return web.json_response({"error": "nozzle_config_not_available"}, status=500)

        nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        bottom_xy = self._get_bottom_camera_xy()
        if not bottom_xy:
            return web.json_response(
                {
                    "error": "bottom_camera_position_not_configured",
                    "message": "Set camera.cameras[BOTTOM].x and .y in config",
                },
                status=409,
            )
        cam_x, cam_y = bottom_xy

        # Submit movement command through job system with nozzle domain
        job_id, canceled_prev = self._submit_domain_command(
            "nozzle",
            f"nozzle_move_to_bottom_camera:{nozzle_name}",
            lambda: self._driver.jog_nozzle_to_camera_position(
                nozzle_cfg, (cam_x, cam_y)
            ),
        )

        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle_name,
            "bottom_camera_position": {"x": cam_x, "y": cam_y},
            "nozzle_offset": {"dx": nozzle_cfg.offset_x, "dy": nozzle_cfg.offset_y},
            "machine_target": {
                "x": cam_x - nozzle_cfg.offset_x,
                "y": cam_y - nozzle_cfg.offset_y,
            },
        })

    async def _api_nozzle_move_camera_here(self, request: web.Request) -> web.Response:
        """Move camera (machine) to align with nozzle's current position.
        
        Path: POST /api/nozzle/{name}/move-camera-here
        """
        raw_name = request.match_info.get("name", "")
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle_name = raw_name.upper()

        # Verify nozzle exists in hardware config
        if not self._nozzle_config_store:
            return web.json_response({"error": "nozzle_config_not_available"}, status=500)

        nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        # Get current machine position
        mach_x = self._position_store.get("X")
        mach_y = self._position_store.get("Y")
        if mach_x is None or mach_y is None:
            return web.json_response(
                {"error": "machine_position_unknown", "message": "Home XY first"},
                status=409,
            )

        # Calculate current nozzle absolute position
        nozzle_x = mach_x + nozzle_cfg.offset_x
        nozzle_y = mach_y + nozzle_cfg.offset_y

        # Submit movement command through job system with nozzle domain
        job_id, canceled_prev = self._submit_domain_command(
            "nozzle",
            f"nozzle_move_camera_here:{nozzle_name}",
            lambda: self._driver.jog_camera_to_nozzle_position(
                nozzle_cfg, (nozzle_x, nozzle_y)
            ),
        )

        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle_name,
            "current_machine_position": {"x": mach_x, "y": mach_y},
            "nozzle_absolute_position": {"x": nozzle_x, "y": nozzle_y},
            "camera_target": {"x": nozzle_x, "y": nozzle_y},
        })

    async def _api_nozzle_calculate_offset_top(self, request: web.Request) -> web.Response:
        """Calculate nozzle XY offset relative to TOP camera from fiducial alignment.

        Workflow expectation:
        1) Move to Homing Fiducial Main location.
        2) Jog XY until nozzle tip is exactly centered over that fiducial.
        3) Call this endpoint for the specific nozzle.

        Since nozzle absolute XY equals fiducial XY in this pose:
            nozzle = camera + offset
            offset = fiducial - camera

        Path: POST /api/nozzle/{name}/calculate-offset-top
        """
        raw_name = request.match_info.get("name", "")
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle_name = raw_name.upper()

        if not self._nozzle_config_store:
            return web.json_response({"error": "nozzle_config_not_available"}, status=500)

        nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        cam_x = self._position_store.get("X")
        cam_y = self._position_store.get("Y")
        if cam_x is None or cam_y is None:
            return web.json_response(
                {"error": "camera_position_unknown", "message": "Home XY first"},
                status=409,
            )

        fiducial = self._location_store.get("fiducial_main")
        if not fiducial:
            return web.json_response(
                {
                    "error": "fiducial_main_not_configured",
                    "message": "Configure locations.fiducial_main with X and Y",
                },
                status=409,
            )

        fid_x = fiducial.get("X")
        fid_y = fiducial.get("Y")
        if fid_x is None or fid_y is None:
            return web.json_response(
                {
                    "error": "fiducial_main_xy_missing",
                    "message": "locations.fiducial_main must include X and Y",
                },
                status=409,
            )

        old_offset_x = nozzle_cfg.offset_x
        old_offset_y = nozzle_cfg.offset_y

        new_offset_x = float(fid_x) - float(cam_x)
        new_offset_y = float(fid_y) - float(cam_y)

        nozzle_cfg.offset_x = new_offset_x
        nozzle_cfg.offset_y = new_offset_y

        persist_error = self._persist_nozzle_offsets()
        persisted = persist_error is None

        return web.json_response({
            "status": "ok",
            "nozzle": nozzle_name,
            "camera_position": {"x": cam_x, "y": cam_y},
            "fiducial_main": {"x": fid_x, "y": fid_y},
            "old_offset_x": old_offset_x,
            "old_offset_y": old_offset_y,
            "new_offset_x": new_offset_x,
            "new_offset_y": new_offset_y,
            "persisted": persisted,
            "persist_path": str(self._nozzle_offsets_persist_path) if self._nozzle_offsets_persist_path else None,
            "persist_error": persist_error,
            "message": "Offsets updated and persisted" if persisted else "Offsets updated in runtime memory only",
        })

    async def _api_nozzle_vacuum(self, request: web.Request) -> web.Response:
        """Control vacuum valve on a nozzle.
        
        Path: POST /api/nozzle/{name}/vacuum?on=true|false
        """
        raw_name = request.match_info.get("name", "")
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle_name = raw_name.upper()

        # Parse on/off from query
        on_str = request.query.get("on", "false").lower()
        on = on_str in ("true", "1", "yes")

        # Verify nozzle exists in hardware config
        if not self._nozzle_config_store or not self._valve_store:
            return web.json_response({"error": "valve_control_not_available"}, status=500)

        nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        if not nozzle_cfg.vacuum_valve:
            return web.json_response({"error": "nozzle_has_no_vacuum_valve"}, status=400)

        # Submit valve command through job system with nozzle domain
        job_id, canceled_prev = self._submit_domain_command(
            "nozzle",
            f"nozzle_vacuum_{nozzle_name}_{on}",
            lambda on_val=on: self._driver.set_nozzle_valve(nozzle_cfg, "vacuum", on_val),
        )

        # Update valve store state
        await self._valve_store.set_vacuum(nozzle_name, on)

        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle_name,
            "valve": "vacuum",
            "on": on,
            "valve_config": {
                "board": nozzle_cfg.vacuum_valve.board,
                "pin": nozzle_cfg.vacuum_valve.pin,
                "io_type": nozzle_cfg.vacuum_valve.io_type,
            },
        })

    async def _api_nozzle_air(self, request: web.Request) -> web.Response:
        """Control air valve on a nozzle (if equipped).
        
        Path: POST /api/nozzle/{name}/air?on=true|false
        """
        raw_name = request.match_info.get("name", "")
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle_name = raw_name.upper()

        # Parse on/off from query
        on_str = request.query.get("on", "false").lower()
        on = on_str in ("true", "1", "yes")

        # Verify nozzle exists in hardware config
        if not self._nozzle_config_store or not self._valve_store:
            return web.json_response({"error": "valve_control_not_available"}, status=500)

        nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        if not nozzle_cfg.air_valve:
            return web.json_response({"error": "nozzle_has_no_air_valve"}, status=400)

        # Submit valve command through job system with nozzle domain
        job_id, canceled_prev = self._submit_domain_command(
            "nozzle",
            f"nozzle_air_{nozzle_name}_{on}",
            lambda on_val=on: self._driver.set_nozzle_valve(nozzle_cfg, "air", on_val),
        )

        # Update valve store state
        await self._valve_store.set_air(nozzle_name, on)

        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle_name,
            "valve": "air",
            "on": on,
            "valve_config": {
                "board": nozzle_cfg.air_valve.board,
                "pin": nozzle_cfg.air_valve.pin,
                "io_type": nozzle_cfg.air_valve.io_type,
            },
        })

    async def _api_status(self, request: web.Request) -> web.Response:
        """Get comprehensive system status including positions, nozzles, and valve states.
        
        Path: GET /api/status
        """
        positions = self._position_store.all()
        nozzle_data = []
        camera_data = []

        for cam_name, cam_state in self._cameras.items():
            cap_open = False
            if cam_state.cap is not None:
                try:
                    cap_open = bool(cam_state.cap.isOpened())
                except Exception:
                    cap_open = False
            camera_data.append({
                "name": cam_name,
                "online": cap_open,
                "stream_path": f"/camera/{cam_name}",
            })

        camera_data.sort(key=lambda c: str(c.get("name", "")))

        if self._nozzle_config_store and self._valve_store:
            cam_x = positions.get("X")
            cam_y = positions.get("Y")
            cam_x_f = cam_x if cam_x is not None else 0.0
            cam_y_f = cam_y if cam_y is not None else 0.0

            for nozzle_name in self._nozzle_config_store.names():
                nozzle_cfg = self._nozzle_config_store.get(nozzle_name)
                valve_state = self._valve_store.get(nozzle_name)

                z_pos = positions.get(nozzle_cfg.z_axis)

                # Derive r_axis from z_axis (same logic as RuntimeNozzleConfig)
                z_axis = nozzle_cfg.z_axis
                if z_axis.startswith("Z") and len(z_axis) > 1:
                    r_axis = f"R{z_axis[1:]}"
                else:
                    r_axis = "R1"

                r_pos = positions.get(r_axis)

                # Calculate nozzle absolute position (None when machine not yet homed)
                nozzle_abs_x = (cam_x_f + nozzle_cfg.offset_x) if cam_x is not None else None
                nozzle_abs_y = (cam_y_f + nozzle_cfg.offset_y) if cam_y is not None else None

                nozzle_data.append({
                    "name": nozzle_name,
                    "z_axis": nozzle_cfg.z_axis,
                    "r_axis": r_axis,
                    "z_position": z_pos,
                    "r_position": r_pos,
                    "offset_x": nozzle_cfg.offset_x,
                    "offset_y": nozzle_cfg.offset_y,
                    "absolute_x": nozzle_abs_x,
                    "absolute_y": nozzle_abs_y,
                    "vacuum_on": valve_state.vacuum_on if valve_state else False,
                    "air_on": valve_state.air_on if valve_state else False,
                    "has_air_valve": nozzle_cfg.air_valve is not None,
                })

        return web.json_response({
            "positions": positions,
            "nozzles": nozzle_data,
            "cameras": camera_data,
            "camera_position": {"x": positions.get("X"), "y": positions.get("Y")},
        })

    async def _ws_coord(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self._coord_ws_clients.add(ws)

        # Send current state immediately on connect
        await ws.send_json({"type": "coord", "positions": self._position_store.all()})

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT and msg.data == "refresh":
                    await ws.send_json({"type": "coord", "positions": self._position_store.all()})
                elif msg.type in {web.WSMsgType.CLOSE, web.WSMsgType.ERROR}:
                    break
        finally:
            self._coord_ws_clients.discard(ws)

        return ws

    async def _on_position_update(self, axis: str, value: float) -> None:
        """Called by PositionStore whenever any axis position changes."""
        await self._broadcast_coord_positions()

    async def _broadcast_coord_positions(self) -> None:
        """Push the current position snapshot to all connected WebSocket clients."""
        if not self._coord_ws_clients:
            return

        payload = self._position_store.all()
        if payload == self._last_coord_broadcast:
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
