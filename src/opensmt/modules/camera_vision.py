from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aiohttp import web

from opensmt.vision import PassthroughPipeline, VisionPipelineBase

from opensmt.hardware.driver import HardwareDriver
from opensmt.runtime.command_runner import CommandRunner
from opensmt.store.feeder_config import FeederConfigStore, feeder_from_dict
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
_UI_LIGHT_MAX = 2


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
    rotation_deg: float
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
    current_rotation_deg: float = field(default_factory=lambda: 0.0)


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
        feeder_config_store: FeederConfigStore | None = None,
        feeders_persist_dir: Path | None = None,
        valve_store: ValveStore | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self._driver = driver
        self._position_store = position_store
        self._location_store = location_store
        self._nozzle_config_store = nozzle_config_store
        self._feeder_config_store = feeder_config_store
        self._feeders_persist_dir = feeders_persist_dir
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
        camera_res_persist_path_raw = config.get("_camera_resolutions_persist_path")
        self._camera_resolutions_persist_path = Path(str(camera_res_persist_path_raw)).expanduser() if camera_res_persist_path_raw else None

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
                rotation_deg=float(cam_cfg.get("rotation_deg", 0.0)),
                lights=lights,
                pipeline_names=pipe_names,
            )
            state = CameraState(config=cfg)
            state.current_rotation_deg = cfg.rotation_deg
            for light_key in cfg.lights:
                state.light_values[light_key] = 0
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
        for state in self._cameras.values():
            await self._open_camera(state)
        await self._start_web()

    async def stop(self) -> None:
        for state in self._cameras.values():
            await self._close_camera(state)

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------------
    # Camera capture
    # ------------------------------------------------------------------

    async def _close_camera(self, state: CameraState) -> None:
        if state.capture_task:
            state.capture_task.cancel()
            try:
                await state.capture_task
            except asyncio.CancelledError:
                pass
            finally:
                state.capture_task = None
        if state.cap:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, state.cap.release)
            state.cap = None

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

        rotation = float(state.current_rotation_deg) % 360.0
        if abs(rotation - 90.0) < 1e-6:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif abs(rotation - 180.0) < 1e-6:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif abs(rotation - 270.0) < 1e-6:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif abs(rotation) > 1e-6:
            h, w = frame.shape[:2]
            center = (w / 2.0, h / 2.0)
            mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
            cos_v = abs(mat[0, 0])
            sin_v = abs(mat[0, 1])

            new_w = int((h * sin_v) + (w * cos_v))
            new_h = int((h * cos_v) + (w * sin_v))

            mat[0, 2] += (new_w / 2.0) - center[0]
            mat[1, 2] += (new_h / 2.0) - center[1]
            frame = cv2.warpAffine(frame, mat, (new_w, new_h))
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
        app.router.add_get("/thumb/{name}", self._web_thumb)
        app.router.add_post("/api/coord/jog", self._api_coord_jog)
        app.router.add_post("/api/coord/home", self._api_coord_home)
        app.router.add_post("/api/coord/home-xy", self._api_coord_home_xy)
        app.router.add_post("/api/head/nozzle/{name}/move", self._api_head_move)
        app.router.add_post("/api/head/nozzle/{name}/move-absolute", self._api_head_move_absolute)
        app.router.add_post("/api/head/nozzle/{name}/move-standard-down", self._api_head_move_standard_down)
        app.router.add_post("/api/head/nozzle/{name}/rotate", self._api_head_rotate)
        app.router.add_post("/api/head/nozzle/{name}/home", self._api_head_home)
        app.router.add_post("/api/head/nozzle/{name}/park", self._api_head_park)
        app.router.add_post("/api/head/nozzle/{name}/vacuum", self._api_head_nozzle_vacuum)
        app.router.add_post("/api/coord/park", self._api_coord_park)
        app.router.add_post("/api/coord/dispose", self._api_coord_dispose)
        app.router.add_post("/api/coord/homing-fiducial-main", self._api_coord_homing_fiducial_main)
        app.router.add_post("/api/coord/secondary-fiducial", self._api_coord_secondary_fiducial)
        app.router.add_post("/api/coord/nozzle-change", self._api_coord_nozzle_change)
        app.router.add_post("/api/coord/calibration-spot", self._api_coord_calibration_spot)
        app.router.add_post("/api/coord/set-home-here", self._api_coord_set_home_here)
        app.router.add_post("/api/coord/set-calibration-spot-here", self._api_coord_set_calibration_spot_here)
        app.router.add_post("/api/coord/move-xy", self._api_coord_move_xy)
        app.router.add_get("/api/coord/positions", self._api_coord_positions)
        app.router.add_post("/api/config/location/{name}", self._api_config_location_set)
        app.router.add_get("/api/feeders", self._api_feeders)
        app.router.add_post("/api/feeders", self._api_feeder_create)
        app.router.add_get("/api/feeders/{feeder_id}", self._api_feeder_get)
        app.router.add_put("/api/feeders/{feeder_id}", self._api_feeder_put)
        app.router.add_post("/api/feeders/{feeder_id}/reset", self._api_feeder_reset)
        app.router.add_post("/api/feeders/{feeder_id}/advance-pick", self._api_feeder_advance_pick)
        app.router.add_post("/api/nozzle/{name}/move-to-camera", self._api_nozzle_move_to_camera)
        app.router.add_post("/api/nozzle/{name}/move-to-bottom-camera", self._api_nozzle_move_to_bottom_camera)
        app.router.add_post("/api/nozzle/{name}/move-camera-here", self._api_nozzle_move_camera_here)
        app.router.add_post("/api/nozzle/{name}/calculate-offset-top", self._api_nozzle_calculate_offset_top)
        app.router.add_post("/api/camera/{name}/light", self._api_camera_light)
        app.router.add_post("/api/camera/{name}/settings", self._api_camera_settings)
        app.router.add_post("/api/camera/{name}/calibrate-resolution", self._api_camera_calibrate_resolution)
        app.router.add_get("/api/status", self._api_status)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._web_host, self._web_port)
        await site.start()
        log.info("Camera web UI on http://%s:%d", self._web_host, self._web_port)

    async def _web_thumb(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            raise web.HTTPNotFound()
        name = raw_name.upper()
        state = self._cameras.get(name)
        if not state:
            raise web.HTTPNotFound()

        frame, _ = await self._apply_pipeline(state)
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

    async def _api_coord_move_xy(self, request: web.Request) -> web.Response:
        if not (self._driver.is_axis_homed("X") and self._driver.is_axis_homed("Y")):
            return web.json_response({"error": "xy_not_homed"}, status=409)

        try:
            body = await request.json()
            x = float(body.get("x"))
            y = float(body.get("y"))
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        job_id, canceled_prev = self._submit_domain_command(
            "coord",
            f"coord_move_xy:{x}:{y}",
            lambda tx=x, ty=y: self._driver.move_axes({"X": tx, "Y": ty}),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "x": x,
            "y": y,
        })

    async def _api_coord_positions(self, request: web.Request) -> web.Response:
        return web.json_response(self._position_store.all())

    def _persist_feeder_config(self, feeder_payload: dict[str, Any]) -> str | None:
        if self._feeders_persist_dir is None:
            return "feeders_persist_dir_not_configured"
        feeder_id = str(feeder_payload.get("feeder_id", "")).upper().strip()
        if not feeder_id:
            return "invalid_feeder_id"

        try:
            self._feeders_persist_dir.mkdir(parents=True, exist_ok=True)
            path = self._feeders_persist_dir / f"{feeder_id}.json"
            path.write_text(json.dumps(feeder_payload, indent=2), encoding="utf-8")
            return None
        except Exception as exc:
            return str(exc)

    async def _api_feeders(self, request: web.Request) -> web.Response:
        if self._feeder_config_store is None:
            return web.json_response({"feeders": []})
        feeders = [feeder.to_status() for feeder in self._feeder_config_store.all()]
        feeders.sort(key=lambda feeder: (str(feeder.get("feeder_type", "")), str(feeder.get("feeder_id", ""))))
        return web.json_response({"feeders": feeders})

    async def _api_feeder_create(self, request: web.Request) -> web.Response:
        if self._feeder_config_store is None:
            return web.json_response({"error": "feeders_not_available"}, status=404)

        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        feeder_type = str(body.get("feeder_type", "")).strip().lower()
        if not feeder_type:
            return web.json_response({"error": "missing_feeder_type"}, status=400)

        # Generate a unique 16-hex feeder id for new entries.
        feeder_id = ""
        for _ in range(100):
            candidate = secrets.token_hex(8).upper()
            if self._feeder_config_store.get(candidate) is None:
                feeder_id = candidate
                break
        if not feeder_id:
            return web.json_response({"error": "unable_to_allocate_feeder_id"}, status=500)

        part_number = str(body.get("manufacturer_part_number", "NEW_PART")).strip() or "NEW_PART"
        pick_x = float(body.get("pick_x", 0.0) or 0.0)
        pick_y = float(body.get("pick_y", 0.0) or 0.0)
        pick_h = float(body.get("pick_height", 0.0) or 0.0)

        merged: dict[str, Any] = {
            "feeder_id": feeder_id,
            "feeder_type": feeder_type,
            "pick_location": {"x": pick_x, "y": pick_y},
            "pick_height": pick_h,
            "manufacturer_part_number": part_number,
            "type_data": {},
            "actual_data": {},
        }

        if feeder_type == "tray_feeder":
            merged["type_data"] = {
                "x_step": 0.0,
                "y_step": 0.0,
                "parts_available_x": 1,
                "parts_available_y": 1,
                "preferred_direction": "X",
            }
            merged["actual_data"] = {
                "parts_picked": 0,
                "current_index_x": 0,
                "current_index_y": 0,
                "current_pick": {"x": pick_x, "y": pick_y},
                "last_pick": {"x": pick_x, "y": pick_y},
            }

        try:
            created = feeder_from_dict(merged)
        except Exception as exc:
            return web.json_response({"error": f"invalid_feeder_payload: {exc}"}, status=400)

        self._feeder_config_store.upsert(created)
        persist_error = self._persist_feeder_config(created.to_status())

        return web.json_response(
            {
                "status": "ok",
                "feeder": created.to_status(),
                "persisted": persist_error is None,
                "persist_error": persist_error,
                "persist_dir": str(self._feeders_persist_dir) if self._feeders_persist_dir else None,
            },
            status=201,
        )

    async def _api_feeder_get(self, request: web.Request) -> web.Response:
        if self._feeder_config_store is None:
            return web.json_response({"error": "feeders_not_available"}, status=404)

        feeder_id = str(request.match_info.get("feeder_id", "")).upper().strip()
        feeder = self._feeder_config_store.get(feeder_id)
        if feeder is None:
            return web.json_response({"error": "unknown_feeder"}, status=404)

        return web.json_response({"feeder": feeder.to_status()})

    async def _api_feeder_put(self, request: web.Request) -> web.Response:
        if self._feeder_config_store is None:
            return web.json_response({"error": "feeders_not_available"}, status=404)

        feeder_id = str(request.match_info.get("feeder_id", "")).upper().strip()
        current = self._feeder_config_store.get(feeder_id)
        if current is None:
            return web.json_response({"error": "unknown_feeder"}, status=404)

        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        merged = current.to_status()
        for key in ("pick_location", "type_data", "actual_data"):
            if key in body and isinstance(body.get(key), dict):
                merged[key] = copy.deepcopy(body[key])
        for key in ("pick_height", "manufacturer_part_number"):
            if key in body:
                merged[key] = body[key]

        merged["feeder_id"] = feeder_id
        merged["feeder_type"] = current.feeder_type

        try:
            updated = feeder_from_dict(merged)
        except Exception as exc:
            return web.json_response({"error": f"invalid_feeder_payload: {exc}"}, status=400)

        self._feeder_config_store.upsert(updated)
        persist_error = self._persist_feeder_config(updated.to_status())

        return web.json_response({
            "status": "ok",
            "feeder": updated.to_status(),
            "persisted": persist_error is None,
            "persist_error": persist_error,
            "persist_dir": str(self._feeders_persist_dir) if self._feeders_persist_dir else None,
        })

    async def _api_feeder_reset(self, request: web.Request) -> web.Response:
        if self._feeder_config_store is None:
            return web.json_response({"error": "feeders_not_available"}, status=404)

        feeder_id = str(request.match_info.get("feeder_id", "")).upper().strip()
        current = self._feeder_config_store.get(feeder_id)
        if current is None:
            return web.json_response({"error": "unknown_feeder"}, status=404)

        merged = current.to_status()
        pick_location = merged.get("pick_location") if isinstance(merged.get("pick_location"), dict) else {}
        base_x = float(pick_location.get("x", 0.0) or 0.0)
        base_y = float(pick_location.get("y", 0.0) or 0.0)
        actual = merged.get("actual_data")
        if not isinstance(actual, dict):
            actual = {}
        actual["parts_picked"] = 0
        actual["current_index_x"] = 0
        actual["current_index_y"] = 0
        actual["current_pick"] = {"x": base_x, "y": base_y}
        actual["last_pick"] = {"x": base_x, "y": base_y}
        merged["actual_data"] = actual

        try:
            updated = feeder_from_dict(merged)
        except Exception as exc:
            return web.json_response({"error": f"invalid_feeder_payload: {exc}"}, status=400)

        self._feeder_config_store.upsert(updated)
        persist_error = self._persist_feeder_config(updated.to_status())

        return web.json_response({
            "status": "ok",
            "feeder": updated.to_status(),
            "persisted": persist_error is None,
            "persist_error": persist_error,
            "persist_dir": str(self._feeders_persist_dir) if self._feeders_persist_dir else None,
        })

    async def _api_feeder_advance_pick(self, request: web.Request) -> web.Response:
        if self._feeder_config_store is None:
            return web.json_response({"error": "feeders_not_available"}, status=404)

        feeder_id = str(request.match_info.get("feeder_id", "")).upper().strip()
        current = self._feeder_config_store.get(feeder_id)
        if current is None:
            return web.json_response({"error": "unknown_feeder"}, status=404)

        merged = current.to_status()
        feeder_type = str(merged.get("feeder_type", "")).lower()
        if feeder_type != "tray_feeder":
            return web.json_response({"error": "advance_pick_only_supported_for_tray_feeder"}, status=400)

        pick_location = merged.get("pick_location") if isinstance(merged.get("pick_location"), dict) else {}
        type_data = merged.get("type_data") if isinstance(merged.get("type_data"), dict) else {}
        actual = merged.get("actual_data") if isinstance(merged.get("actual_data"), dict) else {}

        try:
            base_x = float(pick_location.get("x", 0.0) or 0.0)
            base_y = float(pick_location.get("y", 0.0) or 0.0)
            x_step = float(type_data.get("x_step", 0.0) or 0.0)
            y_step = float(type_data.get("y_step", 0.0) or 0.0)
            x_count = int(type_data.get("parts_available_x", 0) or 0)
            y_count = int(type_data.get("parts_available_y", 0) or 0)
            ix = int(actual.get("current_index_x", 0) or 0)
            iy = int(actual.get("current_index_y", 0) or 0)
            parts_picked = int(actual.get("parts_picked", 0) or 0)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_tray_feeder_data"}, status=400)

        if x_count <= 0 or y_count <= 0:
            return web.json_response({"error": "invalid_parts_available_dimensions"}, status=400)

        max_ix = x_count - 1
        max_iy = y_count - 1
        ix = max(0, min(ix, max_ix))
        iy = max(0, min(iy, max_iy))

        pref = str(type_data.get("preferred_direction", "X")).upper().strip()
        if pref not in {"X", "Y"}:
            pref = "X"

        next_ix = ix
        next_iy = iy
        exhausted = False

        if pref == "X":
            if ix < max_ix:
                next_ix = ix + 1
            elif iy < max_iy:
                next_ix = 0
                next_iy = iy + 1
            else:
                exhausted = True
        else:
            if iy < max_iy:
                next_iy = iy + 1
            elif ix < max_ix:
                next_iy = 0
                next_ix = ix + 1
            else:
                exhausted = True

        if exhausted:
            return web.json_response(
                {
                    "status": "exhausted",
                    "error": "no_next_pick_position",
                    "feeder": merged,
                },
                status=409,
            )

        current_pick = actual.get("current_pick") if isinstance(actual.get("current_pick"), dict) else {}
        actual["last_pick"] = {
            "x": float(current_pick.get("x", base_x) or base_x),
            "y": float(current_pick.get("y", base_y) or base_y),
        }
        actual["current_index_x"] = next_ix
        actual["current_index_y"] = next_iy
        actual["parts_picked"] = max(0, parts_picked + 1)
        actual["current_pick"] = {
            "x": base_x + (next_ix * x_step),
            "y": base_y + (next_iy * y_step),
        }
        merged["actual_data"] = actual

        try:
            updated = feeder_from_dict(merged)
        except Exception as exc:
            return web.json_response({"error": f"invalid_feeder_payload: {exc}"}, status=400)

        self._feeder_config_store.upsert(updated)
        persist_error = self._persist_feeder_config(updated.to_status())

        return web.json_response(
            {
                "status": "ok",
                "feeder": updated.to_status(),
                "persisted": persist_error is None,
                "persist_error": persist_error,
                "persist_dir": str(self._feeders_persist_dir) if self._feeders_persist_dir else None,
            }
        )

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

    def _set_camera_xy(self, camera_name: str, x: float, y: float) -> bool:
        for cam_cfg in self.config.get("cameras", []):
            if str(cam_cfg.get("name", "")).upper() != str(camera_name).upper():
                continue
            cam_cfg["x"] = float(x)
            cam_cfg["y"] = float(y)
            return True
        return False

    def _persist_nozzle_offsets(self) -> str | None:
        """Persist current nozzle offset/tip values into nozzle config chunk file."""
        if self._nozzle_offsets_persist_path is None or self._nozzle_config_store is None:
            return "persistence_not_configured"

        try:
            payload: dict[str, Any] = {}
            if self._nozzle_offsets_persist_path.is_file():
                loaded = json.loads(self._nozzle_offsets_persist_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded

            camera_obj = payload.get("camera")
            if not isinstance(camera_obj, dict):
                camera_obj = {}
                payload["camera"] = camera_obj

            nozzles_list = camera_obj.get("nozzles")
            if not isinstance(nozzles_list, list):
                return "invalid_persist_file:no_camera.nozzles_list"

            cfg_by_name: dict[str, Any] = {}
            for nozzle_name in self._nozzle_config_store.names():
                cfg = self._nozzle_config_store.get(nozzle_name)
                if cfg is not None:
                    cfg_by_name[nozzle_name.upper()] = cfg

            for item in nozzles_list:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).upper().strip()
                cfg = cfg_by_name.get(name)
                if cfg is None:
                    continue
                item["offset_x"] = float(cfg.offset_x)
                item["offset_y"] = float(cfg.offset_y)
                item["tip_id"] = cfg.tip_id
                item["standard_down_z"] = float(cfg.standard_down_z) if cfg.standard_down_z is not None else None

            self._nozzle_offsets_persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._nozzle_offsets_persist_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            return str(exc)

        return None

    def _persist_camera_resolutions(self) -> str | None:
        if self._camera_resolutions_persist_path is None:
            return "persistence_not_configured"

        try:
            payload: dict[str, Any] = {}
            if self._camera_resolutions_persist_path.is_file():
                loaded = json.loads(self._camera_resolutions_persist_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded

            camera_obj = payload.get("camera")
            if not isinstance(camera_obj, dict):
                camera_obj = {}
                payload["camera"] = camera_obj

            cameras_list = camera_obj.get("cameras")
            if not isinstance(cameras_list, list):
                return "invalid_persist_file:no_camera.cameras_list"

            states_by_name = {name.upper(): state for name, state in self._cameras.items()}
            for item in cameras_list:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).upper().strip()
                state = states_by_name.get(name)
                if state is None:
                    continue
                item["resolution_dpcm_x"] = float(state.config.resolution_dpcm_x)
                item["resolution_dpcm_y"] = float(state.config.resolution_dpcm_y)

            self._camera_resolutions_persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._camera_resolutions_persist_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            return str(exc)
        return None

    async def _api_camera_calibrate_resolution(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        cam_name = raw_name.upper()
        state = self._cameras.get(cam_name)
        if state is None:
            return web.json_response({"error": "unknown_camera"}, status=404)

        try:
            body = await request.json()
            dpcm_x = float(body.get("resolution_dpcm_x"))
            dpcm_y = float(body.get("resolution_dpcm_y"))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        if not (math.isfinite(dpcm_x) and math.isfinite(dpcm_y) and dpcm_x > 0.0 and dpcm_y > 0.0):
            return web.json_response({"error": "invalid_resolution_values"}, status=400)

        state.config.resolution_dpcm_x = dpcm_x
        state.config.resolution_dpcm_y = dpcm_y

        persist_error = self._persist_camera_resolutions()
        return web.json_response(
            {
                "status": "ok",
                "camera": cam_name,
                "resolution_dpcm_x": dpcm_x,
                "resolution_dpcm_y": dpcm_y,
                "persisted": persist_error is None,
                "persist_error": persist_error,
                "persist_path": str(self._camera_resolutions_persist_path) if self._camera_resolutions_persist_path else None,
            }
        )

    async def _api_camera_settings(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        cam_name = raw_name.upper()
        state = self._cameras.get(cam_name)
        if state is None:
            return web.json_response({"error": "unknown_camera"}, status=404)

        try:
            body = await request.json()
            device = str(body.get("device", state.config.device)).strip()
            fps = float(body.get("fps", state.config.fps))
            resolution_dpcm_x = float(body.get("resolution_dpcm_x"))
            resolution_dpcm_y = float(body.get("resolution_dpcm_y"))
            rotation_deg = float(body.get("rotation_deg", 0.0))
            flip_horizontal = bool(body.get("flip_horizontal", False))
            flip_vertical = bool(body.get("flip_vertical", False))
            x_raw = body.get("x")
            y_raw = body.get("y")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        pos_x: float | None = None
        pos_y: float | None = None
        if x_raw is not None or y_raw is not None:
            if x_raw is None or y_raw is None:
                return web.json_response({"error": "incomplete_camera_position"}, status=400)
            try:
                pos_x = float(x_raw)
                pos_y = float(y_raw)
            except (TypeError, ValueError):
                return web.json_response({"error": "invalid_camera_position"}, status=400)
            if not (math.isfinite(pos_x) and math.isfinite(pos_y)):
                return web.json_response({"error": "invalid_camera_position"}, status=400)

        if not (
            math.isfinite(resolution_dpcm_x)
            and math.isfinite(resolution_dpcm_y)
            and math.isfinite(fps)
            and math.isfinite(rotation_deg)
            and resolution_dpcm_x > 0.0
            and resolution_dpcm_y > 0.0
            and fps > 0.0
        ):
            return web.json_response({"error": "invalid_camera_settings"}, status=400)

        if not device.startswith("/dev/"):
            return web.json_response({"error": "invalid_device_path"}, status=400)

        reopen_required = device != state.config.device

        state.config.device = device
        state.config.fps = fps
        state.config.resolution_dpcm_x = resolution_dpcm_x
        state.config.resolution_dpcm_y = resolution_dpcm_y
        state.config.flip_horizontal = flip_horizontal
        state.config.flip_vertical = flip_vertical
        state.config.rotation_deg = rotation_deg
        state.current_rotation_deg = rotation_deg

        if pos_x is not None and pos_y is not None:
            self._set_camera_xy(cam_name, pos_x, pos_y)

        if reopen_required:
            await self._close_camera(state)
            await self._open_camera(state)

        return web.json_response(
            {
                "status": "ok",
                "camera": cam_name,
                "device": device,
                "fps": fps,
                "resolution_dpcm_x": resolution_dpcm_x,
                "resolution_dpcm_y": resolution_dpcm_y,
                "flip_horizontal": flip_horizontal,
                "flip_vertical": flip_vertical,
                "rotation_deg": rotation_deg,
                "x": pos_x,
                "y": pos_y,
                "reopened": reopen_required,
            }
        )

    async def _api_config_location_set(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        try:
            body = await request.json()
            x = float(body.get("x"))
            y = float(body.get("y"))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        if not (math.isfinite(x) and math.isfinite(y)):
            return web.json_response({"error": "invalid_coordinates"}, status=400)

        name = raw_name.lower()
        self._location_store.set(name, {"X": x, "Y": y})
        return web.json_response(
            {
                "status": "ok",
                "name": name,
                "x": x,
                "y": y,
                "persist_path": self._location_store.persist_path(),
            }
        )

    async def _api_camera_light(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        cam_name = raw_name.upper()
        state = self._cameras.get(cam_name)
        if state is None:
            return web.json_response({"error": "unknown_camera"}, status=404)

        try:
            body = await request.json()
            light_key = str(body.get("light", "")).strip().lower()
            value = int(body.get("value"))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        if light_key not in state.config.lights:
            return web.json_response({"error": "unknown_light"}, status=404)
        if value < _LIGHT_MIN or value > _UI_LIGHT_MAX:
            return web.json_response({"error": "invalid_light_value", "allowed": [_LIGHT_MIN, 1, _UI_LIGHT_MAX]}, status=400)

        light_cfg = state.config.lights[light_key]
        try:
            await self._driver.set_analog_out(light_cfg.board_id, light_cfg.index, value)
        except Exception as exc:
            return web.json_response({"error": "driver_set_analog_failed", "detail": str(exc)}, status=500)

        state.light_values[light_key] = value
        return web.json_response(
            {
                "status": "ok",
                "camera": cam_name,
                "light": light_key,
                "value": value,
                "board": light_cfg.board_id,
                "index": light_cfg.index,
            }
        )

    async def _api_head_move_absolute(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle = raw_name.upper()
        nozzle_cfg = self._nozzles.get(nozzle)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        try:
            body = await request.json()
            target = float(body.get("z"))
        except (json.JSONDecodeError, TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        clamped = min(max(target, nozzle_cfg.min_z), nozzle_cfg.max_z)

        job_id, canceled_prev = self._submit_domain_command(
            "head",
            f"head_move_abs:{nozzle}:{nozzle_cfg.z_axis}",
            lambda axis=nozzle_cfg.z_axis, position=clamped: self._driver.move_axis(axis, position),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle,
            "requested_z": target,
            "applied_z": clamped,
            "clamped": clamped != target,
        })

    async def _api_head_move_standard_down(self, request: web.Request) -> web.Response:
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle = raw_name.upper()
        nozzle_cfg = self._nozzles.get(nozzle)
        if not nozzle_cfg:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        if not self._nozzle_config_store:
            return web.json_response({"error": "nozzle_config_not_available"}, status=500)

        cfg = self._nozzle_config_store.get(nozzle)
        if cfg is None or cfg.standard_down_z is None:
            return web.json_response(
                {"error": "standard_down_not_set", "message": "Set standard_down_z for this nozzle/tip first"},
                status=409,
            )

        target = float(cfg.standard_down_z)
        clamped = min(max(target, nozzle_cfg.min_z), nozzle_cfg.max_z)

        job_id, canceled_prev = self._submit_domain_command(
            "head",
            f"head_move_standard_down:{nozzle}:{nozzle_cfg.z_axis}",
            lambda axis=nozzle_cfg.z_axis, position=clamped: self._driver.move_axis(axis, position),
        )
        return web.json_response({
            "status": "accepted",
            "job_id": job_id,
            "previous_job_canceled": canceled_prev,
            "nozzle": nozzle,
            "standard_down_z": target,
            "applied_z": clamped,
            "clamped": clamped != target,
            "tip_id": cfg.tip_id,
        })

    async def _api_head_nozzle_vacuum(self, request: web.Request) -> web.Response:
        """Control nozzle vacuum via analog output on board XY.

        POST /api/head/nozzle/{name}/vacuum  body: {"on": bool}
        N1 -> XY analog index 2, N2 -> 3, N3 -> 4, N4 -> 5, …
        Values: 0 = off, 255 = on.
        """
        raw_name = request.match_info["name"]
        if not _NAME_RE.match(raw_name):
            return web.json_response({"error": "invalid_name"}, status=400)

        nozzle = raw_name.upper()
        if nozzle not in self._nozzles:
            return web.json_response({"error": "unknown_nozzle"}, status=400)

        try:
            body = await request.json()
            on = bool(body.get("on", False))
        except (json.JSONDecodeError, TypeError):
            return web.json_response({"error": "invalid_body"}, status=400)

        m = re.search(r"(\d+)$", nozzle)
        if not m:
            return web.json_response({"error": "cannot_determine_nozzle_index"}, status=400)
        analog_index = int(m.group(1)) + 1  # N1->2, N2->3, …
        value = 255 if on else 0

        try:
            await self._driver.set_analog_out("XY", analog_index, value)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

        return web.json_response({
            "nozzle": nozzle,
            "vacuum": "on" if on else "off",
            "board": "XY",
            "analog_index": analog_index,
            "value": value,
        })

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
        old_standard_down_z = nozzle_cfg.standard_down_z

        new_offset_x = float(fid_x) - float(cam_x)
        new_offset_y = float(fid_y) - float(cam_y)

        nozzle_cfg.offset_x = new_offset_x
        nozzle_cfg.offset_y = new_offset_y

        # Capture Z at calibration moment as standard-down reference for the current tip.
        cur_z = self._position_store.get(nozzle_cfg.z_axis)
        if cur_z is not None:
            nozzle_cfg.standard_down_z = float(cur_z)

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
            "old_standard_down_z": old_standard_down_z,
            "new_standard_down_z": nozzle_cfg.standard_down_z,
            "persisted": persisted,
            "persist_path": str(self._nozzle_offsets_persist_path) if self._nozzle_offsets_persist_path else None,
            "persist_error": persist_error,
            "message": "Offsets updated and persisted" if persisted else "Offsets updated in runtime memory only",
        })

    async def _api_status(self, request: web.Request) -> web.Response:
        """Get comprehensive system status including positions, nozzles, and valve states.
        
        Path: GET /api/status
        """
        positions = self._position_store.all()
        nozzle_data = []
        camera_data = []
        feeder_data = []

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
                "resolution_dpcm_x": float(cam_state.config.resolution_dpcm_x),
                "resolution_dpcm_y": float(cam_state.config.resolution_dpcm_y),
                "flip_horizontal": bool(cam_state.config.flip_horizontal),
                "flip_vertical": bool(cam_state.config.flip_vertical),
                "lights": {
                    key: int(cam_state.light_values.get(key, 0))
                    for key in sorted(cam_state.config.lights.keys())
                },
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
                    "tip_id": nozzle_cfg.tip_id,
                    "standard_down_z": nozzle_cfg.standard_down_z,
                    "absolute_x": nozzle_abs_x,
                    "absolute_y": nozzle_abs_y,
                    "vacuum_on": valve_state.vacuum_on if valve_state else False,
                    "air_on": valve_state.air_on if valve_state else False,
                    "has_air_valve": nozzle_cfg.air_valve is not None,
                })

        if self._feeder_config_store is not None:
            feeder_data = [feeder.to_status() for feeder in self._feeder_config_store.all()]
            feeder_data.sort(key=lambda feeder: (str(feeder.get("feeder_type", "")), str(feeder.get("feeder_id", ""))))

        return web.json_response({
            "positions": positions,
            "nozzles": nozzle_data,
            "cameras": camera_data,
            "feeders": feeder_data,
            "camera_position": {"x": positions.get("X"), "y": positions.get("Y")},
        })

