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
    device: int | str
    fps: float
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
        self._web_port = int(config.get("web_port", 8080))
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
            device: int | str = cam_cfg.get("device", 0)
            if isinstance(device, str) and device.isdigit():
                device = int(device)

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
                lights=lights,
                pipeline_names=pipe_names,
            )
            self._cameras[cam_name] = CameraState(config=cfg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self.node.on_query("*", self._handle_query)
        self.node.on_set("*", self._handle_set)

        for state in self._cameras.values():
            await self._open_camera(state)

        await self._start_web()

    async def stop(self) -> None:
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
                async with state.frame_lock:
                    state.frame = frame
            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.001, interval - elapsed))

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
            return processed, results

        return frame, {}

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

    # ------------------------------------------------------------------
    # Web server
    # ------------------------------------------------------------------

    async def _start_web(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._web_main)
        app.router.add_get("/camera/{name}", self._web_camera)
        app.router.add_get("/stream/{name}", self._web_stream)
        app.router.add_get("/thumb/{name}", self._web_thumb)
        app.router.add_post("/api/camera/{name}/light/{light}", self._api_light)
        app.router.add_post("/api/camera/{name}/pipeline/{pipe}", self._api_pipeline)
        app.router.add_get("/api/camera/{name}/pipeline/result", self._api_pipeline_result)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._web_port)
        await site.start()
        log.info("Camera web UI on http://0.0.0.0:%d", self._web_port)

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
  </div>
  <script>
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
        </div>
      </div>

      <!-- Controls column -->
      <div class="col-lg-4">

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
