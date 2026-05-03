from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QPointF, QSize, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Icon helpers
# ---------------------------------------------------------------------------

_ICON_SZ = 22     # individual icon size inside composites (px)
_ARROW_W = 10     # arrow sprite width (px)
_BTN_SQ  = 34     # square side for single-icon buttons (px)

_pm_cache: dict[str, QPixmap] = {}


def _assets_dir() -> Path:
    """Resolve  <workspace-root>/assets/icons/opensmt-ui/32/ from this file."""
    return (
        Path(__file__).resolve()
        .parent   # monitor/
        .parent   # opensmt/
        .parent   # src/
        .parent   # project root
        / "assets" / "icons" / "opensmt-ui" / "32"
    )


def _load_pm(name: str, size: int = _ICON_SZ) -> QPixmap:
    """Load a named PNG from the assets dir, scaled to *size* px square."""
    key = f"{name}@{size}"
    if key not in _pm_cache:
        path = _assets_dir() / f"{name}.png"
        if path.exists():
            pm = QPixmap(str(path))
            if not pm.isNull():
                pm = pm.scaled(
                    size, size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        else:
            pm = QPixmap(size, size)
            pm.fill(Qt.GlobalColor.transparent)
        _pm_cache[key] = pm
    return _pm_cache[key]


def _make_camera_pm(size: int = _ICON_SZ, lens_color: QColor | None = None) -> QPixmap:
    """Draw a simple camera icon programmatically.

    *lens_color* distinguishes top camera (blue, default) from bottom camera (green).
    """
    if lens_color is None:
        lens_color = QColor(60, 120, 200)
    key = f"__cam@{size}@{lens_color.name()}"
    if key in _pm_cache:
        return _pm_cache[key]

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    body_y = int(size * 0.35)
    body_h = int(size * 0.50)
    bump_w = int(size * 0.30)
    bump_h = int(size * 0.15)
    bump_x = int(size * 0.36)

    body_col = QColor(210, 210, 210)
    p.setBrush(body_col)
    p.setPen(QColor(150, 150, 150))
    p.drawRoundedRect(1, body_y, size - 2, body_h, 3, 3)
    p.drawRect(bump_x, body_y - bump_h, bump_w, bump_h)

    lens_r = size * 0.16
    cx, cy = size / 2.0, body_y + body_h / 2.0
    p.setBrush(lens_color)
    p.setPen(lens_color.darker(160))
    p.drawEllipse(
        int(cx - lens_r), int(cy - lens_r),
        int(lens_r * 2), int(lens_r * 2),
    )
    p.end()
    _pm_cache[key] = pm
    return pm


def _make_arrow_pm(w: int = _ARROW_W, h: int = _ICON_SZ) -> QPixmap:
    """Draw a small right-pointing filled triangle."""
    key = f"__arrow@{w}x{h}"
    if key in _pm_cache:
        return _pm_cache[key]
    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(190, 190, 190))
    p.setPen(Qt.PenStyle.NoPen)
    hh = h * 0.32
    cy = h / 2.0
    poly = QPolygonF([
        QPointF(2,       cy - hh),
        QPointF(w - 2,   cy),
        QPointF(2,       cy + hh),
    ])
    p.drawPolygon(poly)
    p.end()
    _pm_cache[key] = pm
    return pm


def _compose_pm(left: QPixmap, right: QPixmap) -> QPixmap:
    """Return a pixmap that renders *left* → arrow → *right* in a single row."""
    gap = 2
    total_w = _ICON_SZ + gap + _ARROW_W + gap + _ICON_SZ
    pm = QPixmap(total_w, _ICON_SZ)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.drawPixmap(0, 0, left)
    p.drawPixmap(_ICON_SZ + gap, 0, _make_arrow_pm())
    p.drawPixmap(_ICON_SZ + gap + _ARROW_W + gap, 0, right)
    p.end()
    return pm


def _icon(name: str) -> QIcon:
    return QIcon(_load_pm(name))


def _sq_btn(pm_or_name: str | QPixmap, tooltip: str = "") -> QPushButton:
    """Create a square icon-only button for single-icon actions."""
    pm = _load_pm(pm_or_name) if isinstance(pm_or_name, str) else pm_or_name
    btn = QPushButton()
    btn.setIcon(QIcon(pm))
    btn.setIconSize(QSize(_ICON_SZ, _ICON_SZ))
    btn.setFixedSize(_BTN_SQ, _BTN_SQ)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def _dual_btn(left: QPixmap, right: QPixmap, tooltip: str = "") -> QPushButton:
    """Create a composite (left → arrow → right) icon button."""
    pm = _compose_pm(left, right)
    gap = 2
    total_w = _ICON_SZ + gap + _ARROW_W + gap + _ICON_SZ
    btn = QPushButton()
    btn.setIcon(QIcon(pm))
    btn.setIconSize(QSize(total_w, _ICON_SZ))
    btn.setFixedSize(total_w + 12, _BTN_SQ)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


class ControlApiClient(QObject):
    request_failed = Signal(str)

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._net = QNetworkAccessManager(self)

    def set_base_url(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def get_json(self, path: str, on_done: Callable[[bool, int, dict[str, Any]], None]) -> None:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        reply = self._net.get(request)
        self._wire_reply(reply, on_done)

    def post_json(
        self,
        path: str,
        payload: dict[str, Any] | None,
        on_done: Callable[[bool, int, dict[str, Any]], None],
    ) -> None:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        reply = self._net.post(request, body)
        self._wire_reply(reply, on_done)

    def _wire_reply(
        self,
        reply: QNetworkReply,
        on_done: Callable[[bool, int, dict[str, Any]], None],
    ) -> None:
        def _finish() -> None:
            raw = bytes(reply.readAll())
            status_obj = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
            status = int(status_obj) if status_obj is not None else 0

            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(data, dict):
                    data = {"value": data}
            except Exception:
                data = {"error": "invalid_json_response"}

            ok = reply.error() == QNetworkReply.NetworkError.NoError and (200 <= status < 300)
            if not ok and not data.get("error"):
                data["error"] = reply.errorString()
            on_done(ok, status, data)
            reply.deleteLater()

        reply.finished.connect(_finish)


class NozzleRow(QWidget):
    action_requested = Signal(str, str)

    # Shared pixmaps — built once, reused for every row
    _nozzle_pm:     QPixmap | None = None
    _cam_top_pm:    QPixmap | None = None
    _cam_bottom_pm: QPixmap | None = None
    _cal_pm:        QPixmap | None = None

    @classmethod
    def _init_pms(cls) -> None:
        if cls._nozzle_pm is not None:
            return
        cls._nozzle_pm     = _load_pm("nozzle_change")
        cls._cam_top_pm    = _make_camera_pm()                              # blue lens = top
        cls._cam_bottom_pm = _make_camera_pm(lens_color=QColor(40, 160, 80))  # green = bottom
        cls._cal_pm        = _load_pm("calibration_spot")

    def __init__(self, nozzle_name: str) -> None:
        super().__init__()
        NozzleRow._init_pms()
        self.nozzle_name = nozzle_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(3)

        self._title  = QLabel(nozzle_name)
        self._title.setMinimumWidth(28)
        self._offset = QLabel("X=--  Y=--")
        self._offset.setMinimumWidth(110)
        self._z = QLabel("Z=--")
        self._r = QLabel("R=--")

        # Composite buttons: nozzle→cam_top, cam_top→nozzle,
        #                    nozzle→cam_bottom, single cal
        b_align  = _dual_btn(self._nozzle_pm,  self._cam_top_pm,    "Align nozzle to top camera")
        b_cam    = _dual_btn(self._cam_top_pm,  self._nozzle_pm,    "Move top camera to nozzle")
        b_bottom = _dual_btn(self._nozzle_pm,  self._cam_bottom_pm, "Move nozzle above bottom camera")
        b_cal    = _sq_btn(self._cal_pm,                            "Calculate nozzle offset at fiducial")

        b_align.clicked.connect( lambda: self.action_requested.emit(self.nozzle_name, "align_to_cam"))
        b_cam.clicked.connect(   lambda: self.action_requested.emit(self.nozzle_name, "cam_to_nozzle"))
        b_bottom.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "above_bottom"))
        b_cal.clicked.connect(   lambda: self.action_requested.emit(self.nozzle_name, "cal_offset"))

        layout.addWidget(self._title)
        layout.addWidget(self._offset)
        layout.addWidget(self._z)
        layout.addWidget(self._r)
        layout.addStretch(1)
        layout.addWidget(b_align)
        layout.addWidget(b_cam)
        layout.addWidget(b_bottom)
        layout.addWidget(b_cal)

    def apply_status(self, nozzle: dict[str, Any]) -> None:
        ox = nozzle.get("offset_x")
        oy = nozzle.get("offset_y")
        z = nozzle.get("z_position")
        r = nozzle.get("r_position")

        self._offset.setText(f"off X={self._fmt(ox)} Y={self._fmt(oy)}")
        self._z.setText(f"Z={self._fmt(z)}")
        self._r.setText(f"R={self._fmt(r)}")

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            if value is None:
                return "--"
            return f"{float(value):.3f}"
        except Exception:
            return "--"


class ControlWindow(QMainWindow):
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.setWindowTitle("openSMT Control")
        self.resize(960, 620)

        base_url = f"http://{host}:{port}"
        self._api = ControlApiClient(base_url)
        self._nozzle_rows: dict[str, NozzleRow] = {}
        self._nozzle_placeholder: QLabel | None = None

        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(4)

        # ── Connection row ──────────────────────────────────────────────────
        top = QHBoxLayout()
        self._host = QLineEdit(host)
        self._port = QLineEdit(str(port))
        self._connect_btn = QPushButton("Apply Host")
        self._conn_state = QLabel("Ready")

        self._host.setMaximumWidth(160)
        self._port.setMaximumWidth(60)

        top.addWidget(QLabel("Host"))
        top.addWidget(self._host)
        top.addWidget(QLabel("Port"))
        top.addWidget(self._port)
        top.addWidget(self._connect_btn)
        top.addStretch(1)
        top.addWidget(self._conn_state)
        outer.addLayout(top)

        # ── XY controls ─────────────────────────────────────────────────────
        xy_group = QGroupBox("XY")
        xy_layout = QVBoxLayout(xy_group)
        xy_layout.setContentsMargins(4, 4, 4, 4)
        xy_layout.setSpacing(3)

        step_row = QHBoxLayout()
        self._xy_step = QComboBox()
        self._xy_steps = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        for val in self._xy_steps:
            self._xy_step.addItem(f"{val:g} mm", val)
        self._xy_step.setCurrentIndex(3)
        self._coord_label = QLabel("X=--  Y=--")

        step_row.addWidget(QLabel("Step"))
        step_row.addWidget(self._xy_step)
        step_row.addStretch(1)
        step_row.addWidget(self._coord_label)
        xy_layout.addLayout(step_row)

        xy_buttons = QGridLayout()
        xy_buttons.setSpacing(3)

        b_home_xy = _sq_btn("home_xy",        "Home XY axes")
        b_fid     = _sq_btn("fiducial_main",   "Move to homing fiducial")
        b_park    = _sq_btn("park_zero",       "Move to park position")
        b_l       = _sq_btn("move_left",       "Jog left")
        b_r       = _sq_btn("move_right",      "Jog right")
        b_u       = _sq_btn("move_up",         "Jog up (Y+)")
        b_d       = _sq_btn("move_down",       "Jog down (Y−)")

        xy_buttons.addWidget(b_home_xy, 0, 0)
        xy_buttons.addWidget(b_fid,     0, 1)
        xy_buttons.addWidget(b_park,    0, 2)
        xy_buttons.addWidget(b_u,       1, 1)
        xy_buttons.addWidget(b_l,       2, 0)
        xy_buttons.addWidget(b_d,  2, 1)
        xy_buttons.addWidget(b_r,  2, 2)
        xy_layout.addLayout(xy_buttons)

        outer.addWidget(xy_group)

        # ── Nozzle controls ─────────────────────────────────────────────────
        noz_group = QGroupBox("Nozzles")
        noz_layout = QVBoxLayout(noz_group)
        noz_layout.setContentsMargins(3, 3, 3, 3)
        noz_scroll = QScrollArea()
        noz_scroll.setWidgetResizable(True)
        noz_container = QWidget()
        self._nozzle_layout = QVBoxLayout(noz_container)
        self._nozzle_layout.setContentsMargins(2, 2, 2, 2)
        self._nozzle_layout.setSpacing(2)
        noz_scroll.setWidget(noz_container)
        noz_layout.addWidget(noz_scroll)
        outer.addWidget(noz_group, 1)

        # ── Log ─────────────────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._log.setMinimumHeight(100)
        self._log.setMaximumHeight(160)
        outer.addWidget(self._log)

        # Wire buttons
        self._connect_btn.clicked.connect(self._apply_host)
        b_home_xy.clicked.connect(lambda: self._post_action("/api/coord/home-xy", None, "Home XY"))
        b_fid.clicked.connect(lambda: self._post_action("/api/coord/homing-fiducial-main", None, "Move to Homing Fiducial Main"))
        b_park.clicked.connect(lambda: self._post_action("/api/coord/park", None, "Move to Park"))
        b_l.clicked.connect(lambda: self._jog_xy(-1.0, 0.0))
        b_r.clicked.connect(lambda: self._jog_xy(1.0, 0.0))
        b_u.clicked.connect(lambda: self._jog_xy(0.0, 1.0))
        b_d.clicked.connect(lambda: self._jog_xy(0.0, -1.0))

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(800)
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.start()

        self._poll_status()

    def _apply_host(self) -> None:
        host = self._host.text().strip() or "127.0.0.1"
        port = self._port.text().strip() or "8080"
        self._api.set_base_url(f"http://{host}:{port}")
        self._conn_state.setText("Host updated")
        self._log_line(f"Base URL set to http://{host}:{port}")
        self._poll_status()

    def _xy_step_mm(self) -> float:
        value = self._xy_step.currentData()
        return float(value) if value is not None else 1.0

    def _jog_xy(self, sx: float, sy: float) -> None:
        step = self._xy_step_mm()
        payload = {"dx": sx * step, "dy": sy * step}
        self._post_action("/api/coord/jog", payload, f"Jog dx={payload['dx']}, dy={payload['dy']}")

    def _post_action(self, path: str, payload: dict[str, Any] | None, title: str) -> None:
        self._api.post_json(path, payload, lambda ok, status, data: self._handle_action_result(title, ok, status, data))

    def _handle_action_result(self, title: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if ok:
            job = data.get("job_id")
            if job:
                self._log_line(f"OK: {title} (job {job})")
            else:
                self._log_line(f"OK: {title}")
            return

        err = data.get("error", "request_failed")
        self._log_line(f"ERR {status}: {title}: {err}")

    def _poll_status(self) -> None:
        self._api.get_json("/api/status", self._handle_status)

    def _handle_status(self, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._conn_state.setText("Disconnected")
            self._log_line(f"ERR {status}: status poll failed: {data.get('error', 'request_failed')}")
            return

        self._conn_state.setText("Connected")
        positions = data.get("positions", {}) if isinstance(data.get("positions"), dict) else {}
        self._coord_label.setText(
            f"X={self._fmt(positions.get('X'))}  Y={self._fmt(positions.get('Y'))}"
        )

        nozzles = data.get("nozzles", []) if isinstance(data.get("nozzles"), list) else []
        self._sync_nozzle_rows(nozzles)

    def _sync_nozzle_rows(self, nozzles: list[dict[str, Any]]) -> None:
        if self._nozzle_placeholder is not None:
            self._nozzle_placeholder.setParent(None)
            self._nozzle_placeholder.deleteLater()
            self._nozzle_placeholder = None

        known = set(self._nozzle_rows.keys())
        incoming = set()

        for nozzle in nozzles:
            name = str(nozzle.get("name", "")).upper()
            if not name:
                continue
            incoming.add(name)

            if name not in self._nozzle_rows:
                row = NozzleRow(name)
                row.action_requested.connect(self._on_nozzle_action)
                self._nozzle_rows[name] = row
                self._nozzle_layout.addWidget(row)

            self._nozzle_rows[name].apply_status(nozzle)

        removed = known - incoming
        for name in removed:
            row = self._nozzle_rows.pop(name)
            row.setParent(None)
            row.deleteLater()

        if not self._nozzle_rows:
            self._nozzle_placeholder = QLabel("No nozzles found in /api/status")
            self._nozzle_placeholder.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._nozzle_layout.addWidget(self._nozzle_placeholder)

    def _on_nozzle_action(self, nozzle: str, action: str) -> None:
        if action == "align_to_cam":
            self._post_action(
                f"/api/nozzle/{nozzle}/move-to-camera",
                None,
                f"{nozzle}: Align to camera",
            )
            return

        if action == "cam_to_nozzle":
            self._post_action(
                f"/api/nozzle/{nozzle}/move-camera-here",
                None,
                f"{nozzle}: Move camera to nozzle",
            )
            return

        if action == "above_bottom":
            self._post_action(
                f"/api/nozzle/{nozzle}/move-to-bottom-camera",
                None,
                f"{nozzle}: Move above bottom camera",
            )
            return

        if action == "cal_offset":
            self._api.post_json(
                f"/api/nozzle/{nozzle}/calculate-offset-top",
                None,
                lambda ok, status, data: self._handle_calibration_result(nozzle, ok, status, data),
            )

    def _handle_calibration_result(self, nozzle: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(
                f"ERR {status}: {nozzle}: Cal offset failed: {data.get('error', 'request_failed')}"
            )
            return

        ox = self._fmt(data.get("new_offset_x"))
        oy = self._fmt(data.get("new_offset_y"))
        persisted = bool(data.get("persisted", False))
        suffix = "persisted" if persisted else "runtime_only"
        if not persisted and data.get("persist_error"):
            suffix += f" ({data.get('persist_error')})"
        self._log_line(f"OK: {nozzle}: offset X={ox} Y={oy} [{suffix}]")
        self._poll_status()

    def _log_line(self, text: str) -> None:
        self._log.append(text)

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            if value is None:
                return "--"
            return f"{float(value):.3f}"
        except Exception:
            return "--"


def run_qt_control(host: str, port: int) -> None:
    app = QApplication.instance() or QApplication([])
    win = ControlWindow(host=host, port=port)
    win.show()
    app.exec()
