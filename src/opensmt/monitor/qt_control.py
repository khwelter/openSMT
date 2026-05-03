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
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_ICON_SZ = 22
_ARROW_W = 10
_BTN_SQ = 34
_JOG_BTN_SQ = 48

_COLOR_RED = QColor(255, 40, 40)
_COLOR_BLUE = QColor(20, 120, 255)

_FEEDER_TYPE_TITLES: list[tuple[str, str]] = [
    ("tray_feeder", "Tray Feeders"),
    ("auto_feeder", "Auto Feeders"),
    ("push_pull_feeder", "Push/Pull Feeders"),
    ("vibration_feeder", "Vibration Feeders"),
    ("label_feeder", "Label Feeders"),
    ("tube_feeder", "Tube Feeders"),
]

_pm_cache: dict[str, QPixmap] = {}


def _assets_dir() -> Path:
    return (
        Path(__file__).resolve()
        .parent
        .parent
        .parent
        .parent
        / "assets"
        / "icons"
        / "opensmt-ui"
        / "32"
    )


def _load_pm(name: str, size: int = _ICON_SZ) -> QPixmap:
    key = f"{name}@{size}"
    if key not in _pm_cache:
        path = _assets_dir() / f"{name}.png"
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        if path.exists():
            src = QPixmap(str(path))
            if not src.isNull():
                pm = src.scaled(
                    size,
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        _pm_cache[key] = pm
    return _pm_cache[key]


def _tint_pm(src: QPixmap, color: QColor) -> QPixmap:
    key = f"__tint@{int(src.cacheKey())}@{color.name()}"
    if key in _pm_cache:
        return _pm_cache[key]

    tinted = QPixmap(src.size())
    tinted.fill(Qt.GlobalColor.transparent)
    p = QPainter(tinted)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(tinted.rect(), color)
    p.end()
    _pm_cache[key] = tinted
    return tinted


def _make_camera_pm(
    size: int = _ICON_SZ,
    body_color: QColor | None = None,
    lens_color: QColor | None = None,
) -> QPixmap:
    if body_color is None:
        body_color = _COLOR_BLUE
    if lens_color is None:
        lens_color = _COLOR_RED

    key = f"__cam@{size}@{body_color.name()}@{lens_color.name()}"
    if key in _pm_cache:
        return _pm_cache[key]

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    body_y = int(size * 0.35)
    body_h = int(size * 0.5)
    bump_w = int(size * 0.3)
    bump_h = int(size * 0.15)
    bump_x = int(size * 0.36)

    p.setBrush(body_color)
    p.setPen(body_color.darker(140))
    p.drawRoundedRect(1, body_y, size - 2, body_h, 3, 3)
    p.drawRect(bump_x, body_y - bump_h, bump_w, bump_h)

    lens_r = size * 0.16
    cx, cy = size / 2.0, body_y + body_h / 2.0
    p.setBrush(lens_color)
    p.setPen(lens_color.darker(145))
    p.drawEllipse(int(cx - lens_r), int(cy - lens_r), int(lens_r * 2), int(lens_r * 2))
    p.end()

    _pm_cache[key] = pm
    return pm


def _make_arrow_pm(w: int = _ARROW_W, h: int = _ICON_SZ) -> QPixmap:
    key = f"__arrow@{w}x{h}"
    if key in _pm_cache:
        return _pm_cache[key]

    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(_COLOR_RED)
    p.setPen(Qt.PenStyle.NoPen)

    hh = h * 0.32
    cy = h / 2.0
    poly = QPolygonF([
        QPointF(2, cy - hh),
        QPointF(w - 2, cy),
        QPointF(2, cy + hh),
    ])
    p.drawPolygon(poly)
    p.end()

    _pm_cache[key] = pm
    return pm


def _compose_pm(left: QPixmap, right: QPixmap) -> QPixmap:
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


def _sq_btn(pm_or_name: str | QPixmap, tooltip: str = "") -> QPushButton:
    pm = _load_pm(pm_or_name) if isinstance(pm_or_name, str) else pm_or_name
    btn = QPushButton()
    btn.setIcon(QIcon(pm))
    btn.setIconSize(QSize(_ICON_SZ, _ICON_SZ))
    btn.setFixedSize(_BTN_SQ, _BTN_SQ)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def _dual_btn(left: QPixmap, right: QPixmap, tooltip: str = "") -> QPushButton:
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


def _xy_btn(
    pm_or_name: str | QPixmap,
    tooltip: str = "",
    *,
    tint: QColor | None = None,
) -> QPushButton:
    """Create larger, high-contrast buttons for the XY pane."""
    pm = _load_pm(pm_or_name) if isinstance(pm_or_name, str) else pm_or_name
    if tint is not None:
        pm = _tint_pm(pm, tint)

    btn = QPushButton()
    btn.setIcon(QIcon(pm))
    btn.setIconSize(QSize(30, 30))
    btn.setFixedSize(_JOG_BTN_SQ, _JOG_BTN_SQ)
    btn.setStyleSheet(
        "QPushButton {"
        " background:#0f1f3a;"
        " border:2px solid #1d4d9a;"
        " border-radius:9px;"
        "}"
        "QPushButton:hover {"
        " background:#18335f;"
        " border:2px solid #ff2a2a;"
        "}"
        "QPushButton:pressed {"
        " background:#24508f;"
        "}"
    )
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

    def put_json(
        self,
        path: str,
        payload: dict[str, Any] | None,
        on_done: Callable[[bool, int, dict[str, Any]], None],
    ) -> None:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        reply = self._net.put(request, body)
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


class CameraTile(QFrame):
    def __init__(self, camera_name: str) -> None:
        super().__init__()
        self.camera_name = camera_name
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        cam_pm = _make_camera_pm(body_color=_COLOR_BLUE, lens_color=_COLOR_RED)

        self._preview = QLabel("No Feed")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(150, 95)
        self._preview.setStyleSheet("background:#070b14; border:1px solid #2a3d66; color:#8ba3cf;")

        hdr = QHBoxLayout()
        self._icon = QLabel()
        self._icon.setPixmap(cam_pm)
        self._name = QLabel(camera_name)
        self._state = QLabel("offline")

        hdr.addWidget(self._icon)
        hdr.addWidget(self._name)
        hdr.addStretch(1)
        hdr.addWidget(self._state)

        layout.addWidget(self._preview)
        layout.addLayout(hdr)

    def apply_status(self, online: bool) -> None:
        self._state.setText("online" if online else "offline")
        self._state.setStyleSheet(
            "color: #1f8a1f;" if online else "color: #bb2b2b;"
        )

    def apply_frame(self, raw: bytes) -> None:
        pm = QPixmap()
        if not pm.loadFromData(raw, "JPG"):
            return
        fitted = pm.scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(fitted)


class NozzleCard(QFrame):
    action_requested = Signal(str, str, float)

    _nozzle_pm: QPixmap | None = None
    _cam_top_pm: QPixmap | None = None
    _cam_bottom_pm: QPixmap | None = None
    _cal_pm: QPixmap | None = None

    _Z_STEPS: list[float] = [10.0, 5.0, 2.5, 1.0, 0.1]
    _R_STEPS: list[float] = [45.0, 15.0, 5.0, 1.0, 0.1]

    @classmethod
    def _init_pms(cls) -> None:
        if cls._nozzle_pm is not None:
            return

        cls._nozzle_pm = _tint_pm(_load_pm("nozzle_change"), _COLOR_RED)
        cls._cam_top_pm = _make_camera_pm(body_color=_COLOR_BLUE, lens_color=_COLOR_RED)
        cls._cam_bottom_pm = _make_camera_pm(body_color=_COLOR_RED, lens_color=_COLOR_BLUE)
        cls._cal_pm = _tint_pm(_load_pm("calibration_spot"), _COLOR_BLUE)

    def __init__(self, nozzle_name: str) -> None:
        super().__init__()
        NozzleCard._init_pms()
        self.nozzle_name = nozzle_name
        self.setMinimumWidth(235)

        self.setFrameShape(QFrame.Shape.StyledPanel)

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 4, 5, 4)
        root.setSpacing(3)

        title_row = QHBoxLayout()
        self._title = QLabel(nozzle_name)
        self._title.setStyleSheet("font-weight: 600;")
        title_row.addWidget(self._title)
        title_row.addStretch(1)
        root.addLayout(title_row)

        self._offset = QLabel("Off X=-- Y=--")
        self._z = QLabel("Z=--")
        self._r = QLabel("R=--")
        root.addWidget(self._offset)

        zr_row = QHBoxLayout()
        zr_row.addWidget(self._z)
        zr_row.addWidget(self._r)
        zr_row.addStretch(1)
        root.addLayout(zr_row)

        # 3x3 compact jog keypad + two vertical step sliders
        jog_block = QHBoxLayout()
        jog_block.setSpacing(4)

        keypad = QGridLayout()
        keypad.setSpacing(2)

        b7_home = _sq_btn("home_axis", "7: Home Z axis")
        b8_up = _sq_btn("z_up", "8: Move up by Z step")
        b9_zero = _sq_btn("park_zero", "9: Move to Z=0.0")

        b4_vac_off = QPushButton("VAC\nOFF")
        b4_vac_off.setFixedSize(_BTN_SQ, _BTN_SQ)
        b4_vac_off.setStyleSheet(
            "QPushButton { background:#1a0000; border:2px solid #cc2200; border-radius:4px;"
            " color:#ff4444; font-size:9px; font-weight:bold; }"
            "QPushButton:hover { background:#2a0000; }"
            "QPushButton:pressed { background:#400000; }"
        )
        b4_vac_off.setToolTip("4: Vacuum OFF")
        b5_park = _sq_btn("park_zero", "5: Park nozzle (Z=0.0)")
        b6_vac_on = QPushButton("VAC\nON")
        b6_vac_on.setFixedSize(_BTN_SQ, _BTN_SQ)
        b6_vac_on.setStyleSheet(
            "QPushButton { background:#001a00; border:2px solid #22aa22; border-radius:4px;"
            " color:#44ee44; font-size:9px; font-weight:bold; }"
            "QPushButton:hover { background:#002a00; }"
            "QPushButton:pressed { background:#004000; }"
        )
        b6_vac_on.setToolTip("6: Vacuum ON")

        b1_rot_ccw = _sq_btn("rotate_ccw", "1: Rotate CCW by angle step")
        b2_down = _sq_btn("z_down", "2: Move down by Z step")
        b3_std_down = _sq_btn("calibration_spot", "3: Move to standard-down Z")

        keypad.addWidget(b7_home, 0, 0)
        keypad.addWidget(b8_up, 0, 1)
        keypad.addWidget(b9_zero, 0, 2)
        keypad.addWidget(b4_vac_off, 1, 0)
        keypad.addWidget(b5_park, 1, 1)
        keypad.addWidget(b6_vac_on, 1, 2)
        keypad.addWidget(b1_rot_ccw, 2, 0)
        keypad.addWidget(b2_down, 2, 1)
        keypad.addWidget(b3_std_down, 2, 2)

        step_cols = QVBoxLayout()
        step_cols.setSpacing(3)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self._z_step_combo = QComboBox()
        for mm in self._Z_STEPS:
            self._z_step_combo.addItem(f"{mm:g} mm", mm)
        self._z_step_combo.setCurrentIndex(3)  # default 1 mm
        self._z_step_combo.setFixedWidth(70)
        z_row.addWidget(self._z_step_combo)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("R:"))
        self._r_step_combo = QComboBox()
        for deg in self._R_STEPS:
            self._r_step_combo.addItem(f"{deg:g}\u00b0", deg)
        self._r_step_combo.setCurrentIndex(2)  # default 5°
        self._r_step_combo.setFixedWidth(70)
        r_row.addWidget(self._r_step_combo)

        step_cols.addLayout(z_row)
        step_cols.addLayout(r_row)
        step_cols.addStretch(1)

        jog_block.addLayout(keypad)
        jog_block.addLayout(step_cols)
        jog_block.addStretch(1)
        root.addLayout(jog_block)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(2)

        b_align = _dual_btn(self._nozzle_pm, self._cam_top_pm, "Align nozzle to top camera")
        b_cam = _dual_btn(self._cam_top_pm, self._nozzle_pm, "Move top camera to nozzle")
        b_bottom = _dual_btn(self._nozzle_pm, self._cam_bottom_pm, "Move nozzle above bottom camera")
        b_cal = _sq_btn(self._cal_pm, "Calculate nozzle offset at fiducial")

        btn_grid.addWidget(b_align, 0, 0)
        btn_grid.addWidget(b_cam, 0, 1)
        btn_grid.addWidget(b_cal, 1, 0)
        btn_grid.addWidget(b_bottom, 1, 1)

        root.addLayout(btn_grid)

        b_align.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "align_to_cam", 0.0))
        b_cam.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "cam_to_nozzle", 0.0))
        b_bottom.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "above_bottom", 0.0))
        b_cal.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "cal_offset", 0.0))
        b7_home.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "nozzle_home", 0.0))
        b8_up.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "z_up", self._z_step_mm()))
        b9_zero.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "z_zero", 0.0))
        b4_vac_off.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "vacuum_off", 0.0))
        b5_park.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "z_park", 0.0))
        b6_vac_on.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "vacuum_on", 0.0))
        b1_rot_ccw.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "rot_ccw", self._angle_step_deg()))
        b2_down.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "z_down", self._z_step_mm()))
        b3_std_down.clicked.connect(lambda: self.action_requested.emit(self.nozzle_name, "z_standard_down", 0.0))

    def apply_status(self, nozzle: dict[str, Any]) -> None:
        ox = nozzle.get("offset_x")
        oy = nozzle.get("offset_y")
        z = nozzle.get("z_position")
        r = nozzle.get("r_position")

        self._offset.setText(f"Off X={self._fmt(ox)} Y={self._fmt(oy)}")
        self._z.setText(f"Z={self._fmt(z, 1)}")
        self._r.setText(f"R={self._fmt(r)}")

    @staticmethod
    def _fmt(value: Any, decimals: int = 3) -> str:
        try:
            if value is None:
                return "--"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return "--"

    def _z_step_mm(self) -> float:
        v = self._z_step_combo.currentData()
        return float(v) if v is not None else 1.0

    def _angle_step_deg(self) -> float:
        v = self._r_step_combo.currentData()
        return float(v) if v is not None else 5.0


class TrayFeederEditor(QWidget):
    save_requested = Signal(str, dict)
    reload_requested = Signal(str)
    move_base_requested = Signal(float, float)
    move_current_requested = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._feeder_id = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        fixed_box = QGroupBox("Tray Feeder - Common")
        fixed_layout = QFormLayout(fixed_box)
        fixed_layout.setContentsMargins(6, 6, 6, 6)
        fixed_layout.setSpacing(4)

        self._id_label = QLabel("--")
        self._part_number = QLineEdit()
        self._pick_x = QDoubleSpinBox()
        self._pick_y = QDoubleSpinBox()
        self._pick_h = QDoubleSpinBox()
        for w in (self._pick_x, self._pick_y, self._pick_h):
            w.setRange(-9999.0, 9999.0)
            w.setDecimals(3)
            w.setSingleStep(0.1)

        pick_xy = QHBoxLayout()
        pick_xy.addWidget(QLabel("X"))
        pick_xy.addWidget(self._pick_x)
        pick_xy.addWidget(QLabel("Y"))
        pick_xy.addWidget(self._pick_y)

        btn_row = QHBoxLayout()
        self._btn_move_base = QPushButton("Top Cam -> Base Pick")
        self._btn_move_current = QPushButton("Top Cam -> Current Pick")
        self._btn_save = QPushButton("Save")
        self._btn_cancel = QPushButton("Cancel Editing")
        btn_row.addWidget(self._btn_move_base)
        btn_row.addWidget(self._btn_move_current)
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_save)

        fixed_layout.addRow("Feeder ID", self._id_label)
        fixed_layout.addRow("Part Number", self._part_number)
        fixed_layout.addRow("Base Pick", pick_xy)
        fixed_layout.addRow("Pick Height", self._pick_h)
        fixed_layout.addRow(btn_row)
        root.addWidget(fixed_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        detail_widget = QWidget()
        detail_layout = QFormLayout(detail_widget)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(4)

        self._step_x = QDoubleSpinBox()
        self._step_y = QDoubleSpinBox()
        self._current_x = QDoubleSpinBox()
        self._current_y = QDoubleSpinBox()
        self._last_x = QDoubleSpinBox()
        self._last_y = QDoubleSpinBox()
        for w in (self._step_x, self._step_y, self._current_x, self._current_y, self._last_x, self._last_y):
            w.setRange(-9999.0, 9999.0)
            w.setDecimals(3)
            w.setSingleStep(0.1)

        self._preferred_direction = QComboBox()
        self._preferred_direction.addItem("X+", "X+")
        self._preferred_direction.addItem("X-", "X-")
        self._preferred_direction.addItem("Y+", "Y+")
        self._preferred_direction.addItem("Y-", "Y-")

        detail_layout.addRow("X step to next pick", self._step_x)
        detail_layout.addRow("Y step to next pick", self._step_y)
        detail_layout.addRow("Preferred next direction", self._preferred_direction)
        detail_layout.addRow(QLabel("Actual Data"))
        detail_layout.addRow("Current pick X", self._current_x)
        detail_layout.addRow("Current pick Y", self._current_y)
        detail_layout.addRow("Last pick X", self._last_x)
        detail_layout.addRow("Last pick Y", self._last_y)

        scroll.setWidget(detail_widget)
        root.addWidget(scroll, 1)

        self._btn_save.clicked.connect(self._emit_save)
        self._btn_cancel.clicked.connect(self._emit_reload)
        self._btn_move_base.clicked.connect(self._emit_move_base)
        self._btn_move_current.clicked.connect(self._emit_move_current)
        self._set_enabled(False)

    def set_feeder(self, feeder: dict[str, Any]) -> None:
        self._feeder_id = str(feeder.get("feeder_id", "")).upper()
        self._id_label.setText(self._feeder_id or "--")

        pick = feeder.get("pick_location") if isinstance(feeder.get("pick_location"), dict) else {}
        type_data = feeder.get("type_data") if isinstance(feeder.get("type_data"), dict) else {}
        actual = feeder.get("actual_data") if isinstance(feeder.get("actual_data"), dict) else {}

        self._part_number.setText(str(feeder.get("manufacturer_part_number", "")))
        self._pick_x.setValue(float(pick.get("x", 0.0) or 0.0))
        self._pick_y.setValue(float(pick.get("y", 0.0) or 0.0))
        self._pick_h.setValue(float(feeder.get("pick_height", 0.0) or 0.0))

        self._step_x.setValue(float(type_data.get("x_step", 0.0) or 0.0))
        self._step_y.setValue(float(type_data.get("y_step", 0.0) or 0.0))
        pref = str(type_data.get("preferred_direction", "X+"))
        idx = self._preferred_direction.findData(pref)
        self._preferred_direction.setCurrentIndex(idx if idx >= 0 else 0)

        current = actual.get("current_pick") if isinstance(actual.get("current_pick"), dict) else {}
        last = actual.get("last_pick") if isinstance(actual.get("last_pick"), dict) else {}
        self._current_x.setValue(float(current.get("x", self._pick_x.value()) or self._pick_x.value()))
        self._current_y.setValue(float(current.get("y", self._pick_y.value()) or self._pick_y.value()))
        self._last_x.setValue(float(last.get("x", self._pick_x.value()) or self._pick_x.value()))
        self._last_y.setValue(float(last.get("y", self._pick_y.value()) or self._pick_y.value()))
        self._set_enabled(bool(self._feeder_id))

    def _set_enabled(self, enabled: bool) -> None:
        for w in (
            self._part_number,
            self._pick_x,
            self._pick_y,
            self._pick_h,
            self._step_x,
            self._step_y,
            self._preferred_direction,
            self._current_x,
            self._current_y,
            self._last_x,
            self._last_y,
            self._btn_move_base,
            self._btn_move_current,
            self._btn_save,
            self._btn_cancel,
        ):
            w.setEnabled(enabled)

    def _emit_reload(self) -> None:
        if self._feeder_id:
            self.reload_requested.emit(self._feeder_id)

    def _emit_move_base(self) -> None:
        self.move_base_requested.emit(self._pick_x.value(), self._pick_y.value())

    def _emit_move_current(self) -> None:
        self.move_current_requested.emit(self._current_x.value(), self._current_y.value())

    def _emit_save(self) -> None:
        if not self._feeder_id:
            return
        payload = {
            "manufacturer_part_number": self._part_number.text().strip(),
            "pick_location": {
                "x": self._pick_x.value(),
                "y": self._pick_y.value(),
            },
            "pick_height": self._pick_h.value(),
            "type_data": {
                "x_step": self._step_x.value(),
                "y_step": self._step_y.value(),
                "preferred_direction": str(self._preferred_direction.currentData()),
            },
            "actual_data": {
                "current_pick": {
                    "x": self._current_x.value(),
                    "y": self._current_y.value(),
                },
                "last_pick": {
                    "x": self._last_x.value(),
                    "y": self._last_y.value(),
                },
            },
        }
        self.save_requested.emit(self._feeder_id, payload)


class ControlWindow(QMainWindow):
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.setWindowTitle("openSMT Control")
        self.resize(1220, 760)

        base_url = f"http://{host}:{port}"
        self._api = ControlApiClient(base_url)
        self._img_net = QNetworkAccessManager(self)

        self._camera_tiles: dict[str, CameraTile] = {}
        self._camera_placeholder: QLabel | None = None
        self._camera_thumb_pending: set[str] = set()

        self._nozzle_cards: dict[str, NozzleCard] = {}
        self._nozzle_placeholder: QLabel | None = None
        self._feeders_by_id: dict[str, dict[str, Any]] = {}
        self._feeder_tab_index: dict[str, int] = {}
        self._selected_feeder_id: str = ""

        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(4)

        top = QHBoxLayout()
        self._host = QLineEdit(host)
        self._port = QLineEdit(str(port))
        self._connect_btn = QPushButton("Apply Host")
        self._conn_state = QLabel("Ready")

        self._host.setMaximumWidth(170)
        self._port.setMaximumWidth(70)

        top.addWidget(QLabel("Host"))
        top.addWidget(self._host)
        top.addWidget(QLabel("Port"))
        top.addWidget(self._port)
        top.addWidget(self._connect_btn)
        top.addStretch(1)
        top.addWidget(self._conn_state)
        outer.addLayout(top)

        pane_grid = QGridLayout()
        pane_grid.setContentsMargins(0, 0, 0, 0)
        pane_grid.setHorizontalSpacing(5)
        pane_grid.setVerticalSpacing(5)
        pane_grid.setColumnStretch(0, 1)
        pane_grid.setColumnStretch(1, 2)
        pane_grid.setRowStretch(0, 1)
        pane_grid.setRowStretch(1, 1)

        cam_group = QGroupBox("Cameras")
        cam_group_layout = QVBoxLayout(cam_group)
        cam_group_layout.setContentsMargins(4, 4, 4, 4)
        cam_scroll = QScrollArea()
        cam_scroll.setWidgetResizable(True)
        cam_container = QWidget()
        self._camera_layout = QGridLayout(cam_container)
        self._camera_layout.setContentsMargins(2, 2, 2, 2)
        self._camera_layout.setHorizontalSpacing(3)
        self._camera_layout.setVerticalSpacing(3)
        cam_scroll.setWidget(cam_container)
        cam_group_layout.addWidget(cam_scroll)

        gp_group = QGroupBox("General Purpose")
        gp_layout = QVBoxLayout(gp_group)
        gp_layout.setContentsMargins(6, 6, 6, 6)
        gp_tabs = QTabWidget()

        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        setup_layout.setContentsMargins(6, 6, 6, 6)
        setup_note = QLabel("Reserved for setup and machine configuration workflows.")
        setup_note.setWordWrap(True)
        setup_layout.addWidget(setup_note)
        setup_layout.addStretch(1)

        production_tab = QWidget()
        production_layout = QVBoxLayout(production_tab)
        production_layout.setContentsMargins(6, 6, 6, 6)
        production_note = QLabel("Reserved for production tools and run-time workflows.")
        production_note.setWordWrap(True)
        production_layout.addWidget(production_note)
        production_layout.addStretch(1)

        feeders_tab = QWidget()
        feeders_layout = QVBoxLayout(feeders_tab)
        feeders_layout.setContentsMargins(6, 6, 6, 6)
        self._feeders_tabs = QTabWidget()

        all_feeders_tab = QWidget()
        all_feeders_layout = QVBoxLayout(all_feeders_tab)
        all_feeders_layout.setContentsMargins(6, 6, 6, 6)
        self._feeder_table = QTableWidget(0, 6)
        self._feeder_table.setHorizontalHeaderLabels([
            "Feeder ID",
            "Type",
            "Part Number",
            "Pick X",
            "Pick Y",
            "Pick Height",
        ])
        self._feeder_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._feeder_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._feeder_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._feeder_table.verticalHeader().setVisible(False)
        self._feeder_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._feeder_table.cellDoubleClicked.connect(self._on_feeder_row_double_clicked)
        all_feeders_layout.addWidget(self._feeder_table)
        self._feeders_tabs.addTab(all_feeders_tab, "All Feeders")

        for feeder_type, title in _FEEDER_TYPE_TITLES:
            type_tab = QWidget()
            type_layout = QVBoxLayout(type_tab)
            type_layout.setContentsMargins(6, 6, 6, 6)

            if feeder_type == "tray_feeder":
                self._tray_editor = TrayFeederEditor()
                self._tray_editor.save_requested.connect(self._on_tray_save)
                self._tray_editor.reload_requested.connect(self._load_feeder_from_api)
                self._tray_editor.move_base_requested.connect(self._move_camera_to_xy)
                self._tray_editor.move_current_requested.connect(self._move_camera_to_xy)
                type_layout.addWidget(self._tray_editor)
            else:
                note = QLabel("Type-specific editor will be added in a later step.")
                note.setWordWrap(True)
                type_layout.addWidget(note)
                type_layout.addStretch(1)

            tab_idx = self._feeders_tabs.addTab(type_tab, title)
            self._feeder_tab_index[feeder_type] = tab_idx

        feeders_layout.addWidget(self._feeders_tabs)

        diagnostics_tab = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_tab)
        diagnostics_layout.setContentsMargins(6, 6, 6, 6)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._log.setMinimumHeight(90)
        diagnostics_layout.addWidget(self._log)

        gp_tabs.addTab(setup_tab, "Setup & Configuration")
        gp_tabs.addTab(production_tab, "Production")
        gp_tabs.addTab(feeders_tab, "Feeders")
        gp_tabs.addTab(diagnostics_tab, "Diagnostics Log")
        gp_layout.addWidget(gp_tabs)

        xy_group = QGroupBox("XY Jogging")
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

        jog_grid = QGridLayout()
        jog_grid.setSpacing(3)

        b_l = _xy_btn("move_left", "Jog left", tint=_COLOR_BLUE)
        b_r = _xy_btn("move_right", "Jog right", tint=_COLOR_BLUE)
        b_u = _xy_btn("move_up", "Jog up (Y+)", tint=_COLOR_BLUE)
        b_d = _xy_btn("move_down", "Jog down (Y-)", tint=_COLOR_BLUE)

        jog_grid.addWidget(b_u, 0, 1)
        jog_grid.addWidget(b_l, 1, 0)
        jog_grid.addWidget(b_d, 1, 1)
        jog_grid.addWidget(b_r, 1, 2)
        xy_layout.addLayout(jog_grid)

        special_grid = QGridLayout()
        special_grid.setSpacing(3)

        b_home_all = _xy_btn("home_all", "Home all axes", tint=_COLOR_RED)
        b_home_xy = _xy_btn("home_xy", "Home XY axes", tint=_COLOR_BLUE)
        b_fid_main = _xy_btn("fiducial_main", "Move to homing fiducial main", tint=_COLOR_RED)
        b_fid_sec = _xy_btn("fiducial_secondary", "Move to secondary fiducial", tint=_COLOR_BLUE)
        b_park = _xy_btn("park_zero", "Move to park", tint=_COLOR_RED)
        b_dispose = _xy_btn("dispose", "Move to dispose", tint=_COLOR_BLUE)
        b_nozchg = _xy_btn("nozzle_change", "Move to nozzle change", tint=_COLOR_RED)
        b_calspot = _xy_btn("calibration_spot", "Move to calibration spot", tint=_COLOR_BLUE)

        special_grid.addWidget(b_home_all, 0, 0)
        special_grid.addWidget(b_home_xy, 0, 1)
        special_grid.addWidget(b_fid_main, 0, 2)
        special_grid.addWidget(b_fid_sec, 0, 3)
        special_grid.addWidget(b_park, 1, 0)
        special_grid.addWidget(b_dispose, 1, 1)
        special_grid.addWidget(b_nozchg, 1, 2)
        special_grid.addWidget(b_calspot, 1, 3)
        xy_layout.addLayout(special_grid)

        noz_group = QGroupBox("Nozzles")
        noz_layout = QVBoxLayout(noz_group)
        noz_layout.setContentsMargins(4, 4, 4, 4)
        noz_scroll = QScrollArea()
        noz_scroll.setWidgetResizable(False)
        noz_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        noz_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._nozzle_container = QWidget()
        self._nozzle_layout = QGridLayout(self._nozzle_container)
        self._nozzle_layout.setContentsMargins(2, 2, 2, 2)
        self._nozzle_layout.setHorizontalSpacing(4)
        self._nozzle_layout.setVerticalSpacing(4)
        noz_scroll.setWidget(self._nozzle_container)
        noz_layout.addWidget(noz_scroll)

        pane_grid.addWidget(cam_group, 0, 0)
        pane_grid.addWidget(gp_group, 0, 1)
        pane_grid.addWidget(xy_group, 1, 0)
        pane_grid.addWidget(noz_group, 1, 1)
        outer.addLayout(pane_grid, 1)

        self._connect_btn.clicked.connect(self._apply_host)

        b_l.clicked.connect(lambda: self._jog_xy(-1.0, 0.0))
        b_r.clicked.connect(lambda: self._jog_xy(1.0, 0.0))
        b_u.clicked.connect(lambda: self._jog_xy(0.0, 1.0))
        b_d.clicked.connect(lambda: self._jog_xy(0.0, -1.0))

        b_home_all.clicked.connect(lambda: self._post_action("/api/coord/home", None, "Home all"))
        b_home_xy.clicked.connect(lambda: self._post_action("/api/coord/home-xy", None, "Home XY"))
        b_fid_main.clicked.connect(
            lambda: self._post_action("/api/coord/homing-fiducial-main", None, "Move to homing fiducial main")
        )
        b_fid_sec.clicked.connect(
            lambda: self._post_action("/api/coord/secondary-fiducial", None, "Move to secondary fiducial")
        )
        b_park.clicked.connect(lambda: self._post_action("/api/coord/park", None, "Move to park"))
        b_dispose.clicked.connect(lambda: self._post_action("/api/coord/dispose", None, "Move to dispose"))
        b_nozchg.clicked.connect(lambda: self._post_action("/api/coord/nozzle-change", None, "Move to nozzle change"))
        b_calspot.clicked.connect(
            lambda: self._post_action("/api/coord/calibration-spot", None, "Move to calibration spot")
        )

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
        self._coord_label.setText(f"X={self._fmt(positions.get('X'))}  Y={self._fmt(positions.get('Y'))}")

        cameras = data.get("cameras", []) if isinstance(data.get("cameras"), list) else []
        if not cameras and not self._camera_tiles:
            # Backward compatibility: older backends may not include camera list in /api/status.
            cameras = [
                {"name": "TOP", "online": False},
                {"name": "BOTTOM", "online": False},
            ]
        self._sync_camera_tiles(cameras)
        self._refresh_camera_thumbs()

        nozzles = data.get("nozzles", []) if isinstance(data.get("nozzles"), list) else []
        self._sync_nozzle_cards(nozzles)

        feeders = data.get("feeders", []) if isinstance(data.get("feeders"), list) else []
        self._sync_feeders(feeders)

    def _sync_camera_tiles(self, cameras: list[dict[str, Any]]) -> None:
        if self._camera_placeholder is not None:
            self._camera_placeholder.setParent(None)
            self._camera_placeholder.deleteLater()
            self._camera_placeholder = None

        known = set(self._camera_tiles.keys())
        incoming: set[str] = set()

        for camera in cameras:
            name = str(camera.get("name", "")).upper()
            if not name:
                continue
            incoming.add(name)

            tile = self._camera_tiles.get(name)
            if tile is None:
                tile = CameraTile(name)
                self._camera_tiles[name] = tile

            tile.apply_status(bool(camera.get("online", False)))

        for name in known - incoming:
            tile = self._camera_tiles.pop(name)
            tile.setParent(None)
            tile.deleteLater()

        for i in reversed(range(self._camera_layout.count())):
            item = self._camera_layout.itemAt(i)
            if item is not None and item.widget() is not None:
                item.widget().setParent(None)

        if not self._camera_tiles:
            self._camera_placeholder = QLabel("No cameras found in /api/status")
            self._camera_placeholder.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._camera_layout.addWidget(self._camera_placeholder, 0, 0)
            return

        for idx, name in enumerate(sorted(self._camera_tiles.keys())):
            row = idx // 2
            col = idx % 2
            self._camera_layout.addWidget(self._camera_tiles[name], row, col)

    def _refresh_camera_thumbs(self) -> None:
        for name, tile in self._camera_tiles.items():
            if name in self._camera_thumb_pending:
                continue

            self._camera_thumb_pending.add(name)
            request = QNetworkRequest(QUrl(f"{self._api._base_url}/thumb/{name}"))
            reply = self._img_net.get(request)

            def _finish(cam_name: str = name, cam_tile: CameraTile = tile, rep: QNetworkReply = reply) -> None:
                try:
                    raw = bytes(rep.readAll())
                    status_obj = rep.attribute(QNetworkRequest.HttpStatusCodeAttribute)
                    status = int(status_obj) if status_obj is not None else 0
                    ok = rep.error() == QNetworkReply.NetworkError.NoError and status == 200 and bool(raw)
                    if ok:
                        cam_tile.apply_frame(raw)
                finally:
                    self._camera_thumb_pending.discard(cam_name)
                    rep.deleteLater()

            reply.finished.connect(_finish)

    def _sync_nozzle_cards(self, nozzles: list[dict[str, Any]]) -> None:
        if self._nozzle_placeholder is not None:
            self._nozzle_placeholder.setParent(None)
            self._nozzle_placeholder.deleteLater()
            self._nozzle_placeholder = None

        known = set(self._nozzle_cards.keys())
        incoming: set[str] = set()

        for nozzle in nozzles:
            name = str(nozzle.get("name", "")).upper()
            if not name:
                continue
            incoming.add(name)

            card = self._nozzle_cards.get(name)
            if card is None:
                card = NozzleCard(name)
                card.action_requested.connect(self._on_nozzle_action)
                self._nozzle_cards[name] = card

            card.apply_status(nozzle)

        for name in known - incoming:
            card = self._nozzle_cards.pop(name)
            card.setParent(None)
            card.deleteLater()

        for i in reversed(range(self._nozzle_layout.count())):
            item = self._nozzle_layout.itemAt(i)
            if item is not None and item.widget() is not None:
                item.widget().setParent(None)

        if not self._nozzle_cards:
            self._nozzle_placeholder = QLabel("No nozzles found in /api/status")
            self._nozzle_placeholder.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._nozzle_layout.addWidget(self._nozzle_placeholder, 0, 0)
            return

        for idx, name in enumerate(sorted(self._nozzle_cards.keys())):
            self._nozzle_layout.addWidget(self._nozzle_cards[name], 0, idx)

        card_count = len(self._nozzle_cards)
        if card_count > 0:
            sample_card = next(iter(self._nozzle_cards.values()))
            card_w = sample_card.minimumWidth()
            spacing = self._nozzle_layout.horizontalSpacing()
            total_w = (card_w * card_count) + (spacing * max(0, card_count - 1)) + 16
            self._nozzle_container.setMinimumWidth(total_w)
            self._nozzle_container.setMinimumHeight(sample_card.sizeHint().height() + 8)

    def _sync_feeders(self, feeders: list[dict[str, Any]]) -> None:
        self._feeders_by_id = {}
        self._feeder_table.setRowCount(len(feeders))

        for row, feeder in enumerate(feeders):
            feeder_id = str(feeder.get("feeder_id", "")).upper()
            if feeder_id:
                self._feeders_by_id[feeder_id] = feeder

            pick_location = feeder.get("pick_location") if isinstance(feeder.get("pick_location"), dict) else {}
            cells = [
                feeder_id,
                self._human_feeder_type(str(feeder.get("feeder_type", ""))),
                str(feeder.get("manufacturer_part_number", "")),
                self._fmt(pick_location.get("x")),
                self._fmt(pick_location.get("y")),
                self._fmt(feeder.get("pick_height")),
            ]
            for col, value in enumerate(cells):
                self._feeder_table.setItem(row, col, QTableWidgetItem(value))

        if self._selected_feeder_id and self._selected_feeder_id in self._feeders_by_id:
            self._open_feeder_editor(self._selected_feeder_id)

    def _on_feeder_row_double_clicked(self, row: int, _col: int) -> None:
        item = self._feeder_table.item(row, 0)
        if item is None:
            return
        feeder_id = item.text().strip().upper()
        if feeder_id:
            self._open_feeder_editor(feeder_id)

    def _open_feeder_editor(self, feeder_id: str) -> None:
        feeder = self._feeders_by_id.get(feeder_id)
        if feeder is None:
            return

        feeder_type = str(feeder.get("feeder_type", "")).strip().lower()
        tab_idx = self._feeder_tab_index.get(feeder_type)
        if tab_idx is not None:
            self._feeders_tabs.setCurrentIndex(tab_idx)

        self._selected_feeder_id = feeder_id
        if feeder_type == "tray_feeder":
            self._tray_editor.set_feeder(feeder)

    def _load_feeder_from_api(self, feeder_id: str) -> None:
        self._api.get_json(
            f"/api/feeders/{feeder_id}",
            lambda ok, status, data, fid=feeder_id: self._on_feeder_loaded(fid, ok, status, data),
        )

    def _on_feeder_loaded(self, feeder_id: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"ERR {status}: feeder reload failed: {data.get('error', 'request_failed')}")
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is None:
            self._log_line("ERR: feeder reload failed: invalid feeder payload")
            return

        fid = str(feeder.get("feeder_id", feeder_id)).upper()
        self._feeders_by_id[fid] = feeder
        self._open_feeder_editor(fid)
        self._log_line(f"OK: feeder {fid} reloaded")
        self._poll_status()

    def _on_tray_save(self, feeder_id: str, payload: dict[str, Any]) -> None:
        self._api.put_json(
            f"/api/feeders/{feeder_id}",
            payload,
            lambda ok, status, data, fid=feeder_id: self._on_tray_saved(fid, ok, status, data),
        )

    def _on_tray_saved(self, feeder_id: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"ERR {status}: feeder {feeder_id} save failed: {data.get('error', 'request_failed')}")
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is not None:
            fid = str(feeder.get("feeder_id", feeder_id)).upper()
            self._feeders_by_id[fid] = feeder
            self._selected_feeder_id = fid

        if data.get("persisted", True):
            self._log_line(f"OK: feeder {feeder_id} saved")
        else:
            self._log_line(f"WARN: feeder {feeder_id} updated but not persisted: {data.get('persist_error')}")
        self._poll_status()

    def _move_camera_to_xy(self, x: float, y: float) -> None:
        self._post_action(
            "/api/coord/move-xy",
            {"x": float(x), "y": float(y)},
            f"Move top camera to X={x:.3f}, Y={y:.3f}",
        )

    def _on_nozzle_action(self, nozzle: str, action: str, value: float) -> None:
        if action == "align_to_cam":
            self._post_action(f"/api/nozzle/{nozzle}/move-to-camera", None, f"{nozzle}: Align to camera")
            return

        if action == "cam_to_nozzle":
            self._post_action(f"/api/nozzle/{nozzle}/move-camera-here", None, f"{nozzle}: Move camera to nozzle")
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
            return

        if action == "nozzle_home":
            self._post_action(f"/api/head/nozzle/{nozzle}/home", None, f"{nozzle}: Home Z")
            return

        if action == "z_up":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/move",
                {"delta": float(value)},
                f"{nozzle}: Z up +{value:.1f}",
            )
            return

        if action == "z_down":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/move",
                {"delta": -float(value)},
                f"{nozzle}: Z down -{value:.1f}",
            )
            return

        if action == "rot_ccw":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/rotate",
                {"delta": -float(value)},
                f"{nozzle}: Rotate CCW {value:.1f}",
            )
            return

        if action == "z_zero":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/move-absolute",
                {"z": 0.0},
                f"{nozzle}: Move to Z=0.0",
            )
            return

        if action == "z_park":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/park",
                None,
                f"{nozzle}: Park",
            )
            return

        if action == "z_standard_down":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/move-standard-down",
                None,
                f"{nozzle}: Move to standard down",
            )
            return

        if action == "vacuum_off":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/vacuum",
                {"on": False},
                f"{nozzle}: Vacuum OFF",
            )
            return

        if action == "vacuum_on":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/vacuum",
                {"on": True},
                f"{nozzle}: Vacuum ON",
            )
            return

        if action == "rot_cw":
            self._post_action(
                f"/api/head/nozzle/{nozzle}/rotate",
                {"delta": float(value)},
                f"{nozzle}: Rotate CW {value:.1f}",
            )
            return

    def _handle_calibration_result(self, nozzle: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"ERR {status}: {nozzle}: Cal offset failed: {data.get('error', 'request_failed')}")
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
    def _human_feeder_type(value: str) -> str:
        key = value.strip().lower()
        for feeder_type, title in _FEEDER_TYPE_TITLES:
            if feeder_type == key:
                return title[:-1] if title.endswith("s") else title
        return key.replace("_", " ").title() if key else "--"

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
