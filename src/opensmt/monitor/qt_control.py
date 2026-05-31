from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QCheckBox,
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
    QMenu,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from opensmt.store.catalog_sqlite import CatalogSQLite

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


def _make_target_marker_pm(label: str, size: int = _ICON_SZ, color: QColor | None = None) -> QPixmap:
    if color is None:
        color = _COLOR_RED
    key = f"__target@{label}@{size}@{color.name()}"
    if key in _pm_cache:
        return _pm_cache[key]

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    r = int(size * 0.34)
    cx = size // 2
    cy = size // 2
    p.setPen(color)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
    p.drawLine(cx - r - 2, cy, cx + r + 2, cy)
    p.drawLine(cx, cy - r - 2, cx, cy + r + 2)

    p.setPen(color.darker(140))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, label)
    p.end()

    _pm_cache[key] = pm
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


class CameraPreviewWidget(QWidget):
    vector_drawn = Signal(float, float)
    square_drawn = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._frame = QPixmap()
        self._zoom = 1.0
        self._source_rect = QRectF()
        self._target_rect = QRectF()
        self._drag_start = QPointF()
        self._drag_end = QPointF()
        self._drag_active = False
        self._square_mode = False
        self._square_reference_mm = 10.0
        self.setMinimumSize(360, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_square_mode(self, enabled: bool) -> None:
        self._square_mode = bool(enabled)
        self.update()

    def set_square_reference_mm(self, value_mm: float) -> None:
        self._square_reference_mm = 20.0 if float(value_mm) >= 20.0 else 10.0
        self.update()

    def set_zoom(self, value: float) -> None:
        self._zoom = max(1.0, min(4.0, float(value)))
        self.update()

    def set_frame(self, frame: QPixmap) -> None:
        self._frame = frame
        self.update()

    def paintEvent(self, _event: Any) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#070b14"))

        if self._frame.isNull():
            p.setPen(QColor("#8ba3cf"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Feed")
            return

        src = self._compute_source_rect()
        target = self._fit_rect(src.width(), src.height())
        self._source_rect = src
        self._target_rect = target

        p.drawPixmap(target, self._frame, src)

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self._target_rect.center().x()
        cy = self._target_rect.center().y()
        cross_len = 10
        p.setPen(QPen(QColor("#49b3ff"), 1))
        p.drawLine(int(cx - cross_len), int(cy), int(cx + cross_len), int(cy))
        p.drawLine(int(cx), int(cy - cross_len), int(cx), int(cy + cross_len))

        if self._drag_active:
            dx = self._drag_end.x() - self._drag_start.x()
            dy = self._drag_end.y() - self._drag_start.y()
            if self._square_mode:
                side = 2.0 * max(abs(dx), abs(dy))
                x0 = self._drag_start.x() - (side / 2.0)
                y0 = self._drag_start.y() - (side / 2.0)
                p.setPen(QPen(QColor("#ffc000"), 2))
                p.drawRect(int(x0), int(y0), int(side), int(side))
                p.setPen(QPen(QColor("#8ba3cf"), 1))
                sample_mm = int(self._square_reference_mm)
                p.drawText(int(x0) + 6, int(y0) - 6, f"{sample_mm}x{sample_mm}mm sample: {side:.1f}px")
            else:
                p.setPen(QPen(QColor("#ff3b3b"), 2))
                p.drawLine(self._drag_start, self._drag_end)
                p.setPen(QPen(QColor("#8ba3cf"), 1))
                p.drawText(
                    int(self._drag_end.x()) + 6,
                    int(self._drag_end.y()) - 6,
                    f"dx={dx:.1f}px dy={dy:.1f}px",
                )

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(event.position())
        if not self._target_rect.contains(pos):
            return
        if self._square_mode:
            center = self._target_rect.center()
            self._drag_start = QPointF(center.x(), center.y())
            self._drag_end = self._constrain_square_corner(self._drag_start, pos)
        else:
            self._drag_start = pos
            self._drag_end = pos
        self._drag_active = True
        self.update()

    def mouseMoveEvent(self, event: Any) -> None:
        if not self._drag_active:
            return
        p = QPointF(event.position())
        if self._square_mode:
            self._drag_end = self._constrain_square_corner(self._drag_start, p)
        else:
            self._drag_end = p
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._drag_active:
            return
        p = QPointF(event.position())
        if self._square_mode:
            self._drag_end = self._constrain_square_corner(self._drag_start, p)
        else:
            self._drag_end = p
        self._drag_active = False

        if self._square_mode:
            center_src = self._widget_to_source(self._drag_start)
            corner_src = self._widget_to_source(self._drag_end)
            if center_src is not None and corner_src is not None:
                side_px_x = 2.0 * abs(corner_src.x() - center_src.x())
                side_px_y = 2.0 * abs(corner_src.y() - center_src.y())
                self.square_drawn.emit(side_px_x, side_px_y)
        else:
            s0 = self._widget_to_source(self._drag_start)
            s1 = self._widget_to_source(self._drag_end)
            if s0 is not None and s1 is not None:
                self.vector_drawn.emit(s1.x() - s0.x(), s1.y() - s0.y())
        self.update()

    @staticmethod
    def _constrain_square_corner(start: QPointF, current: QPointF) -> QPointF:
        dx = current.x() - start.x()
        dy = current.y() - start.y()
        side = max(abs(dx), abs(dy))
        sx = 1.0 if dx >= 0.0 else -1.0
        sy = 1.0 if dy >= 0.0 else -1.0
        return QPointF(start.x() + sx * side, start.y() + sy * side)

    def _compute_source_rect(self) -> QRectF:
        fw = float(self._frame.width())
        fh = float(self._frame.height())
        if fw <= 0 or fh <= 0:
            return QRectF()

        zoom_w = fw / self._zoom
        zoom_h = fh / self._zoom
        x0 = (fw - zoom_w) / 2.0
        y0 = (fh - zoom_h) / 2.0
        return QRectF(x0, y0, zoom_w, zoom_h)

    def _fit_rect(self, src_w: float, src_h: float) -> QRectF:
        ww = float(self.width())
        wh = float(self.height())
        if src_w <= 0 or src_h <= 0 or ww <= 0 or wh <= 0:
            return QRectF(0, 0, ww, wh)

        scale = min(ww / src_w, wh / src_h)
        tw = src_w * scale
        th = src_h * scale
        tx = (ww - tw) / 2.0
        ty = (wh - th) / 2.0
        return QRectF(tx, ty, tw, th)

    def _widget_to_source(self, p: QPointF) -> QPointF | None:
        if self._target_rect.isNull() or self._source_rect.isNull() or not self._target_rect.contains(p):
            return None

        nx = (p.x() - self._target_rect.x()) / self._target_rect.width()
        ny = (p.y() - self._target_rect.y()) / self._target_rect.height()
        sx = self._source_rect.x() + nx * self._source_rect.width()
        sy = self._source_rect.y() + ny * self._source_rect.height()
        return QPointF(sx, sy)


class CameraTile(QFrame):
    vector_move_requested = Signal(str, float, float)
    camera_selected = Signal(str)
    calibrate_requested = Signal(str, float, float)
    light_set_requested = Signal(str, str, int)

    def __init__(self, camera_name: str) -> None:
        super().__init__()
        self.camera_name = camera_name
        self._square_reference_mm = 10.0
        self._resolution_dpcm_x = 0.0
        self._resolution_dpcm_y = 0.0
        self._pending_square_px_x = 0.0
        self._pending_square_px_y = 0.0
        self._flip_h = False
        self._flip_v = False
        self._visible_light_keys: list[str] = []
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._preview = CameraPreviewWidget()
        self._preview.setStyleSheet("background:#070b14; border:1px solid #2a3d66;")
        self._preview.vector_drawn.connect(self._on_vector_drawn)
        self._preview.square_drawn.connect(self._on_square_drawn)
        self._preview.setMinimumHeight(300)

        self._camera_menu_btn = QToolButton(self._preview)
        self._camera_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._camera_menu_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._camera_menu_btn.setAutoRaise(True)
        self._camera_menu_btn.setIcon(QIcon(_make_camera_pm(body_color=_COLOR_BLUE, lens_color=_COLOR_RED)))
        self._camera_menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._camera_menu_btn.setStyleSheet(
            "QToolButton {"
            "background: rgba(7, 11, 20, 185);"
            "color: #d9e8ff;"
            "border: 1px solid #2a3d66;"
            "border-radius: 4px;"
            "padding: 4px 8px;"
            "}"
            "QToolButton::menu-indicator { image: none; width: 0px; }"
        )
        self._camera_menu = QMenu(self._camera_menu_btn)
        self._camera_actions: dict[str, QAction] = {}
        self._camera_menu_btn.setMenu(self._camera_menu)
        self._camera_menu_btn.raise_()

        self._light_dot_buttons: list[QToolButton] = []
        for _ in range(3):
            dot = QToolButton(self._preview)
            dot.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            dot.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            dot.setAutoRaise(True)
            dot.setCursor(Qt.CursorShape.PointingHandCursor)
            dot.setFixedSize(14, 14)
            dot.setText("")
            dot.setStyleSheet(self._light_dot_style(0))
            dot_menu = QMenu(dot)
            for level in (0, 1, 2):
                action = dot_menu.addAction(str(level))
                action.triggered.connect(
                    lambda _checked=False, btn=dot, v=level: self._emit_light_set(btn, v)
                )
            dot.setMenu(dot_menu)
            dot.hide()
            dot.raise_()
            self._light_dot_buttons.append(dot)

        footer = QHBoxLayout()
        footer.setSpacing(4)
        self._name = QLabel(camera_name)
        self._state = QLabel("offline")
        self._name.setStyleSheet("font-weight: 600;")

        controls = QHBoxLayout()
        controls.setSpacing(4)
        controls.addWidget(QLabel("Zoom"))
        self._zoom = QComboBox()
        self._zoom.setMaximumWidth(92)
        for z in (1.0, 1.5, 2.0, 3.0, 4.0):
            self._zoom.addItem(f"{z:g}x", z)
        self._zoom.setCurrentIndex(0)
        self._zoom.currentIndexChanged.connect(self._on_zoom_changed)
        controls.addWidget(self._zoom)

        self._btn_draw_square = QToolButton()
        self._btn_draw_square.setText("Draw Center 10mm")
        self._btn_draw_square.setCheckable(True)
        self._btn_draw_square.toggled.connect(self._preview.set_square_mode)
        controls.addWidget(self._btn_draw_square)

        self._square_size = QComboBox()
        self._square_size.setMaximumWidth(84)
        self._square_size.addItem("10 mm", 10.0)
        self._square_size.addItem("20 mm", 20.0)
        self._square_size.currentIndexChanged.connect(self._on_square_size_changed)
        controls.addWidget(self._square_size)

        self._btn_apply_cal = QPushButton("Apply Cal")
        self._btn_apply_cal.setEnabled(False)
        self._btn_apply_cal.clicked.connect(self._emit_apply_calibration)
        controls.addWidget(self._btn_apply_cal)

        self._cal_info = QLabel("")
        self._cal_info.setStyleSheet("color:#8ba3cf;")
        controls.addWidget(self._cal_info)

        footer.addWidget(self._name)
        footer.addStretch(1)
        footer.addWidget(self._state)
        footer.addSpacing(8)
        footer.addLayout(controls)

        layout.addWidget(self._preview, 1)
        layout.addLayout(footer)

    def apply_status(self, online: bool) -> None:
        self._state.setText("online" if online else "offline")
        self._state.setStyleSheet(
            "color: #1f8a1f;" if online else "color: #bb2b2b;"
        )

    def sync_camera_choices(self, ordered: list[str], status_by_name: dict[str, bool], active_name: str) -> None:
        known = set(self._camera_actions.keys())
        desired = set(ordered)

        for name in known - desired:
            action = self._camera_actions.pop(name)
            self._camera_menu.removeAction(action)
            action.deleteLater()

        for name in ordered:
            action = self._camera_actions.get(name)
            if action is None:
                action = self._camera_menu.addAction("")
                action.triggered.connect(lambda _checked=False, cam=name: self.camera_selected.emit(cam))
                self._camera_actions[name] = action

            state = "online" if status_by_name.get(name, False) else "offline"
            action.setText(f"{name} ({state})")
            action.setCheckable(True)
            action.setChecked(name == active_name)

        self._camera_menu_btn.setText(active_name or self.camera_name)
        self._position_overlay_buttons()

    def sync_lights(self, lights: dict[str, int]) -> None:
        keys = sorted(str(k).strip().lower() for k in lights.keys() if str(k).strip())[:3]
        self._visible_light_keys = keys

        for idx, dot in enumerate(self._light_dot_buttons):
            if idx >= len(keys):
                dot.hide()
                dot.setProperty("light_key", "")
                continue

            key = keys[idx]
            value = int(lights.get(key, 0) or 0)
            value = max(0, min(2, value))
            dot.setProperty("light_key", key)
            dot.setToolTip(f"{key}: {value}")
            dot.setStyleSheet(self._light_dot_style(value))
            dot.show()

        self._position_overlay_buttons()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._position_overlay_buttons()

    def _position_overlay_buttons(self) -> None:
        self._camera_menu_btn.adjustSize()
        margin = 8
        x = max(margin, self._preview.width() - self._camera_menu_btn.width() - margin)
        self._camera_menu_btn.move(x, margin)

        dot_x = margin
        dot_y = margin
        spacing = 4
        for dot in self._light_dot_buttons:
            if dot.isVisible():
                dot.move(dot_x, dot_y)
                dot_x += dot.width() + spacing

    def apply_frame(self, raw: bytes) -> None:
        pm = QPixmap()
        if not pm.loadFromData(raw, "JPG"):
            return
        self._preview.set_frame(pm)

    def set_resolution_dpcm(self, x: float, y: float) -> None:
        self._resolution_dpcm_x = float(x)
        self._resolution_dpcm_y = float(y)

    def set_flip(self, flip_h: bool, flip_v: bool) -> None:
        self._flip_h = bool(flip_h)
        self._flip_v = bool(flip_v)

    def _on_zoom_changed(self, _idx: int) -> None:
        value = self._zoom.currentData()
        self._preview.set_zoom(float(value) if value is not None else 1.0)

    def _on_vector_drawn(self, dx_px: float, dy_px: float) -> None:
        mm_per_px_x = (10.0 / self._resolution_dpcm_x) if self._resolution_dpcm_x > 0.0 else 0.0
        mm_per_px_y = (10.0 / self._resolution_dpcm_y) if self._resolution_dpcm_y > 0.0 else 0.0
        if mm_per_px_x <= 0.0 or mm_per_px_y <= 0.0:
            return

        # flip_horizontal mirrors the image X axis: drag direction is inverted
        # flip_vertical mirrors the image Y axis: combined with the Qt-Y-down correction
        # (base sign is -1 for Y), a vertical flip doubles the inversion back to +1
        x_sign = -1.0 if self._flip_h else 1.0
        y_sign = 1.0 if self._flip_v else -1.0
        dx_mm = x_sign * dx_px * mm_per_px_x
        dy_mm = y_sign * dy_px * mm_per_px_y
        if abs(dx_mm) < 1e-9 and abs(dy_mm) < 1e-9:
            return
        self.vector_move_requested.emit(self.camera_name, dx_mm, dy_mm)

    def _on_square_drawn(self, side_px_x: float, side_px_y: float) -> None:
        self._pending_square_px_x = float(side_px_x)
        self._pending_square_px_y = float(side_px_y)
        self._btn_apply_cal.setEnabled(self._pending_square_px_x > 1.0 and self._pending_square_px_y > 1.0)
        ref_mm = int(self._square_reference_mm)
        self._cal_info.setText(
            f"{self._pending_square_px_x:.1f}px x {self._pending_square_px_y:.1f}px ({ref_mm}mm)"
        )
        self._btn_draw_square.setChecked(False)

    def _emit_apply_calibration(self) -> None:
        if self._pending_square_px_x <= 1.0 or self._pending_square_px_y <= 1.0:
            return
        sample_cm = self._square_reference_mm / 10.0
        dpcm_x = self._pending_square_px_x / sample_cm
        dpcm_y = self._pending_square_px_y / sample_cm
        self.calibrate_requested.emit(self.camera_name, dpcm_x, dpcm_y)

    def _on_square_size_changed(self, _idx: int) -> None:
        value = self._square_size.currentData()
        self._square_reference_mm = 20.0 if float(value or 10.0) >= 20.0 else 10.0
        self._preview.set_square_reference_mm(self._square_reference_mm)
        self._btn_draw_square.setText(f"Draw Center {int(self._square_reference_mm)}mm")

    def _emit_light_set(self, dot: QToolButton, value: int) -> None:
        key = str(dot.property("light_key") or "").strip().lower()
        if not key:
            return
        value = max(0, min(2, int(value)))
        self.light_set_requested.emit(self.camera_name, key, value)

    @staticmethod
    def _light_dot_style(value: int) -> str:
        palette = {
            0: "#4b5563",
            1: "#f59e0b",
            2: "#22c55e",
        }
        color = palette.get(int(value), "#4b5563")
        return (
            "QToolButton {"
            f"background:{color};"
            "border:1px solid #1f2937;"
            "border-radius:7px;"
            "padding:0px;"
            "}"
            "QToolButton:hover { border:1px solid #dbeafe; }"
        )


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

        self._z_step_val: float = self._Z_STEPS[3]   # default 1 mm
        self._r_step_val: float = self._R_STEPS[2]   # default 5°

        _btn_style = (
            "QToolButton {"
            "background:#12243a; color:#d9e8ff;"
            "border:1px solid #2a3d66; border-radius:3px;"
            "padding:2px 6px;"
            "}"
            "QToolButton::menu-indicator { image:none; width:0px; }"
        )

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self._z_step_btn = QToolButton()
        self._z_step_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._z_step_btn.setStyleSheet(_btn_style)
        self._z_step_btn.setFixedWidth(70)
        z_menu = QMenu(self._z_step_btn)
        for mm in self._Z_STEPS:
            action = z_menu.addAction(f"{mm:g} mm")
            action.setData(mm)
            action.triggered.connect(lambda _checked=False, v=mm: self._set_z_step(v))
        self._z_step_btn.setMenu(z_menu)
        self._z_step_btn.setText(f"{self._z_step_val:g} mm")
        z_row.addWidget(self._z_step_btn)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("R:"))
        self._r_step_btn = QToolButton()
        self._r_step_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._r_step_btn.setStyleSheet(_btn_style)
        self._r_step_btn.setFixedWidth(70)
        r_menu = QMenu(self._r_step_btn)
        for deg in self._R_STEPS:
            action = r_menu.addAction(f"{deg:g}\u00b0")
            action.setData(deg)
            action.triggered.connect(lambda _checked=False, v=deg: self._set_r_step(v))
        self._r_step_btn.setMenu(r_menu)
        self._r_step_btn.setText(f"{self._r_step_val:g}\u00b0")
        r_row.addWidget(self._r_step_btn)

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

    def _set_z_step(self, v: float) -> None:
        self._z_step_val = float(v)
        self._z_step_btn.setText(f"{v:g} mm")

    def _set_r_step(self, v: float) -> None:
        self._r_step_val = float(v)
        self._r_step_btn.setText(f"{v:g}\u00b0")

    def _z_step_mm(self) -> float:
        return self._z_step_val

    def _angle_step_deg(self) -> float:
        return self._r_step_val


class _PackageEditorDialog(QDialog):
    def __init__(self, package: dict[str, Any], tip_ids: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Package Editor")

        self._name = QLineEdit(str(package.get("name", "")))
        self._footprint = QLineEdit(str(package.get("footprint", "")))
        self._length = QDoubleSpinBox()
        self._width = QDoubleSpinBox()
        self._height = QDoubleSpinBox()
        self._pins = QSpinBox()
        self._compat_table = QTableWidget(0, 2)

        for widget in (self._length, self._width, self._height):
            widget.setRange(0.0, 1000.0)
            widget.setDecimals(3)
            widget.setSingleStep(0.1)
        self._pins.setRange(0, 9999)

        self._length.setValue(float(package.get("length_mm", 0.0) or 0.0))
        self._width.setValue(float(package.get("width_mm", 0.0) or 0.0))
        self._height.setValue(float(package.get("height_mm", 0.0) or 0.0))
        self._pins.setValue(int(package.get("pin_count", 0) or 0))

        form = QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Footprint", self._footprint)
        form.addRow("Length (mm)", self._length)
        form.addRow("Width (mm)", self._width)
        form.addRow("Height (mm)", self._height)
        form.addRow("Pin Count", self._pins)

        self._compat_table.setHorizontalHeaderLabels(["Nozzle Tip", "Compatible"])
        self._compat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._compat_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._compat_table.verticalHeader().setVisible(False)
        compat_header = self._compat_table.horizontalHeader()
        compat_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        compat_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        compatible_ids = {
            str(item).strip()
            for item in package.get("compatible_nozzle_tips", [])
            if str(item).strip()
        }
        ordered_tips = sorted({str(t).strip() for t in tip_ids if str(t).strip()})
        self._compat_table.setRowCount(len(ordered_tips))
        for row, tip_id in enumerate(ordered_tips):
            tip_item = QTableWidgetItem(tip_id)
            tip_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._compat_table.setItem(row, 0, tip_item)

            compat_item = QTableWidgetItem("")
            compat_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            compat_item.setCheckState(Qt.CheckState.Checked if tip_id in compatible_ids else Qt.CheckState.Unchecked)
            self._compat_table.setItem(row, 1, compat_item)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(QLabel("Nozzle Tip Compatibility"))
        root.addWidget(self._compat_table)
        root.addWidget(buttons)

    def package_data(self) -> dict[str, Any]:
        compatible_tip_ids: list[str] = []
        for row in range(self._compat_table.rowCount()):
            tip_item = self._compat_table.item(row, 0)
            compat_item = self._compat_table.item(row, 1)
            if tip_item is None or compat_item is None:
                continue
            if compat_item.checkState() == Qt.CheckState.Checked:
                tip_id = tip_item.text().strip()
                if tip_id:
                    compatible_tip_ids.append(tip_id)

        return {
            "name": self._name.text().strip().upper(),
            "footprint": self._footprint.text().strip(),
            "length_mm": float(self._length.value()),
            "width_mm": float(self._width.value()),
            "height_mm": float(self._height.value()),
            "pin_count": int(self._pins.value()),
            "compatible_nozzle_tips": compatible_tip_ids,
        }


class _PartEditorDialog(QDialog):
    def __init__(self, part: dict[str, Any], package_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Part Editor")

        self._part_id = QLineEdit(str(part.get("part_id", "")))
        self._description = QLineEdit(str(part.get("description", "")))
        self._package = QComboBox()
        self._package.addItems(package_names)
        self._quantity = QSpinBox()
        self._quantity.setRange(0, 999999)
        self._quantity.setValue(int(part.get("quantity", 0) or 0))

        current_pkg = str(part.get("package", "")).strip().upper()
        if current_pkg:
            idx = self._package.findText(current_pkg)
            if idx >= 0:
                self._package.setCurrentIndex(idx)

        form = QFormLayout()
        form.addRow("Part ID", self._part_id)
        form.addRow("Description", self._description)
        form.addRow("Package", self._package)
        form.addRow("Quantity", self._quantity)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def part_data(self) -> dict[str, Any]:
        return {
            "part_id": self._part_id.text().strip().upper(),
            "description": self._description.text().strip(),
            "package": self._package.currentText().strip().upper(),
            "quantity": int(self._quantity.value()),
        }


class _NozzleTipEditorDialog(QDialog):
    def __init__(self, tip: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nozzle Tip Editor")

        self._tip_id = QLineEdit(str(tip.get("id", "")))
        self._tip_id.setReadOnly(True)
        self._suction_hole = QDoubleSpinBox()
        self._component_min = QDoubleSpinBox()
        self._component_max = QDoubleSpinBox()
        for widget in (self._suction_hole, self._component_min, self._component_max):
            widget.setRange(0.0, 1000.0)
            widget.setDecimals(3)
            widget.setSingleStep(0.1)

        for widget, key in (
            (self._suction_hole, "suction_hole_diameter_mm"),
            (self._component_min, "component_min_mm"),
            (self._component_max, "component_max_mm"),
        ):
            value = tip.get(key)
            widget.setValue(float(value) if value is not None else 0.0)

        form = QFormLayout()
        form.addRow("Tip ID", self._tip_id)
        form.addRow("Suction Hole (mm)", self._suction_hole)
        form.addRow("Component Min (mm)", self._component_min)
        form.addRow("Component Max (mm)", self._component_max)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def tip_data(self) -> dict[str, Any]:
        return {
            "id": self._tip_id.text().strip(),
            "suction_hole_diameter_mm": self._suction_hole.value(),
            "component_min_mm": self._component_min.value(),
            "component_max_mm": self._component_max.value(),
        }


class _PanelImportDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Panel Data")

        self._import_type = QComboBox()
        self._import_type.addItems(["KiCad", "Eagle", "Fusion"])

        self._file_path = QLineEdit()
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse)

        file_row = QHBoxLayout()
        file_row.addWidget(self._file_path, 1)
        file_row.addWidget(self._browse_btn)

        form = QFormLayout()
        form.addRow("Format", self._import_type)
        form.addRow("File", file_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def _on_browse(self) -> None:
        file_path, _filter = QFileDialog.getOpenFileName(self, "Select Import File")
        if file_path:
            self._file_path.setText(file_path)

    def import_type(self) -> str:
        return self._import_type.currentText().strip()

    def file_path(self) -> str:
        return self._file_path.text().strip()


class _NozzleEditorDialog(QDialog):
    def __init__(self, nozzle: dict[str, Any], tip_ids: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nozzle Editor")

        self._name = QLineEdit(str(nozzle.get("name", "")))
        self._name.setReadOnly(True)
        self._z_axis = QLineEdit(str(nozzle.get("z_axis", "")))
        self._min_z = QDoubleSpinBox()
        self._max_z = QDoubleSpinBox()
        self._offset_x = QDoubleSpinBox()
        self._offset_y = QDoubleSpinBox()
        self._standard_down_z = QDoubleSpinBox()
        self._vacuum_board = QLineEdit(str((nozzle.get("vacuum_valve") or {}).get("board", "")))
        self._vacuum_io = QLineEdit(str((nozzle.get("vacuum_valve") or {}).get("io_type", "")))
        self._vacuum_pin = QSpinBox()
        self._air_enabled = QCheckBox("Has Air Valve")
        self._air_board = QLineEdit(str((nozzle.get("air_valve") or {}).get("board", "")))
        self._air_io = QLineEdit(str((nozzle.get("air_valve") or {}).get("io_type", "")))
        self._air_pin = QSpinBox()
        self._tip_id = QComboBox()
        self._tip_id.addItems(tip_ids)

        for widget in (self._min_z, self._max_z, self._offset_x, self._offset_y, self._standard_down_z):
            widget.setRange(-100000.0, 100000.0)
            widget.setDecimals(3)
            widget.setSingleStep(0.1)
        self._vacuum_pin.setRange(0, 9999)
        self._air_pin.setRange(0, 9999)

        self._min_z.setValue(float(nozzle.get("min_z", 0.0) or 0.0))
        self._max_z.setValue(float(nozzle.get("max_z", 0.0) or 0.0))
        self._offset_x.setValue(float(nozzle.get("offset_x", 0.0) or 0.0))
        self._offset_y.setValue(float(nozzle.get("offset_y", 0.0) or 0.0))
        self._standard_down_z.setValue(float(nozzle.get("standard_down_z", 0.0) or 0.0))

        vacuum_valve = nozzle.get("vacuum_valve") if isinstance(nozzle.get("vacuum_valve"), dict) else {}
        self._vacuum_pin.setValue(int(vacuum_valve.get("pin", 0) or 0))

        air_valve = nozzle.get("air_valve") if isinstance(nozzle.get("air_valve"), dict) else None
        if air_valve is not None:
            self._air_enabled.setChecked(True)
            self._air_board.setText(str(air_valve.get("board", "")))
            self._air_io.setText(str(air_valve.get("io_type", "")))
            self._air_pin.setValue(int(air_valve.get("pin", 0) or 0))
        else:
            self._air_enabled.setChecked(False)

        current_tip = str(nozzle.get("tip_id", "")).strip()
        if current_tip:
            idx = self._tip_id.findText(current_tip)
            if idx >= 0:
                self._tip_id.setCurrentIndex(idx)
        elif self._tip_id.count() > 0:
            self._tip_id.setCurrentIndex(0)

        form = QFormLayout()
        form.addRow("Nozzle Name", self._name)
        form.addRow("Z Axis", self._z_axis)
        form.addRow("Min Z", self._min_z)
        form.addRow("Max Z", self._max_z)
        form.addRow("Offset X", self._offset_x)
        form.addRow("Offset Y", self._offset_y)
        form.addRow("Tip ID", self._tip_id)
        form.addRow("Standard Down Z", self._standard_down_z)
        form.addRow("Vacuum Board", self._vacuum_board)
        form.addRow("Vacuum IO Type", self._vacuum_io)
        form.addRow("Vacuum Pin", self._vacuum_pin)
        form.addRow(self._air_enabled)
        form.addRow("Air Board", self._air_board)
        form.addRow("Air IO Type", self._air_io)
        form.addRow("Air Pin", self._air_pin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def nozzle_data(self) -> dict[str, Any]:
        tip_id = self._tip_id.currentText().strip()
        data: dict[str, Any] = {
            "name": self._name.text().strip().upper(),
            "z_axis": self._z_axis.text().strip().upper(),
            "min_z": float(self._min_z.value()),
            "max_z": float(self._max_z.value()),
            "offset_x": float(self._offset_x.value()),
            "offset_y": float(self._offset_y.value()),
            "tip_id": tip_id or None,
            "standard_down_z": float(self._standard_down_z.value()),
            "vacuum_valve": {
                "board": self._vacuum_board.text().strip().upper(),
                "io_type": self._vacuum_io.text().strip().lower(),
                "pin": int(self._vacuum_pin.value()),
            },
        }
        if self._air_enabled.isChecked():
            data["air_valve"] = {
                "board": self._air_board.text().strip().upper(),
                "io_type": self._air_io.text().strip().lower(),
                "pin": int(self._air_pin.value()),
            }
        else:
            data["air_valve"] = None
        return data


class TrayFeederEditor(QWidget):
    save_requested = Signal(str, dict)
    reload_requested = Signal(str)
    move_base_requested = Signal(float, float)
    move_current_requested = Signal(float, float)
    set_last_from_camera_requested = Signal(str)
    back_to_survey_requested = Signal()
    set_pick_from_camera_requested = Signal(str)
    reset_requested = Signal(str)
    advance_requested = Signal(str)
    pick_step_requested = Signal(str, str, int)
    bottom_camera_step_requested = Signal(str, str)
    vision_abort_requested = Signal()
    process_start_requested = Signal(str, str, int, bool)
    process_next_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._feeder_id = ""
        self._baseline_payload: dict[str, Any] = {}
        self._loading_values = False
        self._base_x_prev = 0.0
        self._base_y_prev = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        fixed_box = QGroupBox("Tray Feeder - Common")
        fixed_layout = QVBoxLayout(fixed_box)
        fixed_layout.setContentsMargins(6, 6, 6, 6)
        fixed_layout.setSpacing(4)

        self._id_label = QLabel("--")
        self._part_number = QComboBox()
        self._part_number.setEditable(True)
        self._part_number.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._part_number.setMinimumWidth(75)
        self._part_number.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._part_number.setCompleter(QCompleter([], self._part_number))
        self._part_number.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._part_number.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self._pick_x = QDoubleSpinBox()
        self._pick_y = QDoubleSpinBox()
        self._pick_h = QDoubleSpinBox()
        for w in (self._pick_x, self._pick_y, self._pick_h):
            w.setRange(-9999.0, 9999.0)
            w.setDecimals(3)
            w.setSingleStep(0.1)

        row_top = QHBoxLayout()
        row_top.setSpacing(6)
        row_top.addWidget(QLabel("ID"))
        row_top.addWidget(self._id_label)
        row_top.addSpacing(6)
        row_top.addWidget(QLabel("X"))
        row_top.addWidget(self._pick_x)
        row_top.addWidget(QLabel("Y"))
        row_top.addWidget(self._pick_y)
        row_top.addWidget(QLabel("Z"))
        row_top.addWidget(self._pick_h)
        row_top.addStretch(1)

        row_bottom = QHBoxLayout()
        row_bottom.setSpacing(6)
        row_bottom.addWidget(QLabel("Part"))
        row_bottom.addWidget(self._part_number, 1)

        btn_row_top = QHBoxLayout()
        btn_row_bottom = QHBoxLayout()
        self._btn_move_base = QPushButton()
        self._btn_move_current = QPushButton()
        self._btn_move_base.setIcon(QIcon(_compose_pm(_make_camera_pm(), _make_target_marker_pm("B", color=_COLOR_BLUE))))
        self._btn_move_current.setIcon(QIcon(_compose_pm(_make_camera_pm(), _make_target_marker_pm("C", color=_COLOR_RED))))
        self._btn_move_base.setIconSize(QSize(_ICON_SZ * 2 + _ARROW_W + 4, _ICON_SZ))
        self._btn_move_current.setIconSize(QSize(_ICON_SZ * 2 + _ARROW_W + 4, _ICON_SZ))
        self._btn_move_base.setFixedHeight(_BTN_SQ)
        self._btn_move_current.setFixedHeight(_BTN_SQ)
        self._btn_move_base.setToolTip("Move top camera above base pick location")
        self._btn_move_current.setToolTip("Move top camera above current pick location")
        self._btn_save = QPushButton("Save")
        self._btn_cancel = QPushButton("Cancel Editing")
        self._btn_back = QPushButton("Back to Survey")
        self._btn_pick_from_camera = QPushButton("Use Camera as Pick")
        self._btn_advance = QPushButton("Advance")
        self._btn_reset = QPushButton("Reset")
        self._status = QLabel("")
        self._status.setStyleSheet("color:#5f6b80;")

        pick_pm = _compose_pm(_make_camera_pm(), _make_target_marker_pm("P", color=_COLOR_BLUE))
        self._btn_pick_from_camera.setIcon(QIcon(pick_pm))
        self._btn_pick_from_camera.setIconSize(QSize(_ICON_SZ * 2 + _ARROW_W + 4, _ICON_SZ))

        adv_pm = _compose_pm(_make_target_marker_pm("C", color=_COLOR_RED), _make_target_marker_pm("N", color=_COLOR_BLUE))
        self._btn_advance.setIcon(QIcon(adv_pm))
        self._btn_advance.setIconSize(QSize(_ICON_SZ * 2 + _ARROW_W + 4, _ICON_SZ))
        self._btn_advance.setToolTip("Advance to next tray pick position")

        reset_pm = _compose_pm(_make_target_marker_pm("N", color=_COLOR_BLUE), _make_target_marker_pm("0", color=_COLOR_RED))
        self._btn_reset.setIcon(QIcon(reset_pm))
        self._btn_reset.setIconSize(QSize(_ICON_SZ * 2 + _ARROW_W + 4, _ICON_SZ))
        self._btn_reset.setToolTip("Reset picked count and indices")

        self._btn_pick_from_camera.setFixedHeight(_BTN_SQ)
        self._btn_advance.setFixedHeight(_BTN_SQ)
        self._btn_reset.setFixedHeight(_BTN_SQ)

        self._btn_pick_from_camera.setToolTip("Use current top-camera location as base pick X/Y")
        btn_row_top.addWidget(self._btn_move_base)
        btn_row_top.addWidget(self._btn_move_current)
        btn_row_top.addWidget(self._btn_pick_from_camera)
        btn_row_top.addStretch(1)

        btn_row_bottom.addWidget(self._btn_advance)
        btn_row_bottom.addWidget(self._btn_reset)
        btn_row_bottom.addWidget(self._btn_back)
        btn_row_bottom.addStretch(1)
        btn_row_bottom.addWidget(self._btn_cancel)
        btn_row_bottom.addWidget(self._btn_save)

        fixed_layout.addLayout(row_top)
        fixed_layout.addLayout(row_bottom)
        fixed_layout.addLayout(btn_row_top)
        fixed_layout.addLayout(btn_row_bottom)
        fixed_layout.addWidget(self._status)

        process_box = QGroupBox("Pick and Place Process")
        process_layout = QGridLayout(process_box)
        process_layout.setContentsMargins(6, 6, 6, 6)
        process_layout.setHorizontalSpacing(6)
        process_layout.setVerticalSpacing(4)

        self._pick_nozzle = QComboBox()
        self._pick_nozzle.setMinimumWidth(120)
        self._pick_dwell_ms = QSpinBox()
        self._pick_dwell_ms.setRange(0, 60000)
        self._pick_dwell_ms.setValue(150)
        self._single_step_mode = QCheckBox("Single-step mode")

        self._btn_pick_step = QPushButton("Step 1: Pick")
        self._btn_bottom_step = QPushButton("Step 2: Bottom Camera")
        self._btn_start_process = QPushButton("Start Sequence")
        self._btn_next_process = QPushButton("Run Next Step")
        self._btn_vision_adv = QPushButton("Vision...")
        self._btn_abort_vision = QPushButton("Abort Vision")
        self._vision_pipeline = QTextEdit()
        self._vision_preview_step = QSpinBox()
        self._vision_preview_step.setRange(-1, 999)
        self._vision_preview_step.setValue(-1)
        self._vision_preview_step.setToolTip("-1 = final output, 0..N = show intermediate step")
        self._vision_pipeline.setPlaceholderText(
            '[{"op":"GaussianBlur","args":[[5,5],0]},{"op":"cvtColor","args":["COLOR_BGR2HSV"]}]'
        )
        self._vision_pipeline.setFixedHeight(88)

        self._vision_advanced = QWidget()
        vision_adv_layout = QGridLayout(self._vision_advanced)
        vision_adv_layout.setContentsMargins(0, 0, 0, 0)
        vision_adv_layout.setHorizontalSpacing(6)
        vision_adv_layout.setVerticalSpacing(4)
        vision_adv_layout.addWidget(QLabel("Preview step"), 0, 0)
        vision_adv_layout.addWidget(self._vision_preview_step, 0, 1)
        vision_adv_layout.addWidget(self._btn_abort_vision, 0, 2)
        vision_adv_layout.addWidget(QLabel("Step 2 Vision Pipeline (JSON steps)"), 1, 0, 1, 3)
        vision_adv_layout.addWidget(self._vision_pipeline, 2, 0, 1, 3)
        self._vision_advanced.setVisible(False)

        process_layout.addWidget(QLabel("Nozzle"), 0, 0)
        process_layout.addWidget(self._pick_nozzle, 0, 1)
        process_layout.addWidget(QLabel("Vacuum dwell (ms)"), 0, 2)
        process_layout.addWidget(self._pick_dwell_ms, 0, 3)
        process_layout.addWidget(self._single_step_mode, 0, 4)

        process_layout.addWidget(self._btn_pick_step, 1, 0)
        process_layout.addWidget(self._btn_bottom_step, 1, 1)
        process_layout.addWidget(self._btn_start_process, 1, 2)
        process_layout.addWidget(self._btn_next_process, 1, 3)
        process_layout.addWidget(self._btn_vision_adv, 1, 4)
        process_layout.addWidget(self._vision_advanced, 2, 0, 1, 5)
        process_layout.setColumnStretch(5, 1)

        root.addWidget(process_box)
        root.addWidget(fixed_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        detail_widget = QWidget()
        detail_layout = QFormLayout(detail_widget)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(4)

        self._step_x = QDoubleSpinBox()
        self._step_y = QDoubleSpinBox()
        self._parts_avail_x = QSpinBox()
        self._parts_avail_y = QSpinBox()
        self._index_x = QSpinBox()
        self._index_y = QSpinBox()
        self._parts_picked = QSpinBox()
        self._current_x = QDoubleSpinBox()
        self._current_y = QDoubleSpinBox()
        self._last_x = QDoubleSpinBox()
        self._last_y = QDoubleSpinBox()
        for w in (self._step_x, self._step_y, self._current_x, self._current_y, self._last_x, self._last_y):
            w.setRange(-9999.0, 9999.0)
            w.setDecimals(3)
            w.setSingleStep(0.1)
        for w in (self._current_x, self._current_y, self._last_x, self._last_y):
            w.setReadOnly(True)
            w.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        for w in (self._parts_avail_x, self._parts_avail_y, self._index_x, self._index_y, self._parts_picked):
            w.setRange(0, 1000000)
        self._parts_picked.setReadOnly(True)
        self._parts_picked.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

        self._preferred_direction = QComboBox()
        self._preferred_direction.addItem("X", "X")
        self._preferred_direction.addItem("Y", "Y")

        step_row = QWidget()
        step_row_l = QHBoxLayout(step_row)
        step_row_l.setContentsMargins(0, 0, 0, 0)
        step_row_l.setSpacing(6)
        step_row_l.addWidget(QLabel("X"))
        step_row_l.addWidget(self._step_x)
        step_row_l.addWidget(QLabel("Y"))
        step_row_l.addWidget(self._step_y)
        detail_layout.addRow("Step to next pick", step_row)

        detail_layout.addRow("Preferred next direction", self._preferred_direction)

        avail_row = QWidget()
        avail_row_l = QHBoxLayout(avail_row)
        avail_row_l.setContentsMargins(0, 0, 0, 0)
        avail_row_l.setSpacing(6)
        avail_row_l.addWidget(QLabel("X"))
        avail_row_l.addWidget(self._parts_avail_x)
        avail_row_l.addWidget(QLabel("Y"))
        avail_row_l.addWidget(self._parts_avail_y)
        detail_layout.addRow("Parts available", avail_row)

        index_row = QWidget()
        index_row_l = QHBoxLayout(index_row)
        index_row_l.setContentsMargins(0, 0, 0, 0)
        index_row_l.setSpacing(6)
        index_row_l.addWidget(QLabel("X"))
        index_row_l.addWidget(self._index_x)
        index_row_l.addWidget(QLabel("Y"))
        index_row_l.addWidget(self._index_y)
        detail_layout.addRow("Current index", index_row)

        detail_layout.addRow("Parts picked", self._parts_picked)
        detail_layout.addRow(QLabel("Actual Data"))

        current_row = QWidget()
        current_row_l = QHBoxLayout(current_row)
        current_row_l.setContentsMargins(0, 0, 0, 0)
        current_row_l.setSpacing(6)
        current_row_l.addWidget(QLabel("X"))
        current_row_l.addWidget(self._current_x)
        current_row_l.addWidget(QLabel("Y"))
        current_row_l.addWidget(self._current_y)
        detail_layout.addRow("Current pick", current_row)

        last_row = QWidget()
        last_row_l = QHBoxLayout(last_row)
        last_row_l.setContentsMargins(0, 0, 0, 0)
        last_row_l.setSpacing(6)
        last_row_l.addWidget(QLabel("X"))
        last_row_l.addWidget(self._last_x)
        last_row_l.addWidget(QLabel("Y"))
        last_row_l.addWidget(self._last_y)
        self._btn_last_from_camera = QPushButton()
        self._btn_last_from_camera.setIcon(QIcon(_compose_pm(_make_camera_pm(), _make_target_marker_pm("L", color=_COLOR_RED))))
        self._btn_last_from_camera.setIconSize(QSize(_ICON_SZ * 2 + _ARROW_W + 4, _ICON_SZ))
        self._btn_last_from_camera.setFixedHeight(_BTN_SQ)
        self._btn_last_from_camera.setToolTip("Set last-pick position from current top-camera XY")
        last_row_l.addWidget(self._btn_last_from_camera)
        detail_layout.addRow("Last pick", last_row)

        scroll.setWidget(detail_widget)
        root.addWidget(scroll, 1)

        self._btn_save.clicked.connect(self._emit_save)
        self._btn_cancel.clicked.connect(self._emit_reload)
        self._btn_back.clicked.connect(self.back_to_survey_requested)
        self._btn_move_base.clicked.connect(self._emit_move_base)
        self._btn_move_current.clicked.connect(self._emit_move_current)
        self._btn_last_from_camera.clicked.connect(self._emit_set_last_from_camera)
        self._btn_pick_from_camera.clicked.connect(self._emit_set_pick_from_camera)
        self._btn_advance.clicked.connect(self._emit_advance)
        self._btn_reset.clicked.connect(self._emit_reset)
        self._btn_pick_step.clicked.connect(self._emit_pick_step)
        self._btn_bottom_step.clicked.connect(self._emit_bottom_step)
        self._btn_start_process.clicked.connect(self._emit_start_process)
        self._btn_next_process.clicked.connect(self._emit_next_process)
        self._btn_vision_adv.clicked.connect(self._toggle_vision_advanced)
        self._btn_abort_vision.clicked.connect(self.vision_abort_requested.emit)
        self._part_number.currentTextChanged.connect(self._on_fields_changed)
        if self._part_number.lineEdit() is not None:
            self._part_number.lineEdit().textChanged.connect(self._on_fields_changed)
        self._pick_x.valueChanged.connect(self._on_base_pick_changed)
        self._pick_y.valueChanged.connect(self._on_base_pick_changed)
        for w in (
            self._pick_x,
            self._pick_y,
            self._pick_h,
            self._step_x,
            self._step_y,
            self._parts_avail_x,
            self._parts_avail_y,
            self._index_x,
            self._index_y,
            self._current_x,
            self._current_y,
            self._last_x,
            self._last_y,
        ):
            w.valueChanged.connect(self._on_fields_changed)
        self._preferred_direction.currentIndexChanged.connect(self._on_fields_changed)
        self._step_x.valueChanged.connect(self._sync_current_pick_display)
        self._step_y.valueChanged.connect(self._sync_current_pick_display)
        self._index_x.valueChanged.connect(self._sync_current_pick_display)
        self._index_y.valueChanged.connect(self._sync_current_pick_display)
        self._parts_avail_x.valueChanged.connect(self._sync_last_pick_display)
        self._parts_avail_y.valueChanged.connect(self._sync_last_pick_display)
        self._step_x.valueChanged.connect(self._sync_last_pick_display)
        self._step_y.valueChanged.connect(self._sync_last_pick_display)
        self._set_enabled(False)

    def set_feeder(self, feeder: dict[str, Any]) -> None:
        self._loading_values = True
        self._feeder_id = str(feeder.get("feeder_id", "")).upper()
        self._id_label.setText(self._feeder_id or "--")

        pick = feeder.get("pick_location") if isinstance(feeder.get("pick_location"), dict) else {}
        type_data = feeder.get("type_data") if isinstance(feeder.get("type_data"), dict) else {}
        actual = feeder.get("actual_data") if isinstance(feeder.get("actual_data"), dict) else {}

        self._part_number.setCurrentText(str(feeder.get("manufacturer_part_number", "")))
        self._pick_x.setValue(float(pick.get("x", 0.0) or 0.0))
        self._pick_y.setValue(float(pick.get("y", 0.0) or 0.0))
        self._base_x_prev = self._pick_x.value()
        self._base_y_prev = self._pick_y.value()
        self._pick_h.setValue(float(feeder.get("pick_height", 0.0) or 0.0))

        self._step_x.setValue(float(type_data.get("x_step", 0.0) or 0.0))
        self._step_y.setValue(float(type_data.get("y_step", 0.0) or 0.0))
        self._parts_avail_x.setValue(int(type_data.get("parts_available_x", 0) or 0))
        self._parts_avail_y.setValue(int(type_data.get("parts_available_y", 0) or 0))
        pref = str(type_data.get("preferred_direction", "X"))
        idx = self._preferred_direction.findData(pref)
        self._preferred_direction.setCurrentIndex(idx if idx >= 0 else 0)

        self._index_x.setValue(int(actual.get("current_index_x", 0) or 0))
        self._index_y.setValue(int(actual.get("current_index_y", 0) or 0))
        self._parts_picked.setValue(int(actual.get("parts_picked", 0) or 0))

        current = actual.get("current_pick") if isinstance(actual.get("current_pick"), dict) else {}
        last = actual.get("last_pick") if isinstance(actual.get("last_pick"), dict) else {}
        self._current_x.setValue(float(current.get("x", self._pick_x.value()) or self._pick_x.value()))
        self._current_y.setValue(float(current.get("y", self._pick_y.value()) or self._pick_y.value()))
        self._last_x.setValue(float(last.get("x", self._pick_x.value()) or self._pick_x.value()))
        self._last_y.setValue(float(last.get("y", self._pick_y.value()) or self._pick_y.value()))
        self._baseline_payload = self._build_payload()
        self._loading_values = False
        self._status.setText("")
        self._set_enabled(bool(self._feeder_id))
        self._sync_last_pick_display()
        self._refresh_dirty_state()

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
            self._pick_nozzle,
            self._pick_dwell_ms,
            self._single_step_mode,
            self._btn_pick_step,
            self._btn_bottom_step,
            self._btn_start_process,
            self._btn_next_process,
            self._btn_vision_adv,
            self._btn_abort_vision,
            self._btn_move_base,
            self._btn_move_current,
            self._btn_pick_from_camera,
            self._btn_last_from_camera,
            self._btn_advance,
            self._btn_reset,
            self._btn_back,
            self._btn_save,
            self._btn_cancel,
        ):
            w.setEnabled(enabled)
        self._vision_pipeline.setEnabled(enabled)
        self._vision_preview_step.setEnabled(enabled)
        self._parts_picked.setEnabled(False)

    def _toggle_vision_advanced(self) -> None:
        show = not self._vision_advanced.isVisible()
        self._vision_advanced.setVisible(show)
        self._btn_vision_adv.setText("Vision Hide" if show else "Vision...")

    def _on_fields_changed(self, *_args: Any) -> None:
        if self._loading_values:
            return
        self._refresh_dirty_state()

    def _on_base_pick_changed(self, *_args: Any) -> None:
        if self._loading_values:
            return
        x = self._pick_x.value()
        y = self._pick_y.value()
        dx = x - self._base_x_prev
        dy = y - self._base_y_prev
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return

        self._loading_values = True
        current_x, current_y = self._computed_current_pick()
        self._current_x.setValue(current_x)
        self._current_y.setValue(current_y)
        last_x, last_y = self._computed_last_pick()
        self._last_x.setValue(last_x)
        self._last_y.setValue(last_y)
        self._loading_values = False

        self._base_x_prev = x
        self._base_y_prev = y
        self._refresh_dirty_state()

    def _build_payload(self) -> dict[str, Any]:
        current_x, current_y = self._computed_current_pick()
        return {
            "manufacturer_part_number": self._part_number.currentText().strip(),
            "pick_location": {
                "x": self._pick_x.value(),
                "y": self._pick_y.value(),
            },
            "pick_height": self._pick_h.value(),
            "type_data": {
                "x_step": self._step_x.value(),
                "y_step": self._step_y.value(),
                "preferred_direction": str(self._preferred_direction.currentData()),
                "parts_available_x": int(self._parts_avail_x.value()),
                "parts_available_y": int(self._parts_avail_y.value()),
            },
            "actual_data": {
                "current_index_x": int(self._index_x.value()),
                "current_index_y": int(self._index_y.value()),
                "parts_picked": int(self._parts_picked.value()),
                "current_pick": {
                    "x": current_x,
                    "y": current_y,
                },
                "last_pick": {
                    "x": self._last_x.value(),
                    "y": self._last_y.value(),
                },
            },
        }

    def _refresh_dirty_state(self) -> None:
        dirty = bool(self._feeder_id) and self._build_payload() != self._baseline_payload
        if dirty:
            self._btn_save.setStyleSheet("QPushButton { background:#1f7a1f; color:#ffffff; font-weight:600; }")
            self._btn_cancel.setStyleSheet("QPushButton { background:#b82828; color:#ffffff; font-weight:600; }")
        else:
            self._btn_save.setStyleSheet("")
            self._btn_cancel.setStyleSheet("")
        self._btn_save.setEnabled(bool(self._feeder_id) and dirty)
        self._btn_cancel.setEnabled(bool(self._feeder_id) and dirty)
        self._btn_pick_from_camera.setEnabled(bool(self._feeder_id))
        self._btn_advance.setEnabled(bool(self._feeder_id))
        self._btn_reset.setEnabled(bool(self._feeder_id))
        has_nozzle = bool(self.selected_nozzle_name())
        self._btn_pick_step.setEnabled(bool(self._feeder_id) and has_nozzle)
        self._btn_bottom_step.setEnabled(bool(self._feeder_id) and has_nozzle)
        self._btn_start_process.setEnabled(bool(self._feeder_id) and has_nozzle)
        self._btn_next_process.setEnabled(bool(self._feeder_id))

    def _emit_reload(self) -> None:
        if self._feeder_id:
            self.reload_requested.emit(self._feeder_id)

    def _emit_move_base(self) -> None:
        self.move_base_requested.emit(self._pick_x.value(), self._pick_y.value())

    def _emit_move_current(self) -> None:
        current_x, current_y = self._computed_current_pick()
        self._loading_values = True
        self._current_x.setValue(current_x)
        self._current_y.setValue(current_y)
        self._loading_values = False
        self.move_current_requested.emit(current_x, current_y)

    def _emit_reset(self) -> None:
        if self._feeder_id:
            self.reset_requested.emit(self._feeder_id)

    def _emit_advance(self) -> None:
        if self._feeder_id:
            self.advance_requested.emit(self._feeder_id)

    def _emit_pick_step(self) -> None:
        if self._feeder_id and self.selected_nozzle_name():
            self.pick_step_requested.emit(self._feeder_id, self.selected_nozzle_name(), self.vacuum_dwell_ms())

    def _emit_bottom_step(self) -> None:
        if self._feeder_id and self.selected_nozzle_name():
            self.bottom_camera_step_requested.emit(self._feeder_id, self.selected_nozzle_name())

    def _emit_start_process(self) -> None:
        if self._feeder_id and self.selected_nozzle_name():
            self.process_start_requested.emit(
                self._feeder_id,
                self.selected_nozzle_name(),
                self.vacuum_dwell_ms(),
                self._single_step_mode.isChecked(),
            )

    def _emit_next_process(self) -> None:
        if self._feeder_id:
            self.process_next_requested.emit()

    def _emit_set_pick_from_camera(self) -> None:
        if self._feeder_id:
            self.set_pick_from_camera_requested.emit(self._feeder_id)

    def _emit_set_last_from_camera(self) -> None:
        if self._feeder_id:
            self.set_last_from_camera_requested.emit(self._feeder_id)

    def is_dirty(self) -> bool:
        return bool(self._feeder_id) and self._build_payload() != self._baseline_payload

    def set_pick_location(self, x: float, y: float) -> None:
        self._pick_x.setValue(float(x))
        self._pick_y.setValue(float(y))

    def _computed_current_pick(self) -> tuple[float, float]:
        base_x = self._pick_x.value()
        base_y = self._pick_y.value()
        step_x = self._step_x.value()
        step_y = self._step_y.value()
        idx_x = int(self._index_x.value())
        idx_y = int(self._index_y.value())
        return base_x + (idx_x * step_x), base_y + (idx_y * step_y)

    def _computed_last_pick(self) -> tuple[float, float]:
        base_x = self._pick_x.value()
        base_y = self._pick_y.value()
        step_x = self._step_x.value()
        step_y = self._step_y.value()
        count_x = int(self._parts_avail_x.value())
        count_y = int(self._parts_avail_y.value())
        return base_x + (count_x * step_x), base_y + (count_y * step_y)

    def _sync_current_pick_display(self, *_args: Any) -> None:
        if self._loading_values:
            return
        current_x, current_y = self._computed_current_pick()
        self._loading_values = True
        self._current_x.setValue(current_x)
        self._current_y.setValue(current_y)
        self._loading_values = False

    def _sync_last_pick_display(self, *_args: Any) -> None:
        if self._loading_values:
            return
        last_x, last_y = self._computed_last_pick()
        self._loading_values = True
        self._last_x.setValue(last_x)
        self._last_y.setValue(last_y)
        self._loading_values = False

    def set_last_pick_location(self, x: float, y: float) -> None:
        self._loading_values = True
        self._last_x.setValue(float(x))
        self._last_y.setValue(float(y))
        self._loading_values = False
        self._refresh_dirty_state()

    def set_part_suggestions(self, part_ids: list[str]) -> None:
        current = self._part_number.currentText()
        values = sorted({str(part_id).strip() for part_id in part_ids if str(part_id).strip()})
        self._loading_values = True
        self._part_number.clear()
        self._part_number.addItems(values)
        self._part_number.setCurrentText(current)
        completer = self._part_number.completer()
        if completer is not None:
            completer.setModel(self._part_number.model())
        self._loading_values = False

    def set_nozzle_choices(self, nozzle_states: list[dict[str, Any]]) -> None:
        current = self.selected_nozzle_name()
        options: list[tuple[str, str]] = []
        for nozzle in nozzle_states:
            name = str(nozzle.get("name", "")).strip().upper()
            tip_id = str(nozzle.get("tip_id", "") or "").strip()
            if not name or not tip_id:
                continue
            options.append((name, f"{name} (tip {tip_id})"))

        options.sort(key=lambda item: item[0])

        self._pick_nozzle.blockSignals(True)
        self._pick_nozzle.clear()
        if options:
            for name, label in options:
                self._pick_nozzle.addItem(label, name)
            idx = self._pick_nozzle.findData(current)
            self._pick_nozzle.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._pick_nozzle.addItem("No mounted nozzle tip", "")
            self._pick_nozzle.setCurrentIndex(0)
        self._pick_nozzle.blockSignals(False)
        self._refresh_dirty_state()

    def selected_nozzle_name(self) -> str:
        data = self._pick_nozzle.currentData()
        return str(data).strip().upper() if data is not None else ""

    def vacuum_dwell_ms(self) -> int:
        return int(self._pick_dwell_ms.value())

    def vision_pipeline_steps(self) -> list[dict[str, Any]]:
        raw = self._vision_pipeline.toPlainText().strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except Exception:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def vision_preview_step(self) -> int | None:
        value = int(self._vision_preview_step.value())
        return None if value < 0 else value

    def set_default_vision_pipeline(self, steps: list[dict[str, Any]]) -> None:
        if self._vision_pipeline.toPlainText().strip():
            return
        try:
            self._vision_pipeline.setPlainText(json.dumps(steps, indent=2))
        except Exception:
            pass

    def show_status(self, text: str, ok: bool = True) -> None:
        color = "#1f8a1f" if ok else "#bb2b2b"
        self._status.setStyleSheet(f"color:{color};")
        self._status.setText(text)

    def feeder_id(self) -> str:
        return self._feeder_id

    def _emit_save(self) -> None:
        if not self._feeder_id:
            return
        payload = self._build_payload()
        self.save_requested.emit(self._feeder_id, payload)


class StepperPopup(QDialog):
    play_requested = Signal()
    single_step_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Action Stepper")
        self.setModal(False)
        self.setMinimumWidth(420)

        self._title = QLabel("No action loaded")
        self._steps = QListWidget()
        self._steps.setSelectionMode(QListWidget.SelectionMode.NoSelection)

        self._play_btn = QPushButton("Play")
        self._single_btn = QPushButton("Single Step")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(self._single_btn)
        btn_row.addStretch(1)

        root = QVBoxLayout(self)
        root.addWidget(self._title)
        root.addWidget(self._steps)
        root.addLayout(btn_row)

        self._play_btn.clicked.connect(self.play_requested.emit)
        self._single_btn.clicked.connect(self.single_step_requested.emit)

    def set_action(self, title: str, steps: list[str]) -> None:
        self._title.setText(title)
        self._steps.clear()
        for step in steps:
            self._steps.addItem(QListWidgetItem(str(step)))
        self.update_progress(0, None)

    def update_progress(self, passed_count: int, active_index: int | None) -> None:
        for idx in range(self._steps.count()):
            item = self._steps.item(idx)
            if item is None:
                continue
            item.setBackground(QColor("transparent"))
            item.setForeground(QColor("#d8dee9"))
            if idx < int(passed_count):
                item.setForeground(QColor("#d90429"))
            elif active_index is not None and idx == int(active_index):
                item.setBackground(QColor("#1f8a1f"))
                item.setForeground(QColor("#ffffff"))


class ControlWindow(QMainWindow):
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.setWindowTitle("openSMT Control")
        self.resize(1220, 760)

        base_url = f"http://{host}:{port}"
        self._api = ControlApiClient(base_url)
        self._img_net = QNetworkAccessManager(self)

        self._camera_tiles: dict[str, CameraTile] = {}
        self._camera_order: list[str] = []
        self._camera_status_by_name: dict[str, bool] = {}
        self._active_camera_name: str = ""
        self._shown_camera_name: str = ""
        self._camera_placeholder: QLabel | None = None
        self._camera_thumb_pending: set[str] = set()

        self._nozzle_cards: dict[str, NozzleCard] = {}
        self._nozzle_status_by_name: dict[str, dict[str, Any]] = {}
        self._nozzle_placeholder: QLabel | None = None
        self._feeders_by_id: dict[str, dict[str, Any]] = {}
        self._feeder_tab_index: dict[str, int] = {}
        self._selected_feeder_id: str = ""
        self._packages_by_name: dict[str, dict[str, Any]] = {}
        self._parts_by_id: dict[str, dict[str, Any]] = {}
        self._nozzle_tips_by_id: dict[str, dict[str, Any]] = {}
        self._nozzles_by_name: dict[str, dict[str, Any]] = {}
        self._pcbs_by_board_number: dict[str, dict[str, Any]] = {}
        self._panels_by_name: dict[str, dict[str, Any]] = {}
        self._jobs_by_name: dict[str, dict[str, Any]] = {}
        self._setup_cameras: list[dict[str, Any]] = []
        self._setup_camera_current_row = -1
        self._setup_positions: list[dict[str, Any]] = []
        self._setup_position_current_row = -1
        self._setup_config_dir = Path(__file__).resolve().parents[3] / "config" / "examples"
        self._catalog_db_path = self._setup_config_dir / "catalog.sqlite"
        self._catalog_db = CatalogSQLite(self._catalog_db_path)
        self._packages_config_dir = self._setup_config_dir / "packages"
        self._parts_config_path = self._setup_config_dir / "parts.json"
        self._nozzles_config_path = self._setup_config_dir / "nozzles.json"
        self._setup_camera_config_path = self._setup_config_dir / "camera" / "camera.cameras.json"
        self._setup_locations_config_path = self._setup_config_dir / "system.locations.json"
        self._current_x: float | None = None
        self._current_y: float | None = None
        self._process_queue: list[str] = []
        self._process_feeder_id: str = ""
        self._process_nozzle_name: str = ""
        self._process_dwell_ms: int = 150
        self._process_single_step: bool = False
        self._process_busy: bool = False
        self._vision_abort_requested: bool = False
        self._xy_motion_gate_token: int = 0
        self._xy_motion_in_progress: bool = False
        self._stepper_popup: StepperPopup | None = None
        self._stepper_steps: list[tuple[str, Callable[[Callable[[bool], None]], None]]] = []
        self._stepper_index: int = 0
        self._stepper_running: bool = False
        self._stepper_auto_play: bool = False
        self._stepper_done: Callable[[bool], None] | None = None
        self._stepper_run_token: int = 0
        self._machine_status = QLabel("X=--  Y=--")
        self._catalog_status = QLabel("Catalog DB: --")
        self._top_left_ratio_min = 0.2
        self._top_left_ratio_max = 0.65
        self._bottom_left_ratio_min = 0.2
        self._bottom_left_ratio_max = 0.5
        self._splitter_clamp_active: set[int] = set()

        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(4)

        top = QHBoxLayout()
        self._host = QLineEdit(host)
        self._port = QLineEdit(str(port))
        self._connect_btn = QPushButton("Apply Host")
        self._debug_btn = QPushButton("Debug")
        self._conn_state = QLabel("Ready")

        self._host.setMaximumWidth(170)
        self._port.setMaximumWidth(70)

        top.addWidget(QLabel("Host"))
        top.addWidget(self._host)
        top.addWidget(QLabel("Port"))
        top.addWidget(self._port)
        top.addWidget(self._connect_btn)
        top.addWidget(self._debug_btn)
        top.addStretch(1)
        top.addWidget(self._conn_state)
        outer.addLayout(top)

        split_root = QVBoxLayout()
        split_root.setContentsMargins(0, 0, 0, 0)
        split_root.setSpacing(5)

        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)

        self._bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._bottom_splitter.setChildrenCollapsible(False)

        cam_group = QGroupBox("Cameras")
        cam_group_layout = QVBoxLayout(cam_group)
        cam_group_layout.setContentsMargins(2, 2, 2, 2)
        self._camera_host = QWidget()
        self._camera_host_layout = QVBoxLayout(self._camera_host)
        self._camera_host_layout.setContentsMargins(0, 0, 0, 0)
        self._camera_host_layout.setSpacing(0)
        cam_group_layout.addWidget(self._camera_host, 1)

        gp_group = QGroupBox("General Purpose")
        gp_layout = QVBoxLayout(gp_group)
        gp_layout.setContentsMargins(6, 6, 6, 6)
        gp_tabs = QTabWidget()

        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        setup_layout.setContentsMargins(6, 6, 6, 6)
        self._setup_tabs = QTabWidget()

        setup_cameras_tab = QWidget()
        setup_cameras_layout = QVBoxLayout(setup_cameras_tab)
        setup_cameras_layout.setContentsMargins(4, 2, 4, 4)
        setup_cameras_layout.setSpacing(3)

        setup_cam_actions = QHBoxLayout()
        setup_cam_actions.setContentsMargins(0, 0, 0, 0)
        setup_cam_actions.setSpacing(4)
        self._setup_camera_add_btn = QPushButton("Add Camera")
        self._setup_camera_add_btn.clicked.connect(self._on_setup_camera_add)
        self._setup_camera_save_btn = QPushButton("Save Cameras")
        self._setup_camera_save_btn.clicked.connect(self._on_setup_camera_save)
        setup_cam_actions.addWidget(self._setup_camera_add_btn)
        setup_cam_actions.addWidget(self._setup_camera_save_btn)
        setup_cam_actions.addStretch(1)
        setup_cameras_layout.addLayout(setup_cam_actions)

        self._setup_camera_table = QTableWidget(0, 5)
        self._setup_camera_table.setHorizontalHeaderLabels([
            "Name",
            "Device",
            "FPS",
            "Resolution (dpcm)",
            "Rotation (deg)",
        ])
        self._setup_camera_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._setup_camera_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._setup_camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._setup_camera_table.verticalHeader().setVisible(False)
        sch = self._setup_camera_table.horizontalHeader()
        sch.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        sch.setStretchLastSection(True)
        sch.setMinimumSectionSize(70)
        sch.resizeSection(0, 120)
        sch.resizeSection(1, 170)
        sch.resizeSection(2, 70)
        sch.resizeSection(3, 160)
        sch.resizeSection(4, 110)
        self._setup_camera_table.cellClicked.connect(self._on_setup_camera_row_selected)
        setup_cameras_layout.addWidget(self._setup_camera_table)
        self._set_table_visible_rows(self._setup_camera_table, 4)

        setup_cam_grid = QGridLayout()
        setup_cam_grid.setHorizontalSpacing(8)
        setup_cam_grid.setVerticalSpacing(6)
        self._setup_cam_name = QLineEdit()
        self._setup_cam_device = QLineEdit()
        self._setup_cam_fps = QDoubleSpinBox()
        self._setup_cam_fps.setRange(0.1, 240.0)
        self._setup_cam_fps.setDecimals(2)
        self._setup_cam_res_x = QDoubleSpinBox()
        self._setup_cam_res_x.setRange(0.0, 5000.0)
        self._setup_cam_res_x.setDecimals(3)
        self._setup_cam_res_y = QDoubleSpinBox()
        self._setup_cam_res_y.setRange(0.0, 5000.0)
        self._setup_cam_res_y.setDecimals(3)
        self._setup_cam_flip_h = QCheckBox("Flip Horizontal")
        self._setup_cam_flip_v = QCheckBox("Flip Vertical")
        self._setup_cam_rotation = QDoubleSpinBox()
        self._setup_cam_rotation.setRange(-360.0, 360.0)
        self._setup_cam_rotation.setDecimals(3)

        setup_cam_grid.addWidget(QLabel("Name"), 0, 0)
        setup_cam_grid.addWidget(self._setup_cam_name, 0, 1)
        setup_cam_grid.addWidget(QLabel("Device"), 0, 2)
        setup_cam_grid.addWidget(self._setup_cam_device, 0, 3)

        setup_cam_grid.addWidget(QLabel("FPS"), 1, 0)
        setup_cam_grid.addWidget(self._setup_cam_fps, 1, 1)
        setup_cam_grid.addWidget(QLabel("Rotation (deg)"), 1, 2)
        setup_cam_grid.addWidget(self._setup_cam_rotation, 1, 3)

        setup_cam_grid.addWidget(QLabel("Resolution X (dpcm)"), 2, 0)
        setup_cam_grid.addWidget(self._setup_cam_res_x, 2, 1)
        setup_cam_grid.addWidget(QLabel("Resolution Y (dpcm)"), 2, 2)
        setup_cam_grid.addWidget(self._setup_cam_res_y, 2, 3)

        setup_cam_grid.addWidget(self._setup_cam_flip_h, 3, 1)
        setup_cam_grid.addWidget(self._setup_cam_flip_v, 3, 3)

        setup_cam_grid.setColumnStretch(1, 1)
        setup_cam_grid.setColumnStretch(3, 1)
        setup_cameras_layout.addLayout(setup_cam_grid)

        setup_positions_tab = QWidget()
        setup_positions_layout = QVBoxLayout(setup_positions_tab)
        setup_positions_layout.setContentsMargins(6, 6, 6, 6)

        setup_pos_actions = QHBoxLayout()
        self._setup_position_add_btn = QPushButton("Add Position")
        self._setup_position_add_btn.clicked.connect(self._on_setup_position_add)
        self._setup_position_save_btn = QPushButton("Save Positions")
        self._setup_position_save_btn.clicked.connect(self._on_setup_position_save)
        setup_pos_actions.addWidget(self._setup_position_add_btn)
        setup_pos_actions.addWidget(self._setup_position_save_btn)
        setup_pos_actions.addStretch(1)
        setup_positions_layout.addLayout(setup_pos_actions)

        self._setup_position_table = QTableWidget(0, 4)
        self._setup_position_table.setHorizontalHeaderLabels([
            "Name",
            "Type",
            "X",
            "Y",
        ])
        self._setup_position_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._setup_position_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._setup_position_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._setup_position_table.verticalHeader().setVisible(False)
        sph = self._setup_position_table.horizontalHeader()
        sph.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        sph.setStretchLastSection(True)
        sph.setMinimumSectionSize(70)
        sph.resizeSection(0, 180)
        sph.resizeSection(1, 130)
        sph.resizeSection(2, 110)
        sph.resizeSection(3, 110)
        self._setup_position_table.cellClicked.connect(self._on_setup_position_row_selected)
        setup_positions_layout.addWidget(self._setup_position_table)
        self._set_table_visible_rows(self._setup_position_table, 4)

        setup_pos_form = QFormLayout()
        self._setup_position_name = QLineEdit()
        self._setup_position_kind = QLineEdit()
        self._setup_position_kind.setReadOnly(True)
        self._setup_position_x = QDoubleSpinBox()
        self._setup_position_x.setRange(-100000.0, 100000.0)
        self._setup_position_x.setDecimals(3)
        self._setup_position_y = QDoubleSpinBox()
        self._setup_position_y.setRange(-100000.0, 100000.0)
        self._setup_position_y.setDecimals(3)
        setup_pos_form.addRow("Name", self._setup_position_name)
        setup_pos_form.addRow("Type", self._setup_position_kind)
        setup_pos_form.addRow("X", self._setup_position_x)
        setup_pos_form.addRow("Y", self._setup_position_y)
        setup_positions_layout.addLayout(setup_pos_form)

        setup_pos_detail_actions = QHBoxLayout()
        self._setup_position_move_btn = QPushButton("Move Camera There")
        self._setup_position_move_btn.clicked.connect(self._on_setup_position_move_camera_there)
        self._setup_position_capture_btn = QPushButton("Use Actual Camera Position")
        self._setup_position_capture_btn.clicked.connect(self._on_setup_position_capture_current)
        setup_pos_detail_actions.addWidget(self._setup_position_move_btn)
        setup_pos_detail_actions.addWidget(self._setup_position_capture_btn)
        setup_pos_detail_actions.addStretch(1)
        setup_positions_layout.addLayout(setup_pos_detail_actions)

        setup_other_tab = QWidget()
        setup_other_layout = QVBoxLayout(setup_other_tab)
        setup_other_layout.setContentsMargins(6, 6, 6, 6)
        setup_other_note = QLabel("Reserved for additional setup configuration groups.")
        setup_other_note.setWordWrap(True)
        setup_other_layout.addWidget(setup_other_note)
        setup_other_layout.addStretch(1)

        self._setup_tabs.addTab(setup_cameras_tab, "Cameras")
        self._setup_tabs.addTab(setup_positions_tab, "Special Positions")
        self._setup_tabs.addTab(setup_other_tab, "Other Configurations")
        setup_layout.addWidget(self._setup_tabs)

        production_tab = QWidget()
        production_layout = QVBoxLayout(production_tab)
        production_layout.setContentsMargins(6, 6, 6, 6)
        self._production_tabs = QTabWidget()

        single_pcb_tab = QWidget()
        single_pcb_layout = QVBoxLayout(single_pcb_tab)
        single_pcb_layout.setContentsMargins(6, 6, 6, 6)

        single_top_actions = QHBoxLayout()
        self._single_pcb_select = QComboBox()
        self._single_pcb_select.currentIndexChanged.connect(self._on_single_pcb_selected)
        self._single_pcb_capture_board_btn = QToolButton()
        self._single_pcb_capture_board_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._single_pcb_capture_board_btn.setAutoRaise(True)
        self._single_pcb_capture_board_btn.setIcon(QIcon(_make_camera_pm(body_color=_COLOR_BLUE, lens_color=_COLOR_RED)))
        self._single_pcb_capture_board_btn.setToolTip("Use current camera XY as the board XY position")
        self._single_pcb_capture_board_btn.clicked.connect(self._on_single_pcb_capture_board_xy)
        self._single_pcb_new_btn = QPushButton("New")
        self._single_pcb_save_btn = QPushButton("Save")
        self._single_pcb_delete_btn = QPushButton("Delete")
        self._single_pcb_new_btn.clicked.connect(self._on_single_pcb_new)
        self._single_pcb_save_btn.clicked.connect(self._on_single_pcb_save)
        self._single_pcb_delete_btn.clicked.connect(self._on_single_pcb_delete)
        single_top_actions.addWidget(QLabel("PCB"))
        single_top_actions.addWidget(self._single_pcb_select, 1)
        single_top_actions.addWidget(self._single_pcb_capture_board_btn)
        single_top_actions.addWidget(self._single_pcb_new_btn)
        single_top_actions.addWidget(self._single_pcb_save_btn)
        single_top_actions.addWidget(self._single_pcb_delete_btn)
        single_pcb_layout.addLayout(single_top_actions)

        single_common_group = QGroupBox("Common PCB Data")
        single_common_layout = QGridLayout(single_common_group)
        single_common_layout.setContentsMargins(8, 8, 8, 8)
        single_common_layout.setHorizontalSpacing(8)
        single_common_layout.setVerticalSpacing(6)
        self._single_pcb_board_number = QLineEdit()
        self._single_pcb_name = QLineEdit()
        self._single_pcb_version = QLineEdit()
        self._single_pcb_ll_x = QDoubleSpinBox()
        self._single_pcb_ll_x.setRange(-100000.0, 100000.0)
        self._single_pcb_ll_x.setDecimals(3)
        self._single_pcb_ll_y = QDoubleSpinBox()
        self._single_pcb_ll_y.setRange(-100000.0, 100000.0)
        self._single_pcb_ll_y.setDecimals(3)
        self._single_pcb_rel_z = QDoubleSpinBox()
        self._single_pcb_rel_z.setRange(-1000.0, 1000.0)
        self._single_pcb_rel_z.setDecimals(3)
        self._single_pcb_rotation = QDoubleSpinBox()
        self._single_pcb_rotation.setRange(0.0, 360.0)
        self._single_pcb_rotation.setDecimals(3)
        self._single_pcb_rotation.setSuffix(" deg")
        # Row 1: board number, name, version.
        single_common_layout.addWidget(QLabel("Board Number"), 0, 0)
        single_common_layout.addWidget(self._single_pcb_board_number, 0, 1)
        single_common_layout.addWidget(QLabel("Name"), 0, 2)
        single_common_layout.addWidget(self._single_pcb_name, 0, 3)
        single_common_layout.addWidget(QLabel("Version"), 0, 4)
        single_common_layout.addWidget(self._single_pcb_version, 0, 5)
        # Row 2: lower-left XY.
        single_common_layout.addWidget(QLabel("Lower-Left X (mm)"), 1, 0)
        single_common_layout.addWidget(self._single_pcb_ll_x, 1, 1)
        single_common_layout.addWidget(QLabel("Lower-Left Y (mm)"), 1, 2)
        single_common_layout.addWidget(self._single_pcb_ll_y, 1, 3)
        # Row 3: remaining two fields.
        single_common_layout.addWidget(QLabel("Relative Z Height (mm)"), 2, 0)
        single_common_layout.addWidget(self._single_pcb_rel_z, 2, 1)
        single_common_layout.addWidget(QLabel("Rotation (0-360 deg)"), 2, 2)
        single_common_layout.addWidget(self._single_pcb_rotation, 2, 3)
        single_common_layout.setColumnStretch(1, 1)
        single_common_layout.setColumnStretch(3, 1)
        single_common_layout.setColumnStretch(5, 1)
        single_pcb_layout.addWidget(single_common_group)

        single_items_group = QGroupBox("Placement Items")
        single_items_layout = QVBoxLayout(single_items_group)
        single_items_layout.setContentsMargins(8, 8, 8, 8)

        single_items_actions = QHBoxLayout()
        self._single_pcb_add_item_btn = QPushButton("Add Item")
        self._single_pcb_remove_item_btn = QPushButton("Remove Item")
        self._single_pcb_capture_item_btn = QToolButton()
        self._single_pcb_capture_item_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._single_pcb_capture_item_btn.setAutoRaise(True)
        self._single_pcb_capture_item_btn.setIcon(QIcon(_make_camera_pm(body_color=_COLOR_RED, lens_color=_COLOR_BLUE)))
        self._single_pcb_capture_item_btn.setToolTip("Use current camera XY as the selected part PCB-relative XY")
        self._single_pcb_capture_item_btn.clicked.connect(self._on_single_pcb_capture_selected_item_xy)
        self._single_pcb_move_camera_item_btn = QToolButton()
        self._single_pcb_move_camera_item_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._single_pcb_move_camera_item_btn.setAutoRaise(True)
        self._single_pcb_move_camera_item_btn.setIcon(QIcon(_make_camera_pm(body_color=_COLOR_BLUE, lens_color=_COLOR_RED)))
        self._single_pcb_move_camera_item_btn.setToolTip("Move camera to the selected part position on the PCB")
        self._single_pcb_move_camera_item_btn.clicked.connect(self._on_single_pcb_move_camera_to_selected_item)
        self._single_pcb_add_item_btn.clicked.connect(self._on_single_pcb_add_item)
        self._single_pcb_remove_item_btn.clicked.connect(self._on_single_pcb_remove_item)
        single_items_actions.addWidget(self._single_pcb_add_item_btn)
        single_items_actions.addWidget(self._single_pcb_remove_item_btn)
        single_items_actions.addWidget(self._single_pcb_capture_item_btn)
        single_items_actions.addWidget(self._single_pcb_move_camera_item_btn)
        single_items_actions.addStretch(1)
        single_items_layout.addLayout(single_items_actions)

        self._single_pcb_items_table = QTableWidget(0, 5)
        self._single_pcb_items_table.setHorizontalHeaderLabels([
            "Part Number",
            "Part Code",
            "X (mm)",
            "Y (mm)",
            "Rotation (deg)",
        ])
        self._single_pcb_items_table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        self._single_pcb_items_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._single_pcb_items_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._single_pcb_items_table.verticalHeader().setVisible(False)
        single_hdr = self._single_pcb_items_table.horizontalHeader()
        single_hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        single_hdr.setStretchLastSection(True)
        single_hdr.setMinimumSectionSize(70)
        single_hdr.resizeSection(0, 190)
        single_hdr.resizeSection(1, 120)
        single_hdr.resizeSection(2, 110)
        single_hdr.resizeSection(3, 110)
        single_hdr.resizeSection(4, 110)
        single_items_layout.addWidget(self._single_pcb_items_table)
        single_pcb_layout.addWidget(single_items_group)

        panel_tab = QWidget()
        panel_layout = QVBoxLayout(panel_tab)
        panel_layout.setContentsMargins(6, 6, 6, 6)

        panel_top_actions = QHBoxLayout()
        self._panel_select = QComboBox()
        self._panel_select.currentIndexChanged.connect(self._on_panel_selected)
        self._panel_new_btn = QPushButton("New")
        self._panel_save_btn = QPushButton("Save")
        self._panel_delete_btn = QPushButton("Delete")
        self._panel_import_btn = QPushButton("Import...")
        self._panel_new_btn.clicked.connect(self._on_panel_new)
        self._panel_save_btn.clicked.connect(self._on_panel_save)
        self._panel_delete_btn.clicked.connect(self._on_panel_delete)
        self._panel_import_btn.clicked.connect(self._on_panel_import)
        panel_top_actions.addWidget(QLabel("Panel"))
        panel_top_actions.addWidget(self._panel_select, 1)
        panel_top_actions.addWidget(self._panel_new_btn)
        panel_top_actions.addWidget(self._panel_save_btn)
        panel_top_actions.addWidget(self._panel_delete_btn)
        panel_top_actions.addWidget(self._panel_import_btn)
        panel_layout.addLayout(panel_top_actions)

        panel_common_group = QGroupBox("PCB Panel Definition")
        panel_common_layout = QGridLayout(panel_common_group)
        panel_common_layout.setContentsMargins(8, 8, 8, 8)
        panel_common_layout.setHorizontalSpacing(8)
        panel_common_layout.setVerticalSpacing(6)
        self._panel_name = QLineEdit()
        self._panel_source_board_number = QLineEdit()
        self._panel_count_x = QSpinBox()
        self._panel_count_x.setRange(1, 10)
        self._panel_count_x.setValue(1)
        self._panel_count_y = QSpinBox()
        self._panel_count_y.setRange(1, 10)
        self._panel_count_y.setValue(1)
        self._panel_pitch_x = QDoubleSpinBox()
        self._panel_pitch_x.setRange(0.0, 1000.0)
        self._panel_pitch_x.setDecimals(3)
        self._panel_pitch_y = QDoubleSpinBox()
        self._panel_pitch_y.setRange(0.0, 1000.0)
        self._panel_pitch_y.setDecimals(3)
        self._panel_rotation = QDoubleSpinBox()
        self._panel_rotation.setRange(0.0, 360.0)
        self._panel_rotation.setDecimals(3)
        self._panel_rotation.setSuffix(" deg")
        self._panel_import_type = QLineEdit()
        self._panel_import_type.setReadOnly(True)
        self._panel_import_file = QLineEdit()
        self._panel_import_file.setReadOnly(True)
        # Row 1: key identifiers.
        panel_common_layout.addWidget(QLabel("Panel Name"), 0, 0)
        panel_common_layout.addWidget(self._panel_name, 0, 1)
        panel_common_layout.addWidget(QLabel("Source Board Number"), 0, 2)
        panel_common_layout.addWidget(self._panel_source_board_number, 0, 3)
        # Row 2: matrix counts and rotation.
        panel_common_layout.addWidget(QLabel("PCBs in X (1-10)"), 1, 0)
        panel_common_layout.addWidget(self._panel_count_x, 1, 1)
        panel_common_layout.addWidget(QLabel("PCBs in Y (1-10)"), 1, 2)
        panel_common_layout.addWidget(self._panel_count_y, 1, 3)
        panel_common_layout.addWidget(QLabel("Board Rotation (0-360 deg)"), 1, 4)
        panel_common_layout.addWidget(self._panel_rotation, 1, 5)
        # Row 3: XY pitch and import type.
        panel_common_layout.addWidget(QLabel("Pitch X (mm)"), 2, 0)
        panel_common_layout.addWidget(self._panel_pitch_x, 2, 1)
        panel_common_layout.addWidget(QLabel("Pitch Y (mm)"), 2, 2)
        panel_common_layout.addWidget(self._panel_pitch_y, 2, 3)
        panel_common_layout.addWidget(QLabel("Import Type"), 2, 4)
        panel_common_layout.addWidget(self._panel_import_type, 2, 5)
        # Row 4: import file path takes remaining width.
        panel_common_layout.addWidget(QLabel("Import File"), 3, 0)
        panel_common_layout.addWidget(self._panel_import_file, 3, 1, 1, 5)
        panel_common_layout.setColumnStretch(1, 1)
        panel_common_layout.setColumnStretch(3, 1)
        panel_common_layout.setColumnStretch(5, 2)
        panel_layout.addWidget(panel_common_group)

        panel_note = QLabel(
            "A panel is a matrix of one or more PCBs. Set X/Y counts (max 10 each) "
            "and the board rotation for placement generation."
        )
        panel_note.setWordWrap(True)
        panel_layout.addWidget(panel_note)
        panel_layout.addStretch(1)

        job_tab = QWidget()
        job_layout = QVBoxLayout(job_tab)
        job_layout.setContentsMargins(6, 6, 6, 6)

        job_top_actions = QHBoxLayout()
        self._job_select = QComboBox()
        self._job_select.currentIndexChanged.connect(self._on_job_selected)
        self._job_new_btn = QPushButton("New")
        self._job_save_btn = QPushButton("Save")
        self._job_delete_btn = QPushButton("Delete")
        self._job_new_btn.clicked.connect(self._on_job_new)
        self._job_save_btn.clicked.connect(self._on_job_save)
        self._job_delete_btn.clicked.connect(self._on_job_delete)
        job_top_actions.addWidget(QLabel("Job"))
        job_top_actions.addWidget(self._job_select, 1)
        job_top_actions.addWidget(self._job_new_btn)
        job_top_actions.addWidget(self._job_save_btn)
        job_top_actions.addWidget(self._job_delete_btn)
        job_layout.addLayout(job_top_actions)

        job_form = QFormLayout()
        self._job_name = QLineEdit()
        self._job_description = QLineEdit()
        self._job_target_board_number = QLineEdit()
        self._job_target_panel_name = QLineEdit()
        job_form.addRow("Job Name", self._job_name)
        job_form.addRow("Description", self._job_description)
        job_form.addRow("Target Board Number", self._job_target_board_number)
        job_form.addRow("Target Panel Name", self._job_target_panel_name)
        job_layout.addLayout(job_form)

        job_note = QLabel("Job actions and execution logic will be extended later.")
        job_note.setWordWrap(True)
        job_layout.addWidget(job_note)
        job_layout.addStretch(1)

        self._production_tabs.addTab(single_pcb_tab, "Single PCB")
        self._production_tabs.addTab(panel_tab, "PCB Panel")
        self._production_tabs.addTab(job_tab, "Job")
        production_layout.addWidget(self._production_tabs)

        parts_packages_tab = QWidget()
        parts_packages_layout = QVBoxLayout(parts_packages_tab)
        parts_packages_layout.setContentsMargins(6, 6, 6, 6)
        self._parts_packages_tabs = QTabWidget()

        package_survey_tab = QWidget()
        package_survey_layout = QVBoxLayout(package_survey_tab)
        package_survey_layout.setContentsMargins(6, 6, 6, 6)

        package_actions = QHBoxLayout()
        self._add_package_btn = QPushButton("Add Package")
        self._add_package_btn.clicked.connect(self._on_add_package)
        package_actions.addWidget(self._add_package_btn)
        package_actions.addStretch(1)
        package_survey_layout.addLayout(package_actions)

        self._package_table = QTableWidget(0, 6)
        self._package_table.setHorizontalHeaderLabels([
            "Name",
            "Footprint",
            "Length (mm)",
            "Width (mm)",
            "Height (mm)",
            "Pins",
        ])
        self._package_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._package_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._package_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._package_table.verticalHeader().setVisible(False)
        pkg_header = self._package_table.horizontalHeader()
        pkg_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        pkg_header.setStretchLastSection(True)
        pkg_header.setMinimumSectionSize(70)
        pkg_header.resizeSection(0, 120)
        pkg_header.resizeSection(1, 190)
        pkg_header.resizeSection(2, 95)
        pkg_header.resizeSection(3, 95)
        pkg_header.resizeSection(4, 95)
        pkg_header.resizeSection(5, 70)
        self._package_table.cellDoubleClicked.connect(self._on_package_row_double_clicked)
        package_survey_layout.addWidget(self._package_table)

        parts_survey_tab = QWidget()
        parts_survey_layout = QVBoxLayout(parts_survey_tab)
        parts_survey_layout.setContentsMargins(6, 6, 6, 6)

        part_actions = QHBoxLayout()
        self._add_part_btn = QPushButton("Add Part")
        self._add_part_btn.clicked.connect(self._on_add_part)
        part_actions.addWidget(self._add_part_btn)
        part_actions.addStretch(1)
        parts_survey_layout.addLayout(part_actions)

        self._part_table = QTableWidget(0, 4)
        self._part_table.setHorizontalHeaderLabels([
            "Part ID",
            "Description",
            "Package",
            "Quantity",
        ])
        self._part_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._part_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._part_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._part_table.verticalHeader().setVisible(False)
        part_header = self._part_table.horizontalHeader()
        part_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        part_header.setStretchLastSection(True)
        part_header.setMinimumSectionSize(70)
        part_header.resizeSection(0, 120)
        part_header.resizeSection(1, 220)
        part_header.resizeSection(2, 110)
        part_header.resizeSection(3, 90)
        self._part_table.cellDoubleClicked.connect(self._on_part_row_double_clicked)
        parts_survey_layout.addWidget(self._part_table)

        nozzle_tip_survey_tab = QWidget()
        nozzle_tip_survey_layout = QVBoxLayout(nozzle_tip_survey_tab)
        nozzle_tip_survey_layout.setContentsMargins(6, 6, 6, 6)

        nozzle_tip_actions = QHBoxLayout()
        self._add_nozzle_tip_btn = QPushButton("Add Nozzle Tip")
        self._add_nozzle_tip_btn.clicked.connect(self._on_add_nozzle_tip)
        nozzle_tip_actions.addWidget(self._add_nozzle_tip_btn)
        nozzle_tip_actions.addStretch(1)
        nozzle_tip_survey_layout.addLayout(nozzle_tip_actions)

        self._nozzle_tip_table = QTableWidget(0, 4)
        self._nozzle_tip_table.setHorizontalHeaderLabels([
            "Tip ID",
            "Suction Hole (mm)",
            "Component Min (mm)",
            "Component Max (mm)",
        ])
        self._nozzle_tip_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._nozzle_tip_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._nozzle_tip_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._nozzle_tip_table.verticalHeader().setVisible(False)
        tip_header = self._nozzle_tip_table.horizontalHeader()
        tip_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        tip_header.setStretchLastSection(True)
        tip_header.setMinimumSectionSize(70)
        tip_header.resizeSection(0, 100)
        tip_header.resizeSection(1, 140)
        tip_header.resizeSection(2, 140)
        tip_header.resizeSection(3, 140)
        self._nozzle_tip_table.cellDoubleClicked.connect(self._on_nozzle_tip_row_double_clicked)
        nozzle_tip_survey_layout.addWidget(self._nozzle_tip_table)

        nozzle_survey_tab = QWidget()
        nozzle_survey_layout = QVBoxLayout(nozzle_survey_tab)
        nozzle_survey_layout.setContentsMargins(6, 6, 6, 6)

        nozzle_actions = QHBoxLayout()
        self._add_nozzle_btn = QPushButton("Add Nozzle")
        self._add_nozzle_btn.clicked.connect(self._on_add_nozzle)
        nozzle_actions.addWidget(self._add_nozzle_btn)
        nozzle_actions.addStretch(1)
        nozzle_survey_layout.addLayout(nozzle_actions)

        self._nozzle_table = QTableWidget(0, 7)
        self._nozzle_table.setHorizontalHeaderLabels([
            "Name",
            "Z Axis",
            "Min Z",
            "Max Z",
            "Offset X",
            "Offset Y",
            "Tip ID",
        ])
        self._nozzle_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._nozzle_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._nozzle_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._nozzle_table.verticalHeader().setVisible(False)
        noz_header = self._nozzle_table.horizontalHeader()
        noz_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        noz_header.setStretchLastSection(True)
        noz_header.setMinimumSectionSize(70)
        noz_header.resizeSection(0, 100)
        noz_header.resizeSection(1, 90)
        noz_header.resizeSection(2, 90)
        noz_header.resizeSection(3, 90)
        noz_header.resizeSection(4, 90)
        noz_header.resizeSection(5, 90)
        noz_header.resizeSection(6, 90)
        self._nozzle_table.cellDoubleClicked.connect(self._on_nozzle_row_double_clicked)
        nozzle_survey_layout.addWidget(self._nozzle_table)

        self._parts_packages_tabs.addTab(package_survey_tab, "Packages")
        self._parts_packages_tabs.addTab(parts_survey_tab, "Parts")
        self._parts_packages_tabs.addTab(nozzle_tip_survey_tab, "Nozzle Tips")
        self._parts_packages_tabs.addTab(nozzle_survey_tab, "Nozzles")
        parts_packages_layout.addWidget(self._parts_packages_tabs)

        feeders_tab = QWidget()
        feeders_layout = QVBoxLayout(feeders_tab)
        feeders_layout.setContentsMargins(6, 6, 6, 6)
        self._feeders_tabs = QTabWidget()

        all_feeders_tab = QWidget()
        all_feeders_layout = QVBoxLayout(all_feeders_tab)
        all_feeders_layout.setContentsMargins(6, 6, 6, 6)

        all_actions = QHBoxLayout()
        self._add_feeder_btn = QToolButton()
        self._add_feeder_btn.setText("Add Feeder")
        self._add_feeder_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._add_feeder_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._add_feeder_btn.setIcon(QIcon(_make_camera_pm(body_color=_COLOR_BLUE, lens_color=_COLOR_RED)))
        add_menu = QMenu(self._add_feeder_btn)
        for feeder_type, title in _FEEDER_TYPE_TITLES:
            label = title[:-1] if title.endswith("s") else title
            action = add_menu.addAction(f"Add {label}")
            action.triggered.connect(lambda _checked=False, t=feeder_type: self._create_feeder(t))
        self._add_feeder_btn.setMenu(add_menu)
        all_actions.addWidget(self._add_feeder_btn)
        all_actions.addStretch(1)
        all_feeders_layout.addLayout(all_actions)

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
        _fh = self._feeder_table.horizontalHeader()
        _fh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        _fh.setStretchLastSection(True)
        _fh.setMinimumSectionSize(70)
        _fh.resizeSection(0, 170)
        _fh.resizeSection(1, 120)
        _fh.resizeSection(2, 170)
        _fh.resizeSection(3, 90)
        _fh.resizeSection(4, 90)
        _fh.resizeSection(5, 100)
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
                self._tray_editor.back_to_survey_requested.connect(self._go_to_feeder_survey)
                self._tray_editor.set_pick_from_camera_requested.connect(self._set_tray_pick_from_camera)
                self._tray_editor.set_last_from_camera_requested.connect(self._set_tray_last_from_camera)
                self._tray_editor.advance_requested.connect(self._advance_feeder_pick)
                self._tray_editor.reset_requested.connect(self._reset_feeder)
                self._tray_editor.pick_step_requested.connect(self._on_pick_step_requested)
                self._tray_editor.bottom_camera_step_requested.connect(self._on_bottom_camera_step_requested)
                self._tray_editor.vision_abort_requested.connect(self._on_vision_abort_requested)
                self._tray_editor.process_start_requested.connect(self._on_process_start_requested)
                self._tray_editor.process_next_requested.connect(self._on_process_next_requested)
                self._tray_editor.set_default_vision_pipeline(self._default_bottom_vision_pipeline())
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
        gp_tabs.addTab(parts_packages_tab, "Parts & Packages")
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

        self._top_splitter.addWidget(cam_group)
        self._top_splitter.addWidget(gp_group)
        self._top_splitter.setStretchFactor(0, 1)
        self._top_splitter.setStretchFactor(1, 1)

        self._bottom_splitter.addWidget(xy_group)
        self._bottom_splitter.addWidget(noz_group)
        self._bottom_splitter.setStretchFactor(0, 1)
        self._bottom_splitter.setStretchFactor(1, 1)

        split_root.addWidget(self._top_splitter, 3)
        split_root.addWidget(self._bottom_splitter, 2)
        outer.addLayout(split_root, 1)

        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self._catalog_status, 0)
        status.addPermanentWidget(self._machine_status, 1)

        self._connect_btn.clicked.connect(self._apply_host)
        self._debug_btn.clicked.connect(self._open_stepper_popup)

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

        self._load_packages_from_config()
        self._load_parts_from_config()
        self._load_nozzle_editor_config()
        self._load_setup_cameras_from_config()
        self._load_setup_positions_from_config()
        self._load_production_from_db()
        self._refresh_catalog_status()

        QTimer.singleShot(0, self._init_splitters)

        self._poll_status()

    def _init_splitters(self) -> None:
        self._set_splitter_ratio(self._top_splitter, 0.45)
        self._set_splitter_ratio(self._bottom_splitter, 0.2)
        self._top_splitter.splitterMoved.connect(lambda _pos, _index: self._clamp_splitter(self._top_splitter))
        self._bottom_splitter.splitterMoved.connect(lambda _pos, _index: self._clamp_splitter(self._bottom_splitter))
        self._clamp_splitter(self._top_splitter)
        self._clamp_splitter(self._bottom_splitter)

    def _clamp_splitter(self, splitter: QSplitter) -> None:
        splitter_key = id(splitter)
        if splitter_key in self._splitter_clamp_active:
            return

        sizes = splitter.sizes()
        if len(sizes) != 2:
            return
        total = self._splitter_total_width(splitter)
        if total <= 0:
            return

        if splitter is self._top_splitter:
            desired_min_left = int(total * self._top_left_ratio_min)
            desired_max_left = int(total * self._top_left_ratio_max)
        else:
            desired_min_left = int(total * self._bottom_left_ratio_min)
            desired_max_left = int(total * self._bottom_left_ratio_max)

        left_w = splitter.widget(0)
        right_w = splitter.widget(1)
        left_min = left_w.minimumSizeHint().width() if left_w is not None else 0
        right_min = right_w.minimumSizeHint().width() if right_w is not None else 0

        hard_min_left = max(0, left_min)
        hard_max_left = max(0, total - right_min)

        min_left = max(desired_min_left, hard_min_left)
        max_left = min(desired_max_left, hard_max_left)
        if min_left > max_left:
            min_left = hard_min_left
            max_left = hard_max_left
        if min_left > max_left:
            return

        left = sizes[0]
        clamped = max(min_left, min(max_left, left))
        if clamped != left:
            self._splitter_clamp_active.add(splitter_key)
            try:
                blocker = QSignalBlocker(splitter)
                splitter.setSizes([clamped, total - clamped])
                del blocker
            finally:
                self._splitter_clamp_active.discard(splitter_key)

    def _set_splitter_ratio(self, splitter: QSplitter, left_ratio: float) -> None:
        total = self._splitter_total_width(splitter)
        if total <= 0:
            return
        left = int(total * left_ratio)
        splitter.setSizes([left, total - left])

    @staticmethod
    def _splitter_total_width(splitter: QSplitter) -> int:
        total = splitter.width() - splitter.handleWidth()
        if total > 0:
            return total
        sizes = splitter.sizes()
        return sum(sizes) if sizes else 0

    def _apply_host(self) -> None:
        host = self._host.text().strip() or "127.0.0.1"
        port = self._port.text().strip() or "8080"
        self._api.set_base_url(f"http://{host}:{port}")
        self._conn_state.setText("Host updated")
        self._log_line(f"Base URL set to http://{host}:{port}")
        self._poll_status()

    def _open_stepper_popup(self) -> None:
        if self._stepper_popup is None:
            self._stepper_popup = StepperPopup(self)
            self._stepper_popup.play_requested.connect(self._on_stepper_play)
            self._stepper_popup.single_step_requested.connect(self._on_stepper_single_step)
        self._stepper_popup.show()
        self._stepper_popup.raise_()
        self._stepper_popup.activateWindow()

    def _is_stepper_active(self) -> bool:
        return self._stepper_popup is not None and self._stepper_popup.isVisible()

    def _start_stepper_action(
        self,
        title: str,
        steps: list[tuple[str, Callable[[Callable[[bool], None]], None]]],
        on_done: Callable[[bool], None],
    ) -> None:
        if self._stepper_popup is None:
            self._open_stepper_popup()
        if self._stepper_popup is None:
            on_done(False)
            return

        self._stepper_steps = list(steps)
        self._stepper_index = 0
        self._stepper_running = False
        self._stepper_auto_play = False
        self._stepper_done = on_done
        self._stepper_run_token += 1
        self._stepper_popup.set_action(title, [label for label, _fn in self._stepper_steps])
        self._stepper_popup.update_progress(0, None)
        self._tray_editor.show_status("Debug stepping active: use Play or Single Step.", ok=True)

    def _on_stepper_play(self) -> None:
        if not self._stepper_steps:
            return
        self._stepper_auto_play = True
        self._run_stepper_next()

    def _on_stepper_single_step(self) -> None:
        if not self._stepper_steps:
            return
        self._stepper_auto_play = False
        self._run_stepper_next()

    def _run_stepper_next(self) -> None:
        if self._stepper_running:
            return

        if self._stepper_index >= len(self._stepper_steps):
            total = len(self._stepper_steps)
            done_cb = self._stepper_done
            if self._stepper_popup is not None:
                self._stepper_popup.update_progress(total, None)
            self._stepper_steps = []
            self._stepper_done = None
            if done_cb is not None:
                done_cb(True)
            return

        if self._stepper_popup is not None:
            self._stepper_popup.update_progress(self._stepper_index, self._stepper_index)

        _label, step_fn = self._stepper_steps[self._stepper_index]
        self._stepper_running = True
        token = self._stepper_run_token
        done_called = {"value": False}

        def _done_once(ok: bool) -> None:
            if done_called["value"]:
                return
            done_called["value"] = True
            if token != self._stepper_run_token:
                return
            self._on_stepper_step_finished(ok)

        step_fn(_done_once)

    def _on_stepper_step_finished(self, ok: bool) -> None:
        self._stepper_running = False
        if not ok:
            done_cb = self._stepper_done
            self._stepper_steps = []
            self._stepper_done = None
            if done_cb is not None:
                done_cb(False)
            return

        self._stepper_index += 1
        if self._stepper_popup is not None:
            self._stepper_popup.update_progress(self._stepper_index, None)

        if self._stepper_index >= len(self._stepper_steps):
            done_cb = self._stepper_done
            self._stepper_steps = []
            self._stepper_done = None
            if done_cb is not None:
                done_cb(True)
            return

        if self._stepper_auto_play:
            self._run_stepper_next()

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
            self._machine_status.setText("Disconnected: coordinates unavailable")
            self._log_line(f"ERR {status}: status poll failed: {data.get('error', 'request_failed')}")
            return

        self._conn_state.setText("Connected")

        positions = data.get("positions", {}) if isinstance(data.get("positions"), dict) else {}
        try:
            self._current_x = float(positions.get("X")) if positions.get("X") is not None else None
        except Exception:
            self._current_x = None
        try:
            self._current_y = float(positions.get("Y")) if positions.get("Y") is not None else None
        except Exception:
            self._current_y = None
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
        self._nozzle_status_by_name = {
            str(nozzle.get("name", "")).strip().upper(): nozzle
            for nozzle in nozzles
            if isinstance(nozzle, dict) and str(nozzle.get("name", "")).strip()
        }
        self._sync_nozzle_cards(nozzles)
        self._tray_editor.set_nozzle_choices(nozzles)
        self._update_machine_status_bar(positions, nozzles)

        feeders = data.get("feeders", []) if isinstance(data.get("feeders"), list) else []
        self._sync_feeders(feeders)

    def _sync_camera_tiles(self, cameras: list[dict[str, Any]]) -> None:
        known = set(self._camera_tiles.keys())
        incoming: set[str] = set()
        status_by_name: dict[str, bool] = {}

        for camera in cameras:
            name = str(camera.get("name", "")).upper()
            if not name:
                continue
            incoming.add(name)
            status_by_name[name] = bool(camera.get("online", False))

            tile = self._camera_tiles.get(name)
            if tile is None:
                tile = CameraTile(name)
                tile.vector_move_requested.connect(self._on_camera_vector_move)
                tile.calibrate_requested.connect(self._on_camera_calibrate)
                tile.light_set_requested.connect(self._on_camera_light_set)
                tile.camera_selected.connect(self._on_camera_selected)
                self._camera_tiles[name] = tile

            tile.apply_status(bool(camera.get("online", False)))
            tile.set_resolution_dpcm(
                float(camera.get("resolution_dpcm_x", 0.0) or 0.0),
                float(camera.get("resolution_dpcm_y", 0.0) or 0.0),
            )
            tile.set_flip(
                bool(camera.get("flip_horizontal", False)),
                bool(camera.get("flip_vertical", False)),
            )
            tile.sync_lights(camera.get("lights") if isinstance(camera.get("lights"), dict) else {})

        for name in known - incoming:
            tile = self._camera_tiles.pop(name)
            tile.setParent(None)
            tile.deleteLater()
        if not self._camera_tiles:
            self._camera_order = []
            self._active_camera_name = ""
            self._show_selected_camera()
            return

        ordered = sorted(self._camera_tiles.keys())
        self._camera_order = ordered
        self._camera_status_by_name = status_by_name
        if self._active_camera_name not in self._camera_tiles:
            # Startup/default behavior: prefer TOP camera when available.
            self._active_camera_name = "TOP" if "TOP" in self._camera_tiles else ordered[0]

        for tile in self._camera_tiles.values():
            tile.sync_camera_choices(ordered, status_by_name, self._active_camera_name)
        self._show_selected_camera()

    def _refresh_camera_thumbs(self) -> None:
        name = self._active_camera_name
        if not name:
            return
        tile = self._camera_tiles.get(name)
        if tile is None or name in self._camera_thumb_pending:
            return

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

    def _show_selected_camera(self) -> None:
        selected = self._active_camera_name
        if selected == self._shown_camera_name:
            return

        while self._camera_host_layout.count():
            item = self._camera_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        tile = self._camera_tiles.get(selected)
        if tile is None:
            self._camera_placeholder = QLabel("No cameras found in /api/status")
            self._camera_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._camera_host_layout.addWidget(self._camera_placeholder)
            self._shown_camera_name = ""
            return

        self._camera_host_layout.addWidget(tile)
        self._shown_camera_name = selected
        self._refresh_camera_thumbs()

    def _on_camera_selected(self, name: str) -> None:
        name = str(name or "").upper()
        if not name or name == self._active_camera_name:
            return
        self._active_camera_name = name
        for tile in self._camera_tiles.values():
            tile.sync_camera_choices(self._camera_order, self._camera_status_by_name, self._active_camera_name)
        self._show_selected_camera()

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
            selected = self._feeders_by_id.get(self._selected_feeder_id, {})
            if str(selected.get("feeder_type", "")).strip().lower() == "tray_feeder" and self._tray_editor.is_dirty():
                return
            self._open_feeder_editor(self._selected_feeder_id, activate_tab=False)
        self._refresh_catalog_status()

    def _load_packages_from_config(self) -> None:
        self._packages_by_name = {}
        cfg_root = Path(__file__).resolve().parents[3] / "config" / "examples" / "packages"
        if not cfg_root.exists() or not cfg_root.is_dir():
            self._log_line(f"WARN: package config directory not found: {cfg_root}")
            self._refresh_package_table()
            return

        for path in sorted(cfg_root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("expected JSON object")
                name = str(raw.get("name", "")).strip().upper()
                if not name:
                    raise ValueError("missing package name")
                self._packages_by_name[name] = {
                    "name": name,
                    "footprint": str(raw.get("footprint", "")).strip(),
                    "length_mm": float(raw.get("length_mm", 0.0) or 0.0),
                    "width_mm": float(raw.get("width_mm", 0.0) or 0.0),
                    "height_mm": float(raw.get("height_mm", 0.0) or 0.0),
                    "pin_count": int(raw.get("pin_count", 0) or 0),
                }
            except Exception as exc:
                self._log_line(f"WARN: failed to load package config {path.name}: {exc}")

        self._refresh_package_table()

    def _load_setup_cameras_from_config(self) -> None:
        self._setup_cameras = []
        try:
            raw = json.loads(self._setup_camera_config_path.read_text(encoding="utf-8"))
            cameras: Any = []
            if isinstance(raw, dict):
                camera_block = raw.get("camera")
                if isinstance(camera_block, dict):
                    cameras = camera_block.get("cameras", [])
                else:
                    cameras = raw.get("cameras", [])
            if isinstance(cameras, list):
                for item in cameras:
                    if isinstance(item, dict):
                        self._setup_cameras.append(dict(item))
        except Exception as exc:
            self._log_line(f"WARN: failed to load setup cameras: {exc}")

        self._refresh_setup_camera_table()
        if self._setup_cameras:
            self._setup_camera_table.selectRow(0)
            self._on_setup_camera_row_selected(0, 0)
        else:
            self._setup_camera_current_row = -1

    def _load_setup_positions_from_config(self) -> None:
        self._setup_positions = []
        try:
            loc_raw = json.loads(self._setup_locations_config_path.read_text(encoding="utf-8"))
            if isinstance(loc_raw, dict):
                locations = loc_raw.get("locations", {})
                if isinstance(locations, dict):
                    for name, coords in locations.items():
                        if not isinstance(coords, dict):
                            continue
                        self._setup_positions.append(
                            {
                                "name": str(name).strip().lower(),
                                "kind": "location",
                                "x": float(coords.get("X", 0.0) or 0.0),
                                "y": float(coords.get("Y", 0.0) or 0.0),
                            }
                        )

            cam_raw = json.loads(self._setup_camera_config_path.read_text(encoding="utf-8"))
            if isinstance(cam_raw, dict):
                camera_block = cam_raw.get("camera")
                cameras = camera_block.get("cameras", []) if isinstance(camera_block, dict) else []
                if isinstance(cameras, list):
                    for item in cameras:
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("name", "")).strip().upper() != "BOTTOM":
                            continue
                        if item.get("x") is None or item.get("y") is None:
                            continue
                        self._setup_positions.append(
                            {
                                "name": "bottom_camera",
                                "kind": "camera",
                                "camera_name": "BOTTOM",
                                "x": float(item.get("x", 0.0) or 0.0),
                                "y": float(item.get("y", 0.0) or 0.0),
                            }
                        )
                        break
        except Exception as exc:
            self._log_line(f"WARN: failed to load setup positions: {exc}")

        self._refresh_setup_position_table()
        if self._setup_positions:
            self._setup_position_table.selectRow(0)
            self._on_setup_position_row_selected(0, 0)
        else:
            self._setup_position_current_row = -1

    def _refresh_setup_position_table(self) -> None:
        rows = sorted(self._setup_positions, key=lambda item: (str(item.get("kind", "")), str(item.get("name", ""))))
        self._setup_positions = rows
        self._setup_position_table.setRowCount(len(rows))
        for row, pos in enumerate(rows):
            name = str(pos.get("name", ""))
            kind = str(pos.get("kind", ""))
            cells = [
                name,
                kind,
                self._fmt(pos.get("x")),
                self._fmt(pos.get("y")),
            ]
            for col, value in enumerate(cells):
                self._setup_position_table.setItem(row, col, QTableWidgetItem(value))

    def _store_current_setup_position_editor(self) -> None:
        row = self._setup_position_current_row
        if row < 0 or row >= len(self._setup_positions):
            return
        target = self._setup_positions[row]
        target["name"] = self._setup_position_name.text().strip().lower()
        target["x"] = float(self._setup_position_x.value())
        target["y"] = float(self._setup_position_y.value())

    def _on_setup_position_row_selected(self, row: int, _col: int) -> None:
        self._store_current_setup_position_editor()
        self._refresh_setup_position_table()
        self._setup_position_current_row = int(row)
        if row < 0 or row >= len(self._setup_positions):
            return

        pos = self._setup_positions[row]
        self._setup_position_name.setText(str(pos.get("name", "")))
        self._setup_position_kind.setText(str(pos.get("kind", "")))
        self._setup_position_x.setValue(float(pos.get("x", 0.0) or 0.0))
        self._setup_position_y.setValue(float(pos.get("y", 0.0) or 0.0))

    def _on_setup_position_add(self) -> None:
        self._store_current_setup_position_editor()
        idx = 1
        existing = {str(item.get("name", "")).strip().lower() for item in self._setup_positions}
        while True:
            name = f"location_{idx}"
            if name not in existing:
                break
            idx += 1
        self._setup_positions.append({"name": name, "kind": "location", "x": 0.0, "y": 0.0})
        self._refresh_setup_position_table()
        new_row = next((i for i, item in enumerate(self._setup_positions) if str(item.get("name", "")) == name), len(self._setup_positions) - 1)
        self._setup_position_table.selectRow(new_row)
        self._on_setup_position_row_selected(new_row, 0)

    def _on_setup_position_move_camera_there(self) -> None:
        row = self._setup_position_current_row
        if row < 0 or row >= len(self._setup_positions):
            self._log_line("ERR: move failed: no special position selected")
            return
        self._store_current_setup_position_editor()
        item = self._setup_positions[row]
        x = float(item.get("x", 0.0) or 0.0)
        y = float(item.get("y", 0.0) or 0.0)
        self._move_camera_to_xy(x, y)

    def _on_setup_position_capture_current(self) -> None:
        row = self._setup_position_current_row
        if row < 0 or row >= len(self._setup_positions):
            self._log_line("ERR: capture failed: no special position selected")
            return
        if self._current_x is None or self._current_y is None:
            self._log_line("ERR: capture failed: actual camera XY position unknown")
            return

        self._setup_position_x.setValue(float(self._current_x))
        self._setup_position_y.setValue(float(self._current_y))
        self._store_current_setup_position_editor()
        self._refresh_setup_position_table()
        self._setup_position_table.selectRow(row)
        self._log_line(
            f"OK: updated {self._setup_positions[row].get('name', 'position')} to X={float(self._current_x):.3f}, Y={float(self._current_y):.3f}"
        )

    def _on_setup_position_save(self) -> None:
        self._store_current_setup_position_editor()

        names = [str(item.get("name", "")).strip().lower() for item in self._setup_positions]
        if any(not name for name in names):
            self._log_line("ERR: position save failed: position name must not be empty")
            return
        if len(set(names)) != len(names):
            self._log_line("ERR: position save failed: duplicate position names")
            return

        try:
            locations_out: dict[str, dict[str, float]] = {}
            bottom_xy: tuple[float, float] | None = None
            for item in self._setup_positions:
                kind = str(item.get("kind", "")).strip().lower()
                name = str(item.get("name", "")).strip().lower()
                x = float(item.get("x", 0.0) or 0.0)
                y = float(item.get("y", 0.0) or 0.0)
                if kind == "camera" and name == "bottom_camera":
                    bottom_xy = (x, y)
                    continue
                locations_out[name] = {"X": x, "Y": y}

            loc_raw = json.loads(self._setup_locations_config_path.read_text(encoding="utf-8"))
            if not isinstance(loc_raw, dict):
                raise ValueError("locations config root must be an object")
            loc_raw["locations"] = locations_out
            self._setup_locations_config_path.write_text(json.dumps(loc_raw, indent=2) + "\n", encoding="utf-8")

            if bottom_xy is not None:
                cam_raw = json.loads(self._setup_camera_config_path.read_text(encoding="utf-8"))
                if not isinstance(cam_raw, dict):
                    raise ValueError("camera config root must be an object")
                camera_block = cam_raw.get("camera")
                if isinstance(camera_block, dict):
                    cameras = camera_block.get("cameras", [])
                    if isinstance(cameras, list):
                        for cam in cameras:
                            if not isinstance(cam, dict):
                                continue
                            if str(cam.get("name", "")).strip().upper() != "BOTTOM":
                                continue
                            cam["x"] = bottom_xy[0]
                            cam["y"] = bottom_xy[1]
                            break
                        self._setup_camera_config_path.write_text(json.dumps(cam_raw, indent=2) + "\n", encoding="utf-8")

            self._refresh_setup_position_table()
            self._log_line(
                f"OK: saved positions to {self._setup_locations_config_path}"
                + (f" and BOTTOM camera XY to {self._setup_camera_config_path}" if bottom_xy is not None else "")
            )
            for item in self._setup_positions:
                self._apply_setup_position_runtime(item)
        except Exception as exc:
            self._log_line(f"ERR: position save failed: {exc}")

    def _apply_setup_position_runtime(self, item: dict[str, Any]) -> None:
        kind = str(item.get("kind", "")).strip().lower()
        name = str(item.get("name", "")).strip().lower()
        x = float(item.get("x", 0.0) or 0.0)
        y = float(item.get("y", 0.0) or 0.0)
        if kind == "camera" and name == "bottom_camera":
            camera = next((cam for cam in self._setup_cameras if str(cam.get("name", "")).strip().upper() == "BOTTOM"), None)
            if camera is None:
                self._log_line("WARN: bottom camera runtime position not applied: BOTTOM camera not configured")
                return
            payload = {
                "device": str(camera.get("device", "")).strip(),
                "fps": float(camera.get("fps", 0.0) or 0.0),
                "resolution_dpcm_x": float(camera.get("resolution_dpcm_x", 0.0) or 0.0),
                "resolution_dpcm_y": float(camera.get("resolution_dpcm_y", 0.0) or 0.0),
                "flip_horizontal": bool(camera.get("flip_horizontal", False)),
                "flip_vertical": bool(camera.get("flip_vertical", False)),
                "rotation_deg": float(camera.get("rotation_deg", 0.0) or 0.0),
                "x": x,
                "y": y,
            }
            self._api.post_json(
                "/api/camera/BOTTOM/settings",
                payload,
                lambda ok, status, data: self._on_setup_bottom_camera_position_applied(ok, status, data),
            )
            return

        self._api.post_json(
            f"/api/config/location/{name}",
            {"x": x, "y": y},
            lambda ok, status, data, loc=name: self._on_setup_position_runtime_applied(loc, ok, status, data),
        )

    def _on_setup_position_runtime_applied(self, name: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"WARN {status}: location {name} not applied at runtime: {data.get('error', 'request_failed')}")
            return
        self._log_line(f"OK: location {name} applied at runtime")

    def _on_setup_bottom_camera_position_applied(self, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"WARN {status}: bottom camera position not applied at runtime: {data.get('error', 'request_failed')}")
            return
        self._log_line("OK: bottom camera position applied at runtime")

    def _refresh_setup_camera_table(self) -> None:
        self._setup_camera_table.setRowCount(len(self._setup_cameras))
        for row, cam in enumerate(self._setup_cameras):
            name = str(cam.get("name", "")).strip().upper()
            device = str(cam.get("device", "")).strip()
            fps = self._fmt(cam.get("fps"))
            res_x = self._fmt(cam.get("resolution_dpcm_x"))
            res_y = self._fmt(cam.get("resolution_dpcm_y"))
            rot = self._fmt(cam.get("rotation_deg"))
            cells = [name, device, fps, f"{res_x} / {res_y}", rot]
            for col, value in enumerate(cells):
                self._setup_camera_table.setItem(row, col, QTableWidgetItem(value))

    def _store_current_setup_camera_editor(self) -> None:
        row = self._setup_camera_current_row
        if row < 0 or row >= len(self._setup_cameras):
            return
        target = self._setup_cameras[row]
        target["name"] = self._setup_cam_name.text().strip().upper()
        target["device"] = self._setup_cam_device.text().strip()
        target["fps"] = float(self._setup_cam_fps.value())
        target["resolution_dpcm_x"] = float(self._setup_cam_res_x.value())
        target["resolution_dpcm_y"] = float(self._setup_cam_res_y.value())
        target["flip_horizontal"] = bool(self._setup_cam_flip_h.isChecked())
        target["flip_vertical"] = bool(self._setup_cam_flip_v.isChecked())
        target["rotation_deg"] = float(self._setup_cam_rotation.value())

    def _on_setup_camera_row_selected(self, row: int, _col: int) -> None:
        self._store_current_setup_camera_editor()
        self._refresh_setup_camera_table()
        self._setup_camera_current_row = int(row)
        if row < 0 or row >= len(self._setup_cameras):
            return

        cam = self._setup_cameras[row]
        self._setup_cam_name.setText(str(cam.get("name", "")).strip().upper())
        self._setup_cam_device.setText(str(cam.get("device", "")).strip())
        self._setup_cam_fps.setValue(float(cam.get("fps", 10.0) or 10.0))
        self._setup_cam_res_x.setValue(float(cam.get("resolution_dpcm_x", 0.0) or 0.0))
        self._setup_cam_res_y.setValue(float(cam.get("resolution_dpcm_y", 0.0) or 0.0))
        self._setup_cam_flip_h.setChecked(bool(cam.get("flip_horizontal", False)))
        self._setup_cam_flip_v.setChecked(bool(cam.get("flip_vertical", False)))
        self._setup_cam_rotation.setValue(float(cam.get("rotation_deg", 0.0) or 0.0))

    def _on_setup_camera_add(self) -> None:
        self._store_current_setup_camera_editor()
        idx = len(self._setup_cameras) + 1
        existing = {str(c.get("name", "")).strip().upper() for c in self._setup_cameras}
        while True:
            name = f"CAM{idx}"
            if name not in existing:
                break
            idx += 1
        self._setup_cameras.append(
            {
                "name": name,
                "device": "/dev/video0",
                "fps": 10.0,
                "resolution_dpcm_x": 0.0,
                "resolution_dpcm_y": 0.0,
                "flip_horizontal": False,
                "flip_vertical": False,
                "rotation_deg": 0.0,
                "lights": {},
                "pipelines": [],
            }
        )
        self._refresh_setup_camera_table()
        new_row = len(self._setup_cameras) - 1
        self._setup_camera_table.selectRow(new_row)
        self._on_setup_camera_row_selected(new_row, 0)

    def _on_setup_camera_save(self) -> None:
        self._store_current_setup_camera_editor()

        names = [str(c.get("name", "")).strip().upper() for c in self._setup_cameras]
        if any(not n for n in names):
            self._log_line("ERR: camera save failed: camera name must not be empty")
            return
        if len(set(names)) != len(names):
            self._log_line("ERR: camera save failed: duplicate camera names")
            return

        try:
            raw = json.loads(self._setup_camera_config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("camera config root must be an object")

            camera_block = raw.get("camera")
            if isinstance(camera_block, dict):
                camera_block["cameras"] = self._setup_cameras
            else:
                raise ValueError("camera section missing in camera config file")

            self._setup_camera_config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            self._refresh_setup_camera_table()
            self._log_line(f"OK: saved camera config to {self._setup_camera_config_path}")
            for camera in self._setup_cameras:
                self._apply_setup_camera_runtime(camera)
        except Exception as exc:
            self._log_line(f"ERR: camera save failed: {exc}")

    def _apply_setup_camera_runtime(self, camera: dict[str, Any]) -> None:
        name = str(camera.get("name", "")).strip().upper()
        if not name:
            return

        payload = {
            "device": str(camera.get("device", "")).strip(),
            "fps": float(camera.get("fps", 0.0) or 0.0),
            "resolution_dpcm_x": float(camera.get("resolution_dpcm_x", 0.0) or 0.0),
            "resolution_dpcm_y": float(camera.get("resolution_dpcm_y", 0.0) or 0.0),
            "flip_horizontal": bool(camera.get("flip_horizontal", False)),
            "flip_vertical": bool(camera.get("flip_vertical", False)),
            "rotation_deg": float(camera.get("rotation_deg", 0.0) or 0.0),
        }
        self._api.post_json(
            f"/api/camera/{name}/settings",
            payload,
            lambda ok, status, data, cam=name, applied=payload: self._on_setup_camera_runtime_applied(cam, applied, ok, status, data),
        )

    def _on_setup_camera_runtime_applied(
        self,
        camera_name: str,
        payload: dict[str, Any],
        ok: bool,
        status: int,
        data: dict[str, Any],
    ) -> None:
        if not ok:
            self._log_line(
                f"WARN {status}: {camera_name} runtime camera settings not applied: {data.get('error', 'request_failed')}"
            )
            return

        tile = self._camera_tiles.get(camera_name)
        if tile is not None:
            tile.set_resolution_dpcm(
                float(payload.get("resolution_dpcm_x", 0.0) or 0.0),
                float(payload.get("resolution_dpcm_y", 0.0) or 0.0),
            )
        reopened = bool(data.get("reopened", False))
        if reopened:
            self._log_line(f"OK: {camera_name} runtime camera settings applied and device reopened")
        else:
            self._log_line(f"OK: {camera_name} runtime camera settings applied")
        self._poll_status()

    def _refresh_package_table(self) -> None:
        rows = sorted(self._packages_by_name.values(), key=lambda item: str(item.get("name", "")))
        self._package_table.setRowCount(len(rows))
        for row, pkg in enumerate(rows):
            cells = [
                str(pkg.get("name", "")),
                str(pkg.get("footprint", "")),
                self._fmt(pkg.get("length_mm")),
                self._fmt(pkg.get("width_mm")),
                self._fmt(pkg.get("height_mm")),
                str(int(pkg.get("pin_count", 0) or 0)),
            ]
            for col, value in enumerate(cells):
                self._package_table.setItem(row, col, QTableWidgetItem(value))

    def _package_names(self) -> list[str]:
        return sorted(self._packages_by_name.keys())

    def _on_add_package(self) -> None:
        idx = len(self._packages_by_name) + 1
        while True:
            name = f"NEW_PKG_{idx:03d}"
            if name not in self._packages_by_name:
                break
            idx += 1

        self._packages_by_name[name] = {
            "name": name,
            "footprint": "",
            "length_mm": 1.0,
            "width_mm": 1.0,
            "height_mm": 0.5,
            "pin_count": 2,
            "compatible_nozzle_tips": [],
            "_path": str(self._packages_config_dir / f"{name.lower()}.json"),
        }
        self._refresh_package_table()
        self._open_package_details(name)

    def _on_package_row_double_clicked(self, row: int, _col: int) -> None:
        item = self._package_table.item(row, 0)
        if item is None:
            return
        name = item.text().strip().upper()
        if name:
            self._open_package_details(name)

    def _open_package_details(self, name: str) -> None:
        pkg = self._packages_by_name.get(str(name).strip().upper())
        if pkg is None:
            return

        dialog = _PackageEditorDialog(pkg, sorted(self._nozzle_tips_by_id.keys()), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        updated = dict(pkg)
        updated.update(dialog.package_data())
        old_name = str(pkg.get("name", "")).strip().upper()
        self._save_package_config(updated)
        if updated["name"] != old_name:
            self._catalog_db.delete_package(old_name)
            self._packages_by_name.pop(old_name, None)
        self._packages_by_name[str(updated["name"]).strip().upper()] = updated
        self._refresh_package_table()

    def _load_packages_from_config(self) -> None:
        self._packages_by_name = {}
        self._catalog_db.bootstrap_packages_from_dir(self._packages_config_dir)
        for raw in self._catalog_db.load_packages():
            name = str(raw.get("name", "")).strip().upper()
            if not name:
                continue
            self._packages_by_name[name] = {
                "name": name,
                "footprint": str(raw.get("footprint", "")).strip(),
                "length_mm": float(raw.get("length_mm", 0.0) or 0.0),
                "width_mm": float(raw.get("width_mm", 0.0) or 0.0),
                "height_mm": float(raw.get("height_mm", 0.0) or 0.0),
                "pin_count": int(raw.get("pin_count", 0) or 0),
                "compatible_nozzle_tips": [
                    str(t).strip()
                    for t in raw.get("compatible_nozzle_tips", [])
                    if str(t).strip()
                ],
            }

        self._refresh_package_table()
        self._refresh_catalog_status()

    def _save_package_config(self, package: dict[str, Any]) -> None:
        self._catalog_db.upsert_package(package)
        self._refresh_catalog_status()

    def _refresh_parts_table(self) -> None:
        rows = sorted(self._parts_by_id.values(), key=lambda item: str(item.get("part_id", "")))
        self._part_table.setRowCount(len(rows))
        for row, part in enumerate(rows):
            cells = [
                str(part.get("part_id", "")),
                str(part.get("description", "")),
                str(part.get("package", "")),
                str(int(part.get("quantity", 0) or 0)),
            ]
            for col, value in enumerate(cells):
                self._part_table.setItem(row, col, QTableWidgetItem(value))

    def _refresh_nozzle_tip_table(self) -> None:
        rows = sorted(self._nozzle_tips_by_id.values(), key=lambda item: str(item.get("id", "")))
        self._nozzle_tip_table.setRowCount(len(rows))
        for row, tip in enumerate(rows):
            cells = [
                str(tip.get("id", "")),
                self._fmt(tip.get("suction_hole_diameter_mm")),
                self._fmt(tip.get("component_min_mm")),
                self._fmt(tip.get("component_max_mm")),
            ]
            for col, value in enumerate(cells):
                self._nozzle_tip_table.setItem(row, col, QTableWidgetItem(value))

    def _refresh_nozzle_table(self) -> None:
        rows = sorted(self._nozzles_by_name.values(), key=lambda item: str(item.get("name", "")))
        self._nozzle_table.setRowCount(len(rows))
        for row, nozzle in enumerate(rows):
            cells = [
                str(nozzle.get("name", "")),
                str(nozzle.get("z_axis", "")),
                self._fmt(nozzle.get("min_z")),
                self._fmt(nozzle.get("max_z")),
                self._fmt(nozzle.get("offset_x")),
                self._fmt(nozzle.get("offset_y")),
                str(nozzle.get("tip_id", "") or ""),
            ]
            for col, value in enumerate(cells):
                self._nozzle_table.setItem(row, col, QTableWidgetItem(value))

    def _on_add_part(self) -> None:
        idx = len(self._parts_by_id) + 1
        while True:
            part_id = f"PART-{idx:03d}"
            if part_id not in self._parts_by_id:
                break
            idx += 1

        package_name = self._package_names()[0] if self._packages_by_name else ""

        self._parts_by_id[part_id] = {
            "part_id": part_id,
            "description": "",
            "package": package_name,
            "quantity": 1,
        }
        self._refresh_parts_table()
        self._tray_editor.set_part_suggestions(sorted(self._parts_by_id.keys()))
        self._open_part_details(part_id)

    def _on_part_row_double_clicked(self, row: int, _col: int) -> None:
        item = self._part_table.item(row, 0)
        if item is None:
            return
        part_id = item.text().strip().upper()
        if part_id:
            self._open_part_details(part_id)

    def _open_part_details(self, part_id: str) -> None:
        part = self._parts_by_id.get(str(part_id).strip().upper())
        if part is None:
            return

        dialog = _PartEditorDialog(part, self._package_names(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        old_id = str(part.get("part_id", "")).strip().upper()
        updated = dialog.part_data()
        new_id = str(updated.get("part_id", "")).strip().upper()
        if old_id and old_id != new_id:
            self._parts_by_id.pop(old_id, None)
            self._catalog_db.delete_part(old_id)
        self._parts_by_id[updated["part_id"]] = updated
        self._save_parts_config()
        self._refresh_parts_table()
        self._tray_editor.set_part_suggestions(sorted(self._parts_by_id.keys()))

    def _load_parts_from_config(self) -> None:
        self._parts_by_id = {}
        self._catalog_db.bootstrap_parts_from_file(self._parts_config_path)
        for item in self._catalog_db.load_parts():
            part_id = str(item.get("part_id", "")).strip().upper()
            if not part_id:
                continue
            self._parts_by_id[part_id] = {
                "part_id": part_id,
                "description": str(item.get("description", "")).strip(),
                "package": str(item.get("package", "")).strip().upper(),
                "quantity": int(item.get("quantity", 0) or 0),
            }
        self._refresh_parts_table()
        self._tray_editor.set_part_suggestions(sorted(self._parts_by_id.keys()))
        self._refresh_catalog_status()

    def _save_parts_config(self) -> None:
        for part in self._parts_by_id.values():
            self._catalog_db.upsert_part(part)
        self._refresh_catalog_status()

    def _load_production_from_db(self) -> None:
        self._pcbs_by_board_number = {}
        for item in self._catalog_db.load_pcbs():
            board_number = str(item.get("board_number", "")).strip().upper()
            if board_number:
                self._pcbs_by_board_number[board_number] = item

        self._panels_by_name = {}
        for item in self._catalog_db.load_panels():
            panel_name = str(item.get("panel_name", "")).strip().upper()
            if panel_name:
                self._panels_by_name[panel_name] = item

        self._jobs_by_name = {}
        for item in self._catalog_db.load_jobs():
            job_name = str(item.get("job_name", "")).strip().upper()
            if job_name:
                self._jobs_by_name[job_name] = item

        self._refresh_single_pcb_selector()
        self._refresh_panel_selector()
        self._refresh_job_selector()

    def _refresh_single_pcb_selector(self) -> None:
        names = sorted(self._pcbs_by_board_number.keys())
        self._single_pcb_select.blockSignals(True)
        self._single_pcb_select.clear()
        self._single_pcb_select.addItem("(new)", "")
        for board_number in names:
            self._single_pcb_select.addItem(board_number, board_number)
        self._single_pcb_select.blockSignals(False)
        if names:
            self._single_pcb_select.setCurrentIndex(1)
            self._on_single_pcb_selected(1)
        else:
            self._on_single_pcb_new()

    def _on_single_pcb_selected(self, _index: int) -> None:
        board_number = str(self._single_pcb_select.currentData() or "").strip().upper()
        if not board_number:
            return
        pcb = self._pcbs_by_board_number.get(board_number)
        if pcb is None:
            return
        self._single_pcb_board_number.setText(board_number)
        self._single_pcb_name.setText(str(pcb.get("name", "")))
        self._single_pcb_version.setText(str(pcb.get("version", "")))
        self._single_pcb_ll_x.setValue(float(pcb.get("ll_x_mm", 0.0) or 0.0))
        self._single_pcb_ll_y.setValue(float(pcb.get("ll_y_mm", 0.0) or 0.0))
        self._single_pcb_rel_z.setValue(float(pcb.get("relative_z_mm", 0.0) or 0.0))
        self._single_pcb_rotation.setValue(float(pcb.get("rotation_deg", 0.0) or 0.0))

        items = pcb.get("items", []) if isinstance(pcb.get("items"), list) else []
        self._single_pcb_items_table.setRowCount(0)
        for entry in items:
            if not isinstance(entry, dict):
                continue
            row = self._single_pcb_items_table.rowCount()
            self._single_pcb_items_table.insertRow(row)
            cells = [
                str(entry.get("part_number", "")),
                str(entry.get("part_code", "")),
                str(entry.get("x_mm", "")),
                str(entry.get("y_mm", "")),
                str(entry.get("rotation_deg", "")),
            ]
            for col, value in enumerate(cells):
                self._single_pcb_items_table.setItem(row, col, QTableWidgetItem(value))

    def _on_single_pcb_new(self) -> None:
        self._single_pcb_select.blockSignals(True)
        self._single_pcb_select.setCurrentIndex(0)
        self._single_pcb_select.blockSignals(False)
        self._single_pcb_board_number.clear()
        self._single_pcb_name.clear()
        self._single_pcb_version.clear()
        self._single_pcb_ll_x.setValue(0.0)
        self._single_pcb_ll_y.setValue(0.0)
        self._single_pcb_rel_z.setValue(0.0)
        self._single_pcb_rotation.setValue(0.0)
        self._single_pcb_items_table.setRowCount(0)

    def _collect_single_pcb(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for row in range(self._single_pcb_items_table.rowCount()):
            part_number = self._single_pcb_items_table.item(row, 0)
            part_code = self._single_pcb_items_table.item(row, 1)
            x_cell = self._single_pcb_items_table.item(row, 2)
            y_cell = self._single_pcb_items_table.item(row, 3)
            rotation_cell = self._single_pcb_items_table.item(row, 4)
            x_text = x_cell.text().strip() if x_cell is not None else ""
            y_text = y_cell.text().strip() if y_cell is not None else ""
            rotation_text = rotation_cell.text().strip() if rotation_cell is not None else ""
            try:
                x_mm = float(x_text or 0.0)
            except Exception:
                x_mm = 0.0
            try:
                y_mm = float(y_text or 0.0)
            except Exception:
                y_mm = 0.0
            try:
                rotation_deg = float(rotation_text or 0.0)
            except Exception:
                rotation_deg = 0.0
            items.append(
                {
                    "part_number": part_number.text().strip() if part_number is not None else "",
                    "part_code": part_code.text().strip() if part_code is not None else "",
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "rotation_deg": rotation_deg,
                }
            )

        return {
            "board_number": self._single_pcb_board_number.text().strip().upper(),
            "name": self._single_pcb_name.text().strip(),
            "version": self._single_pcb_version.text().strip(),
            "ll_x_mm": float(self._single_pcb_ll_x.value()),
            "ll_y_mm": float(self._single_pcb_ll_y.value()),
            "relative_z_mm": float(self._single_pcb_rel_z.value()),
            "rotation_deg": float(self._single_pcb_rotation.value()),
            "items": items,
        }

    def _on_single_pcb_save(self) -> None:
        pcb = self._collect_single_pcb()
        board_number = str(pcb.get("board_number", "")).strip().upper()
        if not board_number:
            self._log_line("ERR: cannot save PCB without board number")
            return
        self._catalog_db.upsert_pcb(pcb)
        self._pcbs_by_board_number[board_number] = pcb
        self._refresh_single_pcb_selector()
        idx = self._single_pcb_select.findData(board_number)
        if idx >= 0:
            self._single_pcb_select.setCurrentIndex(idx)
        self._refresh_catalog_status()
        self._log_line(f"OK: saved PCB {board_number}")

    def _on_single_pcb_delete(self) -> None:
        board_number = self._single_pcb_board_number.text().strip().upper()
        if not board_number:
            return
        self._catalog_db.delete_pcb(board_number)
        self._pcbs_by_board_number.pop(board_number, None)
        self._refresh_single_pcb_selector()
        self._refresh_catalog_status()
        self._log_line(f"OK: deleted PCB {board_number}")

    def _on_single_pcb_add_item(self) -> None:
        row = self._single_pcb_items_table.rowCount()
        self._single_pcb_items_table.insertRow(row)
        for col in range(self._single_pcb_items_table.columnCount()):
            self._single_pcb_items_table.setItem(row, col, QTableWidgetItem(""))
        self._single_pcb_items_table.setCurrentCell(row, 0)

    def _on_single_pcb_remove_item(self) -> None:
        row = self._single_pcb_items_table.currentRow()
        if row < 0:
            row = self._single_pcb_items_table.rowCount() - 1
        if row >= 0:
            self._single_pcb_items_table.removeRow(row)

    def _single_pcb_selected_item_row(self) -> int:
        row = self._single_pcb_items_table.currentRow()
        if row >= 0:
            return row
        if self._single_pcb_items_table.rowCount() == 1:
            return 0
        return -1

    def _on_single_pcb_capture_board_xy(self) -> None:
        if self._current_x is None or self._current_y is None:
            self._log_line("ERR: cannot capture board XY because camera position is unknown")
            return
        self._single_pcb_ll_x.setValue(float(self._current_x))
        self._single_pcb_ll_y.setValue(float(self._current_y))
        self._log_line(
            f"OK: board XY captured from camera position X={self._current_x:.3f}, Y={self._current_y:.3f}"
        )

    def _on_single_pcb_capture_selected_item_xy(self) -> None:
        row = self._single_pcb_selected_item_row()
        if row < 0:
            self._log_line("ERR: select a PCB part row before capturing camera XY")
            return
        if self._current_x is None or self._current_y is None:
            self._log_line("ERR: cannot capture part XY because camera position is unknown")
            return

        board_x = float(self._single_pcb_ll_x.value())
        board_y = float(self._single_pcb_ll_y.value())
        rel_x = float(self._current_x) - board_x
        rel_y = float(self._current_y) - board_y

        self._single_pcb_items_table.setItem(row, 2, QTableWidgetItem(f"{rel_x:.3f}"))
        self._single_pcb_items_table.setItem(row, 3, QTableWidgetItem(f"{rel_y:.3f}"))
        self._log_line(
            f"OK: part row {row + 1} XY set to board-relative camera position X={rel_x:.3f}, Y={rel_y:.3f}"
        )

    def _on_single_pcb_move_camera_to_selected_item(self) -> None:
        row = self._single_pcb_selected_item_row()
        if row < 0:
            self._log_line("ERR: select a PCB part row before moving camera")
            return
        x_item = self._single_pcb_items_table.item(row, 2)
        y_item = self._single_pcb_items_table.item(row, 3)
        try:
            rel_x = float(x_item.text().strip() if x_item is not None else "")
            rel_y = float(y_item.text().strip() if y_item is not None else "")
        except Exception:
            self._log_line("ERR: selected part row has invalid X/Y coordinates")
            return

        board_x = float(self._single_pcb_ll_x.value())
        board_y = float(self._single_pcb_ll_y.value())
        self._move_camera_to_xy(board_x + rel_x, board_y + rel_y)

    def _refresh_panel_selector(self) -> None:
        names = sorted(self._panels_by_name.keys())
        self._panel_select.blockSignals(True)
        self._panel_select.clear()
        self._panel_select.addItem("(new)", "")
        for panel_name in names:
            self._panel_select.addItem(panel_name, panel_name)
        self._panel_select.blockSignals(False)
        if names:
            self._panel_select.setCurrentIndex(1)
            self._on_panel_selected(1)
        else:
            self._on_panel_new()

    def _on_panel_selected(self, _index: int) -> None:
        panel_name = str(self._panel_select.currentData() or "").strip().upper()
        if not panel_name:
            return
        panel = self._panels_by_name.get(panel_name)
        if panel is None:
            return
        self._panel_name.setText(str(panel.get("panel_name", "")))
        self._panel_source_board_number.setText(str(panel.get("source_board_number", "")))
        self._panel_count_x.setValue(max(1, min(10, int(panel.get("count_x", 1) or 1))))
        self._panel_count_y.setValue(max(1, min(10, int(panel.get("count_y", 1) or 1))))
        self._panel_pitch_x.setValue(float(panel.get("pitch_x_mm", 0.0) or 0.0))
        self._panel_pitch_y.setValue(float(panel.get("pitch_y_mm", 0.0) or 0.0))
        self._panel_rotation.setValue(float(panel.get("rotation_deg", 0.0) or 0.0))
        self._panel_import_type.setText(str(panel.get("import_type", "")))
        self._panel_import_file.setText(str(panel.get("import_file", "")))

    def _on_panel_new(self) -> None:
        self._panel_select.blockSignals(True)
        self._panel_select.setCurrentIndex(0)
        self._panel_select.blockSignals(False)
        self._panel_name.clear()
        self._panel_source_board_number.clear()
        self._panel_count_x.setValue(1)
        self._panel_count_y.setValue(1)
        self._panel_pitch_x.setValue(0.0)
        self._panel_pitch_y.setValue(0.0)
        self._panel_rotation.setValue(0.0)
        self._panel_import_type.clear()
        self._panel_import_file.clear()

    def _collect_panel(self) -> dict[str, Any]:
        return {
            "panel_name": self._panel_name.text().strip().upper(),
            "source_board_number": self._panel_source_board_number.text().strip().upper(),
            "count_x": int(self._panel_count_x.value()),
            "count_y": int(self._panel_count_y.value()),
            "pitch_x_mm": float(self._panel_pitch_x.value()),
            "pitch_y_mm": float(self._panel_pitch_y.value()),
            "rotation_deg": float(self._panel_rotation.value()),
            "import_type": self._panel_import_type.text().strip(),
            "import_file": self._panel_import_file.text().strip(),
        }

    def _on_panel_save(self) -> None:
        panel = self._collect_panel()
        panel_name = str(panel.get("panel_name", "")).strip().upper()
        if not panel_name:
            self._log_line("ERR: cannot save panel without name")
            return
        self._catalog_db.upsert_panel(panel)
        self._panels_by_name[panel_name] = panel
        self._refresh_panel_selector()
        idx = self._panel_select.findData(panel_name)
        if idx >= 0:
            self._panel_select.setCurrentIndex(idx)
        self._refresh_catalog_status()
        self._log_line(f"OK: saved panel {panel_name}")

    def _on_panel_delete(self) -> None:
        panel_name = self._panel_name.text().strip().upper()
        if not panel_name:
            return
        self._catalog_db.delete_panel(panel_name)
        self._panels_by_name.pop(panel_name, None)
        self._refresh_panel_selector()
        self._refresh_catalog_status()
        self._log_line(f"OK: deleted panel {panel_name}")

    def _on_panel_import(self) -> None:
        dialog = _PanelImportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        import_type = dialog.import_type()
        file_path = dialog.file_path()
        self._panel_import_type.setText(import_type)
        self._panel_import_file.setText(file_path)
        self._log_line(f"REQ: panel import selected type={import_type}, file={file_path}")

    def _refresh_job_selector(self) -> None:
        names = sorted(self._jobs_by_name.keys())
        self._job_select.blockSignals(True)
        self._job_select.clear()
        self._job_select.addItem("(new)", "")
        for job_name in names:
            self._job_select.addItem(job_name, job_name)
        self._job_select.blockSignals(False)
        if names:
            self._job_select.setCurrentIndex(1)
            self._on_job_selected(1)
        else:
            self._on_job_new()

    def _on_job_selected(self, _index: int) -> None:
        job_name = str(self._job_select.currentData() or "").strip().upper()
        if not job_name:
            return
        job = self._jobs_by_name.get(job_name)
        if job is None:
            return
        self._job_name.setText(str(job.get("job_name", "")))
        self._job_description.setText(str(job.get("description", "")))
        self._job_target_board_number.setText(str(job.get("target_board_number", "")))
        self._job_target_panel_name.setText(str(job.get("target_panel_name", "")))

    def _on_job_new(self) -> None:
        self._job_select.blockSignals(True)
        self._job_select.setCurrentIndex(0)
        self._job_select.blockSignals(False)
        self._job_name.clear()
        self._job_description.clear()
        self._job_target_board_number.clear()
        self._job_target_panel_name.clear()

    def _collect_job(self) -> dict[str, Any]:
        return {
            "job_name": self._job_name.text().strip().upper(),
            "description": self._job_description.text().strip(),
            "target_board_number": self._job_target_board_number.text().strip().upper(),
            "target_panel_name": self._job_target_panel_name.text().strip().upper(),
        }

    def _on_job_save(self) -> None:
        job = self._collect_job()
        job_name = str(job.get("job_name", "")).strip().upper()
        if not job_name:
            self._log_line("ERR: cannot save job without name")
            return
        self._catalog_db.upsert_job(job)
        self._jobs_by_name[job_name] = job
        self._refresh_job_selector()
        idx = self._job_select.findData(job_name)
        if idx >= 0:
            self._job_select.setCurrentIndex(idx)
        self._refresh_catalog_status()
        self._log_line(f"OK: saved job {job_name}")

    def _on_job_delete(self) -> None:
        job_name = self._job_name.text().strip().upper()
        if not job_name:
            return
        self._catalog_db.delete_job(job_name)
        self._jobs_by_name.pop(job_name, None)
        self._refresh_job_selector()
        self._refresh_catalog_status()
        self._log_line(f"OK: deleted job {job_name}")

    def _refresh_catalog_status(self) -> None:
        try:
            counts = self._catalog_db.counts()
            self._catalog_status.setText(
                "Catalog DB: "
                f"{self._catalog_db.path.name} "
                "("
                f"packages={counts.get('packages', 0)}, "
                f"parts={counts.get('parts', 0)}, "
                f"feeders={counts.get('feeders', 0)}, "
                f"pcbs={counts.get('pcbs', 0)}, "
                f"panels={counts.get('panels', 0)}, "
                f"jobs={counts.get('jobs', 0)}"
                ")"
            )
            self._catalog_status.setToolTip(str(self._catalog_db.path))
        except Exception:
            self._catalog_status.setText("Catalog DB: unavailable")

    def _on_add_nozzle_tip(self) -> None:
        idx = len(self._nozzle_tips_by_id) + 1
        while True:
            tip_id = f"T{idx:03d}"
            if tip_id not in self._nozzle_tips_by_id:
                break
            idx += 1

        tip = {
            "id": tip_id,
            "suction_hole_diameter_mm": 0.0,
            "component_min_mm": 0.0,
            "component_max_mm": 0.0,
        }
        self._nozzle_tips_by_id[tip_id] = tip
        self._refresh_nozzle_tip_table()
        self._open_nozzle_tip_details(tip_id)

    def _on_nozzle_tip_row_double_clicked(self, row: int, _col: int) -> None:
        item = self._nozzle_tip_table.item(row, 0)
        if item is None:
            return
        tip_id = item.text().strip()
        if tip_id:
            self._open_nozzle_tip_details(tip_id)

    def _open_nozzle_tip_details(self, tip_id: str) -> None:
        tip = self._nozzle_tips_by_id.get(str(tip_id).strip())
        if tip is None:
            return

        dialog = _NozzleTipEditorDialog(tip, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        updated = dialog.tip_data()
        self._nozzle_tips_by_id[updated["id"]] = updated
        self._save_nozzle_editor_config()
        self._refresh_nozzle_tip_table()
        self._refresh_nozzle_table()

    def _on_add_nozzle(self) -> None:
        idx = len(self._nozzles_by_name) + 1
        while True:
            name = f"N{idx}"
            if name not in self._nozzles_by_name:
                break
            idx += 1

        tip_ids = sorted(self._nozzle_tips_by_id.keys())
        nozzle = {
            "name": name,
            "z_axis": f"Z{idx}",
            "min_z": -50.0,
            "max_z": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "tip_id": tip_ids[0] if tip_ids else None,
            "standard_down_z": -10.0,
            "vacuum_valve": {"board": "", "io_type": "gpio", "pin": 0},
            "air_valve": None,
        }
        self._nozzles_by_name[name] = nozzle
        self._refresh_nozzle_table()
        self._open_nozzle_details(name)

    def _on_nozzle_row_double_clicked(self, row: int, _col: int) -> None:
        item = self._nozzle_table.item(row, 0)
        if item is None:
            return
        name = item.text().strip().upper()
        if name:
            self._open_nozzle_details(name)

    def _open_nozzle_details(self, name: str) -> None:
        nozzle = self._nozzles_by_name.get(str(name).strip().upper())
        if nozzle is None:
            return

        tip_ids = sorted(self._nozzle_tips_by_id.keys())
        if not tip_ids:
            self._log_line("ERR: nozzle edit failed: define at least one nozzle tip first")
            return

        dialog = _NozzleEditorDialog(nozzle, tip_ids, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        updated = dialog.nozzle_data()
        if not updated.get("tip_id"):
            self._log_line("ERR: nozzle save failed: a nozzle tip must be selected")
            return
        self._nozzles_by_name[updated["name"]] = updated
        self._save_nozzle_editor_config()
        self._refresh_nozzle_table()
        self._apply_nozzle_runtime(updated)

    def _load_nozzle_editor_config(self) -> None:
        self._nozzle_tips_by_id = {}
        self._nozzles_by_name = {}
        if not self._nozzles_config_path.exists():
            self._refresh_nozzle_tip_table()
            self._refresh_nozzle_table()
            return

        try:
            raw = json.loads(self._nozzles_config_path.read_text(encoding="utf-8"))
            camera = raw.get("camera", {}) if isinstance(raw, dict) else {}
            nozzle_tips = camera.get("nozzle_tips", []) if isinstance(camera, dict) else []
            nozzles = camera.get("nozzles", []) if isinstance(camera, dict) else []

            if isinstance(nozzle_tips, list):
                for item in nozzle_tips:
                    if not isinstance(item, dict):
                        continue
                    tip_id = str(item.get("id", "")).strip()
                    if not tip_id:
                        continue
                    self._nozzle_tips_by_id[tip_id] = {
                        "id": tip_id,
                        "suction_hole_diameter_mm": item.get("suction_hole_diameter_mm"),
                        "component_min_mm": item.get("component_min_mm"),
                        "component_max_mm": item.get("component_max_mm"),
                    }

            if isinstance(nozzles, list):
                for item in nozzles:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "")).strip().upper()
                    if not name:
                        continue
                    self._nozzles_by_name[name] = {
                        "name": name,
                        "z_axis": str(item.get("z_axis", "")).strip().upper(),
                        "min_z": float(item.get("min_z", 0.0) or 0.0),
                        "max_z": float(item.get("max_z", 0.0) or 0.0),
                        "offset_x": float(item.get("offset_x", 0.0) or 0.0),
                        "offset_y": float(item.get("offset_y", 0.0) or 0.0),
                        "tip_id": str(item.get("tip_id", "")).strip() or None,
                        "standard_down_z": float(item.get("standard_down_z", 0.0) or 0.0),
                        "vacuum_valve": dict(item.get("vacuum_valve", {})) if isinstance(item.get("vacuum_valve"), dict) else {},
                        "air_valve": dict(item.get("air_valve", {})) if isinstance(item.get("air_valve"), dict) else None,
                    }
        except Exception as exc:
            self._log_line(f"WARN: failed to load nozzle config: {exc}")
        self._refresh_nozzle_tip_table()
        self._refresh_nozzle_table()

    def _save_nozzle_editor_config(self) -> None:
        self._nozzles_config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = json.loads(self._nozzles_config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}
        camera = raw.get("camera") if isinstance(raw.get("camera"), dict) else {}
        if not isinstance(camera, dict):
            camera = {}
        camera["nozzle_tips"] = [dict(self._nozzle_tips_by_id[key]) for key in sorted(self._nozzle_tips_by_id.keys())]
        camera["nozzles"] = [dict(self._nozzles_by_name[key]) for key in sorted(self._nozzles_by_name.keys())]
        raw["camera"] = camera
        self._nozzles_config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    def _apply_nozzle_runtime(self, nozzle: dict[str, Any]) -> None:
        name = str(nozzle.get("name", "")).strip().upper()
        if not name:
            return
        self._api.post_json(
            f"/api/config/nozzle/{name}",
            nozzle,
            lambda ok, status, data, noz=name: self._on_nozzle_runtime_saved(noz, ok, status, data),
        )

    def _on_nozzle_runtime_saved(self, nozzle_name: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"WARN {status}: nozzle {nozzle_name} runtime update failed: {data.get('error', 'request_failed')}")
            return
        self._log_line(f"OK: nozzle {nozzle_name} applied at runtime")
        self._poll_status()

    def _on_feeder_row_double_clicked(self, row: int, _col: int) -> None:
        item = self._feeder_table.item(row, 0)
        if item is None:
            return
        feeder_id = item.text().strip().upper()
        if feeder_id:
            self._open_feeder_editor(feeder_id)

    def _create_feeder(self, feeder_type: str) -> None:
        payload = {"feeder_type": feeder_type}
        self._api.post_json(
            "/api/feeders",
            payload,
            lambda ok, status, data, ft=feeder_type: self._on_feeder_created(ft, ok, status, data),
        )
        self._log_line(f"REQ: create {self._human_feeder_type(feeder_type)}")

    def _on_feeder_created(self, feeder_type: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"ERR {status}: create {self._human_feeder_type(feeder_type)} failed: {data.get('error', 'request_failed')}")
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is None:
            self._log_line("ERR: feeder create failed: invalid feeder payload")
            return

        feeder_id = str(feeder.get("feeder_id", "")).upper()
        if not feeder_id:
            self._log_line("ERR: feeder create failed: missing feeder_id")
            return

        self._feeders_by_id[feeder_id] = feeder
        self._selected_feeder_id = feeder_id
        self._open_feeder_editor(feeder_id, force=True)

        if data.get("persisted", True):
            self._log_line(f"OK: feeder {feeder_id} created")
        else:
            self._log_line(f"WARN: feeder {feeder_id} created but not persisted: {data.get('persist_error')}")
        self._poll_status()

    def _open_feeder_editor(self, feeder_id: str, force: bool = False, activate_tab: bool = True) -> None:
        feeder = self._feeders_by_id.get(feeder_id)
        if feeder is None:
            return

        feeder_type = str(feeder.get("feeder_type", "")).strip().lower()
        tab_idx = self._feeder_tab_index.get(feeder_type)
        if activate_tab and tab_idx is not None:
            self._feeders_tabs.setCurrentIndex(tab_idx)

        self._selected_feeder_id = feeder_id
        if feeder_type == "tray_feeder":
            if (not force) and self._tray_editor.is_dirty() and feeder_id == self._tray_editor.feeder_id():
                return
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
            self._tray_editor.show_status(f"Save failed ({status}): {data.get('error', 'request_failed')}", ok=False)
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is not None:
            fid = str(feeder.get("feeder_id", feeder_id)).upper()
            self._feeders_by_id[fid] = feeder
            self._selected_feeder_id = fid
            self._tray_editor.set_feeder(feeder)

        if data.get("persisted", True):
            self._log_line(f"OK: feeder {feeder_id} saved")
            self._tray_editor.show_status("Saved and persisted.", ok=True)
        else:
            self._log_line(f"WARN: feeder {feeder_id} updated but not persisted: {data.get('persist_error')}")
            self._tray_editor.show_status("Updated in runtime, but persistence failed.", ok=False)
        self._poll_status()

    def _reset_feeder(self, feeder_id: str) -> None:
        self._api.post_json(
            f"/api/feeders/{feeder_id}/reset",
            None,
            lambda ok, status, data, fid=feeder_id: self._on_feeder_reset(fid, ok, status, data),
        )
        self._log_line(f"REQ: reset feeder {feeder_id}")

    def _advance_feeder_pick(self, feeder_id: str) -> None:
        self._api.post_json(
            f"/api/feeders/{feeder_id}/advance-pick",
            None,
            lambda ok, status, data, fid=feeder_id: self._on_feeder_advanced(fid, ok, status, data),
        )
        self._log_line(f"REQ: advance feeder {feeder_id} pick")

    def _on_feeder_reset(self, feeder_id: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            err = data.get("error", "request_failed")
            self._log_line(f"ERR {status}: reset feeder {feeder_id} failed: {err}")
            self._tray_editor.show_status(f"Reset failed ({status}): {err}", ok=False)
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is not None:
            fid = str(feeder.get("feeder_id", feeder_id)).upper()
            self._feeders_by_id[fid] = feeder
            self._selected_feeder_id = fid
            self._open_feeder_editor(fid, force=True)

        self._log_line(f"OK: feeder {feeder_id} reset")
        self._tray_editor.show_status("Reset complete (picked count and indices set to zero).", ok=True)
        self._poll_status()

    def _on_feeder_advanced(self, feeder_id: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            err = data.get("error", "request_failed")
            self._log_line(f"ERR {status}: advance feeder {feeder_id} failed: {err}")
            self._tray_editor.show_status(f"Advance failed ({status}): {err}", ok=False)
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is not None:
            fid = str(feeder.get("feeder_id", feeder_id)).upper()
            self._feeders_by_id[fid] = feeder
            self._selected_feeder_id = fid
            self._open_feeder_editor(fid, force=True)

        self._log_line(f"OK: feeder {feeder_id} advanced")
        self._tray_editor.show_status("Advanced to next pick position.", ok=True)
        self._poll_status()

    def _on_pick_step_requested(self, feeder_id: str, nozzle_name: str, dwell_ms: int) -> None:
        self._run_pick_part_action(feeder_id, nozzle_name, dwell_ms, lambda _ok: None)

    def _on_bottom_camera_step_requested(self, _feeder_id: str, nozzle_name: str) -> None:
        self._run_bottom_camera_step(
            nozzle_name,
            lambda ok, noz=nozzle_name: self._run_bottom_vision_pipeline(noz, lambda _ok: None) if ok else None,
        )

    def _on_vision_abort_requested(self) -> None:
        self._vision_abort_requested = True
        self._api.post_json(
            "/api/vision/pipeline/abort",
            None,
            lambda _ok, _status, _data: None,
        )
        self._tray_editor.show_status("Vision abort requested.", ok=False)

    def _run_pick_part_action(
        self,
        feeder_id: str,
        nozzle_name: str,
        dwell_ms: int,
        on_done: Callable[[bool], None],
    ) -> None:
        if self._is_stepper_active():
            self._run_pick_part_action_stepper(feeder_id, nozzle_name, dwell_ms, on_done)
            return

        self._run_pick_step(
            feeder_id,
            nozzle_name,
            dwell_ms,
            on_done,
        )

    def _run_pick_part_action_stepper(
        self,
        feeder_id: str,
        nozzle_name: str,
        dwell_ms: int,
        on_done: Callable[[bool], None],
    ) -> None:
        feeder_id = str(feeder_id).strip().upper()
        nozzle_name = str(nozzle_name).strip().upper()
        dwell_ms = max(0, int(dwell_ms))

        nozzle_state = self._nozzle_status_by_name.get(nozzle_name)
        if nozzle_state is None:
            self._tray_editor.show_status(f"Pick failed: nozzle {nozzle_name} unavailable.", ok=False)
            on_done(False)
            return
        if not str(nozzle_state.get("tip_id", "") or "").strip():
            self._tray_editor.show_status(f"Pick failed: nozzle {nozzle_name} has no mounted tip.", ok=False)
            on_done(False)
            return

        ctx: dict[str, float] = {}

        def _advance(done: Callable[[bool], None]) -> None:
            self._api.post_json(
                f"/api/feeders/{feeder_id}/advance-pick",
                None,
                lambda ok, status, data: _after_advance(done, ok, status, data),
            )

        def _after_advance(done: Callable[[bool], None], ok: bool, status: int, data: dict[str, Any]) -> None:
            if not ok:
                self._tray_editor.show_status(f"Advance failed ({status}): {data.get('error', 'request_failed')}", ok=False)
                done(False)
                return

            feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
            if feeder is None:
                self._tray_editor.show_status("Advance failed: invalid feeder payload.", ok=False)
                done(False)
                return

            fid = str(feeder.get("feeder_id", feeder_id)).upper()
            self._feeders_by_id[fid] = feeder
            self._selected_feeder_id = fid
            self._open_feeder_editor(fid, force=True)

            actual = feeder.get("actual_data") if isinstance(feeder.get("actual_data"), dict) else {}
            current_pick = actual.get("current_pick") if isinstance(actual.get("current_pick"), dict) else {}
            try:
                pick_x = float(current_pick.get("x", 0.0) or 0.0)
                pick_y = float(current_pick.get("y", 0.0) or 0.0)
                nozzle_off_x = float(nozzle_state.get("offset_x", 0.0) or 0.0)
                nozzle_off_y = float(nozzle_state.get("offset_y", 0.0) or 0.0)
                feeder_pick_height = float(feeder.get("pick_height", 0.0) or 0.0)
            except Exception:
                self._tray_editor.show_status("Advance failed: invalid pick position data.", ok=False)
                done(False)
                return

            ctx["pick_cam_x"] = pick_x - nozzle_off_x
            ctx["pick_cam_y"] = pick_y - nozzle_off_y
            pick_down_z = self._compute_pick_down_z(nozzle_name, feeder_pick_height)
            if pick_down_z is None:
                self._tray_editor.show_status(
                    f"Pick failed: nozzle {nozzle_name} standard down Z not configured.",
                    ok=False,
                )
                done(False)
                return
            ctx["pick_down_z"] = float(pick_down_z)
            tx = float(ctx.get("pick_cam_x", 0.0))
            ty = float(ctx.get("pick_cam_y", 0.0))
            self._api.post_json(
                "/api/coord/move-xy",
                {"x": tx, "y": ty},
                lambda ok, status, data: _after_xy_pick(done, tx, ty, ok, status, data),
            )

        def _after_xy_pick(done: Callable[[bool], None], tx: float, ty: float, ok: bool, status: int, data: dict[str, Any]) -> None:
            if not ok:
                self._tray_editor.show_status(f"Move XY failed ({status}): {data.get('error', 'request_failed')}", ok=False)
                done(False)
                return
            self._wait_for_xy_target(tx, ty, on_reached=lambda: done(True), on_timeout=lambda: done(False))

        def _move_down(done: Callable[[bool], None]) -> None:
            target_z = float(ctx.get("pick_down_z", 0.0))
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/move-absolute",
                {"z": target_z},
                lambda ok, _status, _data: done(bool(ok)),
            )

        def _vacuum_on(done: Callable[[bool], None]) -> None:
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/vacuum",
                {"on": True},
                lambda ok, _status, _data: done(bool(ok)),
            )

        def _wait_step(done: Callable[[bool], None]) -> None:
            QTimer.singleShot(dwell_ms, lambda: done(True))

        def _z_up(done: Callable[[bool], None]) -> None:
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/move-absolute",
                {"z": 0.0},
                lambda ok, _status, _data: done(bool(ok)),
            )

        def _move_xy_camera(done: Callable[[bool], None]) -> None:
            self._api.post_json(
                f"/api/nozzle/{nozzle_name}/move-to-bottom-camera",
                None,
                lambda ok, status, data: _after_xy_camera(done, ok, status, data),
            )

        def _after_xy_camera(done: Callable[[bool], None], ok: bool, status: int, data: dict[str, Any]) -> None:
            if not ok:
                self._tray_editor.show_status(f"Move to camera failed ({status}): {data.get('error', 'request_failed')}", ok=False)
                done(False)
                return
            target = data.get("machine_target") if isinstance(data.get("machine_target"), dict) else {}
            try:
                tx = float(target.get("x"))
                ty = float(target.get("y"))
            except Exception:
                tx = self._current_x if self._current_x is not None else 0.0
                ty = self._current_y if self._current_y is not None else 0.0
            self._wait_for_xy_target(tx, ty, on_reached=lambda: done(True), on_timeout=lambda: done(False))

        def _camera_level_down(done: Callable[[bool], None]) -> None:
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/move-standard-down",
                None,
                lambda ok, _status, _data: done(bool(ok)),
            )

        def _capture_diag(done: Callable[[bool], None]) -> None:
            self._api.post_json(
                "/api/camera/BOTTOM/capture-diagnostic",
                {"prefix": f"pick_step1_{nozzle_name}", "stage": "after_pick_camera_level"},
                lambda ok, _status, _data: done(bool(ok)),
            )

        self._start_stepper_action(
            f"Pickup Part ({nozzle_name})",
            [
                ("Move to XY", _advance),
                ("Move to down", _move_down),
                ("Switch on vacuum", _vacuum_on),
                ("Wait", _wait_step),
                ("Move Z up", _z_up),
                ("Move XY to camera", _move_xy_camera),
                ("Lower to camera level", _camera_level_down),
                ("Capture diagnostic image", _capture_diag),
            ],
            on_done,
        )

    def _on_process_start_requested(self, feeder_id: str, nozzle_name: str, dwell_ms: int, single_step: bool) -> None:
        if self._process_busy:
            self._log_line("ERR: pick/place process already running")
            self._tray_editor.show_status("Process already running.", ok=False)
            return

        self._process_feeder_id = str(feeder_id).strip().upper()
        self._process_nozzle_name = str(nozzle_name).strip().upper()
        self._process_dwell_ms = max(0, int(dwell_ms))
        self._process_single_step = bool(single_step)
        self._process_queue = [
            "pick_component",
            "run_bottom_vision_pipeline",
            "move_to_final_destination",
        ]
        self._vision_abort_requested = False
        self._log_line(
            f"REQ: start pick/place process feeder={self._process_feeder_id} nozzle={self._process_nozzle_name} single_step={self._process_single_step}"
        )
        self._process_run_next()

    def _on_process_next_requested(self) -> None:
        if self._process_busy:
            self._tray_editor.show_status("Process step currently running.", ok=False)
            return
        if not self._process_queue:
            self._tray_editor.show_status("No pending process step.", ok=False)
            return
        self._process_run_next()

    def _process_run_next(self) -> None:
        if self._process_busy:
            return
        if not self._process_queue:
            self._tray_editor.show_status("Process complete.", ok=True)
            return

        step = self._process_queue[0]
        self._process_busy = True
        self._tray_editor.show_status(f"Running step: {step}", ok=True)

        if step == "pick_component":
            self._run_pick_step(
                self._process_feeder_id,
                self._process_nozzle_name,
                self._process_dwell_ms,
                lambda ok: self._on_process_step_finished("pick_component", ok),
            )
            return

        if step == "run_bottom_vision_pipeline":
            self._run_bottom_vision_pipeline(
                self._process_nozzle_name,
                lambda ok: self._on_process_step_finished("run_bottom_vision_pipeline", ok),
            )
            return

        if step == "move_to_final_destination":
            self._run_move_to_final_destination_placeholder(
                self._process_nozzle_name,
                lambda ok: self._on_process_step_finished("move_to_final_destination", ok),
            )
            return

        self._on_process_step_finished(step, False)

    def _on_process_step_finished(self, step: str, ok: bool) -> None:
        self._process_busy = False
        if not ok:
            self._process_queue = []
            self._tray_editor.show_status(f"Process stopped at step {step}.", ok=False)
            return

        if self._process_queue and self._process_queue[0] == step:
            self._process_queue.pop(0)

        if not self._process_queue:
            self._tray_editor.show_status("Process complete.", ok=True)
            return

        next_step = self._process_queue[0]
        if self._process_single_step:
            self._tray_editor.show_status(f"Step done: {step}. Next: {next_step}", ok=True)
            return

        self._process_run_next()

    def _run_pick_step(
        self,
        feeder_id: str,
        nozzle_name: str,
        dwell_ms: int,
        on_done: Callable[[bool], None],
    ) -> None:
        feeder_id = str(feeder_id).strip().upper()
        nozzle_name = str(nozzle_name).strip().upper()
        nozzle_state = self._nozzle_status_by_name.get(nozzle_name)
        if nozzle_state is None:
            self._log_line(f"ERR: pick step failed: nozzle {nozzle_name} not in runtime status")
            self._tray_editor.show_status(f"Pick failed: nozzle {nozzle_name} unavailable.", ok=False)
            on_done(False)
            return

        tip_id = str(nozzle_state.get("tip_id", "") or "").strip()
        if not tip_id:
            self._log_line(f"ERR: pick step failed: nozzle {nozzle_name} has no mounted tip")
            self._tray_editor.show_status(f"Pick failed: nozzle {nozzle_name} has no mounted tip.", ok=False)
            on_done(False)
            return

        def _fail(status: int, err: str, msg: str) -> None:
            self._log_line(f"ERR {status}: {msg}: {err}")
            self._tray_editor.show_status(f"{msg}: {err}", ok=False)
            on_done(False)

        self._log_line(f"REQ: feeder {feeder_id} advance for pick")
        self._api.post_json(
            f"/api/feeders/{feeder_id}/advance-pick",
            None,
            lambda ok, status, data: self._on_pick_step_advanced(
                feeder_id,
                nozzle_name,
                dwell_ms,
                on_done,
                ok,
                status,
                data,
                _fail,
            ),
        )

    def _on_pick_step_advanced(
        self,
        feeder_id: str,
        nozzle_name: str,
        dwell_ms: int,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
        fail_cb: Callable[[int, str, str], None],
    ) -> None:
        if not ok:
            fail_cb(status, str(data.get("error", "request_failed")), f"advance feeder {feeder_id} failed")
            return

        feeder = data.get("feeder") if isinstance(data.get("feeder"), dict) else None
        if feeder is None:
            fail_cb(500, "invalid_feeder_payload", "advance feeder response invalid")
            return

        fid = str(feeder.get("feeder_id", feeder_id)).upper()
        self._feeders_by_id[fid] = feeder
        self._selected_feeder_id = fid
        self._open_feeder_editor(fid, force=True)

        actual = feeder.get("actual_data") if isinstance(feeder.get("actual_data"), dict) else {}
        current_pick = actual.get("current_pick") if isinstance(actual.get("current_pick"), dict) else {}
        try:
            pick_x = float(current_pick.get("x", 0.0) or 0.0)
            pick_y = float(current_pick.get("y", 0.0) or 0.0)
            feeder_pick_height = float(feeder.get("pick_height", 0.0) or 0.0)
        except Exception:
            fail_cb(400, "invalid_current_pick", "pick position missing after advance")
            return

        nozzle_state = self._nozzle_status_by_name.get(nozzle_name, {})
        try:
            nozzle_off_x = float(nozzle_state.get("offset_x", 0.0) or 0.0)
            nozzle_off_y = float(nozzle_state.get("offset_y", 0.0) or 0.0)
        except Exception:
            nozzle_off_x = 0.0
            nozzle_off_y = 0.0

        cam_x = pick_x - nozzle_off_x
        cam_y = pick_y - nozzle_off_y
        pick_down_z = self._compute_pick_down_z(nozzle_name, feeder_pick_height)
        if pick_down_z is None:
            fail_cb(409, "standard_down_not_set", f"nozzle {nozzle_name} standard down Z not configured")
            return
        self._log_line(
            f"REQ: move nozzle {nozzle_name} to pick X={pick_x:.3f} Y={pick_y:.3f} via camera X={cam_x:.3f} Y={cam_y:.3f}"
        )
        self._api.post_json(
            "/api/coord/move-xy",
            {
                "x": cam_x,
                "y": cam_y,
                "apply_xy_slack_compensation": False,
            },
            lambda move_ok, move_status, move_data: self._on_pick_step_moved(
                nozzle_name,
                cam_x,
                cam_y,
                float(pick_down_z),
                max(0, int(dwell_ms)),
                on_done,
                move_ok,
                move_status,
                move_data,
                fail_cb,
            ),
        )

    def _on_pick_step_moved(
        self,
        nozzle_name: str,
        target_cam_x: float,
        target_cam_y: float,
        target_down_z: float,
        dwell_ms: int,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
        fail_cb: Callable[[int, str, str], None],
    ) -> None:
        if not ok:
            fail_cb(status, str(data.get("error", "request_failed")), "move to pick location failed")
            return

        job_id = str(data.get("job_id", "")).strip()
        if not job_id:
            fail_cb(0, "no_job_id", "move-xy response missing job_id")
            return

        def _after_xy() -> None:
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/move-absolute",
                {"z": float(target_down_z)},
                lambda down_ok, down_status, down_data: self._on_pick_step_down(
                    nozzle_name,
                    dwell_ms,
                    on_done,
                    down_ok,
                    down_status,
                    down_data,
                    fail_cb,
                ),
            )

        self._wait_for_xy_job_and_m114(
            job_id,
            target_cam_x,
            target_cam_y,
            on_success=_after_xy,
            on_failure=lambda err: fail_cb(408, f"xy_move_failed: {err}", "XY move did not complete before Z"),
        )

    def _on_pick_step_down(
        self,
        nozzle_name: str,
        dwell_ms: int,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
        fail_cb: Callable[[int, str, str], None],
    ) -> None:
        if not ok:
            fail_cb(status, str(data.get("error", "request_failed")), "move nozzle to feeder-corrected pick Z failed")
            return

        self._api.post_json(
            f"/api/head/nozzle/{nozzle_name}/vacuum",
            {"on": True},
            lambda vac_ok, vac_status, vac_data: self._on_pick_step_vacuum_on(
                nozzle_name,
                dwell_ms,
                on_done,
                vac_ok,
                vac_status,
                vac_data,
                fail_cb,
            ),
        )

    def _on_pick_step_vacuum_on(
        self,
        nozzle_name: str,
        dwell_ms: int,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
        fail_cb: Callable[[int, str, str], None],
    ) -> None:
        if not ok:
            fail_cb(status, str(data.get("error", "request_failed")), "vacuum on failed")
            return

        self._log_line(f"OK: nozzle {nozzle_name} vacuum on, dwell {dwell_ms} ms")

        def _raise_after_dwell() -> None:
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/move-absolute",
                {"z": 0.0},
                lambda up_ok, up_status, up_data: self._on_pick_step_raised(
                    nozzle_name,
                    on_done,
                    up_ok,
                    up_status,
                    up_data,
                    fail_cb,
                ),
            )

        QTimer.singleShot(max(0, int(dwell_ms)), _raise_after_dwell)

    def _on_pick_step_raised(
        self,
        nozzle_name: str,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
        fail_cb: Callable[[int, str, str], None],
    ) -> None:
        if not ok:
            fail_cb(status, str(data.get("error", "request_failed")), "raise nozzle to Z=0.0 failed")
            return

        def _after_bottom_move(bottom_ok: bool) -> None:
            if not bottom_ok:
                fail_cb(409, "bottom_camera_move_failed", "move nozzle to bottom camera failed after pick")
                return
            self._api.post_json(
                "/api/camera/BOTTOM/capture-diagnostic",
                {
                    "prefix": f"pick_step1_{nozzle_name}",
                    "stage": "after_pick_camera_level",
                },
                lambda cap_ok, cap_status, cap_data: _after_capture(cap_ok, cap_status, cap_data),
            )

        def _after_capture(cap_ok: bool, cap_status: int, cap_data: dict[str, Any]) -> None:
            if not cap_ok:
                fail_cb(cap_status, str(cap_data.get("error", "request_failed")), "pick diagnostic capture failed")
                return
            saved_file = str(cap_data.get("saved_file", ""))
            self._log_line(f"OK: pick diagnostic saved: {saved_file}")
            self._log_line(f"OK: pick step complete for nozzle {nozzle_name}")
            self._tray_editor.show_status(f"Pick complete with nozzle {nozzle_name}.", ok=True)
            self._poll_status()
            on_done(True)

        self._run_bottom_camera_step(nozzle_name, _after_bottom_move)

    def _run_bottom_camera_step(self, nozzle_name: str, on_done: Callable[[bool], None]) -> None:
        nozzle_name = str(nozzle_name).strip().upper()
        if not nozzle_name:
            self._tray_editor.show_status("Bottom-camera step failed: nozzle not selected.", ok=False)
            on_done(False)
            return

        self._api.post_json(
            f"/api/nozzle/{nozzle_name}/move-to-bottom-camera",
            None,
            lambda ok, status, data: self._on_bottom_camera_moved(nozzle_name, on_done, ok, status, data),
        )

    def _default_bottom_vision_pipeline(self) -> list[dict[str, Any]]:
        return [
            {"op": "GaussianBlur", "args": [[5, 5], 0]},
            {"op": "cvtColor", "args": ["COLOR_BGR2HSV"]},
            {"op": "inRange", "args": [[35, 35, 35], [95, 255, 255]]},
            {"op": "morphologyEx", "args": ["MORPH_OPEN", [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]},
            {"op": "morphologyEx", "args": ["MORPH_CLOSE", [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]},
            {"op": "Canny", "args": [60, 160]},
        ]

    def _run_bottom_vision_pipeline(self, nozzle_name: str, on_done: Callable[[bool], None]) -> None:
        steps = self._tray_editor.vision_pipeline_steps()
        if not steps:
            steps = self._default_bottom_vision_pipeline()

        self._vision_abort_requested = False
        payload = {
            "camera": "BOTTOM",
            "steps": steps,
            "preview_step": self._tray_editor.vision_preview_step(),
            "save_diagnostics": True,
            "prefix": f"pick_step2_{str(nozzle_name).strip().upper()}",
        }
        self._api.post_json(
            "/api/vision/pipeline/run",
            payload,
            lambda ok, status, data: self._on_bottom_vision_pipeline_done(nozzle_name, on_done, ok, status, data),
        )

    def _on_bottom_vision_pipeline_done(
        self,
        nozzle_name: str,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
    ) -> None:
        if not ok:
            err = str(data.get("error", "request_failed"))
            self._log_line(f"ERR {status}: bottom vision pipeline failed for {nozzle_name}: {err}")
            self._tray_editor.show_status(f"Bottom vision failed: {err}", ok=False)
            on_done(False)
            return

        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        step_count = int(result.get("step_count", 0) or 0)
        saved_files = data.get("saved_files") if isinstance(data.get("saved_files"), list) else []
        self._log_line(
            f"OK: bottom vision pipeline complete for {nozzle_name}: steps={step_count}, diagnostics={len(saved_files)}"
        )
        self._tray_editor.show_status(
            f"Bottom vision complete ({step_count} steps, {len(saved_files)} diagnostics).",
            ok=True,
        )
        self._poll_status()
        on_done(True)

    def _run_move_to_final_destination_placeholder(self, nozzle_name: str, on_done: Callable[[bool], None]) -> None:
        # Placeholder for future placement target planning and move/placement actions.
        self._log_line(
            f"TODO: step 4 not implemented yet: move nozzle {nozzle_name} to final destination"
        )
        self._tray_editor.show_status(
            "Step 4 placeholder: final destination move TODO.",
            ok=True,
        )
        on_done(True)

    def _on_bottom_camera_moved(
        self,
        nozzle_name: str,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
    ) -> None:
        if not ok:
            err = str(data.get("error", "request_failed"))
            self._log_line(f"ERR {status}: move {nozzle_name} to bottom camera failed: {err}")
            self._tray_editor.show_status(f"Bottom-camera step failed: {err}", ok=False)
            on_done(False)
            return

        job_id = str(data.get("job_id", "")).strip()
        if not job_id:
            self._log_line(f"ERR: move-to-bottom-camera response missing job_id for {nozzle_name}")
            self._tray_editor.show_status("Bottom-camera step failed: no job_id in response", ok=False)
            on_done(False)
            return

        machine_target = data.get("machine_target") if isinstance(data.get("machine_target"), dict) else {}
        try:
            target_x = float(machine_target.get("x"))
            target_y = float(machine_target.get("y"))
        except Exception:
            target_x = self._current_x if self._current_x is not None else 0.0
            target_y = self._current_y if self._current_y is not None else 0.0

        def _after_xy() -> None:
            self._api.post_json(
                f"/api/head/nozzle/{nozzle_name}/move-standard-down",
                None,
                lambda down_ok, down_status, down_data: self._on_bottom_camera_down(
                    nozzle_name,
                    on_done,
                    down_ok,
                    down_status,
                    down_data,
                ),
            )

        def _on_xy_failure(err: str) -> None:
            self._log_line(f"ERR: XY move to bottom camera did not complete for {nozzle_name}: {err}")
            self._tray_editor.show_status("Bottom-camera step failed: XY move did not complete", ok=False)
            on_done(False)

        self._wait_for_xy_job_and_m114(
            job_id,
            target_x,
            target_y,
            on_success=_after_xy,
            on_failure=_on_xy_failure,
        )

    def _on_bottom_camera_xy_timeout(self, nozzle_name: str, on_done: Callable[[bool], None]) -> None:
        self._log_line(f"ERR 408: move {nozzle_name} to bottom camera did not finish in time")
        self._tray_editor.show_status("Bottom-camera step failed: XY move timeout", ok=False)
        on_done(False)

    def _wait_for_xy_target(
        self,
        target_x: float,
        target_y: float,
        on_reached: Callable[[], None],
        on_timeout: Callable[[], None],
        *,
        tolerance_mm: float = 0.02,
        max_attempts: int = 80,
        interval_ms: int = 120,
    ) -> None:
        attempts = {"count": 0}

        def _check_once() -> None:
            self._api.get_json(
                "/api/status",
                lambda ok, _status, data: _on_status(ok, data),
            )

        def _on_status(ok: bool, data: dict[str, Any]) -> None:
            attempts["count"] += 1
            if not ok:
                if attempts["count"] >= max_attempts:
                    on_timeout()
                    return
                QTimer.singleShot(interval_ms, _check_once)
                return

            positions = data.get("positions", {}) if isinstance(data.get("positions"), dict) else {}
            cur_x = positions.get("X")
            cur_y = positions.get("Y")
            try:
                cur_x_f = float(cur_x)
                cur_y_f = float(cur_y)
            except Exception:
                if attempts["count"] >= max_attempts:
                    on_timeout()
                    return
                QTimer.singleShot(interval_ms, _check_once)
                return

            self._current_x = cur_x_f
            self._current_y = cur_y_f
            dx = abs(cur_x_f - float(target_x))
            dy = abs(cur_y_f - float(target_y))
            if dx <= tolerance_mm and dy <= tolerance_mm:
                on_reached()
                return

            if attempts["count"] >= max_attempts:
                on_timeout()
                return

            QTimer.singleShot(interval_ms, _check_once)

        _check_once()

    def _wait_for_job_complete(
        self,
        job_id: str,
        on_success: Callable[[], None],
        on_failure: Callable[[str], None],
        *,
        max_attempts: int = 250,
        interval_ms: int = 100,
    ) -> None:
        """Poll GET /api/jobs/{job_id} until state is 'succeeded', 'failed', or 'canceled'.

        Only returns when the board-level M400 has confirmed the move is done,
        ensuring XY is truly complete before any Z command is sent.
        """
        attempts = {"count": 0}

        def _poll() -> None:
            self._api.get_json(
                f"/api/jobs/{job_id}",
                lambda ok, _status, data: _on_response(ok, data),
            )

        def _on_response(ok: bool, data: dict[str, Any]) -> None:
            attempts["count"] += 1
            if not ok:
                # Transient HTTP error – keep retrying until max_attempts
                if attempts["count"] >= max_attempts:
                    on_failure("job_poll_timeout")
                    return
                QTimer.singleShot(interval_ms, _poll)
                return

            job = data.get("job") if isinstance(data.get("job"), dict) else {}
            state = str(job.get("state", ""))

            if state == "succeeded":
                on_success()
                return
            if state in {"failed", "canceled"}:
                on_failure(f"job_{state}: {job.get('error', '')}")
                return

            # Still queued or running – keep polling
            if attempts["count"] >= max_attempts:
                on_failure("job_poll_timeout")
                return
            QTimer.singleShot(interval_ms, _poll)

        _poll()

    def _wait_for_xy_job_and_m114(
        self,
        job_id: str,
        target_x: float,
        target_y: float,
        on_success: Callable[[], None],
        on_failure: Callable[[str], None],
        *,
        tolerance_mm: float = 0.02,
        max_attempts: int = 500,
        interval_ms: int = 80,
        consecutive_required: int = 3,
    ) -> None:
        """Gate XY completion using both command state and live M114 position.

        Success requires:
        - command job reached 'succeeded'
        - live M114 XY is at target for N consecutive polls
        """
        attempts = {"count": 0}
        stable_hits = {"count": 0}

        def _poll() -> None:
            self._api.get_json(
                f"/api/jobs/{job_id}",
                lambda ok, _status, data: _on_job(ok, data),
            )

        def _on_job(ok: bool, data: dict[str, Any]) -> None:
            attempts["count"] += 1
            if not ok:
                if attempts["count"] >= max_attempts:
                    on_failure("job_poll_timeout")
                    return
                QTimer.singleShot(interval_ms, _poll)
                return

            job = data.get("job") if isinstance(data.get("job"), dict) else {}
            state = str(job.get("state", ""))

            if state in {"failed", "canceled"}:
                on_failure(f"job_{state}: {job.get('error', '')}")
                return

            self._api.get_json(
                "/api/coord/m114",
                lambda m_ok, _m_status, m_data, st=state: _on_m114(m_ok, m_data, st),
            )

        def _on_m114(ok: bool, data: dict[str, Any], job_state: str) -> None:
            if not ok:
                stable_hits["count"] = 0
            else:
                positions = data.get("positions", {}) if isinstance(data.get("positions"), dict) else {}
                cur_x = positions.get("X")
                cur_y = positions.get("Y")
                try:
                    cur_x_f = float(cur_x)
                    cur_y_f = float(cur_y)
                    self._current_x = cur_x_f
                    self._current_y = cur_y_f
                    dx = abs(cur_x_f - float(target_x))
                    dy = abs(cur_y_f - float(target_y))
                    if dx <= tolerance_mm and dy <= tolerance_mm:
                        stable_hits["count"] += 1
                    else:
                        stable_hits["count"] = 0
                except Exception:
                    stable_hits["count"] = 0

            if job_state == "succeeded" and stable_hits["count"] >= max(1, int(consecutive_required)):
                on_success()
                return

            if attempts["count"] >= max_attempts:
                on_failure("xy_not_confirmed_by_m114")
                return

            QTimer.singleShot(interval_ms, _poll)

        _poll()

    def _compute_pick_down_z(self, nozzle_name: str, feeder_pick_height: float) -> float | None:
        nozzle = str(nozzle_name).strip().upper()
        base_raw: Any = None
        status_nozzle = self._nozzle_status_by_name.get(nozzle)
        if isinstance(status_nozzle, dict):
            base_raw = status_nozzle.get("standard_down_z")
        if base_raw is None:
            cfg_nozzle = self._nozzles_by_name.get(nozzle)
            if isinstance(cfg_nozzle, dict):
                base_raw = cfg_nozzle.get("standard_down_z")
        try:
            base_z = float(base_raw)
            feeder_h = float(feeder_pick_height)
        except Exception:
            return None
        # Positive feeder height means higher than PCB, negative means lower.
        return base_z + feeder_h

    def _on_bottom_camera_down(
        self,
        nozzle_name: str,
        on_done: Callable[[bool], None],
        ok: bool,
        status: int,
        data: dict[str, Any],
    ) -> None:
        if not ok:
            err = str(data.get("error", "request_failed"))
            self._log_line(f"ERR {status}: bottom-camera down for {nozzle_name} failed: {err}")
            self._tray_editor.show_status(f"Bottom-camera step failed: {err}", ok=False)
            on_done(False)
            return

        self._log_line(f"OK: bottom-camera step complete for nozzle {nozzle_name}")
        self._tray_editor.show_status(f"Bottom-camera step complete for nozzle {nozzle_name}.", ok=True)
        self._poll_status()
        on_done(True)

    def _set_tray_pick_from_camera(self, feeder_id: str) -> None:
        if self._current_x is None or self._current_y is None:
            self._log_line("ERR: no valid XY position available from status")
            return
        self._tray_editor.set_pick_location(self._current_x, self._current_y)
        self._log_line(f"OK: feeder {feeder_id} pick location set from camera XY ({self._current_x:.3f}, {self._current_y:.3f})")

    def _set_tray_last_from_camera(self, feeder_id: str) -> None:
        if self._current_x is None or self._current_y is None:
            self._log_line("ERR: no valid XY position available from status")
            return
        self._tray_editor.set_last_pick_location(self._current_x, self._current_y)
        self._log_line(f"OK: feeder {feeder_id} last-pick location set from camera XY ({self._current_x:.3f}, {self._current_y:.3f})")

    def _on_camera_vector_move(self, camera_name: str, dx_mm: float, dy_mm: float) -> None:
        if self._current_x is None or self._current_y is None:
            self._log_line(f"ERR: {camera_name}: cannot vector-move camera, XY position unknown")
            return

        target_x = float(self._current_x) + float(dx_mm)
        target_y = float(self._current_y) + float(dy_mm)
        self._log_line(
            f"REQ: {camera_name} vector move dx={dx_mm:.3f}mm dy={dy_mm:.3f}mm -> X={target_x:.3f} Y={target_y:.3f}"
        )
        self._move_camera_to_xy(target_x, target_y)

    def _on_camera_calibrate(self, camera_name: str, dpcm_x: float, dpcm_y: float) -> None:
        self._api.post_json(
            f"/api/camera/{camera_name}/calibrate-resolution",
            {"resolution_dpcm_x": dpcm_x, "resolution_dpcm_y": dpcm_y},
            lambda ok, status, data, cam=camera_name: self._on_camera_calibrated(cam, ok, status, data),
        )
        self._log_line(f"REQ: {camera_name} calibrate resolution ({dpcm_x:.2f}, {dpcm_y:.2f}) dpcm")

    def _on_camera_calibrated(self, camera_name: str, ok: bool, status: int, data: dict[str, Any]) -> None:
        if not ok:
            self._log_line(f"ERR {status}: {camera_name} calibration failed: {data.get('error', 'request_failed')}")
            return

        dpcm_x = float(data.get("resolution_dpcm_x", 0.0) or 0.0)
        dpcm_y = float(data.get("resolution_dpcm_y", 0.0) or 0.0)
        tile = self._camera_tiles.get(camera_name.upper())
        if tile is not None:
            tile.set_resolution_dpcm(dpcm_x, dpcm_y)

        if data.get("persisted", True):
            self._log_line(f"OK: {camera_name} calibrated to {dpcm_x:.2f}/{dpcm_y:.2f} dpcm")
        else:
            self._log_line(
                f"WARN: {camera_name} calibrated in runtime only: {data.get('persist_error', 'persistence_not_configured')}"
            )
        self._poll_status()

    def _on_camera_light_set(self, camera_name: str, light_key: str, value: int) -> None:
        self._api.post_json(
            f"/api/camera/{camera_name}/light",
            {"light": light_key, "value": int(value)},
            lambda ok, status, data, cam=camera_name, key=light_key, v=int(value): self._on_camera_light_set_done(cam, key, v, ok, status, data),
        )
        self._log_line(f"REQ: {camera_name} light {light_key} -> {int(value)}")

    def _on_camera_light_set_done(
        self,
        camera_name: str,
        light_key: str,
        value: int,
        ok: bool,
        status: int,
        data: dict[str, Any],
    ) -> None:
        if not ok:
            self._log_line(f"ERR {status}: {camera_name} light {light_key} failed: {data.get('error', 'request_failed')}")
            return
        self._log_line(f"OK: {camera_name} light {light_key} set to {value}")
        self._poll_status()

    def _go_to_feeder_survey(self) -> None:
        self._feeders_tabs.setCurrentIndex(0)

    def _move_camera_to_xy(self, x: float, y: float) -> None:
        target_x = float(x)
        target_y = float(y)
        self._api.post_json(
            "/api/coord/move-xy",
            {"x": target_x, "y": target_y},
            lambda ok, status, data, tx=target_x, ty=target_y: self._on_move_camera_xy_result(tx, ty, ok, status, data),
        )

    def _on_move_camera_xy_result(self, target_x: float, target_y: float, ok: bool, status: int, data: dict[str, Any]) -> None:
        title = f"Move top camera to X={target_x:.3f}, Y={target_y:.3f}"
        self._handle_action_result(title, ok, status, data)
        if ok:
            job_id = str(data.get("job_id", "")).strip()
            self._start_xy_motion_gate(job_id, title)

    def _start_xy_motion_gate(self, job_id: str, title: str) -> None:
        self._xy_motion_gate_token += 1
        token = self._xy_motion_gate_token
        self._xy_motion_in_progress = True

        if not job_id:
            # No job_id available – gate is advisory only, clear immediately
            self._xy_motion_in_progress = False
            return

        def _on_gate_success() -> None:
            if token != self._xy_motion_gate_token:
                return
            self._xy_motion_in_progress = False
            self._log_line(f"OK: XY movement finished: {title}")

        def _on_gate_failure(err: str) -> None:
            if token != self._xy_motion_gate_token:
                return
            self._xy_motion_in_progress = False
            self._log_line(f"WARN: XY movement gate failed: {title}: {err}")

        self._wait_for_job_complete(job_id, on_success=_on_gate_success, on_failure=_on_gate_failure)

    def _block_if_xy_in_progress(self, title: str) -> bool:
        if not self._xy_motion_in_progress:
            return False
        self._log_line(f"ERR: {title} blocked: XY movement still in progress")
        self._tray_editor.show_status("Z move blocked: wait until XY movement has finished.", ok=False)
        return True

    def _on_nozzle_xy_action_result(
        self,
        nozzle: str,
        action_title: str,
        ok: bool,
        status: int,
        data: dict[str, Any],
    ) -> None:
        title = f"{nozzle}: {action_title}"
        self._handle_action_result(title, ok, status, data)
        if not ok:
            return

        job_id = str(data.get("job_id", "")).strip()
        self._start_xy_motion_gate(job_id, title)

    @staticmethod
    def _set_table_visible_rows(table: QTableWidget, visible_rows: int) -> None:
        rows = max(1, int(visible_rows))
        header_h = table.horizontalHeader().height()
        row_h = table.verticalHeader().defaultSectionSize()
        frame_h = table.frameWidth() * 2
        target_h = header_h + (row_h * rows) + frame_h
        table.setMinimumHeight(target_h)
        table.setMaximumHeight(target_h)

    def _on_nozzle_action(self, nozzle: str, action: str, value: float) -> None:
        if action == "align_to_cam":
            self._api.post_json(
                f"/api/nozzle/{nozzle}/move-to-camera",
                None,
                lambda ok, status, data, noz=nozzle: self._on_nozzle_xy_action_result(
                    noz,
                    "Align to camera",
                    ok,
                    status,
                    data,
                ),
            )
            return

        if action == "cam_to_nozzle":
            self._api.post_json(
                f"/api/nozzle/{nozzle}/move-camera-here",
                None,
                lambda ok, status, data, noz=nozzle: self._on_nozzle_xy_action_result(
                    noz,
                    "Move camera to nozzle",
                    ok,
                    status,
                    data,
                ),
            )
            return

        if action == "above_bottom":
            self._api.post_json(
                f"/api/nozzle/{nozzle}/move-to-bottom-camera",
                None,
                lambda ok, status, data, noz=nozzle: self._on_nozzle_xy_action_result(
                    noz,
                    "Move above bottom camera",
                    ok,
                    status,
                    data,
                ),
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
            if self._block_if_xy_in_progress(f"{nozzle}: Home Z"):
                return
            self._post_action(f"/api/head/nozzle/{nozzle}/home", None, f"{nozzle}: Home Z")
            return

        if action == "z_up":
            if self._block_if_xy_in_progress(f"{nozzle}: Z up"):
                return
            self._post_action(
                f"/api/head/nozzle/{nozzle}/move",
                {"delta": float(value)},
                f"{nozzle}: Z up +{value:.1f}",
            )
            return

        if action == "z_down":
            if self._block_if_xy_in_progress(f"{nozzle}: Z down"):
                return
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
            if self._block_if_xy_in_progress(f"{nozzle}: Move to Z=0.0"):
                return
            self._post_action(
                f"/api/head/nozzle/{nozzle}/move-absolute",
                {"z": 0.0},
                f"{nozzle}: Move to Z=0.0",
            )
            return

        if action == "z_park":
            if self._block_if_xy_in_progress(f"{nozzle}: Park"):
                return
            self._post_action(
                f"/api/head/nozzle/{nozzle}/park",
                None,
                f"{nozzle}: Park",
            )
            return

        if action == "z_standard_down":
            if self._block_if_xy_in_progress(f"{nozzle}: Move to standard down"):
                return
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

    def _update_machine_status_bar(self, positions: dict[str, Any], nozzles: list[dict[str, Any]]) -> None:
        parts: list[str] = [
            f"X={self._fmt(positions.get('X'))}",
            f"Y={self._fmt(positions.get('Y'))}",
        ]

        indexed: list[tuple[int, dict[str, Any]]] = []
        fallback: list[dict[str, Any]] = []
        for nozzle in nozzles:
            name = str(nozzle.get("name", "")).upper()
            suffix = ""
            for ch in reversed(name):
                if ch.isdigit():
                    suffix = ch + suffix
                elif suffix:
                    break
            if suffix:
                indexed.append((int(suffix), nozzle))
            else:
                fallback.append(nozzle)

        indexed.sort(key=lambda t: t[0])
        for idx, nozzle in indexed:
            parts.append(f"Z{idx}={self._fmt(nozzle.get('z_position'))}")
            parts.append(f"R{idx}={self._fmt(nozzle.get('r_position'))}")

        next_idx = (indexed[-1][0] + 1) if indexed else 1
        for nozzle in fallback:
            parts.append(f"Z{next_idx}={self._fmt(nozzle.get('z_position'))}")
            parts.append(f"R{next_idx}={self._fmt(nozzle.get('r_position'))}")
            next_idx += 1

        self._machine_status.setText("  |  ".join(parts))


def run_qt_control(host: str, port: int) -> None:
    app = QApplication.instance() or QApplication([])
    win = ControlWindow(host=host, port=port)
    win.show()
    app.exec()
