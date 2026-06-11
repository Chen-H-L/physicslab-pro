import time
from io import BytesIO
from pathlib import Path
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QSignalBlocker, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .widgets import SliderSpinBox


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _draw_arrow(painter, start, end, color, width=3, head=10):
    painter.setPen(QPen(QColor(color), width))
    painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head * np.cos(angle - np.pi / 7),
        end[1] - head * np.sin(angle - np.pi / 7),
    )
    right = (
        end[0] - head * np.cos(angle + np.pi / 7),
        end[1] - head * np.sin(angle + np.pi / 7),
    )
    painter.drawLine(int(end[0]), int(end[1]), int(left[0]), int(left[1]))
    painter.drawLine(int(end[0]), int(end[1]), int(right[0]), int(right[1]))


def _draw_legend_item(painter, x, y, color, text):
    painter.setPen(QPen(QColor(color), 8))
    painter.drawPoint(int(x), int(y))
    painter.setPen(QColor("#334155"))
    painter.setFont(QFont("Microsoft YaHei", 9))
    painter.drawText(int(x + 12), int(y + 4), text)


class PhaseInputWidget(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._decimals = 4

        self.radians_spinbox = QDoubleSpinBox(self)
        self.radians_spinbox.setRange(-np.pi, np.pi)
        self.radians_spinbox.setDecimals(self._decimals)
        self.radians_spinbox.setSingleStep(0.1)
        self.radians_spinbox.setSuffix(" rad")
        self.radians_spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.radians_spinbox.setKeyboardTracking(False)
        self.radians_spinbox.setAccelerated(True)
        self.radians_spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.radians_spinbox.setMinimumWidth(128)

        self.degrees_spinbox = QDoubleSpinBox(self)
        self.degrees_spinbox.setRange(-180.0, 180.0)
        self.degrees_spinbox.setDecimals(2)
        self.degrees_spinbox.setSingleStep(5.0)
        self.degrees_spinbox.setSuffix(" °")
        self.degrees_spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.degrees_spinbox.setKeyboardTracking(False)
        self.degrees_spinbox.setAccelerated(True)
        self.degrees_spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.degrees_spinbox.setMinimumWidth(112)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.radians_spinbox)
        layout.addWidget(self.degrees_spinbox)

        self.radians_spinbox.valueChanged.connect(self._on_radians_changed)
        self.degrees_spinbox.valueChanged.connect(self._on_degrees_changed)
        self.setValue(value)

    def _clamp_radians(self, value):
        return max(-np.pi, min(np.pi, float(value)))

    def _apply_value(self, value, emit_signal):
        new_value = self._clamp_radians(value)
        old_value = self._value

        rad_blocker = QSignalBlocker(self.radians_spinbox)
        deg_blocker = QSignalBlocker(self.degrees_spinbox)
        self.radians_spinbox.setValue(new_value)
        self.degrees_spinbox.setValue(np.degrees(new_value))
        del deg_blocker
        del rad_blocker

        self._value = new_value
        if emit_signal and not self.signalsBlocked() and abs(old_value - new_value) > 5e-5:
            self.valueChanged.emit(new_value)

    def _on_radians_changed(self, value):
        self._apply_value(value, emit_signal=True)

    def _on_degrees_changed(self, value):
        self._apply_value(np.radians(value), emit_signal=True)

    def value(self):
        return self._value

    def setValue(self, value):
        self._apply_value(value, emit_signal=True)

    def minimum(self):
        return -np.pi

    def maximum(self):
        return np.pi

    def setKeyboardTracking(self, enabled):
        self.radians_spinbox.setKeyboardTracking(enabled)
        self.degrees_spinbox.setKeyboardTracking(enabled)

    def setAccelerated(self, enabled):
        self.radians_spinbox.setAccelerated(enabled)
        self.degrees_spinbox.setAccelerated(enabled)

    def setMinimumHeight(self, min_height):
        super().setMinimumHeight(int(min_height))
        self.radians_spinbox.setMinimumHeight(int(min_height))
        self.degrees_spinbox.setMinimumHeight(int(min_height))


class SpringOscillatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.amplitude_cm = 20.0
        self.displacement_cm = 0.0
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, amplitude_cm: float, displacement_cm: float):
        self.amplitude_cm = max(0.01, float(amplitude_cm))
        self.displacement_cm = float(displacement_cm)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.fillRect(rect, QColor("#ffffff"))

        width = rect.width()
        height = rect.height()
        mid_y = rect.top() + height * 0.58
        base_x = rect.left() + 38
        center_x = rect.left() + width * 0.58
        travel_px = min(width * 0.22, 150)
        view_amplitude = max(12.0, abs(self.amplitude_cm) * 1.25, abs(self.displacement_cm) * 1.25)
        displacement_px = np.clip((self.displacement_cm / view_amplitude) * travel_px, -travel_px, travel_px)
        mass_left = int(center_x + displacement_px)
        mass_width = 74
        mass_height = 52
        mass_rect_left = max(base_x + 120, min(rect.right() - mass_width - 12, mass_left))
        mass_rect_top = int(mid_y - mass_height / 2)

        painter.setPen(QPen(QColor("#9aa4b2"), 2))
        painter.drawLine(rect.left() + 12, int(mid_y), rect.right() - 12, int(mid_y))
        painter.setPen(QPen(QColor("#6b7280"), 3))
        painter.drawLine(base_x, rect.top() + 36, base_x, rect.bottom() - 20)

        spring_start_x = base_x
        spring_end_x = mass_rect_left
        spring_y = int(mid_y)
        segment_count = 12
        lead = 16
        usable = max(32, spring_end_x - spring_start_x - lead * 2)
        step = usable / segment_count
        points = [(spring_start_x, spring_y), (spring_start_x + lead, spring_y)]
        for index in range(segment_count):
            px = spring_start_x + lead + (index + 1) * step
            py = spring_y - 18 if index % 2 == 0 else spring_y + 18
            points.append((int(px), int(py)))
        points.append((spring_end_x, spring_y))

        painter.setPen(QPen(QColor("#2f80ed"), 4))
        for start, end in zip(points, points[1:]):
            painter.drawLine(start[0], start[1], end[0], end[1])

        painter.setPen(QPen(QColor("#d97706"), 2))
        painter.setBrush(QColor("#f59e0b"))
        painter.drawRoundedRect(mass_rect_left, mass_rect_top, mass_width, mass_height, 10, 10)

        marker_left = int(center_x - travel_px)
        marker_right = int(center_x + travel_px)
        painter.setPen(QPen(QColor("#cbd5e1"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(marker_left, rect.top() + 24, marker_left, rect.bottom() - 16)
        painter.drawLine(int(center_x), rect.top() + 24, int(center_x), rect.bottom() - 16)
        painter.drawLine(marker_right, rect.top() + 24, marker_right, rect.bottom() - 16)

        painter.setPen(QColor("#1f2937"))
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(rect.left() + 10, rect.top() + 22, "弹簧振子仿真")
        painter.drawText(
            rect.left() + 10,
            rect.bottom() - 8,
            f"A = {self.amplitude_cm:.2f} cm    x = {self.displacement_cm:.2f} cm",
        )


class PhasorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.amplitude_cm = 20.0
        self.phase_rad = 0.0
        self.omega = 1.0
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, amplitude_cm: float, phase_rad: float, omega: float = 1.0):
        self.amplitude_cm = max(0.01, float(amplitude_cm))
        self.phase_rad = float(phase_rad)
        self.omega = max(float(omega), 1e-9)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(rect, 12, 12)

        footer_height = 40
        diagram_rect = rect.adjusted(12, 10, -12, -footer_height)
        radius = min(diagram_rect.width() * 0.40, diagram_rect.height() * 0.60)
        cx = diagram_rect.center().x()
        cy = diagram_rect.center().y() + 1

        painter.setPen(QPen(QColor("#dbeafe"), 1))
        painter.setBrush(QColor("#f8fbff"))
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
        painter.setPen(QPen(QColor("#cbd5e1"), 2))
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
        painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(cx - radius - 18), int(cy), int(cx + radius + 18), int(cy))
        painter.drawLine(int(cx), int(cy - radius - 18), int(cx), int(cy + radius + 18))

        scale = radius / max(self.amplitude_cm, 1e-6)
        end_x = cx + self.amplitude_cm * np.cos(self.phase_rad) * scale
        end_y = cy - self.amplitude_cm * np.sin(self.phase_rad) * scale
        proj_x = end_x
        proj_y = cy
        x_value = self.amplitude_cm * np.cos(self.phase_rad)
        phase_deg = np.degrees(self.phase_rad)

        painter.setPen(QPen(QColor("#bfdbfe"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(int(end_x), int(end_y), int(proj_x), int(proj_y))
        _draw_arrow(painter, (cx, cy), (end_x, end_y), "#2f80ed", width=4, head=10)
        _draw_arrow(painter, (cx, cy), (proj_x, proj_y), "#f59e0b", width=4, head=9)

        arc_radius = radius * 0.34
        painter.setPen(QPen(QColor("#94a3b8"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            int(cx - arc_radius),
            int(cy - arc_radius),
            int(arc_radius * 2),
            int(arc_radius * 2),
            0,
            int(phase_deg * 16),
        )
        mid_angle = self.phase_rad * 0.5
        theta_x = cx + np.cos(mid_angle) * (arc_radius + 14)
        theta_y = cy - np.sin(mid_angle) * (arc_radius + 14)

        painter.setBrush(QColor("#2f80ed"))
        painter.setPen(QPen(QColor("#2f80ed"), 2))
        painter.drawEllipse(int(end_x - 5), int(end_y - 5), 10, 10)
        painter.setPen(QColor("#334155"))
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(int(end_x + 8), int(end_y - 6), "A")
        painter.drawText(int(proj_x + 6), int(proj_y - 6), "x")
        painter.drawText(int(theta_x), int(theta_y), "θ")

        footer_top = rect.bottom() - footer_height + 7
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawLine(rect.left() + 10, footer_top - 6, rect.right() - 10, footer_top - 6)

        painter.setPen(QColor("#334155"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(
            rect.left() + 14,
            footer_top,
            rect.width() - 28,
            18,
            Qt.AlignmentFlag.AlignLeft,
            f"A = {self.amplitude_cm:.2f} cm    x = {x_value:.2f} cm",
        )
        painter.drawText(
            rect.left() + 14,
            footer_top + 22,
            rect.width() - 28,
            18,
            Qt.AlignmentFlag.AlignLeft,
            f"θ = {self.phase_rad:.2f} rad ({phase_deg:.1f}°)    x = A cos θ",
        )


class CompoundPhasorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.amplitude_1 = 20.0
        self.amplitude_2 = 15.0
        self.phase_1 = 0.0
        self.phase_2 = np.pi / 3
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, amplitude_1: float, amplitude_2: float, phase_1: float, phase_2: float):
        self.amplitude_1 = max(0.01, float(amplitude_1))
        self.amplitude_2 = max(0.01, float(amplitude_2))
        self.phase_1 = float(phase_1)
        self.phase_2 = float(phase_2)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(rect, 12, 12)

        radius = min(rect.width() * 0.36, rect.height() * 0.42)
        cx = rect.center().x() - 8
        cy = rect.center().y() + 2

        v1 = self.amplitude_1 * np.exp(1j * self.phase_1)
        v2 = self.amplitude_2 * np.exp(1j * self.phase_2)
        vr = v1 + v2
        scale = radius / max(abs(v1), abs(v2), abs(vr), 1e-6)

        def to_point(vector: complex):
            return (cx + vector.real * scale, cy - vector.imag * scale)

        p1 = to_point(v1)
        p2 = to_point(v2)
        pr = to_point(vr)
        painter.setPen(QPen(QColor("#dbeafe"), 1))
        painter.setBrush(QColor("#f8fbff"))
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
        painter.setPen(QPen(QColor("#cbd5e1"), 2))
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
        painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(cx - radius - 18), int(cy), int(cx + radius + 18), int(cy))
        painter.drawLine(int(cx), int(cy - radius - 18), int(cx), int(cy + radius + 18))

        _draw_arrow(painter, (cx, cy), p1, "#2f80ed", width=4, head=10)
        painter.setBrush(QColor("#2f80ed"))
        painter.drawEllipse(int(p1[0] - 4), int(p1[1] - 4), 8, 8)

        _draw_arrow(painter, (cx, cy), p2, "#27ae60", width=4, head=10)
        painter.setBrush(QColor("#27ae60"))
        painter.drawEllipse(int(p2[0] - 4), int(p2[1] - 4), 8, 8)

        painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(p1[0]), int(p1[1]), int(pr[0]), int(pr[1]))
        painter.drawLine(int(p2[0]), int(p2[1]), int(pr[0]), int(pr[1]))

        _draw_arrow(painter, (cx, cy), pr, "#e67e22", width=5, head=11)
        painter.setBrush(QColor("#e67e22"))
        painter.drawEllipse(int(pr[0] - 5), int(pr[1] - 5), 10, 10)

        painter.setPen(QColor("#1f2937"))
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(int(p1[0]) + 6, int(p1[1]) - 2, "1")
        painter.drawText(int(p2[0]) + 6, int(p2[1]) - 2, "2")
        painter.drawText(int(pr[0]) + 6, int(pr[1]) - 2, "R")
        _draw_legend_item(painter, rect.left() + 10, rect.bottom() - 10, "#2f80ed", "分振动 1")
        _draw_legend_item(painter, rect.left() + 108, rect.bottom() - 10, "#27ae60", "分振动 2")
        _draw_legend_item(painter, rect.left() + 206, rect.bottom() - 10, "#e67e22", "合振动")


class MotionCurveCanvas(FigureCanvas):
    def __init__(self, titles, ylabels, colors, parent=None):
        self.figure = Figure(figsize=(8, 4.8), dpi=100, facecolor="#f8f9fa")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.titles = list(titles)
        self.ylabels = list(ylabels)
        self.colors = list(colors)
        self.axes = list(self.figure.subplots(len(self.titles), 1, sharex=True))
        self.times = np.array([0.0, 1.0])
        self.series_list = [np.zeros(2) for _ in self.titles]
        self.cursor_lines = []
        self.point_markers = []
        self.current_time = 0.0
        self._render_axes()

    def _render_axes(self):
        self.cursor_lines = []
        self.point_markers = []
        for axis, title, ylabel, color, series in zip(
            self.axes,
            self.titles,
            self.ylabels,
            self.colors,
            self.series_list,
        ):
            axis.clear()
            axis.set_facecolor("#ffffff")
            axis.plot(self.times, series, color=color, linewidth=2.2)
            cursor = axis.axvline(self.current_time, color="#e11d48", linestyle="--", linewidth=1.3)
            current_y = float(np.interp(self.current_time, self.times, series))
            marker, = axis.plot([self.current_time], [current_y], "o", color="#111827", markersize=5)
            axis.grid(True, linestyle="--", alpha=0.28, color="#cbd5e1")
            axis.set_title(title, loc="left", fontsize=11, color="#1f2937", pad=8)
            axis.set_ylabel(ylabel, color="#334155")
            axis.tick_params(colors="#475569")
            for spine in axis.spines.values():
                spine.set_color("#cbd5e1")
            axis.set_xlim(float(self.times[0]), float(self.times[-1]))
            y_min = float(np.min(series))
            y_max = float(np.max(series))
            span = y_max - y_min
            pad = max(1.0, span * 0.18)
            axis.set_ylim(y_min - pad, y_max + pad)
            self.cursor_lines.append(cursor)
            self.point_markers.append(marker)

        self.axes[-1].set_xlabel("t (s)", color="#334155")
        self.figure.tight_layout(pad=1.6)
        self.draw_idle()

    def set_series(self, times, series_list):
        self.times = np.asarray(times, dtype=float)
        self.series_list = [np.asarray(series, dtype=float) for series in series_list]
        if self.current_time < float(self.times[0]) or self.current_time > float(self.times[-1]):
            self.current_time = float(self.times[0])
        self._render_axes()

    def update_current_time(self, current_time: float):
        if self.times.size == 0:
            return
        self.current_time = float(np.clip(current_time, float(self.times[0]), float(self.times[-1])))
        for cursor, marker, series in zip(self.cursor_lines, self.point_markers, self.series_list):
            current_y = float(np.interp(self.current_time, self.times, series))
            cursor.set_xdata([self.current_time, self.current_time])
            marker.set_data([self.current_time], [current_y])
        self.draw_idle()


class EnvelopeCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 2.4), dpi=100, facecolor="#f8f9fa")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axis = self.figure.subplots(1, 1)
        self.times = np.array([0.0, 1.0])
        self.signal = np.zeros(2)
        self.envelope = np.ones(2)
        self.current_time = 0.0
        self.title = "合振动与理论包络"
        self.cursor_line = None
        self.signal_marker = None
        self.upper_marker = None
        self.lower_marker = None
        self._render_axis()

    def _render_axis(self):
        self.axis.clear()
        self.axis.set_facecolor("#ffffff")
        self.axis.fill_between(self.times, self.envelope, -self.envelope, color="#dbeafe", alpha=0.22)
        self.axis.plot(self.times, self.signal, color="#e67e22", linewidth=2.2, label="合振动")
        self.axis.plot(self.times, self.envelope, color="#2563eb", linestyle="--", linewidth=1.8, label="上包络")
        self.axis.plot(self.times, -self.envelope, color="#2563eb", linestyle="--", linewidth=1.8, label="下包络")

        self.cursor_line = self.axis.axvline(self.current_time, color="#e11d48", linestyle="--", linewidth=1.3)
        current_signal = float(np.interp(self.current_time, self.times, self.signal))
        current_env = float(np.interp(self.current_time, self.times, self.envelope))
        self.signal_marker, = self.axis.plot(
            [self.current_time],
            [current_signal],
            "o",
            color="#111827",
            markersize=5,
        )
        self.upper_marker, = self.axis.plot(
            [self.current_time],
            [current_env],
            "o",
            color="#2563eb",
            markersize=4,
        )
        self.lower_marker, = self.axis.plot(
            [self.current_time],
            [-current_env],
            "o",
            color="#2563eb",
            markersize=4,
        )

        self.axis.grid(True, linestyle="--", alpha=0.28, color="#cbd5e1")
        self.axis.set_title(self.title, loc="left", fontsize=11, color="#1f2937", pad=8)
        self.axis.set_ylabel("x (cm)", color="#334155")
        self.axis.set_xlabel("t (s)", color="#334155")
        self.axis.tick_params(colors="#475569")
        for spine in self.axis.spines.values():
            spine.set_color("#cbd5e1")

        self.axis.set_xlim(float(self.times[0]), float(self.times[-1]))
        amplitude_max = max(float(np.max(np.abs(self.signal))), float(np.max(self.envelope)), 1.0)
        pad = max(1.0, amplitude_max * 0.18)
        self.axis.set_ylim(-amplitude_max - pad, amplitude_max + pad)
        self.axis.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        self.figure.tight_layout(pad=1.4)
        self.draw_idle()

    def set_series(self, times, signal, envelope, title):
        self.times = np.asarray(times, dtype=float)
        self.signal = np.asarray(signal, dtype=float)
        self.envelope = np.maximum(0.0, np.asarray(envelope, dtype=float))
        self.title = title
        if self.current_time < float(self.times[0]) or self.current_time > float(self.times[-1]):
            self.current_time = float(self.times[0])
        self._render_axis()

    def update_current_time(self, current_time: float):
        if self.times.size == 0:
            return
        self.current_time = float(np.clip(current_time, float(self.times[0]), float(self.times[-1])))
        current_signal = float(np.interp(self.current_time, self.times, self.signal))
        current_env = float(np.interp(self.current_time, self.times, self.envelope))
        self.cursor_line.set_xdata([self.current_time, self.current_time])
        self.signal_marker.set_data([self.current_time], [current_signal])
        self.upper_marker.set_data([self.current_time], [current_env])
        self.lower_marker.set_data([self.current_time], [-current_env])
        self.draw_idle()


class TrajectoryCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5.8, 3.0), dpi=100, facecolor="#f8f9fa")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axis = self.figure.subplots(1, 1)
        self.times = np.array([0.0, 1.0])
        self.x_values = np.zeros(2)
        self.y_values = np.zeros(2)
        self.current_time = 0.0
        self.title = "合振动轨迹"
        self.cursor_marker = None
        self._render_axis()

    def _render_axis(self):
        self.axis.clear()
        self.axis.set_facecolor("#ffffff")
        self.axis.plot(self.x_values, self.y_values, color="#2f80ed", linewidth=2.2)
        current_x = float(np.interp(self.current_time, self.times, self.x_values))
        current_y = float(np.interp(self.current_time, self.times, self.y_values))
        self.cursor_marker = self.axis.plot(
            [current_x],
            [current_y],
            "o",
            color="#e67e22",
            markersize=6,
        )[0]
        self.axis.axhline(0.0, color="#cbd5e1", linewidth=1.0, linestyle="--")
        self.axis.axvline(0.0, color="#cbd5e1", linewidth=1.0, linestyle="--")
        self.axis.grid(True, linestyle="--", alpha=0.28, color="#cbd5e1")
        self.axis.set_title(self.title, loc="left", fontsize=11, color="#1f2937", pad=8)
        self.axis.set_xlabel("x (cm)", color="#334155")
        self.axis.set_ylabel("y (cm)", color="#334155")
        self.axis.tick_params(colors="#475569")
        for spine in self.axis.spines.values():
            spine.set_color("#cbd5e1")

        max_extent = max(
            float(np.max(np.abs(self.x_values))),
            float(np.max(np.abs(self.y_values))),
            1.0,
        )
        pad = max(1.0, max_extent * 0.18)
        self.axis.set_xlim(-max_extent - pad, max_extent + pad)
        self.axis.set_ylim(-max_extent - pad, max_extent + pad)
        self.axis.set_aspect("equal", adjustable="box")
        self.figure.tight_layout(pad=1.4)
        self.draw_idle()

    def set_series(self, times, x_values, y_values, title):
        self.times = np.asarray(times, dtype=float)
        self.x_values = np.asarray(x_values, dtype=float)
        self.y_values = np.asarray(y_values, dtype=float)
        self.title = title
        if self.current_time < float(self.times[0]) or self.current_time > float(self.times[-1]):
            self.current_time = float(self.times[0])
        self._render_axis()

    def update_current_time(self, current_time: float):
        if self.times.size == 0:
            return
        self.current_time = float(np.clip(current_time, float(self.times[0]), float(self.times[-1])))
        current_x = float(np.interp(self.current_time, self.times, self.x_values))
        current_y = float(np.interp(self.current_time, self.times, self.y_values))
        self.cursor_marker.set_data([current_x], [current_y])
        self.draw_idle()


class VibrationLabTab(QWidget):
    SINGLE_MODE = "单一简谐振动"
    COMPOUND_MODE = "同方向同频率合成"
    DIFF_FREQ_MODE = "同方向不同频率合成"
    ORTHOGONAL_MODE = "方向垂直同频率合成"
    ORTHOGONAL_DIFF_MODE = "方向垂直不同频率合成"
    ORTHOGONAL_DIFF_MANUAL = "手动输入"

    DIFF_SPECIAL_CASE = "特殊情况：A1=A2，φ1=φ2"
    DIFF_GENERAL_CASE = "一般情况：A1、ω1、φ1 与 A2、ω2、φ2 不同"

    SINGLE_TITLES = ["位移 x/t 曲线", "速度 v/t 曲线", "加速度 a/t 曲线"]
    SINGLE_YLABELS = ["x (cm)", "v (cm/s)", "a (cm/s^2)"]
    SINGLE_COLORS = ["#2f80ed", "#27ae60", "#e67e22"]

    COMPOUND_TITLES = ["分振动 1 的 x1/t", "分振动 2 的 x2/t", "合振动 x/t"]
    COMPOUND_YLABELS = ["x1 (cm)", "x2 (cm)", "x (cm)"]
    COMPOUND_COLORS = ["#2f80ed", "#27ae60", "#e67e22"]

    ORTHOGONAL_TITLES = ["x 方向分振动 x/t", "y 方向分振动 y/t"]
    ORTHOGONAL_YLABELS = ["x (cm)", "y (cm)"]
    ORTHOGONAL_COLORS = ["#2f80ed", "#27ae60"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_time = 0.0
        self.duration = 6.0
        self.last_tick = None
        self.micro_lesson_duration = 5.0
        self.micro_lesson_fps = 15

        self.t_values = np.linspace(0.0, self.duration, 1200)
        self.x_values = np.zeros_like(self.t_values)
        self.v_values = np.zeros_like(self.t_values)
        self.a_values = np.zeros_like(self.t_values)
        self.x1_values = np.zeros_like(self.t_values)
        self.x2_values = np.zeros_like(self.t_values)
        self.x_sum_values = np.zeros_like(self.t_values)
        self.diff_envelope_values = np.zeros_like(self.t_values)
        self.orth_x_values = np.zeros_like(self.t_values)
        self.orth_y_values = np.zeros_like(self.t_values)
        self._updating_orthogonal_diff_preset = False

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.on_animation_tick)

        self.init_ui()
        self.on_mode_changed(self.SINGLE_MODE)

    def init_ui(self):
        self.setObjectName("vibrationLabTab")
        self.setStyleSheet(
            """
            QWidget#vibrationLabTab {
                background-color: #f5f7fa;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
                background: #f1f5f9;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e1;
                border-radius: 5px;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 2px solid #d9d9d9;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
                color: #1f2937;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QLabel {
                color: #334155;
                background-color: transparent;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 2px solid #d9d9d9;
                border-radius: 8px;
                padding: 4px;
            }
            """
        )

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_panel.setMaximumWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title_label = QLabel("振动学实验室")
        title_font = QFont("Microsoft YaHei", 15, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1f2937; background: transparent;")
        left_layout.addWidget(title_label)

        self.subtitle_label = QLabel("简谐振动仿真、旋转矢量和运动曲线联动展示")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("color: #64748b; background: transparent;")
        left_layout.addWidget(self.subtitle_label)

        mode_group = QGroupBox("实验模式")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(14, 16, 14, 14)
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumHeight(36)
        self.mode_combo.addItems(
            [
                self.SINGLE_MODE,
                self.COMPOUND_MODE,
                self.DIFF_FREQ_MODE,
                self.ORTHOGONAL_MODE,
                self.ORTHOGONAL_DIFF_MODE,
            ]
        )
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("选择当前实验内容"))
        mode_layout.addWidget(self.mode_combo)
        left_layout.addWidget(mode_group)

        self.single_param_group = self._build_single_param_group()
        self.compound_param_group = self._build_compound_param_group()
        self.diff_freq_param_group = self._build_diff_freq_param_group()
        self.orthogonal_param_group = self._build_orthogonal_param_group()
        self.orthogonal_diff_param_group = self._build_orthogonal_diff_param_group()
        self.single_derived_group = self._build_single_derived_group()
        self.compound_derived_group = self._build_compound_derived_group()
        self.diff_freq_derived_group = self._build_diff_freq_derived_group()
        self.orthogonal_derived_group = self._build_orthogonal_derived_group()
        self.orthogonal_diff_derived_group = self._build_orthogonal_diff_derived_group()
        self.control_group = self._build_control_group()

        left_layout.addWidget(self.single_param_group)
        left_layout.addWidget(self.compound_param_group)
        left_layout.addWidget(self.diff_freq_param_group)
        left_layout.addWidget(self.orthogonal_param_group)
        left_layout.addWidget(self.orthogonal_diff_param_group)
        left_layout.addWidget(self.single_derived_group)
        left_layout.addWidget(self.compound_derived_group)
        left_layout.addWidget(self.diff_freq_derived_group)
        left_layout.addWidget(self.orthogonal_derived_group)
        left_layout.addWidget(self.orthogonal_diff_derived_group)
        left_layout.addWidget(self.control_group)

        self.result_group = QGroupBox("实验说明")
        result_layout = QVBoxLayout(self.result_group)
        result_layout.setContentsMargins(14, 16, 14, 14)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(160)
        result_layout.addWidget(self.result_text)
        left_layout.addWidget(self.result_group)
        left_layout.addStretch(1)
        left_scroll.setWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.display_title = QLabel("简谐振动仿真")
        self.display_title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        self.display_title.setStyleSheet("color: #1f2937; background: transparent;")
        right_layout.addWidget(self.display_title)

        self.single_top_widget = QWidget()
        single_top_layout = QHBoxLayout(self.single_top_widget)
        single_top_layout.setContentsMargins(0, 0, 0, 0)
        single_top_layout.setSpacing(10)

        spring_group = QGroupBox("弹簧振子可视化")
        spring_layout = QVBoxLayout(spring_group)
        spring_layout.setContentsMargins(12, 16, 12, 12)
        self.spring_widget = SpringOscillatorWidget()
        spring_layout.addWidget(self.spring_widget)
        single_top_layout.addWidget(spring_group, 2)

        phasor_group = QGroupBox("旋转矢量示意")
        phasor_layout = QVBoxLayout(phasor_group)
        phasor_layout.setContentsMargins(12, 16, 12, 12)
        self.phasor_widget = PhasorWidget()
        phasor_layout.addWidget(self.phasor_widget)
        single_top_layout.addWidget(phasor_group, 2)
        self.single_top_widget.setMaximumHeight(250)

        self.compound_top_widget = QWidget()
        compound_top_layout = QHBoxLayout(self.compound_top_widget)
        compound_top_layout.setContentsMargins(0, 0, 0, 0)
        compound_top_layout.setSpacing(10)

        compound_visual_group = QGroupBox("旋转矢量法与合成概览")
        compound_visual_layout = QHBoxLayout(compound_visual_group)
        compound_visual_layout.setContentsMargins(12, 16, 12, 12)
        compound_visual_layout.setSpacing(12)

        self.compound_phasor_widget = CompoundPhasorWidget()
        compound_visual_layout.addWidget(self.compound_phasor_widget, 3)

        self.compound_summary_label = QLabel()
        self.compound_summary_label.setWordWrap(True)
        self.compound_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.compound_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.compound_summary_label.setStyleSheet(
            "font-size: 12px; line-height: 1.65; background: transparent; color: #334155; padding-left: 2px;"
        )
        compound_visual_layout.addWidget(self.compound_summary_label, 2)
        compound_visual_layout.setStretch(0, 4)
        compound_visual_layout.setStretch(1, 2)
        compound_top_layout.addWidget(compound_visual_group, 1)
        self.compound_top_widget.setMaximumHeight(250)

        self.diff_freq_top_widget = QWidget()
        diff_top_layout = QHBoxLayout(self.diff_freq_top_widget)
        diff_top_layout.setContentsMargins(0, 0, 0, 0)
        diff_top_layout.setSpacing(10)

        diff_visual_group = QGroupBox("不同频率合成概览")
        diff_visual_layout = QHBoxLayout(diff_visual_group)
        diff_visual_layout.setContentsMargins(12, 16, 12, 12)
        diff_visual_layout.setSpacing(12)

        self.diff_freq_phasor_widget = CompoundPhasorWidget()
        diff_visual_layout.addWidget(self.diff_freq_phasor_widget, 3)

        self.diff_freq_summary_label = QLabel()
        self.diff_freq_summary_label.setWordWrap(True)
        self.diff_freq_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.diff_freq_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.diff_freq_summary_label.setStyleSheet(
            "font-size: 12px; line-height: 1.65; background: transparent; color: #334155; padding-left: 2px;"
        )
        diff_visual_layout.addWidget(self.diff_freq_summary_label, 2)
        diff_visual_layout.setStretch(0, 4)
        diff_visual_layout.setStretch(1, 2)
        diff_top_layout.addWidget(diff_visual_group, 1)
        self.diff_freq_top_widget.setMaximumHeight(250)

        self.orthogonal_top_widget = QWidget()
        orthogonal_top_layout = QHBoxLayout(self.orthogonal_top_widget)
        orthogonal_top_layout.setContentsMargins(0, 0, 0, 0)
        orthogonal_top_layout.setSpacing(10)

        orthogonal_visual_group = QGroupBox("合振动轨迹与概览")
        orthogonal_visual_layout = QHBoxLayout(orthogonal_visual_group)
        orthogonal_visual_layout.setContentsMargins(12, 16, 12, 12)
        orthogonal_visual_layout.setSpacing(12)

        self.trajectory_canvas = TrajectoryCanvas()
        orthogonal_visual_layout.addWidget(self.trajectory_canvas, 3)

        self.orthogonal_summary_label = QLabel()
        self.orthogonal_summary_label.setWordWrap(True)
        self.orthogonal_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.orthogonal_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.orthogonal_summary_label.setStyleSheet(
            "font-size: 12px; line-height: 1.65; background: transparent; color: #334155; padding-left: 2px;"
        )
        orthogonal_visual_layout.addWidget(self.orthogonal_summary_label, 2)
        orthogonal_visual_layout.setStretch(0, 4)
        orthogonal_visual_layout.setStretch(1, 2)
        orthogonal_top_layout.addWidget(orthogonal_visual_group, 1)
        self.orthogonal_top_widget.setMaximumHeight(250)

        self.curve_group = QGroupBox("运动曲线")
        curve_layout = QVBoxLayout(self.curve_group)
        curve_layout.setContentsMargins(12, 16, 12, 12)
        self.single_curve_canvas = MotionCurveCanvas(
            self.SINGLE_TITLES,
            self.SINGLE_YLABELS,
            self.SINGLE_COLORS,
        )
        self.compound_curve_canvas = MotionCurveCanvas(
            self.COMPOUND_TITLES,
            self.COMPOUND_YLABELS,
            self.COMPOUND_COLORS,
        )
        self.diff_freq_curve_canvas = MotionCurveCanvas(
            self.COMPOUND_TITLES,
            self.COMPOUND_YLABELS,
            self.COMPOUND_COLORS,
        )
        self.orthogonal_curve_canvas = MotionCurveCanvas(
            self.ORTHOGONAL_TITLES,
            self.ORTHOGONAL_YLABELS,
            self.ORTHOGONAL_COLORS,
        )
        self.diff_envelope_canvas = EnvelopeCanvas()
        curve_layout.addWidget(self.single_curve_canvas)
        curve_layout.addWidget(self.compound_curve_canvas)
        curve_layout.addWidget(self.diff_freq_curve_canvas)
        curve_layout.addWidget(self.orthogonal_curve_canvas)
        curve_layout.addWidget(self.diff_envelope_canvas)
        self.single_curve_canvas.setMinimumHeight(300)
        self.compound_curve_canvas.setMinimumHeight(300)
        self.diff_freq_curve_canvas.setMinimumHeight(240)
        self.orthogonal_curve_canvas.setMinimumHeight(240)
        self.diff_envelope_canvas.setMinimumHeight(180)

        visual_panel = QWidget()
        visual_layout = QVBoxLayout(visual_panel)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(8)
        visual_layout.addWidget(self.single_top_widget)
        visual_layout.addWidget(self.compound_top_widget)
        visual_layout.addWidget(self.diff_freq_top_widget)
        visual_layout.addWidget(self.orthogonal_top_widget)
        visual_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        right_layout.addWidget(visual_panel, 0)
        right_layout.addWidget(self.curve_group, 1)
        self.record_widget = right_panel

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1080])

    def _build_single_param_group(self):
        group = QGroupBox("单一简谐振动参数")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.amplitude_spin = self._create_spinbox(0.01, 1_000_000.0, 20.0, " cm", 3, 1.0)
        self.omega_spin = self._create_spinbox(0.001, 10_000.0, 3.0, " rad/s", 4, 0.1)
        self.spring_spin = self._create_spinbox(0.001, 1_000_000.0, 10.0, " N/m", 4, 1.0)
        self.phase_spin = self._create_phase_input(0.0)

        rows = [
            ("振幅 A", self.amplitude_spin),
            ("角频率 ω", self.omega_spin),
            ("弹性系数 k", self.spring_spin),
            ("初始相位 φ", self.phase_spin),
        ]
        for row, (text, widget) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(widget, row, 1)

        for spinbox in (
            self.amplitude_spin,
            self.omega_spin,
            self.spring_spin,
            self.phase_spin,
        ):
            spinbox.valueChanged.connect(lambda _: self.refresh_current_mode(reset_time=False))

        return group

    def _build_compound_param_group(self):
        group = QGroupBox("同方向同频率合成参数")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.compound_amplitude_1_spin = self._create_spinbox(0.01, 1_000_000.0, 20.0, " cm", 3, 1.0)
        self.compound_phase_1_spin = self._create_compact_phase_input(0.0)
        self.compound_amplitude_2_spin = self._create_spinbox(0.01, 1_000_000.0, 12.0, " cm", 3, 1.0)
        self.compound_phase_2_spin = self._create_compact_phase_input(1.2)
        self.compound_omega_spin = self._create_spinbox(0.001, 10_000.0, 3.0, " rad/s", 4, 0.1)

        rows = [
            ("分振动 1 振幅 A1", self.compound_amplitude_1_spin),
            ("分振动 1 初相 φ1", self.compound_phase_1_spin),
            ("分振动 2 振幅 A2", self.compound_amplitude_2_spin),
            ("分振动 2 初相 φ2", self.compound_phase_2_spin),
            ("共同角频率 ω", self.compound_omega_spin),
        ]
        for row, (text, widget) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(widget, row, 1)

        for spinbox in (
            self.compound_amplitude_1_spin,
            self.compound_phase_1_spin,
            self.compound_amplitude_2_spin,
            self.compound_phase_2_spin,
            self.compound_omega_spin,
        ):
            spinbox.valueChanged.connect(lambda _: self.refresh_current_mode(reset_time=False))

        return group

    def _build_diff_freq_param_group(self):
        group = QGroupBox("同方向不同频率合成参数")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(8)

        self.diff_case_combo = QComboBox()
        self.diff_case_combo.setMinimumHeight(36)
        self.diff_case_combo.addItems([self.DIFF_SPECIAL_CASE, self.DIFF_GENERAL_CASE])
        self.diff_case_combo.currentTextChanged.connect(self.on_diff_case_changed)
        layout.addWidget(QLabel("选择当前合成情况"))
        layout.addWidget(self.diff_case_combo)

        self.diff_special_widget = QWidget()
        special_layout = QGridLayout(self.diff_special_widget)
        special_layout.setContentsMargins(0, 4, 0, 0)
        special_layout.setHorizontalSpacing(8)
        special_layout.setVerticalSpacing(8)

        self.diff_special_amplitude_spin = self._create_spinbox(0.01, 1_000_000.0, 20.0, " cm", 3, 1.0)
        self.diff_special_omega_1_spin = self._create_spinbox(0.001, 10_000.0, 3.0, " rad/s", 4, 0.1)
        self.diff_special_omega_2_spin = self._create_spinbox(0.001, 10_000.0, 3.8, " rad/s", 4, 0.1)
        self.diff_special_phase_spin = self._create_compact_phase_input(0.0)

        special_rows = [
            ("共同振幅 A", self.diff_special_amplitude_spin),
            ("分振动 1 角频率 ω1", self.diff_special_omega_1_spin),
            ("分振动 2 角频率 ω2", self.diff_special_omega_2_spin),
            ("共同初相 φ", self.diff_special_phase_spin),
        ]
        for row, (text, widget) in enumerate(special_rows):
            special_layout.addWidget(QLabel(text), row, 0)
            special_layout.addWidget(widget, row, 1)

        self.diff_general_widget = QWidget()
        general_layout = QGridLayout(self.diff_general_widget)
        general_layout.setContentsMargins(0, 4, 0, 0)
        general_layout.setHorizontalSpacing(8)
        general_layout.setVerticalSpacing(8)

        self.diff_general_amplitude_1_spin = self._create_spinbox(0.01, 1_000_000.0, 20.0, " cm", 3, 1.0)
        self.diff_general_phase_1_spin = self._create_compact_phase_input(0.0)
        self.diff_general_omega_1_spin = self._create_spinbox(0.001, 10_000.0, 3.0, " rad/s", 4, 0.1)
        self.diff_general_amplitude_2_spin = self._create_spinbox(0.01, 1_000_000.0, 12.0, " cm", 3, 1.0)
        self.diff_general_phase_2_spin = self._create_compact_phase_input(1.1)
        self.diff_general_omega_2_spin = self._create_spinbox(0.001, 10_000.0, 4.3, " rad/s", 4, 0.1)

        general_rows = [
            ("分振动 1 振幅 A1", self.diff_general_amplitude_1_spin),
            ("分振动 1 初相 φ1", self.diff_general_phase_1_spin),
            ("分振动 1 角频率 ω1", self.diff_general_omega_1_spin),
            ("分振动 2 振幅 A2", self.diff_general_amplitude_2_spin),
            ("分振动 2 初相 φ2", self.diff_general_phase_2_spin),
            ("分振动 2 角频率 ω2", self.diff_general_omega_2_spin),
        ]
        for row, (text, widget) in enumerate(general_rows):
            general_layout.addWidget(QLabel(text), row, 0)
            general_layout.addWidget(widget, row, 1)

        for spinbox in (
            self.diff_special_amplitude_spin,
            self.diff_special_omega_1_spin,
            self.diff_special_omega_2_spin,
            self.diff_special_phase_spin,
            self.diff_general_amplitude_1_spin,
            self.diff_general_phase_1_spin,
            self.diff_general_omega_1_spin,
            self.diff_general_amplitude_2_spin,
            self.diff_general_phase_2_spin,
            self.diff_general_omega_2_spin,
        ):
            spinbox.valueChanged.connect(lambda _: self.refresh_current_mode(reset_time=False))

        layout.addWidget(self.diff_special_widget)
        layout.addWidget(self.diff_general_widget)
        self.on_diff_case_changed(self.DIFF_SPECIAL_CASE)
        return group

    def _build_orthogonal_param_group(self):
        group = QGroupBox("方向垂直同频率合成参数")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.orthogonal_amplitude_x_spin = self._create_spinbox(0.01, 1_000_000.0, 20.0, " cm", 3, 1.0)
        self.orthogonal_phase_x_spin = self._create_compact_phase_input(0.0)
        self.orthogonal_amplitude_y_spin = self._create_spinbox(0.01, 1_000_000.0, 12.0, " cm", 3, 1.0)
        self.orthogonal_phase_y_spin = self._create_compact_phase_input(np.pi / 2)
        self.orthogonal_omega_spin = self._create_spinbox(0.001, 10_000.0, 3.0, " rad/s", 4, 0.1)

        rows = [
            ("x 方向振幅 Ax", self.orthogonal_amplitude_x_spin),
            ("x 方向初相 φx", self.orthogonal_phase_x_spin),
            ("y 方向振幅 Ay", self.orthogonal_amplitude_y_spin),
            ("y 方向初相 φy", self.orthogonal_phase_y_spin),
            ("共同角频率 ω", self.orthogonal_omega_spin),
        ]
        for row, (text, widget) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(widget, row, 1)

        for spinbox in (
            self.orthogonal_amplitude_x_spin,
            self.orthogonal_phase_x_spin,
            self.orthogonal_amplitude_y_spin,
            self.orthogonal_phase_y_spin,
            self.orthogonal_omega_spin,
        ):
            spinbox.valueChanged.connect(lambda _: self.refresh_current_mode(reset_time=False))

        return group

    def _build_orthogonal_diff_param_group(self):
        group = QGroupBox("方向垂直不同频率合成参数")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.orthogonal_diff_preset_combo = QComboBox()
        self.orthogonal_diff_preset_combo.setMinimumHeight(36)
        self.orthogonal_diff_preset_combo.addItems(
            [
                self.ORTHOGONAL_DIFF_MANUAL,
                "1:1 直线",
                "1:1 圆",
                "1:2 李萨如",
                "2:1 李萨如",
                "2:3 李萨如",
                "3:4 李萨如",
            ]
        )
        self.orthogonal_diff_preset_combo.currentTextChanged.connect(self.on_orthogonal_diff_preset_changed)
        layout.addWidget(QLabel("常用预设"), 0, 0)
        layout.addWidget(self.orthogonal_diff_preset_combo, 0, 1)

        self.orthogonal_diff_amplitude_x_spin = self._create_spinbox(0.01, 1_000_000.0, 20.0, " cm", 3, 1.0)
        self.orthogonal_diff_phase_x_spin = self._create_compact_phase_input(0.0)
        self.orthogonal_diff_omega_x_spin = self._create_spinbox(0.001, 10_000.0, 3.0, " rad/s", 4, 0.1)
        self.orthogonal_diff_amplitude_y_spin = self._create_spinbox(0.01, 1_000_000.0, 12.0, " cm", 3, 1.0)
        self.orthogonal_diff_phase_y_spin = self._create_compact_phase_input(1.1)
        self.orthogonal_diff_omega_y_spin = self._create_spinbox(0.001, 10_000.0, 4.0, " rad/s", 4, 0.1)

        rows = [
            ("x 方向振幅 A1", self.orthogonal_diff_amplitude_x_spin),
            ("x 方向初相 φ1", self.orthogonal_diff_phase_x_spin),
            ("x 方向角频率 ω1", self.orthogonal_diff_omega_x_spin),
            ("y 方向振幅 A2", self.orthogonal_diff_amplitude_y_spin),
            ("y 方向初相 φ2", self.orthogonal_diff_phase_y_spin),
            ("y 方向角频率 ω2", self.orthogonal_diff_omega_y_spin),
        ]
        for row, (text, widget) in enumerate(rows):
            layout.addWidget(QLabel(text), row + 1, 0)
            layout.addWidget(widget, row + 1, 1)

        self.orthogonal_diff_spinboxes = (
            self.orthogonal_diff_amplitude_x_spin,
            self.orthogonal_diff_phase_x_spin,
            self.orthogonal_diff_omega_x_spin,
            self.orthogonal_diff_amplitude_y_spin,
            self.orthogonal_diff_phase_y_spin,
            self.orthogonal_diff_omega_y_spin,
        )
        for spinbox in self.orthogonal_diff_spinboxes:
            spinbox.valueChanged.connect(self.on_orthogonal_diff_spinbox_changed)

        return group

    def _build_single_derived_group(self):
        group = QGroupBox("单振动推导量")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.omega_label = QLabel()
        self.period_label = QLabel()
        self.mass_label = QLabel()

        rows = [
            ("角频率", self.omega_label),
            ("周期", self.period_label),
            ("等效质量", self.mass_label),
        ]
        for row, (text, label) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(label, row, 1)

        return group

    def _build_compound_derived_group(self):
        group = QGroupBox("合成推导量")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.compound_omega_label = QLabel()
        self.compound_period_label = QLabel()
        self.compound_resultant_amplitude_label = QLabel()
        self.compound_resultant_phase_label = QLabel()

        rows = [
            ("共同角频率", self.compound_omega_label),
            ("共同周期", self.compound_period_label),
            ("合振幅", self.compound_resultant_amplitude_label),
            ("合初相", self.compound_resultant_phase_label),
        ]
        for row, (text, label) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(label, row, 1)

        return group

    def _build_diff_freq_derived_group(self):
        group = QGroupBox("不同频率合成推导量")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.diff_omega_1_label = QLabel()
        self.diff_omega_2_label = QLabel()
        self.diff_delta_omega_label = QLabel()
        self.diff_feature_label = QLabel()
        self.diff_feature_label.setWordWrap(True)

        rows = [
            ("分振动 1 角频率", self.diff_omega_1_label),
            ("分振动 2 角频率", self.diff_omega_2_label),
            ("角频率差 Δω", self.diff_delta_omega_label),
            ("主要特征", self.diff_feature_label),
        ]
        for row, (text, label) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(label, row, 1)

        return group

    def _build_orthogonal_derived_group(self):
        group = QGroupBox("垂直合成推导量")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.orthogonal_omega_label = QLabel()
        self.orthogonal_period_label = QLabel()
        self.orthogonal_delta_phase_label = QLabel()
        self.orthogonal_type_label = QLabel()
        self.orthogonal_type_label.setWordWrap(True)

        rows = [
            ("共同角频率", self.orthogonal_omega_label),
            ("共同周期", self.orthogonal_period_label),
            ("相位差 Δφ", self.orthogonal_delta_phase_label),
            ("轨迹类型", self.orthogonal_type_label),
        ]
        for row, (text, label) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(label, row, 1)

        return group

    def _build_orthogonal_diff_derived_group(self):
        group = QGroupBox("李萨如图形推导量")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.orthogonal_diff_ratio_label = QLabel()
        self.orthogonal_diff_periodicity_label = QLabel()
        self.orthogonal_diff_common_period_label = QLabel()
        self.orthogonal_diff_type_label = QLabel()
        self.orthogonal_diff_type_label.setWordWrap(True)

        rows = [
            ("频率比 ω1:ω2", self.orthogonal_diff_ratio_label),
            ("轨迹判定", self.orthogonal_diff_periodicity_label),
            ("公共周期", self.orthogonal_diff_common_period_label),
            ("图形特征", self.orthogonal_diff_type_label),
        ]
        for row, (text, label) in enumerate(rows):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(label, row, 1)

        return group

    def _build_control_group(self):
        group = QGroupBox("动画控制")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.btn_toggle_animation = QPushButton("开始动画")
        self.btn_reset_animation = QPushButton("重置")
        self.btn_record_micro_lesson = QPushButton("录制微课")
        self.btn_toggle_animation.clicked.connect(self.on_toggle_animation)
        self.btn_reset_animation.clicked.connect(self.on_reset_animation)
        self.btn_record_micro_lesson.clicked.connect(self.on_record_micro_lesson)
        button_row.addWidget(self.btn_toggle_animation)
        button_row.addWidget(self.btn_reset_animation)
        button_row.addWidget(self.btn_record_micro_lesson)
        layout.addLayout(button_row)

        self.live_status_label = QLabel("t = 0.000 s")
        self.live_status_label.setWordWrap(True)
        self.live_status_label.setStyleSheet(
            "padding: 6px 8px; border-radius: 8px; background-color: #eff6ff; color: #1d4ed8;"
        )
        layout.addWidget(self.live_status_label)
        return group

    def _create_spinbox(self, minimum, maximum, value, suffix, decimals, step):
        spinbox = SliderSpinBox(
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            decimals=decimals,
            suffix=suffix,
        )
        spinbox.setMinimumHeight(36)
        spinbox.setKeyboardTracking(False)
        spinbox.setAccelerated(True)
        return spinbox

    def _create_phase_input(self, value=0.0):
        phase_input = PhaseInputWidget(value=value)
        phase_input.setMinimumHeight(36)
        phase_input.setKeyboardTracking(False)
        phase_input.setAccelerated(True)
        return phase_input

    def _create_compact_phase_input(self, value=0.0):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(-np.pi, np.pi)
        spinbox.setDecimals(4)
        spinbox.setValue(value)
        spinbox.setSingleStep(0.1)
        spinbox.setSuffix(" rad")
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setAccelerated(True)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        spinbox.setMinimumHeight(36)
        spinbox.setMinimumWidth(120)
        spinbox.setMaximumWidth(150)
        return spinbox

    def on_diff_case_changed(self, case_text):
        is_special = case_text == self.DIFF_SPECIAL_CASE
        self.diff_special_widget.setVisible(is_special)
        self.diff_general_widget.setVisible(not is_special)
        if hasattr(self, "mode_combo") and self.mode_combo.currentText() == self.DIFF_FREQ_MODE:
            self.refresh_current_mode(reset_time=True)

    def on_orthogonal_diff_preset_changed(self, preset_text):
        if preset_text == self.ORTHOGONAL_DIFF_MANUAL:
            if hasattr(self, "mode_combo") and self.mode_combo.currentText() == self.ORTHOGONAL_DIFF_MODE:
                self.refresh_current_mode(reset_time=True)
            return

        base_amplitude = max(
            self.orthogonal_diff_amplitude_x_spin.value(),
            self.orthogonal_diff_amplitude_y_spin.value(),
            10.0,
        )
        preset_map = {
            "1:1 直线": {
                "A1": base_amplitude,
                "A2": base_amplitude,
                "omega1": 3.0,
                "omega2": 3.0,
                "phi1": 0.0,
                "phi2": 0.0,
            },
            "1:1 圆": {
                "A1": base_amplitude,
                "A2": base_amplitude,
                "omega1": 3.0,
                "omega2": 3.0,
                "phi1": 0.0,
                "phi2": np.pi / 2,
            },
            "1:2 李萨如": {
                "A1": base_amplitude,
                "A2": base_amplitude,
                "omega1": 2.0,
                "omega2": 4.0,
                "phi1": 0.0,
                "phi2": np.pi / 2,
            },
            "2:1 李萨如": {
                "A1": base_amplitude,
                "A2": base_amplitude,
                "omega1": 4.0,
                "omega2": 2.0,
                "phi1": 0.0,
                "phi2": np.pi / 2,
            },
            "2:3 李萨如": {
                "A1": base_amplitude,
                "A2": base_amplitude,
                "omega1": 4.0,
                "omega2": 6.0,
                "phi1": 0.0,
                "phi2": np.pi / 2,
            },
            "3:4 李萨如": {
                "A1": base_amplitude,
                "A2": base_amplitude,
                "omega1": 6.0,
                "omega2": 8.0,
                "phi1": 0.0,
                "phi2": np.pi / 2,
            },
        }
        preset = preset_map.get(preset_text)
        if preset is None:
            return

        self._updating_orthogonal_diff_preset = True
        blocked_states = []
        for spinbox in self.orthogonal_diff_spinboxes:
            blocked_states.append(spinbox.blockSignals(True))

        self.orthogonal_diff_amplitude_x_spin.setValue(preset["A1"])
        self.orthogonal_diff_amplitude_y_spin.setValue(preset["A2"])
        self.orthogonal_diff_omega_x_spin.setValue(preset["omega1"])
        self.orthogonal_diff_omega_y_spin.setValue(preset["omega2"])
        self.orthogonal_diff_phase_x_spin.setValue(preset["phi1"])
        self.orthogonal_diff_phase_y_spin.setValue(preset["phi2"])

        for spinbox, blocked in zip(self.orthogonal_diff_spinboxes, blocked_states):
            spinbox.blockSignals(blocked)
        self._updating_orthogonal_diff_preset = False

        if hasattr(self, "mode_combo") and self.mode_combo.currentText() == self.ORTHOGONAL_DIFF_MODE:
            self.refresh_current_mode(reset_time=True)

    def on_orthogonal_diff_spinbox_changed(self, _value):
        if not self._updating_orthogonal_diff_preset:
            combo = self.orthogonal_diff_preset_combo
            if combo.currentText() != self.ORTHOGONAL_DIFF_MANUAL:
                was_blocked = combo.blockSignals(True)
                combo.setCurrentText(self.ORTHOGONAL_DIFF_MANUAL)
                combo.blockSignals(was_blocked)
        self.refresh_current_mode(reset_time=False)

    def on_mode_changed(self, mode_text):
        if self.timer.isActive():
            self.timer.stop()
        self.btn_toggle_animation.setText("开始动画")
        self.current_time = 0.0
        self.last_tick = None

        is_single = mode_text == self.SINGLE_MODE
        is_compound = mode_text == self.COMPOUND_MODE
        is_diff_freq = mode_text == self.DIFF_FREQ_MODE
        is_orthogonal = mode_text == self.ORTHOGONAL_MODE
        is_orthogonal_diff = mode_text == self.ORTHOGONAL_DIFF_MODE

        self.single_param_group.setVisible(is_single)
        self.single_derived_group.setVisible(is_single)
        self.single_top_widget.setVisible(is_single)
        self.single_curve_canvas.setVisible(is_single)

        self.compound_param_group.setVisible(is_compound)
        self.compound_derived_group.setVisible(is_compound)
        self.compound_top_widget.setVisible(is_compound)
        self.compound_curve_canvas.setVisible(is_compound)

        self.diff_freq_param_group.setVisible(is_diff_freq)
        self.diff_freq_derived_group.setVisible(is_diff_freq)
        self.diff_freq_top_widget.setVisible(False)
        self.diff_freq_curve_canvas.setVisible(is_diff_freq)
        self.diff_envelope_canvas.setVisible(is_diff_freq)

        self.orthogonal_param_group.setVisible(is_orthogonal)
        self.orthogonal_derived_group.setVisible(is_orthogonal)
        self.orthogonal_top_widget.setVisible(is_orthogonal)
        self.orthogonal_curve_canvas.setVisible(is_orthogonal)

        self.orthogonal_diff_param_group.setVisible(is_orthogonal_diff)
        self.orthogonal_diff_derived_group.setVisible(is_orthogonal_diff)
        self.trajectory_canvas.setVisible(is_orthogonal or is_orthogonal_diff)
        self.orthogonal_summary_label.setVisible(is_orthogonal or is_orthogonal_diff)
        self.orthogonal_top_widget.setVisible(is_orthogonal or is_orthogonal_diff)
        self.orthogonal_curve_canvas.setVisible(is_orthogonal or is_orthogonal_diff)

        if is_single:
            self.subtitle_label.setText("弹簧振子、x/t、v/t、a/t 与旋转矢量同步联动")
            self.display_title.setText("简谐振动仿真")
            self.curve_group.setTitle("x/t、v/t、a/t 曲线")
            self.result_group.setTitle("实验说明")
        elif is_compound:
            self.subtitle_label.setText("输入两个同方向、同频率简谐振动，观察旋转矢量法下的合振动")
            self.display_title.setText("同方向同频率简谐振动合成")
            self.curve_group.setTitle("分振动与合振动 x/t 曲线")
            self.result_group.setTitle("实验说明")
        elif is_diff_freq:
            self.subtitle_label.setText("支持特殊拍频情况与一般情况，显示两个分振动、合振动与包络曲线")
            self.display_title.setText("同方向不同频率简谐振动合成")
            self.curve_group.setTitle("不同频率合成 x/t 曲线")
            self.result_group.setTitle("合成说明")
        elif is_orthogonal:
            self.subtitle_label.setText("显示相互垂直且同频率的两个分振动，以及合振动轨迹")
            self.display_title.setText("方向垂直同频率简谐振动合成")
            self.curve_group.setTitle("垂直分振动曲线")
            self.result_group.setTitle("轨迹说明")
        else:
            self.subtitle_label.setText("显示方向垂直且不同频率的合振动轨迹，重点观察整数频率比的李萨如图形")
            self.display_title.setText("方向垂直不同频率简谐振动合成")
            self.curve_group.setTitle("李萨如图形与分振动曲线")
            self.result_group.setTitle("李萨如说明")

        self.refresh_current_mode(reset_time=True)

    def refresh_current_mode(self, reset_time=False):
        if reset_time:
            self.current_time = 0.0
            self.last_tick = time.monotonic()
        elif self.duration > 0:
            self.current_time %= self.duration

        mode = self.mode_combo.currentText()
        if mode == self.SINGLE_MODE:
            self.update_single_simulation(reset_time=reset_time)
        elif mode == self.COMPOUND_MODE:
            self.update_compound_simulation(reset_time=reset_time)
        elif mode == self.DIFF_FREQ_MODE:
            self.update_diff_freq_simulation(reset_time=reset_time)
        elif mode == self.ORTHOGONAL_MODE:
            self.update_orthogonal_simulation(reset_time=reset_time)
        else:
            self.update_orthogonal_diff_simulation(reset_time=reset_time)

    def _get_diff_freq_parameters(self):
        if self.diff_case_combo.currentText() == self.DIFF_SPECIAL_CASE:
            amplitude = self.diff_special_amplitude_spin.value()
            phase = self.diff_special_phase_spin.value()
            return {
                "case": "special",
                "A1": amplitude,
                "A2": amplitude,
                "phi1": phase,
                "phi2": phase,
                "omega1": max(self.diff_special_omega_1_spin.value(), 1e-9),
                "omega2": max(self.diff_special_omega_2_spin.value(), 1e-9),
            }

        return {
            "case": "general",
            "A1": self.diff_general_amplitude_1_spin.value(),
            "A2": self.diff_general_amplitude_2_spin.value(),
            "phi1": self.diff_general_phase_1_spin.value(),
            "phi2": self.diff_general_phase_2_spin.value(),
            "omega1": max(self.diff_general_omega_1_spin.value(), 1e-9),
            "omega2": max(self.diff_general_omega_2_spin.value(), 1e-9),
        }

    def _estimate_diff_duration(self, omega_1, omega_2, show_relative_cycle):
        omega_min = max(min(omega_1, omega_2), 1e-9)
        duration = max(4.0, 3.0 * (2 * np.pi / omega_min))
        delta_omega = abs(omega_2 - omega_1)
        if show_relative_cycle and delta_omega > 1e-9:
            duration = max(duration, 2 * np.pi / delta_omega)
        return duration

    def _calculate_diff_freq_motion(self, times):
        params = self._get_diff_freq_parameters()
        theta_1 = params["omega1"] * times + params["phi1"]
        theta_2 = params["omega2"] * times + params["phi2"]
        x1_values = params["A1"] * np.cos(theta_1)
        x2_values = params["A2"] * np.cos(theta_2)
        return x1_values, x2_values, x1_values + x2_values

    def _calculate_diff_freq_envelope(self, times):
        params = self._get_diff_freq_parameters()
        phase_gap = (params["omega2"] - params["omega1"]) * times + (params["phi2"] - params["phi1"])
        envelope_squared = (
            params["A1"] ** 2
            + params["A2"] ** 2
            + 2.0 * params["A1"] * params["A2"] * np.cos(phase_gap)
        )
        return np.sqrt(np.clip(envelope_squared, 0.0, None))

    def _get_orthogonal_parameters(self):
        return {
            "Ax": self.orthogonal_amplitude_x_spin.value(),
            "Ay": self.orthogonal_amplitude_y_spin.value(),
            "phi_x": self.orthogonal_phase_x_spin.value(),
            "phi_y": self.orthogonal_phase_y_spin.value(),
            "omega": max(self.orthogonal_omega_spin.value(), 1e-9),
        }

    def _get_orthogonal_diff_parameters(self):
        return {
            "A1": self.orthogonal_diff_amplitude_x_spin.value(),
            "A2": self.orthogonal_diff_amplitude_y_spin.value(),
            "phi1": self.orthogonal_diff_phase_x_spin.value(),
            "phi2": self.orthogonal_diff_phase_y_spin.value(),
            "omega1": max(self.orthogonal_diff_omega_x_spin.value(), 1e-9),
            "omega2": max(self.orthogonal_diff_omega_y_spin.value(), 1e-9),
        }

    def _normalize_phase(self, phase_rad):
        return float((phase_rad + np.pi) % (2 * np.pi) - np.pi)

    def _classify_orthogonal_trajectory(self, amplitude_x, amplitude_y, delta_phase):
        if min(amplitude_x, amplitude_y) <= 1e-9:
            return "直线轨迹"

        delta = abs(self._normalize_phase(delta_phase))
        amplitude_ratio = max(amplitude_x, amplitude_y) / max(min(amplitude_x, amplitude_y), 1e-9)
        phase_tolerance = 0.08

        if delta < phase_tolerance or abs(delta - np.pi) < phase_tolerance:
            return "直线轨迹"
        if abs(delta - np.pi / 2) < phase_tolerance and abs(amplitude_ratio - 1.0) < 0.06:
            return "圆轨迹"
        return "椭圆轨迹"

    def _calculate_orthogonal_motion(self, times):
        params = self._get_orthogonal_parameters()
        theta_x = params["omega"] * times + params["phi_x"]
        theta_y = params["omega"] * times + params["phi_y"]
        x_values = params["Ax"] * np.cos(theta_x)
        y_values = params["Ay"] * np.cos(theta_y)
        return x_values, y_values

    def _analyze_frequency_ratio(self, omega_1, omega_2, max_denominator=12):
        ratio = omega_1 / max(omega_2, 1e-9)
        fraction = Fraction(ratio).limit_denominator(max_denominator)
        approx_ratio = fraction.numerator / fraction.denominator
        relative_error = abs(ratio - approx_ratio) / max(abs(ratio), 1e-9)
        is_closed = relative_error < 1e-3
        return {
            "ratio": ratio,
            "fraction": fraction,
            "approx_ratio": approx_ratio,
            "is_closed": is_closed,
            "relative_error": relative_error,
        }

    def _estimate_orthogonal_diff_duration(self, params, ratio_info):
        if ratio_info["is_closed"]:
            common_period = 2 * np.pi * ratio_info["fraction"].denominator / params["omega2"]
            return max(common_period, 2.0), common_period

        slow_period = 2 * np.pi / min(params["omega1"], params["omega2"])
        return max(6.0 * slow_period, 8.0), None

    def _calculate_orthogonal_diff_motion(self, times):
        params = self._get_orthogonal_diff_parameters()
        theta_1 = params["omega1"] * times + params["phi1"]
        theta_2 = params["omega2"] * times + params["phi2"]
        x_values = params["A1"] * np.cos(theta_1)
        y_values = params["A2"] * np.cos(theta_2)
        return x_values, y_values

    def _calculate_single_motion(self, times):
        amplitude = self.amplitude_spin.value()
        omega = self.omega_spin.value()
        phase = self.phase_spin.value()
        theta = omega * times + phase
        x_values = amplitude * np.cos(theta)
        v_values = -amplitude * omega * np.sin(theta)
        a_values = -amplitude * (omega ** 2) * np.cos(theta)
        return x_values, v_values, a_values

    def _calculate_compound_motion(self, times):
        amplitude_1 = self.compound_amplitude_1_spin.value()
        phase_1 = self.compound_phase_1_spin.value()
        amplitude_2 = self.compound_amplitude_2_spin.value()
        phase_2 = self.compound_phase_2_spin.value()
        omega = self.compound_omega_spin.value()

        theta_1 = omega * times + phase_1
        theta_2 = omega * times + phase_2
        x1_values = amplitude_1 * np.cos(theta_1)
        x2_values = amplitude_2 * np.cos(theta_2)
        return x1_values, x2_values, x1_values + x2_values

    def _calculate_compound_resultant(self):
        vector = (
            self.compound_amplitude_1_spin.value() * np.exp(1j * self.compound_phase_1_spin.value())
            + self.compound_amplitude_2_spin.value() * np.exp(1j * self.compound_phase_2_spin.value())
        )
        amplitude = float(abs(vector))
        phase = float(np.angle(vector)) if amplitude > 1e-12 else 0.0
        return amplitude, phase

    def update_single_simulation(self, reset_time=False):
        amplitude = self.amplitude_spin.value()
        omega = max(self.omega_spin.value(), 1e-9)
        spring = self.spring_spin.value()
        phase = self.phase_spin.value()
        period = 2 * np.pi / omega
        frequency = omega / (2 * np.pi)
        self.duration = max(2.0, 3.0 * period)
        self.t_values = np.linspace(0.0, self.duration, 1400)
        self.x_values, self.v_values, self.a_values = self._calculate_single_motion(self.t_values)
        self.current_time = 0.0 if reset_time else float(np.clip(self.current_time, 0.0, self.duration))

        equivalent_mass = spring / (omega ** 2)
        self.omega_label.setText(f"{omega:.4f} rad/s")
        self.period_label.setText(f"{period:.4f} s")
        self.mass_label.setText(f"{equivalent_mass:.4f} kg")

        self.single_curve_canvas.set_series(self.t_values, [self.x_values, self.v_values, self.a_values])
        self.spring_widget.set_state(amplitude, amplitude * np.cos(phase))
        self.phasor_widget.set_state(amplitude, phase, omega)
        self.update_single_live_widgets()
        self.update_single_result_summary(amplitude, omega, phase, spring, period, frequency, equivalent_mass)

    def update_compound_simulation(self, reset_time=False):
        omega = max(self.compound_omega_spin.value(), 1e-9)
        period = 2 * np.pi / omega
        frequency = omega / (2 * np.pi)
        self.duration = max(2.0, 3.0 * period)
        self.t_values = np.linspace(0.0, self.duration, 1400)
        self.x1_values, self.x2_values, self.x_sum_values = self._calculate_compound_motion(self.t_values)
        self.current_time = 0.0 if reset_time else float(np.clip(self.current_time, 0.0, self.duration))

        resultant_amplitude, resultant_phase = self._calculate_compound_resultant()
        self.compound_omega_label.setText(f"{omega:.4f} rad/s")
        self.compound_period_label.setText(f"{period:.4f} s")
        self.compound_resultant_amplitude_label.setText(f"{resultant_amplitude:.4f} cm")
        self.compound_resultant_phase_label.setText(
            f"{resultant_phase:.4f} rad / {np.degrees(resultant_phase):.2f}°"
        )

        self.compound_curve_canvas.set_series(
            self.t_values,
            [self.x1_values, self.x2_values, self.x_sum_values],
        )
        self.update_compound_live_widgets()
        self.update_compound_result_summary(resultant_amplitude, resultant_phase, omega, period, frequency)

    def update_diff_freq_simulation(self, reset_time=False):
        params = self._get_diff_freq_parameters()
        omega_1 = params["omega1"]
        omega_2 = params["omega2"]
        delta_omega = abs(omega_2 - omega_1)
        self.duration = self._estimate_diff_duration(omega_1, omega_2, show_relative_cycle=delta_omega > 1e-9)
        self.t_values = np.linspace(0.0, self.duration, 1600)
        self.x1_values, self.x2_values, self.x_sum_values = self._calculate_diff_freq_motion(self.t_values)
        self.diff_envelope_values = self._calculate_diff_freq_envelope(self.t_values)
        self.current_time = 0.0 if reset_time else float(np.clip(self.current_time, 0.0, self.duration))

        self.diff_omega_1_label.setText(f"{omega_1:.4f} rad/s")
        self.diff_omega_2_label.setText(f"{omega_2:.4f} rad/s")
        self.diff_delta_omega_label.setText(f"{delta_omega:.4f} rad/s")
        if params["case"] == "special":
            if delta_omega > 1e-9:
                beat_period = 2 * np.pi / delta_omega
                feature_text = f"出现拍频现象，包络周期 T_beat = {beat_period:.4f} s"
            else:
                feature_text = "退化为同频同相合成，合振幅为 2A"
        else:
            feature_text = (
                f"瞬时相位差持续变化，合位移理论范围不超过 ±{params['A1'] + params['A2']:.4f} cm"
            )
        self.diff_feature_label.setText(feature_text)

        self.diff_freq_curve_canvas.set_series(
            self.t_values,
            [self.x1_values, self.x2_values, self.x_sum_values],
        )
        envelope_title = "合振动与拍频包络" if params["case"] == "special" else "合振动与理论包络"
        self.diff_envelope_canvas.set_series(
            self.t_values,
            self.x_sum_values,
            self.diff_envelope_values,
            envelope_title,
        )
        self.update_diff_freq_live_widgets()
        self.update_diff_freq_result_summary(params, delta_omega)

    def update_single_live_widgets(self):
        amplitude = self.amplitude_spin.value()
        omega = self.omega_spin.value()
        phase = self.phase_spin.value()
        theta = omega * self.current_time + phase
        x_value = amplitude * np.cos(theta)
        v_value = -amplitude * omega * np.sin(theta)
        a_value = -amplitude * (omega ** 2) * np.cos(theta)

        self.spring_widget.set_state(amplitude, x_value)
        self.phasor_widget.set_state(amplitude, theta, omega)
        self.single_curve_canvas.update_current_time(self.current_time)
        self.live_status_label.setText(
            f"t = {self.current_time:.3f} s | x = {x_value:.3f} cm | "
            f"v = {v_value:.3f} cm/s | a = {a_value:.3f} cm/s^2"
        )

    def update_compound_live_widgets(self):
        amplitude_1 = self.compound_amplitude_1_spin.value()
        phase_1_initial = self.compound_phase_1_spin.value()
        amplitude_2 = self.compound_amplitude_2_spin.value()
        phase_2_initial = self.compound_phase_2_spin.value()
        omega = self.compound_omega_spin.value()

        phase_1 = phase_1_initial + omega * self.current_time
        phase_2 = phase_2_initial + omega * self.current_time
        x1_value = amplitude_1 * np.cos(phase_1)
        x2_value = amplitude_2 * np.cos(phase_2)
        x_sum_value = x1_value + x2_value
        resultant_amplitude, resultant_phase = self._calculate_compound_resultant()
        current_resultant_phase = resultant_phase + omega * self.current_time

        self.compound_phasor_widget.set_state(amplitude_1, amplitude_2, phase_1, phase_2)
        self.compound_curve_canvas.update_current_time(self.current_time)
        self.compound_summary_label.setText(
            (
                "<div style='font-size:12px;'>"
                "<div style='font-weight:600; color:#0f172a; margin-bottom:6px;'>当前合成状态</div>"
                f"<div><span style='color:#2f80ed;'>A1</span> = {amplitude_1:.3f} cm, "
                f"φ1 = {phase_1_initial:.3f} rad</div>"
                f"<div><span style='color:#27ae60;'>A2</span> = {amplitude_2:.3f} cm, "
                f"φ2 = {phase_2_initial:.3f} rad</div>"
                f"<div><span style='color:#e67e22;'>A合</span> = {resultant_amplitude:.3f} cm, "
                f"φ合 = {resultant_phase:.3f} rad</div>"
                f"<div style='margin-top:8px;'>Δφ = {(phase_2_initial - phase_1_initial):.3f} rad</div>"
                f"<div>θ1 = {phase_1:.3f} rad, θ2 = {phase_2:.3f} rad</div>"
                f"<div>θ合 = {current_resultant_phase:.3f} rad</div>"
                f"<div style='margin-top:8px;'>x1 = {x1_value:.3f} cm</div>"
                f"<div>x2 = {x2_value:.3f} cm</div>"
                f"<div style='font-weight:600; color:#0f172a;'>x = {x_sum_value:.3f} cm</div>"
                "</div>"
            )
        )
        self.live_status_label.setText(
            f"t = {self.current_time:.3f} s | x1 = {x1_value:.3f} cm | "
            f"x2 = {x2_value:.3f} cm | x = {x_sum_value:.3f} cm"
        )

    def update_diff_freq_live_widgets(self):
        params = self._get_diff_freq_parameters()
        theta_1 = params["phi1"] + params["omega1"] * self.current_time
        theta_2 = params["phi2"] + params["omega2"] * self.current_time
        x1_value = params["A1"] * np.cos(theta_1)
        x2_value = params["A2"] * np.cos(theta_2)
        x_sum_value = x1_value + x2_value
        delta_omega = abs(params["omega2"] - params["omega1"])
        delta_theta = theta_2 - theta_1
        current_envelope = float(np.interp(self.current_time, self.t_values, self.diff_envelope_values))

        self.diff_freq_phasor_widget.set_state(params["A1"], params["A2"], theta_1, theta_2)
        self.diff_freq_curve_canvas.update_current_time(self.current_time)
        self.diff_envelope_canvas.update_current_time(self.current_time)

        if params["case"] == "special":
            heading = "特殊情况：同振幅、同初相、不同角频率"
            if delta_omega > 1e-9:
                beat_period = 2 * np.pi / delta_omega
                envelope = 2 * params["A1"] * np.cos(0.5 * delta_omega * self.current_time)
                relation_line = "x = 2A cos(Δω t / 2) cos(((ω1 + ω2) t) / 2 + φ)"
                extra_line = (
                    f"当前包络 |Aenv| = {abs(envelope):.3f} cm，包络图已单独绘出，拍周期 T_beat = {beat_period:.3f} s"
                )
            else:
                relation_line = "ω1 = ω2 时退化为 x = 2A cos(ωt + φ)"
                extra_line = "Δω = 0.000 rad/s"
        else:
            heading = "一般情况：不同振幅、不同角频率、不同初相"
            relation_line = "x(t) = A1 cos(ω1 t + φ1) + A2 cos(ω2 t + φ2)"
            extra_line = (
                f"当前相位差 Δθ = {delta_theta:.3f} rad，理论最大振幅不超过 {params['A1'] + params['A2']:.3f} cm"
            )

        self.diff_freq_summary_label.setText(
            (
                "<div style='font-size:12px;'>"
                f"<div style='font-weight:600; color:#0f172a; margin-bottom:6px;'>{heading}</div>"
                f"<div>A1 = {params['A1']:.3f} cm，ω1 = {params['omega1']:.3f} rad/s，φ1 = {params['phi1']:.3f} rad</div>"
                f"<div>A2 = {params['A2']:.3f} cm，ω2 = {params['omega2']:.3f} rad/s，φ2 = {params['phi2']:.3f} rad</div>"
                f"<div style='margin-top:8px;'>θ1 = {theta_1:.3f} rad，θ2 = {theta_2:.3f} rad</div>"
                f"<div>x1 = {x1_value:.3f} cm，x2 = {x2_value:.3f} cm</div>"
                f"<div style='font-weight:600; color:#0f172a;'>x = {x_sum_value:.3f} cm</div>"
                f"<div>当前包络 Aenv = {current_envelope:.3f} cm</div>"
                f"<div style='margin-top:8px;'>{relation_line}</div>"
                f"<div>{extra_line}</div>"
                "</div>"
            )
        )
        self.live_status_label.setText(
            f"t = {self.current_time:.3f} s | x1 = {x1_value:.3f} cm | "
            f"x2 = {x2_value:.3f} cm | x = {x_sum_value:.3f} cm | Aenv = {current_envelope:.3f} cm"
        )

    def update_diff_freq_result_summary(self, params, delta_omega):
        frequency_1 = params["omega1"] / (2 * np.pi)
        frequency_2 = params["omega2"] / (2 * np.pi)
        average_omega = 0.5 * (params["omega1"] + params["omega2"])
        phase_diff_0 = params["phi2"] - params["phi1"]

        if params["case"] == "special":
            if delta_omega > 1e-9:
                beat_period_text = f"拍周期 T_beat = {2 * np.pi / delta_omega:.4f} s\n"
            else:
                beat_period_text = "ω1 = ω2 时退化为同频同相合成\n"
            text = (
                "模型:\n"
                "x1(t) = A cos(ω1 t + φ)\n"
                "x2(t) = A cos(ω2 t + φ)\n"
                "x(t) = x1(t) + x2(t)\n"
                "     = 2A cos(Δω t / 2) cos(((ω1 + ω2) t) / 2 + φ)\n\n"
                f"输入参数:\n"
                f"A = {params['A1']:.4f} cm\n"
                f"ω1 = {params['omega1']:.4f} rad/s, f1 = {frequency_1:.4f} Hz\n"
                f"ω2 = {params['omega2']:.4f} rad/s, f2 = {frequency_2:.4f} Hz\n"
                f"φ = {params['phi1']:.4f} rad ({np.degrees(params['phi1']):.2f}°)\n\n"
                f"推导结果:\n"
                f"平均角频率 ωavg = {average_omega:.4f} rad/s\n"
                f"角频率差 Δω = {delta_omega:.4f} rad/s\n"
                f"{beat_period_text}\n"
                "图像说明:\n"
                "下方三张曲线分别给出两个分振动和合振动的位移随时间变化。\n"
                "新增的包络图会单独绘出合振动与上下包络线。\n"
                "当 ω1 与 ω2 接近时，合振动会出现明显拍频，包络由 cos(Δω t / 2) 控制。"
            )
        else:
            text = (
                "模型:\n"
                "x1(t) = A1 cos(ω1 t + φ1)\n"
                "x2(t) = A2 cos(ω2 t + φ2)\n"
                "x(t) = x1(t) + x2(t)\n\n"
                f"输入参数:\n"
                f"A1 = {params['A1']:.4f} cm, ω1 = {params['omega1']:.4f} rad/s, φ1 = {params['phi1']:.4f} rad ({np.degrees(params['phi1']):.2f}°)\n"
                f"A2 = {params['A2']:.4f} cm, ω2 = {params['omega2']:.4f} rad/s, φ2 = {params['phi2']:.4f} rad ({np.degrees(params['phi2']):.2f}°)\n\n"
                f"推导结果:\n"
                f"f1 = {frequency_1:.4f} Hz, f2 = {frequency_2:.4f} Hz\n"
                f"初始相位差 Δφ0 = {phase_diff_0:.4f} rad ({np.degrees(phase_diff_0):.2f}°)\n"
                f"角频率差 Δω = {delta_omega:.4f} rad/s\n"
                f"合位移理论最大值不超过 ±{params['A1'] + params['A2']:.4f} cm\n"
                f"最小包络可接近 {abs(params['A1'] - params['A2']):.4f} cm\n\n"
                "图像说明:\n"
                "下方三张曲线分别显示两个分振动和合振动的 x/t 变化。\n"
                "底部新增的包络图给出合振动的上下理论包络线。\n"
                "由于振幅、角频率和初相都可以不同，合振动通常表现为非固定振幅的复杂调制。"
            )
        self.result_text.setPlainText(text)

    def update_orthogonal_simulation(self, reset_time=False):
        params = self._get_orthogonal_parameters()
        omega = params["omega"]
        period = 2 * np.pi / omega
        frequency = omega / (2 * np.pi)
        self.duration = max(2.0, 3.0 * period)
        self.t_values = np.linspace(0.0, self.duration, 1400)
        self.orth_x_values, self.orth_y_values = self._calculate_orthogonal_motion(self.t_values)
        self.current_time = 0.0 if reset_time else float(np.clip(self.current_time, 0.0, self.duration))

        delta_phase = self._normalize_phase(params["phi_y"] - params["phi_x"])
        trajectory_type = self._classify_orthogonal_trajectory(params["Ax"], params["Ay"], delta_phase)

        self.orthogonal_omega_label.setText(f"{omega:.4f} rad/s")
        self.orthogonal_period_label.setText(f"{period:.4f} s")
        self.orthogonal_delta_phase_label.setText(f"{delta_phase:.4f} rad / {np.degrees(delta_phase):.2f}°")
        self.orthogonal_type_label.setText(trajectory_type)

        self.orthogonal_curve_canvas.set_series(self.t_values, [self.orth_x_values, self.orth_y_values])
        self.trajectory_canvas.set_series(
            self.t_values,
            self.orth_x_values,
            self.orth_y_values,
            f"合振动轨迹：{trajectory_type}",
        )
        self.update_orthogonal_live_widgets()
        self.update_orthogonal_result_summary(params, delta_phase, trajectory_type, period, frequency)

    def update_orthogonal_live_widgets(self):
        params = self._get_orthogonal_parameters()
        theta_x = params["omega"] * self.current_time + params["phi_x"]
        theta_y = params["omega"] * self.current_time + params["phi_y"]
        x_value = params["Ax"] * np.cos(theta_x)
        y_value = params["Ay"] * np.cos(theta_y)
        delta_phase = self._normalize_phase(params["phi_y"] - params["phi_x"])
        trajectory_type = self._classify_orthogonal_trajectory(params["Ax"], params["Ay"], delta_phase)

        self.orthogonal_curve_canvas.update_current_time(self.current_time)
        self.trajectory_canvas.update_current_time(self.current_time)
        self.orthogonal_summary_label.setText(
            (
                "<div style='font-size:12px;'>"
                "<div style='font-weight:600; color:#0f172a; margin-bottom:6px;'>当前轨迹状态</div>"
                f"<div>Ax = {params['Ax']:.3f} cm，Ay = {params['Ay']:.3f} cm</div>"
                f"<div>φx = {params['phi_x']:.3f} rad，φy = {params['phi_y']:.3f} rad</div>"
                f"<div>Δφ = {delta_phase:.3f} rad ({np.degrees(delta_phase):.1f}°)</div>"
                f"<div>ω = {params['omega']:.3f} rad/s</div>"
                f"<div style='margin-top:8px;'>x = {x_value:.3f} cm</div>"
                f"<div>y = {y_value:.3f} cm</div>"
                f"<div style='font-weight:600; color:#0f172a; margin-top:8px;'>轨迹类型：{trajectory_type}</div>"
                "</div>"
            )
        )
        self.live_status_label.setText(
            f"t = {self.current_time:.3f} s | x = {x_value:.3f} cm | y = {y_value:.3f} cm | 轨迹 = {trajectory_type}"
        )

    def update_orthogonal_result_summary(self, params, delta_phase, trajectory_type, period, frequency):
        if trajectory_type == "直线轨迹":
            condition_text = "当相位差 Δφ 接近 0 或 π 时，轨迹退化为直线。"
        elif trajectory_type == "圆轨迹":
            condition_text = "当 Ax ≈ Ay 且 Δφ 接近 ±π/2 时，轨迹为圆。"
        else:
            condition_text = "当相位差既不为 0/π，又不满足圆轨迹条件时，一般表现为椭圆。"

        text = (
            "模型:\n"
            "x(t) = Ax cos(ωt + φx)\n"
            "y(t) = Ay cos(ωt + φy)\n\n"
            "消去时间后的轨迹方程:\n"
            "(x / Ax)^2 + (y / Ay)^2 - 2xy cos(Δφ) / (Ax Ay) = sin^2(Δφ)\n\n"
            f"输入参数:\n"
            f"Ax = {params['Ax']:.4f} cm, φx = {params['phi_x']:.4f} rad ({np.degrees(params['phi_x']):.2f}°)\n"
            f"Ay = {params['Ay']:.4f} cm, φy = {params['phi_y']:.4f} rad ({np.degrees(params['phi_y']):.2f}°)\n"
            f"ω = {params['omega']:.4f} rad/s, f = {frequency:.4f} Hz, T = {period:.4f} s\n\n"
            f"推导结果:\n"
            f"Δφ = {delta_phase:.4f} rad ({np.degrees(delta_phase):.2f}°)\n"
            f"轨迹类型 = {trajectory_type}\n"
            f"{condition_text}\n\n"
            "图像说明:\n"
            "右上图显示合振动在 x-y 平面上的完整轨迹与当前质点位置。\n"
            "下方两张曲线分别显示 x 方向和 y 方向的分振动随时间变化。\n"
            "通过调整振幅和相位差，可以观察直线、椭圆与圆轨迹之间的变化。"
        )
        self.result_text.setPlainText(text)

    def update_orthogonal_diff_simulation(self, reset_time=False):
        params = self._get_orthogonal_diff_parameters()
        ratio_info = self._analyze_frequency_ratio(params["omega1"], params["omega2"])
        self.duration, common_period = self._estimate_orthogonal_diff_duration(params, ratio_info)
        self.t_values = np.linspace(0.0, self.duration, 1800)
        self.orth_x_values, self.orth_y_values = self._calculate_orthogonal_diff_motion(self.t_values)
        self.current_time = 0.0 if reset_time else float(np.clip(self.current_time, 0.0, self.duration))

        delta_phase = self._normalize_phase(params["phi2"] - params["phi1"])
        ratio_text = f"{ratio_info['fraction'].numerator}:{ratio_info['fraction'].denominator}"
        if ratio_info["is_closed"]:
            if ratio_info["fraction"].numerator == ratio_info["fraction"].denominator:
                periodicity_text = "退化为同频率闭合轨迹"
                feature_text = "频率比为 1:1，可进一步退化为直线、椭圆或圆"
                trajectory_type = "同频率闭合轨迹"
            else:
                periodicity_text = "闭合李萨如图形"
                feature_text = f"整数频率比 {ratio_text}，轨迹在一个公共周期后闭合"
                trajectory_type = f"李萨如图形 {ratio_text}"
        else:
            periodicity_text = "准周期轨迹"
            feature_text = "频率比不是简单整数比，轨迹长期不重复，只在矩形区域内密铺"
            trajectory_type = "准周期轨迹"

        self.orthogonal_diff_ratio_label.setText(
            f"{params['omega1']:.4f}:{params['omega2']:.4f} ≈ {ratio_text}"
        )
        self.orthogonal_diff_periodicity_label.setText(periodicity_text)
        self.orthogonal_diff_common_period_label.setText(
            f"{common_period:.4f} s" if common_period is not None else "无有限公共周期"
        )
        self.orthogonal_diff_type_label.setText(feature_text)

        self.orthogonal_curve_canvas.set_series(self.t_values, [self.orth_x_values, self.orth_y_values])
        self.trajectory_canvas.set_series(
            self.t_values,
            self.orth_x_values,
            self.orth_y_values,
            f"轨迹图：{trajectory_type}",
        )
        self.update_orthogonal_diff_live_widgets()
        self.update_orthogonal_diff_result_summary(
            params,
            delta_phase,
            ratio_info,
            periodicity_text,
            trajectory_type,
            common_period,
        )

    def update_orthogonal_diff_live_widgets(self):
        params = self._get_orthogonal_diff_parameters()
        theta_1 = params["omega1"] * self.current_time + params["phi1"]
        theta_2 = params["omega2"] * self.current_time + params["phi2"]
        x_value = params["A1"] * np.cos(theta_1)
        y_value = params["A2"] * np.cos(theta_2)
        delta_phase = self._normalize_phase(params["phi2"] - params["phi1"])
        ratio_info = self._analyze_frequency_ratio(params["omega1"], params["omega2"])
        ratio_text = f"{ratio_info['fraction'].numerator}:{ratio_info['fraction'].denominator}"

        if ratio_info["is_closed"]:
            trajectory_type = "闭合李萨如图形" if ratio_text != "1:1" else "同频率闭合轨迹"
            ratio_line = f"频率比 ≈ {ratio_text}，轨迹闭合"
        else:
            trajectory_type = "准周期轨迹"
            ratio_line = f"频率比 ≈ {ratio_text}，轨迹不闭合"

        self.orthogonal_curve_canvas.update_current_time(self.current_time)
        self.trajectory_canvas.update_current_time(self.current_time)
        self.orthogonal_summary_label.setText(
            (
                "<div style='font-size:12px;'>"
                "<div style='font-weight:600; color:#0f172a; margin-bottom:6px;'>当前李萨如状态</div>"
                f"<div>A1 = {params['A1']:.3f} cm，ω1 = {params['omega1']:.3f} rad/s，φ1 = {params['phi1']:.3f} rad</div>"
                f"<div>A2 = {params['A2']:.3f} cm，ω2 = {params['omega2']:.3f} rad/s，φ2 = {params['phi2']:.3f} rad</div>"
                f"<div>Δφ = {delta_phase:.3f} rad ({np.degrees(delta_phase):.1f}°)</div>"
                f"<div style='margin-top:8px;'>x = {x_value:.3f} cm</div>"
                f"<div>y = {y_value:.3f} cm</div>"
                f"<div style='margin-top:8px;'>{ratio_line}</div>"
                f"<div style='font-weight:600; color:#0f172a;'>图形类型：{trajectory_type}</div>"
                "</div>"
            )
        )
        self.live_status_label.setText(
            f"t = {self.current_time:.3f} s | x = {x_value:.3f} cm | y = {y_value:.3f} cm | {trajectory_type}"
        )

    def update_orthogonal_diff_result_summary(
        self,
        params,
        delta_phase,
        ratio_info,
        periodicity_text,
        trajectory_type,
        common_period,
    ):
        frequency_1 = params["omega1"] / (2 * np.pi)
        frequency_2 = params["omega2"] / (2 * np.pi)
        ratio_text = f"{ratio_info['fraction'].numerator}:{ratio_info['fraction'].denominator}"

        if ratio_info["is_closed"]:
            period_line = f"公共周期 T = {common_period:.4f} s\n" if common_period is not None else ""
            feature_text = (
                "当两个角频率之比为整数比时，轨迹是闭合的，运动是周期性的，这就是李萨如图形。"
            )
        else:
            period_line = "不存在有限公共周期\n"
            feature_text = (
                "当两个角频率之比不是简单整数比时，轨迹一般不闭合，表现为准周期运动。"
            )

        text = (
            "模型:\n"
            "x(t) = A1 cos(ω1 t + φ1)\n"
            "y(t) = A2 cos(ω2 t + φ2)\n\n"
            f"输入参数:\n"
            f"A1 = {params['A1']:.4f} cm, ω1 = {params['omega1']:.4f} rad/s, φ1 = {params['phi1']:.4f} rad ({np.degrees(params['phi1']):.2f}°)\n"
            f"A2 = {params['A2']:.4f} cm, ω2 = {params['omega2']:.4f} rad/s, φ2 = {params['phi2']:.4f} rad ({np.degrees(params['phi2']):.2f}°)\n\n"
            f"推导结果:\n"
            f"f1 = {frequency_1:.4f} Hz, f2 = {frequency_2:.4f} Hz\n"
            f"Δφ = {delta_phase:.4f} rad ({np.degrees(delta_phase):.2f}°)\n"
            f"频率比 ω1:ω2 ≈ {ratio_text}\n"
            f"轨迹判定 = {periodicity_text}\n"
            f"{period_line}"
            f"图形类型 = {trajectory_type}\n\n"
            "图像说明:\n"
            "右上图显示合振动在 x-y 平面中的轨迹，这就是李萨如图形或其准周期推广。\n"
            "下方两张曲线分别显示 x、y 两个方向上的分振动随时间变化。\n"
            f"{feature_text}"
        )
        self.result_text.setPlainText(text)

    def update_single_result_summary(
        self,
        amplitude,
        omega,
        phase,
        spring,
        period,
        frequency,
        equivalent_mass,
    ):
        text = (
            "模型:\n"
            "x(t) = A cos(ωt + φ)\n"
            "v(t) = -Aω sin(ωt + φ)\n"
            "a(t) = -Aω^2 cos(ωt + φ)\n\n"
            f"当前参数:\n"
            f"A = {amplitude:.4f} cm\n"
            f"ω = {omega:.4f} rad/s\n"
            f"φ = {phase:.4f} rad ({np.degrees(phase):.2f}°)\n"
            f"k = {spring:.4f} N/m\n\n"
            f"推导结果:\n"
            f"f = {frequency:.4f} Hz\n"
            f"T = {period:.4f} s\n"
            f"m = k / ω^2 = {equivalent_mass:.4f} kg\n\n"
            "图像说明:\n"
            "左上区域展示弹簧振子的实时位移，右上区域展示等价旋转矢量，"
            "下方三张图分别给出位移、速度和加速度随时间的变化。"
        )
        self.result_text.setPlainText(text)

    def update_compound_result_summary(self, resultant_amplitude, resultant_phase, omega, period, frequency):
        amplitude_1 = self.compound_amplitude_1_spin.value()
        phase_1 = self.compound_phase_1_spin.value()
        amplitude_2 = self.compound_amplitude_2_spin.value()
        phase_2 = self.compound_phase_2_spin.value()
        delta_phase = phase_2 - phase_1

        text = (
            "模型:\n"
            "x1(t) = A1 cos(ωt + φ1)\n"
            "x2(t) = A2 cos(ωt + φ2)\n"
            "x(t) = x1(t) + x2(t) = A cos(ωt + φ)\n\n"
            f"输入参数:\n"
            f"A1 = {amplitude_1:.4f} cm, φ1 = {phase_1:.4f} rad ({np.degrees(phase_1):.2f}°)\n"
            f"A2 = {amplitude_2:.4f} cm, φ2 = {phase_2:.4f} rad ({np.degrees(phase_2):.2f}°)\n"
            f"ω = {omega:.4f} rad/s\n\n"
            f"合成结果:\n"
            f"Δφ = {delta_phase:.4f} rad ({np.degrees(delta_phase):.2f}°)\n"
            f"A = {resultant_amplitude:.4f} cm\n"
            f"φ = {resultant_phase:.4f} rad ({np.degrees(resultant_phase):.2f}°)\n"
            f"f = {frequency:.4f} Hz\n"
            f"T = {period:.4f} s\n\n"
            "图像说明:\n"
            "下方三张曲线分别给出两个分振动和合振动的位移随时间变化；"
            "右上旋转矢量图同时展示两个分矢量及其合矢量。"
        )
        self.result_text.setPlainText(text)

    def _update_current_mode_live_widgets(self):
        mode = self.mode_combo.currentText()
        if mode == self.SINGLE_MODE:
            self.update_single_live_widgets()
        elif mode == self.COMPOUND_MODE:
            self.update_compound_live_widgets()
        elif mode == self.DIFF_FREQ_MODE:
            self.update_diff_freq_live_widgets()
        elif mode == self.ORTHOGONAL_MODE:
            self.update_orthogonal_live_widgets()
        else:
            self.update_orthogonal_diff_live_widgets()

    def _capture_record_frame(self):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，无法导出 GIF。") from exc

        pixmap = self.record_widget.grab()
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        buffer.close()
        return Image.open(BytesIO(byte_array.data())).convert("RGBA")

    def save_micro_lesson_gif(self, output_path, duration_seconds=None, fps=None):
        duration_seconds = float(duration_seconds or self.micro_lesson_duration)
        fps = int(fps or self.micro_lesson_fps)
        if fps <= 0:
            raise ValueError("GIF 帧率必须大于 0。")
        if duration_seconds <= 0:
            raise ValueError("录制时长必须大于 0。")

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，无法导出 GIF。") from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        was_running = self.timer.isActive()
        original_time = self.current_time
        original_last_tick = self.last_tick
        original_toggle_text = self.btn_toggle_animation.text()

        if was_running:
            self.timer.stop()
        self.btn_toggle_animation.setText("开始动画")

        frame_count = max(2, int(round(duration_seconds * fps)))
        frame_step = duration_seconds / max(frame_count - 1, 1)
        frames = []

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for index in range(frame_count):
                capture_time = original_time + index * frame_step
                if self.duration > 0:
                    capture_time %= self.duration
                self.current_time = capture_time
                self._update_current_mode_live_widgets()
                QApplication.processEvents()
                frames.append(self._capture_record_frame())

            primary = frames[0]
            append_images = frames[1:]
            primary.save(
                output_path,
                save_all=True,
                append_images=append_images,
                duration=max(1, int(round(1000 / fps))),
                loop=0,
                disposal=2,
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.current_time = original_time
            self.last_tick = original_last_tick
            self._update_current_mode_live_widgets()
            self.btn_toggle_animation.setText(original_toggle_text if was_running else "开始动画")
            if was_running:
                self.last_tick = time.monotonic()
                self.timer.start()

        return output_path

    def on_record_micro_lesson(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"vibration_micro_lesson_{timestamp}.gif"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存微课 GIF",
            str(Path.home() / default_name),
            "GIF 动图 (*.gif)",
        )
        if not output_path:
            return

        self.btn_record_micro_lesson.setEnabled(False)
        self.live_status_label.setText(
            f"正在录制微课 GIF（{self.micro_lesson_duration:.0f} s，{self.micro_lesson_fps} fps）..."
        )
        QApplication.processEvents()
        try:
            saved_path = self.save_micro_lesson_gif(output_path)
        except Exception as exc:
            self._update_current_mode_live_widgets()
            QMessageBox.warning(self, "录制失败", f"微课 GIF 导出失败：\n{exc}")
        else:
            QMessageBox.information(self, "录制完成", f"微课 GIF 已保存到：\n{saved_path}")
        finally:
            self.btn_record_micro_lesson.setEnabled(True)

    def on_toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.last_tick = None
            self.btn_toggle_animation.setText("开始动画")
            return

        self.last_tick = time.monotonic()
        self.timer.start()
        self.btn_toggle_animation.setText("暂停动画")

    def on_reset_animation(self):
        if self.timer.isActive():
            self.timer.stop()
        self.btn_toggle_animation.setText("开始动画")
        self.current_time = 0.0
        self.last_tick = None
        self.refresh_current_mode(reset_time=True)

    def on_animation_tick(self):
        now = time.monotonic()
        if self.last_tick is None:
            self.last_tick = now
            return

        elapsed = min(max(now - self.last_tick, 0.0), 0.1)
        self.last_tick = now
        self.current_time += elapsed
        if self.duration > 0:
            self.current_time %= self.duration

        self._update_current_mode_live_widgets()
