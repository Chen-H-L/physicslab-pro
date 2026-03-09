import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


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
        usable = max(24, spring_end_x - spring_start_x - lead * 2)
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


class VibrationLabTab(QWidget):
    SINGLE_MODE = "单一简谐振动"
    COMPOUND_MODE = "同方向同频率合成"

    SINGLE_TITLES = ["位移 x/t 曲线", "速度 v/t 曲线", "加速度 a/t 曲线"]
    SINGLE_YLABELS = ["x (cm)", "v (cm/s)", "a (cm/s^2)"]
    SINGLE_COLORS = ["#2f80ed", "#27ae60", "#e67e22"]

    COMPOUND_TITLES = ["分振动 1 的 x1/t", "分振动 2 的 x2/t", "合振动 x/t"]
    COMPOUND_YLABELS = ["x1 (cm)", "x2 (cm)", "x (cm)"]
    COMPOUND_COLORS = ["#2f80ed", "#27ae60", "#e67e22"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_time = 0.0
        self.duration = 6.0
        self.last_tick = None

        self.t_values = np.linspace(0.0, self.duration, 1200)
        self.x_values = np.zeros_like(self.t_values)
        self.v_values = np.zeros_like(self.t_values)
        self.a_values = np.zeros_like(self.t_values)
        self.x1_values = np.zeros_like(self.t_values)
        self.x2_values = np.zeros_like(self.t_values)
        self.x_sum_values = np.zeros_like(self.t_values)

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
        self.mode_combo.addItems([self.SINGLE_MODE, self.COMPOUND_MODE])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("选择当前实验内容"))
        mode_layout.addWidget(self.mode_combo)
        left_layout.addWidget(mode_group)

        self.single_param_group = self._build_single_param_group()
        self.compound_param_group = self._build_compound_param_group()
        self.single_derived_group = self._build_single_derived_group()
        self.compound_derived_group = self._build_compound_derived_group()
        self.control_group = self._build_control_group()

        left_layout.addWidget(self.single_param_group)
        left_layout.addWidget(self.compound_param_group)
        left_layout.addWidget(self.single_derived_group)
        left_layout.addWidget(self.compound_derived_group)
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

        compound_visual_group = QGroupBox("旋转矢量法与合成概述")
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
        curve_layout.addWidget(self.single_curve_canvas)
        curve_layout.addWidget(self.compound_curve_canvas)
        self.single_curve_canvas.setMinimumHeight(300)
        self.compound_curve_canvas.setMinimumHeight(300)

        visual_panel = QWidget()
        visual_layout = QVBoxLayout(visual_panel)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(8)
        visual_layout.addWidget(self.single_top_widget)
        visual_layout.addWidget(self.compound_top_widget)
        visual_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        right_layout.addWidget(visual_panel, 0)
        right_layout.addWidget(self.curve_group, 1)

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
        self.phase_spin = self._create_spinbox(-1_000.0, 1_000.0, 0.0, " rad", 4, 0.1)

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
        self.compound_phase_1_spin = self._create_spinbox(-1_000.0, 1_000.0, 0.0, " rad", 4, 0.1)
        self.compound_amplitude_2_spin = self._create_spinbox(0.01, 1_000_000.0, 12.0, " cm", 3, 1.0)
        self.compound_phase_2_spin = self._create_spinbox(-1_000.0, 1_000.0, 1.2, " rad", 4, 0.1)
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

    def _build_control_group(self):
        group = QGroupBox("动画控制")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.btn_toggle_animation = QPushButton("开始动画")
        self.btn_reset_animation = QPushButton("重置")
        self.btn_toggle_animation.clicked.connect(self.on_toggle_animation)
        self.btn_reset_animation.clicked.connect(self.on_reset_animation)
        button_row.addWidget(self.btn_toggle_animation)
        button_row.addWidget(self.btn_reset_animation)
        layout.addLayout(button_row)

        self.live_status_label = QLabel("t = 0.000 s")
        self.live_status_label.setWordWrap(True)
        self.live_status_label.setStyleSheet(
            "padding: 6px 8px; border-radius: 8px; background-color: #eff6ff; color: #1d4ed8;"
        )
        layout.addWidget(self.live_status_label)
        return group

    def _create_spinbox(self, minimum, maximum, value, suffix, decimals, step):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(decimals)
        spinbox.setValue(value)
        spinbox.setSingleStep(step)
        spinbox.setSuffix(suffix)
        spinbox.setMinimumHeight(36)
        spinbox.setKeyboardTracking(False)
        spinbox.setAccelerated(True)
        return spinbox

    def on_mode_changed(self, mode_text):
        if self.timer.isActive():
            self.timer.stop()
        self.btn_toggle_animation.setText("开始动画")
        self.current_time = 0.0
        self.last_tick = None

        is_single = mode_text == self.SINGLE_MODE
        self.single_param_group.setVisible(is_single)
        self.single_derived_group.setVisible(is_single)
        self.single_top_widget.setVisible(is_single)
        self.single_curve_canvas.setVisible(is_single)

        self.compound_param_group.setVisible(not is_single)
        self.compound_derived_group.setVisible(not is_single)
        self.compound_top_widget.setVisible(not is_single)
        self.compound_curve_canvas.setVisible(not is_single)

        if is_single:
            self.subtitle_label.setText("弹簧振子、x/t、v/t、a/t 与旋转矢量同步联动")
            self.display_title.setText("简谐振动仿真")
            self.curve_group.setTitle("x/t、v/t、a/t 曲线")
            self.result_group.setTitle("实验说明")
        else:
            self.subtitle_label.setText("两个同方向、同频率简谐振动的图像与旋转矢量合成")
            self.display_title.setText("同方向同频率简谐振动合成")
            self.curve_group.setTitle("分振动与合振动 x/t 曲线")
            self.result_group.setTitle("合成说明")

        self.refresh_current_mode(reset_time=True)

    def refresh_current_mode(self, reset_time=False):
        if reset_time:
            self.current_time = 0.0
            self.last_tick = time.monotonic()
        elif self.duration > 0:
            self.current_time %= self.duration

        if self.mode_combo.currentText() == self.SINGLE_MODE:
            self.update_single_simulation(reset_time=reset_time)
        else:
            self.update_compound_simulation(reset_time=reset_time)

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

        if self.mode_combo.currentText() == self.SINGLE_MODE:
            self.update_single_live_widgets()
        else:
            self.update_compound_live_widgets()
