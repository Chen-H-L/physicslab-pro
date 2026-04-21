import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
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


PROCESS_MODES = {
    "isothermal": "等温改变体积",
    "isochoric": "等体积改变温度",
    "isobaric": "等压改变温度",
}


class GasChamberWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.process_mode = "isothermal"
        self.pressure_kpa = 101.3
        self.volume_l = 4.0
        self.temperature_k = 300.0
        self.reference_pressure_kpa = 101.3
        self.reference_volume_l = 4.0
        self.reference_temperature_k = 300.0
        self.constant_c = self.pressure_kpa * self.volume_l / self.temperature_k
        self.state_error_percent = 0.0

        self.rng = np.random.default_rng(20260421)
        self.particles = []
        self._build_particles()

        self.motion_timer = QTimer(self)
        self.motion_timer.timeout.connect(self._advance_particles)
        self.motion_timer.start(30)

    def _build_particles(self):
        self.particles = []
        for _ in range(38):
            vx = self.rng.uniform(-0.55, 0.55)
            vy = self.rng.uniform(-0.55, 0.55)
            if abs(vx) < 0.18:
                vx = 0.22 if vx >= 0 else -0.22
            if abs(vy) < 0.18:
                vy = 0.22 if vy >= 0 else -0.22
            self.particles.append(
                {
                    "x": self.rng.uniform(0.08, 0.92),
                    "y": self.rng.uniform(0.10, 0.90),
                    "vx": vx,
                    "vy": vy,
                    "radius": self.rng.uniform(2.3, 4.6),
                }
            )

    def set_state(
        self,
        process_mode,
        pressure_kpa,
        volume_l,
        temperature_k,
        reference_pressure_kpa,
        reference_volume_l,
        reference_temperature_k,
        constant_c,
    ):
        self.process_mode = process_mode
        self.pressure_kpa = float(pressure_kpa)
        self.volume_l = float(volume_l)
        self.temperature_k = float(temperature_k)
        self.reference_pressure_kpa = float(reference_pressure_kpa)
        self.reference_volume_l = float(reference_volume_l)
        self.reference_temperature_k = float(reference_temperature_k)
        self.constant_c = float(constant_c)
        current_c = self.pressure_kpa * self.volume_l / max(self.temperature_k, 1e-9)
        self.state_error_percent = abs(current_c - self.constant_c) / max(self.constant_c, 1e-9) * 100.0
        self.update()

    def _advance_particles(self):
        dt = 0.030
        speed_scale = np.clip(np.sqrt(self.temperature_k / max(self.reference_temperature_k, 1e-9)), 0.6, 2.2)
        volume_ratio = np.clip(self.volume_l / max(self.reference_volume_l, 1e-9), 0.48, 1.85)

        for particle in self.particles:
            particle["x"] += particle["vx"] * dt * speed_scale
            particle["y"] += particle["vy"] * dt * speed_scale / max(volume_ratio, 0.6)

            if particle["x"] <= 0.04 or particle["x"] >= 0.96:
                particle["vx"] *= -1
                particle["x"] = np.clip(particle["x"], 0.04, 0.96)

            if particle["y"] <= 0.06 or particle["y"] >= 0.94:
                particle["vy"] *= -1
                particle["y"] = np.clip(particle["y"], 0.06, 0.94)

        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)

        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(rect, 14, 14)

        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        painter.drawText(rect.left() + 16, rect.top() + 26, "理想气体分子动理论模型")

        info_width = min(276, max(228, int(rect.width() * 0.28)))
        model_rect = rect.adjusted(16, 38, -(info_width + 22), -18)
        info_rect = rect.adjusted(rect.width() - info_width - 12, 48, -12, -20)

        chamber_bottom = model_rect.bottom() - 34
        chamber_max_top = model_rect.top() + 30
        chamber_max_height = chamber_bottom - chamber_max_top
        volume_ratio = np.clip(self.volume_l / max(self.reference_volume_l, 1e-9), 0.48, 1.85)
        height_ratio = np.clip(volume_ratio, 0.52, 1.72)
        chamber_height = chamber_max_height * height_ratio / 1.18
        chamber_top = chamber_bottom - chamber_height

        glass_rect = model_rect.adjusted(40, 40, -48, -40)
        glass_rect.setTop(int(chamber_top))
        glass_rect.setBottom(int(chamber_bottom))

        heater_intensity = np.clip((self.temperature_k / max(self.reference_temperature_k, 1e-9) - 0.6) / 1.2, 0.05, 1.0)
        chamber_gradient = QLinearGradient(glass_rect.left(), glass_rect.top(), glass_rect.left(), glass_rect.bottom())
        chamber_gradient.setColorAt(0.0, QColor("#f8fbff"))
        chamber_gradient.setColorAt(0.65, QColor("#eef5ff"))
        chamber_gradient.setColorAt(1.0, QColor(255, 239, 219, int(70 + 60 * heater_intensity)))

        painter.setPen(QPen(QColor("#475569"), 3))
        painter.setBrush(chamber_gradient)
        painter.drawRoundedRect(glass_rect, 16, 16)

        piston_height = 12
        piston_rect = glass_rect.adjusted(-8, -piston_height - 6, 8, -glass_rect.height() + piston_height + 6)
        painter.setPen(QPen(QColor("#64748b"), 2))
        painter.setBrush(QColor("#cbd5e1"))
        painter.drawRoundedRect(piston_rect, 8, 8)
        painter.setBrush(QColor("#94a3b8"))
        painter.drawRoundedRect(
            int(piston_rect.center().x() - 20),
            int(piston_rect.top() - 9),
            40,
            7,
            4,
            4,
        )

        heating_strip = glass_rect.adjusted(18, glass_rect.height() + 12, -18, glass_rect.height() + 22)
        heat_gradient = QLinearGradient(heating_strip.left(), heating_strip.top(), heating_strip.left(), heating_strip.bottom())
        heat_gradient.setColorAt(0.0, QColor(255, 181, 71, 40))
        heat_gradient.setColorAt(0.5, QColor(255, 120, 57, int(140 * heater_intensity)))
        heat_gradient.setColorAt(1.0, QColor(255, 84, 45, int(220 * heater_intensity)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(heat_gradient)
        painter.drawRoundedRect(heating_strip, 8, 8)

        painter.setPen(QPen(QColor("#ff7a1a"), 2))
        flame_base = heating_strip.bottom() + 8
        for index in range(5):
            x = heating_strip.left() + 20 + index * (heating_strip.width() - 40) / 4
            painter.drawLine(int(x), int(flame_base), int(x), int(flame_base + 14))
            painter.drawLine(int(x), int(flame_base + 10), int(x - 4), int(flame_base + 5))
            painter.drawLine(int(x), int(flame_base + 10), int(x + 4), int(flame_base + 5))

        pressure_factor = np.clip(self.pressure_kpa / max(self.reference_pressure_kpa, 1e-9), 0.5, 1.9)
        arrow_count = max(4, int(4 + pressure_factor * 3.2))
        arrow_span = glass_rect.width() / max(arrow_count - 1, 1)
        painter.setPen(QPen(QColor("#ef4444"), 2))
        for index in range(arrow_count):
            x = glass_rect.left() + index * arrow_span
            top_y = piston_rect.top() - 12
            bottom_y = piston_rect.top() - 12 + 10 + pressure_factor * 12
            painter.drawLine(int(x), int(top_y), int(x), int(bottom_y))
            painter.drawLine(int(x), int(bottom_y), int(x - 4), int(bottom_y - 5))
            painter.drawLine(int(x), int(bottom_y), int(x + 4), int(bottom_y - 5))

        speed_factor = np.clip(np.sqrt(self.temperature_k / max(self.reference_temperature_k, 1e-9)), 0.8, 1.8)
        particle_color = QColor(
            min(255, int(90 + 120 * heater_intensity)),
            min(255, int(140 + 60 * (1.0 - heater_intensity))),
            255,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(particle_color)
        for particle in self.particles:
            px = glass_rect.left() + particle["x"] * glass_rect.width()
            py = glass_rect.top() + particle["y"] * glass_rect.height()
            trail_x = px - particle["vx"] * 7.0 * speed_factor
            trail_y = py - particle["vy"] * 7.0 * speed_factor
            painter.setPen(QPen(QColor(191, 219, 254, 95), 1.6))
            painter.drawLine(int(trail_x), int(trail_y), int(px), int(py))
            painter.setPen(Qt.PenStyle.NoPen)
            radius = particle["radius"] * speed_factor * 0.8
            painter.drawEllipse(int(px - radius), int(py - radius), int(radius * 2), int(radius * 2))

        painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(glass_rect.left() - 16), int(glass_rect.top()), int(glass_rect.right() + 16), int(glass_rect.top()))
        painter.setPen(QColor("#334155"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(int(glass_rect.right() + 12), int(glass_rect.top() + 4), "活塞位置")

        self._draw_info_panel(painter, info_rect)

    def _draw_info_panel(self, painter, rect):
        panel_rect = rect.adjusted(0, 0, 0, 0)
        painter.setPen(QPen(QColor("#d6e4ff"), 1))
        painter.setBrush(QColor("#f8fbff"))
        painter.drawRoundedRect(panel_rect, 12, 12)

        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        painter.drawText(panel_rect.left() + 14, panel_rect.top() + 24, "状态信息")

        lines = [
            f"过程：{PROCESS_MODES[self.process_mode]}",
            f"P = {self.pressure_kpa:.2f} kPa",
            f"V = {self.volume_l:.3f} L",
            f"T = {self.temperature_k:.2f} K",
            f"C = PV/T = {self.constant_c:.4f}",
            f"方程偏差 = {self.state_error_percent:.3f}%",
        ]
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor("#334155"))
        y = panel_rect.top() + 52
        for text in lines:
            painter.drawText(panel_rect.left() + 14, y, text)
            y += 21


class ThermoPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8.8, 3.5), dpi=100, facecolor="#f8f9fa")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axis = self.figure.subplots(1, 1)
        self.figure.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.97)

    def update_plots(self, mode, initial_state, history, current_state, limits):
        c_value = initial_state["pressure"] * initial_state["volume"] / initial_state["temperature"]

        history_p = np.asarray(history["pressure"], dtype=float)
        history_v = np.asarray(history["volume"], dtype=float)
        history_t = np.asarray(history["temperature"], dtype=float)

        if mode == "isothermal":
            self._plot_pv(self.axis, mode, initial_state, history_v, history_p, current_state, c_value, limits)
        elif mode == "isochoric":
            self._plot_pt(self.axis, mode, initial_state, history_t, history_p, current_state, c_value, limits)
        else:
            self._plot_vt(self.axis, mode, initial_state, history_t, history_v, current_state, c_value, limits)
        self.draw_idle()

    def _style_axis(self, axis, title, xlabel, ylabel, active):
        axis.clear()
        axis.set_facecolor("#ffffff")
        axis.grid(True, linestyle="--", alpha=0.28, color="#cbd5e1")
        axis.set_title(
            title,
            loc="left",
            fontsize=11,
            color="#0f172a" if active else "#64748b",
            pad=8,
            fontweight="bold" if active else "normal",
        )
        axis.set_xlabel(xlabel, color="#334155")
        axis.set_ylabel(ylabel, color="#334155")
        axis.tick_params(colors="#475569")
        for spine in axis.spines.values():
            spine.set_color("#cbd5e1")

    def _plot_pv(self, axis, mode, initial_state, history_v, history_p, current_state, c_value, limits):
        self._style_axis(axis, "压强-体积关系 P/V", "V (L)", "P (kPa)", mode == "isothermal")
        v_span = np.linspace(limits["volume"][0], limits["volume"][1], 260)
        guide_color = "#2563eb" if mode == "isothermal" else "#94a3b8"

        if mode == "isothermal":
            guide = c_value * initial_state["temperature"] / v_span
            axis.plot(v_span, guide, color=guide_color, linewidth=2.0, label="等温理论曲线")
        elif mode == "isochoric":
            axis.axvline(initial_state["volume"], color=guide_color, linestyle="--", linewidth=1.8, label="V = 常量")
        else:
            axis.axhline(initial_state["pressure"], color=guide_color, linestyle="--", linewidth=1.8, label="P = 常量")

        axis.plot(history_v, history_p, color="#f97316", linewidth=2.2, label="过程轨迹")
        axis.plot(
            [current_state["volume"]],
            [current_state["pressure"]],
            "o",
            color="#111827",
            markersize=6,
            label="当前状态",
        )
        self._set_axis_limits(axis, history_v, history_p, limits["volume"], limits["pressure"])
        axis.legend(loc="upper right", fontsize=8, frameon=False)

    def _plot_pt(self, axis, mode, initial_state, history_t, history_p, current_state, c_value, limits):
        self._style_axis(axis, "压强-温度关系 P/T", "T (K)", "P (kPa)", mode == "isochoric")
        t_span = np.linspace(limits["temperature"][0], limits["temperature"][1], 260)
        guide_color = "#2563eb" if mode == "isochoric" else "#94a3b8"

        if mode == "isochoric":
            guide = c_value * t_span / initial_state["volume"]
            axis.plot(t_span, guide, color=guide_color, linewidth=2.0, label="等体理论直线")
        elif mode == "isothermal":
            axis.axvline(initial_state["temperature"], color=guide_color, linestyle="--", linewidth=1.8, label="T = 常量")
        else:
            axis.axhline(initial_state["pressure"], color=guide_color, linestyle="--", linewidth=1.8, label="P = 常量")

        axis.plot(history_t, history_p, color="#10b981", linewidth=2.2, label="过程轨迹")
        axis.plot(
            [current_state["temperature"]],
            [current_state["pressure"]],
            "o",
            color="#111827",
            markersize=6,
            label="当前状态",
        )
        self._set_axis_limits(axis, history_t, history_p, limits["temperature"], limits["pressure"])
        axis.legend(loc="upper right", fontsize=8, frameon=False)

    def _plot_vt(self, axis, mode, initial_state, history_t, history_v, current_state, c_value, limits):
        self._style_axis(axis, "体积-温度关系 V/T", "T (K)", "V (L)", mode == "isobaric")
        t_span = np.linspace(limits["temperature"][0], limits["temperature"][1], 260)
        guide_color = "#2563eb" if mode == "isobaric" else "#94a3b8"

        if mode == "isobaric":
            guide = c_value * t_span / initial_state["pressure"]
            axis.plot(t_span, guide, color=guide_color, linewidth=2.0, label="等压理论直线")
        elif mode == "isochoric":
            axis.axhline(initial_state["volume"], color=guide_color, linestyle="--", linewidth=1.8, label="V = 常量")
        else:
            axis.axvline(initial_state["temperature"], color=guide_color, linestyle="--", linewidth=1.8, label="T = 常量")

        axis.plot(history_t, history_v, color="#8b5cf6", linewidth=2.2, label="过程轨迹")
        axis.plot(
            [current_state["temperature"]],
            [current_state["volume"]],
            "o",
            color="#111827",
            markersize=6,
            label="当前状态",
        )
        self._set_axis_limits(axis, history_t, history_v, limits["temperature"], limits["volume"])
        axis.legend(loc="upper right", fontsize=8, frameon=False)

    def _set_axis_limits(self, axis, x_data, y_data, x_limit, y_limit):
        x_min = min(np.min(x_data), x_limit[0])
        x_max = max(np.max(x_data), x_limit[1])
        y_min = min(np.min(y_data), y_limit[0])
        y_max = max(np.max(y_data), y_limit[1])
        x_pad = max((x_max - x_min) * 0.08, 0.4 if x_max - x_min < 8 else 0.0)
        y_pad = max((y_max - y_min) * 0.12, 0.4 if y_max - y_min < 8 else 0.0)
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)


class ThermodynamicsLabTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = "isothermal"
        self.animation_duration_s = 3.6
        self.animation_running = False
        self.animation_paused = False
        self.animation_start_time = None
        self.animation_start_variable = None
        self.animation_target_variable = None
        self.history = {"pressure": [], "volume": [], "temperature": []}

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._advance_process)
        self.animation_timer.setInterval(35)

        self._build_ui()
        self._connect_signals()
        self._sync_target_range_labels()
        self.reset_simulation()

    def _build_ui(self):
        self.setObjectName("thermodynamicsLabTab")
        self.setStyleSheet(
            """
            QWidget#thermodynamicsLabTab {
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
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #1f2937;
                border: 1px solid #d9d9d9;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
            }
            """
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        control_area = QScrollArea()
        control_area.setWidgetResizable(True)
        control_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_area.setMinimumWidth(318)
        control_area.setMaximumWidth(360)
        splitter.addWidget(control_area)

        control_widget = QWidget()
        control_area.setWidget(control_widget)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(10)

        header = QLabel("热力学模拟仿真实验")
        header.setStyleSheet("font-size: 20px; font-weight: 700; color: #0f172a;")
        subtitle = QLabel("理想气体 | 活塞容器 | P-V-T 定量关系")
        subtitle.setStyleSheet("font-size: 12px; color: #64748b;")
        control_layout.addWidget(header)
        control_layout.addWidget(subtitle)

        self.mode_group = QGroupBox("实验模式")
        mode_layout = QVBoxLayout(self.mode_group)
        mode_layout.addWidget(QLabel("请选择当前热力学过程"))
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumHeight(36)
        for key, text in PROCESS_MODES.items():
            self.mode_combo.addItem(text, key)
        mode_layout.addWidget(self.mode_combo)
        control_layout.addWidget(self.mode_group)

        self.initial_group = QGroupBox("初始状态")
        initial_layout = QGridLayout(self.initial_group)
        initial_layout.setHorizontalSpacing(10)
        initial_layout.setVerticalSpacing(10)
        self.pressure_spin = self._create_spinbox(60.0, 220.0, 101.3, 0.5, 1, " kPa")
        self.volume_spin = self._create_spinbox(1.5, 9.0, 4.0, 0.1, 3, " L")
        self.temperature_spin = self._create_spinbox(220.0, 520.0, 300.0, 1.0, 1, " K")
        initial_layout.addWidget(QLabel("初始压强 P₀"), 0, 0)
        initial_layout.addWidget(self.pressure_spin, 0, 1)
        initial_layout.addWidget(QLabel("初始体积 V₀"), 1, 0)
        initial_layout.addWidget(self.volume_spin, 1, 1)
        initial_layout.addWidget(QLabel("初始温度 T₀"), 2, 0)
        initial_layout.addWidget(self.temperature_spin, 2, 1)
        control_layout.addWidget(self.initial_group)

        self.process_group = QGroupBox("过程控制")
        process_layout = QGridLayout(self.process_group)
        process_layout.setHorizontalSpacing(10)
        process_layout.setVerticalSpacing(10)
        self.target_label = QLabel("目标体积 V")
        self.target_spin = self._create_spinbox(1.0, 12.0, 2.4, 0.1, 3, " L")
        self.duration_spin = self._create_spinbox(1.5, 8.0, 3.6, 0.1, 1, " s")
        process_layout.addWidget(self.target_label, 0, 0)
        process_layout.addWidget(self.target_spin, 0, 1)
        process_layout.addWidget(QLabel("过程时长"), 1, 0)
        process_layout.addWidget(self.duration_spin, 1, 1)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("开始动画")
        self.pause_button = QPushButton("暂停动画")
        self.reset_button = QPushButton("重置")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.pause_button)
        process_layout.addLayout(button_row, 2, 0, 1, 2)
        process_layout.addWidget(self.reset_button, 3, 0, 1, 2)

        self.status_label = QLabel("等待开始：根据目标参数执行理想气体过程。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #eff6ff; color: #1d4ed8; border-radius: 8px; padding: 10px 12px;"
        )
        process_layout.addWidget(self.status_label, 4, 0, 1, 2)
        control_layout.addWidget(self.process_group)

        self.summary_group = QGroupBox("定量推导量")
        summary_layout = QGridLayout(self.summary_group)
        summary_layout.setHorizontalSpacing(12)
        summary_layout.setVerticalSpacing(8)
        self.summary_labels = {}
        summary_items = [
            ("constant", "状态常量 C"),
            ("pressure", "当前压强 P"),
            ("volume", "当前体积 V"),
            ("temperature", "当前温度 T"),
            ("law", "过程关系"),
            ("consistency", "方程校验"),
        ]
        for row, (key, text) in enumerate(summary_items):
            name_label = QLabel(text)
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: 600; color: #0f172a;")
            summary_layout.addWidget(name_label, row, 0)
            summary_layout.addWidget(value_label, row, 1)
            self.summary_labels[key] = value_label
        control_layout.addWidget(self.summary_group)

        self.note_group = QGroupBox("实验说明")
        note_layout = QVBoxLayout(self.note_group)
        self.note_text = QTextEdit()
        self.note_text.setReadOnly(True)
        self.note_text.setMinimumHeight(220)
        note_layout.addWidget(self.note_text)
        control_layout.addWidget(self.note_group)
        control_layout.addStretch(1)

        right_widget = QWidget()
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)

        self.model_group = QGroupBox("理想气体分子动理论模拟")
        model_layout = QVBoxLayout(self.model_group)
        self.model_hint = QLabel(
            "活塞高度表示体积 V，粒子运动速度对应温度 T，壁面作用与箭头密度体现压强 P。"
        )
        self.model_hint.setStyleSheet("color: #64748b; padding-left: 4px;")
        self.gas_widget = GasChamberWidget()
        model_layout.addWidget(self.model_hint)
        model_layout.addWidget(self.gas_widget, 1)
        right_layout.addWidget(self.model_group, 4)

        self.plot_group = QGroupBox("当前关系图")
        plot_layout = QVBoxLayout(self.plot_group)
        self.plot_canvas = ThermoPlotCanvas()
        plot_layout.addWidget(self.plot_canvas)
        right_layout.addWidget(self.plot_group, 5)
        splitter.setSizes([332, 1048])

    def _connect_signals(self):
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.pressure_spin.valueChanged.connect(self._on_initial_state_changed)
        self.volume_spin.valueChanged.connect(self._on_initial_state_changed)
        self.temperature_spin.valueChanged.connect(self._on_initial_state_changed)
        self.target_spin.valueChanged.connect(self._on_target_changed)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        self.start_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.reset_button.clicked.connect(self.reset_simulation)

    def _create_spinbox(self, minimum, maximum, value, step, decimals, suffix):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setMinimumHeight(36)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        return spin

    def _on_mode_changed(self):
        self.current_mode = self.mode_combo.currentData()
        self._sync_target_range_labels()
        self.reset_simulation()

    def _on_initial_state_changed(self):
        self._sync_target_range_labels()
        self.reset_simulation()

    def _on_target_changed(self):
        if not self.animation_running:
            target_text = self._target_description()
            self.status_label.setText(f"目标已更新：{target_text}。点击“开始动画”以执行过程。")
            self._refresh_views()

    def _on_duration_changed(self):
        self.animation_duration_s = self.duration_spin.value()

    def _sync_target_range_labels(self):
        if self.current_mode == "isothermal":
            base_volume = self.volume_spin.value()
            self.target_label.setText("目标体积 V")
            self.target_spin.blockSignals(True)
            self.target_spin.setDecimals(3)
            self.target_spin.setSuffix(" L")
            self.target_spin.setRange(max(0.8, base_volume * 0.45), min(14.0, base_volume * 1.85))
            target_value = np.clip(self.target_spin.value(), self.target_spin.minimum(), self.target_spin.maximum())
            if abs(target_value - base_volume) < 0.02:
                target_value = min(self.target_spin.maximum(), base_volume * 0.72)
            self.target_spin.setValue(target_value)
            self.target_spin.blockSignals(False)
        else:
            base_temp = self.temperature_spin.value()
            self.target_label.setText("目标温度 T")
            self.target_spin.blockSignals(True)
            self.target_spin.setDecimals(1)
            self.target_spin.setSuffix(" K")
            self.target_spin.setRange(max(180.0, base_temp * 0.70), min(720.0, base_temp * 1.75))
            target_value = np.clip(self.target_spin.value(), self.target_spin.minimum(), self.target_spin.maximum())
            if abs(target_value - base_temp) < 0.5:
                target_value = min(self.target_spin.maximum(), base_temp * 1.28)
            self.target_spin.setValue(target_value)
            self.target_spin.blockSignals(False)

    def _initial_state(self):
        return {
            "pressure": self.pressure_spin.value(),
            "volume": self.volume_spin.value(),
            "temperature": self.temperature_spin.value(),
        }

    def _constant_c(self):
        state = self._initial_state()
        return state["pressure"] * state["volume"] / max(state["temperature"], 1e-9)

    def _limits(self):
        initial = self._initial_state()
        c_value = self._constant_c()
        target = self.target_spin.value()

        temp_min = min(self.temperature_spin.minimum(), initial["temperature"], target)
        temp_max = max(self.temperature_spin.maximum(), initial["temperature"], target)
        vol_min = min(self.volume_spin.minimum(), initial["volume"], target if self.current_mode == "isothermal" else initial["volume"])
        vol_max = max(self.volume_spin.maximum(), initial["volume"], target if self.current_mode == "isothermal" else initial["volume"])

        pressure_candidates = [initial["pressure"]]
        if self.current_mode == "isothermal":
            pressure_candidates.extend(
                [
                    c_value * initial["temperature"] / max(vol_min, 1e-6),
                    c_value * initial["temperature"] / max(vol_max, 1e-6),
                ]
            )
        elif self.current_mode == "isochoric":
            pressure_candidates.extend(
                [
                    c_value * temp_min / initial["volume"],
                    c_value * temp_max / initial["volume"],
                ]
            )
        else:
            pressure_candidates.extend([initial["pressure"], initial["pressure"]])

        pressure_min = max(15.0, min(pressure_candidates) * 0.88)
        pressure_max = max(pressure_candidates) * 1.16
        return {
            "pressure": (pressure_min, pressure_max),
            "volume": (max(0.5, vol_min * 0.88), vol_max * 1.10),
            "temperature": (max(120.0, temp_min * 0.90), temp_max * 1.06),
        }

    def _current_state(self):
        return {
            "pressure": self.current_pressure,
            "volume": self.current_volume,
            "temperature": self.current_temperature,
        }

    def _target_state(self):
        initial = self._initial_state()
        c_value = self._constant_c()
        target = self.target_spin.value()

        if self.current_mode == "isothermal":
            volume = target
            temperature = initial["temperature"]
            pressure = c_value * temperature / volume
        elif self.current_mode == "isochoric":
            temperature = target
            volume = initial["volume"]
            pressure = c_value * temperature / volume
        else:
            temperature = target
            pressure = initial["pressure"]
            volume = c_value * temperature / pressure

        return {
            "pressure": pressure,
            "volume": volume,
            "temperature": temperature,
        }

    def _target_description(self):
        target = self._target_state()
        if self.current_mode == "isothermal":
            return f"V 调整到 {target['volume']:.3f} L，T 保持 {target['temperature']:.2f} K"
        if self.current_mode == "isochoric":
            return f"T 调整到 {target['temperature']:.2f} K，V 保持 {target['volume']:.3f} L"
        return f"T 调整到 {target['temperature']:.2f} K，P 保持 {target['pressure']:.2f} kPa"

    def start_animation(self):
        self.animation_duration_s = self.duration_spin.value()
        target = self._target_state()
        current = self._current_state()

        if self.current_mode == "isothermal":
            self.animation_start_variable = current["volume"]
            self.animation_target_variable = target["volume"]
        else:
            self.animation_start_variable = current["temperature"]
            self.animation_target_variable = target["temperature"]

        self.animation_running = True
        self.animation_paused = False
        self.pause_button.setText("暂停动画")
        self.start_button.setText("重新开始")
        self.animation_start_time = time.perf_counter()
        self.status_label.setText(f"正在执行：{self._target_description()}。")
        self.animation_timer.start()

    def toggle_pause(self):
        if not self.animation_running:
            return

        if self.animation_paused:
            self.animation_paused = False
            self.animation_start_time = time.perf_counter() - self.paused_elapsed
            self.pause_button.setText("暂停动画")
            self.status_label.setText(f"继续执行：{self._target_description()}。")
            self.animation_timer.start()
        else:
            self.animation_paused = True
            self.paused_elapsed = time.perf_counter() - self.animation_start_time
            self.pause_button.setText("继续动画")
            self.status_label.setText("动画已暂停，可继续或重置。")
            self.animation_timer.stop()

    def reset_simulation(self):
        self.animation_timer.stop()
        self.animation_running = False
        self.animation_paused = False
        self.pause_button.setText("暂停动画")
        self.start_button.setText("开始动画")

        initial = self._initial_state()
        self.current_pressure = initial["pressure"]
        self.current_volume = initial["volume"]
        self.current_temperature = initial["temperature"]

        self.history = {
            "pressure": [self.current_pressure],
            "volume": [self.current_volume],
            "temperature": [self.current_temperature],
        }
        self.status_label.setText(f"已重置到初始状态：P₀={self.current_pressure:.2f} kPa，V₀={self.current_volume:.3f} L，T₀={self.current_temperature:.2f} K。")
        self._refresh_views()

    def _advance_process(self):
        if self.animation_paused:
            return

        elapsed = time.perf_counter() - self.animation_start_time
        progress = min(1.0, elapsed / max(self.animation_duration_s, 0.1))
        smooth = progress * progress * (3.0 - 2.0 * progress)
        initial = self._initial_state()
        c_value = self._constant_c()

        variable = self.animation_start_variable + (self.animation_target_variable - self.animation_start_variable) * smooth

        if self.current_mode == "isothermal":
            self.current_volume = variable
            self.current_temperature = initial["temperature"]
            self.current_pressure = c_value * self.current_temperature / max(self.current_volume, 1e-9)
        elif self.current_mode == "isochoric":
            self.current_temperature = variable
            self.current_volume = initial["volume"]
            self.current_pressure = c_value * self.current_temperature / max(self.current_volume, 1e-9)
        else:
            self.current_temperature = variable
            self.current_pressure = initial["pressure"]
            self.current_volume = c_value * self.current_temperature / max(self.current_pressure, 1e-9)

        self.history["pressure"].append(self.current_pressure)
        self.history["volume"].append(self.current_volume)
        self.history["temperature"].append(self.current_temperature)
        self._refresh_views()

        if progress >= 1.0:
            self.animation_timer.stop()
            self.animation_running = False
            self.start_button.setText("再次演示")
            self.pause_button.setText("暂停动画")
            self.status_label.setText(f"过程完成：{self._target_description()}。")

    def _refresh_views(self):
        initial = self._initial_state()
        c_value = self._constant_c()
        current = self._current_state()
        self._update_plot_group_title()
        self.gas_widget.set_state(
            self.current_mode,
            current["pressure"],
            current["volume"],
            current["temperature"],
            initial["pressure"],
            initial["volume"],
            initial["temperature"],
            c_value,
        )
        self.plot_canvas.update_plots(self.current_mode, initial, self.history, current, self._limits())
        self._update_summary()
        self._update_notes()

    def _update_plot_group_title(self):
        titles = {
            "isothermal": "压强-体积关系 P/V",
            "isochoric": "压强-温度关系 P/T",
            "isobaric": "体积-温度关系 V/T",
        }
        self.plot_group.setTitle(titles[self.current_mode])

    def _update_summary(self):
        initial = self._initial_state()
        current = self._current_state()
        c_value = self._constant_c()
        current_c = current["pressure"] * current["volume"] / max(current["temperature"], 1e-9)
        deviation = abs(current_c - c_value) / max(c_value, 1e-9) * 100.0

        if self.current_mode == "isothermal":
            relation = "T = 常量，P 与 V 成反比"
        elif self.current_mode == "isochoric":
            relation = "V = 常量，P 与 T 成正比"
        else:
            relation = "P = 常量，V 与 T 成正比"

        self.summary_labels["constant"].setText(f"{c_value:.4f} kPa·L/K")
        self.summary_labels["pressure"].setText(f"{current['pressure']:.2f} kPa")
        self.summary_labels["volume"].setText(f"{current['volume']:.3f} L")
        self.summary_labels["temperature"].setText(f"{current['temperature']:.2f} K")
        self.summary_labels["law"].setText(relation)
        self.summary_labels["consistency"].setText(f"PV/T = {current_c:.4f}，偏差 {deviation:.3f}%")

    def _update_notes(self):
        initial = self._initial_state()
        target = self._target_state()
        c_value = self._constant_c()

        if self.current_mode == "isothermal":
            relation = (
                "等温过程：\n"
                "T = 常量，P V = 常量。\n"
                "当活塞压缩体积时，粒子与器壁碰撞更频繁，因此压强上升；反之体积增大时压强降低。"
            )
            quantitative = (
                f"本次设置：T = {initial['temperature']:.2f} K，"
                f"V 从 {initial['volume']:.3f} L 调整到 {target['volume']:.3f} L，"
                f"P 将由 {initial['pressure']:.2f} kPa 变化到 {target['pressure']:.2f} kPa。"
            )
        elif self.current_mode == "isochoric":
            relation = (
                "等体过程：\n"
                "V = 常量，P / T = 常量。\n"
                "容器体积固定时，加热让分子平均动能增大，粒子撞击器壁更猛烈，因此压强升高。"
            )
            quantitative = (
                f"本次设置：V = {initial['volume']:.3f} L，"
                f"T 从 {initial['temperature']:.2f} K 调整到 {target['temperature']:.2f} K，"
                f"P 将由 {initial['pressure']:.2f} kPa 变化到 {target['pressure']:.2f} kPa。"
            )
        else:
            relation = (
                "等压过程：\n"
                "P = 常量，V / T = 常量。\n"
                "外界压强固定时，加热使粒子运动增强，活塞上升，体积随温度近似线性增大。"
            )
            quantitative = (
                f"本次设置：P = {initial['pressure']:.2f} kPa，"
                f"T 从 {initial['temperature']:.2f} K 调整到 {target['temperature']:.2f} K，"
                f"V 将由 {initial['volume']:.3f} L 变化到 {target['volume']:.3f} L。"
            )

        self.note_text.setPlainText(
            "\n".join(
                [
                    "核心模型：",
                    "理想气体状态方程 P V / T = C，其中 C = nR。",
                    "",
                    relation,
                    "",
                    quantitative,
                    "",
                    f"本实验初始状态常量：C = {c_value:.4f} kPa·L/K。",
                    "观察建议：关注上方活塞位置、粒子速度和下方三张关系图如何同步变化。",
                ]
            )
        )
