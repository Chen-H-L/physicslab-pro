import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks, sawtooth, square

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
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

from .widgets import SliderSpinBox


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MODULE_FOURIER = "振动频谱分析 / 傅里叶变换实验室"
MODULE_QUANTUM = "量子干涉实验室"

MODE_SINE = "快速傅里叶：简谐振动"
MODE_UNDER_DAMPED = "阻尼振动：欠阻尼"
MODE_CRITICAL_DAMPED = "阻尼振动：临界阻尼"
MODE_OVER_DAMPED = "阻尼振动：过阻尼"
MODE_SQUARE = "其他波形：方波"
MODE_TRIANGLE = "其他波形：三角波"
MODE_SAWTOOTH = "其他波形：锯齿波"
MODE_GAUSSIAN = "高斯脉冲"
MODE_BEAT = "拍振"

MODE_DOUBLE_SLIT = "量子双缝干涉"
MODE_MACH_ZEHNDER = "Mach-Zehnder 单光子干涉仪"

FOURIER_MODES = [
    MODE_SINE,
    MODE_UNDER_DAMPED,
    MODE_CRITICAL_DAMPED,
    MODE_OVER_DAMPED,
    MODE_SQUARE,
    MODE_TRIANGLE,
    MODE_SAWTOOTH,
    MODE_GAUSSIAN,
    MODE_BEAT,
]
QUANTUM_MODES = [MODE_DOUBLE_SLIT, MODE_MACH_ZEHNDER]
DAMPING_MODES = {MODE_UNDER_DAMPED, MODE_CRITICAL_DAMPED, MODE_OVER_DAMPED}


class ExtensionCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9.8, 7.1), dpi=100, facecolor="#ffffff")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axes = self.fig.subplots(2, 2)

    def _style_axis(self, ax):
        ax.set_aspect("auto", adjustable="box")
        try:
            ax.set_box_aspect(None)
        except Exception:
            pass
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#cbd5e1", alpha=0.72, linestyle="-", linewidth=0.8)
        ax.tick_params(colors="#334155", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#64748b")
            spine.set_linewidth(0.9)

    def _reset_axes(self):
        self.fig.clear()
        self.axes = self.fig.subplots(2, 2)
        for ax in self.axes.flat:
            ax.set_visible(True)
            self._style_axis(ax)

    def _integrate(self, y, x):
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))

    def _finish(self):
        self.fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.94, hspace=0.36, wspace=0.26)
        self.draw_idle()

    def plot_fourier(self, result):
        self._reset_axes()
        ax_time, ax_recon, ax_full, ax_zoom = self.axes.flat

        t = result["time"]
        signal = result["signal"]
        freq_shift = result["freq_shift"]
        amp_shift = result["amp_shift"]
        positive_freqs = result["positive_freqs"]
        positive_amp = result["positive_amp"]
        reconstructions = result["reconstructions"]
        dominant_freq = result["dominant_freq"]
        top_peaks = result["top_peaks"]

        ax_time.plot(t, signal, color="#2563eb", linewidth=1.6)
        ax_time.set_title(result["title"], fontsize=11, color="#0f172a")
        ax_time.set_xlabel("Time (s)", color="#334155")
        ax_time.set_ylabel("Amplitude", color="#334155")
        ax_time.margins(x=0.01)

        ax_recon.plot(t, signal, color="#94a3b8", linewidth=1.0, alpha=0.45, label="原始信号")
        colors = {3: "#10b981", 10: "#f59e0b", 50: "#ef4444"}
        for keep_count, reconstructed in reconstructions.items():
            ax_recon.plot(
                t,
                reconstructed,
                color=colors.get(keep_count, "#7c3aed"),
                linewidth=1.25,
                label=f"保留{keep_count}个频率成分",
            )
        ax_recon.set_title("IFFT 重构", fontsize=11, color="#0f172a")
        ax_recon.set_xlabel("Time (s)", color="#334155")
        ax_recon.set_ylabel("Amplitude", color="#334155")
        ax_recon.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax_recon.margins(x=0.01)

        ax_full.plot(freq_shift, amp_shift, color="#1d4ed8", linewidth=1.1)
        ax_full.set_title("完整 FFT 幅度谱", fontsize=11, color="#0f172a")
        ax_full.set_xlabel("Freq (Hz)", color="#334155")
        ax_full.set_ylabel("FFT Amplitude", color="#334155")
        ax_full.margins(x=0.01)

        if positive_freqs.size > 1:
            if dominant_freq > 0:
                resolution = max(float(positive_freqs[1] - positive_freqs[0]), 1e-9)
                half_width = max(1.5, dominant_freq * 0.45, resolution * 36)
                left = max(0.0, dominant_freq - half_width)
                right = min(float(positive_freqs[-1]), dominant_freq + half_width)
                if right <= left:
                    left, right = 0.0, min(float(positive_freqs[-1]), max(5.0, dominant_freq * 2.0))
            else:
                left, right = 0.0, min(float(positive_freqs[-1]), 10.0)

            zoom_mask = (positive_freqs >= left) & (positive_freqs <= right)
            if not np.any(zoom_mask):
                zoom_mask = positive_freqs <= min(float(positive_freqs[-1]), 10.0)

            ax_zoom.plot(positive_freqs[zoom_mask], positive_amp[zoom_mask], color="#2563eb", linewidth=1.25)
            for peak in top_peaks[:5]:
                freq = peak["frequency"]
                if left <= freq <= right:
                    ax_zoom.scatter([freq], [peak["amplitude"]], color="#dc2626", s=34, zorder=4)
                    ax_zoom.annotate(
                        f"{freq:.2f} Hz",
                        xy=(freq, peak["amplitude"]),
                        xytext=(4, 8),
                        textcoords="offset points",
                        fontsize=8,
                        color="#991b1b",
                    )
            ax_zoom.set_xlim(left, right)

        ax_zoom.set_title("主峰区域放大", fontsize=11, color="#0f172a")
        ax_zoom.set_xlabel("Freq (Hz)", color="#334155")
        ax_zoom.set_ylabel("FFT Amplitude", color="#334155")
        self._finish()

    def plot_double_slit(self, result):
        self._reset_axes()
        ax_prob, ax_hits, ax_amp, ax_comp = self.axes.flat

        x_mm = result["x"] * 1000.0
        probability = result["probability"]
        envelope = result["envelope"]
        hits_mm = result["hits"] * 1000.0
        y_hits = result["hit_rows"]
        psi = result["psi"]
        visibility = result["visibility"]
        distinguishability = result["distinguishability"]

        probability_scale = max(float(np.nanmax(probability)), 1e-12)
        envelope_scale = max(float(np.nanmax(envelope)), 1e-12)
        ax_prob.plot(x_mm, probability / probability_scale, color="#2563eb", linewidth=1.5, label="概率密度")
        ax_prob.plot(x_mm, envelope / envelope_scale, color="#94a3b8", linewidth=1.1, linestyle="--", label="单缝包络")
        ax_prob.set_title("屏幕概率密度 P(x)", fontsize=11, color="#0f172a")
        ax_prob.set_xlabel("Screen position x (mm)", color="#334155")
        ax_prob.set_ylabel("Normalized probability", color="#334155")
        ax_prob.legend(loc="upper right", fontsize=8, framealpha=0.92)

        ax_hits.scatter(hits_mm, y_hits, s=4, color="#1d4ed8", alpha=0.28, edgecolors="none")
        ax_hits.set_title("单粒子落点累积", fontsize=11, color="#0f172a")
        ax_hits.set_xlabel("Screen position x (mm)", color="#334155")
        ax_hits.set_ylabel("Accumulation", color="#334155")
        ax_hits.set_yticks([])
        ax_hits.set_xlim(x_mm[0], x_mm[-1])

        ax_amp.plot(x_mm, np.real(psi), color="#2563eb", linewidth=1.1, label="Re ψ")
        ax_amp.plot(x_mm, np.imag(psi), color="#ef4444", linewidth=1.1, label="Im ψ")
        ax_amp.set_title("概率振幅相位结构", fontsize=11, color="#0f172a")
        ax_amp.set_xlabel("Screen position x (mm)", color="#334155")
        ax_amp.set_ylabel("Amplitude", color="#334155")
        ax_amp.legend(loc="upper right", fontsize=8, framealpha=0.92)

        labels = ["V", "D", "V^2 + D^2"]
        values = [visibility, distinguishability, visibility ** 2 + distinguishability ** 2]
        colors = ["#2563eb", "#f59e0b", "#10b981" if values[-1] <= 1.0 else "#dc2626"]
        ax_comp.bar(labels, values, color=colors, width=0.58)
        ax_comp.axhline(1.0, color="#64748b", linewidth=1.0, linestyle="--")
        ax_comp.set_ylim(0.0, max(1.08, values[-1] * 1.15))
        ax_comp.set_title("互补关系读数", fontsize=11, color="#0f172a")
        ax_comp.set_ylabel("Value", color="#334155")
        for index, value in enumerate(values):
            ax_comp.text(index, value + 0.025, f"{value:.3f}", ha="center", color="#0f172a", fontsize=9)

        self._finish()

    def plot_mach_zehnder(self, result):
        self._reset_axes()
        ax_scheme, ax_curve, ax_counts, ax_phase = self.axes.flat

        phase_grid = result["phase_grid"]
        curve_p0 = result["curve_p0"]
        curve_p1 = result["curve_p1"]
        phase = result["phase"]
        p0 = result["p0"]
        p1 = result["p1"]
        clicks = result["clicks"]
        amplitudes = result["amplitudes"]

        ax_scheme.set_title("Mach-Zehnder 单光子干涉仪", fontsize=11, color="#0f172a")
        ax_scheme.set_xlim(0, 10)
        ax_scheme.set_ylim(0, 7)
        ax_scheme.set_aspect("equal", adjustable="box")
        ax_scheme.axis("off")
        ax_scheme.plot([1, 3], [3.5, 3.5], color="#2563eb", linewidth=2)
        ax_scheme.plot([3, 6.8], [3.5, 5.6], color="#2563eb", linewidth=2)
        ax_scheme.plot([3, 6.8], [3.5, 1.4], color="#2563eb", linewidth=2)
        ax_scheme.plot([6.8, 8.8], [5.6, 4.5], color="#2563eb", linewidth=2)
        ax_scheme.plot([6.8, 8.8], [1.4, 2.5], color="#2563eb", linewidth=2)
        ax_scheme.plot([2.75, 3.25], [3.05, 3.95], color="#0f172a", linewidth=4)
        ax_scheme.plot([6.55, 7.05], [5.15, 6.05], color="#0f172a", linewidth=4)
        ax_scheme.text(2.38, 2.65, "BS1", color="#0f172a", fontsize=9)
        ax_scheme.text(6.18, 4.75, "BS2", color="#0f172a", fontsize=9)
        ax_scheme.add_patch(Rectangle((4.75, 4.9), 0.86, 0.42, color="#fde68a", ec="#92400e"))
        ax_scheme.text(4.58, 5.46, f"φ={phase:.2f}", color="#92400e", fontsize=9)
        ax_scheme.scatter([9.0, 9.0], [4.5, 2.5], s=160, color=["#10b981", "#ef4444"], zorder=4)
        ax_scheme.text(9.25, 4.38, f"D0 {p0:.3f}", color="#047857", fontsize=9)
        ax_scheme.text(9.25, 2.38, f"D1 {p1:.3f}", color="#991b1b", fontsize=9)
        ax_scheme.text(0.65, 3.34, "|1>", color="#2563eb", fontsize=10)

        ax_curve.plot(phase_grid, curve_p0, color="#10b981", linewidth=1.5, label="P(D0)")
        ax_curve.plot(phase_grid, curve_p1, color="#ef4444", linewidth=1.5, label="P(D1)")
        ax_curve.scatter([phase], [p0], color="#047857", s=34, zorder=4)
        ax_curve.scatter([phase], [p1], color="#991b1b", s=34, zorder=4)
        ax_curve.set_title("探测概率随相位变化", fontsize=11, color="#0f172a")
        ax_curve.set_xlabel("Phase φ (rad)", color="#334155")
        ax_curve.set_ylabel("Probability", color="#334155")
        ax_curve.set_ylim(-0.03, 1.03)
        ax_curve.legend(loc="upper right", fontsize=8, framealpha=0.92)

        ax_counts.bar(["D0", "D1"], clicks, color=["#10b981", "#ef4444"], width=0.58)
        ax_counts.set_title("单光子点击统计", fontsize=11, color="#0f172a")
        ax_counts.set_ylabel("Counts", color="#334155")
        for index, value in enumerate(clicks):
            ax_counts.text(index, value + max(clicks) * 0.025, str(int(value)), ha="center", color="#0f172a", fontsize=10)

        ax_phase.set_title("输出概率振幅", fontsize=11, color="#0f172a")
        ax_phase.axhline(0, color="#cbd5e1", linewidth=1)
        ax_phase.axvline(0, color="#cbd5e1", linewidth=1)
        ax_phase.set_aspect("equal", adjustable="box")
        ax_phase.set_xlim(-1.1, 1.1)
        ax_phase.set_ylim(-1.1, 1.1)
        phasor_colors = ["#10b981", "#ef4444"]
        for label, amp, color in zip(["D0", "D1"], amplitudes, phasor_colors):
            ax_phase.arrow(
                0,
                0,
                float(np.real(amp)),
                float(np.imag(amp)),
                width=0.012,
                head_width=0.065,
                length_includes_head=True,
                color=color,
                alpha=0.88,
            )
            ax_phase.text(float(np.real(amp)) * 1.08, float(np.imag(amp)) * 1.08, label, color=color, fontsize=9)
        ax_phase.set_xlabel("Re", color="#334155")
        ax_phase.set_ylabel("Im", color="#334155")

        self._finish()


class ExtensionsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.controls = {}
        self.control_rows = {}
        self.init_ui()
        self.on_module_changed(self.module_combo.currentText())

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(430)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 8, 10, 8)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_panel)

        title = QLabel("扩展内容")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        left_layout.addWidget(title)

        module_group = QGroupBox("扩展模块")
        module_layout = QVBoxLayout(module_group)
        module_layout.setContentsMargins(12, 18, 12, 12)
        module_layout.setSpacing(8)
        self.module_combo = QComboBox()
        self.module_combo.addItems([MODULE_FOURIER, MODULE_QUANTUM])
        self.module_combo.currentTextChanged.connect(self.on_module_changed)
        module_layout.addWidget(self.module_combo)
        left_layout.addWidget(module_group)

        experiment_group = QGroupBox("实验类型")
        experiment_layout = QVBoxLayout(experiment_group)
        experiment_layout.setContentsMargins(12, 18, 12, 12)
        experiment_layout.setSpacing(8)
        self.mode_combo = QComboBox()
        self.mode_combo.currentTextChanged.connect(self.on_mode_selected)
        experiment_layout.addWidget(self.mode_combo)
        left_layout.addWidget(experiment_group)

        param_group = QGroupBox("参数")
        param_layout = QGridLayout(param_group)
        param_layout.setContentsMargins(12, 18, 12, 12)
        param_layout.setHorizontalSpacing(8)
        param_layout.setVerticalSpacing(9)

        row = 0
        row = self._add_control(param_layout, row, "振幅 A", "amplitude", 0.05, 5.0, 1.0, 0.05, 3, "")
        row = self._add_control(param_layout, row, "频率 f1", "frequency_1", 0.1, 120.0, 8.0, 0.1, 3, " Hz")
        row = self._add_control(param_layout, row, "频率 f2", "frequency_2", 0.1, 120.0, 9.0, 0.1, 3, " Hz")
        row = self._add_control(param_layout, row, "初相 φ", "phase", -np.pi, np.pi, 0.0, 0.05, 4, " rad")
        row = self._add_control(param_layout, row, "阻尼 β", "damping", 0.0, 40.0, 0.8, 0.05, 3, " s^-1")
        row = self._add_control(param_layout, row, "脉冲宽度 σ", "pulse_width", 0.01, 5.0, 0.12, 0.01, 3, " s")
        row = self._add_control(param_layout, row, "采样率 fs", "sample_rate", 20.0, 5000.0, 1000.0, 10.0, 1, " Hz")
        row = self._add_control(param_layout, row, "采样时长", "duration", 0.2, 30.0, 2.0, 0.1, 3, " s")

        row = self._add_control(param_layout, row, "波长 λ", "wavelength_nm", 10.0, 1000.0, 632.8, 0.1, 2, " nm")
        row = self._add_control(param_layout, row, "双缝间距 d", "slit_distance_um", 5.0, 800.0, 120.0, 1.0, 2, " μm")
        row = self._add_control(param_layout, row, "单缝宽度 a", "slit_width_um", 1.0, 300.0, 28.0, 1.0, 2, " μm")
        row = self._add_control(param_layout, row, "屏幕距离 L", "screen_distance_m", 0.05, 5.0, 1.2, 0.05, 3, " m")
        row = self._add_control(param_layout, row, "相干度 γ", "coherence", 0.0, 1.0, 0.95, 0.01, 3, "")
        row = self._add_control(param_layout, row, "路径信息 D", "path_info", 0.0, 1.0, 0.15, 0.01, 3, "")
        row = self._add_control(param_layout, row, "粒子数 N", "particle_count", 100.0, 10000.0, 2200.0, 100.0, 0, "")

        row = self._add_control(param_layout, row, "相位差 φ", "mz_phase", 0.0, 2.0 * np.pi, 1.1, 0.05, 4, " rad")
        row = self._add_control(param_layout, row, "分束器反射率 R", "beam_splitter_r", 0.01, 0.99, 0.5, 0.01, 3, "")
        row = self._add_control(param_layout, row, "光子数 N", "photon_count", 100.0, 20000.0, 3000.0, 100.0, 0, "")
        row = self._add_control(param_layout, row, "相位噪声 σφ", "phase_noise", 0.0, 2.0, 0.0, 0.02, 3, " rad")

        self.window_label = QLabel("窗函数")
        self.window_combo = QComboBox()
        self.window_combo.addItems(["无", "Hann", "Hamming", "Blackman"])
        param_layout.addWidget(self.window_label, row, 0)
        param_layout.addWidget(self.window_combo, row, 1)

        left_layout.addWidget(param_group)

        action_group = QGroupBox("分析")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(12, 18, 12, 12)
        action_layout.setSpacing(8)
        self.btn_load_sample = QPushButton("加载样例")
        self.btn_analyze = QPushButton("生成分析图")
        self.btn_load_sample.clicked.connect(self.load_current_sample)
        self.btn_analyze.clicked.connect(self.update_analysis)
        action_layout.addWidget(self.btn_load_sample)
        action_layout.addWidget(self.btn_analyze)
        left_layout.addWidget(action_group)

        readout_group = QGroupBox("读数")
        readout_layout = QVBoxLayout(readout_group)
        readout_layout.setContentsMargins(12, 18, 12, 12)
        self.readout_text = QTextEdit()
        self.readout_text.setReadOnly(True)
        self.readout_text.setMinimumHeight(150)
        self.readout_text.setStyleSheet(
            "QTextEdit { background:#ffffff; color:#1f2937; border:1px solid #d9d9d9; border-radius:6px; }"
        )
        readout_layout.addWidget(self.readout_text)
        left_layout.addWidget(readout_group)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 8, 10, 8)
        right_layout.setSpacing(8)

        self.heading_label = QLabel("振动频谱分析")
        self.heading_label.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        right_layout.addWidget(self.heading_label)

        self.canvas = ExtensionCanvas(self)
        right_layout.addWidget(self.canvas, 1)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1000])

        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d9d9d9;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #1f2937;
            }
            QComboBox {
                min-height: 32px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 4px 8px;
            }
            """
        )

    def _add_control(self, layout, row, label_text, name, minimum, maximum, value, step, decimals, suffix):
        label = QLabel(label_text)
        control = SliderSpinBox(
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            decimals=decimals,
            suffix=suffix,
        )
        control.setMinimumHeight(38)
        control.setKeyboardTracking(False)
        control.setAccelerated(True)
        layout.addWidget(label, row, 0)
        layout.addWidget(control, row, 1)
        self.controls[name] = control
        self.control_rows[name] = (label, control)
        return row + 1

    def _set_control_visible(self, name, visible):
        label, control = self.control_rows[name]
        label.setVisible(visible)
        control.setVisible(visible)

    def _set_window_visible(self, visible):
        self.window_label.setVisible(visible)
        self.window_combo.setVisible(visible)

    def _current_module(self):
        return self.module_combo.currentText()

    def _current_mode(self):
        return self.mode_combo.currentText()

    def on_module_changed(self, module):
        modes = FOURIER_MODES if module == MODULE_FOURIER else QUANTUM_MODES
        was_blocked = self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(modes)
        self.mode_combo.blockSignals(was_blocked)
        self.load_current_sample()

    def on_mode_selected(self, _mode):
        self.load_current_sample()

    def _update_control_visibility(self):
        module = self._current_module()
        mode = self._current_mode()
        fourier_controls = {
            "amplitude",
            "frequency_1",
            "frequency_2",
            "phase",
            "damping",
            "pulse_width",
            "sample_rate",
            "duration",
        }
        double_slit_controls = {
            "wavelength_nm",
            "slit_distance_um",
            "slit_width_um",
            "screen_distance_m",
            "coherence",
            "path_info",
            "particle_count",
        }
        mach_controls = {"mz_phase", "beam_splitter_r", "photon_count", "phase_noise"}

        for name in self.controls:
            self._set_control_visible(name, False)

        if module == MODULE_FOURIER:
            for name in fourier_controls:
                self._set_control_visible(name, True)
            self._set_control_visible("frequency_2", mode == MODE_BEAT)
            self._set_control_visible("damping", mode in DAMPING_MODES)
            self._set_control_visible("pulse_width", mode == MODE_GAUSSIAN)
            self._set_control_visible("phase", mode not in {MODE_CRITICAL_DAMPED, MODE_OVER_DAMPED})
            self._set_window_visible(True)
            self.heading_label.setText("振动频谱分析")
        elif mode == MODE_DOUBLE_SLIT:
            for name in double_slit_controls:
                self._set_control_visible(name, True)
            self._set_window_visible(False)
            self.heading_label.setText("量子双缝干涉")
        else:
            for name in mach_controls:
                self._set_control_visible(name, True)
            self._set_window_visible(False)
            self.heading_label.setText("Mach-Zehnder 单光子干涉仪")

    def _set_values(self, values):
        for name, value in values.items():
            if name == "window":
                self.window_combo.setCurrentText(value)
            elif name in self.controls:
                self.controls[name].setValue(value)

    def load_current_sample(self):
        mode = self._current_mode()
        samples = {
            MODE_SINE: {
                "amplitude": 1.0,
                "frequency_1": 40.0,
                "phase": 0.0,
                "sample_rate": 1000.0,
                "duration": 1.0,
                "window": "无",
            },
            MODE_UNDER_DAMPED: {
                "amplitude": 1.0,
                "frequency_1": 0.42,
                "phase": 0.0,
                "damping": 0.55,
                "sample_rate": 80.0,
                "duration": 20.0,
                "window": "Hann",
            },
            MODE_CRITICAL_DAMPED: {
                "amplitude": 1.0,
                "frequency_1": 0.22,
                "damping": 1.4,
                "sample_rate": 80.0,
                "duration": 20.0,
                "window": "Hann",
            },
            MODE_OVER_DAMPED: {
                "amplitude": 1.0,
                "frequency_1": 0.25,
                "damping": 2.2,
                "sample_rate": 80.0,
                "duration": 20.0,
                "window": "Hann",
            },
            MODE_SQUARE: {
                "amplitude": 1.0,
                "frequency_1": 5.0,
                "phase": 0.0,
                "sample_rate": 1000.0,
                "duration": 2.0,
                "window": "无",
            },
            MODE_TRIANGLE: {
                "amplitude": 1.0,
                "frequency_1": 5.0,
                "phase": 0.0,
                "sample_rate": 1000.0,
                "duration": 2.0,
                "window": "无",
            },
            MODE_SAWTOOTH: {
                "amplitude": 1.0,
                "frequency_1": 5.0,
                "phase": 0.0,
                "sample_rate": 1000.0,
                "duration": 2.0,
                "window": "无",
            },
            MODE_GAUSSIAN: {
                "amplitude": 1.0,
                "frequency_1": 18.0,
                "phase": 0.0,
                "pulse_width": 0.08,
                "sample_rate": 1200.0,
                "duration": 1.5,
                "window": "无",
            },
            MODE_BEAT: {
                "amplitude": 1.0,
                "frequency_1": 8.0,
                "frequency_2": 9.2,
                "phase": 0.0,
                "sample_rate": 500.0,
                "duration": 8.0,
                "window": "Hann",
            },
            MODE_DOUBLE_SLIT: {
                "wavelength_nm": 632.8,
                "slit_distance_um": 120.0,
                "slit_width_um": 28.0,
                "screen_distance_m": 1.2,
                "coherence": 0.95,
                "path_info": 0.15,
                "particle_count": 2200.0,
            },
            MODE_MACH_ZEHNDER: {
                "mz_phase": 1.1,
                "beam_splitter_r": 0.5,
                "photon_count": 3000.0,
                "phase_noise": 0.0,
            },
        }
        self._set_values(samples.get(mode, {}))
        self._update_control_visibility()
        self.update_analysis()

    def _control_value(self, name):
        return self.controls[name].value()

    def _time_axis(self):
        duration = max(self._control_value("duration"), 1e-6)
        sample_rate = max(self._control_value("sample_rate"), 1e-6)
        sample_count = int(round(duration * sample_rate))
        sample_count = max(128, min(sample_count, 65536))
        time = np.linspace(0.0, duration, sample_count, endpoint=False)
        effective_sample_rate = sample_count / duration
        return time, effective_sample_rate

    def _generate_signal(self):
        mode = self._current_mode()
        t, fs = self._time_axis()
        amplitude = self._control_value("amplitude")
        frequency_1 = self._control_value("frequency_1")
        frequency_2 = self._control_value("frequency_2")
        phase = self._control_value("phase")
        damping = self._control_value("damping")
        pulse_width = self._control_value("pulse_width")

        if mode == MODE_SINE:
            signal = amplitude * np.sin(2.0 * np.pi * frequency_1 * t + phase)
        elif mode == MODE_UNDER_DAMPED:
            omega_0 = 2.0 * np.pi * frequency_1
            beta = min(max(damping, 0.0), omega_0 * 0.98)
            omega_d = np.sqrt(max(omega_0 ** 2 - beta ** 2, 1e-12))
            signal = amplitude * np.exp(-beta * t) * np.cos(omega_d * t + phase)
        elif mode == MODE_CRITICAL_DAMPED:
            omega_0 = 2.0 * np.pi * max(frequency_1, 1e-9)
            signal = amplitude * (1.0 + omega_0 * t) * np.exp(-omega_0 * t)
        elif mode == MODE_OVER_DAMPED:
            omega_0 = 2.0 * np.pi * max(frequency_1, 1e-9)
            beta = max(damping, omega_0 * 1.05)
            root = np.sqrt(max(beta ** 2 - omega_0 ** 2, 1e-12))
            r1 = -beta + root
            r2 = -beta - root
            signal = amplitude * (0.70 * np.exp(r1 * t) + 0.30 * np.exp(r2 * t))
        elif mode == MODE_SQUARE:
            signal = amplitude * square(2.0 * np.pi * frequency_1 * t + phase)
        elif mode == MODE_TRIANGLE:
            signal = amplitude * sawtooth(2.0 * np.pi * frequency_1 * t + phase, width=0.5)
        elif mode == MODE_SAWTOOTH:
            signal = amplitude * sawtooth(2.0 * np.pi * frequency_1 * t + phase)
        elif mode == MODE_GAUSSIAN:
            center = 0.5 * t[-1]
            sigma = max(pulse_width, 1e-6)
            envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
            signal = amplitude * envelope * np.cos(2.0 * np.pi * frequency_1 * (t - center) + phase)
        elif mode == MODE_BEAT:
            signal = 0.5 * amplitude * (
                np.sin(2.0 * np.pi * frequency_1 * t + phase)
                + np.sin(2.0 * np.pi * frequency_2 * t + phase)
            )
        else:
            signal = amplitude * np.sin(2.0 * np.pi * frequency_1 * t + phase)

        return t, signal.astype(float), fs

    def _window(self, length):
        window_name = self.window_combo.currentText()
        if window_name == "Hann":
            return np.hanning(length)
        if window_name == "Hamming":
            return np.hamming(length)
        if window_name == "Blackman":
            return np.blackman(length)
        return np.ones(length)

    def _reconstruct_by_top_components(self, signal, counts):
        fft_raw = np.fft.fft(signal)
        freqs = np.fft.fftfreq(signal.size)
        positive_indices = np.flatnonzero(freqs > 0)
        ranked = positive_indices[np.argsort(np.abs(fft_raw[positive_indices]))[::-1]]
        reconstructions = {}

        for count in counts:
            filtered = np.zeros_like(fft_raw)
            filtered[0] = fft_raw[0]
            for index in ranked[: min(count, ranked.size)]:
                filtered[index] = fft_raw[index]
                filtered[-index] = fft_raw[-index]
            reconstructions[count] = np.fft.ifft(filtered).real

        return reconstructions

    def _find_main_peaks(self, positive_freqs, positive_amp):
        if positive_freqs.size < 3:
            return [], 0.0, None

        valid = positive_freqs > 0
        if not np.any(valid):
            return [], 0.0, None

        amp_for_peaks = positive_amp.copy()
        amp_for_peaks[~valid] = 0.0
        max_amp = float(np.max(amp_for_peaks))
        if max_amp <= 0:
            return [], 0.0, None

        prominence = max(max_amp * 0.04, float(np.std(amp_for_peaks[valid])) * 0.45, 1e-12)
        distance = max(2, int(round(positive_freqs.size * 0.004)))
        peak_indices, _ = find_peaks(amp_for_peaks, prominence=prominence, distance=distance)
        peak_indices = [idx for idx in peak_indices if positive_freqs[idx] > 0]

        if not peak_indices:
            peak_indices = [int(np.argmax(amp_for_peaks))]

        peak_indices = sorted(peak_indices, key=lambda idx: positive_amp[idx], reverse=True)
        top_peaks = [
            {
                "index": int(index),
                "frequency": float(positive_freqs[index]),
                "amplitude": float(positive_amp[index]),
            }
            for index in peak_indices[:8]
        ]
        dominant = top_peaks[0]["frequency"] if top_peaks else 0.0
        return top_peaks, dominant, peak_indices[0] if peak_indices else None

    def _estimate_fwhm(self, freqs, amp, peak_index):
        if peak_index is None or peak_index <= 0 or peak_index >= len(amp) - 1:
            return None

        peak_amp = amp[peak_index]
        if peak_amp <= 0:
            return None

        half = peak_amp * 0.5
        left = peak_index
        while left > 0 and amp[left] >= half:
            left -= 1
        right = peak_index
        while right < len(amp) - 1 and amp[right] >= half:
            right += 1

        if right <= left:
            return None
        width = float(freqs[right] - freqs[left])
        if width <= 0:
            return None
        return width

    def _analyze_signal(self):
        time, signal, sample_rate = self._generate_signal()
        n = signal.size
        window = self._window(n)
        coherent_gain = max(float(np.mean(window)), 1e-9)
        windowed_signal = signal * window

        fft_values = np.fft.fft(windowed_signal)
        frequencies = np.fft.fftfreq(n, d=1.0 / sample_rate)
        amp = np.abs(fft_values) / (n * coherent_gain)
        freq_shift = np.fft.fftshift(frequencies)
        amp_shift = np.fft.fftshift(amp)

        positive_mask = frequencies >= 0
        positive_freqs = frequencies[positive_mask]
        positive_amp = amp[positive_mask].copy()
        if positive_amp.size > 2:
            positive_amp[1:-1] *= 2.0

        top_peaks, dominant_freq, peak_index = self._find_main_peaks(positive_freqs, positive_amp)
        fwhm = self._estimate_fwhm(positive_freqs, positive_amp, peak_index)
        reconstructions = self._reconstruct_by_top_components(signal, (3, 10, 50))

        return {
            "time": time,
            "signal": signal,
            "sample_rate": sample_rate,
            "freq_shift": freq_shift,
            "amp_shift": amp_shift,
            "positive_freqs": positive_freqs,
            "positive_amp": positive_amp,
            "top_peaks": top_peaks,
            "dominant_freq": dominant_freq,
            "fwhm": fwhm,
            "reconstructions": reconstructions,
            "title": self._current_mode(),
            "window": self.window_combo.currentText(),
        }

    def _format_fourier_readout(self, result):
        dominant = result["dominant_freq"]
        period = 1.0 / dominant if dominant > 1e-12 else 0.0
        omega = 2.0 * np.pi * dominant
        n = result["signal"].size
        sample_rate = result["sample_rate"]
        resolution = sample_rate / n

        lines = [
            f"模式：{result['title']}",
            f"N = {n}，fs = {sample_rate:.3f} Hz，Δf = {resolution:.5f} Hz",
            f"主频 f0 = {dominant:.5f} Hz",
            f"角频率 ω = {omega:.5f} rad/s",
            f"周期 T = {period:.5f} s" if period > 0 else "周期 T = --",
            f"窗函数：{result['window']}",
        ]

        if result["fwhm"] and dominant > 0:
            q_value = dominant / result["fwhm"]
            lines.append(f"谱峰半高宽 FWHM = {result['fwhm']:.5f} Hz，Q ≈ {q_value:.3f}")

        mode = self._current_mode()
        if mode == MODE_BEAT:
            f1 = self._control_value("frequency_1")
            f2 = self._control_value("frequency_2")
            lines.append(f"设定拍频 Δf = |f2 - f1| = {abs(f2 - f1):.5f} Hz")

        if self._control_value("frequency_1") > sample_rate / 2:
            lines.append("提示：f1 超过奈奎斯特频率，频谱会出现混叠。")
        if mode == MODE_BEAT and self._control_value("frequency_2") > sample_rate / 2:
            lines.append("提示：f2 超过奈奎斯特频率，频谱会出现混叠。")

        if result["top_peaks"]:
            lines.append("")
            lines.append("主要频率峰：")
            for peak in result["top_peaks"][:5]:
                lines.append(f"  f = {peak['frequency']:.5f} Hz，A = {peak['amplitude']:.6g}")

        return "\n".join(lines)

    def _analyze_double_slit(self):
        wavelength = self._control_value("wavelength_nm") * 1e-9
        slit_distance = self._control_value("slit_distance_um") * 1e-6
        slit_width = self._control_value("slit_width_um") * 1e-6
        screen_distance = self._control_value("screen_distance_m")
        coherence = self._control_value("coherence")
        distinguishability = self._control_value("path_info")
        particle_count = int(round(self._control_value("particle_count")))

        fringe_spacing = wavelength * screen_distance / max(slit_distance, 1e-12)
        screen_half_width = max(0.010, min(0.050, fringe_spacing * 7.5))
        x = np.linspace(-screen_half_width, screen_half_width, 2600)
        alpha = np.pi * slit_width * x / max(wavelength * screen_distance, 1e-18)
        beta = np.pi * slit_distance * x / max(wavelength * screen_distance, 1e-18)
        envelope = np.sinc(alpha / np.pi) ** 2
        visibility = coherence * np.sqrt(max(0.0, 1.0 - distinguishability ** 2))
        probability = envelope * (1.0 + visibility * np.cos(2.0 * beta))
        probability = np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
        probability = np.clip(probability, 0.0, None)
        area = self.canvas._integrate(probability, x)
        if not np.isfinite(area) or area <= 1e-18 or float(np.sum(probability)) <= 1e-18:
            probability = np.ones_like(x, dtype=float)
            area = self.canvas._integrate(probability, x)
        probability /= max(area, 1e-18)

        discrete_probability = probability / np.sum(probability)
        discrete_probability = np.nan_to_num(discrete_probability, nan=0.0, posinf=0.0, neginf=0.0)
        probability_sum = float(np.sum(discrete_probability))
        if not np.isfinite(probability_sum) or probability_sum <= 0:
            discrete_probability = np.full_like(probability, 1.0 / probability.size)
        else:
            discrete_probability /= probability_sum
        rng = np.random.default_rng(20260610)
        draw_count = max(100, min(particle_count, 10000))
        hits = rng.choice(x, size=draw_count, p=discrete_probability)
        hit_rows = rng.random(draw_count)

        amp_envelope = np.sqrt(np.maximum(envelope, 0.0))
        psi = amp_envelope * (np.exp(1j * beta) + visibility * np.exp(-1j * beta)) / np.sqrt(2.0)
        psi /= max(float(np.max(np.abs(psi))), 1e-12)

        return {
            "x": x,
            "probability": probability,
            "envelope": envelope,
            "hits": hits,
            "hit_rows": hit_rows,
            "psi": psi,
            "visibility": visibility,
            "distinguishability": distinguishability,
            "fringe_spacing": fringe_spacing,
            "coherence": coherence,
            "particle_count": draw_count,
        }

    def _format_double_slit_readout(self, result):
        v = result["visibility"]
        d = result["distinguishability"]
        return "\n".join(
            [
                "模式：量子双缝干涉",
                f"条纹间距 Δx = λL/d = {result['fringe_spacing'] * 1000.0:.4f} mm",
                f"相干度 γ = {result['coherence']:.3f}",
                f"路径可区分度 D = {d:.3f}",
                f"有效可见度 V = γ√(1-D²) = {v:.3f}",
                f"互补关系 V^2 + D^2 = {v ** 2 + d ** 2:.3f}",
                f"单粒子累积数 N = {result['particle_count']}",
            ]
        )

    def _mach_probabilities_for_phase(self, phase_values, reflectivity):
        r = np.sqrt(reflectivity)
        t = np.sqrt(max(0.0, 1.0 - reflectivity))
        bs = np.array([[t, 1j * r], [1j * r, t]], dtype=complex)
        input_state = np.array([1.0 + 0.0j, 0.0 + 0.0j])

        phase_values = np.asarray(phase_values, dtype=float)
        p0 = np.zeros_like(phase_values)
        p1 = np.zeros_like(phase_values)
        amplitudes = []
        for index, phase in np.ndenumerate(phase_values):
            phase_matrix = np.array([[1.0, 0.0], [0.0, np.exp(1j * phase)]], dtype=complex)
            out = bs @ phase_matrix @ bs @ input_state
            norm = max(float(np.sum(np.abs(out) ** 2)), 1e-12)
            p0[index] = float(np.abs(out[0]) ** 2 / norm)
            p1[index] = float(np.abs(out[1]) ** 2 / norm)
            if phase_values.size == 1:
                amplitudes = [out[0] / np.sqrt(norm), out[1] / np.sqrt(norm)]
        return p0, p1, amplitudes

    def _analyze_mach_zehnder(self):
        phase = self._control_value("mz_phase")
        reflectivity = self._control_value("beam_splitter_r")
        photon_count = int(round(self._control_value("photon_count")))
        phase_noise = self._control_value("phase_noise")

        phase_grid = np.linspace(0.0, 2.0 * np.pi, 420)
        curve_p0, curve_p1, _ = self._mach_probabilities_for_phase(phase_grid, reflectivity)
        p0_single, p1_single, amplitudes = self._mach_probabilities_for_phase(np.array([phase]), reflectivity)
        p0 = float(p0_single[0])
        p1 = float(p1_single[0])

        rng = np.random.default_rng(20260611)
        if phase_noise > 1e-9:
            noisy_phases = phase + rng.normal(0.0, phase_noise, 3200)
            noisy_p0, noisy_p1, _ = self._mach_probabilities_for_phase(noisy_phases, reflectivity)
            p0 = float(np.mean(noisy_p0))
            p1 = float(np.mean(noisy_p1))

        total = max(p0 + p1, 1e-12)
        p0 /= total
        p1 /= total
        clicks = rng.multinomial(max(1, photon_count), [p0, p1])

        ideal_visibility = 2.0 * reflectivity * (1.0 - reflectivity) / max(
            reflectivity ** 2 + (1.0 - reflectivity) ** 2,
            1e-12,
        )
        visibility = ideal_visibility * np.exp(-0.5 * phase_noise ** 2)

        return {
            "phase_grid": phase_grid,
            "curve_p0": curve_p0,
            "curve_p1": curve_p1,
            "phase": phase,
            "p0": p0,
            "p1": p1,
            "clicks": clicks,
            "amplitudes": amplitudes,
            "reflectivity": reflectivity,
            "photon_count": photon_count,
            "phase_noise": phase_noise,
            "visibility": visibility,
        }

    def _format_mach_readout(self, result):
        return "\n".join(
            [
                "模式：Mach-Zehnder 单光子干涉仪",
                f"相位差 φ = {result['phase']:.4f} rad",
                f"分束器反射率 R = {result['reflectivity']:.3f}",
                f"相位噪声 σφ = {result['phase_noise']:.3f} rad",
                f"P(D0) = {result['p0']:.5f}",
                f"P(D1) = {result['p1']:.5f}",
                f"干涉可见度 V ≈ {result['visibility']:.5f}",
                f"点击统计：D0 = {int(result['clicks'][0])}，D1 = {int(result['clicks'][1])}",
            ]
        )

    def update_analysis(self):
        try:
            module = self._current_module()
            mode = self._current_mode()
            self._update_control_visibility()
            if module == MODULE_FOURIER:
                result = self._analyze_signal()
                self.canvas.plot_fourier(result)
                self.readout_text.setPlainText(self._format_fourier_readout(result))
            elif mode == MODE_DOUBLE_SLIT:
                result = self._analyze_double_slit()
                self.canvas.plot_double_slit(result)
                self.readout_text.setPlainText(self._format_double_slit_readout(result))
            else:
                result = self._analyze_mach_zehnder()
                self.canvas.plot_mach_zehnder(result)
                self.readout_text.setPlainText(self._format_mach_readout(result))
        except Exception as exc:
            self.readout_text.setPlainText(
                "扩展模块生成失败。\n\n"
                f"错误信息：{exc}\n\n"
                "请点击“加载样例”恢复默认参数后再试。"
            )
