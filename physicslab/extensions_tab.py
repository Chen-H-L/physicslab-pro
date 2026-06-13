from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks, sawtooth, square

from PyQt6.QtCore import Qt, QTimer
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


ASSET_DIR = Path(__file__).resolve().parent / "assets"
DOUBLE_SLIT_MODEL_IMAGE = ASSET_DIR / "double_slit_theory_model.png"
PHOTOELECTRIC_MODEL_IMAGE = ASSET_DIR / "photoelectric_theory_model.png"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MODULE_FOURIER = "振动频谱分析 / 傅里叶变换实验室"
MODULE_QUANTUM = "量子干涉实验室"
MODULE_PHOTOELECTRIC = "光电效应实验室"

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
MODE_PHOTOELECTRIC = "光电效应方程与伏安特性"

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
QUANTUM_MODES = [MODE_DOUBLE_SLIT]
PHOTOELECTRIC_MODES = [MODE_PHOTOELECTRIC]
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

    def _show_model_image(self, ax, image_path, title):
        ax.set_facecolor("#ffffff")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, color="#0f172a", pad=4)
        if image_path.exists():
            image = mpimg.imread(str(image_path))
            ax.imshow(image)
            ax.set_anchor("C")
        else:
            ax.text(
                0.5,
                0.5,
                "理论模型图资源缺失",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#b91c1c",
            )

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
        self.fig.clear()
        grid = self.fig.add_gridspec(
            2,
            3,
            height_ratios=[1.35, 1.0],
            left=0.055,
            right=0.985,
            bottom=0.075,
            top=0.955,
            hspace=0.34,
            wspace=0.30,
        )
        ax_model = self.fig.add_subplot(grid[0, :])
        ax_prob = self.fig.add_subplot(grid[1, 0])
        ax_hits = self.fig.add_subplot(grid[1, 1])
        ax_amp = self.fig.add_subplot(grid[1, 2])
        self.axes = np.asarray([ax_model, ax_prob, ax_hits, ax_amp], dtype=object)
        self._show_model_image(ax_model, DOUBLE_SLIT_MODEL_IMAGE, "实验原理理论模型图")
        for ax in (ax_prob, ax_hits, ax_amp):
            self._style_axis(ax)

        x_mm = result["x"] * 1000.0
        probability = result["probability"]
        envelope = result["envelope"]
        hits_mm = result["hits"] * 1000.0
        y_hits = result["hit_rows"]
        psi = result["psi"]

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

        self.draw_idle()

    def plot_photoelectric(self, result):
        self.fig.clear()
        grid = self.fig.add_gridspec(
            2,
            3,
            height_ratios=[1.25, 1.0],
            left=0.055,
            right=0.985,
            bottom=0.075,
            top=0.955,
            hspace=0.34,
            wspace=0.30,
        )
        ax_model = self.fig.add_subplot(grid[0, :])
        ax_u0 = self.fig.add_subplot(grid[1, 0])
        ax_is = self.fig.add_subplot(grid[1, 1])
        ax_iv = self.fig.add_subplot(grid[1, 2])
        self.axes = np.asarray([ax_model, ax_u0, ax_is, ax_iv], dtype=object)
        self._show_model_image(ax_model, PHOTOELECTRIC_MODEL_IMAGE, "实验原理理论模型图")
        for ax in (ax_u0, ax_is, ax_iv):
            self._style_axis(ax)

        frequency = float(result["frequency_thz"])
        work_function = float(result["work_function_ev"])
        stopping_voltage = float(result["stopping_voltage"])
        intensity = float(result["intensity"])
        saturation_current = float(result["saturation_current_ua"])
        selected_voltage = float(result["voltage"])
        selected_current = float(result["current_at_voltage"])
        h_ev_per_thz = 4.135667696e-3

        cutoff = float(result["cutoff_frequency_thz"])
        freq_left = max(50.0, min(cutoff, frequency) * 0.55)
        freq_right = max(1200.0, frequency * 1.20, cutoff * 1.65)
        frequency_scan = np.linspace(freq_left, freq_right, 560)
        u0_scan = np.maximum(h_ev_per_thz * frequency_scan - work_function, 0.0)
        u0_top = max(float(np.max(u0_scan)) * 1.12, stopping_voltage * 1.25, 0.8)
        ax_u0.plot(frequency_scan, u0_scan, color="#1d4ed8", linewidth=1.7, label="$U_0=h\\nu-W_0$")
        ax_u0.scatter([frequency], [stopping_voltage], color="#dc2626", s=42, zorder=5, label="当前条件")
        ax_u0.plot([frequency, frequency], [0.0, stopping_voltage], color="#dc2626", linestyle="--", linewidth=1.1)
        ax_u0.plot([freq_left, frequency], [stopping_voltage, stopping_voltage], color="#dc2626", linestyle="--", linewidth=1.1)
        ax_u0.axvline(cutoff, color="#64748b", linestyle=":", linewidth=1.0, label="截止频率")
        ax_u0.set_xlim(freq_left, freq_right)
        ax_u0.set_ylim(0.0, u0_top)
        ax_u0.set_title("遏止电压与入射频率", fontsize=11, color="#0f172a")
        ax_u0.set_xlabel("入射频率 ν (THz)", color="#334155")
        ax_u0.set_ylabel("遏止电压 U0 (V)", color="#334155")
        ax_u0.legend(loc="upper left", fontsize=8, framealpha=0.9)

        intensity_scan = np.linspace(0.0, 10.0, 420)
        current_per_intensity = 3.5 if result["emission"] else 0.0
        saturation_scan = current_per_intensity * intensity_scan
        is_top = max(float(np.max(saturation_scan)) * 1.12, saturation_current * 1.25, 1.0)
        ax_is.plot(intensity_scan, saturation_scan, color="#2563eb", linewidth=1.7, label="$I_s \\propto$ 光强")
        ax_is.scatter([intensity], [saturation_current], color="#dc2626", s=42, zorder=5, label="当前条件")
        ax_is.plot([intensity, intensity], [0.0, saturation_current], color="#dc2626", linestyle="--", linewidth=1.1)
        ax_is.plot([0.0, intensity], [saturation_current, saturation_current], color="#dc2626", linestyle="--", linewidth=1.1)
        if not result["emission"]:
            ax_is.text(0.5, 0.62, "未达到截止频率", transform=ax_is.transAxes, ha="center", color="#b91c1c", fontsize=10)
        ax_is.set_xlim(0.0, 10.0)
        ax_is.set_ylim(0.0, is_top)
        ax_is.set_title("光强与饱和光电流", fontsize=11, color="#0f172a")
        ax_is.set_xlabel("光强 I", color="#334155")
        ax_is.set_ylabel("饱和光电流 Is (μA)", color="#334155")
        ax_is.legend(loc="upper left", fontsize=8, framealpha=0.9)

        voltage = result["voltage_grid"]
        current = result["current_grid"]
        iv_top = max(float(np.max(current)) * 1.16 if len(current) else 0.0, selected_current * 1.25, 1.0)
        ax_iv.plot(voltage, current, color="#1d4ed8", linewidth=1.7, label="伏安特性")
        ax_iv.scatter([selected_voltage], [selected_current], color="#dc2626", s=42, zorder=5, label="当前工作点")
        ax_iv.plot([selected_voltage, selected_voltage], [0.0, selected_current], color="#dc2626", linestyle="--", linewidth=1.1)
        ax_iv.plot([float(voltage[0]), selected_voltage], [selected_current, selected_current], color="#dc2626", linestyle="--", linewidth=1.1)
        if result["emission"]:
            ax_iv.axvline(-stopping_voltage, color="#64748b", linestyle=":", linewidth=1.0, label="遏止电压")
        ax_iv.set_xlim(float(voltage[0]), float(voltage[-1]))
        ax_iv.set_ylim(0.0, iv_top)
        ax_iv.set_title("光电流与外加电压", fontsize=11, color="#0f172a")
        ax_iv.set_xlabel("外加电压 U (V)", color="#334155")
        ax_iv.set_ylabel("光电流 I (μA)", color="#334155")
        ax_iv.legend(loc="lower right", fontsize=8, framealpha=0.9)

        self.draw_idle()


class ExtensionsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.controls = {}
        self.control_rows = {}
        self._suspend_auto_update = False
        self._analysis_update_timer = QTimer(self)
        self._analysis_update_timer.setSingleShot(True)
        self._analysis_update_timer.setInterval(120)
        self._analysis_update_timer.timeout.connect(self.update_analysis)
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
        self.module_combo.addItems([MODULE_FOURIER, MODULE_QUANTUM, MODULE_PHOTOELECTRIC])
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

        row = self._add_control(param_layout, row, "入射频率 ν", "pe_frequency_thz", 100.0, 1200.0, 650.0, 1.0, 1, " THz")
        row = self._add_control(param_layout, row, "逸出功 W₀", "pe_work_function_ev", 1.0, 6.0, 2.28, 0.01, 3, " eV")
        row = self._add_control(param_layout, row, "光强 I", "pe_intensity", 0.1, 10.0, 3.0, 0.1, 2, "")
        row = self._add_control(param_layout, row, "外加电压 U", "pe_voltage_v", -5.0, 5.0, 0.0, 0.05, 3, " V")

        self.window_label = QLabel("窗函数")
        self.window_combo = QComboBox()
        self.window_combo.addItems(["无", "Hann", "Hamming", "Blackman"])
        self.window_combo.currentTextChanged.connect(self.schedule_analysis_update)
        param_layout.addWidget(self.window_label, row, 0)
        param_layout.addWidget(self.window_combo, row, 1)

        left_layout.addWidget(param_group)

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(12, 18, 12, 12)
        action_layout.setSpacing(8)
        self.btn_load_sample = QPushButton("加载样例")
        self.btn_load_sample.clicked.connect(self.load_current_sample)
        action_layout.addWidget(self.btn_load_sample)
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
        control.valueChanged.connect(self.schedule_analysis_update)
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

    def schedule_analysis_update(self, *_args):
        if self._suspend_auto_update:
            return
        self._analysis_update_timer.start()

    def _current_module(self):
        return self.module_combo.currentText()

    def _current_mode(self):
        return self.mode_combo.currentText()

    def on_module_changed(self, module):
        module_modes = {
            MODULE_FOURIER: FOURIER_MODES,
            MODULE_QUANTUM: QUANTUM_MODES,
            MODULE_PHOTOELECTRIC: PHOTOELECTRIC_MODES,
        }
        modes = module_modes.get(module, FOURIER_MODES)
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
        photoelectric_controls = {
            "pe_frequency_thz",
            "pe_work_function_ev",
            "pe_intensity",
            "pe_voltage_v",
        }

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
            for name in photoelectric_controls:
                self._set_control_visible(name, True)
            self._set_window_visible(False)
            self.heading_label.setText("光电效应实验室")

    def _set_values(self, values):
        self._suspend_auto_update = True
        try:
            for name, value in values.items():
                if name == "window":
                    self.window_combo.setCurrentText(value)
                elif name in self.controls:
                    self.controls[name].setValue(value)
        finally:
            self._suspend_auto_update = False

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
            MODE_PHOTOELECTRIC: {
                "pe_frequency_thz": 650.0,
                "pe_work_function_ev": 2.28,
                "pe_intensity": 3.0,
                "pe_voltage_v": 0.0,
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

    def _photoelectric_current(self, voltage, stopping_voltage, saturation_current):
        voltage = np.asarray(voltage, dtype=float)
        if saturation_current <= 0.0 or stopping_voltage <= 0.0:
            return np.zeros_like(voltage)

        collection_width = 1.6
        normalized = (voltage + stopping_voltage) / max(stopping_voltage + collection_width, 1e-9)
        normalized = np.clip(normalized, 0.0, 1.0)
        return saturation_current * normalized ** 1.35

    def _analyze_photoelectric(self):
        h_ev_s = 4.135667696e-15
        c = 299792458.0

        frequency_thz = self._control_value("pe_frequency_thz")
        frequency_hz = frequency_thz * 1e12
        work_function_ev = self._control_value("pe_work_function_ev")
        intensity = self._control_value("pe_intensity")
        voltage = self._control_value("pe_voltage_v")

        photon_energy_ev = h_ev_s * frequency_hz
        kinetic_energy_ev = max(photon_energy_ev - work_function_ev, 0.0)
        emission = kinetic_energy_ev > 1e-9
        stopping_voltage = kinetic_energy_ev
        cutoff_frequency_thz = work_function_ev / h_ev_s / 1e12
        wavelength_nm = c / max(frequency_hz, 1e-12) * 1e9
        saturation_current_ua = 3.5 * intensity if emission else 0.0

        left_voltage = -max(5.0, stopping_voltage + 0.8)
        right_voltage = 5.0
        voltage_grid = np.linspace(left_voltage, right_voltage, 640)
        current_grid = self._photoelectric_current(voltage_grid, stopping_voltage, saturation_current_ua)
        current_at_voltage = float(self._photoelectric_current(np.array([voltage]), stopping_voltage, saturation_current_ua)[0])

        scan_left = max(80.0, cutoff_frequency_thz * 0.45)
        scan_right = max(1200.0, frequency_thz * 1.16, cutoff_frequency_thz * 1.65)
        frequency_scan_thz = np.linspace(scan_left, scan_right, 720)
        kinetic_scan_ev = np.maximum(h_ev_s * frequency_scan_thz * 1e12 - work_function_ev, 0.0)

        return {
            "frequency_thz": frequency_thz,
            "frequency_hz": frequency_hz,
            "wavelength_nm": wavelength_nm,
            "work_function_ev": work_function_ev,
            "intensity": intensity,
            "voltage": voltage,
            "photon_energy_ev": photon_energy_ev,
            "kinetic_energy_ev": kinetic_energy_ev,
            "emission": emission,
            "stopping_voltage": stopping_voltage,
            "cutoff_frequency_thz": cutoff_frequency_thz,
            "cutoff_wavelength_nm": c / max(cutoff_frequency_thz * 1e12, 1e-12) * 1e9,
            "saturation_current_ua": saturation_current_ua,
            "current_at_voltage": current_at_voltage,
            "voltage_grid": voltage_grid,
            "current_grid": current_grid,
            "frequency_scan_thz": frequency_scan_thz,
            "kinetic_scan_ev": kinetic_scan_ev,
        }

    def _format_photoelectric_readout(self, result):
        status = "已发生光电效应" if result["emission"] else "未发生光电效应"
        lines = [
            "模式：光电效应方程与伏安特性",
            f"入射频率 ν = {result['frequency_thz']:.3f} THz，波长 λ = {result['wavelength_nm']:.3f} nm",
            f"光子能量 hν = {result['photon_energy_ev']:.5f} eV",
            f"逸出功 W₀ = {result['work_function_ev']:.5f} eV",
            f"截止频率 ν0 = {result['cutoff_frequency_thz']:.3f} THz，截止波长 λ0 = {result['cutoff_wavelength_nm']:.3f} nm",
            f"最大初动能 Kmax = {result['kinetic_energy_ev']:.5f} eV",
            f"遏止电压 Us = {result['stopping_voltage']:.5f} V",
            f"饱和光电流 Is ≈ {result['saturation_current_ua']:.5f} μA",
            f"当前电压 U = {result['voltage']:.3f} V，光电流 I ≈ {result['current_at_voltage']:.5f} μA",
            f"判据：{status}",
        ]
        if not result["emission"]:
            lines.append("提示：提高入射频率或降低金属逸出功后，光电子才会逸出。")
        return "\n".join(lines)

    def update_analysis(self):
        try:
            self._analysis_update_timer.stop()
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
                result = self._analyze_photoelectric()
                self.canvas.plot_photoelectric(result)
                self.readout_text.setPlainText(self._format_photoelectric_readout(result))
        except Exception as exc:
            self.readout_text.setPlainText(
                "扩展模块生成失败。\n\n"
                f"错误信息：{exc}\n\n"
                "请点击“加载样例”恢复默认参数后再试。"
            )
