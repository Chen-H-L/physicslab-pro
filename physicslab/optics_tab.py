import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms import (
    analyze_interference_pattern,
    count_peaks_in_signal,
    extract_center_intensity,
    smooth_signal,
)

from .simulation import SimulationWidget
from .widgets import ClickableImageLabel, MatplotlibCanvas
from .workers import VideoWorker


class OpticsLabTab(QWidget):
    """页面 1: 光学 AI 实验室 (AI Optics Lab)"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None  # 当前加载的图像
        self.video_path = None  # 当前视频路径
        self.video_worker = None  # 视频处理线程
        self.video_first_frame = None  # 视频第一帧
        self.intensities_raw = []  # 原始亮度数据
        self.intensities_smoothed = []  # 平滑后的亮度数据
        self.frame_numbers = []  # 帧号数组
        self.peak_count = 0  # 波峰计数
        self.init_ui()
    
    def init_ui(self):
        """初始化界面布局：类似虚拟仿真实验室的布局"""
        main_layout = QHBoxLayout()
        
        # 创建 QSplitter 进行左右分屏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：控制面板
        left_panel = QFrame()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # 实验类型选择
        experiment_label = QLabel("实验类型:")
        experiment_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(experiment_label)
        
        self.experiment_combo = QComboBox()
        self.experiment_combo.addItems(["分析图片", "分析视频", "虚拟仿真"])
        self.experiment_combo.currentTextChanged.connect(self.on_experiment_changed)
        left_layout.addWidget(self.experiment_combo)
        
        left_layout.addSpacing(20)
        
        # 图片分析控制组
        self.image_control_group = QGroupBox("图片分析控制")
        image_control_layout = QVBoxLayout()
        
        self.btn_load_image = QPushButton("加载图片")
        self.btn_load_image.clicked.connect(self.on_load_image)
        image_control_layout.addWidget(self.btn_load_image)
        
        self.btn_start_analysis = QPushButton("开始分析")
        self.btn_start_analysis.clicked.connect(self.on_start_analysis)
        image_control_layout.addWidget(self.btn_start_analysis)
        
        self.btn_clear_points = QPushButton("清除选择")
        self.btn_clear_points.clicked.connect(self.on_clear_points)
        image_control_layout.addWidget(self.btn_clear_points)
        
        self.image_control_group.setLayout(image_control_layout)
        left_layout.addWidget(self.image_control_group)
        
        # 视频分析控制组
        self.video_control_group = QGroupBox("视频分析控制")
        video_control_layout = QVBoxLayout()
        
        self.btn_load_video = QPushButton("加载视频")
        self.btn_load_video.clicked.connect(self.on_load_video)
        video_control_layout.addWidget(self.btn_load_video)
        
        self.btn_start_stop = QPushButton("开始/停止")
        self.btn_start_stop.clicked.connect(self.on_start_stop)
        video_control_layout.addWidget(self.btn_start_stop)
        
        # 进度条（用于视频处理）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        video_control_layout.addWidget(self.progress_bar)
        
        self.video_control_group.setLayout(video_control_layout)
        left_layout.addWidget(self.video_control_group)
        
        # 虚拟仿真控制组
        self.simulation_control_group = QGroupBox("虚拟仿真控制")
        simulation_control_layout = QVBoxLayout()
        
        # 仿真实验类型选择
        sim_experiment_label = QLabel("仿真实验:")
        simulation_control_layout.addWidget(sim_experiment_label)
        
        self.sim_experiment_combo = QComboBox()
        self.sim_experiment_combo.addItems(["牛顿环实验", "劈尖干涉", "双缝干涉"])
        self.sim_experiment_combo.currentTextChanged.connect(self.on_sim_experiment_changed)
        simulation_control_layout.addWidget(self.sim_experiment_combo)
        
        # 波长参数
        wavelength_layout = QHBoxLayout()
        wavelength_label = QLabel("波长 (nm):")
        wavelength_label.setMinimumWidth(100)
        wavelength_layout.addWidget(wavelength_label)
        
        self.wavelength_spinbox = QDoubleSpinBox()
        self.wavelength_spinbox.setMinimum(400.0)
        self.wavelength_spinbox.setMaximum(700.0)
        self.wavelength_spinbox.setValue(632.8)
        self.wavelength_spinbox.setDecimals(1)
        self.wavelength_spinbox.setSuffix(" nm")
        self.wavelength_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        wavelength_layout.addWidget(self.wavelength_spinbox)
        
        simulation_control_layout.addLayout(wavelength_layout)
        
        # 牛顿环参数：曲率半径
        self.radius_layout = QHBoxLayout()
        radius_label = QLabel("曲率半径 (mm):")
        radius_label.setMinimumWidth(100)
        self.radius_layout.addWidget(radius_label)
        
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(500.0)
        self.radius_spinbox.setMaximum(5000.0)
        self.radius_spinbox.setValue(1000.0)
        self.radius_spinbox.setDecimals(1)
        self.radius_spinbox.setSuffix(" mm")
        self.radius_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        self.radius_layout.addWidget(self.radius_spinbox)
        
        simulation_control_layout.addLayout(self.radius_layout)
        
        # 牛顿环和劈尖共用：间隙距离
        self.gap_layout = QHBoxLayout()
        gap_label = QLabel("间隙距离 (nm):")
        gap_label.setMinimumWidth(100)
        self.gap_layout.addWidget(gap_label)
        
        self.gap_spinbox = QDoubleSpinBox()
        self.gap_spinbox.setMinimum(0.0)
        self.gap_spinbox.setMaximum(2000.0)
        self.gap_spinbox.setValue(0.0)
        self.gap_spinbox.setDecimals(1)
        self.gap_spinbox.setSuffix(" nm")
        self.gap_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        self.gap_layout.addWidget(self.gap_spinbox)
        
        simulation_control_layout.addLayout(self.gap_layout)
        
        # 视野缩放
        self.scale_layout = QHBoxLayout()
        scale_label = QLabel("视野范围 (mm):")
        scale_label.setMinimumWidth(100)
        self.scale_layout.addWidget(scale_label)
        
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setMinimum(0.1)
        self.scale_spinbox.setMaximum(100.0)
        self.scale_spinbox.setValue(5.0)
        self.scale_spinbox.setDecimals(2)
        self.scale_spinbox.setSuffix(" mm")
        self.scale_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        self.scale_layout.addWidget(self.scale_spinbox)
        
        simulation_control_layout.addLayout(self.scale_layout)
        
        # 劈尖干涉参数：夹角
        self.angle_layout = QHBoxLayout()
        angle_label = QLabel("劈尖夹角 (度):")
        angle_label.setMinimumWidth(100)
        self.angle_layout.addWidget(angle_label)
        
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setMinimum(0.01)
        self.angle_spinbox.setMaximum(10.0)
        self.angle_spinbox.setValue(0.057)
        self.angle_spinbox.setDecimals(3)
        self.angle_spinbox.setSuffix(" °")
        self.angle_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        self.angle_layout.addWidget(self.angle_spinbox)
        
        simulation_control_layout.addLayout(self.angle_layout)
        
        # 双缝干涉参数：缝宽
        self.slit_width_layout = QHBoxLayout()
        slit_width_label = QLabel("缝宽 (μm):")
        slit_width_label.setMinimumWidth(100)
        self.slit_width_layout.addWidget(slit_width_label)
        
        self.slit_width_spinbox = QDoubleSpinBox()
        self.slit_width_spinbox.setMinimum(1.0)
        self.slit_width_spinbox.setMaximum(100.0)
        self.slit_width_spinbox.setValue(10.0)
        self.slit_width_spinbox.setDecimals(1)
        self.slit_width_spinbox.setSuffix(" μm")
        self.slit_width_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        self.slit_width_layout.addWidget(self.slit_width_spinbox)
        
        simulation_control_layout.addLayout(self.slit_width_layout)
        
        # 双缝干涉参数：缝间距
        self.slit_spacing_layout = QHBoxLayout()
        slit_spacing_label = QLabel("缝间距 (μm):")
        slit_spacing_label.setMinimumWidth(100)
        self.slit_spacing_layout.addWidget(slit_spacing_label)
        
        self.slit_spacing_spinbox = QDoubleSpinBox()
        self.slit_spacing_spinbox.setMinimum(10.0)
        self.slit_spacing_spinbox.setMaximum(200.0)
        self.slit_spacing_spinbox.setValue(50.0)
        self.slit_spacing_spinbox.setDecimals(1)
        self.slit_spacing_spinbox.setSuffix(" μm")
        self.slit_spacing_spinbox.valueChanged.connect(self.on_simulation_parameter_changed)
        self.slit_spacing_layout.addWidget(self.slit_spacing_spinbox)
        
        simulation_control_layout.addLayout(self.slit_spacing_layout)
        
        self.simulation_control_group.setLayout(simulation_control_layout)
        left_layout.addWidget(self.simulation_control_group)
        
        # 分析结果文本框（左下角位置）
        result_label = QLabel("分析结果:")
        result_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setMinimumHeight(120)
        self.result_text.setMaximumHeight(180)
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("分析结果将显示在这里...")
        left_layout.addWidget(self.result_text)
        
        left_layout.addStretch()
        
        # 设置左侧面板样式
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        
        # 右侧：显示区域容器
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_container.setLayout(right_layout)
        
        # 添加标题
        display_label = QLabel("显示区域")
        display_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(display_label)
        
        # 创建带边框的显示区域容器
        display_frame = QFrame()
        display_frame.setFrameShape(QFrame.Shape.Box)
        display_frame.setFrameShadow(QFrame.Shadow.Raised)
        display_frame.setLineWidth(2)
        display_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 3px solid #0078d4;
                border-radius: 8px;
            }
        """)
        
        display_frame_layout = QVBoxLayout()
        display_frame_layout.setContentsMargins(5, 5, 5, 5)
        display_frame.setLayout(display_frame_layout)
        
        # 图像/视频显示区域
        self.image_label = ClickableImageLabel(self)
        self.image_label.setFixedSize(500, 350)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        display_frame_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        right_layout.addWidget(display_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 分析结果区域 - 只保留图表
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()
        
        # Matplotlib 画布
        self.plot_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100, theme='modern')
        self.plot_canvas.setMinimumSize(400, 300)
        self.plot_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_layout.addWidget(self.plot_canvas)
        
        # 添加点击事件，显示完整信号图
        self.plot_canvas.mpl_connect('button_press_event', self.on_plot_clicked)
        
        # 导出按钮
        export_layout = QHBoxLayout()
        self.btn_export_signal = QPushButton("导出信号数据")
        self.btn_export_signal.clicked.connect(self.on_export_signal)
        export_layout.addWidget(self.btn_export_signal)
        export_layout.addStretch()
        result_layout.addLayout(export_layout)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        right_layout.addStretch()
        
        # 添加到 Splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)  # 左侧不拉伸
        splitter.setStretchFactor(1, 1)  # 右侧拉伸
        
        # 设置 Splitter 的初始比例（左侧 30%，右侧 70%）
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # 初始化 OpenGL 仿真组件
        self.simulation_widget = SimulationWidget()
        self.simulation_widget.setFixedSize(500, 350)
        self.simulation_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # 初始化：显示图片分析控件
        self.on_experiment_changed("分析图片")
        self.on_sim_experiment_changed("牛顿环实验")
    
    def on_experiment_changed(self, text):
        """实验类型改变时的回调"""
        self.image_control_group.setVisible(False)
        self.video_control_group.setVisible(False)
        self.simulation_control_group.setVisible(False)

        if text == "分析图片":
            self.image_control_group.setVisible(True)
            # 显示图片分析控件
            for i in range(self.image_control_group.layout().count()):
                item = self.image_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(True)
            # 隐藏视频分析控件
            for i in range(self.video_control_group.layout().count()):
                item = self.video_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(False)
            # 隐藏仿真控制控件
            for i in range(self.simulation_control_group.layout().count()):
                item = self.simulation_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(False)
            # 设置图像标签为两点模式
            self.image_label.set_mode('two_points')
            # 显示图像标签，隐藏仿真控件
            self.image_label.show()
            self.simulation_widget.hide()
            # 确保图像标签在布局中
            display_frame = self.image_label.parent()
            display_frame_layout = display_frame.layout()
            if self.image_label not in [display_frame_layout.itemAt(i).widget() for i in range(display_frame_layout.count())]:
                display_frame_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
            # 清空之前的结果
            self.result_text.clear()
            self.plot_canvas.ax.clear()
            self.plot_canvas.draw()
        elif text == "分析视频":
            self.video_control_group.setVisible(True)
            # 显示视频分析控件
            for i in range(self.video_control_group.layout().count()):
                item = self.video_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(True)
            # 隐藏图片分析控件
            for i in range(self.image_control_group.layout().count()):
                item = self.image_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(False)
            # 隐藏仿真控制控件
            for i in range(self.simulation_control_group.layout().count()):
                item = self.simulation_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(False)
            # 设置图像标签为中心点模式
            self.image_label.set_mode('center_point')
            # 显示图像标签，隐藏仿真控件
            self.image_label.show()
            self.simulation_widget.hide()
            # 确保图像标签在布局中
            display_frame = self.image_label.parent()
            display_frame_layout = display_frame.layout()
            if self.image_label not in [display_frame_layout.itemAt(i).widget() for i in range(display_frame_layout.count())]:
                display_frame_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
            # 清空之前的结果
            self.result_text.clear()
            self.plot_canvas.ax.clear()
            self.plot_canvas.draw()
        elif text == "虚拟仿真":
            self.simulation_control_group.setVisible(True)
            # 隐藏图片分析控件
            for i in range(self.image_control_group.layout().count()):
                item = self.image_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(False)
            # 隐藏视频分析控件
            for i in range(self.video_control_group.layout().count()):
                item = self.video_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(False)
            # 显示仿真控制控件
            for i in range(self.simulation_control_group.layout().count()):
                item = self.simulation_control_group.layout().itemAt(i)
                if item.widget():
                    item.widget().setVisible(True)
            # 隐藏图像标签，显示仿真控件
            self.image_label.hide()
            # 在显示区域添加仿真控件
            display_frame = self.image_label.parent()
            display_frame_layout = display_frame.layout()
            if self.simulation_widget not in [display_frame_layout.itemAt(i).widget() for i in range(display_frame_layout.count())]:
                display_frame_layout.addWidget(self.simulation_widget, alignment=Qt.AlignmentFlag.AlignCenter)
            self.simulation_widget.show()
            # 清空之前的结果
            self.result_text.clear()
            self.plot_canvas.ax.clear()
            self.plot_canvas.draw()
            # 更新仿真参数
            self.on_simulation_parameter_changed()
    
    def on_sim_experiment_changed(self, text):
        """仿真实验类型改变时的回调"""
        # 根据仿真实验类型显示/隐藏相应的参数控件
        if text == "牛顿环实验":
            # 显示：波长、曲率半径、间隙距离、视野范围
            for i in range(self.radius_layout.count()):
                self.radius_layout.itemAt(i).widget().setVisible(True)
            for i in range(self.gap_layout.count()):
                self.gap_layout.itemAt(i).widget().setVisible(True)
            for i in range(self.scale_layout.count()):
                self.scale_layout.itemAt(i).widget().setVisible(True)
            # 隐藏：夹角、缝宽、缝间距
            for i in range(self.angle_layout.count()):
                self.angle_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.slit_width_layout.count()):
                self.slit_width_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.slit_spacing_layout.count()):
                self.slit_spacing_layout.itemAt(i).widget().setVisible(False)
        elif text == "劈尖干涉":
            # 显示：波长、夹角、间隙距离、视野范围
            for i in range(self.angle_layout.count()):
                self.angle_layout.itemAt(i).widget().setVisible(True)
            for i in range(self.gap_layout.count()):
                self.gap_layout.itemAt(i).widget().setVisible(True)
            for i in range(self.scale_layout.count()):
                self.scale_layout.itemAt(i).widget().setVisible(True)
            # 隐藏：曲率半径、缝宽、缝间距
            for i in range(self.radius_layout.count()):
                self.radius_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.slit_width_layout.count()):
                self.slit_width_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.slit_spacing_layout.count()):
                self.slit_spacing_layout.itemAt(i).widget().setVisible(False)
        elif text == "双缝干涉":
            # 显示：波长、缝宽、缝间距
            for i in range(self.slit_width_layout.count()):
                self.slit_width_layout.itemAt(i).widget().setVisible(True)
            for i in range(self.slit_spacing_layout.count()):
                self.slit_spacing_layout.itemAt(i).widget().setVisible(True)
            # 隐藏：曲率半径、间隙距离、夹角、视野范围
            for i in range(self.radius_layout.count()):
                self.radius_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.gap_layout.count()):
                self.gap_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.angle_layout.count()):
                self.angle_layout.itemAt(i).widget().setVisible(False)
            for i in range(self.scale_layout.count()):
                self.scale_layout.itemAt(i).widget().setVisible(False)
        # 更新仿真参数
        self.on_simulation_parameter_changed()
    
    def on_simulation_parameter_changed(self):
        """仿真参数改变时的回调"""
        wavelength = self.wavelength_spinbox.value()
        current_text = self.sim_experiment_combo.currentText()
        
        # 根据仿真实验类型更新参数
        if current_text == "牛顿环实验":
            radius = self.radius_spinbox.value()
            gap_distance = self.gap_spinbox.value()
            scale = self.scale_spinbox.value()
            self.simulation_widget.update_parameters(
                0, wavelength, scale=scale,
                radius=radius, gap_distance=gap_distance
            )
        elif current_text == "劈尖干涉":
            angle_degrees = self.angle_spinbox.value()
            angle_radians = angle_degrees * 3.14159265359 / 180.0  # 度转弧度
            gap_distance = self.gap_spinbox.value()
            scale = self.scale_spinbox.value()
            self.simulation_widget.update_parameters(
                1, wavelength, scale=scale,
                angle=angle_radians, gap_distance=gap_distance
            )
        elif current_text == "双缝干涉":
            slit_width = self.slit_width_spinbox.value()
            slit_spacing = self.slit_spacing_spinbox.value()
            self.simulation_widget.update_parameters(
                2, wavelength,
                slit_width=slit_width, slit_spacing=slit_spacing
            )
    
    def on_load_image(self):
        """加载图片按钮点击事件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择图像文件",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*.*)"
            )
            
            if file_path:
                # 如果正在处理视频，先停止
                if self.video_worker is not None and self.video_worker.isRunning():
                    self.video_worker.stop()
                    self.video_worker.wait()
                    self.video_worker = None
                
                # 重置视频相关状态
                self.video_path = None
                self.video_first_frame = None
                self.intensities_raw = []
                self.intensities_smoothed = []
                self.frame_numbers = []
                self.peak_count = 0
                
                # 切换到图像分析模式
                self.image_label.set_mode('two_points')
                
                # 加载图像
                self.image_label.load_image(file_path)
                self.current_image = self.image_label.get_original_image()
                
                # 清空之前的结果
                self.result_text.clear()
                self.plot_canvas.ax.clear()
                self.plot_canvas.draw()
                
                # 隐藏进度条
                self.progress_bar.setVisible(False)
                self.btn_start_stop.setText("开始/停止")
                
                QMessageBox.information(self, "成功", f"图像已加载: {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载图像失败:\n{str(e)}")
    
    def on_load_video(self):
        """加载视频按钮点击事件"""
        try:
            # 如果正在处理视频，先停止
            if self.video_worker is not None and self.video_worker.isRunning():
                self.video_worker.stop()
                self.video_worker.wait()
            
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
            )
            
            if file_path:
                self.video_path = file_path
                
                # 读取第一帧（处理中文路径）
                # OpenCV 无法直接读取中文路径，需要转换
                import os
                if os.path.exists(file_path):
                    # 使用 numpy 和文件流读取视频（对于第一帧，直接读取图像更简单）
                    # 但视频文件需要特殊处理，这里先尝试直接打开
                    cap = cv2.VideoCapture()
                    # 尝试使用文件路径
                    cap.open(file_path)
                    if not cap.isOpened():
                        raise Exception("无法打开视频文件，请检查文件格式和路径")
                    
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        raise Exception("无法读取视频第一帧")
                    
                    cap.release()
                else:
                    raise Exception("视频文件不存在")
                
                # 保存第一帧并显示
                self.video_first_frame = frame
                self.image_label.original_image = frame
                self.image_label.set_mode('center_point')
                self.image_label.center_point = None
                self.image_label.update_display()
                
                # 重置数据
                self.intensities_raw = []
                self.intensities_smoothed = []
                self.frame_numbers = []
                self.peak_count = 0
                
                # 清空结果
                self.result_text.clear()
                self.plot_canvas.ax.clear()
                self.plot_canvas.draw()
                
                # 显示提示
                QMessageBox.information(
                    self, 
                    "提示", 
                    "视频已加载！\n\n"
                    "请在图像上点击选择干涉条纹的中心点，\n"
                    "然后点击\"开始/停止\"按钮开始分析。"
                )
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载视频失败:\n{str(e)}")
    
    def on_start_analysis(self):
        """开始分析按钮点击事件"""
        try:
            # 检查是否已加载图像
            if self.current_image is None:
                QMessageBox.warning(self, "警告", "请先加载图像！")
                return
            
            # 获取选择的两个点
            point_a, point_b = self.image_label.get_selected_points()
            
            if point_a is None or point_b is None:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "请在图像上点击选择两个点（Point A 和 Point B）！\n"
                    "第一个点击为 Point A（红色），第二个点击为 Point B（蓝色）。"
                )
                return
            
            # 执行分析
            result = analyze_interference_pattern(
                self.current_image,
                point_a,
                point_b
            )
            
            # 显示结果
            self.display_analysis_result(result)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"分析失败:\n{str(e)}")
    
    def on_points_selected(self):
        """当用户选择了两个点后自动触发分析"""
        # 如果已经选择了两个点，自动进行分析
        point_a, point_b = self.image_label.get_selected_points()
        if point_a is not None and point_b is not None:
            # 延迟一下，让界面更新完成
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self.on_start_analysis)
    
    def on_clear_points(self):
        """清除选择点"""
        self.image_label.point_a = None
        self.image_label.point_b = None
        self.image_label.update_display()
        self.result_text.clear()
        self.plot_canvas.ax.clear()
        self.plot_canvas.draw()
    
    def on_start_stop(self):
        """开始/停止按钮点击事件"""
        try:
            if self.video_path is None:
                QMessageBox.warning(self, "警告", "请先加载视频！")
                return
            
            # 检查是否已选择中心点
            center_point = self.image_label.get_center_point()
            if center_point is None:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "请先在视频第一帧上点击选择中心点！\n"
                    "中心点用于分析干涉条纹的亮度变化。"
                )
                return
            
            # 如果正在运行，则停止
            if self.video_worker is not None and self.video_worker.isRunning():
                self.video_worker.stop()
                self.video_worker.wait()
                self.video_worker = None
                self.btn_start_stop.setText("开始/停止")
                self.progress_bar.setVisible(False)
                return
            
            # 开始处理视频
            self.video_worker = VideoWorker(self.video_path, center_point)
            
            # 连接信号
            self.video_worker.frame_ready.connect(self.on_frame_ready)
            self.video_worker.intensity_ready.connect(self.on_intensity_ready)
            self.video_worker.progress_updated.connect(self.on_progress_updated)
            self.video_worker.finished_signal.connect(self.on_video_finished)
            
            # 重置数据
            self.intensities_raw = []
            self.intensities_smoothed = []
            self.frame_numbers = []
            self.peak_count = 0
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 更新按钮文本
            self.btn_start_stop.setText("停止")
            
            # 启动线程
            self.video_worker.start()
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动视频分析失败:\n{str(e)}")
    
    def on_frame_ready(self, frame):
        """接收视频帧信号"""
        # 更新图像显示（每 5 帧更新一次，避免过于频繁）
        if len(self.frame_numbers) % 5 == 0:
            # 在帧上标记中心点
            display_frame = frame.copy()
            center_point = self.image_label.get_center_point()
            if center_point is not None:
                x, y = center_point
                cv2.circle(display_frame, center_point, 8, (0, 255, 255), -1)
                cv2.circle(display_frame, center_point, 12, (0, 255, 255), 2)
                cv2.rectangle(display_frame, 
                             (x - 2, y - 2),
                             (x + 2, y + 2),
                             (0, 255, 255), 1)
            
            # 更新显示
            self.image_label.original_image = display_frame
            self.image_label.update_display()
    
    def on_intensity_ready(self, frame_num, intensity_raw, intensity_smoothed, peak_count):
        """接收亮度数据信号"""
        # 保存数据
        self.frame_numbers.append(frame_num)
        self.intensities_raw.append(intensity_raw)
        self.intensities_smoothed.append(intensity_smoothed)
        self.peak_count = peak_count
        
        # 更新图表（实时滚动显示最近 100 帧）
        self.plot_canvas.plot_realtime_signal(
            self.frame_numbers,
            self.intensities_raw,
            self.intensities_smoothed,
            window_size=100
        )
        
        # 更新文本结果
        result_str = f"""实时分析结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
当前帧数: {frame_num}
当前亮度: {intensity_raw:.2f}
平滑亮度: {intensity_smoothed:.2f}
已移动条纹数: {peak_count}

说明：
• 实时监测中心点的亮度变化
• 使用 Savitzky-Golay 滤波器平滑信号
• 自动统计检测到的波峰数量（条纹级数）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        self.result_text.setText(result_str)
    
    def on_progress_updated(self, progress):
        """更新进度条"""
        self.progress_bar.setValue(progress)
    
    def on_plot_clicked(self, event):
        """点击图表时显示完整信号图"""
        if len(self.frame_numbers) > 0 and len(self.intensities_raw) > 0:
            # 显示完整信号图
            self.plot_canvas.ax.clear()
            
            # 绘制原始信号
            self.plot_canvas.ax.plot(self.frame_numbers, self.intensities_raw, 'b-', linewidth=1, 
                                   alpha=0.6, label='\u539f\u59cb\u4fe1\u53f7', zorder=1, clip_on=True)
            
            # 绘制平滑后的信号
            if self.intensities_smoothed and len(self.intensities_smoothed) > 0:
                min_len = min(len(self.frame_numbers), len(self.intensities_smoothed))
                if min_len > 0:
                    self.plot_canvas.ax.plot(self.frame_numbers[:min_len], self.intensities_smoothed[:min_len], 
                                           'lime', linewidth=2, label='\u5e73\u6ed1\u4fe1\u53f7', zorder=2, clip_on=True)
            
            self.plot_canvas.ax.set_xlabel('帧数', color=self.plot_canvas.theme['text'])
            self.plot_canvas.ax.set_ylabel('亮度值', color=self.plot_canvas.theme['text'])
            self.plot_canvas.ax.set_title('完整亮度信号', color=self.plot_canvas.theme['text'])
            self.plot_canvas.ax.grid(True, alpha=0.3, color=self.plot_canvas.theme['grid'], linestyle='--')
            
            # 固定图例位置
            self.plot_canvas.ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
            
            # Adjust the axes using both raw and smoothed signals.
            self.plot_canvas.ax.set_xlim(self.frame_numbers[0] - 2, self.frame_numbers[-1] + 2)
            min_len = min(len(self.frame_numbers), len(self.intensities_smoothed)) if self.intensities_smoothed else 0
            signal_limits = self.plot_canvas.compute_signal_limits(
                self.intensities_raw,
                self.intensities_smoothed[:min_len] if min_len > 0 else None,
            )
            if signal_limits is not None:
                self.plot_canvas.ax.set_ylim(*signal_limits)
            self.plot_canvas.ax.margins(x=0.02, y=0.0)

            self.plot_canvas.apply_compact_layout()
            self.plot_canvas.draw()
    
    def on_export_signal(self):
        """导出信号数据"""
        if len(self.frame_numbers) > 0 and len(self.intensities_raw) > 0:
            # 打开保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出信号数据",
                "",
                "CSV 文件 (*.csv);;所有文件 (*.*)"
            )
            
            if file_path:
                try:
                    import pandas as pd
                    # 创建数据框
                    data = {
                        '帧号': self.frame_numbers,
                        '原始信号': self.intensities_raw
                    }
                    
                    # 如果有平滑信号，也添加进去
                    if self.intensities_smoothed and len(self.intensities_smoothed) == len(self.frame_numbers):
                        data['平滑信号'] = self.intensities_smoothed
                    
                    df = pd.DataFrame(data)
                    # 导出为 CSV
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    
                    QMessageBox.information(self, "成功", "信号数据已成功导出！")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")
    
    def on_video_finished(self):
        """视频处理完成"""
        self.btn_start_stop.setText("开始/停止")
        self.progress_bar.setValue(100)
        
        # 显示最终结果
        if len(self.frame_numbers) > 0:
            final_result = f"""分析完成！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总帧数: {self.frame_numbers[-1] + 1}
最终亮度: {self.intensities_raw[-1]:.2f}
检测到的条纹移动数: {self.peak_count}

说明：
• 分析已完成，可以查看右侧的实时曲线图
• 点击图表可查看完整的信号曲线
• 条纹移动数表示干涉条纹经过中心点的完整周期数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            self.result_text.setText(final_result)
    
    def on_center_point_selected(self):
        """中心点选择完成后的回调"""
        # 可以在这里添加一些提示
        pass
    
    def display_analysis_result(self, result: dict):
        """显示分析结果"""
        distances = result['distances']
        intensities = result['intensities']
        peaks = result['peaks']
        peak_count = result['peak_count']
        avg_spacing = result['avg_spacing']
        peak_positions = result['peak_positions']
        
        # 绘制图表
        self.plot_canvas.plot_intensity_curve(
            distances, 
            intensities, 
            peaks=peaks,
            peak_positions=peak_positions
        )
        
        # 显示文本结果
        result_str = f"""分析结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检测到的波峰数量（条纹级数）: {peak_count}
平均条纹间距: {avg_spacing:.2f} 像素

说明：
• 波峰（红色 × 标记）代表亮条纹的位置
• 条纹级数表示从 Point A 到 Point B 之间包含的完整条纹数量
• 平均间距可用于计算波长（需要已知实验参数）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        self.result_text.setText(result_str)
