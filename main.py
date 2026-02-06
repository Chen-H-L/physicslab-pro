"""
PhysicsLab Pro - 物理实验辅助工具
主窗口程序，包含选项卡界面和基础布局
"""

import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, 
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QFrame, QComboBox, QGroupBox, QSlider, QDoubleSpinBox,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QPoint, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QImage, QVector2D
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLVertexArrayObject

try:
    from OpenGL import GL as gl
except ImportError:
    # 如果 PyOpenGL 未安装，使用替代方案
    print("警告: PyOpenGL 未安装，OpenGL 功能可能不可用")
    gl = None

# Matplotlib 集成
# 注意：matplotlib 3.5+ 支持 PyQt6，使用 qt5agg 后端即可
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 导入算法模块
from algorithms import (
    analyze_interference_pattern,
    extract_center_intensity,
    smooth_signal,
    count_peaks_in_signal,
    auto_fit,
    calculate_uncertainty
)


class ClickableImageLabel(QLabel):
    """可点击的图像标签，支持鼠标点击选择两个点或单个中心点"""
    
    def __init__(self, parent=None, mode='two_points'):
        """
        参数:
            mode: 'two_points' 用于图像分析（选择两个点）
                  'center_point' 用于视频分析（选择单个中心点）
        """
        super().__init__(parent)
        self.mode = mode
        self.original_image = None  # 原始图像（OpenCV格式）
        self.display_pixmap = None  # 显示的QPixmap
        self.point_a = None  # 第一个点
        self.point_b = None  # 第二个点
        self.center_point = None  # 中心点（用于视频分析）
        self.scale_factor = 1.0  # 图像缩放因子
        self.offset_x = 0  # 图像显示偏移
        self.offset_y = 0
        
        if mode == 'two_points':
            self.setText("点击\"加载图片\"按钮选择图像\n然后在图像上点击两个点进行分析")
        else:
            self.setText("点击\"加载视频\"按钮选择视频\n然后在第一帧上点击选择中心点")
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            background-color: #1a1d29;
            color: #e0e0e0;
            border: 2px solid #3a3f5c;
            border-radius: 8px;
            font-size: 13px;
        """)
    
    def load_image(self, image_path: str):
        """加载图像文件"""
        try:
            # OpenCV 无法直接读取中文路径，需要使用 numpy 和文件流
            import numpy as np
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
                self.original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if self.original_image is None:
                raise ValueError("无法读取图像文件，请检查文件格式是否支持")
            
            # 重置选择点
            self.point_a = None
            self.point_b = None
            
            # 更新显示
            self.update_display()
        except Exception as e:
            raise Exception(f"加载图像失败: {str(e)}")
    
    def update_display(self):
        """更新图像显示，包括绘制选择点和连接线"""
        if self.original_image is None:
            return
        
        # 创建显示用的图像副本
        display_image = self.original_image.copy()
        
        # 根据模式绘制不同的标记
        if self.mode == 'center_point':
            # 中心点模式：只显示中心点
            if self.center_point is not None:
                x, y = self.center_point
                # 绘制中心点（黄色，较大）
                cv2.circle(display_image, self.center_point, 8, (0, 255, 255), -1)
                cv2.circle(display_image, self.center_point, 12, (0, 255, 255), 2)
                cv2.putText(display_image, "Center", 
                           (x + 15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # 绘制 5x5 区域框
                half_size = 2
                cv2.rectangle(display_image, 
                             (x - half_size, y - half_size),
                             (x + half_size, y + half_size),
                             (0, 255, 255), 1)
        else:
            # 两点模式：显示两个点和连接线
            if self.point_a is not None and self.point_b is not None:
                # 绘制连接线
                cv2.line(display_image, self.point_a, self.point_b, (0, 255, 0), 2)
                # 绘制点 A（红色）
                cv2.circle(display_image, self.point_a, 5, (0, 0, 255), -1)
                cv2.putText(display_image, "A", 
                           (self.point_a[0] + 10, self.point_a[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # 绘制点 B（蓝色）
                cv2.circle(display_image, self.point_b, 5, (255, 0, 0), -1)
                cv2.putText(display_image, "B", 
                           (self.point_b[0] + 10, self.point_b[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            elif self.point_a is not None:
                # 只绘制点 A
                cv2.circle(display_image, self.point_a, 5, (0, 0, 255), -1)
                cv2.putText(display_image, "A", 
                           (self.point_a[0] + 10, self.point_a[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 转换为 QPixmap 并缩放以适应标签大小
        h, w = display_image.shape[:2]
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        
        # 计算缩放以适应标签
        label_size = self.size()
        scale_w = label_size.width() / w
        scale_h = label_size.height() / h
        self.scale_factor = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        # 计算居中偏移
        self.offset_x = (label_size.width() - new_w) // 2
        self.offset_y = (label_size.height() - new_h) // 2
        
        # 缩放图像
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, 
                                     Qt.TransformationMode.SmoothTransformation)
        
        self.display_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)
    
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if self.original_image is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            # 获取点击位置（相对于标签）
            label_x = event.position().x()
            label_y = event.position().y()
            
            # 转换为图像坐标
            img_x = int((label_x - self.offset_x) / self.scale_factor)
            img_y = int((label_y - self.offset_y) / self.scale_factor)
            
            # 检查坐标是否在图像范围内
            h, w = self.original_image.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                if self.mode == 'center_point':
                    # 中心点模式：只选择一个点
                    self.center_point = (img_x, img_y)
                    self.update_display()
                    # 触发中心点选择信号
                    parent = self.parent()
                    while parent is not None:
                        if hasattr(parent, 'on_center_point_selected'):
                            parent.on_center_point_selected()
                            break
                        parent = parent.parent()
                else:
                    # 两点模式：选择两个点
                    if self.point_a is None:
                        # 选择第一个点
                        self.point_a = (img_x, img_y)
                        self.point_b = None
                    else:
                        # 选择第二个点
                        self.point_b = (img_x, img_y)
                    
                    # 更新显示
                    self.update_display()
                    
                    # 触发分析信号（通过父组件处理）
                    parent = self.parent()
                    while parent is not None:
                        if hasattr(parent, 'on_points_selected'):
                            parent.on_points_selected()
                            break
                        parent = parent.parent()
    
    def resizeEvent(self, event):
        """处理窗口大小变化"""
        super().resizeEvent(event)
        if self.original_image is not None:
            self.update_display()
    
    def get_selected_points(self):
        """获取选择的两个点（图像坐标）"""
        return self.point_a, self.point_b
    
    def get_original_image(self):
        """获取原始图像"""
        return self.original_image
    
    def get_center_point(self):
        """获取中心点（用于视频分析）"""
        return self.center_point
    
    def set_mode(self, mode):
        """设置选择模式"""
        self.mode = mode
        if mode == 'two_points':
            self.setText("点击\"加载图片\"按钮选择图像\n然后在图像上点击两个点进行分析")
        else:
            self.setText("点击\"加载视频\"按钮选择视频\n然后在第一帧上点击选择中心点")


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib 画布，用于嵌入到 PyQt 界面中"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # 使用深色主题
        plt.style.use('dark_background')
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e2132')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e2132')
        # 使用更柔和的颜色
        axis_color = '#b0b8c4'
        text_color = '#e0e4eb'
        self.ax.tick_params(colors=axis_color)
        self.ax.spines['bottom'].set_color(axis_color)
        self.ax.spines['top'].set_color(axis_color)
        self.ax.spines['right'].set_color(axis_color)
        self.ax.spines['left'].set_color(axis_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)
    
    def plot_intensity_curve(self, distances, intensities, peaks=None, peak_positions=None):
        """绘制亮度分布曲线（静态图像分析）"""
        self.ax.clear()
        
        # 绘制亮度曲线
        self.ax.plot(distances, intensities, 'b-', linewidth=1.5, label='亮度分布')
        
        # 标记波峰位置
        if peaks is not None and len(peaks) > 0 and peak_positions is not None:
            peak_intensities = intensities[peaks]
            self.ax.plot(peak_positions, peak_intensities, 'rx', 
                        markersize=8, markeredgewidth=2, label='检测到的波峰')
        
        self.ax.set_xlabel('距离 (像素)', color='#e0e4eb')
        self.ax.set_ylabel('亮度值', color='#e0e4eb')
        self.ax.set_title('光强分布 vs 像素位置', color='#e0e4eb')
        self.ax.grid(True, alpha=0.2, color='#3a3f5c', linestyle='--')
        self.ax.legend(loc='best', framealpha=0.8)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_realtime_signal(self, frame_numbers, intensities, smoothed_intensities=None, 
                            window_size=100):
        """绘制实时信号曲线（视频分析）"""
        self.ax.clear()
        
        # 只显示最近 window_size 帧
        if len(frame_numbers) > window_size:
            start_idx = len(frame_numbers) - window_size
            frame_numbers = frame_numbers[start_idx:]
            intensities = intensities[start_idx:]
            if smoothed_intensities is not None and len(smoothed_intensities) > start_idx:
                smoothed_intensities = smoothed_intensities[start_idx:]
            else:
                smoothed_intensities = None
        else:
            start_idx = 0
        
        # 绘制原始信号（较淡，作为背景）
        self.ax.plot(frame_numbers, intensities, 'b-', linewidth=1, 
                    alpha=0.3, label='原始信号', zorder=1)
        
        # 绘制平滑后的信号（更明显）
        if smoothed_intensities is not None and len(smoothed_intensities) > 0:
            # 确保长度匹配
            min_len = min(len(frame_numbers), len(smoothed_intensities))
            if min_len > 0:
                self.ax.plot(frame_numbers[:min_len], smoothed_intensities[:min_len], 
                           'lime', linewidth=2.5, label='平滑信号', zorder=2)
        
        self.ax.set_xlabel('帧数', color='#e0e4eb')
        self.ax.set_ylabel('亮度值', color='#e0e4eb')
        self.ax.set_title('实时亮度信号（最近 100 帧）', color='#e0e4eb')
        self.ax.grid(True, alpha=0.2, color='#3a3f5c', linestyle='--')
        
        # 固定图例位置，避免乱飞
        self.ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        
        # 自动调整坐标轴范围
        if len(frame_numbers) > 0:
            self.ax.set_xlim(frame_numbers[0] - 2, frame_numbers[-1] + 2)
            if len(intensities) > 0:
                y_min = min(intensities)
                y_max = max(intensities)
                y_range = y_max - y_min
                if y_range > 0:
                    self.ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        self.fig.tight_layout()
        self.draw()


class VideoWorker(QThread):
    """视频处理工作线程"""
    
    # 定义信号
    frame_ready = pyqtSignal(np.ndarray)  # 当前帧图像
    intensity_ready = pyqtSignal(int, float, float, int)  # 帧号, 原始亮度, 平滑亮度, 波峰计数
    progress_updated = pyqtSignal(int)  # 进度百分比
    finished_signal = pyqtSignal()  # 处理完成
    
    def __init__(self, video_path, center_point):
        super().__init__()
        self.video_path = video_path
        self.center_point = center_point
        self.is_running = False
        self.is_paused = False
    
    def run(self):
        """线程主函数：处理视频"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            intensities_raw = []
            intensities_smoothed = []
            frame_numbers = []
            peak_count = 0
            
            self.is_running = True
            
            while self.is_running:
                if self.is_paused:
                    self.msleep(100)  # 暂停时等待
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 提取中心点亮度
                intensity = extract_center_intensity(frame, self.center_point, region_size=5)
                intensities_raw.append(intensity)
                frame_numbers.append(frame_count)
                
                # 平滑处理（需要至少 11 个数据点）
                if len(intensities_raw) >= 11:
                    smoothed = smooth_signal(np.array(intensities_raw), window_length=11, polyorder=3)
                    intensities_smoothed = smoothed.tolist()
                    
                    # 实时波峰检测（在平滑后的信号上）
                    if len(intensities_smoothed) >= 10:
                        current_smoothed = intensities_smoothed[-1]
                        peak_count = count_peaks_in_signal(np.array(intensities_smoothed))
                else:
                    intensities_smoothed.append(intensity)
                
                # 发送信号更新 UI
                current_smoothed = intensities_smoothed[-1] if intensities_smoothed else intensity
                self.intensity_ready.emit(frame_count, intensity, current_smoothed, peak_count)
                self.frame_ready.emit(frame)
                
                # 更新进度
                if total_frames > 0:
                    progress = int((frame_count + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)
                
                frame_count += 1
                
                # 控制处理速度（避免过快）
                if fps > 0:
                    self.msleep(int(1000 / fps))
            
            cap.release()
            self.finished_signal.emit()
            
        except Exception as e:
            print(f"视频处理错误: {str(e)}")
            self.finished_signal.emit()
    
    def stop(self):
        """停止处理"""
        self.is_running = False
    
    def pause(self):
        """暂停处理"""
        self.is_paused = True
    
    def resume(self):
        """恢复处理"""
        self.is_paused = False


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
        self.experiment_combo.addItems(["分析图片", "分析视频"])
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
                background-color: #1a1d29;
                border: 3px solid #4f7cff;
                border-radius: 8px;
            }
        """)
        
        display_frame_layout = QVBoxLayout()
        display_frame_layout.setContentsMargins(5, 5, 5, 5)
        display_frame.setLayout(display_frame_layout)
        
        # 图像/视频显示区域
        self.image_label = ClickableImageLabel(self)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        display_frame_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        right_layout.addWidget(display_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 分析结果区域
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()
        
        # 嵌入 Matplotlib 画布
        self.plot_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        self.plot_canvas.setMinimumSize(400, 300)
        result_layout.addWidget(self.plot_canvas)
        
        # 结果显示文本
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(120)
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("分析结果将显示在这里...")
        result_layout.addWidget(self.result_text)
        
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
        
        # 初始化：显示图片分析控件
        self.on_experiment_changed("分析图片")
    
    def on_experiment_changed(self, text):
        """实验类型改变时的回调"""
        if text == "分析图片":
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
            # 设置图像标签为两点模式
            self.image_label.set_mode('two_points')
        elif text == "分析视频":
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
            # 设置图像标签为中心点模式
            self.image_label.set_mode('center_point')
    
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


class DataWorkstationTab(QWidget):
    """页面 2: 数据工作台 (Data Workstation)"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化界面布局：上半部分数据输入，下半部分可视化"""
        main_layout = QVBoxLayout()
        
        # 上半部分：数据输入区域
        input_section = QWidget()
        input_layout = QVBoxLayout()
        
        # 标题
        input_title = QLabel("数据输入")
        input_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        input_layout.addWidget(input_title)
        
        # 表格控件用于手动输入数据
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        self.data_table.setHorizontalHeaderLabels(["X", "Y"])
        self.data_table.setMinimumHeight(200)
        
        # 设置表格为可编辑
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        
        # 添加初始行，方便用户输入
        self.data_table.setRowCount(10)
        for i in range(10):
            # 创建可编辑的单元格
            x_item = QTableWidgetItem("")
            y_item = QTableWidgetItem("")
            self.data_table.setItem(i, 0, x_item)
            self.data_table.setItem(i, 1, y_item)
        
        # 设置列宽
        self.data_table.setColumnWidth(0, 200)
        self.data_table.setColumnWidth(1, 200)
        
        input_layout.addWidget(self.data_table)
        
        # 按钮组
        button_layout = QHBoxLayout()
        self.btn_import_csv = QPushButton("导入 CSV")
        self.btn_start_fit = QPushButton("开始拟合")
        self.btn_calculate_uncertainty = QPushButton("计算不确定度")
        
        button_layout.addWidget(self.btn_import_csv)
        button_layout.addWidget(self.btn_start_fit)
        button_layout.addWidget(self.btn_calculate_uncertainty)
        input_layout.addLayout(button_layout)
        
        # 不确定度计算输入框
        uncertainty_label = QLabel("不确定度计算（输入逗号分隔的数字）：")
        input_layout.addWidget(uncertainty_label)
        self.uncertainty_input = QLineEdit()
        self.uncertainty_input.setPlaceholderText("例如: 1.23, 1.25, 1.24, 1.26, 1.22")
        input_layout.addWidget(self.uncertainty_input)
        
        input_section.setLayout(input_layout)
        
        # 下半部分：可视化绘图区域
        plot_section = QWidget()
        plot_layout = QVBoxLayout()
        
        # 标题
        plot_title = QLabel("可视化结果")
        plot_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        plot_layout.addWidget(plot_title)
        
        # 嵌入 Matplotlib 画布
        self.plot_canvas = MatplotlibCanvas(self, width=8, height=4, dpi=100)
        self.plot_canvas.setMinimumSize(800, 400)
        plot_layout.addWidget(self.plot_canvas)
        
        # 结果显示文本
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(150)
        self.result_text.setReadOnly(True)
        plot_layout.addWidget(self.result_text)
        
        plot_section.setLayout(plot_layout)
        
        # 将上下两部分添加到主布局
        main_layout.addWidget(input_section, 1)
        main_layout.addWidget(plot_section, 1)
        
        self.setLayout(main_layout)
        
        # 连接信号（功能逻辑后续实现）
        self.btn_import_csv.clicked.connect(self.on_import_csv)
        self.btn_start_fit.clicked.connect(self.on_start_fit)
        self.btn_calculate_uncertainty.clicked.connect(self.on_calculate_uncertainty)
    
    def get_data_from_table(self):
        """从表格中读取 X, Y 数据"""
        x_data = []
        y_data = []
        
        row_count = self.data_table.rowCount()
        for i in range(row_count):
            x_item = self.data_table.item(i, 0)
            y_item = self.data_table.item(i, 1)
            
            if x_item is not None and y_item is not None:
                try:
                    x_val = float(x_item.text())
                    y_val = float(y_item.text())
                    x_data.append(x_val)
                    y_data.append(y_val)
                except ValueError:
                    continue  # 跳过无效的行
        
        return np.array(x_data), np.array(y_data)
    
    def on_import_csv(self):
        """导入 CSV 按钮点击事件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择 CSV 文件",
                "",
                "CSV 文件 (*.csv);;所有文件 (*.*)"
            )
            
            if file_path:
                # 使用 pandas 读取 CSV
                import pandas as pd
                df = pd.read_csv(file_path)
                
                # 检查列数
                if df.shape[1] < 2:
                    QMessageBox.warning(self, "错误", "CSV 文件至少需要 2 列数据（X 和 Y）")
                    return
                
                # 清空表格
                self.data_table.setRowCount(0)
                
                # 填充表格
                for i, row in df.iterrows():
                    row_pos = self.data_table.rowCount()
                    self.data_table.insertRow(row_pos)
                    
                    # 设置 X 值
                    x_item = QTableWidgetItem(str(row.iloc[0]))
                    self.data_table.setItem(row_pos, 0, x_item)
                    
                    # 设置 Y 值
                    y_item = QTableWidgetItem(str(row.iloc[1]))
                    self.data_table.setItem(row_pos, 1, y_item)
                
                QMessageBox.information(self, "成功", f"已导入 {len(df)} 行数据")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导入 CSV 失败:\n{str(e)}")
    
    def on_start_fit(self):
        """开始拟合按钮点击事件"""
        try:
            # 从表格读取数据
            x_data, y_data = self.get_data_from_table()
            
            if len(x_data) < 2:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "数据点不足！\n\n"
                    "请在表格中输入至少 2 行数据，或使用\"导入 CSV\"按钮导入数据。"
                )
                return
            
            # 执行自动拟合
            result = auto_fit(x_data, y_data)
            
            # 显示拟合结果
            self.display_fit_result(result)
            
        except ValueError as e:
            QMessageBox.warning(self, "错误", f"数据错误:\n{str(e)}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"拟合失败:\n{str(e)}")
    
    def display_fit_result(self, result: dict):
        """显示拟合结果"""
        # 绘制图表
        self.plot_canvas.ax.clear()
        
        # 绘制原始数据点（散点）
        self.plot_canvas.ax.scatter(
            result['x_data'], 
            result['y_data'], 
            color='cyan', 
            s=50, 
            alpha=0.7, 
            label='原始数据点',
            edgecolors='white',
            linewidths=0.5
        )
        
        # 绘制拟合曲线（实线）
        self.plot_canvas.ax.plot(
            result['x_fitted'], 
            result['y_fitted'], 
            'r-', 
            linewidth=2, 
            label=f'拟合曲线 ({result["best_formula"]})'
        )
        
        self.plot_canvas.ax.set_xlabel('X', color='#e0e4eb')
        self.plot_canvas.ax.set_ylabel('Y', color='#e0e4eb')
        self.plot_canvas.ax.set_title('数据拟合结果', color='#e0e4eb')
        self.plot_canvas.ax.grid(True, alpha=0.2, color='#3a3f5c', linestyle='--')
        self.plot_canvas.ax.legend(loc='best', framealpha=0.8)
        
        # 设置坐标轴颜色
        axis_color = '#b0b8c4'
        self.plot_canvas.ax.tick_params(colors=axis_color)
        self.plot_canvas.ax.spines['bottom'].set_color(axis_color)
        self.plot_canvas.ax.spines['top'].set_color(axis_color)
        self.plot_canvas.ax.spines['right'].set_color(axis_color)
        self.plot_canvas.ax.spines['left'].set_color(axis_color)
        
        self.plot_canvas.fig.tight_layout()
        self.plot_canvas.draw()
        
        # 显示文本结果
        result_str = f"""拟合结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最佳拟合公式: {result['best_formula']}
公式表达式: {result['best_formula_str']}
拟合优度 (R²): {result['best_r_squared']:.6f}

所有尝试的拟合结果：
"""
        
        for r in result['all_results']:
            if r['r_squared'] != -np.inf:
                result_str += f"• {r['formula']}: R² = {r['r_squared']:.6f}\n"
            else:
                result_str += f"• {r['formula']}: 拟合失败"
                if 'error' in r:
                    result_str += f" ({r['error']})"
                result_str += "\n"
        
        result_str += """
说明：
• R² 值越接近 1，表示拟合效果越好
• 软件自动尝试了线性、指数、余弦三种拟合方式
• 选择了 R² 值最高的拟合结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        self.result_text.setText(result_str)
    
    def on_calculate_uncertainty(self):
        """计算不确定度按钮点击事件"""
        try:
            # 获取输入数据
            data_str = self.uncertainty_input.text().strip()
            
            if not data_str:
                QMessageBox.warning(self, "警告", "请输入数据！\n\n例如: 10.1, 10.2, 10.5, 10.3")
                return
            
            # 计算不确定度
            result = calculate_uncertainty(data_str)
            
            # 显示结果
            self.display_uncertainty_result(result)
            
        except ValueError as e:
            QMessageBox.warning(self, "错误", f"计算失败:\n{str(e)}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"发生错误:\n{str(e)}")
    
    def display_uncertainty_result(self, result: dict):
        """显示不确定度计算结果"""
        # 绘制数据分布图
        self.plot_canvas.ax.clear()
        
        data = result['data']
        mean = result['mean']
        
        # 绘制数据点
        x_positions = np.arange(len(data))
        self.plot_canvas.ax.scatter(
            x_positions, 
            data, 
            color='cyan', 
            s=80, 
            alpha=0.7, 
            label='测量数据',
            edgecolors='white',
            linewidths=1
        )
        
        # 绘制平均值线
        self.plot_canvas.ax.axhline(
            y=mean, 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'平均值 = {mean:.4f}'
        )
        
        # 绘制误差带（± 标准误差）
        std_error = result['standard_error']
        self.plot_canvas.ax.fill_between(
            x_positions,
            mean - std_error,
            mean + std_error,
            alpha=0.2,
            color='yellow',
            label=f'不确定度范围 (±{std_error:.4f})'
        )
        
        # 标记异常值
        if result['outliers']:
            outlier_indices = [idx for idx, _, _ in result['outliers']]
            outlier_values = [val for _, val, _ in result['outliers']]
            self.plot_canvas.ax.scatter(
                outlier_indices,
                outlier_values,
                color='red',
                s=150,
                marker='x',
                linewidths=3,
                label='异常值 (Z-score > 3)'
            )
        
        self.plot_canvas.ax.set_xlabel('测量序号', color='#e0e4eb')
        self.plot_canvas.ax.set_ylabel('测量值', color='#e0e4eb')
        self.plot_canvas.ax.set_title('测量数据分布', color='#e0e4eb')
        self.plot_canvas.ax.grid(True, alpha=0.2, color='#3a3f5c', linestyle='--')
        self.plot_canvas.ax.legend(loc='best', framealpha=0.8)
        
        # 设置坐标轴颜色
        axis_color = '#b0b8c4'
        self.plot_canvas.ax.tick_params(colors=axis_color)
        self.plot_canvas.ax.spines['bottom'].set_color(axis_color)
        self.plot_canvas.ax.spines['top'].set_color(axis_color)
        self.plot_canvas.ax.spines['right'].set_color(axis_color)
        self.plot_canvas.ax.spines['left'].set_color(axis_color)
        
        self.plot_canvas.fig.tight_layout()
        self.plot_canvas.draw()
        
        # 显示文本结果
        result_str = f"""不确定度计算结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据点数量: {result['count']}
平均值 (Mean): {result['mean']:.6f}
标准差 (Std Dev): {result['std_dev']:.6f}
标准误差 (Standard Error): {result['standard_error']:.6f}

最终结果: {result['mean']:.6f} ± {result['standard_error']:.6f}
"""
        
        if result['outliers']:
            result_str += f"\n⚠️  检测到 {len(result['outliers'])} 个异常值 (Z-score > 3):\n"
            for idx, val, z_score in result['outliers']:
                result_str += f"  • 第 {idx + 1} 个数据: {val:.6f} (Z-score = {z_score:.2f})\n"
            result_str += "\n建议：考虑剔除这些异常值后重新计算。\n"
        else:
            result_str += "\n✓ 未检测到异常值。\n"
        
        result_str += """
说明：
• 平均值：所有测量数据的算术平均
• 标准差：测量数据的离散程度
• 标准误差（A类不确定度）：平均值的不确定度
• 异常值检测：使用 Z-score 方法，Z-score > 3 的数据被认为是异常值
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        self.result_text.setText(result_str)


class SimulationWidget(QOpenGLWidget):
    """OpenGL 仿真显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shader_program = None
        self.vao = None
        self.vbo = None
        
        # 窗口尺寸
        self.width = 800
        self.height = 600
        
        # 实验类型
        self.experiment_type = 0  # 0=牛顿环, 1=劈尖干涉, 2=双缝干涉
        
        # 通用参数
        self.wavelength = 632.8  # nm
        self.scale = 5.0  # 视野缩放因子 (mm) - 屏幕中心到边缘的物理半径
        
        # 牛顿环参数
        self.radius = 1000.0  # mm
        self.gap_distance = 0.0  # nm (间隙距离)
        
        # 劈尖干涉参数
        self.angle = 0.001  # 弧度 (约 0.057 度)
        
        # 双缝干涉参数
        self.slit_width = 10.0  # μm
        self.slit_spacing = 50.0  # μm
        
    def initializeGL(self):
        """初始化 OpenGL"""
        if gl is None:
            return
        
        # 设置 OpenGL 版本
        gl.glClearColor(0.12, 0.13, 0.18, 1.0)  # 深蓝灰色背景
        
        # 创建并编译 Shader
        self.shader_program = QOpenGLShaderProgram(self)
        
        # Vertex Shader - 绘制全屏矩形
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        
        out vec2 TexCoord;
        
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
        """
        
        # Fragment Shader - 多实验类型物理渲染
        fragment_shader_source = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        uniform vec2 u_resolution;      // 画布分辨率 (width, height)
        uniform int u_experiment_type; // 实验类型: 0=牛顿环, 1=劈尖干涉, 2=双缝干涉
        uniform float u_wavelength;    // 光波长 (nm)
        uniform float u_scale;         // 视野缩放因子 (mm) - 屏幕中心到边缘的物理半径
        
        // 牛顿环参数
        uniform float u_radius;        // 透镜曲率半径 R (mm)
        uniform float u_gap;           // 空气间隙 h (nm)
        
        // 劈尖干涉参数
        uniform float u_angle;         // 劈尖夹角 (弧度)
        
        // 双缝干涉参数
        uniform float u_slit_width;    // 缝宽 a (μm)
        uniform float u_slit_spacing;  // 缝间距 d (μm)
        
        // 波长到颜色的转换函数（可见光范围 400-700 nm）
        vec3 wavelength_to_rgb(float wavelength_nm) {
            float r = 0.0;
            float g = 0.0;
            float b = 0.0;
            
            if (wavelength_nm >= 400.0 && wavelength_nm < 440.0) {
                r = -(wavelength_nm - 440.0) / (440.0 - 400.0);
                g = 0.0;
                b = 1.0;
            }
            else if (wavelength_nm >= 440.0 && wavelength_nm < 490.0) {
                r = 0.0;
                g = (wavelength_nm - 440.0) / (490.0 - 440.0);
                b = 1.0;
            }
            else if (wavelength_nm >= 490.0 && wavelength_nm < 510.0) {
                r = 0.0;
                g = 1.0;
                b = -(wavelength_nm - 510.0) / (510.0 - 490.0);
            }
            else if (wavelength_nm >= 510.0 && wavelength_nm < 580.0) {
                r = (wavelength_nm - 510.0) / (580.0 - 510.0);
                g = 1.0;
                b = 0.0;
            }
            else if (wavelength_nm >= 580.0 && wavelength_nm < 645.0) {
                r = 1.0;
                g = -(wavelength_nm - 645.0) / (645.0 - 580.0);
                b = 0.0;
            }
            else if (wavelength_nm >= 645.0 && wavelength_nm <= 700.0) {
                r = 1.0;
                g = 0.0;
                b = 0.0;
            }
            
            // 在边界处平滑过渡
            if (wavelength_nm < 400.0) {
                r = 0.5; g = 0.0; b = 1.0;  // 紫色
            }
            if (wavelength_nm > 700.0) {
                r = 1.0; g = 0.0; b = 0.0;  // 红色
            }
            
            return vec3(r, g, b);
        }
        
        void main() {
            // 统一使用归一化坐标系统，确保中心对齐
            // 将屏幕坐标归一化到 [-1, 1]，中心在 (0, 0)
            // 使用 min 保持宽高比，确保圆形不变形
            vec2 uv = (gl_FragCoord.xy - u_resolution * 0.5) / min(u_resolution.x, u_resolution.y);
            
            // 确保中心在 (0, 0)
            // gl_FragCoord.xy 是像素坐标，从 (0,0) 到 (width, height)
            // 减去中心后除以最小边，得到归一化坐标 [-1, 1]
            
            float intensity = 0.0;
            float d = 0.0;
            
            if (u_experiment_type == 0) {
                // 牛顿环干涉
                // 使用归一化坐标计算径向距离
                float r_mm = length(uv) * u_scale;
                
                // 单位换算：r_mm (mm) -> r_nm (nm)
                float r_nm = r_mm * 1000000.0;
                
                // R 从 mm 转换为 nm
                float R_nm = u_radius * 1000000.0;
                
                // 计算空气层厚度 d(r) = r²/(2R) + h
                d = (r_nm * r_nm) / (2.0 * R_nm) + u_gap;
                
                // 应用干涉光强公式 I = cos²(2πd/λ)
                float phase = 2.0 * 3.14159265359 * d / u_wavelength;
                intensity = cos(phase);
                intensity = intensity * intensity;  // cos²
            }
            else if (u_experiment_type == 1) {
                // 劈尖干涉
                // 空气厚度随 X 轴线性增加: d = x * tan(θ) + h₀
                // 使用归一化坐标，转换为物理尺寸
                float x_normalized = uv.x;  // 归一化坐标 [-1, 1]
                float scale_factor = u_scale;  // 使用与牛顿环相同的缩放因子
                float x_mm = x_normalized * scale_factor;
                float x_nm = x_mm * 1000000.0;  // mm to nm
                d = x_nm * tan(u_angle) + u_gap;  // u_gap 作为初始厚度 h₀
                float phase = 2.0 * 3.14159265359 * d / u_wavelength;
                intensity = cos(phase);
                intensity = intensity * intensity;  // cos²
            }
            else if (u_experiment_type == 2) {
                // 双缝干涉（竖着的条纹）
                // 使用归一化坐标，确保中心对齐
                float x_normalized = uv.x;  // 归一化坐标 [-1, 1]，中心在 0
                
                // 计算角度: sin(θ) ≈ tan(θ) ≈ x / L (小角度近似)
                // 使用合理的比例：观察屏距离 L = 1000 mm
                // 屏幕边缘到中心代表约 100 mm 的观察范围（调整比例使条纹更合理）
                float L_mm = 1000.0;  // 观察屏距离（固定）
                float x_mm = x_normalized * 100.0;  // 调整比例因子，使条纹密度更合理
                float sin_theta = x_mm / L_mm;  // 小角度近似
                
                // 单位转换: μm to nm
                float slit_spacing_nm = u_slit_spacing * 1000.0;
                float slit_width_nm = u_slit_width * 1000.0;
                
                // 双缝干涉项: cos²(πd sin(θ)/λ)
                float interference_phase = 3.14159265359 * slit_spacing_nm * sin_theta / u_wavelength;
                float interference = cos(interference_phase);
                interference = interference * interference;
                
                // 单缝衍射包络: sinc²(πa sin(θ)/λ)
                float diffraction_phase = 3.14159265359 * slit_width_nm * sin_theta / u_wavelength;
                float sinc_value = 1.0;
                if (abs(diffraction_phase) > 0.001) {
                    sinc_value = sin(diffraction_phase) / diffraction_phase;
                }
                float diffraction = sinc_value * sinc_value;
                
                // 总强度 = 干涉项 × 衍射包络
                intensity = interference * diffraction;
            }
            
            // 根据波长计算颜色
            vec3 base_color = wavelength_to_rgb(u_wavelength);
            
            // 将强度映射到颜色
            vec3 final_color = base_color * intensity;
            
            // 输出彩色
            FragColor = vec4(final_color, 1.0);
        }
        """
        
        # 编译 Shader
        if not self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader_source):
            print(f"Vertex Shader 编译失败: {self.shader_program.log()}")
            return
        
        if not self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader_source):
            print(f"Fragment Shader 编译失败: {self.shader_program.log()}")
            return
        
        # 链接 Shader Program
        if not self.shader_program.link():
            print(f"Shader Program 链接失败: {self.shader_program.log()}")
            return
        
        # 创建全屏矩形顶点数据
        # 位置坐标 (NDC: -1 到 1) 和纹理坐标 (0 到 1)
        vertices = np.array([
            # 位置 (x, y)    纹理坐标 (u, v)
            -1.0, -1.0,      0.0, 0.0,  # 左下
             1.0, -1.0,      1.0, 0.0,  # 右下
             1.0,  1.0,      1.0, 1.0,  # 右上
            -1.0,  1.0,      0.0, 1.0,  # 左上
        ], dtype=np.float32)
        
        # 创建 VAO 和 VBO
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer()
        
        self.vao.create()
        self.vbo.create()
        
        self.vao.bind()
        self.vbo.bind()
        self.vbo.allocate(vertices.tobytes(), vertices.nbytes)
        
        # 设置顶点属性
        # 位置属性 (location = 0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, None)
        gl.glEnableVertexAttribArray(0)
        
        # 纹理坐标属性 (location = 1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.GLvoidp(2 * 4))
        gl.glEnableVertexAttribArray(1)
        
        self.vao.release()
        self.vbo.release()
    
    def resizeGL(self, width, height):
        """处理窗口大小调整"""
        if gl is None:
            return
        gl.glViewport(0, 0, width, height)
        self.width = width
        self.height = height
        # 注意：OpenGL 的 gl_FragCoord.y 是从下往上的，但我们的计算应该没问题
        self.update()  # 触发重绘以更新分辨率
    
    def paintGL(self):
        """执行绘制"""
        if gl is None:
            return
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        if self.shader_program is None or not self.shader_program.isLinked():
            return
        
        # 使用 Shader Program
        self.shader_program.bind()
        
        # 获取当前窗口尺寸
        width = self.width if hasattr(self, 'width') else self.size().width()
        height = self.height if hasattr(self, 'height') else self.size().height()
        
        # 如果尺寸无效，使用默认值
        if width <= 0:
            width = 800
        if height <= 0:
            height = 600
        
        # 设置 uniform 变量
        self.shader_program.setUniformValue("u_resolution", QVector2D(float(width), float(height)))
        self.shader_program.setUniformValue("u_experiment_type", int(self.experiment_type))
        self.shader_program.setUniformValue("u_wavelength", float(self.wavelength))
        self.shader_program.setUniformValue("u_scale", float(self.scale))
        
        # 牛顿环参数
        self.shader_program.setUniformValue("u_radius", float(self.radius))
        self.shader_program.setUniformValue("u_gap", float(self.gap_distance))
        
        # 劈尖干涉参数
        self.shader_program.setUniformValue("u_angle", float(self.angle))
        
        # 双缝干涉参数
        self.shader_program.setUniformValue("u_slit_width", float(self.slit_width))
        self.shader_program.setUniformValue("u_slit_spacing", float(self.slit_spacing))
        
        # 绑定 VAO 并绘制
        if self.vao is not None:
            self.vao.bind()
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
            self.vao.release()
        
        self.shader_program.release()
    
    def update_parameters(self, experiment_type, wavelength, scale=None, **kwargs):
        """更新实验参数"""
        self.experiment_type = experiment_type
        self.wavelength = wavelength
        if scale is not None:
            self.scale = scale
        
        # 根据实验类型更新相应参数
        if experiment_type == 0:  # 牛顿环
            self.radius = kwargs.get('radius', self.radius)
            self.gap_distance = kwargs.get('gap_distance', self.gap_distance)
        elif experiment_type == 1:  # 劈尖干涉
            self.angle = kwargs.get('angle', self.angle)
            self.gap_distance = kwargs.get('gap_distance', self.gap_distance)
        elif experiment_type == 2:  # 双缝干涉
            self.slit_width = kwargs.get('slit_width', self.slit_width)
            self.slit_spacing = kwargs.get('slit_spacing', self.slit_spacing)
        
        self.update()  # 触发重绘


class VirtualLabTab(QWidget):
    """页面 3: 虚拟仿真实验室 (Virtual Lab)"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化界面布局：左右分屏"""
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
        self.experiment_combo.addItems(["牛顿环实验", "劈尖干涉", "双缝干涉"])
        self.experiment_combo.currentTextChanged.connect(self.on_experiment_changed)
        left_layout.addWidget(self.experiment_combo)
        
        left_layout.addSpacing(20)
        
        # 参数控制组
        self.param_group = QGroupBox("实验参数")
        param_layout = QVBoxLayout()
        
        # 波长参数（所有实验都需要）
        wavelength_layout = QHBoxLayout()
        wavelength_label = QLabel("波长 (nm):")
        wavelength_label.setMinimumWidth(120)
        wavelength_layout.addWidget(wavelength_label)
        
        self.wavelength_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelength_slider.setMinimum(400)
        self.wavelength_slider.setMaximum(700)
        self.wavelength_slider.setValue(633)
        self.wavelength_slider.valueChanged.connect(self.on_parameter_changed)
        wavelength_layout.addWidget(self.wavelength_slider)
        
        self.wavelength_spinbox = QDoubleSpinBox()
        self.wavelength_spinbox.setMinimum(400.0)
        self.wavelength_spinbox.setMaximum(700.0)
        self.wavelength_spinbox.setValue(632.8)
        self.wavelength_spinbox.setDecimals(1)
        self.wavelength_spinbox.setSuffix(" nm")
        self.wavelength_spinbox.valueChanged.connect(self.on_parameter_changed)
        wavelength_layout.addWidget(self.wavelength_spinbox)
        
        param_layout.addLayout(wavelength_layout)
        
        # 牛顿环参数：曲率半径
        self.radius_layout = QHBoxLayout()
        radius_label = QLabel("曲率半径 (mm):")
        radius_label.setMinimumWidth(120)
        self.radius_layout.addWidget(radius_label)
        
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setMinimum(500)
        self.radius_slider.setMaximum(5000)
        self.radius_slider.setValue(1000)
        self.radius_slider.valueChanged.connect(self.on_parameter_changed)
        self.radius_layout.addWidget(self.radius_slider)
        
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(500.0)
        self.radius_spinbox.setMaximum(5000.0)
        self.radius_spinbox.setValue(1000.0)
        self.radius_spinbox.setDecimals(1)
        self.radius_spinbox.setSuffix(" mm")
        self.radius_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.radius_layout.addWidget(self.radius_spinbox)
        
        param_layout.addLayout(self.radius_layout)
        
        # 牛顿环和劈尖共用：间隙距离
        self.gap_layout = QHBoxLayout()
        gap_label = QLabel("间隙距离 (nm):")
        gap_label.setMinimumWidth(120)
        self.gap_layout.addWidget(gap_label)
        
        self.gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.gap_slider.setMinimum(0)
        self.gap_slider.setMaximum(2000)
        self.gap_slider.setValue(0)
        self.gap_slider.valueChanged.connect(self.on_parameter_changed)
        self.gap_layout.addWidget(self.gap_slider)
        
        self.gap_spinbox = QDoubleSpinBox()
        self.gap_spinbox.setMinimum(0.0)
        self.gap_spinbox.setMaximum(2000.0)
        self.gap_spinbox.setValue(0.0)
        self.gap_spinbox.setDecimals(1)
        self.gap_spinbox.setSuffix(" nm")
        self.gap_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.gap_layout.addWidget(self.gap_spinbox)
        
        param_layout.addLayout(self.gap_layout)
        
        # 牛顿环参数：视野缩放（仅牛顿环显示）
        self.scale_layout = QHBoxLayout()
        scale_label = QLabel("视野范围 (mm):")
        scale_label.setMinimumWidth(120)
        self.scale_layout.addWidget(scale_label)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)  # 1 mm
        self.scale_slider.setMaximum(100)  # 100 mm
        self.scale_slider.setValue(5)  # 默认 5 mm
        self.scale_slider.valueChanged.connect(self.on_parameter_changed)
        self.scale_layout.addWidget(self.scale_slider)
        
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setMinimum(0.1)
        self.scale_spinbox.setMaximum(100.0)
        self.scale_spinbox.setValue(5.0)
        self.scale_spinbox.setDecimals(2)
        self.scale_spinbox.setSuffix(" mm")
        self.scale_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.scale_layout.addWidget(self.scale_spinbox)
        
        param_layout.addLayout(self.scale_layout)
        
        # 劈尖干涉参数：夹角
        self.angle_layout = QHBoxLayout()
        angle_label = QLabel("劈尖夹角 (度):")
        angle_label.setMinimumWidth(120)
        self.angle_layout.addWidget(angle_label)
        
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setMinimum(1)  # 0.01 度
        self.angle_slider.setMaximum(1000)  # 10 度
        self.angle_slider.setValue(57)  # 约 0.057 度
        self.angle_slider.valueChanged.connect(self.on_parameter_changed)
        self.angle_layout.addWidget(self.angle_slider)
        
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setMinimum(0.01)
        self.angle_spinbox.setMaximum(10.0)
        self.angle_spinbox.setValue(0.057)
        self.angle_spinbox.setDecimals(3)
        self.angle_spinbox.setSuffix(" °")
        self.angle_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.angle_layout.addWidget(self.angle_spinbox)
        
        param_layout.addLayout(self.angle_layout)
        
        # 双缝干涉参数：缝宽
        self.slit_width_layout = QHBoxLayout()
        slit_width_label = QLabel("缝宽 (μm):")
        slit_width_label.setMinimumWidth(120)
        self.slit_width_layout.addWidget(slit_width_label)
        
        self.slit_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.slit_width_slider.setMinimum(1)
        self.slit_width_slider.setMaximum(100)
        self.slit_width_slider.setValue(10)
        self.slit_width_slider.valueChanged.connect(self.on_parameter_changed)
        self.slit_width_layout.addWidget(self.slit_width_slider)
        
        self.slit_width_spinbox = QDoubleSpinBox()
        self.slit_width_spinbox.setMinimum(1.0)
        self.slit_width_spinbox.setMaximum(100.0)
        self.slit_width_spinbox.setValue(10.0)
        self.slit_width_spinbox.setDecimals(1)
        self.slit_width_spinbox.setSuffix(" μm")
        self.slit_width_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.slit_width_layout.addWidget(self.slit_width_spinbox)
        
        param_layout.addLayout(self.slit_width_layout)
        
        # 双缝干涉参数：缝间距
        self.slit_spacing_layout = QHBoxLayout()
        slit_spacing_label = QLabel("缝间距 (μm):")
        slit_spacing_label.setMinimumWidth(120)
        self.slit_spacing_layout.addWidget(slit_spacing_label)
        
        self.slit_spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self.slit_spacing_slider.setMinimum(10)
        self.slit_spacing_slider.setMaximum(200)
        self.slit_spacing_slider.setValue(50)
        self.slit_spacing_slider.valueChanged.connect(self.on_parameter_changed)
        self.slit_spacing_layout.addWidget(self.slit_spacing_slider)
        
        self.slit_spacing_spinbox = QDoubleSpinBox()
        self.slit_spacing_spinbox.setMinimum(10.0)
        self.slit_spacing_spinbox.setMaximum(200.0)
        self.slit_spacing_spinbox.setValue(50.0)
        self.slit_spacing_spinbox.setDecimals(1)
        self.slit_spacing_spinbox.setSuffix(" μm")
        self.slit_spacing_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.slit_spacing_layout.addWidget(self.slit_spacing_spinbox)
        
        param_layout.addLayout(self.slit_spacing_layout)
        
        self.param_group.setLayout(param_layout)
        left_layout.addWidget(self.param_group)
        
        left_layout.addStretch()
        
        # 设置左侧面板样式
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        
        # 右侧：创建一个容器来包裹 OpenGL 显示区域
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_container.setLayout(right_layout)
        
        # 添加标题
        display_label = QLabel("仿真显示区域")
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
                background-color: #1a1d29;
                border: 3px solid #4f7cff;
                border-radius: 8px;
            }
        """)
        
        display_frame_layout = QVBoxLayout()
        display_frame_layout.setContentsMargins(5, 5, 5, 5)
        display_frame.setLayout(display_frame_layout)
        
        # OpenGL 显示区域（固定尺寸，类似画布）
        self.simulation_widget = SimulationWidget()
        # 设置固定大小，保持 4:3 的宽高比
        self.simulation_widget.setFixedSize(800, 600)
        self.simulation_widget.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Fixed
        )
        
        display_frame_layout.addWidget(self.simulation_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(display_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        
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
        
        # 同步滑块和数值框
        self.wavelength_slider.valueChanged.connect(
            lambda v: self.wavelength_spinbox.setValue(float(v))
        )
        self.wavelength_spinbox.valueChanged.connect(
            lambda v: self.wavelength_slider.setValue(int(v))
        )
        
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_spinbox.setValue(float(v))
        )
        self.radius_spinbox.valueChanged.connect(
            lambda v: self.radius_slider.setValue(int(v))
        )
        
        self.gap_slider.valueChanged.connect(
            lambda v: self.gap_spinbox.setValue(float(v))
        )
        self.gap_spinbox.valueChanged.connect(
            lambda v: self.gap_slider.setValue(int(v))
        )
        
        # 视野缩放滑块同步
        self.scale_slider.valueChanged.connect(
            lambda v: self.scale_spinbox.setValue(float(v))
        )
        self.scale_spinbox.valueChanged.connect(
            lambda v: self.scale_slider.setValue(int(v))
        )
        
        # 夹角滑块同步（度转弧度）
        self.angle_slider.valueChanged.connect(
            lambda v: self.angle_spinbox.setValue(float(v) / 1000.0)
        )
        self.angle_spinbox.valueChanged.connect(
            lambda v: self.angle_slider.setValue(int(v * 1000.0))
        )
        
        self.slit_width_slider.valueChanged.connect(
            lambda v: self.slit_width_spinbox.setValue(float(v))
        )
        self.slit_width_spinbox.valueChanged.connect(
            lambda v: self.slit_width_slider.setValue(int(v))
        )
        
        self.slit_spacing_slider.valueChanged.connect(
            lambda v: self.slit_spacing_spinbox.setValue(float(v))
        )
        self.slit_spacing_spinbox.valueChanged.connect(
            lambda v: self.slit_spacing_slider.setValue(int(v))
        )
        
        # 初始化：显示牛顿环参数
        self.on_experiment_changed("牛顿环实验")
        self.on_parameter_changed()
    
    def on_experiment_changed(self, text):
        """实验类型改变时的回调"""
        # 获取实验类型索引
        experiment_types = ["牛顿环实验", "劈尖干涉", "双缝干涉"]
        experiment_type = experiment_types.index(text) if text in experiment_types else 0
        
        # 根据实验类型显示/隐藏相应的参数控件
        if experiment_type == 0:  # 牛顿环
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
        
        elif experiment_type == 1:  # 劈尖干涉
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
        
        elif experiment_type == 2:  # 双缝干涉
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
        
        # 触发参数更新
        self.on_parameter_changed()
    
    def on_parameter_changed(self):
        """参数改变时的回调"""
        wavelength = self.wavelength_spinbox.value()
        experiment_types = ["牛顿环实验", "劈尖干涉", "双缝干涉"]
        current_text = self.experiment_combo.currentText()
        experiment_type = experiment_types.index(current_text) if current_text in experiment_types else 0
        
        # 根据实验类型传递不同的参数
        # 所有实验都使用视野范围参数
        scale = self.scale_spinbox.value()
        
        if experiment_type == 0:  # 牛顿环
            radius = self.radius_spinbox.value()
            gap_distance = self.gap_spinbox.value()
            self.simulation_widget.update_parameters(
                experiment_type, wavelength, scale=scale,
                radius=radius, gap_distance=gap_distance
            )
        
        elif experiment_type == 1:  # 劈尖干涉
            angle_degrees = self.angle_spinbox.value()
            angle_radians = angle_degrees * 3.14159265359 / 180.0  # 度转弧度
            gap_distance = self.gap_spinbox.value()
            self.simulation_widget.update_parameters(
                experiment_type, wavelength, scale=scale,
                angle=angle_radians, gap_distance=gap_distance
            )
        
        elif experiment_type == 2:  # 双缝干涉
            slit_width = self.slit_width_spinbox.value()
            slit_spacing = self.slit_spacing_spinbox.value()
            # 双缝干涉不使用 scale 参数，使用固定比例
            self.simulation_widget.update_parameters(
                experiment_type, wavelength,
                slit_width=slit_width, slit_spacing=slit_spacing
            )


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化主窗口界面"""
        self.setWindowTitle("PhysicsLab Pro - 物理实验辅助工具")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        
        # 创建三个主要页面
        self.optics_tab = OpticsLabTab()
        self.data_tab = DataWorkstationTab()
        self.virtual_lab_tab = VirtualLabTab()
        
        # 添加选项卡
        self.tab_widget.addTab(self.optics_tab, "光学 AI 实验室")
        self.tab_widget.addTab(self.data_tab, "数据工作台")
        self.tab_widget.addTab(self.virtual_lab_tab, "虚拟仿真实验室")
        
        # 设置选项卡为中央部件
        self.setCentralWidget(self.tab_widget)
        
        # 设置窗口样式
        self.setStyleSheet("""
            /* 主窗口背景 - 深蓝灰色渐变风格 */
            QMainWindow {
                background-color: #1a1d29;
            }
            
            /* 选项卡面板 */
            QTabWidget::pane {
                border: 2px solid #2d3142;
                border-radius: 8px;
                background-color: #252836;
                top: -1px;
            }
            
            /* 选项卡标签 */
            QTabBar::tab {
                background-color: #2d3142;
                color: #b0b8c4;
                padding: 10px 24px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            
            QTabBar::tab:hover {
                background-color: #3a3f5c;
                color: #ffffff;
            }
            
            QTabBar::tab:selected {
                background-color: #4f7cff;
                color: #ffffff;
                font-weight: 600;
            }
            
            /* 按钮样式 - 现代蓝色渐变 */
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5b8fff, stop:1 #4a7cff);
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6b9fff, stop:1 #5a8cff);
            }
            
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a7cff, stop:1 #3a6cff);
            }
            
            /* 表格样式 */
            QTableWidget {
                background-color: #1e2132;
                color: #e0e4eb;
                border: 2px solid #2d3142;
                border-radius: 6px;
                gridline-color: #2d3142;
                font-size: 12px;
            }
            
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            
            QTableWidget::item:selected {
                background-color: #4f7cff;
                color: #ffffff;
            }
            
            QHeaderView::section {
                background-color: #2d3142;
                color: #b0b8c4;
                padding: 8px;
                border: none;
                font-weight: 600;
                font-size: 12px;
            }
            
            /* 输入框和文本区域 */
            QLineEdit, QTextEdit {
                background-color: #1e2132;
                color: #e0e4eb;
                border: 2px solid #2d3142;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
                selection-background-color: #4f7cff;
            }
            
            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #4f7cff;
                background-color: #252836;
            }
            
            /* 标签样式 */
            QLabel {
                color: #e0e4eb;
                font-size: 12px;
            }
            
            /* 进度条样式 */
            QProgressBar {
                border: 2px solid #2d3142;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-weight: 600;
                background-color: #1e2132;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5b8fff, stop:1 #4a7cff);
                border-radius: 4px;
            }
        """)


def main():
    """主函数：启动应用程序"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

