import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


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
            background-color: #f5f7fa;
            color: #333333;
            border: 2px solid #d9d9d9;
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

    # 配色方案
    THEMES = {
        'dark': {
            'background': '#1e2132',
            'face': '#1e2132',
            'axis': '#b0b8c4',
            'text': '#e0e4eb',
            'grid': '#3a3f5c',
            'primary': '#4a90e2',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c'
        },
        'modern': {
            'background': '#f8f9fa',
            'face': '#ffffff',
            'axis': '#495057',
            'text': '#212529',
            'grid': '#dee2e6',
            'primary': '#3498db',
            'success': '#2ecc71',
            'warning': '#e67e22',
            'danger': '#e74c3c',
            'purple': '#9b59b6'
        }
    }

    def __init__(self, parent=None, width=5, height=4, dpi=100, theme='dark'):
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        self.theme_name = theme
        self.theme = self.THEMES[theme]

        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=self.theme['background'])
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.theme['face'])
        self.apply_theme()

    def set_theme(self, theme_name):
        """切换主题：'dark' 或 'modern'"""
        if theme_name in self.THEMES:
            self.theme_name = theme_name
            self.theme = self.THEMES[theme_name]
            self.fig.set_facecolor(self.theme['background'])
            self.ax.set_facecolor(self.theme['face'])
            self.apply_theme()

    def apply_theme(self):
        """应用当前主题样式"""
        self.ax.tick_params(colors=self.theme['axis'])
        self.ax.spines['bottom'].set_color(self.theme['axis'])
        self.ax.spines['top'].set_color(self.theme['axis'])
        self.ax.spines['right'].set_color(self.theme['axis'])
        self.ax.spines['left'].set_color(self.theme['axis'])
        self.ax.xaxis.label.set_color(self.theme['text'])
        self.ax.yaxis.label.set_color(self.theme['text'])
        self.ax.title.set_color(self.theme['text'])

    def apply_compact_layout(self):
        """Use fixed margins to keep plots inside the canvas."""
        self.fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.18)

    @staticmethod
    def compute_signal_limits(*series):
        """Compute stable y-axis limits from one or more signals."""
        valid_arrays = []
        for data in series:
            if data is None:
                continue
            array = np.asarray(data, dtype=float)
            if array.size == 0:
                continue
            valid_arrays.append(array)

        if not valid_arrays:
            return None

        combined = np.concatenate(valid_arrays)
        y_min = float(np.min(combined))
        y_max = float(np.max(combined))
        y_range = y_max - y_min
        padding = max(y_range * 0.12, 1.0) if y_range > 0 else max(abs(y_max) * 0.08, 1.0)
        return y_min - padding, y_max + padding
    
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
        
        self.ax.set_xlabel('距离 (像素)', color=self.theme['text'])
        self.ax.set_ylabel('亮度值', color=self.theme['text'])
        self.ax.set_title('光强分布 vs 像素位置', color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--')
        self.ax.legend(loc='best', framealpha=0.8)
        
        self.apply_compact_layout()
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
                    alpha=0.6, label='\u539f\u59cb\u4fe1\u53f7', zorder=1, clip_on=True)
        
        # 绘制平滑后的信号（更明显）
        if smoothed_intensities is not None and len(smoothed_intensities) > 0:
            # 确保长度匹配
            min_len = min(len(frame_numbers), len(smoothed_intensities))
            if min_len > 0:
                self.ax.plot(frame_numbers[:min_len], smoothed_intensities[:min_len],
                           'lime', linewidth=2.5, label='\u5e73\u6ed1\u4fe1\u53f7', zorder=2, clip_on=True)
        
        self.ax.set_xlabel('帧数', color=self.theme['text'])
        self.ax.set_ylabel('亮度值', color=self.theme['text'])
        self.ax.set_title('实时亮度信号（最近 100 帧）', color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--')
        
        # 固定图例位置，避免乱飞
        self.ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        
        # Adjust the axes using both raw and smoothed signals.
        if len(frame_numbers) > 0:
            self.ax.set_xlim(frame_numbers[0] - 2, frame_numbers[-1] + 2)
            signal_limits = self.compute_signal_limits(intensities, smoothed_intensities)
            if signal_limits is not None:
                self.ax.set_ylim(*signal_limits)
            self.ax.margins(x=0.02, y=0.0)

        self.apply_compact_layout()
        self.draw()

    def plot_boxplot(self, data, title='箱型图', xlabel='数据', ylabel='数值'):
        """绘制箱型图"""
        self.ax.clear()

        # 绘制箱型图
        if isinstance(data, list):
            data = np.array(data)

        if len(data) == 0:
            self.ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                       transform=self.ax.transAxes, color=self.theme['text'])
        else:
            box = self.ax.boxplot(data, patch_artist=True, widths=0.5,
                               boxprops=dict(facecolor=self.theme['primary'], alpha=0.7),
                               medianprops=dict(color=self.theme['danger'], linewidth=2),
                               whiskerprops=dict(color=self.theme['axis'], linewidth=1.5),
                               capprops=dict(color=self.theme['axis'], linewidth=1.5))

            # 添加数据点
            x = np.random.normal(1, 0.04, size=len(data))
            self.ax.plot(x, data, 'o', alpha=0.4, color=self.theme['success'],
                       markersize=6, markeredgecolor=self.theme['axis'])

        self.ax.set_xticks([1])
        self.ax.set_xticklabels([xlabel])
        self.ax.set_xlabel(xlabel, color=self.theme['text'])
        self.ax.set_ylabel(ylabel, color=self.theme['text'])
        self.ax.set_title(title, color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--')

        self.apply_theme()
        self.apply_compact_layout()
        self.draw()

    def plot_line_chart(self, x_data, y_data, title='折线图', xlabel='X', ylabel='Y'):
        """绘制折线图"""
        self.ax.clear()

        if isinstance(x_data, list):
            x_data = np.array(x_data)
        if isinstance(y_data, list):
            y_data = np.array(y_data)

        if len(x_data) == 0 or len(y_data) == 0:
            self.ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                       transform=self.ax.transAxes, color=self.theme['text'])
        else:
            # 绘制折线
            self.ax.plot(x_data, y_data, 'o-', color=self.theme['primary'],
                       linewidth=2, markersize=8, markerfacecolor=self.theme['face'],
                       markeredgecolor=self.theme['primary'], markeredgewidth=2)

            # 添加渐变填充区域
            self.ax.fill_between(x_data, y_data, alpha=0.2, color=self.theme['primary'])

        self.ax.set_xlabel(xlabel, color=self.theme['text'])
        self.ax.set_ylabel(ylabel, color=self.theme['text'])
        self.ax.set_title(title, color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--')

        self.apply_theme()
        self.apply_compact_layout()
        self.draw()

    def plot_bar_chart(self, x_data, y_data, title='柱状图', xlabel='X', ylabel='Y'):
        """绘制柱状图"""
        self.ax.clear()

        if isinstance(x_data, list):
            x_data = np.array(x_data)
        if isinstance(y_data, list):
            y_data = np.array(y_data)

        if len(x_data) == 0 or len(y_data) == 0:
            self.ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                       transform=self.ax.transAxes, color=self.theme['text'])
        else:
            # 绘制柱状图
            bars = self.ax.bar(x_data, y_data, color=self.theme['primary'],
                              alpha=0.8, edgecolor=self.theme['primary'],
                              linewidth=1)

            # 为每个柱子添加渐变效果
            for bar in bars:
                bar.set_facecolor(self.theme['primary'])
                bar.set_alpha(0.8)

        self.ax.set_xlabel(xlabel, color=self.theme['text'])
        self.ax.set_ylabel(ylabel, color=self.theme['text'])
        self.ax.set_title(title, color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--', axis='y')

        self.apply_theme()
        self.apply_compact_layout()
        self.draw()

    def plot_scatter(self, x_data, y_data, title='散点图', xlabel='X', ylabel='Y'):
        """绘制散点图"""
        self.ax.clear()

        if isinstance(x_data, list):
            x_data = np.array(x_data)
        if isinstance(y_data, list):
            y_data = np.array(y_data)

        if len(x_data) == 0 or len(y_data) == 0:
            self.ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                       transform=self.ax.transAxes, color=self.theme['text'])
        else:
            # 绘制散点
            self.ax.scatter(x_data, y_data, s=80, alpha=0.7,
                          color=self.theme['primary'], edgecolors=self.theme['face'],
                          linewidths=1.5)

            # 添加趋势线
            if len(x_data) > 1 and len(y_data) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(np.min(x_data), np.max(x_data), 100)
                self.ax.plot(x_trend, p(x_trend), '--', color=self.theme['warning'],
                           linewidth=2, alpha=0.7, label='趋势线')

        self.ax.set_xlabel(xlabel, color=self.theme['text'])
        self.ax.set_ylabel(ylabel, color=self.theme['text'])
        self.ax.set_title(title, color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--')

        self.apply_theme()
        self.apply_compact_layout()
        self.draw()

    def plot_histogram(self, data, title='直方图', xlabel='数值', ylabel='频数', bins=None):
        """绘制直方图"""
        self.ax.clear()

        if isinstance(data, list):
            data = np.array(data)

        if len(data) == 0:
            self.ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                       transform=self.ax.transAxes, color=self.theme['text'])
        else:
            # 自动确定分箱数量
            if bins is None:
                bins = min(30, max(5, int(len(data) ** 0.5)))

            # 绘制直方图
            n, bins, patches = self.ax.hist(data, bins=bins, alpha=0.7,
                                         color=self.theme['primary'],
                                         edgecolor=self.theme['axis'],
                                         linewidth=1)

            # 添加均值线
            mean = np.mean(data)
            self.ax.axvline(mean, color=self.theme['danger'], linestyle='--',
                           linewidth=2, label=f'均值 = {mean:.2f}')

            # 添加标准差范围
            std = np.std(data)
            self.ax.axvspan(mean - std, mean + std, alpha=0.2,
                            color=self.theme['success'], label='±1σ')

        self.ax.set_xlabel(xlabel, color=self.theme['text'])
        self.ax.set_ylabel(ylabel, color=self.theme['text'])
        self.ax.set_title(title, color=self.theme['text'])
        self.ax.grid(True, alpha=0.3, color=self.theme['grid'], linestyle='--', axis='y')
        self.ax.legend(loc='best', framealpha=0.9)

        self.apply_theme()
        self.apply_compact_layout()
        self.draw()
