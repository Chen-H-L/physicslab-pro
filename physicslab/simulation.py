import math

import numpy as np
from OpenGL import GL as gl
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QVector2D
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PyQt6.QtOpenGL import QOpenGLVertexArrayObject
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class LegacyOpenGLSimulationWidget(QOpenGLWidget):
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


class SimulationWidget(QWidget):
    """Stable CPU simulation widget."""

    _EXPERIMENT_NAMES = {
        0: "Newton Rings",
        1: "Wedge Interference",
        2: "Double Slit",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumSize(320, 240)

        self.experiment_type = 0
        self.wavelength = 632.8
        self.scale = 5.0
        self.radius = 1000.0
        self.gap_distance = 0.0
        self.angle = 0.001
        self.slit_width = 10.0
        self.slit_spacing = 50.0

        self._cached_image = None
        self._cache_key = None

    def resizeEvent(self, event):
        self._invalidate_cache()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#171e29'))

        image = self._render_pattern()
        if image is not None:
            painter.drawImage(self.rect(), image)

        self._draw_overlay(painter)
        painter.end()

    def update_parameters(self, experiment_type, wavelength, scale=None, **kwargs):
        self.experiment_type = experiment_type
        self.wavelength = wavelength
        if scale is not None:
            self.scale = scale

        if experiment_type == 0:
            self.radius = kwargs.get('radius', self.radius)
            self.gap_distance = kwargs.get('gap_distance', self.gap_distance)
        elif experiment_type == 1:
            self.angle = kwargs.get('angle', self.angle)
            self.gap_distance = kwargs.get('gap_distance', self.gap_distance)
        elif experiment_type == 2:
            self.slit_width = kwargs.get('slit_width', self.slit_width)
            self.slit_spacing = kwargs.get('slit_spacing', self.slit_spacing)

        self._invalidate_cache()
        self.update()

    def _invalidate_cache(self):
        self._cached_image = None
        self._cache_key = None

    def _render_pattern(self):
        width = max(2, self.width())
        height = max(2, self.height())
        cache_key = (
            width,
            height,
            int(self.experiment_type),
            round(float(self.wavelength), 4),
            round(float(self.scale), 4),
            round(float(self.radius), 4),
            round(float(self.gap_distance), 4),
            round(float(self.angle), 8),
            round(float(self.slit_width), 4),
            round(float(self.slit_spacing), 4),
        )
        if cache_key == self._cache_key and self._cached_image is not None:
            return self._cached_image

        side = float(min(width, height))
        x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / side
        y = (((height - 1) / 2.0) - np.arange(height, dtype=np.float32)) / side
        xx, yy = np.meshgrid(x, y)

        intensity = self._compute_intensity(xx, yy)
        intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32)
        intensity = np.power(intensity, 0.85)

        background = np.array([0.08, 0.10, 0.16], dtype=np.float32)
        base_color = np.array(self._wavelength_to_rgb(self.wavelength), dtype=np.float32)
        tinted_color = np.clip(base_color * 0.95 + 0.05, 0.0, 1.0)
        rgb = background + intensity[..., None] * (tinted_color - background)
        rgb = np.clip(rgb, 0.0, 1.0)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255.0).astype(np.uint8)
        rgba[..., 3] = 255

        image = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
        self._cached_image = image
        self._cache_key = cache_key
        return image

    def _compute_intensity(self, xx, yy):
        wavelength = max(float(self.wavelength), 1e-6)

        if self.experiment_type == 0:
            r_mm = np.hypot(xx, yy) * float(self.scale)
            r_nm = r_mm * 1_000_000.0
            radius_nm = max(float(self.radius), 1e-6) * 1_000_000.0
            d = (r_nm * r_nm) / (2.0 * radius_nm) + float(self.gap_distance)
            return np.cos(2.0 * np.pi * d / wavelength) ** 2

        if self.experiment_type == 1:
            x_mm = xx * float(self.scale)
            d = x_mm * 1_000_000.0 * np.tan(float(self.angle)) + float(self.gap_distance)
            return np.cos(2.0 * np.pi * d / wavelength) ** 2

        x_mm = xx * 100.0
        sin_theta = x_mm / 1000.0
        slit_spacing_nm = max(float(self.slit_spacing), 1e-6) * 1000.0
        slit_width_nm = max(float(self.slit_width), 1e-6) * 1000.0

        interference_phase = np.pi * slit_spacing_nm * sin_theta / wavelength
        interference = np.cos(interference_phase) ** 2

        diffraction_phase = np.pi * slit_width_nm * sin_theta / wavelength
        sinc_value = np.ones_like(diffraction_phase)
        mask = np.abs(diffraction_phase) > 1e-4
        sinc_value[mask] = np.sin(diffraction_phase[mask]) / diffraction_phase[mask]
        diffraction = sinc_value ** 2
        return interference * diffraction

    def _draw_overlay(self, painter):
        painter.setPen(QPen(QColor(255, 255, 255, 55), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 8, 8)

    @staticmethod
    def _wavelength_to_rgb(wavelength_nm):
        wavelength_nm = float(wavelength_nm)
        if wavelength_nm < 400.0:
            return (0.5, 0.0, 1.0)
        if wavelength_nm > 700.0:
            return (1.0, 0.0, 0.0)
        if wavelength_nm < 440.0:
            return (
                -(wavelength_nm - 440.0) / 40.0,
                0.0,
                1.0,
            )
        if wavelength_nm < 490.0:
            return (0.0, (wavelength_nm - 440.0) / 50.0, 1.0)
        if wavelength_nm < 510.0:
            return (0.0, 1.0, -(wavelength_nm - 510.0) / 20.0)
        if wavelength_nm < 580.0:
            return ((wavelength_nm - 510.0) / 70.0, 1.0, 0.0)
        if wavelength_nm < 645.0:
            return (1.0, -(wavelength_nm - 645.0) / 65.0, 0.0)
        return (1.0, 0.0, 0.0)


class SimulationModelWidget(QWidget):
    """Schematic apparatus model linked to the optical simulation parameters."""

    _TITLE_MAP = {
        0: "牛顿环实验模型",
        1: "劈尖干涉实验模型",
        2: "双缝干涉实验模型",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 210)

        self.experiment_type = 0
        self.wavelength = 632.8
        self.scale = 5.0
        self.radius = 1000.0
        self.gap_distance = 0.0
        self.angle = 0.001
        self.slit_width = 10.0
        self.slit_spacing = 50.0

    def update_parameters(self, experiment_type, wavelength, scale=None, **kwargs):
        self.experiment_type = experiment_type
        self.wavelength = wavelength
        if scale is not None:
            self.scale = scale

        if experiment_type == 0:
            self.radius = kwargs.get('radius', self.radius)
            self.gap_distance = kwargs.get('gap_distance', self.gap_distance)
        elif experiment_type == 1:
            self.angle = kwargs.get('angle', self.angle)
            self.gap_distance = kwargs.get('gap_distance', self.gap_distance)
        elif experiment_type == 2:
            self.slit_width = kwargs.get('slit_width', self.slit_width)
            self.slit_spacing = kwargs.get('slit_spacing', self.slit_spacing)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#ffffff'))

        panel_rect = QRectF(self.rect().adjusted(10, 10, -10, -10))
        painter.setPen(QPen(QColor('#d5dfeb'), 1.2))
        painter.setBrush(QColor('#fbfdff'))
        painter.drawRoundedRect(panel_rect, 12, 12)

        content_rect = panel_rect.adjusted(18, 14, -18, -16)
        self._draw_model_title(
            painter,
            QRectF(content_rect.left(), content_rect.top(), content_rect.width(), 22),
            self._TITLE_MAP.get(self.experiment_type, "实验模型"),
        )

        diagram_rect = content_rect.adjusted(0, 28, 0, 0)
        if self.experiment_type == 0:
            self._draw_newton_rings_model(painter, diagram_rect)
        elif self.experiment_type == 1:
            self._draw_wedge_model(painter, diagram_rect)
        else:
            self._draw_double_slit_model(painter, diagram_rect)

        painter.end()

    def _draw_newton_rings_model(self, painter, rect):
        light_color = self._qt_color_from_wavelength(self.wavelength, 230)
        outline_pen = QPen(QColor('#202a38'), 2.0)
        thin_pen = QPen(QColor('#7f91a6'), 1.1, Qt.PenStyle.DashLine)
        label_font = QFont()
        label_font.setPointSize(10)

        plate_top = rect.bottom() - 34
        plate_rect = QRectF(rect.left() + 14, plate_top, rect.width() - 28, 22)
        painter.setPen(QPen(QColor('#2b3646'), 1.8))
        painter.setBrush(QColor(234, 241, 250, 225))
        painter.drawRoundedRect(plate_rect, 2, 2)

        axis_x = rect.left() + rect.width() * 0.48
        center_gap_px = self._clamp(2.0 + self.gap_distance / 180.0, 2.0, 12.0)
        contact_y = plate_top - center_gap_px
        max_radius_px = max(contact_y - (rect.top() + 34.0), 46.0)
        radius_px = min(float(np.interp(self.radius, [500.0, 5000.0], [92.0, 138.0])), max_radius_px)
        center_y = contact_y - radius_px
        max_dx = radius_px * 0.82

        curve_points = []
        for dx in np.linspace(-max_dx, max_dx, 180):
            y = center_y + math.sqrt(max(radius_px * radius_px - dx * dx, 0.0))
            curve_points.append(QPointF(axis_x + dx, y))
        surface_path = QPainterPath(curve_points[0])
        for point in curve_points[1:]:
            surface_path.lineTo(point)

        cap_height = max(radius_px * 0.55, 28.0)
        cap_top_y = max(rect.top() + 8.0, curve_points[0].y() - cap_height)
        top_left = QPointF(axis_x - max_dx * 0.86, cap_top_y)
        top_right = QPointF(axis_x + max_dx * 0.86, cap_top_y)
        body_path = QPainterPath(curve_points[0])
        body_path.quadTo(QPointF(axis_x, cap_top_y - 10.0), curve_points[-1])
        body_path.lineTo(top_right)
        body_path.quadTo(QPointF(axis_x, max(rect.top() + 4.0, cap_top_y - 22.0)), top_left)
        body_path.closeSubpath()

        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QColor(225, 234, 246, 170))
        painter.drawPath(body_path)
        painter.setPen(outline_pen)
        painter.drawPath(surface_path)

        optical_axis_top = rect.top() + 18
        painter.setPen(thin_pen)
        painter.drawLine(QPointF(axis_x, optical_axis_top), QPointF(axis_x, plate_rect.bottom() + 8))

        for x_factor in (-0.26, -0.08, 0.10, 0.28):
            x_pos = axis_x + rect.width() * x_factor
            self._draw_arrow(
                painter,
                QPointF(x_pos, rect.top() + 6),
                QPointF(x_pos, rect.top() + 56),
                light_color,
                width=2.0,
                head_size=7.0,
            )
        painter.setFont(label_font)
        painter.setPen(QColor('#34495e'))
        painter.drawText(QPointF(rect.left() + 22, rect.top() + 12), "入射光")

        lambda_mm = max(self.wavelength * 1e-6, 1e-9)
        gap_mm = max(self.gap_distance * 1e-6, 0.0)
        sample_order = 5.0
        sample_r_mm = math.sqrt(max(self.radius * (sample_order * lambda_mm + 2.0 * gap_mm), 0.0))
        sample_fraction = self._clamp(sample_r_mm / 5.8, 0.20, 0.76)
        sample_dx = max_dx * sample_fraction
        sample_x = axis_x + sample_dx
        sample_y = center_y + math.sqrt(max(radius_px * radius_px - sample_dx * sample_dx, 0.0))

        radius_origin = QPointF(axis_x, center_y)
        painter.setPen(QPen(QColor('#24374f'), 1.7))
        painter.drawLine(radius_origin, QPointF(sample_x, sample_y))
        painter.setFont(label_font)
        painter.setPen(QColor('#1f2733'))
        painter.drawText(
            QPointF((radius_origin.x() + sample_x) / 2.0 + 8, (radius_origin.y() + sample_y) / 2.0 - 2),
            "R",
        )

        self._draw_arrow(
            painter,
            QPointF(sample_x, sample_y),
            QPointF(sample_x, plate_top),
            QColor('#50657b'),
            width=1.6,
            head_size=5.5,
            both_ends=True,
        )
        painter.setFont(label_font)
        painter.setPen(QColor('#1f2733'))
        painter.drawText(QPointF(sample_x + 8, (sample_y + plate_top) / 2.0 + 4), "d")

        radius_line_y = plate_top - 8
        self._draw_arrow(
            painter,
            QPointF(axis_x, radius_line_y),
            QPointF(sample_x, radius_line_y),
            QColor(light_color.red(), light_color.green(), light_color.blue(), 220),
            width=1.8,
            head_size=5.2,
            both_ends=True,
        )
        painter.setFont(label_font)
        painter.setPen(QColor('#1f2733'))
        painter.drawText(QPointF((axis_x + sample_x) / 2.0 - 4, radius_line_y - 6), "r")

        painter.setBrush(QColor('#ffffff'))
        painter.setPen(QPen(QColor('#ff5a2c'), 1.8))
        painter.drawEllipse(QPointF(sample_x, radius_line_y), 3.2, 3.2)
        painter.setFont(label_font)
        painter.setPen(QColor('#1f2733'))
        painter.drawText(QPointF(axis_x - 10, plate_top + 18), "O")

        parameter_rect = QRectF(rect.right() - 156, rect.top() + 4, 146, 88)
        self._draw_parameter_panel(
            painter,
            parameter_rect,
            [
                ("λ", f"{self.wavelength:.1f} nm"),
                ("R", f"{self.radius:.1f} mm"),
                ("d", f"{self.gap_distance:.1f} nm"),
                ("L", f"{self.scale:.2f} mm"),
            ],
        )

    def _draw_wedge_model(self, painter, rect):
        light_color = self._qt_color_from_wavelength(self.wavelength, 230)
        outline_pen = QPen(QColor('#202a38'), 2.0)
        label_font = QFont()
        label_font.setPointSize(10)
        panel_width = 146.0
        panel_gap = 22.0
        base_y = rect.bottom() - 34
        plate_rect = QRectF(rect.left() + 20, base_y, rect.width() - 40, 18)

        painter.setPen(QPen(QColor('#2b3646'), 1.6))
        painter.setBrush(QColor(234, 241, 250, 225))
        painter.drawRoundedRect(plate_rect, 2, 2)

        left_x = rect.left() + 42
        right_x = rect.right() - (panel_width + panel_gap)
        gap_px = self._clamp(3.0 + self.gap_distance / 180.0, 3.0, 14.0)
        opening_px = self._clamp(14.0 + self.angle * 9000.0, 14.0, 52.0)
        thin_y = base_y - gap_px
        thick_y = thin_y - opening_px

        upper_glass = QPainterPath(QPointF(left_x, thin_y))
        upper_glass.lineTo(QPointF(right_x, thick_y))
        upper_glass.lineTo(QPointF(right_x, thick_y - 14))
        upper_glass.lineTo(QPointF(left_x, thin_y - 14))
        upper_glass.closeSubpath()

        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QColor(223, 232, 246, 180))
        painter.drawPath(upper_glass)
        painter.setPen(outline_pen)
        painter.drawLine(QPointF(left_x, thin_y), QPointF(right_x, thick_y))

        for x_factor in (-0.32, -0.14, 0.04, 0.22):
            x_pos = rect.center().x() + rect.width() * x_factor
            target_y = thin_y - 10 if x_pos <= rect.center().x() else thick_y - 10
            self._draw_arrow(
                painter,
                QPointF(x_pos, rect.top() + 8),
                QPointF(x_pos, target_y),
                light_color,
                width=2.0,
                head_size=7.0,
            )
        painter.setFont(label_font)
        painter.setPen(QColor('#34495e'))
        painter.drawText(QPointF(rect.left() + 22, rect.top() + 14), "入射光")

        marker_x = right_x - (right_x - left_x) * 0.18
        marker_y = thin_y + (thick_y - thin_y) * ((marker_x - left_x) / max(right_x - left_x, 1.0))
        self._draw_arrow(
            painter,
            QPointF(marker_x + 20, marker_y),
            QPointF(marker_x + 20, base_y),
            QColor('#50657b'),
            width=1.6,
            head_size=5.5,
            both_ends=True,
        )
        painter.setFont(label_font)
        painter.setPen(QColor('#1f2733'))
        painter.drawText(QPointF(marker_x + 26, (marker_y + base_y) / 2.0 + 4), "d")

        alpha_rect = QRectF(left_x + 14, thin_y - 4, 40, 30)
        painter.setPen(QPen(QColor('#51657c'), 1.5))
        painter.drawArc(alpha_rect, -16 * 8, -16 * 26)
        painter.setFont(label_font)
        painter.drawText(QPointF(left_x + 44, thin_y - 2), "α")

        spacing_px = self._clamp(10.0 + 180.0 * self.wavelength / max(self.angle * 1e6, 1.0), 10.0, 30.0)
        painter.setPen(QPen(QColor(light_color.red(), light_color.green(), light_color.blue(), 145), 1.8))
        stripe_x = left_x + 30
        while stripe_x < right_x - 18:
            top_y = thin_y + (thick_y - thin_y) * ((stripe_x - left_x) / max(right_x - left_x, 1.0))
            painter.drawLine(QPointF(stripe_x, base_y - 2), QPointF(stripe_x, top_y + 2))
            stripe_x += spacing_px

        self._draw_parameter_panel(
            painter,
            QRectF(rect.right() - panel_width, rect.top() + 4, panel_width, 88),
            [
                ("λ", f"{self.wavelength:.1f} nm"),
                ("α", f"{math.degrees(self.angle):.3f}°"),
                ("d", f"{self.gap_distance:.1f} nm"),
                ("L", f"{self.scale:.2f} mm"),
            ],
        )

    def _draw_double_slit_model(self, painter, rect):
        light_color = self._qt_color_from_wavelength(self.wavelength, 235)
        outline_pen = QPen(QColor('#202a38'), 2.0)
        label_font = QFont()
        label_font.setPointSize(10)
        panel_width = 138.0
        panel_gap = 26.0
        source_x = rect.left() + 34
        barrier_x = rect.left() + rect.width() * 0.40
        screen_x = rect.right() - (panel_width + panel_gap)
        center_y = rect.center().y()

        painter.setPen(outline_pen)
        painter.drawLine(QPointF(barrier_x, rect.top() + 14), QPointF(barrier_x, rect.bottom() - 14))
        painter.drawLine(QPointF(screen_x, rect.top() + 8), QPointF(screen_x, rect.bottom() - 8))

        slit_spacing_px = self._clamp(18.0 + (self.slit_spacing - 10.0) * 0.28, 18.0, 64.0)
        slit_half_height = self._clamp(3.0 + self.slit_width * 0.08, 3.0, 11.0)
        upper_slit_y = center_y - slit_spacing_px / 2.0
        lower_slit_y = center_y + slit_spacing_px / 2.0

        painter.setPen(QPen(QColor('#fbfdff'), 7))
        painter.drawLine(QPointF(barrier_x, upper_slit_y - slit_half_height), QPointF(barrier_x, upper_slit_y + slit_half_height))
        painter.drawLine(QPointF(barrier_x, lower_slit_y - slit_half_height), QPointF(barrier_x, lower_slit_y + slit_half_height))

        source_rect = QRectF(source_x - 10, center_y - 10, 20, 20)
        painter.setPen(QPen(QColor('#32465a'), 1.5))
        painter.setBrush(QColor(light_color.red(), light_color.green(), light_color.blue(), 60))
        painter.drawEllipse(source_rect)
        painter.setFont(label_font)
        painter.setPen(QColor('#1f2733'))
        painter.drawText(QPointF(source_x - 4, center_y - 16), "S")

        self._draw_arrow(
            painter,
            QPointF(source_x + 12, center_y),
            QPointF(barrier_x - 18, center_y),
            light_color,
            width=2.4,
            head_size=8.0,
        )

        fringe_spacing = self._clamp(5.0 + 220.0 * self.wavelength / max(self.slit_spacing * 1000.0, 1.0), 6.0, 16.0)
        envelope_half = self._clamp(22.0 + 220.0 * self.wavelength / max(self.slit_width * 1000.0, 1.0), 28.0, rect.height() * 0.34)

        painter.setPen(QPen(light_color, 1.8))
        painter.drawLine(QPointF(barrier_x, upper_slit_y), QPointF(screen_x, center_y - fringe_spacing * 2.0))
        painter.drawLine(QPointF(barrier_x, lower_slit_y), QPointF(screen_x, center_y + fringe_spacing * 2.0))

        painter.setPen(QPen(Qt.PenStyle.NoPen))
        band_index = 0
        offset = -envelope_half
        while offset <= envelope_half:
            distance_ratio = 1.0 - abs(offset) / max(envelope_half, 1.0)
            alpha = int(self._clamp(50 + distance_ratio * 160 - (band_index % 2) * 70, 25, 210))
            band_color = QColor(light_color.red(), light_color.green(), light_color.blue(), alpha)
            painter.setBrush(band_color)
            painter.drawRoundedRect(
                QRectF(screen_x - 8, center_y + offset - 1.8, 16, 3.6),
                2,
                2,
            )
            band_index += 1
            offset += fringe_spacing

        painter.setPen(QPen(QColor('#34495e'), 1.5))
        self._draw_arrow(
            painter,
            QPointF(barrier_x + 18, upper_slit_y),
            QPointF(barrier_x + 18, lower_slit_y),
            QColor('#34495e'),
            width=1.5,
            head_size=5.0,
            both_ends=True,
        )
        painter.setFont(label_font)
        painter.drawText(QPointF(barrier_x + 24, center_y + 4), "d")

        self._draw_arrow(
            painter,
            QPointF(barrier_x - 14, upper_slit_y - slit_half_height),
            QPointF(barrier_x - 14, upper_slit_y + slit_half_height),
            QColor('#34495e'),
            width=1.4,
            head_size=4.5,
            both_ends=True,
        )
        painter.setFont(label_font)
        painter.drawText(QPointF(barrier_x - 34, upper_slit_y + 4), "a")

        painter.setFont(label_font)
        painter.setPen(QColor('#34495e'))
        painter.drawText(QPointF(rect.left() + 22, rect.top() + 14), "单色光")

        self._draw_parameter_panel(
            painter,
            QRectF(rect.right() - panel_width, rect.top() + 6, panel_width, 74),
            [
                ("λ", f"{self.wavelength:.1f} nm"),
                ("a", f"{self.slit_width:.1f} μm"),
                ("d", f"{self.slit_spacing:.1f} μm"),
            ],
        )

    def _draw_model_title(self, painter, rect, title):
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor('#223247'))
        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

    def _draw_parameter_panel(self, painter, rect, entries):
        painter.setPen(QPen(QColor('#d6e0ed'), 1.0))
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawRoundedRect(rect, 10, 10)

        label_font = QFont()
        label_font.setPointSize(9)
        label_font.setBold(True)
        value_font = QFont()
        value_font.setPointSize(9)

        line_height = 18
        y = rect.top() + 18
        for symbol, value in entries:
            painter.setFont(label_font)
            painter.setPen(QColor('#2e4257'))
            painter.drawText(QPointF(rect.left() + 10, y), f"{symbol} =")
            painter.setFont(value_font)
            painter.setPen(QColor('#55697c'))
            painter.drawText(QPointF(rect.left() + 36, y), value)
            y += line_height

    @staticmethod
    def _draw_arrow(painter, start, end, color, width=2.0, head_size=7.0, both_ends=False):
        def draw_head(tail, tip):
            angle = math.atan2(tip.y() - tail.y(), tip.x() - tail.x())
            left = QPointF(
                tip.x() - head_size * math.cos(angle - math.pi / 6.0),
                tip.y() - head_size * math.sin(angle - math.pi / 6.0),
            )
            right = QPointF(
                tip.x() - head_size * math.cos(angle + math.pi / 6.0),
                tip.y() - head_size * math.sin(angle + math.pi / 6.0),
            )
            painter.drawLine(tip, left)
            painter.drawLine(tip, right)

        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start, end)
        draw_head(start, end)
        if both_ends:
            draw_head(end, start)

    @staticmethod
    def _qt_color_from_wavelength(wavelength_nm, alpha=255):
        red, green, blue = SimulationWidget._wavelength_to_rgb(wavelength_nm)
        color = QColor(int(red * 255), int(green * 255), int(blue * 255))
        color.setAlpha(int(alpha))
        return color

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))


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
                background-color: #ffffff;
                border: 3px solid #0078d4;
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
