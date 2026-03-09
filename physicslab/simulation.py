import numpy as np
from OpenGL import GL as gl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QVector2D
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
