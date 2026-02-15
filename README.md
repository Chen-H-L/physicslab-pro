# PhysicsLab Pro - 物理实验辅助工具

一个基于 PyQt6 开发的物理实验辅助桌面应用程序，集成了 OpenGL 仿真、数据分析和 AI 虚拟助教功能。

## 项目结构

```
physicslab-pro/
├── main.py              # 主窗口程序，包含界面布局和事件处理
├── algorithms.py        # 物理算法模块
├── requirements.txt     # Python 依赖包列表
├── models/              # 预训练模型文件夹
│   └── michelson_center.pth  # CNN 模型
└── README.md            # 项目说明文档
```

## 快速开始

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行程序

```bash
python main.py
```

## 功能模块

### 1. 🔬 光学 AI 实验室
- **静态图像分析**：干涉/衍射条纹检测与分析
- **视频信号分析**：实时处理迈克尔逊干涉仪视频，提取圆环中心和信号
- **波峰检测**：自动识别光强分布中的条纹级数

### 2. 📊 数据工作台
- **智能公式探索**：通过数据拟合自动探索物理规律（线性、指数、余弦）
- **不确定度计算器**：计算测量数据的标准差、标准误差和 A 类不确定度
- **异常值检测**：使用 Z-score 方法识别和标记离群点

### 3. 🌐 虚拟仿真实验室
- **牛顿环实验**：实时仿真干涉条纹，可调节曲率半径和光波长
- **劈尖干涉**：观察条纹如何随夹角变化
- **双缝干涉**：展示衍射包络与干涉纹样的结合
- 基于 OpenGL 和 GLSL Shader 的实时渲染

### 4. 🤖 AI 虚拟助教（NEW）
- **智能问答**：基于 DeepSeek API 的实时对话
- **实验指导**：解答物理实验原理和实验方法
- **数据分析**：帮助理解数据和不确定度
- **LaTeX 支持**：支持数学公式输入输出
- **后台异步处理**：不阻塞主 UI 线程

## 技术栈

| 组件 | 技术 |
|------|------|
| **GUI 框架** | PyQt6 |
| **图像处理** | OpenCV、NumPy |
| **数据分析** | Pandas、SciPy、Matplotlib |
| **3D/图形** | PyOpenGL、GLSL Shader |
| **深度学习** | PyTorch |
| **AI** | DeepSeek API、OpenAI Python SDK |

## AI 虚拟助教配置

### 获取 DeepSeek API Key

1. 访问 [DeepSeek 平台](https://platform.deepseek.com/)
2. 注册并登录账号
3. 进入 [API Keys](https://platform.deepseek.com/api_keys) 页面
4. 创建新的 API Key（格式为 `sk-...`）

### 配置 API Key 到代码

打开 `main.py` 文件，搜索 `API_KEY = "sk-` 找到以下代码（约 760 行）：

```python
class AIAssistantDock(QDockWidget):
    """AI 虚拟助教停靠窗口"""
    
    # API Key 配置常量（方便使用者修改）
    API_KEY = "sk-在此处填入你的DeepSeek Key"
```

将 `"sk-在此处填入你的DeepSeek Key"` 替换为您的实际 API Key：

```python
API_KEY = "sk-abcdef1234567890abcdef1234567890"
```

保存文件后，重新启动应用即可使用 AI 虚拟助教。

### 使用 AI 虚拟助教

1. 启动应用后，右侧会显示 **"🤖 AI 虚拟助教"** 停靠窗口
2. 在输入框中输入您的问题
3. 按 **Enter 键**或点击 **"📤 发送"** 按钮
4. 等待 AI 回答（通常 3-10 秒）
5. 点击 **"🗑️ 清空"** 可清除对话记录

### 示例问题

```
• "解释一下牛顿环实验的原理"
• "什么是光的干涉现象？"
• "如何计算测量数据的不确定度？"
• "给我讲解波长公式：λ = 2nd cos θ / m"
• "我的实验数据是 10.1, 10.2, 10.5，怎么处理异常值？"
```

### 故障排除

| 问题 | 解决方案 |
|------|--------|
| "openai 库未安装" | 运行 `pip install openai` |
| "API Key 无效" | 检查密钥是否复制完整，确认在 DeepSeek 平台有效 |
| "网络连接失败" | 检查网络连接，确保能访问 api.deepseek.com |
| "没有响应" | 等待处理、检查状态提示、查看控制台日志 |

## CNN 圆环中心定位

项目包含预训练的 CNN 模型用于定位迈克尔逊干涉仪的中心位置。

### 训练模型（使用合成样本）

```bash
python michelson_center_cnn.py train --epochs 30 --out-model models/michelson_center.pth
```

### 使用模型进行推理

```bash
python michelson_center_cnn.py infer --model models/michelson_center.pth --image input.png --out-image output.png
```

## 依赖包列表

| 包名 | 版本 | 用途 |
|------|------|------|
| PyQt6 | ≥6.6.0 | GUI 框架 |
| opencv-python | ≥4.8.0 | 图像处理 |
| numpy | ≥1.24.0 | 数值计算 |
| matplotlib | ≥3.7.0 | 数据绘图 |
| pandas | ≥2.0.0 | 数据分析 |
| scipy | ≥1.11.0 | 科学计算 |
| PyOpenGL | ≥3.1.6 | 3D 图形 |
| torch | ≥2.1.0 | 深度学习 |
| openai | ≥2.0.0 | AI API 调用 |

