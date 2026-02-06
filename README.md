# PhysicsLab Pro - 物理实验辅助工具

一个基于 PyQt6 开发的物理实验辅助桌面应用程序。

## 项目结构

```
physicslab-pro/
├── main.py              # 主窗口程序，包含界面布局和事件处理
├── algorithms.py        # 物理算法模块（待实现）
├── requirements.txt     # Python 依赖包列表
└── README.md           # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行程序

```bash
python main.py
```

## 功能模块

### 1. 光学 AI 实验室
- 静态图像分析（干涉/衍射条纹检测）
- 视频信号分析（迈克尔逊干涉仪）

### 2. 数据工作台
- 智能公式探索（数据拟合）
- 不确定度计算器

## CNN 圆环中心定位

新增脚本 `michelson_center_cnn.py`，用于训练 CNN 并定位迈克尔逊干涉圆环中心。

### 训练模型（合成样本）

```bash
python michelson_center_cnn.py train --epochs 30 --out-model models/michelson_center.pth
```

### 推理并标注中心

```bash
python michelson_center_cnn.py infer --model models/michelson_center.pth --image 牛顿环.png --out-image outputs/center_marked.png
```

## 技术栈

- **GUI 框架:** PyQt6
- **图像处理:** OpenCV, NumPy
- **绘图:** Matplotlib
- **数据分析:** Pandas, SciPy

