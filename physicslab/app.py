import sys
from pathlib import Path

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QStackedWidget, QTabWidget, QWidget

from .ai_assistant import AIAssistantDock
from .data_workstation import DataWorkstationTab
from .extensions_tab import ExtensionsTab
from .optics_tab import OpticsPatternAnalysisTab, OpticsSimulationTab
from .thermodynamics_tab import ThermodynamicsLabTab
from .vibration_tab import VibrationLabTab

APP_NAME_SHORT = "物演智启"
APP_NAME_FULL = "物演智启：科学实验仿真辅助工具"


def asset_path(filename):
    """Return a source or PyInstaller-bundled asset path."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "physicslab" / "assets" / filename
    return Path(__file__).resolve().parent / "assets" / filename


class StartScreen(QWidget):
    """Game-like launch screen shown before entering the main lab workspace."""

    start_requested = pyqtSignal()
    usage_requested = pyqtSignal()
    intro_requested = pyqtSignal()

    _BUTTON_RECTS = {
        "start": QRect(548, 318, 581, 136),
        "usage": QRect(548, 474, 581, 132),
        "intro": QRect(548, 627, 581, 130),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(960, 600)
        self.background = QPixmap(str(asset_path("start_screen.png")))
        self._draw_rect = QRect()

        self.start_button = self._create_hotspot("开始实验", self.start_requested.emit)
        self.usage_button = self._create_hotspot("使用说明", self.usage_requested.emit)
        self.intro_button = self._create_hotspot("程序介绍", self.intro_requested.emit)

        if self.background.isNull():
            self.start_button.setText("开始实验")
            self.usage_button.setText("使用说明")
            self.intro_button.setText("程序介绍")

    def _create_hotspot(self, tooltip, slot):
        button = QPushButton("", self)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip)
        button.clicked.connect(slot)
        button.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 2px solid transparent;
                border-radius: 12px;
                color: #ffffff;
                font-size: 28px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 22);
                border: 2px solid rgba(147, 197, 253, 180);
            }
            QPushButton:pressed {
                background: rgba(37, 99, 235, 60);
                border: 2px solid rgba(191, 219, 254, 220);
            }
            """
        )
        return button

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self.background.isNull():
            painter.fillRect(self.rect(), QColor("#0f172a"))
            return

        painter.fillRect(self.rect(), QColor("#0b1624"))
        scaled = self.background.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        left = (self.width() - scaled.width()) // 2
        top = (self.height() - scaled.height()) // 2
        self._draw_rect = QRect(left, top, scaled.width(), scaled.height())
        painter.drawPixmap(self._draw_rect, scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.background.isNull():
            button_width = min(520, max(280, int(self.width() * 0.42)))
            button_height = 78
            left = (self.width() - button_width) // 2
            top = int(self.height() * 0.38)
            gap = 24
            self.start_button.setGeometry(left, top, button_width, button_height)
            self.usage_button.setGeometry(left, top + button_height + gap, button_width, button_height)
            self.intro_button.setGeometry(left, top + 2 * (button_height + gap), button_width, button_height)
            return

        scaled = self.background.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._draw_rect = QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
        for button, source_rect in (
            (self.start_button, self._BUTTON_RECTS["start"]),
            (self.usage_button, self._BUTTON_RECTS["usage"]),
            (self.intro_button, self._BUTTON_RECTS["intro"]),
        ):
            button.setGeometry(self._map_source_rect(source_rect))
            button.raise_()

    def _map_source_rect(self, source_rect):
        scale_x = self._draw_rect.width() / max(self.background.width(), 1)
        scale_y = self._draw_rect.height() / max(self.background.height(), 1)
        return QRect(
            int(self._draw_rect.left() + source_rect.left() * scale_x),
            int(self._draw_rect.top() + source_rect.top() * scale_y),
            int(source_rect.width() * scale_x),
            int(source_rect.height() * scale_y),
        )


class MainWindow(QMainWindow):
    """主窗口。"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化主窗口界面。"""
        self.setWindowTitle(APP_NAME_FULL)
        self.setGeometry(100, 100, 1400, 900)

        self.stack = QStackedWidget()
        self.start_screen = StartScreen()
        self.start_screen.start_requested.connect(self.enter_lab_workspace)
        self.start_screen.usage_requested.connect(self.show_usage_instructions)
        self.start_screen.intro_requested.connect(self.show_program_intro)

        self.tab_widget = QTabWidget()

        self.optics_simulation_tab = OpticsSimulationTab()
        self.optics_pattern_tab = OpticsPatternAnalysisTab()
        self.data_tab = DataWorkstationTab()
        self.thermodynamics_tab = ThermodynamicsLabTab()
        self.vibration_tab = VibrationLabTab()
        self.extensions_tab = ExtensionsTab()
        self.optics_tab = self.optics_pattern_tab

        self.tab_widget.addTab(self.optics_simulation_tab, "光学虚拟仿真实验")
        self.tab_widget.addTab(self.optics_pattern_tab, "光学图样分析")
        self.tab_widget.addTab(self.thermodynamics_tab, "热力学模拟仿真实验")
        self.tab_widget.addTab(self.vibration_tab, "振动学实验室")
        self.tab_widget.addTab(self.data_tab, "数据工作台")

        self.tab_widget.addTab(self.extensions_tab, "扩展内容")

        self.stack.addWidget(self.start_screen)
        self.stack.addWidget(self.tab_widget)
        self.setCentralWidget(self.stack)

        self.ai_assistant = AIAssistantDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ai_assistant)
        self.ai_assistant.hide()

        menubar = self.menuBar()
        menubar.setStyleSheet(
            """
            QMenuBar {
                background-color: #f5f7fa;
                color: #333333;
                border-bottom: 2px solid #d9d9d9;
            }
            QMenuBar::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QMenu {
                background-color: #ffffff;
                color: #333333;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            """
        )

        help_menu = menubar.addMenu("帮助(&H)")

        toggle_ai_action = help_menu.addAction("显示/隐藏 AI 助手")
        toggle_ai_action.triggered.connect(self.toggle_ai_assistant)

        return_start_action = help_menu.addAction("回到开始界面")
        return_start_action.triggered.connect(self.return_to_start_screen)

        ai_settings_action = help_menu.addAction("AI 接口设置")
        ai_settings_action.triggered.connect(self.ai_assistant.open_settings_dialog)

        help_menu.addSeparator()

        about_action = help_menu.addAction(f"关于 {APP_NAME_SHORT}")
        about_action.triggered.connect(self.show_about)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f5f7fa;
            }

            QMenuBar {
                background-color: #f5f7fa;
                color: #333333;
                border-bottom: 1px solid #d9d9d9;
            }

            QTabWidget::pane {
                border: 2px solid #d9d9d9;
                border-radius: 8px;
                background-color: #ffffff;
                top: -1px;
            }

            QTabBar::tab {
                background-color: #ffffff;
                color: #666666;
                padding: 10px 24px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                border: 1px solid #d9d9d9;
            }

            QTabBar::tab:hover {
                background-color: #f0f5ff;
                color: #333333;
            }

            QTabBar::tab:selected {
                background-color: #0078d4;
                color: #ffffff;
                font-weight: 600;
                border: 1px solid #0078d4;
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

            QTableWidget {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #d9d9d9;
                border-radius: 6px;
                gridline-color: #d9d9d9;
                font-size: 12px;
            }

            QTableWidget::item {
                padding: 4px;
                border: none;
            }

            QTableWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }

            QHeaderView::section {
                background-color: #f5f7fa;
                color: #333333;
                padding: 8px;
                border: 1px solid #d9d9d9;
                font-weight: 600;
                font-size: 12px;
            }

            QLineEdit, QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #d9d9d9;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
            }

            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #0078d4;
                background-color: #ffffff;
            }

            QLabel {
                color: #333333;
                font-size: 12px;
            }

            QProgressBar {
                border: 2px solid #d9d9d9;
                border-radius: 6px;
                text-align: center;
                color: #333333;
                font-weight: 600;
                background-color: #f5f7fa;
            }

            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
            """
        )
        menubar.hide()
        self.stack.setCurrentWidget(self.start_screen)

    def enter_lab_workspace(self):
        """进入主实验工作区。"""
        self.stack.setCurrentWidget(self.tab_widget)
        self.menuBar().show()
        self.ai_assistant.show()

    def return_to_start_screen(self):
        """返回启动页。"""
        self.stack.setCurrentWidget(self.start_screen)
        self.ai_assistant.hide()
        self.menuBar().hide()

    def toggle_ai_assistant(self):
        """显示或隐藏 AI 助手。"""
        if self.ai_assistant.isVisible():
            self.ai_assistant.hide()
        else:
            self.ai_assistant.show()

    def show_usage_instructions(self):
        """显示启动页的使用说明。"""
        usage_text = f"""
<h3>{APP_NAME_SHORT} 使用说明</h3>
<p>点击“开始实验”进入主实验工作区后，可通过顶部标签页切换不同实验模块。</p>
<ul>
    <li><b>光学虚拟仿真实验</b>：选择实验类型后调节左侧参数，右侧会实时显示干涉/衍射图样和实验模型。</li>
    <li><b>光学图样分析</b>：可加载图片或视频，也可以点击“加载样例”查看分析流程。</li>
    <li><b>热力学模拟仿真实验</b>：调节气体状态参数，观察 P-V-T 关系和分子运动。</li>
    <li><b>振动学实验室</b>：支持单一简谐振动、多振动合成和相位参数调节。</li>
    <li><b>数据工作台</b>：导入实验数据，进行拟合、绘图和不确定度计算。</li>
    <li><b>扩展内容</b>：包含傅里叶频谱分析、量子干涉、光电效应等拓展实验。</li>
</ul>
<p>右侧 AI 虚拟助教可用于提问实验原理、参数含义和数据处理思路。</p>
"""
        QMessageBox.information(self, "使用说明", usage_text)

    def show_program_intro(self):
        """显示启动页的程序介绍。"""
        intro_text = f"""
<h3>{APP_NAME_SHORT}</h3>
<p>{APP_NAME_FULL}</p>
<p>本程序是一款面向物理实验教学与自主学习的综合辅助工具，整合虚拟仿真、图样分析、数据处理和 AI 助教能力，帮助用户在同一平台中完成实验观察、参数调节、数据分析与原理理解。</p>
<ul>
    <li><b>模块化实验环境</b>：覆盖光学、热力学、振动学、数据分析与扩展物理内容。</li>
    <li><b>交互式仿真</b>：通过模型示意、动态图样和参数联动呈现实验过程。</li>
    <li><b>数据分析支持</b>：提供样例数据、曲线拟合、统计读数和图表输出等常用工具。</li>
    <li><b>AI 辅助学习</b>：结合虚拟助教提供原理解释、操作提示和实验问题解答。</li>
</ul>
"""
        QMessageBox.information(self, "程序介绍", intro_text)

    def show_about(self):
        """显示关于对话框。"""
        about_text = f"""
<h3>{APP_NAME_SHORT}</h3>
<p>科学实验仿真辅助工具</p>
<hr>
<p><b>功能模块：</b></p>
<ul>
    <li><b>光学虚拟仿真实验</b> - 牛顿环、劈尖干涉、双缝干涉、迈克尔逊干涉仪、光栅衍射等光学仿真实验</li>
    <li><b>光学图样分析</b> - 干涉/衍射图片分析与视频亮度时序分析</li>
    <li><b>数据工作台</b> - 数据拟合与不确定度计算</li>
    <li><b>热力学模拟仿真实验</b> - 理想气体分子动理论模拟、等温/等体/等压过程的 P-V-T 联动分析</li>
    <li><b>振动学实验室</b> - 简谐振动仿真及多种简谐运动合成</li>
    <li><b>扩展内容</b> - 傅里叶频谱分析、量子干涉、光电效应等高阶拓展实验</li>
    <li><b>AI 虚拟助教</b> - 基于 DeepSeek API 的智能问答</li>
</ul>
<hr>
<p><b>技术栈：</b></p>
<ul>
    <li>GUI: PyQt6</li>
    <li>图像处理: OpenCV, NumPy</li>
    <li>数据分析: Pandas, SciPy, Matplotlib</li>
    <li>深度学习: PyTorch</li>
    <li>AI: DeepSeek API</li>
</ul>
<hr>
<p>© 2024 {APP_NAME_SHORT}. All rights reserved.</p>
"""

        QMessageBox.about(self, f"关于 {APP_NAME_SHORT}", about_text)


def main():
    """启动应用程序。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
