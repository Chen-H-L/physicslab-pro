import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from .ai_assistant import AIAssistantDock
from .data_workstation import DataWorkstationTab
from .extensions_tab import ExtensionsTab
from .optics_tab import OpticsPatternAnalysisTab, OpticsSimulationTab
from .thermodynamics_tab import ThermodynamicsLabTab
from .vibration_tab import VibrationLabTab

APP_NAME_SHORT = "物演智启"
APP_NAME_FULL = "物演智启：科学实验仿真辅助工具"


class MainWindow(QMainWindow):
    """主窗口。"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化主窗口界面。"""
        self.setWindowTitle(APP_NAME_FULL)
        self.setGeometry(100, 100, 1400, 900)

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

        self.setCentralWidget(self.tab_widget)

        self.ai_assistant = AIAssistantDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ai_assistant)
        self.ai_assistant.show()

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

    def toggle_ai_assistant(self):
        """显示或隐藏 AI 助手。"""
        if self.ai_assistant.isVisible():
            self.ai_assistant.hide()
        else:
            self.ai_assistant.show()

    def show_about(self):
        """显示关于对话框。"""
        about_text = f"""
<h3>{APP_NAME_SHORT}</h3>
<p>科学实验仿真辅助工具</p>
<hr>
<p><b>功能模块：</b></p>
<ul>
    <li><b>光学虚拟仿真实验</b> - 牛顿环、劈尖干涉、双缝干涉等光学仿真实验</li>
    <li><b>光学图样分析</b> - 干涉/衍射图片分析与视频亮度时序分析</li>
    <li><b>数据工作台</b> - 数据拟合与不确定度计算</li>
    <li><b>热力学模拟仿真实验</b> - 理想气体分子动理论模拟、等温/等体/等压过程的 P-V-T 联动分析</li>
    <li><b>振动学实验室</b> - 简谐振动仿真及多种简谐运动合成</li>
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
