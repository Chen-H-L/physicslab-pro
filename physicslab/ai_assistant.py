import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .workers import LLMWorker


class AIAssistantDock(QDockWidget):
    """AI 虚拟助教停靠窗口"""
    
    # API Key 配置常量（方便使用者修改）
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
    
    def __init__(self, parent=None):
        super().__init__("🤖 AI 虚拟助教", parent)
        self.llm_worker = None  # LLM 工作线程
        self.init_ui()
    
    def init_ui(self):
        """初始化 UI 组件"""
        # 创建主容器
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 聊天记录显示区域（QTextBrowser 支持 HTML）
        self.chat_display = QTextBrowser()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextBrowser {
                background-color: #ffffff;
                color: #333333;
                border: none;
                padding: 20px;
                font-size: 14px;
                line-height: 1.6;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }
        """)
        self.chat_display.setHtml("""
            <html>
            <body style="margin: 0; padding: 0;">
                <div style="text-align: center; padding: 40px 20px; color: #666;">
                    <h2 style="margin-bottom: 10px; color: #333;">新对话</h2>
                    <p>欢迎使用 AI 虚拟助教</p>
                    <p style="font-size: 12px; color: #999; margin-top: 20px;">请输入您的问题，开始对话</p>
                </div>
            </body>
            </html>
        """)
        main_layout.addWidget(self.chat_display)
        
        # 输入区域
        input_container = QWidget()
        input_container.setStyleSheet("background-color: #f8f9fa; border-top: 1px solid #e9ecef;")
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(15, 10, 15, 15)
        input_layout.setSpacing(10)
        
        # 用户输入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("给 AI 发送消息...")
        self.input_field.setMinimumHeight(40)
        self.input_field.returnPressed.connect(self.on_send_question)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 14px;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
                selection-background-color: #e3f2fd;
                selection-color: #1976d2;
            }
            QLineEdit:focus {
                border: 1px solid #1976d2;
                background-color: #ffffff;
                outline: none;
            }
        """)
        input_layout.addWidget(self.input_field)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # 清空聊天记录按钮
        self.clear_button = QPushButton("清空")
        self.clear_button.setMinimumHeight(36)
        self.clear_button.setMaximumWidth(80)
        self.clear_button.clicked.connect(self.on_clear_chat)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #666666;
                border: 1px solid #e9ecef;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #dee2e6;
            }
        """)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.setMinimumHeight(36)
        self.send_button.setMaximumWidth(100)
        self.send_button.clicked.connect(self.on_send_question)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: #ffffff;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        button_layout.addWidget(self.send_button)
        
        input_layout.addLayout(button_layout)
        
        # 提示信息标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_layout.addWidget(self.status_label)
        
        input_container.setLayout(input_layout)
        main_layout.addWidget(input_container)
        
        # 设置主小部件
        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)
        
        # 设置停靠窗口的样式
        self.setStyleSheet("""
            QDockWidget {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #e9ecef;
                titlebar-close-icon: url(none);
            }
            QDockWidget::title {
                text-align: center;
                padding: 10px;
                background-color: #ffffff;
                font-weight: 600;
                font-size: 14px;
                color: #333333;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }
        """)
    
    def on_send_question(self):
        """发送问题按钮点击事件"""
        user_input = self.input_field.text().strip()
        
        if not user_input:
            QMessageBox.warning(self, "警告", "请输入问题！")
            return
        
        # 检查 API Key 是否已配置
        if not self.API_KEY:
            QMessageBox.warning(
                self, 
                "⚠️ 未配置 API Key", 
                "请先配置您的 DeepSeek API Key：\n\n"
                "PowerShell:\n"
                "$env:DEEPSEEK_API_KEY='你的 DeepSeek Key'\n\n"
                "然后重新启动程序。\n\n"
                "获取密钥：https://platform.deepseek.com/api_keys"
            )
            return
        
        
        # 禁用输入框和按钮
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.send_button.setText("🕐 AI 正在思考...")
        self.status_label.setText("⏳ 正在处理您的问题...")
        self.status_label.setStyleSheet("color: #ffb347;")  # 橙色
        
        # 在聊天窗口中显示用户问题
        self.append_message("👤 您", user_input, is_user=True)
        
        # 清空输入框
        self.input_field.clear()
        
        # 创建并启动 LLM 工作线程
        # 可选：从应用程序的某个地方获取上下文数据
        context_data = ""  # 暂时不使用上下文
        
        self.llm_worker = LLMWorker(self.API_KEY, user_input, context_data)
        self.llm_worker.response_ready.connect(self.on_response_received)
        self.llm_worker.error_occurred.connect(self.on_error_occurred)
        self.llm_worker.finished.connect(self.on_worker_finished)
        self.llm_worker.start()
    
    def append_message(self, role: str, message: str, is_user: bool = False):
        """向聊天窗口追加消息"""
        # 获取当前的 HTML 内容
        current_html = self.chat_display.toHtml()
        
        # 处理消息内容，改进公式显示
        processed_message = self.process_message(message)
        
        # 构造消息 HTML（类似网页版 DeepSeek 风格）
        if is_user:
            # 用户消息（蓝色，右对齐）
            message_html = f"""
<div style="display: flex; justify-content: flex-end; margin: 10px 0;">
    <div style="max-width: 75%; background-color: #e3f2fd; border-radius: 18px 18px 4px 18px; padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
        <div style="color: #1976d2; font-weight: 500; margin-bottom: 4px;">👤 {role}</div>
        <div style="color: #333333; white-space: pre-wrap; word-wrap: break-word; line-height: 1.6;">{processed_message}</div>
    </div>
</div>
"""
        else:
            # AI 消息（白色，左对齐）
            message_html = f"""
<div style="display: flex; margin: 10px 0;">
    <div style="max-width: 75%; background-color: #ffffff; border: 1px solid #e9ecef; border-radius: 4px 18px 18px 4px; padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        <div style="color: #1976d2; font-weight: 500; margin-bottom: 4px;">🤖 {role}</div>
        <div style="color: #333333; white-space: pre-wrap; word-wrap: break-word; line-height: 1.6;">{processed_message}</div>
    </div>
</div>
"""
        
        # 追加到聊天窗口（去掉</body></html> 标签后添加新内容）
        new_html = current_html.replace("</body></html>", "") + message_html + "</body></html>"
        self.chat_display.setHtml(new_html)
        
        # 自动滚动到底部
        scroll_bar = self.chat_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
    
    def process_message(self, message: str) -> str:
        """处理消息内容，改进公式显示"""
        # 处理消息内容，改进公式显示
        
        # 替换常见的数学符号
        replacements = {
            "\\lambda": "λ",
            "\\theta": "θ",
            "\\sin": "sin",
            "\\cos": "cos",
            "\\tan": "tan",
            "\\pi": "π",
            "\\cdot": "·",
            "\\times": "×",
            "\\div": "÷",
            "\\pm": "±",
            "\\infty": "∞",
            "\\sqrt": "√",
            "\\frac": "/",
            "\\sum": "Σ",
            "\\int": "∫",
            "\\partial": "∂",
            "\\nabla": "∇",
            "\\ldots": "...",
            "\\dots": "...",
            "\\cdots": "...",
            "\\ldots": "...",
            "\\approx": "≈",
            "\\equiv": "≡",
            "\\neq": "≠",
            "\\leq": "≤",
            "\\geq": "≥",
            "\\lt": "<",
            "\\gt": ">",
            "\\quad": " ",
            "\\Delta": "Δ",
            "\\alpha": "α",
            "\\beta": "β",
            "\\gamma": "γ",
            "\\delta": "δ",
            "\\epsilon": "ε",
            "\\phi": "φ",
            "\\psi": "ψ",
            "\\omega": "ω",
        }
        
        for latex, symbol in replacements.items():
            message = message.replace(latex, symbol)
        
        # 处理美元符号包围的数学表达式
        import re
        # 处理行内公式 $...$
        message = re.sub(r'\$(.*?)\$', r'<span style="font-family: Cambria Math, Times New Roman, serif; font-style: italic;">\1</span>', message)
        
        # 处理Markdown粗体标记 **...**
        message = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', message)
        
        # 处理多余的星号
        message = re.sub(r'\*+', r'', message)
        
        # 处理多行文本，添加适当的换行和缩进
        lines = message.split('\n')
        processed_lines = []
        
        # 检测是否在公式块中
        in_equation = False
        equation_lines = []
        
        for line in lines:
            line = line.strip()
            
            # 处理公式块
            if line.startswith('$$') and line.endswith('$$'):
                # 单行公式
                eq_content = line.strip('$$').strip()
                processed_lines.append(f"<div style='margin: 12px 0; padding: 10px; background-color: #f8f9fa; border-radius: 6px; text-align: center; font-family: Cambria Math, Times New Roman, serif;'>{eq_content}</div>")
            elif line.startswith('$$'):
                # 公式块开始
                in_equation = True
                equation_lines = []
            elif line.endswith('$$'):
                # 公式块结束
                in_equation = False
                eq_content = ' '.join(equation_lines).strip()
                processed_lines.append(f"<div style='margin: 12px 0; padding: 12px; background-color: #f8f9fa; border-radius: 6px; text-align: center; font-family: Cambria Math, Times New Roman, serif; font-size: 15px;'>{eq_content}</div>")
            elif in_equation:
                # 公式块内容
                equation_lines.append(line)
            elif line:
                # 处理标题
                if line.startswith("#"):
                    level = line.count("#")
                    text = line.lstrip("#").strip()
                    if level == 1:
                        processed_lines.append(f"<h3 style='margin: 20px 0 12px 0; color: #333; font-weight: 600;'>{text}</h3>")
                    elif level == 2:
                        processed_lines.append(f"<h4 style='margin: 16px 0 10px 0; color: #333; font-weight: 600;'>{text}</h4>")
                    elif level == 3:
                        processed_lines.append(f"<h5 style='margin: 14px 0 8px 0; color: #333; font-weight: 600;'>{text}</h5>")
                # 处理列表
                elif line.startswith("-") or line.startswith("*"):
                    # 检查是否包含公式
                    if any(sym in line for sym in ["λ", "θ", "π", "Δ", "α", "β", "γ", "δ", "ε", "φ", "ψ", "ω", "sin", "cos", "tan"]):
                        processed_lines.append(f"<div style='margin-left: 20px; margin-bottom: 8px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>• {line.lstrip('-* ').strip()}</div>")
                    else:
                        processed_lines.append(f"<div style='margin-left: 20px; margin-bottom: 5px;'>• {line.lstrip('-* ').strip()}</div>")
                # 处理引用
                elif line.startswith("> "):
                    processed_lines.append(f"<div style='margin: 10px 0; padding: 12px; background-color: #f0f7ff; border-left: 4px solid #1976d2; border-radius: 0 6px 6px 0;'>{line.lstrip('> ').strip()}</div>")
                # 处理包含公式的普通文本
                elif any(sym in line for sym in ["λ", "θ", "π", "Δ", "α", "β", "γ", "δ", "ε", "φ", "ψ", "ω", "sin", "cos", "tan", "=", "<", ">", "±"]):
                    processed_lines.append(f"<div style='margin: 8px 0; padding: 8px; background-color: #fafafa; border-radius: 4px; font-family: Cambria Math, Times New Roman, serif;'>{line}</div>")
                # 处理普通文本
                else:
                    processed_lines.append(f"<p style='margin: 8px 0; line-height: 1.6;'>{line}</p>")
        
        return "".join(processed_lines)
    
    def on_response_received(self, response: str):
        """接收 AI 响应"""
        self.append_message("AI 助教", response, is_user=False)
        self.status_label.setText("✅ 回答完成，您可以继续提问")
        self.status_label.setStyleSheet("color: #90ee90;")  # 绿色
    
    def on_error_occurred(self, error_msg: str):
        """处理错误"""
        self.append_message("⚠️ 系统", error_msg, is_user=False)
        self.status_label.setText(f"❌ 错误: {error_msg[:30]}...")
        self.status_label.setStyleSheet("color: #ff6b6b;")  # 红色
    
    def on_worker_finished(self):
        """工作线程完成"""
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.send_button.setText("📤 发送")
    
    def on_clear_chat(self):
        """清空聊天记录"""
        reply = QMessageBox.question(
            self, 
            "确认清空",
            "确定要清空所有聊天记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_display.setMarkdown("### 聊天记录已清空\n\n准备开始新的对话...")
            self.status_label.setText("🔄 已清空，准备就绪")
            self.status_label.setStyleSheet("color: #90ee90;")
