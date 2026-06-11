import re
from html import escape
from urllib.parse import urlparse

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import load_ai_settings, save_ai_settings
from .workers import LLMWorker


class AISettingsDialog(QDialog):
    """AI 接口配置对话框。"""

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 接口设置")
        self.setModal(True)
        self.resize(460, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        hint_label = QLabel(
            "在程序内保存 AI 助手的接口配置。默认兼容 DeepSeek，也可以填写兼容 OpenAI Chat Completions 的其他接口。"
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #64748b; line-height: 1.5;")
        layout.addWidget(hint_label)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("请输入 API Key")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(settings.get("api_key", ""))
        form_layout.addRow("API Key", self.api_key_input)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("例如：https://api.deepseek.com")
        self.base_url_input.setText(settings.get("base_url", "https://api.deepseek.com"))
        form_layout.addRow("Base URL", self.base_url_input)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("例如：deepseek-chat")
        self.model_input.setText(settings.get("model", "deepseek-chat"))
        form_layout.addRow("模型名称", self.model_input)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = button_box.button(QDialogButtonBox.StandardButton.Save)
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if save_button is not None:
            save_button.setText("保存")
        if cancel_button is not None:
            cancel_button.setText("取消")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
            }
            QLabel {
                color: #334155;
                font-size: 12px;
            }
            QLineEdit {
                min-height: 36px;
                border: 1px solid #d9d9d9;
                border-radius: 8px;
                padding: 0 12px;
                background-color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #1976d2;
            }
            QPushButton {
                min-width: 88px;
                min-height: 34px;
                border-radius: 6px;
            }
            """
        )

    def get_settings(self) -> dict:
        return {
            "api_key": self.api_key_input.text().strip(),
            "base_url": self.base_url_input.text().strip(),
            "model": self.model_input.text().strip(),
        }

    def accept(self):
        settings = self.get_settings()
        if not settings["api_key"]:
            QMessageBox.warning(self, "配置不完整", "请至少填写 API Key。")
            return
        if not settings["base_url"]:
            self.base_url_input.setText("https://api.deepseek.com")
        if not settings["model"]:
            self.model_input.setText("deepseek-chat")
        super().accept()


class AIAssistantDock(QDockWidget):
    """AI 虚拟助教停靠窗口。"""

    def __init__(self, parent=None):
        super().__init__("AI 虚拟助教", parent)
        self.llm_worker = None
        self.ai_settings = load_ai_settings()
        self.init_ui()
        self.update_config_status()

    def init_ui(self):
        """初始化 UI 组件。"""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.chat_display = QTextBrowser()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            """
            QTextBrowser {
                background-color: #f8fafc;
                color: #1f2937;
                border: none;
                padding: 18px;
                font-size: 14px;
                line-height: 1.6;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }
            """
        )
        self.chat_display.setHtml(
            """
            <html>
            <body style="margin: 0; padding: 0; background-color: #f8fafc;">
                <div style="text-align: center; padding: 40px 20px; color: #666;">
                    <h2 style="margin-bottom: 10px; color: #333;">新对话</h2>
                    <p>欢迎使用 AI 虚拟助教</p>
                    <p style="font-size: 12px; color: #999; margin-top: 20px;">请输入您的问题，开始对话</p>
                </div>
            </body>
            </html>
            """
        )
        main_layout.addWidget(self.chat_display)

        input_container = QWidget()
        input_container.setStyleSheet("background-color: #f8f9fa; border-top: 1px solid #e9ecef;")
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(15, 10, 15, 15)
        input_layout.setSpacing(10)

        self.config_label = QLabel()
        self.config_label.setWordWrap(True)
        self.config_label.setStyleSheet("color: #64748b; font-size: 11px;")
        input_layout.addWidget(self.config_label)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("给 AI 发送消息...")
        self.input_field.setMinimumHeight(40)
        self.input_field.returnPressed.connect(self.on_send_question)
        self.input_field.setStyleSheet(
            """
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
            }
            """
        )
        input_layout.addWidget(self.input_field)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.settings_button = QPushButton("AI 设置")
        self.settings_button.setMinimumHeight(36)
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.settings_button.setStyleSheet(
            """
            QPushButton {
                background-color: #ffffff;
                color: #1976d2;
                border: 1px solid #cfe3fb;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #f4f9ff;
                border: 1px solid #9cc7f7;
            }
            """
        )
        button_layout.addWidget(self.settings_button)

        self.clear_button = QPushButton("清空")
        self.clear_button.setMinimumHeight(36)
        self.clear_button.setMaximumWidth(80)
        self.clear_button.clicked.connect(self.on_clear_chat)
        self.clear_button.setStyleSheet(
            """
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
            """
        )
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        self.send_button = QPushButton("发送")
        self.send_button.setMinimumHeight(36)
        self.send_button.setMaximumWidth(100)
        self.send_button.clicked.connect(self.on_send_question)
        self.send_button.setStyleSheet(
            """
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
            """
        )
        button_layout.addWidget(self.send_button)

        input_layout.addLayout(button_layout)

        self.status_label = QLabel("准备就绪")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_layout.addWidget(self.status_label)

        input_container.setLayout(input_layout)
        main_layout.addWidget(input_container)

        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)

        self.setStyleSheet(
            """
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
            """
        )

    def masked_api_key(self) -> str:
        api_key = self.ai_settings.get("api_key", "")
        if len(api_key) <= 8:
            return "已配置"
        return f"{api_key[:4]}...{api_key[-4:]}"

    def update_config_status(self):
        """刷新配置状态提示。"""
        if self.ai_settings.get("api_key"):
            host = urlparse(self.ai_settings.get("base_url", "")).netloc or self.ai_settings.get("base_url", "")
            model = self.ai_settings.get("model", "deepseek-chat")
            self.config_label.setText(
                f"当前 AI 配置：{model} @ {host}，Key：{self.masked_api_key()}。"
            )
            self.config_label.setStyleSheet("color: #64748b; font-size: 11px;")
        else:
            self.config_label.setText("尚未配置 AI 接口。点击“AI 设置”填写 API Key、Base URL 和模型名称。")
            self.config_label.setStyleSheet("color: #d97706; font-size: 11px;")

    def open_settings_dialog(self):
        """打开 AI 接口配置对话框。"""
        dialog = AISettingsDialog(self.ai_settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.ai_settings = dialog.get_settings()
        save_ai_settings(self.ai_settings)
        self.update_config_status()
        self.status_label.setText("AI 配置已保存")
        self.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
        QMessageBox.information(
            self,
            "保存成功",
            "AI 接口配置已保存到本地。下次启动程序时会自动加载该配置。",
        )

    def on_send_question(self):
        """发送问题按钮点击事件。"""
        user_input = self.input_field.text().strip()
        if not user_input:
            QMessageBox.warning(self, "警告", "请输入问题。")
            return

        self.ai_settings = load_ai_settings()
        self.update_config_status()
        if not self.ai_settings.get("api_key"):
            QMessageBox.warning(
                self,
                "未配置 AI 接口",
                "请先点击“AI 设置”填写 API Key，再开始对话。",
            )
            self.open_settings_dialog()
            return

        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.send_button.setText("AI 思考中...")
        self.status_label.setText("正在处理您的问题...")
        self.status_label.setStyleSheet("color: #ffb347; font-size: 11px;")

        self.append_message("您", user_input, is_user=True)
        self.input_field.clear()

        context_data = ""
        self.llm_worker = LLMWorker(
            api_key=self.ai_settings["api_key"],
            user_question=user_input,
            context_data=context_data,
            base_url=self.ai_settings["base_url"],
            model=self.ai_settings["model"],
        )
        self.llm_worker.response_ready.connect(self.on_response_received)
        self.llm_worker.error_occurred.connect(self.on_error_occurred)
        self.llm_worker.finished.connect(self.on_worker_finished)
        self.llm_worker.start()

    def append_message(self, role: str, message: str, is_user: bool = False):
        """向聊天窗口追加消息。"""
        current_html = self.chat_display.toHtml()
        processed_message = self.process_message(message)

        if is_user:
            message_html = f"""
<table width="100%" cellspacing="0" cellpadding="0" style="margin: 12px 0;">
    <tr>
        <td align="right">
            <table width="78%" cellspacing="0" cellpadding="0" style="background-color: #eaf4ff; border: 1px solid #cfe3fb;">
                <tr><td style="padding: 10px 14px;">
                    <p style="margin: 0 0 4px 0; color: #1d4ed8; font-weight: 600; font-size: 12px;">{escape(role)}</p>
                    <div style="color: #1f2937; line-height: 1.65;">{processed_message}</div>
                </td></tr>
            </table>
        </td>
    </tr>
</table>
"""
        else:
            message_html = f"""
<table width="100%" cellspacing="0" cellpadding="0" style="margin: 14px 0; background-color: #ffffff; border: 1px solid #e2e8f0;">
    <tr><td style="padding: 14px 16px;">
        <p style="margin: 0 0 10px 0; padding-bottom: 8px; color: #0f172a; font-weight: 600; font-size: 13px; border-bottom: 1px solid #eef2f7;">{escape(role)}</p>
        <div style="color: #1f2937; line-height: 1.7;">{processed_message}</div>
    </td></tr>
</table>
"""

        new_html = current_html.replace("</body></html>", "") + message_html + "</body></html>"
        self.chat_display.setHtml(new_html)

        scroll_bar = self.chat_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _extract_brace_group(self, text: str, start: int):
        if start >= len(text) or text[start] != "{":
            return None

        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1:index], index + 1
        return None

    def _replace_group_command(self, text: str, command: str, group_count: int, formatter) -> str:
        cursor = 0
        output = []
        while True:
            index = text.find(command, cursor)
            if index < 0:
                output.append(text[cursor:])
                break

            output.append(text[cursor:index])
            pos = index + len(command)
            groups = []
            ok = True
            for _ in range(group_count):
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                group = self._extract_brace_group(text, pos)
                if group is None:
                    ok = False
                    break
                groups.append(group[0])
                pos = group[1]

            if ok:
                output.append(formatter(*groups))
                cursor = pos
            else:
                output.append(command)
                cursor = index + len(command)

        return "".join(output)

    def _apply_scripts(self, text: str) -> str:
        superscript_map = str.maketrans({
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
            "n": "ⁿ", "i": "ⁱ",
        })
        subscript_map = str.maketrans({
            "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
            "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
            "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
            "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
            "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
            "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
            "v": "ᵥ", "x": "ₓ",
        })

        def convert(content: str, table, marker: str) -> str:
            cleaned = content.replace(" ", "")
            converted = cleaned.translate(table)
            if all(char != original for char, original in zip(converted, cleaned)) or converted != cleaned:
                return converted
            return f"{marker}({content})"

        text = re.sub(r"\^\{([^{}]+)\}", lambda m: convert(m.group(1), superscript_map, "^"), text)
        text = re.sub(r"_\{([^{}]+)\}", lambda m: convert(m.group(1), subscript_map, "_"), text)
        text = re.sub(r"\^([A-Za-z0-9+\-=()])", lambda m: convert(m.group(1), superscript_map, "^"), text)
        text = re.sub(r"_([A-Za-z0-9+\-=()])", lambda m: convert(m.group(1), subscript_map, "_"), text)
        return text

    def _latex_to_readable(self, formula: str) -> str:
        formula = formula.strip()
        formula = formula.replace("\\left", "").replace("\\right", "")
        formula = formula.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
        formula = formula.replace("\\quad", " ").replace("\\qquad", " ")

        formula = self._replace_group_command(
            formula,
            "\\frac",
            2,
            lambda numerator, denominator: (
                f"({self._latex_to_readable(numerator)})/({self._latex_to_readable(denominator)})"
            ),
        )
        formula = self._replace_group_command(
            formula,
            "\\sqrt",
            1,
            lambda value: f"√({self._latex_to_readable(value)})",
        )
        formula = self._replace_group_command(
            formula,
            "\\mathrm",
            1,
            lambda value: self._latex_to_readable(value),
        )
        formula = self._replace_group_command(
            formula,
            "\\text",
            1,
            lambda value: value,
        )
        formula = self._replace_group_command(
            formula,
            "\\hat",
            1,
            lambda value: f"{self._latex_to_readable(value)}̂",
        )
        formula = self._replace_group_command(
            formula,
            "\\bar",
            1,
            lambda value: f"{self._latex_to_readable(value)}̄",
        )
        formula = re.sub(
            r"√\s*\{([^{}]+)\}",
            lambda match: f"√({self._latex_to_readable(match.group(1))})",
            formula,
        )

        for function_name in ("sin", "cos", "tan", "exp", "ln", "log"):
            formula = re.sub(rf"\\{function_name}\b", f" {function_name}", formula)

        replacements = {
            "\\lambda": "λ",
            "\\theta": "θ",
            "\\pi": "π",
            "\\Delta": "Δ",
            "\\alpha": "α",
            "\\beta": "β",
            "\\gamma": "γ",
            "\\delta": "δ",
            "\\epsilon": "ε",
            "\\varepsilon": "ε",
            "\\phi": "φ",
            "\\varphi": "φ",
            "\\psi": "ψ",
            "\\omega": "ω",
            "\\Omega": "Ω",
            "\\mu": "μ",
            "\\rho": "ρ",
            "\\sigma": "σ",
            "\\sin": "sin",
            "\\cos": "cos",
            "\\tan": "tan",
            "\\exp": "exp",
            "\\ln": "ln",
            "\\log": "log",
            "\\cdot": "·",
            "\\times": "×",
            "\\div": "÷",
            "\\pm": "±",
            "\\infty": "∞",
            "\\sum": "∑",
            "\\int": "∫",
            "\\partial": "∂",
            "\\nabla": "∇",
            "\\propto": "∝",
            "\\approx": "≈",
            "\\equiv": "≡",
            "\\neq": "≠",
            "\\leq": "≤",
            "\\le": "≤",
            "\\geq": "≥",
            "\\ge": "≥",
            "\\lt": "<",
            "\\gt": ">",
            "\\rightarrow": "→",
            "\\to": "→",
            "\\Rightarrow": "⇒",
            "\\ldots": "...",
            "\\dots": "...",
            "\\cdots": "...",
        }
        for latex, symbol in replacements.items():
            formula = formula.replace(latex, symbol)

        formula = self._apply_scripts(formula)
        formula = formula.replace("{", "").replace("}", "")
        formula = re.sub(r"\s+", " ", formula).strip()
        return formula

    def _formula_html(self, formula: str, block: bool = False) -> str:
        readable = escape(self._latex_to_readable(formula))
        if block:
            return (
                "<div style='margin: 12px 0; padding: 12px 14px; "
                "background-color: #f1f7ff; border: 1px solid #dbeafe; "
                "border-radius: 8px; text-align: center; "
                "font-family: Cambria Math, STIX Two Math, Times New Roman, serif; "
                "font-size: 16px; color: #0f172a; line-height: 1.8;'>"
                f"{readable}</div>"
            )
        return (
            "<span style='font-family: Cambria Math, STIX Two Math, Times New Roman, serif; "
            "font-size: 15px; color: #0f172a; background-color: #f8fafc; padding: 0 3px; border-radius: 3px;'>"
            f"{readable}</span>"
        )

    def _clean_markdown_noise(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^\s*[-*•]\s*", "", text)
        text = re.sub(r"^\s*\*+", "", text)
        text = re.sub(r"\*+\s*$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _format_inline_text(self, text: str) -> str:
        formulas = []
        text = self._clean_markdown_noise(text)

        def stash_formula(match):
            formulas.append(self._formula_html(match.group(1), block=False))
            return f"@@FORMULA{len(formulas) - 1}@@"

        text = re.sub(r"\\\((.+?)\\\)", stash_formula, text)
        text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", stash_formula, text)
        if "\\" in text or re.search(r"[A-Za-zα-ωΑ-Ω][_^][{A-Za-z0-9]", text):
            text = self._latex_to_readable(text)
        text = escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"`(.+?)`", r"<code style='background:#f3f4f6; padding:1px 4px; border-radius:4px;'>\1</code>", text)
        text = text.replace("**", "").replace("*", "")

        for index, formula_html in enumerate(formulas):
            text = text.replace(f"@@FORMULA{index}@@", formula_html)
        return text

    def _looks_like_formula(self, line: str) -> bool:
        if "$" in line or "\\(" in line or "\\)" in line:
            return False
        if re.search(r"[\u4e00-\u9fff]", line):
            return False
        if any(token in line for token in ("\\frac", "\\sqrt", "\\lambda", "\\Delta", "\\propto", "\\approx")):
            return True
        if re.search(r"[A-Za-zα-ωΑ-Ω][_^][{A-Za-z0-9]", line):
            return True
        return "=" in line and any(token in line for token in ("√", "^", "_", "/", "λ", "Δ", "π", "∝", "≈"))

    def process_message(self, message: str) -> str:
        """处理消息内容，改善公式显示。"""
        message = message.replace("\r\n", "\n").replace("\r", "\n")
        lines = message.split("\n")
        processed_lines = []
        in_equation = False
        equation_lines = []

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("$$") and line.endswith("$$") and len(line) > 4:
                processed_lines.append(self._formula_html(line[2:-2].strip(), block=True))
                continue
            if line.startswith("\\[") and line.endswith("\\]"):
                processed_lines.append(self._formula_html(line[2:-2].strip(), block=True))
                continue
            if line.startswith("$$") or line.startswith("\\["):
                in_equation = True
                equation_lines = [line.lstrip("$").lstrip("\\[").strip()]
                continue
            if line.endswith("$$") or line.endswith("\\]"):
                equation_lines.append(line.rstrip("$").rstrip("\\]").strip())
                processed_lines.append(self._formula_html(" ".join(equation_lines), block=True))
                in_equation = False
                equation_lines = []
                continue
            if in_equation:
                equation_lines.append(line)
                continue

            list_candidate = re.sub(r"^\s*[-*•]\s*", "", line).strip()
            list_candidate = re.sub(r"^\*+|\*+$", "", list_candidate).strip()
            numbered_match = re.match(r"^(\d+)[\.、]\s*(.+)$", list_candidate)

            if line.startswith("#"):
                level = min(line.count("#"), 3)
                text = self._format_inline_text(line.lstrip("#").strip())
                tag = "h3" if level == 1 else "h4" if level == 2 else "h5"
                margin = "18px 0 10px 0" if level == 1 else "14px 0 8px 0"
                processed_lines.append(
                    f"<{tag} style='margin: {margin}; color: #1f2937; font-weight: 600;'>{text}</{tag}>"
                )
            elif numbered_match:
                number, item_text = numbered_match.groups()
                processed_lines.append(
                    "<p style='margin: 8px 0; line-height: 1.65;'>"
                    f"<span style='color: #1d4ed8; font-weight: 600;'>{escape(number)}.</span> "
                    f"{self._format_inline_text(item_text)}</p>"
                )
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                clean_line = line.lstrip("-*• ").strip()
                processed_lines.append(
                    "<p style='margin: 6px 0 6px 14px; line-height: 1.65;'>"
                    f"{self._format_inline_text(clean_line)}</p>"
                )
            elif line.startswith("> "):
                processed_lines.append(
                    "<div style='margin: 10px 0; padding: 10px 12px; background-color: #f0f7ff; "
                    "border-left: 4px solid #1976d2; border-radius: 0 6px 6px 0;'>"
                    f"{self._format_inline_text(line.lstrip('> ').strip())}</div>"
                )
            elif self._looks_like_formula(line):
                processed_lines.append(self._formula_html(line, block=True))
            else:
                processed_lines.append(
                    f"<p style='margin: 8px 0; line-height: 1.65;'>{self._format_inline_text(line)}</p>"
                )

        if in_equation and equation_lines:
            processed_lines.append(self._formula_html(" ".join(equation_lines), block=True))

        return "".join(processed_lines)

    def on_response_received(self, response: str):
        """接收 AI 响应。"""
        self.append_message("AI 助教", response, is_user=False)
        self.status_label.setText("回答完成，您可以继续提问")
        self.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")

    def on_error_occurred(self, error_msg: str):
        """处理错误。"""
        self.append_message("系统", error_msg, is_user=False)
        self.status_label.setText(f"发生错误：{error_msg[:36]}")
        self.status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")

    def on_worker_finished(self):
        """工作线程完成。"""
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.send_button.setText("发送")

    def on_clear_chat(self):
        """清空聊天记录。"""
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有聊天记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.chat_display.setMarkdown("### 聊天记录已清空\n\n准备开始新的对话...")
            self.status_label.setText("已清空，准备就绪")
            self.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
