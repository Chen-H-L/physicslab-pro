import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms import auto_fit, calculate_statistics, calculate_uncertainty

from .widgets import MatplotlibCanvas


class DataWorkstationTab(QWidget):
    """页面 2: 数据工作台 (Data Workstation)"""

    def __init__(self):
        super().__init__()
        self.data_mode = 'single'  # 'single' 或 'double'
        self.chart_type = 'scatter'  # 默认图表类型
        self._update_timer = None  # 初始化定时器实例变量
        self.init_ui()

    def init_ui(self):
        """初始化界面布局：左右分屏"""
        main_layout = QHBoxLayout()

        # 创建 QSplitter 进行左右分屏
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：数据输入和统计结果显示
        left_panel = QFrame()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # 标题
        input_title = QLabel("📊 数据输入")
        input_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(input_title)

        # 数据输入模式选择
        mode_layout = QHBoxLayout()
        mode_label = QLabel("输入模式：")
        self.btn_single_mode = QRadioButton("单列数据")
        self.btn_single_mode.setChecked(True)
        self.btn_double_mode = QRadioButton("双列数据 (X, Y)")
        self.btn_single_mode.toggled.connect(self.on_mode_changed)
        self.btn_double_mode.toggled.connect(self.on_mode_changed)

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.btn_single_mode)
        mode_layout.addWidget(self.btn_double_mode)
        mode_layout.addStretch()
        left_layout.addLayout(mode_layout)

        left_layout.addSpacing(10)

        # 表格控件用于手动输入数据
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        # 默认显示为单列模式
        self.data_table.setHorizontalHeaderLabels(["数值 (Y)"])
        self.data_table.setColumnHidden(0, True)
        self.data_table.setMinimumHeight(200)
        self.data_table.setMaximumHeight(250)

        # 设置表格样式
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        
        # 为表格添加明确的样式，确保字体颜色和背景颜色有足够的对比度
        self.data_table.setStyleSheet("""
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
                color: #333333;
                background-color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTableWidget::item:selected:editable {
                background-color: #ffffff;
                color: #333333;
            }
            QTableWidget QLineEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #d9d9d9;
                padding: 2px;
            }
            QHeaderView::section {
                background-color: #f5f7fa;
                color: #333333;
                padding: 8px;
                border: 1px solid #d9d9d9;
                font-weight: 600;
                font-size: 12px;
            }
        """)

        # 添加初始行
        self.data_table.setRowCount(15)
        for i in range(15):
            x_item = QTableWidgetItem("")
            y_item = QTableWidgetItem("")
            self.data_table.setItem(i, 0, x_item)
            self.data_table.setItem(i, 1, y_item)

        left_layout.addWidget(self.data_table)

        # 连接单元格变化信号
        self.data_table.cellChanged.connect(self.on_cell_changed)

        # 按钮组
        button_layout = QHBoxLayout()
        self.btn_import_csv = QPushButton("📁 导入 CSV")
        self.btn_clear = QPushButton("🗑️ 清空数据")

        button_layout.addWidget(self.btn_import_csv)
        button_layout.addWidget(self.btn_clear)
        left_layout.addLayout(button_layout)

        left_layout.addSpacing(15)

        # 统计结果显示区
        stats_group = QGroupBox("📈 统计结果")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("输入数据后自动计算...")
        self.stats_label.setFont(QFont("Arial", 9))
        self.stats_label.setStyleSheet("color: #6c757d; line-height: 1.6;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()

        # 设置左侧面板样式
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(400)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
        """)

        # 右侧：绘图区域
        right_panel = QFrame()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # 图表类型选择
        chart_title = QLabel("📊 图表类型")
        chart_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(chart_title)

        chart_type_layout = QHBoxLayout()
        chart_type_layout.setSpacing(8)

        self.btn_chart_boxplot = QPushButton("箱型图")
        self.btn_chart_line = QPushButton("折线图")
        self.btn_chart_bar = QPushButton("柱状图")
        self.btn_chart_scatter = QPushButton("散点图")
        self.btn_chart_histogram = QPushButton("直方图")

        # 设置按钮样式
        for btn in [self.btn_chart_boxplot, self.btn_chart_line, self.btn_chart_bar,
                   self.btn_chart_scatter, self.btn_chart_histogram]:
            btn.setMinimumHeight(35)
            btn.setCheckable(True)
            btn.clicked.connect(self.on_chart_type_changed)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    color: #495057;
                    border: 1px solid #bdc3c7;
                    border-radius: 5px;
                    padding: 5px 12px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:checked {
                    background-color: #3498db;
                    color: #ffffff;
                    border: 1px solid #3498db;
                }
            """)

        # 默认选中散点图
        self.btn_chart_scatter.setChecked(True)

        chart_type_layout.addWidget(self.btn_chart_boxplot)
        chart_type_layout.addWidget(self.btn_chart_line)
        chart_type_layout.addWidget(self.btn_chart_bar)
        chart_type_layout.addWidget(self.btn_chart_scatter)
        chart_type_layout.addWidget(self.btn_chart_histogram)
        chart_type_layout.addStretch()

        # 创建容器并设置最大宽度
        chart_type_container = QWidget()
        chart_type_container.setLayout(chart_type_layout)
        chart_type_container.setMaximumWidth(500)
        right_layout.addWidget(chart_type_container)
        right_layout.addSpacing(10)

        # Matplotlib 画布（使用现代清新风）
        self.plot_canvas = MatplotlibCanvas(self, width=8, height=5, dpi=100, theme='modern')
        self.plot_canvas.setMinimumSize(600, 400)
        right_layout.addWidget(self.plot_canvas)

        # 设置右侧面板样式
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
        """)

        # 添加到 Splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 600])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # 连接信号
        self.btn_import_csv.clicked.connect(self.on_import_csv)
        self.btn_clear.clicked.connect(self.on_clear_data)

    def on_mode_changed(self):
        """数据输入模式切换"""
        if self.btn_single_mode.isChecked():
            self.data_mode = 'single'
            self.data_table.setHorizontalHeaderLabels(["数值 (Y)"])
            self.data_table.setColumnHidden(0, True)
        else:
            self.data_mode = 'double'
            self.data_table.setHorizontalHeaderLabels(["X", "Y"])
            self.data_table.setColumnHidden(0, False)

        self.update_statistics()
        self.update_chart()

    def on_cell_changed(self, row, column):
        """单元格内容变化时触发"""
        from PyQt6.QtCore import QTimer
        if self._update_timer is not None:
            self._update_timer.stop()
        self._update_timer = QTimer.singleShot(300, self._delayed_update)

    def _delayed_update(self):
        """延迟更新统计和图表"""
        self.update_statistics()
        self.update_chart()

    def on_chart_type_changed(self):
        """图表类型切换"""
        sender = self.sender()
        for btn in [self.btn_chart_boxplot, self.btn_chart_line, self.btn_chart_bar,
                   self.btn_chart_scatter, self.btn_chart_histogram]:
            btn.setChecked(False)
        sender.setChecked(True)

        if sender == self.btn_chart_boxplot:
            self.chart_type = 'boxplot'
        elif sender == self.btn_chart_line:
            self.chart_type = 'line'
        elif sender == self.btn_chart_bar:
            self.chart_type = 'bar'
        elif sender == self.btn_chart_scatter:
            self.chart_type = 'scatter'
        elif sender == self.btn_chart_histogram:
            self.chart_type = 'histogram'

        self.update_chart()

    def get_data_from_table(self):
        """从表格中读取数据，支持单列和双列模式"""
        x_data = []
        y_data = []

        row_count = self.data_table.rowCount()
        for i in range(row_count):
            y_item = self.data_table.item(i, 1)

            if self.data_mode == 'double':
                x_item = self.data_table.item(i, 0)
                if x_item is not None and y_item is not None:
                    try:
                        x_val = float(x_item.text())
                        y_val = float(y_item.text())
                        x_data.append(x_val)
                        y_data.append(y_val)
                    except ValueError:
                        continue
            else:  # single 模式
                if y_item is not None:
                    try:
                        y_val = float(y_item.text())
                        x_data.append(i + 1)  # 序号作为 X
                        y_data.append(y_val)
                    except ValueError:
                        continue

        return np.array(x_data), np.array(y_data)

    def update_statistics(self):
        """更新统计结果"""
        x_data, y_data = self.get_data_from_table()

        if len(y_data) == 0:
            self.stats_label.setText("暂无数据")
            return

        # 计算统计量
        result = calculate_statistics(y_data)

        # 构建统计显示文本
        stats_text = f"""📊 统计结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据点数量: {result['count']}
平均值: {result['mean']:.6f}
标准差: {result['std_dev']:.6f}
标准误差: {result['standard_error']:.6f}
方差: {result['variance']:.6f}
最小值: {result['min']:.6f}
最大值: {result['max']:.6f}
中位数: {result['median']:.6f}
极差: {result['range']:.6f}
"""

        if result['outliers']:
            stats_text += f"\n⚠️  检测到 {len(result['outliers'])} 个异常值:\n"
            for idx, val, z_score in result['outliers']:
                stats_text += f"  • 第 {idx + 1} 个数据: {val:.6f} (Z-score = {z_score:.2f})\n"
            stats_text += "\n建议：考虑剔除这些异常值后重新计算。\n"
        else:
            stats_text += "\n✓ 未检测到异常值。\n"

        stats_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        self.stats_label.setText(stats_text)

    def update_chart(self):
        """根据当前数据和图表类型更新图表"""
        x_data, y_data = self.get_data_from_table()

        if len(y_data) == 0:
            self.plot_canvas.ax.clear()
            self.plot_canvas.ax.text(0.5, 0.5, '请输入数据后显示图表',
                                   ha='center', va='center',
                                   transform=self.plot_canvas.ax.transAxes,
                                   color=self.plot_canvas.theme['text'])
            self.plot_canvas.draw()
            return

        # 根据图表类型绘制
        if self.chart_type == 'boxplot':
            xlabel = '数据' if self.data_mode == 'single' else 'Y'
            self.plot_canvas.plot_boxplot(y_data, title='箱型图', xlabel=xlabel, ylabel='数值')
        elif self.chart_type == 'line':
            xlabel = '序号' if self.data_mode == 'single' else 'X'
            ylabel = '数值 (Y)' if self.data_mode == 'single' else 'Y'
            self.plot_canvas.plot_line_chart(x_data, y_data, title='折线图', xlabel=xlabel, ylabel=ylabel)
        elif self.chart_type == 'bar':
            xlabel = '序号' if self.data_mode == 'single' else 'X'
            ylabel = '数值 (Y)' if self.data_mode == 'single' else 'Y'
            self.plot_canvas.plot_bar_chart(x_data, y_data, title='柱状图', xlabel=xlabel, ylabel=ylabel)
        elif self.chart_type == 'scatter':
            xlabel = '序号' if self.data_mode == 'single' else 'X'
            ylabel = '数值 (Y)' if self.data_mode == 'single' else 'Y'
            self.plot_canvas.plot_scatter(x_data, y_data, title='散点图', xlabel=xlabel, ylabel=ylabel)
        elif self.chart_type == 'histogram':
            xlabel = '数值'
            ylabel = '频数'
            self.plot_canvas.plot_histogram(y_data, title='直方图', xlabel=xlabel, ylabel=ylabel)

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
                import pandas as pd
                df = pd.read_csv(file_path)

                if df.shape[1] < 1:
                    QMessageBox.warning(self, "错误", "CSV 文件至少需要 1 列数据")
                    return

                # 根据列数自动切换模式
                if df.shape[1] == 1:
                    self.data_mode = 'single'
                    self.btn_single_mode.setChecked(True)
                    self.data_table.setHorizontalHeaderLabels(["数值 (Y)"])
                    self.data_table.setColumnHidden(0, True)
                else:
                    self.data_mode = 'double'
                    self.btn_double_mode.setChecked(True)
                    self.data_table.setHorizontalHeaderLabels(["X", "Y"])
                    self.data_table.setColumnHidden(0, False)

                # 清空并填充表格
                self.data_table.setRowCount(0)
                for i, row in df.iterrows():
                    row_pos = self.data_table.rowCount()
                    self.data_table.insertRow(row_pos)

                    if df.shape[1] == 1:
                        y_item = QTableWidgetItem(str(row.iloc[0]))
                        x_item = QTableWidgetItem(str(i + 1))
                        self.data_table.setItem(row_pos, 0, x_item)
                        self.data_table.setItem(row_pos, 1, y_item)
                    else:
                        x_item = QTableWidgetItem(str(row.iloc[0]))
                        y_item = QTableWidgetItem(str(row.iloc[1]))
                        self.data_table.setItem(row_pos, 0, x_item)
                        self.data_table.setItem(row_pos, 1, y_item)

                # 更新统计和图表
                self.update_statistics()
                self.update_chart()

                QMessageBox.information(self, "成功", f"已导入 {len(df)} 行数据")

        except Exception as e:
            QMessageBox.warning(self, "错误", f"导入 CSV 失败:\n{str(e)}")

    def on_clear_data(self):
        """清空数据"""
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有数据吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.data_table.setRowCount(15)
            for i in range(15):
                x_item = QTableWidgetItem("")
                y_item = QTableWidgetItem("")
                self.data_table.setItem(i, 0, x_item)
                self.data_table.setItem(i, 1, y_item)

            self.update_statistics()
            self.update_chart()
