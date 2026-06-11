import cv2
import numpy as np
from openai import OpenAI
from PyQt6.QtCore import QThread, pyqtSignal

from algorithms import count_peaks_in_signal, extract_center_intensity, smooth_signal


class LLMWorker(QThread):
    """LLM 对话工作线程。"""
    
    # 定义信号
    response_ready = pyqtSignal(str)  # 传输 AI 返回的文本
    error_occurred = pyqtSignal(str)  # 传输错误信息
    
    def __init__(
        self,
        api_key: str,
        user_question: str,
        context_data: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        """
        初始化 LLMWorker
        
        参数:
            api_key: API 密钥
            user_question: 用户提问
            context_data: 上下文数据（可选）
            base_url: 接口地址
            model: 模型名称
        """
        super().__init__()
        self.api_key = api_key
        self.user_question = user_question
        self.context_data = context_data
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def run(self):
        """执行 LLM 对话"""
        try:
            # 初始化 OpenAI 客户端
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # 构造 System Prompt
            system_prompt = """你是 PhysicsLab Pro 的智能助教，擅长物理实验原理、误差分析和数据解释。
            
你的特点：
• 回答要简洁专业，逻辑清晰
• 针对物理实验提供指导和分析
• 如果用户提供实验数据，帮助分析和解释
• 对于计算错误或不合理的数据，提出改进建议

回答格式：
- 简洁明了，避免冗长的解释
- 公式优先使用可直接阅读的标准数学符号，不要输出原始 LaTeX 代码
- 使用 λ、θ、Δ、π、α、β、ω、≈、∝、≤、≥、√、²、ₖ 等符号
- 例如写成 rₖ = √(kλR)、k = r²/(λR)、Δr ≈ λR/(2r)，不要写成 \\frac{}{} 或 \\sqrt{}
- 复杂公式单独成行，普通解释用中文短句
- 不要使用 Markdown 加粗星号，不要写 **标题** 或 **重点**；重点直接用中文表达
- 列表最多 3 点，使用“1. 2. 3.”编号，避免密集项目符号
- 提供必要的单位和数值精度
- 如需补充信息，以友好的方式请求"""
            
            # 构造用户消息
            user_message = self.user_question
            if self.context_data.strip():
                user_message = f"{self.user_question}\n\n背景信息:\n{self.context_data}"
            
            # 调用 AI 接口
            message = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                stream=False,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 提取返回内容
            response_text = message.choices[0].message.content
            self.response_ready.emit(response_text)
            
        except Exception as e:
            # 捕获所有错误并发射错误信号
            error_msg = f"❌ API 错误: {str(e)}"
            if "401" in str(e):
                error_msg = "❌ API Key 无效，请检查密钥是否正确"
            elif "connection" in str(e).lower():
                error_msg = "❌ 网络连接失败，请检查网络"
            elif "timeout" in str(e).lower():
                error_msg = "❌ 请求超时，请稍后重试"
            self.error_occurred.emit(error_msg)


class VideoWorker(QThread):
    """视频处理工作线程"""
    
    # 定义信号
    frame_ready = pyqtSignal(np.ndarray)  # 当前帧图像
    intensity_ready = pyqtSignal(int, float, float, int)  # 帧号, 原始亮度, 平滑亮度, 波峰计数
    progress_updated = pyqtSignal(int)  # 进度百分比
    finished_signal = pyqtSignal()  # 处理完成
    
    def __init__(self, video_path, center_point):
        super().__init__()
        self.video_path = video_path
        self.center_point = center_point
        self.is_running = False
        self.is_paused = False
    
    def run(self):
        """线程主函数：处理视频"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            intensities_raw = []
            intensities_smoothed = []
            frame_numbers = []
            peak_count = 0
            
            self.is_running = True
            
            while self.is_running:
                if self.is_paused:
                    self.msleep(100)  # 暂停时等待
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 提取中心点亮度
                intensity = extract_center_intensity(frame, self.center_point, region_size=5)
                intensities_raw.append(intensity)
                frame_numbers.append(frame_count)
                
                # Use a larger SG window for longer sequences.
                if len(intensities_raw) >= 31:
                    smoothed = smooth_signal(np.array(intensities_raw), window_length=31, polyorder=2)
                    intensities_smoothed = smoothed.tolist()
                    
                    # 实时波峰检测（在平滑后的信号上）
                    if len(intensities_smoothed) >= 10:
                        current_smoothed = intensities_smoothed[-1]
                        peak_count = count_peaks_in_signal(np.array(intensities_smoothed))
                elif len(intensities_raw) >= 15:
                    smoothed = smooth_signal(np.array(intensities_raw), window_length=15, polyorder=2)
                    intensities_smoothed = smoothed.tolist()
                    if len(intensities_smoothed) >= 10:
                        current_smoothed = intensities_smoothed[-1]
                        peak_count = count_peaks_in_signal(np.array(intensities_smoothed))
                else:
                    # 当数据点少于11个时，直接使用原始值
                    intensities_smoothed.append(intensity)
                
                # 发送信号更新 UI
                current_smoothed = intensities_smoothed[-1] if intensities_smoothed else intensity
                self.intensity_ready.emit(frame_count, intensity, current_smoothed, peak_count)
                self.frame_ready.emit(frame)
                
                # 更新进度
                if total_frames > 0:
                    progress = int((frame_count + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)
                
                frame_count += 1
                
                # 控制处理速度（避免过快）
                if fps > 0:
                    self.msleep(int(1000 / fps))
            
            cap.release()
            self.finished_signal.emit()
            
        except Exception as e:
            print(f"视频处理错误: {str(e)}")
            self.finished_signal.emit()
    
    def stop(self):
        """停止处理"""
        self.is_running = False
    
    def pause(self):
        """暂停处理"""
        self.is_paused = True
    
    def resume(self):
        """恢复处理"""
        self.is_paused = False
