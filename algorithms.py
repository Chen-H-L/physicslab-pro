"""
PhysicsLab Pro - 物理算法模块
包含所有物理计算和分析的核心算法
"""

import numpy as np
import cv2
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional, Dict


def extract_line_intensity(
    image: np.ndarray, 
    point_a: Tuple[int, int], 
    point_b: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取图像上两点之间线段上的像素亮度分布
    
    参数:
        image: 灰度图像数组 (numpy.ndarray)
        point_a: 起点坐标 (x, y)
        point_b: 终点坐标 (x, y)
    
    返回:
        distances: 沿线段的距离数组（像素）
        intensities: 对应的亮度值数组
    """
    # 转换为灰度图（如果输入是彩色图）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    x1, y1 = point_a
    x2, y2 = point_b
    
    # 计算线段长度（像素数）
    length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    
    if length == 0:
        return np.array([]), np.array([])
    
    # 生成沿线段均匀分布的点
    distances = np.linspace(0, length, length)
    x_coords = np.linspace(x1, x2, length).astype(int)
    y_coords = np.linspace(y1, y2, length).astype(int)
    
    # 确保坐标在图像范围内
    h, w = gray.shape
    x_coords = np.clip(x_coords, 0, w - 1)
    y_coords = np.clip(y_coords, 0, h - 1)
    
    # 提取亮度值
    intensities = gray[y_coords, x_coords]
    
    return distances, intensities

def _adaptive_smoothing_window(signal_length: int) -> int:
    """Return an odd Savitzky-Golay window suitable for image line profiles."""
    if signal_length < 5:
        return 0

    window = int(round(signal_length * 0.004))
    window = max(5, min(window, 9, signal_length))
    if window % 2 == 0:
        window -= 1
    return window if window >= 5 else 0


def _merge_peaks_without_clear_valley(
    signal: np.ndarray,
    peaks: np.ndarray,
    min_valley_drop: float
) -> np.ndarray:
    """Merge local maxima that do not have a clear dark valley between them."""
    if len(peaks) <= 1:
        return peaks

    merged = []
    i = 0
    while i < len(peaks):
        cluster = [int(peaks[i])]
        j = i + 1

        while j < len(peaks):
            left = cluster[-1]
            right = int(peaks[j])
            if right <= left:
                j += 1
                continue

            between = signal[left:right + 1]
            valley = float(np.min(between))
            lower_peak = float(min(signal[left], signal[right]))
            if lower_peak - valley < min_valley_drop:
                cluster.append(right)
                j += 1
            else:
                break

        heights = signal[cluster]
        max_height = float(np.max(heights))
        near_top = [
            p for p in cluster
            if max_height - float(signal[p]) <= max(min_valley_drop * 0.25, 1.0)
        ]
        center = float(np.mean(cluster))
        merged.append(min(near_top, key=lambda p: abs(p - center)))
        i = j

    return np.asarray(merged, dtype=int)


def detect_peaks(
    intensities: np.ndarray,
    min_height: Optional[float] = None,
    min_distance: Optional[int] = None,
    prominence: Optional[float] = None,
    merge_shallow_valleys: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    使用 scipy.signal.find_peaks 检测波峰（亮条纹）
    
    参数:
        intensities: 亮度值数组
        min_height: 最小波峰高度（可选）
        min_distance: 波峰之间的最小距离（像素）
        prominence: 波峰的突出度（可选，用于过滤噪声）
    
    返回:
        peaks: 波峰位置的索引数组
        properties: 波峰属性字典
    """
    intensities = np.asarray(intensities, dtype=float)
    if intensities.size < 3:
        return np.array([], dtype=int), {}

    finite_mask = np.isfinite(intensities)
    if not np.any(finite_mask):
        return np.array([], dtype=int), {}

    if not np.all(finite_mask):
        intensities = np.interp(
            np.arange(len(intensities)),
            np.flatnonzero(finite_mask),
            intensities[finite_mask],
        )

    signal_min = float(np.percentile(intensities, 5))
    signal_max = float(np.percentile(intensities, 95))
    signal_range = max(signal_max - signal_min, 1.0)

    # 如果没有指定最小高度，使用平均值作为阈值
    if min_height is None:
        min_height = float(np.mean(intensities))
    
    # 如果没有指定突出度，使用标准差的 0.5 倍
    if prominence is None:
        prominence = max(float(np.std(intensities)) * 0.18, signal_range * 0.035, 0.8)

    if min_distance is None:
        min_distance = max(2, int(round(len(intensities) * 0.003)))
    
    # 检测波峰
    peaks, properties = find_peaks(
        intensities,
        height=min_height,
        distance=min_distance,
        prominence=prominence
    )

    if merge_shallow_valleys and len(peaks) > 1:
        valley_drops = []
        for left, right in zip(peaks[:-1], peaks[1:]):
            between = intensities[int(left):int(right) + 1]
            lower_peak = min(float(intensities[int(left)]), float(intensities[int(right)]))
            valley_drops.append(lower_peak - float(np.min(between)))

        typical_valley_drop = float(np.median(valley_drops)) if valley_drops else 0.0
        min_valley_drop = max(
            signal_range * 0.12,
            typical_valley_drop * 0.65,
            float(np.std(intensities)) * 0.35,
            2.0,
        )
        peaks = _merge_peaks_without_clear_valley(intensities, peaks, min_valley_drop)
    
    return peaks, properties


def analyze_interference_pattern(
    image: np.ndarray,
    point_a: Tuple[int, int],
    point_b: Tuple[int, int]
) -> dict:
    """
    分析干涉图像：提取线段上的亮度分布并检测条纹
    
    参数:
        image: 输入图像（可以是彩色或灰度）
        point_a: 起点坐标 (x, y)
        point_b: 终点坐标 (x, y)
    
    返回:
        包含分析结果的字典:
        - distances: 距离数组
        - intensities: 亮度数组
        - peaks: 波峰位置索引
        - peak_count: 波峰数量（条纹级数）
        - avg_spacing: 平均条纹间距（像素）
    """
    # 提取线段上的亮度分布
    distances, intensities = extract_line_intensity(image, point_a, point_b)
    
    if len(intensities) == 0:
        return {
            'distances': np.array([]),
            'intensities': np.array([]),
            'analysis_intensities': np.array([]),
            'peaks': np.array([]),
            'peak_count': 0,
            'avg_spacing': 0.0
        }
    
    analysis_intensities = intensities.astype(float)
    smoothing_window = _adaptive_smoothing_window(len(analysis_intensities))
    if smoothing_window > 0:
        analysis_intensities = smooth_signal(
            analysis_intensities,
            window_length=smoothing_window,
            polyorder=2,
        )

    # 检测波峰
    peaks, properties = detect_peaks(analysis_intensities)
    
    # 计算波峰数量
    peak_count = len(peaks)
    
    # 计算平均条纹间距
    avg_spacing = 0.0
    if peak_count > 1:
        # 计算相邻波峰之间的距离
        peak_distances = np.diff(distances[peaks])
        avg_spacing = np.mean(peak_distances)
    
    return {
        'distances': distances,
        'intensities': intensities,
        'analysis_intensities': analysis_intensities,
        'peaks': peaks,
        'peak_count': peak_count,
        'avg_spacing': avg_spacing,
        'peak_positions': distances[peaks] if peak_count > 0 else np.array([])
    }


def extract_center_intensity(
    image: np.ndarray,
    center_point: Tuple[int, int],
    region_size: int = 5
) -> float:
    """
    提取图像中心点（或周围区域）的平均亮度值
    
    参数:
        image: 输入图像（可以是彩色或灰度）
        center_point: 中心点坐标 (x, y)
        region_size: 区域大小（默认 5x5）
    
    返回:
        平均亮度值
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    x, y = center_point
    h, w = gray.shape
    
    # 计算区域边界
    half_size = region_size // 2
    x1 = max(0, x - half_size)
    x2 = min(w, x + half_size + 1)
    y1 = max(0, y - half_size)
    y2 = min(h, y + half_size + 1)
    
    # 提取区域并计算平均亮度
    region = gray[y1:y2, x1:x2]
    avg_intensity = np.mean(region)
    
    return float(avg_intensity)


def smooth_signal(
    signal: np.ndarray,
    window_length: int = 15,
    polyorder: int = 2
) -> np.ndarray:
    """
    使用 Savitzky-Golay 滤波器平滑信号
    
    参数:
        signal: 输入信号数组
        window_length: 窗口长度（必须是奇数）
        polyorder: 多项式阶数
    
    返回:
        平滑后的信号数组
    """
    if len(signal) < window_length:
        # 如果信号太短，直接返回原信号
        return signal
    
    # 确保 window_length 是奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 确保 window_length 不超过信号长度
    window_length = min(window_length, len(signal))
    if window_length % 2 == 0:
        window_length -= 1
    
    # 确保 polyorder < window_length
    polyorder = min(polyorder, window_length - 1)
    
    try:
        smoothed = savgol_filter(signal, window_length, polyorder)
        return smoothed
    except Exception:
        # 如果滤波失败，返回原信号
        return signal


def detect_peak_crossing(
    current_value: float,
    previous_value: float,
    threshold: float = 0.0
) -> bool:
    """
    检测是否出现新的波峰（通过过零点或峰值检测）
    
    参数:
        current_value: 当前亮度值
        previous_value: 前一个亮度值
        threshold: 阈值（用于过滤噪声）
    
    返回:
        如果检测到新的波峰，返回 True
    """
    # 简单的峰值检测：当前值大于前一个值，且超过阈值
    # 这是一个简化的检测方法，实际应用中可能需要更复杂的逻辑
    if current_value > previous_value + threshold:
        return False
    
    # 检测从上升转为下降的转折点（峰值）
    # 这里使用简单的差分检测
    return False  # 这个函数主要用于实时检测，更精确的检测在平滑后的信号上进行


def count_peaks_in_signal(
    signal: np.ndarray,
    min_distance: int = 5,
    min_prominence: Optional[float] = None
) -> int:
    """
    统计信号中的波峰数量
    
    参数:
        signal: 输入信号数组
        min_distance: 波峰之间的最小距离
        min_prominence: 最小突出度
    
    返回:
        波峰数量
    """
    if len(signal) < min_distance:
        return 0
    
    if min_prominence is None:
        min_prominence = np.std(signal) * 0.3
    
    try:
        peaks, _ = find_peaks(
            signal,
            distance=min_distance,
            prominence=min_prominence
        )
        return len(peaks)
    except Exception:
        return 0


# ==================== 数据拟合相关函数 ====================

def linear_func(x, a, b):
    """线性函数: y = ax + b"""
    return a * x + b


def exponential_func(x, a, b):
    """指数函数: y = a * exp(b * x)"""
    return a * np.exp(b * x)


def cosine_func(x, a, b, c):
    """余弦函数: y = a * cos(b * x) + c"""
    return a * np.cos(b * x) + c


def calculate_r_squared(y_true, y_pred):
    """
    计算 R² (决定系数，拟合优度)
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
    
    返回:
        R² 值（越接近 1 表示拟合越好）
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
    
    if ss_tot == 0:
        return 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def auto_fit(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    自动拟合数据，尝试多个公式并选择最佳拟合
    
    参数:
        x: X 数据数组
        y: Y 数据数组
    
    返回:
        包含拟合结果的字典:
        - best_formula: 最佳拟合公式名称
        - best_params: 最佳拟合参数
        - best_r_squared: 最佳 R² 值
        - y_fitted: 拟合曲线数据
        - all_results: 所有尝试的拟合结果
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("数据长度不匹配或数据点太少（至少需要 2 个点）")
    
    # 移除 NaN 和 Inf 值
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 2:
        raise ValueError("有效数据点太少")
    
    results = []
    
    # 1. 尝试线性拟合: y = ax + b
    try:
        params_linear, _ = curve_fit(linear_func, x, y, maxfev=10000)
        y_pred_linear = linear_func(x, *params_linear)
        r2_linear = calculate_r_squared(y, y_pred_linear)
        results.append({
            'formula': '线性',
            'formula_str': f'y = {params_linear[0]:.4f}x + {params_linear[1]:.4f}',
            'params': params_linear,
            'r_squared': r2_linear,
            'y_fitted': y_pred_linear,
            'func': linear_func
        })
    except Exception as e:
        results.append({
            'formula': '线性',
            'formula_str': '拟合失败',
            'params': None,
            'r_squared': -np.inf,
            'y_fitted': None,
            'func': linear_func,
            'error': str(e)
        })
    
    # 2. 尝试指数拟合: y = a * exp(b * x)
    try:
        # 指数拟合需要初始猜测，且 y 值必须为正
        if np.all(y > 0):
            # 使用对数线性化进行初始猜测
            log_y = np.log(y)
            try:
                p0_exp = np.polyfit(x, log_y, 1)
                initial_guess = [np.exp(p0_exp[1]), p0_exp[0]]
            except:
                initial_guess = [1.0, 0.01]
            
            params_exp, _ = curve_fit(exponential_func, x, y, p0=initial_guess, maxfev=10000)
            y_pred_exp = exponential_func(x, *params_exp)
            r2_exp = calculate_r_squared(y, y_pred_exp)
            results.append({
                'formula': '指数',
                'formula_str': f'y = {params_exp[0]:.4f} * exp({params_exp[1]:.4f} * x)',
                'params': params_exp,
                'r_squared': r2_exp,
                'y_fitted': y_pred_exp,
                'func': exponential_func
            })
        else:
            results.append({
                'formula': '指数',
                'formula_str': '拟合失败（Y 值必须为正）',
                'params': None,
                'r_squared': -np.inf,
                'y_fitted': None,
                'func': exponential_func,
                'error': 'Y 值包含非正数'
            })
    except Exception as e:
        results.append({
            'formula': '指数',
            'formula_str': '拟合失败',
            'params': None,
            'r_squared': -np.inf,
            'y_fitted': None,
            'func': exponential_func,
            'error': str(e)
        })
    
    # 3. 尝试余弦拟合: y = a * cos(b * x) + c
    try:
        # 余弦拟合需要初始猜测
        y_range = np.max(y) - np.min(y)
        y_mean = np.mean(y)
        x_range = np.max(x) - np.min(x)
        
        if x_range > 0:
            # 估算频率（假设至少有一个完整周期）
            freq_guess = 2 * np.pi / x_range
        else:
            freq_guess = 0.1
        
        initial_guess = [y_range / 2, freq_guess, y_mean]
        params_cos, _ = curve_fit(cosine_func, x, y, p0=initial_guess, maxfev=10000)
        y_pred_cos = cosine_func(x, *params_cos)
        r2_cos = calculate_r_squared(y, y_pred_cos)
        results.append({
            'formula': '余弦',
            'formula_str': f'y = {params_cos[0]:.4f} * cos({params_cos[1]:.4f} * x) + {params_cos[2]:.4f}',
            'params': params_cos,
            'r_squared': r2_cos,
            'y_fitted': y_pred_cos,
            'func': cosine_func
        })
    except Exception as e:
        results.append({
            'formula': '余弦',
            'formula_str': '拟合失败',
            'params': None,
            'r_squared': -np.inf,
            'y_fitted': None,
            'func': cosine_func,
            'error': str(e)
        })
    
    # 选择 R² 值最大的拟合结果
    valid_results = [r for r in results if r['r_squared'] != -np.inf and r['y_fitted'] is not None]
    
    if not valid_results:
        raise ValueError("所有拟合公式都失败了")
    
    best_result = max(valid_results, key=lambda r: r['r_squared'])
    
    # 生成更密集的拟合曲线用于绘图
    x_fitted = np.linspace(np.min(x), np.max(x), 200)
    y_fitted_dense = best_result['func'](x_fitted, *best_result['params'])
    
    return {
        'best_formula': best_result['formula'],
        'best_formula_str': best_result['formula_str'],
        'best_params': best_result['params'],
        'best_r_squared': best_result['r_squared'],
        'x_fitted': x_fitted,
        'y_fitted': y_fitted_dense,
        'x_data': x,
        'y_data': y,
        'all_results': results
    }


# ==================== 不确定度计算相关函数 ====================

def calculate_uncertainty(data_str: str) -> Dict:
    """
    计算测量数据的不确定度
    
    参数:
        data_str: 逗号分隔的数字字符串，例如 "10.1, 10.2, 10.5"
    
    返回:
        包含计算结果的字典:
        - mean: 平均值
        - std_dev: 标准差
        - standard_error: 标准误差（A类不确定度）
        - outliers: 异常值列表（Z-score > 3）
        - data: 原始数据数组
    """
    # 解析字符串
    try:
        # 移除空格并分割
        data_list = [float(x.strip()) for x in data_str.split(',') if x.strip()]
    except ValueError as e:
        raise ValueError(f"数据格式错误: {str(e)}\n请确保输入的是逗号分隔的数字，例如: 10.1, 10.2, 10.5")
    
    if len(data_list) < 2:
        raise ValueError("至少需要 2 个数据点才能计算不确定度")
    
    data = np.array(data_list)
    
    # 计算统计量
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # 样本标准差（使用 n-1）
    standard_error = std_dev / np.sqrt(len(data))  # 标准误差（A类不确定度）
    
    # 检测异常值（Z-score > 3）
    if std_dev > 0:
        z_scores = np.abs((data - mean) / std_dev)
        outlier_indices = np.where(z_scores > 3)[0]
        outliers = [(i, data[i], z_scores[i]) for i in outlier_indices]
    else:
        outliers = []
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'standard_error': standard_error,
        'outliers': outliers,
        'data': data,
        'count': len(data)
    }


def calculate_statistics(data: np.ndarray) -> Dict:
    """
    直接从数组计算统计数据（无需字符串解析）

    参数:
        data: 输入数据数组 (numpy.ndarray)

    返回:
        包含计算结果的字典:
        - count: 数据点数量
        - mean: 平均值
        - std_dev: 标准差
        - standard_error: 标准误差（A类不确定度）
        - variance: 方差
        - min: 最小值
        - max: 最大值
        - median: 中位数
        - range: 极差（最大值-最小值）
        - outliers: 异常值列表（Z-score > 3）
        - data: 原始数据数组
    """
    # 确保是 numpy 数组
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # 移除 NaN 和 Inf 值
    valid_mask = np.isfinite(data)
    data = data[valid_mask]

    if len(data) == 0:
        raise ValueError("没有有效数据点")

    # 计算基本统计量
    count = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # 样本标准差（使用 n-1）
    variance = np.var(data, ddof=1)  # 样本方差
    standard_error = std_dev / np.sqrt(count) if count > 0 else 0
    data_min = np.min(data)
    data_max = np.max(data)
    median = np.median(data)
    data_range = data_max - data_min

    # 检测异常值（Z-score > 3）
    outliers = []
    if std_dev > 0 and count > 1:
        z_scores = np.abs((data - mean) / std_dev)
        outlier_indices = np.where(z_scores > 3)[0]
        outliers = [(i, data[i], z_scores[i]) for i in outlier_indices]

    return {
        'count': count,
        'mean': mean,
        'std_dev': std_dev,
        'standard_error': standard_error,
        'variance': variance,
        'min': data_min,
        'max': data_max,
        'median': median,
        'range': data_range,
        'outliers': outliers,
        'data': data
    }
