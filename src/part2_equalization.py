"""
Part 2：信道均衡实验

学生需要完成 ZF 均衡器估计、FIR 滤波应用和 LMS 自适应均衡。
"""

import numpy as np
from utils import (
    bpsk_demodulate,
    bpsk_modulate,
    calculate_ber,
    generate_bits,
    multipath_channel,
    plot_equalization_results,
    plot_mse_curve,
)


def estimate_zf_equalizer(channel, num_taps):
    """
    估计迫零（Zero-Forcing, ZF）FIR 均衡器。

    参数:
        channel: 一维信道冲激响应，例如 np.array([0.9, 0.3, -0.2])。
        num_taps: 均衡器抽头数，建议为奇数。

    返回:
        taps: 一维 FIR 均衡器系数。

    提示:
        1. 构造信道与均衡器卷积的线性方程 A @ taps ≈ d。
        2. d 为中心位置为 1 的冲激响应。
        3. 使用 np.linalg.lstsq 求最小二乘解。
    """
    channel = np.asarray(channel, dtype=float)
    if channel.ndim != 1 or len(channel) == 0:
        raise ValueError('channel 必须是一维非空数组')
    if num_taps < 1:
        raise ValueError('num_taps 必须为正整数')

    # TODO: 构造卷积矩阵并求解 ZF 均衡器抽头。
    from scipy.linalg import convolution_matrix # 可能需要导入此工具或自己构造
    # 1. 构造卷积矩阵 A
    # channel 长度为 L，taps 长度为 N，卷积结果长度为 L + N - 1
    L = len(channel)
    A = np.zeros((L + num_taps - 1, num_taps))
    for i in range(num_taps):
        A[i:i+L, i] = channel

    # 2. 构造目标冲激响应 d
    d = np.zeros(L + num_taps - 1)
    delay = num_taps // 2  # 通常将 1 放在中心位置
    d[delay] = 1.0

    # 3. 使用最小二乘解求 taps
    taps, _, _, _ = np.linalg.lstsq(A, d, rcond=None)

    # 4. 返回 taps
    return taps
    #raise NotImplementedError('请实现 ZF 均衡器估计')


def apply_fir_filter(signal, taps):
    """
    对信号应用 FIR 滤波器，并返回与输入等长的输出。

    参数:
        signal: 输入序列。
        taps: FIR 滤波器系数。

    返回:
        filtered: 与 signal 等长的滤波输出。
    """
    signal = np.asarray(signal, dtype=float)
    taps = np.asarray(taps, dtype=float)
    if signal.ndim != 1 or taps.ndim != 1:
        raise ValueError('signal 和 taps 必须是一维数组')

    # TODO: 使用 np.convolve，并截取与 signal 等长的输出。
    # 建议实现步骤极其简单：
    filtered_signal = np.convolve(signal, taps, mode='full')
    return filtered_signal[:len(signal)]
    #raise NotImplementedError('请实现 FIR 滤波')


def lms_equalizer(rx_train, tx_train, num_taps, step_size=0.01):
    """
    使用训练序列实现 LMS 自适应均衡。

    参数:
        rx_train: 接收训练序列。
        tx_train: 期望发送训练符号。
        num_taps: 均衡器抽头数。
        step_size: LMS 步长 μ。

    返回:
        taps: 训练后的均衡器系数。
        errors: 每次迭代的误差 e[n]。

    提示:
        1. 抽头向量可初始化为中心抽头为 1。
        2. y[n] = w^T x[n]
        3. e[n] = d[n] - y[n]
        4. w = w + μ e[n] x[n]
    """
    rx_train = np.asarray(rx_train, dtype=float)
    tx_train = np.asarray(tx_train, dtype=float)
    if len(rx_train) != len(tx_train):
        raise ValueError('rx_train 和 tx_train 长度必须一致')
    if num_taps < 1:
        raise ValueError('num_taps 必须为正整数')

    # TODO: 实现 LMS 自适应均衡训练。
    # 1. 初始化 taps，中心抽头为1，其余为0
    taps = np.zeros(num_taps)
    taps[num_taps // 2] = 1.0

    errors = []
    # 2. 从第 num_taps - 1 个样本开始迭代
    for i in range(num_taps - 1, len(rx_train)):
        # 3. 构造当前输入向量 x (注意时间反转，越新的样本在越前面)
        x = rx_train[i : i - num_taps : -1] if i - num_taps >= -1 else rx_train[i::-1]
        # 如果切片不够长(边界情况)，通常补零，但从 num_taps-1 开始切片长度刚好是 num_taps
        if len(x) < num_taps:
            x = np.pad(x, (0, num_taps - len(x)))

        # 4. 计算输出 y
        y = np.dot(taps, x)
        
        # 5. 计算误差 e
        d = tx_train[i - num_taps // 2] # 对齐中心延迟
        e = d - y
        errors.append(e)
        
        # 6. 更新 taps
        taps = taps + step_size * e * x

    return taps, np.array(errors)
    #raise NotImplementedError('请实现 LMS 均衡器')


def run_equalization_demo():
    """运行 Part 2 演示并生成均衡效果图。"""
    print('=' * 60)
    print('Part 2：信道均衡实验')
    print('=' * 60)

    try:
        bits = generate_bits(2000, seed=2027)
        symbols = bpsk_modulate(bits)
        channel = np.array([0.9, 0.35, -0.25])
        rx = multipath_channel(symbols, channel, noise_std=0.12, seed=7)

        zf_taps = estimate_zf_equalizer(channel, num_taps=7)
        zf_output = apply_fir_filter(rx, zf_taps)

        lms_taps, errors = lms_equalizer(rx[:800], symbols[:800], num_taps=7, step_size=0.01)
        lms_output = apply_fir_filter(rx, lms_taps)

        raw_bits = bpsk_demodulate(rx[: len(bits)])
        eq_bits = bpsk_demodulate(lms_output[: len(bits)])
        print(f'均衡前 BER: {calculate_ber(bits, raw_bits):.4f}')
        print(f'LMS 均衡后 BER: {calculate_ber(bits, eq_bits):.4f}')

        plot_equalization_results(symbols, rx, lms_output, 'equalization_eye_comparison.png')
        plot_mse_curve(errors, 'equalization_mse_curve.png')
        print('✅ 已生成均衡结果图')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 2 运行失败：{error}')


if __name__ == '__main__':
    run_equalization_demo()
