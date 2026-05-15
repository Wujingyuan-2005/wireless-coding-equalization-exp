"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)


def hamming74_encode(bits):
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # TODO: 将 bits reshape 为 (-1, 4)，再与 HAMMING_G 相乘并对 2 取模。
    # 建议实现步骤：
    # 1. 将 `bits` reshape 成形状 `(-1, 4)`
    blocks = bits.reshape(-1, 4)
    # 2. 计算 `blocks @ HAMMING_G`
    encoded_blocks = np.dot(blocks, HAMMING_G)
    # 3. 对结果取模 2
    encoded_blocks = encoded_blocks % 2
    # 4. 将二维数组 flatten 成一维数组返回
    return encoded_blocks.flatten()
    #raise NotImplementedError('请实现 Hamming(7,4) 编码')


def hamming74_syndrome(codewords):
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    # TODO: 计算 s = r H^T mod 2。
    # 建议实现步骤：
     # 1. 如果输入是一维数组，reshape 为 `(-1, 7)`
    if codewords.ndim == 1:
        codewords = codewords.reshape(-1, 7)
    # 2. 计算 `codewords @ HAMMING_H.T`
    syndrome = np.dot(codewords, HAMMING_H.T)
    # 3. 对结果取模 2
    syndrome = syndrome % 2
    # 4. 返回伴随式矩阵
    return syndrome
    #raise NotImplementedError('请实现伴随式计算')


def hamming74_decode(received):
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    # TODO: 使用 hamming74_syndrome 完成单比特纠错，并返回前 4 个信息位。
    # 建议实现步骤：
    # 1. 将 `received` reshape 为 `(-1, 7)`，复制一份避免修改原数据
    recv_blocks = received.copy().reshape(-1, 7)
    # 2. 调用上面的函数计算伴随式
    syndromes = hamming74_syndrome(recv_blocks)

    # 3 & 4. 寻找非零伴随式并翻转对应比特
    for i in range(len(recv_blocks)):
        s = syndromes[i]
        if np.any(s != 0): # 检测到错误
        # 遍历校验矩阵H的每一列，寻找与伴随式匹配的列
            for col_idx in range(7):
                if np.array_equal(s, HAMMING_H[:, col_idx]):
                    # 找到错误位置，翻转该比特 (0变1，1变0)
                    recv_blocks[i, col_idx] ^= 1 
                    break

    # 5. 取每个码字前 4 位 (信息位) 并 flatten 返回
    return recv_blocks[:, :4].flatten()
    #raise NotImplementedError('请实现 Hamming(7,4) 译码')


def convolutional_encode(bits):
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # TODO: 选做任务，可参考课件第6章卷积码部分。
        
    raise NotImplementedError('选做：请实现卷积码编码')


def viterbi_decode_hard(received_bits):
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。
    """
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    # TODO: 选做任务，可使用汉明距离作为路径度量。
    raise NotImplementedError('选做：请实现 Viterbi 硬判决译码')


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')


if __name__ == '__main__':
    run_coding_demo()
