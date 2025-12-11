import numpy as np
import torch
from torch import nn
import math


def get_sinusoid_relative_encoding(max_rel_pos, d_hid, n_head=None, symmetric=True):
    """
    正弦相对位置编码

    参数说明:
        max_rel_pos: 最大相对位置距离（正负方向）
        d_hid: 每个相对位置编码向量的维度
        n_head: 注意力头的数量（可选）
        symmetric: 是否对称（即相对位置i-j和j-i的编码是否对称）

    返回:
        相对位置编码张量，形状根据参数不同：
        - 如果n_head为None: 形状为(2*max_rel_pos+1, d_hid)
        - 如果n_head不为None: 形状为(n_head, 2*max_rel_pos+1, d_hid)

    说明:
        相对位置编码表示两个位置之间的相对关系，常用于添加到注意力分数中。
        每个相对距离都有一个d_hid维的编码向量。
    """
    # 生成相对位置索引: [-max_rel_pos, -max_rel_pos+1, ..., max_rel_pos]
    positions = torch.arange(-max_rel_pos, max_rel_pos + 1).float()

    def get_position_angle_vec(position):
        """计算单个相对位置的编码向量"""
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # 生成正弦/余弦相对位置编码表
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in positions])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度使用正弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度使用余弦函数

    # 转换为PyTorch张量
    rel_pos_enc = torch.FloatTensor(sinusoid_table)  # 形状: (2*max_rel_pos+1, d_hid)

    # 如果不使用对称编码，为正向和负向分别生成编码
    if not symmetric:
        # 重新生成，这次分别处理正向和负向距离
        positive_positions = torch.arange(0, max_rel_pos + 1).float()
        negative_positions = torch.arange(-max_rel_pos, 0).float()

        # 正向相对位置编码
        pos_table = np.array([get_position_angle_vec(pos_i) for pos_i in positive_positions])
        pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])

        # 负向相对位置编码（使用不同的角度计算）
        neg_table = np.array([get_position_angle_vec(pos_i) for pos_i in negative_positions])
        neg_table[:, 0::2] = np.sin(neg_table[:, 0::2])
        neg_table[:, 1::2] = np.cos(neg_table[:, 1::2])

        # 合并正向和负向编码
        rel_pos_enc = torch.FloatTensor(np.concatenate([neg_table, pos_table], axis=0))

    # 如果需要为每个注意力头生成独立的相对位置编码
    if n_head is not None:
        # 为每个头创建稍微不同的相对位置编码
        rel_pos_enc = rel_pos_enc.unsqueeze(0)  # (1, 2*max_rel_pos+1, d_hid)
        # 使用不同的频率为每个头生成编码
        head_factors = torch.arange(1, n_head + 1).float().view(n_head, 1, 1)
        rel_pos_enc = rel_pos_enc * head_factors / n_head
        # 形状: (n_head, 2*max_rel_pos+1, d_hid)

    return rel_pos_enc


def get_learnable_relative_encoding(max_rel_pos, d_hid, n_head=None):
    """
    可学习的相对位置编码

    参数说明:
        max_rel_pos: 最大相对位置距离
        d_hid: 每个相对位置编码向量的维度
        n_head: 注意力头的数量（可选）

    返回:
        可学习的相对位置编码参数

    说明:
        许多现代Transformer模型使用可学习的相对位置编码，因为它们可以更好地适应数据。
    """
    if n_head is None:
        # 形状: (2*max_rel_pos+1, d_hid)
        return nn.Parameter(torch.randn(2 * max_rel_pos + 1, d_hid) * 0.02)
    else:
        # 形状: (n_head, 2*max_rel_pos+1, d_hid) 或 (n_head, 2*max_rel_pos+1)
        # 这里提供两种选择：
        # 1. 每个头每个相对位置有一个d_hid维向量
        # 2. 每个头每个相对位置只有一个标量（更常见）

        # 选择1: 多维编码（更灵活但参数更多）
        # return nn.Parameter(torch.randn(n_head, 2 * max_rel_pos + 1, d_hid) * 0.02)

        # 选择2: 标量编码（更常用）
        return nn.Parameter(torch.randn(n_head, 2 * max_rel_pos + 1) * 0.02)


def get_relative_position_bias(max_rel_pos, n_head, method='sinusoid'):
    """
    获取相对位置偏置（用于直接添加到注意力分数）

    参数说明:
        max_rel_pos: 最大相对位置距离
        n_head: 注意力头的数量
        method: 编码方法，'sinusoid'（正弦）或 'learnable'（可学习）

    返回:
        相对位置偏置，形状为(n_head, 2*max_rel_pos+1)或(n_head, 2*max_rel_pos+1, d_hid)

    说明:
        这是最常见的相对位置编码形式，直接为每个注意力头生成标量偏置。
    """
    if method == 'sinusoid':
        # 使用正弦函数生成相对位置偏置（标量形式）
        positions = torch.arange(-max_rel_pos, max_rel_pos + 1).float()

        # 为每个头使用不同的频率
        biases = []
        for h in range(n_head):
            # 每个头使用不同的基础频率
            freq = 10000.0 ** (2 * h / n_head)
            # 计算正弦偏置
            bias = torch.sin(positions / freq)
            biases.append(bias)

        bias_table = torch.stack(biases, dim=0)  # (n_head, 2*max_rel_pos+1)
        return bias_table

    elif method == 'learnable':
        # 可学习的相对位置偏置（标量形式）
        return nn.Parameter(torch.randn(n_head, 2 * max_rel_pos + 1) * 0.02)

    else:
        raise ValueError(f"不支持的相对位置编码方法: {method}")


def get_relative_position_index(q_len, k_len, max_rel_pos=None, device=None):
    """
    获取相对位置索引矩阵

    参数说明:
        q_len: 查询序列长度
        k_len: 键序列长度
        max_rel_pos: 最大相对位置距离（可选，用于裁剪）
        device: 计算设备

    返回:
        相对位置索引矩阵，形状为(q_len, k_len)，值在[-max_rel_pos, max_rel_pos]范围内

    说明:
        这个函数生成一个矩阵，其中每个元素(i,j)表示位置i和位置j之间的相对距离。
    """
    if device is None:
        device = torch.device('cpu')

    # 创建位置索引矩阵
    q_pos = torch.arange(q_len, device=device).view(-1, 1)
    k_pos = torch.arange(k_len, device=device).view(1, -1)

    # 计算相对位置
    rel_pos = q_pos - k_pos  # (q_len, k_len)

    # 如果指定了最大相对位置，进行裁剪
    if max_rel_pos is not None:
        rel_pos = rel_pos.clamp(-max_rel_pos, max_rel_pos)

    return rel_pos


def add_relative_position_bias(attn_scores, rel_pos_bias, rel_pos_index):
    """
    将相对位置偏置添加到注意力分数中

    参数说明:
        attn_scores: 注意力分数，形状为(batch_size, n_head, q_len, k_len)
        rel_pos_bias: 相对位置偏置，形状为(n_head, 2*max_rel_pos+1)
        rel_pos_index: 相对位置索引矩阵，形状为(q_len, k_len)

    返回:
        添加相对位置偏置后的注意力分数
    """
    batch_size, n_head, q_len, k_len = attn_scores.shape

    # 将相对位置索引转换为非负索引
    max_rel_pos = (rel_pos_bias.shape[1] - 1) // 2
    rel_pos_index_shifted = rel_pos_index + max_rel_pos  # 将范围从[-max, max]映射到[0, 2*max]

    # 从偏置表中获取对应位置的偏置
    # rel_pos_bias: (n_head, 2*max_rel_pos+1)
    # rel_pos_index_shifted: (q_len, k_len)
    bias = rel_pos_bias[:, rel_pos_index_shifted]  # (n_head, q_len, k_len)

    # 扩展维度以匹配批次大小并添加到注意力分数
    bias = bias.unsqueeze(0)  # (1, n_head, q_len, k_len)
    attn_scores_with_bias = attn_scores + bias

    return attn_scores_with_bias


# 使用示例
if __name__ == "__main__":
    # 1. 获取正弦相对位置编码（对标绝对位置编码的get_sinusoid_encoding）
    max_rel_pos = 128
    d_hid = 512
    rel_pos_enc = get_sinusoid_relative_encoding(max_rel_pos, d_hid)
    print(f"正弦相对位置编码形状: {rel_pos_enc.shape}")  # (257, 512)

    # 2. 获取带有注意力头的相对位置编码
    n_head = 8
    rel_pos_enc_multi_head = get_sinusoid_relative_encoding(max_rel_pos, d_hid, n_head)
    print(f"多头正弦相对位置编码形状: {rel_pos_enc_multi_head.shape}")  # (8, 257, 512)

    # 3. 获取相对位置偏置（标量形式，更常用）
    rel_pos_bias = get_relative_position_bias(max_rel_pos, n_head, method='sinusoid')
    print(f"相对位置偏置形状: {rel_pos_bias.shape}")  # (8, 257)

    # 4. 获取可学习的相对位置偏置
    learnable_bias = get_learnable_relative_encoding(max_rel_pos, d_hid, n_head)
    print(f"可学习相对位置编码形状: {learnable_bias.shape}")  # (8, 257) 或 (8, 257, 512)

    # 5. 获取相对位置索引矩阵
    q_len = 64
    k_len = 64
    rel_pos_index = get_relative_position_index(q_len, k_len, max_rel_pos)
    print(f"相对位置索引矩阵形状: {rel_pos_index.shape}")  # (64, 64)

    # 6. 演示如何将相对位置偏置添加到注意力分数中
    batch_size = 4
    attn_scores = torch.randn(batch_size, n_head, q_len, k_len)
    attn_scores_with_bias = add_relative_position_bias(attn_scores, rel_pos_bias, rel_pos_index)
    print(f"添加相对位置偏置后的注意力分数形状: {attn_scores_with_bias.shape}")  # (4, 8, 64, 64)