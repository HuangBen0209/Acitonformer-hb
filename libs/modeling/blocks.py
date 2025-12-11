import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    带掩码的1D卷积层

    这是专门为处理变长序列设计的卷积层，确保padding部分不参与计算。
    只支持内核大小为奇数且padding = kernel_size//2的情况，这是时序模型的标准配置。

    参数说明:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（必须是奇数）
        stride: 步幅
        padding: 填充大小（代码中会强制设为kernel_size//2）
        dilation: 空洞率
        groups: 分组卷积
        bias: 是否使用偏置
        padding_mode: 填充模式
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # 要求内核大小必须是奇数，并且填充必须是内核大小的一半
        # 这样设计可以确保时序对齐，特别是在因果卷积或掩码卷积中
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # 如果使用偏置，初始化为0，这是一种常见的做法以避免初始偏差过大
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (B, C, T)，其中B是批次大小，C是通道数，T是序列长度
            mask: 掩码张量，形状为 (B, 1, T)，bool类型，True表示该位置有效（不是padding）

        返回:
            out_conv: 卷积输出，形状为 (B, out_channels, T')
            out_mask: 更新后的掩码，形状为 (B, 1, T')
        """
        B, C, T = x.size()
        # 输入序列长度必须能被步幅整除，这是许多下采样操作的前提条件
        assert T % self.stride == 0

        # 执行标准卷积操作
        out_conv = self.conv(x)

        # 对掩码进行相应的下采样/对齐
        if self.stride > 1:
            # 当步幅大于1时（下采样），使用最近邻插值来保持掩码的0/1特性
            # 使用最近邻插值可以避免产生非整数的掩码值
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # 当步幅为1时，直接使用原始掩码
            out_mask = mask.to(x.dtype)

        # 将无效位置（padding）的输出置为0，同时防止梯度回传到掩码
        # 使用detach()确保掩码不参与梯度计算，只作为掩码使用
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()  # 将掩码转换回bool类型
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    支持(B, C, T)形状输入的层归一化

    传统的LayerNorm是对最后一个维度进行归一化，而这里我们设计为对通道维度(C)进行归一化。
    这种设计在时序建模中很常见，特别是当我们将通道视为特征维度时。

    参数说明:
        num_channels: 通道数
        eps: 防止除以零的小常数
        affine: 是否使用可学习的缩放和偏置参数
        device: 设备类型
        dtype: 数据类型
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            # 创建可学习的缩放和偏置参数，形状为(1, C, 1)，便于广播到整个批次和时间维度
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # 验证输入形状
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # 沿着通道维度(C)进行归一化
        # 计算每个时间步上的通道均值
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # 应用可学习的缩放和偏置（如果启用）
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


def get_sinusoid_encoding(n_position, d_hid):
    """
    正弦位置编码

    这是Transformer原版的位置编码方法。通过正弦和余弦函数生成位置编码，
    使得模型能够感知序列中元素的位置信息。

    参数说明:
        n_position: 位置数量（序列最大长度）
        d_hid: 编码维度（通常等于特征维度）

    返回:
        位置编码张量，形状为(1, d_hid, n_position)，便于直接与特征相加
    """

    def get_position_angle_vec(position):
        # 计算每个维度的角度，基于10000的幂次
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # 生成正弦/余弦位置编码表
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度使用正弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度使用余弦函数

    # 转换为PyTorch张量并调整形状为(1, d_hid, n_position)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class MaskedMHA(nn.Module):
    """
    带掩码的多头自注意力机制

    这是基础的全局注意力实现，模仿minGPT的实现方式，但增加了掩码支持，
    确保padding位置不参与注意力计算。

    参数说明:
        n_embd: 输入特征维度（embedding维度）
        n_head: 注意力头的数量
        attn_pdrop: 注意力dropout率
        proj_pdrop: 投影层dropout率
    """

    def __init__(
            self,
            n_embd,  # 输入特征维度
            n_head,  # 注意力头数量
            attn_pdrop=0.0,  # 注意力dropout率
            proj_pdrop=0.0  # 投影层dropout率
    ):
        super().__init__()
        assert n_embd % n_head == 0  # 确保特征维度能被头数整除
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head  # 每个头的维度
        self.scale = 1.0 / math.sqrt(self.n_channels)  # 缩放因子，用于稳定softmax

        # 使用1x1卷积实现Q、K、V投影
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # 正则化
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # 输出投影层
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        """
        前向传播
        参数:
            x: 输入张量，形状为(B, C, T)
            mask: 掩码张量，形状为(B, 1, T)，bool类型
        返回:
            out: 注意力输出，形状为(B, C, T)
            mask: 与输入相同的掩码
        """
        B, C, T = x.size()

        # 计算Q、K、V
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # 重塑为多头形式：(B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # 计算注意力分数：(B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)

        # 应用掩码：将无效位置的注意力分数设为负无穷
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))

        # softmax归一化
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 注意力加权求和
        out = att @ (v * mask[:, :, :, None].to(v.dtype))

        # 重塑回原始形状
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # 输出投影和dropout
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    带深度可分离卷积的多头注意力

    在标准的MHA基础上，为Q、K、V添加了深度可分离卷积（depthwise convolution）。
    这些额外的卷积操作有三个作用：
    1. 编码局部时序信息（替代部分位置编码）
    2. 实现下采样（当stride>1时）
    3. 使Q和KV可以有不同大小的感受野

    这是许多SOTA时序动作定位模型（如ActionFormer、TriDet）中常用的"Conv-Attention"。

    参数说明:
        n_embd: 特征维度
        n_head: 注意力头数量
        n_qx_stride: 查询(Q)和输入(x)的下采样步幅
        n_kv_stride: 键(K)和值(V)的下采样步幅
        attn_pdrop: 注意力dropout率
        proj_pdrop: 投影层dropout率
    """

    def __init__(
            self,
            n_embd,  # 输出特征维度
            n_head,  # 注意力头数量
            n_qx_stride=1,  # 查询和输入的下采样步幅
            n_kv_stride=1,  # 键和值的下采样步幅
            attn_pdrop=0.0,  # 注意力dropout率
            proj_pdrop=0.0,  # 投影层dropout率
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # 卷积/池化操作的步幅必须是1或偶数
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # 查询的深度可分离卷积
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # 键和值的深度可分离卷积
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)

        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # 标准的Q、K、V投影（1x1卷积）
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # 正则化
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # 输出投影
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        """
        前向传播

        步骤:
        1. 使用深度可分离卷积处理Q、K、V
        2. 应用层归一化
        3. 使用1x1卷积进行投影
        4. 计算注意力
        5. 输出投影

        注意: 下采样后的特征会与步幅s+1的时间步对齐，这样可以方便地插值对应的位置编码。
        """
        B, C, T = x.size()

        # 步骤1: 深度卷积 -> (B, nh*hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # 键和值卷积 -> (B, nh*hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # 步骤2: 投影
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # 步骤3: 重塑为多头形式
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # 步骤4: 计算注意力
        att = (q * self.scale) @ k.transpose(-2, -1)
        # 防止查询关注无效标记
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 注意力加权求和
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))

        # 重塑回原始形状
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # 步骤5: 输出投影
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class LocalMaskedMHCA(nn.Module):
    """
    局部窗口多头卷积注意力（Longformer风格）

    专门为超长视频序列（几千帧）设计，将计算复杂度从O(T²)降低到O(T×window_size)。
    这是通过滑动窗口（sliding window）注意力实现的，每个位置只关注其附近窗口内的其他位置。
    实现参考自Longformer论文和HuggingFace实现。
    参数说明:
        n_embd: 特征维度
        n_head: 注意力头数量
        window_size: 局部窗口大小（必须是奇数，常见65、129）
        n_qx_stride: 查询和输入的下采样步幅
        n_kv_stride: 键和值的下采样步幅
        attn_pdrop: 注意力dropout率
        proj_pdrop: 投影层dropout率
        use_rel_pe: 是否使用相对位置编码
    """
    def __init__(
            self,
            n_embd,  # 输出特征维度
            n_head,  # 注意力头数量
            window_size,  # 局部窗口大小
            n_qx_stride=1,  # 查询和输入的下采样步幅
            n_kv_stride=1,  # 键和值的下采样步幅
            attn_pdrop=0.0,  # 注意力dropout率
            proj_pdrop=0.0,  # 投影层dropout率
            use_rel_pe=False  # 是否使用相对位置编码
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2  # 窗口重叠部分，通常是窗口大小的一半
        # 必须使用奇数窗口大小，并且头数至少为1
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # 卷积/池化操作的步幅必须是1或偶数
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # 查询的深度可分离卷积
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # 键和值的深度可分离卷积
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # 标准的Q、K、V投影（1x1卷积）
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # 正则化
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # 输出投影
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # 相对位置编码
        if self.use_rel_pe:
            # 相对位置偏置，只在局部窗口内有效
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd) ** 0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """
        将序列分割成重叠的chunk

        每个chunk的大小为2w，重叠部分为w。
        这是实现滑动窗口注意力的关键步骤。

        参数:
            x: 输入张量，形状为(B*nh, T, hs)
            window_overlap: 窗口重叠大小

        返回:
            分割后的张量，形状为(B*nh, #chunks, 2w, hs)
        """
        # 将序列分成不重叠的chunk，每个chunk大小为2w
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # 使用as_strided使chunk重叠，重叠大小为window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1  # 由于重叠，chunk数量增加
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2  # 在序列维度上的步幅减半

        # 返回重叠的chunk：B*nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """填充行，然后翻转行和列"""
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        """屏蔽无效位置（超出窗口范围的位置）"""
        # 创建左下三角掩码（用于序列开始部分）
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        # 创建右上三角掩码（用于序列结束部分）
        ending_mask = beginning_mask.flip(dims=(1, 3))

        # 应用开始部分掩码
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))

        # 应用结束部分掩码
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        将每行向右移动一步，将列转换为对角线

        这是滑动窗口注意力计算中的关键步骤，用于将chunked的注意力分数
        转换为对角线形式的矩阵。
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # 在最后一个维度上填充，为对角线化做准备
        x = nn.functional.pad(x, (0, window_overlap + 1))
        x = x.view(total_num_heads, num_chunks, -1)
        x = x[:, :, :-window_overlap]  # 移除多余的填充
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]  # 调整形状
        return x

    def _sliding_chunks_query_key_matmul(self, query, key, num_heads, window_overlap):
        """
        使用滑动窗口注意力模式计算查询和键的矩阵乘法

        这个实现将输入分割成大小为2w的重叠chunk，重叠部分为w。
        这是Longformer论文中的核心算法。
        """
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        # 序列长度必须能被2w整除
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # 将查询和键分割成chunk
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # 矩阵乘法计算chunk之间的注意力分数
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # 将对角线转换为列
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # 为整体注意力矩阵分配空间
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # 将chunked注意力分数复制到整体注意力矩阵中
        # - 复制主对角线和上三角部分
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - 复制下三角部分
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap:
        ]

        # 分离批次和头维度
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        # 屏蔽无效位置
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, num_heads, window_overlap):
        """
        与_sliding_chunks_query_key_matmul类似，但用于注意力概率和值的乘法

        返回的张量形状与attn_probs相同。
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1

        # 重塑注意力概率
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # 在序列开始和结束处填充值
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # 将填充后的值分割成chunk
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        # 对角线化注意力概率
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        # 计算上下文向量
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        """
        前向传播

        步骤:
        1. 深度卷积
        2. 查询、键、值变换和重塑
        3. 计算带相对位置编码和掩码的局部自注意力
        4. 计算注意力值乘积和输出投影
        """
        B, C, T = x.size()

        # 步骤1: 深度卷积
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # 步骤2: 查询、键、值变换和重塑
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # 重塑为(B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # 视图为(B*nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # 步骤3: 计算带相对位置编码和掩码的局部自注意力
        q *= self.scale
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # 相对位置编码
        if self.use_rel_pe:
            att += self.rel_pe

        # 应用键/值掩码
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)

        # 计算对角线掩码（针对每个局部窗口）
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # softmax归一化
        att = nn.functional.softmax(att, dim=-1)
        # 如果所有位置都被掩码，softmax可能产生NaN，用0替换
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # 步骤4: 计算注意力值乘积和输出投影
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    一个简单的Transformer块（Post-LayerNorm版本）

    支持以下特性:
    - 全局/卷积/局部三种注意力机制
    - Q/X和K/V不同的下采样率（适合金字塔结构）
    - DropPath（随机深度）
    - 可插拔的位置编码

    参数说明:
        n_embd: 输入特征维度
        n_head: 注意力头数量
        n_ds_strides: Q&X和K&V的下采样步幅
        n_out: 输出维度，如果为None则设置为输入维度
        n_hidden: MLP隐藏层维度
        act_layer: MLP中使用的激活函数，默认为GELU
        attn_pdrop: 注意力dropout率
        proj_pdrop: 投影层dropout率
        path_pdrop: DropPath概率
        mha_win_size: >0时使用窗口注意力，-1表示使用全局注意力
        use_rel_pe: 是否在注意力中添加相对位置编码
    """

    def __init__(
            self,
            n_embd,  # 输入特征维度
            n_head,  # 注意力头数量
            n_ds_strides=(1, 1),  # Q&X和K&V的下采样步幅
            n_out=None,  # 输出维度
            n_hidden=None,  # MLP隐藏层维度
            act_layer=nn.GELU,  # MLP激活函数
            attn_pdrop=0.0,  # 注意力dropout率
            proj_pdrop=0.0,  # 投影层dropout率
            path_pdrop=0.0,  # DropPath概率
            mha_win_size=-1,  # >0时使用窗口注意力
            use_rel_pe=False  # 是否使用相对位置编码
    ):
        super().__init__()
        assert len(n_ds_strides) == 2

        # 层归一化（适用于B, C, T顺序）
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # 选择注意力模块
        if mha_win_size > 1:
            # 使用局部窗口注意力
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # 仅对局部注意力有效
            )
        else:
            # 使用标准卷积注意力
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # 跳跃连接的下采样处理
        if n_ds_strides[0] > 1:
            # 当Q&X有下采样时，需要对跳跃连接也进行下采样
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # 两层MLP（使用1x1卷积实现）
        if n_hidden is None:
            n_hidden = 4 * n_embd  # 默认隐藏层维度是输入维度的4倍
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),  # 扩展维度
            act_layer(),  # 非线性激活
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),  # 压缩维度
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # DropPath（随机深度）
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        """
        前向传播

        采用Pre-LN结构：https://arxiv.org/pdf/2002.04745.pdf
        Pre-LN比传统的Post-LN更稳定，更容易训练深层的Transformer。
        """
        # 注意力部分
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)

        # 残差连接 + DropPath
        # 注意：跳跃连接需要与注意力输出进行掩码对齐
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)

        # FFN部分
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)

        # 可选：添加位置编码到输出
        if pos_embd is not None:
            out += pos_embd * out_mask_float

        return out, out_mask


class ConvBlock(nn.Module):
    """
    简单的卷积块（类似ResNet中的基础块）

    这个块包含两个卷积层，带有残差连接，可以与Transformer块混合使用，
    构建Conv-Transformer混合骨干网络。

    参数说明:
        n_embd: 输入特征维度
        kernel_size: 卷积核大小
        n_ds_stride: 当前层的下采样步幅
        expansion_factor: 特征维度的扩展因子
        n_out: 输出维度
        act_layer: 卷积后使用的激活函数，默认为ReLU
    """

    def __init__(
            self,
            n_embd,  # 输入特征维度
            kernel_size=3,  # 卷积核大小
            n_ds_stride=1,  # 下采样步幅
            expansion_factor=2,  # 扩展因子
            n_out=None,  # 输出维度
            act_layer=nn.ReLU,  # 激活函数
    ):
        super().__init__()
        # 必须使用奇数大小的卷积核
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

        # 第一个卷积：扩展维度
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)

        # 第二个卷积：压缩回原始维度
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # 下采样跳跃连接
        if n_ds_stride > 1:
            # 使用1x1卷积进行下采样（与ResNet相同）
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        """
        前向传播

        采用标准的残差连接结构：
        输入 -> 卷积1 -> 激活 -> 卷积2 -> + 跳跃连接 -> 激活
        """
        identity = x

        # 第一个卷积层
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)

        # 第二个卷积层
        out, out_mask = self.conv2(out, out_mask)

        # 下采样跳跃连接（如果需要）
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # 残差连接
        out += identity
        out = self.act(out)

        return out, out_mask


class Scale(nn.Module):
    """
    通过学习一个可学习的常数来缩放回归输出范围

    这在时序动作定位任务中很常见，用于调整边界框回归的输出尺度。

    参数说明:
        init_value: 标量的初始值
    """

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        输入 -> 缩放 * 输入
        """
        return x * self.scale


def drop_path(x, drop_prob=0.0, training=False):
    """
    随机深度（Stochastic Depth）实现

    这是Dropout的一种变体，不是丢弃神经元，而是丢弃整个网络层。
    在训练时随机丢弃一些层，在测试时使用所有层。

    参数说明:
        x: 输入张量
        drop_prob: 丢弃概率
        training: 是否处于训练模式

    返回:
        经过随机深度处理后的张量
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # 创建与输入形状匹配的掩码
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # 二值化
    output = x.div(keep_prob) * mask  # 缩放以保持期望值
    return output


class DropPath(nn.Module):
    """
    DropPath（随机深度）模块

    在残差块的主路径中应用随机深度。
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    带有每通道缩放系数的DropPath

    这是DropPath的一个变体，为每个通道添加了一个可学习的缩放系数，
    并且初始化为一个很小的值（如1e-4），有助于稳定深度网络的训练。

    参考论文：https://arxiv.org/pdf/2103.17239.pdf

    参数说明:
        num_dim: 通道维度大小
        drop_prob: 丢弃概率
        init_scale_value: 缩放系数的初始值
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        # 创建每通道的缩放系数，初始化为很小的值
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        # 先缩放，再应用DropPath
        return drop_path(self.scale * x, self.drop_prob, self.training)