import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding,get_relative_position_encoding,TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)
import numpy as np


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
    卷积与Transformer结合的骨干网络

    参数说明：
    - n_in: 输入特征的维度，如果是多特征融合，可以是列表/元组
    - n_embd: 经过卷积后的嵌入维度
    - n_head: Transformer中自注意力的头数
    - n_embd_ks: 嵌入网络的卷积核大小
    - max_len: 最大序列长度，用于位置编码
    - arch: 架构配置元组 (卷积层数, stem Transformer层数, branch Transformer层数)
    - mha_win_size: 多头注意力局部窗口大小列表，长度需等于1+branch层数
    - scale_factor: branch部分的降采样率
    - with_ln: 是否在卷积后添加LayerNorm
    - attn_pdrop: 注意力图的dropout概率
    - proj_pdrop: 投影/MLP的dropout概率
    - path_pdrop: drop path的概率
    - lbc_win_size=5,  LBC窗口大小,表示一次看多少帧
    - lbc_fusion_gate=0.2,  LBC融合门控系数
    - use_abs_pe: 是否使用绝对位置编码
    - use_rel_pe: 是否使用相对位置编码
    """

    def __init__(
            self,
            n_in,  # 输入特征维度
            n_embd,  # 嵌入维度（卷积后）
            n_head,  # Transformer自注意力头数
            n_embd_ks,  # 嵌入网络的卷积核大小
            max_len,  # 最大序列长度
            arch=(2, 2, 5),  # (卷积层数, stem Transformer层数, branch Transformer层数)
            mha_win_size=[-1] * 6,  # MHA局部窗口大小
            scale_factor=2,  # branch的降采样率
            with_ln=False,  # 是否在卷积后添加LayerNorm
            attn_pdrop=0.0,  # 注意力dropout概率
            proj_pdrop=0.0,  # 投影dropout概率
            path_pdrop=0.0,  # drop path概率
            lbc_win_size=5, # LBC窗口大小
            lbc_fusion_gate=0.2, # LBC融合门控系数
            use_abs_pe=False,  # 是否使用绝对位置编码
            use_rel_pe=False,  # 是否使用相对位置编码
            use_lbc=False,  # 是否使用边界增强模块
    ):
        super().__init__()
        # 参数验证
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])

        # 保存参数
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.use_lbc = use_lbc

        # 特征投影层：如果输入是多特征，分别投影后拼接
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            # 验证：n_in和n_embd都是列表且长度相同
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)  # 更新为拼接后的总维度
        else:
            self.proj = None
        # 嵌入网络：使用多个卷积层提取特征
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            # 第一层使用n_in，后续层使用n_embd
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
                )
            )
            # 可选的LayerNorm
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())
        # 绝对位置编码：生成并注册为缓冲区（不参与梯度更新）
        if self.use_abs_pe:
            # 生成正弦位置编码，除以sqrt(n_embd)进行缩放
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)
        # ========== 新增相对位置编码调用（和绝对编码结构完全对齐） ==========
        if self.use_rel_pe:
            # 生成相对位置编码，除以sqrt(n_embd)进行缩放（和绝对编码保持一致的缩放逻辑）
            rel_pos_embd = get_relative_position_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("rel_pos_embd", rel_pos_embd, persistent=False)
        # stem网络：使用标准Transformer块（无降采样）
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),  # 步长均为1，不降采样
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    lbc_win_size=lbc_win_size,  # LBC窗口大小
                    lbc_fusion_gate=lbc_fusion_gate, # LBC融合门控系数
                    mha_win_size=self.mha_win_size[0],  # stem使用第一个窗口大小
                    use_rel_pe=self.use_rel_pe,
                    use_lbc=self.use_lbc, #LBC使用
                )
            )
        # 主分支：使用带池化的Transformer块（降采样）
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),  # 降采样
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    lbc_win_size=lbc_win_size,  # LBC窗口大小
                    lbc_fusion_gate=lbc_fusion_gate,  # LBC融合门控系数
                    mha_win_size=self.mha_win_size[1 + idx],  # branch使用后续窗口大小
                    use_rel_pe=self.use_rel_pe,
                    use_lbc=self.use_lbc, #LBC使用
                )
            )

        # 初始化权重
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        """初始化权重：将线性层和卷积层的偏置设为0"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        """
        输入：
        - x: 形状为 [批大小, 特征通道数, 序列长度] 的特征张量
        - mask: 形状为 [批大小, 1, 序列长度] 的布尔掩码，True表示有效位置
        输出：
        - out_feats: 多尺度特征元组，包含stem和branch各层的输出
        - out_masks: 对应的掩码元组
        """
        B, C, T = x.size()
        # 1. 特征投影：如果是多特征输入，分别投影后拼接
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                 for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )
        # 2. 嵌入网络：卷积层 + ReLU + LayerNorm
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
        # 3. 训练时的绝对位置编码：使用预计算的位置编码
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "序列长度超过最大长度限制"
            pe = self.pos_embd
            # 将位置编码加到特征上，并乘以掩码确保无效位置不参与计算
            x = x + pe[:, :, :T] * mask.to(x.dtype)
        # 4. 推理时的绝对位置编码：如果序列过长，重新插值位置编码
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                # 序列超过预设最大长度，需要插值扩展位置编码
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            x = x + pe[:, :, :T] * mask.to(x.dtype)
        # 5. stem Transformer：不降采样，保持分辨率
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
        # 6. 准备输出：stem的输出作为第一个特征图
        out_feats = (x,)
        out_masks = (mask,)
        # 7. 主分支Transformer：逐步降采样，生成多尺度特征
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
    纯卷积骨干网络（无Transformer）

    参数说明：
    - n_in: 输入特征维度
    - n_embd: 嵌入维度
    - n_embd_ks: 卷积核大小
    - arch: 架构配置 (卷积层数, stem卷积层数, branch卷积层数)
    - scale_factor: branch的降采样率
    - with_ln: 是否使用LayerNorm
    """

    def __init__(
            self,
            n_in,  # 输入特征维度
            n_embd,  # 嵌入维度
            n_embd_ks,  # 嵌入网络的卷积核大小
            arch=(2, 2, 5),  # (卷积层数, stem卷积层数, branch卷积层数)
            scale_factor=2,  # branch的降采样率
            with_ln=False,  # 是否使用LayerNorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # 特征投影层（与ConvTransformerBackbone相同）
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # 嵌入网络：多层卷积
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
                )
            )
            # 可选的LayerNorm
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem网络：标准卷积块（无降采样）
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # 主分支：带降采样的卷积块
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # 初始化权重
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        """初始化权重：偏置置零"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        """
        前向传播（结构类似ConvTransformerBackbone，但全为卷积操作）
        """
        B, C, T = x.size()

        # 特征投影
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                 for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        # 嵌入网络
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem卷积
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # 准备输出
        out_feats = (x,)
        out_masks = (mask,)

        # 主分支（降采样）
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks