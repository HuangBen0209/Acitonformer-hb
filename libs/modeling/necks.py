import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm

@register_neck("fpn")
class FPN1D(nn.Module):
    """
    特征金字塔网络（Feature Pyramid Network，FPN）
    """
    def __init__(
        self,
        in_channels,      # 输入特征通道数，len(in_channels) = 特征层级数量
        out_channel,      # 输出特征通道数
        scale_factor=2.0, # 相邻两个 FPN 层级之间的下采样率
        start_level=0,    # 开始的 FPN 层级
        end_level=-1,     # 结束的 FPN 层级
        with_ln=True,     # 是否在最后应用层归一化（LayerNorm）
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels  # 输入特征通道数
        self.out_channel = out_channel  # 输出特征通道数
        self.scale_factor = scale_factor  # 下采样率

        self.start_level = start_level  # 开始层级
        if end_level == -1:
            self.end_level = len(in_channels)  # 如果 end_level 为 -1，则默认为输入特征的总层级数
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)  # 确保结束层级不超过输入特征的总层级数
        assert (self.start_level >= 0) and (self.start_level < self.end_level)  # 确保开始层级有效

        self.lateral_convs = nn.ModuleList()  # 横向连接卷积层 1x1的卷积是吗？--是的
        self.fpn_convs = nn.ModuleList()  # FPN 卷积层
        self.fpn_norms = nn.ModuleList()  # FPN 归一化层
        for i in range(self.start_level, self.end_level):
            # 横向连接卷积，将输入特征通道数转换为输出通道数
            l_conv = MaskedConv1D( #横向连接卷积层 1x1的卷积
                in_channels[i], out_channel, 1, bias=(not with_ln) #归一化也有偏置，所以这里不加偏置
            )
            # 使用深度可分离卷积提高效率
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel  #这里group是通道数，为每个通道分配一组卷积核
            )
            # 层归一化，适用于 (B, C, T) 的数据格式
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs 必须是列表或元组
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # 构建横向连接特征
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # 构建自顶向下的路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.scale_factor, mode='nearest'
            )

        # FPN 卷积 / 归一化 -> 输出特征
        # mask 保持不变
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(used_backbone_levels):
            x, new_mask = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )
            new_fpn_masks += (new_mask, )

        return fpn_feats, new_fpn_masks


@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # 输入特征通道数，len(in_channels) = 特征层级数量
        out_channel,      # 输出特征通道数
        scale_factor=2.0, # 相邻两个 FPN 层级之间的下采样率
        start_level=0,    # 开始的 FPN 层级
        end_level=-1,     # 结束的 FPN 层级
        with_ln=True,     # 是否在最后应用层归一化（LayerNorm）
    ):
        super().__init__()

        self.in_channels = in_channels  # 输入特征通道数
        self.out_channel = out_channel  # 输出特征通道数
        self.scale_factor = scale_factor  # 下采样率

        self.start_level = start_level  # 开始层级
        if end_level == -1:
            self.end_level = len(in_channels)  # 如果 end_level 为 -1，则默认为输入特征的总层级数
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)  # 确保结束层级不超过输入特征的总层级数
        assert (self.start_level >= 0) and (self.start_level < self.end_level)  # 确保开始层级有效

        self.fpn_norms = nn.ModuleList()  # FPN 归一化层
        for i in range(self.start_level, self.end_level):
            # 检查特征维度是否一致
            assert self.in_channels[i] == self.out_channel
            # 层归一化，适用于 (B, C, T) 的数据格式
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs 必须是列表或元组
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # 应用归一化，mask 保持不变
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks