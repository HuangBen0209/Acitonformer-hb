import torch
from torch import nn
from torch.nn import functional as F
from .models import register_generator
class BufferList(nn.Module):
    """
    类似于 nn.ParameterList，但用于缓冲区

    代码取自 https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # 使用非持久化缓冲区，这样值就不会保存在检查点中
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


@register_generator('point')
class PointGenerator(nn.Module):
    """
    用于生成时间“点”的生成器

    max_seq_len 可以远大于实际的序列长度
    """
    def __init__(
        self,
        max_seq_len,        # 生成器将缓冲的最大序列长度
        fpn_strides,        # FPN（特征金字塔网络）各层的步幅
        regression_range,   # 特征网格上的回归范围
        use_offset=False    # 是否将点对齐到网格中心
    ):
        super().__init__()
        # 检查 FPN 层的数量以及长度是否可被整除
        fpn_levels = len(fpn_strides)
        assert len(regression_range) == fpn_levels

        # 保存参数
        self.max_seq_len = max_seq_len
        self.fpn_levels = fpn_levels
        self.fpn_strides = fpn_strides
        self.regression_range = regression_range
        self.use_offset = use_offset

        # 生成所有点并将它们缓冲到列表中
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # 遍历每个金字塔层的所有点
        for l, stride in enumerate(self.fpn_strides):
            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            # 如果需要，添加偏移量（在当前模型中未使用）
            if self.use_offset:
                points += 0.5 * stride
            # 使用额外的回归范围 / 步幅来填充时间戳
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # 大小：T x 4（时间戳，回归范围，步幅）
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))

        return BufferList(points_list)

    def forward(self, feats):
        # feats 是一个 PyTorch 张量列表
        assert len(feats) == self.fpn_levels
        pts_list = []
        feat_lens = [feat.shape[-1] for feat in feats]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            assert feat_len <= buffer_pts.shape[0], "点生成器达到最大缓冲长度"
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list