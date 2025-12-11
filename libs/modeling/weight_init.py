# from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
import torch
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    无梯度的截断正态分布初始化函数。
    基于 https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf 的方法。
    """
    def norm_cdf(x):
        """
        计算标准正态分布的累积分布函数（CDF）。
        """
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 如果均值超出截断范围 [a, b] 的两倍标准差，发出警告
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("均值超出截断范围 [a, b] 的两倍标准差。"
                      "生成的值可能不正确。",
                      stacklevel=2)

    with torch.no_grad():
        # 使用截断均匀分布生成值，然后通过逆 CDF 转换为截断正态分布
        # 计算截断范围的 CDF 值
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 使用均匀分布填充张量，范围为 [2l-1, 2u-1]
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # 使用逆误差函数（erfinv）将均匀分布转换为标准正态分布
        tensor.erfinv_()

        # 转换为指定的均值和标准差
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # 通过截断确保值在指定范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    使用截断正态分布填充输入张量。
    值从正态分布 :math:`\mathcal{N}(\text{mean}, \text{std}^2)` 中抽取，
    超出范围 :math:`[a, b]` 的值将重新抽取，直到它们在指定范围内。
    生成随机值的方法在 :math:`a \leq \text{mean} \leq b` 时效果最佳。
    参数：
        tensor: 一个 n 维的 `torch.Tensor`
        mean: 正态分布的均值
        std: 正态分布的标准差
        a: 最小截断值
        b: 最大截断值
    示例：
        >>> w = torch.empty(3, 5)
        >>> trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)