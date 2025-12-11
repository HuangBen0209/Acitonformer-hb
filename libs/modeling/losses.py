import torch
from torch.nn import functional as F

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    RetinaNet 中用于密集检测的损失函数：https://arxiv.org/abs/1708.02002 。
    代码取自
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # 版权所有 (c) Facebook, Inc. 及其附属机构。保留所有权利。

    参数：
        inputs：任意形状的浮点张量。
                每个样本的预测结果。
        targets：与 inputs 形状相同的浮点张量。存储每个元素在 inputs 中的二分类标签
                （负类为 0，正类为 1）。
        alpha：（可选）权重因子，范围在 (0,1) 内，用于平衡正负样本。默认值为 0.25。
        gamma：调节因子 (1 - p_t) 的指数，用于平衡易分类和难分类样本。
        reduction：'none' | 'mean' | 'sum'
                 'none'：不对输出应用任何归约操作。
                 'mean'：将输出取平均值。
                 'sum'：将输出求和。
    返回：
        应用了归约选项的损失张量。
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    通用交并比损失（Generalized Intersection over Union Loss，Hamid Rezatofighi 等人）
    https://arxiv.org/abs/1902.09630
    这是一个假设一维事件使用相同的中心点和不同的偏移量来表示的实现，例如，
    (t1, t2) = (c - o_1, c + o_2)，其中 o_i >= 0
    参考代码来自
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py
    参数：
        input/target_offsets (Tensor)：大小为 (N, 2) 的一维偏移量。
        reduction：'none' | 'mean' | 'sum'
                 'none'：不对输出应用任何归约操作。
                 'mean'：将输出取平均值。
                 'sum'：将输出求和。
        eps (float)：一个小数，用于防止除以零。
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # 检查所有一维事件是否有效
    assert (input_offsets >= 0.0).all(), "预测的偏移量必须是非负的"
    assert (target_offsets >= 0.0).all(), "GT 偏移量必须是非负的"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # 交集关键点
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # 在我们的设置中，giou 简化为 iou，跳过不必要的步骤
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    距离交并比损失（Distance-IoU Loss，Zheng 等人）
    https://arxiv.org/abs/1911.08287

    这是一个假设一维事件使用相同的中心点和不同的偏移量来表示的实现，例如，
    (t1, t2) = (c - o_1, c + o_2)，其中 o_i >= 0

    参考代码来自
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    参数：
        input/target_offsets (Tensor)：大小为 (N, 2) 的一维偏移量。
        reduction：'none' | 'mean' | 'sum'
                 'none'：不对输出应用任何归约操作。
                 'mean'：将输出取平均值。
                 'sum'：将输出求和。
        eps (float)：一个小数，用于防止除以零。
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # 检查所有一维事件是否有效
    assert (input_offsets >= 0.0).all(), "预测的偏移量必须是非负的"
    assert (target_offsets >= 0.0).all(), "GT 偏移量必须是非负的"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # 交集关键点
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # 最小闭包框
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # 中心点之间的偏移量
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss