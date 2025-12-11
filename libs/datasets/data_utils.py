import os
import copy
import random
import numpy as np
import torch


def trivial_batch_collator(batch):
    """
    一个什么都不做的批量收集器
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
    为每个工作进程重置随机种子
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    截断字典项中的特征和时间戳

    data_dict = {
        'video_id': str,  # 视频 ID
        'feats': Tensor C x T,  # 特征张量，C 表示通道数，T 表示时间序列长度
        'segments': Tensor N x 2,  # 段信息，N 表示段的数量，每段用两个时间点表示（在特征网格中）
        'labels': Tensor N,  # 每个段的标签
        'fps': float,  # 帧率
        'feat_stride': int,  # 特征步幅
        'feat_num_frames': int  # 特征帧数
    }
    """
    # 获取元数据信息
    feat_len = data_dict['feats'].shape[1]  # 特征的时间序列长度
    num_segs = data_dict['segments'].shape[0]  # 段的数量

    # 如果序列长度小于等于最大序列长度
    if feat_len <= max_seq_len:
        # 如果没有指定裁剪比例，则直接返回原始数据
        if crop_ratio is None:
            return data_dict
        # 随机裁剪序列，通过设置 max_seq_len 为 [l, r] 范围内的一个值
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # 特殊情况：如果特征长度等于最大序列长度，则直接返回原始数据
            if feat_len == max_seq_len:
                return data_dict

    # 否则，深拷贝字典
    data_dict = copy.deepcopy(data_dict)

    # 尝试多次，直到找到一个有效的截断，至少包含一个动作
    for _ in range(max_num_trials):

        # 随机采样视频特征的一个截断
        st = random.randint(0, feat_len - max_seq_len)  # 截断的起始点
        ed = st + max_seq_len  # 截断的结束点
        window = torch.as_tensor([st, ed], dtype=torch.float32)  # 截断窗口

        # 计算采样窗口与所有段的交集
        window = window[None].repeat(num_segs, 1)  # 扩展窗口维度以匹配段数量
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])  # 交集的左边界
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])  # 交集的右边界
        inter = (right - left).clamp(min=0)  # 计算交集长度
        area_segs = torch.abs(data_dict['segments'][:, 1] - data_dict['segments'][:, 0])  # 段的长度
        inter_ratio = inter / area_segs  # 交集比例

        # 仅选择交集比例大于阈值的段
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # 至少包含一个动作，并且不截断任何动作
            seg_trunc_idx = torch.logical_and((inter_ratio > 0.0), (inter_ratio < 1.0))
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # 至少包含一个动作
            if seg_idx.sum().item() > 0:
                break
        else:
            # 没有任何约束
            break

    # 截断特征：C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # 截断段信息：N x 2（在特征网格中）
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # 由于截断，调整时间戳
    data_dict['segments'] = data_dict['segments'] - st
    # 截断标签：N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict