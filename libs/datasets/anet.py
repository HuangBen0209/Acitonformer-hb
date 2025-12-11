import os
import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset  # 数据集注册装饰器
from .data_utils import truncate_feats  # 特征截断工具函数
from ..utils import remove_duplicate_annotations  # 移除重复标注工具函数

@register_dataset("anet")  # 注册为"anet"数据集
class ActivityNetDataset(Dataset):
    """
    ActivityNet-1.3 数据集类
    用于加载ActivityNet数据集的特征和标注，支持训练/测试模式，处理固定/可变长度特征，
    实现特征下采样/上采样、标注过滤、特征截断等功能
    """
    def __init__(
        self,
        is_training,      # 是否为训练模式
        split,            # 数据集划分（元组/列表，支持多个子集拼接，如('train', 'val')）
        feat_folder,      # 特征文件存储文件夹路径
        json_file,        # 标注信息JSON文件路径
        feat_stride,      # 特征的时间步长（每个特征对应原始视频的帧数）
        num_frames,       # 每个特征覆盖的原始视频帧数
        default_fps,      # 默认帧率（FPS），若为None则从标注文件读取
        downsample_rate,  # 特征下采样率（>1时对特征序列进行降采样）
        max_seq_len,      # 训练时允许的最大特征序列长度
        trunc_thresh,     # 动作片段截断阈值（保留超过该比例的动作片段）
        crop_ratio,       # 随机裁剪比例范围（如(0.9, 1.0)，None表示不裁剪）
        input_dim,        # 输入特征维度
        num_classes,      # 动作类别数量
        file_prefix,      # 特征文件前缀（如有）
        file_ext,         # 特征文件扩展名（如有，如.npy/.pth）
        force_upsampling  # 是否强制将特征上采样至max_seq_len
    ):
        # 文件路径检查
        assert os.path.exists(feat_folder) and os.path.exists(json_file), "特征文件夹或标注文件不存在"
        assert isinstance(split, tuple) or isinstance(split, list), "split必须是元组或列表类型"
        assert crop_ratio == None or len(crop_ratio) == 2, "crop_ratio应为None或长度为2的元组"
        self.feat_folder = feat_folder
        self.use_hdf5 = '.hdf5' in feat_folder  # 判断是否使用HDF5格式存储特征
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''  # 默认无文件前缀
        self.file_ext = file_ext
        self.json_file = json_file

        # ActivityNet使用固定长度特征，确保无额外下采样（特殊处理）
        self.force_upsampling = force_upsampling

        # 数据集划分/训练模式
        self.split = split
        self.is_training = is_training

        # 特征元信息
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None  # 类别名称到类别ID的映射字典
        self.crop_ratio = crop_ratio

        # 加载标注数据库并筛选指定子集
        dict_db, label_dict = self._load_json_db(self.json_file)
        # 验证类别数量（1表示仅检测动作存在性，否则需匹配标注类别数）
        assert (num_classes == 1) or (len(label_dict) == num_classes), "类别数量不匹配"
        self.data_list = dict_db  # 处理后的数据集列表
        self.label_dict = label_dict  # 类别映射字典

        # 数据集特有属性（评估时使用）
        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',  # 数据集名称
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),  # 时间IOU评估阈值（10个梯度）
            'empty_label_ids': []  # 无标注的类别ID（ActivityNet无空类别）
        }

    def get_attributes(self):
        """获取数据集属性（供评估模块使用）"""
        return self.db_attributes

    def _load_json_db(self, json_file):
        """
        加载JSON格式的标注数据库并筛选指定子集
        参数:
            json_file: 标注文件路径
        返回:
            dict_db: 处理后的数据集元组（每个元素为单视频信息）
            label_dict: 类别名称到ID的映射字典
        """
        # 加载标注文件
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']  # 核心标注数据

        # 若类别映射字典未初始化，则从标注中构建
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']  # 构建类别-ID映射

        # 填充数据库（构建完成后不可变）
        dict_db = tuple()
        for key, value in json_db.items():
            # 跳过不在指定划分中的视频
            if value['subset'].lower() not in self.split:
                continue

            # 获取视频帧率
            if self.default_fps is not None:
                fps = self.default_fps  # 使用默认帧率
            elif 'fps' in value:
                fps = value['fps']  # 从标注读取帧率
            else:
                assert False, "未知的视频帧率（FPS），请检查标注文件"
            duration = value['duration']  # 视频总时长（秒）

            # 加载动作标注（如有）
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])  # 移除重复标注
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)  # 动作时间段（开始/结束，秒）
                labels = np.zeros([num_acts, ], dtype=np.int64)  # 动作类别ID
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0  # 仅检测动作存在性时统一标为0
                    else:
                        labels[idx] = label_dict[act['label']]  # 使用类别映射ID
            else:
                segments = None  # 无标注时设为None
                labels = None
            # 将单视频信息加入数据库
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        获取指定索引的样本数据
        直接返回（可能截断的）数据点（保证读取速度），后续DataLoader禁用自动批处理，
        由模型自行决定批处理/预处理方式
        参数:
            idx: 样本索引
        返回:
            data_dict: 包含特征、标注、元信息的字典
        """
        video_item = self.data_list[idx]  # 获取单视频信息

        # 加载特征文件
        if self.use_hdf5:
            # 从HDF5文件读取特征
            with h5py.File(self.feat_folder, 'r') as h5_fid:
                feats = np.asarray(
                    h5_fid[self.file_prefix + video_item['id']][()],
                    dtype=np.float32
                )
        else:
            # 从普通文件（如.npy）读取特征
            filename = os.path.join(self.feat_folder,
                                    self.file_prefix + video_item['id'] + self.file_ext)
            feats = np.load(filename).astype(np.float32)

        # 支持固定长度/可变长度特征处理
        # 情况1: 训练时使用可变长度特征（不强制上采样）
        if self.feat_stride > 0 and (not self.force_upsampling):
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # 仅在此处应用下采样
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]  # 按步长下采样
                feat_stride = self.feat_stride * self.downsample_rate  # 更新时间步长
        # 情况2: 输入为可变长度特征，但训练时强制调整为固定长度
        elif self.feat_stride > 0 and self.force_upsampling:
            # 重新计算时间步长（适配max_seq_len）
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            num_frames = feat_stride  # 特征中心对齐
        # 情况3: 输入为固定长度特征
        else:
            # 处理固定长度特征，重新计算时间步长和帧数
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len, "特征序列长度超过最大限制"
            if self.force_upsampling:
                seq_len = self.max_seq_len  # 重置为最大序列长度
            # 按视频总帧数重新计算时间步长
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            num_frames = feat_stride  # 特征中心对齐

        # 特征偏移量（用于将时间戳映射到特征网格）
        feat_offset = 0.5 * num_frames / feat_stride

        # 维度转换: T x C（时间步x特征维度） -> C x T（特征维度x时间步）
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # 若需要，调整特征长度至max_seq_len
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),  # 添加batch维度（插值要求4D输入: B x C x T）
                size=self.max_seq_len,  # 目标长度
                mode='linear',  # 线性插值
                align_corners=False  # 不强制对齐角落（避免边界失真）
            )
            feats = resize_feats.squeeze(0)  # 移除batch维度

        # 将时间戳（秒）转换为时间特征网格坐标
        # 允许少量负值（后续会裁剪）
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # ActivityNet部分视频存在大量缺失帧，训练时快速修复
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset  # 有效特征长度
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # 跳过超出特征范围的动作
                        continue
                    # 计算动作在特征范围内的占比
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        # 保留占比超过阈值的动作
                        valid_seg_list.append(seg.clamp(max=vid_len))  # 裁剪超出部分
                        valid_label_list.append(label.view(1))  # 转为1维张量（避免维度错误）
                # 重新拼接有效标注
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # 构建数据字典
        data_dict = {
            'video_id'        : video_item['id'],          # 视频ID
            'feats'           : feats,                     # 特征张量 (C x T)
            'segments'        : segments,                  # 动作时间段张量 (N x 2)
            'labels'          : labels,                    # 动作类别张量 (N)
            'fps'             : video_item['fps'],         # 视频帧率
            'duration'        : video_item['duration'],    # 视频时长（秒）
            'feat_stride'     : feat_stride,               # 特征时间步长
            'feat_num_frames' : num_frames                 # 每个特征覆盖的帧数
        }

        # 训练阶段对特征进行截断（测试阶段不截断）
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict