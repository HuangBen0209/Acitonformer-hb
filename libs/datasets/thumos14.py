import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    """
    THUMOS14 数据集类，用于加载和处理 THUMOS14 数据集。
    """
    def __init__(
        self,
        is_training,     # 是否处于训练模式
        split,           # 数据集划分，可以是一个元组或列表，允许合并子集
        feat_folder,     # 特征文件夹路径
        json_file,       # 注释文件的 JSON 路径
        feat_stride,     # 特征的时间步长
        num_frames,      # 每个特征的时间帧数
        default_fps,     # 默认帧率
        downsample_rate, # 特征的下采样率
        max_seq_len,     # 训练时的最大序列长度
        trunc_thresh,    # 截断动作段的阈值
        crop_ratio,      # 随机裁剪的比例范围，例如 (0.9, 1.0)
        input_dim,       # 输入特征的维度
        num_classes,     # 动作类别的数量
        file_prefix,     # 特征文件的前缀（如果有）
        file_ext,        # 特征文件的扩展名（如果有）
        force_upsampling # 是否强制上采样到最大序列长度
    ):
        # 文件路径
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # 数据集划分 / 训练模式
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
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # 加载数据库并选择子集
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # 数据集特定属性
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # 我们将屏蔽悬崖跳水类别
            'empty_label_ids': [],
        }
    def get_attributes(self):
        """
        获取数据集的属性。
        """
        return self.db_attributes

    def _load_json_db(self, json_file):
        """
        加载 JSON 数据库并选择子集。
        """
        # 加载数据库并选择子集
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # 如果没有标签字典
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # 填充数据库（之后不可变）
        dict_db = tuple()
        for key, value in json_db.items():
            # 如果视频不在指定的划分中，则跳过
            if value['subset'].lower() not in self.split:
                continue
            # 或者没有特征文件
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # 获取帧率（如果可用）
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "未知视频帧率。"

            # 获取视频时长（如果可用）
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # 获取注释（如果可用）
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # 一个有趣的事实：THUMOS 中悬崖跳水（4）是跳水（7）的一个子集
                # 我们的代码现在可以处理这个特殊情况
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        直接返回一个（截断的）数据点（因此非常快！）
        在后续的数据加载器中将禁用自动批处理，
        相反，模型需要决定如何批处理/预处理数据。
        """
        video_item = self.data_list[idx]

        # 加载特征
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # 处理下采样（= 增加特征步长）
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # 将时间戳（以秒为单位）转换为时间特征网格
        # 在这里允许有小的负值
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # 返回一个数据字典
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # 在训练时截断特征
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict