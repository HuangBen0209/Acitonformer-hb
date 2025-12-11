"""
EPIC-Kitchens-100 数据集加载模块
===============================
功能说明：
1. 实现适配EPIC-Kitchens-100数据集的PyTorch Dataset类，支持训练/测试模式下的数据集加载
2. 核心特性：
   - 处理EPIC-Kitchens特有的“空类别”问题（部分动作类别无标注样本）
   - 适配该数据集的标注格式（第一人称厨房动作，包含verb/noun组合类别）
   - 自动完成特征下采样、时间戳到特征网格的映射、训练阶段特征截断
   - 定义该数据集专属的评估参数（时间IOU阈值范围：0.1~0.5）
3. 适配场景：
   - EPIC-Kitchens-100数据集的动作检测/时序定位任务
   - 支持可变长度特征输入，兼容训练/测试不同的数据处理策略
"""
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset  # 数据集注册装饰器
from .data_utils import truncate_feats  # 特征截断工具函数（训练时截断超长序列）


@register_dataset("epic")  # 将该数据集类注册为"epic"，可通过数据集名称动态调用
class EpicKitchensDataset(Dataset):
    """
    EPIC-Kitchens-100数据集类（继承自PyTorch Dataset）
    负责加载EPIC-Kitchens-100数据集的特征文件和标注信息，完成数据预处理，核心特性：
    1. 识别并记录“空类别”（无标注样本的类别ID），供模型训练时跳过这些类别
    2. 加载.npz格式的特征文件（需包含'feats'键存储特征数组）
    3. 训练阶段自动截断超长特征序列，支持随机裁剪增强
    4. 适配EPIC-Kitchens的标注格式，完成时间戳到特征网格的映射
    """

    def __init__(
            self,
            is_training,  # 是否为训练模式（bool）：训练模式执行特征截断/随机裁剪，测试模式保留完整数据
            split,  # 数据集划分（tuple/list）：指定加载的子集，如('training',)、('validation',)
            feat_folder,  # 特征文件存储文件夹路径（str）：EPIC-Kitchens特征通常为.npz格式
            json_file,  # 标注信息JSON文件路径（str）：包含视频元信息、动作标注、类别ID等
            feat_stride,  # 特征时间步长（int）：每个特征向量对应的原始视频帧数（如16表示每16帧提取1个特征）
            num_frames,  # 每个特征覆盖的原始视频帧数（int）：与feat_stride共同决定特征时间分辨率
            default_fps,  # 默认帧率（float/None）：若为None则从标注文件读取，否则使用指定值（如25.0）
            downsample_rate,  # 特征下采样率（int）：>1时对特征序列降采样（如2表示每隔1个特征取1个）
            max_seq_len,  # 训练时最大特征序列长度（int）：超过该长度的特征会被截断（适配显存）
            trunc_thresh,  # 动作片段截断阈值（float）：保留动作片段在特征范围内的比例≥该值（如0.5表示保留超50%的动作）
            crop_ratio,  # 随机裁剪比例范围（tuple/None）：如(0.9, 1.0)表示随机裁剪90%-100%的特征序列，None表示不裁剪
            input_dim,  # 输入特征维度（int）：单个特征向量的维度（如2048/4096，需与加载的特征匹配）
            num_classes,  # 动作类别总数（int）：需包含空类别（EPIC-Kitchens部分类别无样本）
            file_prefix,  # 特征文件前缀（str/None）：特征文件名的统一前缀（如有，如"epic_"）
            file_ext,  # 特征文件扩展名（str/None）：EPIC-Kitchens常用.npz
            force_upsampling  # 强制上采样标志（bool）：EPIC-Kitchens中未启用，仅为接口兼容保留
    ):
        # 路径有效性校验
        assert os.path.exists(feat_folder) and os.path.exists(json_file), "特征文件夹或标注文件不存在"
        assert isinstance(split, tuple) or isinstance(split, list), "split必须是元组/列表类型"
        assert crop_ratio == None or len(crop_ratio) == 2, "crop_ratio应为None或长度为2的元组"

        # 路径相关属性
        self.feat_folder = feat_folder  # 特征文件夹路径
        self.file_prefix = file_prefix if file_prefix is not None else ''  # 特征文件前缀
        self.file_ext = file_ext  # 特征文件扩展名（如.npz）
        self.json_file = json_file  # 标注文件路径

        # 数据集模式/划分
        self.split = split  # 数据集划分（如('training',)）
        self.is_training = is_training  # 训练/测试模式标志

        # 特征元信息（核心参数）
        self.feat_stride = feat_stride  # 原始特征时间步长
        self.num_frames = num_frames  # 每个特征覆盖的帧数
        self.input_dim = input_dim  # 输入特征维度
        self.default_fps = default_fps  # 默认帧率
        self.downsample_rate = downsample_rate  # 特征下采样率
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.trunc_thresh = trunc_thresh  # 动作截断阈值
        self.num_classes = num_classes  # 总类别数（含空类别）
        self.label_dict = None  # 类别名称→ID映射字典（延迟初始化）
        self.crop_ratio = crop_ratio  # 随机裁剪比例

        # 加载标注数据库并筛选指定子集
        dict_db, label_dict = self._load_json_db(self.json_file)
        # EPIC-Kitchens存在空类别，标注中的类别数≤总类别数
        assert len(label_dict) <= num_classes, "标注中的类别数超过指定的总类别数"
        self.data_list = dict_db  # 处理后的数据集列表（每个元素为单视频信息）
        self.label_dict = label_dict  # 类别映射字典

        # 识别空类别（无标注样本的类别ID），构建数据集特有属性（供评估使用）
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'epic-kitchens-100',  # 数据集名称
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),  # 时间IOU评估阈值（0.1/0.2/0.3/0.4/0.5）
            'empty_label_ids': empty_label_ids  # 空类别ID列表（无标注样本的类别）
        }

    def find_empty_cls(self, label_dict, num_classes):
        """
        查找EPIC-Kitchens数据集中无标注样本的“空类别”ID
        参数：
            label_dict (dict): 类别名称→ID的映射字典（仅包含有样本的类别）
            num_classes (int): 数据集总类别数（含空类别）
        返回：
            list: 空类别ID列表（如[5, 12]表示ID为5、12的类别无标注样本）
        """
        # 若标注类别数等于总类别数，说明无空类别
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        # 提取所有有样本的类别ID
        label_ids = [v for _, v in label_dict.items()]
        # 遍历所有类别ID，找出无样本的ID
        for cls_id in range(num_classes):
            if cls_id not in label_ids:
                empty_label_ids.append(cls_id)
        return empty_label_ids

    def get_attributes(self):
        """
        获取数据集属性（供评估模块使用）
        返回：
            dict: 包含数据集名称、时间IOU阈值、空类别ID的字典
        """
        return self.db_attributes

    def _load_json_db(self, json_file):
        """
        加载JSON格式的标注数据库，构建视频样本列表和类别映射字典
        参数：
            json_file (str): 标注文件路径
        返回：
            tuple: (dict_db, label_dict)
                - dict_db (tuple): 视频样本元组，每个元素为字典，包含：
                    'id': 视频ID（str）
                    'fps': 视频帧率（float）
                    'duration': 视频时长（秒，float）
                    'segments': 动作时间段（N×2 numpy数组，None表示无标注）
                    'labels': 动作类别ID（N×1 numpy数组，None表示无标注）
                - label_dict (dict): 类别名称→ID映射字典（仅包含有样本的类别）
        """
        # 加载标注文件
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']  # 核心标注数据

        # 构建类别名称→ID映射字典（仅当未初始化时）
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    # 仅记录有标注样本的类别
                    label_dict[act['label']] = act['label_id']

        # 构建数据集列表（不可变元组）
        dict_db = tuple()
        for key, value in json_db.items():
            # 1. 跳过不在指定划分中的视频
            if value['subset'].lower() not in self.split:
                continue

            # 2. 获取视频帧率
            if self.default_fps is not None:
                fps = self.default_fps  # 使用默认帧率
            elif 'fps' in value:
                fps = value['fps']  # 从标注读取帧率
            else:
                assert False, "未知的视频帧率（FPS），请检查标注文件"

            # 3. 获取视频时长（秒）
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8  # 无时长时设为极大值（不影响后续处理）

            # 4. 加载动作标注
            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)  # N×2，存储每个动作的开始/结束时间（秒）
                labels = np.zeros([num_acts, ], dtype=np.int64)  # N×1，存储每个动作的类别ID
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]  # 动作开始时间（秒）
                    segments[idx][1] = act['segment'][1]  # 动作结束时间（秒）
                    labels[idx] = label_dict[act['label']]  # 动作类别ID（仅有样本的类别）
            else:
                segments = None  # 无标注时设为None
                labels = None

            # 将当前视频信息加入数据集列表
            dict_db += ({
                            'id': key,  # 视频ID
                            'fps': fps,  # 帧率
                            'duration': duration,  # 时长（秒）
                            'segments': segments,  # 动作时间段（N×2）
                            'labels': labels  # 动作类别ID（N）
                        },)

        return dict_db, label_dict

    def __len__(self):
        """
        返回数据集样本数量
        返回：
            int: 有效视频样本数（已过滤非指定划分的视频）
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        加载指定索引的视频样本，完成特征预处理和标注转换
        核心流程：加载.npz特征→下采样→维度转换→时间戳映射→训练截断
        参数：
            idx (int): 样本索引（0 ≤ idx < __len__()）
        返回：
            dict: 预处理后的样本字典，包含以下字段：
                'video_id': 视频ID（str）
                'feats': 特征张量（C×T，C=输入特征维度，T=特征序列长度）
                'segments': 动作时间段张量（N×2，None表示无标注）
                'labels': 动作类别张量（N，None表示无标注）
                'fps': 视频帧率（float）
                'duration': 视频时长（float）
                'feat_stride': 下采样后的特征时间步长（int）
                'feat_num_frames': 每个特征覆盖的帧数（int）
        """
        # 获取视频元信息
        video_item = self.data_list[idx]

        # 1. 加载特征文件（EPIC-Kitchens特征为.npz格式，需读取'feats'键）
        filename = os.path.join(
            self.feat_folder, self.file_prefix + video_item['id'] + self.file_ext
        )
        with np.load(filename) as data:
            feats = data['feats'].astype(np.float32)  # 加载后形状：T×C（T=时间步，C=特征维度）

        # 2. 特征下采样（降低序列长度，提升训练效率）
        feats = feats[::self.downsample_rate, :]  # 按步长下采样，形状仍为T×C
        feat_stride = self.feat_stride * self.downsample_rate  # 更新时间步长（下采样后步长变大）
        feat_offset = 0.5 * self.num_frames / feat_stride  # 特征偏移量（用于时间戳对齐）

        # 3. 维度转换：T×C → C×T（适配PyTorch模型输入格式）
        # np.ascontiguousarray确保内存连续，提升张量转换效率
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))  # 转换后形状：C×T

        # 4. 将动作时间戳（秒）转换为特征网格坐标
        # 公式：特征坐标 = (视频时间(秒) × 帧率) / 特征步长 - 偏移量
        # 允许少量负值（后续截断时会处理）
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )  # 形状：N×2
            labels = torch.from_numpy(video_item['labels'])  # 形状：N
        else:
            segments, labels = None, None

        # 5. 构建样本字典（标准化输出格式）
        data_dict = {
            'video_id': video_item['id'],  # 视频ID
            'feats': feats,  # 特征张量（C×T）
            'segments': segments,  # 动作时间段张量（N×2）
            'labels': labels,  # 动作类别张量（N）
            'fps': video_item['fps'],  # 视频帧率
            'duration': video_item['duration'],  # 视频时长（秒）
            'feat_stride': feat_stride,  # 下采样后的特征步长
            'feat_num_frames': self.num_frames  # 每个特征覆盖的帧数
        }

        # 6. 训练阶段截断超长特征序列（测试阶段不截断）
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict