import yaml

# 默认配置字典，包含模型训练/测试的所有核心参数
# 所有参数均可通过外部YAML配置文件覆盖
DEFAULTS = {
    # 随机种子（用于保证实验结果可复现，建议使用较大的数值以避免冲突）
    "init_rand_seed": 1234567891,
    # 数据集名称，指定训练/测试使用的数据集（支持epic/epic-kitchens、thumos14、activitynet等）
    "dataset_name": "epic",
    # 训练/测试使用的设备列表，默认单GPU（cuda:0），多GPU配置示例：['cuda:0', 'cuda:1']
    "devices": ['cuda:0'],
    # 训练集划分名称，不同数据集划分命名不同（如ActivityNet为'train'，EPIC为'training'）
    "train_split": ('training',),
    # 验证集划分名称，与训练集划分对应
    "val_split": ('validation',),
    # 模型名称，指定使用的核心模型架构（此处为LocPointTransformer）
    "model_name": "LocPointTransformer",
    # 数据集相关配置（特征维度、序列长度、类别数等核心参数）
    "dataset": {
        # 特征的时间步长（每个特征向量对应原始视频的帧数，如16表示每16帧提取一个特征）
        "feat_stride": 16,
        # 每个特征向量覆盖的原始视频帧数（与feat_stride共同决定特征的时间分辨率）
        "num_frames": 32,
        # 数据集默认帧率（FPS），设为None则从标注JSON文件中自动读取（适配不同数据集帧率差异）
        "default_fps": None,
        # 输入特征维度（如RGB特征(1024)+光流特征(1280)=2304，需与提取的特征匹配）
        "input_dim": 2304,
        # 数据集的类别数量（EPIC-Kitchens-100为97类，可根据实际数据集调整）
        "num_classes": 97,
        # 特征下采样率，1表示使用原始特征分辨率，>1则对特征序列进行均匀下采样
        "downsample_rate": 1,
        # 训练时允许的最大特征序列长度（超过该长度的序列会被截断，需适配显存大小）
        "max_seq_len": 2304,
        # 动作截断阈值，用于处理动作边界的截断逻辑（0.5表示保留超过50%的动作片段）
        "trunc_thresh": 0.5,
        # 随机特征裁剪比例范围（如(0.9, 1.0)表示随机裁剪90%-100%的特征序列）
        # 注意：该功能可能未在部分数据集加载器中实现
        "crop_ratio": None,
        # 是否强制将输入特征上采样到固定尺寸（仅ActivityNet数据集需要此配置）
        "force_upsampling": False,
    },
    # 数据加载器配置（影响数据读取效率和显存占用）
    "loader": {
        # 每个GPU的批次大小（多GPU时总批次=batch_size*GPU数量，需根据显存调整）
        "batch_size": 8,
        # 数据加载的工作线程数（建议设置为CPU核心数的一半，避免线程竞争）
        "num_workers": 4,
    },
    # 网络架构核心配置（模型结构的关键参数）
    "model": {
        # 骨干网络类型（convTransformer：卷积+Transformer混合架构 | conv：纯卷积架构）
        "backbone_type": 'convTransformer',
        # FPN（特征金字塔网络）类型（fpn：使用特征金字塔融合多尺度特征 | identity：不使用FPN）
        "fpn_type": "identity",
        # 骨干网络的层级配置，元组内数值表示各阶段Transformer块的数量（如(2,2,5)表示3个阶段各2/2/5层）
        "backbone_arch": (2, 2, 5),
        # 特征金字塔各层级之间的缩放因子（控制不同层级特征的时间分辨率比例）
        "scale_factor": 2,
        # 各金字塔层级的回归范围（每个元组表示该层级负责预测的动作时长范围，单位：特征步长）
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # 多头自注意力的头数（需满足embd_dim能被n_head整除，如512/4=128）
        "n_head": 4,
        # 自注意力窗口大小；<=1表示使用全局注意力（覆盖整个序列），>1表示局部窗口注意力
        "n_mha_win_size": -1,
        # 嵌入网络的卷积核大小（用于特征维度转换和空间编码）
        "embd_kernel_size": 3,
        # 嵌入网络的输出特征维度（Transformer的输入维度）
        "embd_dim": 512,
        # 是否在嵌入网络输出后添加层归一化（Layer Normalization），提升训练稳定性
        "embd_with_ln": True,
        # FPN网络的特征维度（需与embd_dim匹配以保证特征融合效果）
        "fpn_dim": 512,
        # 是否在FPN各层级输出后添加层归一化
        "fpn_with_ln": True,
        # FPN的起始层级（从骨干网络的第几个层级开始构建金字塔，0表示最底层）
        "fpn_start_level": 0,
        # 预测头（分类/回归）的特征维度
        "head_dim": 512,
        # 分类/回归/中心预测头的卷积核大小（用于特征提取）
        "head_kernel_size": 3,
        # 预测头的网络层数（包含最后一层输出层，层数越多拟合能力越强但易过拟合）
        "head_num_layers": 3,
        # 是否在预测头中添加层归一化，提升模型泛化能力
        "head_with_ln": True,
        # 缓冲点的最大长度因子（决定模型能处理的缓冲点序列最大长度，需大于max_seq_len）
        "max_buffer_len_factor": 6.0,
        # 是否使用绝对位置编码（绝对位置编码会添加到输入嵌入中，提供位置信息）
        "use_abs_pe": False,
        # 是否使用相对位置编码（相对位置编码添加到自注意力中，更适合长序列建模）
        "use_rel_pe": False,
        # 是否使用LBC模块，增强边界区分能力
        "use_lbc": True,
    },
    # 训练过程配置（损失函数、正则化、采样策略等）
    "train_cfg": {
        # 中心采样策略（radius：基于固定半径采样正样本 | none：不使用中心采样）
        "center_sample": "radius",
        # 中心采样的半径大小（控制正样本的采样范围，平衡正负样本比例）
        "center_sample_radius": 1.5,
        # 回归损失的权重（设为-1时启用自动损失平衡机制，自适应调整分类/回归损失权重）
        "loss_weight": 1.0,
        # 分类头的先验概率（用于初始化分类层偏置，解决类别不平衡问题，通常设为0.01）
        "cls_prior_prob": 0.01,
        # 损失归一化的初始值（用于稳定训练初期的损失值）
        "init_loss_norm": 2000,
        # 梯度裁剪的L2范数阈值（Pre-LN Transformer通常无需梯度裁剪，设为-1禁用）
        "clip_grad_l2norm": -1,
        # 无数据的分类头列表（针对EPIC-Kitchens/THUMOS数据集的特殊处理，部分类别无训练数据）
        "head_empty_cls": [],
        # Transformer层的dropout比率（随机丢弃神经元，防止过拟合）
        "dropout": 0.0,

        # DropPath比率（随机丢弃网络路径，比dropout更温和的正则化方式）
        "droppath": 0.1,
        #lbc 的窗口大小
        "lbc_win_size" : 5,
        # lbc 的融合权重
        "lbc_fusion_gate" :0.2,
        # 标签平滑系数（>0时启用标签平滑，将硬标签转换为软标签，缓解过拟合）
        "label_smoothing": 0.0,
    },
    # 测试/推理配置（后处理、NMS、分数过滤等）
    "test_cfg": {
        # NMS（非极大值抑制）前的分数阈值，低于该值的预测框直接过滤，减少计算量
        "pre_nms_thresh": 0.001,
        # NMS前保留的最高分数预测框数量（控制候选框数量，平衡精度和速度）
        "pre_nms_topk": 5000,
        # NMS的IOU阈值（重叠度超过该值的预测框会被抑制）
        "iou_threshold": 0.1,
        # 最终输出的最小分数阈值，低于该值的预测结果不参与评估
        "min_score": 0.01,
        # 单视频输出的最大片段数量（防止过多冗余预测）
        "max_seg_num": 1000,
        # NMS方法（soft：软NMS（分数平滑衰减） | hard：硬NMS（直接删除） | none：不使用NMS）
        "nms_method": 'soft',
        # 软NMS的sigma参数（控制分数衰减速度，sigma越大衰减越慢）
        "nms_sigma": 0.5,
        # 最小片段时长阈值（低于该值的预测片段视为无效，过滤短噪声预测）
        "duration_thresh": 0.05,
        # 是否使用多类别NMS（对每个类别独立执行NMS，适合多类别动作检测）
        "multiclass_nms": True,
        # 外部分数文件路径（用于融合外部模型的预测分数，设为None则不使用）
        "ext_score_file": None,
        # 投票阈值（多模型集成时的投票决策阈值，高于该值的类别才会被选中）
        "voting_thresh": 0.75,
    },
    # 优化器配置（训练的优化策略）
    "opt": {
        # 优化器类型（SGD：随机梯度下降 | AdamW：带权重衰减的Adam优化器，适合Transformer）
        "type": "AdamW",
        # 动量参数（仅SGD使用，提升收敛速度和稳定性）
        "momentum": 0.9,
        # 权重衰减系数（L2正则化，防止模型权重过大）
        "weight_decay": 0.0,
        # 初始学习率（需根据批次大小调整，批次越大学习率可适当提高）
        "learning_rate": 1e-3,
        # 训练总轮数（不包含热身（warmup）轮数）
        "epochs": 30,
        # 是否启用学习率热身（warmup），训练初期逐步提升学习率，避免梯度爆炸
        "warmup": True,
        # 热身轮数（前5轮学习率从低到高线性提升至初始学习率）
        "warmup_epochs": 5,
        # 学习率调度器类型（cosine：余弦退火调度 | multistep：多步衰减调度）
        "schedule_type": "cosine",
        # 多步衰减的步数（仅multistep调度生效，指定学习率衰减的轮数，不含warmup）
        "schedule_steps": [],
        # 多步衰减的衰减因子（仅multistep调度生效，每到指定步数学习率乘以该因子）
        "schedule_gamma": 0.1,
    }
}


def _merge(src, dst):
    """
    递归合并两个字典（将src中的键值对合并到dst中）
    若键在dst中存在且对应值为字典，则递归合并；否则直接赋值

    参数:
        src (dict): 待合并的源字典（如默认配置）
        dst (dict): 目标字典（如用户自定义配置）
    """
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    """
    加载默认配置

    返回:
        dict: 完整的默认配置字典
    """
    config = DEFAULTS
    return config


def _update_config(config):
    """
    更新配置字典，填充派生字段（将数据集和训练/测试配置同步到模型配置中）
    保证模型配置与数据集、训练策略的一致性

    参数:
        config (dict): 合并后的配置字典

    返回:
        dict: 更新后的完整配置字典
    """
    # 将数据集的核心参数同步到模型配置中
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    # 将训练/测试配置嵌入到模型配置中，方便模型调用
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config


def load_config(config_file, defaults=DEFAULTS):
    """
    从YAML文件加载配置，并与默认配置合并，最后更新派生字段

    参数:
        config_file (str): YAML配置文件路径
        defaults (dict): 默认配置字典，默认为DEFAULTS

    返回:
        dict: 最终的完整配置字典
    """
    with open(config_file, "r") as fd:
        # 加载用户自定义配置
        config = yaml.load(fd, Loader=yaml.FullLoader)
    # 合并默认配置和用户配置（用户配置会覆盖默认配置）
    _merge(defaults, config)
    # 更新派生字段，保证配置一致性
    config = _update_config(config)
    return config