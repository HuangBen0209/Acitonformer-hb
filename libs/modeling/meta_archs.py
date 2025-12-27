import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms
class PtTransformerClsHead(nn.Module):
    """
    用于分类的 1D 卷积头
    通过特征金字塔网络（FPN）的多尺度特征
    对每个特征层进行分类预测
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()
        # 构建头部
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())
        # 分类器
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )
        # 在模型初始化中使用先验概率以提高稳定性
        # 这将覆盖其他权重初始化
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
        # 对空类别进行快速修复：
        # 与这些类别相关的权重将保持不变
        # 我们将其偏置设置为一个很大的负值，以防止它们的输出
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        # 对每个金字塔层应用分类器
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )
        # fpn_masks 保持不变
        return out_logits

class PtTransformerRegHead(nn.Module):
    """
    共享的1D卷积头部，用于回归任务
    与PtTransformerClsHead有类似的逻辑，但为了清晰起见，采用分开实现的方式
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # 构建卷积头部
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # 用于目标检测的边界框回归
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # 对每个金字塔层级的特征应用回归头部
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )
        # fpn_masks保持不变
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        基于Transformer的单阶段动作定位模型
    """
    def __init__(
            self,
            backbone_type,  # 定义使用的骨干网络类型（字符串）
            fpn_type,  # 定义使用的FPN类型（字符串）
            backbone_arch,  # 定义embed/stem/branch的层数（元组）
            scale_factor,  # 分支层之间的缩放因子
            input_dim,  # 输入特征维度
            max_seq_len,  # 最大序列长度（用于训练）
            max_buffer_len_factor,  # 最大缓冲区大小（定义为max_seq_len的倍数）
            n_head,  # Transformer中自注意力的头数
            n_mha_win_size,  # 自注意力窗口大小；-1表示使用完整序列
            embd_kernel_size,  # 嵌入网络的卷积核大小
            embd_dim,  # 嵌入网络的输出特征通道数
            embd_with_ln,  # 是否在嵌入网络后添加LayerNorm
            fpn_dim,  # FPN的特征维度
            fpn_with_ln,  # 是否在FPN末端应用LayerNorm
            fpn_start_level,  # FPN的起始层级
            head_dim,  # 头部网络的特征维度
            regression_range,  # FPN每一层的回归范围
            head_num_layers,  # 头部网络的层数（包含分类器）
            head_kernel_size,  # 回归/分类头的卷积核大小
            head_with_ln,  # 是否在回归/分类头添加LayerNorm
            use_abs_pe,  # 是否使用绝对位置编码
            use_rel_pe,  # 是否使用相对位置编码
            use_lbc,  # 是否使用LBC模块
            num_classes,  # 动作类别数
            train_cfg,  # 训练配置
            test_cfg  # 测试配置
    ):
        super().__init__()
        # 将参数重新分配到backbone/neck/head
        self.fpn_strides = [scale_factor ** i for i in range(
            fpn_start_level, backbone_arch[-1] + 1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # 类别数 = num_classes + 1（背景），最后一种类别作为背景
        # 例如：num_classes = 10 -> 0,1,...,9为动作，10为背景
        self.num_classes = num_classes

        # 检查特征金字塔和局部注意力窗口大小
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len必须能被fpn步长和窗口大小整除"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # 训练时配置
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.lbc_win_size = train_cfg['lbc_win_size']
        self.lbc_fusion_gate = train_cfg['lbc_fusion_gate']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # 测试时配置
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # 需要更好的方式来分发参数到backbones/necks
        # 骨干网络：卷积 + Transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch': backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                    'attn_pdrop': 0.0,
                    'proj_pdrop': self.train_dropout,
                    'path_pdrop': self.train_droppath,
                    'lbc_win_size':self.lbc_win_size ,
                    'lbc_fusion_gate': self.lbc_fusion_gate,
                    'use_abs_pe': use_abs_pe,
                    'use_rel_pe': use_rel_pe,
                    'use_lbc': use_lbc,
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # FPN网络：卷积层
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln
            }
        )
        # ==============================================
        # 位置生成器：1D时序场景下的Anchor-free参考点生成器
        # 核心作用：替代传统Anchor（锚框），在FPN多尺度特征图上生成时序参考点，
        #          作为后续时序检测/回归任务的基准锚点（逐点做分类+回归预测）
        # 所属范式：Anchor-free（无锚框），源自2D目标检测的CenterNet/FCOS思路迁移
        # ==============================================
        self.point_generator = make_generator(
            # 生成器类型：指定为"point"（点生成器），适配1D时序场景（2D场景常用"anchor"锚框生成器）
            'point',
            # 解包关键字参数：向点生成器传递核心配置（**表示将字典拆分为关键字参数传入）
            **{
                # 参数1：点生成器支持的最大时序序列长度（带缓冲）
                # - max_seq_len：模型原本支持的最大时序序列长度（如语音/时序信号的最大步数）
                # - max_buffer_len_factor：缓冲因子（如1.2），避免推理时序列略长于max_seq_len导致点不足
                # - 最终值：保证生成的参考点能覆盖所有可能的输入序列长度（含冗余缓冲）
                'max_seq_len': max_seq_len * max_buffer_len_factor,

                # 参数2：FPN各层级的下采样步长（stride）
                # - 对应1D FPN（FPN1D）的scale_factor，如[fpn_strides=[2,4,8]]表示：
                #   FPN低层特征图（步长2）：1个参考点对应原始序列2个时序步（细粒度）
                #   FPN高层特征图（步长8）：1个参考点对应原始序列8个时序步（粗粒度）
                # - 作用：控制不同FPN层级生成的参考点数量（步长大→点数少，步长小→点数多）
                'fpn_strides': self.fpn_strides,

                # 参数3：FPN各层级的时序回归范围（分层回归）
                # - 格式示例：[[0,16], [16,64], [64,256]]（单位：时序步）
                # - 作用：限制不同FPN层级参考点的预测范围，避免跨层级无效回归：
                #   低层FPN（细粒度）→ 负责小范围/短时长事件（如10ms语音关键词）
                #   高层FPN（粗粒度）→ 负责大范围/长时长事件（如100ms语音句子）
                # - 优势：提升多尺度时序事件的检测精度（源自FCOS的分层回归思路）
                'regression_range': self.reg_range
            }
        )
        # 分类和回归头
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # 维护前景数量的EMA以稳定损失归一化器，对小批量训练很有用
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # 获取设备类型的简易方法（如果参数在不同设备上会抛出错误）
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # 将视频列表批处理为特征(B, C, T)和掩码(B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)
        # 前向传播网络（backbone -> neck -> heads）
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # 计算FPN上的点坐标，用于计算GT或解码最终结果
        # points: 列表[T x 4]，长度 = FPN层级数（在小批量中所有样本共享）
        points = self.point_generator(fpn_feats)
        # 输出分类：List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # 输出偏移：List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        # 重排输出维度
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # 训练时返回损失
        if self.training:
            # 生成片段/标签列表：长度 = B，每个元素形状为[N x 2] / [N]
            assert video_list[0]['segments'] is not None, "GT动作标签不存在"
            assert video_list[0]['labels'] is not None, "GT动作标签不存在"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # 计算分类和回归的GT标签
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            # 计算损失并返回
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            return losses

        else:
            # 解码动作（sigmoid/stride等处理）
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            从字典项列表生成批处理特征和掩码
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "训练时输入长度必须小于max_seq_len"
            # 将max_len设为self.max_seq_len
            max_len = self.max_seq_len
            # 批输入形状 B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "推理时只支持batch_size=1"
            # 输入长度 < self.max_seq_len，填充到max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # 将输入填充到下一个可整除的大小
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # 生成掩码
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # 推送到设备
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # 合并所有FPN层级的点：List[T x 4] -> F T x 4（小批量中所有样本共享）
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # 遍历每个视频样本
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # 添加到列表（长度 = 图像数，每个大小为 FT x C）
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, 回归范围, 步长)
        # gt_segment : N (#事件) x 2
        # gt_label : N (#事件) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # 当前样本没有动作的边缘情况
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # 计算所有片段的长度 -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # 计算每个点到每个片段边界的距离
        # 所有回归目标的自动广播 -> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # 所有片段的中心点 F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # 基于步长半径的中心采样
            # 计算新边界：concat_points[:, 3]存储步长
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # 防止t_mins/t_maxs超出动作边界
            # 左边界：torch.maximum(t_mins, gt_segs[:, :, 0])
            # 右边界：torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N（到新边界的距离）
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # 在GT动作内部
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # 限制每个位置的回归范围
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # 如果一个时刻仍有多个动作，选择持续时间最短的（最容易回归）
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # 边缘情况：多个动作持续时间非常相似（例如THUMOS14数据集）
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # 防止具有相同标签和边界的多个GT动作
        cls_targets.clamp_(min=0.0, max=1.0)
        # 可以使用min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # 基于步长归一化
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (列表) [B, T_i, C]
        # gt_* : B (列表) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. 分类损失
        # 堆叠列表 -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # 拼接预测的偏移 -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # 更新损失归一化器
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls已经是one-hot编码，直接掩码处理
        gt_target = gt_cls[valid_mask]

        # 可选标签平滑
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # 焦点损失
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. 回归损失（在正样本上使用IoU/GIoU损失）
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # 在正样本上定义GIoU损失
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # 返回损失字典
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'final_loss': final_loss}

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets
    ):
        # video_list B (列表) [字典]
        # points F (列表) [T_i, 4]
        # fpn_masks, out_*: F (列表) [B, T_i, C]
        results = []

        # 1: 收集视频元信息
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: 对每个单独视频进行推理并收集结果
        # 到目前为止，所有结果都使用特征网格上定义的时间戳
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # 收集每个视频的输出
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # 在单个视频上推理（应始终是这种情况）
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # 传递视频元信息
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # 步骤3: 后处理
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
    ):
        # points F (列表) [T_i, 4]
        # fpn_masks, out_*: F (列表) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # 遍历FPN层级
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
        ):
            # 对输出logits进行sigmoid归一化
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # 应用过滤以加速NMS（遵循detectron2方法）
            # 1. 保留置信度分数大于阈值的片段
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. 仅保留前k个最高得分的框
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # 修复pytorch 1.9中的警告
            pt_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. 收集预测偏移
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. 计算预测片段（对输出偏移进行步长反归一化）
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. 保留持续时间大于阈值（相对于特征网格）的片段
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N（过滤后的片段数）x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # 沿FPN层级拼接（F N_i, C）
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # 输入：字典项列表
        # (1) 转移到CPU；(2) NMS；(3) 转换为实际时间戳
        processed_results = []
        for results_per_vid in results:
            # 解包元信息
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: 解包结果并移动到CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: 批量NMS（仅在CPU上实现）
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: 从特征网格转换为秒
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # 截断所有边界到[0, duration]内
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen

            # 4: 重新打包结果
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels}
            )

        return processed_results