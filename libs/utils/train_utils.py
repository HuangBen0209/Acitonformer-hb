"""
训练工具模块
该模块提供了深度学习模型训练和验证过程中所需的各类工具函数和类，主要用于基于PyTorch的时序动作检测（Temporal Action Detection）任务，核心功能包括：
1. 随机种子固定：确保实验结果可复现，支持CPU/GPU环境的随机性控制
2. Checkpoint管理：保存模型权重、优化器状态等，支持最佳模型单独保存
3. 优化器构建：支持SGD/AdamW，实现参数分组的权重衰减（不同层不同decay策略）
4. 学习率调度器：支持带线性预热的余弦退火/多步衰减策略，支持按迭代步数更新
5. 训练/验证流程：实现单轮次训练和验证的完整逻辑，包含损失跟踪、日志记录、梯度裁剪
6. 辅助工具：指标统计（AverageMeter）、模型EMA（指数移动平均）、模型参数打印等
"""
import os
import shutil
import time
import pickle
import numpy as np
import random
from copy import deepcopy
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm


def fix_random_seed(seed, include_cuda=True):
    """
    固定随机种子，确保实验结果可复现
    参数：
        seed (int): 随机种子值
        include_cuda (bool): 是否固定CUDA相关的随机种子（GPU训练时需设为True）
    返回：
        torch.Generator: PyTorch的随机数生成器
    注意：
        1. 固定CUDA种子时会关闭cudnn.benchmark以保证确定性，但可能降低训练速度
        2. CUDA>=10.2时需设置CUBLAS_WORKSPACE_CONFIG环境变量确保确定性
    """
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # 训练阶段：关闭cudnn基准模式以保证可复现性
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDA >= 10.2 必需的配置，确保CUDA算法确定性
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    """
    保存模型训练的Checkpoint文件
    参数：
        state (dict): 需保存的状态字典，典型结构包括：
            - epoch: 当前训练轮次
            - state_dict: 模型权重参数
            - optimizer: 优化器状态
            - scheduler: 学习率调度器状态
            - mAP: 验证集mAP指标
        is_best (bool): 是否为当前最优模型（如最高mAP）
        file_folder (str): 保存文件的文件夹路径
        file_name (str): 普通checkpoint的文件名，默认'checkpoint.pth.tar'
    功能：
        1. 若文件夹不存在则创建
        2. 保存完整的checkpoint（包含优化器/调度器状态）
        3. 若为最优模型，保存仅含模型权重的文件（model_best.pth.tar），便于后续推理
    """
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # 移除优化器和调度器状态，仅保存模型相关参数
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    """
    打印模型所有可训练参数的最小值、最大值和平均值
    参数：
        model (torch.nn.Module): 待打印参数的PyTorch模型
    返回：
        None
    用途：
        调试用，检查参数初始化或训练过程中的参数分布是否异常
    """
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """
    构建优化器，支持SGD/AdamW，实现精细化的参数分组权重衰减策略
    参数：
        model (torch.nn.Module): 待优化的PyTorch模型
        optimizer_config (dict): 优化器配置字典，结构如下：
            {
                "type": str,  # 优化器类型，可选"SGD"或"AdamW"
                "learning_rate": float,  # 基础学习率
                "weight_decay": float,  # 权重衰减系数（仅应用于指定参数）
                "momentum": float,  # SGD专用，动量系数（AdamW无需此参数）
            }
    返回：
        torch.optim.Optimizer: 配置好的PyTorch优化器
    权重衰减分组规则：
        1. 所有偏置（bias）参数：不施加权重衰减
        2. 线性层(Linear)/1D卷积层(Conv1d)/MaskedConv1D的权重：施加权重衰减
        3. 归一化层(LayerNorm/GroupNorm)的权重：不施加权重衰减
        4. Scale/AffineDropPath层的scale参数：不施加权重衰减
        5. 相对位置编码(rel_pe)参数：不施加权重衰减
    验证逻辑：
        确保所有参数仅属于"衰减"或"不衰减"一组，无重叠/遗漏
    """
    # 分离需要/不需要权重衰减的参数集合
    decay = set()
    no_decay = set()
    # 白名单模块：权重需要衰减
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    # 黑名单模块：权重不需要衰减
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # 遍历模型所有模块和参数
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # 拼接完整参数名（模块名.参数名，根模块直接用参数名）
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                # 所有偏置参数不衰减
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # 白名单模块的权重参数需要衰减
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # 黑名单模块的权重参数不衰减
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # Scale/AffineDropPath层的scale参数不衰减
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # 相对位置编码参数不衰减
                no_decay.add(fpn)

    # 验证参数分组的完整性和互斥性
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"参数 {str(inter_params)} 同时出现在衰减/不衰减集合中！"
    assert len(param_dict.keys() - union_params) == 0, \
        f"参数 {str(param_dict.keys() - union_params)} 未被分配到任何分组！"

    # 构建优化器参数组
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # 根据配置创建对应优化器
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("不支持的优化器类型！仅支持SGD/AdamW")

    return optimizer


def make_scheduler(optimizer, optimizer_config, num_iters_per_epoch, last_epoch=-1):
    """
    构建学习率调度器，支持带线性预热的余弦退火/多步衰减策略
    参数：
        optimizer (torch.optim.Optimizer): 已配置好的PyTorch优化器
        optimizer_config (dict): 优化器/调度器配置字典，扩展结构如下：
            {
                "type": str,  # 优化器类型（SGD/AdamW）
                "learning_rate": float,  # 基础学习率
                "weight_decay": float,  # 权重衰减系数
                "momentum": float,  # SGD动量（可选）
                "warmup": bool,  # 是否启用线性预热
                "warmup_epochs": int,  # 预热轮次（warmup=True时必需）
                "epochs": int,  # 总训练轮次（不含预热）
                "schedule_type": str,  # 调度器类型，可选"cosine"或"multistep"
                "schedule_steps": list,  # multistep专用，衰减轮次列表（如[10,20]）
                "schedule_gamma": float,  # multistep专用，衰减系数（如0.1）
            }
        num_iters_per_epoch (int): 每个训练轮次的迭代步数（len(train_loader)）
        last_epoch (int): 上一次训练的最后迭代步数，用于断点续训，默认-1（从头开始）
    返回：
        torch.optim.lr_scheduler._LRScheduler: 配置好的学习率调度器
    调度器规则：
        1. 带预热（warmup=True）：使用自定义的LinearWarmupXXXLR，按迭代步数更新
        2. 无预热（warmup=False）：使用PyTorch原生调度器，按迭代步数更新
    """
    if optimizer_config["warmup"]:
        # 带预热：总轮次=训练轮次+预热轮次
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch
        # 获取预热参数
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # 带线性预热的调度器
        if optimizer_config["schedule_type"] == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )
        if optimizer_config["schedule_type"] == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,  # 修正：warmup_steps → warmup_iters
                max_steps,  # 修正：max_steps → total_iters
                # last_epoch=last_epoch  # 移除：该类不支持 last_epoch，若需初始epoch用 t_initial
                last_epoch if last_epoch != -1 else 0  # 可选：设置初始迭代数
            )
        elif optimizer_config["schedule_type"] == "multistep":
            # 多步衰减调度器：将轮次转换为迭代步数
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                milestones=steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("不支持的调度器类型！仅支持cosine/multistep")
    else:
        # 无预热：总轮次=训练轮次
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # 原生调度器（无预热）
        if optimizer_config["schedule_type"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps,
                last_epoch=last_epoch
            )
        elif optimizer_config["schedule_type"] == "multistep":
            # 多步衰减调度器：将轮次转换为迭代步数
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("不支持的调度器类型！仅支持cosine/multistep")

    return scheduler


class AverageMeter(object):
    """
    指标统计工具类
    用于计算并存储指标的当前值、累计和、平均值，主要用于批量训练/验证时的指标统计（如损失、准确率）
    属性：
        initialized (bool): 是否完成初始化
        val (float): 当前批次的指标值
        avg (float): 累计平均指标值
        sum (float): 累计指标总和
        count (float): 累计样本数/批次总数
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        """
        初始化统计器
        参数：
            val (float): 初始指标值（如第一个批次的损失值）
            n (int/float): 初始样本数/批次大小
        """
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        """
        更新统计器（核心方法）
        参数：
            val (float): 当前批次的指标值
            n (int/float): 当前批次的样本数/批次大小，默认1
        """
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        """
        累加指标值（内部方法，由update调用）
        参数：
            val (float): 当前批次的指标值
            n (int/float): 当前批次的样本数/批次大小
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    """
    模型指数移动平均（EMA）类
    维护一个模型权重的移动平均副本，用于提升模型泛化能力，验证/推理时使用EMA模型可获得更稳定的结果
    属性：
        module (torch.nn.Module): EMA模型副本（深拷贝自原始模型）
        decay (float): 衰减系数（越接近1，EMA模型更新越平缓）
        device (torch.device/None): EMA模型所在设备，None则与原始模型同设备
    """
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # 深拷贝原始模型，创建EMA模型副本（设置为评估模式）
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        """
        内部更新方法：按指定更新函数更新EMA模型权重
        参数：
            model (torch.nn.Module): 原始训练中的模型
            update_fn (function): 更新函数，输入(ema权重, 原始模型权重)，输出新的ema权重
        """
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """
        常规更新EMA模型权重（指数移动平均）
        参数：
            model (torch.nn.Module): 原始训练中的模型
        更新公式：
            ema_weight = decay * ema_weight + (1 - decay) * model_weight
        """
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        """
        直接将EMA模型权重设置为原始模型权重（初始化/重置时使用）
        参数：
            model (torch.nn.Module): 原始模型
        更新公式：
            ema_weight = model_weight
        """
        self._update(model, update_fn=lambda e, m: m)


def train_one_epoch(train_loader, model, optimizer, scheduler, curr_epoch, model_ema=None, clip_grad_l2norm=-1, tb_writer=None, print_freq=20):
    """
    执行单轮次模型训练
    参数：
        train_loader (torch.utils.data.DataLoader): 训练数据加载器
            - 每个迭代返回：video_list (list)，时序动作检测的视频样本列表
        model (torch.nn.Module): 待训练的模型
            - 输入：video_list (list)
            - 输出：losses (dict)，包含"final_loss"（总损失）和各分项损失
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器（按迭代更新）
        curr_epoch (int): 当前训练轮次（从0开始）
        model_ema (ModelEma/None): EMA模型实例，None则不更新EMA
        clip_grad_l2norm (float): 梯度L2范数裁剪阈值，<=0则不裁剪
        tb_writer (torch.utils.tensorboard.SummaryWriter/None): TensorBoard日志写入器
        print_freq (int): 日志打印频率（每多少迭代打印一次）
    返回：
        None
    """
    # 初始化统计器
    batch_time = AverageMeter()
    losses_tracker = {}
    # 每个轮次的迭代总数
    num_iters = len(train_loader)
    # 切换到训练模式
    model.train()
    # 主训练循环
    print(f"\n[训练]: 轮次 {curr_epoch} 开始")
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # 梯度清零（set_to_none=True更高效）
        optimizer.zero_grad(set_to_none=True)
        # 前向传播 + 反向传播
        losses = model(video_list)
        losses['final_loss'].backward()
        # 梯度裁剪（可选）
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        # 更新优化器和学习率调度器
        optimizer.step()
        scheduler.step()

        # 更新EMA模型（可选）
        if model_ema is not None:
            model_ema.update(model)

        # 定期打印日志
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # 同步CUDA内核，确保时间统计准确
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # 更新损失跟踪器
            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())

            # 记录到TensorBoard
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                # 记录所有分项损失
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars('train/all_losses', tag_dict, global_step)
                # 记录总损失
                tb_writer.add_scalar('train/final_loss', losses_tracker['final_loss'].val, global_step)

            # 打印终端日志
            block1 = f'Epoch: [{curr_epoch:03d}][{iter_idx:05d}/{num_iters:05d}]'
            block2 = f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'
            block3 = f'Loss {losses_tracker["final_loss"].val:.2f} ({losses_tracker["final_loss"].avg:.2f})\n'
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += f'\t{key} {value.val:.2f} ({value.avg:.2f})'

            print('\t'.join([block1, block2, block3, block4]))

    # 轮次结束打印
    lr = scheduler.get_last_lr()[0]
    print(f"[训练]: 轮次 {curr_epoch} 结束，当前学习率={lr:.8f}\n")
    return


def valid_one_epoch(val_loader, model, curr_epoch, ext_score_file=None, evaluator=None, output_file=None, tb_writer=None, print_freq=20):
    """
    执行单轮次模型验证
    参数：
        val_loader (torch.utils.data.DataLoader): 验证数据加载器
            - 每个迭代返回：video_list (list)，验证集视频样本列表
        model (torch.nn.Module): 待验证的模型
            - 输入：video_list (list)
            - 输出：output (list[dict])，每个元素对应一个视频的预测结果，结构如下：
                {
                    "video_id": str,  # 视频ID
                    "segments": torch.Tensor,  # 预测时间段，形状[N,2]，N为预测片段数
                    "labels": torch.Tensor,  # 预测类别标签，形状[N]
                    "scores": torch.Tensor,  # 预测置信度，形状[N]
                }
        curr_epoch (int): 当前训练轮次（用于日志记录）
        ext_score_file (str/None): 外部分数文件路径，用于结果后处理
        evaluator (object/None): 评估器实例，需包含evaluate(results, verbose)方法
        output_file (str/None): 预测结果保存路径（pickle格式）
        tb_writer (torch.utils.tensorboard.SummaryWriter/None): TensorBoard日志写入器
        print_freq (int): 日志打印频率（每多少迭代打印一次）
    返回：
        float: 验证集mAP值（若无evaluator则返回0.0）
    注意：
        evaluator和output_file必须至少指定一个
    """
    # 校验参数：必须指定评估器或输出文件
    assert (evaluator is not None) or (output_file is not None), "必须指定evaluator或output_file"

    # 初始化统计器
    batch_time = AverageMeter()
    # 切换到评估模式
    model.eval()
    # 结果收集字典（符合ActivityNet评估格式）
    results = {
        'video-id': [],  # 视频ID列表
        't-start': [],   # 片段起始时间列表
        't-end': [],     # 片段结束时间列表
        'label': [],     # 类别标签列表
        'score': []      # 置信度列表
    }

    # 验证主循环
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # 无梯度前向传播
        with torch.no_grad():
            output = model(video_list)

            # 解析预测结果到ActivityNet格式
            num_vids = len(output)
            for vid_idx in range(num_vids):
                # 仅处理有预测结果的视频
                if output[vid_idx]['segments'].shape[0] > 0:
                    # 扩展视频ID列表（每个预测片段对应一个视频ID）
                    results['video-id'].extend([output[vid_idx]['video_id']] * output[vid_idx]['segments'].shape[0])
                    # 收集预测结果（后续拼接）
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # 定期打印时间日志
        if (iter_idx != 0) and iter_idx % print_freq == 0:
            # 同步CUDA内核，确保时间统计准确
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # 打印时间信息
            print(f'Test: [{iter_idx:05d}/{len(val_loader):05d}]\t'
                  f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})')

    # 整理结果：拼接张量并转换为列表（适配评估器输入格式）
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    # 评估或保存结果
    if evaluator is not None:
        # 结果后处理（可选）
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # 调用评估器计算mAP
        _, mAP, _ = evaluator.evaluate(results, verbose=True)
    else:
        # 保存结果到pickle文件（用于后续评估）
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # 记录mAP到TensorBoard
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP