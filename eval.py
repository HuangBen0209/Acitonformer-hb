"""
评估模块 (Evaluation Module)
===================================

这个模块用于评估训练好的动作定位模型。主要功能包括：
1. 加载配置文件 (Load config file)
2. 加载训练好的模型检查点 (Load trained model checkpoint)
3. 在验证集或测试集上进行推理 (Perform inference on validation/test set)
4. 评估模型性能并计算mAP (Evaluate model performance and compute mAP)
5. 可选地保存预测结果 (Optionally save prediction results)

模块特点：
- 支持从检查点恢复模型 (Supports resuming from checkpoint)
- 支持多种数据集 (Supports multiple datasets)
- 支持EMA模型评估 (Supports EMA model evaluation)
- 支持日志记录到文件 (Supports logging to file)
- 支持限制输出动作数量 (Supports limiting number of output actions)

主要流程：
1. 配置加载与参数解析 (Config loading and argument parsing)
2. 数据集和数据加载器创建 (Dataset and dataloader creation)
3. 模型初始化与权重加载 (Model initialization and weight loading)
4. 评估器设置 (Evaluator setup)
5. 推理与性能评估 (Inference and performance evaluation)

输入参数说明：
- config: 配置文件路径 (Path to config file)
- ckpt: 检查点文件或文件夹路径 (Path to checkpoint file or folder)
- epoch: 指定加载哪个epoch的检查点 (Specify which epoch's checkpoint to load)
- topk: 最大输出动作数量 (Maximum number of output actions)
- saveonly: 仅保存输出结果而不评估 (Only save outputs without evaluation)
- print_freq: 打印频率 (Print frequency)

输出：
- 控制台打印评估结果 (Console prints evaluation results)
- 可选：保存评估结果到pkl文件 (Optional: Save evaluation results to pkl file)
- 生成日志文件 (Generate log file)

典型用法：
python eval.py configs/my_config.yaml checkpoints/my_model --topk 100 --print_freq 20

注意：这个模块通常用于模型开发的验证阶段，以及最终的测试阶段。
"""

# python 导入
import argparse
import os
import glob
import time
from pprint import pprint

# torch 导入
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# 我们的代码
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

import sys
from datetime import datetime


def main(args):
    """主函数：执行模型评估流程"""

    """0. 加载配置"""
    # 设置日志文件路径
    log_dir = os.path.split(args.ckpt)[0]
    log_file = os.path.join(log_dir, f'eval_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    # 重定向输出到文件和控制台
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')

        def write(self, message):
            """同时写入终端和日志文件"""
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            """刷新输出缓冲区"""
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)

    print(f"评估开始时间: {datetime.now()}")
    print(f"配置文件: {args.config}")
    print(f"检查点: {args.ckpt}")
    print("=" * 50)

    # 检查配置文件是否存在并加载
    if os.path.isfile(args.config):
        cfg = load_config(args.config)  # 加载YAML配置文件
    else:
        raise ValueError("配置文件不存在。")

    # 确保指定了测试集
    assert len(cfg['val_split']) > 0, "必须指定测试集！"

    # 确定检查点文件路径
    # 检查点可以是具体的文件路径，也可以是包含多个检查点的文件夹
    if ".pth.tar" in args.ckpt:
        # 如果参数包含.pth.tar扩展名，则认为是具体的文件路径
        assert os.path.isfile(args.ckpt), "检查点文件不存在！"
        ckpt_file = args.ckpt
    else:
        # 否则认为是文件夹，需要从中选择检查点
        assert os.path.isdir(args.ckpt), "检查点文件夹不存在！"
        if args.epoch > 0:
            # 如果指定了epoch，加载对应的检查点
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            # 否则加载最新的检查点（按文件名排序后的最后一个）
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    # 如果指定了topk参数，修改配置中的最大片段数
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    # 打印配置信息
    pprint(cfg)

    """1. 固定所有随机性"""
    # 固定随机种子以确保可重复性（包括CUDA随机性）
    _ = fix_random_seed(0, include_cuda=True)

    """2. 创建数据集和数据加载器"""
    # 创建验证/测试数据集
    # 参数说明：
    # - cfg['dataset_name']: 数据集名称
    # - False: 表示不是训练模式
    # - cfg['val_split']: 验证集划分名称
    # - **cfg['dataset']: 数据集的其他配置参数
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )

    # 创建数据加载器
    # 参数说明：
    # - val_dataset: 验证数据集
    # - False: 不进行shuffle（测试时不需要打乱顺序）
    # - None: 不使用随机数生成器
    # - 1: 批大小为1（通常测试时设为1）
    # - cfg['loader']['num_workers']: 数据加载的工作进程数
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. 创建模型和评估器"""
    # 创建模型架构
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    # 使用DataParallel进行多GPU并行（即使测试时也可能有多个GPU）
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. 加载检查点"""
    print("=> 加载检查点 '{}'".format(ckpt_file))

    # 加载检查点
    # map_location参数确保将权重加载到正确的GPU设备上
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(int(cfg['devices'][0]))
    )

    # 加载EMA（指数移动平均）模型的权重，通常EMA模型泛化性能更好
    print("从EMA模型加载...")
    model.load_state_dict(checkpoint['state_dict_ema'])

    # 删除检查点以释放内存
    del checkpoint

    # 设置评估器
    det_eval, output_file = None, None

    if not args.saveonly:
        # 获取数据集的属性（如tIoU阈值等）
        val_db_vars = val_dataset.get_attributes()

        # 创建ANETdetection评估器
        # 参数说明：
        # - val_dataset.json_file: 数据集标注文件路径
        # - val_dataset.split[0]: 数据集划分名称
        # - tiou_thresholds: 用于计算mAP的tIoU阈值列表
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        # 如果只需保存结果而不进行评估，设置输出文件路径
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. 测试模型"""
    print("\n开始测试模型 {:s} ...".format(cfg['model_name']))
    start = time.time()

    # 执行一个epoch的验证（测试）
    # 参数说明：
    # - val_loader: 验证数据加载器
    # - model: 待评估的模型
    # - -1: 当前epoch编号（测试时设为-1）
    # - evaluator: 评估器对象
    # - output_file: 输出结果文件路径
    # - ext_score_file: 外部分数文件路径（可选）
    # - tb_writer: TensorBoard写入器（测试时设为None）
    # - print_freq: 打印频率
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )

    end = time.time()
    print("全部完成！总时间: {:0.2f} 秒".format(end - start))

    # 返回mAP值（如果进行了评估）
    return


################################################################################
if __name__ == '__main__':
    """程序入口点"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='训练一个基于点的Transformer用于动作定位')

    # 定义命令行参数
    parser.add_argument('config', type=str, metavar='DIR',
                        help='配置文件路径')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='检查点文件或文件夹路径')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='检查点的epoch编号（默认：-1表示最新）')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='最大输出动作数量（默认：-1表示无限制）')
    parser.add_argument('--saveonly', action='store_true',
                        help='仅保存输出结果而不评估（例如，用于测试集）')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='打印频率（默认：每10次迭代）')

    # 解析参数
    args = parser.parse_args()

    # 执行主函数
    main(args)