import argparse
import os
import time
import datetime
from pprint import pprint
import torch
import torch.nn as nn
import torch.utils.data
# 用于可视化
from torch.utils.tensorboard import SummaryWriter

# 我们的代码
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def main(args):
    """主函数，处理训练/推理过程"""

    """1. 设置参数/文件夹"""
    # 解析命令行参数
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("配置文件不存在。")
    pprint(cfg)

    # 准备输出文件夹（基于时间戳）
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            str(cfg['output_folder']), str(cfg_filename + '_' + str(ts)))
    else:
        ckpt_folder = os.path.join(
            str(cfg['output_folder']), str(cfg_filename + '_' + str(args.output)))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # 创建TensorBoard写入器
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # 固定随机种子（这会固定所有随机性）
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # 根据GPU数量重新调整学习率/工作进程数
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    """2. 创建数据集/数据加载器"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # 根据数据集属性更新cfg（针对epic-kitchens数据集）
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # 数据加载器
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. 创建模型、优化器和调度器"""
    # 模型
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    print(f"--------------------cfg['model_name'] = {cfg['model_name']}-------------------------")
    # 对于多GPU训练来说不是最理想的方法，但暂时可用
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # 优化器
    optimizer = make_optimizer(model, cfg['opt'])
    # 调度器
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # 启用模型EMA（指数移动平均）
    print("使用模型EMA...")
    model_ema = ModelEma(model)

    """4. 从检查点恢复/其他杂项"""
    # 从检查点恢复？
    if args.resume:
        if os.path.isfile(args.resume):
            # 加载检查点，重置epoch/最佳RMSE
            checkpoint = torch.load(args.resume,
                map_location=lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # 如果需要，也加载优化器/调度器
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> 加载检查点 '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> 在'{}'未找到检查点".format(args.resume))
            return

    # 保存当前配置
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. 训练/验证循环"""
    print("\n开始训练模型 {:s} ...".format(cfg['model_name']))

    # 开始训练
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        # 训练一个epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        # 定期保存检查点
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # 收尾工作
    tb_writer.close()
    print("全部完成！")
    return

################################################################################
if __name__ == '__main__':
    """程序入口点"""
    # 参数解析器
    parser = argparse.ArgumentParser(
        description='训练一个基于点的Transformer用于动作定位')
    parser.add_argument('config', metavar='DIR',
                        help='配置文件路径')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='打印频率（默认：每10次迭代）')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='检查点保存频率（默认：每5个epoch）')
    parser.add_argument('--output', default='', type=str,
                        help='实验文件夹名称（默认：无）')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='检查点路径（默认：无）')
    args = parser.parse_args()
    main(args)