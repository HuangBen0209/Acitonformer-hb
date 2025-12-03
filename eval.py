# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    # 修复后：先解析成 device 对象，再取索引
    device = torch.device(cfg['devices'][0])  # 解析 'cuda:0' 成 device 对象
    checkpoint = torch.load(
        args.ckpt,
        map_location=lambda storage, loc: storage.cuda(device.index)  # 用 device.index 取整数索引
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
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
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""  # 程序入口（当直接运行该脚本时，以下代码才会执行）

    # 创建参数解析器，描述信息为“训练用于动作定位的点基Transformer”
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')

    # 1. 必选参数：配置文件路径（type=str 表示参数是字符串类型，metavar=DIR 是命令行显示别名）
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')  # 帮助信息：配置文件的路径

    # 2. 必选参数：模型权重（checkpoint）路径（必选，需传入 .pth.tar 文件路径）
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')  # 帮助信息：权重文件的路径

    # 3. 可选参数：指定权重的epoch（默认-1，表示自动加载最新/最优权重）
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')  # 帮助信息：权重对应的训练轮次

    # 4. 可选参数：输出动作的最大数量（默认-1，表示不限制，按模型默认输出）
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')  # 帮助信息：输出动作的最大个数

    # 5. 可选参数：仅保存预测结果，不进行评估（用于测试集，后续单独评估）
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    # 注：action='store_true' 表示只要命令行加了这个参数，就设为 True，不加则为 False

    # 6. 可选参数：打印频率（默认每10个迭代打印一次日志）
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')  # 帮助信息：日志打印频率

    # 解析命令行传入的参数，存入 args 对象（后续 main 函数会使用这些参数）
    args = parser.parse_args()
    # 调用主函数，传入解析后的参数
    main(args)