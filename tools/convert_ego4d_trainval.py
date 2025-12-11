import os
import json

import torch
import numpy as np

"""
处理前的目录结构：

本文件夹
│  convert_ego4d_trainval.py
│  ego4d_label_map.txt
│  ... 
│
└───features
│    └───slowfast8x8_r100_k400
│    └───omnivore_video_swinl
│
└───annotations
│    └───moments_train.json
│    └───moments_val.json
│  ...
"""

# 从 Ego4D 官网下载的“整段视频”特征
slowfast_dir = 'features/slowfast8x8_r101_k400'
omnivore_dir = 'features/omnivore_video_swinl'

# 从 Ego4D 官网下载的标注文件
train_annot_path = 'annotations/moments_train.json'
val_annot_path = 'annotations/moments_val.json'

# 标签映射表
label_map_path = 'ego4d_label_map.txt'

# 处理后的特征保存路径
slowfast_out_dir = 'features/slowfast_features'
omnivore_out_dir = 'features/omnivore_features'
os.makedirs(slowfast_out_dir, exist_ok=True)
os.makedirs(omnivore_out_dir, exist_ok=True)

# 处理后的统一标注文件保存路径
annot_out_path = 'annotations/ego4d.json'

# 特征提取时使用的“clip 长度 / 步长”
clip_size = 32
stride = 16

# 读取训练集与验证集标注
with open(train_annot_path, 'r') as f:
    train_videos = json.load(f)['videos']
with open(val_annot_path, 'r') as f:
    val_videos = json.load(f)['videos']
videos = train_videos + val_videos

# 读取标签→id 映射表
label_map = dict()
with open(label_map_path, 'r') as f:
    lines = [l.strip().split('\t') for l in f.readlines()]
    for v, k in lines:
        label_map[k] = int(v)

database = dict()

# 逐个视频处理
for video in videos:
    vid = video['video_uid']
    print('正在处理视频 {:s} ...'.format(vid))
    subset = video['split']
    if subset == 'train':
        subset = 'training'
    elif subset == 'val':
        subset = 'validation'

    # 加载整段视频的特征
    slowfast_path = os.path.join(slowfast_dir, vid + '.pt')
    omnivore_path = os.path.join(omnivore_dir, vid + '.pt')
    # 如果特征文件缺失就跳过
    if not os.path.exists(slowfast_path):
        print('> 缺少 slowfast 特征')
    if not os.path.exists(omnivore_path):
        print('> 缺少 omnivore 特征')
        continue
    slowfast_video = torch.load(slowfast_path).numpy()
    omnivore_video = torch.load(omnivore_path).numpy()

    # 逐个 clip 处理
    clips = video['clips']
    for clip in clips:
        cid = clip['clip_uid']
        ss = max(float(clip['video_start_sec']), 0)  # 起始秒
        es = float(clip['video_end_sec'])            # 结束秒
        sf = max(int(clip['video_start_frame']), 0)  # 起始帧
        ef = int(clip['video_end_frame'])            # 结束帧
        duration = es - ss      # clip 时长（秒）
        frames = ef - sf        # clip 帧数
        fps = frames / duration
        if fps < 10 or fps > 100:
            continue

        # 把事件起止对齐到特征网格
        prepend_frames = sf % stride
        prepend_sec = prepend_frames / fps
        duration += prepend_sec
        frames += prepend_frames

        append_frames = append_sec = 0
        if (frames - clip_size) % stride:
            append_frames = stride - (frames - clip_size) % stride
            append_sec = append_frames / fps
            duration += append_sec
            frames += append_frames

        # 计算在整段特征中的起止索引
        si = (sf - prepend_frames) // stride
        ei = (ef + append_frames - clip_size) // stride
        if ei > len(slowfast_video):
            raise ValueError('结束索引超出 slowfast 特征长度')
        if ei > len(omnivore_video):
            raise ValueError('结束索引超出 omnivore 特征长度')

        # 保存该 clip 的特征
        slowfast_clip = slowfast_video[si:ei]
        omnivore_clip = omnivore_video[si:ei]
        np.save(
            os.path.join(slowfast_out_dir, cid + '.npy'),
            slowfast_clip.astype(np.float32),
        )
        np.save(
            os.path.join(omnivore_out_dir, cid + '.npy'),
            omnivore_clip.astype(np.float32),
        )

        annotations = []

        # 多标注人
        annotators = clip['annotations']
        for annotator in annotators:
            # 每个标注人的动作列表
            items = annotator['labels']
            for item in items:
                # 只保留 primary 类别
                if not item['primary']:
                    continue

                # 计算相对该 clip 的起止时间/帧
                ssi = item['video_start_time'] - ss + prepend_sec
                esi = item['video_end_time'] - ss + prepend_sec
                sfi = item['video_start_frame'] - sf + prepend_frames
                efi = item['video_end_frame'] - sf + prepend_frames

                # 过滤掉太短的动作
                if esi - ssi < 0.25:
                    continue

                label = item['label']
                annotations += [{
                    'label': label,
                    'segment': [round(ssi, 2), round(esi, 2)],
                    'segment(frames)': [sfi, efi],
                    'label_id': label_map[label],
                }]

        # 如果该 clip 没有任何有效动作，就跳过不写
        if len(annotations) == 0:
            continue

        # 写入数据库
        database[cid] = {
            'subset': subset,
            'duration': round(duration, 2),
            'fps': round(fps, 2),
            'annotations': annotations,
        }

# 最终把 database 写成统一的 json
out = {'version': 'v1', 'database': database}
with open(annot_out_path, 'w') as f:
    json.dump(out, f)