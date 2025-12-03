import numpy as np
import pandas as pd

# 1. 加载特征数组
feat_array = np.load(r"D:\Code\DeepLearning\TAL\actionformer\actionformer-hb\data\thumos\i3d_features\video_test_0000004.npy", allow_pickle=False)
print(f"特征数组形状：{feat_array.shape}")  # 确认 (249, 2048)

# 2. 视频ID映射（关键！需和你的特征数组行顺序完全一致）
# 两种常见情况，选一种适配你的数据：
## 情况A：视频ID是连续的（比如从 video_test_0000001 开始到 0000249）
video_ids = [f"video_test_{i:07d}" for i in range(1, 1 + len(feat_array))]

## 情况B：有配套的视频ID列表文件（比如 txt/csv，每行一个视频ID）
# video_ids = pd.read_csv("video_ids.csv")["video_id"].tolist()  # 替换为你的ID文件路径

# 3. 构建特征表格（视频ID + 2048维特征）
feat_cols = [f"feat_{i}" for i in range(2048)]  # 特征列名：feat_0 到 feat_2047
df_feat = pd.DataFrame(feat_array, columns=feat_cols)
df_feat["video_id"] = video_ids  # 绑定视频ID
df_feat = df_feat[["video_id"] + feat_cols]  # 调整列顺序（视频ID在前，方便查看）

# 查看结果（前3行，只显示前10个特征列，避免刷屏）
print("\n特征表格（前3行+前10维特征）：")
print(df_feat[["video_id"] + feat_cols[:10]].head())

# 保存为CSV（或parquet，更节省空间）
df_feat.to_csv("thumos14_video_features_2048d.csv", index=False, encoding="utf-8")
# df_feat.to_parquet("thumos14_video_features_2048d.parquet", index=False)  # 需安装 pyarrow：pip install pyarrow
print("\n已保存特征表格到 CSV 文件")