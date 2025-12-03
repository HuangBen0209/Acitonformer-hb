import pickle

# 1. 定义 PKL 文件路径（绝对路径或相对路径）
pkl_path = r"D:\Code\DeepLearning\TAL\actionformer\actionformer-hb\data\thumos\annotations\thumos14_cls_scores.pkl"  # 替换为你的文件路径

# 2. 解析 PKL 文件
try:
    with open(pkl_path, "rb") as f:
        # 反序列化，得到原始对象
        data = pickle.load(f)

    # 3. 查看结果（根据对象类型调整）
    print("解析成功！对象类型：", type(data))
    print("对象内容：", data)  # 若数据量大，可打印前几行或关键键值

except FileNotFoundError:
    print(f"错误：未找到文件 {pkl_path}")
except Exception as e:
    print(f"解析失败：{str(e)}")  # 可能是版本不兼容、文件损坏等