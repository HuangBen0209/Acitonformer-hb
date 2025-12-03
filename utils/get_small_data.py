import json
import random
import os
import shutil
from collections import defaultdict

# ======================== 集中配置参数（可直接修改）========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 原数据路径
FULL_ANNOTATION_PATH = os.path.join(PROJECT_ROOT, "data", "thumos", "annotations", "thumos14.json")
FULL_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "thumos", "i3d_features")
# 小数据集输出路径
SMALL_DATA_ROOT = "data/thumos_small"
SMALL_TRAIN_ANNOT = f"{SMALL_DATA_ROOT}/annotations/thumos14_train_small.json"  # 单独训练集
SMALL_TEST_ANNOT = f"{SMALL_DATA_ROOT}/annotations/thumos14_test_small.json"   # 单独测试集
SMALL_MERGED_ANNOT = f"{SMALL_DATA_ROOT}/annotations/thumos14_small.json"  # 合并集（新增）
SMALL_FEATURE_DIR = f"{SMALL_DATA_ROOT}/i3d_features"

# 抽样配置（二选一，修改后注释掉另一个）
SAMPLING_MODE = "RATIO"  # 抽样模式："COUNT"（按数量）或 "RATIO"（按比例）
# 模式1：按数量抽样（每个类别抽指定个数）
TRAIN_NUM_PER_CLASS = 3    # 每个类别训练样本数
TEST_NUM_PER_CLASS = 2     # 每个类别测试样本数
# 模式2：按比例抽样（每个类别抽指定比例）
TRAIN_RATIO_PER_CLASS = 0.2  # 每个类别训练样本抽样比例（如 0.2=20%）
TEST_RATIO_PER_CLASS = 0.1   # 每个类别测试样本抽样比例（如 0.1=10%）

# 兜底限制（避免样本过多）
TRAIN_MAX_TOTAL = 100  # 训练集最大总样本数
TEST_MAX_TOTAL = 50    # 测试集最大总样本数
RANDOM_SEED = 42       # 固定种子（保证每次抽样结果一致）
# ==========================================================================


def check_input_paths():
    """检查原数据路径是否存在"""
    if not os.path.exists(FULL_ANNOTATION_PATH):
        raise FileNotFoundError(f"原标注文件不存在：{FULL_ANNOTATION_PATH}")
    if not os.path.exists(FULL_FEATURE_DIR):
        raise FileNotFoundError(f"原特征目录不存在：{FULL_FEATURE_DIR}")
    print("✅ 原数据路径检查通过")


def parse_full_annotation():
    """解析标注，返回：
    - 训练池（原validation）、测试池（原test）的视频信息
    - 每个类别的样本分布（训练池+测试池）
    - 类别映射
    """
    with open(FULL_ANNOTATION_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # 兼容标注格式（database dict 或 直接列表）
    if "database" in full_data and isinstance(full_data["database"], dict):
        database = full_data["database"]
    elif isinstance(full_data, list):
        database = {item.get("video_name", str(i)): item for i, item in enumerate(full_data)}
    else:
        raise ValueError("标注格式错误！需含 'database' 字段或直接为样本列表")

    # 初始化变量
    train_pool = {}  # 训练池：原validation视频 {vid: info}
    test_pool = {}   # 测试池：原test视频 {vid: info}
    class_map = {}   # {label_id: label_name}
    # 统计每个类别的样本分布（训练池+测试池）
    class_distribution = {
        "train_pool": defaultdict(list),  # {label_id: [vid1, vid2...]}
        "test_pool": defaultdict(list)    # {label_id: [vid1, vid2...]}
    }

    for vid, info in database.items():
        # 提取subset（兼容大小写和字段名）
        subset = info.get("subset", info.get("split", "")).strip().lower()
        annotations = info.get("annotations", [])
        if not annotations:
            continue  # 跳过无标注视频

        # 构建类别映射和分布统计
        for ann in annotations:
            label_id = ann.get("label_id")
            label_name = ann.get("label", ann.get("category", f"class_{label_id}"))
            if label_id not in class_map:
                class_map[label_id] = label_name
            # 按subset添加到对应池的类别分布中
            if subset == "validation":
                if vid not in class_distribution["train_pool"][label_id]:
                    class_distribution["train_pool"][label_id].append(vid)
            elif subset == "test":
                if vid not in class_distribution["test_pool"][label_id]:
                    class_distribution["test_pool"][label_id].append(vid)

        # 按subset添加到训练池/测试池
        if subset == "validation":
            train_pool[vid] = info
        elif subset == "test":
            test_pool[vid] = info

    # 打印原数据集类别分布（核心需求）
    print("\n" + "="*80)
    print("📊 原数据集每个类别样本数统计（训练池=原validation，测试池=原test）")
    print("="*80)
    print(f"{'类别ID':<10} {'类别名称':<20} {'训练池样本数':<15} {'测试池样本数':<15}")
    print("-"*80)
    for label_id in sorted(class_map.keys()):
        label_name = class_map[label_id]
        train_cnt = len(class_distribution["train_pool"].get(label_id, []))
        test_cnt = len(class_distribution["test_pool"].get(label_id, []))
        print(f"{label_id:<10} {label_name:<20} {train_cnt:<15} {test_cnt:<15}")
    print("-"*80)
    print(f"{'总计':<10} {'-':<20} {len(train_pool):<15} {len(test_pool):<15}")
    print("="*80 + "\n")

    return train_pool, test_pool, class_map, class_distribution


def sample_small_dataset(class_distribution, pool_type):
    """根据配置抽样小数据集
    Args:
        class_distribution: 类别分布 dict
        pool_type: 池类型 "train_pool" 或 "test_pool"
    Returns:
        small_vids: 抽样后的视频ID列表
        sample_log: 抽样日志（每个类别抽了多少）
    """
    small_vids = []
    sample_log = defaultdict(dict)  # {label_id: {"total": 原数量, "sampled": 抽样数量}}

    for label_id, vids in class_distribution[pool_type].items():
        total = len(vids)
        if total == 0:
            sample_log[label_id] = {"total": 0, "sampled": 0}
            continue

        # 按模式抽样
        if SAMPLING_MODE == "COUNT":
            # 按数量抽样（不超过该类别总数量）
            if pool_type == "train_pool":
                sample_num = min(TRAIN_NUM_PER_CLASS, total)
            else:
                sample_num = min(TEST_NUM_PER_CLASS, total)
        else:  # SAMPLING_MODE == "RATIO"
            # 按比例抽样（四舍五入，最少抽1个）
            if pool_type == "train_pool":
                sample_num = max(1, int(total * TRAIN_RATIO_PER_CLASS))
            else:
                sample_num = max(1, int(total * TEST_RATIO_PER_CLASS))

        # 抽样（固定种子保证可复现）
        sampled_vids = random.sample(vids, sample_num)
        small_vids.extend(sampled_vids)
        sample_log[label_id] = {"total": total, "sampled": sample_num}

    # 去重 + 兜底限制（不超过最大总样本数）
    small_vids = list(set(small_vids))
    if pool_type == "train_pool" and len(small_vids) > TRAIN_MAX_TOTAL:
        small_vids = random.sample(small_vids, TRAIN_MAX_TOTAL)
        print(f"⚠️  训练集抽样数超过上限 {TRAIN_MAX_TOTAL}，随机截取到 {TRAIN_MAX_TOTAL} 个")
    elif pool_type == "test_pool" and len(small_vids) > TEST_MAX_TOTAL:
        small_vids = random.sample(small_vids, TEST_MAX_TOTAL)
        print(f"⚠️  测试集抽样数超过上限 {TEST_MAX_TOTAL}，随机截取到 {TEST_MAX_TOTAL} 个")

    return small_vids, sample_log


def create_small_annotations():
    """生成小数据集标注（含：单独训练集+单独测试集+合并集）"""
    train_pool, test_pool, class_map, class_dist = parse_full_annotation()

    # 1. 抽样小训练集（从原validation训练池）
    print("[步骤1/2] 抽样小训练集...")
    small_train_vids, train_sample_log = sample_small_dataset(class_dist, "train_pool")

    # 2. 抽样小测试集（从原test测试池）
    print("[步骤2/2] 抽样小测试集...")
    small_test_vids, test_sample_log = sample_small_dataset(class_dist, "test_pool")

    # 打印抽样日志（核心需求）
    print("\n" + "="*80)
    print(f"📝 小数据集抽样日志（种子={RANDOM_SEED}，模式={SAMPLING_MODE}）")
    print("="*80)
    print(f"{'类别ID':<10} {'类别名称':<20} {'训练集（原val）':<25} {'测试集（原test）':<25}")
    print("-"*80)
    for label_id in sorted(class_map.keys()):
        label_name = class_map[label_id]
        # 训练集抽样信息
        train_total = train_sample_log[label_id]["total"]
        train_sampled = train_sample_log[label_id]["sampled"]
        train_info = f"原{train_total} → 抽{train_sampled}"
        # 测试集抽样信息
        test_total = test_sample_log[label_id]["total"]
        test_sampled = test_sample_log[label_id]["sampled"]
        test_info = f"原{test_total} → 抽{test_sampled}"
        print(f"{label_id:<10} {label_name:<20} {train_info:<25} {test_info:<25}")
    print("-"*80)
    print(f"{'总计':<10} {'-':<20} 原{len(train_pool)} → 抽{len(small_train_vids)} "
          f"{'':<5} 原{len(test_pool)} → 抽{len(small_test_vids)}")
    print("="*80 + "\n")

    # 构建标注数据（单独训练集+单独测试集+合并集）
    small_train_db = {vid: train_pool[vid] for vid in small_train_vids}
    small_test_db = {vid: test_pool[vid] for vid in small_test_vids}
    small_merged_db = {**small_train_db, **small_test_db}  # 合并训练+测试样本（新增）

    # 保持原标注格式
    small_train_ann = {"version": "Thumos14-30fps", "database": small_train_db}
    small_test_ann = {"version": "Thumos14-30fps", "database": small_test_db}
    small_merged_ann = {"version": "Thumos14-30fps", "database": small_merged_db}  # 合并集（新增）

    # 创建输出目录并保存
    os.makedirs(os.path.dirname(SMALL_TRAIN_ANNOT), exist_ok=True)
    with open(SMALL_TRAIN_ANNOT, "w", encoding="utf-8") as f:
        json.dump(small_train_ann, f, indent=2)
    with open(SMALL_TEST_ANNOT, "w", encoding="utf-8") as f:
        json.dump(small_test_ann, f, indent=2)
    with open(SMALL_MERGED_ANNOT, "w", encoding="utf-8") as f:  # 保存合并集（新增）
        json.dump(small_merged_ann, f, indent=2)

    print(f"✅ 小标注文件保存完成：")
    print(f"   - 单独训练集：{len(small_train_vids)} 个样本 → {SMALL_TRAIN_ANNOT}")
    print(f"   - 单独测试集：{len(small_test_vids)} 个样本 → {SMALL_TEST_ANNOT}")
    print(f"   - 合并集（训练+测试）：{len(small_merged_db)} 个样本 → {SMALL_MERGED_ANNOT}")  # 新增

    return small_train_vids, small_test_vids


def copy_small_features(small_train_vids, small_test_vids):
    """复制抽样后的特征文件"""
    all_small_vids = list(set(small_train_vids + small_test_vids))
    os.makedirs(SMALL_FEATURE_DIR, exist_ok=True)
    print(f"\n📥 开始复制 {len(all_small_vids)} 个特征文件...")

    copied_cnt = 0
    missing_cnt = 0
    missing_vids = []
    for vid in all_small_vids:
        # 兼容文件名（直接vid.npy 或 含vid的前缀文件名）
        src_file = os.path.join(FULL_FEATURE_DIR, f"{vid}.npy")
        if not os.path.exists(src_file):
            for fname in os.listdir(FULL_FEATURE_DIR):
                if fname.endswith(".npy") and vid in fname:
                    src_file = os.path.join(FULL_FEATURE_DIR, fname)
                    break
        # 复制文件
        dst_file = os.path.join(SMALL_FEATURE_DIR, os.path.basename(src_file))
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            copied_cnt += 1
        else:
            missing_cnt += 1
            missing_vids.append(vid)

    # 输出复制结果
    print(f"✅ 特征复制完成：")
    print(f"   - 成功复制：{copied_cnt} 个文件")
    if missing_cnt > 0:
        print(f"   - 缺失文件：{missing_cnt} 个（示例：{', '.join(missing_vids[:5])}...）")


def main():
    print("="*60)
    print(f"📌 小数据集生成工具（模式：{SAMPLING_MODE}，种子：{RANDOM_SEED}）")
    print("="*60)

    # 固定种子（保证可复现）
    random.seed(RANDOM_SEED)

    try:
        # 步骤1：检查路径
        check_input_paths()
        # 步骤2：生成小标注（含合并集）
        small_train_vids, small_test_vids = create_small_annotations()
        # 步骤3：复制特征文件
        copy_small_features(small_train_vids, small_test_vids)

        # 最终提示（更新配置参考）
        print("\n" + "="*60)
        print("🎉 小数据集生成成功！")
        print(f"📁 小数据集目录：{SMALL_DATA_ROOT}")
        print("💡 配置文件修改参考（推荐用合并集）：")
        print(f"   训练+实时验证：json_file = {SMALL_MERGED_ANNOT}（模型自动筛选训练/验证样本）")
        print(f"   单独训练：json_file = {SMALL_TRAIN_ANNOT}")
        print(f"   单独测试：json_file = {SMALL_TEST_ANNOT}")
        print(f"   feat_folder: {SMALL_FEATURE_DIR}")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 生成失败：{str(e)}")
        raise


if __name__ == "__main__":
    main()