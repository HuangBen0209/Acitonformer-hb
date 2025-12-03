import os
from pathlib import Path

def count_test_val_npy(feat_folder):
    """
    仅通过 .npy 文件名统计 test 和 validation 样本数
    匹配规则：
    - test 样本：文件名包含 "test"（不区分大小写，如 video_test_xxx.npy）
    - validation 样本：文件名包含 "val"（不区分大小写，如 video_val_xxx.npy 或 video_validation_xxx.npy）
    """
    # 1. 检查特征文件夹是否存在
    if not os.path.exists(feat_folder):
        print(f"❌ 错误：找不到特征文件夹 {feat_folder}")
        return

    # 2. 初始化统计变量
    test_count = 0          # test 样本数
    val_count = 0           # validation 样本数
    other_count = 0         # 其他样本数（既不含 test 也不含 val）
    test_files = []         # 存储 test 文件名（可选，用于核对）
    val_files = []          # 存储 val 文件名（可选，用于核对）

    # 3. 遍历文件夹中的所有 .npy 文件
    for filename in os.listdir(feat_folder):
        # 只处理 .npy 后缀的文件
        if filename.lower().endswith('.npy'):
            filename_lower = filename.lower()  # 转为小写，避免大小写敏感
            if 'test' in filename_lower:
                test_count += 1
                test_files.append(filename)
            elif 'val' in filename_lower:  # 匹配 val 或 validation
                val_count += 1
                val_files.append(filename)
            else:
                other_count += 1

    # 4. 输出统计结果
    print("=" * 60)
    print("📊 .npy 文件分类统计结果（按文件名关键词）")
    print("=" * 60)
    print(f"特征文件夹路径：{feat_folder}")
    print("-" * 60)
    print(f"✅ 含 'test' 的 .npy 文件（test 样本）：{test_count:>4} 个")
    print(f"✅ 含 'val' 的 .npy 文件（validation 样本）：{val_count:>2} 个")
    print(f"⚠️  其他 .npy 文件（无 test/val 关键词）：{other_count:>2} 个")
    print(f"📝 总 .npy 文件数：{test_count + val_count + other_count:>8} 个")
    print("-" * 60)

    # 可选：显示前 5 个 test/val 文件名（方便核对）
    if test_files:
        print(f"\nTest 文件名示例（前 5 个）：")
        for i, fname in enumerate(test_files[:5], 1):
            print(f"  {i}. {fname}")
    if val_files:
        print(f"\nValidation 文件名示例（前 5 个）：")
        for i, fname in enumerate(val_files[:5], 1):
            print(f"  {i}. {fname}")
    print("=" * 60)

if __name__ == "__main__":
    # 自动获取脚本所在目录（项目根目录），拼接特征文件夹路径
    ROOT_PATH_DIR = os.path.dirname(os.path.abspath(__file__))
    # 特征文件夹路径：根目录 → data → i3d_features（根据你的实际路径修改）
    FEAT_FOLDER_PATH = os.path.join(
        ROOT_PATH_DIR,
        "data",
        "thumos",
        "i3d_features"  # 若你的特征文件夹路径不同，修改这里即可
    )

    # 打印路径供核对
    print(f"📌 项目根目录：{ROOT_PATH_DIR}")
    print(f"📌 特征文件夹路径：{FEAT_FOLDER_PATH}")
    print()

    # 调用函数统计
    count_test_val_npy(FEAT_FOLDER_PATH)