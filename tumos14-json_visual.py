import json
import pandas as pd
from typing import Dict, List, Optional

# ===================== 新增：动作类别中英文映射字典 =====================
ACTION_CN_MAP = {
    "BaseballPitch": "棒球投球",
    "BasketballDunk": "篮球扣篮",
    "Billiards": "台球",
    "CleanAndJerk": "挺举（举重）",
    "CliffDiving": "悬崖跳水",
    "CricketBowling": "板球投球",
    "CricketShot": "板球击球",
    "Diving": "跳水",
    "FrisbeeCatch": "飞盘接盘",
    "GolfSwing": "高尔夫挥杆",
    "HammerThrow": "链球投掷",
    "HighJump": "跳高",
    "JavelinThrow": "标枪投掷",
    "LongJump": "跳远",
    "PoleVault": "撑杆跳高",
    "Shotput": "铅球投掷",
    "SoccerPenalty": "足球点球",
    "TennisSwing": "网球挥拍",
    "ThrowDiscus": "铁饼投掷",
    "VolleyballSpiking": "排球扣球",
    "未知动作": "未知动作"
}


def load_thumos_data(json_path: str) -> Dict:
    """加载Thumos14数据集JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载数据，数据集版本：{data.get('version', '未知')}")
        return data
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {json_path}")
        raise
    except json.JSONDecodeError:
        print(f"❌ 错误：{json_path} 不是有效的JSON文件")
        raise


def clean_annotations(annotations: List[Dict]) -> List[Dict]:
    """清洗标注数据（修正拼写错误、补全缺失字段）"""
    cleaned = []
    error_count = 0
    for idx, ann in enumerate(annotations):
        # 修正label_id拼写错误（原数据中"labe"→"label_id"）
        if 'labe' in ann and 'label_id' not in ann:
            ann['label_id'] = ann.pop('labe')
            error_count += 1
            print(f"⚠️  修正标注{idx + 1}的'label_id'拼写错误")
        # 补全缺失的label_id（若存在）
        if 'label_id' not in ann:
            ann['label_id'] = '未知'
            error_count += 1
            print(f"⚠️  标注{idx + 1}缺失label_id，已标记为'未知'")
        cleaned.append(ann)
    if error_count == 0:
        print("✅ 标注数据无明显错误")
    return cleaned


def calculate_action_duration(segment: List[float]) -> float:
    """计算动作持续时间（结束时间-开始时间）"""
    return round(segment[1] - segment[0], 2)


def analyze_thumos_data(data: Dict) -> Dict:
    """核心分析：提取视频信息、统计动作数据（包含中英文动作类别）"""
    database = data.get('database', {})
    analysis_result = {
        'total_videos': len(database),
        'videos': [],
        'action_summary': {},  # 动作类别统计：{动作名: {总次数: N, 总时长: T, 最长时长: M, 动作ID: ID}}
        'data_issues': []
    }

    for video_id, video_info in database.items():
        # 提取视频基础信息
        video_base = {
            'video_id': video_id,
            'subset': video_info.get('subset', '未知'),
            'duration': video_info.get('duration', 0.0),
            'fps': video_info.get('fps', 0.0),
            'annotation_count': len(video_info.get('annotations', []))
        }

        # 清洗标注并处理动作数据（包含中英文动作类别）
        annotations = clean_annotations(video_info.get('annotations', []))
        video_actions = []
        for ann in annotations:
            # 提取英文动作类别
            action_category_en = ann.get('label', '未知动作')
            # 映射中文动作类别
            action_category_cn = ACTION_CN_MAP[action_category_en]

            segment = ann.get('segment', [0.0, 0.0])
            duration = calculate_action_duration(segment)
            frame_segment = ann.get('segment(frames)', [0.0, 0.0])
            label_id = ann.get('label_id', '未知')

            # 记录单条动作数据（新增中文类别列）
            action_detail = {
                '动作类别（英文）': action_category_en,
                '动作类别（中文）': action_category_cn,  # 新增：中文翻译列
                '动作时间区间[始,末]': segment,
                '动作持续时间(秒)': duration,
                '动作帧区间[始,末]': frame_segment,
                '动作ID': label_id
            }
            video_actions.append(action_detail)

            # 更新动作类别统计
            if action_category_en not in analysis_result['action_summary']:
                analysis_result['action_summary'][action_category_en] = {
                    '总次数': 1,
                    '总时长': duration,
                    '最长单次时长': duration,
                    '动作ID': label_id,
                    '动作类别（中文）': action_category_cn
                }
            else:
                summary = analysis_result['action_summary'][action_category_en]
                summary['总次数'] += 1
                summary['总时长'] += duration
                if duration > summary['最长单次时长']:
                    summary['最长单次时长'] = duration

        # 整合视频数据
        video_base['actions'] = video_actions
        analysis_result['videos'].append(video_base)

    return analysis_result


def generate_report(analysis_result: Dict, save_excel: bool = True,
                    excel_path: str = 'thumos14_analysis_result.xlsx') -> None:
    """生成分析报告并保存Excel（包含中英文动作类别）"""
    print("\n" + "=" * 50)
    print("Thumos14-30fps数据集分析报告（含中文动作类别）")
    print("=" * 50)

    # 1. 基础统计
    print(f"\n1. 数据集基础信息")
    print(f"   - 总视频数：{analysis_result['total_videos']}")
    print(f"   - 动作类别数：{len(analysis_result['action_summary'])}")
    total_annotations = sum([v['annotation_count'] for v in analysis_result['videos']])
    print(f"   - 总标注动作数：{total_annotations}")

    # 2. 视频详情（含中英文动作类别）
    print(f"\n2. 各视频动作详情（示例：前3个视频）")
    for i, video in enumerate(analysis_result['videos'][:3]):  # 只打印前3个视频避免输出过长
        print(f"   - 视频ID：{video['video_id']}")
        print(
            f"     子集：{video['subset']} | 时长：{video['duration']}s | 帧率：{video['fps']} | 动作数：{video['annotation_count']}")
        for action in video['actions'][:3]:  # 每个视频只打印前3个动作
            print(
                f"     - 动作（中）：{action['动作类别（中文）']} | 动作（英）：{action['动作类别（英文）']} | 持续时长：{action['动作持续时间(秒)']}s | 动作ID：{action['动作ID']}")

    # 3. 动作类别统计（含中英文）
    print(f"\n3. 动作类别统计（完整）")
    for action_en, stats in analysis_result['action_summary'].items():
        print(
            f"   - 动作（中）：{stats['动作类别（中文）']} | 动作（英）：{action_en} | 动作ID：{stats['动作ID']} | 总次数：{stats['总次数']} | 总时长：{round(stats['总时长'], 2)}s")

    # 4. 保存Excel（字段包含中英文动作类别）
    if save_excel:
        # 构建视频动作详情表（匹配你的表格+新增中文列）
        video_data = []
        for video in analysis_result['videos']:
            for action in video['actions']:
                video_data.append({
                    '视频ID': video['video_id'],
                    '子集': video['subset'],
                    '视频时长(秒)': video['duration'],
                    '帧率': video['fps'],
                    '动作类别（英文）': action['动作类别（英文）'],
                    '动作类别（中文）': action['动作类别（中文）'],  # 新增中文列
                    '动作时间区间[始,末]': action['动作时间区间[始,末]'],
                    '动作持续时间(秒)': action['动作持续时间(秒)'],
                    '动作帧区间[始,末]': action['动作帧区间[始,末]'],
                    '动作ID': action['动作ID']
                })
        video_df = pd.DataFrame(video_data)

        # 构建动作统计汇总表（含中英文）
        action_stats_data = []
        for action_en, stats in analysis_result['action_summary'].items():
            action_stats_data.append({
                '动作类别（英文）': action_en,
                '动作类别（中文）': stats['动作类别（中文）'],
                '动作ID': stats['动作ID'],
                '总出现次数': stats['总次数'],
                '总持续时长(秒)': round(stats['总时长'], 2),
                '最长单次时长(秒)': stats['最长单次时长']
            })
        action_df = pd.DataFrame(action_stats_data)

        # 保存到Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            video_df.to_excel(writer, sheet_name='视频动作详情', index=False)
            action_df.to_excel(writer, sheet_name='动作统计汇总', index=False)
        print(f"\n✅ 分析结果已保存到Excel：{excel_path}")
        print(f"   - 新增列：动作类别（中文），包含所有动作的中文翻译")


if __name__ == "__main__":
    # --------------------------
    # 配置参数（用户可修改）
    # --------------------------
    JSON_FILE_PATH = "data/thumos/annotations/thumos14.json"  # 你的JSON文件路径
    SAVE_EXCEL_PATH = "data/thumos/annotations/thumos14.xlsx"  # 输出Excel路径

    # --------------------------
    # 执行分析流程
    # --------------------------
    try:
        # 1. 加载数据
        raw_data = load_thumos_data(JSON_FILE_PATH)
        # 2. 核心分析（含中英文动作类别）
        analysis_result = analyze_thumos_data(raw_data)
        # 3. 生成报告与保存结果
        generate_report(analysis_result, save_excel=True, excel_path=SAVE_EXCEL_PATH)
        print("\n🎉 分析完成！Excel文件中已包含中英文动作类别列")
    except Exception as e:
        print(f"\n❌ 分析失败：{str(e)}")