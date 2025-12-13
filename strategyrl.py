import pandas as pd
import numpy as np
import os

# --- 1. 配置参数 (路径已修改适配当前文件夹) ---
# ✅ 修改点：直接使用文件名，因为脚本和csv在同一层级
INPUT_FILE = 'state.csv'       # 成员 A 生成的文件
OUTPUT_FILE = 'weights.csv'    # 你要生成给 C 的文件

# 难度系数权重 (根据 A 的情报：FP误报高，FN漏检低)
# 策略思路：重点惩罚误报(FP)，同时保持对漏检(FN)的高敏感度
ALPHA_FP = 2.0   # 误报惩罚
BETA_FN = 3.0    # 漏检惩罚
GAMMA_CONF = 1.0 # 不确定性惩罚 (1 - conf)

# 权重动作空间 (Bandit Actions)
# 含义: [难度分数阈值, 分配的权重]
# 逻辑: 如果难度 >= 阈值, 则分配对应权重
WEIGHT_LEVELS = [
    (2.5, 3.0),  # 极难样本 -> 3倍权重 (通常是严重误报或漏检)
    (1.0, 2.0),  # 困难样本 -> 2倍权重 (通常是置信度低或轻微误报)
    (0.0, 1.0)   # 普通样本 -> 1倍权重 (Base)
]

def calculate_difficulty(row):
    """
    计算每张图片的难度分数
    公式: D = α*FP + β*FN + γ*(1-conf)
    """
    # 归一化 conf 带来的不确定性 (conf 越低，uncertainty 越高)
    # 如果 avg_conf 是 0 (没检测到东西)，这里设为 1.0
    conf = row.get('avg_conf', 1.0)
    uncertainty = 1.0 - conf
    
    # 确保读取数值，防止空值报错
    fp = row.get('FP', 0)
    fn = row.get('FN', 0)
    
    score = (ALPHA_FP * fp) + \
            (BETA_FN * fn) + \
            (GAMMA_CONF * uncertainty)
    return score

def get_weight_strategy(difficulty_score):
    """
    根据难度分数选择权重 (简单的阈值策略 / Contextual Bandit Policy)
    """
    for threshold, weight in WEIGHT_LEVELS:
        if difficulty_score >= threshold:
            return weight
    return 1.0 # 默认权重

def main():
    print(f"📂 当前工作目录: {os.getcwd()}")
    
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到 {INPUT_FILE}")
        print("   请确认终端路径是否正确 (应该在 screw3.0-Member A 文件夹下运行)")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✅ 成功加载 {len(df)} 张图片的统计数据。")
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 2. 计算难度 (State)
    # 确保列名匹配，防止大小写差异
    # 假设 CSV 列名为: filename, gt, pred, TP, FP, FN, avg_conf
    required_cols = ['FP', 'FN', 'avg_conf']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ 警告: CSV 中缺少列 '{col}'，请检查 state.csv 格式。")
            return

    df['difficulty'] = df.apply(calculate_difficulty, axis=1)

    # 3. 分配权重 (Action)
    df['weight'] = df['difficulty'].apply(get_weight_strategy)

    # --- 数据分析 (供你自己检查策略是否合理) ---
    total_imgs = len(df)
    hard_samples = df[df['weight'] > 1.0]
    high_weight_ratio = len(hard_samples) / total_imgs if total_imgs > 0 else 0

    print("\n📊 策略分布分析:")
    print(f"  - 总图片数: {total_imgs}")
    print(f"  - 被加权的困难样本数 (w > 1): {len(hard_samples)}")
    print(f"  - 困难样本占比: {high_weight_ratio:.2%}")
    print(f"  - 平均难度分: {df['difficulty'].mean():.4f}")
    
    if high_weight_ratio > 0.4:
        print("⚠️ 提示: 加权样本有点多 (>40%)，建议在代码顶部调高 WEIGHT_LEVELS 的阈值。")
    elif high_weight_ratio < 0.05:
        print("⚠️ 提示: 加权样本太少 (<5%)，建议降低阈值，否则 RL 效果不明显。")
    else:
        print("✅ 提示: 加权比例适中 (5%-40%)，策略看起来很健康。")

    # 4. 保存结果
    # 只保留 filename 和 weight 给成员 C 即可
    output_cols = ['filename', 'weight']
    if 'filename' not in df.columns:
        # 兼容性处理：如果没有 filename 列，尝试用 image_id 或 index
        print("⚠️ 注意: 输入文件中没有 'filename' 列，尝试保存所有列。")
        output_df = df
    else:
        output_df = df[output_cols]
        
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n权重文件已保存至: {OUTPUT_FILE}")
   

if __name__ == "__main__":
    main()