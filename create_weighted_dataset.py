import pandas as pd
import shutil
import os
from tqdm import tqdm  # 进度条库，如果没有安装，后面会教你装

# --- 1. 配置路径 (根据你的截图适配) ---
# 都在当前目录下，所以不需要加 'screw3.0-Member A/' 前缀
WEIGHTS_FILE = 'weights.csv'             
SOURCE_IMG_DIR = 'dataset/images/train'   # 假设 YOLO 标准结构
SOURCE_LBL_DIR = 'dataset/labels/train'   # 假设 YOLO 标准结构
OUTPUT_BASE_DIR = 'dataset_weighted'      # 生成的新数据集文件夹名

def create_weighted_dataset():
    # 1. 检查输入文件是否存在
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ 错误: 找不到 {WEIGHTS_FILE}")
        return
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"❌ 错误: 找不到图片文件夹 {SOURCE_IMG_DIR}")
        print("   请确认 dataset 文件夹里是否有 images/train 结构？")
        return

    # 2. 准备输出目录 (如果已存在则清空，防止混淆)
    if os.path.exists(OUTPUT_BASE_DIR):
        print(f"⚠️  警告: 输出目录 {OUTPUT_BASE_DIR} 已存在，正在删除重建...")
        shutil.rmtree(OUTPUT_BASE_DIR)
    
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'images/train'))
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'labels/train'))
    
    # 3. 读取权重
    df = pd.read_csv(WEIGHTS_FILE)
    print(f"🚀 开始生成加权数据集...")
    print(f"   源目录: {SOURCE_IMG_DIR}")
    print(f"   目标目录: {OUTPUT_BASE_DIR}")
    
    total_files = 0
    
    # 4. 循环处理每张图
    # 尝试使用 tqdm 显示进度条，如果没装库则退化为普通循环
    try:
        iterator = tqdm(df.iterrows(), total=len(df), unit="img")
    except ImportError:
        iterator = df.iterrows()
        print("💡 提示: 安装 tqdm 库可以看到进度条 (pip install tqdm)")

    for _, row in iterator:
        filename = row['filename']
        weight = float(row['weight'])
        
        # 计算复制次数 (权重 3.0 = 复制 3 份)
        repeat_times = int(round(weight))
        if repeat_times < 1: repeat_times = 1
        
        # 构建源文件路径
        src_img_path = os.path.join(SOURCE_IMG_DIR, filename)
        
        # 推导 Label 文件名 (把 .jpg/.png 换成 .txt)
        file_name_no_ext = os.path.splitext(filename)[0]
        label_filename = file_name_no_ext + ".txt"
        src_lbl_path = os.path.join(SOURCE_LBL_DIR, label_filename)
        
        # 检查源图片是否存在
        if not os.path.exists(src_img_path):
            # 这里的 print 可能会刷屏，如果缺图多可以注释掉
            # print(f"⚠️ 跳过丢失图片: {filename}")
            continue
            
        # 开始复制 (核心逻辑)
        for i in range(repeat_times):
            # 生成新名字: 001.jpg -> 001_copy0.jpg, 001_copy1.jpg
            suffix = f"_copy{i}" if i > 0 else ""
            
            # 组合新文件名
            new_img_name = f"{file_name_no_ext}{suffix}{os.path.splitext(filename)[1]}"
            new_lbl_name = f"{file_name_no_ext}{suffix}.txt"
            
            dst_img_path = os.path.join(OUTPUT_BASE_DIR, 'images/train', new_img_name)
            dst_lbl_path = os.path.join(OUTPUT_BASE_DIR, 'labels/train', new_lbl_name)
            
            # 复制文件
            shutil.copy2(src_img_path, dst_img_path)
            
            # 如果有对应的标签文件，也复制一份
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)
            
            total_files += 1

    print("-" * 30)
    print(f"✅ 处理完成！")
    print(f"📊 原始图片数: {len(df)}")
    print(f"📈 加权后总数: {total_files} (扩充了 {total_files - len(df)} 张)")
    print(f"📂 新数据集位置: {os.path.abspath(OUTPUT_BASE_DIR)}")

if __name__ == "__main__":
    create_weighted_dataset()