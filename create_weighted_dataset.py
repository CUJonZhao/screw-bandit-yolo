import pandas as pd
import shutil
import os
from tqdm import tqdm  

WEIGHTS_FILE = 'weights.csv'             
SOURCE_IMG_DIR = 'dataset/images/train'   
SOURCE_LBL_DIR = 'dataset/labels/train'   
OUTPUT_BASE_DIR = 'dataset_weighted'      

def create_weighted_dataset():
    if not os.path.exists(WEIGHTS_FILE):
        print(f"can't find {WEIGHTS_FILE}")
        return
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"can not find {SOURCE_IMG_DIR}")
        return

    if os.path.exists(OUTPUT_BASE_DIR):
        print(f" {OUTPUT_BASE_DIR} already exist")
        shutil.rmtree(OUTPUT_BASE_DIR)
    
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'images/train'))
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'labels/train'))
    
    df = pd.read_csv(WEIGHTS_FILE)
   
    total_files = 0
    
    try:
        iterator = tqdm(df.iterrows(), total=len(df), unit="img")
    except ImportError:
        iterator = df.iterrows()

    for _, row in iterator:
        filename = row['filename']
        weight = float(row['weight'])
        
        repeat_times = int(round(weight))
        if repeat_times < 1: repeat_times = 1

        src_img_path = os.path.join(SOURCE_IMG_DIR, filename)
        
        file_name_no_ext = os.path.splitext(filename)[0]
        label_filename = file_name_no_ext + ".txt"
        src_lbl_path = os.path.join(SOURCE_LBL_DIR, label_filename)
        
        if not os.path.exists(src_img_path):
            continue
            
        for i in range(repeat_times):
        
            suffix = f"_copy{i}" if i > 0 else ""
            
            new_img_name = f"{file_name_no_ext}{suffix}{os.path.splitext(filename)[1]}"
            new_lbl_name = f"{file_name_no_ext}{suffix}.txt"
            
            dst_img_path = os.path.join(OUTPUT_BASE_DIR, 'images/train', new_img_name)
            dst_lbl_path = os.path.join(OUTPUT_BASE_DIR, 'labels/train', new_lbl_name)
            
            shutil.copy2(src_img_path, dst_img_path)
            
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)
            
            total_files += 1


if __name__ == "__main__":
    create_weighted_dataset()
