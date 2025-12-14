import pandas as pd
import numpy as np
import os

INPUT_FILE = 'state.csv'       
OUTPUT_FILE = 'weights.csv'   

ALPHA_FP = 2.0   
BETA_FN = 3.0    
GAMMA_CONF = 1.0 

WEIGHT_LEVELS = [
    (2.5, 3.0),  
    (1.0, 2.0),  
    (0.0, 1.0)   
]

def calculate_difficulty(row):
    """
     D = α*FP + β*FN + γ*(1-conf)
    """
    conf = row.get('avg_conf', 1.0)
    uncertainty = 1.0 - conf
    
    fp = row.get('FP', 0)
    fn = row.get('FN', 0)
    
    score = (ALPHA_FP * fp) + \
            (BETA_FN * fn) + \
            (GAMMA_CONF * uncertainty)
    return score

def get_weight_strategy(difficulty_score):
    """
    Contextual Bandit Policy
    """
    for threshold, weight in WEIGHT_LEVELS:
        if difficulty_score >= threshold:
            return weight
    return 1.0 

def main():
    print(f"current working category: {os.getcwd()}")
    
    if not os.path.exists(INPUT_FILE):
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"success")
    except Exception as e:
        print(f"fail to get {e}")
        return


    required_cols = ['FP', 'FN', 'avg_conf']
    for col in required_cols:
        if col not in df.columns:
            return

    df['difficulty'] = df.apply(calculate_difficulty, axis=1)

   
    df['weight'] = df['difficulty'].apply(get_weight_strategy)

    total_imgs = len(df)
    hard_samples = df[df['weight'] > 1.0]
    high_weight_ratio = len(hard_samples) / total_imgs if total_imgs > 0 else 0


    output_cols = ['filename', 'weight']
    if 'filename' not in df.columns:
        output_df = df
    else:
        output_df = df[output_cols]
        
    output_df.to_csv(OUTPUT_FILE, index=False)
   

if __name__ == "__main__":
    main()
