import pandas as pd
import numpy as np
import os

# 設定資料路徑
BASE_DIR = r'D:\VS\機器學習概論\Baseball_ML'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

def process_pitching_data(pitching_files):
    print("正在處理投手數據 (計算 ERA, WHIP)...")
    
    # 讀取所有年份的 pitching.csv
    p_dfs = []
    for f in pitching_files:
        if os.path.exists(f):
            df = pd.read_csv(f, low_memory=False)
            p_dfs.append(df)
            
    if not p_dfs:
        return pd.DataFrame()
        
    p_df = pd.concat(p_dfs, ignore_index=True)
    
    # 1. 只篩選先發投手 (p_seq == 1)
    # 這是最關鍵的一步，我們只關心先發投手的數據
    starters = p_df[p_df['p_seq'] == 1].copy()
    
    # 確保日期格式與排序
    starters['date'] = pd.to_datetime(starters['date'])
    starters = starters.sort_values(by=['id', 'date'])
    
    # 2. 特徵工程：計算「賽前」累積數據 (Avoid Data Leakage)
    # 我們使用 expanding().sum() 來計算生涯(或該區間)累積，使用 shift(1) 排除當場
    
    # 分組運算 (Group by Player ID)
    grouped = starters.groupby('id')
    
    # 累積 責失分 (ER)
    starters['cum_er'] = grouped['p_er'].transform(lambda x: x.shift(1).expanding().sum())
    # 累積 抓到的出局數 (IPOuts) -> 局數 = IPOuts / 3
    starters['cum_ipouts'] = grouped['p_ipouts'].transform(lambda x: x.shift(1).expanding().sum())
    # 累積 被安打 + 保送 (for WHIP)
    starters['cum_walks_hits'] = grouped['p_h'].transform(lambda x: x.shift(1).expanding().sum()) + \
                                 grouped['p_w'].transform(lambda x: x.shift(1).expanding().sum())

    # 3. 計算指標
    # ERA = (Earned Runs * 27) / IPOuts  (註：標準公式是 *9 / IP，這裡 IPOuts 是局數*3)
    starters['sp_era'] = (starters['cum_er'] * 27) / starters['cum_ipouts']
    
    # WHIP = (Walks + Hits) * 3 / IPOuts
    starters['sp_whip'] = (starters['cum_walks_hits'] * 3) / starters['cum_ipouts']
    
    # 處理除以 0 或 NaN 的情況 (給予聯盟平均水準的預設值)
    starters['sp_era'] = starters['sp_era'].fillna(4.50)
    starters['sp_whip'] = starters['sp_whip'].fillna(1.35)
    
    # 處理無限大 (剛出道第一場可能投不滿一局)
    starters = starters.replace([np.inf, -np.inf], 4.50)
    
    # 只保留需要的欄位進行合併
    # gid 是連接 Key, team 是為了確保對應正確
    return starters[['gid', 'team', 'id', 'sp_era', 'sp_whip']].rename(columns={'id': 'starter_id'})

def load_and_process_data():
    print(f"正在從 {DATA_DIR} 讀取資料...")
    all_team_dfs = []
    pitching_files = []
    
    for year in range(2013, 2025):
        t_file = os.path.join(DATA_DIR, f"{year}teamstats.csv")
        p_file = os.path.join(DATA_DIR, f"{year}pitching.csv")
        
        if os.path.exists(t_file):
            df = pd.read_csv(t_file, low_memory=False)
            df['season'] = year
            all_team_dfs.append(df)
        
        if os.path.exists(p_file):
            pitching_files.append(p_file)

    if not all_team_dfs:
        raise ValueError("錯誤：沒有讀取到任何 teamstats 資料。")

    # 合併球隊資料
    df = pd.concat(all_team_dfs, ignore_index=True)
    
    # --- 處理投手資料 ---
    sp_stats = process_pitching_data(pitching_files)
    
    # --- 資料清洗 ---
    df['date'] = pd.to_datetime(df['date'])
    df['gid'] = df['gid'].astype(str).str.strip()
    df['target'] = df['win'].apply(lambda x: 1 if str(x).upper() in ['Y', 'W', '1', 'TRUE'] else 0)
    df['vishome'] = df['vishome'].astype(str).str.lower().str.strip()
    
    # --- 合併投手資料 (Merge) ---
    # sp_stats 裡面的 team 是投手所屬隊伍
    print("正在合併投手數據...")
    
    # 確保 gid 格式一致
    sp_stats['gid'] = sp_stats['gid'].astype(str).str.strip()
    
    # 將投手數據併入主資料表
    # 我們用 gid 和 team 雙重鍵來確保對應到正確的球隊
    df = pd.merge(df, sp_stats, on=['gid', 'team'], how='left')
    
    # 填補缺失值 (如果某些比賽沒有投手紀錄)
    df['sp_era'] = df['sp_era'].fillna(4.50)
    df['sp_whip'] = df['sp_whip'].fillna(1.35)
    
    # --- 特徵工程 (球隊數據) ---
    df = df.sort_values(by=['team', 'season', 'date'])
    
    cols_to_roll = ['b_r', 'b_h', 'b_hr', 'p_r', 'p_h', 'd_e']
    for col in cols_to_roll:
        if col in df.columns:
            df[f'roll_{col}'] = df.groupby(['team', 'season'])[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
            )
        else:
            df[f'roll_{col}'] = 0
            
    df['cum_wins'] = df.groupby(['team', 'season'])['target'].transform(lambda x: x.shift(1).cumsum())
    df['cum_games'] = df.groupby(['team', 'season']).cumcount()
    df['pre_win_rate'] = np.where(df['cum_games'] > 0, df['cum_wins'] / df['cum_games'], 0.5)
    
    df = df.fillna(0)

    # --- 轉換對戰格式 ---
    print("正在合併主客隊資料...")
    home_df = df[df['vishome'] == 'h'].add_prefix('home_')
    vis_df = df[df['vishome'] == 'v'].add_prefix('vis_')
    
    matchups = pd.merge(home_df, vis_df, left_on='home_gid', right_on='vis_gid')

    if len(matchups) == 0:
        print("錯誤：合併後資料為空。")
    else:
        matchups.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"資料處理完成！已儲存至 {PROCESSED_DATA_PATH}")
        print("新增關鍵特徵: home_sp_era, vis_sp_era, home_sp_whip, vis_sp_whip")

if __name__ == "__main__":
    load_and_process_data()