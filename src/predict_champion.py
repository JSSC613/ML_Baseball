#src/predict_champion.py
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import os
import itertools

# --- 設定路徑 ---
BASE_DIR = r'D:\VS\機器學習概論\Baseball_ML'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'keras_model.h5')

def get_latest_team_stats(df, season=2024):
    """
    取得指定賽季每支球隊『最後一場比賽』的數據狀態
    """
    # 篩選該賽季 (確保 home_season 存在)
    if 'home_season' not in df.columns:
        print("❌ 錯誤：資料中找不到 'home_season' 欄位")
        return {}
        
    df_season = df[df['home_season'] == season].copy()
    
    if len(df_season) == 0:
        print(f"警告：找不到 {season} 年的資料。")
        return {}

    # --- 關鍵修正：處理日期欄位 ---
    # processed_data.csv 裡面的日期欄位叫 'home_date'
    if 'date' not in df_season.columns:
        if 'home_date' in df_season.columns:
            df_season['date'] = pd.to_datetime(df_season['home_date'])
        else:
            print("❌ 錯誤：找不到日期欄位 (date 或 home_date)")
            return {}
    
    # 依日期排序，確保我們抓到的是最後一場
    df_season = df_season.sort_values('date')
    
    team_stats = {}
    
    # 基礎體質數據
    stat_cols = ['pre_win_rate', 'roll_b_r', 'roll_p_r', 'roll_b_h', 'roll_d_e']
    
    for idx, row in df_season.iterrows():
        # 更新主隊數據
        h_team = row['home_team']
        team_stats[h_team] = {col: row[f'home_{col}'] for col in stat_cols}
        
        # 手動計算 pyth
        h_r = row['home_roll_b_r']
        h_ra = row['home_roll_p_r']
        denom_h = (h_r**1.83 + h_ra**1.83)
        team_stats[h_team]['pyth'] = (h_r**1.83) / denom_h if denom_h > 0 else 0.5

        # 更新客隊數據
        v_team = row['vis_team']
        team_stats[v_team] = {col: row[f'vis_{col}'] for col in stat_cols}
        
        v_r = row['vis_roll_b_r']
        v_ra = row['vis_roll_p_r']
        denom_v = (v_r**1.83 + v_ra**1.83)
        team_stats[v_team]['pyth'] = (v_r**1.83) / denom_v if denom_v > 0 else 0.5
        
    return team_stats

def simulate_2025_season():
    print(f"正在載入 {DATA_PATH} ...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ 錯誤：找不到資料檔，請先執行 data_processing.py")
        return
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ 錯誤：找不到模型檔，請先執行 train_models.py")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. 獲取每隊 2024 年底的最終實力
    print("正在提取 2024 球隊最終狀態...")
    team_stats_db = get_latest_team_stats(df, season=2024)
    teams = list(team_stats_db.keys())
    
    if not teams:
        print("❌ 錯誤：無法提取球隊數據，請檢查 CSV 內容或年份。")
        return
        
    print(f"已載入 {len(teams)} 支球隊數據。")
    print("正在進行 2025 賽季全循環模擬 (Round Robin Simulation)...")

    # 2. 產生所有可能的對戰組合
    matchups = list(itertools.permutations(teams, 2))
    simulation_data = []
    
    for home, vis in matchups:
        h_s = team_stats_db[home]
        v_s = team_stats_db[vis]
        
        # 計算對戰特徵 (必須與 train_models.py 順序一致)
        diff_win_rate = h_s['pre_win_rate'] - v_s['pre_win_rate']
        diff_pyth = h_s['pyth'] - v_s['pyth']
        diff_run_diff = (h_s['roll_b_r'] - h_s['roll_p_r']) - (v_s['roll_b_r'] - v_s['roll_p_r'])
        
        row = [
            diff_win_rate, diff_pyth, diff_run_diff,
            h_s['pre_win_rate'], v_s['pre_win_rate'],
            h_s['pyth'], v_s['pyth'],
            h_s['roll_b_r'], v_s['roll_b_r'],
            h_s['roll_p_r'], v_s['roll_p_r'],
            h_s['roll_b_h'], v_s['roll_b_h'],
            h_s['roll_d_e'], v_s['roll_d_e']
        ]
        simulation_data.append(row)

    # 3. 載入模型並預測
    print("載入模型中...")
    try:
        scaler = joblib.load(SCALER_PATH)
        model = load_model(MODEL_PATH)
        
        X_sim = np.array(simulation_data)
        X_scaled = scaler.transform(X_sim)
        
        print("計算勝率中...")
        probs = model.predict(X_scaled, verbose=0).flatten()
        
    except ValueError as e:
        print("\n!!!! 發生錯誤 !!!!")
        print(f"錯誤訊息: {e}")
        print("這通常是因為特徵數量不符。請務必重新執行 train_models.py")
        return
    
    # 4. 統計積分
    leaderboard = {team: 0 for team in teams}
    
    for (home, vis), p_home_win in zip(matchups, probs):
        leaderboard[home] += p_home_win
        leaderboard[vis] += (1.0 - p_home_win)
        
    # 5. 排序與顯示
    scale_factor = 162 / (len(teams) - 1) / 2 
    sorted_ranking = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("🏆 2025 MLB 賽季模擬預測 (基於 2024 最終數據 + Keras)")
    print("="*60)
    print(f"{'排名':<5} {'球隊':<20} {'模擬積分':<10} {'預估勝場':<10}")
    print("-" * 60)
    
    for rank, (team, score) in enumerate(sorted_ranking, 1):
        proj_wins = score * scale_factor
        print(f"{rank:<5} {team:<20} {score:<10.2f} {proj_wins:<10.1f}")
    print("="*60)

if __name__ == "__main__":
    simulate_2025_season()