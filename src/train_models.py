import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

BASE_DIR = r'D:\VS\機器學習概論\Baseball_ML'
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train():
    print(f"正在從 {PROCESSED_DATA_PATH} 載入資料...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("❌ 錯誤：找不到資料檔")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("正在進行特徵工程...")
    
    # 1. 畢達哥拉斯期望勝率
    df['home_pyth'] = (df['home_roll_b_r']**1.83) / ((df['home_roll_b_r']**1.83 + df['home_roll_p_r']**1.83) + 1e-9)
    df['vis_pyth'] = (df['vis_roll_b_r']**1.83) / ((df['vis_roll_b_r']**1.83 + df['vis_roll_p_r']**1.83) + 1e-9)
    
    # 2. 投手數據差異 (新特徵 🔥)
    # 負值表示主隊投手ERA較低(較好)，對模型來說是正向訊號
    df['diff_sp_era'] = df['home_sp_era'] - df['vis_sp_era']
    df['diff_sp_whip'] = df['home_sp_whip'] - df['vis_sp_whip']

    # 3. 其他差異特徵
    df['diff_win_rate'] = df['home_pre_win_rate'] - df['vis_pre_win_rate']
    df['diff_run_diff'] = (df['home_roll_b_r'] - df['home_roll_p_r']) - (df['vis_roll_b_r'] - df['vis_roll_p_r'])
    df['diff_pyth'] = df['home_pyth'] - df['vis_pyth']

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # 定義特徵列表 (現在有 17 個特徵)
    features = [
        'diff_win_rate', 'diff_pyth', 'diff_run_diff',
        'diff_sp_era', 'diff_sp_whip',  # 新增：投手差異
        'home_pre_win_rate', 'vis_pre_win_rate',
        'home_pyth', 'vis_pyth',
        'home_roll_b_r', 'vis_roll_b_r',
        'home_roll_p_r', 'vis_roll_p_r',
        'home_sp_era', 'vis_sp_era',    # 新增：投手ERA
        'home_sp_whip', 'vis_sp_whip'   # 新增：投手WHIP
    ]
    target = 'home_target'

    # 分割資料
    train_df, test_df = train_test_split(
        df, test_size=0.02, random_state=42, shuffle=True, stratify=df[target]
    )

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"訓練特徵數: {len(features)}")
    print(f"訓練資料: {len(X_train)} 筆")

    # --- 標準化 ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # --- XGBoost ---
    print("訓練 XGBoost...")
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.4f}")
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))

    # --- Random Forest ---
    print("訓練 Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))

    # --- Keras ---
    print("訓練 Keras...")
    model = Sequential([
        Input(shape=(len(features),)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)
    _, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Keras Accuracy: {acc:.4f}")
    model.save(os.path.join(MODEL_DIR, 'keras_model.h5'))

if __name__ == "__main__":
    train()