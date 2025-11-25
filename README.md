MLB Game Prediction & Season Simulation System
MLB 賽事勝率預測與 2025 賽季模擬系統
這是一個基於機器學習 (Machine Learning) 與深度學習 (Deep Learning) 的全端數據分析專案。本系統利用 2013 至 2024 年的 MLB 歷史數據，透過特徵工程提取關鍵指標，並訓練 Random Forest、XGBoost 與 Keras (Neural Network) 三種模型來預測單場比賽勝率。此外，系統還具備「賽季模擬」功能，能依據球隊實力模擬 2025 年完整賽季與季後賽流程，預測總冠軍。
![alt text](https://img.shields.io/badge/Python-3.8%2B-blue)
![alt text](https://img.shields.io/badge/Flask-2.0%2B-green)
![alt text](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![alt text](https://img.shields.io/badge/XGBoost-Latest-red)
✨ 功能亮點 (Features)
多模型集成預測：同時展示 Random Forest、XGBoost 與 Deep Learning 的勝率預測結果，提供不同角度的信心水準。
進階特徵工程：
引入 先發投手 (Starting Pitcher) 歷史數據 (ERA, WHIP)。
計算 畢達哥拉斯期望勝率 (Pythagorean Expectation)。
計算球隊近況 (Rolling Stats) 與對戰數據差異 (Differentials)。
完整賽季模擬：
模擬 162 場例行賽 (依分區/聯盟權重排賽程)。
依據 MLB 規則決定分區冠軍與外卡資格。
模擬完整季後賽樹狀圖 (外卡戰 -> 分區賽 -> 冠軍賽 -> 世界大賽)。
視覺化 Web 介面：使用 Flask 搭建，操作直觀，支援即時對戰分析與重新模擬賽季。
📂 檔案結構說明 (Project Structure)
code
Text
Baseball_ML/
│
├── data/                   # 原始數據與處理後的資料夾
│   ├── 2013gameinfo.csv ~ 2024gameinfo.csv  # 比賽資訊
│   ├── 2013teamstats.csv ~ 2024teamstats.csv # 球隊攻守數據
│   ├── 2013pitching.csv ~ 2024pitching.csv   # 投手詳細數據
│   └── processed_data.csv  # [自動生成] 經過清洗與特徵工程後的訓練資料
│
├── models/                 # [自動生成] 訓練好的模型檔案
│   ├── rf_model.pkl        # Random Forest 模型
│   ├── xgb_model.pkl       # XGBoost 模型
│   ├── keras_model.h5      # Keras 深度學習模型
│   └── scaler.pkl          # 數據標準化處理器 (StandardScaler)
│
├── src/                    # 核心程式碼
│   ├── data_processing.py  # 資料清洗、特徵提取、合併資料 (ETL)
│   ├── train_models.py     # 模型訓練、評估與儲存
│   └── team_info.py        # 定義 MLB 球隊、聯盟與分區結構
│
├── templates/              # Web 前端頁面
│   └── index.html          # 主頁面 (包含預測表單與模擬結果)
│
├── app.py                  # Flask 啟動檔 (後端邏輯、路由、模擬演算法)
└── README.md               # 專案說明文件
🚀 安裝與執行 (Installation & Usage)
1. 環境設定
請確保已安裝 Python 3.8 以上版本，並安裝以下套件：
code
Bash
pip install pandas numpy scikit-learn xgboost tensorflow flask joblib
2. 資料準備
請將 2013~2024 年的 gameinfo.csv, teamstats.csv, pitching.csv 檔案放入 data/ 資料夾中。
3. 執行資料處理 (ETL)
執行此腳本以讀取原始 CSV，計算進階特徵（如滾動平均、投手 ERA），並產出 processed_data.csv。
code
Bash
python src/data_processing.py
4. 訓練模型
執行此腳本以訓練三種模型，並將訓練好的模型儲存至 models/ 資料夾。
code
Bash
python src/train_models.py
執行完畢後，你會看到各模型的準確率 (Accuracy) 與特徵重要性排行。
5. 啟動 Web 應用程式
啟動 Flask 伺服器。
code
Bash
python app.py
開啟瀏覽器訪問 http://127.0.0.1:5000 即可使用系統。
🧠 模型與方法論 (Methodology)
使用特徵 (Features)
本專案使用了 17 個關鍵特徵進行預測，重點在於捕捉「雙方差距」與「投手實力」：
實力差距 (Differentials)：勝率差、畢達哥拉斯期望勝率差、得失分差。
先發投手 (Starting Pitcher)：主客隊先發投手的生涯/賽季 ERA 與 WHIP，以及兩者差距。
球隊近況 (Recent Form)：過去 10 場的平均得分、失分、安打數、失誤數。
賽前狀態 (Pre-game Stats)：賽季至今的累積勝率。
模型選擇
XGBoost：目前表現最佳的模型，擅長處理表格型數據與特徵間的非線性關係。
Random Forest：作為基準模型 (Baseline)，提供穩定的預測結果。
Keras (Deep Learning)：使用多層感知機 (MLP) 架構，經過 StandardScaler 標準化後進行訓練。
模擬演算法
賽季模擬採用 蒙地卡羅方法 (Monte Carlo Simulation)：
不直接使用固定勝率，而是根據模型預測的機率進行隨機擲骰 (random() < prob)。
模擬 162 場賽程，同分區對手權重較高。
季後賽依照真實 MLB 規則（外卡 -> 分區賽 -> 冠軍賽 -> 世界大賽）進行樹狀對決。
📝 資料來源
資料格式基於 Retrosheet 
