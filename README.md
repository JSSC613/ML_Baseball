# ⚾ MLB 賽事預測與賽季模擬系統 (MLB Prediction & Season Simulation)

這是一個基於機器學習 (Machine Learning) 與深度學習 (Deep Learning) 的 MLB 賽事分析系統。透過分析 2013 至 2024 年的歷史數據（包含球隊攻守數據、先發投手數據），預測單場比賽的勝率，並透過蒙地卡羅模擬 (Monte Carlo Simulation) 預測 2025 年賽季的總冠軍。

## ✨ 功能特色 (Features)

*   **多模型預測**：同時使用 Random Forest、XGBoost 與 Deep Learning (Keras) 三種模型進行勝率分析，提供多元視角。
*   **進階特徵工程**：
    *   引入 畢達哥拉斯期望勝率 (Pythagorean Expectation)。
    *   計算對戰球隊的數據差異 (Differentials)。
    *   整合 先發投手 (Starting Pitcher) 的歷史防禦率 (ERA) 與 WHIP 數據，大幅提升預測準確度。
*   **完整賽季模擬**：
    *   模擬 162 場加權例行賽（同分區對戰權重較高）。
    *   依據 MLB 規則（分區冠軍、外卡）決定季後賽名單。
    *   模擬外卡賽、分區系列賽、聯盟冠軍賽至世界大賽的完整晉級流程。
*   **視覺化 Web 介面**：使用 Flask 建置網頁，提供直觀的操作介面與即時戰況模擬日誌。

## 📂 專案結構

```text
MLB_Prediction/
├── data/                   # 存放原始 CSV (gameinfo, teamstats, pitching) 與處理後的資料
├── models/                 # 存放訓練好的模型 (.pkl, .h5)
├── src/                    # 核心程式碼
│   ├── data_processing.py  # 資料清洗與特徵工程
│   ├── train_models.py     # 模型訓練與評估
│   └── team_info.py        # 球隊與聯盟結構定義
├── templates/              # Flask 網頁模板 (index.html)
├── app.py                  # Flask 啟動程式
└── requirements.txt        # 套件需求清單
```
## 🚀 安裝說明 (Installation)
```
1. 克隆專案 (Clone Repository)
code
Bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
```
2. 建立虛擬環境 (Optional but Recommended)
code
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
```
3. 安裝依賴套件
請確保你的環境已安裝 Python。
code
Bash
pip install -r requirements.txt
## 🛠️ 使用方法 (Usage)
本系統分為三個階段，請依序執行：
步驟 1：資料處理 (ETL)
讀取原始 CSV 檔案，清洗資料，解析先發投手，並計算滾動平均數據 (Rolling Stats)。
code
Bash
python src/data_processing.py
執行成功後，會在 data/ 資料夾產生 processed_data.csv。
步驟 2：模型訓練 (Training)
使用處理好的資料訓練 RF, XGBoost, Keras 模型，並進行標準化 (Scaler) 處理。
code
Bash
python src/train_models.py
執行成功後，會在 models/ 資料夾產生模型檔案。
步驟 3：啟動 Web 應用程式
啟動 Flask 伺服器。
code
Bash
python app.py
開啟瀏覽器輸入 http://127.0.0.1:5000 即可使用。
```
## 📊 使用的技術 (Tech Stack)
```
語言: Python
Web 框架: Flask, Bootstrap 5
資料處理: Pandas, NumPy
機器學習: Scikit-Learn (Random Forest), XGBoost
深度學習: TensorFlow / Keras
儲存: Joblib
```
## 📝 資料來源 : Retrosheet
