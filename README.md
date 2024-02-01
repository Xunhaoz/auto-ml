# auto-ml-api

📦 是一款整合各類機器學習模型的機器學習預測策平台

## 安裝

1. 克隆倉庫到本地

    ```shell
    git clone https://github.com/username/repo.git
    ```

2. 進入專案目錄

    ```shell
    cd repo
    ```

3. 安裝依賴

    ```shell
    pip install -r requirements.txt
    ```

## 使用

1. 啟動API伺服器

    ```shell
    python app.py
    ```

2. 測試API連接

    ```shell
    curl http://localhost/api/
    ```

## API

### 測試連接

- 請求方法：GET
- 路徑：/api/
- 功能：測試API連接

### 上傳CSV文件

- 請求方法：POST
- 路徑：/api/upload_csv
- 功能：上傳CSV文件
- 參數：
   - file: 上傳的CSV文件
   - project_name: 專案名稱

### 取得CSV檔案信息

- 請求方法：GET
- 路徑：/api/csv_info
- 功能：取得CSV檔案信息
- 參數：
   - file_id: 文件ID

### 取得所有CSV文件

- 請求方法：GET
- 路徑：/api/all_csv
- 功能：取得所有CSV文件

### 取得CSV檔案相關性圖

- 請求方法：GET
- 路徑：/api/csv_corr
- 功能：取得CSV檔案相關性圖
- 參數：
   - file_id: 文件ID

### 訓練模型

- 請求方法：POST
- 路徑：/api/train_model
- 功能：訓練模型
- 參數：
   - file_id: 文件ID
   - mission_type: 任務類型
   - feature: 特徵列表
   - label: 標籤

### 取得訓練進度

- 請求方法：GET
- 路徑：/api/train_progressing
- 功能：取得訓練進度
- 參數：
   - file_id: 文件ID

### 取得訓練結果

- 請求方法：GET
- 路徑：/api/train_result
- 功能：取得訓練結果
- 參數：
   - file_id: 文件ID

### 取得訓練圖表

- 請求方法：GET
- 路徑：/api/train_pic
- 功能：取得訓練圖表
- 參數：
   - file_id: 文件ID
