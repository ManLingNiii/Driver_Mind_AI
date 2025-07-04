# BlindGuard 專案工作日誌（JOURNAL.md）

本文件用於記錄各階段的專案進度、問題、調整紀錄與重要學習。

---

```
## 2025-5-20（二）

### 記錄人員：Eric

### 今日進度：
- 決定專案初版架構圖，架構圖檔可參照 docs/images/architecture_v1.png
- 成功轉換 BDD100K JSON ➝ YOLO txt 格式
- 決定YOLO v8 影像標籤種類，並建立bdd100k.yaml
- 將 convert_bdd_to_yolo.py 移至 datasets/ 並優化路徑
- 更改專案名稱為 Driver Mind AI

---
15:30跟老師討論後，改成以下方式實作：

1.定義 ROI（Region of Interest）區域座標
    •可用 OpenCV 或程式碼畫出 trapezoid/rectangle
    •例如設定風險區為畫面中下方 1/3 的區域
    
2.每幀 YOLO 推論出 bounding box
    •判斷物件的~~中心點~~(可自定義碰撞體積)是否落在 ROI 區域內
    
3.分區等級邏輯（舉例）
    •ROI 分為紅區（近）、橙區（中）、綠區（遠）
    •根據落點層級輸出「高風險」「中風險」回饋
    
4.加值功能（選用）：
    •可用 OpenCV cv2.polylines / cv2.fillPoly 視覺化警示區域
    •用 pyttsx3 或文字輸出提示：「前方高風險物件靠近」
---

### 學習內容：
- 了解 YOLO 格式標註需為中心點與寬高（normalized）
- 釐清YOLO 標籤框的碰撞交集判定(IoU)，決定好所使用的標籤類別

### 問題與修復：
- 錯誤：join() argument must not be NoneType
- 解法：從 JSON 檔名推回圖片名稱，而不是取錯誤的欄位

### 待辦事項：
- [ ] 台灣行車資料收集與特徵工程
- [ ] 整合 pyttsx3 警示語音模組
- [ ] 加入臉部視覺辨識
- [ ] 加入模型訓練流程（YOLOv8 CLI）
- [ ] 訓練日誌與 checkpoint 儲存
```

---

```
## 2025-5-21（三）

### 記錄人員：劉元新

### 今日進度：
- BDD100K資料集過於龐大，經測試colab環境難以負荷訓練，討論後改用BDD10K訓練。
- 決定YOLO v8 影像標籤種類，並建立bdd10k.yaml
- 討論後定義各類別台灣行車影像畫框範圍(在roboflow平台實作)，統一規範標記格式
- 開始進行roboflow切幀、畫框作業，製作五百張台灣行車紀錄
- 初步進行資料集EDA
- 用segmentation嘗試訓練，結果CPU無法負荷需要GPU加速，討論後改用openCV定義警戒區範圍。
- 用openCV依據分隔線做警示區 
---

### 學習內容：
- 了解模型訓練的環境資源配置/資料集大小影響
- 釐清類別畫框區域（影響後續物件偵測靈敏度與物件碰撞範圍）
- 模型訓練五百張耗費一小時，最終決定用三千到五千張資料做最終訓練
- 嘗試釐清風險RoI的判斷標準


### 問題與修復：
- 

### 待辦事項：
- [ ] 台灣行車資料收集與特徵工程
- [ ] 整合 pyttsx3 警示語音模組
- [ ] 加入臉部視覺辨識
- [ ] 加入模型訓練流程（YOLOv8 CLI）
- [ ] 訓練日誌與 checkpoint 儲存
- [ ] 測試台灣行車紀錄五百份資料訓練
- [ ] 前端展示
- [ ] 風險判定
```
```

---

```
## 2025-5-22（四）

### 今日進度：
-  將臉部辨識從dlib改成mediapipe，解決戴眼鏡時辨識率低的問題
-  建立GUI，可用按鈕選擇開啟臉部辨識與車道偵測
-  統整第一次 Roboflow 模型結果及改善方法
-  增加影像標註統一格式的事項
-  風險區域分析與評估
-  優化ROI
-  今日roboflow上Annotating數量100張         
-  整理BDD10K資料集，與製作隨機批次抽取腳本
-  下週請假，交接相關工作給旻翰
-  今晚預計更新雲端硬碟共享資料夾
-  模型訓練模板（含自動化調參）
-  資料集EDA模板


-  新增3部夜間行車紀錄器影片至roboflow,　總資料量約2600張
-  新增1部自行車影片至roboflow, 資料量約1300張
-  今日roboflow上Annotating 數約400張

### 學習內容：

### 待辦事項：
   

### 2025-5-23（五）

### 今日進度：
-  臉部偵測計算使用者打哈欠次數
-  建立工作流程圖與會議紀錄
-  roboflow 照片標註
-  風險影片跳針處理（一次跳5幀）
-  動態ROI
-  語音模組基本架構討論
-  研究roboflow複製方法


### 2025-5-27（二）

### 今日進度：
-  將疲勞辨識與車道風險辨識模組化(變成函式)
-  使用全域變數讓疲勞辨識與車道風險辨識可以互通資訊
-  利用thread平行處理疲勞辨識與車道風險辨識
-  利用wait讓疲勞辨識先完成初始化再開啟車道風險辨識
-  解決yolov8無法使用evolve問題(改用tune)，解決超參數問題
-  roboflow 照片標註

### 2025-5-28（三）

### 今日進度：
-  讀取車道風險辨識檔案路徑會有問題->將driver_risk_alert_system資料夾放入GUI資料夾中
-  將疲勞辨識與車道風險辨識模組化(變成函式)
-  使用全域變數讓疲勞辨識與車道風險辨識可以互通資訊
-  利用thread平行處理疲勞辨識與車道風險辨識
-  利用wait讓疲勞辨識先完成初始化再開啟車道風險辨識
-  Roboflow 影像標註
-  YOLOv8 tune參數 100 epochs 模型訓練
-  BERT測試 (跟老師開會後不採用)
-  語音模組測試
-  製做並串接driver_risk_alert和GUI的提示音（尚未測試教室無音響）
-  會議記錄


### 2025-5-29（四）

### 今日進度：
-  測試ultrafast->對於台灣較不適合
-  將按鈕加入gui，可以個別暫停臉部辨識與車道辨識
-  Roboflow 影像標註資料集優化及調整
-  openCV ROI 優化及 debug
-  測試聲音功能，目前已能發出聲音，但會閃退且沒出現Error，需再debug
-  標註roboflow



### 2025-6-03（二）

### 今日進度：
-  修改程式使用GPU
-  tkinter造成畫面卡頓，換成使用opencv
-  整合加入聲音
-  Roboflow 影像標註修正
-  測試1280*1280影像格式的模型訓練
-  測試yolov8
-  優化速度模組
-  改善ROI車道線偏移問題
-  協助模型訓練
-   語音功能模組更換從playsound更換至pygame，目前順利執行無Bug，已交給瑋傑串接
