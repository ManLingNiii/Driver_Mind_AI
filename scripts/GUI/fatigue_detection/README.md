# Drowsiness Detection Using MediaPipe

本專案使用 OpenCV 與 Google MediaPipe 的 Face Mesh 功能，實作即時「駕駛疲勞與打哈欠偵測」系統。透過眼睛與嘴巴的特徵點計算 EAR（Eye Aspect Ratio）與 MAR（Mouth Aspect Ratio），可有效辨識閉眼、打瞌睡與打哈欠狀況。

## 📌 功能特色

- 🔍 使用 MediaPipe Face Mesh 進行臉部特徵點偵測（精準且效能佳）
- 👁 自動校準個人 EAR 閾值，提升每位使用者的準確性
- 😴 閉眼超過 2 秒會觸發「DROWSINESS」警示
- 😮 打哈欠超過 3 次會觸發「YAWN」警示
- 🔄 按下空白鍵可隨時重置校準狀態
- 🖥 即時在畫面中顯示警示與哈欠次數

## 🧱 專案結構

```
drowsiness_detection_mediapipe.py  # 主程式，包含所有邏輯
```

## 🚀 如何執行

可單獨執行主程式：

```bash
python drowsiness_detection_mediapipe.py
```

畫面會即時顯示來自攝影機的畫面、校準資訊、偵測狀態與警示訊息。

## 🎯 應用場景

- 駕駛安全監測系統
- 長途司機疲勞預警
- 學習/工作過勞偵測
- 健康行為監控研究

## 🔑 關鍵技術

- **MediaPipe Face Mesh**：提取 468 個臉部特徵點
- **EAR（Eye Aspect Ratio）**：判斷眼睛是否閉合
- **MAR（Mouth Aspect Ratio）**：判斷是否張口打哈欠
- **自動校準機制**：前 3 秒收集 EAR 平均作為基準


