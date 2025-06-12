# YOLOv8 模型訓練與資料探索模組

本專案包含兩個主要的 Jupyter Notebook，針對自訂資料集進行探索性資料分析（EDA）與 YOLOv8 模型的訓練流程說明。

---

## 📁 Notebook 模組說明

### `dataset_eda.ipynb`
此 Notebook 用於分析標註完成的資料集，功能包含：
- 圖像與標籤分布統計
- 類別數量可視化
- 圖片大小、比例、長寬統計
- 標註框數量與面積分布
- YOLO 格式標註可視化（含座標解析）

適合於確認資料集是否平衡、標註是否異常。

---

### `YOLOv8 模型訓練模組.ipynb`
此 Notebook 用於訓練 YOLOv8 模型（使用 Ultralytics 提供的 API）：
- 模型選擇（n/s/m/l/x）
- 訓練參數設定（batch size, epochs, img size）
- 使用 YOLOv8 `train()` 指令訓練自訂資料集
- 顯示訓練過程圖與 metrics（精確度、mAP 等）
- 儲存最佳模型（`best.pt`）

---

## 🔧 執行環境與依賴套件

建議安裝 Ultralytics 官方套件與必要依賴：

```bash
pip install ultralytics opencv-python matplotlib pandas seaborn
```

---

## 📦 資料與模型結構建議

```
project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset_eda.ipynb
├── YOLOv8 模型訓練模組.ipynb
└── runs/
    └── detect/
        └── train/   ← 訓練結果自動儲存
```

---

## 🧑‍💻 適用對象

- YOLO 初學者進行模型訓練練習
- 專題開發使用者需了解資料集分布與訓練成效
- 快速建立小型物件偵測應用原型

---

## 📌 備註

- 訓練前請確認 `data.yaml` 路徑與資料集格式正確
- 若使用 GPU 訓練，請安裝正確的 CUDA + PyTorch 版本

