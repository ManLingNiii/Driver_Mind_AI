from ultralytics import YOLO
import os

# === 設定模型路徑 ===
weight_path = r"C:\Users\USER\Downloads\Driver_Mind_AI2_Modified\scripts\GUI\driver_risk_alert_system\weight\best3.pt"

# === 載入模型 ===
model = YOLO(weight_path)

# === 匯出 ONNX 模型（FP32）===
print("▶ 匯出 ONNX (FP32)...")
model.export(format="onnx", dynamic=True, simplify=True)
print("✅ ONNX FP32 匯出成功\n")

# === 匯出 ONNX 模型（FP16）===
print("▶ 匯出 ONNX (FP16)...")
model.export(format="onnx", half=True, dynamic=True, simplify=True)
print("✅ ONNX FP16 匯出成功\n")

# === 匯出 TFLite 模型（INT8 量化）===
print("▶ 匯出 TFLite (INT8)...")
try:
    model.export(format="tflite", int8=True)
    print("✅ TFLite 匯出成功\n")
except Exception as e:
    print("❌ TFLite 匯出失敗，請確認是否已安裝 tensorflow")
    print(e)
