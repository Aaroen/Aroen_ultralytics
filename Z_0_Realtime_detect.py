from ultralytics import YOLO
import os

# 初始化模型
model = YOLO('yolo12n.pt')  # 自动下载官方模型

# 实时摄像头检测
results = model.predict(
    source=0,
    show=True,
    conf=0.5,
    save=True
)
