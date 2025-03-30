from ultralytics import YOLO
import os

# 解决路径空格问题（针对Windows的program files目录）
os.chdir(os.path.dirname(__file__))

# 初始化模型
model = YOLO('yolov8n.pt')  # 自动下载官方模型

# 实时摄像头检测
results = model.predict(
    source=0,
    show=True,
    conf=0.5,
    save=True
)

# 导出为ONNX
model.export(format='onnx')