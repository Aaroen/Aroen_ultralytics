from ultralytics import YOLO
import os

os.chdir(os.path.dirname(__file__))

# 初始化分类模型 (使用预训练分类模型)
model = YOLO('yolov8m-cls.pt')  # 注意使用-cls后缀的分类模型

# 图片分类预测
results = model.predict(
    source='all\  (80).jpg',  # 替换为你的图片路径
    show=True,
    save=True,
    conf=0.5  # 显示置信度阈值
)

# （可选）输出分类结果
for result in results:
    top5 = result.probs.top5
    top5_conf = result.probs.top5_conf
    print(f"预测结果：{result.names[top5[0]]} (置信度：{top5_conf[0]:.2f})")

# 导出为ONNX（分类模型同样支持）
model.export(format='onnx')