from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows多进程支持
    
    # 正确加载模型（保留一个有效的加载方式即可）
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 合并YAML配置和预训练权重
    
    # 训练模型
    # 建议优化项（在现有代码基础上新增）
    results = model.train(
        data="coco8.yaml",
        epochs=30,  # ← 增加训练轮次
        imgsz=640,
        device=0,
        workers=4,    # ← 适当启用多线程
        batch=32,     # ← 增大批次大小
        mixup=0.1     # ← 启用数据增强
    )