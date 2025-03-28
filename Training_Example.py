from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows多进程支持
    
    # 加载模型
    model = YOLO("yolov8s.yaml").load("yolov8s.pt")  
    # 训练模型
    results = model.train(
        data="ultralytics\cfg\datasets\coco.yaml",  # 改用内置数据集名称自动下载
        epochs=0,
        imgsz=640,
        device=0,
        workers=8,
        batch=0.8,
        mixup=0.1,
        exist_ok=True,
        close_mosaic=15,
        cache=True  # 启用数据集缓存
    )
    