from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows多进程支持
    
    # 加载模型
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  
    # 训练模型
    results = model.train(
        data="ultralytics\cfg\datasets\coco128.yaml",
        epochs=1,
        imgsz=640,          
        device=0,
        workers=3,
    )