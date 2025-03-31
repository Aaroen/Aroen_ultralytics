from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 加载模型
    model = YOLO("ultralytics/cfg/models/12/yolo12n.yaml").load("yolo12n.pt")  
    # 训练模型
    results = model.train(
        data="ultralytics\cfg\datasets\VOC.yaml",
        epochs=3,
        imgsz=640,          
        device=0,
    )
