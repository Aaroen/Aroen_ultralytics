from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run with arguments
model.predict('E:/Data/coco/images/val2017/000000022396.jpg', save=True, imgsz=320, conf=0.5)
