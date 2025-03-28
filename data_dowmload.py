from ultralytics import YOLO
import os

def download_dataset_yaml(dataset_name, dataset_dir='datasets'):
    """
    下载指定的数据集
    Args:
        dataset_name: 数据集配置文件名称（不包含.yaml后缀）
        dataset_dir: 保存数据集的目录
    """
    try:
        # 确保目标目录存在
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 使用YOLO模型加载一个预定义的模型结构 (例如 yolov8n.yaml)，并使用train方法下载数据集，
        # data 参数指定数据集配置文件
        model = YOLO("yolov8n.yaml") # 加载一个模型结构，这里使用 yolov8n.yaml 作为示例
        model.train(data=f"ultralytics/cfg/datasets/{dataset_name}.yaml", epochs=1, imgsz=640, exist_ok=True, device='cpu')
        print(f"数据集 {dataset_name} 下载成功！")
        
    except Exception as e:
        print(f"下载数据集 {dataset_name} 时发生错误: {str(e)}")

if __name__ == '__main__':
    # 可选数据集示例
    available_datasets = [
        'coco',          # COCO 数据集
        'coco8',         # COCO 数据集的小型版本（仅8张图像）
        'coco128',       # COCO 数据集的小型版本（128张图像）
        'VOC',           # Pascal VOC 数据集
        'ImageNet',      # ImageNet 数据集
        'VisDrone',      # VisDrone 数据集
    ]
    
    print("可用的数据集列表：")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")
    
    # 让用户选择要下载的数据集
    choice = input("\n请输入要下载的数据集编号（1-6）: ")
    try:
        dataset_name = available_datasets[int(choice)-1]
        download_dataset_yaml(dataset_name)
    except (ValueError, IndexError):
        print("无效的选择，请输入1-6之间的数字。")