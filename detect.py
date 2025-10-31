import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    # model = YOLO('runs/train/yolo11-use-sure/weights/best.pt') # select your model.pt path
    model = YOLO('runs/train/All-change-PCB/weights/best.pt') # select your model.pt path
    # model = YOLO('runs/distill/All_Change_distill_all_tau=2.0/weights/best.pt') # select your model.pt path
    # model = YOLO('Distill_module_without_pruning/teacher/best.pt') # select your model.pt path
    # model.predict(source=r'E:\Data_Set\ExDark_datasets\YOLODataset\images\val',
    # model.predict(source=r'E:/Data_Set/Pin_Detection/Pin_datasets/images/val_test',
    # model.predict(source=r'E:/Data_Set/Pin_Detection/Low_quality_dataset_second/images/val',
    model.predict(source=r'E:/Data_Set/PCBDataSet/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='PCB',
                  # name='ALL_change_detect',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  save_txt=True, # save results as .txt file
                  save_crop=True, # save cropped images with results
                )