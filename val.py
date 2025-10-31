import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO('Trained_model/Pin_datasets_train_model/best.pt') # Pin_datasets train
    # model = YOLO('Trained_model/Pin_low_quality_dataset_train_model/best.pt') # Pin_low_quality_dataset train
    # model = YOLO('Trained_model/Pin_datasets_Pruning_and_Distill_model/best.pt') # Pin_datasets Pruning and Distill
    model = YOLO('Trained_model/Pin_low_quality_dataset_Pruning_and_Distill/best.pt') # Pin_low_quality_dataset Pruning and Distill


    # model.val(data='dataset/Pin_datasets.yaml',
    model.val(data='dataset/Pin_low_quality_dataset.yaml',
              split='val',
              imgsz=640,
              batch=32,
              device='0',
              # iou=0.95,
              # rect=False,
              save_json=True,
              project='runs/val',
              name='exp',
              )
