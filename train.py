import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/My_Structure/yolo11-Slimneck-p2-DySample-DWConv-AFGCATT.yaml')
    model.load('yolo11n.pt')
    # model.train(data='dataset/Pin_datasets.yaml',
    model.train(data='dataset/Pin_low_quality_dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=2,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                resume=True,
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='Pin_datasets',
                )

