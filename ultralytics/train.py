import sys
# 现在就可以导入Yolo类了
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('/content/YOLOv8-multi-task-PLCable-Colab/YOLOv8-multi-task-PLCable-Colab/ultralytics/models/v8/PowerLine-MTYOLO.yaml', task='multi')#.load('yolov8n.pt') # build a new model from YAML
    #model = YOLO("C:/Users/FRDISI/Desktop/[phD]yolov8_improve/YOLOv8-multi-task-PLCable/ultralytics/models/v8/yolov8-seg.yaml", task='multi')#.load('yolov8n.pt') # build a new model from YAML

    model.train(data='/content/YOLOv8-multi-task-PLCable-Colab/YOLOv8-multi-task-PLCable-Colab/ultralytics/datasets/bdd-multi-Cable.yaml', rect=True, batch=32, epochs=150, imgsz=(640,640), device=0, name='PaperBadr-Colab', val=True, task='multi', pretrained=False, cache=False) #rect=True
