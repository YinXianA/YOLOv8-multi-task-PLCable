import sys
sys.path.insert(0, "C:/Users/FRDISI/Desktop/[phD]yolov8_improve/YOLOv8-multi-task-PLCable/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

if __name__ == '__main__':


    # model = YOLO('yolov8s-seg.pt')
    #number = 1 #input how many tasks in your work
    model = YOLO('C:/Users/FRDISI/Desktop/[phD]yolov8_improve/YOLOv8-multi-task-PLCable/runs/multi/IEEE23/weights/best.pt')  # 加载自己训练的模型# Validate the model
    # metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[4],task='multi',name='v3-model-val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered

    metrics = model.val(data='C:/Users/FRDISI/Desktop/[phD]yolov8_improve/YOLOv8-multi-task-PLCable/ultralytics/datasets/bdd-multi-Cable.yaml',device=[0],task='multi',name='val',iou=0.6,conf=0.001,batch=1)  # batch 1 cause onnx and tensort do only batch 1 #no arguments needed, dataset and settings remembered
"""    for i in range(number):
         print(f'This is for {i} work')
         print(metrics[i].box.map)    # map50-95
         print(metrics[i].box.map50)  # map50
         print(metrics[i].box.map75)  # map75
         print(metrics[i].box.maps)   # a list contains map50-95 of each category"""