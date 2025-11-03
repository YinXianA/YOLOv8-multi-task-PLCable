import sys
import os
import cv2
import numpy as np
from pathlib import Path
# Add current project directory to Python path to ensure local ultralytics is used
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)
# 现在就可以导入Yolo类了
from ultralytics import YOLO

def draw_contours_on_image(results, save_dir="runs/predict_contours"):
    """
    在原图上绘制线缆分割的轮廓并保存可视化结果
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        # 获取原始图像
        img = result.orig_img.copy()
        
        # 检查是否有分割结果
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # 获取分割掩码
            
            # 为每个分割掩码绘制轮廓
            for j, mask in enumerate(masks):
                # 将掩码转换为uint8格式
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # 查找轮廓
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 在原图上绘制轮廓
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽2
                
                # 可选：填充半透明区域
                overlay = img.copy()
                cv2.fillPoly(overlay, contours, (0, 255, 0))
                img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
        
        # 如果有检测框，也绘制出来
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                # 绘制检测框（红色）
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 添加置信度标签
                cv2.putText(img, f'Broken Cable: {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存结果图像
        img_name = f"contour_result_{i:04d}.jpg"
        output_path = save_path / img_name
        cv2.imwrite(str(output_path), img)
        print(f"轮廓可视化结果已保存到: {output_path}")

if __name__ == '__main__':
    number = 2 #input how many tasks in your work
    model = YOLO("PowerLine-MTYOLO-NANO-150Epochs.pt")  # Load the model
    
    # 执行预测
    results = model.predict(
        source='content/MulticableData/images/val2017', 
        imgsz=(640,640), 
        device=[0],
        name='FTMAPS', 
        augment=False,
        save=True,
        task='multi', 
        conf=0.25, 
        iou=0.45,  
        show_labels=True
    )
    
    # 生成轮廓可视化
    print("正在生成轮廓可视化图片...")
    draw_contours_on_image(results, save_dir="runs/contour_visualization")
    print("轮廓可视化完成！")
