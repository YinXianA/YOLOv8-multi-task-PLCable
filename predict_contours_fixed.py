#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 多任务线缆分割轮廓可视化 - 修复版本
直接访问predictor.results来获取预测结果
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

def draw_contours_on_image(det_results, seg_results, image_path, output_dir):
    """
    在图像上绘制检测框和分割轮廓
    
    Args:
        det_results: 检测结果 (Results对象列表)
        seg_results: 分割结果 (torch.Tensor)
        image_path: 原始图像路径
        output_dir: 输出目录
    """
    # 读取原始图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    original_image = image.copy()
    
    # 绘制检测框
    if det_results and len(det_results) > 0:
        result = det_results[0]  # 取第一个结果
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f"{result.names[int(cls)]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 绘制分割轮廓
    if seg_results is not None and isinstance(seg_results, torch.Tensor):
        # 将分割结果转换为numpy数组
        seg_mask = seg_results.cpu().numpy().astype(np.uint8)
        
        # 如果是批次结果，取第一个
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask[0]
        
        # 调整分割掩码大小以匹配原始图像
        if seg_mask.shape != image.shape[:2]:
            seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # 查找轮廓
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓
        if contours:
            cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
            print(f"在图像 {Path(image_path).name} 中找到 {len(contours)} 个轮廓")
        else:
            print(f"在图像 {Path(image_path).name} 中未找到轮廓")
    
    # 保存结果
    output_path = Path(output_dir) / f"contour_{Path(image_path).name}"
    cv2.imwrite(str(output_path), image)
    print(f"轮廓可视化结果已保存到: {output_path}")

def main():
    # 加载模型
    model_path = "PowerLine-MTYOLO-NANO-150Epochs.pt"
    model = YOLO(model_path)
    
    # 测试图像路径
    test_images_dir = "content/MulticableData/images/val2017"
    output_dir = "runs/predict_contours_fixed"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取测试图像列表（只处理前5张）
    image_files = list(Path(test_images_dir).glob("*.jpg"))
    
    if not image_files:
        print("错误: 在指定路径下没有找到图片文件！")
        return
    
    print(f"开始处理 {len(image_files)} 张图像...")
    
    # 设置模型为多任务模式
    model.predictor = None  # 重置predictor
    
    # 逐张处理图像
    for i, image_path in enumerate(image_files):
        print(f"\n处理图像 {i+1}/{len(image_files)}: {image_path.name}")
        
        # 执行预测
        try:
            # 使用predict方法，但设置为多任务
            _ = model.predict(
                source=str(image_path),
                imgsz=(640, 640),
                device=[0] if torch.cuda.is_available() else 'cpu',
                task='multi',
                conf=0.25,
                iou=0.45,
                save=False,
                verbose=False
            )
            
            # 直接访问predictor的结果
            if hasattr(model, 'predictor') and model.predictor is not None:
                if hasattr(model.predictor, 'results') and model.predictor.results:
                    results = model.predictor.results
                    print(f"获取到 {len(results)} 个结果")
                    
                    # 分离检测和分割结果
                    det_results = None
                    seg_results = None
                    
                    for j, result in enumerate(results):
                        if isinstance(result, list):
                            # 检测结果
                            det_results = result
                            print(f"检测结果: {len(result)} 个对象")
                        elif isinstance(result, torch.Tensor):
                            # 分割结果
                            seg_results = result
                            print(f"分割结果形状: {result.shape}")
                    
                    # 绘制轮廓
                    draw_contours_on_image(det_results, seg_results, image_path, output_dir)
                else:
                    print("预测器中没有找到结果")
            else:
                print("预测器未初始化或为空")
                
        except Exception as e:
            print(f"处理图像 {image_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    print("=== YOLOv8 多任务线缆分割轮廓可视化 - 修复版本 ===")
    
    # 检查模型文件是否存在
    model_path = "PowerLine-MTYOLO-NANO-150Epochs.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在！")
        sys.exit(1)
    
    main()
    print("\n=== 轮廓可视化完成！ ===")
    print("请查看 'runs/predict_contours_fixed' 目录中的结果图片")