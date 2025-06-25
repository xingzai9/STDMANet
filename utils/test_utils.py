import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_prediction_masks(predictions, seq_ids, frame_ids, output_dir, threshold=0.5):
    """
    保存预测的二值掩码
    
    参数:
        predictions: 预测热图 [B, 1, H, W]
        seq_ids: 序列ID列表
        frame_ids: 帧ID列表
        output_dir: 输出目录
        threshold: 二值化阈值
    """
    # 创建输出目录
    ensure_dir(output_dir)
    mask_dir = os.path.join(output_dir, 'masks')
    ensure_dir(mask_dir)
    
    # 二值化预测
    binary_masks = (predictions > threshold).cpu().numpy()
    
    # 保存掩码
    for i in range(len(seq_ids)):
        # 创建序列目录
        seq_dir = os.path.join(mask_dir, seq_ids[i])
        ensure_dir(seq_dir)
        
        # 构建文件名
        frame_name = os.path.splitext(frame_ids[i])[0]
        mask_path = os.path.join(seq_dir, f'{frame_name}.png')
        
        # 保存二值掩码 - 确保正确处理二值图像
        mask = (binary_masks[i, 0] * 255).astype(np.uint8)
        # 可选：应用形态学操作来改善掩码质量
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        cv2.imwrite(mask_path, mask)


def generate_overlay_visualization(image, prediction, mask=None, threshold=0.5):
    """
    生成预测结果的叠加可视化
    
    参数:
        image: 输入图像 [H, W]
        prediction: 预测热图 [H, W]
        mask: 真实掩码 [H, W]，可选
        threshold: 二值化阈值
        
    返回:
        可视化图像
    """
    # 归一化图像
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建RGB图像
    rgb_image = np.stack([image] * 3, axis=2)
    
    # 二值化预测
    binary_pred = (prediction > threshold).astype(np.float32)
    
    # 创建叠加图像
    overlay = rgb_image.copy()
    
    # 添加预测掩码（红色）
    overlay[binary_pred > 0, 0] = 1.0
    overlay[binary_pred > 0, 1] = 0.0
    overlay[binary_pred > 0, 2] = 0.0
    
    # 如果有真实掩码，添加真实掩码（绿色）
    if mask is not None:
        overlay[mask > 0, 0] = 0.0
        overlay[mask > 0, 1] = 1.0
        overlay[mask > 0, 2] = 0.0
        
        # 预测和真实的重叠区域（黄色）
        overlap = (binary_pred > 0) & (mask > 0)
        overlay[overlap, 0] = 1.0
        overlay[overlap, 1] = 1.0
        overlay[overlap, 2] = 0.0
    
    return overlay


def batch_save_results(images, predictions, targets=None, masks=None, seq_ids=None, frame_ids=None, 
                      output_dir='results', save_masks=True, save_visualizations=True, threshold=0.5):
    """
    批量保存测试结果
    
    参数:
        images: 输入图像 [B, C, H, W]
        predictions: 预测热图 [B, 1, H, W]
        targets: 目标热图 [B, 1, H, W]，可选
        masks: 真实掩码 [B, 1, H, W]，可选
        seq_ids: 序列ID列表
        frame_ids: 帧ID列表
        output_dir: 输出目录
        save_masks: 是否保存掩码
        save_visualizations: 是否保存可视化结果
        threshold: 二值化阈值
    """
    # 创建输出目录
    ensure_dir(output_dir)
    
    if save_masks:
        mask_dir = os.path.join(output_dir, 'masks')
        ensure_dir(mask_dir)
    
    if save_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        ensure_dir(vis_dir)
    
    # 获取批次大小
    batch_size = images.shape[0]
    
    # 二值化预测
    binary_preds = (predictions > threshold).cpu().numpy()
    
    # 处理每个样本
    for i in range(batch_size):
        # 获取当前样本数据
        image = images[i, 0].cpu().numpy() if images.shape[1] == 1 else images[i].cpu().numpy()
        pred = predictions[i, 0].cpu().numpy()
        binary_pred = binary_preds[i, 0]
        
        # 获取序列ID和帧ID
        seq_id = seq_ids[i] if seq_ids is not None else f'seq_{i}'
        frame_id = frame_ids[i] if frame_ids is not None else f'frame_{i}'
        frame_name = os.path.splitext(frame_id)[0] if '.' in frame_id else frame_id
        
        # 创建序列目录
        if save_masks:
            seq_mask_dir = os.path.join(mask_dir, seq_id)
            ensure_dir(seq_mask_dir)
            
            # 保存二值掩码
            mask_path = os.path.join(seq_mask_dir, f'{frame_name}.png')
            cv2.imwrite(mask_path, (binary_pred * 255).astype(np.uint8))
        
        if save_visualizations:
            seq_vis_dir = os.path.join(vis_dir, seq_id)
            ensure_dir(seq_vis_dir)
            
            # 获取真实掩码和目标（如果有）
            mask = masks[i, 0].cpu().numpy() if masks is not None else None
            target = targets[i, 0].cpu().numpy() if targets is not None else None
            
            # 创建可视化图像
            plt.figure(figsize=(15, 10))
            
            # 显示原始图像
            plt.subplot(2, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Input Image')
            plt.axis('off')
            
            # 显示预测热图
            plt.subplot(2, 3, 2)
            plt.imshow(pred, cmap='jet')
            plt.title('Prediction Heatmap')
            plt.axis('off')
            
            # 显示二值预测掩码
            plt.subplot(2, 3, 3)
            plt.imshow(binary_pred, cmap='gray')
            plt.title('Binary Prediction')
            plt.axis('off')
            
            # 如果有真实掩码和目标
            if mask is not None and target is not None:
                plt.subplot(2, 3, 4)
                plt.imshow(mask, cmap='gray')
                plt.title('Ground Truth Mask')
                plt.axis('off')
                
                plt.subplot(2, 3, 5)
                plt.imshow(target, cmap='jet')
                plt.title('Ground Truth Heatmap')
                plt.axis('off')
                
                # 显示叠加结果
                overlay = generate_overlay_visualization(image, pred, mask, threshold)
                plt.subplot(2, 3, 6)
                plt.imshow(overlay)
                plt.title('Overlay (Green: GT, Red: Pred)')
                plt.axis('off')
            else:
                # 只显示叠加到原图的预测
                overlay = generate_overlay_visualization(image, pred, None, threshold)
                plt.subplot(2, 3, 4)
                plt.imshow(overlay)
                plt.title('Prediction Overlay')
                plt.axis('off')
            
            # 保存可视化图像
            plt.tight_layout()
            plt.savefig(os.path.join(seq_vis_dir, f'{frame_name}_result.png'))
            plt.close()
