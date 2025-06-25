import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from model.STDMANet import STDMANet
from utils.dataloader import IRTargetDataset
from utils.metrics import evaluate_model, calculate_precision_recall, calculate_f1_score, calculate_pd_fa


def parse_args():
    parser = argparse.ArgumentParser(description='Test STDMANet model')
    parser.add_argument('--data_dir', type=str, default="/mnt/c/Users/admin/Desktop/STDMANet/dataset/IRDST", help='Dataset directory')
    parser.add_argument('--checkpoint', type=str, default="/mnt/c/Users/admin/Desktop/STDMANet/checkpoints/best.pth", help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames in sequence')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save visualization results')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--visualization', action='store_true', help='Save visualization results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--save_masks', action='store_true', help='Save predicted binary masks')
    
    return parser.parse_args()


def visualize_results(image, mask, target, prediction, output_path, binary_pred=None):
    """
    可视化检测结果
    
    参数:
        image: 输入图像 [H, W]
        mask: 掩码 [H, W]
        target: 目标热图 [H, W]
        prediction: 预测热图 [H, W]
        output_path: 保存路径
        binary_pred: 二值化预测掩码 [H, W]
    """
    if binary_pred is not None:
        # 如果有二值预测，创建2x3网格
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 显示原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # 显示掩码
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # 显示二值预测掩码
        axes[0, 2].imshow(binary_pred, cmap='gray')
        axes[0, 2].set_title('Predicted Mask')
        axes[0, 2].axis('off')
        
        # 显示目标热图
        axes[1, 0].imshow(target, cmap='jet')
        axes[1, 0].set_title('Ground Truth Heatmap')
        axes[1, 0].axis('off')
        
        # 显示预测热图
        axes[1, 1].imshow(prediction, cmap='jet')
        axes[1, 1].set_title('Predicted Heatmap')
        axes[1, 1].axis('off')
        
        # 第三个位置留空或者显示热图叠加到原图上
        overlay = 0.7 * image + 0.3 * prediction
        axes[1, 2].imshow(overlay, cmap='jet')
        axes[1, 2].set_title('Heatmap Overlay')
        axes[1, 2].axis('off')
    else:
        # 原来的2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 显示原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # 显示掩码
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # 显示目标热图
        axes[1, 0].imshow(target, cmap='jet')
        axes[1, 0].set_title('Ground Truth Heatmap')
        axes[1, 0].axis('off')
        
        # 显示预测热图
        axes[1, 1].imshow(prediction, cmap='jet')
        axes[1, 1].set_title('Predicted Heatmap')
        axes[1, 1].axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_binary_mask(binary_mask, save_path):
    """
    保存二值掩码图像
    
    参数:
        binary_mask: 二值掩码图像 [H, W]
        save_path: 保存路径
    """
    # 确保二值掩码是0和255
    mask_image = (binary_mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_image)


def test_model(model, test_loader, device, output_dir=None, visualization=False, threshold=0.5, save_masks=False):
    """
    测试模型并评估性能
    
    参数:
        model: 待测试模型
        test_loader: 测试数据加载器
        device: 使用的设备 (CPU/GPU)
        output_dir: 结果保存目录
        visualization: 是否保存可视化结果
        threshold: 二值化阈值
        save_masks: 是否保存预测的二值掩码
        
    返回:
        metrics: 包含各种评估指标的字典
    """
    model.eval()
    
    # 创建输出目录
    if (visualization or save_masks) and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if save_masks:
            mask_dir = os.path.join(output_dir, 'predicted_masks')
            os.makedirs(mask_dir, exist_ok=True)
    
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_pd = 0
    total_fa = 0
    
    # 获取图像大小
    sample_batch = next(iter(test_loader))
    img_size = sample_batch['images'].shape[-1]  # 假设图像是正方形的
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            # 获取数据
            images = batch['images'].to(device)  # [B, T, C, H, W]
            targets = batch['target'].to(device)  # [B, 1, H, W]
            masks = batch['mask'].to(device)  # [B, 1, H, W]
            seq_ids = batch['seq_id']
            frame_ids = batch['frame_id']
            
            # 前向传播
            outputs = model(images)  # [B, 1, H, W]
            
            # 应用sigmoid激活
            pred_probs = torch.sigmoid(outputs)
            
            # 计算评估指标
            precision, recall = calculate_precision_recall(pred_probs.cpu(), targets.cpu(), threshold)
            f1 = calculate_f1_score(precision, recall)
            
            # 计算PD和FA
            pd_rate, fa_rate, _ = calculate_pd_fa(pred_probs.cpu(), targets.cpu(), img_size, bins=10)
            
            # 累加指标
            total_precision += precision
            total_recall += recall
            total_f1_score += f1
            total_pd += pd_rate
            total_fa += fa_rate
            
            # 二值化预测掩码
            binary_masks = (pred_probs > threshold).float().cpu().numpy()
            
            # 可视化或保存掩码
            if (visualization or save_masks) and output_dir:
                for b in range(len(seq_ids)):
                    # 获取当前样本
                    current_image = images[b, -1, 0].cpu().numpy()  # 获取最后一帧
                    current_mask = masks[b, 0].cpu().numpy()
                    current_target = targets[b, 0].cpu().numpy()
                    current_pred = pred_probs[b, 0].cpu().numpy()
                    current_binary = binary_masks[b, 0]
                    current_seq = seq_ids[b]
                    current_frame = frame_ids[b]
                    
                    # 创建序列目录
                    seq_dir = os.path.join(output_dir, current_seq)
                    os.makedirs(seq_dir, exist_ok=True)
                    
                    # 保存可视化结果
                    if visualization:
                        save_path = os.path.join(seq_dir, f'{os.path.splitext(current_frame)[0]}_result.png')
                        visualize_results(current_image, current_mask, current_target, current_pred, save_path, current_binary)
                    
                    # 保存预测掩码
                    if save_masks:
                        # 创建掩码保存目录
                        masks_seq_dir = os.path.join(mask_dir, current_seq)
                        os.makedirs(masks_seq_dir, exist_ok=True)
                        mask_save_path = os.path.join(masks_seq_dir, f'{os.path.splitext(current_frame)[0]}.png')
                        save_binary_mask(current_binary, mask_save_path)
    
    # 计算平均指标
    num_batches = len(test_loader)
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1_score = total_f1_score / num_batches
    avg_pd = total_pd / num_batches
    avg_fa = total_fa / num_batches
    
    # 计算所有评估指标
    metrics = evaluate_model(model, test_loader, device, threshold)
    
    # 确保手动计算的指标与evaluate_model结果一致
    metrics['pd'] = avg_pd
    metrics['fa'] = avg_fa
    
    return metrics


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # 创建测试数据集
    test_dataset = IRTargetDataset(
        data_dir=args.data_dir,
        num_frames=args.num_frames,
        transform=None,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f'Test samples: {len(test_dataset)}')
    
    # 创建模型
    model = STDMANet(in_channels=1, num_frames=args.num_frames)
    
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        print(f'Loading checkpoint from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
    else:
        raise FileNotFoundError(f'Checkpoint not found at {args.checkpoint}')
    
    # 测试模型
    metrics = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        visualization=args.visualization,
        threshold=args.threshold,
        save_masks=args.save_masks
    )
    
    # 打印结果
    print(f"\nTest Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"AP: {metrics['ap']:.4f}")
    print(f"PD (Detection Probability): {metrics['pd']:.4f}")  # 新增
    print(f"FA (False Alarm Rate): {metrics['fa']:.4f}")       # 新增
    
    # 保存结果到文本文件
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, 'results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"IoU: {metrics['iou']:.4f}\n")
            f.write(f"AP: {metrics['ap']:.4f}\n")
            f.write(f"PD (Detection Probability): {metrics['pd']:.4f}\n")  # 新增
            f.write(f"FA (False Alarm Rate): {metrics['fa']:.4f}\n")       # 新增
            f.write(f"Threshold: {args.threshold:.2f}\n")


if __name__ == '__main__':
    main()