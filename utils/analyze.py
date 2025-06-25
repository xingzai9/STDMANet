import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc

from model.STDMANet import STDMANet
from dataset.dataloader import get_data_loaders
from utils.metrics import calculate_metrics, calculate_pd_fa


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze STDMANet model performance')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames in sequence')
    parser.add_argument('--output_dir', type=str, default='analysis', help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def calculate_pr_curve(model, data_loader, device):
    """计算精确率-召回率曲线"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Calculating PR curve'):
            # 获取数据
            images = batch['images'].to(device)  # [B, T, C, H, W]
            targets = batch['target']  # [B, 1, H, W]
            
            # 前向传播
            outputs = model(images)  # [B, 1, H, W]
            
            # 应用sigmoid激活
            pred_probs = torch.sigmoid(outputs)
            
            # 收集预测和目标
            all_preds.append(pred_probs.cpu().view(-1).numpy())
            all_targets.append(targets.view(-1).numpy())
    
    # 连接所有预测和目标
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    
    # 计算AP
    ap = auc(recall, precision)
    
    return precision, recall, thresholds, ap


def calculate_metrics(pred, target):
    """计算精确率、召回率、F1分数和误检率"""
    # 展平预测和目标
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 计算TP, FP, FN
    true_positive = (pred_flat * target_flat).sum()
    false_positive = pred_flat.sum() - true_positive
    false_negative = target_flat.sum() - true_positive
    
    # 计算评价指标
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    false_alarm = false_positive / (target_flat.sum() + 1e-6)  # 计算误检率
    
    # 获取第一个样本的形状来估计图像大小
    img_size = int(np.sqrt(pred_flat.shape[0]))
    
    # 计算PD和FA
    pd_rate, fa_rate, _ = calculate_pd_fa(pred, target, img_size, bins=10)
    
    return precision.item(), recall.item(), f1.item(), false_alarm.item(), pd_rate, fa_rate


def analyze_threshold_impact(model, data_loader, device):
    """分析阈值对检测性能的影响"""
    model.eval()
    
    thresholds = np.linspace(0.1, 0.9, 9)
    precision_list = []
    recall_list = []
    f1_list = []
    pd_list = []
    fa_list = []
    
    with torch.no_grad():
        for threshold in tqdm(thresholds, desc='Analyzing thresholds'):
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            total_pd = 0
            total_fa = 0
            
            for batch in data_loader:
                # 获取数据
                images = batch['images'].to(device)  # [B, T, C, H, W]
                targets = batch['target'].to(device)  # [B, 1, H, W]
                
                # 前向传播
                outputs = model(images)  # [B, 1, H, W]
                
                # 应用sigmoid激活
                pred_probs = torch.sigmoid(outputs)
                
                # 二值化预测
                pred_binary = (pred_probs > threshold).float()
                
                # 计算评价指标
                precision, recall, f1, false_alarm, pd_rate, fa_rate = calculate_metrics(pred_binary, targets)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_pd += pd_rate
                total_fa += fa_rate
            
            # 计算平均指标
            num_batches = len(data_loader)
            avg_precision = total_precision / num_batches
            avg_recall = total_recall / num_batches
            avg_f1 = total_f1 / num_batches
            avg_pd = total_pd / num_batches
            avg_fa = total_fa / num_batches
            
            precision_list.append(avg_precision)
            recall_list.append(avg_recall)
            f1_list.append(avg_f1)
            pd_list.append(avg_pd)
            fa_list.append(avg_fa)
    
    return thresholds, precision_list, recall_list, f1_list, pd_list, fa_list


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取数据加载器
    _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames
    )
    
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
    
    # 计算PR曲线
    print('Calculating PR curve...')
    precision, recall, thresholds, ap = calculate_pr_curve(model, test_loader, device)
    
    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'))
    plt.close()
    
    # 分析阈值影响
    print('Analyzing threshold impact...')
    th_values, precision_list, recall_list, f1_list, pd_list, fa_list = analyze_threshold_impact(model, test_loader, device)
    
    # 绘制阈值影响图 - 常规指标
    plt.figure(figsize=(10, 8))
    plt.plot(th_values, precision_list, 'r-', linewidth=2, label='Precision')
    plt.plot(th_values, recall_list, 'g-', linewidth=2, label='Recall')
    plt.plot(th_values, f1_list, 'b-', linewidth=2, label='F1 Score')
    
    # 找到最佳F1分数对应的阈值
    best_f1_idx = np.argmax(f1_list)
    best_threshold = th_values[best_f1_idx]
    best_f1 = f1_list[best_f1_idx]
    
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
    plt.title(f'Impact of Threshold on Metrics (Best F1 = {best_f1:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'threshold_analysis.png'))
    plt.close()
    
    # 绘制PD和FA图
    plt.figure(figsize=(10, 8))
    plt.plot(th_values, pd_list, 'm-', linewidth=2, label='PD (Detection Probability)')
    plt.plot(th_values, fa_list, 'y-', linewidth=2, label='FA (False Alarm Rate)')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
    plt.title('Impact of Threshold on PD and FA')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'pd_fa_analysis.png'))
    plt.close()
    
    # 保存分析结果
    results_path = os.path.join(args.output_dir, 'analysis_results.txt')
    with open(results_path, 'w') as f:
        f.write(f'AP: {ap:.4f}\n\n')
        f.write(f'Best Threshold: {best_threshold:.4f}\n')
        f.write(f'Best F1 Score: {best_f1:.4f}\n')
        f.write(f'Precision at Best Threshold: {precision_list[best_f1_idx]:.4f}\n')
        f.write(f'Recall at Best Threshold: {recall_list[best_f1_idx]:.4f}\n')
        f.write(f'PD at Best Threshold: {pd_list[best_f1_idx]:.4f}\n')
        f.write(f'FA at Best Threshold: {fa_list[best_f1_idx]:.4f}\n\n')
        
        f.write('Threshold Analysis:\n')
        f.write('Threshold\tPrecision\tRecall\t\tF1 Score\tPD\t\tFA\n')
        for i, th in enumerate(th_values):
            f.write(f'{th:.2f}\t\t{precision_list[i]:.4f}\t\t{recall_list[i]:.4f}\t\t{f1_list[i]:.4f}\t\t{pd_list[i]:.4f}\t\t{fa_list[i]:.4f}\n')
    
    print(f'Analysis completed. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
