import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from utils.dataloader import IRTargetDataset
from model.STDMANet import STDMANet

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize STDMANet intermediate representations')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualization results')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames in sequence')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


class FeatureExtractor:
    """用于提取模型中间特征的类"""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hook(self, module_name, module):
        """注册钩子以获取中间特征"""
        hook = module.register_forward_hook(
            lambda m, input, output: self._get_features(module_name, output)
        )
        self.hooks.append(hook)
        
    def _get_features(self, name, output):
        """存储中间特征"""
        self.features[name] = output.detach().cpu()
        
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def visualize_features(features, output_dir):
    """可视化特征图"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, feature in features.items():
        # 确保特征是4D张量：[B, C, H, W]
        if len(feature.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = feature.shape
            feature = feature.view(B * T, C, H, W)
        
        # 获取通道数和空间维度
        B, C, H, W = feature.shape
        
        # 只处理第一个样本
        feature = feature[0]  # [C, H, W]
        
        # 确定要显示的通道数
        num_channels = min(64, C)
        
        # 计算子图布局
        rows = int(np.ceil(np.sqrt(num_channels)))
        cols = int(np.ceil(num_channels / rows))
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        
        # 扁平化轴数组以便于索引
        axes = axes.flatten() if num_channels > 1 else [axes]
        
        # 可视化每个通道
        for i in range(num_channels):
            channel_data = feature[i].numpy()
            
            # 归一化到[0, 1]
            channel_min = channel_data.min()
            channel_max = channel_data.max()
            if channel_max > channel_min:
                channel_data = (channel_data - channel_min) / (channel_max - channel_min)
            
            # 显示特征图
            axes[i].imshow(channel_data, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Ch {i}')
        
        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}.png'))
        plt.close()
        
        # 如果通道数很多，保存通道平均和最大值投影
        if C > 64:
            plt.figure(figsize=(10, 5))
            
            # 平均投影
            avg_proj = feature.mean(dim=0).numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(avg_proj, cmap='viridis')
            plt.title(f'{name} - Avg Projection')
            plt.axis('off')
            
            # 最大值投影
            max_proj = feature.max(dim=0)[0].numpy()
            plt.subplot(1, 2, 2)
            plt.imshow(max_proj, cmap='viridis')
            plt.title(f'{name} - Max Projection')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{name}_projection.png'))
            plt.close()


def visualize_attention_maps(features, output_dir):
    """可视化注意力图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 特定查找动态路径注意力图
    attention_keys = [k for k in features.keys() if 'dynamic' in k.lower()]
    
    for key in attention_keys:
        attention_map = features[key]
        
        # 确保特征是4D张量：[B, C, H, W]
        if len(attention_map.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = attention_map.shape
            attention_map = attention_map.view(B * T, C, H, W)
        
        # 只使用第一个样本的特征
        attention_map = attention_map[0]  # [C, H, W]
        
        # 计算空间注意力图
        spatial_attention = attention_map.mean(dim=0)  # [H, W]
        
        # 归一化到[0, 1]
        spatial_attention = spatial_attention - spatial_attention.min()
        spatial_attention = spatial_attention / (spatial_attention.max() + 1e-8)
        
        # 创建热图colormap
        plt.figure(figsize=(10, 10))
        plt.imshow(spatial_attention, cmap='jet')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Spatial Attention Map - {key}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_{key}.png'))
        plt.close()


def visualize_prediction_process(image, pred_heatmap, threshold=0.5, output_dir=None):
    """可视化预测过程"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 归一化预测热图
    pred_norm = pred_heatmap - pred_heatmap.min()
    pred_norm = pred_norm / (pred_norm.max() + 1e-8)
    
    # 二值化预测
    binary_pred = (pred_norm > threshold).astype(np.float32)
    
    # 创建结果图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 预测热图
    im = axes[1].imshow(pred_norm, cmap='jet')
    axes[1].set_title('Predicted Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 二值化预测结果
    axes[2].imshow(binary_pred, cmap='gray')
    axes[2].set_title(f'Binary Prediction (th={threshold})')
    axes[2].axis('off')
    
    # 保存结果
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'prediction_process.png'))
    plt.close()


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
    
    # 创建测试数据集
    test_dataset = IRTargetDataset(
        data_dir=args.data_dir,
        num_frames=args.num_frames,
        transform=None,
        is_train=False
    )
    
    # 确保样本索引有效
    if args.sample_idx >= len(test_dataset):
        print(f'Error: Sample index {args.sample_idx} exceeds dataset size {len(test_dataset)}')
        return
    
    # 获取样本
    sample = test_dataset[args.sample_idx]
    images = sample['images'].unsqueeze(0).to(device)  # [1, T, C, H, W]
    target = sample['target'].to(device)  # [1, H, W]
    mask = sample['mask'].to(device)  # [1, H, W]
    seq_id = sample['seq_id']
    frame_id = sample['frame_id']
    
    print(f'Visualizing sequence: {seq_id}, frame: {frame_id}')
    
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
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model)
    
    # 注册钩子
    feature_extractor.register_hook('temporal_extractor.differential_path', model.temporal_extractor.differential_path)
    feature_extractor.register_hook('temporal_extractor.dynamic_path', model.temporal_extractor.dynamic_path)
    feature_extractor.register_hook('temporal_extractor.static_path', model.temporal_extractor.static_path)
    feature_extractor.register_hook('temporal_features', model.temporal_extractor)
    feature_extractor.register_hook('spatial_features', model.spatial_refiner)
    feature_extractor.register_hook('prediction', model.prediction_head)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(images)  # [1, 1, H, W]
    
    # 可视化输入图像序列
    input_dir = os.path.join(args.output_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    
    for t in range(args.num_frames):
        img = images[0, t, 0].cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f'Frame {t+1}')
        plt.axis('off')
        plt.savefig(os.path.join(input_dir, f'frame_{t+1}.png'))
        plt.close()
    
    # 可视化目标和掩码
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(mask[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(target[0, 0].cpu().numpy(), cmap='jet')
    plt.title('Ground Truth Heatmap')
    plt.axis('off')
    
    plt.savefig(os.path.join(args.output_dir, 'ground_truth.png'))
    plt.close()
    
    # 可视化预测结果
    pred = torch.sigmoid(outputs)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(pred[0, 0].cpu().numpy(), cmap='jet')
    plt.title('Predicted Heatmap')
    plt.colorbar(label='Probability')
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'prediction.png'))
    plt.close()
    
    # 可视化中间特征
    features_dir = os.path.join(args.output_dir, 'features')
    visualize_features(feature_extractor.features, features_dir)
    
    # 可视化注意力图
    attention_dir = os.path.join(args.output_dir, 'attention')
    visualize_attention_maps(feature_extractor.features, attention_dir)
    
    # 可视化预测过程
    last_frame = images[0, -1, 0].cpu().numpy()
    pred_heatmap = pred[0, 0].cpu().numpy()
    process_dir = os.path.join(args.output_dir, 'process')
    visualize_prediction_process(last_frame, pred_heatmap, output_dir=process_dir)
    
    # 移除钩子
    feature_extractor.remove_hooks()
    
    print(f'Visualization completed. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
