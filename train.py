import os
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.STDMANet import STDMANet
from utils.dataloader import get_data_loaders  # 修改为使用更新后的数据加载器
from utils.loss import SoftIoULoss, SLSIoULoss
from utils.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train STDMANet model')
    parser.add_argument('--data_dir', type=str, default="/mnt/c/Users/admin/Desktop/STDMANet/dataset/IRDST", help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames in sequence')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from checkpoint')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint saving interval (epochs)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, device, epoch, writer):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        images = batch['images'].to(device)  # [B, T, C, H, W]
        masks = batch['mask'].to(device)  # [B, 1, H, W]
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)  # [B, 1, H, W]
        
        # 计算损失（直接用outputs，不要二值化）
        loss = criterion(outputs, masks, epoch)  # 使用epoch参数以支持SLSIoULoss的warm-up

        # 反向传播
        loss.backward()
        
        # 优化
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item()
        })

    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)

    return avg_loss


def validate(model, val_loader, criterion, device, epoch, writer):
    """验证模型"""
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # 获取数据
            images = batch['images'].to(device)  # [B, T, C, H, W]
            masks = batch['mask'].to(device)  # [B, 1, H, W]
            
            # 前向传播
            outputs = model(images)  # [B, 1, H, W]
            
            # 计算损失（直接用outputs，不要二值化）
            loss = criterion(outputs, masks, epoch)  # 使用epoch参数以支持SLSIoULoss的warm-up

            # 记录损失
            total_loss += loss.item()
    
    # 计算平均损失
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    
    # 评估模型
    metrics = evaluate_model(model, val_loader, device)
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Metrics/precision', metrics['precision'], epoch)
    writer.add_scalar('Metrics/recall', metrics['recall'], epoch)
    writer.add_scalar('Metrics/f1_score', metrics['f1_score'], epoch)
    writer.add_scalar('Metrics/iou', metrics['iou'], epoch)
    writer.add_scalar('Metrics/ap', metrics['ap'], epoch)
    
    print(f'Validation Epoch {epoch}: Loss: {avg_loss:.4f}, '
          f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, '
          f'F1: {metrics["f1_score"]:.4f}, IoU: {metrics["iou"]:.4f}, AP: {metrics["ap"]:.4f}')
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, save_dir, is_best=True):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    # 保存最新的检查点
    torch.save(state, os.path.join(save_dir, 'latest.pth'))
    
    # 保存特定epoch的检查点
    torch.save(state, os.path.join(save_dir, f'epoch_{epoch}.pth'))
    
    # 如果是最佳模型，也保存为best.pth
    if is_best:
        torch.save(state, os.path.join(save_dir, 'best.pth'))


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames
    )
    print(f'Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}')
    
    # 创建模型
    model = STDMANet(in_channels=1, num_frames=args.num_frames)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = SLSIoULoss()  # 使用自定义的SLSIoULoss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 初始化变量
    start_epoch = 1
    best_val_loss = float('inf')
    best_f1_score = 0
    
    # 如果指定了resume，则加载检查点
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('loss', float('inf'))
            best_f1_score = checkpoint.get('metrics', {}).get('f1_score', 0)
            
            print(f'Resumed training from epoch {start_epoch}')
        else:
            print(f'No checkpoint found at {args.resume}, starting from scratch')
    
    # 开始训练
    print(f'Starting training for {args.epochs} epochs')
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # 周期性验证
        if (epoch) % args.val_interval == 0:
            val_loss, metrics = validate(model, val_loader, criterion, device, epoch, writer)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 检查是否为最佳模型
            is_best_loss = val_loss < best_val_loss
            is_best_f1 = metrics['f1_score'] > best_f1_score
            
            if is_best_loss:
                best_val_loss = val_loss
                print(f'New best validation loss: {best_val_loss:.4f}')
            
            if is_best_f1:
                best_f1_score = metrics['f1_score']
                print(f'New best F1 score: {best_f1_score:.4f}')
        
        # 周期性保存检查点
        if (epoch) % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, val_loss, metrics,
                args.save_dir, is_best=is_best_loss or is_best_f1
            )
    
    print(f'Training completed. Best validation loss: {best_val_loss:.4f}, Best F1 score: {best_f1_score:.4f}')
    writer.close()


if __name__ == '__main__':
    main()