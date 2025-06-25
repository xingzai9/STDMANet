import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IRTargetDataset(Dataset):
    """
    红外小目标数据集
    """
    def __init__(self, data_dir, num_frames=4, transform=None, target_size=(256, 256), is_train=True):
        """
        参数:
            data_dir: 数据目录，应该包含images和masks子目录
            num_frames: 使用的帧序列长度
            transform: 数据增强转换
            target_size: 调整图像大小
            is_train: 是否为训练模式
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.transform = transform
        self.target_size = target_size
        self.is_train = is_train
        
        # 设置图像和掩码目录
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        
        # 读取训练/测试分割
        if is_train:
            list_file = os.path.join(data_dir, 'ImageSets', 'train_new.txt')
        else:
            list_file = os.path.join(data_dir, 'ImageSets', 'val_new.txt')
        
        self.sequence_ids = []
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                for line in f:
                    seq_id = line.strip()
                    if seq_id:  # 确保不是空行
                        self.sequence_ids.append(seq_id)
        else:
            # 如果列表文件不存在，尝试列出所有序列
            print(f"警告: 找不到列表文件 {list_file}，尝试使用所有可用序列")
            self.sequence_ids = [d for d in os.listdir(self.img_dir) 
                               if os.path.isdir(os.path.join(self.img_dir, d))]
        
        # 为每个序列收集有效帧
        self.samples = []
        
        for seq_id in self.sequence_ids:
            seq_img_path = os.path.join(self.img_dir, seq_id)
            
            if not os.path.exists(seq_img_path):
                print(f"警告: 序列目录不存在: {seq_img_path}")
                continue
                
            # 获取所有图像文件
            frames = [f for f in os.listdir(seq_img_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            frames.sort()  # 确保按顺序排序
            
            # 确保每个序列有足够的帧
            if len(frames) >= self.num_frames:
                for i in range(len(frames) - self.num_frames + 1):
                    # 收集num_frames个连续帧
                    frame_seq = frames[i:i+self.num_frames]
                    self.samples.append((seq_id, frame_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_id, frames = self.samples[idx]
        
        # 读取图像序列
        img_seq = []
        for frame in frames:
            img_path = os.path.join(self.img_dir, seq_id, frame)
            
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在: {img_path}")
                # 创建空白图像
                img = np.zeros(self.target_size[::-1], dtype=np.float32)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"警告: 无法读取图像: {img_path}")
                    img = np.zeros(self.target_size[::-1], dtype=np.float32)
            
            # 调整大小
            if img.shape[:2] != self.target_size:
                img = cv2.resize(img, self.target_size)
                
            # 归一化到[0,1]
            img = img.astype(np.float32) / 255.0
            
            if self.transform:
                img = self.transform(img)
                
            # 添加通道维度
            img = np.expand_dims(img, axis=0)
            img_seq.append(img)
            
        # 读取最后一帧的掩码（目标帧）
        target_frame = frames[-1]
        mask_name = os.path.splitext(target_frame)[0] + '.png'  # 确保使用.png作为掩码扩展名
        mask_path = os.path.join(self.mask_dir, seq_id, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取掩码: {mask_path}")
                mask = np.zeros(self.target_size[::-1], dtype=np.float32)
        else:
            # 如果掩码不存在，创建一个空掩码
            print(f"警告: 掩码不存在: {mask_path}，创建空掩码")
            mask = np.zeros(self.target_size[::-1], dtype=np.float32)
        
        # 调整掩码大小
        if mask.shape[:2] != self.target_size:
            mask = cv2.resize(mask, self.target_size)
            
        # 二值化掩码
        mask = (mask > 0).astype(np.float32)
        
        # 生成高斯热图作为目标
        target = self._generate_gaussian_heatmap(mask)
        
        # 转换为torch张量
        img_seq_tensor = torch.from_numpy(np.stack(img_seq, axis=0)).float()  # [T, C, H, W]
        mask_tensor = torch.from_numpy(np.expand_dims(mask, axis=0)).float()  # [1, H, W]
        target_tensor = torch.from_numpy(np.expand_dims(target, axis=0)).float()  # [1, H, W]
        
        return {
            'images': img_seq_tensor,  # [T, C, H, W]
            'mask': mask_tensor,  # [1, H, W]
            'target': target_tensor,  # [1, H, W]
            'seq_id': seq_id,
            'frame_id': frames[-1]
        }
    
    def _generate_gaussian_heatmap(self, mask, sigma=5):
        """
        从二值掩码生成高斯热图
        """
        H, W = mask.shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        # 找到掩码中的目标点
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            # 如果没有目标，则返回空热图
            return heatmap
        
        # 计算目标的中心点
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        # 生成网格坐标
        y, x = np.ogrid[:H, :W]
        
        # 计算每个像素到中心的距离
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        
        # 应用高斯函数
        heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
        
        return heatmap


def get_data_loaders(data_dir, batch_size=8, num_frames=4, num_workers=4):
    """
    获取训练和测试数据加载器
    """
    # 创建数据集
    train_dataset = IRTargetDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        transform=None,  # 我们已经在自定义Dataset中进行了预处理
        is_train=True
    )
    
    test_dataset = IRTargetDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        transform=None,
        is_train=False
    )
    
    print(f"已加载 {len(train_dataset)} 个训练样本和 {len(test_dataset)} 个测试样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
