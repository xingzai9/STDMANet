import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvBlock(in_channels + i * growth_rate, growth_rate))
            
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class BackgroundAlignment(nn.Module):
    def __init__(self, channels=64):
        super(BackgroundAlignment, self).__init__()
        self.conv1 = ConvBlock(1, channels//2)
        self.conv2 = ConvBlock(channels//2, channels)
        
    def forward(self, frames):
        # frames: [B, T, C, H, W]
        B, T, C, H, W = frames.shape
        aligned_features = []
        
        # 使用最后一帧作为参考帧
        reference = frames[:, -1]  # [B, C, H, W]
        reference_features = self.conv1(reference)
        reference_features = self.conv2(reference_features)
        
        # 将参考帧添加到对齐特征中
        aligned_features.append(reference_features)
        
        # 使用SIFT和RANSAC对齐前面的帧
        for t in range(T-1):
            current = frames[:, t]  # [B, C, H, W]
            
            # 处理批次中的每个样本
            batch_aligned = []
            for b in range(B):
                # 将张量转换为numpy数组以便使用OpenCV
                curr_frame = current[b, 0].detach().cpu().numpy()
                ref_frame = reference[b, 0].detach().cpu().numpy()
                
                # 确保数据类型和值范围正确
                curr_frame = (curr_frame * 255).astype(np.uint8)
                ref_frame = (ref_frame * 255).astype(np.uint8)
                
                # 使用SIFT提取特征点和描述符
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(curr_frame, None)
                kp2, des2 = sift.detectAndCompute(ref_frame, None)
                
                aligned_frame = curr_frame.copy()
                
                # 确保找到了足够的特征点
                if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
                    # 特征匹配
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
                    
                    # 应用比率测试筛选好的匹配
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    
                    # 如果有足够的好匹配点，估计透视变换
                    if len(good_matches) > 10:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # 使用RANSAC估计单应性矩阵
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if H is not None:
                            # 应用变换
                            aligned_frame = cv2.warpPerspective(curr_frame, H, (W, H))
                
                # 转换回PyTorch张量
                aligned_tensor = torch.from_numpy(aligned_frame.astype(np.float32) / 255.0).unsqueeze(0)
                
                # 确保在正确的设备上
                aligned_tensor = aligned_tensor.to(current.device)
                
                # 提取特征
                aligned_feat = self.conv1(aligned_tensor)
                aligned_feat = self.conv2(aligned_feat)
                
                batch_aligned.append(aligned_feat)
            
            # 将批次结果堆叠
            batch_tensor = torch.stack(batch_aligned, dim=0)
            aligned_features.insert(0, batch_tensor)  # 注意顺序，我们从前往后处理，但需要保持时间顺序
        
        # 堆叠时间维度上的特征
        aligned_features = torch.stack(aligned_features, dim=1)  # [B, T, C, H, W]
        
        return aligned_features


class TemporalMultiscaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, channels=64):
        super(TemporalMultiscaleFeatureExtractor, self).__init__()
        self.background_alignment = BackgroundAlignment(channels)
        
        # 差分路径
        self.differential_path = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )
        
        # 动态路径
        self.dynamic_path = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )
        
        # 静态路径
        self.static_path = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )
        
        # 特征聚合
        self.feature_aggregation = ConvBlock(channels * 3, channels * 2)
        
    def forward(self, x):
        # x: [B, T, C, H, W], 输入为时序帧
        B, T, C, H, W = x.shape
        
        # 背景对齐
        aligned_features = self.background_alignment(x)  # [B, T, C', H, W]
        
        # 提取当前帧特征
        current_frame = aligned_features[:, -1]  # [B, C', H, W]
        
        # 差分路径 - 计算当前帧与前一帧的差异
        if T > 1:
            prev_frame = aligned_features[:, -2]  # [B, C', H, W]
            diff_feature = current_frame - prev_frame
            diff_path = self.differential_path(diff_feature)
        else:
            # 如果没有前一帧，则使用零张量
            diff_path = self.differential_path(torch.zeros_like(current_frame))
        
        # 动态路径 - 专注于移动物体
        dynamic_path = self.dynamic_path(current_frame)
        
        # 静态路径 - 处理背景信息
        static_path = self.static_path(current_frame)
        
        # 特征聚合
        combined_features = torch.cat([diff_path, dynamic_path, static_path], dim=1)
        st_features = self.feature_aggregation(combined_features)
        
        return st_features


class SpatialMultiscaleFeatureRefiner(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(SpatialMultiscaleFeatureRefiner, self).__init__()
        
        # Stage 1 - 顶层特征
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2),
            ConvBlock(in_channels//2, in_channels//2)
        )
        
        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 输出层
        self.output_conv = ConvBlock(in_channels * 2, out_channels)
        
    def forward(self, x):
        # Stage 1
        s1 = self.stage1(x)
        
        # Stage 2
        s2_in = F.max_pool2d(s1, kernel_size=2, stride=2)
        s2 = self.stage2(s2_in)
        
        # Stage 3
        s3_in = F.max_pool2d(s2, kernel_size=2, stride=2)
        s3 = self.stage3(s3_in)
        
        # Stage 4
        s4_in = F.max_pool2d(s3, kernel_size=2, stride=2)
        s4 = self.stage4(s4_in)
        
        # 上采样并连接特征
        s4_up = self.upsample(s4)
        s3_out = torch.cat([s3, s4_up], dim=1)
        
        s3_up = self.upsample(s3_out)
        s2_out = torch.cat([s2, s3_up], dim=1)
        
        s2_up = self.upsample(s2_out)
        s1_out = torch.cat([s1, s2_up], dim=1)
        
        # 最终输出
        output = self.output_conv(s1_out)
        
        return output


class PredictionHead(nn.Module):
    def __init__(self, in_channels=128):
        super(PredictionHead, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.output(x)
        return x


class STDMANet(nn.Module):
    def __init__(self, in_channels=1, num_frames=4):
        super(STDMANet, self).__init__()
        self.num_frames = num_frames
        
        # 时间多尺度特征提取器
        self.temporal_extractor = TemporalMultiscaleFeatureExtractor(in_channels)
        
        # 空间多尺度特征细化器
        self.spatial_refiner = SpatialMultiscaleFeatureRefiner(128, 128)
        
        # 预测头
        self.prediction_head = PredictionHead(128)
        
    def forward(self, x):
        """
        x: [B, T, C, H, W] - 批次大小、时间步长、通道数、高度、宽度
        """
        # 时间多尺度特征提取
        st_features = self.temporal_extractor(x)
        
        # 空间多尺度特征细化
        refined_features = self.spatial_refiner(st_features)
        
        # 预测头输出
        pred_map = self.prediction_head(refined_features)
        
        return pred_map
    
    def compute_centers(self, heatmaps):
        """计算目标中心点"""
        B, C, H, W = heatmaps.shape
        centers = []
        
        for b in range(B):
            heatmap = heatmaps[b, 0]  # 单通道热图
            
            # 找到热图中最大值的位置
            flat_indices = torch.argmax(heatmap.view(-1))
            y = flat_indices // W
            x = flat_indices % W
            
            # 归一化坐标
            x_norm = x.float() / W
            y_norm = y.float() / H
            
            centers.append(torch.tensor([x_norm, y_norm]))
            
        return centers
    
    def extract_instances(self, heatmaps, threshold=0.5):
        """从预测热图中提取实例"""
        B, C, H, W = heatmaps.shape
        instances = []
        
        for b in range(B):
            heatmap = heatmaps[b, 0]  # 单通道热图
            
            # 二值化热图
            binary_map = (heatmap > threshold).float()
            
            # 这里简化了实例提取过程，实际上可能需要更复杂的连通组件分析
            # 在实际应用中，可以使用cv2.connectedComponentsWithStats等方法
            instances.append(binary_map)
            
        return instances