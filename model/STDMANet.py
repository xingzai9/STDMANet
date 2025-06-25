import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# -----------------------------------
# 1) CTSAM 模块
# -----------------------------------
class CTSAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Conv1 + Conv2
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        # 通道-时域注意力的 MLP W
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False))
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, 7, 1, 3)
    
    def forward(self, x):
        # 1) 基础卷积
        H = self.conv2(self.conv1(x))   # [B, C, H, W]
        B, C, Ht, Wt = H.shape
        
        # 2) 通道-时域注意力 MCT
        # 平面池化得 [B, C]
        max_pool = F.adaptive_max_pool2d(H, 1).view(B, C)
        avg_pool = F.adaptive_avg_pool2d(H, 1).view(B, C)
        MCT = torch.sigmoid(
            self.mlp(max_pool) + self.mlp(avg_pool)
        ).view(B, C, 1, 1)
        HCT = H * MCT
        
        # 3) 空间注意力 MS
        # 对通道做全局池化，得到 [B,1,H,W] 两个
        mp = torch.max(HCT, dim=1, keepdim=True)[0]
        ap = torch.mean(HCT, dim=1, keepdim=True)
        MS = torch.sigmoid(self.spatial_conv(torch.cat([mp, ap], dim=1)))
        Hout = HCT * MS
        
        return Hout

# -----------------------------------
# 2) 背景对齐 + 时序多尺度特征提取器
# -----------------------------------
class BackgroundAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        # 参考帧特征提取
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, 3,1,1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
    
    def forward(self, frames):
        # frames: [B, T, 1, H, W]
        B, T, C, H, W = frames.shape
        ref = frames[:,-1]                    # [B,1,H,W]
        ref_feat = self.feat(ref)             # [B,64,H,W]
        
        aligned = []                          # 保存每帧对齐后特征
        
        # 对前 T-1 帧进行批处理
        for t in range(T-1):
            batch_aligned = []
            
            # 处理批次中的每个样本
            for b in range(B):
                cur_frame = frames[b, t, 0].detach().cpu().numpy()
                ref_frame = frames[b, -1, 0].detach().cpu().numpy()
                
                # 确保数据类型和值范围正确
                cur_frame = (cur_frame * 255).astype(np.uint8)
                ref_frame = (ref_frame * 255).astype(np.uint8)
                
                # 使用SIFT提取特征点和描述符
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(cur_frame, None)
                kp2, des2 = sift.detectAndCompute(ref_frame, None)
                
                aligned_frame = cur_frame.copy()
                
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
                        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if homography is not None:
                            # 应用变换，注意dsize参数应该是(width, height)
                            aligned_frame = cv2.warpPerspective(cur_frame, homography, (W, H))
                
                # 转换回PyTorch张量
                aligned_tensor = torch.from_numpy(aligned_frame.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                # 确保在正确的设备上
                aligned_tensor = aligned_tensor.to(frames.device)
                # 提取特征
                aligned_feat = self.feat(aligned_tensor)  # [1, 64, H, W]
                batch_aligned.append(aligned_feat.squeeze(0))  # 移除batch维度变为[64, H, W]
            
            # 将批次结果堆叠
            batch_tensor = torch.stack(batch_aligned, dim=0)  # [B, 64, H, W]
            aligned.append(batch_tensor)
        
        # 添加参考帧特征到最后
        aligned.append(ref_feat)  # [B, 64, H, W]
        
        # 堆叠时间维度上的特征
        aligned_features = torch.stack(aligned, dim=1)  # [B, T, 64, H, W]
        
        return aligned_features

class TemporalMultiscaleFeatureExtractor(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.align = BackgroundAlignment()
        # 修复差分路径的通道数，应该是 64*(k-1)
        self.diff_conv = CTSAM(64*(k-1))  # 修改通道数以匹配输入
        # 修复动态路径的通道数，应该是 64*(k-1) + 64*k
        self.dynamic_conv = CTSAM(64*(k-1) + 64*k)
        # 静态路径的通道数，应该是 64
        self.static_conv = CTSAM(64)
        # 最后特征聚合的通道数
        self.agg = CTSAM(64*(k-1) + 64*(k-1) + 64*k + 64)
        # 添加额外的卷积层调整最终输出通道为固定值
        self.final_conv = nn.Conv2d(64*(k-1) + 64*(k-1) + 64*k + 64, 256, 1, 1, 0)
    
    def forward(self, x):
        # x: [B, T, 1, H, W]
        B, T, C, H, W = x.shape
        aligned = self.align(x)             # [B, T, 64, H, W]
        # current
        cur = aligned[:,-1]                 # [B,64,H,W]
        # 差分 inputs & mask Md
        diffs = []
        for i in range(T-1):
            prev = aligned[:,i]
            diff = cur - prev
            # 计算 mask，简单忽略零区
            md = (prev!=0).float()
            diffs.append(diff * md)
        # 修改 DIt 的计算方式，保持通道维度清晰
        DIt = torch.cat(diffs, dim=1)       # [B,64*(T-1),H,W]
        # dynamic path input [abs(diff), aligned all]
        # 将 aligned 正确重塑为 [B, 64*T, H, W]
        aligned_flat = aligned.view(B, T*64, H, W)
        DYt = torch.cat([torch.abs(DIt), aligned_flat], dim=1)
        # static
        SIt = cur
        # 提取
        FDI = self.diff_conv(DIt)
        FDY = self.dynamic_conv(DYt)
        FSP = self.static_conv(SIt)
        # 聚合
        Fcat = torch.cat([FDI, FDY, FSP], dim=1)
        FT = self.agg(Fcat)
        # 添加最终卷积调整通道数为固定值
        FT = self.final_conv(FT)
        return FT  # [B, 256, H, W]

# -----------------------------------
# 3) 空间多尺度特征细化器
# -----------------------------------
# class SpatialMultiscaleFeatureRefiner(nn.Module):
#     def __init__(self, C_in, L=2):
#         super().__init__()
#         self.L = L
#         self.stages = nn.ModuleList()
#         self.channel_adjust = nn.ModuleList()
#         self.C_in = C_in
        
#         # 计算每个阶段和层的输入通道数
#         for s in range(4):
#             blocks = nn.ModuleList()
#             adjust_layers = nn.ModuleList()
#             for l in range(L):
#                 # 为每个位置计算确切的输入通道数
#                 # TODO：错误
#                 if l == 0:
#                     # 第一层只接收初始输入
#                     in_channels = C_in
#                 else:
#                     # 后续层接收前一层输出
#                     in_channels = C_in
                    
#                 # 添加跨尺度连接的通道数
#                 cross_channels = 0
#                 if s > 0:  # 有上层连接
#                     cross_channels += C_in
#                 if s < 3:  # 有下层连接
#                     cross_channels += C_in
                
#                 total_channels = in_channels + cross_channels
                
#                 # 添加通道调整层，将总通道数调整为C_in
#                 if total_channels != C_in:
#                     adjust_layers.append(nn.Conv2d(total_channels, C_in, 1, 1, 0))
#                 else:
#                     adjust_layers.append(nn.Identity())
                    
#                 blocks.append(CTSAM(C_in))
#             self.stages.append(blocks)
#             self.channel_adjust.append(adjust_layers)
class SpatialMultiscaleFeatureRefiner(nn.Module):
    def __init__(self, C_in, L=2, growth_rate=32):
        super().__init__()
        self.L = L
        self.growth_rate = growth_rate
        self.stages = nn.ModuleList()
        self.channel_adjust = nn.ModuleList()
        
        # 用于初始输入通道调整的卷积层
        self.init_conv = nn.ModuleList([
            nn.Conv2d(C_in, growth_rate, 1, 1, 0) for _ in range(4)
        ])
        
        for s in range(4):  # 4个尺度
            blocks = nn.ModuleList()
            adjust_layers = nn.ModuleList()
            
            for l in range(L):  # 每个尺度有L层
                # 当前层的输入通道数 (更准确的计算)
                # 每个尺度在l层的通道数=(l+1)*growth_rate
                # 例如：第0层后有1*growth_rate，第1层后有2*growth_rate等
                dense_channels = (l + 1) * growth_rate
                
                # 跨尺度连接通道数
                cross_channels = 0
                if s > 0:  # 上层连接
                    cross_channels += growth_rate  # 每个连接只贡献growth_rate通道
                if s < 3:  # 下层连接
                    cross_channels += growth_rate  # 每个连接只贡献growth_rate通道
                
                total_channels = dense_channels + cross_channels
                
                # 通道调整层 - 将输入调整为growth_rate，作为该层的输出
                adjust_layers.append(nn.Conv2d(total_channels, growth_rate, 1, 1, 0))
                blocks.append(CTSAM(growth_rate))
                
            self.stages.append(blocks)
            self.channel_adjust.append(adjust_layers)

    def forward(self, x):
        # x: [B, C, H, W] = [B, 256, H, W]
        B, C, H, W = x.shape
        
        # 下采样得到四个尺度的初始特征
        G0 = x  # [B, 256, H, W]
        G1 = F.max_pool2d(G0, 2)  # [B, 256, H/2, W/2]
        G2 = F.max_pool2d(G1, 2)  # [B, 256, H/4, W/4]
        G3 = F.max_pool2d(G2, 2)  # [B, 256, H/8, W/8]
        
        # 存储各尺度各层的特征
        features = [[], [], [], []]  # [尺度][层]
        
        # 首先将初始输入调整为growth_rate通道
        features[0].append(self.init_conv[0](G0))
        features[1].append(self.init_conv[1](G1))
        features[2].append(self.init_conv[2](G2))
        features[3].append(self.init_conv[3](G3))
        
        # 对每个尺度处理L层DenseBlock
        for l in range(self.L):
            for s in range(4):
                # 所有现有特征 - 这就是DenseBlock的核心
                # 只取到当前层的特征(索引l+1)，不要包括还未生成的后续层
                dense_features = features[s][:l+1]
                
                # 收集跨尺度连接 - 只取相邻尺度的最新特征(索引l)
                cross_scale_feats = []
                if s > 0:  # 上层连接
                    up_feat = F.max_pool2d(features[s-1][l], 2)
                    cross_scale_feats.append(up_feat)
                    
                if s < 3:  # 下层连接
                    down_feat = F.interpolate(
                        features[s+1][l], 
                        size=(features[s][0].shape[2], features[s][0].shape[3]),
                        mode='bilinear', 
                        align_corners=False
                    )
                    cross_scale_feats.append(down_feat)
                
                # 连接所有特征
                all_features = dense_features + cross_scale_feats
                cat_feats = torch.cat(all_features, dim=1)
                
                # 调整通道并应用CTSAM
                adjusted_feats = self.channel_adjust[s][l](cat_feats)
                output = self.stages[s][l](adjusted_feats)
                
                # 添加到当前尺度的特征列表
                features[s].append(output)
        
        # 汇聚四个尺度最后一层输出
        out0 = features[0][-1]  # [B, growth_rate, H, W]
        out1 = F.interpolate(features[1][-1], scale_factor=2, mode='bilinear', align_corners=False)
        out2 = F.interpolate(features[2][-1], scale_factor=4, mode='bilinear', align_corners=False)
        out3 = F.interpolate(features[3][-1], scale_factor=8, mode='bilinear', align_corners=False)
        
        return torch.cat([out0, out1, out2, out3], dim=1)  # [B, growth_rate*4, H, W]

# -----------------------------------
# 4) Prediction Head
# -----------------------------------
class PredictionHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ctsam = CTSAM(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 1, 1)
    def forward(self, x):
        x = self.ctsam(x)
        return self.conv1(x)  # [B,1,H,W] heatmap

# -----------------------------------
# 5) 完整 STDMANet
# -----------------------------------
class STDMANet(nn.Module):
    def __init__(self, in_channels=1, num_frames=5):
        super().__init__()
        self.temp = TemporalMultiscaleFeatureExtractor(k=num_frames)
        # 现在 FT 通道数固定为 256
        self.spat = SpatialMultiscaleFeatureRefiner(C_in=256, L=2, growth_rate=32)
        # 修正：输出是 growth_rate*4 = 32*4 = 128 通道
        self.head = PredictionHead(in_ch=32*4)  # 4 scales * growth_rate
    def forward(self, x):
        # x: [B, T, 1, H, W]
        FT = self.temp(x)            # [B, 256, H, W]
        FS = self.spat(FT)           # [B, 128, H, W]
        heat = self.head(FS)         # [B, 1, H, W]
        return heat