import cv2
import numpy as np
import torch


def align_frames_sift_ransac(source_frame, target_frame, min_matches=10):
    """
    使用SIFT和RANSAC算法对齐两帧图像
    
    参数:
        source_frame: 源图像(numpy数组，单通道，值范围0-255的uint8类型)
        target_frame: 目标图像(numpy数组，单通道，值范围0-255的uint8类型)
        min_matches: 最小匹配点数量
        
    返回:
        aligned_frame: 对齐后的源图像
        success: 是否成功对齐
    """
    # 确保输入图像是正确的格式
    if source_frame.dtype != np.uint8:
        source_frame = (source_frame * 255).astype(np.uint8)
    if target_frame.dtype != np.uint8:
        target_frame = (target_frame * 255).astype(np.uint8)
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测特征点并计算描述符
    kp1, des1 = sift.detectAndCompute(source_frame, None)
    kp2, des2 = sift.detectAndCompute(target_frame, None)
    
    # 如果没有足够的特征点，返回原始图像
    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return source_frame, False
    
    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        # 如果FLANN匹配失败，尝试使用暴力匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
    
    # 应用Lowe比率测试筛选好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # 如果没有足够的好匹配点，返回原始图像
    if len(good_matches) < min_matches:
        return source_frame, False
    
    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 使用RANSAC计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return source_frame, False
    
    # 应用变换
    h, w = target_frame.shape
    aligned_frame = cv2.warpPerspective(source_frame, H, (w, h))
    
    return aligned_frame, True


def batch_align_frames(batch_frames, reference_idx=-1, device=None):
    """
    批量对齐帧序列
    
    参数:
        batch_frames: 批次帧张量 [B, T, C, H, W]
        reference_idx: 参考帧索引，默认为最后一帧
        device: 计算设备
        
    返回:
        aligned_frames: 对齐后的帧张量 [B, T, C, H, W]
    """
    B, T, C, H, W = batch_frames.shape
    
    # 确保在CPU上处理numpy数组
    cpu_frames = batch_frames.detach().cpu()
    
    # 初始化结果数组
    aligned_frames = torch.zeros_like(batch_frames)
    
    # 将参考帧复制到结果中
    aligned_frames[:, reference_idx] = batch_frames[:, reference_idx]
    
    # 对每个批次样本进行处理
    for b in range(B):
        ref_frame = cpu_frames[b, reference_idx, 0].numpy()
        
        # 对齐除参考帧外的所有帧
        for t in range(T):
            if t == reference_idx:
                continue
                
            source_frame = cpu_frames[b, t, 0].numpy()
            
            # 应用SIFT+RANSAC对齐
            aligned_frame, success = align_frames_sift_ransac(source_frame, ref_frame)
            
            # 转换回张量并保存
            aligned_tensor = torch.from_numpy(aligned_frame.astype(np.float32) / 255.0)
            aligned_frames[b, t, 0] = aligned_tensor
    
    # 确保返回张量在正确的设备上
    if device is not None:
        aligned_frames = aligned_frames.to(device)
    elif batch_frames.device.type != 'cpu':
        aligned_frames = aligned_frames.to(batch_frames.device)
    
    return aligned_frames
