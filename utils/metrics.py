import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from skimage import measure


class PD_FA():
    """
    计算检测概率(PD)和虚警率(FA)
    
    参数:
        nclass: 类别数
        bins: 分箱数量
        size: 图像大小
    """
    def __init__(self, nclass, bins, size):
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.size = size
        
    def update(self, preds, labels):
        """
        更新PD和FA计算
        
        参数:
            preds: 预测热图
            labels: 目标标签
        """
        # 获取实际输入的形状
        if len(preds.shape) > 1:
            # 如果输入是多维张量，直接使用最后两个维度作为图像尺寸
            actual_height, actual_width = preds.shape[-2], preds.shape[-1]
        else:
            # 如果是一维张量，尝试计算最接近的平方数
            total_elements = preds.numel()
            actual_size = int(np.sqrt(total_elements))
            if actual_size * actual_size != total_elements:
                raise ValueError(f"无法将大小为 {total_elements} 的一维数组重塑为正方形图像")
            actual_height = actual_width = actual_size
        
        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits = np.array((preds > score_thresh/255).cpu()).astype('int64')
            
            # 使用实际大小进行重塑
            if len(predits.shape) == 1:
                predits = np.reshape(predits, (actual_height, actual_width))
            elif len(predits.shape) > 2:
                # 如果是批量数据，只取第一个样本
                predits = predits.reshape(-1, actual_height, actual_width)[0]
            
            labelss = np.array((labels).cpu()).astype('int64')
            if len(labelss.shape) == 1:
                labelss = np.reshape(labelss, (actual_height, actual_width))
            elif len(labelss.shape) > 2:
                # 如果是批量数据，只取第一个样本
                labelss = labelss.reshape(-1, actual_height, actual_width)[0]

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            # 创建一个副本而不是修改原始列表
            remaining_coord_image = coord_image.copy()
            
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                matched = False
                
                # 找到最近的中心点
                min_distance = float('inf')
                min_index = -1
                
                for m, region in enumerate(remaining_coord_image):
                    centroid_image = np.array(list(region.centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    
                    # 自适应阈值，根据图像大小设置
                    adaptive_threshold = max(3, min(actual_height, actual_width) * 0.05)
                    if distance < adaptive_threshold and distance < min_distance:
                        min_distance = distance
                        min_index = m
                        matched = True
                
                if matched:
                    self.distance_match.append(min_distance)
                    self.image_area_match.append(np.array(remaining_coord_image[min_index].area))
                    # 从列表中移除已匹配的项
                    remaining_coord_image.pop(min_index)

            # 计算未匹配的区域
            # 将numpy数组转换为标量值，使其可哈希
            matched_areas_values = set([float(area) for area in self.image_area_match])
            self.dismatch = [area for area in self.image_area_total if float(area) not in matched_areas_values]
            
            self.FA[iBin] += len(self.dismatch)
            self.PD[iBin] += len(self.distance_match)
    
    def get_metrics(self):
        """
        获取计算结果
        
        返回:
            pd_values: 检测概率值
            fa_values: 虚警率值
        """
        pd_values = self.PD / (self.target + 1e-6)
        fa_values = self.FA / (self.target + 1e-6)
        return pd_values, fa_values


def calculate_precision_recall(pred, target, threshold=0.5):
    """
    计算精确率和召回率
    
    参数:
        pred: 预测热图 [N, H, W]
        target: 目标热图 [N, H, W]
        threshold: 二值化阈值
        
    返回:
        precision: 精确率
        recall: 召回率
    """
    # 将预测和目标展平为1D张量
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 二值化预测
    pred_binary = (pred_flat > threshold).float()
    
    # 计算TP, FP, FN
    true_positive = (pred_binary * target_flat).sum()
    false_positive = pred_binary.sum() - true_positive
    false_negative = target_flat.sum() - true_positive
    
    # 计算精确率和召回率
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    
    return precision.item(), recall.item()


def calculate_f1_score(precision, recall):
    """
    计算F1分数
    
    参数:
        precision: 精确率
        recall: 召回率
        
    返回:
        f1_score: F1分数
    """
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1_score


def calculate_iou(pred, target, threshold=0.5):
    """
    计算IoU (Intersection over Union)
    
    参数:
        pred: 预测热图 [N, H, W]
        target: 目标热图 [N, H, W]
        threshold: 二值化阈值
        
    返回:
        iou: IoU值
    """
    # 将预测和目标展平为1D张量
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 二值化预测
    pred_binary = (pred_flat > threshold).float()
    
    # 计算交集和并集
    intersection = (pred_binary * target_flat).sum()
    union = pred_binary.sum() + target_flat.sum() - intersection
    
    # 计算IoU
    iou = intersection / (union + 1e-6)
    
    return iou.item()


def calculate_ap(pred, target, threshold=0.5):
    """
    计算平均精度 (Average Precision)
    
    参数:
        pred: 预测热图 [N, H, W]
        target: 目标热图 [N, H, W]
        threshold: 用于将连续目标值转换为二进制格式的阈值
        
    返回:
        ap: 平均精度
    """
    # 确保输入是numpy数组
    if isinstance(pred, torch.Tensor):
        pred_flat = pred.flatten().cpu().numpy()
    else:
        pred_flat = np.array(pred).flatten()
        
    if isinstance(target, torch.Tensor):
        target_flat = target.flatten().cpu().numpy()
    else:
        target_flat = np.array(target).flatten()
    
    # 将连续目标值转换为二进制格式
    target_binary = (target_flat > threshold).astype(np.int64)
    
    # 检查数据有效性
    if len(np.unique(target_binary)) < 2:
        return 0.0
    
    # 确保预测值在[0,1]范围内
    pred_flat = np.clip(pred_flat, 0, 1)
    
    try:
        ap = average_precision_score(target_binary, pred_flat)
    except Exception as e:
        print(f"计算AP时出错: {str(e)}")
        ap = 0.0
    
    return ap


def calculate_false_alarm_rate(pred, target, threshold=0.5):
    """
    计算误检率 (False Alarm Rate)
    """
    # 将预测和目标展平为1D张量
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 二值化预测
    pred_binary = (pred_flat > threshold).float()
    
    # 计算FP（误检）和地面真实负样本总数
    true_positive = (pred_binary * target_flat).sum()
    false_positive = pred_binary.sum() - true_positive
    
    # 问题修正: FA应该是FP与负样本总数的比值，而不是与正样本总数的比值
    true_negative_total = (target_flat == 0).sum()  # 真实负样本总数
    
    # 计算误检率，即误检目标数与真实负样本总数的比值
    fa = false_positive / (true_negative_total + 1e-6)
    
    return fa.item()


def calculate_detection_metrics(pred_centers, gt_centers, distance_threshold=0.1):
    """
    计算目标检测相关指标
    
    参数:
        pred_centers: 预测的目标中心点列表 [(x1, y1), (x2, y2), ...]
        gt_centers: 真实的目标中心点列表 [(x1, y1), (x2, y2), ...]
        distance_threshold: 距离阈值，用于确定检测是否为真阳性
        
    返回:
        metrics: 包含各类检测指标的字典
    """
    if len(gt_centers) == 0:
        if len(pred_centers) == 0:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0
            }
        else:
            return {
                'precision': 0.0,
                'recall': 1.0,
                'f1_score': 0.0
            }
    
    if len(pred_centers) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # 计算预测中心点与真实中心点之间的距离矩阵
    distances = np.zeros((len(pred_centers), len(gt_centers)))
    
    for i, pred in enumerate(pred_centers):
        for j, gt in enumerate(gt_centers):
            # 计算欧几里得距离
            dist = np.sqrt(((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2).item())
            distances[i, j] = dist
    
    # 标记匹配的中心点
    matched_pred = set()
    matched_gt = set()
    
    # 贪心匹配 - 每次找到最近的一对点进行匹配
    while True:
        if len(matched_pred) == len(pred_centers) or len(matched_gt) == len(gt_centers):
            break
            
        # 找到距离最小的未匹配点对
        min_dist = float('inf')
        min_i, min_j = -1, -1
        
        for i in range(len(pred_centers)):
            if i in matched_pred:
                continue
                
            for j in range(len(gt_centers)):
                if j in matched_gt:
                    continue
                    
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    min_i, min_j = i, j
        
        # 如果最小距离大于阈值，则停止匹配
        if min_dist > distance_threshold:
            break
        
        # 标记匹配的点对
        matched_pred.add(min_i)
        matched_gt.add(min_j)
    
    # 计算真阳性、假阳性和假阴性
    tp = len(matched_pred)
    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    
    # 计算精确率、召回率和F1分数
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def calculate_pd_fa(pred, target, img_size=128, bins=10):
    """
    计算检测概率(PD)和虚警率(FA)
    
    参数:
        pred: 预测热图 [N, H, W]
        target: 目标热图 [N, H, W]
        img_size: 图像大小
        bins: 分箱数量
        
    返回:
        pd_rate: 最佳阈值下的检测概率
        fa_rate: 最佳阈值下的虚警率
        best_threshold: 最佳阈值
    """
    # 预处理
    pred_np = pred.clone().detach()
    target_np = target.clone().detach()
    
    # 获取实际图像大小
    actual_size = pred_np.shape[-1]
    
    # 计算多个阈值下的PD和FA
    pd_fa_calculator = PD_FA(1, bins, actual_size)
    pd_fa_calculator.update(pred_np, target_np)
    pd_values, fa_values = pd_fa_calculator.get_metrics()
    
    # 计算F1分数以找到最佳阈值
    f1_scores = 2 * pd_values / (pd_values + fa_values + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = best_idx * (255/bins) / 255
    
    return pd_values[best_idx], fa_values[best_idx], best_threshold


def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    评估模型性能
    
    参数:
        model: 待评估模型
        data_loader: 数据加载器
        device: 使用的设备 (CPU/GPU)
        threshold: 二值化阈值
        
    返回:
        metrics: 包含各类评估指标的字典
    """
    model.eval()
    
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_iou = 0
    total_ap = 0
    total_fa = 0
    total_pd = 0
    
    # 获取数据集中的图像大小和键名
    sample_batch = next(iter(data_loader))
    # 检测批次中的键名以确定图像和掩码的键
    keys = list(sample_batch.keys())
    
    # 假设第一个键是图像，第二个键是掩码
    # 或者尝试常见的键名组合
    if 'images' in keys and ('masks' in keys or 'targets' in keys or 'labels' in keys):
        image_key = 'images'
        mask_key = 'masks' if 'masks' in keys else ('targets' if 'targets' in keys else 'labels')
    elif len(keys) >= 2:
        # 如果没有标准键名，使用第一个和第二个键
        image_key, mask_key = keys[0], keys[1]
    else:
        raise ValueError(f"无法确定数据批次中的图像和掩码键，可用键: {keys}")
    
    img_size = sample_batch[image_key].shape[-1]  # 假设图像是正方形的
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch[image_key].to(device)
            targets = batch[mask_key].to(device)
            
            # 前向传播
            outputs = model(images)
            pred_probs = outputs['out'] if isinstance(outputs, dict) else outputs
            
            # 计算各种指标
            precision, recall = calculate_precision_recall(pred_probs.cpu(), targets.cpu(), threshold)
            f1 = calculate_f1_score(precision, recall)
            iou = calculate_iou(pred_probs.cpu(), targets.cpu(), threshold)
            ap = calculate_ap(pred_probs.cpu(), targets.cpu(), threshold)
            fa = calculate_false_alarm_rate(pred_probs.cpu(), targets.cpu(), threshold)
            pd_rate, fa_rate, _ = calculate_pd_fa(pred_probs.cpu(), targets.cpu(), img_size, bins=10)
            
            # 累加指标
            total_precision += precision
            total_recall += recall
            total_f1_score += f1
            total_iou += iou
            total_ap += ap
            total_fa += fa
            total_pd += pd_rate
    
    # 计算平均指标
    num_batches = len(data_loader)
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1_score = total_f1_score / num_batches
    avg_iou = total_iou / num_batches
    avg_ap = total_ap / num_batches
    avg_fa = total_fa / num_batches
    avg_pd = total_pd / num_batches
    
    metrics = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score,
        'iou': avg_iou,
        'ap': avg_ap,
        'fa': avg_fa,
        'pd': avg_pd
    }
    
    return metrics
