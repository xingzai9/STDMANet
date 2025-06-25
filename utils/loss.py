import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskWeightedHeatmapLoss(nn.Module):
    """
    带掩码权重的热图损失，如图1所示
    L = (p_t - g_t)^2 ⊙ (mask + λ(1 - mask))
    
    其中：
    p_t: 预测热图
    g_t: 目标热图
    mask: 掩码，突出目标区域
    λ: 权重因子，平衡目标区域和背景区域的贡献
    """
    def __init__(self, lambda_weight=0.1):
        super(MaskWeightedHeatmapLoss, self).__init__()
        self.lambda_weight = lambda_weight
        
    def forward(self, pred, target, mask):
        """
        参数:
            pred: 预测热图 [B, 1, H, W]
            target: 目标热图 [B, 1, H, W]
            mask: 目标区域掩码 [B, 1, H, W]
        返回:
            loss: 加权热图损失
        """
        # 计算均方误差
        mse = (pred - target) ** 2
        
        # 计算权重掩码
        weight = mask + self.lambda_weight * (1 - mask)
        
        # 应用掩码权重
        weighted_mse = mse * weight
        
        # 计算平均损失
        loss = weighted_mse.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for红外小目标检测
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        
    def forward(self, pred, target):
        """
        参数:
            pred: 预测热图 [B, 1, H, W]
            target: 目标热图 [B, 1, H, W]
        返回:
            loss: Focal Loss
        """
        # 将预测和目标压缩到[0,1]区间
        pred = torch.sigmoid(pred)
        
        # 计算二值交叉熵
        bce = F.binary_cross_entropy(pred + self.eps, target, reduction='none')
        
        # 计算p_t
        pt = target * pred + (1 - target) * (1 - pred)
        
        # 计算权重因子
        weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # 计算Focal Loss
        focal_loss = weight * (1 - pt).pow(self.gamma) * bce
        
        # 应用reduction策略
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 确保pred经过sigmoid激活到[0,1]范围
        pred = torch.sigmoid(pred)
        
        # 计算交集和并集
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        # 计算IoU
        iou = (intersection_sum + self.smooth) / \
              (pred_sum + target_sum - intersection_sum + self.smooth)
    
        # 返回1-IoU作为损失
        loss = 1 - iou.mean()

        return loss

class IoULoss(nn.Module):
    """
    IoU Loss计算预测热图和目标热图之间的IoU损失
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        参数:
            pred: 预测热图 [B, 1, H, W]
            target: 目标热图 [B, 1, H, W]
        返回:
            loss: 1 - IoU
        """
        # 将预测输出转换为概率
        pred = torch.sigmoid(pred)
        
        # 计算交集
        intersection = (pred * target).sum(dim=(2, 3))
        
        # 计算并集
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 计算损失: 1 - IoU
        loss = 1 - iou.mean()
        
        return loss




class CombinedLoss(nn.Module):
    """
    组合损失函数：结合掩码加权损失、Focal Loss和IoU损失
    """
    def __init__(self, lambda_mask=1.0, lambda_focal=0.5, lambda_iou=0.5, mask_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mask_loss = MaskWeightedHeatmapLoss(lambda_weight=mask_weight)
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss()
        
        # 权重因子
        self.lambda_mask = lambda_mask
        self.lambda_focal = lambda_focal
        self.lambda_iou = lambda_iou
        
    def forward(self, pred, target, mask):
        """
        参数:
            pred: 预测热图 [B, 1, H, W]
            target: 目标热图 [B, 1, H, W]
            mask: 目标区域掩码 [B, 1, H, W]
        返回:
            loss: 组合损失
        """
        # 计算各个损失项
        mask_loss = self.mask_loss(pred, target, mask)
        focal_loss = self.focal_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        
        # 计算总损失
        total_loss = self.lambda_mask * mask_loss + self.lambda_focal * focal_loss + self.lambda_iou * iou_loss
        
        return total_loss, {
            'mask_loss': mask_loss.item(),
            'focal_loss': focal_loss.item(),
            'iou_loss': iou_loss.item()
        }
    

class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()


    def forward(self, pred_log, target, epoch, warm_epoch = 5, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        
        
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum  + smooth)       
        lloss = LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss
    
    

def LLoss(pred, target):
        loss = torch.tensor(0.0, requires_grad=True).to(pred)

        patch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]        
        x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
        y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
        smooth = 1e-8
        for i in range(patch_size):  

            pred_centerx = (x_index*pred[i]).mean()
            pred_centery = (y_index*pred[i]).mean()

            target_centerx = (x_index*target[i]).mean()
            target_centery = (y_index*target[i]).mean()
           
            angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                            - torch.arctan((target_centery) / (target_centerx + smooth))))

            pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
            target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)
            
            length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
        
            loss = loss + (1 - length_loss + angle_loss) / patch_size
        
        return loss