from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BalancedLoss', 'FocalLoss', 'GHMCLoss', 'OHNMLoss']


def log_sigmoid(x):
    # for x > 0: 0 - log(1 + exp(-x))
    # for x < 0: x - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)


def log_minus_sigmoid(x):
    # for x > 0: -x - log(1 + exp(-x))
    # for x < 0:  0 - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)

# class BalancedLoss(nn.Module):
#     def __init__(self, neg_weight=1.0, use_soft_target=True, use_iou_loss=False, iou_weight=1.0):
#         super(BalancedLoss, self).__init__()
#         self.neg_weight = neg_weight
#         self.use_soft_target = use_soft_target
#         self.use_iou_loss = use_iou_loss
#         self.iou_weight = iou_weight
#
#     def forward(self, pred_logits, target_logits, pred_boxes=None, target_boxes=None):
#         """
#         pred_logits: (N, H, W) logits map
#         target_logits: (N, H, W) same size as pred
#         pred_boxes: (N, 4) [x1, y1, x2, y2], optional
#         target_boxes: (N, 4), optional
#         """
#         # Soft / Hard target 分支
#         if self.use_soft_target:
#             
#             weight = target_logits.clone()
#             weight = torch.clamp(weight, 1e-6, 1.0)  # 避免除0
#             weight = weight / weight.sum()           # 归一化
#             loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_logits, weight, reduction='sum')
#         else:
#             # 传统 hard 0-1 标签
#             pos_mask = (target_logits == 1)
#             neg_mask = (target_logits == 0)
#
#             pos_num = pos_mask.sum().float().clamp(min=1.0)
#             neg_num = neg_mask.sum().float().clamp(min=1.0)
#
#             weight = target_logits.new_zeros(target_logits.size())
#             weight[pos_mask] = 1.0 / pos_num
#             weight[neg_mask] = self.neg_weight / neg_num
#             weight /= weight.sum()
#             loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_logits, weight, reduction='sum')
#
#         
#         if self.use_iou_loss and pred_boxes is not None and target_boxes is not None:
#             iou = self._compute_iou(pred_boxes, target_boxes)
#             loss_iou = 1 - iou.mean()
#             return loss_cls + self.iou_weight * loss_iou
#
#         return loss_cls
#
#     def _compute_iou(self, boxes1, boxes2):
#         # boxes: [N, 4], format: x1, y1, x2, y2
#         x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
#         y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
#         x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
#         y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
#
#         inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
#         area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#         area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
#         union = area1 + area2 - inter
#         return inter / union.clamp(min=1e-6)



class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')


class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)

        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)

        loss = -(target * pos_weight * pos_log_sig + \
            (1 - target) * neg_weight * neg_log_sig)
        
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        loss /= avg_weight.mean()

        return loss.mean()


class GHMCLoss(nn.Module):
    
    def __init__(self, bins=30, momentum=0.5):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [t / bins for t in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
    
    def forward(self, input, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        tot = input.numel()
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights /= weights.mean()

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        
        return loss


class OHNMLoss(nn.Module):
    
    def __init__(self, neg_ratio=3.0):
        super(OHNMLoss, self).__init__()
        self.neg_ratio = neg_ratio
    
    def forward(self, input, target):
        pos_logits = input[target > 0]
        pos_labels = target[target > 0]

        neg_logits = input[target == 0]
        neg_labels = target[target == 0]

        pos_num = pos_logits.numel()
        neg_num = int(pos_num * self.neg_ratio)
        neg_logits, neg_indices = neg_logits.topk(neg_num)
        neg_labels = neg_labels[neg_indices]

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits, neg_logits]),
            torch.cat([pos_labels, neg_labels]),
            reduction='mean')
        
        return loss
