# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
import kornia.filters as KF
import kornia.losses as KL
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def Fusion_loss(vi, ir, fu, weights=[10, 10, 1], device=None):
    """
    计算图像融合损失，包含梯度损失和强度损失。

    参数:
    vi (torch.Tensor): 可见光图像张量，形状通常为 (batch_size, channels, height, width)。
    ir (torch.Tensor): 红外图像张量，形状通常为 (batch_size, 1, height, width)。
    fu (torch.Tensor): 融合后的图像张量，形状通常为 (batch_size, channels, height, width)。
    weights (list, 可选): 梯度损失和强度损失的权重，默认值为 [10, 10]。
    device (torch.device, 可选): 计算设备，如 'cpu' 或 'cuda'，默认为 None。

    返回:
    tuple: 包含总损失、强度损失和梯度损失的元组。
    """
    # 将可见光图像转换为灰度图像，在通道维度上求均值
    vi_gray = torch.mean(vi, 1, keepdim=True)
    # 将融合后的图像转换为灰度图像，在通道维度上求均值
    fu_gray = torch.mean(fu, 1, keepdim=True)
    # 初始化 Sobel 算子卷积模块，用于计算图像的梯度
    sobelconv = Sobelxy(device)

    # 梯度损失计算部分
    # 计算可见光图像在 x 和 y 方向的梯度
    vi_grad_x, vi_grad_y = sobelconv(vi_gray)
    # 计算红外图像在 x 和 y 方向的梯度
    ir_grad_x, ir_grad_y = sobelconv(ir)
    # 计算融合后图像在 x 和 y 方向的梯度
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    # 取可见光和红外图像在 x 方向梯度的最大值，作为联合梯度
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    # 取可见光和红外图像在 y 方向梯度的最大值，作为联合梯度
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    # 计算梯度损失，使用 L1 损失函数分别计算 x 和 y 方向的梯度损失并求和
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    loss_ssim = final_ssim(ir, vi, fu)
    # 强度损失计算部分
    # 计算融合图像与可见光图像的均方误差，加上融合灰度图像小于红外图像部分的绝对误差
    loss_intensity = torch.mean(torch.pow((fu - vi), 2)) + torch.mean((fu_gray < ir) * torch.abs((fu_gray - ir)))

    # 计算总损失，将梯度损失和强度损失按给定的权重加权求和
    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity +  weights[1] * loss_ssim
    return loss_total, loss_intensity, loss_grad,loss_ssim


class Sobelxy(nn.Module):
    """
    自定义 PyTorch 模块，用于计算输入图像在 x 和 y 方向的 Sobel 梯度。
    Sobel 算子是一种常用的边缘检测算子，可用于提取图像的边缘信息。
    """
    def __init__(self, device):
        """
        初始化 Sobelxy 模块。

        参数:
        device (torch.device): 计算设备，用于指定模型参数存储的设备，如 'cpu' 或 'cuda'。
        """
        # 调用父类 nn.Module 的构造函数
        super(Sobelxy, self).__init__()
        # 定义 Sobel 算子在 x 方向的卷积核
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        # 定义 Sobel 算子在 y 方向的卷积核
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        # 将 x 方向的卷积核转换为 PyTorch 的 FloatTensor 类型，
        # 并添加两个维度以符合卷积层输入的形状要求 (out_channels, in_channels, height, width)
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        # 将 y 方向的卷积核转换为 PyTorch 的 FloatTensor 类型，
        # 并添加两个维度以符合卷积层输入的形状要求 (out_channels, in_channels, height, width)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        # 将 x 方向的卷积核作为不可训练的参数，移动到指定设备
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        # 将 y 方向的卷积核作为不可训练的参数，移动到指定设备
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)

    def forward(self, x):
        """
        前向传播函数，计算输入图像在 x 和 y 方向的 Sobel 梯度。

        参数:
        x (torch.Tensor): 输入图像张量，形状通常为 (batch_size, channels, height, width)。

        返回:
        tuple: 包含 x 和 y 方向 Sobel 梯度绝对值的元组。
        """
        # 使用 x 方向的卷积核进行 2D 卷积操作，填充为 1 以保持输出尺寸与输入相同
        sobelx = F.conv2d(x, self.weightx, padding=1)
        # 使用 y 方向的卷积核进行 2D 卷积操作，填充为 1 以保持输出尺寸与输入相同
        sobely = F.conv2d(x, self.weighty, padding=1)
        # 返回 x 和 y 方向 Sobel 梯度的绝对值
        return torch.abs(sobelx), torch.abs(sobely)


def Seg_loss(pred, label, device, criteria=None):
    '''
    利用预训练好的分割网络,计算在融合结果上的分割结果与真实标签之间的语义损失
    :param fused_image:
    :param label:
    :param seg_model: 分割模型在主函数中提前加载好,避免每次充分load分割模型
    :return seg_loss:
    fused_image 在输入Seg_loss函数之前需要由YCbCr色彩空间转换至RGB色彩空间
    '''
    # 计算语义损失
    lb = torch.squeeze(label, 1)
    if criteria is None:
        raise ValueError("criteria cannot be None")
    seg_loss = criteria(pred, lb)
    return seg_loss

class OhemCELoss(nn.Module):
    """
    实现 Online Hard Example Mining (OHEM) 的交叉熵损失函数。
    OHEM 方法在训练过程中动态选择损失较大的难样本进行训练，有助于提高模型的性能。
    """
    def __init__(self, thresh, n_min, device, ignore_lb=255, *args, **kwargs):
        """
        初始化 OhemCELoss 类。

        参数:
        thresh (float): 损失阈值，用于筛选难样本。损失值大于该阈值的样本会被选中。
        n_min (int): 最小样本数量。当按阈值筛选出的样本数量不足时，选取损失值最大的前 n_min 个样本。
        device (torch.device): 计算设备，如 'cpu' 或 'cuda'，用于将张量放置到相应设备上。
        ignore_lb (int, 可选): 要忽略的标签值，在计算损失时会忽略该标签对应的样本。默认为 255。
        """
        super(OhemCELoss, self).__init__()
        # 计算损失阈值的负对数，并将其移动到指定设备上
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(device)
        # 存储最小样本数量
        self.n_min = n_min
        # 存储要忽略的标签值
        self.ignore_lb = ignore_lb
        # 初始化交叉熵损失函数，忽略指定标签值，且不进行损失值的缩减，返回每个样本的损失
        #self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        self.criteria = DynamicLabelSmoothSoftmaxCEV1()

    def forward(self, logits, labels):
        """
        前向传播函数，计算 OHEM 交叉熵损失。

        参数:
        logits (torch.Tensor): 模型的输出 logits 张量，形状通常为 (N, C, H, W)，
                               其中 N 是批量大小，C 是类别数，H 和 W 是图像的高度和宽度。
        labels (torch.Tensor): 真实标签张量，形状通常为 (N, H, W)。

        返回:
        torch.Tensor: 计算得到的 OHEM 交叉熵损失。
        """
        # 获取 logits 张量的形状信息
        N, C, H, W = logits.size()
        # 使用交叉熵损失函数计算每个样本的损失，并将结果展平为一维张量
        loss = self.criteria(logits, labels).view(-1)
        # 对损失值进行降序排序
        loss, _ = torch.sort(loss, descending=True)
        
        # 确保索引不会越界
        if loss.size(0) <= self.n_min:
            # 如果有效损失数量小于等于 n_min，直接返回所有损失的平均值
            return torch.mean(loss)
        
        # 判断排序后第 n_min 个样本的损失值是否大于阈值
        if loss[self.n_min] > self.thresh:
            # 若大于阈值，则选取损失值大于阈值的样本
            loss = loss[loss>self.thresh]
        else:
            # 若小于等于阈值，则选取损失值最大的前 n_min 个样本
            loss = loss[:self.n_min]
        # 计算选中样本的平均损失
        return torch.mean(loss)


class DynamicLabelSmoothSoftmaxCEV1(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=255):
        super(DynamicLabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        logits = input.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = target.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg) \
                .scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        dynamic_weight = self.get_weight(input)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss = loss * dynamic_weight
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


    @staticmethod
    # 动态权重调整
    def get_weight(input):
        num_classes = input.size(1)
        x = 1e-6
        logits = input.float()
        #计算全局置信度
        probs_logits = F.softmax(logits, dim=1)
        max_pred_b, max_idx_b = torch.max(probs_logits, dim=1)
        max_pred_c, max_idx_c = torch.max(probs_logits, dim=0)
        # 确保 u_t 是标量
        u_t = torch.mean(max_pred_c)#全局置平均置信度
        u_t_tensor = torch.full_like(max_pred_b, u_t.item())
        diff_squared = (max_pred_c - u_t) ** 2
        # 确保 variance_t 是标量
        variance_t = torch.mean(diff_squared)

        #低置信度增加权重
        weight = torch.ones_like(max_pred_b)
        lambda_max = 1.0
        update = max_pred_b < u_t_tensor
        if torch.any(update):
            # 确保有更新的元素才进行计算，避免空张量运算
            update_indices = torch.where(update)[0]
            if len(update_indices) > 0:
                weight[update] = lambda_max * torch.exp(
                    -((max_pred_b[update] - u_t_tensor[update]) ** 2) / (2 * variance_t + x))

        return weight


def symmetric_cross_entropy(outputs, labels, alpha=1.0, beta=0.1):  # 调整参数比例
    ce = F.cross_entropy(outputs, labels, reduction='none')

    probs = F.softmax(outputs, dim=-1)
    targets_one_hot = F.one_hot(labels, num_classes=outputs.size(-1)).float()

    # 更安全的数值处理
    probs_clamped = torch.clamp(probs, min=1e-10, max=1.0)
    targets_clamped = torch.clamp(targets_one_hot, min=1e-10, max=1.0)

    rce = -torch.sum(probs_clamped * torch.log(targets_clamped), dim=1)

    return (alpha * ce + beta * rce).mean()

# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret





def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window

def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma1

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# 添加SSIM损失函数，基于kornia.losses.SSIMLoss
def SSIM_loss(img1, img2, window_size=11):
    """
    使用kornia库计算SSIM损失
    参数:
    img1 (torch.Tensor): 第一张图像
    img2 (torch.Tensor): 第二张图像
    window_size (int): 窗口大小
    返回:
    torch.Tensor: SSIM损失值
    """
    ssim_loss = KL.SSIMLoss(window_size=window_size)
    return ssim_loss(img1, img2)

# 添加多尺度SSIM损失函数
def Multi_Scale_SSIM_loss(img1, img2, window_size=11, scales=[1, 2, 4]):
    """
    计算多尺度SSIM损失
    参数:
    img1 (torch.Tensor): 第一张图像
    img2 (torch.Tensor): 第二张图像
    window_size (int): 窗口大小
    scales (list): 不同尺度
    返回:
    torch.Tensor: 多尺度SSIM损失值
    """
    total_loss = 0.0
    weight = 1.0 / len(scales)
    
    for scale in scales:
        # 根据尺度调整图像大小
        if scale != 1:
            # 使用插值调整图像大小
            img1_scaled = F.interpolate(img1, scale_factor=1.0/scale, mode='bilinear', align_corners=False)
            img2_scaled = F.interpolate(img2, scale_factor=1.0/scale, mode='bilinear', align_corners=False)
        else:
            img1_scaled = img1
            img2_scaled = img2
            
        # 计算该尺度的SSIM损失
        ssim_loss = SSIM_loss(img1_scaled, img2_scaled, window_size)
        total_loss += weight * ssim_loss
        
    return total_loss

# 添加动态SSIM权重调整函数
def adaptive_ssim_weight(epoch, total_epochs, initial_weight=0.1, final_weight=1.0):
    """
    根据训练进度动态调整SSIM损失权重
    参数:
    epoch (int): 当前epoch
    total_epochs (int): 总epoch数
    initial_weight (float): 初始权重
    final_weight (float): 最终权重
    返回:
    float: 当前SSIM权重
    """
    # 使用sigmoid函数实现平滑过渡
    progress = epoch / total_epochs
    weight = initial_weight + (final_weight - initial_weight) * progress
    return weight

# 添加增强的融合损失函数
def Enhanced_Fusion_loss(vi, ir, fu, epoch=0, total_epochs=100, weights=[10, 10, 0.5], device=None):
    """
    增强的图像融合损失，包含梯度损失、强度损失和SSIM损失
    参数:
    vi (torch.Tensor): 可见光图像
    ir (torch.Tensor): 红外图像
    fu (torch.Tensor): 融合图像
    epoch (int): 当前训练epoch
    total_epochs (int): 总训练epoch数
    weights (list): [梯度损失权重, 强度损失权重, SSIM损失权重]
    device (torch.device): 设备
    返回:
    tuple: (总损失, 强度损失, 梯度损失, SSIM损失)
    """
    # 将可见光图像转换为灰度图像
    vi_gray = torch.mean(vi, 1, keepdim=True)
    # 将融合后的图像转换为灰度图像
    fu_gray = torch.mean(fu, 1, keepdim=True)
    # 初始化 Sobel 算子卷积模块
    sobelconv = Sobelxy(device)

    # 梯度损失计算
    vi_grad_x, vi_grad_y = sobelconv(vi_gray)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    # 强度损失计算
    loss_intensity = torch.mean(torch.pow((fu - vi), 2)) + torch.mean((fu_gray < ir) * torch.abs((fu_gray - ir)))

    # SSIM损失计算
    # 对红外图像和融合图像计算SSIM损失
    ssim_loss_ir = Multi_Scale_SSIM_loss(ir, fu_gray)
    # 对可见光图像和融合图像计算SSIM损失
    ssim_loss_vi = Multi_Scale_SSIM_loss(vi_gray, fu_gray)
    # 综合SSIM损失
    ssim_loss = (ssim_loss_ir + ssim_loss_vi) / 2.0

    # 动态调整SSIM权重
    ssim_weight = adaptive_ssim_weight(epoch, total_epochs, 0.1, 1.0)
    
    # 计算总损失
    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity + ssim_weight * weights[2] * ssim_loss
    
    return loss_total, loss_intensity, loss_grad, ssim_loss
