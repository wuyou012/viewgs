#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import random
from torchmetrics.functional.regression import pearson_corrcoef

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l1_loss_mask_downsample(network_output, gt, mode='bilinear', scale_factor=0.25, mask=None):
    """
    将输入下采样后计算L1损失

    Args:
        network_output (torch.Tensor): 预测输出，形状为[B, C, H, W]或[C, H, W]
        gt (torch.Tensor): 真实标签，形状为[B, C, H, W]或[C, H, W]
        scale_factor (float): 下采样缩放因子，例如0.25表示缩小到原来的1/4
        mode (str): 插值模式，'nearest', 'linear', 'bilinear', 'bicubic'或'trilinear'
        mask (torch.Tensor, optional): 掩码，指定哪些像素参与损失计算

    Returns:
        torch.Tensor: 下采样后的L1损失
    """
    # 确保输入维度正确
    if len(network_output.shape) == 3:  # [C, H, W]
        network_output = network_output.unsqueeze(0)  # 添加批次维度 [1, C, H, W]
        gt = gt.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    # 应用下采样
    align_corners = None if mode == 'nearest' else False
    network_output_down = F.interpolate(
        network_output, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    gt_down = F.interpolate(
        gt, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    # 如果有掩码，也需要下采样
    if mask is not None:
        # 确保掩码是单通道
        if len(mask.shape) == 4 and mask.shape[1] > 1:
            mask = mask[:, 0:1, :, :]  # 只取第一个通道

        # 掩码应该使用最近邻插值以保持二值性质
        mask_down = F.interpolate(mask, scale_factor=scale_factor, mode='nearest')

        # 计算掩码下的L1损失
        loss = torch.abs((network_output_down - gt_down) * mask_down).sum() / (mask_down.sum() + 1e-8)
    else:
        # 计算普通L1损失
        loss = torch.abs(network_output_down - gt_down).mean()

    return loss

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def get_vit_feature(x, ext):
    """
    :x: shape(batch, channel, height, width)
    """
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=x.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                        device=x.device).reshape(1, 3, 1, 1)
    x = F.interpolate(x, size=(224, 224))
    x = (x - mean) / std
    return ext.get_feature_from_input(x)[-1][0, 0, :]

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_partition2(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape

    # Calculate the new height and width that are divisible by window_size
    new_H = H - (H % window_size)
    new_W = W - (W % window_size)

    # Trim the tensor to the new dimensions
    x = x[:, :, :new_H, :new_W]

    # Reshape into windows
    x = x.view(B, C, new_H // window_size, window_size, new_W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)

    return windows

def window_partition3(x, window_size, step_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
        step_size (int): step size for sliding window

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # Calculate the number of windows along height and width
    num_windows_h = (H - window_size) // step_size + 1
    num_windows_w = (W - window_size) // step_size + 1

    # Allocate memory for windows
    windows = x.new_zeros(B, C, num_windows_h, num_windows_w, window_size, window_size)

    # Slide the window over the input tensor
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            y0 = i * step_size
            x0 = j * step_size
            windows[:, :, i, j] = x[:, :, y0:y0+window_size, x0:x0+window_size]

    # Reshape the windows tensor
    windows = windows.view(-1, C, window_size, window_size)

    return windows

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
            (weight_x.sum() + weight_y.sum())
    return loss

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def normalize0(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=0, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=0, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def random_crop(tensor, crop_size):

    c, h, w = tensor.shape

    crop_height, crop_width = crop_size
    if crop_height > h or crop_width > w:
        raise ValueError("Crop size must be less than or equal to the original size.")

    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)

    cropped_tensor = tensor[:, top:top + crop_height, left:left + crop_width]
    return cropped_tensor

def random_crop1(gt_tensor, re_tensor, crop_size):

    _, h, w = gt_tensor.shape
    _, h_re, w_re = re_tensor.shape

    crop_height, crop_width = crop_size
    if crop_height > h or crop_width > w:
        raise ValueError("Crop size must be less than or equal to the original size.")

    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)

    cropped_gt_tensor = gt_tensor[:, top:top + crop_height, left:left + crop_width]

    re_crop_height = 2 * crop_height
    re_crop_width = 2 * crop_width

    re_top = max(top - crop_height // 2, 0)  # 确保不超出上边界
    re_left = max(left - crop_width // 2, 0)  # 确保不超出左边界

    re_bottom = min(re_top + re_crop_height, h_re)
    re_right = min(re_left + re_crop_width, w_re)

    if re_bottom - re_top < re_crop_height:
        re_top = re_bottom - re_crop_height
    if re_right - re_left < re_crop_width:
        re_left = re_right - re_crop_width

    cropped_re_tensor = re_tensor[:, re_top:re_bottom, re_left:re_right]
    # cropped_re_tensor = re_tensor[:, top - 0.5*crop_height:top + 1.5*crop_height, left - 0.5*crop_width:left + 1.5*crop_width]
    # print('gt',top,top + crop_height,left,left + crop_width)
    # print('re',re_top, re_bottom, re_left, re_right)
    return cropped_gt_tensor, cropped_re_tensor
def random_crop3(gt_tensor, re_tensor, crop_size):
    _, h, w = gt_tensor.shape
    _, h_re, w_re = re_tensor.shape

    # Assert that both tensors have the same dimensions
    assert (h == h_re) and (w == w_re), f"Image size mismatch: gt_tensor is {h}x{w}, re_tensor is {h_re}x{w_re}"

    # Ensure crop size does not exceed the image size
    crop_height = min(crop_size[0], h)
    crop_width = min(crop_size[1], w)

    # Generate random top-left corner for the crop within the valid range
    top = torch.randint(0, h - crop_height + 1, (1,)).item()
    left = torch.randint(0, w - crop_width + 1, (1,)).item()

    # Crop both tensors
    gt_cropped = gt_tensor[:, top:top + crop_height, left:left + crop_width]
    re_cropped = re_tensor[:, top:top + crop_height, left:left + crop_width]

    return gt_cropped, re_cropped
def random_crop0(tensor, crop_size):

    b, c, h, w = tensor.shape

    crop_height, crop_width = crop_size
    if crop_height > h or crop_width > w:
        raise ValueError("Crop size must be less than or equal to the original size.")

    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)

    cropped_tensor = tensor[:, :, top:top + crop_height, left:left + crop_width]
    return cropped_tensor

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def mmnormal(input,eps):
    """
    Min-Max normalization
    return [0-1]
    """
    input_norm = (input - input.min()) / (input.max() - input.min() + eps)
    return input_norm

class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256, c=3, l2=False):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj
        self.l2 = l2
        self.c = c
        self.sample_projections()

    def sample_projections(self):
        # Sample random normalized projections
        rand = torch.randn(self.num_proj, self.c*self.patch_size**2) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        self.rand = rand.reshape(self.num_proj, self.c, self.patch_size, self.patch_size)

    def forward(self, x, y, reset_projections=True):
        if reset_projections:
            self.sample_projections()
        self.rand = self.rand.to(x.device)
        # Project patches
        projx = F.conv2d(x, self.rand).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(y, self.rand).transpose(1,0).reshape(self.num_proj, -1)

        # Duplicate patches if number does not equal
        projx, projy = duplicate_to_match_lengths(projx, projy)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        if self.l2:
            loss = ((projx - projy)**2).mean()
        else:
            loss = torch.abs(projx - projy).mean()

        return loss


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2

def depth_correlation_loss(rendered_depth, midas_depth):
    """
    pearson correlation loss。

    Args:
        rendered_depth (torch.Tensor): rendered depth
        midas_depth (torch.Tensor): MiDaS, depth from DPT
    """
    eps = 1e-8
    rendered_depth = mmnormal(rendered_depth,eps)
    midas_depth = mmnormal(midas_depth,eps)
    # check input shape
    if len(rendered_depth.shape) == 3:  # [H, W, 1] or [1, H, W]
        if rendered_depth.shape[0] == 1 or rendered_depth.shape[2] == 1:
            rendered_depth = rendered_depth.reshape(-1, 1)
        else:
            raise ValueError(f"Invalid depth tensor shape: {rendered_depth.shape}")
    elif len(rendered_depth.shape) == 2:  # [H, W]
        rendered_depth = rendered_depth.reshape(-1, 1)
    elif len(rendered_depth.shape) == 1:  # [H*W]
        rendered_depth = rendered_depth.unsqueeze(1)

    if len(midas_depth.shape) == 3:
        if midas_depth.shape[0] == 1 or midas_depth.shape[2] == 1:
            midas_depth = midas_depth.reshape(-1, 1)
        else:
            raise ValueError(f"Invalid depth tensor shape: {midas_depth.shape}")
    elif len(midas_depth.shape) == 2:
        midas_depth = midas_depth.reshape(-1, 1)
    elif len(midas_depth.shape) == 1:
        midas_depth = midas_depth.unsqueeze(1)

    # same device
    if rendered_depth.device != midas_depth.device:
        midas_depth = midas_depth.to(rendered_depth.device)

    loss = min(
        (1 - pearson_corrcoef(-midas_depth, rendered_depth)),
        (1 - pearson_corrcoef(1 / (midas_depth + 0.05), rendered_depth))
    )

    return loss

def edge_aware_depth_correlation_loss(depth_pred, depth_gt, edge_weight=5.0):
    """
    边缘感知的深度相关性损失，更强调深度边缘的保留

    Args:
        depth_pred: 预测的深度图
        depth_gt: 真实深度图
        edge_weight: 边缘权重系数，越大对边缘的重视程度越高
    """
    # 计算深度梯度
    depth_grad_x_pred = torch.abs(depth_pred[:, :-1] - depth_pred[:, 1:])
    depth_grad_y_pred = torch.abs(depth_pred[:-1, :] - depth_pred[1:, :])

    depth_grad_x_gt = torch.abs(depth_gt[:, :-1] - depth_gt[:, 1:])
    depth_grad_y_gt = torch.abs(depth_gt[:-1, :] - depth_gt[1:, :])

    # 计算边缘强度图 - 用于判断哪里是重要边缘
    edge_loss_x = F.l1_loss(depth_grad_x_pred, depth_grad_x_gt, reduction='none')
    edge_loss_y = F.l1_loss(depth_grad_y_pred, depth_grad_y_gt, reduction='none')

    # 单独计算水平和垂直方向的边缘重要性
    edge_importance_x = torch.clamp(depth_grad_x_gt, 0, 1)
    edge_importance_y = torch.clamp(depth_grad_y_gt, 0, 1)

    # 加权边缘损失 - 边缘区域权重更高
    weighted_edge_loss_x = edge_loss_x * (1.0 + edge_importance_x * edge_weight)
    weighted_edge_loss_y = edge_loss_y * (1.0 + edge_importance_y * edge_weight)

    # 计算最终损失
    edge_loss = (weighted_edge_loss_x.mean() + weighted_edge_loss_y.mean()) * 0.5

    # 全局相关性损失，但权重较低
    global_corr = depth_correlation_loss(depth_pred, depth_gt)

    # 组合损失：更重视边缘，较轻视整体相关性
    return edge_loss + 0.2 * global_corr

def multi_scale_edge_aware_depth_loss(depth_pred, depth_gt, scales=[1.0, 0.5, 0.25], edge_weight=5.0):
    """
    多尺度边缘感知深度损失，在多个分辨率下保留边缘结构
    if args.dif_pear_loss_weight != 0:
        dif_pear_loss = loss_utils.multi_scale_edge_aware_depth_loss(
            dif_depth, dif_midas_depth, scales=[1.0, 0.5, 0.25], edge_weight=3.0)
        dif_pear_loss = args.dif_pear_loss_weight * dif_pear_loss
    """
    total_loss = 0.0
    weight = 1.0

    for scale in scales:
        if scale != 1.0:
            # 下采样到当前尺度
            h, w = depth_pred.shape
            new_h, new_w = int(h * scale), int(w * scale)
            depth_pred_scaled = F.interpolate(depth_pred.unsqueeze(0).unsqueeze(0),
                                             size=(new_h, new_w),
                                             mode='bilinear',
                                             align_corners=False).squeeze()
            depth_gt_scaled = F.interpolate(depth_gt.unsqueeze(0).unsqueeze(0),
                                           size=(new_h, new_w),
                                           mode='bilinear',
                                           align_corners=False).squeeze()
        else:
            depth_pred_scaled = depth_pred
            depth_gt_scaled = depth_gt

        # 计算当前尺度的边缘感知损失
        edge_loss = edge_aware_depth_correlation_loss(
            depth_pred_scaled, depth_gt_scaled, edge_weight=edge_weight)

        # 累加损失，给较大尺度更多权重
        total_loss += weight * edge_loss
        weight *= 0.5

    return total_loss

def multi_scale_depth_correlation_loss(depth_pred, depth_gt, scales=[1.0, 0.5, 0.25], weights=None):
    """
    Multi-scale Pearson depth correlation loss.

    Args:
        depth_pred: predicted/rendered depth map
        depth_gt: reference monocular depth map
        scales: downsampling scales
        weights: optional per-scale weights. If omitted, use the historical
            decayed weights [1.0, 0.5, 0.25, ...].
    """
    total_loss = 0.0
    if weights is None:
        weights = [0.5 ** i for i in range(len(scales))]
    elif len(weights) < len(scales):
        weights = list(weights) + [weights[-1]] * (len(scales) - len(weights))

    for scale_idx, scale in enumerate(scales):
        weight = float(weights[scale_idx])
        if scale != 1.0:
            h, w = depth_pred.shape
            new_h, new_w = int(h * scale), int(w * scale)
            depth_pred_scaled = F.interpolate(depth_pred.unsqueeze(0).unsqueeze(0),
                                             size=(new_h, new_w),
                                             mode='bilinear',
                                             align_corners=False).squeeze()
            depth_gt_scaled = F.interpolate(depth_gt.unsqueeze(0).unsqueeze(0),
                                           size=(new_h, new_w),
                                           mode='bilinear',
                                           align_corners=False).squeeze()
        else:
            depth_pred_scaled = depth_pred
            depth_gt_scaled = depth_gt

        corr_loss = depth_correlation_loss(depth_pred_scaled, depth_gt_scaled)
        total_loss += weight * corr_loss

    return total_loss

def enhance_depth_edges(depth, sigma=1.0, amount=1.5):
    """增强深度图的边缘
    dif_midas_depth = estimate_depth(dif_image, mode='train')
    dif_midas_depth = enhance_depth_edges(dif_midas_depth, sigma=1.0, amount=1.8)
    """
    # 转换为NumPy进行处理
    if torch.is_tensor(depth):
        is_tensor = True
        device = depth.device
        depth_np = depth.cpu().numpy()
    else:
        is_tensor = False
        depth_np = depth

    # 使用高斯模糊获取平滑版本
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(depth_np, sigma=sigma)

    # 计算边缘增强
    edge_enhanced = depth_np + (depth_np - blurred) * amount

    # 转回原始格式
    if is_tensor:
        return torch.tensor(edge_enhanced, device=device)
    return edge_enhanced

def windowed_depth_loss(rendered_depth, midas_depth, window_size=126, stride=64):
    """
    depth and window loss。

    Args:
        rendered_depth (torch.Tensor): 渲染的深度图，形状为 [H, W] 或 [1, H, W]
        midas_depth (torch.Tensor): MiDaS深度图，形状为 [H, W] 或 [1, H, W]
        window_size (int): 窗口大小
        stride (int): 窗口步长

    Returns:
        tuple: (全局深度损失, 窗口化深度损失)
    """
    # check input shape [B, C, H, W]
    if rendered_depth.dim() == 2:  # [H, W]
        rendered_depth = rendered_depth.unsqueeze(0).unsqueeze(0)
    elif rendered_depth.dim() == 3:  # [1, H, W] or [H, W, 1]
        if rendered_depth.shape[0] == 1:  # [1, H, W]
            rendered_depth = rendered_depth.unsqueeze(0)
        else:  # [H, W, 1]
            rendered_depth = rendered_depth.permute(2, 0, 1).unsqueeze(0)

    if midas_depth.dim() == 2:  # [H, W]
        midas_depth = midas_depth.unsqueeze(0).unsqueeze(0)
    elif midas_depth.dim() == 3:  # [1, H, W] or [H, W, 1]
        if midas_depth.shape[0] == 1:  # [1, H, W]
            midas_depth = midas_depth.unsqueeze(0)
        else:  # [H, W, 1]
            midas_depth = midas_depth.permute(2, 0, 1).unsqueeze(0)

    # 计算全局深度损失
    rendered_depth_flat = rendered_depth.reshape(-1, 1)
    midas_depth_flat = midas_depth.reshape(-1, 1)
    global_depth_loss = (1 - pearson_corrcoef(rendered_depth_flat, -midas_depth_flat)).mean()

    # 分割窗口
    rendered_windows = window_partition3(rendered_depth, window_size, stride)
    midas_windows = window_partition3(midas_depth, window_size, stride)

    # 从窗口中随机选择一个
    if rendered_windows.shape[0] > 1:
        rn = random.randint(0, rendered_windows.shape[0] - 1)
        rendered_window = rendered_windows[rn].reshape(-1, 1)
        midas_window = midas_windows[rn].reshape(-1, 1)
    else:
        rendered_window = rendered_windows.reshape(-1, 1)
        midas_window = midas_windows.reshape(-1, 1)

    # 标准化窗口值
    normalized_rendered = normalize0(rendered_window)
    normalized_midas = normalize0(midas_window)

    # 计算窗口化损失
    window_depth_loss = (1 - pearson_corrcoef(normalized_rendered, -normalized_midas)).mean()

    return global_depth_loss, window_depth_loss

def gaussian_loss(rendered_rgb, gt_rgb, lambda_dssim, downsample=False, scale_factor=0.25):
    """
    Gaussian basic loss function"
    """

    if downsample:
        Ll1 =  l1_loss_mask_downsample(rendered_rgb, gt_rgb, mode='bicubic', scale_factor=scale_factor)
    else:
        Ll1 =  l1_loss_mask(rendered_rgb, gt_rgb)

    L_Ssim = ssim(rendered_rgb, gt_rgb)
    loss_l1_ssim = ((1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - L_Ssim))

    return loss_l1_ssim, Ll1

def rgb_to_grayscale(image):
    """
    将RGB图像转换为灰度图。

    Args:
        image (torch.Tensor): RGB图像，形状为[B, 3, H, W]或[3, H, W]

    Returns:
        torch.Tensor: 灰度图，与输入保持相同维度，但通道数变为1
    """
    if len(image.shape) == 4:  # [B, 3, H, W]
        # 使用ITU-R BT.601标准权重将RGB转换为灰度
        return 0.299 * image[:, 0, ...] + 0.587 * image[:, 1, ...] + 0.114 * image[:, 2, ...]
    elif len(image.shape) == 3:  # [3, H, W]
        return 0.299 * image[0, ...] + 0.587 * image[1, ...] + 0.114 * image[2, ...]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}. Expected [B, 3, H, W] or [3, H, W]")

def grayscale_correlation_loss(image1, image2):
    """
    计算两个图像转换为灰度后的Pearson相关系数损失。

    Args:
        image1 (torch.Tensor): 第一个图像，形状为[B, 3, H, W]或[3, H, W]
        image2 (torch.Tensor): 第二个图像，形状为[B, 3, H, W]或[3, H, W]

    Returns:
        torch.Tensor: 1减去灰度图之间的Pearson相关系数
    """
    # 转换为灰度图
    gray1 = rgb_to_grayscale(image1)
    gray2 = rgb_to_grayscale(image2)

    eps = 1e-8
    gray1_norm = (gray1 - gray1.min()) / (gray1.max() - gray1.min() + eps)
    gray2_norm = (gray2 - gray2.min()) / (gray2.max() - gray2.min() + eps)

    if gray1_norm.shape[1] != 1:
        gray1_norm = gray1_norm.reshape(-1)
    if gray2_norm.shape[1] != 1:
        gray2_norm = gray2_norm.reshape(-1)

    corr_positive = pearson_corrcoef(gray1_norm, gray2_norm)
    corr_negative = pearson_corrcoef(-gray1_norm, gray2_norm)

    corr_abs_max = torch.max(torch.abs(corr_positive), torch.abs(corr_negative))
    gray_loss = 1 - corr_abs_max

    return gray_loss

def channel_weighted_correlation_loss(image1, image2):
    """
    计算两个图像各通道分别的Pearson相关系数损失，并按灰度权重加权。

    Args:
        image1 (torch.Tensor): 第一个图像，形状为[B, 3, H, W]或[3, H, W]
        image2 (torch.Tensor): 第二个图像，形状为[B, 3, H, W]或[3, H, W]

    Returns:
        torch.Tensor: 加权通道相关系数损失
    """
    # 定义通道权重（与灰度转换权重相同）
    channel_weights = torch.tensor([0.299, 0.587, 0.114], device=image1.device)

    # 确保输入形状正确
    if len(image1.shape) == 3:  # [3, H, W]
        image1 = image1.unsqueeze(0)  # [1, 3, H, W]
        image2 = image2.unsqueeze(0)  # [1, 3, H, W]

    # 检查通道数
    if image1.shape[1] != 3 or image2.shape[1] != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {image1.shape[1]} and {image2.shape[1]}")

    batch_size = image1.shape[0]
    eps = 1e-8

    # 存储每个通道的损失
    channel_losses = []

    # 单独处理每个通道
    for i in range(3):
        # 提取当前通道 [B, H, W]
        ch1 = image1[:, i, ...]
        ch2 = image2[:, i, ...]

        # 归一化到[0,1]范围
        ch1_norm = (ch1 - ch1.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)) / \
                   (ch1.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) -
                    ch1.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) + eps)

        ch2_norm = (ch2 - ch2.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)) / \
                   (ch2.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) -
                    ch2.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) + eps)

        # 展平为向量进行相关系数计算 [B, H*W]
        ch1_flat = ch1_norm.reshape(batch_size, -1)
        ch2_flat = ch2_norm.reshape(batch_size, -1)

        # 计算正相关和负相关
        corr_positive = torch.stack([pearson_corrcoef(ch1_flat[b], ch2_flat[b]) for b in range(batch_size)])
        corr_negative = torch.stack([pearson_corrcoef(-ch1_flat[b], ch2_flat[b]) for b in range(batch_size)])

        # 取绝对值最大的相关系数
        corr_abs_max = torch.max(torch.abs(corr_positive), torch.abs(corr_negative))

        # 计算当前通道的损失 (1 - 相关系数)
        channel_loss = 1 - corr_abs_max
        channel_losses.append(channel_loss)

    # 将通道损失转换为张量 [B, 3]
    channel_losses = torch.stack(channel_losses, dim=1)

    # 按灰度权重加权通道损失
    weighted_loss = torch.sum(channel_losses * channel_weights.unsqueeze(0), dim=1)

    # 返回批次平均损失
    return weighted_loss.mean()

def multi_scale_channel_correlation_loss(image1, image2, scales=[1.0, 0.5, 0.25]):
    """
    多尺度多通道相关系数损失，在不同分辨率下计算通道加权相关系数损失。

    Args:
        image1 (torch.Tensor): 第一个图像，形状为[B, 3, H, W]或[3, H, W]
        image2 (torch.Tensor): 第二个图像，形状为[B, 3, H, W]或[3, H, W]
        scales (list): 缩放因子列表，默认为[1.0, 0.5, 0.25]

    Returns:
        torch.Tensor: 多尺度通道加权相关系数损失
    """
    # 确保输入形状正确
    if len(image1.shape) == 3:  # [3, H, W]
        image1 = image1.unsqueeze(0)  # [1, 3, H, W]
        image2 = image2.unsqueeze(0)  # [1, 3, H, W]

    # 存储每个尺度的损失
    scale_losses = []

    # 在不同尺度上计算损失
    for scale in scales:
        if scale != 1.0:
            # 对图像进行下采样
            img1_scaled = F.interpolate(image1, scale_factor=scale, mode='bilinear', align_corners=False)
            img2_scaled = F.interpolate(image2, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            img1_scaled = image1
            img2_scaled = image2

        # 计算当前尺度的损失
        scale_loss = channel_weighted_correlation_loss(img1_scaled, img2_scaled)
        scale_losses.append(scale_loss)

    # 计算所有尺度的平均损失
    multi_scale_loss = torch.mean(torch.stack(scale_losses))

    return multi_scale_loss

def advanced_channel_correlation_loss(image1, image2, use_multi_scale=True, scales=[1.0, 0.5, 0.25],
                                     use_gradient_images=True, use_combined_loss=True):
    """
    高级通道相关性损失函数，结合多尺度、梯度图像和组合损失。

    Args:
        image1 (torch.Tensor): 第一个图像，形状为[B, 3, H, W]或[3, H, W]
        image2 (torch.Tensor): 第二个图像，形状为[B, 3, H, W]或[3, H, W]
        use_multi_scale (bool): 是否使用多尺度评估
        scales (list): 缩放因子列表
        use_gradient_images (bool): 是否计算图像梯度的相关性
        use_combined_loss (bool): 是否同时使用原始灰度相关和通道加权相关

    Returns:
        torch.Tensor: 综合相关系数损失
    """
    # 确保输入形状正确
    if len(image1.shape) == 3:  # [3, H, W]
        image1 = image1.unsqueeze(0)  # [1, 3, H, W]
        image2 = image2.unsqueeze(0)  # [1, 3, H, W]

    losses = []

    # 1. 计算通道加权相关性损失
    if use_multi_scale:
        channel_loss = multi_scale_channel_correlation_loss(image1, image2, scales)
    else:
        channel_loss = channel_weighted_correlation_loss(image1, image2)
    losses.append(channel_loss)

    # 2. 计算图像梯度的相关性（边缘相关性）
    if use_gradient_images:
        # 计算x和y方向的梯度
        dx_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=image1.device)
        dy_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=image1.device)

        # 为每个RGB通道重复卷积核
        dx_kernel = dx_kernel.repeat(3, 1, 1, 1)
        dy_kernel = dy_kernel.repeat(3, 1, 1, 1)

        # 计算图像梯度
        grad1_x = F.conv2d(image1, dx_kernel, padding=1, groups=3)
        grad1_y = F.conv2d(image1, dy_kernel, padding=1, groups=3)
        grad2_x = F.conv2d(image2, dx_kernel, padding=1, groups=3)
        grad2_y = F.conv2d(image2, dy_kernel, padding=1, groups=3)

        # 合并x和y方向的梯度
        grad1 = torch.sqrt(grad1_x**2 + grad1_y**2)
        grad2 = torch.sqrt(grad2_x**2 + grad2_y**2)

        # 计算梯度图像的相关性损失
        grad_loss = channel_weighted_correlation_loss(grad1, grad2)
        losses.append(grad_loss)

    # 3. 同时使用原始灰度相关（如果需要）
    if use_combined_loss:
        gray_loss = grayscale_correlation_loss(image1, image2)
        losses.append(gray_loss)

    # 组合所有损失
    combined_loss = torch.mean(torch.stack(losses))

    return combined_loss

def structural_consistency_loss(image1, image2):
    """
    计算两个图像之间的结构一致性，不关注亮度值
    使用SSIM的结构部分而非完整SSIM
    # 为深度图添加结构一致性损失
    struct_loss = structural_consistency_loss(dif_depth, dif_midas_depth)
    dif_pear_loss += args.struct_weight * struct_loss  # 新增参数

    """
    # 提取结构信息（通过局部方差和协方差）
    window_size = 11
    pad = window_size // 2

    # 填充图像
    img1_pad = F.pad(image1.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='replicate')
    img2_pad = F.pad(image2.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='replicate')

    # 创建窗口
    window = torch.ones(1, 1, window_size, window_size, device=image1.device) / (window_size * window_size)

    # 计算局部均值
    mu1 = F.conv2d(img1_pad, window).squeeze()
    mu2 = F.conv2d(img2_pad, window).squeeze()

    # 计算局部方差和协方差
    sigma1_sq = F.conv2d(img1_pad * img1_pad, window).squeeze() - mu1 * mu1
    sigma2_sq = F.conv2d(img2_pad * img2_pad, window).squeeze() - mu2 * mu2
    sigma12 = F.conv2d(img1_pad * img2_pad, window).squeeze() - mu1 * mu2

    # 计算结构相似性
    C3 = 0.03**2  # 小常数防止除零
    cs = (2 * sigma12 + C3) / (sigma1_sq + sigma2_sq + C3)

    # 返回结构一致性损失（1-cs）
    return (1 - cs).mean()

def laplacian_pyramid_loss(x, target, max_levels=3, kernel_size=5, sigma=1.0):
    """
    使用拉普拉斯金字塔计算多尺度图像损失

    Args:
        x: 输入图像 [B, C, H, W] 或 [C, H, W]
        target: 目标图像 [B, C, H, W] 或 [C, H, W]
        max_levels: 拉普拉斯金字塔的层数
        kernel_size: 高斯核的大小
        sigma: 高斯核的标准差

    Returns:
        损失值: 所有层级损失的加权和
    """
    # 确保输入是4D张量 [B, C, H, W]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    # 获取通道数
    _, channels, _, _ = x.shape
    device = x.device

    # 创建高斯核
    kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, device=device)

    # 计算拉普拉斯金字塔
    x_pyramid = create_laplacian_pyramid(x, kernel, max_levels)
    target_pyramid = create_laplacian_pyramid(target, kernel, max_levels)

    # 计算所有层级的加权损失
    total_loss = 0.0
    level_weights = [0.8**i for i in range(max_levels)]  # 赋予较低层级更高的权重
    level_weights.append(0.8**max_levels)  # 为最顶层高斯残差添加权重

    for i, (x_level, target_level) in enumerate(zip(x_pyramid, target_pyramid)):
        level_loss = F.l1_loss(x_level, target_level)
        total_loss += level_weights[i] * level_loss

    return total_loss

def gaussian_kernel(size=5, channels=3, sigma=1.0, device=torch.device('cuda')):
    """
    创建高斯核

    Args:
        size: 核大小
        channels: 通道数
        sigma: 高斯标准差
        device: 计算设备

    Returns:
        高斯核张量 [channels, 1, size, size]
    """
    # 创建高斯核（在Numpy中）
    x = torch.linspace(-(size - 1) / 2., (size - 1) / 2., size, device=device)
    x = x.view(-1, 1)
    y = x.clone().view(1, -1)

    # 计算高斯函数
    kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # 调整形状为 [channels, 1, size, size]
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    return kernel

def downsample(x):
    """
    对图像进行2倍降采样

    Args:
        x: 输入图像 [B, C, H, W]

    Returns:
        下采样图像 [B, C, H/2, W/2]
    """
    return x[:, :, ::2, ::2]

def create_laplacian_pyramid(x, kernel, levels):
    """
    创建拉普拉斯金字塔，处理任意尺寸的输入图像

    Args:
        x: 输入图像 [B, C, H, W]
        kernel: 高斯核
        levels: 金字塔层数

    Returns:
        拉普拉斯金字塔层列表
    """
    pyramids = []
    current_x = x

    for _ in range(levels):
        # 高斯滤波
        gauss_filtered = gaussian_conv2d(current_x, kernel)
        # 下采样
        down = downsample(gauss_filtered)

        # 使用目标尺寸进行上采样，确保与current_x尺寸完全匹配
        up = F.interpolate(down, size=(current_x.shape[2], current_x.shape[3]),
                          mode='bilinear', align_corners=False)

        # 计算拉普拉斯层 = 当前层 - 上采样(下采样层)
        laplacian = current_x - up

        # 将拉普拉斯层添加到金字塔中
        pyramids.append(laplacian)
        current_x = down

    # 添加最顶层的高斯残差
    pyramids.append(current_x)

    return pyramids

def gaussian_conv2d(x, kernel):
    """
    应用高斯卷积

    Args:
        x: 输入图像 [B, C, H, W]
        kernel: 高斯核 [C, 1, K, K]

    Returns:
        卷积后的图像 [B, C, H, W]
    """
    # 确定通道数和填充大小
    channels = kernel.shape[0]
    padding = kernel.shape[-1] // 2

    # 使用分组卷积，为每个通道应用相同的卷积核
    return F.conv2d(x, weight=kernel, padding=padding, groups=channels)