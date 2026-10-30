import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

# 全局变量，用于缓存光流模型
flow_model = None

def load_keypoint_data(keypoints_dir, view_idx1, view_idx2, mid_view_idx=None):
    """
    从保存的NPZ文件中加载关键点和投影数据

    Args:
        keypoints_dir: 关键点目录路径
        view_idx1, view_idx2: 视图索引
        mid_view_idx: 中间视图索引，如果为None则随机选择

    Returns:
        dict: 包含关键点数据和相机参数的字典
    """
    # 加载匹配点数据
    keypoints_file = os.path.join(keypoints_dir, f'keypoints_{view_idx1}_{view_idx2}.npz')
    if not os.path.exists(keypoints_file):
        raise FileNotFoundError(f"找不到匹配点文件: {keypoints_file}")

    keypoints_data = np.load(keypoints_file)

    # 加载中间相机参数
    mid_cameras_file = os.path.join(keypoints_dir, f'mid_cameras_{view_idx1}_{view_idx2}.npz')
    if not os.path.exists(mid_cameras_file):
        raise FileNotFoundError(f"找不到中间相机参数文件: {mid_cameras_file}")

    mid_cameras_data = np.load(mid_cameras_file)

    # 选择中间视图
    n_mid_views = len(mid_cameras_data['R'])
    if mid_view_idx is None:
        mid_view_idx = np.random.randint(0, n_mid_views)
    elif mid_view_idx >= n_mid_views:
        raise ValueError(f"中间视图索引 {mid_view_idx} 超出范围 (0-{n_mid_views-1})")

    # 加载对应的投影点
    projected_file = os.path.join(keypoints_dir, f'projected_{view_idx1}_{view_idx2}_view{mid_view_idx:03d}.npy')
    if not os.path.exists(projected_file):
        # 尝试不带前导零的格式
        projected_file = os.path.join(keypoints_dir, f'projected_{view_idx1}_{view_idx2}_view{mid_view_idx}.npy')
        if not os.path.exists(projected_file):
            raise FileNotFoundError(f"找不到投影点文件: {projected_file}")

    projected_points = np.load(projected_file)

    # 构造返回结果
    result = {
        'matches_im0': keypoints_data['matches_im0'],
        'matches_im1': keypoints_data['matches_im1'],
        'points3D': keypoints_data['points3D'],
        'colors': keypoints_data['colors'],
        'projected_points': projected_points,
        'mid_camera_R': mid_cameras_data['R'][mid_view_idx],
        'mid_camera_T': mid_cameras_data['T'][mid_view_idx],
        'mid_view_idx': mid_view_idx,
        'intrinsics': mid_cameras_data['intrinsics']
    }

    return result

def load_ply_point_cloud(ply_file):
    """
    从PLY文件加载点云数据

    Args:
        ply_file: PLY文件路径

    Returns:
        points: 点云坐标 [N,3]
        colors: 点云颜色 [N,3] (如果有)
    """
    # 简单实现 - 实际使用时可能需要更强大的PLY解析器
    with open(ply_file, 'r') as f:
        lines = f.readlines()

    # 查找数据开始的行
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break

    # 检查是否有颜色信息
    has_colors = False
    for line in lines[:header_end]:
        if "property uchar red" in line:
            has_colors = True
            break

    # 读取点云数据
    points = []
    colors = [] if has_colors else None

    for line in lines[header_end:]:
        values = line.strip().split()
        if len(values) >= 3:
            points.append([float(values[0]), float(values[1]), float(values[2])])

            if has_colors and len(values) >= 6:
                colors.append([int(values[3]), int(values[4]), int(values[5])])

    return np.array(points), np.array(colors) if colors else None

def get_flow_model():
    """获取光流模型，使用单例模式避免重复加载"""
    global flow_model
    if flow_model is None:
        # 使用'unimatch'模型并加载预训练权重
        flow_model = ptlflow.get_model('unimatch', pretrained_ckpt='mix')
        print("已加载光流模型: unimatch")
    return flow_model

def compute_optical_flow(source_img, target_img):
    """
    计算从源图像到目标图像的光流

    Args:
        source_img: 源图像 [3,H,W] 或 [H,W,3]
        target_img: 目标图像 [3,H,W] 或 [H,W,3]

    Returns:
        flow: 光流场 [2,H,W]
    """
    # 确保图像格式正确
    if torch.is_tensor(source_img):
        if source_img.dim() == 4:  # [B,C,H,W]
            source_img = source_img.squeeze(0)

        if source_img.shape[0] == 3:  # [C,H,W]
            source_img = source_img.permute(1, 2, 0)  # 转换为 [H,W,C]

        source_img = source_img.detach().cpu().numpy()
        # 确保值范围在0-255
        if source_img.max() <= 1.0:
            source_img = (source_img * 255).astype(np.uint8)

    if torch.is_tensor(target_img):
        if target_img.dim() == 4:  # [B,C,H,W]
            target_img = target_img.squeeze(0)

        if target_img.shape[0] == 3:  # [C,H,W]
            target_img = target_img.permute(1, 2, 0)  # 转换为 [H,W,C]

        target_img = target_img.detach().cpu().numpy()
        # 确保值范围在0-255
        if target_img.max() <= 1.0:
            target_img = (target_img * 255).astype(np.uint8)

    # 确保图像是uint8类型
    source_img = source_img.astype(np.uint8)
    target_img = target_img.astype(np.uint8)

    # 获取光流模型
    model = get_flow_model()

    # 创建输入适配器
    io_adapter = IOAdapter(model, source_img.shape[:2])

    # 准备输入
    inputs = io_adapter.prepare_inputs([source_img, target_img])

    # 计算光流
    with torch.no_grad():
        predictions = model(inputs)

    # 提取光流
    flows = predictions['flows']
    flow = flows[0, 0]  # [2,H,W]

    return flow

def warp_points_with_flow(points, flow):
    """
    使用光流将点从源图像变形到目标图像

    Args:
        points: 点坐标 [N,2]
        flow: 光流场 [2,H,W]

    Returns:
        warped_points: 变形后的点坐标 [N,2]
    """
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32, device=flow.device)

    # 确保点在有效范围内
    H, W = flow.shape[1], flow.shape[2]
    points = torch.clamp(points, 0, torch.tensor([W-1, H-1], device=flow.device))

    # 获取每个点位置的光流向量
    point_flows = []
    for point in points:
        x, y = int(point[0].item()), int(point[1].item())
        point_flow = flow[:, y, x]  # [2]
        point_flows.append(point_flow)

    point_flows = torch.stack(point_flows)  # [N,2]

    # 应用光流得到变形后的点
    warped_points = points + point_flows

    # 确保变形后的点在有效范围内
    warped_points = torch.clamp(warped_points, 0, torch.tensor([W-1, H-1], device=flow.device))

    return warped_points

def keypoint_distance_loss(pred_points, gt_points, distance_threshold=10.0):
    """
    计算预测点和真实点之间的距离损失

    Args:
        pred_points: 预测点坐标 [N,2]
        gt_points: 真实点坐标 [M,2]
        distance_threshold: 距离阈值，超过此值的匹配将被忽略

    Returns:
        loss: 点距离损失
    """
    device = pred_points.device if torch.is_tensor(pred_points) else torch.device('cuda')

    # 确保输入是PyTorch张量
    if not torch.is_tensor(pred_points):
        pred_points = torch.tensor(pred_points, dtype=torch.float32, device=device)

    if not torch.is_tensor(gt_points):
        gt_points = torch.tensor(gt_points, dtype=torch.float32, device=device)

    # 计算所有点对之间的欧几里得距离
    N, M = pred_points.shape[0], gt_points.shape[0]

    # 如果任一集合为空，则返回零损失
    if N == 0 or M == 0:
        return torch.tensor(0.0, device=device)

    # 计算距离矩阵 [N,M]
    pred_expanded = pred_points.unsqueeze(1)  # [N,1,2]
    gt_expanded = gt_points.unsqueeze(0)  # [1,M,2]
    distances = torch.sqrt(torch.sum((pred_expanded - gt_expanded)**2, dim=2))  # [N,M]

    # 对每个预测点，找到最近的真实点
    min_distances, _ = torch.min(distances, dim=1)  # [N]

    # 过滤掉距离太远的匹配
    valid_matches = min_distances < distance_threshold

    # 如果没有有效匹配，返回零损失
    if not torch.any(valid_matches):
        return torch.tensor(0.0, device=device)

    # 计算有效匹配的平均距离作为损失
    loss = min_distances[valid_matches].mean()

    return loss

def compute_keypoint_constraint_loss(rendered_image, target_image, keypoints_data):
    """
    计算关键点约束损失

    Args:
        rendered_image: 3DGS渲染的图像 [3,H,W]
        target_image: 目标GT图像 [3,H,W]
        keypoints_data: 关键点数据字典

    Returns:
        loss: 关键点约束损失
        metadata: 调试元数据
    """
    # 计算光流
    flow = compute_optical_flow(rendered_image, target_image)

    # 提取投影点
    projected_points = torch.tensor(keypoints_data['projected_points'], dtype=torch.float32, device=flow.device)

    # 提取在图像边界内的点
    H, W = flow.shape[1], flow.shape[2]
    valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < W) & \
                (projected_points[:, 1] >= 0) & (projected_points[:, 1] < H)
    valid_points = projected_points[valid_mask]

    # 使用光流变形点
    warped_points = warp_points_with_flow(valid_points, flow)

    # 计算距离损失
    loss = keypoint_distance_loss(warped_points, valid_points)

    # 收集元数据
    metadata = {
        'projected_points': valid_points.cpu().numpy(),
        'warped_points': warped_points.cpu().numpy(),
        'flow': flow.cpu().numpy(),
        'valid_count': valid_mask.sum().item(),
        'total_count': len(projected_points)
    }

    return loss, metadata

def visualize_keypoint_constraint(rendered_image, target_image, metadata, save_path=None):
    """
    可视化关键点约束

    Args:
        rendered_image: 3DGS渲染图像
        target_image: 目标GT图像
        metadata: 约束元数据
        save_path: 保存路径
    """
    # 转换为numpy数组
    if torch.is_tensor(rendered_image):
        if rendered_image.dim() == 4:  # [B,C,H,W]
            rendered_image = rendered_image.squeeze(0)

        if rendered_image.shape[0] == 3:  # [C,H,W]
            rendered_image = rendered_image.permute(1, 2, 0)  # 转换为 [H,W,C]

        rendered_image = rendered_image.detach().cpu().numpy()
        # 确保值范围在0-1
        if rendered_image.max() > 1.0:
            rendered_image = rendered_image / 255.0

    if torch.is_tensor(target_image):
        if target_image.dim() == 4:  # [B,C,H,W]
            target_image = target_image.squeeze(0)

        if target_image.shape[0] == 3:  # [C,H,W]
            target_image = target_image.permute(1, 2, 0)  # 转换为 [H,W,C]

        target_image = target_image.detach().cpu().numpy()
        # 确保值范围在0-1
        if target_image.max() > 1.0:
            target_image = target_image / 255.0

    # 创建图像
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 渲染图像
    axes[0].imshow(rendered_image)
    axes[0].set_title("Rendered Image")
    axes[0].axis('off')

    # 目标图像
    axes[1].imshow(target_image)
    axes[1].set_title("Target Image")
    axes[1].axis('off')

    # 关键点对比
    axes[2].imshow(rendered_image, alpha=0.7)

    # 绘制投影点
    projected_points = metadata['projected_points']
    axes[2].scatter(projected_points[:, 0], projected_points[:, 1], c='red', s=10, label='Projected')

    # 绘制变形点
    warped_points = metadata['warped_points']
    axes[2].scatter(warped_points[:, 0], warped_points[:, 1], c='blue', s=10, label='Warped')

    # 绘制连接线
    for i in range(len(projected_points)):
        axes[2].plot([projected_points[i, 0], warped_points[i, 0]],
                    [projected_points[i, 1], warped_points[i, 1]],
                    'g-', linewidth=0.5)

    axes[2].set_title(f"Keypoint Matches ({len(projected_points)} points)")
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def triangulate_matches_gpu(matches_im0, matches_im1, extrinsics_w2c, intrinsics, view_idx1, view_idx2, save_dir=None, images=None):
    """
    使用GPU加速的三角测量实现，重建3D点云

    Args:
        matches_im0, matches_im1: 匹配点坐标 (PyTorch张量)
        extrinsics_w2c, intrinsics: 相机参数
        view_idx1, view_idx2: 视图索引
        save_dir: 保存点云的目录
    """
    import torch

    # 获取设备
    device = matches_im0.device

    # 获取相机参数
    K1 = torch.tensor(intrinsics[view_idx1], device=device, dtype=torch.float32)
    K2 = torch.tensor(intrinsics[view_idx2], device=device, dtype=torch.float32)

    R1 = torch.tensor(extrinsics_w2c[view_idx1][:3, :3], device=device, dtype=torch.float32)
    t1 = torch.tensor(extrinsics_w2c[view_idx1][:3, 3], device=device, dtype=torch.float32)

    R2 = torch.tensor(extrinsics_w2c[view_idx2][:3, :3], device=device, dtype=torch.float32)
    t2 = torch.tensor(extrinsics_w2c[view_idx2][:3, 3], device=device, dtype=torch.float32)

    # 构建投影矩阵
    P1 = torch.zeros((3, 4), device=device, dtype=torch.float32)
    P1[:3, :3] = R1
    P1[:3, 3] = t1
    P1 = K1 @ P1

    P2 = torch.zeros((3, 4), device=device, dtype=torch.float32)
    P2[:3, :3] = R2
    P2[:3, 3] = t2
    P2 = K2 @ P2

    # 确保匹配点在GPU上
    if not torch.is_tensor(matches_im0):
        matches_im0 = torch.tensor(matches_im0, device=device, dtype=torch.float32)
    if not torch.is_tensor(matches_im1):
        matches_im1 = torch.tensor(matches_im1, device=device, dtype=torch.float32)

    # 将齐次坐标添加为1
    num_points = matches_im0.shape[0]
    ones = torch.ones((num_points, 1), device=device, dtype=torch.float32)

    # 形成齐次坐标 [x, y, 1]
    points1_h = torch.cat([matches_im0, ones], dim=1)  # [N, 3]
    points2_h = torch.cat([matches_im1, ones], dim=1)  # [N, 3]

    # 线性三角测量 (DLT算法)
    points3D = []
    for i in range(num_points):
        # 获取当前匹配点对
        p1 = points1_h[i]  # [3]
        p2 = points2_h[i]  # [3]

        # 构建DLT矩阵
        A = torch.zeros((4, 4), device=device, dtype=torch.float32)
        A[0] = p1[0] * P1[2] - P1[0]
        A[1] = p1[1] * P1[2] - P1[1]
        A[2] = p2[0] * P2[2] - P2[0]
        A[3] = p2[1] * P2[2] - P2[1]

        # 使用SVD求解最小二乘解
        _, _, Vt = torch.linalg.svd(A)
        X = Vt[-1]  # 最小奇异值对应的特征向量

        # 归一化齐次坐标
        X = X / X[3]
        points3D.append(X[:3])

    # 将结果堆叠为点云
    points3D = torch.stack(points3D, dim=0)  # [N, 3]
    if images is not None and view_idx1 < len(images) and 'img' in images[view_idx1]:
        # 获取图像
        img = images[view_idx1]['img']
        if torch.is_tensor(img):
            if len(img.shape) == 4:  # [B, C, H, W]
                img = img[0]  # 移除批次维度
            img = img.cpu().numpy()

        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # 确保图像值在0-255范围内
        if img.max() <= 1.0:
            img = img * 255

        # 取整数值
        img = img.astype(np.uint8)

        # 从第一张图像中获取颜色
        H, W = img.shape[:2]
        colors = []
        for point_idx in range(num_points):
            x, y = int(matches_im0[point_idx, 0].item()), int(matches_im0[point_idx, 1].item())

            # 确保索引在有效范围内
            x = max(0, min(x, W-1))
            y = max(0, min(y, H-1))

            # 获取颜色
            color = img[y, x]
            colors.append(color)

        colors = np.array(colors)

    # 如果需要保存，先转回CPU和NumPy
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # 保存带颜色的PLY
        points3D_np = points3D.cpu().numpy()
        colored_ply_path = os.path.join(save_dir, f'colored_points_{view_idx1}_to_{view_idx2}.ply')
        save_colored_point_cloud_ply(points3D_np, colors, colored_ply_path)

        # 保存普通PLY
        save_path = os.path.join(save_dir, f'triangulated_points_{view_idx1}_to_{view_idx2}.ply')
        save_point_cloud_ply(points3D_np, save_path)

        if images is not None and view_idx1 < len(images) and 'img' in images[view_idx1]:
            proj_img_path = os.path.join(save_dir, f'projected_points_{view_idx1}.png')
            project_points_to_image(
                points3D_np,
                extrinsics_w2c,
                intrinsics,
                view_idx1,
                image=images[view_idx1]['img'],
                save_path=proj_img_path
            )
        # viz_path = os.path.join(save_dir, f'triangulated_viz_{view_idx1}_to_{view_idx2}.png')
        # visualize_point_cloud(points3D_np, viz_path, colors=colors)

        print(f">> 已保存三角测量重建的点云: {save_path}")
        print(f">> 已保存彩色点云: {colored_ply_path}")
    # # 如果需要保存，先转回CPU和NumPy
    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f'triangulated_points_{view_idx1}_to_{view_idx2}.ply')
    #     save_point_cloud_ply(points3D.cpu().numpy(), save_path)
    #     print(f">> 已保存三角测量重建的点云: {save_path}")
    #     viz_path = os.path.join(save_dir, f'triangulated_viz_{view_idx1}_to_{view_idx2}.png')
    #     visualize_point_cloud(points3D, viz_path)

    return points3D

def project_points_to_image(points3D, extrinsics_w2c, intrinsics, view_idx, image=None, save_path=None, colors=None):
    """
    将3D点云投影回图像平面，并可视化结果

    Args:
        points3D: 3D点云坐标数组 [N,3]
        extrinsics_w2c: 世界坐标系到相机坐标系的变换矩阵 [4,4]
        intrinsics: 相机内参矩阵 [3,3]
        view_idx: 要投影的视图索引
        image: 原始图像(可选)，如果提供则在原图上绘制投影点
        save_path: 保存结果图像的路径
        colors: 点云颜色数组 [N,3]，如果提供则使用这些颜色

    Returns:
        投影后的2D点坐标 [N,2]
        投影点的有效掩码 [N]
        投影点的深度 [N]
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # 确保点云是numpy数组
    if torch.is_tensor(points3D):
        points3D = points3D.cpu().numpy()

    # 获取相机参数
    K = intrinsics[view_idx]
    R = extrinsics_w2c[view_idx][:3, :3]
    t = extrinsics_w2c[view_idx][:3, 3]

    # 转换为齐次坐标
    n_points = points3D.shape[0]
    points_h = np.hstack((points3D, np.ones((n_points, 1))))

    # 构建相机变换矩阵
    RT = np.zeros((3, 4))
    RT[:3, :3] = R
    RT[:3, 3] = t

    # 投影: P = K[R|t]
    P = K @ RT

    # 应用投影矩阵
    points_proj_h = P @ points_h.T  # [3, N]

    # 计算点的深度
    depths = points_proj_h[2]

    # 齐次坐标归一化
    points_proj = points_proj_h[:2] / points_proj_h[2]  # [2, N]
    points_proj = points_proj.T  # [N, 2]

    # 如果提供了图像尺寸，创建有效点的掩码
    if image is not None:
        if torch.is_tensor(image):
            if len(image.shape) == 4:  # [B, C, H, W]
                height, width = image.shape[2], image.shape[3]
            else:  # [C, H, W]
                height, width = image.shape[1], image.shape[2]
        else:
            height, width = image.shape[:2]
    else:
        height, width = 512, 512  # 默认图像尺寸

    # 创建有效点掩码
    valid_mask = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < width) & \
                (points_proj[:, 1] >= 0) & (points_proj[:, 1] < height) & \
                (depths > 0)

    # 如果提供了保存路径，可视化投影结果
    if save_path is not None:
        # 准备图像
        if image is None:
            # 创建空白图像
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
        else:
            # 使用提供的图像
            if torch.is_tensor(image):
                img = image.cpu().numpy()

                # 如果是PyTorch格式 [C,H,W]，转换为 [H,W,C]
                if len(img.shape) == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                # 如果有额外的批次维度，则去除
                elif len(img.shape) == 4:
                    img = img[0].transpose(1, 2, 0)

                # 确保图像值在0-255范围内
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

        # 创建图像
        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        valid_points = points_proj[valid_mask]

        # 如果提供了颜色信息，使用这些颜色
        if colors is not None and valid_mask.shape[0] == colors.shape[0]:
            valid_colors = colors[valid_mask]
            # 确保颜色在0-1范围内
            if valid_colors.dtype == np.uint8:
                valid_colors = valid_colors.astype(float) / 255.0

            # 绘制有颜色的投影点
            plt.scatter(valid_points[:, 0], valid_points[:, 1], c=valid_colors, s=3, alpha=0.7)

        # 添加标题和轴标签
        plt.title(f'3D点云投影结果 (视图 {view_idx})')

        # 关闭坐标轴
        plt.axis('off')

        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f">> 已保存投影可视化结果: {save_path}")

    # 返回投影点、有效掩码和深度
    return points_proj, valid_mask, depths