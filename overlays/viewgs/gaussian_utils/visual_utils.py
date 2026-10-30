import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import hsv_to_rgb
import json
import torch
import random

def save_point_cloud_ply(points, filename):
    """保存点云为PLY格式"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"已保存点云到: {filename}")
    return 1

def save_colored_point_cloud_ply(points, colors, filename):
    """
    保存带颜色的点云为PLY格式

    Args:
        points: Nx3 点坐标数组
        colors: Nx3 RGB颜色数组，值范围0-255
        filename: 保存文件路径
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i, point in enumerate(points):
            color = colors[i]
            # 确保颜色值在0-255范围内
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")

    print(f"已保存彩色点云到: {filename}")
    return 1

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
def visualize_point_cloud(points, image_path=None, colors=None, max_points=50000):
    """
    可视化点云并保存为图像

    Args:
        points: 点云坐标数组
        image_path: 保存图像的路径
        colors: 点云颜色数组
        max_points: 最大可视化点数量
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 如果点数过多，进行下采样
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    # 创建3D图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    if colors is not None:
        # 确保颜色值在0-1范围内
        if colors.max() > 1:
            colors = colors / 255.0
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, alpha=0.5)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, alpha=0.5)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置轴范围，使点云居中
    mean = np.mean(points, axis=0)
    max_range = np.max(np.ptp(points, axis=0))
    ax.set_xlim(mean[0] - max_range/2, mean[0] + max_range/2)
    ax.set_ylim(mean[1] - max_range/2, mean[1] + max_range/2)
    ax.set_zlim(mean[2] - max_range/2, mean[2] + max_range/2)

    # 保存图像
    if image_path:
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"已保存点云可视化图像: {image_path}")
        plt.close()
    else:
        plt.show()

    return 1

def triangulate_matches_gpu(matches_im0, matches_im1, extrinsics_w2c, intrinsics, view_idx1, view_idx2, save_dir=None, images=None):
    """
    使用GPU加速的三角测量实现，重建3D点云

    Args:
        matches_im0, matches_im1: 匹配点坐标 (PyTorch张量)
        extrinsics_w2c, intrinsics: 相机参数
        view_idx1, view_idx2: 视图索引
        save_dir: 保存点云的目录
    """
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
        # 如果需要保存，先转回CPU和NumPy
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'triangulated_points_{view_idx1}_to_{view_idx2}.ply')
            save_point_cloud_ply(points3D.cpu().numpy(), save_path)
            print(f">> 已保存三角测量重建的点云: {save_path}")
            viz_path = os.path.join(save_dir, f'triangulated_viz_{view_idx1}_to_{view_idx2}.png')
            visualize_point_cloud(points3D, viz_path)

    return points3D
# 添加辅助函数，将NumPy数据类型转换为Python原生类型
def numpy_to_python(obj):
    """将NumPy数据类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    return obj
def visualize_feature_matches(self, camera_idx, scale=1.0, save_dir=None, max_display=50, line_thickness=1, save_keypoints=False, target_size=None, triangulate=False):
    """
    可视化指定相机与其匹配相机之间的特征匹配点

    Args:
        camera_idx: 指定相机的索引
        scale: 分辨率比例
        save_dir: 保存可视化结果的目录，如果为None则显示而不保存
        max_display: 最多显示的匹配点数量，避免图像过于拥挤
        line_thickness: 匹配线的粗细
        save_keypoints: 是否保存匹配点数据为单独的文件
        target_size: 目标图像尺寸，形式为(宽, 高)。如果为None，则不调整图像大小

    Returns:
        如果save_dir不为None，返回保存的图像路径列表；否则返回None
    """


    # 设置特征提取时的图像尺寸，如果未指定target_size
    if target_size is None:
        target_size = (512, 288)  # 特征提取时的图像尺寸

    if not hasattr(self, 'camera_matches') or camera_idx not in self.camera_matches:
        print(f"相机 {camera_idx} 没有匹配关系")
        return None

    if camera_idx >= len(self.train_cameras[scale]):
        print(f"相机索引 {camera_idx} 超出范围，当前只有 {len(self.train_cameras[scale])} 个训练相机")
        return None

    # 获取当前相机
    current_camera = self.train_cameras[scale][camera_idx]

    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_keypoints:
            keypoints_dir = os.path.join(save_dir, 'keypoints')
            os.makedirs(keypoints_dir, exist_ok=True)
        if triangulate:
            triangulate_dir = os.path.join(save_dir, 'triangulated')
            os.makedirs(triangulate_dir, exist_ok=True)

    extrinsics_w2c = []
    intrinsics = []
    images = []

    if triangulate:
        # 为所有相机收集参数
        for cam in self.train_cameras[scale]:
            # 外参矩阵
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = cam.R
            extrinsic[:3, 3] = cam.T
            extrinsics_w2c.append(extrinsic)

            # 计算内参矩阵
            height, width = cam.image_height, cam.image_width
            fx = width / (2 * np.tan(cam.FoVx / 2))
            fy = height / (2 * np.tan(cam.FoVy / 2))
            cx = width / 2
            cy = height / 2

            intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            intrinsics.append(intrinsic)

            # 收集图像以便在三角测量后进行投影可视化
            if hasattr(cam, 'original_image'):
                img_dict = {'img': cam.original_image}
                images.append(img_dict)
            else:
                images.append(None)

    saved_paths = []
    # 为每个匹配的相机创建可视化
    for matched_idx, match_data in self.camera_matches[camera_idx].items():
        if matched_idx >= len(self.train_cameras[scale]):
            continue

        matched_camera = self.train_cameras[scale][matched_idx]

        # 获取匹配点
        matches_im0 = match_data['points']
        matches_im1 = match_data['matching_points']

        if triangulate:
            print(f">> 对相机 {camera_idx} 和相机 {matched_idx} 之间的匹配点进行三角测量...")

            # 确保点是PyTorch张量
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if not torch.is_tensor(matches_im0):
                matches_im0_tensor = torch.tensor(matches_im0, device=device, dtype=torch.float32)
            else:
                matches_im0_tensor = matches_im0.to(device)

            if not torch.is_tensor(matches_im1):
                matches_im1_tensor = torch.tensor(matches_im1, device=device, dtype=torch.float32)
            else:
                matches_im1_tensor = matches_im1.to(device)

            # 使用收集的内参和外参进行三角测量
            points3D = triangulate_matches_gpu(
                matches_im0_tensor,
                matches_im1_tensor,
                extrinsics_w2c,
                intrinsics,
                camera_idx,
                matched_idx,
                save_dir=os.path.join(triangulate_dir, f'camera_{camera_idx}_to_{matched_idx}'),
                images=images
            )
            print(f">> 成功三角测量重建 {len(points3D)} 个3D点")


        # 限制显示点的数量
        num_matches = len(matches_im0)
        if num_matches > max_display:
            # 使用官方示例的均匀采样方法
            match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, max_display)).astype(int)
            viz_matches_im0 = matches_im0[match_idx_to_viz]
            viz_matches_im1 = matches_im1[match_idx_to_viz]
        else:
            viz_matches_im0 = matches_im0
            viz_matches_im1 = matches_im1

        num_viz = len(viz_matches_im0)

        # 获取当前相机的图像
        img0 = None
        if hasattr(current_camera, 'image_path') and os.path.exists(current_camera.image_path):
            img0 = cv2.imread(current_camera.image_path)
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        elif hasattr(current_camera, 'original_image') and current_camera.original_image is not None:
            img0 = current_camera.original_image
            # 如果是PyTorch张量，转换为NumPy数组
            if torch.is_tensor(img0):
                img0 = img0.detach().cpu().numpy()
                # 如果是通道优先，转换为通道最后
                if img0.shape[0] == 3:
                    img0 = np.transpose(img0, (1, 2, 0))
        else:
            print(f"无法加载相机 {camera_idx} 的图像")
            continue
        if img0.dtype == np.float32 or img0.dtype == np.float64:
            if img0.max() <= 1.0:  # 如果是0-1范围
                img0 = (img0 * 255).astype(np.uint8)
            else:  # 如果已经是0-255范围
                img0 = img0.astype(np.uint8)
        # 获取匹配相机的图像
        img1 = None
        if hasattr(matched_camera, 'image_path') and os.path.exists(matched_camera.image_path):
            img1 = cv2.imread(matched_camera.image_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        elif hasattr(matched_camera, 'original_image') and matched_camera.original_image is not None:
            img1 = matched_camera.original_image
            # 如果是PyTorch张量，转换为NumPy数组
            if torch.is_tensor(img1):
                img1 = img1.detach().cpu().numpy()
                # 如果是通道优先，转换为通道最后
                if img1.shape[0] == 3:
                    img1 = np.transpose(img1, (1, 2, 0))
        else:
            print(f"无法加载相机 {matched_idx} 的图像")
            continue

        if img1.dtype == np.float32 or img1.dtype == np.float64:
            if img1.max() <= 1.0:  # 如果是0-1范围
                img1 = (img1 * 255).astype(np.uint8)
            else:  # 如果已经是0-255范围
                img1 = img1.astype(np.uint8)
        # 保存原始图像，用于后续单独保存
        original_img0 = img0.copy()
        original_img1 = img1.copy()

        # 记录原始图像尺寸
        original_size0 = img0.shape[:2][::-1]  # (宽, 高)
        original_size1 = img1.shape[:2][::-1]  # (宽, 高)

        # 调整图像尺寸到target_size
        if target_size:
            img0 = cv2.resize(img0, target_size)
            img1 = cv2.resize(img1, target_size)

        # 按照官方代码的方式拼接图像
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]

        # 使用更大的高度填充两张图像
        img0_padded = np.pad(img0, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1_padded = np.pad(img1, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)

        # 水平拼接图像
        combined_img = np.concatenate((img0_padded, img1_padded), axis=1)

        # 使用matplotlib的方式来处理可视化
        plt.figure(figsize=(15, 10))
        plt.imshow(combined_img)

        # 使用官方的彩色映射
        cmap = plt.get_cmap('jet')

        # 记录匹配信息
        match_info = {
            "matched_camera_idx": matched_idx,
            "matched_image_name": getattr(matched_camera, "image_name", f"camera_{matched_idx}"),
            "original_size0": original_size0,
            "original_size1": original_size1,
            "target_size": target_size,
            "keypoints": []
        }

        # 绘制匹配点和连线
        for i in range(num_viz):
            x0, y0 = viz_matches_im0[i]
            x1, y1 = viz_matches_im1[i]

            # 绘制连线和点
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (num_viz - 1)), scalex=False, scaley=False)

            # 记录点信息
            match_info["keypoints"].append({
                "point1_original": viz_matches_im0[i].tolist() if isinstance(viz_matches_im0[i], np.ndarray) else viz_matches_im0[i],
                "point2_original": viz_matches_im1[i].tolist() if isinstance(viz_matches_im1[i], np.ndarray) else viz_matches_im1[i],
                "point1_display": [x0, y0],
                "point2_display": [x1, y1]
            })

        # 添加相机索引标签
        plt.text(10, 30, f"Camera {camera_idx}", color='white', fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.5))
        plt.text(W0 + 10, 30, f"Camera {matched_idx}", color='white', fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.5))

        plt.title(f"Feature Matches between Camera {camera_idx} and Camera {matched_idx}")
        plt.axis('off')

        # 保存或显示
        if save_dir:
            save_path = os.path.join(save_dir, f"match_{camera_idx}_{matched_idx}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            saved_paths.append(save_path)
            print(f"已保存匹配可视化: {save_path}")

            # 保存单独的图像和关键点
            if save_keypoints:
                # 保存原始尺寸图像
                cam1_path = os.path.join(keypoints_dir, f"camera_{camera_idx}_original.png")
                if not os.path.exists(cam1_path):
                    cv2.imwrite(cam1_path, cv2.cvtColor(original_img0, cv2.COLOR_RGB2BGR))

                cam2_path = os.path.join(keypoints_dir, f"camera_{matched_idx}_original.png")
                cv2.imwrite(cam2_path, cv2.cvtColor(original_img1, cv2.COLOR_RGB2BGR))

                # 保存缩放后的图像
                if target_size:
                    cam1_resized_path = os.path.join(keypoints_dir, f"camera_{camera_idx}_resized.png")
                    if not os.path.exists(cam1_resized_path):
                        cv2.imwrite(cam1_resized_path, cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))

                    cam2_resized_path = os.path.join(keypoints_dir, f"camera_{matched_idx}_resized.png")
                    cv2.imwrite(cam2_resized_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                else:
                    cam1_resized_path = cam1_path
                    cam2_resized_path = cam2_path

                # 保存单独带有关键点标注的图像
                plt.figure(figsize=(8, 6))
                plt.imshow(img0)
                plt.title(f"Camera {camera_idx} with Keypoints")
                plt.axis('off')
                for i in range(num_viz):
                    x0, y0 = match_info["keypoints"][i]["point1_display"]
                    plt.plot(x0, y0, '+', color=cmap(i / (num_viz - 1)), markersize=8)
                    plt.text(x0+5, y0-5, str(i), fontsize=8, color=cmap(i / (num_viz - 1)))
                plt.savefig(os.path.join(keypoints_dir, f"keypoints_vis_{camera_idx}.png"), bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.imshow(img1)
                plt.title(f"Camera {matched_idx} with Keypoints")
                plt.axis('off')
                for i in range(num_viz):
                    x1, y1 = match_info["keypoints"][i]["point2_display"]
                    plt.plot(x1, y1, '+', color=cmap(i / (num_viz - 1)), markersize=8)
                    plt.text(x1+5, y1-5, str(i), fontsize=8, color=cmap(i / (num_viz - 1)))
                plt.savefig(os.path.join(keypoints_dir, f"keypoints_vis_{matched_idx}.png"), bbox_inches='tight')
                plt.close()

                # 保存匹配点数据
                # 修改 JSON 写入部分
                keypoints_path = os.path.join(keypoints_dir, f"keypoints_{camera_idx}_{matched_idx}.json")
                with open(keypoints_path, 'w') as f:
                    json_data = {
                        "camera1_idx": int(camera_idx),  # 确保是Python原生类型
                        "camera2_idx": int(matched_idx),
                        "original_image1_path": cam1_path,
                        "original_image2_path": cam2_path,
                        "resized_image1_path": cam1_resized_path,
                        "resized_image2_path": cam2_resized_path,
                        "original_size0": [int(s) for s in original_size0],
                        "original_size1": [int(s) for s in original_size1],
                        "target_size": [int(s) for s in target_size] if target_size else None,
                        "matches_im0": numpy_to_python(viz_matches_im0),
                        "matches_im1": numpy_to_python(viz_matches_im1),
                        "displayed_matches_im0": numpy_to_python([match_info["keypoints"][i]["point1_display"] for i in range(num_viz)]),
                        "displayed_matches_im1": numpy_to_python([match_info["keypoints"][i]["point2_display"] for i in range(num_viz)])
                    }
                    json.dump(json_data, f, indent=2)
        else:
            plt.show()
            plt.close()

    # 保存所有匹配信息的摘要
    if save_dir and save_keypoints:
        all_matches_info = {
            "camera_idx": camera_idx,
            "image_name": getattr(current_camera, "image_name", f"camera_{camera_idx}"),
            "target_size": target_size,
            "matches_count": len(self.camera_matches[camera_idx]),
            "matching_cameras": list(self.camera_matches[camera_idx].keys())
        }

        summary_path = os.path.join(save_dir, f"matches_summary_{camera_idx}.json")
        with open(summary_path, 'w') as f:
            json.dump(all_matches_info, f, indent=2)
        print(f"已保存匹配摘要信息: {summary_path}")

    return saved_paths if save_dir else None

def visualize_all_matches(self, scale=1.0, save_dir="./match_visualization", max_display=50, line_thickness=1, save_keypoints=False):
    """
    可视化所有相机的特征匹配

    Args:
        scale: 分辨率比例
        save_dir: 保存可视化结果的目录
        max_display: 每对相机最多显示的匹配点数量
        line_thickness: 连接线的粗细
        save_keypoints: 是否保存匹配点数据为单独的文件

    Returns:
        保存的图像路径列表
    """
    if not hasattr(self, 'camera_matches'):
        print("没有加载特征匹配数据")
        return []

    os.makedirs(save_dir, exist_ok=True)

    all_saved_paths = []

    # 创建摘要文件
    summary = {
        "total_cameras": len(self.camera_matches),
        "cameras": []
    }

    for camera_idx in sorted(self.camera_matches.keys()):
        print(f"处理相机 {camera_idx}...")
        camera_save_dir = os.path.join(save_dir, f"camera_{camera_idx}")

        saved_paths = self.visualize_feature_matches(
            camera_idx,
            scale=scale,
            save_dir=camera_save_dir,
            max_display=max_display,
            line_thickness=line_thickness,
            save_keypoints=save_keypoints
        )

        if saved_paths:
            all_saved_paths.extend(saved_paths)

            # 添加到摘要
            camera_info = {
                "camera_idx": camera_idx,
                "matches_count": len(self.camera_matches[camera_idx]),
                "matching_cameras": list(self.camera_matches[camera_idx].keys())
            }
            summary["cameras"].append(camera_info)

    # 保存摘要
    summary_path = os.path.join(save_dir, "visualization_summary.json")
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"完成所有匹配可视化，共处理 {len(summary['cameras'])} 个相机")
    print(f"可视化结果已保存至: {save_dir}")
    print(f"摘要信息已保存至: {summary_path}")

    return all_saved_paths

def export_keypoints_for_camera(self, camera_idx, scale=1.0, format='colmap', output_dir=None):
    """
    为指定相机导出特征点以及其与其他相机的匹配关系

    Args:
        camera_idx: 相机索引
        scale: 分辨率比例
        format: 导出格式，支持'colmap'或'json'
        output_dir: 输出目录，如果为None则使用当前目录

    Returns:
        导出文件的路径
    """


    if not hasattr(self, 'camera_matches') or camera_idx not in self.camera_matches:
        print(f"相机 {camera_idx} 没有匹配关系")
        return None

    if output_dir is None:
        output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    if format == 'json':
        # JSON格式
        output_data = {
            "camera_idx": camera_idx,
            "matches": {}
        }

        for matched_idx, match_data in self.camera_matches[camera_idx].items():
            output_data["matches"][str(matched_idx)] = {
                "points": match_data["points"].tolist(),
                "matching_points": match_data["matching_points"].tolist()
            }

        output_path = os.path.join(output_dir, f"camera_{camera_idx}_keypoints.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        return output_path

    elif format == 'colmap':
        # COLMAP格式
        # 收集所有特征点
        all_keypoints = {}
        for matched_idx, match_data in self.camera_matches[camera_idx].items():
            points = match_data["points"]
            if camera_idx not in all_keypoints:
                all_keypoints[camera_idx] = points

            matching_points = match_data["matching_points"]
            if matched_idx not in all_keypoints:
                all_keypoints[matched_idx] = matching_points

        # 写入keypoints文件
        keypoints_path = os.path.join(output_dir, "keypoints.txt")
        with open(keypoints_path, 'w') as f:
            for cam_id, points in all_keypoints.items():
                f.write(f"{cam_id} {len(points)} 6\n")
                for point in points:
                    # COLMAP格式: x, y, scale, orientation, [descriptor]
                    # 我们没有scale和orientation等信息，使用默认值
                    x, y = point
                    f.write(f"{x} {y} 1.0 0.0\n")

        # 写入matches文件
        matches_path = os.path.join(output_dir, "matches.txt")
        with open(matches_path, 'w') as f:
            for matched_idx, match_data in self.camera_matches[camera_idx].items():
                points1 = match_data["points"]
                points2 = match_data["matching_points"]
                f.write(f"{camera_idx} {matched_idx} {len(points1)}\n")
                for i in range(len(points1)):
                    # COLMAP格式: index1 index2
                    # 这里我们使用点的索引
                    f.write(f"{i} {i}\n")

        return keypoints_path

    else:
        print(f"不支持的导出格式: {format}")
        return None