import os
import glob
from natsort import natsorted
import numpy as np
import torch
from plyfile import PlyData
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def save_camera_params(camera_traj, focals, save_dir, pc_image_files, root_name):

    # 保存 camera_traj.R, camera_traj.T, focals, H, W, 图片名和图片绝对路径到 txt 文件
    camera_params_file_path = os.path.join(save_dir, root_name)
    with open(camera_params_file_path, 'w') as f:
        for R, T in zip(camera_traj.R, camera_traj.T):
            f.write(f"R: {R.tolist()}\n")
            f.write(f"T: {T.tolist()}\n")
        f.write(f"focals: {focals.tolist()}\n")
        f.write(f"H: {camera_traj.image_size[0][0]}\n")
        f.write(f"W: {camera_traj.image_size[0][1]}\n")
        for img_path in pc_image_files:
            img_name = os.path.basename(img_path).split('.')[0]
            f.write(f"image_name: {img_name}\n")
            f.write(f"image_path: {os.path.abspath(img_path)}\n")

def save_camera_params_crop(camera_traj, focals, save_dir, pc_image_files, root_name):

    # 保存 camera_traj.R, camera_traj.T, focals, H, W, 图片名和图片绝对路径到 txt 文件
    camera_params_file_path = os.path.join(save_dir, root_name)
    with open(camera_params_file_path, 'w') as f:
        for R, T in zip(camera_traj.R, camera_traj.T):
            f.write(f"R: {R.tolist()}\n")
            f.write(f"T: {T.tolist()}\n")
        f.write(f"focals: {focals.tolist()}\n")
        f.write(f"H: {camera_traj.image_size[0][0]}\n")
        f.write(f"W: {camera_traj.image_size[0][1]}\n")
        for i in range(len(pc_image_files)):
            img_path = pc_image_files[i]
            img_name = os.path.basename(img_path).split('.')[0]
            f.write(f"image_name: {img_name}\n")
            f.write(f"image_path: {os.path.join(os.path.abspath(save_dir),'gt',f'{i+1}.png')}\n")

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def read_ply_file(ply_path):
    """
    从 .ply 文件中读取点云数据，并将其转换为指定形状。

    Args:
        ply_path (str): .ply 文件路径。
        image_shape (tuple): 图像的形状 (h, w, c)。

    Returns:
        list: 点云数据列表，每个元素为形状为 (h, w, c) 的 np.ndarray。
    """
    pcd = fetchPly(ply_path)
    fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float32).cuda()
    fused_color = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32).cuda()

    # fused_point_cloud = fused_point_cloud.reshape(h, w, c)
    # fused_color = fused_color.reshape(h, w, c)
    return fused_point_cloud, fused_color