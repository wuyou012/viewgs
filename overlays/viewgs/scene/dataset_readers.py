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
import glob
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
import imageio
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, \
    read_camera_params, split_camera_params_mid, read_camera_params_without_mid, verify_camera_params
from gaussian_utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gaussian_utils.general_utils import chamfer_dist
from gaussian_utils.pose_utils import generate_mid_cam
import numpy as np
import json
import cv2
import math
import torch
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussian_utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from natsort import natsorted

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array

class CameraInfo_endo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    intrinsics: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class DustCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    focal: float
class SceneInfo_noply(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def find_image_folders(base_path, suffix="imgs"):
    """
    查找以指定前缀 (如 'imgs') 开头的所有图像文件夹

    参数:
        base_path: 基础路径
        suffix: 前缀，默认为 "imgs"

    返回:
        找到的图像文件夹路径（如果找到多个，返回第一个）
    """
    # 首先检查是否存在精确匹配的文件夹
    exact_match = os.path.join(base_path, suffix)
    if os.path.isdir(exact_match):
        print(f"找到精确匹配的图像文件夹: {exact_match}")
        return exact_match

    # 查找所有以 suffix 开头的目录
    imgs_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith(suffix):
            imgs_dirs.append(item_path)

    if imgs_dirs:
        # 如果找到多个匹配的文件夹，打印所有找到的并使用第一个
        if len(imgs_dirs) > 1:
            print(f"警告: 找到多个图像文件夹: {imgs_dirs}")
            print(f"将使用第一个: {imgs_dirs[0]}")
        else:
            print(f"找到图像文件夹: {imgs_dirs[0]}")
        return imgs_dirs[0]

    # 如果没有找到匹配的文件夹，尝试使用基础路径
    print(f"警告: 在 {base_path} 中未找到以 '{suffix}' 开头的文件夹，将使用基础路径")
    return base_path

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
def readEndoscopeSceneInfo(path, pose, eval, llffhold=8, start_cam=0, end_cam=0):
    """
    读取内窥镜数据集的场景信息（简化版本）

    Args:
        path: 数据集根目录路径
        images: 图像子目录名称（如果为None则使用默认值）
        eval: 是否进行评估（如果为True则分割训练和测试集）
        llffhold: 每隔多少帧取一个测试图像
        start_cam: 起始相机索引
        end_cam: 结束相机索引（如果为None则使用所有相机）

    Returns:
        SceneInfo: 包含场景信息的命名元组
    """

    camera_params = {
        "image_height": 540,
        "image_width": 675,
        "fx": 401.1595,
        "fy": 400.9425,
        "cx": 334.143,
        "cy": 273.8665,
        "png_depth_scale": 2.55,
        "crop_edge": 0
    }

    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    width = camera_params["image_width"]
    height = camera_params["image_height"]

    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)

    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    print(f"相机内参:")
    print(f"  图像尺寸: {width}x{height}")
    print(f"  焦距: fx={fx}, fy={fy}")
    print(f"  主点: cx={cx}, cy={cy}")
    print(f"  视场角: FovX={FovX:.2f}°, FovY={FovY:.2f}°")

    # 读取外参文件
    pose_path = os.path.join(path, pose)
    print(f"从 {pose_path} 读取相机外参...")

    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"未找到相机外参文件: {pose_path}")

    # 读取相机位姿
    with open(pose_path, "r") as f:
        lines = f.readlines()

    # reading_dir = "images" if images is None else images
    # images_folder = os.path.join(path, reading_dir)

    # if not os.path.exists(images_folder):
        # raise FileNotFoundError(f"未找到图像文件夹: {images_folder}")

    # image_files = natsorted([f for f in os.listdir(images_folder) if f.endswith(".png") or f.endswith(".jpg")])

    if end_cam == 0:
        end_cam = len(lines)

    # 限制处理的图像范围
    # image_files = image_files[start_cam:end_cam]
    pose_lines = lines[start_cam:end_cam]

    print(f"从 {pose} 读取 {len(pose_lines)} 张图像...")

    # 直接创建相机信息列表
    cam_infos = []

    for line in pose_lines:
        # 解析位姿
        pose_values = list(map(float, line.strip().split(',')))
        pose_matrix = np.array(pose_values, dtype=np.float32).reshape(4, 4)

        # 从位姿矩阵中直接提取旋转矩阵和平移向量
        R = pose_matrix[:3, :3]
        T = pose_matrix[:3, 3]

        # 构建图像路径
        # image_path = os.path.join(images_folder, image_file)
        # image_name = os.path.basename(image_path).split(".")[0]

        # 加载图像
        # try:
        #     image = Image.open(image_path)
        # except Exception as e:
        #     print(f"无法加载图像 {image_path}: {e}")
        #     continue

        # 创建相机信息对象
        cam_info = CameraInfo_endo(
            uid=1,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,
            intrinsics=intrinsic_matrix,
            image_path=None,
            image_name=None,
            width=width,
            height=height
        )

        cam_infos.append(cam_info)

    print(f"共加载 {len(cam_infos)} 个相机")

    if eval:
        sample_rate = llffhold
        ids = np.arange(len(cam_infos))

        # 修改测试样本选择逻辑：每sample_rate张图取最后一张
        i_test = []
        for i in range(0, len(ids) - sample_rate + 1, sample_rate):  # 确保至少有一个完整组
            # 选择这个组的最后一个
            end_idx = i + sample_rate - 1
            if end_idx < len(ids):  # 双重检查，确保不会越界
                i_test.append(end_idx)

        i_test = np.array(i_test)
        i_train = np.array([i for i in ids if i not in i_test])

        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
        test_names = [cam_infos[i].image_name for i in i_test]
        print(f"测试集图像名称: {test_names}")
        print(f"训练集: {len(train_cam_infos)} 个相机, 测试集: {len(test_cam_infos)} 个相机")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        print(f"训练集: {len(train_cam_infos)} 个相机")

    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 不加载点云
    pcd = None
    ply_path = None

    # 构造并返回场景信息
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )

    return scene_info
def readColmapCameras2(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

        break

    def normalize(x):
        return x / np.linalg.norm(x)

    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m


    c2w = np.concatenate([R, T], dim=1)
    print(c2w.shape)
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    Num_views = 120
    rots = 2

    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, Num_views + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * 0.5), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))


    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping):
    cam_infos = []
    ids = []
    for idx, key in enumerate(natsorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id # same camera, same uid?
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # ids.append([uid,extr.camera_id])
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)

        # if image.width != width or image.height != height:
        #     print(f"\nResizing image {rgb_name} from {image.width}x{image.height} to {width}x{height} (内参定义尺寸)")
        #     image = image.resize((width, height), Image.LANCZOS)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=None, bounds=bounds)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readinstantCameras(cam_extrinsics, cam_intrinsics, images_folder, path):
    cam_infos = []
    ids = []
    for idx, key in enumerate(cam_extrinsics.keys()):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[1]
        height = intr.height
        width = intr.width

        uid = intr.id # same camera, same uid?
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # ids.append([uid,extr.camera_id])

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])

        image = Image.open(image_path)

        if image.width != width or image.height != height:
            print(f"\nResizing image {image_name} from {image.width}x{image.height} to {width}x{height}")
            image = image.resize((width, height), Image.LANCZOS)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=None, bounds=None)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readDustCameras(camera_params, colmap_path, scene):
    cam_infos = []
    rotations = camera_params["R"]
    translations = camera_params["T"]
    image_infos = camera_params["images"]
    height = camera_params["H"]
    width = camera_params["W"]
    focals = camera_params["focals"]

    n_cameras = len(rotations)
    assert len(translations) == n_cameras, "R and T have different lengths"
    assert len(image_infos) == n_cameras, "camera and image have different lengths"

    if isinstance(focals, list):
        if isinstance(focals[0], list):
            focal_lengths = [f[0] for f in focals]
            print(f"Using individual focal length for each camera: {len(focal_lengths)} values")
        else:
            focal_lengths = focals
            print(f"Using individual focal length for each camera: {len(focal_lengths)} values")
    else:
        print("Error: focal length not found")


    for idx, (R, T, image_info) in enumerate(zip(camera_params["R"], camera_params["T"], camera_params["images"])):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(camera_params["R"])))
        sys.stdout.flush()

        image_path = image_info["path"]
        image_name = image_info["name"]

        focal_length_x = focal_lengths[idx]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        image = Image.open(image_path)
        image = image.resize((width, height), Image.LANCZOS)

        cam_info = DustCameraInfo(uid=1, R=np.array(R), T=np.array(T), FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              mask=None,focal=focal_length_x)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def farthest_point_sampling(points, k):
    """
    Sample k points from input pointcloud data points using Farthest Point Sampling.

    Parameters:
    points: numpy.ndarray
        The input pointcloud data, a numpy array of shape (N, D) where N is the
        number of points and D is the dimensionality of each point.
    k: int
        The number of points to sample.

    Returns:
    sampled_points: numpy.ndarray
        The sampled pointcloud data, a numpy array of shape (k, D).
    """
    N, D = points.shape
    farthest_pts = np.zeros((k, D))
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(k):
        farthest_pts[i] = points[farthest]
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return farthest_pts

def farthest_point_sampling_indices(points, k):
    """
    使用最远点采样算法选择点的索引

    参数:
        points (np.ndarray): 形状为(N, 3)的点云数组
        k (int): 要采样的点数量

    返回:
        np.ndarray: 选定点的索引数组
    """
    N, D = points.shape
    # 确保k不大于N
    k = min(k, N)

    # 初始化
    selected_indices = np.zeros(k, dtype=np.int32)
    distances = np.full(N, np.inf)

    # 随机选择第一个点
    farthest = np.random.randint(0, N)

    for i in range(k):
        selected_indices[i] = farthest
        centroid = points[farthest]

        # 计算所有点到当前点的欧几里得距离的平方
        dist = np.sum((points - centroid) ** 2, axis=1)

        # 更新每个点到已选点集的最短距离
        distances = np.minimum(distances, dist)

        # 选择距离最大的点作为下一个点
        farthest = np.argmax(distances)

    return selected_indices

def fetchPly_o(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPly(path, subsample=True, subsample_ratio=20, method='random', seed=42):

    np.random.seed(seed)

    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    total_points = len(vertices)

    if not subsample:
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        print(f"use original point cloud: {total_points} points")
        return BasicPointCloud(points=positions, colors=colors, normals=normals)

    num_samples = max(100, total_points // subsample_ratio)
    num_samples = min(num_samples, total_points)
    print(f"pcd has {total_points} points, subsample to {num_samples} points (ratio 1/{total_points/num_samples:.1f})")

    if method == 'fps' and total_points > 100000:

        print(f"random, fps")
        # 预采样到10万点
        pre_sample_size = min(100000, total_points)
        pre_indices = np.random.choice(total_points, pre_sample_size, replace=False)

        pre_positions = np.vstack([vertices['x'][pre_indices],
                                  vertices['y'][pre_indices],
                                  vertices['z'][pre_indices]]).T

        fps_indices = farthest_point_sampling_indices(pre_positions, num_samples)
        selected_indices = pre_indices[fps_indices]
    else:
        selected_indices = np.random.choice(total_points, num_samples, replace=False)

    positions = np.vstack([vertices['x'][selected_indices],
                          vertices['y'][selected_indices],
                          vertices['z'][selected_indices]]).T
    colors = np.vstack([vertices['red'][selected_indices],
                       vertices['green'][selected_indices],
                       vertices['blue'][selected_indices]]).T / 255.0
    normals = np.vstack([vertices['nx'][selected_indices],
                        vertices['ny'][selected_indices],
                        vertices['nz'][selected_indices]]).T

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readDustInfo(args, path, images, eval, colmap_path, n_views=0, llffhold=8, ):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")

    camera_params_file_path = os.path.join(path, "camera_params_full.txt")
    c2w_path = os.path.join(path, "camera_trajectory.npy")

    if not os.path.exists(c2w_path):
        print(f"无法找到相机轨迹文件: {c2w_path}")

    c2ws_np = np.load(c2w_path)
    c2ws = torch.from_numpy(c2ws_np)[:n_views]

    if os.path.exists(camera_params_file_path):
        camera_params = read_camera_params(camera_params_file_path)
        real_camera_params, mid_camera_params =  split_camera_params_mid(camera_params)
        if isinstance(mid_camera_params, dict) and "R" in mid_camera_params and "T" in mid_camera_params:
            mid_camera_params["R"] = mid_camera_params["R"][:(n_views-1)*23]
            mid_camera_params["T"] = mid_camera_params["T"][:(n_views-1)*23]
        ply_path = os.path.join(path, "pcd.ply")
    else:
        raise Exception("Camera params file not found")

    print('sub sample point cloud using ratio', args.sub_ratio)
    pcd = fetchPly(ply_path,
                  subsample=True,
                  subsample_ratio=args.sub_ratio,
                  method='random')
    # reading_dir = "images" if images == None else images
    scene = path.split('/')[-1]

    cam_infos_unsorted = readDustCameras(real_camera_params, colmap_path, scene)

    if eval == True:
        train_cam_infos = cam_infos_unsorted[:n_views]
        test_cam_infos = cam_infos_unsorted[n_views:]
    else:
        train_cam_infos = cam_infos_unsorted
        test_cam_infos = cam_infos_unsorted

    if args.use_diff:
        mid_cam = generate_mid_cam(train_cam_infos)

    # print('train camera infos', train_cam_infos, '/n')
    # print('test camera infos', test_cam_infos, '/n')
    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, mid_camera_params, mid_cam

def readColmapSceneInfo(path, images, eval, n_views=0, llffhold=8):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    ply_path = os.path.join(path, "3_views/dense/fused.ply")
    # ply_path = os.path.join(path, "dense/0/points3D.ply")

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     xyz, rgb, _ = read_points3D_binary(bin_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     extended_pcd = fetchPly(ply_path)
    # except:
    #     extended_pcd = None
        # have to read the original pcd for initialization

    pcd = fetchPly(ply_path)


    reading_dir = "images" if images == None else images
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir),  path=path, rgb_mapping=rgb_mapping)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readinstantSceneInfo(args, path, images, eval, n_views=0, llffhold=8):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    ply_path = os.path.join(path, "0/points3D.ply")
    # ply_path = os.path.join(path, "dense/0/points3D.ply")


    cameras_extrinsic_file_train = os.path.join(path, "0/images.txt")
    cameras_intrinsic_file_train = os.path.join(path, "0/cameras.txt")
    cam_extrinsics_train = read_extrinsics_text(cameras_extrinsic_file_train)
    cam_intrinsics_train = read_intrinsics_text(cameras_intrinsic_file_train)

    cameras_extrinsic_file_test = os.path.join(path, "1/images.txt")
    cameras_intrinsic_file_test = os.path.join(path, "1/cameras.txt")
    cam_extrinsics_test = read_extrinsics_text(cameras_extrinsic_file_test)
    cam_intrinsics_test = read_intrinsics_text(cameras_intrinsic_file_test)

    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     xyz, rgb, _ = read_points3D_binary(bin_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     extended_pcd = fetchPly(ply_path)
    # except:
    #     extended_pcd = None
        # have to read the original pcd for initialization

    if hasattr(args, 'sub_ratio'):
        pcd = fetchPly(ply_path,
                    subsample=True,
                    subsample_ratio=args.sub_ratio,
                    method='random')
    else:
        pcd = fetchPly_o(ply_path)

    reading_dir = "images" if images == None else images
    # image_folder = find_image_folders(path, reading_dir)
    image_folder = os.path.join(os.path.dirname(path), reading_dir)
    cam_extrinsics_train = {cam_extrinsics_train[k].name: cam_extrinsics_train[k] for k in cam_extrinsics_train}
    cam_infos_unsorted_train = readinstantCameras(cam_extrinsics=cam_extrinsics_train, cam_intrinsics=cam_intrinsics_train,
                             images_folder=image_folder,  path=path)

    cam_extrinsics_test = {cam_extrinsics_test[k].name: cam_extrinsics_test[k] for k in cam_extrinsics_test}
    cam_infos_unsorted_test = readinstantCameras(cam_extrinsics=cam_extrinsics_test, cam_intrinsics=cam_intrinsics_test,
                             images_folder=image_folder,  path=path)

    mid_cam = None

    if eval:
        train_cam_infos = cam_infos_unsorted_train
        test_cam_infos = cam_infos_unsorted_test

    else:
        train_cam_infos = cam_infos_unsorted_train + cam_infos_unsorted_test
        test_cam_infos = []
    test_name = [cam_infos_unsorted_test[i].image_name for i in range(len(cam_infos_unsorted_test))]
    print(f"测试集图像名称: {test_name}")
    if args.use_diff:
        mid_cam = generate_mid_cam(cam_infos_unsorted_train)


    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, mid_cam

def readinstantSceneInfo_full(path, images, eval, n_views=0, llffhold=8):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    ply_path = os.path.join(path, "0/points3D.ply")
    # ply_path = os.path.join(path, "dense/0/points3D.ply")


    cameras_extrinsic_file = os.path.join(path, "0/images.txt")
    cameras_intrinsic_file = os.path.join(path, "0/cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    pcd = fetchPly_o(ply_path)

    reading_dir = "images" if images == None else images
    # image_folder = find_image_folders(path, reading_dir)
    image_folder = os.path.join(os.path.dirname(path), reading_dir)
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readinstantCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=image_folder,  path=path)


    if eval:
        train_cam_infos = cam_infos_unsorted[:n_views]
        test_cam_infos = cam_infos_unsorted[n_views:]
    else:
        train_cam_infos = cam_infos_unsorted
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo_noply(path, images, eval, n_views=0, llffhold=8):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    # ply_path = os.path.join(path, "3_views/dense/fused.ply")
    # ply_path = os.path.join(path, "dense/0/points3D.ply")

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     xyz, rgb, _ = read_points3D_binary(bin_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     extended_pcd = fetchPly(ply_path)
    # except:
    #     extended_pcd = None
        # have to read the original pcd for initialization

    # pcd = fetchPly(ply_path)


    reading_dir = "images" if images == None else images
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir),  path=path, rgb_mapping=rgb_mapping)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo_noply(train_cameras=train_cam_infos,
                                test_cameras=test_cam_infos,
                                nerf_normalization=nerf_normalization)
    return scene_info

def readColmapSceneInfo2(path, images, eval, n_views=0, llffhold=8):
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    ply_path = os.path.join(path, "dense/0/points3D.ply")

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     xyz, rgb, _ = read_points3D_binary(bin_path)
    #     storePly(ply_path, xyz, rgb)
    try:
        extended_pcd = fetchPly(ply_path)
    except:
        extended_pcd = None
        # have to read the original pcd for initialization

    # pcd = fetchPly(ply_path)


    reading_dir = "images" if images == None else images
    rgb_mapping = [f for f in natsorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir),  path=path, rgb_mapping=rgb_mapping)
    cam_infos = natsorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=extended_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            mask = norm_data[:, :, 3:4]
            if skip == 1:
                # depth_image = np.load('../SparseNeRF/depth_midas_temp_DPT_Hybrid/Blender/' +
                #                       image_path.split('/')[-4]+'/'+image_name+'_depth.npy')
                depth_image = None
            else:
                depth_image = None

            arr = cv2.resize(arr, (400, 400))
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            depth_image = None if depth_image is None else cv2.resize(depth_image, (400, 400))
            mask = None if mask is None else cv2.resize(mask, (400, 400))


            # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
            #                             image_name=image_name, width=image.size[0], height=image.size[1],
            #                             depth_image=depth_image, mask=mask))
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                            image_name=image_name, width=image.size[0], height=image.size[1], mask=mask, bounds = None))
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, n_views=0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    pseudo_cam_infos = train_cam_infos #train_cam_infos
    if n_views > 0:
        train_cam_infos = train_cam_infos[:n_views]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")

    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 30000
    #     print(f"Generating random point cloud ({num_pts})...")
    #
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                        #    pseudo_cameras=pseudo_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def compare_camera_params(cam_infos, source_cam_infos):
    differences = []
    source_cam_dict = {cam.image_name: cam for cam in source_cam_infos}

    for cam in cam_infos:
        if cam.image_name in source_cam_dict:
            source_cam = source_cam_dict[cam.image_name]
            R_diff = np.dot(cam.R, source_cam.R.T)
            T_diff = cam.T - source_cam.T
            differences.append({
                "image_name": cam.image_name,
                "R_diff": R_diff,
                "T_diff": T_diff
            })
        else:
            differences.append({
                "image_name": cam.image_name,
                "R_diff": None,
                "T_diff": None
            })

    return differences

sceneLoadTypeCallbacks = {
    "Dust": readDustInfo,
    "instant": readinstantSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Colmap_dense": readColmapSceneInfo2,
    "Blender" : readNerfSyntheticInfo
}
