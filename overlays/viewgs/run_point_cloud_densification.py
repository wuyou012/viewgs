


"""
TODO:
This program is used to read the train depth images and train camera info and create a very dense point cloud
"""


import os

from gaussian_scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

from gaussian_scene.dataset_readers import readColmapCameras, getNerfppNorm, storePly, fetchPly, SceneInfo

import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2


def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for Debuger to Attach on Port 5678...")
    debugpy.wait_for_client()
    print("Attached!")

# attach_debugger()

def depth_to_point_cloud(depth_image, rgb_image, fovx, fovy, R, T, depth_scaling_factor):
    # Scale the depth image
    depth_image = depth_image / depth_scaling_factor

    # Get the shape of the depth image
    height, width = depth_image.shape

    # Compute the focal lengths from the field of view
    fx = width / (2 * math.tan(fovx / 2))
    fy = height / (2 * math.tan(fovy / 2))

    # Compute the optical center
    cx = width / 2
    cy = height / 2

    # Create the intrinsic matrix
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Create a 4x4 extrinsic matrix
    extrinsic = np.eye(4)

    # Set the rotation part
    extrinsic[:3, :3] = R

    # Set the translation part
    extrinsic[:3, 3] = T

    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Homogeneous coordinates
    homogeneous_pixel_coordinates = np.stack([x, y, np.ones_like(x)], axis=-1)

    # Reshape to (height*width, 3)
    homogeneous_pixel_coordinates = homogeneous_pixel_coordinates.reshape(-1, 3)

    # Invert the intrinsic matrix
    intrinsic_inv = np.linalg.inv(intrinsic)

    # Apply the inverse intrinsic matrix to get normalized pixel coordinates
    normalized_pixel_coordinates = homogeneous_pixel_coordinates @ intrinsic_inv.T

    # Reshape the depth image to (height*width, 1)
    depth = depth_image.reshape(-1, 1)

    # Multiply depth with normalized pixel coordinates to get camera coordinates
    camera_coordinates = depth.cpu() * normalized_pixel_coordinates

    # Add a column of ones for homogeneous coordinates
    homogeneous_camera_coordinates = np.concatenate([camera_coordinates, np.ones((height*width, 1))], axis=-1)

    # Invert the extrinsic matrix
    extrinsic_inv = np.linalg.inv(extrinsic)

    # Apply the inverse extrinsic matrix to get world coordinates
    world_coordinates = homogeneous_camera_coordinates @ extrinsic_inv.T

    # Discard the homogeneous coordinate (the last column)
    world_coordinates = world_coordinates[:, :3]

    # Reshape back to (height, width, 3)
    point_cloud = world_coordinates.reshape(height, width, 3)

    # Reshape the RGB image to (height*width, 3)
    # rgb = rgb_image.reshape(-1, 3)
    # point_cloud= point_cloud.reshape(-1, 3)
    print('++++++',point_cloud.shape)
    print('------',rgb_image.shape)
    # Concatenate the point cloud and the RGB values along the last dimension
    point_cloud_rgb = np.concatenate([point_cloud, rgb_image.cpu()], axis=-1)

    return point_cloud_rgb


def point_cloud_to_depth(point_cloud_rgb, output_size, camera, depth_scaling_factor):
    point_cloud = point_cloud_rgb.points
    height, width = camera.height, camera.width
    world_coordinates = point_cloud.reshape(-1, 3)
    # Add a column of ones to get homogeneous coordinates
    world_coordinates = torch.from_numpy(world_coordinates)
    homogeneous_world_coordinates = torch.cat([world_coordinates, torch.ones((world_coordinates.shape[0], 1), device=world_coordinates.device)], dim=-1)
    # Apply the extrinsic matrix to get camera coordinates
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = camera.R
    extrinsic[:3, 3] = camera.T
    extrinsic = torch.from_numpy(extrinsic)

    camera_coordinates = torch.mm(homogeneous_world_coordinates, extrinsic.float().T)
    camera_coordinates = camera_coordinates[:, :3]
    # Divide by the depth to get normalized pixel coordinates
    depth = torch.norm(camera_coordinates, dim=-1, keepdim=True)
    normalized_pixel_coordinates = camera_coordinates / depth
    # Add a column of ones to get homogeneous coordinates
    # homogeneous_pixel_coordinates = torch.cat([normalized_pixel_coordinates, torch.ones((normalized_pixel_coordinates.shape[0], 1), device=normalized_pixel_coordinates.device)], dim=-1)
    # Compute the intrinsic matrix from the camera parameters
    fx = width / (2 * math.tan(camera.FovX / 2))
    fy = height / (2 * math.tan(camera.FovY / 2))
    cx = width / 2
    cy = height / 2
    intrinsic = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device='cpu')
    # Apply the intrinsic matrix to get pixel coordinates
    pixel_coordinates_3d = torch.mm(normalized_pixel_coordinates, intrinsic.T)
    pixel_coordinates = torch.round(pixel_coordinates_3d[:, :2]).int()
    pixel_value = pixel_coordinates_3d[:,-1]
    # Reshape to (height, width, 2)
    # pixel_coordinates = pixel_coordinates.reshape(height, width, 2)
    # Create an empty depth image with default value 0
    depth_image = torch.zeros((height, width), device=depth.device)
    # For each pixel coordinate, set the corresponding depth value if it is smaller than the current value
    # for i in range(height):
    #     for j in range(width):
    #         if depth[i*width + j] < depth_image[int(pixel_coordinates[i, j, 1]), int(pixel_coordinates[i, j, 0])]:
    #             depth_image[int(pixel_coordinates[i, j, 1]), int(pixel_coordinates[i, j, 0])] = depth[i*width + j]
    # # Scale the depth image
    depth_image = depth_image * depth_scaling_factor
    # return depth_image
    for i, c in enumerate(pixel_coordinates):
        if c[1]>=width or c[1]<0:
            continue
        if c[0]>=height or c[0]<0:
            continue
        if depth_image[c[0]][c[1]] < pixel_value[i]:
            depth_image[c[0]][c[1]]  = pixel_value[i]

    depth_image = depth_image.unsqueeze(0).unsqueeze(0)
    depth_image = F.interpolate(depth_image, size=output_size, mode='bilinear', align_corners=False)
    return depth_image.squeeze(0).squeeze(0)







def save_point_cloud(filename, point_cloud_rgb, global_index):
    with open(filename, 'a') as f:  # 'a' for append mode
        for i in range(point_cloud_rgb.shape[0]):
            for j in range(point_cloud_rgb.shape[1]):
                x, y, z, r, g, b = point_cloud_rgb[i, j]
                f.write(f"{global_index[0]} {x} {y} {z} {r:.0f} {g:.0f} {b:.0f}\n")
                global_index[0] = global_index[0] + 1


def colmapPointsDensification(path, path_help, images, depth_scaling_factor = float(1000)):
    try:
        cameras_extrinsic_file = os.path.join(path, path_help, "0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, path_help, "0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, path_help, "0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, path_help, "0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    global_index = [1]
    point_cloud_txt_file = os.path.join(path, path_help, "0/points3D.txt")
    point_cloud_bin_file = point_cloud_txt_file.replace("txt", "bin")
    os.remove(point_cloud_txt_file)
    os.remove(point_cloud_bin_file)
    for idx, c in enumerate(cam_infos):
        rgb_fname = c.image_path
        depth_fname = rgb_fname.replace("rgb", "target_depth").replace('jpg', 'png')
        rgb_image = cv2.imread(rgb_fname)
        depth_image = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
        point_cloud_rgb = depth_to_point_cloud(depth_image, rgb_image, c.FovX, c.FovY, c.R, c.T, depth_scaling_factor)
        save_point_cloud(point_cloud_txt_file, point_cloud_rgb, global_index)
    print(f"""save point cloud to {os.path.join(path, path_help, "points3D.txt")}""")
    print(f"""total number of points in the point cloud is {global_index[0]-1}""")

def realtimePointsDensification(path, path_help, rgb=None, depths=None, pcd_file=None, depth_scaling_factor = float(1000), file_path = None):
    path_help = ''
    try:
        cameras_extrinsic_file = os.path.join(path, path_help, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, path_help, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, path_help, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, path_help, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    images = os.path.join(path, path_help, "images")
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    global_index = [1]
    # point_cloud_txt_file = os.path.join(path, "dense/0/points3D.txt")
    point_cloud_txt_file = pcd_file
    print('pcd_file',pcd_file)
    dense_dir = os.path.join(path, path_help, "dense")
    print('dense_dir',dense_dir)
    if not os.path.exists(dense_dir):
        os.makedirs(dense_dir)
        os.makedirs(os.path.join(dense_dir, "0"))
    # point_cloud_bin_file = point_cloud_txt_file.replace("txt", "bin")
    train_view_fname = []
    # train_view_dir = os.path.join(path, "train/rgb")
    # for filename in os.listdir(train_view_dir):
    #     if filename.endswith('.jpg'):
    #         train_view_fname.append(filename.split('.')[0])
    for filename in file_path:
        train_view_fname.append(filename.split('/')[-1].split('.')[0])
    train_view_fname = np.array(train_view_fname)
    for c in cam_infos:
        idx = np.where(train_view_fname == c.image_name)[0][0] if c.image_name in train_view_fname else False
        print(idx)
        if idx is not False:
            rgb_image = rgb[idx]
            depth_image = depths[idx]
            point_cloud_rgb = depth_to_point_cloud(depth_image[:, :, 0], rgb_image, c.FovX, c.FovY, c.R, c.T, depth_scaling_factor)
            save_point_cloud(point_cloud_txt_file, point_cloud_rgb, global_index)
    print(f"""save point cloud to {os.path.join(path, path_help, "points3D.txt")}""")
    print(f"""total number of points in the point cloud is {global_index[0]-1}""")


if __name__=='__main__':
    root = '/media/pinguin/DATA_DISK/NERF_DATA/depth_nerf_download/dataset/scene0708_00'
    # root = '/media/pinguin/DATA_DISK/NERF_DATA/depth_nerf_download/scene0710_00-20231116T062022Z-001/scene0710_00'

    path = os.path.join(root, 'colmap')

    data_split = 'sparse_train' #or 'sparse'

    images = os.path.join(root, 'train/rgb')



    realtimePointsDensification(path, data_split, images)
