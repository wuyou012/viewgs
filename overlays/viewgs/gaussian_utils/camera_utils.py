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

from scene.cameras import Camera,MidCamera
import torch
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
from gaussian_utils.general_utils import PILtoTorch
from gaussian_utils.graphics_utils import fov2focal
from gaussian_utils.depth_utils import estimate_depth
from natsort import natsorted

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    depth = estimate_depth(gt_image.cuda()).cpu().numpy()

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,  image=gt_image, gt_alpha_mask=loaded_mask,
                  uid=id, data_device=args.data_device, image_name=cam_info.image_name,
                  depth_image=depth, mask=mask)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def loadMidCam(args, id, R, T, FoVx, FoVy, width, height, image_path=None, image_name=None, resolution_scale=1.0):

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(width/(resolution_scale * args.resolution)), round(height/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if width > 1600:
                global_down = width / 1600
            else:
                global_down = 1
        else:
            global_down = width / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(width / scale), int(height / scale))

    gt_image = None
    loaded_mask = None

    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]

            if resized_image_rgb.shape[1] == 4:
                loaded_mask = resized_image_rgb[3:4, ...]
        except Exception as e:
            print(f"Failed to load mid camera image {image_path}: {e}")
    else:
        print('image_path do not exist')

    depth = estimate_depth(gt_image.cuda()).cpu().numpy()

    return MidCamera(R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                    width=resolution[0], height=resolution[1],
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=image_name,
                    uid=id, data_device=args.data_device,
                    depth_image=depth, mask=None, bounds=None)

def midCameraList_from_params(mid_camera_params, diffusion_dir, resolution_scale, args, view_template):
    """
    从参数创建中间相机列表

    Args:
        mid_camera_params: 中间相机参数字典
        diffusion_dir: 扩散图像目录
        resolution_scale: 分辨率缩放
        args: 参数对象
        view_template: 模板视图，用于获取相机参数

    Returns:
        list: 中间相机列表
    """
    from tqdm import tqdm
    import os
    import copy

    mid_camera_list = []
    processed_params = copy.deepcopy(mid_camera_params)

    if diffusion_dir and os.path.exists(diffusion_dir):
        all_image_files = natsorted([f for f in os.listdir(diffusion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {len(all_image_files)} images in diffusion directory: {diffusion_dir}")
        indices_to_remove = [24*k for k in range((len(all_image_files) // 24) + 1)]

        image_files = [img for i, img in enumerate(all_image_files) if i not in indices_to_remove]
        removed_count = len(all_image_files) - len(image_files)
        print(f"Removed {removed_count} images at indices {indices_to_remove[:10]}{'...' if len(indices_to_remove) > 10 else ''}")
        print(f"Remaining {len(image_files)} images for processing")
        if len(processed_params["R"]) != len(image_files):
            indices_to_remove = [24*k for k in range((len(all_image_files) // 24) + 1)]

            valid_indices = [i for i in range(len(all_image_files)) if i not in indices_to_remove]
            valid_indices = valid_indices[:len(image_files)]  # 确保不超过图像文件数量

            # 更新相机参数
            processed_params["R"] = [R for i, R in enumerate(processed_params["R"]) if i in valid_indices]
            processed_params["T"] = [T for i, T in enumerate(processed_params["T"]) if i in valid_indices]
            print(f"After adjustment: {len(processed_params['R'])} camera parameters, {len(image_files)} image files")
    else:
        image_files = []
        print(f"No diffusion images found in {diffusion_dir}")

    for id, i in tqdm(enumerate(range(len(processed_params["R"]))), desc="Loading mid cameras"):
        R = np.array(processed_params["R"][i])
        T = np.array(processed_params["T"][i])

        image_path = None
        image_name = None

        if id < len(image_files):
            image_name = image_files[id]
            image_path = os.path.join(diffusion_dir, image_name)

        cam = loadMidCam(
            args=args,
            id=id,
            R=R,
            T=T,
            FoVx=view_template.FoVx,
            FoVy=view_template.FoVy,
            width=view_template.image_width,
            height=view_template.image_height,
            image_path=image_path,
            image_name=image_name,
            resolution_scale=resolution_scale
        )
        mid_camera_list.append(cam)
    mid_name_lsit = []
    mid_R_list = []
    for cam in mid_camera_list:
        mid_name_lsit.append(cam.image_name)
        mid_R_list.append(cam.R)

    print('load mid camera name list', mid_name_lsit)
    # print('load mid camera R list', mid_R_list)
    return mid_camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
