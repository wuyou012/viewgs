import glob
import os
import sys
from PIL import Image
from typing import NamedTuple
import numpy as np
from natsort import natsorted

from utils.cam_utils import focal2fov, interpolate_translation, interpolate_rotation
from utils.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary

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
    principal_points: np.array
    focals: np.array

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list

def convert_cam_params(cam_infos):
    Rs = np.array([cam.R for cam in cam_infos])
    Ts = np.array([cam.T for cam in cam_infos])
    focals = np.array([cam.focals for cam in cam_infos])
    principal_points = np.array([cam.principal_points for cam in cam_infos])
    return [Rs, Ts, focals, principal_points]

def generate_pseudo_cameras(cam1, cam2, num_pseudo_cams=23, cam1_index=0):
    pseudo_cams = []
    for j in range(1, num_pseudo_cams + 1):
        t = round(j / (num_pseudo_cams + 1), 2)
        r = t / 6
        position = interpolate_translation(cam1.T, cam2.T, t)
        rotation = interpolate_rotation(cam1.R, cam2.R, t)  # 使用 t 进行均匀插值
        pseudo_cam = CameraInfo(
            uid=-1,  # 假相机没有唯一ID
            R=rotation,
            T=position,
            FovY=cam1.FovY,
            FovX=cam1.FovX,
            image=None,
            image_path="",
            image_name=f"pseudo_{cam1_index}_{j}",
            width=cam1.width,
            height=cam1.height,
            mask=None,
            principal_points=cam1.principal_points,
            focals=cam1.focals
        )
        pseudo_cams.append(pseudo_cam)
    return pseudo_cams

def combine_train_and_pseudo_cams(train_cam_infos, num_pseudo_cams=23):
    combined_cams = []
    for i in range(len(train_cam_infos) - 1):
        combined_cams.append(train_cam_infos[i])
        pseudo_cams = generate_pseudo_cameras(train_cam_infos[i], train_cam_infos[i + 1], num_pseudo_cams, cam1_index=i)
        combined_cams.extend(pseudo_cams)
    combined_cams.append(train_cam_infos[-1])  # 添加最后一个训练相机
    return combined_cams

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping):
    cam_infos = []
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

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL": #llff
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x  # use the same focal_length
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE": #mip?
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

        image_width, image_height = image.size
        principal_points = np.array([image_height / 2, image_width / 2], dtype=np.float32) # 256, 144
        scale = height / image_height # == image_8
        if scale != int(images_folder.split("/")[-1].split("_")[-1]):
            # print(f"Scale mismatch: {scale} != {images_folder.split('_')[-1]}")
            # scale = images_folder.split("_")[-1]
            assert False, "image scale mismatch"

        focals = np.array([focal_length_x/scale, focal_length_y/scale], dtype=np.float32)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, \
                              image_path=image_path, image_name=image_name, width=width, height=height, \
                              mask=None, principal_points=principal_points, focals=focals)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path, images, eval, n_views=0, n_trains = 3, llffhold=8):

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

    combined_cams = combine_train_and_pseudo_cams(train_cam_infos[:n_trains], num_pseudo_cams=23)
    train_cam_params = convert_cam_params(train_cam_infos)
    combined_cam_params = convert_cam_params(combined_cams)
    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos)
    return scene_info, train_cam_params

# numbers = [round(i * 0.1, 1) for i in range(1, 6)]
# t = random.choice(numbers)
# r = random.choice(numbers)/6.0
# near_viewpoint_cam = scene.getNearCamera(viewpoint_cam.image_name)
# random_position = interpolate_translation(viewpoint_cam.T, near_viewpoint_cam.T, t)
# random_rotation = interpolate_rotation(viewpoint_cam.R, near_viewpoint_cam.R, r)
# image_size = ((H, W),)  # (h, w)
# pseudo_cam = PseudoCamera(
#             R=random_rotation, T=random_position, FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
#             width=viewpoint_cam.image_width, height=viewpoint_cam.image_height
#         )


if __name__ == "__main__":
    def attach_debugger():
        import debugpy
        debugpy.listen(5678)
        print("Waiting for Debugger to Attach on Port 5678...")
        debugpy.wait_for_client()
        print("Attached!")
    attach_debugger()

    images = "images_8"
    path = "/home/vip2024/he/nerf_llff_data/flower"
    scene_info, combined_cam_params = readColmapSceneInfo(path, images=images, eval=True, n_views=3)
