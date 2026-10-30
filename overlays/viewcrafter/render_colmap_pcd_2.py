import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    look_at_view_transform
)
from torchvision.utils import save_image as save_torch_image
from utils.loader import read_ply_file
from utils.dataset_reader import readColmapSceneInfo
import os
from utils.colmap_utils import read_cameras_from_sparse
from natsort import natsorted
import glob

def load_colmap_data(colmap_path, eval, n_views=0, llffhold=8):
    # 读取点云数据


    images_folder = "images_8"
    colmap_sparse_dir = os.path.join(colmap_path, 'sparse/0')
    cameras_unsorted = read_cameras_from_sparse(colmap_sparse_dir)
    cam_infos = natsorted(cameras_unsorted.copy(), key=lambda x: x['img_name'])

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos) - 1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views

    return train_cam_infos

def setup_renderer(cameras, device):
    print(cameras)
    down = 8
    Rs, Ts = [], []

    if cameras is not None:
        for camera in cameras:
            """
            The camera coordinate sysmte in COLMAP is right-down-forward
            Pytorch3D is left-up-forward
            """
            c2w = torch.inverse(camera['w2c']) # to c2w
            R, T = c2w[:3, :3], c2w[:3, 3:]
            R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)  # from RDF to LUF for Rotation

            new_c2w = torch.cat([R, T], 1)
            w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0, 0, 0, 1]])), 0))
            R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]  # convert R to row-major matrix
            Rs.append(R)
            Ts.append(T)

        Rs = torch.stack(Rs)
        Ts = torch.stack(Ts)

        """ Downsample images size for faster rendering """
        H, W = cameras[0]['H'], cameras[0]['W']
        H, W = int(H / down), int(W / down)
        intrinsics = cameras[0]['intrinsics'] / down

        image_size = (H, W)  # (h, w)
        fcl_screen = ((intrinsics[0], intrinsics[0]),)  # fcl_ndc * min(image_size) / 2
        prp_screen = ((W / 2, H / 2),)  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2

        cameras = PerspectiveCameras(
            focal_length=fcl_screen,
            principal_point=prp_screen,
            in_ndc=False,
            image_size=[image_size] * len(Rs),
            R=Rs,
            T=Ts,
            device=device
        )

    print('Camera R T', Rs, Ts)

    # Define the settings for rasterization and shading.
    print(image_size)

    # raster_settings = PointsRasterizationSettings(
    #     image_size=image_size,
    #     radius=0.01,
    #     points_per_pixel=10
    # )

    # renderer = PointsRenderer(
    #     rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
    #     compositor=AlphaCompositor()
    # )

    return cameras

def save_image(tensor, path):
    save_torch_image(tensor, path)

def check_ply_path(ply_path):

    if isinstance(ply_path, list):
        if not ply_path:
            raise FileNotFoundError("No .ply files containing 'fwd' found in the specified directory.")
        ply_path = ply_path[0]
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"The file {ply_path} does not exist.")

    return ply_path

def genarate_col_traj(colmap_path, device):
    eval = True
    train_cam_infos = load_colmap_data(colmap_path, eval, n_views=3)
    cameras = setup_renderer(train_cam_infos, device)
    return cameras

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_root = "/home/vip2024/he/nerf_llff_data"
    scene = 'flower'
    pcd_folder = 'tmp/pcd'
    suffix = 'fwd'
    colmap_path = os.path.join(input_root, scene)
    ply_folder = os.path.join(colmap_path, pcd_folder)
    ply_files = glob.glob(os.path.join(ply_folder, '*.ply'))
    ply_path = [file for file in ply_files if suffix in os.path.basename(file)]
    ply_path = check_ply_path(ply_path)
    # ply_path = '/home/vip2024/he/nerf_llff_data/flower/tmp/pcd/1_0.ply'
    ply_path = os.path.join(colmap_path, 'sparse/0/points3D.ply')
    # ply_path = '/home/vip2024/he/work1/ViewCrafter/output/20250202_1725_flower/pcd.ply'
    image_size = (378, 504)
    save_dir = "/home/vip2024/he/work1/ViewCrafter/output_tmp"

    os.makedirs(save_dir, exist_ok=True)
    eval = True
    fused_point_cloud, fused_color = read_ply_file(ply_path)
    train_cam_infos = load_colmap_data(colmap_path, eval, n_views=3)
    renderer, cameras = setup_renderer(train_cam_infos, device)
    point_cloud = Pointclouds(points=[fused_point_cloud], features=[fused_color])
    images = renderer(point_cloud.extend(len(cameras.R)))
    for i, image in enumerate(images):
        save_image(image.permute(2, 0, 1), os.path.join(save_dir, f"rendered_image_{i}.png"))

if __name__ == "__main__":
    def attach_debugger():
        import debugpy
        debugpy.listen(5678)
        print("Waiting for Debugger to Attach on Port 5678...")
        debugpy.wait_for_client()
        print("Attached!")
    # attach_debugger()
    main()
