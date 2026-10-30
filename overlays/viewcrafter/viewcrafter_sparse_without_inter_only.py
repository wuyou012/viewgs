import sys
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images,load_images_original, load_images_resized
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path
from torchvision.utils import save_image
from natsort import natsorted
from utils.loader import save_camera_params, read_ply_file, save_camera_params_crop
from render_colmap_pcd_2 import genarate_col_traj

def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for Debuger to Attach on Port 8765...")
    debugpy.wait_for_client()
    print("Attached!")
# attach_debugger()

class ViewCrafter:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        self.setup_diffusion()
        # initialize ref images, pcd
        if not gradio:
            if os.path.isfile(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):

                self.images, self.img_ori, self.img_raws = self.load_initial_dir_crop(image_dir=self.opts.image_dir)
                # self.images, self.img_ori, self.imginfo = self.load_initial_dir(image_dir=self.opts.image_dir)
                # self.images, self.img_ori = self.load_initial_dir_o(image_dir=self.opts.image_dir)
                self.run_dust3r_sparse(input_images=self.images, clean_pc = True)
            else:
                print(f"{self.opts.image_dir} doesn't exist")

    def run_dust3r(self, input_images, clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def run_dust3r_sparse(self, input_images, clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=False)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):

        imgs = to_numpy(imgs) #(B,288,512,3) B is number of input images
        pts3d = to_numpy(pts3d) #list of B x (288, 512, 3)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device) # h*w*2
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)

        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else:
            view_masks = None

        return images, view_masks

    def setup_pcd(self,pts3d,imgs,masks,device):

        imgs = to_numpy(imgs) #(B,288,512,3) B is number of input images
        pts3d = to_numpy(pts3d) #list of B x (288, 512, 3)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device) # h*w*2
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)

        return pts, col

    def render_gs_pcd(self,pts,col,masks,views,renderer,device,nbv=False):

        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else:
            view_masks = None

        return images, view_masks

    def run_render(self, pcd, imgs, masks, H, W, camera_traj, num_views, nbv=False):
        torch.cuda.empty_cache()
        batch_size = 30  # 每批处理的相机数量
        render_results_list = []

        # 获取所有相机参数，以便分批处理
        all_R = camera_traj.R
        all_T = camera_traj.T
        focal_length = camera_traj.focal_length
        principal_point = camera_traj.principal_point
        image_size = camera_traj.image_size

        for i in range(0, num_views, batch_size):
            # 清除上一批的缓存
            torch.cuda.empty_cache()

            end_idx = min(i + batch_size, num_views)
            current_batch_size = end_idx - i

            # 为当前批次创建新的相机对象
            batch_R = all_R[i:end_idx]
            batch_T = all_T[i:end_idx]
            batch_fl = focal_length[i:end_idx]
            batch_pp = principal_point[i:end_idx]
            batch_is = image_size[i:end_idx]
            # 使用相同的相机内参，但只处理当前批次的外参
            batch_camera = pytorch3d.renderer.PerspectiveCameras(
                focal_length=batch_fl,
                principal_point=batch_pp,
                in_ndc=False,
                image_size=batch_is,
                R=batch_R,
                T=batch_T,
                device=self.device
            )

            # 为当前批次设置渲染器
            render_setup = setup_renderer(batch_camera, image_size=(H, W))
            renderer = render_setup['renderer']

            print(f"渲染视角 {i} 到 {end_idx-1} (共 {num_views} 个)")

            # 渲染当前批次
            batch_results, _ = self.render_pcd(pcd, imgs, masks, current_batch_size, renderer, self.device, nbv=nbv)

            # 将结果移到CPU以节省GPU内存
            render_results_list.append(batch_results.cpu())

            # 强制执行垃圾回收
            import gc
            gc.collect()

        # 合并所有渲染结果
        render_results = torch.cat(render_results_list, dim=0).to(self.device)
        viewmask = None  # 如果需要viewmask，可以在分批处理中收集

        return render_results, viewmask

    def run_gs_render(self, pts, col, masks, camera_traj, nbv=False):
        [H,W] = camera_traj.image_size[0]
        H,W = int(H.item()), int(W.item())
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_gs_pcd(pts, col, masks, len(camera_traj.R),renderer,self.device,nbv=False)
        return render_results, viewmask

    def run_diffusion(self, renderings):

        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        if videos.shape[-2:] != (self.opts.height, self.opts.width):
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            videos = F.interpolate(videos, size=(self.opts.height, self.opts.width), mode='bilinear', align_corners=False)
            videos = videos.reshape(b, t, c, self.opts.height, self.opts.width).permute(0, 2, 1, 3, 4)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.)


    def nvs_sparse_view_interp(self):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach() #([B,2]) float32
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = self.scene.get_masks()
            depth = self.scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)]
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(self.scene.imgs) #(2,288,512,3) 0-1

        save_videoimages(torch.stack(self.img_raws), os.path.join(self.opts.save_dir, 'gt'))

        #---- original viewcrafter code ----
        camera_traj,num_views = generate_traj_interp_gs(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        save_camera_params_crop(camera_traj, focals, self.opts.save_dir, self.pc_image_files, 'camera_params_full.txt')
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # 如果可能，将不再需要的大型变量移到CPU或释放
        if hasattr(self, 'dust3r'):
            self.dust3r = self.dust3r.cpu()  # 将模型暂时移到CPU
        if isinstance(pcd, list) and len(pcd) > 0:
            total_points = sum(p.numel() // 3 for p in pcd[:self.opts.n_views])
            print(f"点云包含约 {total_points} 个点")
            if total_points > 1000000:  # 如果点数超过100万
                print("点云过大，尝试降低精度...")
        camera_traj,num_views = generate_traj_interp(c2ws[:self.opts.n_views], H, W, focals[:self.opts.n_views], principal_points[:self.opts.n_views], self.opts.video_length, self.device)
        render_results, viewmask = self.run_render(pcd[:self.opts.n_views], imgs[:self.opts.n_views], masks, H, W, camera_traj,num_views)

        for i in range(len(self.img_raws[:self.opts.n_views])):
            render_results[i*(self.opts.video_length - 1)] = self.img_raws[i]
        save_videoimages(render_results, os.path.join(self.opts.save_dir, 'render0'))
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)

        save_pointcloud_with_normals(imgs[:self.opts.n_views], pcd[:self.opts.n_views], msk=masks, \
                                     save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        # =========== 新增内存释放代码 ===========
        # 保存需要的渲染结果
        render_results_cpu = render_results.cpu()

        # 释放Dust3r相关内存
        self.scene = None

        # 清理不需要的大型变量
        del c2ws, pcd, principal_points, focals, imgs, masks
        del camera_traj, viewmask
        if 'masks_new' in locals():
            del masks_new
        if 'bgs_mask' in locals():
            del bgs_mask
        if 'depth' in locals():
            del depth

        # 强制执行垃圾回收
        import gc
        gc.collect()

        # 清理CUDA缓存
        torch.cuda.empty_cache()
        print("已释放Dust3r相关内存，准备运行Diffusion...")

        # 将渲染结果移回GPU
        render_results = render_results_cpu.to(self.device)
        # =========== 内存释放代码结束 ===========

        diffusion_results = []
        # print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori[:self.opts.n_views])-1 ):
            print(f'Generating clip {i} ...\n')
            diffusion_result = self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)])
            if i < len(self.img_ori[:self.opts.n_views])-2:
                diffusion_result = diffusion_result[:-1]
            diffusion_results.append(diffusion_result)

        print(f'Finish!\n')
        diffusion_results = torch.cat(diffusion_results)
        # save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        save_videoimages((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion'))
        # # torch.Size([25, 576, 1024, 3])
        return diffusion_results


    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)

    def load_initial_images(self, image_dir):
        print('-----------------------')
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
        images = load_images([image_dir], size=512,force_1024 = False)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori
    def split_images_train_test(self, image_files, n_views=0, llffhold=8):
        """
        Split image files into training and testing sets.

        Args:
            image_files (list): List of image file paths
            n_views (int, optional): Number of views to select for training.
                                    If 0, use all available training views.
            llffhold (int, optional): Hold every llffhold-th image for testing. Default is 8.

        Returns:
            tuple: (training_files, testing_files) - Lists of file paths
        """
        train_files = [img for idx, img in enumerate(image_files) if idx % llffhold != 0]
        test_files = [img for idx, img in enumerate(image_files) if idx % llffhold == 0]

        num_test = 12
        if len(test_files) > num_test:
            # 计算整数间隔，确保索引为整数且不重复
            if len(test_files) <= 1:
                indices = [0] * num_test  # 边缘情况处理
            else:
                # 计算实际需要的间隔，保证能够均匀分布
                step = (len(test_files) - 1) / (num_test - 1) if num_test > 1 else 1

                # 生成整数索引
                indices = []
                for i in range(num_test):
                    # 计算当前索引并确保它是整数
                    idx = int(round(i * step))

                    # 确保索引不超过范围
                    idx = min(idx, len(test_files) - 1)

                    # 添加到索引列表
                    indices.append(idx)

                # 确保索引唯一（移除重复）
                indices = sorted(list(set(indices)))

                # 如果去重后索引数量不足，需要补充
                while len(indices) < num_test:
                    # 寻找最大间隔并在中间添加一个索引
                    max_gap = 0
                    insert_pos = 0
                    for i in range(len(indices) - 1):
                        gap = indices[i+1] - indices[i]
                        if gap > max_gap:
                            max_gap = gap
                            insert_pos = i

                    # 在最大间隔中间插入一个索引
                    if max_gap > 1:  # 确保有空间插入
                        new_idx = indices[insert_pos] + max_gap // 2
                        indices.insert(insert_pos + 1, new_idx)
                    else:
                        # 如果没有足够的间隔，就在末尾添加（但确保不超出范围）
                        new_idx = min(indices[-1] + 1, len(test_files) - 1)
                        if new_idx not in indices:
                            indices.append(new_idx)
                        else:
                            # 如果尝试添加的索引已存在，尝试在开头添加
                            new_idx = max(indices[0] - 1, 0)
                            if new_idx not in indices:
                                indices.insert(0, new_idx)
                            else:
                                # 如果仍然无法添加，在现有索引之间寻找任何可能的空隙
                                found = False
                                for i in range(len(test_files)):
                                    if i not in indices:
                                        indices.append(i)
                                        indices.sort()
                                        found = True
                                        break
                                if not found:
                                    # 如果所有可能的索引都已使用，只能重复使用末尾索引
                                    indices.append(indices[-1])

            # 确保最终选择的索引数量正确
            indices = indices[:num_test]
            print(f'selected {num_test} test files: {indices}')
            test_files = [test_files[i] for i in indices]

            if n_views > 0 and n_views < len(train_files):
                idx_sub = np.linspace(0, len(train_files) - 1, n_views)
                idx_sub = [round(i) for i in idx_sub]
                train_files = [train_files[i] for i in idx_sub]
                assert len(train_files) == n_views, f"Selected {len(train_files)} views, expected {n_views}"

            print(f"Split images: {len(train_files)} for training, {len(test_files)} for testing")
        return train_files, test_files

    def load_initial_dir(self, image_dir):

        image_files = [f for f in sorted(glob.glob(os.path.join(image_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        # image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        image_files = natsorted(image_files)
        train_files, test_files = self.split_images_train_test(image_files, n_views=3)
        image_files = train_files
        # image_files = image_files[:3]
        # image_files = [image_files[0],image_files[-1]]
        print(f'load image path total number = {len(image_files)}')
        # images, images_info = load_images_original(image_files, size=512,force_1024 = True)
        images, images_info = load_images_resized(image_files)

        img_gts = []
        img_raws = []
        img_names = []
        img_crops = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img'].squeeze(0).permute(1,2,0)+1.)/2.)
            img_raws.append((images_info[i]['img_raw'].squeeze(0).permute(1,2,0)+1.)/2.)
            img_names.append(images_info[i]['img_name'])
            img_crops.append(images_info[i]['img_crop'])

        return images, img_gts, [img_raws, img_names, img_crops]

    def load_initial_dir_crop(self, image_dir):

        image_files = glob.glob(os.path.join(image_dir, "*"))

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = natsorted(image_files)
        train_files, test_files = self.split_images_train_test(image_files, n_views=self.opts.n_views)

        image_files = train_files

        self.pc_image_files = image_files
        images = load_images(image_files, size=512,force_1024 = True)

        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)

        img_raws = []
        for i in range(len(image_files)):
            img_raws.append((images[i]['img'].squeeze(0).permute(1,2,0)+1.)/2.)

        return images, img_gts, img_raws

    def run_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed):
        self.opts.elevation = float(i2v_elevation)
        self.opts.center_scale = float(i2v_center_scale)
        self.opts.ddim_steps = i2v_steps
        self.gradio_traj = [float(i) for i in i2v_d_phi.split()],[float(i) for i in i2v_d_theta.split()],[float(i) for i in i2v_d_r.split()]
        seed_everything(i2v_seed)
        torch.cuda.empty_cache()
        img_tensor = torch.from_numpy(i2v_input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = center_crop_image(img_tensor) #1,3,h,w
        images = get_input_dict(image_tensor_resized,idx = 0,dtype = torch.float32)
        images = [images, copy.deepcopy(images)]
        images[1]['idx'] = 1
        self.images = images
        self.img_ori = (image_tensor_resized.squeeze(0).permute(1,2,0) + 1.)/2.

        # self.images: torch.Size([1, 3, 288, 512]), [-1,1]
        # self.img_ori:  torch.Size([576, 1024, 3]), [0,1]
        # self.images, self.img_ori = self.load_initial_images(image_dir=i2v_input_image)
        self.run_dust3r(input_images=self.images)
        self.nvs_single_view(gradio=True)

        traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion0.mp4")

        return traj_dir, gen_dir
