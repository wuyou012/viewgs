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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
import gaussian_utils.loss_utils as loss_utils
from gaussian_utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, get_vit_feature, PatchSWDLoss
from gaussian_utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from gaussian_utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from gaussian_utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
# from FDL_pytorch import FDL_loss
from gaussian_utils.extractor import VitExtractor
import torch.nn.functional as F
# from gaussian_scene.cameras import Camera
import random
from run_point_cloud_densification import realtimePointsDensification, point_cloud_to_depth
import imageio

def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for Debuger to Attach on Port 5678...")
    debugpy.wait_for_client()
    print("Attached!")
if os.environ.get("VIEWGS_ATTACH_DEBUGGER") == "1":
    attach_debugger()

def _parse_index_list(index_text):
    if index_text is None or str(index_text).strip() == "":
        return None
    return [int(x.strip()) for x in str(index_text).split(',') if x.strip() != ""]


def _parse_float_list(text, default):
    if text is None or str(text).strip() == "":
        return list(default)
    return [float(x.strip()) for x in str(text).split(',') if x.strip() != ""]


def select_fixed_cameras(cameras, count=0, index_text=None):
    """Select a deterministic subset of CVI cameras while still loading all cameras."""
    if not cameras or cameras[0] is None:
        return cameras, []
    n_cameras = len(cameras)
    explicit_indices = _parse_index_list(index_text)
    if explicit_indices is not None:
        raw_indices = explicit_indices
    elif count is None or int(count) <= 0 or int(count) >= n_cameras:
        raw_indices = list(range(n_cameras))
    else:
        count = int(count)
        raw_indices = np.rint(np.linspace(0, n_cameras - 1, count)).astype(int).tolist()

    indices = []
    for idx in raw_indices:
        idx = max(0, min(n_cameras - 1, int(idx)))
        if idx not in indices:
            indices.append(idx)
    if explicit_indices is None and count is not None and int(count) > 0:
        fill_idx = 0
        while len(indices) < min(int(count), n_cameras):
            if fill_idx not in indices:
                indices.append(fill_idx)
            fill_idx += 1
    indices = indices[:n_cameras]
    return [cameras[i] for i in indices], indices


def training(dataset, opt, pipe, args, metrics):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(args)
    scene_data_dir = dataset.source_path

    scene = Scene(args, gaussians, shuffle=False)
    selected_mid_cameras, selected_mid_indices = [], []
    if args.use_diff:
        all_mid_cameras = scene.getMidCameras().copy()
        selected_mid_cameras, selected_mid_indices = select_fixed_cameras(
            all_mid_cameras, args.diff_view_count, args.diff_view_indices
        )
        selected_names = [getattr(cam, "image_name", str(idx)) for idx, cam in zip(selected_mid_indices, selected_mid_cameras)]
        print(f"CVI supervision cameras: selected {len(selected_mid_cameras)}/{len(all_mid_cameras)} "
              f"indices={selected_mid_indices} names={selected_names}")
    args.selected_diff_view_indices = selected_mid_indices
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    opt.iterations = args.training_iters

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    dif_depth = None
    dif_midas_depth = None
    ema_loss_for_log = 0.0
    first_iter += 1


    vit_ext0 = VitExtractor(model_name='dino_vits16', device="cuda:0")

    if args.disable_gt:
        print('disable gt after 5000 iterations')

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        loss = 0
        loss_l1_ssim = 0
        dif_gs_loss = 0
        dif_pear_loss = 0
        dif_corr_loss = 0
        lap_loss = 0
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        names = []
        for camera in viewpoint_stack:
            names.append(camera.image_name)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 =  l1_loss_mask(image, gt_image)
        loss_l1_ssim, Ll1 = loss_utils.gaussian_loss(image, gt_image, opt.lambda_dssim)
        if args.disable_gt and iteration > 5000:
            loss_l1_ssim = 0
        loss += loss_l1_ssim


        rendered_depth = render_pkg["depth"][0] # (378 504)
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()

        depth_loss = loss_utils.depth_correlation_loss(rendered_depth, midas_depth)

        loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001

        if iteration > args.start_sample_pseudo:
            if args.use_diff:

                dif_stack = selected_mid_cameras.copy()
                dif_cam = dif_stack.pop(randint(0, len(dif_stack) - 1))
                dif_image = dif_cam.original_image.cuda()
                render_pkg_dif = render(dif_cam, gaussians, pipe, background)
                render_dif = render_pkg_dif["render"]
                dif_depth = render_pkg_dif["depth"][0]
                dif_midas_depth = estimate_depth(dif_image, mode='train')

                depth_percentile_60 = torch.quantile(dif_midas_depth, 0.60)  # 获取60%深度阈值
                depth_mask = (dif_midas_depth >= depth_percentile_60).float()  # 创建二值掩码

                # 可选: 对掩码应用轻微模糊以避免硬边界
                if depth_mask.sum() > 0:  # 确保掩码不是空的
                    # 手动实现高斯模糊
                    kernel_size = 5
                    sigma = 1.0

                    # 创建高斯核
                    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device='cuda')
                    kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
                    kernel_1d = kernel_1d / kernel_1d.sum()

                    # 转换为2D核
                    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
                    kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)

                    # 应用卷积来模拟高斯模糊
                    padding = kernel_size // 2
                    depth_mask = depth_mask.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
                    depth_mask = F.conv2d(depth_mask, kernel_2d, padding=padding)
                    depth_mask = depth_mask.squeeze()  # 移除批次和通道维度

                # 可选: 如果想每1000次迭代保存一次掩码图像用于可视化
                if args.use_wandb and iteration % 1000 == 0:
                    min_depth = dif_midas_depth.min().item()
                    max_depth = dif_midas_depth.max().item()
                    mean_depth = dif_midas_depth.mean().item()
                    threshold = depth_percentile_60.item()

                    # 记录深度统计信息
                    wandb.log({
                        "depth_stats/min": min_depth,
                        "depth_stats/max": max_depth,
                        "depth_stats/mean": mean_depth,
                        "depth_stats/threshold": threshold,
                        "depth_stats/mask_ratio": depth_mask.mean().item(),  # 掩码覆盖比例
                        "iteration": iteration
                    })
                    depth_mask_vis = depth_mask.cpu().numpy()*255.0
                    wandb.log({"depth/mask": wandb.Image(depth_mask_vis, caption="Depth Mask (<60%)")})

                if args.dif_pear_loss_weight != 0:
                    masked_dif_depth = dif_depth * depth_mask
                    masked_dif_midas_depth = dif_midas_depth * depth_mask
                    # dif_pear_loss = loss_utils.depth_correlation_loss(render_dif, dif_image)

                    # dif_pear_loss = loss_utils.depth_correlation_loss(dif_depth, dif_midas_depth)

                    psg_scales = _parse_float_list(args.psg_scales, [1.0, 0.5, 0.25, 0.125])
                    psg_weights = [args.psg_ws for _ in psg_scales]
                    depth_loss = loss_utils.multi_scale_depth_correlation_loss(
                        dif_depth, dif_midas_depth, scales=psg_scales, weights=psg_weights)
                    warmup_iter = args.training_iters * args.psg_alpha
                    if iteration < warmup_iter:
                        combined_depth_loss = depth_loss
                    else:
                        if args.psg_disable_spatial_mask:
                            combined_depth_loss = depth_loss
                        else:
                            masked_depth_loss = loss_utils.multi_scale_depth_correlation_loss(
                                masked_dif_depth, masked_dif_midas_depth, scales=psg_scales, weights=psg_weights)
                            combined_depth_loss = depth_loss + args.psg_beta * masked_depth_loss

                    if args.psg_disable_progressive:
                        depth_weight_factor = 1.0
                    elif iteration < warmup_iter:
                        depth_weight_factor = 1.0
                    else:
                        depth_weight_factor = max(args.psg_eta_min, 1.0 - (iteration - warmup_iter) / (args.training_iters * 0.5))
                    dif_pear_loss = args.dif_pear_loss_weight * depth_weight_factor * combined_depth_loss

                if args.struct_weight != 0:
                    struct_loss = loss_utils.structural_consistency_loss(dif_depth, dif_midas_depth)
                    dif_pear_loss += args.struct_weight * struct_loss  # 新增参数

                if args.dif_l1_loss_weight != 0:
                    dif_gs_loss, diff_l1 = loss_utils.gaussian_loss(render_dif, dif_image, opt.lambda_dssim, args.loss_downsample, args.l_down_ratio)
                    dif_gs_loss = args.dif_l1_loss_weight * dif_gs_loss

                if args.gray_corr_weight != 0:
                    # dif_corr_loss = loss_utils.grayscale_correlation_loss(render_dif, dif_image)
                    # dif_corr_loss = loss_utils.channel_weighted_correlation_loss(render_dif, dif_image)
                    dif_corr_loss = loss_utils.multi_scale_channel_correlation_loss(render_dif, dif_image, scales=[1.0, 0.5, 0.25])
                    dif_corr_loss = args.gray_corr_weight * dif_corr_loss

                if args.lap_weight != 0:
                    lap_loss = loss_utils.laplacian_pyramid_loss(
                                render_dif,
                                dif_image,
                                max_levels=args.lap_levels,  # 使用指定的金字塔层级
                                kernel_size=5,  # 5x5高斯核
                                sigma=1.0       # 标准差
                            )
                    lap_loss = args.lap_weight * lap_loss

                loss += dif_gs_loss + dif_pear_loss + dif_corr_loss + lap_loss
                # # 简单使用（仅通道加权相关）
                # loss = channel_weighted_correlation_loss(rendered_image, target_image)

                # # 多尺度使用
                # loss = loss_utils.multi_scale_channel_correlation_loss(
                #     render_dif,
                #     dif_image,
                #     scales=[1.0, 0.5, 0.25]
                #     )

                # # 完整高级使用
                # loss = advanced_channel_correlation_loss(
                    # rendered_image,
                    # target_image,
                    # use_multi_scale=True,
                    # scales=[1.0, 0.5, 0.25],
                    # use_gradient_images=True,
                    # use_combined_loss=True
                    # )
                if iteration == 2100:
                    print('-------- args: pear l1 gray_corr --------',args.dif_pear_loss_weight, \
                          args.dif_l1_loss_weight, args.gray_corr_weight)
                    print('-------- down sample l1 --------',args.l_down_ratio)

        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:

            if args.use_pseudo:
                if not pseudo_stack:
                    pseudo_stack = scene.getPseudoCameras().copy()
                pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

                render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
                rendered_depth_pseudo = render_pkg_pseudo["depth"][0] # (378,504) (260 398) (411 618)
                midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')
                if iteration == 2100:
                    print('----------------args:',args.W, args.D, args.A)

                depth_loss_pseudo, patch_depth_loss_pseudo = loss_utils.windowed_depth_loss(
                    rendered_depth_pseudo,
                    midas_depth_pseudo,
                    window_size=126,
                    stride=64
                )

                if torch.isnan(depth_loss_pseudo).sum() == 0:
                    loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                    loss += loss_scale *args.A *(depth_loss_pseudo + args.W * patch_depth_loss_pseudo)

                #---------------------- crop dino psoude ----------------------
                rendered_image_pseudo = render_pkg_pseudo["render"]
                gt_image = gt_image
                # rendered_crop = loss_utils.random_crop(rendered_image_pseudo,(84,63))
                # gt_crop = loss_utils.random_crop(gt_image,(84,63))
                gt_crop, rendered_crop = loss_utils.random_crop3(gt_image, rendered_image_pseudo ,(83,63))
                render_crop_pseudo_vit = get_vit_feature(rendered_crop.unsqueeze(0), vit_ext0)
                gt_crop_vit = get_vit_feature(gt_crop.unsqueeze(0), vit_ext0)
                side_loss_dino = F.mse_loss(render_crop_pseudo_vit, gt_crop_vit)
                loss += args.D *side_loss_dino

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.set_postfix({
                    "points": f"{scene.gaussians.get_xyz.shape[0]}",
                    "L1_ssim": f"{loss_l1_ssim:.3f}",
                    "dif_gs": f"{dif_gs_loss:.3f}" if args.use_diff else "Nan",
                    "dif_pear": f"{dif_pear_loss:.3f}" if args.use_diff else "Nan",
                    "gray_corr": f"{dif_corr_loss:.3f}" if args.use_diff else "Nan",
                    "lap_pyr": f"{lap_loss:.3f}" if args.use_diff else "Nan",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background),metrics)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()

            if args.use_wandb and iteration % 10 == 0:
                log_dict = {
                    "iteration": iteration,
                    "loss/total": loss.item(),
                    "loss/l1_ssim": loss_l1_ssim.item() if isinstance(loss_l1_ssim, torch.Tensor) else loss_l1_ssim,
                    "training/points": scene.gaussians.get_xyz.shape[0],
                    "training/lr": gaussians.optimizer.param_groups[0]['lr'],
                }

                # 添加其他损失项（如果存在）
                if args.use_diff and iteration > args.start_sample_pseudo:
                    log_dict.update({
                        "loss/dif_gs": dif_gs_loss.item() if isinstance(dif_gs_loss, torch.Tensor) else dif_gs_loss,
                        "loss/dif_pear": dif_pear_loss.item() if isinstance(dif_pear_loss, torch.Tensor) else dif_pear_loss,
                        "loss/gray_corr": dif_corr_loss.item() if isinstance(dif_corr_loss, torch.Tensor) else dif_corr_loss,
                        "loss/lap_pyr": lap_loss.item() if isinstance(lap_loss, torch.Tensor) else lap_loss,
                    })

                if args.use_diff and dif_depth is not None and dif_midas_depth is not None and render_dif is not None:
                    # 每1000次迭代记录深度图和RGB图像
                    if iteration % 1000 == 0:
                        # 归一化深度图用于可视化
                        pred_depth_vis = (dif_depth - dif_depth.min()) / (dif_depth.max() - dif_depth.min() + 1e-8)
                        gt_depth_vis = (dif_midas_depth - dif_midas_depth.min()) / (dif_midas_depth.max() - dif_midas_depth.min() + 1e-8)

                        # 确保图像已被正确转换为numpy数组
                        render_dif_np = render_dif.permute(1, 2, 0).detach().cpu().numpy()
                        dif_image_np = dif_image.permute(1, 2, 0).detach().cpu().numpy()

                        # 检查图像是否全黑
                        if render_dif_np.max() <= 0.01:
                            print("警告: 渲染图像几乎全黑")

                        # 确保保存一张图像而不是图像列表
                        log_dict.update({
                            "depth/predicted": wandb.Image(pred_depth_vis.cpu().numpy()),
                            "depth/estimated": wandb.Image(gt_depth_vis.cpu().numpy()),
                            "rgb/rendered": wandb.Image(render_dif_np),
                            "rgb/target": wandb.Image(dif_image_np),
                        })

                wandb.log(log_dict)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 初始化 wandb (仅在启用时)
    if args.use_wandb:
        # 提取场景名称用于项目分组
        scene_name = args.source_path.split('/')[-1]
        if scene_name == "":
            scene_name = args.source_path.split('/')[-2]

        # 创建一个有意义的运行名称
        run_name = f"{scene_name}_views{args.n_views}"
        if args.use_diff:
            run_name += "_diff"
        if args.use_pseudo:
            run_name += "_pseudo"

        # 配置 wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)  # 上传所有参数
        )

        # 添加系统信息和代码文件
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # 创建 Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs,metrics):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                rendered_images = []  # 为每个配置存储渲染的图像
                gt_images_list = []   # 为每个配置存储GT图像

                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 保存用于wandb记录的图像
                    if idx < 4:  # 仅保存前4张图像用于wandb
                        rendered_images.append({
                            'image': image.detach().clone(),
                            'name': viewpoint.image_name
                        })
                        gt_images_list.append({
                            'image': gt_image.detach().clone(),
                            'name': viewpoint.image_name
                        })

                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {:.5f} PSNR {:.5f} SSIM {:.5f} LPIPS {:.5f} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))

                if config['name'] == 'test':
                    metrics.setdefault('ITER', []).append(iteration)
                    metrics['PSNR'].append(float(f"{psnr_test.item():.4f}"))
                    metrics['SSIM'].append(float(f"{ssim_test.item():.4f}"))
                    metrics['LPIPS'].append(float(f"{lpips_test.item():.4f}"))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                # wandb日志记录
                if 'use_wandb' in args.__dict__ and args.use_wandb:
                    # 每1000次迭代或在最后一次迭代时记录一张图像
                    if iteration % 1000 == 0 or iteration == args.training_iters:
                        # 只取第一张图像
                        if rendered_images and gt_images_list:
                            # 确保使用CPU张量并正确转换格式
                            render_img_np = rendered_images[0]['image'].permute(1, 2, 0).cpu().numpy()
                            gt_img_np = gt_images_list[0]['image'].permute(1, 2, 0).cpu().numpy()

                            # 为每个配置记录指标和单个图像
                            log_dict = {
                                f"{config['name']}/psnr": psnr_test.item(),
                                f"{config['name']}/ssim": ssim_test.item(),
                                f"{config['name']}/lpips": lpips_test.item(),
                                f"{config['name']}/render": wandb.Image(render_img_np, caption=f"Render {rendered_images[0]['name']}"),
                                f"{config['name']}/ground_truth": wandb.Image(gt_img_np, caption=f"GT {gt_images_list[0]['name']}"),
                                "iteration": iteration,
                            }
                            wandb.log(log_dict)
                    else:
                        # 只记录指标，不记录图像
                        log_dict = {
                            f"{config['name']}/psnr": psnr_test.item(),
                            f"{config['name']}/ssim": ssim_test.item(),
                            f"{config['name']}/lpips": lpips_test.item(),
                            "iteration": iteration,
                        }
                        wandb.log(log_dict)


        torch.cuda.empty_cache()

def add_wandb_args(parser):
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument("--use_wandb", action="store_true", help="启用 Weights & Biases 日志")
    group.add_argument("--wandb_project", type=str, default="3dgs-depth", help="WandB 项目名称")
    group.add_argument("--wandb_entity", type=str, default=None, help="WandB 实体名称 (可选)")
    return group

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_wandb_args(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--use_diff', action='store_true', help='Enable differential rendering')
    parser.add_argument('--no_diff', action='store_true', help='Disable differential rendering')
    parser.add_argument('--use_pseudo', action='store_true', help='Enable pseudo views')
    parser.add_argument('--no_pseudo', action='store_true', help='Disable pseudo views')
    parser.add_argument('--disable_gt', action='store_true', help='Disable gt after 5000 iterations')
    parser.add_argument('--loss_downsample', action='store_true', help='downsample image for loss calculation')
    parser.add_argument('--diff_path', type=str, default="./")
    parser.add_argument("--edge_weight", type=float, default=3.0, help="Edge importance weight in depth loss")

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000,4_000,6_000,8_000,10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for repeatable runs")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--l_down_ratio", type=float, default = 0.25)
    parser.add_argument("--sub_ratio", type=int, default = 20)
    parser.add_argument("--A", type=float, default = 0.05)
    parser.add_argument("--D", type=float, default = 0.1)
    parser.add_argument("--W", type=float, default = 0.25)
    parser.add_argument("--dif_l1_loss_weight", type=float, default = 0)
    parser.add_argument("--dif_pear_loss_weight", type=float, default = 0)
    parser.add_argument("--gray_corr_weight", type=float, default=0,
                    help="Weight for grayscale correlation loss")
    parser.add_argument("--lap_weight", type=float, default=0, help="Weight for laplacian pyramid loss")
    parser.add_argument("--lap_levels", type=int, default=3, help="Number of Laplacian pyramid levels")
    parser.add_argument("--psg_scales", type=str, default="1,0.5,0.25,0.125", help="Comma-separated PSG depth scales")
    parser.add_argument("--psg_ws", type=float, default=1.0, help="Uniform weight for each PSG depth scale")
    parser.add_argument("--psg_alpha", type=float, default=0.5, help="Warm-up ratio before enabling the spatial-aware PSG term")
    parser.add_argument("--psg_beta", type=float, default=0.25, help="Weight of the spatial-aware PSG masked term")
    parser.add_argument("--psg_eta_min", type=float, default=0.5, help="Minimum progressive PSG schedule factor")
    parser.add_argument("--diff_view_count", type=int, default=0, help="Deterministic number of CVI mid views to supervise; 0 means all")
    parser.add_argument("--diff_view_indices", type=str, default="", help="Comma-separated fixed CVI mid-view indices; overrides diff_view_count")
    parser.add_argument("--psg_disable_progressive", action="store_true", help="Use PSG depth loss without progressive schedule")
    parser.add_argument("--psg_disable_spatial_mask", action="store_true", help="Use PSG depth loss without spatial-aware masked term")
    parser.add_argument("--struct_weight", type=float, default=0, help="Weight for structural consistency loss")
    args = parser.parse_args(sys.argv[1:])

    # 优先使用明确指定的参数
    if args.no_diff:
        print('------ not use diffusion ------')
        args.use_diff = False
    if args.no_pseudo:
        print('------ not use pseudo view ------')
        args.use_pseudo = False


    args.test_iterations.append(args.training_iters)
    scene_data_dir = lp.extract(args).source_path
    scene = scene_data_dir.split('/')[-3]
    opt_dir = args.model_path.split('/')[0]

    print(args.test_iterations)

    print("Optimizing " + args.model_path)
    # test_result = []
    metrics = {
    'ITER': [],
    'PSNR': [],
    'SSIM': [],
    'LPIPS': []
    }
    op.densify_until_iter = args.training_iters
    op.position_lr_max_steps = args.training_iters
    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, metrics)

    if args.use_wandb:
        # 记录最终结果
        wandb.log({
            "final/psnr": metrics['PSNR'][-1],
            "final/ssim": metrics['SSIM'][-1],
            "final/lpips": metrics['LPIPS'][-1]
        })

        # 创建摘要表格
        result_table = wandb.Table(columns=["Iteration", "PSNR", "SSIM", "LPIPS"])
        for i, iter_num in enumerate(args.test_iterations):
            if i < len(metrics['PSNR']):
                result_table.add_data(iter_num, metrics['PSNR'][i], metrics['SSIM'][i], metrics['LPIPS'][i])

        wandb.log({"results": result_table})

        wandb.finish()

    # All done
    print("\nTraining complete.")
    import json
    output_dir = args.model_path
    output_data = {
        "ITER": metrics.get('ITER', []),
        "PSNR": metrics['PSNR'],
        "SSIM": metrics['SSIM'],
        "LPIPS": metrics['LPIPS'],
        "scene": scene,
        "seed": args.seed,
        "diff_view_count": args.diff_view_count,
        "diff_view_indices": getattr(args, "selected_diff_view_indices", []),
        "lap_levels": args.lap_levels,
        "psg_disable_progressive": args.psg_disable_progressive,
        "psg_disable_spatial_mask": args.psg_disable_spatial_mask,
        "psg_scales": args.psg_scales,
        "psg_ws": args.psg_ws,
        "psg_alpha": args.psg_alpha,
        "psg_beta": args.psg_beta,
        "psg_eta_min": args.psg_eta_min
    }
    output_path = os.path.join(output_dir, "test_results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
