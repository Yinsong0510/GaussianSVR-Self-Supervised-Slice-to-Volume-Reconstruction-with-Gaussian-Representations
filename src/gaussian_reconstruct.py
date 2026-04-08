import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from kornia.losses import ssim_loss

from models.gaussian_model import GaussianModel
from transform import RigidTransform
from data.rescan import ReScanner
from utils import create_grid_3d, tv_regularization


class OptimizationParams:
    def __init__(self):
        self.position_lr_init = 0.002
        self.position_lr_final = 0.00002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.intensity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, decay_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.decay_steps = decay_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        elif self.warmup_steps <= step < self.decay_steps:
            return 1
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.decay_steps)),
        )


def gaussian_reconstruct(config, stacks, initial_transforms, initial_volume,
                         resolution, resolution_slice, slice_thickness,
                         device, output_dir=None, log_dir=None):
    """Joint Gaussian + transform optimization to reconstruct a 3D volume.

    Args:
        config: dict of hyperparameters (from config/default.yaml).
        stacks: [N, 1, H, W] original acquired slices.
        initial_transforms: [N, 3, 4] initial rigid transform matrices
            from SVoRTv2 motion estimation.
        initial_volume: [1, 1, D, H, W] initial volume estimate.
        resolution: volume voxel size (mm).
        resolution_slice: slice in-plane pixel spacing (mm).
        slice_thickness: slice thickness (mm).
        device: torch device.
        output_dir: optional path for saving checkpoints and images.
        log_dir: optional path for tensorboard logs.

    Returns:
        volume: [1, 1, D, H, W] reconstructed volume tensor.
        transforms: [N, 6] optimized rigid transforms (axis-angle).
    """
    slice_size = stacks.shape[3]
    re_scanner = ReScanner(slice_size=slice_size)

    # Prepare volume: [1, D, H, W, 1]
    init_vol = initial_volume.permute(0, 2, 3, 4, 1)
    _, d, h, w, _ = init_vol.shape

    grid = create_grid_3d(d, h, w).to(device).unsqueeze(0)
    mask = (init_vol > 0).to(torch.uint8).to(device)

    grid_low = grid[:, ::2, ::2, ::2, :]
    image_low = init_vol[:, ::2, ::2, ::2, :]
    mask_low = mask[:, ::2, ::2, ::2, :]
    seg_low = mask_low[..., 0]

    # --- Initialize Gaussians ---
    op = OptimizationParams()
    gaussians = GaussianModel()
    gaussians.random_initialize(
        seg_low, image_low,
        ini_intensity=config['ini_intensity'],
        ini_sigma=config['ini_sigma'],
        spatial_lr_scale=config['spatial_lr_scale'],
        fg_num_samples=config['num_fg_gaussian'],
        bg_num_samples=config['num_bg_gaussian'],
    )
    gaussians.training_setup(op)

    # --- Initialize transform parameters ---
    transforms_rt = RigidTransform(initial_transforms)
    rot = transforms_rt.axisangle()[..., :3]
    trans = transforms_rt.axisangle()[..., -3:]
    rot_params = torch.nn.Parameter(rot, requires_grad=True)
    trans_params = torch.nn.Parameter(trans, requires_grad=True)

    optimizer_motion = torch.optim.Adam([
        {'params': [rot_params], 'lr': 0.0001, "name": "rotation"},
        {'params': [trans_params], 'lr': 0.001, "name": "translation"},
    ], lr=0.0, eps=1e-15)

    scheduler = WarmupLinearSchedule(
        optimizer_motion,
        warmup_steps=1500,
        decay_steps=4000,
        t_total=config['max_iter'],
    )

    writer = SummaryWriter(log_dir) if log_dir else None

    # --- Training loop ---
    print("Starting Gaussian reconstruction...")
    for iteration in range(config['max_iter']):
        gaussians.update_learning_rate(iteration)
        gaussians.optimizer.zero_grad(set_to_none=True)
        optimizer_motion.zero_grad(set_to_none=True)

        total_params = torch.cat([rot_params, trans_params], dim=-1)

        if iteration < config['low_reso_stage']:
            train_output = gaussians.grid_sample(grid_low)
            volume = torch.nn.functional.interpolate(
                train_output.permute(0, 4, 1, 2, 3),
                size=init_vol.shape[1:4], mode='trilinear', align_corners=False,
            )
        else:
            train_output = gaussians.grid_sample(grid)
            volume = train_output.permute(0, 4, 1, 2, 3)

        re_slices = re_scanner.scan(
            volume, total_params, resolution, resolution_slice, slice_thickness,
        )

        l1_loss = config['l1_weight'] * torch.nn.functional.l1_loss(re_slices, stacks)
        ssim = (1 - config['l1_weight']) * ssim_loss(re_slices, stacks, window_size=11)
        tv_loss = config['tv_weight'] * tv_regularization(train_output)
        loss = l1_loss + tv_loss + ssim

        loss.backward()
        gaussians.optimizer.step()
        optimizer_motion.step()

        # Densification
        if config['do_density_control']:
                with torch.no_grad():
                    if iteration > config['densify_from_iter'] and iteration % config['densification_interval'] == 0:
                        # Densification (only if under max_gaussians and before densify_until_iter)
                        if gaussians.get_xyz.shape[0] < config['max_gaussians'] and iteration < config['densify_until_iter']:
                            gaussians.densify(config['max_grad'], sigma_extent=config['sigma_extent'])
                        # Pruning (always, independent of densification)
                        gaussians.prune(config['min_intensity'])

        log_msg = (
            f"\rIteration {iteration + 1:>{len(str(config['max_iter']))}}"
            f"/{config['max_iter']} | "
            f"L1: {l1_loss.item():.6f} | "
            f"TV: {tv_loss.item():.6f} | "
            f"Total: {loss.item():.6f}"
        )
        sys.stdout.write(log_msg)
        sys.stdout.flush()

        if writer:
            writer.add_scalar('Loss/total', loss.item(), iteration)
            writer.add_scalar('Loss/L1', l1_loss.item(), iteration)
            writer.add_scalar('Loss/TV', tv_loss.item(), iteration)

        if output_dir and (iteration + 1) % config['image_save_iter'] == 0:
            print(f"\nSaving model at iteration {iteration + 1}...")
            gaussians.save_model(output_dir, iteration + 1)

        scheduler.step()

    if writer:
        writer.close()
    print("\nReconstruction complete.")

    # Final volume at full resolution
    with torch.no_grad():
        final_output = gaussians.grid_sample(grid)
        final_volume = final_output.permute(0, 4, 1, 2, 3)

    final_transforms = torch.cat([rot_params, trans_params], dim=-1).detach()

    return final_volume, final_transforms
