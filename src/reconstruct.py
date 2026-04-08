"""Reconstruct a 3D volume using pretrained SVoRT + Gaussian representation.

Usage::

    python reconstruct.py \\
        --stacks stack1.nii.gz stack2.nii.gz stack3.nii.gz \\
        --slice_thickness 3.0 \\
        --config ../config/default.yaml \\
        --output output/
"""

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import seed_everything

from svort import run_svort
from transform import RigidTransform
from gaussian_reconstruct import gaussian_reconstruct
from utils import get_config, prepare_sub_folder, save_img


def main():
    parser = argparse.ArgumentParser(description="Reconstruct 3D volume using GaussianSVR")
    parser.add_argument("--stacks", nargs="+", required=True, help="Paths to NIfTI stack files")
    parser.add_argument("--slice_thickness", type=float, required=True, help="Slice thickness / gap (mm)")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "config", "default.yaml"), help="Path to GaussianSVR config yaml")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    opts = parser.parse_args()

    device = torch.device(opts.device)
    cudnn.benchmark = True
    seed_everything(opts.seed)

    config = get_config(opts.config)
    os.makedirs(opts.output, exist_ok=True)
    checkpoint_dir, image_dir = prepare_sub_folder(opts.output)
    log_dir = os.path.join(opts.output, "logs")

    # --- Stage 1: pretrained SVoRT registration ---
    print(f"Loading {len(opts.stacks)} stacks and running SVoRT...")
    result = run_svort(opts.stacks, opts.slice_thickness, device)

    volume_svort = result["volume_svort"]
    save_img(
        volume_svort[0, 0].detach().cpu().numpy(),
        os.path.join(opts.output, "volume_svort_init.nii.gz"),
    )

    # Concatenate per-stack transforms and slices for Gaussian stage
    svort_transforms = RigidTransform.cat(result["transforms_svort_full"])
    initial_transforms = svort_transforms.matrix()
    ori_stacks = torch.cat(result["stacks_ori"], dim=0)

    res_s = result["res_s"]
    res_r = result["res_r"]
    s_thick = result["slice_thickness"]

    print(f"  stacks: {ori_stacks.shape}  transforms: {initial_transforms.shape}")
    print(f"  res_s={res_s}  res_r={res_r}  slice_thickness={s_thick}")

    # --- Stage 2: Gaussian + transform joint optimisation ---
    print("Starting Gaussian reconstruction...")
    final_volume, final_transforms = gaussian_reconstruct(
        config=config,
        stacks=ori_stacks,
        initial_transforms=initial_transforms,
        initial_volume=volume_svort,
        resolution=res_r,
        resolution_slice=res_s,
        slice_thickness=s_thick,
        device=device,
        output_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # --- Save results ---
    save_img(
        final_volume[0, 0].detach().cpu().numpy(),
        os.path.join(opts.output, "volume_reconstructed.nii.gz"),
    )
    torch.save(final_transforms, os.path.join(opts.output, "transforms.pt"))

    print(f"Results saved to {opts.output}")


if __name__ == "__main__":
    main()
