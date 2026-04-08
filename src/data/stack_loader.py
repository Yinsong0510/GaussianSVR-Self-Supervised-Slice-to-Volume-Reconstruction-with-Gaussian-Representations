from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from data.rescan import get_PSF
from transform import RigidTransform


def _load_nii_volume(path: str):
    """Load a NIfTI file and return (volume, resolutions, affine).

    * volume is transposed to (D, H, W) — slice-first — matching NeSVoR.
    * resolutions is the pixdim[1:4] array.
    """
    img = nib.load(path)
    volume = img.get_fdata().astype(np.float32)
    while volume.ndim > 3:
        volume = volume.squeeze(-1)
    # NeSVoR convention: (x, y, z) on disk -> (D=z, H=y, W=x) in tensor
    volume = volume.transpose(2, 1, 0)
    resolutions = np.array(img.header["pixdim"][1:4], dtype=np.float64)
    affine = img.affine.copy()
    if np.any(np.isnan(affine)):
        affine = img.get_qform()
    return volume, resolutions, affine


def _affine2transformation(
    volume: torch.Tensor,
    mask: torch.Tensor,
    resolutions: np.ndarray,
    affine: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, RigidTransform]:
    """Convert NIfTI affine to per-slice RigidTransform (NeSVoR convention).
    """
    device = volume.device
    d, h, w = volume.shape

    R = affine[:3, :3].copy()
    negative_det = np.linalg.det(R) < 0

    T = affine[:3, -1:].copy()  # T = R @ (-T0 + T_r)
    R = R @ np.linalg.inv(np.diag(resolutions))

    T0 = np.array([(w - 1) / 2 * resolutions[0],
                    (h - 1) / 2 * resolutions[1],
                    0.0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    tz = (
        torch.arange(0, d, device=device, dtype=torch.float32) * resolutions[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R_t = torch.tensor(R, device=device, dtype=torch.float32).unsqueeze(0).expand(d, -1, -1)

    if negative_det:
        volume = torch.flip(volume, (-1,))
        mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1
        R_t = R_t.clone()
        R_t[:, :, 0] *= -1

    transformation = RigidTransform(
        torch.cat((R_t, t), -1).to(torch.float32), trans_first=True
    )
    return volume, mask, transformation


def _resample_2d(x: torch.Tensor, res_old: Tuple[float, float],
                 res_new: Tuple[float, float]) -> torch.Tensor:
    """Resample a batch of 2-D slices [N, 1, H, W] to new in-plane res."""
    if res_old[0] == res_new[0] and res_old[1] == res_new[1]:
        return x
    grids = []
    for i in range(2):  # last two spatial dims (W, H) reversed for grid_sample
        fac = res_old[1 - i] / res_new[1 - i]
        size_new = int(x.shape[-1 - i] * fac)
        grid_max = (size_new - 1) / fac / (x.shape[-1 - i] - 1)
        grids.append(
            torch.linspace(-grid_max, grid_max, size_new,
                           dtype=x.dtype, device=x.device)
        )
    grid = torch.stack(torch.meshgrid(*grids, indexing="ij")[::-1], -1)
    return F.grid_sample(
        x, grid[None].expand(x.shape[0], -1, -1, -1),
        align_corners=True, mode="bilinear",
    )


def load_stacks(
    stack_paths: List[str],
    slice_thickness: float,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    
    res_s = 1.0    # NeSVoR hard-codes in-plane resolution for SVoRT
    res_r = 0.8    # NeSVoR hard-codes recon resolution

    stacks_svort: List[torch.Tensor] = []        # cropped / normalised
    stacks_ori: List[torch.Tensor] = []           # resampled, un-cropped
    transforms_svort: List[RigidTransform] = []   # reset, cropped
    transforms_full: List[RigidTransform] = []    # reset, full (with crop offsets)
    transforms_ori: List[RigidTransform] = []     # original from affine
    crop_idx_list: List[torch.Tensor] = []
    gaps: List[float] = []

    for path in stack_paths:
        volume_np, resolutions, affine = _load_nii_volume(path)
        vol_t = torch.tensor(volume_np, device=device)
        mask_t = (vol_t > 0)

        vol_t, mask_t, transformation = _affine2transformation(vol_t, mask_t, resolutions, affine)

        # shape: [D, 1, H, W]
        slices = vol_t.unsqueeze(1)
        masks = mask_t.unsqueeze(1).float()

        gap = float(resolutions[2])  # z-spacing from header
        gaps.append(gap)

        # --- Resample to res_s = 1.0 mm in-plane ---
        slices = _resample_2d(slices, (resolutions[1], resolutions[0]),
                              (res_s, res_s))
        slices_ori = slices.clone()

        # --- Crop x, y to 128 × 128 centred on brain ROI ---
        # Find bounding box of the slice with the most non-zero pixels
        s = slices[torch.argmax((slices > 0).sum((1, 2, 3))), 0]
        i1, i2 = 0, s.shape[0] - 1
        j1, j2 = 0, s.shape[1] - 1
        while i1 < s.shape[0] and s[i1, :].sum() == 0:
            i1 += 1
        while i2 > 0 and s[i2, :].sum() == 0:
            i2 -= 1
        while j1 < s.shape[1] and s[:, j1].sum() == 0:
            j1 += 1
        while j2 > 0 and s[:, j2].sum() == 0:
            j2 -= 1

        pad_margin = 64
        slices = F.pad(slices, (pad_margin, pad_margin, pad_margin, pad_margin),
                       "constant", 0)
        ci = pad_margin + (i1 + i2) // 2
        cj = pad_margin + (j1 + j2) // 2
        slices = slices[:, :, ci - 64: ci + 64, cj - 64: cj + 64]

        # --- Crop z: remove empty slices (keep contiguous range) ---
        idx = (slices > 0).float().sum((1, 2, 3)) > 0
        nz = torch.nonzero(idx)
        idx[int(nz[0, 0]): int(nz[-1, 0] + 1)] = True
        crop_idx_list.append(idx)
        slices = slices[idx]

        # --- Normalise ---
        stacks_svort.append(slices / torch.quantile(slices[slices > 0], 0.99))
        stacks_ori.append(slices_ori)

        # --- Transform handling ---
        transforms_ori.append(transformation)

        transform_full_ax = transformation.axisangle().clone()
        transform_ax = transform_full_ax[idx].clone()

        # (a) Full reset with crop offsets (for later recovery)
        transform_full_ax[:, :-1] = 0
        transform_full_ax[:, 3] = -(
            (j1 + j2) // 2 - slices_ori.shape[-1] / 2
        ) * res_s
        transform_full_ax[:, 4] = -(
            (i1 + i2) // 2 - slices_ori.shape[-2] / 2
        ) * res_s
        transform_full_ax[:, -1] -= transform_ax[:, -1].mean()

        # (b) SVoRT input: clean reset (identity rotation + centred tz)
        transform_ax[:, :-1] = 0
        transform_ax[:, -1] -= transform_ax[:, -1].mean()

        transforms_svort.append(RigidTransform(transform_ax))
        transforms_full.append(RigidTransform(transform_full_ax))

    # --- Assemble SVoRT data dict ---
    s_thick = float(np.mean([slice_thickness] * len(stack_paths)))

    positions = torch.cat([
        torch.stack((
            torch.arange(len(s), dtype=torch.float32, device=device) - len(s) // 2,
            torch.full((len(s),), float(i), dtype=torch.float32, device=device),
        ), dim=-1)
        for i, s in enumerate(stacks_svort)
    ], dim=0)

    all_stacks = torch.cat(stacks_svort, dim=0)
    all_transforms = RigidTransform.cat(transforms_svort).matrix()

    volume_shape = (200, 200, 200)  # NeSVoR hard-codes this

    psf_rec = get_PSF(
        res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
        device=device,
    )

    data_svort = {
        "stacks": all_stacks,
        "transforms": all_transforms,
        "positions": positions,
        "psf_rec": psf_rec,
        "slice_shape": all_stacks.shape[-2:],    # (128, 128)
        "resolution_slice": res_s,
        "resolution_recon": res_r,
        "slice_thickness": s_thick,
        "volume_shape": volume_shape,
    }

    recovery = {
        "stacks_svort": stacks_svort,            # list of [Ni, 1, 128, 128]
        "stacks_ori": stacks_ori,                 # list of [D, 1, H', W'] resampled
        "transforms_svort": transforms_svort,     # list of RigidTransform (reset, cropped)
        "transforms_full": transforms_full,       # list of RigidTransform (reset + crop offsets)
        "transforms_ori": transforms_ori,         # list of RigidTransform (original from affine)
        "crop_idx": crop_idx_list,                # list of bool tensors
        "gaps": gaps,
        "res_s": res_s,
        "res_r": res_r,
        "slice_thickness": s_thick,
        "psf_rec": psf_rec,
    }

    return {"data_svort": data_svort, "recovery": recovery}
