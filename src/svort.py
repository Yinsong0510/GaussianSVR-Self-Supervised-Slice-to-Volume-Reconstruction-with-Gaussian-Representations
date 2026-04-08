"""SVoRT inference using pretrained weights.

Usage::

    python svort.py \\
        --stacks stack1.nii.gz stack2.nii.gz stack3.nii.gz \\
        --slice_thickness 3.0 \\
        --output output/

Or use ``run_svort()`` programmatically from ``reconstruct.py``.
"""

import argparse
import os
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F

from data.stack_loader import load_stacks
from models import SVoRTv2
from transform import RigidTransform, mat_update_resolution
from slice_acquisition import slice_acquisition
from utils import save_img


SVORT_V2_URL = ("https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def _load_svortv2(device: torch.device) -> SVoRTv2:
    """Download (or load cached) SVoRTv2 checkpoint and return model."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cp = torch.hub.load_state_dict_from_url(
        url=SVORT_V2_URL,
        model_dir=_CACHE_DIR,
        map_location=device,
        file_name="SVoRT_v2.pt",
    )
    model = SVoRTv2(n_iter=4, iqa=True, vol=True, pe=True)
    model.load_state_dict(cp["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def _get_transform_diff_mean(
    transform_out: RigidTransform,
    transform_in: RigidTransform,
    mean_r: int = 3,
) -> Tuple[RigidTransform, RigidTransform]:
    """Compute per-slice diff and its central mean (stack-level correction)."""
    transform_diff = transform_out.compose(transform_in.inv())
    length = len(transform_diff)
    mid = length // 2
    left = max(0, mid - mean_r)
    right = min(length, mid + mean_r)
    transform_diff_mean = transform_diff[left:right].mean(simple_mean=False)
    return transform_diff_mean, transform_diff


def _ncc_loss(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-slice negative NCC (lower is better)."""
    eps = 1e-6
    n = mask.sum((1, 2, 3)).clamp(min=1)
    mx = (x * mask).sum((1, 2, 3)) / n
    my = (y * mask).sum((1, 2, 3)) / n
    xc = (x - mx.view(-1, 1, 1, 1)) * mask
    yc = (y - my.view(-1, 1, 1, 1)) * mask
    vx = (xc ** 2).sum((1, 2, 3))
    vy = (yc ** 2).sum((1, 2, 3))
    cov = (xc * yc).sum((1, 2, 3))
    ncc = cov / (torch.sqrt(vx * vy) + eps)
    return -ncc  # negative so lower = better


def _run_model_all_stack(
    stacks_svort: List[torch.Tensor],
    transforms_svort: List[RigidTransform],
    model: SVoRTv2,
    data_svort: Dict[str, Any],
    device: torch.device,
) -> Tuple[List[RigidTransform], torch.Tensor]:
    """Run SVoRTv2 on all stacks at once (NeSVoR ``run_model_all_stack``)."""

    positions = data_svort["positions"]

    with torch.no_grad():
        trans_out, volumes, _, _ = model(data_svort)
    t_out = trans_out[-1]
    volume = volumes[-1]

    # Split per-stack
    transforms_out = []
    for i in range(len(stacks_svort)):
        idx = positions[:, -1] == i
        transforms_out.append(t_out[idx])

    return transforms_out, volume


def _simulate_slices(
    transforms: RigidTransform,
    volume: torch.Tensor,
    psf: torch.Tensor,
    slice_shape: Tuple[int, int],
    res_s: float,
    res_r: float,
) -> torch.Tensor:
    """Forward-simulate slices from a volume."""
    mat = mat_update_resolution(transforms.matrix(), 1, res_r)
    return slice_acquisition(
        mat, volume, None, None, psf, slice_shape, res_s / res_r, False, False,
    )


def _correct_svort(
    stacks_out: List[RigidTransform],
    stacks_in: List[RigidTransform],
    stacks_data: List[torch.Tensor],
    volume: torch.Tensor,
    psf: torch.Tensor,
    slice_shape: Tuple[int, int],
    res_s: float,
    res_r: float,
) -> List[RigidTransform]:
    """Per-slice: choose SVoRT prediction or stack-corrected, by NCC.
    """
    stacks_corrected: List[RigidTransform] = []
    for j in range(len(stacks_out)):
        diff_mean, _ = _get_transform_diff_mean(stacks_out[j], stacks_in[j])
        stacks_corrected.append(diff_mean.compose(stacks_in[j]))

    # Evaluate NCC for both candidates
    all_svort = RigidTransform.cat(stacks_out)
    all_stack = RigidTransform.cat(stacks_corrected)
    all_slices = torch.cat(stacks_data, dim=0)
    masks = (all_slices > 0).float()

    sim_svort = _simulate_slices(all_svort, volume, psf, slice_shape, res_s, res_r)
    sim_stack = _simulate_slices(all_stack, volume, psf, slice_shape, res_s, res_r)
    ncc_svort = _ncc_loss(sim_svort, all_slices, masks)
    ncc_stack = _ncc_loss(sim_stack, all_slices, masks)

    # Pick per-slice winner
    idx = 0
    result: List[RigidTransform] = []
    for j in range(len(stacks_out)):
        ns = len(stacks_out[j])
        choose_svort = (ncc_svort[idx: idx + ns] <= ncc_stack[idx: idx + ns])
        t = torch.where(
            choose_svort.view(-1, 1, 1),
            stacks_out[j].matrix(),
            stacks_corrected[j].matrix(),
        )
        result.append(RigidTransform(t))
        idx += ns

    return result


def _get_transforms_full(
    stacks_out: List[RigidTransform],
    stacks_in: List[RigidTransform],
    stacks_full: List[RigidTransform],
    crop_idx: List[torch.Tensor],
) -> Tuple[List[RigidTransform], List[RigidTransform]]:
    """Map SVoRT per-slice transforms back to the full (uncropped) stacks.

    Returns:
        svort_full: per-slice SVoRT transforms on the full stack
        stack_full: stack-level (rigid) transforms on the full stack
    """
    svort_full: List[RigidTransform] = []
    stack_full: List[RigidTransform] = []

    for j in range(len(stacks_in)):
        diff_mean, diff = _get_transform_diff_mean(
            stacks_out[j], stacks_in[j],
        )
        # Stack-level: apply mean diff to the full (uncropped) transform
        t_stack = diff_mean.compose(stacks_full[j])
        stack_full.append(t_stack)

        # Per-slice: SVoRT diff applied where available, stack-level elsewhere
        t_svort_mat = t_stack.matrix().clone()
        t_svort_mat[crop_idx[j]] = diff.compose(
            stacks_full[j][crop_idx[j]]
        ).matrix()
        svort_full.append(RigidTransform(t_svort_mat))

    return svort_full, stack_full


def run_svort(
    stack_paths: List[str],
    slice_thickness: float,
    device: torch.device = torch.device("cuda:0"),
) -> Dict[str, Any]:
    """End-to-end: load stacks → SVoRT → post-process → return data dict.

    Returns a dict with:
        ``transforms_svort_full``: list of per-stack RigidTransform in
            original (resampled, uncropped) coordinate space — use these
            as initial transforms for Gaussian reconstruction.
        ``transforms_stack_full``: list of per-stack RigidTransform using
            only the stack-level correction (less accurate per-slice but
            more robust).
        ``stacks_ori``: list of resampled (1 mm) stack tensors [D, 1, H, W]
            in the original (uncropped) geometry.
        ``volume_svort``: initial volume produced by SVoRTv2 [1, 1, D, H, W].
        ``res_s``, ``res_r``, ``slice_thickness``, ``psf_rec``: parameters
            for downstream reconstruction.
    """
    # 1. Load and preprocess
    loaded = load_stacks(stack_paths, slice_thickness, device)
    data_svort = loaded["data_svort"]
    recovery = loaded["recovery"]

    # move to device
    for k in data_svort:
        if torch.is_tensor(data_svort[k]):
            data_svort[k] = data_svort[k].to(device)

    # 2. Load pretrained SVoRTv2
    model = _load_svortv2(device)

    # 3. Run SVoRT
    transforms_out, volume = _run_model_all_stack(
        recovery["stacks_svort"],
        recovery["transforms_svort"],
        model,
        data_svort,
        device,
    )

    # 4. Stack-level correction
    corrected = _correct_svort(
        stacks_out=transforms_out,
        stacks_in=recovery["transforms_svort"],
        stacks_data=recovery["stacks_svort"],
        volume=volume,
        psf=recovery["psf_rec"],
        slice_shape=data_svort["slice_shape"],
        res_s=recovery["res_s"],
        res_r=recovery["res_r"],
    )

    # 5. Propagate to full stacks
    svort_full, stack_full = _get_transforms_full(
        stacks_out=corrected,
        stacks_in=recovery["transforms_svort"],
        stacks_full=recovery["transforms_full"],
        crop_idx=recovery["crop_idx"],
    )

    return {
        "transforms_svort_full": svort_full,
        "transforms_stack_full": stack_full,
        "stacks_ori": recovery["stacks_ori"],
        "transforms_ori": recovery["transforms_ori"],
        "volume_svort": volume,
        "res_s": recovery["res_s"],
        "res_r": recovery["res_r"],
        "slice_thickness": recovery["slice_thickness"],
        "psf_rec": recovery["psf_rec"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run pretrained SVoRT registration on MRI stacks")
    parser.add_argument("--stacks", nargs="+", required=True, help="Paths to NIfTI stack files")
    parser.add_argument("--slice_thickness", type=float, required=True, help="Slice thickness / gap (mm)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    opts = parser.parse_args()

    device = torch.device(opts.device)
    os.makedirs(opts.output, exist_ok=True)

    result = run_svort(opts.stacks, opts.slice_thickness, device)

    # Save the initial SVoRT volume
    vol = result["volume_svort"]
    save_img(vol[0, 0].detach().cpu().numpy(), os.path.join(opts.output, "volume_svort.nii.gz"))

    # Save transforms
    torch.save(
        {
            "svort_full": [t.matrix().cpu() for t in result["transforms_svort_full"]],
            "stack_full": [t.matrix().cpu() for t in result["transforms_stack_full"]],
        },
        os.path.join(opts.output, "transforms_svort.pt"),
    )

    print(f"SVoRT registration done. Results saved to {opts.output}")


if __name__ == "__main__":
    main()
