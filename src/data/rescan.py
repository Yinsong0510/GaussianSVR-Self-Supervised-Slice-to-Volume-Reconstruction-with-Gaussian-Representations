import torch
import numpy as np
from transform import RigidTransform, mat_update_resolution
from slice_acquisition import slice_acquisition


def get_PSF(r_max=None, res_ratio=(1, 1, 3), threshold=1e-3, device=torch.device("cpu")):
    sigma_x = 1.2 * res_ratio[0] / 2.3548
    sigma_y = 1.2 * res_ratio[1] / 2.3548
    sigma_z = res_ratio[2] / 2.3548
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    psf = torch.exp(
        -0.5 * (grid_x ** 2 / sigma_x ** 2
                 + grid_y ** 2 / sigma_y ** 2
                 + grid_z ** 2 / sigma_z ** 2)
    )
    psf[psf.abs() < threshold] = 0
    rx = torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item()
    ry = torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item()
    rz = torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item()
    psf = psf[
        rz: 2 * r_max + 1 - rz,
        ry: 2 * r_max + 1 - ry,
        rx: 2 * r_max + 1 - rx,
    ].contiguous()
    psf = psf / psf.sum()
    return psf


class ReScanner:
    """Re-acquire slices from a volume using known transforms and PSF.

    Uses fixed resolution parameters (not randomly sampled) so that the
    re-acquired slices are directly comparable with the input stacks.
    """

    def __init__(self, slice_size=None):
        self.slice_size = slice_size

    def scan(self, volume, transform, resolution, resolution_slice, slice_thickness):
        """Forward model: volume + transforms -> slices.

        Args:
            volume: [1, 1, D, H, W] reconstructed volume.
            transform: [N, 6] axis-angle rigid transform parameters.
            resolution: volume voxel size (mm).
            resolution_slice: slice in-plane pixel spacing (mm).
            slice_thickness: slice thickness (mm).

        Returns:
            slices: [N, 1, ss, ss] re-acquired 2D slices.
        """
        device = volume.device
        res = resolution
        res_s = resolution_slice
        s_thick = slice_thickness
        ss = self.slice_size

        psf_acq = get_PSF(
            res_ratio=(res_s / res, res_s / res, s_thick / res), device=device
        )

        mat = mat_update_resolution(RigidTransform(transform).matrix(), 1, res)
        slices = slice_acquisition(
            mat, volume, None, None, psf_acq, (ss, ss), res_s / res,
            False, False,
        )

        return slices
