import os

import nibabel as nib
import numpy as np
import torch
import yaml


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, "r") as f:
            return yaml.load(f, Loader)


Loader.add_constructor("!include", Loader.include)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=Loader)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def save_img(I_img, savename, header=None, affine=None):
    if affine is None:
        affine = np.eye(4)
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)
    nib.save(new_img, savename)


def create_grid_3d(d, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, steps=d),
        torch.linspace(0, 1, steps=h),
        torch.linspace(0, 1, steps=w),
    )
    return torch.stack([grid_z, grid_y, grid_x], dim=-1)


def tv_regularization(image: torch.Tensor) -> torch.Tensor:
    img = image.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
    depth_tv = torch.mean(torch.abs(img[:, :, 1:, :, :] - img[:, :, :-1, :, :]))
    height_tv = torch.mean(torch.abs(img[:, :, :, 1:, :] - img[:, :, :, :-1, :]))
    width_tv = torch.mean(torch.abs(img[:, :, :, :, 1:] - img[:, :, :, :, :-1]))
    return depth_tv + height_tv + width_tv
