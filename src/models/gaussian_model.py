import torch
from gs_utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from gs_utils.general_utils import strip_symmetric, build_scaling_rotation, build_rotation
import torch.nn.functional as F
import numpy as np
from gs_utils.Compute_intensity import compute_intensity
import os


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)

            return actual_covariance

        self._scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.intensity_activation = torch.sigmoid
        self.inverse_intensity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._intensity = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._intensity,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._intensity,
            opt_dict,
            self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self._scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_intensity(self):
        return self.intensity_activation(self._intensity)

    @property
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    @property
    def get_inv_covariance(self, scaling_modifier=1):
        scaling = self.get_scaling
        rotation = self.get_rotation
        scaling_inv_squared = 1.0 / (scaling * scaling_modifier) ** 2
        S_inv_squared = torch.diag_embed(scaling_inv_squared)
        R = build_rotation(rotation)
        R_transpose = R.transpose(1, 2)
        covariance_inv = torch.matmul(R, torch.matmul(S_inv_squared, R_transpose))
        return covariance_inv

    def random_initialize(self, seg_low_reso, image_low_reso, ini_intensity=1.0, ini_sigma=0.01, spatial_lr_scale=1,
                          fg_num_samples=10000, bg_num_samples=50
                          ):
        self.spatial_lr_scale = spatial_lr_scale
        bs, D, H, W = seg_low_reso.shape

        mask = seg_low_reso.detach().cpu().numpy()[0]
        coords = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W)), dim=-1).reshape(-1,
                                                                                                                3).cuda()
        fg_coords = np.argwhere(mask == 1)
        fg_indices = np.random.choice(len(fg_coords), fg_num_samples, replace=False)
        fg_coords = coords[fg_indices]

        bg_coords = np.argwhere(mask == 0)
        bg_indices = np.random.choice(len(bg_coords), bg_num_samples, replace=False)
        bg_coords = coords[bg_indices]

        coords = torch.cat((fg_coords, bg_coords), dim=0)
        indices = np.concatenate((fg_indices, bg_indices), axis=0)
        image_array = image_low_reso.reshape(-1)[indices]

        grid = torch.zeros((D, H, W), dtype=torch.int32, device="cuda")
        # Increase the count in the grid at the location of each sampled point
        indices_3d = coords.long()
        grid[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]] += 1
        # Apply a 3D convolution to count neighbours
        kernel_size = 5  # Define the size of the neighbourhood
        padding = kernel_size // 2
        conv_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device="cuda", dtype=torch.float32)
        neighbours_count = F.conv3d(grid.unsqueeze(0).unsqueeze(0).float(), conv_kernel, padding=padding).squeeze()
        # Retrieve the number of neighbours for each sampled point
        num_neighbours = neighbours_count[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]]
        # Adjust scaling based on the number of neighbours
        scaling = ini_sigma / num_neighbours.float()

        intensities = ini_intensity * image_array + 1e-4
        coords = coords.float()
        coords = coords / torch.tensor([D, H, W], dtype=torch.float, device="cuda")

        intensities = inverse_sigmoid(intensities).unsqueeze(1)
        scaling = torch.log(scaling).unsqueeze(1).repeat(1, 3)
        rotation = torch.zeros((fg_num_samples + bg_num_samples, 4), device="cuda")
        rotation[:, 0] = 1

        self._xyz = nn.Parameter(coords.requires_grad_(True))
        self._intensity = nn.Parameter(intensities.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._intensity], 'lr': training_args.intensity_lr, "name": "intensity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('intensity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_intensities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "intensity": new_intensities,
             "scaling": new_scaling,
             "rotation": new_rotation
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        new_xyz = samples + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_intensities = self._intensity[selected_pts_mask].repeat(N, 1)
        new_scale = self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scale = torch.clamp(new_scale, min=2e-4)
        new_scaling = self.scaling_inverse_activation(new_scale)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_intensities, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        print("Splitting {} points".format(selected_pts_mask.sum()))

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]

        original_intensity = self.intensity_activation(self._intensity[selected_pts_mask]) / 2
        original_intensity = self.inverse_intensity_activation(original_intensity)
        new_intensity = original_intensity
        self._intensity[selected_pts_mask] = original_intensity

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation)

        print("Cloning {} points".format(selected_pts_mask.sum()))

    def densify(self, max_grad, sigma_extent):
        """Densify Gaussians by cloning and splitting based on gradients."""
        grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)
        self.densify_and_clone(grads, max_grad, sigma_extent)
        self.densify_and_split(grads, max_grad, sigma_extent)

    def prune(self, min_intensity, min_scale=0.0002):
        """Prune Gaussians with low intensity, small scale."""
        prune_mask = (
                (self.get_intensity < min_intensity).squeeze()
                | (torch.min(self.get_scaling, dim=1).values < min_scale)
        )

        if prune_mask.any():
            print("Pruning {} points".format(prune_mask.sum()))
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def grid_sample(self, grid):
        # grid: [batchsize, z, x, y, 3]
        grid_shape = grid.shape
        # intensity_grid = torch.empty(grid_shape, device="cuda")
        # expand dimensions for broadcasting
        grid_expanded = grid.unsqueeze(-2)  # [batchsize, z, x, y, 1, 3]
        intensity_grid = self.compute_intensity(self._xyz, grid_expanded, self.get_intensity, self.get_inv_covariance,
                                                self.get_scaling)

        return intensity_grid

    def compute_intensity(self, gaussian_centers, grid_point, intensity, inv_covariance, scaling):
        # grid_point: [1, z, x, y, 1, 3]
        z, x, y = grid_point.shape[1:4]
        # Initialize intensity_grid outside the loop
        intensity_grid = torch.zeros(1, z, x, y, 1, device='cuda', requires_grad=True)

        intensity_grid = compute_intensity(
            gaussian_centers.contiguous(),
            grid_point.contiguous(),
            intensity.contiguous(),
            inv_covariance.contiguous(),
            scaling.contiguous(),
            intensity_grid.contiguous(),
        )

        return intensity_grid

    def save_model(self, path, iteration):
        scaling_path = os.path.join(path, 'scaling_' + str(iteration) + '.pt')
        rotation_path = os.path.join(path, 'rotation_' + str(iteration) + '.pt')
        pixel_coords_path = os.path.join(path, 'pixel_coords_' + str(iteration) + '.pt')
        intensities_path = os.path.join(path, 'intensities_' + str(iteration) + '.pt')
        torch.save(self._scaling, scaling_path)
        torch.save(self._rotation, rotation_path)
        torch.save(self._xyz, pixel_coords_path)
        torch.save(self._intensity, intensities_path)
        print(f"Saved torch parameters output to {path}")

    def load_model(self, path, iteration):
        scaling_path = os.path.join(path, 'scaling_' + str(iteration) + '.pt')
        rotation_path = os.path.join(path, 'rotation_' + str(iteration) + '.pt')
        pixel_coords_path = os.path.join(path, 'pixel_coords_' + str(iteration) + '.pt')
        intensities_path = os.path.join(path, 'intensities_' + str(iteration) + '.pt')
        self._scaling = torch.load(scaling_path, map_location='cuda', weights_only=True)
        self._rotation = torch.load(rotation_path, map_location='cuda', weights_only=True)
        self._xyz = torch.load(pixel_coords_path, map_location='cuda', weights_only=True)
        self._intensity = torch.load(intensities_path, map_location='cuda', weights_only=True)
        print(f"Loaded torch parameters output from {path}")

    @property
    def motion_params(self):
        return self._motion_params