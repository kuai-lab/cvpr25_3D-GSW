import torch
import numpy as np

# load scene, dataset

from glob import glob
import torch
from scene import Scene, GaussianModel

import numpy as np
from tqdm import tqdm
from gaussian_renderer import render
#from gaussian_renderer import GaussianModel
from numpy.fft import fft2, fftshift
from pytorch_wavelets import DWTInverse, DWTForward


def high_frequency_strength(patch):
    # Compute the 2D FFT of the patch
    f = np.fft.fft2(patch)
    # Center the zero-frequency component
    fshift = np.fft.fftshift(f)
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fshift)
    
    # Define high-frequency region (corners of the array)
    # For simplicity, consider a square around the corners to be the high-frequency region
    # Adjust the size of the region as needed
    size = patch.shape[0]
    corner_size = size // 4  # Example: consider outer 25% of the array as high-frequency region
    high_freq_region = np.r_[
        magnitude_spectrum[:corner_size, :corner_size].flat,
        magnitude_spectrum[-corner_size:, :corner_size].flat,
        magnitude_spectrum[:corner_size, -corner_size:].flat,
        magnitude_spectrum[-corner_size:, -corner_size:].flat,
    ]
    
    # Calculate the strength of the high-frequency signal (you can also use other metrics)
    
    hor_weight = np.linspace(-1, 1, f.shape[1]).reshape(1, -1, 1) ** 2
    ver_weight = np.linspace(-1, 1, f.shape[0]).reshape(-1, 1, 1) ** 2
    
    f_weighted = np.abs(fshift) * hor_weight * ver_weight
    return f_weighted.sum() / f_weighted.size


def patchify_and_get_fdomain(image, patch_size):
    frequency_patches = []
    high_frequency_score_list = []

    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            # Extract the patch
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]

            # Check if the patch size is as expected (it might not be at the edges)
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                # Compute the 2D Fourier Transform of the patch
                fft_patch = fft2(patch)
                # Shift the zero frequency component to the center
                fft_patch_shifted = fftshift(fft_patch)

                # Save the transformed patch
                frequency_patches.append(fft_patch_shifted)

                strength = high_frequency_strength(patch)
                high_frequency_score_list.append(strength)
    return frequency_patches, high_frequency_score_list


def find_guassian_within_patches(
    patch_size, patch_rcw_list, means2d, visibility_filter
) :
    ridx_arr, cidx_arr, width_arr = torch.tensor(patch_rcw_list).T 
    mean2d_repeat = means2d.transpose(0, 1).repeat(len(ridx_arr), 1, 1)
    return torch.any(
        (
            mean2d_repeat[:, 0, :] > cidx_arr.reshape(-1, 1) * patch_size[0]
        ) & (
            mean2d_repeat[:, 0, :] < (cidx_arr + width_arr).reshape(-1, 1) * patch_size[0]
        ) & (
            mean2d_repeat[:, 1, :] > ridx_arr.reshape(-1, 1) * patch_size[1]
        ) & (
            mean2d_repeat[:, 1, :] < (ridx_arr + 1).reshape(-1, 1) * patch_size[1]
        ),
        dim=0
    )

def get_frequency_map(image, patch_size) :
    # get pixel sum within patches
    patch_sum = torch.sum(
        torch.nn.functional.avg_pool2d(image, patch_size, stride=patch_size),
        dim=0
    )
    return patch_sum

def get_top_k_percent_mask(fmap, k) :
    unique_value, unique_count = torch.unique(fmap, return_counts=True)
    unique_count_accum = torch.cumsum(unique_count, dim=0)
    
    thresh_idx = (unique_count_accum < (1 - k) * unique_count_accum[-1]).sum()
    thresh_val = unique_value[thresh_idx]
    
    return fmap > thresh_val  

def densify_high_frequency(scene, gaussians, dataset, opt, pipe):
    patch_size = (dataset.patch, dataset.patch)
    high_freq_proportion = dataset.proportion
    train_camera_list = scene.getTrainCameras()
    per_cam_data_dict = {}
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg_color = torch.rand((3), device="cuda") if opt.random_background else background

    override_c_mask_list = []
    high_freq_override_c_mask_list = []
    high_freq_gaussian_mask_list = []

    with tqdm(total=len(train_camera_list)) as pbar:
        for cam_idx, camera in enumerate(train_camera_list) :
            # find blur by comparing frequency patches
            image_gt_torch  = camera.original_image.detach()
            image_gt_np     = (image_gt_torch.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)            
            render_pkg  = render(
                camera, gaussians, pipe, bg_color=bg_color,
            )
            image_rendered_torch = render_pkg["render"].detach().cpu()
            image_rendered_np = image_rendered_torch.clamp(0, 1).permute(1,2,0).numpy()
            
            override_c = torch.zeros((gaussians._xyz.shape[0], 3)).cuda().requires_grad_(True)
            render_pkg_new = render(
                camera, gaussians, pipe, bg_color=bg_color,
                override_color = override_c
            )

            aux_loss = torch.mean((torch.abs(render_pkg_new['render'] - image_gt_torch).detach()) * render_pkg_new['render']) # == 0
            aux_loss.backward()

            override_c_mask = override_c.grad <= 1e-8
            override_c_mask = override_c_mask.sum(dim=1) > 0
            override_c_mask_list.append(override_c_mask)

            high_freq_override_c_mask = override_c.grad <= 8e-8
            high_freq_override_c_mask = high_freq_override_c_mask.sum(dim=1) > 0
            high_freq_override_c_mask_list.append(high_freq_override_c_mask)

            freq_patches_gt, high_freq_score_gt = patchify_and_get_fdomain(image_gt_np, patch_size)
            freq_feature_map_gt = np.array(high_freq_score_gt).reshape(
                int(image_gt_np.shape[0]/patch_size[0]), int(image_gt_np.shape[1]/patch_size[1])
            )
            freq_patches_rendered, high_freq_score_rendered = patchify_and_get_fdomain(image_rendered_np, patch_size)
            freq_feature_map_render = np.array(high_freq_score_rendered).reshape(
                int(image_rendered_np.shape[0]/patch_size[0]), int(image_rendered_np.shape[1]/patch_size[1])
            )
            
            unique_freq_value, unique_freq_count = np.unique(freq_feature_map_render, return_counts=True)
            unique_freq_count_accum = np.cumsum(unique_freq_count)

            thresh_idx = (unique_freq_count_accum < (1-high_freq_proportion) * unique_freq_count_accum[-1]).sum()
            high_freq_thresh = unique_freq_value[thresh_idx]
            
            high_freq_patch_ridx_list, high_freq_patch_cidx_list = np.where(freq_feature_map_render > high_freq_thresh)
            patch_rcw_list = [[high_freq_patch_ridx_list[0], high_freq_patch_cidx_list[0], 1]]
            for ridx, cidx in zip(high_freq_patch_ridx_list, high_freq_patch_cidx_list):
                b = patch_rcw_list[-1][0]
                r = patch_rcw_list[-1][1] + patch_rcw_list[-1][2]
                if ridx == b and cidx == r:
                    patch_rcw_list[-1][2] += 1
                else:
                    patch_rcw_list.append([ridx, cidx, 1])
            high_freq_gaussian_mask = find_guassian_within_patches(
                patch_size, patch_rcw_list, render_pkg["means2D"], None  
            )
            high_freq_gaussian_mask_list.append(high_freq_gaussian_mask)

            pbar.update(1)

    override_c_stacked_tensors = torch.stack(override_c_mask_list)
    contribution_mask = torch.all(override_c_stacked_tensors, dim=0)

    high_freq_stacked_tensors = torch.stack(high_freq_gaussian_mask_list)
    high_frequency_mask = torch.any(high_freq_stacked_tensors, dim=0).to("cuda")
 
    gaussians.prune_points_for_FGD(contribution_mask.to("cuda"))

    indices_to_remove = torch.nonzero(contribution_mask, as_tuple=False).squeeze().to("cuda")
    mask = torch.zeros(contribution_mask.size(0), dtype=torch.bool, device='cuda') 
    mask[indices_to_remove] = True
    mask = ~mask
    high_freq_override_c_stacked_tensors = torch.stack(high_freq_override_c_mask_list)
    high_freq_contribution_mask = torch.all(high_freq_override_c_stacked_tensors, dim=0)

    high_freq_contribution_mask_after_fgd = high_freq_contribution_mask[mask]
    high_frequency_mask_after_fgd = high_frequency_mask[mask]

    combined_mask_after_fgd = torch.stack([high_freq_contribution_mask_after_fgd, high_frequency_mask_after_fgd])


    combined_mask_after_fgd = torch.all(combined_mask_after_fgd, dim=0)

    gaussians.densify_and_split_by_frequency(combined_mask_after_fgd.to("cuda"))

    # scene.frequency_save(30001)

    return scene, gaussians