import mmcv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet.models.builder import build_backbone
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.policynet_models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin import SPConvVoxelization
from projects.mmdet3d_plugin import DifferentiableSPConvVoxelization
import sys
import numpy as np
import pickle
import sys
import numba
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_bilateral
from skimage.morphology import remove_small_objects, remove_small_holes
from collections import deque
from skimage import filters, morphology


import math
import torch
import numpy as np
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_bilateral
from collections import deque

# Your 10 nuScenes class names:
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# Set of small-object label indices
SMALL_LABELS = {
    class_names.index(name)
    for name in ('motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
}
def geodesic_distance(seeds: np.ndarray, support: np.ndarray) -> np.ndarray:
    H, W = support.shape
    D = np.full((H, W), np.inf, dtype=np.float32)
    q = deque()
    for y, x in zip(*np.nonzero(seeds)):
        D[y, x] = 0.0
        q.append((y, x))
    nbrs = [(-1,0), (1,0), (0,-1), (0,1)]
    while q:
        y, x = q.popleft()
        d0 = D[y, x]
        for dy, dx in nbrs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and support[ny, nx] and D[ny, nx] > d0 + 1:
                D[ny, nx] = d0 + 1
                q.append((ny, nx))
    return D

def smart_geodesic_mask_soft(
    mask_probs: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.1,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Continuous (soft) dynamic-threshold geodesic mask.
    Non-support pixels are explicitly set to 0 (no NaNs).
    """
    # 1) extract & smooth class-1 prob
    p = mask_probs[0,1].cpu().numpy()
    p_s = denoise_bilateral(
        p,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        multichannel=False
    )

    # 2) thresholds & support
    T_high = threshold_otsu(p_s)
    seeds   = p_s >= T_high
    support = p_s >= alpha * T_high

    # 3) compute geodesic distance & threshold map
    D     = geodesic_distance(seeds, support)
    T_map = T_high - lam * D

    # 4) linear ramp: only inside support, zeros elsewhere
    M_cont = np.zeros_like(p_s, dtype=np.float32)
    denom = 1.0 - T_map
    # compute ramp for valid pixels
    valid = support
    M_cont[valid] = (p_s[valid] - T_map[valid]) / (denom[valid] + eps)

    # replace any NaN/inf with 0
    M_cont = np.nan_to_num(M_cont, nan=0.0, posinf=1.0, neginf=0.0)

    # clip into [0,1]
    M_cont = np.clip(M_cont, 0.0, 1.0)

    # 5) back to tensor and two-channel format
    M_t = torch.from_numpy(M_cont).to(mask_probs.device).unsqueeze(0).unsqueeze(0)
    out = torch.zeros_like(mask_probs)
    out[0,1] = M_t[0,0]
    out[0,0] = 1.0 - M_t[0,0]
    return out
def smart_geodesic_mask_soft_batch(
    mask_probs: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.1,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch version of continuous geodesic mask.
    Input:  mask_probs [B,2,H,W]
    Output: [B,2,H,W], channel1 ∈ [0,1], channel0 = 1-channel1
    """
    B, C, H, W = mask_probs.shape
    device = mask_probs.device
    out = torch.zeros_like(mask_probs)

    for b in range(B):
        # 1) smooth class-1 probability
        p = mask_probs[b,1].detach().cpu().numpy()
        p_s = denoise_bilateral(
            p,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            multichannel=False
        )

        # 2) thresholds & support
        T_high = threshold_otsu(p_s)
        seeds   = p_s >= T_high
        support = p_s >= alpha * T_high

        # 3) distance & threshold map
        D     = geodesic_distance(seeds, support)
        T_map = T_high - lam * D

        # 4) linear ramp within support, zeros outside
        M_cont = np.zeros_like(p_s, dtype=np.float32)
        valid = support
        denom = 1.0 - T_map
        M_cont[valid] = (p_s[valid] - T_map[valid]) / (denom[valid] + eps)
        # Replace NaN/Inf, clip
        M_cont = np.nan_to_num(M_cont, nan=0.0, posinf=1.0, neginf=0.0)
        M_cont = np.clip(M_cont, 0.0, 1.0)

        # 5) write back to tensor
        M_t = torch.from_numpy(M_cont).to(device=device, dtype=mask_probs.dtype)
        out[b,1] = M_t
        out[b,0] = 1.0 - M_t

    return out
def fill_class1_mask_prob(mask: torch.Tensor,
                          kernel_size: int = 31,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Takes a soft mask [B,2,H,W] and returns a locally-smoothed soft mask
    of the same shape, then rescales so that the max in channel 1 is 1.

    Args:
      mask:        Tensor [B,2,H,W], soft probabilities in [0,1]
      kernel_size: smoothing window size
      eps:         small constant to avoid div0

    Returns:
      Tensor [B,2,H,W] where channel1 has been box-filtered and then
      divided by its per-sample max (clamped to [0,1]), and channel0=1-channel1.
    """
    # 1) extract class-1 probabilities [B,1,H,W]
    class1 = mask[:, 1:2, :, :]

    # 2) build normalized box filter
    k = kernel_size
    kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype) / (k*k)
    pad = k // 2

    # 3) convolve → local average probability
    prob1 = F.conv2d(class1, kernel, padding=pad)  # [B,1,H,W]

    # 4) rescale so that max per-sample = 1
    #    compute per-sample max over spatial dims
    max_vals = prob1.amax(dim=[2,3], keepdim=True)  # [B,1,1,1]
    prob1 = prob1 / (max_vals + eps)

    # 5) clamp tiny numerical drift
    prob1 = prob1.clamp(0.0, 1.0)

    # 6) re-assemble two channels
    prob0 = 1.0 - prob1
    return torch.cat([prob0, prob1], dim=1)

def batch_soft_bernoulli_mask(
    mask_probs: torch.Tensor,
    offset: float = 0.0,
    quantize_levels: list = None
) -> torch.Tensor:
    """
    Applies advisor’s "soft-masking": uses each block’s probability
    as a sampling rate to randomly keep/drop pixels.

    Args:
        mask_probs:      [B,2,H,W] tensor of block-wise soft probabilities.
                         channel 1 is the keep-confidence p ∈ [0,1].
        offset:          minimum sampling probability (broadcast to all p).
        quantize_levels: optional list of floats (e.g. [0.0625,0.125,0.25,0.5,1.0])
                         to which p is snapped before sampling.

    Returns:
        mask: [B,2,H,W] binary mask tensor (0 or 1), same dtype & device.
    """
    B, C, H, W = mask_probs.shape
    device = mask_probs.device
    dtype = mask_probs.dtype

    # 1) extract and clamp probabilities
    p = mask_probs[:, 1, :, :].clamp(min=offset)  # [B,H,W]

    # 2) optional quantization
    if quantize_levels is not None:
        levels = torch.tensor(quantize_levels, device=device, dtype=dtype)  # [L]
        # compute |p - level| and pick nearest level
        # p.unsqueeze(-1) -> [B,H,W,1], broadcast against [L]
        idx = (p.unsqueeze(-1) - levels).abs().argmin(dim=-1)  # [B,H,W]
        p = levels[idx]  # [B,H,W]

    # 3) sample uniform random and build keep-mask
    rand = torch.rand((B, H, W), device=device, dtype=dtype)
    keep = (rand < p).to(dtype).unsqueeze(1)  # [B,1,H,W]

    # 4) assemble two-channel binary mask
    mask = torch.cat([1.0 - keep, keep], dim=1)  # [B,2,H,W]
    return mask


def hysteresis_mask_single(
    p: np.ndarray,
    low_frac: float = 0.5,
    min_size: int = 64,
    area_threshold: int = 64
) -> np.ndarray:
    """
    Single-image hysteresis thresholding on a probability map p [H,W].
    Returns uint8 binary mask [H,W].
    """
    # high and low thresholds
    th_high = filters.threshold_otsu(p)
    th_low  = th_high * low_frac

    # seeds and support
    seeds       = (p >= th_high).astype(np.uint8)
    mask_region = (p >= th_low).astype(np.uint8)

    # grow seeds within support
    grown = morphology.reconstruction(seed=seeds, mask=mask_region, method='dilation')

    # cleanup
    clean = morphology.remove_small_objects(grown.astype(bool), min_size=min_size)
    clean = morphology.remove_small_holes(clean, area_threshold=area_threshold)

    return clean.astype(np.uint8)

def batch_hysteresis_mask(
    mask_probs: torch.Tensor,
    low_frac: float = 0.5,
    min_size: int = 64,
    area_threshold: int = 64
) -> torch.Tensor:
    """
    Args:
      mask_probs:     [B,2,H,W] float tensor of soft probabilities.
      low_frac:       fraction of high threshold for low threshold.
      min_size:       min connected-component size.
      area_threshold: min hole size to fill.

    Returns:
      [B,2,H,W] tensor of 0/1 masks, same dtype & device.
    """
    B, C, H, W = mask_probs.shape
    device = mask_probs.device
    dtype  = mask_probs.dtype

    output = torch.zeros((B, C, H, W), dtype=dtype, device=device)

    for b in range(B):
        # get class-1 map as numpy
        p = mask_probs[b, 1].detach().cpu().numpy()
        # single-image hysteresis
        M = hysteresis_mask_single(p, low_frac, min_size, area_threshold)
        # back to tensor
        M_t = torch.from_numpy(M).to(device=device, dtype=dtype)
        # fill channels
        output[b, 1] = M_t
        output[b, 0] = 1.0 - M_t

    return output

def geodesic_distance(seeds: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Compute grid-graph geodesic distance within a support region.
    seeds:   bool array [H,W], True where p >= T_high
    support: bool array [H,W], True where p >= T_low
    Returns: float array [H,W], distance (inf outside support)
    """
    H, W = support.shape
    D = np.full((H, W), np.inf, dtype=np.float32)
    q = deque()
    ys, xs = np.nonzero(seeds)
    for y, x in zip(ys, xs):
        D[y, x] = 0.0
        q.append((y, x))
    nbrs = [(-1,0), (1,0), (0,-1), (0,1)]
    while q:
        y, x = q.popleft()
        d0 = D[y, x]
        for dy, dx in nbrs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and support[ny, nx]:
                if D[ny, nx] > d0 + 1:
                    D[ny, nx] = d0 + 1
                    q.append((ny, nx))
    return D

def smart_geodesic_mask_single(
    p: np.ndarray,
    alpha: float = 0.5,
    lam: float = 0.1,
    min_area: int = 64,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0
) -> np.ndarray:
    """
    Single-image dynamic-threshold geodesic region-growing.
    p:           numpy array [H,W] of soft class-1 probabilities
    Returns:     uint8 array [H,W] of 0/1 mask
    """
    # 1) bilateral smoothing
    p_s = denoise_bilateral(
        p,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        multichannel=False
    )
    # 2) thresholds
    T_high = threshold_otsu(p_s)
    T_low  = alpha * T_high
    # 3) seeds & support
    seeds   = p_s >= T_high
    support = p_s >= T_low
    # 4) geodesic distance
    D = geodesic_distance(seeds, support)
    # 5) dynamic threshold map
    T_map = T_high - lam * D
    # 6) initial mask
    M = (p_s >= T_map) & support
    # 7) cleanup
    M = remove_small_objects(M, min_size=min_area)
    M = remove_small_holes(M, area_threshold=min_area)
    return M.astype(np.uint8)

def batch_smart_geodesic_mask(
    mask_probs: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.1,
    min_area: int = 64,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0
) -> torch.Tensor:
    """
    Apply dynamic-geodesic mask to each sample in [B,2,H,W] tensor.
    Returns: [B,2,H,W] binary tensor (0/1) same dtype & device.
    """
    B, _, H, W = mask_probs.shape
    device = mask_probs.device
    dtype  = mask_probs.dtype
    output = torch.zeros((B,2,H,W), device=device, dtype=dtype)
    for b in range(B):
        # extract class-1 probability map as numpy
        p = mask_probs[b,1].detach().cpu().numpy()
        # compute H×W binary mask
        M = smart_geodesic_mask_single(
            p, alpha=alpha, lam=lam,
            min_area=min_area,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial
        )
        # back to tensor
        M_t = torch.from_numpy(M).to(device=device, dtype=dtype)
        output[b,1] = M_t
        output[b,0] = 1.0 - M_t
    return output

def focal_loss(inputs, targets, alpha=[0.25, 0.75], gamma=2.0, reduction='mean'):
    # Compute cross entropy loss without reduction, shape: [N, H, W]
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    # Flatten the loss to shape [N*H*W]
    ce_loss_flat = ce_loss.view(-1)
    
    # Get the predicted probability for the true class
    pt = torch.exp(-ce_loss_flat)  # shape [N*H*W]
    
    # Flatten targets so they match ce_loss_flat
    targets_flat = targets.view(-1)
    
    # Create an alpha tensor and index using the targets
    alpha_tensor = torch.tensor(alpha, dtype=ce_loss_flat.dtype, device=ce_loss_flat.device)
    alpha_factor = alpha_tensor[targets_flat]  # shape [N*H*W]
    
    # Compute the focal loss for each sample
    focal_loss_val = alpha_factor * (1 - pt) ** gamma * ce_loss_flat
    
    # Apply the requested reduction
    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val

# Numba-based scatter function
@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array

# Generate range image
def generate_range_image(points, H=32, W=1080, min_depth=0.0, max_depth=100.0, scan_unfolding=False):
    xyz = points[:, :3]  # Extract xyz
    x, y, z = xyz[:, [0]], xyz[:, [1]], xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    d_adding=points.shape[1]
    # Masking points by depth
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)


    h_up, h_down = np.deg2rad(10), np.deg2rad(-30)
    elevation = np.arcsin(z / depth) + abs(h_down)
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)


    # horizontal grid
    azimuth = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)
    # Store row, column indices and depth information
    saving_info = {
        'depth': depth,          # Depth for each point
        'row_indices': grid_h,   # Row indices (vertical)
        'col_indices': grid_w    # Column indices (horizontal)
    }
    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W,d_adding + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points,saving_info
def find_2d_bbox_from_3d_corners(box_corners, H=32, W=1080):
    """
    box_corners: numpy array of shape [N, 8, 3] (N boxes, each with 8 corners in 3D)
    H: Height of the range image
    W: Width of the range image
    """
    all_bboxes_2d = []
    
    for i in range(box_corners.shape[0]):  # Iterate over each bounding box
        corners = box_corners[i]  # Shape [8, 3]
        
        # Project the corners using the same logic as generate_range_image
        proj_points, saving_info = generate_range_image(corners, H, W)

        # Extract the row and column indices for the corners
        row_indices = saving_info['row_indices'].squeeze()
        col_indices = saving_info['col_indices'].squeeze()

        # Calculate the 2D bounding box
        min_row, max_row =np.clip(row_indices.min() , 0, 31),  np.clip(row_indices.max(), 0, 31)
        min_col, max_col = np.clip(col_indices.min(), 0, 1079),  np.clip(col_indices.max() , 0, 1079)
        
        bbox_2d = [min_row, min_col, max_row, max_col]  # [ymin, xmin, ymax, xmax]
        all_bboxes_2d.append(bbox_2d)
    
    return np.array(all_bboxes_2d)  # Shape [N, 4]
def generate_binary_mask(all_corners, H=32, W=1080):
    """
    Generates a binary mask that includes the bounding boxes in a 2D range image map.
    
    Parameters:
    - all_corners: List or array of 2D bounding boxes in [ymin, xmin, ymax, xmax] format.
    - H: Height of the range image.
    - W: Width of the range image.
    
    Returns:
    - mask: Binary mask with shape (H, W), where 1 indicates the bounding box region.
    """
    # Initialize the binary mask
    mask = np.zeros((H, W), dtype=np.uint8)

    # Iterate through each bounding box
    for bbox in all_corners:
        ymin, xmin, ymax, xmax = bbox

        if xmax - xmin > W // 2:  # Handle wrapping case
            # Fill region from 0 to xmin
            mask[ymin:ymax, 0:xmin] = 1
            # Fill region from xmax to W
            mask[ymin:ymax, xmax:W] = 1
        else:  # Normal case
            # Fill the bounding box region
            mask[ymin:ymax, xmin:xmax] = 1

    return mask

def fill_class1_mask_same_shape(mask, kernel_size=31, threshold_ratio=0.1): # non diff for inference
    """
    Fill holes in the class 1 channel of a binary mask tensor, and output a mask of the same
    shape and type as the input.

    Args:
        mask (torch.Tensor): A tensor of shape [bs, 2, H, W] (e.g., [bs, 2, 64, 2048])
                               where channel 0 corresponds to class 0 and channel 1 to class 1.
        kernel_size (int): Size of the convolution kernel (must be odd).
        threshold_ratio (float): Ratio to decide threshold based on the kernel window sum.
                                 For example, if threshold_ratio=0.1 and kernel_size=31,
                                 then if a 31x31 window has at least 10% ones, it is set to 1.

    Returns:
        torch.Tensor: A filled mask of the same shape and type as the input mask.
    """
    # Ensure we work on the class 1 channel: shape [bs, 1, H, W]
    class1 = mask[:, 1:2, :, :]
    
    # Create a kernel of ones for convolution
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    pad = kernel_size // 2  # to maintain same spatial dimensions
    
    # Convolve the class1 channel with the kernel. This sums the values in a local neighborhood.
    conv_result = F.conv2d(class1, kernel, padding=pad)
    
    # Maximum possible sum in the window is kernel_size * kernel_size (if all values are 1).
    max_sum = kernel_size * kernel_size
    threshold = threshold_ratio * max_sum
    
    # Create the filled version: if the sum in the window is above threshold, set that pixel to 1, else 0.
    filled_class1 = (conv_result >= threshold).to(mask.dtype)
    
    # Create the output mask as a copy of the input mask
    output_mask = mask.clone()
    # Replace channel 1 with the filled version. Output_mask remains of shape [bs, 2, H, W]
    output_mask[:, 1:2, :, :] = filled_class1
    
    return output_mask
def differentiable_fill_class1_mask(mask, kernel_size=31, threshold_ratio=0.1, steepness=50): # differentiable 
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [bs, 1, H, W]
    
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    pad = kernel_size // 2
    conv_result = F.conv2d(mask[:, 1:2, :, :], kernel, padding=pad)
    
    max_sum = kernel_size * kernel_size
    threshold = threshold_ratio * max_sum
    
    # Use a sigmoid to approximate the step function:
    filled_class1 = torch.sigmoid(steepness * (conv_result - threshold))
    # Optionally, if you need a hard mask during inference, you might threshold this later.
    
    output_mask = mask.clone()
    output_mask[:, 1:2, :, :] = filled_class1
    return output_mask

@DETECTORS.register_module()
class PolicyDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(PolicyDetector, self).__init__(**kwargs)
        
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
            #self.pts_voxel_layer = DifferentiableSPConvVoxelization(**pts_voxel_cfg)
            #self.pts_voxel_layer =DifferentiableSPConvVoxelization(**pts_voxel_cfg)

    def init_weights(self):
        """Initialize model weights."""
        super(PolicyDetector, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def ste_point_selection(self, mask_logits, range_image, lidar_points, 
                        selected_class=1, tau=1.0):
        """
            Differentiable point selection using a binary mask directly.
    
            This function assumes that mask_logits comes from a mask generator that uses
            Gumbel-softmax with hard=True, so its forward pass is binary but gradients 
            are computed via the underlying soft probabilities. Instead of using a 
            non-differentiable boolean mask, we multiply the lidar points by the mask 
            values (which are nearly 0 or 1 in the forward pass). This way, the operation 
            remains differentiable.
    
            Args:
                mask_logits (Tensor): Differentiable mask of shape [B, num_classes, H, W].
                              (For example, produced by your MaskGenerator module.)
                range_image (Tensor): Tensor of shape [B, N, 3] where:
                              - range_image[b][:, 0] contains row indices,
                              - range_image[b][:, 1] contains col indices.
                lidar_points (list[Tensor]): List of length B. Each element is a tensor of shape [N, 5]
                                     with columns (x, y, z, row_idx, col_idx).
                selected_class (int): The channel index used for selection (e.g. class 1).
                tau (float): Temperature for Gumbel softmax (not used directly here, but relevant to the mask).
    
            Returns:
                list[Tensor]: A list of length B, where each element is a tensor of shape [N, 5].
                      Each tensor is the result of multiplying the lidar points by the binary mask.
                      Points not selected are zeroed out, while selected points retain their value.
                      (Gradients flow through the mask thanks to the STE.)
        """
        #print(mask_logits[0,1,:,:])
        B = mask_logits.shape[0]
        selected_points_list = []
    
        for b in range(B):
            # Extract row and column indices from the range image.
            # Here we assume range_image[b] is of shape [1, N, 3].
            rows = range_image[b][0, :, 0].long()  # shape: [N]
            cols = range_image[b][0, :, 1].long()  # shape: [N]
        
            # Use the mask for the selected class.
            # Although the forward pass gives zeros or ones, the gradient flows
            # through the underlying soft probabilities due to the STE.
            binary_mask = mask_logits[b, selected_class, rows, cols]  # shape: [N]
        
            # Multiply each lidar point by its corresponding mask value.
            # This operation is fully differentiable.
            selected_pts = binary_mask.unsqueeze(1) * lidar_points[b]  # shape: [N, 5]
        
            selected_points_list.append(selected_pts)
        
        return selected_points_list
    
    def ste_point_selection_inference(self, mask_logits, range_image, lidar_points, 
                        selected_class=1, tau=1.0):
        """
            Differentiable point selection using a binary mask directly.
    
            This function assumes that mask_logits comes from a mask generator that uses
            Gumbel-softmax with hard=True, so its forward pass is binary but gradients 
            are computed via the underlying soft probabilities. Instead of using a 
            non-differentiable boolean mask, we multiply the lidar points by the mask 
            values (which are nearly 0 or 1 in the forward pass). This way, the operation 
            remains differentiable.
    
            Args:
                mask_logits (Tensor): Differentiable mask of shape [B, num_classes, H, W].
                              (For example, produced by your MaskGenerator module.)
                range_image (Tensor): Tensor of shape [B, N, 3] where:
                              - range_image[b][:, 0] contains row indices,
                              - range_image[b][:, 1] contains col indices.
                lidar_points (list[Tensor]): List of length B. Each element is a tensor of shape [N, 5]
                                     with columns (x, y, z, row_idx, col_idx).
                selected_class (int): The channel index used for selection (e.g. class 1).
                tau (float): Temperature for Gumbel softmax (not used directly here, but relevant to the mask).
    
            Returns:
                list[Tensor]: A list of length B, where each element is a tensor of shape [N, 5].
                      Each tensor is the result of multiplying the lidar points by the binary mask.
                      Points not selected are zeroed out, while selected points retain their value.
                      (Gradients flow through the mask thanks to the STE.)
        """
        #print(mask_logits[0,1,:,:])
        B = mask_logits.shape[0]
        selected_points_list = []
    
        for b in range(B):
            # Extract row and column indices from the range image.
            # Here we assume range_image[b] is of shape [1, N, 3].
            rows = range_image[b][0, :, 0].long()  # shape: [N]
            cols = range_image[b][0, :, 1].long()  # shape: [N]
        
            # Use the mask for the selected class.
            # Although the forward pass gives zeros or ones, the gradient flows
            # through the underlying soft probabilities due to the STE.
            binary_mask = mask_logits[b, selected_class, rows, cols]  # shape: [N]
            keep = binary_mask.bool()
        
            # Multiply each lidar point by its corresponding mask value.
            # This operation is fully differentiable.
            selected_pts = lidar_points[b][keep]
        
            selected_points_list.append(selected_pts)
        
        return selected_points_list

    def forward_train(self,
                      queries=None,
                      range_image=None,
                      pred_results=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        num_frames = queries.shape[1]
        queries = [queries[:, i] for i in range(num_frames)]
        for key in pred_results.keys():
            pred_results[key] = [pred_results[key][:, i] for i in range(num_frames)]
        mask_logits, temp=self.pts_bbox_head.mask_generator(queries, img_metas, pred_results)
        # --- Convert gt_bboxes_3d into binary guidance masks ---

        B, C, H, W = mask_logits.shape  # C == 2, two class
        # Prepare empty guidance & small masks
        guidance_masks = torch.zeros((B, H, W), dtype=torch.long, device=mask_logits.device)
        small_masks    = torch.zeros((B, H, W), dtype=torch.bool, device=mask_logits.device)

        for b in range(B):
            boxes = gt_bboxes_3d[b]
            labels = gt_labels_3d[b].cpu().numpy()   # [num_boxes]
            corners = boxes.corners.detach().cpu().numpy()  # [num_boxes, 8, 3]
            if corners.shape[0] == 0:
                continue  # leave both masks as zeros
            bboxes2d = find_2d_bbox_from_3d_corners(corners, H=H, W=W)  # list of [4]-arrays
            dense_mask_np = generate_binary_mask(bboxes2d, H=H, W=W)   # uint8 [H,W]
            guidance_masks[b] = torch.from_numpy(dense_mask_np).to(mask_logits.device)

            small_mask_np = np.zeros((H, W), dtype=bool)
            for lbl, box2d in zip(labels, bboxes2d):
                if lbl in SMALL_LABELS:
                    small_mask_np |= generate_binary_mask([box2d], H=H, W=W).astype(bool)
            small_masks[b] = torch.from_numpy(small_mask_np).to(mask_logits.device)

        
        # --- compute per-pixel focal loss once ---
        # 1) CE per-pixel, flattened
        ce = F.cross_entropy(mask_logits, guidance_masks, reduction='none')  # [B,H,W]
        flat_ce = ce.view(-1)      
        # 2) compute pt and alpha-factor
        pt = torch.exp(-flat_ce)
        t_flat = guidance_masks.view(-1)
        alpha_tensor = mask_logits.new_tensor([0.2, 0.8])
        alpha_factor = alpha_tensor[t_flat]       
        # 3) focal map flat
        focal_flat = alpha_factor * (1 - pt)**2 * flat_ce                   # [B*H*W]

        # --- global focal loss ---
        loss_focal = 20 * focal_flat.mean() 
        small_flat = small_masks.view(-1)
        small_losses = focal_flat[small_flat]                                # [M]
        M = small_losses.numel()     
        lambda_cvar = 0.5
        if M > 0: #loss of CVaR consididering small objects as 
            beta = 0.3
            k = int(math.ceil((1.0 - beta) * M))    # ceil(0.7 * M)
            sorted_l, _ = torch.sort(small_losses, descending=True)
            m_star = sorted_l[k - 1]               # the (1-beta)-quantile

            tail = F.relu(small_losses - m_star).sum()
            loss_cvar = lambda_cvar * (m_star + tail / (beta * M))
        else:
            loss_cvar = mask_logits.new_tensor(0.0, requires_grad=True)

        current_tau = self.pts_bbox_head.mask_generator.mask_generator.temperature
        points=self.ste_point_selection(mask_logits,range_image, points,tau=current_tau)
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore, queries, temp)
            losses.update(losses_pts)
        # --- mask losses ---
        losses.update(dict(loss_mask=loss_focal, loss_cvar=loss_cvar))
        return losses


    
    def forward_train_org(self,
                      queries=None,
                      range_image=None,
                      pred_results=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        num_frames = queries.shape[1]
        queries = [queries[:, i] for i in range(num_frames)]
        for key in pred_results.keys():
            pred_results[key] = [pred_results[key][:, i] for i in range(num_frames)]
        mask, temp=self.pts_bbox_head.mask_generator(queries, img_metas, pred_results)
        # --- Convert gt_bboxes_3d into binary guidance masks ---
        guidance_masks = []
        for gt_box in gt_bboxes_3d:
            # Assume each gt_box has a 'corners' attribute of shape [N, 8, 3] (N boxes)
            corners = gt_box.corners.detach().cpu().numpy()  # shape: [N, 8, 3]
            if corners.shape[0] == 0:
                # No boxes: create an empty (all zeros) mask.
                mask_np = np.zeros((32, 1080), dtype=np.uint8)
            else:
                # Compute the 2D bounding boxes from 3D corners.
                all_corners = find_2d_bbox_from_3d_corners(corners, H=32, W=1080)
                # Generate the binary mask from the 2D bounding boxes.
                mask_np = generate_binary_mask(all_corners, H=32, W=1080)
            guidance_masks.append(mask_np)
        # Stack guidance masks to create a batch; shape: [B, 64, 2048]
        guidance_masks = np.stack(guidance_masks, axis=0)
        # Convert to a torch tensor; ensure type is Long (required for cross entropy) and add a channel dimension if needed.
        guidance_masks = torch.tensor(guidance_masks, dtype=torch.long, device=mask.device)
    
        lambda_sparse = 1
        loss_mask =10 * focal_loss(mask, guidance_masks, alpha=[0.2, 0.8], gamma=2.0)
       # … inside your loop or function …
        current_tau = self.pts_bbox_head.mask_generator.mask_generator.temperature
        #filled_mask = batch_hysteresis_mask(mask, low_frac=0.1)
        #soft_mask= batch_soft_bernoulli_mask(
        #        mask,
        #        offset=0,
        #        quantize_levels=[0.03125,0.0625,0.125,0.25,0.5,1.0]
        #)
        points=self.ste_point_selection_inference(mask,range_image, points,tau=current_tau)
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore, queries, temp)
            losses.update(losses_pts)
        losses.update(dict(loss_mask=loss_mask))
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None, 
                          queries=None,
                          temp=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs, outs_dec = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        #print(type(outs[0][0][0]))
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs,temp, queries[0]]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     queries=None,
                     range_image=None,
                     pred_results=None,
                     points=None,
                     img_metas=None,
                     img=None, frame_index=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(queries[0],pred_results[0], range_image[0],points[0], img_metas[0], img[0], frame_index, **kwargs)
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs, outs_dec = self.pts_bbox_head(x, x_img, img_metas)
        #sys.stdout.write("we are here in CMT head 225 \n")
        #print(outs)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results, outs_dec

    def simple_test(self, queries, pred_results, range_image, points, img_metas, img=None, frame_index=None, rescale=False):
        #print(frame_index)
        num_frames = queries.shape[1]
        queries = [queries[:, i] for i in range(num_frames)]
        for key in pred_results.keys():
            pred_results[key] = [pred_results[key][:, i] for i in range(num_frames)]
        mask, temp=self.pts_bbox_head.mask_generator(queries, img_metas, pred_results)
    
        selected_class = 1
        current_tau = self.pts_bbox_head.mask_generator.mask_generator.temperature
        #print(points)
        #if frame_index is not None and frame_index < 4:
        #    soft_mask[:, 1, ...] = torch.ones_like(soft_mask[:, 1, ...])
        filled_mask = batch_hysteresis_mask(mask, low_frac=0.1)
        soft_mask= batch_soft_bernoulli_mask(
                filled_mask,
                offset=0,
                quantize_levels=[0.03125,0.0625,0.125,0.25,0.5,1.0]
        )
        points=self.ste_point_selection_inference(soft_mask,range_image, points,tau=current_tau)

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        #print("we are in test 242")
        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts, outs_dec  = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list , outs_dec


