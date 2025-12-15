import sys
sys.path.append('../')
sys.path.append("../submodules")
sys.path.append('../submodules/RoMa')

#from typing import Tuple, Dict, Any
from matplotlib import pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist

import torch.nn.functional as F
from romatch import roma_outdoor, roma_indoor
from utils.sh_utils import RGB2SH
from romatch.utils import get_tuple_transform_ops

import time
from collections import defaultdict
from tqdm import tqdm

def _stack_KHW(x):
    """
    Returns a torch.Tensor of shape (K,H,W) on CPU from:
      - list/tuple of Tensors/ndarrays each (H,W)
      - Tensor of shape (H,W) or (K,H,W)
      - ndarray of shape (H,W) or (K,H,W)
    """
    import torch, numpy as np
    if torch.is_tensor(x):
        if x.ndim == 2:         # (H,W) => (1,H,W)
            return x.unsqueeze(0).detach().cpu()
        elif x.ndim == 3:       # (K,H,W)
            return x.detach().cpu()
        else:
            raise ValueError(f"Unexpected tensor shape {tuple(x.shape)} for certainties_all")
    elif isinstance(x, (list, tuple)):
        xs = []
        for xi in x:
            if torch.is_tensor(xi):
                xs.append(xi.detach().cpu())
            else:  # numpy
                xs.append(torch.from_numpy(np.asarray(xi)))
        return torch.stack(xs, dim=0)
    else:  # numpy
        x = np.asarray(x)
        if x.ndim == 2:
            return torch.from_numpy(x[None, ...])
        elif x.ndim == 3:
            return torch.from_numpy(x)
        else:
            raise ValueError(f"Unexpected ndarray shape {x.shape} for certainties_all")

def _resize_label_nearest(arr_np, target_hw):
    """Nearest-neighbor resize for integer label maps."""
    Ht, Wt = target_hw
    pil = Image.fromarray(arr_np.astype(np.int32))
    pil = pil.resize((Wt, Ht), resample=Image.NEAREST)
    return np.array(pil, dtype=np.int32)

def _warp_seg_to_ref(seg_nbr_np, grid_AB, out_hw):
    """
    Warp neighbor integer label map into the reference canvas using RoMa A->B grid.
    seg_nbr_np: (Hn, Wn) int label map (SAM at your feature level)
    grid_AB:    (Hc, Wc, 2) RoMa coords in B (normalized [-1,1]); last dim must be (x_B, y_B)
    out_hw:     (Hc, Wc)
    Returns:    (Hc, Wc) int label map on reference canvas
    """
    import torch.nn.functional as F
    Hc, Wc = out_hw
    # grid_sample needs (N,H,W,2) with order (x,y)
    grid = grid_AB.clone()
    if grid.shape[-1] != 2:
        raise RuntimeError("grid_AB must have last dim=2")
    # Many RoMa impls store (y,x); ensure (x,y)
    grid = torch.stack([grid[...,1], grid[...,0]], dim=-1)
    grid = grid[None]  # (1,Hc,Wc,2)
    seg_t = torch.from_numpy(seg_nbr_np.astype(np.int64))[None,None].float().to(grid.device)  # (1,1,Hn,Wn)
    # Nearest to preserve labels
    proj = F.grid_sample(seg_t, grid, mode="nearest", align_corners=False)  # (1,1,Hc,Wc)
    return proj[0,0].round().to(torch.int64).cpu().numpy()

def _best_iou_for_mask(seg_ref_hw, seg_proj_hw, mid, counts_proj=None):
    """
    For a reference mask id 'mid', find the neighbor-projected label in seg_proj_hw
    that maximizes IoU with the ref mask region.
    Returns IoU* (float in [0,1]).
    """
    m = (seg_ref_hw == int(mid))
    if not np.any(m): return 0.0
    area_m = int(m.sum())
    # label histogram inside the mask
    inside = seg_proj_hw[m]
    if inside.size == 0: return 0.0
    max_label = int(inside.max())
    binc_in  = np.bincount(inside, minlength=max_label+1)
    if counts_proj is None:
        counts_proj = np.bincount(seg_proj_hw.reshape(-1), minlength=max_label+1)
    inter = binc_in.astype(np.int64)
    union = area_m + counts_proj[:inter.shape[0]] - inter
    valid = union > 0
    if not np.any(valid): return 0.0
    iou = np.zeros_like(union, dtype=np.float32)
    iou[valid] = inter[valid] / union[valid]
    return float(iou.max())

def _calibrate_sigmoid(x, pct=(25,50,75), rho=1.0):
    """
    Per-image calibration: robust z-score via IQR, then sigmoid, then power rho.
    x is 1D array of raw scores per mask id (zeros for missing ids ok).
    """
    nz = x[x > 0]
    if nz.size == 0: return x.astype(np.float32)
    q25,q50,q75 = np.percentile(nz, pct)
    denom = max(q75 - q25, 1e-6)
    z = (x - q50) / denom
    z[~np.isfinite(z)] = 0.0
    #w = 1.0/(1.0 + np.exp(-z))
    w = sigmoid_stable(z)
    if rho != 1.0:
        w = np.power(w, float(rho))
    return w.astype(np.float32)

def sigmoid_stable(z):
    z = np.asarray(z, dtype=np.float32)
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    # For z >= 0: 1 / (1 + exp(-z)) is safe
    out[pos] = 1.0 / (1.0 + np.exp(-np.clip(-z[pos], -20.0, 20.0)))
    # For z < 0: exp(z) / (1 + exp(z)) avoids exp(-z) overflow
    zneg = np.clip(z[~pos], -20.0, 20.0)
    ez   = np.exp(zneg)
    out[~pos] = ez / (1.0 + ez)
    return out

def _binary_erode_bool(mask_bool, radius=1):
    """Binary erosion using torch 3x3 ones kernel; returns numpy bool mask."""
    if radius <= 0:
        return mask_bool
    x = torch.from_numpy(mask_bool.astype(np.float32))[None,None]  # 1x1xH xW
    k = torch.ones(1,1,2*radius+1,2*radius+1)
    y = F.conv2d(x, k, padding=radius)
    need = float(k.numel())
    out = (y >= need - 1e-6).squeeze(0).squeeze(0).numpy().astype(bool)
    return out

def _trimmed_mean(x, trim_pct=10.0):
    """Percentile trimmed mean (drop trim_pct/2 from each tail). x is 1D float array."""
    if x.size == 0: return 0.0
    lo = np.percentile(x, trim_pct/2.0)
    hi = np.percentile(x, 100.0 - trim_pct/2.0)
    keep = (x >= lo) & (x <= hi)
    if not np.any(keep): return float(x.mean())
    return float(x[keep].mean())

def _entropy01(counts):
    """Normalized entropy in [0,1] for a 1D non-negative array."""
    s = counts.sum()
    if s <= 0: return 1.0
    p = counts / s
    p = p[p > 0]
    H = -(p*np.log(p)).sum()
    Hmax = np.log(counts.size) if counts.size > 0 else 1.0
    return float(H / max(Hmax, 1e-6))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# ============================================================================

def _make_vis_mask_for_ref(viewpoint_ref, gaussians, pipe, background, H, W, tauT=0.05):
    """
    Try to get a per-pixel 'visible' mask on the RoMa canvas (H,W).
    If a GS renderer is available, we render transmittance T and keep T<1 (hit).
    Otherwise fall back to 'all visible' -> the code path remains functional.
    """
    try:
        from gaussian_renderer import count_render  # available in Dr.Splat
        # Render at (H,W) regardless of the original camera size
        vp = viewpoint_ref.clone_with_resolution(H, W) if hasattr(viewpoint_ref, "clone_with_resolution") else viewpoint_ref
        pkg = count_render(vp, gaussians, pipe, background)
        # pkg['accumulated_alpha'] or 'transmittance' naming differs across repos;
        # prefer contributions sum as 'hit' proxy if transmittance key absent.
        if "transmittance" in pkg:
            T = pkg["transmittance"].detach().clamp(0, 1)      # [H,W]
            vis = (T < (1.0 - tauT))
        elif "accumulated_alpha" in pkg:
            A = pkg["accumulated_alpha"].detach().clamp(0, 1)  # [H,W]
            vis = (A > tauT)
        else:
            contrib = pkg.get("per_pixel_gaussian_contributions", None)
            vis = contrib is not None and contrib.sum(dim=-1) > tauT if contrib is not None else None
        if vis is None:
            return np.ones((H, W), dtype=bool)
        return vis.bool().cpu().numpy()
    except Exception:
        # Renderer not available in EDGS corr-init; fall back gracefully
        return np.ones((H, W), dtype=bool)

def pairwise_distances(matrix):
    """
    Computes the pairwise Euclidean distances between all vectors in the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N, N].
    """
    # Compute squared pairwise distances
    squared_diff = torch.cdist(matrix, matrix, p=2)
    return squared_diff


def k_closest_vectors(matrix, k):
    """
    Finds the k-closest vectors for each vector in the input matrix based on Euclidean distance.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.
        k (int): Number of closest vectors to return for each vector.

    Returns:
        torch.Tensor: Indices of the k-closest vectors for each vector, excluding the vector itself.
    """
    # Compute pairwise distances
    distances = pairwise_distances(matrix)

    # For each vector, sort distances and get the indices of the k-closest vectors (excluding itself)
    # Set diagonal distances to infinity to exclude the vector itself from the nearest neighbors
    distances.fill_diagonal_(float('inf'))

    # Get the indices of the k smallest distances (k-closest vectors)
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    return indices


def select_cameras_kmeans(cameras, K):
    """
    Selects K cameras from a set using K-means clustering.

    Args:
        cameras: NumPy array of shape (N, 16), representing N cameras with their 4x4 homogeneous matrices flattened.
        K: Number of clusters (cameras to select).

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
    """
    # Ensure input is a NumPy array
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)

    if cameras.shape[1] != 16:
        raise ValueError("Each camera must have 16 values corresponding to a flattened 4x4 matrix.")

    # Perform K-means clustering
    cluster_centers, _ = kmeans(cameras, K)

    # Assign each camera to a cluster and find distances to cluster centers
    cluster_assignments, _ = vq(cameras, cluster_centers)

    # Find the camera nearest to each cluster center
    selected_indices = []
    for k in range(K):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)

    return selected_indices


def compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, device="cuda", verbose=False, output_dict={}):
    """
    Computes the warp and confidence between two viewpoint cameras using the roma_model.

    Args:
        viewpoint_cam1: Source viewpoint camera.
        viewpoint_cam2: Target viewpoint camera.
        roma_model: Pre-trained Roma model for correspondence matching.
        device: Device to run the computation on.
        verbose: If True, displays the images.

    Returns:
        certainty: Confidence tensor.
        warp: Warp tensor.
        imB: Processed image B as numpy array.
    """
    # Prepare images
    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
    imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        cax1 = ax[0].imshow(imA)
        ax[0].set_title("Image 1")
        cax2 = ax[1].imshow(imB)
        ax[1].set_title("Image 2")
        fig.colorbar(cax1, ax=ax[0])
        fig.colorbar(cax2, ax=ax[1])
    
        for axis in ax:
            axis.axis('off')
        # Save the figure into the dictionary
        output_dict[f'image_pair'] = fig
   
    # Transform images
    ws, hs = roma_model.w_resized, roma_model.h_resized
    test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
    im_A, im_B = test_transform((imA, imB))
    batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}

    # Forward pass through Roma model
    corresps = roma_model.forward(batch) if not roma_model.symmetric else roma_model.forward_symmetric(batch)
    finest_scale = 1
    hs, ws = roma_model.upsample_res if roma_model.upsample_preds else (hs, ws)

    # Process certainty and warp
    certainty = corresps[finest_scale]["certainty"]
    im_A_to_im_B = corresps[finest_scale]["flow"]
    if roma_model.attenuate_cert:
        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty -= 0.5 * low_res_certainty * (low_res_certainty < 0)

    # Upsample predictions if needed
    if roma_model.upsample_preds:
        im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode="bilinear"
        )

    # Convert predictions to final format
    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    im_A_coords = torch.stack(torch.meshgrid(
        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
        indexing='ij'
    ), dim=0).permute(1, 2, 0).unsqueeze(0).expand(im_A_to_im_B.size(0), -1, -1, -1)

    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    certainty = certainty.sigmoid()

    return certainty[0, 0], warp[0], np.array(imB)


def resize_batch(tensors_3d, tensors_4d, target_shape):
    """
    Resizes a batch of tensors with shapes [B, H, W] and [B, H, W, 4] to the target spatial dimensions.

    Args:
        tensors_3d: Tensor of shape [B, H, W].
        tensors_4d: Tensor of shape [B, H, W, 4].
        target_shape: Tuple (target_H, target_W) specifying the target spatial dimensions.

    Returns:
        resized_tensors_3d: Tensor of shape [B, target_H, target_W].
        resized_tensors_4d: Tensor of shape [B, target_H, target_W, 4].
    """
    target_H, target_W = target_shape

    # Resize [B, H, W] tensor
    resized_tensors_3d = F.interpolate(
        tensors_3d.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).squeeze(1)

    # Resize [B, H, W, 4] tensor
    B, _, _, C = tensors_4d.shape
    resized_tensors_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    return resized_tensors_3d, resized_tensors_4d


def aggregate_confidences_and_warps(viewpoint_stack, closest_indices, roma_model, source_idx, verbose=False, output_dict={}):
    """
    Aggregates confidences and warps by iterating over the nearest neighbors of the source viewpoint.

    Args:
        viewpoint_stack: Stack of viewpoint cameras.
        closest_indices: Indices of the nearest neighbors for each viewpoint.
        roma_model: Pre-trained Roma model.
        source_idx: Index of the source viewpoint.
        verbose: If True, displays intermediate results.

    Returns:
        certainties_max: Aggregated maximum confidences.
        warps_max: Aggregated warps corresponding to maximum confidences.
        certainties_max_idcs: Pixel-wise index of the image  from which we taken the best matching.
        imB_compound: List of the neighboring images.
    """
    certainties_all, warps_all, imB_compound = [], [], []

    for nn in tqdm(closest_indices[source_idx]):

        viewpoint_cam1 = viewpoint_stack[source_idx]
        viewpoint_cam2 = viewpoint_stack[nn]

        certainty, warp, imB = compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, verbose=verbose, output_dict=output_dict)
        certainties_all.append(certainty)
        warps_all.append(warp)
        imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)
    target_shape = imB_compound[0].shape[:2]
    if verbose: 
        print("certainties_all.shape:", certainties_all.shape)
        print("torch.stack(warps_all, dim=0).shape:", torch.stack(warps_all, dim=0).shape)
        print("target_shape:", target_shape)        

    certainties_all_resized, warps_all_resized = resize_batch(certainties_all,
                                                              torch.stack(warps_all, dim=0),
                                                              target_shape
                                                              )

    if verbose:
        print("warps_all_resized.shape:", warps_all_resized.shape)
        for n, cert in enumerate(certainties_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence")
            output_dict[f'certainty_{n}'] = fig

        for n, warp in enumerate(warps_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp")
            output_dict[f'warp_resized_{n}'] = fig

        for n, cert in enumerate(certainties_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence resized")
            output_dict[f'certainty_resized_{n}'] = fig

        for n, warp in enumerate(warps_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp resized")
            output_dict[f'warp_resized_{n}'] = fig

    certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)
    H, W = certainties_max.shape

    warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]

    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = np.clip(imA * 255, 0, 255).astype(np.uint8)

    return certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all_resized, warps_all_resized



def extract_keypoints_and_colors(imA, imB_compound, certainties_max, certainties_max_idcs, matches, roma_model,
                                 verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB_compound: List of target images as NumPy arrays [(H_B, W_B, C), ...].
        certainties_max: Tensor of pixel-wise maximum confidences.
        certainties_max_idcs: Tensor of pixel-wise indices for the best matches.
        matches: Matches in normalized coordinates.
        roma_model: Roma model instance for keypoint operations.
        verbose: if to show intermediate outputs and visualize results

    Returns:
        kptsA_np: Keypoints in imA in normalized coordinates.
        kptsB_np: Keypoints in imB in normalized coordinates.
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB based on certainties_max_idcs.
    """
    # Ensure inputs are numpy uint8
    if isinstance(imA, Image.Image):
        imA = np.array(imA)

    # imB_compound is a list of images; ensure all np arrays
    imB_compound = [
        np.array(imB) if isinstance(imB, Image.Image) else imB
        for imB in imB_compound
    ]

    # Enforce dtype uint8 (handles cases where arrays are float in [0,1])
    if imA.dtype != np.uint8:
        imA = (np.clip(imA, 0, 1) * 255).astype(np.uint8) if imA.max() <= 1.0 else imA.astype(np.uint8)

    for i in range(len(imB_compound)):
        if imB_compound[i].dtype != np.uint8:
            imB = imB_compound[i]
            imB_compound[i] = (np.clip(imB, 0, 1) * 255).astype(np.uint8) if imB.max() <= 1.0 else imB.astype(np.uint8)

    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    # Convert matches to pixel coordinates
    kptsA, kptsB = roma_model.to_pixel_coordinates(
        matches, W_A, H_A, H, W  # W, H
    )

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()
    kptsA_np = kptsA_np[:, [1, 0]]

    if verbose:
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imA)
        ax.set_title("Reference image, imA")
        output_dict[f'reference_image'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imB_compound[0])
        ax.set_title("Image to compare to image, imB_compound")
        output_dict[f'imB_compound'] = fig
    
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imA))
        cax = ax.scatter(kptsA_np[:, 0], H_A - kptsA_np[:, 1], s=.03)
        ax.set_title("Keypoints in imA")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsA'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_compound[0]))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("Keypoints in imB")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsB'] = fig

    # Keypoints are in format (row, column) so the first value is alwain in range [0;height] and second is in range[0;width]

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()

    # Extract colors for keypoints in imA (vectorized)
    # New experimental version
    kptsA_x = np.round(kptsA_np[:, 0] / 1.).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1] / 1.).astype(int)
    kptsA_color = imA[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
   
    # Create a composite image from imB_compound
    imB_compound_np = np.stack(imB_compound, axis=0)
    H_B, W_B, _ = imB_compound[0].shape

    # Extract colors for keypoints in imB using certainties_max_idcs
    imB_np = imB_compound_np[
            certainties_max_idcs.detach().cpu().numpy(),
            np.arange(H).reshape(-1, 1),
            np.arange(W)
        ]
    
    if verbose:
        print("imB_np.shape:", imB_np.shape)
        print("imB_np:", imB_np)
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_np))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("np.flipud(imB_np[0]")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'np.flipud(imB_np[0]'] = fig


    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()
    kptsB_proj_matrices_idx = certainties_max_idcs_np[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
    kptsB_color = imB_compound_np[kptsB_proj_matrices_idx, np.clip(kptsB_y, 0, H - 1), np.clip(kptsB_x, 0, W - 1)]

    # Normalize keypoints in both images
    kptsA_np[:, 0] = kptsA_np[:, 0] / H * 2.0 - 1.0
    kptsA_np[:, 1] = kptsA_np[:, 1] / W * 2.0 - 1.0
    kptsB_np[:, 0] = kptsB_np[:, 0] / W_B * 2.0 - 1.0
    kptsB_np[:, 1] = kptsB_np[:, 1] / H_B * 2.0 - 1.0

    return kptsA_np[:, [1, 0]], kptsB_np, kptsB_proj_matrices_idx, kptsA_color, kptsB_color

def prepare_tensor(input_array, device):
    """
    Converts an input array to a torch tensor, clones it, and detaches it for safe computation.
    Args:
        input_array (array-like): The input array to convert.
        device (str or torch.device): The device to move the tensor to.
    Returns:
        torch.Tensor: A detached tensor clone of the input array on the specified device.
    """
    if not isinstance(input_array, torch.Tensor):
        return torch.tensor(input_array, dtype=torch.float32).to(device).clone().detach()
    return input_array.clone().detach().to(device).to(torch.float32)

def triangulate_points(P1, P2, k1_x, k1_y, k2_x, k2_y, device="cuda"):
    """
    Solves for a batch of 3D points given batches of projection matrices and corresponding image points.

    Parameters:
    - P1, P2: Tensors of projection matrices of size (batch_size, 4, 4) or (4, 4)
    - k1_x, k1_y: Tensors of shape (batch_size,)
    - k2_x, k2_y: Tensors of shape (batch_size,)

    Returns:
    - X: A tensor containing the 3D homogeneous coordinates, shape (batch_size, 4)
    """
    EPS = 1e-4
    # Ensure inputs are tensors

    P1 = prepare_tensor(P1, device)
    P2 = prepare_tensor(P2, device)
    k1_x = prepare_tensor(k1_x, device)
    k1_y = prepare_tensor(k1_y, device)
    k2_x = prepare_tensor(k2_x, device)
    k2_y =  prepare_tensor(k2_y, device)
    batch_size = k1_x.shape[0]

    # Expand P1 and P2 if they are not batched
    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract columns from P1 and P2
    P1_0 = P1[:, :, 0]  # Shape: (batch_size, 4)
    P1_1 = P1[:, :, 1]
    P1_2 = P1[:, :, 2]

    P2_0 = P2[:, :, 0]
    P2_1 = P2[:, :, 1]
    P2_2 = P2[:, :, 2]

    # Reshape kx and ky to (batch_size, 1)
    k1_x = k1_x.view(-1, 1)
    k1_y = k1_y.view(-1, 1)
    k2_x = k2_x.view(-1, 1)
    k2_y = k2_y.view(-1, 1)

    # Construct the equations for each batch
    # For camera 1
    A1 = P1_0 - k1_x * P1_2  # Shape: (batch_size, 4)
    A2 = P1_1 - k1_y * P1_2
    # For camera 2
    A3 = P2_0 - k2_x * P2_2
    A4 = P2_1 - k2_y * P2_2

    # Stack the equations
    A = torch.stack([A1, A2, A3, A4], dim=1)  # Shape: (batch_size, 4, 4)

    # Right-hand side (constants)
    b = -A[:, :, 3]  # Shape: (batch_size, 4)
    A_reduced = A[:, :, :3]  # Coefficients of x, y, z

    # Solve using torch.linalg.lstsq (supports batching)
    X_xyz = torch.linalg.lstsq(A_reduced, b.unsqueeze(2)).solution.squeeze(2)  # Shape: (batch_size, 3)

    # Append 1 to get homogeneous coordinates
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=X_xyz.device)
    X = torch.cat([X_xyz, ones], dim=1)  # Shape: (batch_size, 4)

    # Now compute the errors of projections.
    seeked_splats_proj1 = (X.unsqueeze(1) @ P1).squeeze(1)
    seeked_splats_proj1 = seeked_splats_proj1 / (EPS + seeked_splats_proj1[:, [3]])
    seeked_splats_proj2 = (X.unsqueeze(1) @ P2).squeeze(1)
    seeked_splats_proj2 = seeked_splats_proj2 / (EPS + seeked_splats_proj2[:, [3]])
    proj1_target = torch.concat([k1_x, k1_y], dim=1)
    proj2_target = torch.concat([k2_x, k2_y], dim=1)
    errors_proj1 = torch.abs(seeked_splats_proj1[:, :2] - proj1_target).sum(1).detach().cpu().numpy()
    errors_proj2 = torch.abs(seeked_splats_proj2[:, :2] - proj2_target).sum(1).detach().cpu().numpy()

    return X, errors_proj1, errors_proj2



def select_best_keypoints(
        NNs_triangulated_points, NNs_errors_proj1, NNs_errors_proj2, device="cuda"):
    """
    From all the points fitted to  keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        NNs_triangulated_points:  torch tensor with keypoints coordinates (num_nns, num_points, dim). dim can be arbitrary,
            usually 3 or 4(for homogeneous representation).
        NNs_errors_proj1:  numpy array with projection error of the estimated keypoint on the reference frame (num_nns, num_points).
        NNs_errors_proj2:  numpy array with projection error of the estimated keypoint on the neighbor frame (num_nns, num_points).
    Returns:
        selected_keypoints: torch tensor (num_points, dim) — triangulated points from the winning neighbor per sample.
        selected_proj_errors: numpy array (num_points,) — min over neighbors of max(error1, error2) per sample.
        winning_indices: torch.long tensor (num_points,) — argmin neighbor index per sample (on the provided device).
    """

    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)

    # Convert indices to PyTorch tensor
    indices = torch.from_numpy(np.argmin(NNs_errors_proj, axis=0)).long().to(device)

    # Create index tensor for the second dimension
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)

    # Use advanced indexing to select elements
    NNs_triangulated_points_selected = NNs_triangulated_points[indices, n_indices, :]  # Shape: [N, k]
    selected_proj_errors = np.min(NNs_errors_proj, axis=0)

    # NOTE: now we also return the winning neighbor indices
    return NNs_triangulated_points_selected, selected_proj_errors, indices


def _resize_label_nearest(arr_np, target_hw):
    """Nearest-neighbor resize for integer label maps."""
    Ht, Wt = target_hw
    pil = Image.fromarray(arr_np.astype(np.int32))
    pil = pil.resize((Wt, Ht), resample=Image.NEAREST)
    return np.array(pil, dtype=np.int32)

def _mask_consensus_score(seg_ref_hw, cert_max_hw, nn_idx_hw, ignore_val=-1, alpha=0.7):
    """
    seg_ref_hw : (H,W) int labels for the reference view at the RoMa (H,W)
    cert_max_hw: (H,W) float in [0,1], certainty map after max over neighbors
    nn_idx_hw  : (H,W) int, argmax neighbor index per pixel
    Returns:
        c_vec: (num_masks_ref,) float32 in [0,1], index = mask id in seg_ref_hw
               Missing ids get 0.
    """
    H, W = seg_ref_hw.shape
    valid = (seg_ref_hw != ignore_val)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32)

    seg = seg_ref_hw[valid]                 # (N,)
    cert = cert_max_hw[valid].astype(np.float32)  # (N,)
    nn   = nn_idx_hw[valid].astype(np.int32)      # (N,)

    # List of mask ids present in this image
    mask_ids = np.unique(seg)
    mask_ids = mask_ids[mask_ids != ignore_val]
    if mask_ids.size == 0:
        return np.zeros((0,), dtype=np.float32)

    # Build scores per mask:
    #  - mean certainty inside the mask
    #  - neighbor-consistency: fraction of pixels whose per-pixel best neighbor equals the modal neighbor for that mask
    max_mask_id = int(mask_ids.max())
    c_vec = np.zeros((max_mask_id + 1,), dtype=np.float32)

    for mid in mask_ids:
        m = (seg == mid)
        if not np.any(m):
            continue
        cert_m = cert[m]
        nn_m   = nn[m]

        mean_cert = float(cert_m.mean())

        # modal neighbor
        binc = np.bincount(nn_m)
        dom_nn = int(binc.argmax())
        dom_frac = float((nn_m == dom_nn).mean())

        # combine
        score = alpha * mean_cert + (1.0 - alpha) * dom_frac
        # clamp for safety
        score = float(np.clip(score, 0.0, 1.0))
        c_vec[mid] = score

    return c_vec

def init_gaussians_with_corr(gaussians, scene, cfg, device, verbose = False, roma_model=None):
    """
    For a given input gaussians and a scene we instantiate a RoMa model(change to indoors if necessary) and process scene
    training frames to extract correspondences. Those are used to initialize gaussians
    Args:
        gaussians: object gaussians of the class GaussianModel that we need to enrich with gaussians.
        scene: object of the Scene class.
        cfg: configuration. Use init_wC
    Returns:
        gaussians: inplace transforms object gaussians of the class GaussianModel.

    """
    if roma_model is None:
        if cfg.roma_model == "indoors":
            roma_model = roma_indoor(device=device)
        else:
            roma_model = roma_outdoor(device=device)
        roma_model.upsample_preds = False
        roma_model.symmetric = False

    M = cfg.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    scaling_factor = cfg.scaling_factor
    expansion_factor = 1
    keypoint_fit_error_tolerance = cfg.proj_err_tolerance

    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(cfg.num_refs, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = min(cfg.nns_per_ref , len(viewpoint_stack))

    # Select cameras using K-means
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)
   

    # Find the k-closest vectors for each vector
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    if verbose: print("Indices of k-closest vectors for each vector:\n", closest_indices)

    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    # Accumulators for new gaussians (unchanged)
    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []

    # ADDED: accumulators for per-seed metadata aligned with all_new_xyz order
    all_seed_ref_cam_idx = []
    all_seed_nbr_cam_idx = []
    all_seed_uv_ref_norm = []
    all_seed_uv_nbr_norm = []
    all_seed_reproj_err_mean = []
    all_seed_reproj_err_max = []
    all_seed_match_conf = []

    # Run roma_model.match once to kinda initialize the model 
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)
        print("Once run full roma_model.match warp.shape:", warp.shape)
        print("Once run full roma_model.match certainty_warp.shape:", certainty_warp.shape)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    for source_idx in tqdm(sorted(selected_indices)):

        viewpoint_cam1 = viewpoint_stack[source_idx]

        # Compute per-neighbor warps & certainties (unchanged)
        certainties_all, warps_all, imB_compound = [], [], []

        for nn in tqdm(closest_indices[source_idx]):
            viewpoint_cam2 = viewpoint_stack[nn]
            certainty, warp, imB = compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, verbose=verbose, output_dict=visualizations)
            certainties_all.append(certainty)
            warps_all.append(warp)
            imB_compound.append(imB)
        certainties_all = torch.stack(certainties_all, dim=0)

        # Resize and aggregate (unchanged)
        target_shape = imB_compound[0].shape[:2]
        certainties_all_resized, warps_all_resized = resize_batch(certainties_all, torch.stack(warps_all, dim=0), target_shape)
        certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)
        H, W = certainties_max.shape
        warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]

        # Visibility (GS) mask on the RoMa canvas (True = keep). Falls back to all-True if renderer not found.
        vis_mask = _make_vis_mask_for_ref(viewpoint_stack[source_idx], gaussians, cfg.pipe if hasattr(cfg, "pipe") else None,
                                          torch.tensor([1,1,1], dtype=torch.float32, device=certainties_max.device) if not hasattr(cfg, "background") else cfg.background,
                                          H, W, tauT=getattr(cfg, "cluster_tauT", 0.05))
        
        # ===== Multi-view clustering (RoMa canvas) =====
        try:
            if hasattr(cfg, "langfeat_dir") and (cfg.langfeat_dir is not None):
                ref_name = viewpoint_stack[source_idx].image_name
                seg_path_ref = os.path.join(cfg.langfeat_dir, ref_name + "_s.npy")
                if os.path.isfile(seg_path_ref):
                    lvl = int(getattr(cfg, "langfeat_level", 1))
                    seg_pyr_ref = np.load(seg_path_ref, allow_pickle=True)
                    seg_ref = seg_pyr_ref[lvl]  # (H0,W0) ints, -1 ignored

                    # Resize ref seg to RoMa canvas (H,W) with nearest
                    seg_ref_t  = torch.from_numpy(seg_ref.astype(np.int64))[None,None].float()
                    seg_ref_hw = torch.nn.functional.interpolate(seg_ref_t, size=(H, W), mode="nearest")[0,0].to(torch.int64).cpu().numpy()

                    # Collect neighbor projections and names
                    proj_list, weight_list, nbr_names = [], [], []
                    for k, nn in enumerate(closest_indices_selected[source_idx, :]):
                        nbr_idx  = int(nn)
                        nbr_name = viewpoint_stack[nbr_idx].image_name
                        seg_path_nbr = os.path.join(cfg.langfeat_dir, nbr_name + "_s.npy")
                        if not os.path.isfile(seg_path_nbr):
                            continue
                        seg_pyr_nbr = np.load(seg_path_nbr, allow_pickle=True)
                        seg_nbr = seg_pyr_nbr[lvl]  # (Hn,Wn)

                        # RoMa A->B grid: use last two channels from your resized warp for this neighbor
                        grid_AB = warps_all_resized[k][..., 2:4].detach().float().cpu()
                        # ensure grid order (x,y)
                        grid = torch.stack([grid_AB[...,1], grid_AB[...,0]], dim=-1)[None]  # (1,H,W,2)

                        seg_t = torch.from_numpy(seg_nbr.astype(np.int64))[None,None].float()
                        seg_proj_hw = torch.nn.functional.grid_sample(seg_t, grid, mode="nearest", align_corners=False)[0,0].round().to(torch.int64).cpu().numpy()
                        # --- occlusion pruning ---
                        if isinstance(vis_mask, np.ndarray):
                            seg_proj_hw = np.where(vis_mask, seg_proj_hw, -1).astype(seg_proj_hw.dtype)
                        # -------------------------------
                        proj_list.append(seg_proj_hw)
                        weight_list.append(certainties_all_resized[k].detach().cpu().numpy())
                        nbr_names.append(nbr_name)

                    if len(proj_list) == 0:
                        raise RuntimeError("No neighbor segmentations available to cluster.")

                    # Map labels to compact ids per view to keep arrays dense/small
                    def _compact_ids(arr):
                        vals = arr[arr >= 0]
                        if vals.size == 0: return arr, np.array([-1], dtype=np.int32)
                        uniq = np.unique(vals)
                        lut  = -np.ones(int(uniq.max())+1, dtype=np.int32)
                        lut[uniq] = np.arange(uniq.size, dtype=np.int32)
                        out = -np.ones_like(arr, dtype=np.int32)
                        m = (arr >= 0)
                        out[m] = lut[arr[m]]
                        return out, uniq  # out has ids in [0..K-1], uniq maps back to original labels
        
                    ref_c, ref_labels = _compact_ids(seg_ref_hw)  # ref_c in [0..R-1], -1 for ignored
                    nbr_c_list, nbr_labels_list = [], []
                    for seg_proj in proj_list:
                        nc, nl = _compact_ids(seg_proj)
                        nbr_c_list.append(nc)
                        nbr_labels_list.append(nl)

                    tau_iou = float(getattr(cfg, "cluster_tau_iou", 0.3))   # IoU threshold for edges
                    tau_bbox = float(getattr(cfg, "cluster_tau_bbox", 0.05)) # optional bbox prefilter

                    R = int(ref_labels.size)
                    iou_mats = []     # per neighbor: [R, Lk] IoU
                    bbox_ok = []      # per neighbor: [R, Lk] bool (for prefilter)

                    # Precompute ref areas and bboxes
                    ref_flat = ref_c.reshape(-1)
                    ref_area = np.bincount(ref_flat.clip(min=0), minlength=R)
                    # bboxes per ref id
                    ref_bbox = [None]*R
                    for rid in range(R):
                        ys, xs = np.where(ref_c == rid)
                        if ys.size == 0: ref_bbox[rid] = (0,0,0,0)
                        else: ref_bbox[rid] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                    #optional    
                    # --- size-aware IoU thresholds (small masks need higher IoU) ---
                    Htot, Wtot = ref_c.shape
                    pix_total = float(max(Htot * Wtot, 1))
                    small_mask_frac = float(getattr(cfg, "cluster_small_mask_frac", 0.005))  # e.g., 0.5% of canvas
                    small_mask_tau  = float(getattr(cfg, "cluster_small_mask_tau", max(tau_iou, 0.30)))
                    rid_small = (ref_area / pix_total) < small_mask_frac  # [R] bool

                    # --- precompute ref regions once (for certainty gating) ---
                    # R is usually 20-80, so building boolean regions is fine
                    ref_regions = []
                    for rid in range(R):
                        ref_regions.append(ref_c == rid)  # (H,W) bool

                    #end of 
        
                    for nc in nbr_c_list:
                        L = int(nc.max()) + 1 if nc.max() >= 0 else 0
                        if L == 0:
                            iou_mats.append(np.zeros((R,0), dtype=np.float32))
                            bbox_ok.append(np.zeros((R,0), dtype=bool))
                            continue

                        # areas & bboxes per neighbor label
                        nbr_flat = nc.reshape(-1)
                        nbr_area = np.bincount(nbr_flat.clip(min=0), minlength=L)
                        nbr_bbox = [None]*L
                        for lid in range(L):
                            ys, xs = np.where(nc == lid)
                            if ys.size == 0: nbr_bbox[lid] = (0,0,0,0)
                            else: nbr_bbox[lid] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                        
                        # bbox prefilter
                        bb_ok = np.zeros((R,L), dtype=bool)
                        for rid in range(R):
                            x1r,y1r,x2r,y2r = ref_bbox[rid]
                            ar = (x2r-x1r+1)*(y2r-y1r+1)
                            if ar <= 0: continue
                            for lid in range(L):
                                x1n,y1n,x2n,y2n = nbr_bbox[lid]
                                an = (x2n-x1n+1)*(y2n-y1n+1)
                                if an <= 0: continue
                                # bbox IoU
                                xi1, yi1 = max(x1r, x1n), max(y1r, y1n)
                                xi2, yi2 = min(x2r, x2n), min(y2r, y2n)
                                iw, ih = max(0, xi2-xi1+1), max(0, yi2-yi1+1)
                                inter_bb = iw*ih
                                uni_bb   = ar + an - inter_bb
                                if uni_bb > 0 and (inter_bb/uni_bb) >= tau_bbox:
                                    bb_ok[rid, lid] = True
                        bbox_ok.append(bb_ok)

                        # pixel overlaps by joint bincount on valid pixels
                        m = (ref_flat >= 0) & (nbr_flat >= 0)
                        rf = ref_flat[m]
                        nf = nbr_flat[m]
                        # encode pairs into single index
                        P = np.bincount(rf * L + nf, minlength=R*L).reshape(R, L)
                        # IoU
                        den = (ref_area[:,None] + nbr_area[None,:] - P).astype(np.float64)
                        iou = np.zeros_like(P, dtype=np.float32)
                        v = den > 0
                        iou[v] = (P[v] / den[v]).astype(np.float32)
                        iou_mats.append(iou)

                    # Build bipartite edges with IoU >= tau and mutual top across that neighbor
                    # Node indexing: ref nodes [0..R-1], neighbor nodes get offset per neighbor
                    offsets = [0]
                    for nc in nbr_c_list:
                        Lk = int(nc.max()) + 1 if nc.max() >= 0 else 0
                        offsets.append(offsets[-1] + Lk)
                    Nnbr_total = offsets[-1]  # total neighbor nodes
                    N_nodes = R + Nnbr_total

                    parent = np.arange(N_nodes, dtype=np.int32)
                    def find(a):
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a
                    def union(a,b):
                        ra, rb = find(a), find(b)
                        if ra != rb: parent[rb] = ra

                    '''
                    for j, (iou, bb_ok_j) in enumerate(zip(iou_mats, bbox_ok)):
                        L = iou.shape[1]
                        if L == 0: continue
                        # mutual NN under bbox prefilter
                        iou_mask = (iou >= tau_iou) & bb_ok_j
                        if not iou_mask.any(): continue
                        # for ref side: best neighbor per ref
                        best_nbr = iou.copy()
                        best_nbr[~iou_mask] = -1
                        best_n = best_nbr.argmax(axis=1)  # [R]
                        # for nbr side: best ref per label
                        best_ref = iou.copy()
                        best_ref[~iou_mask] = -1
                        best_r = best_ref.argmax(axis=0)  # [L]

                        for rid in range(R):
                            lid = best_n[rid]
                            if lid < 0 or not iou_mask[rid, lid]:
                                continue
                            if best_r[lid] != rid:
                                continue  # not mutual
                            # connect ref node rid with neighbor-j node offset+lid
                            node_ref = rid
                            node_nbr = R + offsets[j] + lid
                            union(node_ref, node_nbr)'''
                    
                    min_conf = int(getattr(cfg, "cluster_min_confirmations", 2))  # require ≥2 neighbors

                    # First pass: collect candidate mutual edges per neighbor
                    candidate_edges = []  # list of arrays [K_j, 2] with (rid,lid)
                    for j, (iou, final_mask_j) in enumerate(zip(iou_mats, bbox_ok)):
                        L = iou.shape[1]
                        if L == 0:
                            candidate_edges.append(np.zeros((0,2), dtype=np.int32))
                            continue

                        if not final_mask_j.any():
                            candidate_edges.append(np.zeros((0,2), dtype=np.int32))
                            continue

                        # keep only allowed pairs
                        iou_mask = final_mask_j  # already includes bbox+certainty+size-aware IoU

                        # mutual NN under the mask
                        iou_ref = iou.copy()
                        iou_ref[~iou_mask] = -1.0
                        best_n = iou_ref.argmax(axis=1)   # [R], index in [0..L-1]
                        iou_n  = iou_ref[np.arange(iou.shape[0]), best_n]

                        iou_nbr = iou.copy()
                        iou_nbr[~iou_mask] = -1.0
                        best_r = iou_nbr.argmax(axis=0)   # [L], index in [0..R-1]

                        pairs = []
                        for rid in range(iou.shape[0]):
                            lid = best_n[rid]
                            if lid < 0: 
                                continue
                            if best_r[lid] != rid:
                                continue  # not mutual
                            if iou_mask[rid, lid] and (iou_n[rid] >= 0):
                                pairs.append((rid, lid))
                        candidate_edges.append(np.array(pairs, dtype=np.int32))

                    # Count confirmations per ref across neighbors
                    confirm_counts = np.zeros((R,), dtype=np.int32)
                    for arr in candidate_edges:
                        if arr.size == 0: 
                            continue
                        rids = arr[:,0]
                        np.add.at(confirm_counts, rids, 1)

                    # Union only edges whose ref has enough confirmations
                    parent = np.arange(R + sum((int(nc.max()) + 1 if nc.max() >= 0 else 0) for nc in nbr_c_list), dtype=np.int32)
                    def find(a):
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a
                    def union(a,b):
                        ra, rb = find(a), find(b)
                        if ra != rb: parent[rb] = ra

                    offsets = [0]
                    for nc in nbr_c_list:
                        Lk = int(nc.max()) + 1 if nc.max() >= 0 else 0
                        offsets.append(offsets[-1] + Lk)

                    for j, arr in enumerate(candidate_edges):
                        if arr.size == 0:
                            continue
                        for rid, lid in arr:
                            if confirm_counts[rid] < min_conf:
                                continue
                            node_ref = rid
                            node_nbr = R + offsets[j] + int(lid)
                            union(node_ref, node_nbr)

                    # Assign cluster ids from connected components
                    N_nodes = parent.size
                    roots = np.array([find(i) for i in range(N_nodes)], dtype=np.int32)
                    uniq_roots, cluster_ids = np.unique(roots, return_inverse=True)
                    ref_cluster_of_label = cluster_ids[:R]
                    cluster_of_label_per_nbr = []
                    for j, nc in enumerate(nbr_c_list):
                        Lk = int(nc.max()) + 1 if nc.max() >= 0 else 0
                        if Lk == 0:
                            cluster_of_label_per_nbr.append(np.zeros((0,), dtype=np.int32))
                        else:
                            s = R + offsets[j]
                            e = s + Lk
                            cluster_of_label_per_nbr.append(cluster_ids[s:e].copy())
                    #end of



                    # Assign cluster ids: all connected components get a unique id
                    roots = np.array([find(i) for i in range(N_nodes)], dtype=np.int32)
                    uniq_roots, cluster_ids = np.unique(roots, return_inverse=True)
                    # cluster id per ref label id in [0..R-1]
                    ref_cluster_of_label = cluster_ids[:R]  # shape [R]
                    # cluster id per neighbor label with offsets
                    cluster_of_label_per_nbr = []
                    for j, nc in enumerate(nbr_c_list):
                        L = int(nc.max()) + 1 if nc.max() >= 0 else 0
                        if L == 0:
                            cluster_of_label_per_nbr.append(np.zeros((0,), dtype=np.int32))
                        else:
                            s = R + offsets[j]
                            e = s + L
                            cluster_of_label_per_nbr.append(cluster_ids[s:e].copy())
                    # Save clusters
                    out = {
                        "ref_cluster_of_label": ref_cluster_of_label.astype(np.int32),
                        "ref_label_values":      ref_labels.astype(np.int32),  # map compact id -> original label
                        "neighbors":             np.array(nbr_names, dtype=object),
                        "tau_iou":               np.array([tau_iou], dtype=np.float32),
                        "tau_bbox":              np.array([tau_bbox], dtype=np.float32),
                    }
                    for j, (nbr_name, labvals, cl_ids) in enumerate(zip(nbr_names, nbr_labels_list, cluster_of_label_per_nbr)):
                        out[f"cluster_of_label__{nbr_name}"] = cl_ids.astype(np.int32)
                        out[f"label_values__{nbr_name}"]     = labvals.astype(np.int32)

                    np.savez(os.path.join(cfg.langfeat_dir, ref_name + "_clusters.npz"), **out)

                    # Optional: derive per-ref-mask consensus w from its cluster (averaged IoU across linked labels)
                    # This keeps compatibility with Dr.Splat today.
                    # We reuse iou_mats and ref_cluster_of_label.
                    #not using _c.npy for now
                    '''Rmax = int(ref_labels.max()) if ref_labels.size>0 else -1
                    c_raw = np.zeros(Rmax+1, dtype=np.float32)
                    # Build reverse: for each ref compact id rid, which neighbor labels are in same cluster?
                    for rid in range(R):
                        cluster_id = ref_cluster_of_label[rid]
                        if cluster_id < 0: continue
                        iou_acc, w_acc = 0.0, 0.0
                        for j, (iou, nc) in enumerate(zip(iou_mats, nbr_c_list)):
                            L = iou.shape[1]
                            if L == 0: continue
                            # find neighbor labels that share the same cluster
                            cl_ids = cluster_of_label_per_nbr[j]
                            lids   = np.where(cl_ids == cluster_id)[0]  # labels in neighbor j in same cluster
                            if lids.size == 0: continue
                            # weight by certainty mean inside overlap with rid
                            cert_map = weight_list[j]  # (H,W)
                            region   = (ref_c == rid)
                            if not region.any(): continue
                            c_mean = float(cert_map[region].mean()) if cert_map[region].size>0 else 0.0
                            # take max IoU among lids in that neighbor for stability
                            iou_star = float(iou[rid, lids].max())
                            w = 0.5*c_mean + 0.5*iou_star
                            iou_acc += w * iou_star
                            w_acc   += w
                        if w_acc > 0:
                            orig_label = int(ref_labels[rid])
                            c_raw[orig_label] = iou_acc / w_acc

                    # calibrate -> _c.npy
                    rho = float(getattr(cfg, "langfeat_consensus_rho", 1.0))
                    c_w = _calibrate_sigmoid(c_raw, rho=rho)
                    np.save(os.path.join(cfg.langfeat_dir, ref_name + "_c.npy"), c_w)'''

                    #print(f"[EDGS][CLUSTER] {ref_name}: ref_masks={R} nbrs={len(nbr_names)} clusters={uniq_roots.size}")
                    # --- detailed cluster stats (uses existing vars) ---
                    # clusters that touch at least one ref mask
                    ref_cids = ref_cluster_of_label[ref_cluster_of_label >= 0]
                    clusters_touching_ref = int(np.unique(ref_cids).size) if ref_cids.size > 0 else 0
                    
                    # union of neighbor cluster ids (>=0)
                    nbr_cid_parts = []
                    for cl_ids in cluster_of_label_per_nbr:
                        if cl_ids is None or cl_ids.size == 0:
                            continue
                        nbr_cid_parts.append(cl_ids[cl_ids >= 0])
                    if len(nbr_cid_parts) > 0:
                        nbr_cids_unique = np.unique(np.concatenate(nbr_cid_parts, axis=0))
                    else:
                        nbr_cids_unique = np.array([], dtype=np.int32)

                    # ref masks that matched at least one neighbor (cluster id appears in neighbor set)
                    matched_ref = int(np.isin(ref_cids, nbr_cids_unique).sum()) if ref_cids.size > 0 else 0

                    print(f"[EDGS][CLUSTER] {ref_name}: ref_masks={R}, nbrs={len(nbr_names)}, "
                          f"clusters={uniq_roots.size}, clusters_touching_ref={clusters_touching_ref}, matched_ref={matched_ref}")
                    # --- end detailed cluster stats ---
        except Exception as ex:
            print(f"[EDGS][CLUSTER] Failed for {viewpoint_stack[source_idx].image_name}: {ex}")
        # ===== end clustering =====





        matches = warps_max
        certainty = certainties_max.clone()
        certainty[certainty > upper_thresh] = 1
        matches, certainty = (matches.reshape(-1, 4), certainty.reshape(-1))
        good_samples = torch.multinomial(certainty, num_samples=min(expansion_factor * M, len(certainty)), replacement=False)

        reference_image_dict = {
            "ref_image": imA,
            "NNs_images": imB_compound,
            "certainties_all": certainties_all,
            "warps_all": warps_all,
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": [],
            # ADDED: we log normalized keypoints for each neighbor to extract per-seed UVs of the chosen neighbor
            #enabled when seed meta is needed
            #"kptsA_norm_all": [],
            #"kptsB_norm_all": [],
            # Keep the sample indices to compute (row,col) for confidence
            #"good_samples": good_samples.detach().cpu().numpy(),
            #"ref_cam_idx": source_idx,

        }

        with torch.no_grad():
            for NN_idx in tqdm(range(len(warps_all_resized))):
                matches_NN = warps_all_resized[NN_idx].reshape(-1, 4)[good_samples]

                # Extract keypoints and colors
                kptsA_np, kptsB_np, kptsB_proj_matrices_idcs, kptsA_color, kptsB_color = extract_keypoints_and_colors(
                    imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
                )
                # ADDED: Save per-neighbor normalized UVs for later selection
                #reference_image_dict["kptsA_norm_all"].append(kptsA_np[:M])
                #reference_image_dict["kptsB_norm_all"].append(kptsB_np[:M])

                proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
                proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, NN_idx]].full_proj_transform
                triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
                    P1=torch.stack([proj_matrices_A] * M, axis=0),
                    P2=torch.stack([proj_matrices_B] * M, axis=0),
                    k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
                    k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

                reference_image_dict["triangulated_points"].append(triangulated_points)
                reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
                reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)
        
        with torch.no_grad():
            NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors, winning_indices = select_best_keypoints(
                NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
                NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
                NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))
            
        # === ADDED: Build per-seed meta for this reference using the chosen neighbor ===
        #only for seed meta
        '''with torch.no_grad():
            # Shapes
            N = len(NNs_triangulated_points_selected)
            sample_idx = np.arange(N)
            win_np = winning_indices.detach().cpu().numpy()  # (N,)

            # Neighbor camera indices for the winners
            nbr_for_ref = closest_indices_selected[source_idx, :]  # (num_nns,)
            nbr_cam_idx_sel = nbr_for_ref[win_np]                  # (N,)

            # UVs for the winners
            kptsA_stack = np.stack(reference_image_dict["kptsA_norm_all"], axis=0)  # (num_nns, N, 2)
            kptsB_stack = np.stack(reference_image_dict["kptsB_norm_all"], axis=0)  # (num_nns, N, 2)
            uv_ref_norm_sel = kptsA_stack[win_np, sample_idx, :]                    # (N, 2)
            uv_nbr_norm_sel = kptsB_stack[win_np, sample_idx, :]                    # (N, 2)

            # Per-view reprojection errors for the winners
            err1_stack = np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0)  # (num_nns, N)
            err2_stack = np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0)  # (num_nns, N)
            err1_sel = err1_stack[win_np, sample_idx]                                               # (N,)
            err2_sel = err2_stack[win_np, sample_idx]                                               # (N,)
            reproj_err_mean = (err1_sel + err2_sel) * 0.5
            reproj_err_max  = np.maximum(err1_sel, err2_sel)

            # Flattened gather on the same per-neighbor certainty tensor used for sampling
            flat_idx = reference_image_dict["good_samples"].astype(np.int64)
            cert_t = certainties_all_resized
            if cert_t.shape[1:] != certainties_max.shape:
            # nearest is fine — these are certainty maps
                cert_t = torch.nn.functional.interpolate(
                cert_t.unsqueeze(1), size=certainties_max.shape, mode="nearest"
                ).squeeze(1)
            cert_flat = cert_t.reshape(cert_t.shape[0], -1).detach().cpu().numpy()
            match_conf_sel = cert_flat[win_np, flat_idx]                          

            # Accumulate in the same order as all_new_xyz append
            all_seed_ref_cam_idx.append(np.full((N,), source_idx, dtype=np.int32))
            all_seed_nbr_cam_idx.append(nbr_cam_idx_sel.astype(np.int32))
            all_seed_uv_ref_norm.append(uv_ref_norm_sel.astype(np.float32))
            all_seed_uv_nbr_norm.append(uv_nbr_norm_sel.astype(np.float32))
            all_seed_reproj_err_mean.append(reproj_err_mean.astype(np.float32))
            all_seed_reproj_err_max.append(reproj_err_max.astype(np.float32))
            all_seed_match_conf.append(match_conf_sel.astype(np.float32))'''

        # 4. Save as gaussians
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        with torch.no_grad():
            new_xyz = NNs_triangulated_points_selected[:, :-1]
            all_new_xyz.append(new_xyz)  # seeked_splats
            all_new_features_dc.append(RGB2SH(torch.tensor(kptsA_color.astype(np.float32) / 255.)).unsqueeze(1))
            all_new_features_rest.append(torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0))
            # new version that sets points with large error invisible
            # TODO: remove those points instead. However it doesn't affect the performance.
            mask_bad_points = torch.tensor(
                NNs_triangulated_points_selected_proj_errors > keypoint_fit_error_tolerance,
                dtype=torch.float32).unsqueeze(1).to(device)
            all_new_opacities.append(torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0) * 0. - mask_bad_points * (1e1))

            dist_points_to_cam1 = torch.linalg.norm(viewpoint_cam1.camera_center.clone().detach() - new_xyz,
                                                    dim=1, ord=2)
            #all_new_scaling.append(torch.log(((dist_points_to_cam1) / 1. * scaling_factor).unsqueeze(1).repeat(1, 3)))
            all_new_scaling.append(gaussians.scaling_inverse_activation((dist_points_to_cam1 * scaling_factor).unsqueeze(1).repeat(1, 3)))
            all_new_rotation.append(torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0))

    all_new_xyz = torch.cat(all_new_xyz, dim=0) 
    all_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    new_tmp_radii = torch.zeros(all_new_xyz.shape[0])
    prune_mask = torch.ones(all_new_xyz.shape[0], dtype=torch.bool)

    # ADDED: pack per-seed meta in the same order as all_new_xyz
    '''if all_new_xyz.shape[0] > 0:
        visualizations["seed_ref_cam_idx"] = np.concatenate(all_seed_ref_cam_idx, axis=0)
        visualizations["seed_nbr_cam_idx"] = np.concatenate(all_seed_nbr_cam_idx, axis=0)
        visualizations["seed_uv_ref_norm"] = np.concatenate(all_seed_uv_ref_norm, axis=0)
        visualizations["seed_uv_nbr_norm"] = np.concatenate(all_seed_uv_nbr_norm, axis=0)
        visualizations["seed_reproj_err_mean"] = np.concatenate(all_seed_reproj_err_mean, axis=0)
        visualizations["seed_reproj_err_max"]  = np.concatenate(all_seed_reproj_err_max,  axis=0)
        visualizations["seed_match_conf"]      = np.concatenate(all_seed_match_conf,      axis=0)
        # Map to the slice that will be inserted by densification_postfix
        visualizations["seed_new_gauss_start"] = int(gaussians._xyz.shape[0])
        visualizations["seed_new_gauss_count"] = int(all_new_xyz.shape[0])
        visualizations["seed_xyz"] = all_new_xyz.detach().cpu().numpy()'''
    
    gaussians.densification_postfix(all_new_xyz[prune_mask].to(device),
                                    all_new_features_dc[prune_mask].to(device),
                                    torch.cat(all_new_features_rest, dim=0)[prune_mask].to(device),
                                    torch.cat(all_new_opacities, dim=0)[prune_mask].to(device),
                                    torch.cat(all_new_scaling, dim=0)[prune_mask].to(device),
                                    torch.cat(all_new_rotation, dim=0)[prune_mask].to(device))
    
    return viewpoint_stack, closest_indices_selected, visualizations



def extract_keypoints_and_colors_single(imA, imB, matches, roma_model, verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from a source image (imA) and a single target image (imB).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB: Target image as a NumPy array (H_B, W_B, C).
        matches: Matches in normalized coordinates (torch.Tensor).
        roma_model: Roma model instance for keypoint operations.
        verbose: If True, outputs intermediate visualizations.
    Returns:
        kptsA_np: Keypoints in imA (normalized).
        kptsB_np: Keypoints in imB (normalized).
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB.
    """
    H_A, W_A, _ = imA.shape
    H_B, W_B, _ = imB.shape

    # Convert matches to pixel coordinates
    # Matches format: (B, 4) = (x1_norm, y1_norm, x2_norm, y2_norm)
    kptsA = matches[:, :2]  # [N, 2]
    kptsB = matches[:, 2:]  # [N, 2]

    # Scale normalized coordinates [-1,1] to pixel coordinates
    kptsA_pix = torch.zeros_like(kptsA)
    kptsB_pix = torch.zeros_like(kptsB)

    # Important! [Normalized to pixel space]
    kptsA_pix[:, 0] = (kptsA[:, 0] + 1) * (W_A - 1) / 2
    kptsA_pix[:, 1] = (kptsA[:, 1] + 1) * (H_A - 1) / 2

    kptsB_pix[:, 0] = (kptsB[:, 0] + 1) * (W_B - 1) / 2
    kptsB_pix[:, 1] = (kptsB[:, 1] + 1) * (H_B - 1) / 2

    kptsA_np = kptsA_pix.detach().cpu().numpy()
    kptsB_np = kptsB_pix.detach().cpu().numpy()

    # Extract colors
    kptsA_x = np.round(kptsA_np[:, 0]).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1]).astype(int)
    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    kptsA_color = imA[np.clip(kptsA_y, 0, H_A-1), np.clip(kptsA_x, 0, W_A-1)]
    kptsB_color = imB[np.clip(kptsB_y, 0, H_B-1), np.clip(kptsB_x, 0, W_B-1)]

    # Normalize keypoints into [-1, 1] for downstream triangulation
    kptsA_np_norm = np.zeros_like(kptsA_np)
    kptsB_np_norm = np.zeros_like(kptsB_np)

    kptsA_np_norm[:, 0] = kptsA_np[:, 0] / (W_A - 1) * 2.0 - 1.0
    kptsA_np_norm[:, 1] = kptsA_np[:, 1] / (H_A - 1) * 2.0 - 1.0

    kptsB_np_norm[:, 0] = kptsB_np[:, 0] / (W_B - 1) * 2.0 - 1.0
    kptsB_np_norm[:, 1] = kptsB_np[:, 1] / (H_B - 1) * 2.0 - 1.0

    return kptsA_np_norm, kptsB_np_norm, kptsA_color, kptsB_color



def init_gaussians_with_corr_fast(gaussians, scene, cfg, device, verbose=False, roma_model=None):
    timings = defaultdict(list)

    if roma_model is None:
        if cfg.roma_model == "indoors":
            roma_model = roma_indoor(device=device)
        else:
            roma_model = roma_outdoor(device=device)
        roma_model.upsample_preds = False
        roma_model.symmetric = False

    M = cfg.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    scaling_factor = cfg.scaling_factor
    expansion_factor = 1
    keypoint_fit_error_tolerance = cfg.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(cfg.num_refs, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = 1  # Only ONE neighbor now!

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []

    # Dummy first pass to initialize model
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    # Main Loop over source_idx
    for source_idx in tqdm(sorted(selected_indices), desc="Profiling source frames"):

        # =================== Step 1: Compute Warp and Certainty ===================
        start = time.time()
        viewpoint_cam1 = viewpoint_stack[source_idx]
        NNs=closest_indices_selected.shape[1]
        viewpoint_cam2 = viewpoint_stack[closest_indices_selected[source_idx, np.random.randint(NNs)]]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)

        certainties_max = certainty_warp  # New manual sampling
        timings['aggregation_warp_certainty'].append(time.time() - start)

        # =================== Step 2: Good Samples Selection ===================
        start = time.time()
        certainty = certainties_max.reshape(-1).clone()
        certainty[certainty > upper_thresh] = 1
        good_samples = torch.multinomial(certainty, num_samples=min(expansion_factor * M, len(certainty)), replacement=False)
        timings['good_samples_selection'].append(time.time() - start)

        # =================== Step 3: Triangulate Keypoints ===================
        reference_image_dict = {
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []
        }

        start = time.time()
        matches_NN = warp.reshape(-1, 4)[good_samples]

        # Convert matches to pixel coordinates
        kptsA_np, kptsB_np, kptsA_color, kptsB_color = extract_keypoints_and_colors_single(
            np.array(imA).astype(np.uint8), 
            np.array(imB).astype(np.uint8), 
            matches_NN, 
            roma_model
        )

        proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
        proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, 0]].full_proj_transform

        triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
            P1=torch.stack([proj_matrices_A] * M, axis=0),
            P2=torch.stack([proj_matrices_B] * M, axis=0),
            k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
            k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

        reference_image_dict["triangulated_points"].append(triangulated_points)
        reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
        reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)
        timings['triangulation_per_NN'].append(time.time() - start)

        # =================== Step 4: Select Best Triangulated Points ===================
        start = time.time()
        NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
            NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
            NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
            NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))
        timings['select_best_keypoints'].append(time.time() - start)

        # =================== Step 5: Create New Gaussians ===================
        start = time.time()
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        new_xyz = NNs_triangulated_points_selected[:, :-1]
        all_new_xyz.append(new_xyz)
        all_new_features_dc.append(RGB2SH(torch.tensor(kptsA_color.astype(np.float32) / 255.)).unsqueeze(1))
        all_new_features_rest.append(torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0))

        mask_bad_points = torch.tensor(
            NNs_triangulated_points_selected_proj_errors > keypoint_fit_error_tolerance,
            dtype=torch.float32).unsqueeze(1).to(device)

        all_new_opacities.append(torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0) * 0. - mask_bad_points * (1e1))

        dist_points_to_cam1 = torch.linalg.norm(viewpoint_cam1.camera_center.clone().detach() - new_xyz, dim=1, ord=2)
        all_new_scaling.append(gaussians.scaling_inverse_activation((dist_points_to_cam1 * scaling_factor).unsqueeze(1).repeat(1, 3)))
        all_new_rotation.append(torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0))
        timings['save_gaussians'].append(time.time() - start)

    # =================== Final Densification Postfix ===================
    start = time.time()
    all_new_xyz = torch.cat(all_new_xyz, dim=0) 
    all_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    new_tmp_radii = torch.zeros(all_new_xyz.shape[0])
    prune_mask = torch.ones(all_new_xyz.shape[0], dtype=torch.bool)

    gaussians.densification_postfix(
        all_new_xyz[prune_mask].to(device),
        all_new_features_dc[prune_mask].to(device),
        torch.cat(all_new_features_rest, dim=0)[prune_mask].to(device),
        torch.cat(all_new_opacities, dim=0)[prune_mask].to(device),
        torch.cat(all_new_scaling, dim=0)[prune_mask].to(device),
        torch.cat(all_new_rotation, dim=0)[prune_mask].to(device)
    )
    timings['final_densification_postfix'].append(time.time() - start)

    # =================== Print Profiling Results ===================
    print("\n=== Profiling Summary (average per frame) ===")
    for key, times in timings.items():
        print(f"{key:35s}: {sum(times) / len(times):.4f} sec (total {sum(times):.2f} sec)")

    return viewpoint_stack, closest_indices_selected, visualizations