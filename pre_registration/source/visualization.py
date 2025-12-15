from matplotlib import pyplot as plt
import numpy as np
import torch

import numpy as np
from typing import List
import sys
sys.path.append('./submodules/gaussian-splatting/')
from scene.cameras import Camera
from PIL import Image
import imageio
from scipy.interpolate import splprep, splev

import cv2
import numpy as np
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from typing import List
from sklearn.mixture import GaussianMixture

def render_gaussians_rgb(generator3DGS, viewpoint_cam, visualize=False):
    """
    Simply render gaussians from the generator3DGS from the viewpoint_cam.
    Args:
        generator3DGS : instance of the Generator3DGS class from the networks.py file
        viewpoint_cam : camera instance
        visualize : boolean flag. If True, will call pyplot function and render image inplace
    Returns:
        uint8 numpy array with shape (H, W, 3) representing the image
    """
    with torch.no_grad():
        render_pkg = generator3DGS(viewpoint_cam)
        image = render_pkg["render"]
        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            plt.show()

        return image_np

def render_gaussians_D_scores(generator3DGS, viewpoint_cam, mask=None, mask_channel=0, visualize=False):
    """
        Simply render D_scores of gaussians from the generator3DGS from the viewpoint_cam.
        Args:
            generator3DGS : instance of the Generator3DGS class from the networks.py file
            viewpoint_cam : camera instance
            visualize : boolean flag. If True, will call pyplot function and render image inplace
            mask : optional mask to highlight specific gaussians. Must be of shape (N) where N is the numnber
                of gaussians in generator3DGS.gaussians. Must be a torch tensor of floats, please scale according
                to how much color you want to have. Recommended mask value is 10.
            mask_channel: to which color channel should we add mask
        Returns:
            uint8 numpy array with shape (H, W, 3) representing the generator3DGS.gaussians.D_scores rendered as colors
        """
    with torch.no_grad():
        # Visualize D_scores
        generator3DGS.gaussians._features_dc = generator3DGS.gaussians._features_dc * 1e-4 + \
                                               torch.stack([generator3DGS.gaussians.D_scores] * 3, axis=-1)
        generator3DGS.gaussians._features_rest = generator3DGS.gaussians._features_rest * 1e-4
        if mask is not None:
            generator3DGS.gaussians._features_dc[..., mask_channel] += mask.unsqueeze(-1)
        render_pkg = generator3DGS(viewpoint_cam)
        image = render_pkg["render"]
        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            plt.show()

        if mask is not None:
            generator3DGS.gaussians._features_dc[..., mask_channel] -= mask.unsqueeze(-1)

        generator3DGS.gaussians._features_dc = (generator3DGS.gaussians._features_dc - \
                                                     torch.stack([generator3DGS.gaussians.D_scores] * 3, axis=-1)) * 1e4
        generator3DGS.gaussians._features_rest = generator3DGS.gaussians._features_rest * 1e4

        return image_np
    


def normalize(v):
    """
    Normalize a vector to unit length.

    Parameters:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Unit vector in the same direction as `v`.
    """
    return v / np.linalg.norm(v)

def look_at_rotation(camera_position: np.ndarray, target: np.ndarray, world_up=np.array([0, 1, 0])):
    """
    Compute a rotation matrix for a camera looking at a target point.

    Parameters:
        camera_position (np.ndarray): The 3D position of the camera.
        target (np.ndarray): The point the camera should look at.
        world_up (np.ndarray): A vector that defines the global 'up' direction.

    Returns:
        np.ndarray: A 3x3 rotation matrix (camera-to-world) with columns [right, up, forward].
    """
    z_axis = normalize(target - camera_position)         # Forward direction
    x_axis = normalize(np.cross(world_up, z_axis))       # Right direction
    y_axis = np.cross(z_axis, x_axis)                    # Recomputed up
    return np.stack([x_axis, y_axis, z_axis], axis=1)

    
def generate_circular_camera_path(existing_cameras: List[Camera], N: int = 12, radius_scale: float = 1.0, d: float = 2.0) -> List[Camera]:
    """
    Generate a circular path of cameras around an existing camera group, 
    with each new camera oriented to look at the average viewing direction.

    Parameters:
        existing_cameras (List[Camera]): List of existing camera objects to estimate average orientation and layout.
        N (int): Number of new cameras to generate along the circular path.
        radius_scale (float): Scale factor to adjust the radius of the circle.
        d (float): Distance ahead of each camera used to estimate its look-at point.

    Returns:
        List[Camera]: A list of newly generated Camera objects forming a circular path and oriented toward a shared view center.
    """
    # Step 1: Compute average camera position
    center = np.mean([cam.T for cam in existing_cameras], axis=0)

    # Estimate where each camera is looking
    # d denotes how far ahead each camera sees — you can scale this
    look_targets = [cam.T + cam.R[:, 2] * d for cam in existing_cameras]
    center_of_view = np.mean(look_targets, axis=0)

    # Step 2: Define circular plane basis using fixed up vector
    avg_forward = normalize(np.mean([cam.R[:, 2] for cam in existing_cameras], axis=0))
    up_guess = np.array([0, 1, 0])
    right = normalize(np.cross(avg_forward, up_guess))
    up = normalize(np.cross(right, avg_forward))

    # Step 3: Estimate radius
    avg_radius = np.mean([np.linalg.norm(cam.T - center) for cam in existing_cameras]) * radius_scale

    # Step 4: Create cameras on a circular path
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reference_cam = existing_cameras[0]
    new_cameras = []

    
    for i, a in enumerate(angles):
        position = center + avg_radius * (np.cos(a) * right + np.sin(a) * up)

        if d < 1e-5 or radius_scale < 1e-5:
            # Use same orientation as the first camera
            R = reference_cam.R.copy()
        else:
            # Change orientation
            R = look_at_rotation(position, center_of_view)
        new_cameras.append(Camera(
            R=R, 
            T=position,                                   # New position
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"circular_a={a:.3f}",
            uid=i
        ))

    return new_cameras


def save_numpy_frames_as_gif(frames, output_path="animation.gif", duration=100):
    """
    Save a list of RGB NumPy frames as a looping GIF animation.

    Parameters:
        frames (List[np.ndarray]): List of RGB images as uint8 NumPy arrays (shape HxWx3).
        output_path (str): Path to save the output GIF.
        duration (int): Duration per frame in milliseconds.

    Returns:
        None
    """
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,  # duration per frame in ms
        loop=0
    )
    print(f"GIF saved to: {output_path}")

def center_crop_frame(frame: np.ndarray, crop_fraction: float) -> np.ndarray:
    """
    Crop the central region of the frame by the given fraction.

    Parameters:
        frame (np.ndarray): Input RGB image (H, W, 3).
        crop_fraction (float): Fraction of the original size to retain (e.g., 0.8 keeps 80%).

    Returns:
        np.ndarray: Cropped RGB image.
    """
    if crop_fraction >= 1.0:
        return frame

    h, w, _ = frame.shape
    new_h, new_w = int(h * crop_fraction), int(w * crop_fraction)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    return frame[start_y:start_y + new_h, start_x:start_x + new_w, :]



def generate_smooth_closed_camera_path(existing_cameras: List[Camera], N: int = 120, d: float = 2.0, s=.25) -> List[Camera]:
    """
    Generate a smooth, closed path interpolating the positions of existing cameras.

    Parameters:
        existing_cameras (List[Camera]): List of existing cameras.
        N (int): Number of points (cameras) to sample along the smooth path.
        d (float): Distance ahead for estimating the center of view.

    Returns:
        List[Camera]: A list of smoothly moving Camera objects along a closed loop.
    """
    # Step 1: Extract camera positions
    positions = np.array([cam.T for cam in existing_cameras])
    
    # Step 2: Estimate center of view
    look_targets = [cam.T + cam.R[:, 2] * d for cam in existing_cameras]
    center_of_view = np.mean(look_targets, axis=0)

    # Step 3: Fit a smooth closed spline through the positions
    positions = np.vstack([positions, positions[0]])  # close the loop
    tck, u = splprep(positions.T, s=s, per=True)  # periodic=True for closed loop

    # Step 4: Sample points along the spline
    u_fine = np.linspace(0, 1, N)
    smooth_path = np.stack(splev(u_fine, tck), axis=-1)

    # Step 5: Generate cameras along the smooth path
    reference_cam = existing_cameras[0]
    new_cameras = []

    for i, pos in enumerate(smooth_path):
        R = look_at_rotation(pos, center_of_view)
        new_cameras.append(Camera(
            R=R,
            T=pos,
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"smooth_path_i={i}",
            uid=i
        ))

    return new_cameras


def save_numpy_frames_as_mp4(frames, output_path="animation.mp4", fps=10, center_crop: float = 1.0):
    """
    Save a list of RGB NumPy frames as an MP4 video with optional center cropping.

    Parameters:
        frames (List[np.ndarray]): List of RGB images as uint8 NumPy arrays (shape HxWx3).
        output_path (str): Path to save the output MP4.
        fps (int): Frames per second for playback speed.
        center_crop (float): Fraction (0 < center_crop <= 1.0) of central region to retain. 
                             Use 1.0 for no cropping; 0.8 to crop to 80% center region.

    Returns:
        None
    """
    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            cropped = center_crop_frame(frame, center_crop)
            writer.append_data(cropped)
    print(f"MP4 saved to: {output_path}")


    
def put_text_on_image(img: np.ndarray, text: str) -> np.ndarray:
    """
    Draws multiline white text on a copy of the input image, positioned near the bottom
    and around 80% of the image width. Handles '\n' characters to split text into multiple lines.

    Args:
        img (np.ndarray): Input image as a (H, W, 3) uint8 numpy array.
        text (str): Text string to draw on the image. Newlines '\n' are treated as line breaks.

    Returns:
        np.ndarray: The output image with the text drawn on it.
    
    Notes:
        - The function automatically adjusts line spacing and prevents text from going outside the image.
        - Text is drawn in white with small font size (0.5) for minimal visual impact.
    """
    img = img.copy()
    height, width, _ = img.shape
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.
    color = (255, 255, 255)
    thickness = 2
    line_spacing = 5  # extra pixels between lines
    
    lines = text.split('\n')
    
    # Precompute the maximum text width to adjust starting x
    max_text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    
    x = int(0.8 * width)
    x = min(x, width - max_text_width - 30)  # margin on right
    #x = int(0.03 * width)
    
    # Start near the bottom, but move up depending on number of lines
    total_text_height = len(lines) * (cv2.getTextSize('A', font, font_scale, thickness)[0][1] + line_spacing)
    y_start = int(height*0.9) - total_text_height  # 30 pixels from bottom

    for i, line in enumerate(lines):
        y = y_start + i * (cv2.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing)
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img




def catmull_rom_spline(P0, P1, P2, P3, n_points=20):
    """
    Compute Catmull-Rom spline segment between P1 and P2.
    """
    t = np.linspace(0, 1, n_points)[:, None]

    M = 0.5 * np.array([
        [-1,  3, -3, 1],
        [ 2, -5,  4, -1],
        [-1,  0,  1, 0],
        [ 0,  2,  0, 0]
    ])

    G = np.stack([P0, P1, P2, P3], axis=0)
    T = np.concatenate([t**3, t**2, t, np.ones_like(t)], axis=1)

    return T @ M @ G

def sort_cameras_pca(existing_cameras: List[Camera]):
    """
    Sort cameras along the main PCA axis.
    """
    positions = np.array([cam.T for cam in existing_cameras])
    pca = PCA(n_components=1)
    scores = pca.fit_transform(positions)
    sorted_indices = np.argsort(scores[:, 0])
    return sorted_indices

def generate_fully_smooth_cameras(existing_cameras: List[Camera], 
                                  n_selected: int = 30, 
                                  n_points_per_segment: int = 20, 
                                  d: float = 2.0,
                                  closed: bool = False) -> List[Camera]:
    """
    Generate a fully smooth camera path using PCA ordering, global Catmull-Rom spline for positions, and global SLERP for orientations.

    Args:
        existing_cameras (List[Camera]): List of input cameras.
        n_selected (int): Number of cameras to select after sorting.
        n_points_per_segment (int): Number of interpolated points per spline segment.
        d (float): Distance ahead for estimating center of view.
        closed (bool): Whether to close the path.

    Returns:
        List[Camera]: List of smoothly moving Camera objects.
    """
    # 1. Sort cameras along PCA axis
    sorted_indices = sort_cameras_pca(existing_cameras)
    sorted_cameras = [existing_cameras[i] for i in sorted_indices]
    positions = np.array([cam.T for cam in sorted_cameras])

    # 2. Subsample uniformly
    idx = np.linspace(0, len(positions) - 1, n_selected).astype(int)
    sampled_positions = positions[idx]
    sampled_cameras = [sorted_cameras[i] for i in idx]

    # 3. Prepare for Catmull-Rom
    if closed:
        sampled_positions = np.vstack([sampled_positions[-1], sampled_positions, sampled_positions[0], sampled_positions[1]])
    else:
        sampled_positions = np.vstack([sampled_positions[0], sampled_positions, sampled_positions[-1], sampled_positions[-1]])

    # 4. Generate smooth path positions
    path_positions = []
    for i in range(1, len(sampled_positions) - 2):
        segment = catmull_rom_spline(sampled_positions[i-1], sampled_positions[i], sampled_positions[i+1], sampled_positions[i+2], n_points_per_segment)
        path_positions.append(segment)
    path_positions = np.concatenate(path_positions, axis=0)

    # 5. Global SLERP for rotations
    rotations = R.from_matrix([cam.R for cam in sampled_cameras])
    key_times = np.linspace(0, 1, len(rotations))
    slerp = Slerp(key_times, rotations)

    query_times = np.linspace(0, 1, len(path_positions))
    interpolated_rotations = slerp(query_times)

    # 6. Generate Camera objects
    reference_cam = existing_cameras[0]
    smooth_cameras = []

    for i, pos in enumerate(path_positions):
        R_interp = interpolated_rotations[i].as_matrix()

        smooth_cameras.append(Camera(
            R=R_interp,
            T=pos,
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"fully_smooth_path_i={i}",
            uid=i
        ))

    return smooth_cameras


def plot_cameras_and_smooth_path_with_orientation(existing_cameras: List[Camera], smooth_cameras: List[Camera], scale: float = 0.1):
    """
    Plot input cameras and smooth path cameras with their orientations in 3D.

    Args:
        existing_cameras (List[Camera]): List of original input cameras.
        smooth_cameras (List[Camera]): List of smooth path cameras.
        scale (float): Length of orientation arrows.

    Returns:
        None
    """
    # Input cameras
    input_positions = np.array([cam.T for cam in existing_cameras])

    # Smooth cameras
    smooth_positions = np.array([cam.T for cam in smooth_cameras])

    fig = go.Figure()

    # Plot input camera positions
    fig.add_trace(go.Scatter3d(
        x=input_positions[:, 0], y=input_positions[:, 1], z=input_positions[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Input Cameras'
    ))

    # Plot smooth path positions
    fig.add_trace(go.Scatter3d(
        x=smooth_positions[:, 0], y=smooth_positions[:, 1], z=smooth_positions[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=2, color='red'),
        name='Smooth Path Cameras'
    ))

    # Plot input camera orientations
    for cam in existing_cameras:
        origin = cam.T
        forward = cam.R[:, 2]  # Forward direction

        fig.add_trace(go.Cone(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            u=[forward[0]], v=[forward[1]], w=[forward[2]],
            colorscale=[[0, 'blue'], [1, 'blue']],
            sizemode="absolute",
            sizeref=scale,
            anchor="tail",
            showscale=False,
            name='Input Camera Direction'
        ))

    # Plot smooth camera orientations
    for cam in smooth_cameras:
        origin = cam.T
        forward = cam.R[:, 2]  # Forward direction

        fig.add_trace(go.Cone(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            u=[forward[0]], v=[forward[1]], w=[forward[2]],
            colorscale=[[0, 'red'], [1, 'red']],
            sizemode="absolute",
            sizeref=scale,
            anchor="tail",
            showscale=False,
            name='Smooth Camera Direction'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Input Cameras and Smooth Path with Orientations",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


def solve_tsp_nearest_neighbor(points: np.ndarray):
    """
    Solve TSP approximately using nearest neighbor heuristic.

    Args:
        points (np.ndarray): (N, 3) array of points.

    Returns:
        List[int]: Optimal visiting order of points.
    """
    N = points.shape[0]
    dist = distance_matrix(points, points)
    visited = [0]
    unvisited = set(range(1, N))

    while unvisited:
        last = visited[-1]
        next_city = min(unvisited, key=lambda city: dist[last, city])
        visited.append(next_city)
        unvisited.remove(next_city)

    return visited

def solve_tsp_2opt(points: np.ndarray, n_iter: int = 1000) -> np.ndarray:
    """
    Solve TSP approximately using Nearest Neighbor + 2-Opt.

    Args:
        points (np.ndarray): Array of shape (N, D) with points.
        n_iter (int): Number of 2-opt iterations.

    Returns:
        np.ndarray: Ordered list of indices.
    """
    n_points = points.shape[0]

    # === 1. Start with Nearest Neighbor
    unvisited = list(range(n_points))
    current = unvisited.pop(0)
    path = [current]

    while unvisited:
        dists = np.linalg.norm(points[unvisited] - points[current], axis=1)
        next_idx = unvisited[np.argmin(dists)]
        unvisited.remove(next_idx)
        path.append(next_idx)
        current = next_idx

    # === 2. Apply 2-Opt improvements
    def path_length(path):
        return np.sum(np.linalg.norm(points[path[i]] - points[path[i+1]], axis=0) for i in range(len(path)-1))

    best_length = path_length(path)
    improved = True

    for _ in range(n_iter):
        if not improved:
            break
        improved = False
        for i in range(1, n_points - 2):
            for j in range(i + 1, n_points):
                if j - i == 1: continue
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_length = path_length(new_path)
                if new_length < best_length:
                    path = new_path
                    best_length = new_length
                    improved = True
                    break
            if improved:
                break

    return np.array(path)

def generate_fully_smooth_cameras_with_tsp(existing_cameras: List[Camera], 
                                           n_selected: int = 30, 
                                           n_points_per_segment: int = 20, 
                                           d: float = 2.0,
                                           closed: bool = False) -> List[Camera]:
    """
    Generate a fully smooth camera path using TSP ordering, global Catmull-Rom spline for positions, and global SLERP for orientations.

    Args:
        existing_cameras (List[Camera]): List of input cameras.
        n_selected (int): Number of cameras to select after ordering.
        n_points_per_segment (int): Number of interpolated points per spline segment.
        d (float): Distance ahead for estimating center of view.
        closed (bool): Whether to close the path.

    Returns:
        List[Camera]: List of smoothly moving Camera objects.
    """
    positions = np.array([cam.T for cam in existing_cameras])

    # 1. Solve approximate TSP
    order = solve_tsp_nearest_neighbor(positions)
    ordered_cameras = [existing_cameras[i] for i in order]
    ordered_positions = positions[order]

    # 2. Subsample uniformly
    idx = np.linspace(0, len(ordered_positions) - 1, n_selected).astype(int)
    sampled_positions = ordered_positions[idx]
    sampled_cameras = [ordered_cameras[i] for i in idx]

    # 3. Prepare for Catmull-Rom
    if closed:
        sampled_positions = np.vstack([sampled_positions[-1], sampled_positions, sampled_positions[0], sampled_positions[1]])
    else:
        sampled_positions = np.vstack([sampled_positions[0], sampled_positions, sampled_positions[-1], sampled_positions[-1]])

    # 4. Generate smooth path positions
    path_positions = []
    for i in range(1, len(sampled_positions) - 2):
        segment = catmull_rom_spline(sampled_positions[i-1], sampled_positions[i], sampled_positions[i+1], sampled_positions[i+2], n_points_per_segment)
        path_positions.append(segment)
    path_positions = np.concatenate(path_positions, axis=0)

    # 5. Global SLERP for rotations
    rotations = R.from_matrix([cam.R for cam in sampled_cameras])
    key_times = np.linspace(0, 1, len(rotations))
    slerp = Slerp(key_times, rotations)

    query_times = np.linspace(0, 1, len(path_positions))
    interpolated_rotations = slerp(query_times)

    # 6. Generate Camera objects
    reference_cam = existing_cameras[0]
    smooth_cameras = []

    for i, pos in enumerate(path_positions):
        R_interp = interpolated_rotations[i].as_matrix()

        smooth_cameras.append(Camera(
            R=R_interp,
            T=pos,
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"fully_smooth_path_i={i}",
            uid=i
        ))

    return smooth_cameras

from typing import List
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image

def generate_clustered_smooth_cameras_with_tsp(existing_cameras: List[Camera], 
                                                n_selected: int = 30, 
                                                n_points_per_segment: int = 20, 
                                                d: float = 2.0,
                                                n_clusters: int = 5,
                                                closed: bool = False) -> List[Camera]:
    """
    Generate a fully smooth camera path using clustering + TSP between nearest cluster centers + TSP inside clusters.
    Positions are normalized before clustering and denormalized before generating final cameras.

    Args:
        existing_cameras (List[Camera]): List of input cameras.
        n_selected (int): Number of cameras to select after ordering.
        n_points_per_segment (int): Number of interpolated points per spline segment.
        d (float): Distance ahead for estimating center of view.
        n_clusters (int): Number of GMM clusters.
        closed (bool): Whether to close the path.

    Returns:
        List[Camera]: Smooth path of Camera objects.
    """
    # Extract positions and rotations
    positions = np.array([cam.T for cam in existing_cameras])
    rotations = np.array([R.from_matrix(cam.R).as_quat() for cam in existing_cameras])

    # === Normalize positions
    mean_pos = np.mean(positions, axis=0)
    scale_pos = np.std(positions, axis=0)
    scale_pos[scale_pos == 0] = 1.0  # avoid division by zero

    positions_normalized = (positions - mean_pos) / scale_pos

    # === Features for clustering (only positions, not rotations)
    features = positions_normalized

    # === 1. GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(features)

    clusters = {}
    cluster_centers = []

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        clusters[cluster_id] = cluster_indices
        cluster_center = np.mean(features[cluster_indices], axis=0)
        cluster_centers.append(cluster_center)

    cluster_centers = np.stack(cluster_centers)

    # === 2. Remap cluster centers to nearest existing cameras
    if False:
        mapped_centers = []
        for center in cluster_centers:
            dists = np.linalg.norm(features - center, axis=1)
            nearest_idx = np.argmin(dists)
            mapped_centers.append(features[nearest_idx])
        mapped_centers = np.stack(mapped_centers)
        cluster_centers = mapped_centers
    # === 3. Solve TSP between mapped cluster centers
    cluster_order = solve_tsp_2opt(cluster_centers)

    # === 4. For each cluster, solve TSP inside cluster
    final_indices = []
    for cluster_id in cluster_order:
        cluster_indices = clusters[cluster_id]
        cluster_positions = features[cluster_indices]

        if len(cluster_positions) == 1:
            final_indices.append(cluster_indices[0])
            continue

        local_order = solve_tsp_nearest_neighbor(cluster_positions)
        ordered_cluster_indices = cluster_indices[local_order]
        final_indices.extend(ordered_cluster_indices)

    ordered_cameras = [existing_cameras[i] for i in final_indices]
    ordered_positions = positions_normalized[final_indices]

    # === 5. Subsample uniformly
    idx = np.linspace(0, len(ordered_positions) - 1, n_selected).astype(int)
    sampled_positions = ordered_positions[idx]
    sampled_cameras = [ordered_cameras[i] for i in idx]

    # === 6. Prepare for Catmull-Rom spline
    if closed:
        sampled_positions = np.vstack([sampled_positions[-1], sampled_positions, sampled_positions[0], sampled_positions[1]])
    else:
        sampled_positions = np.vstack([sampled_positions[0], sampled_positions, sampled_positions[-1], sampled_positions[-1]])

    # === 7. Smooth path positions
    path_positions = []
    for i in range(1, len(sampled_positions) - 2):
        segment = catmull_rom_spline(sampled_positions[i-1], sampled_positions[i], sampled_positions[i+1], sampled_positions[i+2], n_points_per_segment)
        path_positions.append(segment)
    path_positions = np.concatenate(path_positions, axis=0)

    # === 8. Denormalize
    path_positions = path_positions * scale_pos + mean_pos

    # === 9. SLERP for rotations
    rotations = R.from_matrix([cam.R for cam in sampled_cameras])
    key_times = np.linspace(0, 1, len(rotations))
    slerp = Slerp(key_times, rotations)

    query_times = np.linspace(0, 1, len(path_positions))
    interpolated_rotations = slerp(query_times)

    # === 10. Generate Camera objects
    reference_cam = existing_cameras[0]
    smooth_cameras = []

    for i, pos in enumerate(path_positions):
        R_interp = interpolated_rotations[i].as_matrix()

        smooth_cameras.append(Camera(
            R=R_interp,
            T=pos,
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"clustered_smooth_path_i={i}",
            uid=i
        ))

    return smooth_cameras


# def generate_clustered_path(existing_cameras: List[Camera], 
#                              n_points_per_segment: int = 20, 
#                              d: float = 2.0,
#                              n_clusters: int = 5,
#                              closed: bool = False) -> List[Camera]:
#     """
#     Generate a smooth camera path using GMM clustering and TSP on cluster centers.

#     Args:
#         existing_cameras (List[Camera]): List of input cameras.
#         n_points_per_segment (int): Number of interpolated points per spline segment.
#         d (float): Distance ahead for estimating center of view.
#         n_clusters (int): Number of GMM clusters (zones).
#         closed (bool): Whether to close the path.

#     Returns:
#         List[Camera]: Smooth path of Camera objects.
#     """
#     # Extract positions and rotations
#     positions = np.array([cam.T for cam in existing_cameras])

#     # === Normalize positions
#     mean_pos = np.mean(positions, axis=0)
#     scale_pos = np.std(positions, axis=0)
#     scale_pos[scale_pos == 0] = 1.0

#     positions_normalized = (positions - mean_pos) / scale_pos

#     # === 1. GMM clustering (only positions)
#     gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
#     cluster_labels = gmm.fit_predict(positions_normalized)

#     cluster_centers = []
#     for cluster_id in range(n_clusters):
#         cluster_indices = np.where(cluster_labels == cluster_id)[0]
#         if len(cluster_indices) == 0:
#             continue
#         cluster_center = np.mean(positions_normalized[cluster_indices], axis=0)
#         cluster_centers.append(cluster_center)

#     cluster_centers = np.stack(cluster_centers)

#     # === 2. Solve TSP between cluster centers
#     cluster_order = solve_tsp_2opt(cluster_centers)

#     # === 3. Reorder cluster centers
#     ordered_centers = cluster_centers[cluster_order]

#     # === 4. Prepare Catmull-Rom spline
#     if closed:
#         ordered_centers = np.vstack([ordered_centers[-1], ordered_centers, ordered_centers[0], ordered_centers[1]])
#     else:
#         ordered_centers = np.vstack([ordered_centers[0], ordered_centers, ordered_centers[-1], ordered_centers[-1]])

#     # === 5. Generate smooth path positions
#     path_positions = []
#     for i in range(1, len(ordered_centers) - 2):
#         segment = catmull_rom_spline(ordered_centers[i-1], ordered_centers[i], ordered_centers[i+1], ordered_centers[i+2], n_points_per_segment)
#         path_positions.append(segment)
#     path_positions = np.concatenate(path_positions, axis=0)

#     # === 6. Denormalize back
#     path_positions = path_positions * scale_pos + mean_pos

#     # === 7. Generate dummy rotations (constant forward facing)
#     reference_cam = existing_cameras[0]
#     default_rotation = R.from_matrix(reference_cam.R)

#     # For simplicity, fixed rotation for all
#     smooth_cameras = []

#     for i, pos in enumerate(path_positions):
#         R_interp = default_rotation.as_matrix()

#         smooth_cameras.append(Camera(
#             R=R_interp,
#             T=pos,
#             FoVx=reference_cam.FoVx,
#             FoVy=reference_cam.FoVy,
#             resolution=(reference_cam.image_width, reference_cam.image_height),
#             colmap_id=-1,
#             depth_params=None,
#             image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
#             invdepthmap=None,
#             image_name=f"cluster_path_i={i}",
#             uid=i
#         ))

#     return smooth_cameras

from typing import List
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image

def generate_clustered_path(existing_cameras: List[Camera], 
                             n_points_per_segment: int = 20, 
                             d: float = 2.0,
                             n_clusters: int = 5,
                             closed: bool = False) -> List[Camera]:
    """
    Generate a smooth camera path using K-Means clustering and TSP on cluster centers.

    Args:
        existing_cameras (List[Camera]): List of input cameras.
        n_points_per_segment (int): Number of interpolated points per spline segment.
        d (float): Distance ahead for estimating center of view.
        n_clusters (int): Number of KMeans clusters (zones).
        closed (bool): Whether to close the path.

    Returns:
        List[Camera]: Smooth path of Camera objects.
    """
    # Extract positions
    positions = np.array([cam.T for cam in existing_cameras])

    # === Normalize positions
    mean_pos = np.mean(positions, axis=0)
    scale_pos = np.std(positions, axis=0)
    scale_pos[scale_pos == 0] = 1.0

    positions_normalized = (positions - mean_pos) / scale_pos

    # === 1. K-Means clustering (only positions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(positions_normalized)

    cluster_centers = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_center = np.mean(positions_normalized[cluster_indices], axis=0)
        cluster_centers.append(cluster_center)

    cluster_centers = np.stack(cluster_centers)

    # === 2. Solve TSP between cluster centers
    cluster_order = solve_tsp_2opt(cluster_centers)

    # === 3. Reorder cluster centers
    ordered_centers = cluster_centers[cluster_order]

    # === 4. Prepare Catmull-Rom spline
    if closed:
        ordered_centers = np.vstack([ordered_centers[-1], ordered_centers, ordered_centers[0], ordered_centers[1]])
    else:
        ordered_centers = np.vstack([ordered_centers[0], ordered_centers, ordered_centers[-1], ordered_centers[-1]])

    # === 5. Generate smooth path positions
    path_positions = []
    for i in range(1, len(ordered_centers) - 2):
        segment = catmull_rom_spline(ordered_centers[i-1], ordered_centers[i], ordered_centers[i+1], ordered_centers[i+2], n_points_per_segment)
        path_positions.append(segment)
    path_positions = np.concatenate(path_positions, axis=0)

    # === 6. Denormalize back
    path_positions = path_positions * scale_pos + mean_pos

    # === 7. Generate dummy rotations (constant forward facing)
    reference_cam = existing_cameras[0]
    default_rotation = R.from_matrix(reference_cam.R)

    # For simplicity, fixed rotation for all
    smooth_cameras = []

    for i, pos in enumerate(path_positions):
        R_interp = default_rotation.as_matrix()

        smooth_cameras.append(Camera(
            R=R_interp,
            T=pos,
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"cluster_path_i={i}",
            uid=i
        ))

    return smooth_cameras




def visualize_image_with_points(image, points):
    """
    Visualize an image with points overlaid on top. This is useful for correspondences visualizations

    Parameters:
    - image: PIL Image object
    - points: Numpy array of shape [N, 2] containing (x, y) coordinates of points

    Returns:
    - None (displays the visualization)
    """

    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7,7))

    # Display the image
    ax.imshow(img_array)

    # Scatter plot the points on top of the image
    ax.scatter(points[:, 0], points[:, 1], color='red', marker='o', s=1)

    # Show the plot
    plt.show()


def visualize_correspondences(image1, points1, image2, points2):
    """
    Visualize two images concatenated horizontally with key points and correspondences.

    Parameters:
    - image1: PIL Image object (left image)
    - points1: Numpy array of shape [N, 2] containing (x, y) coordinates of key points for image1
    - image2: PIL Image object (right image)
    - points2: Numpy array of shape [N, 2] containing (x, y) coordinates of key points for image2

    Returns:
    - None (displays the visualization)
    """

    # Concatenate images horizontally
    concatenated_image = np.concatenate((np.array(image1), np.array(image2)), axis=1)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10,10))

    # Display the concatenated image
    ax.imshow(concatenated_image)

    # Plot key points on the left image
    ax.scatter(points1[:, 0], points1[:, 1], color='red', marker='o', s=10)

    # Plot key points on the right image
    ax.scatter(points2[:, 0] + image1.width, points2[:, 1], color='blue', marker='o', s=10)

    # Draw lines connecting corresponding key points
    for i in range(len(points1)):
        ax.plot([points1[i, 0], points2[i, 0] + image1.width], [points1[i, 1], points2[i, 1]])#, color='green')

    # Show the plot
    plt.show()

