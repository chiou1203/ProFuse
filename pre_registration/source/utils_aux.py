# Perlin noise code taken from https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
from types import SimpleNamespace
import random
import numpy as np
import torch
import torchvision
import wandb
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from PIL import Image

def parse_dict_to_namespace(dict_nested):
    """Turns nested dictionary into nested namespaces"""
    if type(dict_nested) != dict and type(dict_nested) != list: return dict_nested
    x = SimpleNamespace()
    for key, val in dict_nested.items():
        if type(val) == dict:
            setattr(x, key, parse_dict_to_namespace(val))
        elif type(val) == list:
            setattr(x, key, [parse_dict_to_namespace(v) for v in val])
        else:
            setattr(x, key, val)
    return x

def set_seed(seed=42, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)



def log_samples(samples, scores, iteration, caption="Real Samples"):
    # Create a grid of images
    grid = torchvision.utils.make_grid(samples)

    # Log the images and scores to wandb
    wandb.log({
        f"{caption}_images": [wandb.Image(grid, caption=f"{caption}: {scores}")],
    }, step = iteration)



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

def process_image(image_tensor):
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.clip(image_np * 255, 0, 255).astype(np.uint8))


def normalize_keypoints(kpts_np, width, height):
    kpts_np[:, 0] = kpts_np[:, 0] / width * 2. - 1.
    kpts_np[:, 1] = kpts_np[:, 1] / height * 2. - 1.
    return kpts_np