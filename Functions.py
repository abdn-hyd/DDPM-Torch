import math
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import glob
import re

DEFAULT_DTYPE = torch.float32


@torch.jit.script
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embed_dim: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(
        -torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed
    )
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed


def create_images_grid(
    images: np.ndarray,
    rows: int,
    cols: int,
):
    """
    Args:
        images (np.ndarray): Array of images to be arranged in a grid.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    images = [Image.fromarray(image) for image in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def create_denoising_gif(
    input_folder: str, 
    output_path: str, 
    duration: int = 100,
    reverse_order: bool = False,
):
    """
    Args:
        input_folder (str): Path to the folder containing generated images.
        output_path (str): Path to save the generated GIF.
        duration (int): Duration of each frame in milliseconds.
        reverse_order (bool): If True, reverse the order of frames.
    """
    files = glob.glob(os.path.join(input_folder, "generated_images_*.png"))

    def extract_number(file_path):
        match = re.search(r"generated_images_(\d+)\.png", file_path)
        return int(match.group(1)) if match else -1

    files.sort(key=extract_number, reverse=not reverse_order)
    if not files:
        raise ValueError(f"No matching images found in {input_folder}")
    frames = [Image.open(f) for f in files]
    frames[0].save(
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    return output_path


if __name__ == "__main__":
    # Example usage
    input_folder = "/Users/gunneo/ai/codes/Diffusion/outputs/samples/epoch_140"
    output_path = "/Users/gunneo/ai/codes/Diffusion/outputs/samples/epoch_140/denoising_process.gif"
    create_denoising_gif(input_folder, output_path, duration=100, reverse_order=False)
    print(f"GIF saved at {output_path}")
