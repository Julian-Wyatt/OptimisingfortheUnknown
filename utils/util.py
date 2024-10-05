import math
import os
from inspect import isfunction

import imageio
import torch
import numpy as np
from einops import repeat
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm import tqdm
import torch.cuda.amp as amp


def count_params(model, verbose=False):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d



def conv_nd(dims, *args, **kwargs) -> nn.Module:
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def normalisation(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def avg_pool_nd(dims, *args, **kwargs):
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GRN(nn.Module):
    """
    CONVNEXT V2
    GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm(nn.Module):
    """
    CONVNEXT V2
    https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_timestep_embedding(timesteps, embedding_dim):
    """
    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L698
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def invert_coordinate_embeddings(embeddings, embedding_matrix):
    """

    :param embeddings: embeded tensor of shape [b, k, 2, d]
    :param embedding_matrix: nn.Embedding of shape [k, d]
    :return:
    """
    embedding_matrix = embedding_matrix.clone()
    while len(embeddings.shape) + 1 > len(embedding_matrix.shape):
        embedding_matrix = embedding_matrix[:, None, :]
    distance = torch.norm(embedding_matrix - embeddings, dim=-1)
    return torch.argmin(distance, dim=0)


def checkpoint(func, inputs, params, flag):
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# class CheckpointFunction(torch.autograd.Function):
#     # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#
#         with torch.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors
#
#     @staticmethod
#     @amp.custom_bwd
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with torch.enable_grad():
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             with amp.autocast():
#                 output_tensors = ctx.run_function(*shallow_copies)
#         input_grads = torch.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + ctx.input_params,
#             output_grads,
#             allow_unused=True,
#         )
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + input_grads

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # add this
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @torch.cuda.amp.custom_bwd  # add this
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def output_video(tensor, path="output.mp4", fps=None):
    """
    :param tensor: video tensor of shape [t,c,h,w] for t timesteps, c channels, h height, w width
    :param path: output video path
    :param fps: frames per second of video
    """

    if fps is None:
        fps = 8
    with imageio.get_writer(path, fps=fps) as writer:  # Adjust fps as needed
        for frame_tensor in tensor:
            frame_array = frame_tensor.permute(1, 2, 0).clip(0, 255).cpu().numpy().astype(np.uint8)
            writer.append_data(frame_array)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class noVal_testProgressBar(TQDMProgressBar):
    """https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning"""

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            disable=True,
        )
        return bar


class noProgressBar(noVal_testProgressBar):
    """https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning"""

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            disable=True,
        )
        return bar


def plot_variance_schedule(timesteps=1000):
    import matplotlib.pyplot as plt
    t = np.array(range(timesteps)) / timesteps
    for variance in ["linear", "cosine", "sqrt_linear", "sqrt"]:
        betas = make_beta_schedule(variance, timesteps)
        alphas = 1 - betas
        alpha_bar = np.cumprod(alphas)
        plt.plot(alpha_bar, t, label=variance)
    plt.legend()
    plt.title(f"Variance Schedule - t={timesteps}")
    plt.show()


def save_to_csv(saving_root_dir, id, coordinates_all_batch, batch, total_landmarks):
    file_path = f"{saving_root_dir}/tmp/{id}/test_landmarks.csv"
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            header = ["image file"] + [f"p{i + 1}x,p{i + 1}y" for i in range(total_landmarks)]
            f.write(",".join(header) + "\n")

    for i in range(len(batch["name"])):
        with open(file_path, "a") as f:
            coordinates = coordinates_all_batch[i]
            output = [f"{int(batch['name'][i]):03d}.bmp"] + [str(i) for i in
                                                             coordinates.flatten().tolist()]
            f.write(",".join(output) + "\n")


if __name__ == "__main__":
    for timestep in [10, 50, 100, 500, 1000]:
        plot_variance_schedule(timestep)
