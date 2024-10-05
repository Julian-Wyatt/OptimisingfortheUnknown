# Originally forked from
# https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

from abc import abstractmethod
import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from models.attention import SpatialTransformer
from utils.util import conv_nd, zero_module, normalisation, avg_pool_nd, checkpoint, LayerNorm


# dummy replace
def convert_module_to_f16(param):
    """
    Convert primitive modules to float16.
    """
    if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        param.weight.data = param.weight.data.half()
        if param.bias is not None:
            param.bias.data = param.bias.data.half()


def convert_module_to_f32(param):
    """
    Convert primitive modules to float32.
    """
    if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        param.weight.data = param.weight.data.float()
        if param.bias is not None:
            param.bias.data = param.bias.data.float()



## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, kernel_size=3, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, kernel_size, padding=padding)

    def forward(self, x):
        assert x.shape[
                   1] == self.channels, f"Expected {self.channels} channels, got shape {x.shape}"
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, kernel_size=3, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalisation(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalisation(self.out_channels),
            # Swish6(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        with torch.cuda.amp.autocast():
            if self.updown:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                h = in_rest(x)
                h = self.h_upd(h)
                x = self.x_upd(x)
                h = in_conv(h)
            else:
                h = self.in_layers(x)
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
            return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalisation(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        # split qkv before split heads
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(),
                          True)  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        with torch.cuda.amp.autocast():
            b, c, *spatial = x.shape
            x = x.reshape(b, c, -1)
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
            return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class ResisdualConvBlock(nn.Module):
    def __init__(self, channels, dropout, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        if channels == out_channels:
            self.skip_conv = nn.Identity()
        else:
            self.skip_conv = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return self.skip_conv(x) + h


class ConvNeXtBlock(nn.Module):
    """
    https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
    A convnext block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            dropout,
            embed_channels=None,
            out_channels=None,
            dims=2,
            use_checkpoint=False,
            channel_mult=4,
            kernel_size=7,
            dilation=1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        if embed_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    embed_channels,
                    self.out_channels,
                ),
            )
        if dilation == 2:
            padding *= 2
        self.in_conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size, padding=padding,
                                 dilation=dilation, groups=channels)

        # self.group_norm = normalisation(self.out_channels)
        self.layer_norm = LayerNorm(self.out_channels, data_format="channels_last")
        self.linear1 = nn.Linear(self.out_channels, self.out_channels * channel_mult)
        self.act = nn.GELU()
        self.GRN = GRN(self.out_channels * channel_mult)
        self.linear2 = nn.Linear(self.out_channels * channel_mult, self.out_channels)
        # self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        # elif use_conv:
        #     self.skip_connection = conv_nd(
        #         dims, channels, self.out_channels, 3, padding=1
        #     )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None):
        h = self.in_conv(x)
        if hasattr(self, 'emb_layers'):
            emb = self.emb_layers(emb)
            h = h + emb
        h = h.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        h = self.layer_norm(h)
        h = self.linear1(h)
        h = self.act(h)
        h = self.GRN(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = h.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # h = self.dropout(h)
        return self.drop_path(self.skip_connection(x)) + h


class UpBlock(TimestepBlock):
    def __init__(self, in_channels, skip_channels, out_channels, dropout, conv_next_channel_mult, num_blocks=2,
                 up_sample=True, kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv_next_channel_mult = conv_next_channel_mult

        if up_sample:
            self.up_sample = nn.Sequential(
                LayerNorm(self.in_channels, eps=1e-6, data_format="channels_first"),
                Upsample(self.in_channels, True, dims=2, kernel_size=3, padding=1, out_channels=self.in_channels)
            )
        else:
            self.up_sample = nn.Identity()

        if in_channels != out_channels:
            self.inner_skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.inner_skip_conv = nn.Identity()

        if skip_channels + in_channels != out_channels:
            self.skip_conv = ResisdualConvBlock(channels=skip_channels + in_channels, dropout=max(dropout) + 0.05,
                                                out_channels=out_channels)
        else:
            self.skip_conv = nn.Identity()
        layers = []

        for i in range(num_blocks):
            layers.append(
                ConvNeXtBlock(channels=out_channels, dropout=dropout[i],
                              out_channels=out_channels,
                              channel_mult=conv_next_channel_mult, kernel_size=kernel_size)
            )

        self.conv_blocks = TimestepEmbedSequential(*layers)


    def forward(self, x, skip=None, emb=None):
        x = self.up_sample(x)
        inner_skip = self.inner_skip_conv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.skip_conv(x)
        x = self.conv_blocks(x, emb=emb)
        return x + inner_skip


def two_d_softmax(x: torch.Tensor):
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    x_exp = th.exp(x)
    x_sum = th.sum(x_exp, dim=(2, 3), keepdim=True)
    return x_exp / x_sum


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim, shape=2):
        super().__init__()
        if shape == 2:
            g_zero = torch.zeros(1, dim)
            b_zero = torch.zeros(1, dim)
        elif shape == 4:
            g_zero = torch.zeros(1, 1, 1, dim)
            b_zero = torch.zeros(1, 1, 1, dim)
        else:
            raise ValueError(f"Invalid shape: {shape}")
        self.gamma = nn.Parameter(g_zero)
        self.beta = nn.Parameter(b_zero)

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x