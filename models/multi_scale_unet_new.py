# Originally forked from
# https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

import torch
import torch as th
import torch.nn as nn
from timm.layers import DropPath
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile
from torchinfo import summary
from torchvision.transforms.v2 import Resize

from models.attention import SpatialSelfAttention
from models.unet_utils import TimestepEmbedSequential, Downsample, Upsample, \
    TimestepBlock
from utils.util import conv_nd, normalisation, GRN, checkpoint, LayerNorm
import torch.nn.functional as F
import torchvision.transforms.functional as F_vis


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


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
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
            out_channels=None,
            embed_channels=None,
            use_conv=True,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalisation(channels),
            # LayerNorm(self.out_channels, data_format="channels_first"),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        if embed_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    embed_channels,
                    self.out_channels,
                ),
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

        self.out_layers = nn.Sequential(
            normalisation(self.out_channels),
            nn.SiLU(),
            # nn.Dropout(p=dropout),
            nn.Dropout2d(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)

        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None):
        with torch.cuda.amp.autocast():
            if self.updown:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                h = in_rest(x)
                h = self.h_upd(h)
                x = self.x_upd(x)
                h = in_conv(h)
            else:
                h = self.in_layers(x)
            if hasattr(self, 'emb_layers'):
                emb = self.emb_layers(emb)
                h = h + emb

            h = self.out_layers(h)
            return self.skip_connection(x) + h


class DropChannelButNotBoth(nn.Module):
    def __init__(self, prob: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob

    def forward(self, x):
        if self.prob == 0 or not self.training:
            return x
        rand_val = th.rand(x.shape[1], device=x.device)
        if th.all(rand_val < self.prob):
            x[:, th.randint(0, x.shape[1], (1,), device=x.device)] = 0
        else:
            x[:, rand_val < self.prob] = 0
        return x


class InceptionNeXtBlock(nn.Module):
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
            out_channels=None,
            dims=2,
            use_checkpoint=False,
            channel_mult=2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.conv7 = conv_nd(dims, channels, self.out_channels * channel_mult, kernel_size=7, padding=3)
        self.conv5 = conv_nd(dims, channels, self.out_channels * channel_mult, kernel_size=5, padding=2)
        self.conv3 = conv_nd(dims, channels, self.out_channels * channel_mult, kernel_size=3, padding=1)
        # stack
        # group norm
        self.group_norm = normalisation(self.out_channels * 3 * channel_mult)
        self.act = nn.SiLU()

        self.out_conv = nn.Sequential(
            conv_nd(dims, self.out_channels * channel_mult * 3, self.out_channels * channel_mult, 3, padding=1),
            normalisation(self.out_channels * channel_mult),
            nn.SiLU(),
            conv_nd(dims, self.out_channels * channel_mult, self.out_channels, 1, padding=0)
        )
        self.dropout = nn.Dropout(dropout + 0.1)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        # instead
        # in conv with high kernel size
        # layer norm
        # upsize conv kernel 1 padding 0 to 4x
        # gelu
        # grn
        # downsize conv kernel 3 padding 1 to 1x

        h_7 = self.conv7(x)
        h_5 = self.conv5(x)
        h_3 = self.conv3(x)
        h = torch.cat([h_7, h_5, h_3], dim=1)
        h = self.group_norm(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.out_conv(h)

        return self.drop_path(self.skip_connection(x)) + h


class CustomConvNeXtBlock(nn.Module):
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
            padding=3,
    ):
        super().__init__()
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

        self.in_conv = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)

        self.group_norm = normalisation(self.out_channels)
        # self.layer_norm = LayerNorm(self.out_channels, data_format="channels_first")
        self.mid_conv = conv_nd(dims, self.out_channels, self.out_channels * channel_mult, 1, padding=0)
        self.act = nn.SiLU()
        # self.GRN = GRN(self.out_channels * channel_mult)
        self.out_conv = conv_nd(dims, self.out_channels * channel_mult, self.out_channels, 3, padding=1)
        # self.dropout = nn.Dropout(dropout + 0.1)
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
        # instead
        # in conv with high kernel size
        # layer norm
        # upsize conv kernel 1 padding 0 to 4x
        # gelu
        # grn
        # downsize conv kernel 3 padding 1 to 1x

        h = self.in_conv(x)
        if hasattr(self, 'emb_layers'):
            emb = self.emb_layers(emb)
            h = h + emb
        h = self.group_norm(h)
        h = self.mid_conv(h)
        h = self.act(h)
        # h = self.GRN(h)
        h = self.dropout(h)
        h = self.out_conv(h)

        return self.drop_path(self.skip_connection(x)) + h


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


class DownConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, use_lap_pyramid, dropout, conv_next_channel_mult, num_blocks=2, out_channels=None,
                 use_attn=False, use_dilation=False, use_downsample=True):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if use_downsample:
            self.down_sample = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                Downsample(in_channels, True, dims=2, kernel_size=2, padding=0, out_channels=out_channels)
            )
        else:
            self.down_sample = nn.Identity()

        self.in_channels = in_channels
        self.use_lap_pyramid = use_lap_pyramid
        self.dropout = dropout
        self.conv_next_channel_mult = conv_next_channel_mult

        if use_lap_pyramid:
            self.lap_conv = nn.Conv2d(out_channels + 1, out_channels, kernel_size=1, padding=0)

        layers = []
        for i in range(num_blocks):
            dilation = 1
            if use_dilation:
                dilation = (i % 2) + 1
            layers.append(
                ConvNeXtBlock(channels=out_channels, dropout=dropout[i], out_channels=out_channels,
                              channel_mult=conv_next_channel_mult, dilation=dilation)
            )
        self.conv_blocks = TimestepEmbedSequential(*layers)
        if use_attn:
            self.attn = SpatialSelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, label_emb=None, lap_pyramid_input=None):
        x = self.down_sample(x)
        inner_skip = x
        if self.use_lap_pyramid:
            x = th.cat([x, lap_pyramid_input], dim=1)
            # conv to remove lap pyramid
            x = self.lap_conv(x)

        x = self.conv_blocks(x, label_emb)
        x = self.attn(x + inner_skip)
        return x


class UpBlock(TimestepBlock):
    def __init__(self, in_channels, skip_channels, out_channels, dropout, conv_next_channel_mult, num_blocks=2,
                 block_type='convnext', use_attn=False, up_sample=True, kernel_size=7):
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
            if block_type == 'convnext':
                layers.append(
                    ConvNeXtBlock(channels=out_channels, dropout=dropout[i],
                                  out_channels=out_channels,
                                  channel_mult=conv_next_channel_mult, kernel_size=kernel_size)
                )
            elif block_type == 'resblock':
                layers.append(
                    ResBlock(channels=out_channels, dropout=dropout[i],
                             out_channels=out_channels)
                )
            elif block_type == "custom_convnext":
                layers.append(
                    CustomConvNeXtBlock(channels=out_channels, dropout=dropout[i],
                                        out_channels=out_channels,
                                        channel_mult=conv_next_channel_mult)
                )
            else:
                raise ValueError("Invalid block type")

        self.conv_blocks = TimestepEmbedSequential(*layers)

        if use_attn:
            self.attn = SpatialSelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, skip=None, emb=None):
        x = self.up_sample(x)
        inner_skip = self.inner_skip_conv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.skip_conv(x)
        x = self.conv_blocks(x, emb=emb)
        return self.attn(x + inner_skip)



def two_d_softmax(x: torch.Tensor):
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    x_exp = th.exp(x)
    x_sum = th.sum(x_exp, dim=(2, 3), keepdim=True)
    return x_exp / x_sum


