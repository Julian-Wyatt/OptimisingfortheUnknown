import torch
from timm.layers import trunc_normal_, DropPath
from torch import nn
import torch.nn.functional as F
from torch._C._profiler import ProfilerActivity
from torch.profiler import record_function
from torch.profiler import profile
from torchinfo import summary

from core import config
from models.unet_utils import TimestepEmbedSequential, Upsample
from models.multi_scale_unet_new import UpBlock
from utils.util import LayerNorm, normalisation


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    # https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/encoders/_utils.py#L5
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size,
            )
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

class IMG_MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        img_shape = x.shape[-2:]
        x = x.flatten(2).permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1).reshape(-1, self.out_channels, *img_shape)

        return x


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


class SegformerDecoder(nn.Module):
    def __init__(self, in_channels: list, embedding_dim_base, num_classes, dropout=0.1, embedding_dim_mult=4,
                 conv_next_channel_mult=4, drop_path_rate=0.1):
        super().__init__()

        embedding_dim = embedding_dim_base * embedding_dim_mult
        # 4 linear layers
        self.linears = nn.ModuleList(
            [IMG_MLP(i, embedding_dim) for i in in_channels])
        # fuse layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), embedding_dim, kernel_size=1, padding=0),
            normalisation(embedding_dim),
            # nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        # UpBlock
        print(in_channels)

        depths = [2, 2]
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)][::-1][sum(depths):]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate - 0.1, sum(depths))][::-1]
        if len(in_channels) == 5:

            self.up_blocks_0 = UpBlock(embedding_dim, in_channels[0], embedding_dim_base, dp_rates[:depths[0]],
                                       conv_next_channel_mult,
                                       depths[0],
                                       "convnext", False,
                                       False, kernel_size=7)

            self.up_blocks = TimestepEmbedSequential(
                Upsample(embedding_dim_base, True, dims=2, kernel_size=3, padding=1,
                         out_channels=embedding_dim_base // 2),
            )
        else:
            self.up_blocks_0 = UpBlock(embedding_dim, in_channels[0], embedding_dim_base * 2, dp_rates[:depths[0]],
                                       conv_next_channel_mult,
                                       depths[0],
                                       "convnext", False,
                                       False, kernel_size=7)

            self.up_blocks = TimestepEmbedSequential(
                Upsample(embedding_dim_base * 2, True, dims=2, kernel_size=3, padding=1,
                         out_channels=embedding_dim_base),
                UpBlock(embedding_dim_base, 0, embedding_dim_base, dp_rates[depths[0]:],
                        conv_next_channel_mult, depths[1],
                        "convnext",
                        False, False, kernel_size=7),
                Upsample(embedding_dim_base, True, dims=2, kernel_size=1, padding=0,
                         out_channels=embedding_dim_base // 2)
            )

        self.prediction = nn.Sequential(
            normalisation(embedding_dim_base // 2),
            nn.GELU(),
            nn.Conv2d(embedding_dim_base // 2, num_classes, kernel_size=1, padding=0)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, skips, label=None):
        # skips = [c1, c2, c3, c4, c5]
        skip_to_up_block = skips[0]
        skips = [linear(skip) for skip, linear in zip(skips, self.linears)]
        skips = [skips[0]] + [F.interpolate(skip, size=skips[0].shape[2:], mode='bilinear', align_corners=False) for
                              i, skip in enumerate(skips) if i > 0]
        x = torch.cat(skips, dim=1)
        x = self.linear_fuse(x)
        x = self.up_blocks_0(x, emb=label, skip=skip_to_up_block)
        x = self.up_blocks(x, emb=label)
        x = self.prediction(x)
        return x


class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0., GRN_shape=2):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim, shape=GRN_shape)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, depths=None, dims=None, drop_path_rate=0., GRN_shape=2):
        super().__init__()
        if dims is None:
            dims = [96, 192, 384, 768]

        if depths is None:
            depths = [3, 3, 9, 3]
        self.dims = dims
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j], GRN_shape=GRN_shape) for j in
                  range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], 1000)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def unfreeze(self, to_unfreeze=["all"]):
        for param in self.named_parameters():
            if to_unfreeze == ["all"]:
                param[1].requires_grad = True
            elif "norm" in to_unfreeze and "norm" in param[0] or "grn" in param[0]:
                param[1].requires_grad = True
            elif "downsample" in to_unfreeze and "downsample_layers" in param[0]:
                param[1].requires_grad = True

    def freeze(self, to_freeze=["all"]):
        for param in self.named_parameters():
            if to_freeze == ["all"]:
                param[1].requires_grad = False
            elif "norm" in to_freeze and "norm" in param[0] or "grn" in param[0]:
                param[1].requires_grad = False
            elif "downsample" in to_freeze and "downsample_layers" in param[0]:
                param[1].requires_grad = False

    def forward_features(self, x):
        skips = []
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            skips.append(x)
        return skips

    def forward(self, x):
        x = self.forward_features(x)
        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


class ConvNeXtUNet(nn.Module):
    def __init__(self, model_type="nano", use_pretrained=True, embedding_dim_base=128, num_classes=1, dropout=0.1,
                 drop_path_rate=0.1,
                 embedding_dim_mult=4, conv_next_channel_mult=4,
                 grayscale_to_rgb="weighted_sum", in_channels=1,
                 use_patchify_stem=True,
                 ):
        super(ConvNeXtUNet, self).__init__()
        convnext_kwargs = {
            "drop_path_rate": drop_path_rate
        }

        use_finetuned = False
        if model_type == "atto":
            self.encoder = convnextv2_atto(**convnext_kwargs)
            if use_pretrained:
                checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_atto_1k_224_fcmae.pt")
                self.encoder.load_state_dict(state_dict=checkpoint["model"])
        elif model_type == "femto":
            self.encoder = convnextv2_femto(**convnext_kwargs)
            if use_pretrained:
                checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_femto_1k_224_fcmae.pt")
                self.encoder.load_state_dict(state_dict=checkpoint["model"])
        elif model_type == "pico":
            self.encoder = convnext_pico(**convnext_kwargs)
            if use_pretrained:
                checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_pico_1k_224_fcmae.pt")
                self.encoder.load_state_dict(state_dict=checkpoint["model"])
        elif model_type == "nano":
            if use_pretrained:

                convnext_kwargs["GRN_shape"] = 2
                self.encoder = convnextv2_nano(**convnext_kwargs)
                if use_finetuned:
                    self.encoder.norm = nn.LayerNorm(640, eps=1e-6)  # final norm layer
                    self.encoder.head = nn.Linear(640, 1000)
                    # checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_nano_22k_384_ema.pt")
                    checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_nano_1k_224_ema.pt")
                    for k in checkpoint["model"].keys():
                        if "grn" in k:
                            checkpoint["model"][k] = checkpoint["model"][k].reshape(1, -1)
                else:
                    checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_nano_1k_224_fcmae.pt")
                self.encoder.load_state_dict(state_dict=checkpoint["model"])
                if use_finetuned:
                    del self.encoder.head, self.encoder.norm
            else:
                self.encoder = convnextv2_nano(**convnext_kwargs)
        elif model_type == "tiny":
            if use_pretrained:
                convnext_kwargs["GRN_shape"] = 2
                self.encoder = convnextv2_tiny(**convnext_kwargs)
                if use_finetuned:
                    self.encoder.norm = nn.LayerNorm(768, eps=1e-6)  # final norm layer
                    self.encoder.head = nn.Linear(768, 1000)

                    # checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_tiny_22k_384_ema.pt")
                    checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_tiny_1k_224_ema.pt")
                    for k in checkpoint["model"].keys():
                        if "grn" in k:
                            checkpoint["model"][k] = checkpoint["model"][k].reshape(1, -1)
                else:
                    checkpoint = torch.load("checkpoints/ConvNeXtV2/convnextv2_tiny_1k_224_fcmae.pt")
                self.encoder.load_state_dict(state_dict=checkpoint["model"])
                if use_finetuned:
                    del self.encoder.head, self.encoder.norm

            else:
                self.encoder = convnextv2_tiny(**convnext_kwargs)
        elif model_type == "base":
            self.encoder = convnextv2_base(**convnext_kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        if grayscale_to_rgb == "conv":
            self.input_conv = nn.Conv2d(1, 3, kernel_size=1)
        elif grayscale_to_rgb == "weighted_sum":
            patch_first_conv(
                model=self, new_in_channels=in_channels, pretrained=use_pretrained
            )

        if not use_patchify_stem:
            # swap self.encoder.downsample_layers[0] with patchify stem
            # self.encoder.downsample_layers[0] = nn.Sequential(
            #     nn.Conv2d(in_channels, self.encoder.dims[0], kernel_size=9, stride=4, padding=4),
            #     LayerNorm(self.encoder.dims[0], eps=1e-6, data_format="channels_first")
            # )
            self.encoder.downsample_layers = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_channels, self.encoder.dims[0], kernel_size=7, stride=2, padding=3),
                LayerNorm(self.encoder.dims[0], eps=1e-6, data_format="channels_first")),
                nn.Sequential(
                    LayerNorm(self.encoder.dims[0], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.encoder.dims[0], self.encoder.dims[0], kernel_size=2, stride=2, padding=0)
                ),
                *self.encoder.downsample_layers[1:]])

            self.encoder.stages = nn.ModuleList([nn.Identity(), *self.encoder.stages])
            self.decoder = SegformerDecoder([self.encoder.dims[0]] + self.encoder.dims,
                                            embedding_dim_base=embedding_dim_base,
                                            num_classes=num_classes,
                                            dropout=dropout, embedding_dim_mult=embedding_dim_mult,
                                            conv_next_channel_mult=conv_next_channel_mult,
                                            drop_path_rate=drop_path_rate)
        else:
            self.decoder = SegformerDecoder(self.encoder.dims, embedding_dim_base=embedding_dim_base,
                                            num_classes=num_classes,
                                            dropout=dropout, embedding_dim_mult=embedding_dim_mult,
                                            conv_next_channel_mult=conv_next_channel_mult,
                                            drop_path_rate=drop_path_rate)

        self.grayscale_to_rgb = grayscale_to_rgb

    def forward(self, x):
        if self.grayscale_to_rgb == "conv":
            x = self.input_conv(x)
        elif self.grayscale_to_rgb == "repeat":
            x = x.repeat(1, 3, 1, 1)
        elif self.grayscale_to_rgb == "weighted_sum":
            pass
        else:
            raise ValueError(f"Invalid grayscale_to_rgb: {self.grayscale_to_rgb}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # model = ConvNeXtUNet(model_type="atto", use_pretrained=True)
    cfg = config.get_config("configs/docker_configs/next_ensemble_tiny.yaml")
    cfg.DENOISE_MODEL.NAME = "tiny"
    # cfg.DENOISE_MODEL.GRAYSCALE_TO_RGB = "repeat"
    cfg.DENOISE_MODEL.GRAYSCALE_TO_RGB = "weighted_sum"

    model = ConvNeXtUNet(
        model_type=cfg.DENOISE_MODEL.NAME, use_pretrained=cfg.DENOISE_MODEL.USE_PRETRAINED_IMAGENET_WEIGHTS,
        embedding_dim_base=cfg.DENOISE_MODEL.DECODER_CHANNELS[0], num_classes=cfg.DATASET.NUMBER_KEY_POINTS,
        dropout=cfg.DENOISE_MODEL.DROPOUT,
        drop_path_rate=cfg.DENOISE_MODEL.DROP_PATH_RATE,
        embedding_dim_mult=cfg.DENOISE_MODEL.SEGFORMER_DECODER_CH_MULT,
        conv_next_channel_mult=cfg.DENOISE_MODEL.CONVNEXT_CH_MULT,
        grayscale_to_rgb=cfg.DENOISE_MODEL.GRAYSCALE_TO_RGB,
        use_patchify_stem=True
    )
    # model.encoder.freeze(to_freeze=["all"])
    # model.encoder.unfreeze(to_unfreeze=["all"])

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            # summary(model,
            #         input_size=[(1, 1, 640, 512), (1,)], dtypes=[th.float32, th.int64],
            #         device="cpu", depth=5)
            #
            summary(model,
                    input_size=[(1, 1, 800, 704)], dtypes=[torch.float32],
                    device="cpu", depth=5)
            # summary(model,
            #         input_size=[(1, 1, 256, 256)], dtypes=[torch.float32],
            #         device="cpu", depth=5)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
