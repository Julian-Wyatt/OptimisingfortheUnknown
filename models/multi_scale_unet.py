# Originally forked from
# https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
import torch
import torch as th
import torch.nn as nn
from torchvision.transforms.v2 import Resize

from models.attention import SpatialTransformer
from models.unet_utils import TimestepEmbedSequential, ResBlock, AttentionBlock, Downsample, Upsample, \
    convert_module_to_f16, convert_module_to_f32
from models.multi_scale_unet_new import ConvNeXtBlock

from utils.util import conv_nd, zero_module, normalisation, get_timestep_embedding


def Res_or_ConvNeXtBlock(convnext, in_channels, out_channels, dropout, use_checkpoint,
                         use_scale_shift_norm, dims, time_embed_channels=None, channel_mult=4):
    if convnext:
        return TimestepEmbedSequential(ConvNeXtBlock(in_channels, out_channels=out_channels, dropout=dropout,
                                                     use_checkpoint=use_checkpoint, channel_mult=channel_mult,
                                                     dims=dims))
    else:
        return TimestepEmbedSequential(
            ResBlock(in_channels, time_embed_channels, dropout, out_channels, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm))


class MultiScaleUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    To compute noise prediction - ie given a noisy image, predict the noise
    Or general unet generator without timestep embedding but instead label embedding.

    :param in_channels: channels in the input Tensor.
    :param encoder_channels : a list of channels for each encoder block.
    :param decoder_channels : a list of channels for each decoder block.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_upsample: works with num_heads to set a different number
            of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            encoder_channels,
            decoder_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            use_timestep_embedding=True,
            use_img_as_context=False,
            final_act="tanh",
            dropout=0,
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            down_sample_context=False,
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            cat_final=False,
            convnext=True,
            conv_next_channel_mult=2,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
            if context_dim != image_size[1]:
                context_dim = image_size[1]
            # if context_dim is not None:
            #     assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # context_dim = list(context_dim)

        assert len(encoder_channels) == len(
            decoder_channels), 'The number of encoder and decoder channels must be the same'

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.cat_final = cat_final
        self.in_channels = in_channels if not use_img_as_context else in_channels + 1
        self.use_img_as_context = use_img_as_context
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.down_sample_context = down_sample_context
        assert down_sample_context is True, "Down sample context must be true for multi-scale unet model"
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        self.combine_channels = 32

        time_embed_dim = self.encoder_channels[0] * 4
        self.time_embed_dim = time_embed_dim
        self.use_timestep_embedding = use_timestep_embedding
        if use_timestep_embedding:
            self.time_embed = nn.Sequential(
                nn.Linear(self.encoder_channels[0], time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, time_embed_dim)
            self.label_embed = nn.Linear(time_embed_dim, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, self.in_channels, self.encoder_channels[0], 3, padding=1)
                )
            ]
        )
        self._feature_size = self.encoder_channels[0]
        input_block_chans = [self.encoder_channels[0]]
        ch_prev = self.encoder_channels[0]
        ds = 1

        if self.down_sample_context:
            self.context_blocks = nn.ModuleList()

        for level, ch in enumerate(self.encoder_channels):

            for _ in range(num_res_blocks):
                layers = [
                    Res_or_ConvNeXtBlock(convnext, ch_prev, ch, dropout, use_checkpoint,
                                         use_scale_shift_norm, dims=dims, time_embed_channels=time_embed_dim,
                                         channel_mult=conv_next_channel_mult)
                ]
                ch_prev = ch
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.encoder_channels) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
                if self.down_sample_context:
                    context_dim = context_dim // 2
                    self.context_blocks.append(Downsample(1, True, dims=2))

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            Res_or_ConvNeXtBlock(convnext, ch_prev, ch, dropout, use_checkpoint,
                                 use_scale_shift_norm, dims=dims, time_embed_channels=time_embed_dim,
                                 channel_mult=conv_next_channel_mult),
            Res_or_ConvNeXtBlock(convnext, ch_prev, ch, dropout, use_checkpoint,
                                 use_scale_shift_norm, dims=dims, time_embed_channels=time_embed_dim,
                                 channel_mult=conv_next_channel_mult),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=dim_head,
            # ) if not use_spatial_transformer else SpatialTransformer(
            #     ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            # ),
            # ResBlock(
            #     ch,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.multiscale_output_blocks = nn.ModuleList([])
        ch_prev = ch
        for level, ch_next in enumerate(self.decoder_channels):

            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()

                layers = [
                    Res_or_ConvNeXtBlock(convnext, ch_prev + ich, ch_next, dropout, use_checkpoint,
                                         use_scale_shift_norm, dims=dims, time_embed_channels=time_embed_dim,
                                         channel_mult=conv_next_channel_mult)
                ]
                ch_prev = ch_next
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch_prev // num_heads
                    else:
                        num_heads = ch_prev // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch_prev,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch_prev, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level < len(self.decoder_channels) - 1 and i == num_res_blocks:
                    out_ch = ch_prev
                    layers.append(
                        ResBlock(
                            ch_prev,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch_prev, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                    if self.down_sample_context:
                        context_dim = context_dim * 2
                        # context_blocks.append(Upsample(1, True, dims=2))
                    self.multiscale_output_blocks.append(
                        nn.Sequential(
                            # normalisation(out_ch),
                            nn.Tanh(),
                            zero_module(conv_nd(dims, out_ch, self.combine_channels, 3, padding=1)),
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch_prev

        self.final_block = Upsample(ch_prev, conv_resample, dims=dims, out_channels=ch_prev)

        self.multiscale_output_blocks.append(
            nn.Sequential(
                # normalisation(ch_prev),
                nn.Tanh(),
                zero_module(conv_nd(dims, ch_prev, self.combine_channels, 3, padding=1)),
            ))
        self.combine_conv_1 = conv_nd(3, len(self.multiscale_output_blocks), out_channels=1, kernel_size=3, stride=1,
                                      padding=1)
        self.combine_conv_2 = nn.Sequential(normalisation(self.combine_channels),
                                            conv_nd(2, self.combine_channels, out_channels, kernel_size=3, stride=1,
                                                    padding=1, bias=False))

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def two_d_softmax(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_exp = th.exp(x)
        x_sum = th.sum(x_exp, dim=(2, 3), keepdim=True)
        return x_exp / x_sum

    def forward(self, x, timesteps=None, img=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        context = img if context is None else context
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []

        # if use timesteps
        #        then get timestep embedding, embed that vector
        # if use y
        #         then get label embedding, embed that vector
        t_emb = torch.zeros(x.shape[0], self.time_embed_dim, device=x.device, dtype=x.dtype)
        if self.use_timestep_embedding:
            t_emb = get_timestep_embedding(timesteps, self.encoder_channels[0])
            t_emb = self.time_embed(t_emb)
        l_emb = torch.zeros(x.shape[0], self.time_embed_dim, device=x.device, dtype=x.dtype)
        if self.num_classes is not None:
            l_emb = self.label_embedding(y)
            l_emb = self.label_embed(l_emb)
        emb = t_emb + l_emb

        if img is not None and self.use_img_as_context:
            # cat non noisy image across channel dimension
            x = torch.cat([x, img], dim=1)
            del img
        current_shape = x.shape[2]
        cxs = []
        if self.down_sample_context:
            cxs.append(context)
        context_idx = 0
        h = x.type(self.dtype)
        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context.squeeze(1))
            hs.append(h)
            if self.down_sample_context and current_shape != h.shape[2]:
                context = self.context_blocks[context_idx](context)
                context_idx += 1
                if context_idx < len(self.context_blocks):
                    cxs.append(context)
            current_shape = h.shape[2]
        h = self.middle_block(h, emb, context.squeeze(1))

        outputs = []
        multi_scale_idx = 0
        for idx, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context.squeeze(1))
            if self.down_sample_context and current_shape != h.shape[2]:
                context = cxs.pop()
                outputs.append(self.multiscale_output_blocks[multi_scale_idx](h))
                multi_scale_idx += 1
            current_shape = h.shape[2]

        final_h = self.final_block(h)
        if self.cat_final:
            final_h = torch.cat([final_h, Resize(final_h.shape[2:])(context)], dim=1)
        outputs.append(self.multiscale_output_blocks[-1](final_h))
        return outputs

    def combine_heatmaps(self, heatmaps, img_size=None, scale_by_resolution=True):

        if img_size is None:
            img_size = self.image_size
        if scale_by_resolution:
            heatmaps = [val * heatmaps[i].shape[-1] / img_size[1] for i, val in enumerate(heatmaps)]

        # combine all the heatmaps
        final_heatmaps = [
            Resize((img_size[0], img_size[1]))(heatmaps[i]) for i in range(len(heatmaps))]

        if heatmaps[0].device.type == "mps":
            return [final_heatmaps[-1], final_heatmaps[-1]]
        # contribution factor varies based on heatmap resolution
        conv_1_output = self.combine_conv_1(torch.stack(final_heatmaps, dim=1)).squeeze(1)
        return self.combine_conv_2(conv_1_output)
