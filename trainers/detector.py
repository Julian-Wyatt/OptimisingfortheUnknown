from typing import Any

import segmentation_models_pytorch
from einops import reduce, rearrange
import torch
from torch import nn

from dataset_utils.dataset_preprocessing_utils import get_coordinates_from_heatmap, renormalise
from dataset_utils.generate_landmark_images import create_radial_mask_batch, make_multi_scale_landmark_image_final_only, \
    generate_offset_maps
from dataset_utils.visualisations import plot_heatmaps, plot_heatmaps_and_landmarks_over_img, plot_landmarks_from_img
from models.convnextv2 import ConvNeXtUNet
from trainers.LandmarkDetection import LandmarkDetection
from utils.ema import LitEma

from utils.metrics import euclidean_distance
from torchinfo import summary

from models.multi_scale_unet import MultiScaleUNetModel
from models.multi_scale_unet_new import MultiScaleUNetModel as MultiScaleUNetModelNEW, two_d_softmax

from torchvision.transforms import Resize

import numpy as np


def scale_heatmap_for_plotting(heatmap):
    scale_factor = torch.amax(heatmap, dim=(2, 3), keepdim=True)
    softmax_output = reduce(heatmap / scale_factor * 255, "b c h w -> b 1 h w",
                            "max")
    return softmax_output, scale_factor


class RandomLandmarkDetector(LandmarkDetection):

    def __init__(self, cfg, use_ema=True, show_summary=True, load_external_weights=True):
        super().__init__(cfg)

        self.automatic_optimization = True
        # make a generator ie unet
        if self.cfg.DENOISE_MODEL.USE_NEW_MODEL == "ConvNeXtUNet":
            self.model = ConvNeXtUNet(
                model_type=cfg.DENOISE_MODEL.NAME,
                use_pretrained=cfg.DENOISE_MODEL.USE_PRETRAINED_IMAGENET_WEIGHTS and load_external_weights,
                embedding_dim_base=cfg.DENOISE_MODEL.DECODER_CHANNELS[0], num_classes=cfg.DATASET.NUMBER_KEY_POINTS,
                dropout=cfg.DENOISE_MODEL.DROPOUT,
                drop_path_rate=cfg.DENOISE_MODEL.DROP_PATH_RATE,
                embedding_dim_mult=cfg.DENOISE_MODEL.SEGFORMER_DECODER_CH_MULT,
                conv_next_channel_mult=cfg.DENOISE_MODEL.CONVNEXT_CH_MULT,
                grayscale_to_rgb=cfg.DENOISE_MODEL.GRAYSCALE_TO_RGB,
                use_patchify_stem=True
            )
            if show_summary:
                summary(self.model,
                        input_size=[(1, 1, *self.image_size)],
                        dtypes=[torch.float32], device="cpu", depth=5)

        elif self.cfg.DENOISE_MODEL.USE_NEW_MODEL is True:
            self.model = MultiScaleUNetModelNEW(
                in_channels=cfg.DATASET.CHANNELS,
                out_channels=cfg.DATASET.NUMBER_KEY_POINTS,
                encoder_channels=cfg.DENOISE_MODEL.ENCODER_CHANNELS,
                decoder_channels=cfg.DENOISE_MODEL.DECODER_CHANNELS,
                # num_blocks=cfg.DENOISE_MODEL.NUM_RES_BLOCKS,
                conv_next_channel_mult=self.cfg.DENOISE_MODEL.CONVNEXT_CH_MULT,
                dropout=cfg.DENOISE_MODEL.DROPOUT,
                use_checkpoint=False,
                use_lap_pyramid=cfg.DENOISE_MODEL.USE_LAP_PYRAMID,
                use_upscaled_heatmap=self.cfg.DIFFUSION.ADV_USE_UPSCALED_HEATMAP,
                blocks_per_level=self.cfg.DENOISE_MODEL.BLOCKS_PER_LEVEL,
                train_multi_objective=self.cfg.DIFFUSION.ADV_TRAIN_MULTI_OBJECTIVE,
                segformer_decoder_ch_mult=self.cfg.DENOISE_MODEL.SEGFORMER_DECODER_CH_MULT,
                # multi_scale_combine_channels=64
            )
            # x, timesteps = None, img = None, context = None, y = None,
            if show_summary:
                summary(self.model,
                        input_size=[(1, 1, *self.image_size)],
                        dtypes=[torch.float32], device="cpu", depth=5)
        elif self.cfg.DENOISE_MODEL.USE_NEW_MODEL == "UNet++":
            self.model = segmentation_models_pytorch.UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                # encoder_depth=len(cfg.DENOISE_MODEL.ENCODER_CHANNELS),
                # decoder_channels=cfg.DENOISE_MODEL.DECODER_CHANNELS,  # - 256, 128, 64,32,32
                decoder_channels=[256, 128, 64, 64, 32],
                in_channels=cfg.DATASET.CHANNELS,
                classes=64,
                # activation="identity"
            )
            if show_summary:
                summary(self.model,
                        input_size=[(1, 1, *self.image_size)],
                        dtypes=[torch.float32], device="cpu", depth=5)
        elif self.cfg.DENOISE_MODEL.USE_NEW_MODEL == "UNet":
            self.model = segmentation_models_pytorch.Unet(
                encoder_name='resnet34',
                # encoder_weights='imagenet',
                encoder_weights=None,
                # encoder_depth=len(cfg.DENOISE_MODEL.ENCODER_CHANNELS),
                # decoder_channels=cfg.DENOISE_MODEL.DECODER_CHANNELS,  # - 256, 128, 64,32,32
                # decoder_channels=[256, 128, 64, 64, 64],
                decoder_channels=[256, 256, 256, 128, 64],
                # decoder_channels=[512, 384, 256, 128, 64],
                in_channels=cfg.DATASET.CHANNELS,
                classes=cfg.DATASET.NUMBER_KEY_POINTS,
                # activation="identity"
            )

            def swap_relu_to_swish(model):
                for param in model.named_children():
                    if isinstance(param[1], nn.ReLU):
                        setattr(model, param[0], nn.SiLU())
                    elif isinstance(param[1], nn.BatchNorm2d):
                        setattr(model, param[0], nn.GroupNorm(num_groups=32, num_channels=param[1].num_features))
                    else:
                        swap_relu_to_swish(param[1])

            swap_relu_to_swish(self.model)

            # self.model.segmentation_head = nn.Identity()
            if show_summary:
                summary(self.model,
                        input_size=[(1, 1, *self.image_size)],
                        dtypes=[torch.float32], device="cpu", depth=5)
        else:

            self.model = MultiScaleUNetModel(
                image_size=cfg.DATASET.IMG_SIZE,
                in_channels=cfg.DATASET.CHANNELS,
                use_checkpoint=False,
                out_channels=cfg.DATASET.NUMBER_KEY_POINTS,
                final_act=cfg.DENOISE_MODEL.FINAL_ACT,
                dropout=cfg.DENOISE_MODEL.DROPOUT,
                encoder_channels=cfg.DENOISE_MODEL.ENCODER_CHANNELS,
                decoder_channels=cfg.DENOISE_MODEL.DECODER_CHANNELS,
                num_res_blocks=cfg.DENOISE_MODEL.NUM_RES_BLOCKS,
                context_dim=cfg.DENOISE_MODEL.CONTEXT_DIM,
                use_spatial_transformer=cfg.CLASSIFIER_MODEL.CONTEXT_DIM is not None,
                attention_resolutions=cfg.DENOISE_MODEL.ATTN_RESOLUTIONS,
                use_img_as_context=False,  # input is the image
                num_heads=cfg.DENOISE_MODEL.NUM_HEADS,
                down_sample_context=True,
                use_timestep_embedding=False,
                num_classes=1,
                cat_final=False,
                conv_next_channel_mult=4,
                convnext=True,

            )
            # x, timesteps = None, img = None, context = None, y = None,
            if show_summary:
                summary(self.model,
                        input_size=[(1, 1, *self.image_size), [1], (1, 1, *self.image_size), (1, 1, *self.image_size),
                                    [1]],
                        dtypes=[torch.float32, torch.long, torch.float32, torch.float32, torch.int], device="cpu",
                        depth=5)

        self.learning_rate = cfg.TRAIN.LR

        self.bce = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        self.binaryLoss = torch.nn.BCEWithLogitsLoss(None, reduction='none')
        self.l1Loss = torch.nn.SmoothL1Loss(reduction="none")


        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=0.9999)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_losses_by_resolution = self.cfg.DIFFUSION.ADV_SCALE_LOSSES_BY_RESOLUTION

        self.use_upscaled_heatmap = self.cfg.DIFFUSION.ADV_USE_UPSCALED_HEATMAP

        # self.model = torch.jit.script(self.model)

    def compute_losses(self, landmarks, img, batch):

        log_prefix = "train" if self.training else "val"
        loss_dict = {}

        target = batch["y_img"]

        heatmap_prediction = self.model(img)
        act = two_d_softmax

        nll = torch.tensor(0.0).to(self.device)

        if self.cfg.DIFFUSION.ADV_TRAIN_MULTI_OBJECTIVE:
            model_predictions = self.model.head(heatmap_prediction)
            heatmap_prediction, mask_prediction, offset_prediction_x, offset_prediction_y = torch.chunk(
                model_predictions, 4, dim=1)
            if self.device.type != "mps":
                heatmap_prediction = heatmap_prediction.double()
            # mask_gt
            mask_gt = create_radial_mask_batch(landmarks, img.shape[2:], batch["pixel_size"],
                                               radius=self.cfg.DATASET.NEGATIVE_LEARNING_MAX_RADIUS, min_radius=0,
                                               do_normalise=False)

            # offset_gt = torch.zeros(
            #     (landmarks.shape[0], landmarks.shape[1], self.image_size[0], self.image_size[1]),
            #     device=landmarks.device,
            #     dtype=torch.float32)
            # for b in range(landmarks.shape[0]):
            #     offset_gt[b] = generate_combined_offset_map(landmarks[b], self.image_size,
            #                                                 pixel_size=batch["pixel_size"][b],
            #                                                 radius=self.cfg.DATASET.NEGATIVE_LEARNING_MAX_RADIUS)

            offset_gt_x = torch.zeros(
                (landmarks.shape[0], landmarks.shape[1], img.shape[2], img.shape[3]),
                device=landmarks.device,
                dtype=torch.float32)
            offset_gt_y = torch.zeros(
                (landmarks.shape[0], landmarks.shape[1], img.shape[2], img.shape[3]),
                device=landmarks.device,
                dtype=torch.float32)
            for b in range(landmarks.shape[0]):
                offset_gt = generate_offset_maps(landmarks[b], img.shape[2:],
                                                 mask=mask_gt[b])
                offset_gt_x[b] = offset_gt[0]
                offset_gt_y[b] = offset_gt[1]

            use_mask_on_nll = False
            heatmap_prediction_softmax = act(heatmap_prediction)
            if use_mask_on_nll:
                combined_nll = -target[mask_gt == 1] * torch.log(heatmap_prediction_softmax[mask_gt == 1])
                combined_nll = torch.mean(combined_nll)
                loss_dict.update({f'{log_prefix}/combined_nll': combined_nll.detach().item()})
            else:
                combined_nll = -target * torch.log(heatmap_prediction_softmax)
                combined_nll = torch.mean(torch.sum(combined_nll, dim=(2, 3)))
                loss_dict.update({f'{log_prefix}/combined_nll': combined_nll.detach().item()})
                nll /= 4
            nll += combined_nll

            # mask loss

            # B, N, H, W
            # -(1-p_t)^gamma log p_t
            # gamma = 1.5
            # mask_prediction = torch.sigmoid(mask_prediction)
            # p_t = mask_prediction * mask_gt + (1 - mask_prediction) * (1 - mask_gt)
            # mask_loss = -((1 - p_t) ** gamma) * torch.log(p_t)
            # mask_loss = torch.mean(torch.sum(mask_loss, dim=(2, 3))) / 10

            # ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
            # pt = torch.exp(-ce_loss)
            # focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
            # return focal_loss

            # mask_loss = self.binaryLoss(mask_prediction, mask_gt)
            # mask_loss = torch.mean(torch.sum(mask_loss, dim=(2, 3))) / 1000

            # reshape mask to be B*H*W,C

            distance_x = offset_prediction_x[mask_gt == 1] - offset_gt_x[mask_gt == 1]
            distance_y = offset_prediction_y[mask_gt == 1] - offset_gt_y[mask_gt == 1]
            distances = torch.sqrt(distance_x * distance_x + distance_y * distance_y)
            zero_distances = torch.zeros(distance_x.shape, requires_grad=True).to(self.device)

            offset_loss = self.l1Loss(distances, zero_distances)
            offset_loss = torch.mean(offset_loss)

            # reshape mask to be B*H*W,C
            # gamma = 1.5
            # mask_prediction_permuted = mask_prediction.clone().permute(0, 2, 3, 1).reshape(-1, target.shape[1])
            # mask_gt_permuted = mask_gt.clone().permute(0, 2, 3, 1).reshape(-1, target.shape[1])
            #
            # # loss = -1 * (1-pt)**gamma * logpt
            # logpt = torch.nn.functional.cross_entropy(mask_prediction_permuted, mask_gt_permuted, reduction='none')
            # mask_loss = -1 * (1 - logpt.exp()) ** gamma * logpt
            # mask_loss = mask_loss.mean()

            # mask_loss = self.binaryLoss(mask_prediction, mask_gt).mean()
            mask_prediction = torch.sigmoid(mask_prediction.double())
            mask_prediction_sigmoid = torch.clamp(mask_prediction, min=1e-3, max=1 - 1e-3)

            pos_inds = mask_gt.gt(0.9)
            neg_inds = mask_gt.lt(0.9)
            neg_weights = torch.pow(1 - mask_gt[neg_inds], 4)  # negative weights | 负样本权重

            pos_pred = mask_prediction_sigmoid[pos_inds]
            neg_pred = mask_prediction_sigmoid[neg_inds]
            pos_loss = torch.log2(pos_pred) * torch.pow(1 - pos_pred, 2)  # positive loss | 正样本损失
            neg_loss = torch.log2(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights  # negative loss | 负样本损失
            num_pos = pos_inds.float().sum()

            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if num_pos == 0:
                mask_loss = -neg_loss  # if no positive samples, only consider negative loss | 如果没有正样本，只考虑负样本损失
            else:
                mask_loss = -(pos_loss + neg_loss) / num_pos  # average loss | 平均损失
            mask_loss /= 5

            loss_dict.update({f"{log_prefix}/mask_loss": mask_loss})
            loss_dict.update({f"{log_prefix}/offset_loss": offset_loss})

            with torch.no_grad():
                distance_x = offset_prediction_x.detach()
                distance_y = offset_prediction_y.detach()
                total_distances = torch.sqrt(distance_x * distance_x + distance_y * distance_y)
                mask_prediction_sigmoid = torch.sigmoid(mask_prediction.detach())
                heatmap_prediction = mask_prediction.detach().clone()

            nll += (mask_loss + offset_loss * 3)

        # combined_heatmaps = self.model.combine_heatmaps(heatmap_predictions, target.shape[2:],
        #                                                 scale_by_resolution=self.scale_losses_by_resolution)
        #
        # heatmap_prediction = Resize(target.shape[2:])(combined_heatmaps)
        else:
            # heatmap_prediction = self.model.head(model_output_pre_head)  # point prediction
            if self.device.type != "mps":
                heatmap_prediction = heatmap_prediction.double()
            heatmap_prediction_softmax = act(heatmap_prediction)
            combined_nll = -target * torch.log(heatmap_prediction_softmax)
            combined_nll = torch.mean(torch.sum(combined_nll, dim=(2, 3)))
            loss_dict.update({f'{log_prefix}/combined_nll': combined_nll.detach().item()})
            nll += combined_nll

        if torch.isnan(nll):
            print("NAN", heatmap_prediction.min(), heatmap_prediction.max(), act(heatmap_prediction).min(),
                  act(heatmap_prediction).max())

        with torch.no_grad():
            gt = Resize(img.shape[2:])(target)
            heatmap_prediction_softmax = Resize(img.shape[2:])(heatmap_prediction_softmax)


        loss_dict.update({f'{log_prefix}/nll': nll.detach().item()})
        loss = nll
        loss_dict.update({f'{log_prefix}/loss': loss.detach().item()})

        with torch.no_grad():
            coordinates = get_coordinates_from_heatmap(heatmap_prediction_softmax,
                                                       k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS).flip(-1)
            l2_coordinate_estimate = torch.mean(euclidean_distance(coordinates, landmarks.float()))
            loss_dict.update({f'{log_prefix}/l2_est': l2_coordinate_estimate.item()})
            pixel_sizes = batch["pixel_size"].unsqueeze(1)
            coordinates_scaled = coordinates * pixel_sizes
            real_landmarks_scaled = batch["y"].float() * pixel_sizes
            l2_coordinate_estimate_scaled = torch.mean(euclidean_distance(coordinates_scaled, real_landmarks_scaled))
            loss_dict.update({f'{log_prefix}/l2_est_scaled': l2_coordinate_estimate_scaled.item()})
        # log image
        frequency = 10
        if self.cfg.DIFFUSION.ADV_TRAIN_MULTI_OBJECTIVE:
            frequency //= 2
        if self.device.type == "mps":
            frequency = 1
        if self.batch_idx == 0 and self.current_epoch % frequency == 0 and self.do_img_logging or (
                not self.training and self.do_img_logging and self.batch_idx == 0):
            with torch.no_grad():
                # log training samples
                # log predictions w/ gt, heatmap, heatmap for channel 15, gt for channel 15
                heatmap_prediction_renorm = heatmap_prediction_softmax
                softmax_output, scale_factor = scale_heatmap_for_plotting(heatmap_prediction_renorm)

                img_pred_log = [plot_landmarks_from_img(renormalise(img, method=self.cfg.DATASET.NORMALISATION),
                                                        heatmap_prediction_renorm,
                                                        true_landmark=batch["y_img_initial"]).cpu().int(),
                                softmax_output.repeat(1, 3, 1, 1).cpu()]
                img_pred_log.append(
                    (heatmap_prediction_renorm[:, 15] / heatmap_prediction_renorm[:, 15].max() * 240).unsqueeze(1)
                    .repeat(1, 3, 1, 1).cpu()
                )
                img_pred_log.append(
                    (gt[:, 15] * 240).clamp(0, 240).unsqueeze(1).repeat(1, 3, 1, 1).cpu()
                )
                # img_pred_log.append(reduce((batch["y_img_initial"] + 1) * 127.5, "b c h w -> b 1 h w",
                #                            "max").repeat(1, 3, 1, 1).cpu())
                # img_pred_log.append(reduce(gt * 240, "b c h w -> b 1 h w",
                #                            "max").repeat(1, 3, 1, 1).cpu())

                # for i, val in enumerate(img_pred_log):
                #     print(i, val.min(), val.max(), val.float().mean(), val.float().std())
                img_pred_log = self._get_rows_from_list(torch.stack(img_pred_log))

                try:

                    self.logger.log_image(key=f"Media/{log_prefix}/predictions",
                                          caption=[
                                              f"Images {batch['name']} step pixel mre {l2_coordinate_estimate.item():.4f} step mm mre {l2_coordinate_estimate_scaled.item():.4f} scaled by {scale_factor.min().item():.2f}, {scale_factor.max().item():.2f}, {scale_factor.mean().item():.2f}, {scale_factor.std().item():.2f}"],
                                          images=[img_pred_log])
                except OSError as e:
                    print(e)
                del scale_factor, img_pred_log
                if self.cfg.DIFFUSION.ADV_TRAIN_MULTI_OBJECTIVE:
                    img_pred_log_2 = []

                    for i in [heatmap_prediction_softmax, gt, mask_prediction_sigmoid, mask_gt,
                              total_distances, (offset_gt_x ** 2 + offset_gt_y ** 2).sqrt()]:
                        img_pred_log_2.append(
                            scale_heatmap_for_plotting(i)[0].clamp(0, 255).repeat(1, 3, 1, 1).cpu())
                    img_pred_log_2 = self._get_rows_from_list(torch.stack(img_pred_log_2))
                    try:
                        self.logger.log_image(key=f"Media/{log_prefix}/multi_objective_predictions",
                                              caption=["combined gt"],
                                              images=[img_pred_log_2])
                    except OSError as e:
                        print(e)

        return loss, loss_dict, heatmap_prediction_softmax.detach()

    def forward(self, query_img, label=None) -> Any:
        keypoint_hat = self.model(query_img)

        if self.cfg.DIFFUSION.ADV_TRAIN_MULTI_OBJECTIVE:
            _, keypoint_hat, _, _ = torch.chunk(keypoint_hat, 4, dim=1)

        return two_d_softmax(keypoint_hat)


    def shared_step(self, batch):

        landmarks = batch["y"]
        image = rearrange(batch["x"], 'b h w c -> b c h w')

        loss, loss_dict, output = self.compute_losses(landmarks, img=image, batch=batch)

        return loss, loss_dict, output

    def output_to_img_log(self, output, batch, batch_idx=None):
        # img_log returns dict with {"heatmaps": torch.Tensor, "video": list of torch.Tensor,
        #                            "heatmaps_figure": plt.Figure, "final": plt.Figure}
        if not self.cfg.TRAIN.LOG_WHOLE_VAL and batch_idx > 0:
            return None
        with torch.no_grad():
            img_log = {}
            img_log["heatmaps"] = output
            img_log["final"] = plot_heatmaps_and_landmarks_over_img(self.get_input(batch, "x"), output,
                                                                    batch["y"],
                                                                    normalisation_method=self.cfg.DATASET.NORMALISATION)
            img_log["heatmaps_figure"] = plot_heatmaps(output, batch["y"])
        return img_log

    @torch.no_grad()
    def unique_test_step(self, batch, batch_idx):
        query_image = self.get_input(batch, "x")
        landmarks = batch["y"]
        with self.ema_scope():
            output = self(query_image)
        if self.use_upscaled_heatmap:
            output = Resize(query_image.shape[2:])(output)
        img_log = dict()
        img_log["heatmaps"] = output
        img_log["final"] = plot_heatmaps_and_landmarks_over_img(self.get_input(batch, "x"), output,
                                                                landmarks,
                                                                normalisation_method=self.cfg.DATASET.NORMALISATION)
        img_log["heatmaps_figure"] = plot_heatmaps(output, landmarks)

        return img_log

    def on_train_epoch_start(self) -> None:
        if self.cfg.TRAIN.USE_SCHEDULER == "cosine" or self.cfg.TRAIN.USE_SCHEDULER == "exp":
            self.anneal_lr()

    def anneal_lr(self):

        # linear warmup with exponential decay
        if self.current_epoch < self.cfg.TRAIN.WARMUP_EPOCHS:
            lr = (self.cfg.TRAIN.LR * (self.current_epoch + 1) / self.cfg.TRAIN.WARMUP_EPOCHS)
            # lr = self.cfg.TRAIN.LR
        else:
            # elif self.current_epoch < self.cfg.TRAIN.WARMUP_EPOCHS * 2:
            # cosine decay
            if self.cfg.TRAIN.USE_SCHEDULER == "cosine":
                lr = self.cfg.TRAIN.MIN_LR + (self.cfg.TRAIN.LR - self.cfg.TRAIN.MIN_LR) * 0.5 * \
                     (1. + np.cos(np.pi * (self.current_epoch - self.cfg.TRAIN.WARMUP_EPOCHS) / (
                             self.cfg.TRAIN.EPOCHS - self.cfg.TRAIN.WARMUP_EPOCHS)))
            elif self.cfg.TRAIN.USE_SCHEDULER == "exp":
                # exp decay
                lr = max(
                    self.cfg.TRAIN.LR * self.cfg.TRAIN.EXP_LR_DECAY ** (
                            self.current_epoch - self.cfg.TRAIN.WARMUP_EPOCHS),
                    self.cfg.TRAIN.MIN_LR)
            else:
                lr = self.cfg.TRAIN.LR

        for param_group in self.optimizers().param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        self.log('Charts/lr', lr, prog_bar=False, logger=True, on_epoch=True)
        return lr

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMISER.lower() == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                     betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "adamw":
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                      betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                    momentum=0.9)
        else:
            raise ValueError(f"Optimiser {self.cfg.TRAIN.OPTIMISER} not recognised")
        if self.cfg.TRAIN.USE_SCHEDULER == "multistep":
            # scheduler = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 65], gamma=0.2)]
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 45], gamma=0.25)]
            # scheduler = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 55], gamma=0.25)]
        elif self.cfg.TRAIN.USE_SCHEDULER == "reduce_on_plateau":
            scheduler = [{
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2,
                                                                        cooldown=2,
                                                                        threshold=0.0001, threshold_mode="rel",
                                                                        verbose=True,
                                                                        min_lr=5e-6),
                "strict": False,
                "monitor": "val/l2_scaled",
            }]
        else:
            scheduler = []

        return [opt], scheduler
