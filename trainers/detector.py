from typing import Any

import segmentation_models_pytorch
from einops import reduce, rearrange
import torch
from torch import nn

from dataset_utils.dataset_preprocessing_utils import get_coordinates_from_heatmap, renormalise
from dataset_utils.visualisations import plot_heatmaps, plot_heatmaps_and_landmarks_over_img, plot_landmarks_from_img
from models.convnextv2 import ConvNeXtUNet
from trainers.LandmarkDetection import LandmarkDetection
from utils.ema import LitEma

from utils.metrics import euclidean_distance
from torchinfo import summary

from models.unet_utils import two_d_softmax

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


        self.learning_rate = cfg.TRAIN.LR

        self.bce = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        self.binaryLoss = torch.nn.BCEWithLogitsLoss(None, reduction='none')
        self.l1Loss = torch.nn.SmoothL1Loss(reduction="none")


        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=0.9999)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # self.model = torch.jit.script(self.model)

    def compute_losses(self, landmarks, img, batch):

        log_prefix = "train" if self.training else "val"
        loss_dict = {}

        target = batch["y_img"]

        heatmap_prediction = self.model(img)
        act = two_d_softmax

        nll = torch.tensor(0.0).to(self.device)


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

        return loss, loss_dict, heatmap_prediction_softmax.detach()

    def forward(self, query_img, label=None) -> Any:
        keypoint_hat = self.model(query_img)


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
