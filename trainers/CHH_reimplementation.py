from core import init_paths
import torch
import torch.nn as nn
from einops import rearrange

from core import config
from models.convnextv2 import ConvNeXtV2Block
from models.denoising_unet import Upsample
from models.multi_scale_unet_new import UpBlock
from trainers.LandmarkDetection import LandmarkDetection
from trainers.detector import scale_heatmap_for_plotting
from dataset_utils.dataset_preprocessing_utils import renormalise, get_coordinates_from_heatmap
from dataset_utils.visualisations import plot_landmarks_from_img, plot_heatmaps_and_landmarks_over_img, \
    plot_heatmaps

from torchinfo import summary

import segmentation_models_pytorch as smp

from utils.metrics import euclidean_distance


class CHH(LandmarkDetection):

    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self, cfg: config.Config, show_summary=True, load_external_weights=True):
        """
        TRAIN:
          BATCH_SIZE: 4
          LR: 0.001
          EPOCHS: 15

        MODEL:
          ENCODER_NAME: 'resnet34'
          ENCODER_WEIGHTS: 'imagenet'
          DECODER_CHANNELS:
            - 256
            - 128
            - 64
            - 32
            - 32
          IN_CHANNELS: 1
        :param cfg:
        """
        super().__init__(cfg)

        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet' if cfg.DENOISE_MODEL.USE_PRETRAINED_IMAGENET_WEIGHTS and load_external_weights else None,
            decoder_channels=cfg.DENOISE_MODEL.DECODER_CHANNELS,  # - 256, 128, 64,32,32
            in_channels=cfg.DATASET.CHANNELS,
            classes=cfg.DATASET.NUMBER_KEY_POINTS,
        )

        def swap_relu_to_swish(model):
            for param in model.named_children():
                if isinstance(param[1], nn.ReLU):
                    # setattr(model, param[0], nn.Sequential(nn.ReLU(), nn.Dropout(0.1)))
                    setattr(model, param[0], nn.Sequential(nn.GELU(), nn.Dropout2d(cfg.DENOISE_MODEL.DROPOUT)))
                elif isinstance(param[1], nn.BatchNorm2d) and not cfg.DENOISE_MODEL.USE_PRETRAINED_IMAGENET_WEIGHTS:
                    setattr(model, param[0], nn.GroupNorm(num_groups=32, num_channels=param[1].num_features))
                else:
                    swap_relu_to_swish(param[1])
            return model

        self.model = swap_relu_to_swish(self.model)
        #
        # self.model.encoder.layer1[0] = ConvNeXtV2Block(self.model.encoder.layer1[0].conv1.in_channels, 0.05)
        # self.model.encoder.layer2[1] = ConvNeXtV2Block(self.model.encoder.layer2[1].conv1.in_channels, 0.1)
        # self.model.encoder.layer3[1] = ConvNeXtV2Block(self.model.encoder.layer3[1].conv1.in_channels, 0.15)
        # self.model.encoder.layer3[2] = ConvNeXtV2Block(self.model.encoder.layer3[2].conv1.in_channels, 0.175)
        # self.model.encoder.layer4[1] = ConvNeXtV2Block(self.model.encoder.layer4[1].conv1.in_channels, 0.2)

        self.temperatures = nn.Parameter(torch.ones(1, cfg.DATASET.NUMBER_KEY_POINTS, 1, 1), requires_grad=False)
        if show_summary:
            summary(self.model, input_size=[(1, cfg.DATASET.CHANNELS, *self.cfg.DATASET.IMG_SIZE)],
                    dtypes=[torch.float32], device="cpu", depth=5)

    def nll_across_batch(self, output, target):
        if self.device.type != "mps":
            output = output.double()
        nll = -target * torch.log(output)
        return torch.mean(torch.sum(nll, dim=(2, 3)))

    def two_d_softmax(self, x):
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

    def scale(self, x):
        y = x / self.temperatures
        return y

    def forward(self, img):
        output = self.model(img.float())
        return self.two_d_softmax(output)

    def shared_step(self, batch):
        # image is batch["x"], landmark coordinates are batch["y"]
        # randomly generate class A or B
        # generate image based on class
        landmarks = batch["y_img"]
        # image = self.get_input(batch, "x")
        if batch["x"].shape[-1] == 1:
            image = rearrange(batch["x"], 'b h w c -> b c h w')
        else:
            image = batch["x"]

        loss, loss_dict, output = self.compute_losses(landmarks, img=image, batch=batch)

        return loss, loss_dict, output

    def compute_losses(self, landmarks, img, batch=None):
        # returns loss, loss_dict, output
        log_prefix = "train" if self.training else "val"
        loss_dict = {}

        target = landmarks
        output = self(img)
        loss = self.nll_across_batch(output, target)

        coordinates = get_coordinates_from_heatmap(output, k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS).flip(-1)
        l2_coordinate_estimate = torch.mean(euclidean_distance(coordinates, batch["y"].float()))
        loss_dict.update({f'{log_prefix}/l2_est': l2_coordinate_estimate.item()})
        loss_dict.update({f'{log_prefix}/nll': loss})
        loss_dict.update({f'{log_prefix}/loss': loss})

        pixel_sizes = batch["pixel_size"].unsqueeze(1)
        coordinates_scaled = coordinates * pixel_sizes
        real_landmarks_scaled = batch["y"].float() * pixel_sizes
        l2_coordinate_estimate_scaled = torch.mean(euclidean_distance(coordinates_scaled, real_landmarks_scaled))
        loss_dict.update({f'{log_prefix}/l2_est_scaled': l2_coordinate_estimate_scaled.item()})

        frequency = 2
        if self.device.type == "mps":
            frequency = 1
        if self.batch_idx == 0 and self.current_epoch % frequency == 0 and self.do_img_logging or (
                not self.training and self.do_img_logging and self.batch_idx == 0):
            # if chosen_class == 0:
            #     gt = gt_fake
            # else:
            #     gt = gt_real
            with torch.no_grad():
                # log training samples
                # log predictions w/ gt, heatmap, heatmap for channel 15, gt for channel 15
                heatmap_prediction_renorm = output
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
                    (target[:, 15] * 240).clamp(0, 240).unsqueeze(1).repeat(1, 3, 1, 1).cpu()
                )

                # for i, val in enumerate(img_pred_log):
                #     print(i, val.min(), val.max(), val.float().mean(), val.float().std())
                img_pred_log = self._get_rows_from_list(torch.stack(img_pred_log))

                try:

                    self.logger.log_image(key=f"Media/{log_prefix}/predictions",
                                          caption=[
                                              f"Images {batch['name']} step pixel mre {l2_coordinate_estimate:.4f} step mm mre {l2_coordinate_estimate_scaled:.4f} scaled by {scale_factor.min().item():.2f}, {scale_factor.max().item():.2f}, {scale_factor.mean().item():.2f}, {scale_factor.std().item():.2f}"],
                                          images=[img_pred_log])
                except OSError as e:
                    print(e)
                del scale_factor, img_pred_log

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

    def unique_test_step(self, batch, batch_idx):
        query_image = self.get_input(batch, "x")
        landmarks = batch["y"]
        output = self(query_image)
        img_log = {}
        img_log["heatmaps"] = output
        img_log["final"] = plot_heatmaps_and_landmarks_over_img(query_image, output, landmarks,
                                                                normalisation_method=self.cfg.DATASET.NORMALISATION)
        img_log["heatmaps_figure"] = plot_heatmaps(output, landmarks)
        return img_log

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 6, 10], gamma=0.1)
        # pretrained scheduler:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 7, 10], gamma=0.1)
        if self.cfg.DENOISE_MODEL.USE_PRETRAINED_IMAGENET_WEIGHTS:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 28, 35], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32, 36, 38], gamma=0.1)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    cfg = config.Config()
    cfg.DATASET.IMG_SIZE = (800, 704)
    cfg.DATASET.NUMBER_KEY_POINTS = 16
    model = CHH(cfg)
