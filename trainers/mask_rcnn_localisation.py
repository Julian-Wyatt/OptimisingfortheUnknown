import os.path
from typing import Any

import lightning as L
import torch
import torchvision
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchinfo import summary
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.utils import make_grid

from core import config
from dataset_utils import dataset_preprocessing_utils
from dataset_utils.visualisations import plot_img_bounding_box_landmarks


class RCNN(L.LightningModule):
    def __init__(self, cfg: config.Config, resize_dir=None, load_external_weights=True):
        super().__init__()
        self.cfg = cfg
        if load_external_weights:
            backbone = torchvision.models.mobilenet_v3_small(weights="DEFAULT").features
        else:
            backbone = torchvision.models.mobilenet_v3_small().features
        backbone.out_channels = 576

        anchor_generator = AnchorGenerator(
            sizes=((128, 256, 320, 512),),
            aspect_ratios=((0.5, 0.75, 1.0, 1.25, 1.5, 1.75),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '4'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = torchvision.models.detection.FasterRCNN(backbone, num_classes=6,
                                                             box_roi_pool=roi_pooler,
                                                             rpn_anchor_generator=anchor_generator,
                                                             rpn_nms_thresh=0.55,
                                                             # rpn_head=rpn_head,
                                                             box_detections_per_img=5,
                                                             )

        self.box_landmark_indices = [None, None]
        if self.cfg.DATASET.NUMBER_KEY_POINTS == 53:
            for i in [[0, 3, 16, 18, 23, 24, 26, 27, 28],
                      [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                      [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 29, 30, 31, 32, 35, 36,
                       37],
                      [1, 2, 33, 34, 51, 52]]:
                self.box_landmark_indices.append(i)

            self.label_to_str = {0: "Background", 1: "full", 2: "ear", 3: "spine", 4: "chin", 5: "eye"}
        else:
            self.label_to_str = {0: "Background", 1: "full"}

        if resize_dir is None:
            self.resize_dir = None
        else:
            self.resize_dir = resize_dir

    def log_image(self, batch_idx, frequency, output, batch, images, log_prefix):
        if (batch_idx == 0 and self.current_epoch % frequency == 0 and self.cfg.TRAIN.LOG_IMAGE) or (
                not self.training and self.cfg.TRAIN.LOG_IMAGE and batch_idx == 0) or (batch["name"] in ["396"]):

            with torch.no_grad():
                # log val samples

                pred_labels = [f"{self.label_to_str[int(label)]}: {score:.3f}" for label, score in
                               zip(output[0]["labels"], output[0]["scores"])]
                gt_labels = [f"{self.label_to_str[int(label)]}" for label in batch["labels"][0]]
                img_pred_log = []
                img_pred_log.append(
                    plot_img_bounding_box_landmarks(images[0], output[0]["boxes"], batch["y"][0], pred_labels))
                img_pred_log.append(
                    plot_img_bounding_box_landmarks(images[0], batch["boxes"][0], batch["y"][0], gt_labels))
                img_pred_log = self._get_rows_from_list(torch.stack(img_pred_log))
                try:
                    self.logger.log_image(key=f"Media/{log_prefix}/predictions",
                                          caption=[
                                              f"Image: {batch['name']} scores {output[0]['scores'][:5]}", ],
                                          images=[img_pred_log])
                except OSError as e:
                    print(e)

    def training_step(self, batch, batch_idx):
        log_prefix = "train"

        images = rearrange(batch["x"], 'b h w c -> b c h w').to(torch.float32)
        images = list(image for image in images)
        targets = [{"boxes": t, "labels": l} for t, l in zip(batch["boxes"], batch["labels"])]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        for k, v in loss_dict.items():
            self.log(f"{log_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        frequency = 5
        if self.current_epoch % frequency == 0 and (batch_idx == 0 or (batch["name"] in ["396"])):
            self.model.eval()
            output = self.model(images)
            self.log_image(batch_idx, frequency, output, batch, images, log_prefix)
            self.model.train()

        return losses

    def calculate_percentage_of_landmarks_in_bounding_box(self, bounding_boxes, landmarks, labels):
        percentages = {}
        for idx, box in enumerate(bounding_boxes):
            count = 0
            if self.box_landmark_indices[labels[idx]] is None:
                selected_landmarks = landmarks
            else:
                selected_landmarks = landmarks[self.box_landmark_indices[labels[idx]]]
            for landmark in selected_landmarks:
                if box[0] <= landmark[0] <= box[2] and box[1] - 8 <= landmark[1] <= box[3] + 8:
                    count += 1
            percentages[labels[idx].cpu().item()] = (count / len(selected_landmarks)) * 100
        return percentages

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        log_prefix = "val"
        images = rearrange(batch["x"], 'b h w c -> b c h w').to(torch.float32)
        images_list = list(image for image in images)

        targets = [{"boxes": t, "labels": l} for t, l in zip(batch["boxes"], batch["labels"])]
        output = self.model(images_list, targets)

        average_percentage = [self.calculate_percentage_of_landmarks_in_bounding_box(output[i]["boxes"], batch["y"][i],
                                                                                     output[i]["labels"]) for i in
                              range(len(images_list))]
        try:
            for idx, percentage in enumerate(average_percentage):
                for k, v in percentage.items():
                    self.log(f"{log_prefix}/percentage_within_box/{self.label_to_str[k]}", v, on_step=False,
                             on_epoch=True,
                             prog_bar=False, logger=True)
        except Exception as e:
            print(e)

        frequency = 5

        if self.device.type == "mps":
            frequency = 1
        self.log_image(batch_idx, frequency, output, batch, images, log_prefix)

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        log_prefix = "test"
        images = rearrange(batch["x"], 'b h w c -> b c h w').to(torch.float32)
        images_list = list(image for image in images)

        targets = [{"boxes": t, "labels": l} for t, l in zip(batch["boxes"], batch["labels"])]
        output = self.model(images_list, targets)

        self.save_subimages(batch, output[0]["boxes"], output[0]["labels"])

        average_percentage = [self.calculate_percentage_of_landmarks_in_bounding_box(output[i]["boxes"], batch["y"][i],
                                                                                     output[i]["labels"]) for i in
                              range(len(images_list))]
        try:
            for idx, percentage in enumerate(average_percentage):
                for k, v in percentage.items():
                    if v != 100:
                        print(
                            f"Percentage of landmarks in bounding box for {self.label_to_str[k]}: {v:.2f}%, name {batch['name']}, step {self.global_step}")
                    self.log(f"{log_prefix}/percentage_within_box/{self.label_to_str[k]}", v, on_step=False,
                             on_epoch=True,
                             prog_bar=False, logger=True)
        except Exception as e:
            print(e)

        self.log_image(0, 1, output, batch, images, log_prefix)

    def save_subimages(self, batch, boxes, labels):

        if not self.cfg.DATASET.SAVE_RCNN_IMAGES:
            return
        if self.resize_dir is None:
            self.resize_dir = os.path.join(self.cfg.DATASET.CACHE_DIR, self.cfg.DATASET.NAME,
                                           f"{self.cfg.DATASET.IMG_SIZE[0]}x{self.cfg.DATASET.IMG_SIZE[1]}-{self.logger.experiment.id}")

        if not os.path.exists(self.resize_dir):
            os.makedirs(self.resize_dir)

        for key, value in batch.items():
            if type(value) == torch.Tensor:
                batch[key] = value.cpu()
            elif type(value) == list and type(value[0]) == torch.Tensor:
                batch[key] = value.cpu()

        dataset_preprocessing_utils.rcnn_save_image(batch, boxes.cpu(), labels.cpu(), self.resize_dir, self.cfg)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Override this method to customize the batch transfer to device
        batch_on_device = {}
        for key, value in batch.items():
            if key == 'image_original':
                batch_on_device[key] = value  # Keep metadata on CPU
            else:
                if type(value) == torch.Tensor:
                    batch_on_device[key] = value.to(device)
                elif type(value) == list and type(value[0]) == torch.Tensor:
                    batch_on_device[key] = [v.to(device) for v in value]  # Move the rest to the specified device
                else:
                    batch_on_device[key] = value
        return batch_on_device

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        grid = rearrange(samples, 'n b c h w -> b n c h w')
        grid = rearrange(grid, 'b n c h w -> (b n) c h w')
        grid = make_grid(grid, nrow=n_imgs_per_row)
        return grid

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMISER.lower() == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                   weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                   betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "adamw":
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                    weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                    betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        else:
            raise ValueError(f"Unknown optimiser: {self.cfg.TRAIN.OPTIMISER}")
        if self.cfg.TRAIN.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8, 16], gamma=0.25, verbose=True)
            return [opt], [scheduler]
        return opt


if __name__ == "__main__":
    cfg = config.Config()
    model = RCNN(cfg)
    # print(torch.tensor([[1, 2, 3, 4]]))
    summary(model.model,
            input_size=[(1, 1, 800, 704)], dtypes=[torch.float32],
            device="cpu", depth=5)
