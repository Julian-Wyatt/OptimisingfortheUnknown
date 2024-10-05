import os
from collections import defaultdict
from typing import Any

import lightning as L
import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from core import config
from core.config import Config
from dataset_utils.dataset_preprocessing_utils import get_coordinates_from_heatmap
from dataset_utils.visualisations import plot_heatmaps_and_landmarks_over_img
from trainers.CHH_reimplementation import CHH

from utils.metrics import success_detection_rates, euclidean_distance
from utils.util import save_to_csv


def make_model_paths(paths, saving_root_dir, name, model_type):
    return [f"{saving_root_dir}/tmp/checkpoints/{name}-{model_type}-{path}" for path in paths]


class LandmarkEnsembleDetector(L.LightningModule):
    def __init__(self, cfg: Config, saving_root_dir="./", model_states=None):
        super(LandmarkEnsembleDetector, self).__init__()
        self.cfg = cfg

        if cfg.TRAIN.MODEL_TYPE.lower() == "main":
            from trainers.detector import RandomLandmarkDetector
            baseModel = RandomLandmarkDetector
        elif cfg.TRAIN.MODEL_TYPE.lower() == "chh":
            baseModel = CHH
        else:
            raise ValueError(f"Model type {cfg.TRAIN.MODEL_TYPE} not recognised")

        # f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{cfg.TRAIN.CHECKPOINT_FILE}"
        print(make_model_paths(cfg.TRAIN.ENSEMBLE_MODEL_PATHS, saving_root_dir, cfg.DATASET.NAME,
                               cfg.TRAIN.MODEL_TYPE))

        # Initialize models, either from model paths or model states
        if model_states is None:
            # Assume we have paths and load models
            model_paths = make_model_paths(cfg.TRAIN.ENSEMBLE_MODEL_PATHS, saving_root_dir, cfg.DATASET.NAME,
                                           cfg.TRAIN.MODEL_TYPE)
            self.models = nn.ModuleList(
                [baseModel.load_from_checkpoint(path, cfg=cfg, show_summary=False, load_external_weights=True).cpu()
                 for path in model_paths])
        else:
            # Initialize models and load state dicts from the checkpoint
            self.models = nn.ModuleList(
                [baseModel(cfg=cfg, show_summary=False, load_external_weights=False) for _ in range(len(model_states))])
            for model, state_dict in zip(self.models, model_states):
                model.load_state_dict(state_dict)

        # average heatmap or average coordinate
        # "heatmap" or "coordinate" or "max"
        self.ensemble_strategy = cfg.TRAIN.ENSEMBLE_STRATEGY
        self.test_coordinates_errors = defaultdict(list)
        self.automatic_optimization = False

    def forward(self, x):
        x = x.to(self.device)
        # x is B, C, H, W

        # heatmaps is N, B, C, H, W

        heatmaps = torch.zeros(
            (len(self.models), x.shape[0], self.cfg.DATASET.NUMBER_KEY_POINTS, x.shape[2], x.shape[3]),
            device=self.device)

        for i, model in enumerate(self.models):
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                heatmap = model(x)
            # heatmap = heatmap.cpu()

            heatmaps[i] = heatmap

            self.models[i] = model.cpu()
            torch.cuda.empty_cache()
        return heatmaps

    def get_img(self, batch):
        if batch["x"].shape[-1] == 1:
            image = rearrange(batch["x"], 'b h w c -> b c h w')
        else:
            image = batch["x"]
        return image

    def training_step(self, batch, batch_idx):
        optims = self.optimizers()
        image = self.get_img(batch)
        heatmaps = self.forward(image)
        heatmaps = torch.mean(heatmaps, dim=0)
        nll = -batch["y_img"] * torch.log(heatmaps)
        loss = torch.mean(torch.sum(nll, dim=(2, 3)))
        self.manual_backward(loss)
        for optim in optims:
            optim.step()
        for optim in optims:
            optim.zero_grad()
        return loss

    def landmark_prediction(self, x, visualise=False, ground_truths=None):
        heatmaps = self.forward(x).cpu()

        final_landmarks = []

        if self.ensemble_strategy == "coordinate" or self.ensemble_strategy == "all":
            total_landmark_predictions = np.zeros((len(self.models), self.cfg.DATASET.NUMBER_KEY_POINTS, 2))
            for i in range(len(self.models)):
                coordinate_prediction = get_coordinates_from_heatmap(heatmaps[i], k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
                total_landmark_predictions[i] = coordinate_prediction
                # if ground_truths is not None and self.logger is not None:
                #     final_overlay = plot_heatmaps_and_landmarks_over_img(x, heatmaps[i], ground_truths,
                #                                                          return_as_array=True)
                #     self.logger.log_image(key=f"Media/test/predictions_{i}",
                #                           caption=[
                #                               f"{i}'th model"],
                #                           images=[final_overlay])

            final_landmarks.append(torch.tensor(np.mean(total_landmark_predictions, axis=0)).unsqueeze(0))

        if self.ensemble_strategy == "max" or self.ensemble_strategy == "all":
            heatmap, _ = torch.max(heatmaps, dim=0)
            maxed_landmarks = get_coordinates_from_heatmap(heatmap, k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
            final_landmarks.append(maxed_landmarks)
        if self.ensemble_strategy == "heatmap" or self.ensemble_strategy == "all":
            heatmap = torch.mean(heatmaps, dim=0)
            coordinate_prediction = get_coordinates_from_heatmap(heatmap, k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
            final_landmarks.append(coordinate_prediction)

        if visualise and self.logger is not None and ground_truths is not None:
            final_overlay = plot_heatmaps_and_landmarks_over_img(x, torch.mean(heatmaps, dim=0), ground_truths,
                                                                 return_as_array=True,
                                                                 normalisation_method=self.cfg.DATASET.NORMALISATION)
            return final_landmarks, final_overlay

        return final_landmarks

    def on_test_start(self) -> None:
        if not os.path.exists(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}"):
            os.makedirs(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.test_step(batch, batch_idx, log_prefix="val")

    def test_step(self, batch, batch_idx, log_prefix="test") -> STEP_OUTPUT:
        image = self.get_img(batch)
        batch["y"] = batch["y"].cpu()
        batch[f"pixel_size"] = batch[f"pixel_size"].cpu()
        if batch_idx % 10 == 0:
            predicted_landmarks, overlay = self.landmark_prediction(image, visualise=True,
                                                                    ground_truths=batch["y"])
            l2_coordinate_estimate = torch.mean(
                euclidean_distance(predicted_landmarks[0].clone().flip(-1), batch["y"].float()))
            pixel_sizes = batch["pixel_size"].unsqueeze(1)
            coordinates_prediction_scaled = predicted_landmarks[0].clone().flip(-1) * pixel_sizes
            real_landmarks_scaled = batch["y"].float() * pixel_sizes
            l2_coordinate_estimate_scaled = torch.mean(
                euclidean_distance(coordinates_prediction_scaled, real_landmarks_scaled))
            # if batch['name'][0] == '018':
            #     print("---------prediction------")
            #     print(coordinates_prediction_scaled)
            #     print("---------gt------")
            #     print(real_landmarks_scaled)
            #     print("---------error------")
            #     print(euclidean_distance(coordinates_prediction_scaled, real_landmarks_scaled))
            try:
                self.logger.log_image(key=f"Media/test/predictions",
                                      caption=[
                                          f"Images {batch['name']} step pixel mre {l2_coordinate_estimate.item():.4f} step mm mre {l2_coordinate_estimate_scaled.item():.4f}"],
                                      images=[overlay])
            except OSError as e:
                print(e)
        else:
            predicted_landmarks = self.landmark_prediction(image)

        if self.cfg.TRAIN.ENSEMBLE_STRATEGY != "all":
            predicted_landmarks = predicted_landmarks[0]
            # log mre
            coordinate_prediction_flipped = predicted_landmarks.flip(-1)
            l2_coordinate_estimate = torch.mean(euclidean_distance(coordinate_prediction_flipped, batch[f"y"]))

            pixel_sizes = batch[f"pixel_size"].unsqueeze(1)
            coordinates_prediction_scaled = coordinate_prediction_flipped * pixel_sizes
            real_landmarks_scaled = batch[f"y"].float() * pixel_sizes
            l2_coordinate_estimate_scaled = torch.mean(
                euclidean_distance(coordinates_prediction_scaled, real_landmarks_scaled))
            self.log(f"{log_prefix}/l2_est_epoch", l2_coordinate_estimate.item(), prog_bar=False, logger=True,
                     on_epoch=True)
            self.log(f"{log_prefix}/l2_scaled_epoch", l2_coordinate_estimate_scaled.item(), prog_bar=False, logger=True,
                     on_epoch=True)
            self.test_coordinates_errors[f"{log_prefix}_l2_scaled"].append(
                euclidean_distance(coordinates_prediction_scaled, real_landmarks_scaled).numpy())
        else:
            for i in zip(predicted_landmarks, ["coordinate", "max", "heatmap"]):
                prediction, strategy = i

                coordinate_prediction_flipped = prediction.clone().flip(-1)
                l2_coordinate_estimate = torch.mean(euclidean_distance(coordinate_prediction_flipped, batch[f"y"]))

                pixel_sizes = batch[f"pixel_size"].unsqueeze(1)
                coordinates_prediction_scaled = coordinate_prediction_flipped * pixel_sizes
                real_landmarks_scaled = batch[f"y"].float() * pixel_sizes
                l2_coordinate_estimate_scaled = torch.mean(
                    euclidean_distance(coordinates_prediction_scaled, real_landmarks_scaled))
                self.log(f"{log_prefix}/l2_est_epoch_{strategy}", l2_coordinate_estimate.item(), prog_bar=False,
                         logger=True,
                         on_epoch=True, on_step=True)
                self.log(f"{log_prefix}/l2_scaled_epoch_{strategy}", l2_coordinate_estimate_scaled.item(),
                         prog_bar=False,
                         logger=True,
                         on_epoch=True, on_step=True)
                if strategy == "coordinate":
                    self.test_coordinates_errors[f"{log_prefix}_l2_scaled"].append(
                        euclidean_distance(coordinates_prediction_scaled, real_landmarks_scaled).numpy())
            predicted_landmarks = predicted_landmarks[0]

        if log_prefix == "test":  # and self.trainer.test_dataloaders.dataset.partition == "testing":
            inverted_landmarks = self.invert_heatmap_coordinate_to_original_res(predicted_landmarks[0],
                                                                                batch['shift'][0].cpu(),
                                                                                batch['scale_factor'][
                                                                                    0].cpu()).unsqueeze(0)
            save_to_csv(self.cfg.TRAIN.SAVING_ROOT_DIR,
                        self.logger.experiment.id + "-"+self.trainer.test_dataloaders.dataset.partition, inverted_landmarks,
                        batch,
                        self.cfg.DATASET.NUMBER_KEY_POINTS)

    def on_validation_epoch_end(self) -> None:
        self.calculate_sdr_stats(prefix="val")

    def on_test_epoch_end(self) -> None:
        self.calculate_sdr_stats(prefix="test")
        print(f"Saved to {self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/test_landmarks.csv")

    def invert_heatmap_coordinate_to_original_res(self, landmarks, shift, scale):

        landmarks[:] = landmarks[:] * scale
        landmarks[:, 0] += shift[0]
        landmarks[:, 1] += shift[1]
        return landmarks

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, cfg=None, **kwargs):
        # Load checkpoint and retrieve the saved states
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        checkpoint_state_dict = checkpoint['state_dict']
        # Pass the saved model states into the init method
        model_states = [checkpoint_state_dict[f'model_{i}_state_dict'] for i in range(len(checkpoint_state_dict) - 1)]

        # Initialize the model using the loaded states
        model = cls(cfg=cfg, model_states=model_states, **kwargs)

        # Load the rest of the state dict
        model.load_state_dict(checkpoint_state_dict['lightning_state_dict'])
        return model

    def state_dict(self, **kwargs):
        # Create a dictionary containing the state dicts of all models
        ensemble_state_dict = {}
        for i, model in enumerate(self.models):
            ensemble_state_dict[f'model_{i}_state_dict'] = model.state_dict()
        ensemble_state_dict['lightning_state_dict'] = super().state_dict()
        return ensemble_state_dict

    # def load_state_dict(self, state_dict, **kwargs):
    #     # Load the state dicts for all models
    #     super().load_state_dict(state_dict['lightning_state_dict'])
    #     for i, model in enumerate(self.models):
    #         model.load_state_dict(state_dict[f'model_{i}_state_dict'])

    def configure_optimizers(self):
        optims = []
        for model in self.models:
            if self.cfg.TRAIN.OPTIMISER.lower() == "adam":
                opt_g = torch.optim.Adam(model.parameters(), lr=self.cfg.TRAIN.LR,
                                         weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                         betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
            elif self.cfg.TRAIN.OPTIMISER.lower() == "adamw":
                opt_g = torch.optim.AdamW(model.parameters(), lr=self.cfg.TRAIN.LR,
                                          weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                          betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
            elif self.cfg.TRAIN.OPTIMISER.lower() == "sgd":
                opt_g = torch.optim.SGD(model.parameters(), lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                        momentum=0.9)
            elif self.cfg.TRAIN.OPTIMISER.lower() == "rmsprop":
                opt_g = torch.optim.RMSprop(model.parameters(), lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                            alpha=0.9)
            else:
                raise ValueError(f"Optimiser {self.cfg.TRAIN.OPTIMISER} not recognised")
            optims.append(opt_g)
        return optims, []

    def calculate_sdr_stats(self, prefix="test") -> None:
        errors_dict = {key: np.concatenate(val).reshape(-1, self.cfg.DATASET.NUMBER_KEY_POINTS) for key, val in
                       self.test_coordinates_errors.items() if key in [f"{prefix}_l2_scaled"]}

        sdr_metric = f"{prefix}_l2"
        if f"{prefix}_l2_scaled" in errors_dict:
            sdr_metric = f"{prefix}_l2_scaled"
        for k in [sdr_metric]:
            if k in self.test_coordinates_errors:
                if "scaled" in k:
                    pixel_sizes = [2.0, 2.5, 3.0, 4.0]
                else:
                    pixel_sizes = [5, 10, 20, 40]
                sdrs = success_detection_rates(errors_dict[k].flatten(), pixel_sizes)

                for i, sdr in enumerate(sdrs):
                    self.log(f"{prefix}/sdr_{k[4:]}_{pixel_sizes[i]}", sdr, prog_bar=False, logger=True)

                self.log(f"{prefix}/{k[4:]}_std", np.std(errors_dict[k]), prog_bar=False, logger=True)
        self.test_coordinates_errors = defaultdict(list)


if __name__ == "__main__":
    ckpt = torch.load("checkpoints/CustomUnet/ensemble_convnext_tiny_v1.ckpt")
    cfg = config.get_config("configs/docker_configs/next_ensemble_tiny.yaml")

    model = LandmarkEnsembleDetector.load_from_checkpoint("checkpoints/CustomUnet/ensemble_convnext_tiny_v1.ckpt",
                                                          cfg=cfg)
