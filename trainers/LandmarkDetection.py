import json
import os.path
import traceback
from collections import defaultdict

from contextlib import contextmanager
from typing import Any

import imgaug
import lightning as L
import matplotlib.pyplot as plt
import torch

from utils import metrics
from core import config

from einops import rearrange

import numpy as np

import abc
from torchvision.utils import make_grid
from dataset_utils.dataset_preprocessing_utils import get_coordinates_from_heatmap


class LandmarkDetection(L.LightningModule):
    # Landmark Detection Parent Class
    cfg: config.Config
    batch_idx: int

    def __init__(self, cfg: config.Config, use_ema=False):
        super(LandmarkDetection, self).__init__()
        self.cfg = cfg
        self.batch_idx = 0
        self.channels = 0
        self.total_annotators = 1
        self.do_video_logging = False
        self.do_img_logging = False
        self.test_coordinates_errors = defaultdict(list)
        self.test_output = dict()
        self.channels = cfg.DATASET.NUMBER_KEY_POINTS

        self.do_img_logging = cfg.TRAIN.LOG_IMAGE
        self.image_size = cfg.DATASET.IMG_SIZE
        self.use_ema = use_ema

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:
            #     print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:
                #     print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        if ignore_keys is None:
            ignore_keys = list()
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd,
                                                   strict=False) if not only_model else self.model.load_state_dict(sd)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.shape[-1] in [1, 3]:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_landmark_input(self, batch):
        # return dataset.create_landmark_image(batch["y"], self.image_size,
        #                                      eps_window_size=self.cfg.DATASET.LANDMARK_POINT_EPSILON,
        #                                      device=self.device)
        return batch["y_img"].to(self.device)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # returns loss, loss _dict, output
        pass

    def shared_step(self, batch):
        # torch.cuda.empty_cache()
        # image is batch["x"], landmark coordinates are batch["y"]
        landmarks = self.get_landmark_input(batch)
        image = self.get_input(batch, "x")

        loss, loss_dict, output = self(landmarks, img=image, batch=batch)

        return loss, loss_dict, output

    def training_step(self, batch, batch_idx):

        self.batch_idx = batch_idx
        loss, loss_dict, _ = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_start(self) -> None:
        output_str = f"Epoch {self.current_epoch}"
        if "train/loss" in self.trainer.callback_metrics:
            output_str += f" - Train Loss: {self.trainer.callback_metrics['train/loss']:0.4f}"
        if "val/loss" in self.trainer.callback_metrics:
            output_str += f" - Val Loss: {self.trainer.callback_metrics['val/loss']:0.4f}"
        if "val/l2" in self.trainer.callback_metrics:
            output_str += f" - Val L2: {self.trainer.callback_metrics['val/l2']:0.4f}"
        print(output_str)
        # imgaug.seed(np.random.randint(0, 100000))

    @abc.abstractmethod
    def output_to_img_log(self, output, batch, batch_idx=None):
        # img_log returns dict with {"heatmaps": torch.Tensor, "video": list of torch.Tensor,
        #                            "heatmaps_figure": plt.Figure, "final": plt.Figure}
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        _, loss_dict_no_ema, output = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema, output = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        img_log = self.output_to_img_log(output, batch, batch_idx)
        if img_log is None:
            return
        # img_log returns dict with {"heatmaps": torch.Tensor, "video": list of torch.Tensor,
        #                            "heatmaps_figure": plt.Figure, "final": plt.Figure}
        log = metrics.evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                  ddh_metrics=self.cfg.DATASET.LOG_DDH_METRICS,
                                                  pixel_sizes=torch.Tensor([[1, 1]]).to(
                                                      img_log["heatmaps"].device),
                                                  top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
        if batch["landmarks_per_annotator"].shape[1] > 1:

            for annotator in range(batch["landmarks_per_annotator"].shape[1]):
                annotations = batch["landmarks_per_annotator"][:, annotator, :, :]
                log_annotator = metrics.evaluate_landmark_detection(img_log["heatmaps"], annotations,
                                                                    ddh_metrics=False,
                                                                    pixel_sizes=batch["pixel_size"],
                                                                    top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
                self.log(f"val/l2_annotator_{annotator}", np.mean(log_annotator["l2"]), prog_bar=False,
                         logger=True,
                         on_step=False, on_epoch=True)

        if self.cfg.TRAIN.DEBUG:
            if not os.path.exists(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}"):
                os.makedirs(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}")
            with open(
                    f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/epoch{self.current_epoch}_debug.json",
                    "w") as f:
                coords = get_coordinates_from_heatmap(img_log["heatmaps"]).flip(-1)
                output = {"real": batch["y"].tolist(), "pred": coords.tolist(), "name": batch["name"]}
                json.dump(output, f)

        if torch.mean(batch["pixel_size"]) != 1:
            log_scaled = metrics.evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                             ddh_metrics=self.cfg.DATASET.LOG_DDH_METRICS,
                                                             pixel_sizes=batch["pixel_size"],
                                                             top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
            self.log("val/l2_scaled", np.mean(log_scaled["l2"]), prog_bar=False, logger=True, on_step=False,
                     on_epoch=True)
            log["l2_scaled"] = log_scaled["l2"]
        else:
            log["l2_scaled"] = log["l2"]
        sweep_minimiser = np.mean(log["l2_scaled"])
        self.log("val/l2", np.mean(log["l2"]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/l1", np.mean(log["l1"]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/ere", np.mean(log["ere"]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.cfg.DATASET.LOG_DDH_METRICS:
            self.log("val/line_distance", np.mean(log["line_dist"]), prog_bar=False, logger=True, on_step=False,
                     on_epoch=True)
            self.log("val/angle_distance_degrees", np.mean(log["angle_dist"]), prog_bar=False,
                     logger=True, on_step=False,
                     on_epoch=True)
            sweep_minimiser += np.mean(log["line_dist"]) + np.mean(log["angle_dist"])
        self.log("val/sweep_minimiser", sweep_minimiser, prog_bar=False, logger=True, on_step=False,
                 on_epoch=True)
        if self.do_video_logging and "video" in img_log:
            self.log_video(img_log["video"], "val/denoising video",
                           f"denoising_{self.current_epoch}_{batch_idx}")
        if self.do_img_logging and self.batch_idx % 100 == 0:
            if "final" in img_log:
                self.log_image_to_wandb(img_log["final"], "Media/val/Final", ",".join(batch["name"]),
                                        np.mean(log["l2"]))
            # if "heatmaps_figure" in img_log:
            #     self.log_image_to_wandb(img_log["heatmaps_figure"], "Media/val/Heatmaps", ",".join(batch["name"]))
        if "final" in img_log and type(img_log["final"]) is plt.Figure:
            img_log["final"].clf()
            plt.close("all")
        elif "final" in img_log:
            del img_log["final"]
        if "heatmaps_figure" in img_log and type(img_log["heatmaps_figure"]) is plt.Figure:
            img_log["heatmaps_figure"].clf()
            plt.close("all")
        elif "heatmaps_figure" in img_log:
            del img_log["heatmaps_figure"]

        for key in log:
            self.test_coordinates_errors[f"VAL_{key}"].append(log[key])

    def on_validation_epoch_end(self) -> None:
        errors_dict = {key: np.concatenate(val).reshape(-1, self.channels) for key, val in
                       self.test_coordinates_errors.items() if key in ["VAL_l2", "VAL_l2_scaled"]}
        sdr_metric = "VAL_l2"
        if "VAL_l2_scaled" in errors_dict:
            sdr_metric = "VAL_l2_scaled"
        for k in [sdr_metric]:
            if k in self.test_coordinates_errors:
                if "scaled" in k:
                    pixel_sizes = [2.0, 2.5, 3.0, 4.0]
                else:
                    pixel_sizes = [5, 10, 20, 40]
                sdrs = metrics.success_detection_rates(errors_dict[k].flatten(), pixel_sizes)

                for i, sdr in enumerate(sdrs):
                    self.log(f"val/sdr_{k[4:]}_{pixel_sizes[i]}", sdr, prog_bar=False, logger=True)

                self.log(f"val/{k[4:]}_std", np.std(errors_dict[k]), prog_bar=False, logger=True)

                self.current_val_2mm_sdr = sdrs[0]
        self.current_val_mre = np.mean(errors_dict[sdr_metric])

        self.test_coordinates_errors = defaultdict(list)

    @abc.abstractmethod
    def unique_test_step(self, batch, batch_idx):
        pass

    def on_test_start(self) -> None:
        if not os.path.exists(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}"):
            os.makedirs(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        img_log = self.unique_test_step(batch, batch_idx)

        # {}

        if self.cfg.TRAIN.LOG_TEST_METRICS:

            log = metrics.evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                      ddh_metrics=self.cfg.DATASET.LOG_DDH_METRICS,
                                                      pixel_sizes=torch.Tensor([[1, 1]]).to(
                                                          img_log["heatmaps"].device),
                                                      top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
            log_scaled = metrics.evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                             ddh_metrics=self.cfg.DATASET.LOG_DDH_METRICS,
                                                             pixel_sizes=batch["pixel_size"],
                                                             top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)

            if batch["landmarks_per_annotator"].shape[1] > 1:

                for annotator in range(batch["landmarks_per_annotator"].shape[1]):
                    annotations = batch["landmarks_per_annotator"][:, annotator, :, :]
                    log_annotator = metrics.evaluate_landmark_detection(img_log["heatmaps"], annotations,
                                                                        ddh_metrics=False,
                                                                        pixel_sizes=batch["pixel_size"],
                                                                        top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
                    self.log(f"test/l2_annotator_{annotator}", np.mean(log_annotator["l2"]), prog_bar=False,
                             logger=True,
                             on_step=False, on_epoch=True)
                    self.test_coordinates_errors[f"l2_scaled_annotator_{annotator}"].append(log_annotator["l2"])

            if not os.path.exists(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}"):
                os.makedirs(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}")

            if self.cfg.TRAIN.DEBUG:
                with open(
                        f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/test_batch_idx{batch_idx}_debug.json",
                        "w") as f:
                    coords = get_coordinates_from_heatmap(img_log["heatmaps"]).flip(-1)
                    output = {"real": batch["y"].tolist(), "pred": coords.tolist(), "name": batch["name"]}
                    json.dump(output, f)
                if np.any(np.isnan(log["ere"])):
                    torch.save(img_log["heatmaps"], f"test_batch_idx{batch_idx}_debug_nan.json")

            for i in range(len(batch["name"])):
                msg = f"Image: {batch['name'][i]}\t"
                for radial_error in log_scaled["l2"][i]:
                    msg += f"\t{radial_error:06.3f} mm"
                msg += f"\taverage: {np.mean(log_scaled['l2'][i]):06.3f} mm"
                print(msg)

                with open(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/test_results.csv",
                          "a") as f:
                    output = [batch['name'][i], *[str(x) for x in log["l2"][i]], str(np.mean(log['l2'][i]))]
                    if self.cfg.DATASET.LOG_DDH_METRICS:
                        output += [str(np.mean(log["line_dist"][i]))]
                        output += [str(np.mean(log["angle_dist"][i]))]

                    f.write(",".join(output) + "\n")

            for k, v in log.items():
                self.test_coordinates_errors[k].append(v)
                self.log(f"test/{k}", np.mean(v), prog_bar=False, logger=True, on_step=True, on_epoch=True)

            for k, v in log_scaled.items():
                self.test_coordinates_errors[f"{k}_scaled"].append(v)
                if "l2" in k:
                    self.log(f"test/{k}_scaled", np.mean(v), prog_bar=False, logger=True, on_step=True,
                             on_epoch=True)
        else:
            # output to csv with format
            # image name, landmark0x, landmark0y, ...,n
            file_path = f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/test_landmarks.csv"
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    header = ["image file"] + [f"p{i + 1}x,p{i + 1}y" for i in range(self.channels)]
                    f.write(",".join(header) + "\n")
            coordinates_all_batch = get_coordinates_from_heatmap(img_log["heatmaps"],
                                                                 k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)

            for i in range(len(batch["name"])):
                with open(file_path, "a") as f:
                    # convert heatmaps to coordinates
                    coordinates = coordinates_all_batch[i]
                    # scale coordinates to the original screen size
                    # ie 800x640 -> 1935x2400 or other resolutions
                    # coordinates = coordinates.scale

                    invert_augs_dict = {"shift": batch["shift"][i],
                                        "pre-resize shape": batch["pre-resize shape"][i],
                                        }

                    if "scale_factor" in batch:
                        invert_augs_dict["scale_factor"] = batch["scale_factor"][i]

                    coordinates = self.trainer.test_dataloaders.dataset.preprocess.invert(coordinates, invert_augs_dict)

                    output = [f"{int(batch['name'][i]):03d}.bmp"] + [str(int(i)) for i in
                                                                     coordinates.flatten().tolist()]
                    f.write(",".join(output) + "\n")

        coordinates = get_coordinates_from_heatmap(img_log["heatmaps"],
                                                   k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS).flip(-1)
        l2_coordinate_estimate = torch.mean(metrics.euclidean_distance(coordinates, batch["y"].float()))
        if not self.cfg.TRAIN.LOG_TEST_METRICS:
            self.log(f'test/l2', l2_coordinate_estimate.item(), prog_bar=False, logger=True, on_step=True,
                     on_epoch=True)
        pixel_sizes = batch["pixel_size"].unsqueeze(1)
        coordinates_scaled = coordinates * pixel_sizes
        real_landmarks_scaled = batch["y"].float() * pixel_sizes
        l2_coordinate_estimate_scaled = torch.mean(
            metrics.euclidean_distance(coordinates_scaled, real_landmarks_scaled))
        if not self.cfg.TRAIN.LOG_TEST_METRICS:
            self.log(f'test/l2_scaled', l2_coordinate_estimate_scaled.item(), prog_bar=False, logger=True,
                     on_step=True, )

        if self.do_video_logging and batch_idx % 20 == 0 and "video" in img_log:
            self.log_video(img_log["video"], "test/Denoising", f"denoising_{self.current_epoch}_{batch_idx}")
        if self.do_img_logging and "final" in img_log:
            self.log_image_to_wandb(img_log["final"], "Media/test/Overlay", ",".join(batch["name"]),
                                    l2_coordinate_estimate_scaled.item())

    def on_test_epoch_end(self) -> None:
        # output to csv with format
        # image name, landmark0,...,n average l2, if ddh metrics then angle and line distance
        # columns: image name, landmark0, landmark1, landmark2, landmark3, landmark4, average l2, average angle, average line distance

        if not self.cfg.TRAIN.LOG_TEST_METRICS:
            print(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/test_landmarks.csv")
            self.logger.log_hyperparams({"test_landmarks_csv": f"{self.logger.root_dir}/test_landmarks.csv"})
            return
        print({key: np.concatenate(val).shape for key, val in self.test_coordinates_errors.items()})

        def print_error_stats(error_name, error_values, is_scaled=False, tostdout=True):
            mean_error = np.mean(error_values)
            std_error = np.std(error_values)
            unit = "mm" if is_scaled else "pixels"
            if "angle" in error_name:
                unit = "degrees"
            if tostdout:
                print(f"{error_name} {mean_error:.3f} +- {std_error:.3f} {unit}")
            return f" {mean_error:.3f} +- {std_error:.3f} {unit} "

        def print_sdr_stats(error_name, error_values, tostdout=True, prefix=""):
            if "scaled" in error_name or "x0" in error_name:
                pixel_sizes = [2.0, 2.5, 3.0, 4.0]
                unit = "mm"
            else:
                pixel_sizes = [5, 10, 20, 40]
                unit = "pixels"
            sdr_stats = metrics.success_detection_rates(error_values.flatten(), pixel_sizes)

            if tostdout:
                print(
                    f"{prefix} test {error_name} sdr stats for {unit} sizes {pixel_sizes} {[f'{sdr:.3f}' for sdr in sdr_stats]}")

        error_metrics = ['l2', "l2_scaled", 'l1', 'ere']
        if self.cfg.DIFFUSION.GEN_BIG_T_HEATMAP:
            error_metrics.append("l2_x0")
            error_metrics.append("ere_x0")
        if self.total_annotators > 1:
            for i in range(self.total_annotators):
                error_metrics.append(f"l2_scaled_annotator_{i}")
        errors_dict = {key: np.concatenate(val).reshape(-1, self.channels) for key, val in
                       self.test_coordinates_errors.items() if key in error_metrics}
        with open(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/summary_test_results.csv",
                  "a") as f:
            output = [np.mean(errors_dict["l2"]),
                      *[metrics.success_detection_rates(errors_dict["l2"].flatten(), [5, 10, 20, 40])]]
            f.write(",".join([str(x) for x in output]) + "\n")
        self.logger.log_hyperparams({"test_csv": f"{self.logger.root_dir}/test_results.csv"})
        self.logger.log_hyperparams({"summary_test_csv": f"{self.logger.root_dir}/summary_test_results.csv"})
        print("-----------------------------------")
        print("Test Results")
        print("-----------------------------------")
        print("Channel wise mean errors")

        # change to output landmark detail all in one line and new line for new metric []
        # average l2 error for each landmark
        try:
            outputs = ["" for _ in range(len(errors_dict.keys()))]
            for i in range(self.channels):

                for j, val in enumerate(errors_dict.items()):
                    keys, values = val
                    # self.log(f"test/landmark_{i + 1}_{keys}", np.mean(values[:, i]))
                    outputs[j] += print_error_stats(keys, values[:, i], "scaled" in keys or "x0" in keys,
                                                    tostdout=False)

            for j, val in enumerate(errors_dict.items()):
                keys, values = val
                print(f"{keys} {outputs[j]}")

            for keys in ["l2", "l2_scaled", "l2_x0"]:
                if keys in errors_dict:
                    print(f"{keys} SDR stats")
                    for i in range(self.channels):
                        print_sdr_stats(keys, errors_dict[keys][:, i], prefix=f"Landmark {i}")
        except Exception as e:
            print(e, traceback.format_exc())

        print("\n-----------------------------------")

        print("Overall mean errors")

        if self.cfg.DATASET.LOG_DDH_METRICS:
            error_metrics.extend(['line_dist', "line_dist_x0", 'angle_dist', "angle_dist_x0"])
            errors_dict["line_dist"] = np.concatenate(self.test_coordinates_errors["line_dist"]).reshape(-1, 4)

            errors_dict["angle_dist"] = np.concatenate(self.test_coordinates_errors["angle_dist"]).reshape(-1, 2)

            if self.cfg.DIFFUSION.GEN_BIG_T_HEATMAP:
                errors_dict["line_dist_x0"] = np.concatenate(self.test_coordinates_errors["line_dist_x0"]).reshape(
                    -1,
                    4)
                errors_dict["angle_dist_x0"] = np.concatenate(
                    self.test_coordinates_errors["angle_dist_x0"]).reshape(-1,
                                                                           2)

        for error in error_metrics:
            print_error_stats(error, errors_dict[error], "scaled" in error or "x0" in error)

        print("-----------------------------------")

        for error in ['l2', 'l2_x0']:
            if error in errors_dict:
                print_sdr_stats(error, errors_dict[error])

        print("-----------------------------------")

        for error in ['l2_scaled']:
            print_sdr_stats(error, errors_dict[error])
        sdr_stats = metrics.success_detection_rates(errors_dict["l2_scaled"].flatten(), [2.0, 2.5, 3.0, 4.0])
        for i, dist in enumerate([2, 2.5, 3, 4]):
            self.log(f"test/sdr/{dist}", sdr_stats[i])

        print("-----------------------------------")

        if self.cfg.DATASET.LOG_DDH_METRICS:
            print_sdr_stats("line_dist", errors_dict["line_dist"])
            if self.cfg.DIFFUSION.GEN_BIG_T_HEATMAP:
                print_sdr_stats("line_dist_x0", errors_dict["line_dist_x0"])

        print("-----------------------------------")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def log_image_to_wandb(self, figure: plt.Figure, log_name, filename, mre=None):
        import wandb
        if not os.path.exists(f"./{self.logger.experiment.dir}/images"):
            os.makedirs(f"./{self.logger.experiment.dir}/images")

        figure.savefig(f"./{self.logger.experiment.dir}/images/{filename}.png")

        if not log_name.lower().startswith("media"):
            log_name = f"Media/{log_name}"
        self.logger.experiment.log(
            {f"{log_name}": wandb.Image(f"./{self.logger.experiment.dir}/images/{filename}.png",
                                        caption=filename + f"| MRE: {mre:.4f}")})

        plt.clf()
        plt.close("all")

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            with self.ema_scope():
                checkpoint['state_dict'] = self.state_dict()

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                               betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        return opt
