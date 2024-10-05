"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script utilizes the DetectionAlgorithm class to make predictions using a trained model,
and it saves the prediction results as a CSV file.
Email: xiehaoyu2022@email.szu.edu.cn

"""
import argparse
import init_paths
import os.path
import torch
import os

import tqdm
from einops import rearrange

from core import config
from trainers.mask_rcnn_localisation import RCNN
import imgaug.augmenters as iaa
import cv2
import numpy as np
import shutil


class RCNNExtraction:
    def __init__(self, cfg: config.Config, images_dir: str = "/input/images/lateral-dental-x-rays/"):
        self.cfg = cfg
        # self.images_dir = "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation Set/images/"
        # self.images_dir = "/input/images/lateral-dental-x-rays/"
        self.images_dir = images_dir
        self.images_list = os.listdir(self.images_dir)

        resize_dir = os.path.join(self.images_dir, "..", "processed_images")
        metas = os.path.join(self.images_dir, "..", "processed_images", "meta")

        resize_dir = os.path.abspath(resize_dir)
        metas = os.path.abspath(metas)

        self.RCNN_model = RCNN.load_from_checkpoint(cfg.TRAIN.CHECKPOINT_PATH, cfg=cfg, resize_dir=resize_dir,
                                                    load_external_weights=False)
        self.RCNN_model.eval()

        self.preprocess_pad = iaa.PadToMultiplesOf(32, 32, position="right-bottom")
        self.preprocess_resize = iaa.Resize({"width": 512, "height": "keep-aspect-ratio"})
        # Cache dir should be "/input/images"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.RCNN_model.to(self.device)

        self.loop()

        if not os.path.exists(metas):
            os.makedirs(metas)

        for file in os.listdir(resize_dir):
            if file.endswith("txt") or file.endswith("json"):
                shutil.move(resize_dir + "/" + file, metas)

    def make_batch(self, image_name_full):
        # load image, rescale
        img = cv2.imread(image_name_full, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.uint8)
        original_resolution = img.shape
        img_original = img.copy()
        img = self.preprocess_resize(images=img)
        img_pre_padded_resolution = img.shape
        img = self.preprocess_pad(images=img)
        img = img[0]

        img = img.astype(np.float32) / 255

        landmarks = np.zeros((self.cfg.DATASET.NUMBER_KEY_POINTS, 2)).astype(np.float32)
        landmarks_all_annotators = np.zeros((1, self.cfg.DATASET.NUMBER_KEY_POINTS, 2)).astype(np.float32)
        output = {"x": img, "y": landmarks, "name": f"{image_name_full.split('/')[-1].split('.')[0]}"}
        if len(output["x"].shape) == 2:
            output["x"] = np.expand_dims(output["x"], axis=-1)
        output["image_original"] = img_original
        output["image_original_shape"] = np.array(original_resolution)
        output["landmarks_original"] = landmarks
        output["img_pre_padded_resolution"] = np.array(img_pre_padded_resolution).astype(np.int32)
        output["pixels_per_mm"] = np.array([1, 1])
        output["landmarks_all_annotators"] = landmarks_all_annotators

        for key in output.keys():
            if key != "name":
                output[key] = torch.from_numpy(output[key]).unsqueeze(0).to(self.device)
            else:
                output[key] = [output[key]]

        return output

    def loop(self):
        for i in tqdm.tqdm(range(len(self.images_list))):
            if self.images_list[i].split(".")[-1] not in ["bmp", "png"]:
                continue

            batch = self.make_batch(self.images_dir + self.images_list[i])
            self.preprocess(batch)

    @torch.no_grad()
    def preprocess(self, batch):
        images = rearrange(batch["x"], 'b h w c -> b c h w').to(torch.float32)
        images_list = list(image for image in images)
        output = self.RCNN_model.model(images_list)

        self.RCNN_model.save_subimages(batch, output[0]["boxes"], output[0]["labels"])


if __name__ == "__main__":
    def parse_args():
        # Create an argument parser
        parser = argparse.ArgumentParser(
            description='Detect landmarks conditional on an image')

        parser.add_argument('--config_path', type=str, help='Path to the configuration file',
                            default="configs/default.yaml", required=False)
        parser.add_argument("--saving_root_dir", type=str, help='Path to where to save project files',
                            default="./", required=False)
        parser.add_argument("--input_images_dir", type=str, help='Path to images',
                            default="/input/images/lateral-dental-x-rays/", required=False)
        parser.add_argument("--desc", type=str, help="Description of the run", default="", required=False)
        # Parse the command-line arguments
        args = parser.parse_args()
        return args


    args = parse_args()
    # args.config_path = "./configs/docker_configs/rcnn_sub_images.yaml"
    # args.config_path = "./configs/docker_configs/rcnn_full_images.yaml"
    # args.input_images_dir = "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation Set/images/"
    cfg = config.get_config(args.config_path, args.saving_root_dir)
    algorithm = RCNNExtraction(cfg, args.input_images_dir)
