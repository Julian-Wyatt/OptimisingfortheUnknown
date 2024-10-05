import glob
import json
import multiprocessing
import os
import time
from collections import defaultdict

import imgaug
import matplotlib.pyplot as plt
import skimage.exposure
import torchvision.transforms
from einops import rearrange, reduce
# from line_profiler import profile
from torch.utils.data import Dataset, DataLoader

import cv2

from utils import util
from core import config
from dataset_utils import generate_landmark_images_numpy
from dataset_utils.dataset_preprocessing_utils import LandmarkPreprocessing, renormalise, \
    simulate_x_ray_artefacts

from dataset_utils.generate_landmark_images import *
from core.config import Config

import imgaug as ia

ia.seed(42)
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

import pandas as pd


class LandmarkDataset(Dataset):

    def __init__(self, cfg: Config, partition="training", augment=False):

        # handle directories
        self.root_dir = cfg.DATASET.ROOT_DIR

        self.img_dirs = cfg.DATASET.IMG_DIRS
        self.img_ext = cfg.DATASET.IMG_EXT
        self.tensor_device = util.get_device()

        if type(cfg.DATASET.LABEL_DIR) is str:

            if cfg.DATASET.LABEL_DIR.endswith(".csv"):
                self.annotation_dirs = [""]
                self.annotations_dataframe = pd.read_csv(
                    f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{cfg.DATASET.LABEL_DIR}")
            else:
                self.annotation_dirs = [cfg.DATASET.LABEL_DIR]
        else:
            self.annotation_dirs = cfg.DATASET.LABEL_DIR

        self.total_landmarks = cfg.DATASET.NUMBER_KEY_POINTS

        self.dataset_pixels_per_mm = cfg.DATASET.PIXELS_PER_MM

        # handle partitions
        self.partition = partition
        self.partition_file = f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}"
        if cfg.DATASET.PARTITION_FILE == "":
            self.partition_file += "partitions/partition_0.7_0.15_0.15_shuffled.json"
        else:
            self.partition_file += cfg.DATASET.PARTITION_FILE

        self.handle_partition()
        self.annotations = []
        self.images = []
        self.metas = []

        # perform augmentation
        self.augment = augment

        self.preprocess = LandmarkPreprocessing(cfg)

        self.IMG_SIZE = cfg.DATASET.IMG_SIZE

        self.SIGMAS = np.ones(self.total_landmarks) * cfg.DATASET.GT_SIGMA

        self.USE_GAUSSIAN = cfg.DATASET.GT_SIGMA > 0

        self.scale_transform_skew = iaa.Affine(scale={"x": (1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                                                      "y": (1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE)},
                                               order=3)
        self.scale_transform = iaa.Affine(scale=(1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                                          order=3)
        self.USE_SKEWED_SCALE_RATE = cfg.AUGMENTATIONS.USE_SKEWED_SCALE_RATE
        self.SIMULATE_XRAY_ARTEFACTS_RATE = cfg.AUGMENTATIONS.SIMULATE_XRAY_ARTEFACTS_RATE
        # augmentations
        self.transform = iaa.Sequential([
            iaa.Cutout(nb_iterations=(0, cfg.AUGMENTATIONS.CUTOUT_ITERATIONS),
                       size=(cfg.AUGMENTATIONS.CUTOUT_SIZE_MIN, cfg.AUGMENTATIONS.CUTOUT_SIZE_MAX),
                       squared=False,
                       cval=(0, 255)),
            iaa.Affine(rotate=(-cfg.AUGMENTATIONS.ROTATION, cfg.AUGMENTATIONS.ROTATION),
                       # scale=(1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                       translate_px={"x": cfg.AUGMENTATIONS.TRANSLATION_X, "y": cfg.AUGMENTATIONS.TRANSLATION_Y},
                       mode=["reflect", "edge", "constant"], shear=(-cfg.AUGMENTATIONS.SHEAR, cfg.AUGMENTATIONS.SHEAR)),
            iaa.Multiply(mul=(1 - cfg.AUGMENTATIONS.MULTIPLY, 1 + cfg.AUGMENTATIONS.MULTIPLY)),
            iaa.Sometimes(cfg.AUGMENTATIONS.BLUR_RATE, iaa.GaussianBlur(sigma=(0, 1.5))),
            iaa.GammaContrast((cfg.AUGMENTATIONS.CONTRAST_GAMMA_MIN, cfg.AUGMENTATIONS.CONTRAST_GAMMA_MAX)),
            iaa.ElasticTransformation(alpha=(0, cfg.AUGMENTATIONS.ELASTIC_TRANSFORM_ALPHA),
                                      sigma=cfg.AUGMENTATIONS.ELASTIC_TRANSFORM_SIGMA, order=3),
        ], random_order=False)
        self.low_transform = iaa.Sequential([
            iaa.Affine(rotate=2,
                       scale=(0.95, 1.05),
                       translate_px={"x": [-3, 3], "y": [-2, 3]},
                       mode="edge", ),
            iaa.Multiply(mul=(0.65, 1.35)),
            iaa.GammaContrast((0.5, 2)),
            iaa.ElasticTransformation(alpha=(0, 250),
                                      sigma=30, order=3),
        ], random_order=False)

        self.invert_transform = iaa.Invert(cfg.AUGMENTATIONS.INVERT_RATE)

        self.coarse_dropout = iaa.CoarseDropout(0.02, size_percent=0.08)
        self.addative_gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, cfg.AUGMENTATIONS.GAUSSIAN_NOISE * 255))
        self.coarse_dropout_rate = cfg.AUGMENTATIONS.COARSE_DROPOUT_RATE
        self.addative_gaussian_noise_rate = cfg.AUGMENTATIONS.ADDATIVE_GAUSSIAN_NOISE_RATE

        self.FLIP_INITIAL_COORDINATES = cfg.AUGMENTATIONS.FLIP_INITIAL_COORDINATES
        self.LANDMARK_POINT_EPSILON = cfg.DATASET.LANDMARK_POINT_EPSILON

        self.cfg_INT_TO_FLOAT = cfg.DATASET.INT_TO_FLOAT
        self.cfg_NORMALISATION_METHOD = cfg.DATASET.NORMALISATION

        self.store_in_ram = True
        self.ram = defaultdict(dict)

        # normalisation funciton
        self.normalise = normalise

        if cfg.DATASET.CACHE_DIR == "":
            cfg.DATASET.CACHE_DIR = "datasets/dataset_cache"

        self.USE_ALL_RCNN_IMAGES = cfg.DATASET.USE_ALL_RCNN_IMAGES

        self.USE_PROCESSED_IMAGES = cfg.DATASET.PROCESSED_IMGS_DIR != ""
        if cfg.DATASET.PROCESSED_IMGS_DIR == "" or cfg.DATASET.PROCESSED_IMGS_DIR == "deterministic_challenge_preprocessing":
            if cfg.DATASET.PROCESSED_IMGS_DIR == "deterministic_challenge_preprocessing":
                cfg.DATASET.CHALLENGE_PREPROCESSING = True
            cfg.DATASET.PROCESSED_IMGS_DIR = ""

            self.cache_data(cfg)
        else:
            self.get_processed_filenames(cfg)

        self.STANDARDISE_SIZES = cfg.DATASET.STANDARDISE_SIZES
        if self.STANDARDISE_SIZES:
            self.resize = iaa.Sequential([
                iaa.PadToAspectRatio(cfg.DATASET.IMG_SIZE[1] / cfg.DATASET.IMG_SIZE[0], position="right-bottom"),
                iaa.Resize({"height": cfg.DATASET.IMG_SIZE[0], "width": cfg.DATASET.IMG_SIZE[1]})
            ])


    def __len__(self):
        return len(self.data)

    def handle_partition(self):
        """
        training | validation | testing
        :return:
        """
        with open(self.partition_file, "r") as partition_file:
            self.data = json.load(partition_file)[self.partition]
        self.data_set = set(self.data)

    def cache_data(self, cfg: Config):
        resize_dir = os.path.join(cfg.DATASET.CACHE_DIR, cfg.DATASET.NAME,
                                  f"{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}_CH_Preprocessing_{cfg.DATASET.CHALLENGE_PREPROCESSING}")

        # loop over all image files
        # for each image file, load the image and the landmarks
        # resize the image
        # save the image and the landmarks w/ scale factor

        if not os.path.exists(resize_dir):
            os.makedirs(resize_dir)

        files = []
        for i in self.img_dirs:
            files.extend(glob.glob(f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{i}/*"))
        for file in sorted(files):
            image_name = file.split("/")[-1].split(".")[0]
            if image_name not in self.data_set:
                continue

            cached_image_name = f"{resize_dir}/{image_name}.png"
            cached_meta_name = f"{resize_dir}/{image_name}_meta.json"
            annotations_names = []
            if self.annotation_dirs == [""]:
                annotations_names.append(f"{resize_dir}/{image_name}_annotations.txt")
            else:
                for annotation_dir in self.annotation_dirs:
                    cached_annotation_name = f"{resize_dir}/{image_name}_{annotation_dir.split('/')[-1]}.txt"
                    annotations_names.append(cached_annotation_name)
            self.images.append(cached_image_name)
            self.annotations.append(annotations_names)
            self.metas.append(cached_meta_name)

            # if the image has not been resized
            if not os.path.exists(cached_image_name):

                image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                annotation_original_image_shape = image.shape
                # preprocess from dataset_utils

                image, _, operations = self.preprocess(image)

                cv2.imwrite(cached_image_name, image)

                if type(self.dataset_pixels_per_mm) is list:
                    pixels_per_mm = self.dataset_pixels_per_mm[0]
                else:
                    pixels_per_mm = self.dataset_pixels_per_mm

                NO_LABEL = False

                # load the annotations
                for i, annotation_dir in enumerate(self.annotation_dirs):
                    if annotation_dir == "":
                        image_data = self.annotations_dataframe.loc[
                            self.annotations_dataframe["image file"] == f"{image_name}.bmp"]

                        if image_data.empty:
                            # raise FileNotFoundError(
                            #     f"Annotation file not found - image {self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{annotation_dir}/{image_name}.txt")
                            landmarks = np.zeros((self.total_landmarks, 2))
                            pixels_per_mm = 1
                            NO_LABEL = True
                        else:
                            landmarks = image_data.iloc[0, 2:].values.astype('float').reshape(-1, 2)
                            pixels_per_mm = self.annotations_dataframe.loc[
                                self.annotations_dataframe["image file"] == f"{image_name}.bmp"].iloc[0, 1]

                    elif os.path.exists(
                            f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{annotation_dir}/{image_name}.txt"):
                        landmarks = np.loadtxt(
                            f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{annotation_dir}/{image_name}.txt",
                            delimiter=",", max_rows=self.total_landmarks)
                        if self.FLIP_INITIAL_COORDINATES and image_name.startswith("A"):
                            landmarks = np.flip(landmarks, -1)
                    else:
                        raise FileNotFoundError(
                            f"Annotation file not found - image {self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{annotation_dir}/{image_name}.txt")
                    # resize
                    operations["image_name"] = image_name
                    preprocess_landmarks = self.preprocess.preprocess_just_landmarks(landmarks, operations)
                    if NO_LABEL:
                        preprocess_landmarks = np.zeros((self.total_landmarks, 2))
                    np.savetxt(
                        annotations_names[i], preprocess_landmarks, fmt="%.14g", delimiter=","
                    )

                original_image_height, original_image_width = operations['pre-resize shape']
                original_aspect_ratio = original_image_width / original_image_height
                downsampled_aspect_ratio = self.IMG_SIZE[1] / self.IMG_SIZE[0]
                if original_aspect_ratio > downsampled_aspect_ratio:
                    scale_factor = original_image_width / self.IMG_SIZE[1]
                else:
                    scale_factor = original_image_height / self.IMG_SIZE[0]

                with open(cached_meta_name, "w") as meta_file:
                    json.dump({"scale_factor": scale_factor, "filename": image_name,
                               "pixels_per_mm": pixels_per_mm, "shift": operations["shift"],
                               "pre-resize shape": operations['pre-resize shape'], "no_label_exists": NO_LABEL},
                              meta_file)

    def get_processed_filenames(self, cfg: Config):
        resize_dir = os.path.join(cfg.DATASET.CACHE_DIR, cfg.DATASET.NAME,
                                  cfg.DATASET.PROCESSED_IMGS_DIR)
        files = os.listdir(resize_dir)
        for file in sorted(files):

            image_name = file.split("/")[-1].split("_")[0]
            if len(file.split("/")[-1].split("_")) == 1:
                file_type = "Image"
            else:
                file_type = file.split("/")[-1].split("_")[1].split(".")[0]
            if image_name.split(".")[0] not in self.data_set:
                continue

            if file.endswith("_meta.json"):
                self.metas.append(f"{resize_dir}/{file}")
                continue
            if len(file.split("/")[-1].split("_")) <= 2:
                subimage = "1"
            elif len(file.split("/")[-1].split("_")) == 3:
                subimage = file.split("/")[-1].split("_")[2].split(".")[0]
                continue
            else:
                raise ValueError(f"File type {file_type} {file}, {image_name}, {file_type}, {subimage}")

            if int(subimage) > 1:
                continue
            if file_type == "annotations":
                filename = f"{resize_dir}/{image_name}"
                if not self.USE_ALL_RCNN_IMAGES:
                    filename += f"_annotations.{file.split('.')[-1]}"
                self.annotations.append([filename])
            elif file_type == "cropped" or file_type == "Image":
                filename = f"{resize_dir}/{image_name}"
                if not self.USE_ALL_RCNN_IMAGES and file_type == "cropped":
                    filename += f"_cropped.{file.split('.')[-1]}"
                    # filename += f".{file.split('.')[-1]}"
                self.images.append(filename)
            else:
                raise ValueError(f"File type {file_type} {file}, {image_name}, {file_type}, {subimage}")

    def img_int_to_float(self, img):
        # # 0-255 (none), image/255 (standard), max-min/max (0-1), adaptive
        int_to_float = self.cfg_INT_TO_FLOAT
        if self.cfg_INT_TO_FLOAT == "random" and self.augment:
            int_to_float = random.choices(["none", "standard", "minmax", "adaptive"], weights=[0.25, 0.25, 0.25, 0.25])[
                0]
        elif self.cfg_INT_TO_FLOAT == "random" and not self.augment:
            int_to_float = "standard"

        if int_to_float == "none":
            return img.astype(np.float32)
        elif int_to_float == "standard":
            return img.astype(np.float32) / 255
        elif int_to_float == "minmax":
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            return img
        elif int_to_float == "adaptive":
            return skimage.exposure.equalize_adapthist(img)
        else:
            raise ValueError(f"Invalid int to float method {self.cfg_INT_TO_FLOAT}")

    # @profile
    def __getitem__(self, idx):
        # if self.augment:
        #     imgaug.seed(np.random.randint(0, 10000))
        #     np.random.seed(np.random.randint(0, 10000))
        image_name = self.images[idx]
        annotations = self.annotations[idx]


        label = 0
        total_landmarks = self.total_landmarks
        meta_suffix = ""

        # load data
        if self.store_in_ram and label not in self.ram[idx]:
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            self.ram[idx][label] = img
        elif self.store_in_ram:
            img = self.ram[idx][label]
        else:
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.uint8)
        landmarks_all_annotators = np.loadtxt(annotations[0], delimiter=",", max_rows=total_landmarks)

        landmarks_all_annotators = np.array(landmarks_all_annotators).reshape(-1, 2)
        meta = json.load(open(self.metas[idx], "r"))

        # preprocess and augment
        kps = KeypointsOnImage.from_xy_array(landmarks_all_annotators, shape=img.shape)
        if self.STANDARDISE_SIZES:
            img, kps = self.resize(image=img, keypoints=kps)

        if self.augment:

            img_aug_colour = self.invert_transform(image=img)
            if random.random() < self.SIMULATE_XRAY_ARTEFACTS_RATE:
                img_aug_colour = simulate_x_ray_artefacts(img_aug_colour)

            img_aug, kps_aug = img_aug_colour, kps
            use_skewed_scaled = random.random() < self.USE_SKEWED_SCALE_RATE

            for i in range(5):
                img_aug, kps_aug = self.transform(image=img_aug_colour, keypoints=kps)
                if use_skewed_scaled:
                    img_aug, kps_aug = self.scale_transform_skew(image=img_aug, keypoints=kps_aug)
                else:
                    img_aug, kps_aug = self.scale_transform(image=img_aug, keypoints=kps_aug)
                if any([kp.x < 0 or kp.x >= img.shape[1] or kp.y < 0 or kp.y >= img.shape[0] for kp in
                        kps_aug.keypoints]):

                    if i == 4:
                        # print(image_name, "lower augs")
                        # plt.imshow(img_aug)
                        # plt.scatter(kps_aug.to_xy_array()[:, 0], kps_aug.to_xy_array()[:, 1], c='r', s=2)
                        # plt.title("outside augs")
                        # plt.show()
                        img_aug, kps_aug = self.low_transform(image=img_aug_colour, keypoints=kps)
                        break
                    continue
                else:
                    break

            img, kps = img_aug, kps_aug

        landmarks_all_annotators = kps.to_xy_array().reshape(-1, total_landmarks, 2)

        landmarks = np.mean(landmarks_all_annotators, axis=0)

        # if not self.USE_PROCESSED_IMAGES:
        landmarks = np.flip(landmarks, axis=-1).astype(np.float32)
        landmarks_all_annotators = np.flip(landmarks_all_annotators, axis=-1).astype(np.float32)

        img = self.img_int_to_float(img)
        img = self.normalise(img, method=self.cfg_NORMALISATION_METHOD)
        img = img.astype(np.float32)
        output = {"x": img, "y": landmarks, "name": f"{image_name.split('/')[-1].split('.')[0].split('_')[0]}"}
        output["label"] = np.array(label, dtype=np.int32)
        output["scale_factor"] = np.array(meta["scale_factor" + meta_suffix], dtype=np.float32)

        if len(output["x"].shape) == 2:
            output["x"] = np.expand_dims(output["x"], axis=-1)
        if type(meta["pixels_per_mm"]) is list:
            if len(meta["pixels_per_mm"]) == 2:
                output["pixel_size"] = (
                        np.array([meta["pixels_per_mm"][0], meta["pixels_per_mm"][1]], dtype=np.float32) *
                        output["scale_factor"])
            else:
                output["pixel_size"] = (
                        np.array([meta["pixels_per_mm"][0][0], meta["pixels_per_mm"][0][1]], dtype=np.float32) *
                        output["scale_factor"])
        else:
            output["pixel_size"] = (np.array([meta["pixels_per_mm"], meta["pixels_per_mm"]], dtype=np.float32) *
                                    output["scale_factor"])

        output["shift"] = np.array(meta["shift" + meta_suffix], dtype=np.int16)
        if "pre-resize shape" not in meta:
            meta["pre-resize shape"] = img.shape
        output["pre-resize shape"] = np.array(meta["pre-resize shape"], dtype=np.int16)

        output["landmarks_per_annotator"] = landmarks_all_annotators
        landmarks_rounded = np.round(landmarks).astype(int)
        # output["y_img_initial"] = create_landmark_image_no_batch(landmarks_rounded, img.shape)
        output["y_img_initial"] = generate_landmark_images_numpy.create_landmark_image(
            # np.flip(landmarks_rounded, axis=-1),
            landmarks_rounded,
            img.shape,
            use_gaussian=self.USE_GAUSSIAN,
            sigma=self.SIGMAS)
        output["y_img"] = output["y_img_initial"]

        return output

    @classmethod
    def get_loaders(cls, root_dir, batch_size, num_workers, augment_train=False, partition="training", shuffle=None):
        if shuffle is None:
            shuffle = partition == "training"
        dataset = cls(root_dir, partition=partition, augment=augment_train and partition == "training")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers,
                                persistent_workers=True)

        return dataloader


def main():
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.rcParams["image.cmap"] = "gray"
    plt.rcParams["figure.dpi"] = 300

    cfg = config.get_config("configs/local_test_ceph_challenge.yaml")


    cfg.DATASET.PROCESSED_IMGS_DIR = "800x704-54pbdf90"
    # cfg.DATASET.PROCESSED_IMGS_DIR = "deterministic_challenge_preprocessing"
    # cfg.DATASET.PROCESSED_IMGS_DIR = ""
    cfg.DATASET.CHALLENGE_PREPROCESSING = False
    cfg.DATASET.USE_ALL_RCNN_IMAGES = False
    cfg.DATASET.USE_GAUSSIAN_GT = True
    cfg.DATASET.GT_SIGMA = 1

    train_loader = LandmarkDataset.get_loaders(cfg, 1, 0, False, partition="training", shuffle=True)
    # train_loader = LandmarkDataset.get_loaders(cfg, 1, 1, False, partition="validation", shuffle=False)
    # train_loader = LandmarkDataset.get_loaders(cfg, 1, 1, False, partition="testing", shuffle=True)
    start = time.time()

    ia.seed(np.random.randint(0, 1000))
    for b, batch in enumerate(train_loader):

        x = rearrange(batch["x"], 'b h w c -> b c h w')

        if batch["name"][0] == "454":
            plt.imshow((renormalise(x[0, 0], True)).clamp(0, 255).long().cpu().numpy())
            plt.scatter(batch["y"][0, :, 1], batch["y"][0, :, 0], c='r', s=2)
            plt.title(f"Image {batch['name'][0]}")
            plt.show()

            plt.imshow(reduce(batch["y_img"][0], "c h w -> h w", "max").cpu().numpy())
            # plt.scatter(batch["y" + label][0, :, 1], batch["y" + label][0, :, 0], c='r', s=2)
            plt.title(f"Image {batch['name'][0]}")
            plt.show()

    print(time.time() - start)


if __name__ == "__main__":
    # get_mean_std()
    main()
    # mnist_loader(4, 1)
    # check_file()
    # check_partitions()
