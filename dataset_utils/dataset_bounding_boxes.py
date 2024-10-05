import json
import os
import time

from einops import rearrange
# from line_profiler import profile
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision.tv_tensors import BoundingBoxes

from utils import util
from core import config

from dataset_utils.dataset_preprocessing_utils import colour_augmentation, rcnn_save_image

from dataset_utils.visualisations import plot_img_bounding_box_landmarks
from dataset_utils.generate_landmark_images import *
from core.config import Config

import imgaug as ia

ia.seed(42)
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
import pandas as pd


class LandmarkDatasetBoundingBoxes(Dataset):

    def __init__(self, cfg: Config, partition="training", augment=False):

        # handle directories
        self.root_dir = cfg.DATASET.ROOT_DIR

        self.box_eps = cfg.DATASET.BOUNDING_BOX_EPS

        self.img_dirs = cfg.DATASET.IMG_DIRS
        self.img_ext = cfg.DATASET.IMG_EXT
        self.tensor_device = util.get_device()
        self.cfg = cfg

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
        self.VARY_GT_W_DIFFUSION = cfg.DIFFUSION.VARY_GT_W_DIFFUSION
        self.USE_MULTI_SCALE = cfg.DENOISE_MODEL.USE_MULTI_SCALE
        self.num_scales = len(cfg.DENOISE_MODEL.ENCODER_CHANNELS)
        self.USE_NEGATIVE_LEARNING = cfg.DIFFUSION.ADV_USE_NEGATIVE_LEARNING
        self.Diffusion_T = cfg.DIFFUSION.DIFFUSION_STEPS
        self.use_multi_annotation = cfg.DATASET.USE_MULTI_ANNOTATION
        self.NEGATIVE_LEARNING_MAX_RADIUS = cfg.DATASET.NEGATIVE_LEARNING_MAX_RADIUS
        self.augment = augment

        self.IMG_SIZE = cfg.DATASET.IMG_SIZE

        self.preprocess_pad = iaa.PadToMultiplesOf(32, 32, position="right-bottom")
        self.preprocess_resize = iaa.Resize({"width": 512, "height": "keep-aspect-ratio"})

        # augmentations
        self.transform = iaa.Sequential([
            iaa.Affine(rotate=(-cfg.AUGMENTATIONS.ROTATION, cfg.AUGMENTATIONS.ROTATION),
                       scale={"x": (1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                              "y": (1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE)},
                       translate_px={"x": cfg.AUGMENTATIONS.TRANSLATION_X, "y": cfg.AUGMENTATIONS.TRANSLATION_Y},
                       mode="constant", shear=(-cfg.AUGMENTATIONS.SHEAR, cfg.AUGMENTATIONS.SHEAR)),
            iaa.Multiply(mul=(1 - cfg.AUGMENTATIONS.MULTIPLY, 1 + cfg.AUGMENTATIONS.MULTIPLY)),

            # iaa.GammaContrast((0.5, 2)),
            # iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
            # iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
            iaa.ElasticTransformation(alpha=(0, cfg.AUGMENTATIONS.ELASTIC_TRANSFORM_ALPHA),
                                      sigma=cfg.AUGMENTATIONS.ELASTIC_TRANSFORM_SIGMA, order=3),
            iaa.Cutout(nb_iterations=(0, cfg.AUGMENTATIONS.CUTOUT_ITERATIONS),
                       size=(cfg.AUGMENTATIONS.CUTOUT_SIZE_MIN, cfg.AUGMENTATIONS.CUTOUT_SIZE_MAX),
                       squared=False),
            # iaa.Invert(0.25),
            # iaa.Fliplr(0.4)

        ], random_order=True)
        # self.Contrasts = [iaa.GammaContrast((0.5, 2)), iaa.LinearContrast((0.5, 1.4)),
        #                   iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.5, 0.7))]

        self.coarse_dropout = iaa.CoarseDropout(0.02, size_percent=0.08)
        self.addative_gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, cfg.AUGMENTATIONS.GAUSSIAN_NOISE * 255))
        self.coarse_dropout_rate = cfg.AUGMENTATIONS.COARSE_DROPOUT_RATE
        self.addative_gaussian_noise_rate = cfg.AUGMENTATIONS.ADDATIVE_GAUSSIAN_NOISE_RATE

        self.FLIP_INITIAL_COORDINATES = cfg.AUGMENTATIONS.FLIP_INITIAL_COORDINATES
        self.LANDMARK_POINT_EPSILON = cfg.DATASET.LANDMARK_POINT_EPSILON
        self.LOCALISED_LOSS = cfg.DIFFUSION.LOCALISED_LOSS
        self.USE_OFFSETS = cfg.DIFFUSION.USE_OFFSETS
        self.RADIUS = cfg.DIFFUSION.MASK_RADIUS

        self.store_in_ram = True
        self.ram = {}

        self.normalise = normalise

        # if cfg.DATASET.CACHE_DIR == "":
        #     cfg.DATASET.CACHE_DIR = "datasets/dataset_cache"
        # self.cache_data(cfg)

    def __len__(self):
        return len(self.data)

    def indices_to_target(self, landmarks, indices=None):
        if indices is None:

            x_min = np.min(landmarks[:, 0], axis=0)
            x_max = np.max(landmarks[:, 0], axis=0)
            y_min = np.min(landmarks[:, 1], axis=0)
            y_max = np.max(landmarks[:, 1], axis=0)
        else:
            selected_landmarks = landmarks[indices]
            x_min = np.min(selected_landmarks[:, 0], axis=0)
            x_max = np.max(selected_landmarks[:, 0], axis=0)
            y_min = np.min(selected_landmarks[:, 1], axis=0)
            y_max = np.max(selected_landmarks[:, 1], axis=0)
        return x_min, x_max, y_min, y_max

    def make_target(self, landmarks, img_shape):
        # index to label = {0: "full", 1: "ear", 2: "spine", 3: "chin", 4: "eye"}
        box_landmark_indices = [None]
        if self.total_landmarks == 53 and self.cfg.DATASET.USE_ALL_RCNN_IMAGES:
            for i in [[0, 3, 16, 18, 23, 24, 26, 27, 28],
                      [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                      [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 29, 30, 31, 32, 35, 36,
                       37],
                      [1, 2, 33, 34, 51, 52]]:
                box_landmark_indices.append(i)

        output = np.zeros((len(box_landmark_indices), 4))
        # Percentage of landmarks in bounding box for eye: 83.33%, name ['153'], step 7920
        # 46 Percentage of landmarks in bounding box for chin: 96.00%, name ['208'], step 7920 - add half eps and add 9 to included indices
        # 47 Percentage of landmarks in bounding box for full: 98.11%, name ['396'], step 7920 - move
        # 48 Percentage of landmarks in bounding box for chin: 96.00%, name ['407'], step 7920
        # 49 Percentage of landmarks in bounding box for chin: 96.00%, name ['408'], step 7920
        # 50 Percentage of landmarks in bounding box for chin: 96.00%, name ['424'], step 7920
        for i, indices in enumerate(box_landmark_indices):
            x_min, x_max, y_min, y_max = self.indices_to_target(landmarks, indices)
            if i == 1:  # ear
                # y_min -= 3
                y_max += 15
                # x_min -= 10
                x_min -= 10
                x_max += 15
            if i == 2:  # spine
                y_min -= 45
                y_max += 5
                x_min += 5
                x_max -= 5
            elif i == 3:  # chin
                y_max += 20
                x_min += 5
                x_max -= 10

                # x_min -= 10
            elif i == 4:  # eye
                x_min -= 35
                x_max -= 10

            elif i == 0:  # full
                y_min -= 5
                if 20 < self.box_eps < 40:
                    y_max += 20
                    x_max -= 10
                elif self.box_eps > 40:
                    y_max += 15
                    x_max -= 20

            # eye left 10
            # chin left 10
            # full should be enclosed by other 4 boxes (caliper out of bounds mainly)

            x_min = max(0, x_min - self.box_eps)
            y_min = max(0, y_min - self.box_eps)
            x_max = min(img_shape[1] - 2, x_max + self.box_eps)
            y_max = min(img_shape[0] - 2, y_max + self.box_eps)

            output[i] = [x_min, y_min, x_max, y_max]
        return output

    def handle_partition(self):
        """
        training | validation | testing
        :return:
        """
        with open(self.partition_file, "r") as partition_file:
            self.data = json.load(partition_file)[self.partition]
        self.data_set = set(self.data)

    def __getitem__(self, idx):
        image_name = self.data[idx]
        image_name_full = None
        for img_dir in self.img_dirs:
            if os.path.exists(
                    f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{img_dir}/{image_name}{self.img_ext}"):
                image_name_full = f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{img_dir}/{image_name}{self.img_ext}"
                break
        if image_name_full is None:
            raise FileNotFoundError(f"Image not found - {image_name_full}")
        # load data
        if self.store_in_ram and idx not in self.ram:
            img = cv2.imread(image_name_full, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.uint8)
            self.ram[idx] = img
        elif self.store_in_ram:
            img = self.ram[idx]
        else:
            img = cv2.imread(image_name_full, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.uint8)
        original_resolution = img.shape

        if self.annotations_dataframe is not None:
            annotation_data = self.annotations_dataframe.loc[
                self.annotations_dataframe["image file"] == f"{image_name}.bmp"]
            if annotation_data.empty:
                # raise FileNotFoundError(
                #     f"Annotation file not found - image {self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{annotation_dir}/{image_name}.txt")
                landmarks_all_annotators = np.zeros((self.total_landmarks, 2))
                pixels_per_mm = 1
            else:
                landmarks_all_annotators = annotation_data.iloc[0, 2:].values.astype('float').reshape(-1, 2)
                pixels_per_mm = self.annotations_dataframe.loc[
                    self.annotations_dataframe["image file"] == f"{image_name}.bmp"].iloc[0, 1]
            pixels_per_mm = [pixels_per_mm, pixels_per_mm]
        else:

            landmarks_all_annotators = [
                np.loadtxt(f"{annotation}/{image_name}.txt", delimiter=",", max_rows=self.total_landmarks)
                for annotation in self.annotation_dirs if os.path.exists(annotation)]

            landmarks_all_annotators = np.array(landmarks_all_annotators).reshape(-1, 2)
            pixels_per_mm = self.dataset_pixels_per_mm

        # preprocess and augment
        # kps = KeypointsOnImage.from_xy_array(
        #     landmarks, shape=img.shape
        # )
        kps = KeypointsOnImage.from_xy_array(landmarks_all_annotators, shape=img.shape)

        img_original = img.copy()
        landmarks_original = landmarks_all_annotators.copy()

        img, kps = self.preprocess_resize(images=img, keypoints=kps)
        img_pre_padded_resolution = img.shape
        img, kps = self.preprocess_pad(images=img, keypoints=kps)

        img = img[0]

        if self.augment:
            img_aug_colour = colour_augmentation(img)
            i = 0
            while True:
                img_aug, kps_aug = self.transform(image=img_aug_colour, keypoints=kps)
                if any([kp.x < 0 or kp.x > img.shape[1] or kp.y < 0 or kp.y > img.shape[0] for kp in
                        kps_aug.keypoints]):
                    i += 1
                    if i > 3:
                        # output which image and keypoint value i s causing the issue
                        # temp = kps_aug.to_xy_array()
                        # print("Recommend Smaller Augs")
                        # print(f"Image {image_name}")
                        # print(f" has keypoints {temp}")
                        break
                    continue
                else:
                    break

            random_1, random_2, random_3 = random.random(), random.random(), random.random()
            if random_1 < self.coarse_dropout_rate:
                img_aug = self.coarse_dropout(image=img_aug)
            if random_2 < self.addative_gaussian_noise_rate:
                img_aug = self.addative_gaussian_noise(image=img_aug)
            # contr = random.randint(0, len(self.Contrasts) - 1)
            # img_aug = self.Contrasts[random.randint(0, len(self.Contrasts) - 1)].augment_image(img_aug)

            img, kps = img_aug, kps_aug

        img = img.astype(np.float32) / 255

        landmarks_all_annotators = kps.to_xy_array().reshape(-1, self.total_landmarks, 2)
        landmarks = np.mean(landmarks_all_annotators, axis=0)
        # landmarks = np.flip(landmarks, axis=-1).astype(np.float32)

        output = {"x": img, "y": landmarks, "name": f"{image_name.split('/')[-1].split('.')[0]}"}
        output["landmarks_all_annotators"] = landmarks_all_annotators
        # output["x"] = output["x"].astype(np.float32) / 255

        output["boxes"] = BoundingBoxes(self.make_target(landmarks, img.shape),
                                        canvas_size=(img.shape[1], img.shape[0]),
                                        format="xyxy")
        output["labels"] = torch.arange(0, output["boxes"].shape[0], dtype=torch.int64) + 1

        output["image_original"] = torch.Tensor(img_original).to(torch.float32)
        output["image_original_shape"] = np.array(original_resolution)
        output["landmarks_original"] = landmarks_original
        output["img_pre_padded_resolution"] = np.array(img_pre_padded_resolution)
        output["pixels_per_mm"] = np.array(pixels_per_mm)

        if len(output["x"].shape) == 2:
            output["x"] = np.expand_dims(output["x"], axis=-1)

        return output

    @classmethod
    def get_loaders(cls, root_dir, batch_size, num_workers, augment_train=False, partition="training", shuffle=None):
        if shuffle is None:
            shuffle = partition == "training"
        dataset = cls(root_dir, partition=partition, augment=augment_train and partition == "training")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers)

        return dataloader


def main():
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.rcParams["image.cmap"] = "gray"
    plt.rcParams["figure.dpi"] = 75
    cfg = config.get_config("configs/local_test_ceph_challenge.yaml")
    # cfg = config.get_config("configs/local_test_ceph_adv.yaml")
    # cfg = config.get_config("configs/local_test_ceph.yaml")

    # cfg.DATASET.USE_MULTI_ANNOTATION = 1
    # cfg = config.get_config("./configs/local_test_ceph.yaml")
    cfg.DATASET.IMG_SIZE = [800, 704]
    cfg.AUGMENTATIONS.ELASTIC_TRANSFORM_ALPHA = 170
    cfg.AUGMENTATIONS.ELASTIC_TRANSFORM_SIGMA = 45
    cfg.DATASET.BOUNDING_BOX_EPS = 128
    train_loader = LandmarkDatasetBoundingBoxes.get_loaders(cfg, 1, 0, False, partition="training", shuffle=True)
    # train_loader = LandmarkDataset.get_loaders(cfg, 1, 1, False, partition="validation", shuffle=True)
    # train_loader = LandmarkDataset.get_loaders(cfg, 1, 1, False, partition="testing", shuffle=True)
    start = time.time()
    for i, batch in enumerate(train_loader):
        x = rearrange(batch["x"], 'b h w c -> b c h w').to(torch.float32)

        # boxes_numpy = batch["boxes"].numpy()
        print(batch["name"][0], x.shape, batch["name"], type(batch["name"]), batch["img_pre_padded_resolution"].dtype)
        # print(boxes_numpy[:, :, 2] - boxes_numpy[:, :, 0], boxes_numpy[:, :, 3] - boxes_numpy[:, :, 1],
        #       (boxes_numpy[:, :, 2] - boxes_numpy[:, :, 0]) / (boxes_numpy[:, :, 3] - boxes_numpy[:, :, 1]))
        cfg.DATASET.SAVE_ALL_RCNN_IMAGES = False
        img = plot_img_bounding_box_landmarks(x, batch["boxes"], batch["y"], convert_to_tensor=False,
                                              show_landmark_indices=True)
        plt.imshow(img)
        plt.show()

        # rcnn_save_image(batch, batch["boxes"][0], batch["labels"][0], "./datasets/dataset_cache/TEST", cfg)

        # def try_again(name, use_original_landmarks=False):
        #
        #     if use_original_landmarks:
        #
        #         landmarks_original = train_loader.dataset.annotations_dataframe.loc[
        #                                  train_loader.dataset.annotations_dataframe[
        #                                      "image file"] == f"{name}.bmp"].iloc[0,
        #                              2:].values.astype(
        #             'float').reshape(-1, 2)
        #         # landmarks_original = np.array(
        #         #     [835, 996, 1473, 1029, 1318, 1272, 623, 1238, 1342, 1639, 1373, 2029, 1333, 2200, 1313, 2244, 1266,
        #         #      2279, 694, 1805, 1457, 1865, 1450, 1864, 1587, 1757, 1569, 2005, 1514, 1620, 1431, 2236, 934, 1538,
        #         #      1387, 1552, 670, 1343, 1276, 2182, 1340, 1661, 1321, 2035, 1588, 1592, 938, 1412, 736, 1185, 1663,
        #         #      1528, 498, 1440, 931, 1222, 363, 1388, 1558, 1718, 1537, 2028, 1385, 2304, 1266, 2356, 1599, 892,
        #         #      1500, 1146, 960, 2309, 1175, 1777, 1208, 1848, 432, 1790, 499, 1798, 567, 1845, 416, 1822, 551,
        #         #      1896, 375, 1952, 447, 1968, 529, 2023, 352, 1988, 504, 2078, 325, 2123, 398, 2137, 485, 2187, 1711,
        #         #      953, 1712, 1044])
        #         landmarks_original = landmarks_original.reshape(53, 2)
        #         # plt.scatter(landmarks_original[:, 0], landmarks_original[:, 1], s=1)
        #         landmarks = landmarks_original
        #     else:
        #         meta = json.load(open(f"datasets/dataset_cache/TEST/{name}_meta.json", "r"))
        #         shift = meta["shift"]
        #         # pre_resize_shape = meta["pre-resize shape_1"]
        #         # img = cv2.imread(f"datasets/dataset_cache/TEST/{name}_cropped_1.png", cv2.IMREAD_GRAYSCALE)
        #         landmarks_np = np.loadtxt(f"datasets/dataset_cache/TEST/{name}_annotations.txt", delimiter=",",
        #                                   max_rows=54).astype(np.float64).reshape(-1, 2)
        #
        #         landmarks_np[:] = landmarks_np[:] * meta["scale_factor"]
        #
        #         landmarks_np[:, 0] += shift[0]
        #         landmarks_np[:, 1] += shift[1]
        #         # plt.scatter(landmarks_np[:, 0], landmarks_np[:, 1], s=1)
        #         landmarks = landmarks_np
        #     img_original = cv2.imread(
        #         f"datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images/{name}.bmp",
        #         cv2.IMREAD_GRAYSCALE)
        #     # plt.imshow(img_original, cmap="gray")
        #     # plt.title(f"Original: {use_original_landmarks}")
        #     # plt.show()
        #     return landmarks
        #
        # landmarks = try_again(batch["name"][0])
        # landmarks_gt = try_again(batch["name"][0], use_original_landmarks=True)
        #
        # # output error
        # print(np.mean(landmarks_gt - landmarks, axis=0))

        # if i >= 2:
        #     break

        # indices = [None, [0, 3, 16, 18, 23, 24, 26, 27, 28],
        #            [9, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        #            [4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 29, 30, 31, 32, 35, 36,
        #             37],
        #            [1, 2, 33, 34, 51, 52]]
        #
        # def calculate_percentage_of_landmarks_in_bounding_box(bounding_boxes, landmarks, labels):
        # def calculate_percentage_of_landmarks_in_bounding_box(bounding_boxes, landmarks, labels):
        #     percentages = []
        #     for idx, box in enumerate(bounding_boxes):
        #         count = 0
        #         if indices[labels[idx]] is None:
        #             selected_landmarks = landmarks
        #         else:
        #             selected_landmarks = landmarks[indices[labels[idx]]]
        #         for landmark in selected_landmarks:
        #             if box[0] <= landmark[0] <= box[2] and box[1] <= landmark[1] <= box[3]:
        #                 count += 1
        #         percentages.append(count / len(selected_landmarks))
        #     return percentages
        #
        # print(calculate_percentage_of_landmarks_in_bounding_box(batch["boxes"][0], batch["y"][0], batch["labels"][0]))
        #
        # output = plot_img_bounding_box_landmarks(x, batch["boxes"], batch["y"], convert_to_tensor=False,
        #                                          show_landmark_indices=True)
        # plt.title(f"Image {batch['name'][0]}")
        # plt.imshow(output)

        # boxed_landmarks = torch.round(batch["y"]).to(torch.int)
        # plt.scatter(boxed_landmarks[0, :, 0], boxed_landmarks[0, :, 1], c='r', s=2)
        # for landmark_i, (coordinate_x, coordinate_y) in enumerate(boxed_landmarks[0]):
        #     plt.text(coordinate_x, coordinate_y, str(landmark_i), color='red', fontsize=8)
        # plt.title(f"Image {batch['name'][0]}")
        # plt.show()

        if i > 5:
            break
    print(time.time() - start)


if __name__ == "__main__":
    # get_mean_std()
    main()
    # mnist_loader(4, 1)
    # check_file()
    # check_partitions()
