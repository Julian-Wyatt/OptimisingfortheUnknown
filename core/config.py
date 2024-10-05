import yaml


class SubConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def update(self, entries: dict):
        for key, value in entries.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Key {key} not found in config")

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class Dataset(SubConfig):
    def __init__(self, **entries):
        self.ROOT_DIR = ""
        self.NAME = ""
        self.IMG_DIRS = [""]
        self.IMG_EXT = ""
        self.LABEL_DIR = [""]
        self.PARTITION_FILE = ""
        self.CACHE_DIR = ""

        # Image-related attributes
        self.NUMBER_KEY_POINTS = 0
        self.COORDINATE_DIMENSION = 2
        self.CHANNELS = 1
        self.PIXELS_PER_MM = [1, 1]
        self.CENTER_CROP = [-1, -1]
        self.CENTER_CROP_POSITION = "center"
        self.IMG_SIZE = [128, 128]
        self.LANDMARK_POINT_EPSILON = 1
        self.USE_MULTI_ANNOTATION = -1
        self.GT_SIGMA = 0
        # 0 is randomly select one annotator
        # 1 is random linear combination between landmark annotations
        # 2 is random gaussian jitter around the mean of the annotations

        # Training-related attributes
        self.AUGMENT_TRAIN = True

        self.CHALLENGE_PREPROCESSING = False

        self.BOUNDING_BOX_EPS = 15

        self.SAVE_RCNN_IMAGES = True
        self.USE_ALL_RCNN_IMAGES = False

        self.PROCESSED_IMGS_DIR = ""
        self.STANDARDISE_SIZES = False

        # int to float methods:
        # 0-255 (none), image/255 (standard), max-min/max (0-1), adaptive
        # normalisation methods:
        # 0-1 (none), -1-1 (mu=0.5, sigma=0.5), dataset mean and std

        self.INT_TO_FLOAT = "none"  # ["none", "standard", "minmax", "adaptive"]
        self.NORMALISATION = "none"  # ["none", "mu=0.5,sig=0.5", "dataset"]

        super().__init__(**entries)


class Augmentations(SubConfig):
    def __init__(self, **entries):
        self.ROTATION = 5
        self.SCALE = 0.1
        self.TRANSLATION_X = [-10, 10]
        self.TRANSLATION_Y = [-10, 10]
        self.SHEAR = 0
        self.MULTIPLY = 0.5
        self.GAUSSIAN_NOISE = 0.02
        self.ELASTIC_TRANSFORM_ALPHA = 400
        self.ELASTIC_TRANSFORM_SIGMA = 30
        self.COARSE_DROPOUT_RATE = 0
        self.ADDATIVE_GAUSSIAN_NOISE_RATE = 0
        self.FLIP_INITIAL_COORDINATES = False
        self.CHANNEL_DROPOUT = 0
        self.CUTOUT_ITERATIONS = 1
        self.CUTOUT_SIZE_MIN = 0
        self.CUTOUT_SIZE_MAX = 0.3
        self.BLUR_RATE = 0.25
        self.CONTRAST_GAMMA_MIN = 0.4
        self.CONTRAST_GAMMA_MAX = 2.4
        self.INVERT_RATE = 0
        self.USE_SKEWED_SCALE_RATE = 0.1
        self.SIMULATE_XRAY_ARTEFACTS_RATE = 0.75

        super().__init__(**entries)


class DenoiseModel(SubConfig):
    # class conditional timestep unet with attention
    def __init__(self, **entries):
        self.NAME = "Image Unet"
        # should remain 2 for 2D coordinates
        self.USE_MULTI_SCALE = False  # multi scale training
        self.USE_NEW_MODEL = False
        self.CONVNEXT_CH_MULT = 4
        self.DROPOUT = 0.1
        self.NUM_RES_BLOCKS = 2
        self.CONTEXT_DIM = None
        self.ENCODER_CHANNELS = (32, 32, 64, 128, 256)
        self.DECODER_CHANNELS = (256, 128, 64, 32, 32)
        self.ATTN_RESOLUTIONS = (32, 16, 8)
        self.BLOCKS_PER_LEVEL = (2, 2, 2)
        self.SEGFORMER_DECODER_CH_MULT = 2
        self.NUM_HEADS = 2
        self.FINAL_ACT = "tanh"
        self.DOWN_SAMPLE_CONTEXT = False
        self.USE_PRETRAINED_IMAGENET_WEIGHTS = True
        self.GRAYSCALE_TO_RGB = "repeat"
        self.DROP_PATH_RATE = 0.1
        super().__init__(**entries)




# class Diffusion(SubConfig):
#     def __init__(self, **entries):
#         self.BETA_SCHEDULE = "sqrt_linear"  # ["linear", "cosine", "sqrt_linear", "sqrt"]
#         self.BETA_START = 1e-4
#         self.BETA_END = 2e-2
#         self.DIFFUSION_STEPS = 100
#         self.LEARN_LOGVAR = False
#         self.PARAMETERISATION = "eps"  # ["eps", "x0"]
#         self.LOG_EVERY_t = 10
#         self.USE_EMBEDDING = True
#         self.CLASSIFIER_SCALE = 1
#         self.CLASSIFIER_CHECKPOINT_FILE = ""
#         self.COORDINATE_LOSS_WEIGHT = 0
#         self.L_SIMPLE_WEIGHT = 1
#         self.ORIGINAL_ELBO_WEIGHT = 0
#         self.NLL_WEIGHT = 0
#         self.OFFSET_LOSS_WEIGHT = 0
#         self.BCE_WEIGHT = 0
#
#         self.MAX_BLUR_STD = 10
#         self.BLUR_KERNEL_SIZE = 5
#
#         self.VARY_GT_W_DIFFUSION = False
#         self.RANDOM_VALIDATION = False
#
#         self.MASK_RADIUS = 0
#         self.USE_OFFSETS = False
#         self.LOCALISED_LOSS = False
#
#         self.GEN_BIG_T_HEATMAP = False
#
#         self.ADV_SCALE_LOSSES_BY_RESOLUTION = False
#         self.ADV_USE_UPSCALED_HEATMAP = False
#         self.ADV_USE_NEGATIVE_LEARNING = False
#         self.ADV_TRAIN_MULTI_OBJECTIVE = False
#
#         super().__init__(**entries)


class Train(SubConfig):
    def __init__(self, **entries):
        self.BATCH_SIZE = 1
        self.LR = 0.01
        self.EPOCHS = 10
        self.NUM_WORKERS = 4
        self.WEIGHT_DECAY = 0.0
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.OPTIMISER = "adamw"
        self.LOG_VIDEO = False
        self.LOG_IMAGE = False
        self.LOG_INTERVAL = 25
        self.MODEL_TYPE = "DDPM"
        self.SAVING_ROOT_DIR = ""
        self.LOG_WHOLE_VAL = False
        self.DEBUG = False
        self.DESCRIPTION = ""
        self.USE_SCHEDULER = False
        self.RUN_TEST = True
        self.RUN_TRAIN = True
        self.LOG_TEST_METRICS = True
        self.CHECKPOINT_FILE = ""
        self.TOP_K_HOTTEST_POINTS = 1

        self.VAL_EVERY_N_EPOCHS = 20

        self.MIN_LR = 1e-6
        self.WARMUP_EPOCHS = 15

        self.ENSEMBLE_STRATEGY = "coordinate"
        self.ENSEMBLE_MODEL_PATHS = []

        self.PROJECT = "DiffLand"  # wandb project name

        self.EXP_LR_DECAY = 0.94
        super().__init__(**entries)


class Config:
    def __init__(self, **entries):
        self.DATASET = Dataset(**entries.get("DATASET", {}))
        self.DENOISE_MODEL = DenoiseModel(**entries.get("DENOISE_MODEL", {}))
        self.AUGMENTATIONS = Augmentations(**entries.get("AUGMENTATIONS", {}))
        self.TRAIN = Train(**entries.get("TRAIN", {}))
        self.PATH = ""

    def __str__(self):
        sections = ["DATASET", "TRAIN", "DENOISE_MODEL", "AUGMENTATIONS"]
        output = ""
        for section in sections:
            output += f"{section}\n"
            section_obj = getattr(self, section)
            output += "\n".join([f"\t{key}: {value}" for key, value in section_obj.__dict__.items()])
            output += "\n\n"
        return output

    def to_dict(self):
        return {section: getattr(self, section).__dict__ for section in
                ["DATASET", "TRAIN", "DENOISE_MODEL", "AUGMENTATIONS"]}

    def __copy__(self):
        return Config(**self.to_dict())

    def update(self, entries: dict):
        for key, value in entries.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Key {key} not found in config")

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


def get_config(cfg_path, saving_root_dir="./") -> Config:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    _cfg = Config(**cfg)
    _cfg.PATH = cfg_path
    _cfg.TRAIN.SAVING_ROOT_DIR = saving_root_dir
    if _cfg.DATASET.ROOT_DIR.startswith("/data/") and _cfg.DATASET.CACHE_DIR == "":
        _cfg.DATASET.CACHE_DIR = f"{'/'.join(_cfg.TRAIN.SAVING_ROOT_DIR.split('/')[:-1])}/dataset_cache/"
    elif _cfg.DATASET.CACHE_DIR == "":
        _cfg.DATASET.CACHE_DIR = "../datasets/dataset_cache"
    return _cfg


if __name__ == "__main__":
    # cfg = get_config("./configs/default.yaml")
    # print(get_config("./configs/autoencoder.yaml"))
    print(get_config("../configs/local_test_ceph.yaml"))
