import init_paths
import argparse
import os

import wandb
from trainers.CHH_reimplementation import CHH
from trainers.detector import RandomLandmarkDetector
from trainers.multi_image_landmark_detection import MultiImageLandmarkDetector
from trainers.baseline import Baseline
from dataset_utils.dataset import LandmarkDataset
from trainers.test_ensemble import LandmarkEnsembleDetector

os.environ["WANDB__SERVICE_WAIT"] = "240"
import torch

torch.manual_seed(42)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from utils import util
from core import config

from trainers import diffusion


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Train a generative model to generate landmarks conditional on an image')

    parser.add_argument('--config_path', type=str, help='Path to the configuration file',
                        default="configs/default.yaml", required=False)
    parser.add_argument("--saving_root_dir", type=str, help='Path to where to save project files',
                        default="./", required=False)
    parser.add_argument("--desc", type=str, help="Description of the run", default="", required=False)
    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main(args):
    cfg = config.get_config(args.config_path, args.saving_root_dir)
    cfg.TRAIN.DESCRIPTION = args.desc
    import dotenv
    dotenv.load_dotenv()

    if not os.path.exists(f"{args.saving_root_dir}/wandb") and args.saving_root_dir != "./":
        os.makedirs(f"{args.saving_root_dir}/wandb")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(project=f"{cfg.TRAIN.PROJECT}", name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}",
                     dir=f"{args.saving_root_dir}/wandb",
                     notes=args.desc, reinit=True)

    logger = WandbLogger(project=f"{cfg.TRAIN.PROJECT}",
                         name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}",
                         save_dir=f"{args.saving_root_dir}/wandb",
                         experiment=run,
                         id=run.id)
    if any(i in wandb.config.__dict__["_items"] for i in
           ["DATASET", "TRAIN", "DIFFUSION", "DENOISE_MODEL", "AUGMENTATIONS"]):
        sweep_config = dict(wandb.config.__dict__["_items"])
        # {'AUGMENTATIONS': {'ELASTIC_TRANSFORM_ALPHA': 28, 'ELASTIC_TRANSFORM_SIGMA': 13, 'GAUSSIAN_NOISE': 0.03753207276761211, 'MULTIPLY': 0.5885496080689955, 'ROTATION': 10, 'SCALE': 0.35906491281628156, 'SHEAR': 10, 'TRANSLATION_X': {'LOWERBOUND': -0.17911522501872462, 'UPPERBOUND': 0.07071682391980352}, 'TRANSLATION_Y': {'LOWERBOUND': -0.18743152448644856, 'UPPERBOUND': 0.2588030545780289}}}
        cfg.TRAIN.LOG_IMAGE = False
        cfg.TRAIN.LOG_VIDEO = False
        for config_parent in ["DATASET", "TRAIN", "DIFFUSION", "DENOISE_MODEL", "AUGMENTATIONS"]:
            if config_parent in wandb.config:

                if config_parent == "DIFFUSION" and "DIFFUSION" in wandb.config:
                    if "NLL_WEIGHT" in wandb.config["DIFFUSION"] and "L_SIMPLE_WEIGHT" in wandb.config["DIFFUSION"]:
                        if wandb.config["DIFFUSION"]["NLL_WEIGHT"] + wandb.config["DIFFUSION"]["L_SIMPLE_WEIGHT"] == 0:
                            return

                if config_parent == "AUGMENTATIONS" and "AUGMENTATIONS" in wandb.config:
                    if "TRANSLATION_X" in wandb.config["AUGMENTATIONS"]:
                        sweep_config["AUGMENTATIONS"]["TRANSLATION_X"] = sorted([
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_X"]["LOWERBOUND"],
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_X"]["UPPERBOUND"]])
                    if "TRANSLATION_Y" in wandb.config["AUGMENTATIONS"]:
                        sweep_config["AUGMENTATIONS"]["TRANSLATION_Y"] = sorted([
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_Y"]["LOWERBOUND"],
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_Y"]["UPPERBOUND"]])
                getattr(cfg, config_parent).update(sweep_config[config_parent])
                
    logger.log_hyperparams(args)
    logger.log_hyperparams(cfg.to_dict())

    logger.experiment.save("./dataset_utils/*", policy="now")

    if cfg.TRAIN.MODEL_TYPE.lower() == "ddpm":
        logger.experiment.save("./trainers/diffusion.py", policy="now")
        logger.experiment.save("./models/denoising_unet.py", policy="now")
    elif cfg.TRAIN.MODEL_TYPE.lower() == "baseline":
        logger.experiment.save("./trainers/baseline.py", policy="now")
        logger.experiment.save("./models/denoising_unet.py", policy="now")
    elif cfg.TRAIN.MODEL_TYPE.lower() == "main":
        logger.experiment.save("./models/multi_scale_unet_new.py", policy="now")
        if cfg.DATASET.USE_ALL_RCNN_IMAGES:
            logger.experiment.save("./trainers/multi_image_landmark_detection.py", policy="now")
        else:
            logger.experiment.save("./trainers/detector.py", policy="now")
            logger.experiment.save("./models/discriminator_model.py", policy="now")
    elif cfg.TRAIN.MODEL_TYPE.lower() == "chh":
        logger.experiment.save("./CHH_reimplementation.py", policy="now")
    else:
        raise ValueError(f"Model type {cfg.TRAIN.MODEL_TYPE} not recognised")
    model = LandmarkEnsembleDetector(cfg, args.saving_root_dir)

    logger.experiment.save("./dataset_utils/*", policy="now")
    logger.experiment.save("./utils/*", policy="now")
    logger.experiment.save("./core/train.py", policy="now")

    train_dataloader = LandmarkDataset.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=cfg.DATASET.AUGMENT_TRAIN, partition="training")
    validation_dataloader = LandmarkDataset.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=False, partition="validation")

    checkpoint_dir = f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{logger.experiment.id}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        save_last=True,
        monitor="val/sdr_l2_scaled_2.0",
        mode="max", )

    if util.get_device() == "mps":
        precision = "32-true"
    else:
        precision = "16-mixed"

    trainer = L.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator=util.get_device(),
                        logger=logger, check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY_N_EPOCHS, precision=precision,
                        callbacks=[checkpoint_callback],
                        default_root_dir=checkpoint_dir,
                        enable_progress_bar=False,
                        gradient_clip_algorithm="norm",
                        )

    print(f"EXPERIMENT ID {logger.experiment.id}")
    checkpoint_file = f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{cfg.TRAIN.CHECKPOINT_FILE}"
    if cfg.TRAIN.RUN_TRAIN:
        trainer.fit(model, train_dataloader, validation_dataloader)

        trainer.save_checkpoint(f"{checkpoint_dir}/train_end.ckpt")

        logger.log_hyperparams({"last_checkpoint": f"{checkpoint_dir}/train_end.ckpt"})
        logger.log_hyperparams({"best_checkpoint": checkpoint_callback.best_model_path})
        checkpoint_file = checkpoint_callback.best_model_path

    if cfg.TRAIN.RUN_TEST:

        if cfg.TRAIN.CHECKPOINT_FILE == "" and not cfg.TRAIN.RUN_TRAIN:
            model = LandmarkEnsembleDetector(cfg, args.saving_root_dir)
        else:
            model = LandmarkEnsembleDetector.load_from_checkpoint(checkpoint_file, cfg=cfg,
                                                                  saving_root_dir=args.saving_root_dir)

        for partition in ["validation", "testing"]:
            test_dataloader = LandmarkDataset.get_loaders(
                cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
                augment_train=False, partition=partition)
            trainer.test(model, test_dataloader)

        trainer.save_checkpoint(
            f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{logger.experiment.id}/ensemble_test.ckpt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
