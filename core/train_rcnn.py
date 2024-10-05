import init_paths
import argparse
import os

import wandb
from dataset_utils.dataset_bounding_boxes import LandmarkDatasetBoundingBoxes
from trainers.mask_rcnn_localisation import RCNN

os.environ["WANDB__SERVICE_WAIT"] = "240"
import torch

torch.manual_seed(42)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from utils import util
from core import config


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
    run = wandb.init(project=f"{cfg.TRAIN.PROJECT}", name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}-RCNN",
                     dir=f"{args.saving_root_dir}/wandb",
                     notes=args.desc, reinit=True)

    logger = WandbLogger(project=f"{cfg.TRAIN.PROJECT}",
                         name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}",
                         save_dir=f"{args.saving_root_dir}/wandb",
                         experiment=run,
                         id=run.id)

    logger.log_hyperparams(args)
    logger.log_hyperparams(cfg.to_dict())

    model = RCNN(cfg)
    logger.experiment.save("./dataset_utils/*", policy="now")
    logger.experiment.save("./mask_rcnn_localisation.py", policy="now")
    logger.experiment.save("./train.py", policy="now")
    logger.experiment.save("./util.py", policy="now")
    logger.experiment.save("./denoising_unet.py", policy="now")
    logger.experiment.save("./multi_scale_unet.py", policy="now")
    logger.experiment.save("./multi_scale_unet_new.py", policy="now")

    train_dataloader = LandmarkDatasetBoundingBoxes.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=cfg.DATASET.AUGMENT_TRAIN, partition="training")
    validation_dataloader = LandmarkDatasetBoundingBoxes.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=False, partition="validation")

    checkpoint_dir = f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{logger.experiment.id}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        save_last=True,
        mode="min", )

    early_stopping = EarlyStopping(monitor="train/loss_box_reg_epoch",
                                   min_delta=0.00,
                                   patience=3,
                                   verbose=True, mode="min")

    trainer = L.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator=util.get_device(),
                        logger=logger, check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY_N_EPOCHS,  # precision="16-mixed",
                        callbacks=[checkpoint_callback, early_stopping],
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
        model = RCNN.load_from_checkpoint(
            checkpoint_file,
            cfg=cfg)
        for partition in ["training", "validation", "testing"]:
            test_dataloader = LandmarkDatasetBoundingBoxes.get_loaders(
                cfg, batch_size=1, num_workers=cfg.TRAIN.NUM_WORKERS,
                augment_train=False, partition=partition, shuffle=False)
            trainer.test(model, test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
