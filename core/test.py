import argparse
import os

from trainers.CHH_reimplementation import CHH
from trainers.detector import RandomLandmarkDetector
from dataset_utils.dataset import LandmarkDataset

os.environ["WANDB__SERVICE_WAIT"] = "240"

# torch.manual_seed(42)
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
    parser.add_argument('--project', type=str, help='Wandb project name',
                        default="DiffLand", required=False)
    parser.add_argument("--desc", type=str, help="Description of the run", default="", required=False)
    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = config.get_config(args.config_path, saving_root_dir=args.saving_root_dir)
    cfg.TRAIN.DESCRIPTION = args.desc
    import dotenv
    dotenv.load_dotenv()

    validation_dataloader = LandmarkDataset.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=False, partition="validation")

    test_dataloader = LandmarkDataset.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=False, partition="testing")

    import wandb

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    logger = WandbLogger(project=f"{args.project}",
                         name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}-TEST",
                         offline=False,
                         dir=f"{args.saving_root_dir}/wandb")

    logger.log_hyperparams(args)
    logger.log_hyperparams(cfg.to_dict())

    if cfg.TRAIN.MODEL_TYPE.lower() == "main":
        logger.experiment.save("./trainers/detector.py")
        model = RandomLandmarkDetector.load_from_checkpoint(
            f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{cfg.TRAIN.CHECKPOINT_FILE}",
            cfg=cfg)
    elif cfg.TRAIN.MODEL_TYPE.lower() == "chh":
        logger.experiment.save("./trainers/CHH_reimplementation.py")
        model = CHH.load_from_checkpoin(
            f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{cfg.TRAIN.CHECKPOINT_FILE}",
            cfg=cfg)
    else:
        raise ValueError(f"Model type {cfg.TRAIN.MODEL_TYPE} not recognised")
    model.eval()
    logger.experiment.save("./dataset_utils/*", policy="now")
    logger.experiment.save("./utils/*")

    if not os.path.exists(
            f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{logger.experiment.id}"):
        os.makedirs(
            f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{logger.experiment.id}")
    trainer = L.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator=util.get_device(),  # precision="16-mixed",
                        logger=logger, check_val_every_n_epoch=1,
                        enable_progress_bar=False, )

    trainer.validate(model, validation_dataloader)
    trainer.test(model, test_dataloader)
    # pass args

    # load checkpoint
    # load data
    # run model
    # save results


if __name__ == '__main__':
    main()
