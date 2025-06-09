import os
import numpy as np
from tqdm.auto import tqdm
import typing
import json
import time
from omegaconf import OmegaConf
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import (
    LightningDataModule,
    Trainer,
    seed_everything,
)

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.utilities import rank_zero_only

from diffusion import Diffusion
from data import BuildDataset
import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


def main(config):
    """
    Main function to set up and start the training process.
    """
    if config.training.n_epochs < 5 * config.model.sample_interval:
        config.model.sample_interval = config.training.n_epochs // 6
    else:
        pass

    if (
        config.training.n_gpu * config.training.n_nodes > 1
    ) and config.training.divide_batch:
        config.training.batch_size = config.training.batch_size // (
            config.training.n_gpu * config.training.n_nodes
        )
    else:
        pass

    seed_everything(config.training.random_state, workers=True)
    image_size = [config.data.img_size] * 3

    dm = MicroData(
        data_path=config.data.path,
        characteristics=config.data.characteristics,
        apply_sym=config.data.apply_sym,
        batch_size=config.training.batch_size,
        num_workers=config.training.n_cpu,
    )

    model = Diffusion(
        config.model.channels,
        *image_size,
        config.model.base_dim,
        config.training.lr,
        config.model.sample_interval,
        config.model.diff_timesteps,
        config.model.inf_timesteps,
        config.model.sample_size,
        config.training.scheduler_gamma,
        config.model.n_blocks,
        config.model.ch_mul,
        config.model.is_attn,
        config.model.beta_start,
        config.model.beta_end,
        config.model.var_schedule,
        config.model.dropout,
        config.model.cond_dim,
        config.model.cross_attn,
        config.model.train_base_model,
        config.training.conditional_validation_frequency,  # Add this line
        config.training.use_ema,  # Add this line
        config.training.ema_decay,  # Add this line
        config.training.validate_with_ema,  # Add this line
        config.model.condition_fn,
    )

    if config.logging.uncond_path is not None and config.logging.ckpt is None:
        model.load_unconditional_weights(config.logging.uncond_path)

    # Compile the model
    model = torch.compile(model)

    if (config.training.n_gpu * config.training.n_nodes) > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss",
        mode="min",
        filename="best_loss-{epoch:03d}-{loss:.6f}",
    )

    best_val_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best_val_loss-{epoch:03d}-{val_loss:.6f}",
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="epoch",
        mode="max",
        every_n_epochs=config.training.n_epochs // 2,
        save_last=True,
        filename="loss-{epoch:03d}-{loss:.6f}",
    )

    # swa_callback = StochasticWeightAveraging(
    #     swa_lrs=0.0001 if config.model.train_base_model else 0.001,
    #     swa_epoch_start=0.6,
    #     annealing_epochs=int(0.1 * config.training.n_epochs),
    # )

    refresh_rate = 16
    tqdm_callback = TQDMProgressBar(refresh_rate=refresh_rate)

    logger = TensorBoardLogger(
        save_dir=config.logging.dir,
        name="lightning_logs",
    )
    
    trainer = Trainer(
        logger=logger,
        default_root_dir=config.logging.dir,
        accelerator="auto",
        devices=config.training.n_gpu,
        num_nodes=config.training.n_nodes,
        max_epochs=config.training.n_epochs,
        strategy=strategy,
        deterministic=True,
        callbacks=[
            tqdm_callback,
            checkpoint_callback,
            best_callback,
            best_val_callback,
            # swa_callback,
        ],
        precision=config.training.precision,
        gradient_clip_val=config.training.clip_val,
    )
    
    save_config(config,logger.log_dir)
    
    trainer.fit(model, dm, ckpt_path=config.logging.ckpt)

@rank_zero_only
def save_config(config, log_dir):
    """
    Save the configuration file to the logging directory.
    This function will only run on rank 0 (main process) in distributed training.
    
    Args:
        config: OmegaConf configuration object
        log_dir: Directory where the config should be saved
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Save the config as YAML
    config_save_path = os.path.join(log_dir, "training_config.yaml")
    OmegaConf.save(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")

@rank_zero_only
def get_next_version(save_dir):
    """
    Get the next version number by checking existing version directories.
    
    Args:
        save_dir (str): Base directory where lightning_logs will be created
        
    Returns:
        int: Next version number
    """
    lightning_logs_dir = os.path.join(save_dir, "lightning_logs")
    
    if not os.path.exists(lightning_logs_dir):
        os.makedirs(lightning_logs_dir, exist_ok=True)
        return 0
    
    # Find existing version directories
    version_dirs = glob.glob(os.path.join(lightning_logs_dir, "version_*"))
    
    if not version_dirs:
        return 0
    
    # Extract version numbers and find the maximum
    version_numbers = []
    for version_dir in version_dirs:
        dir_name = os.path.basename(version_dir)
        if dir_name.startswith("version_"):
            try:
                version_num = int(dir_name.split("_")[1])
                version_numbers.append(version_num)
            except (ValueError, IndexError):
                continue
    
    return max(version_numbers) + 1 if version_numbers else 0


class MicroData(LightningDataModule):
    def __init__(
        self,
        data_path: str = ".",
        characteristics: list = [],
        batch_size: int = 32,
        num_workers: int = 1,
        apply_sym: bool = True,
    ):
        """
        Initialize the MicroData module.

        Args:
            data_path (str): Path to the data folder.
            cond_path (str): Path to the conditional data dataframe.
            characteristics (list): List of characteristics for the dataset.
            batch_size (int): Batch size for training and validation.
            num_workers (int): Number of workers for data loading.
            apply_sym (bool): Whether to apply symmetry to the data.
        """
        super().__init__()

        self.data_dir = data_path
        self.characteristics = characteristics
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.apply_symmetry = apply_sym

    def setup(self, stage: typing.Optional[str] = None):
        if stage == "fit" or stage is None:
            characteristics = self.characteristics  # or [] if you want no conditions

            self.data_train = BuildDataset(
                data_path=self.data_dir,
                conditional_csv=os.path.join(self.data_dir, "train.csv"),
                characteristics=characteristics,
                apply_symmetry=self.apply_symmetry,
            )
            self.data_val = BuildDataset(
                data_path=self.data_dir,
                conditional_csv=os.path.join(self.data_dir, "validation.csv"),
                characteristics=characteristics,
                apply_symmetry=self.apply_symmetry,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified configuration file."
    )
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(config)
