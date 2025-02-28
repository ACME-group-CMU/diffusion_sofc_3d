import os
import numpy as np
from tqdm.auto import tqdm
import typing
import json
from omegaconf import OmegaConf
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from data import Microstructures
from unet_attention import UNet
from diffusers import DDPMScheduler, DDIMScheduler
from skimage.filters import threshold_multiotsu

import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

def main(config):
    """
    Main function to set up and start the training process.
    """
    if config.training.n_epochs < 5 * config.model.sample_interval:
        config.model.sample_interval = config.training.n_epochs // 6
    else:
        pass

    if (config.training.n_gpu * config.training.n_nodes > 1) and config.training.divide_batch:
        config.training.batch_size = config.training.batch_size // (config.training.n_gpu * config.training.n_nodes)
    else:
        pass

    print(json.dumps(config,indent=2, default=str))

    seed_everything(config.training.random_state, workers=True)
    image_size = [config.data.img_size] * 3

    dm = MicroData(
        data_path=config.data.path,
        cond_path=config.data.cond_path,
        img_size=image_size,
        data_length=config.data.length,
        apply_sym=config.data.apply_sym,
        val_indices=config.data.val_indices,
        subset=config.data.subset,
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
        config.model.mse_sum,
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
    )

    if config.logging.uncond_path is not None and config.logging.ckpt is None:
        model.load_unconditional_weights(config.logging.uncond_path)

    if (config.training.n_gpu * config.training.n_nodes) > 1:
        strategy = "ddp"
    else:
        strategy = None

    best_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="loss",
        mode="min",
        filename="best_loss-{epoch:03d}-{loss:.6f}",
    )

    best_val_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        filename="best_val_loss-{epoch:03d}-{val_loss:.6f}",
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="epoch",
        mode="max",
        every_n_epochs=config.training.n_epochs // 5,
        save_last=True,
        filename="loss-{epoch:03d}-{loss:.6f}",
    )

    swa_callback = StochasticWeightAveraging(
        swa_lrs=0.0001 if config.model.train_base_model else 0.001,
        swa_epoch_start=0.8,
        annealing_epochs=int(0.1 * config.training.n_epochs),
    )

    trainer = Trainer(
        default_root_dir=config.logging.dir,
        accelerator="auto",
        devices=config.training.n_gpu,
        num_nodes=config.training.n_nodes,
        max_epochs=config.training.n_epochs,
        strategy=strategy,
        deterministic=True,
        callbacks=[
            TQDMProgressBar(
                refresh_rate=int(0.05 * (config.data.length // config.training.batch_size)) + 1
            ),
            checkpoint_callback,
            best_callback,
            best_val_callback,
            swa_callback,
        ],
        precision=config.training.precision,
        resume_from_checkpoint=config.logging.ckpt,
        gradient_clip_val=config.training.clip_val,
        track_grad_norm=2,
    )

    trainer.fit(model, dm)


class MicroData(LightningDataModule):
    def __init__(
        self,
        data_path: str = ".",
        cond_path: str = ".",
        img_size: typing.Tuple[int, int, int] = (96, 96, 96),
        data_length: int = 10000,
        apply_sym: bool = True,
        val_indices: typing.Optional[str] = None,
        subset: typing.Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        """
        Initialize the MicroData module.
        
        Args:
            data_path (str): Path to the data.
            cond_path (str): Path to the conditional data.
            img_size (tuple): Size of the images.
            data_length (int): Length of the dataset.
            apply_sym (bool): Whether to apply symmetry.
            val_indices (str, optional): Path to validation indices.
            subset (str, optional): Subset of the data.
            batch_size (int): Batch size.
            num_workers (int): Number of workers.
        """
        super().__init__()

        self.data_dir = data_path
        self.cond_dir = cond_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.length = data_length
        self.apply_symmetry = apply_sym
        self.val_indices = np.load(val_indices) if val_indices is not None else None
        self.subset = subset

    def setup(self, stage: typing.Optional[str] = None):
        """
        Set up the dataset for training and validation.
        
        Args:
            stage (str, optional): Stage of the setup (fit or None).
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.data_train = Microstructures(
                self.data_dir,
                self.cond_dir,
                self.img_size,
                self.length,
                apply_symmetry=self.apply_symmetry,
                indices=None,
                subset=self.subset,
            )

            self.data_val = Microstructures(
                self.data_dir,
                self.cond_dir,
                self.img_size,
                self.length,
                apply_symmetry=self.apply_symmetry,
                indices=self.val_indices,
                subset=None,
            )

            if (self.val_indices is None) and (self.subset is None):
                data_full = Microstructures(
                    self.data_dir,
                    self.cond_dir,
                    self.img_size,
                    self.length + (self.batch_size * 4),
                    apply_symmetry=self.apply_symmetry,
                    indices=None,
                    subset=None,
                )

                self.data_train, self.data_val = random_split(
                    data_full, [self.length, (self.batch_size * 4)]
                )

    def train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader.
        
        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            self.data_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(
            self.data_val, batch_size=self.batch_size * 2, num_workers=self.num_workers
        )

class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.params = [p.data for p in self.model.parameters() if p.requires_grad]
        self.shadow_params = [p.data for p in self.shadow.parameters() if p.requires_grad]
        self.backup = []

    def update(self):
        decay = self.decay
        for param, shadow_param in zip(self.params, self.shadow_params):
            shadow_param.copy_(shadow_param * decay + (1 - decay) * param)

    def apply_shadow(self):
        self.backup = [p.clone() for p in self.params]
        for param, shadow_param in zip(self.params, self.shadow_params):
            param.data.copy_(shadow_param)

    def restore(self):
        for param, backup in zip(self.params, self.backup):
            param.data.copy_(backup)
        self.backup = []


class Diffusion(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        depth,
        base_dim,
        lr: float = 0.0001,
        save_freq: int = 20,
        dif_timesteps: int = 1000,
        inf_timesteps: int = 50,
        sample_amt: int = 36,
        scheduler_gamma: float = 0.8,
        n_blocks: int = 1,
        mse_sum: bool = False,
        ch_mul: tuple = (1, 2, 2, 4),
        is_attn: tuple = (0, 0, 1, 1),
        beta_start=0.0001,
        beta_end=0.1,
        var_sched="linear",
        dropout=0,
        condition_dim=None,
        cross_attn=False,
        train_base_model=False,
        conditional_validation_frequency: int = 5,
        use_ema: bool = True,  # Add EMA flag
        ema_decay: float = 0.995,  # Add EMA decay rate
        validate_with_ema: bool = True, # Add validation with EMA flag
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_freq = save_freq
        self.unet = UNet(
            image_channels=channels,
            n_channels=base_dim,
            ch_mults=ch_mul,
            is_attn=is_attn,
            n_blocks=n_blocks,
            dropout=dropout,
            condition_dim=condition_dim,
            cross_attn=cross_attn,
        )

        self.ema = None # Initialize EMA to None
        if use_ema:
            self.ema = EMA(self.unet, decay=ema_decay)

        if var_sched == "cosine":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=dif_timesteps,
                beta_schedule="squaredcos_cap_v2",
                timestep_spacing="linspace",
            )
        elif var_sched == "linear":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=dif_timesteps,
                beta_schedule="linear",
                beta_start=beta_start,
                beta_end=beta_end,
                timestep_spacing="linspace",
            )


        self.mse = nn.MSELoss()
        self.lr = lr

        self.channels = channels
        self.width = width
        self.height = height
        self.depth = depth
        self.sample_amt = sample_amt
        self.condition_dim = condition_dim

        self.automatic_optimization = False
        self.train_base_model = train_base_model

    def configure_optimizers(self):
        opt = AdamW(filter(lambda p: p.requires_grad, self.unet.parameters()), lr=self.lr)
        scheduler = ExponentialLR(opt, gamma=self.hparams.scheduler_gamma)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        if self.condition_dim is not None:
            imgs, condition = batch
            bs = imgs.shape[0]
            # Randomly set condition to None
            if torch.rand(1) < 0.8:
                condition = None
        else:
            imgs, _ = batch
            condition = None
            bs = imgs.shape[0]

        # sample noise
        noise = torch.randn_like(imgs)

        # sample a timestep for each image in the batch
        timesteps = torch.randint(
            0, self.hparams.dif_timesteps, (bs,), device=imgs.device
        ).long()

        # add noise to the images
        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, timesteps)

        # predict noise
        noise_pred = self.unet(noisy_imgs, timesteps, condition)

        # calculate loss
        loss = self.mse(noise_pred.flatten(), noise.flatten())

        # Backpropagation
        self.manual_backward(loss)
        opt.step()

        if self.ema:
            self.ema.update()

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # EMA shadow model context
        if self.ema and self.hparams.validate_with_ema:
            self.ema.apply_shadow()
            model = self.ema.model
        else:
            model = self.unet.eval()

        # Conditional validation
        if (self.condition_dim is not None) and ((self.current_epoch+1)% int(self.trainer.max_epochs * 0.1))==0:
            imgs, conditions = batch
            batch_size = self.sample_amt
            sample_shape = (batch_size, *img.shape[1:])  

            # Generate samples using the diffusion model
            generated_images = self.generate(
                inf_timesteps=self.hparams.inf_timesteps,
                sample_shape=sample_shape,
                condition=conditions.cpu().numpy(),
                model=model # Pass the model to generate
            )
            generated_images = torch.tensor(generated_images).type_as(imgs)

            # Estimate volume fractions from the generated images
            estimated_percs = []
            for img in generated_images:
                percs = self.get_exp_vf_otsu(img)
                estimated_percs.append(percs.unsqueeze(0))  # Keep the batch dimension

            estimated_percs = torch.cat(estimated_percs, dim=0).type_as(conditions)

            # Calculate the MSE loss between estimated and true volume fractions
            loss = self.mse(estimated_percs.flatten(), conditions.flatten())

            self.log(
                "val_loss_generation",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            if self.ema and self.hparams.validate_with_ema:
                self.ema.restore()
            
            return loss

        else:
            # Standard validation step: predict noise
            imgs, conditions = batch # Unpack both images and conditions

            # sample noise
            noise = torch.randn_like(imgs)
            bs = imgs.shape[0]

            timesteps = torch.randint(
                0, self.hparams.dif_timesteps, (bs,), device=imgs.device
            ).long()

            noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, timesteps)

            # Predict noise, pass condition if available
            if self.condition_dim is not None:
                noise_pred = model(noisy_imgs, timesteps, conditions)
            else:
                noise_pred = model(noisy_imgs, timesteps, None)

            loss = self.mse(noise_pred.flatten(), noise.flatten())

            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            if self.ema and self.hparams.validate_with_ema:
                self.ema.restore()

            return loss

    @torch.no_grad()
    def generate(
        self, inf_timesteps=None, sample_shape=(10, 1, 96, 96, 96), condition=None, model=None, w=2
    ):
        if model is None:
            model = self.unet

        if inf_timesteps is not None:
            self.noise_scheduler.set_timesteps(inf_timesteps)

        if condition is not None:
            if np.array(condition).shape[0] != sample_shape[0]:
                condition = np.array([condition] * sample_shape[0])

            assert (
                condition.shape[0] == sample_shape[0]
            ), "number of conditions not matching the number of samples"
        
        x = torch.randn(sample_shape).to(next(model.parameters()).device)

        if condition is not None:
            condition = torch.tensor(condition).type_as(x)
            w = w
        else:
            w = -1

        for i in tqdm(self.noise_scheduler.timesteps):
            i = i.item()
            timestep = x.new_full((x.shape[0],), i, dtype=torch.long)

            with torch.cuda.amp.autocast():
                residual_noise_cond = model(x, timestep, condition)
                residual_noise = model(x, timestep, None)
            
            residual_noise = residual_noise = ((1 + w) * residual_noise_cond) - (w * residual_noise)

            x = self.noise_scheduler.step(residual_noise, i, x).prev_sample

        x = x.cpu().numpy().squeeze()
        return x

    def get_exp_vf_otsu(self, microstructures):

        np_micro = microstructures.data.cpu().numpy()
        np_micro = (((np_micro + 1) / 2) * 255).astype(int)
        try:
            thresholds = threshold_multiotsu(np_micro)
        except:
            thresholds = [50, 130]

        # print(thresholds)
        torch.use_deterministic_algorithms(False)
        hist = (
            torch.histc(microstructures, bins=256, min=-1, max=1)
            / microstructures.numel()
        )
        torch.use_deterministic_algorithms(True)
        percs = torch.empty((3,))

        areas = torch.split(
            hist,
            (
                thresholds[0],
                thresholds[1] - thresholds[0],
                hist.numel() - thresholds[1],
            ),
        )

        for i in range(3):
            percs[i] = torch.sum(areas[i])

        return percs

    def load_unconditional_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)["state_dict"]

        # Remove 'unet.' prefix from keys
        unet_state_dict = {}
        for k, v in state_dict.items():
            unet_state_dict[k.replace("unet.", "")] = v

        self.unet.load_state_dict(unet_state_dict, strict=False)

        # Freeze the base model parameters
        if not self.train_base_model:
            for param in self.unet.parameters():
                param.requires_grad = False

        # Unfreeze the conditional layers if they exist
        if hasattr(self.unet, "condition_emb"):
            if self.unet.condition_emb is not None:
                for param in self.unet.condition_emb.parameters():
                    param.requires_grad = True

        if hasattr(self.unet, "cross_attn"):
            if self.unet.cross_attn is not None:
                for param in self.unet.cross_attn.parameters():
                    param.requires_grad = True

        if hasattr(self.unet, "time_concat"):
            if self.unet.time_concat is not None:
                for param in self.unet.time_concat.parameters():
                    param.requires_grad = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with a specified configuration file.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)
