import numpy as np
from tqdm.auto import tqdm
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from unet_attention import UNet
from diffusers import DDPMScheduler
from utils import *


import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
from ema import EMA


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
        use_ema: bool = True,  # import EMA flag
        ema_decay: float = 0.995,  # Add EMA decay rate
        validate_with_ema: bool = True,  # Add validation with EMA flag
        condition_fn: typing.Optional[str] = None,
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

        self.ema = None  # Initialize EMA to None
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

        self.train_base_model = train_base_model

        try:
            self.condition_fn = eval(condition_fn)
        except:
            self.condition_fn = None

    def configure_optimizers(self):
        optimizer = AdamW(self.unet.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.hparams.scheduler_gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 5,
                "interval": "epoch",
            },
        }

    def get_input(self, batch):

        if self.condition_dim is not None:
            imgs, condition = batch
            # Randomly set condition to None
            if self.train_base_model:
                if torch.rand(1) > 0.8:
                    condition = None
        else:
            imgs = batch
            condition = None

        return imgs, condition

    def shared_step(self, batch, model):

        imgs, condition = self.get_input(batch)
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
        noise_pred = model(noisy_imgs, timesteps, condition)

        # calculate loss
        loss = self.mse(noise_pred.flatten(), noise.flatten())

        return loss

    def training_step(self, batch, batch_idx):

        loss = self.shared_step(batch, self.unet)

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

        loss = self.shared_step(batch, model)

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

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema:
            self.ema.update()

    @torch.no_grad()
    def on_train_epoch_end(self):
        # Check if we should perform conditional validation this epoch
        if (self.condition_dim is not None) and (
            (self.current_epoch + 1) % self.hparams.conditional_validation_frequency
            == 0
        ):

            # EMA shadow model context
            if self.ema and self.hparams.validate_with_ema:
                self.ema.apply_shadow()
                model = self.ema.model
            else:
                model = self.unet.eval()

            # Get a random batch from the validation dataloader
            # Each GPU gets its own batch
            val_dataloader = self.trainer.val_dataloaders

            try:
                imgs, conditions = next(iter(val_dataloader))
            except (StopIteration, IndexError):
                if self.trainer.is_global_zero:
                    print(
                        "No validation data available, skipping conditional validation."
                    )
                if self.ema and self.hparams.validate_with_ema:
                    self.ema.restore()
                return

            sample_shape = imgs.shape


            # Generate samples using the diffusion model
            # Each GPU generates its own batch
            generated_images = self.generate(
                inf_timesteps=self.hparams.inf_timesteps,
                sample_shape=sample_shape,
                condition=conditions.cpu().numpy(),
                model=model,
            )
            generated_images = torch.tensor(generated_images).to(self.device)

            # Estimate volume fractions from the generated images
            # Each GPU processes its own batch
            gen_vols = generated_images.clone().cpu().detach().numpy()
            
            if gen_vols.ndim !=4:
                if gen_vols.ndim > 4:
                    gen_vols = gen_vols.squeeze()
                elif gen_vols.ndim ==3:
                    gen_vols = gen_vols[None,...]
                else:
                    raise ValueError(
                        f"Generated volumes should be 4D (N,H, W, D), got less {gen_vols.ndim} dimensions.")
            
            gen_conditions_local, correct_segment = self.condition_fn(gen_vols)

            # Convert to tensors for distributed operations
            try:
                gen_conditions_tensor = torch.tensor(gen_conditions_local).to(self.device)[correct_segment]
                conditions_tensor = conditions.to(self.device)[correct_segment]
            except:
                gen_conditions_tensor = torch.zeros_like(conditions).to('cpu')
                conditions_tensor = conditons.to('cpu')

            # Calculate the MSE loss between estimated and true volume fractions
            # This calculation happens on all GPUs but with the same gathered data
            loss = self.mse(gen_conditions_tensor.flatten(), conditions_tensor.flatten())

            # Log the results (sync_dist=True will handle averaging across GPUs)
            self.log(
                "generation_loss",
                loss,
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

            # Restore original model if using EMA
            if self.ema and self.hparams.validate_with_ema:
                self.ema.restore()

    @torch.no_grad()
    def generate(
        self,
        inf_timesteps=None,
        sample_shape=(10, 1, 96, 96, 96),
        condition=None,
        model=None,
        w=3,
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

        for i in tqdm(self.noise_scheduler.timesteps, miniters=100):
            i = i.item()
            timestep = x.new_full((x.shape[0],), i, dtype=torch.long)

            with torch.cuda.amp.autocast():
                residual_noise_cond = model(x, timestep, condition)
                residual_noise = model(x, timestep, None)

            residual_noise = ((1 + w) * residual_noise_cond) - (w * residual_noise)
            x = self.noise_scheduler.step(residual_noise, i, x).prev_sample

        x = x.cpu().numpy().squeeze()
        return x

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
        trainable_params_found = False

        if hasattr(self.unet, "condition_emb") and self.unet.condition_emb is not None:
            for param in self.unet.condition_emb.parameters():
                param.requires_grad = True
                trainable_params_found = True

        if hasattr(self.unet, "cross_attn") and self.unet.cross_attn is not None:
            for param in self.unet.cross_attn.parameters():
                param.requires_grad = True
                trainable_params_found = True

        if hasattr(self.unet, "time_concat") and self.unet.time_concat is not None:
            for param in self.unet.time_concat.parameters():
                param.requires_grad = True
                trainable_params_found = True

        # Ensure we have at least some trainable parameters
        if not trainable_params_found:
            print(
                "Warning: No conditional layers found. Making all parameters trainable."
            )
            for param in self.unet.parameters():
                param.requires_grad = True

        # Reinitialize EMA if it was previously set
        self.ema = EMA(self.unet, decay=self.hparams.ema_decay)
        self.configure_optimizers()
