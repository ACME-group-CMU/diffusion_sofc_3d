import os
import argparse
import numpy as np
from tqdm import tqdm
import typing


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
from pytorch_lightning.callbacks import ModelCheckpoint

from data import Microstructures

# from unet import UNet
from unet_attention import UNet

# from sampler import DDIMSampler
from diffusers import DDPMScheduler, DDIMScheduler

import warnings


warnings.filterwarnings("ignore")


def main():

    global args

    if args.n_epochs < 5 * args.sample_interval:
        args.sample_interval = args.n_epochs // 6
    else:
        pass

    if (args.n_gpu * args.n_nodes > 1) and args.divide_batch:
        args.batch_size = args.batch_size // (args.n_gpu * args.n_nodes)
    else:
        pass

    print(args)

    seed_everything(args.random_state, workers=True)
    image_size = [args.img_size] * 3

    dm = MicroData(
        data_path=args.data_path,
        img_size=image_size,
        data_length=args.data_length,
        apply_sym=args.apply_sym,
        val_indices=args.val_indices,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.n_cpu,
    )

    model = Diffusion(
        args.channels,
        *image_size,
        args.base_dim,
        args.lr,
        args.sample_interval,
        args.dif_timesteps,
        args.inf_timesteps,
        args.sample_size,
        args.scheduler_gamma,
        args.n_blocks,
        args.mse_sum,
        args.ch_mul,
        args.is_attn,
        args.beta_start,
        args.beta_end,
        args.var_schedule,
        args.dropout,
    )

    if (args.n_gpu * args.n_nodes) > 1:
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
        every_n_epochs=args.n_epochs // 5,
        save_last=True,
        filename="loss-{epoch:03d}-{loss:.6f}",
    )

    trainer = Trainer(
        default_root_dir=args.dir,
        accelerator="gpu",
        devices=args.n_gpu,
        num_nodes=args.n_nodes,
        max_epochs=args.n_epochs,
        strategy=strategy,
        deterministic=True,
        callbacks=[
            TQDMProgressBar(refresh_rate=(args.data_length // (40 * args.batch_size))),
            checkpoint_callback,
            best_callback,
            best_val_callback,
        ],
        precision=16,
        resume_from_checkpoint=args.ckpt,
        gradient_clip_val=args.clip_val,
    )

    trainer.fit(model, dm)


class MicroData(LightningDataModule):
    def __init__(
        self,
        data_path: str = ".",
        img_size=(96, 96, 96),
        data_length=10000,
        apply_sym=True,
        val_indices=None,
        subset=None,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        super().__init__()

        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.length = data_length
        self.apply_symmetry = apply_sym
        self.val_indices = np.load(val_indices) if val_indices is not None else None
        self.subset = subset

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.data_train = Microstructures(
                self.data_dir,
                self.img_size,
                self.length,
                apply_symmetry=self.apply_symmetry,
                indices=None,
                subset=self.subset,
            )

            self.data_val = Microstructures(
                self.data_dir,
                self.img_size,
                self.length,
                apply_symmetry=self.apply_symmetry,
                indices=self.val_indices,
                subset=None,
            )

    def train_dataloader(self):

        return DataLoader(
            self.data_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):

        return DataLoader(
            self.data_val, batch_size=self.batch_size, num_workers=self.num_workers
        )


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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_freq = save_freq

        # network
        # self.unet = Unet(size=height,timesteps=den_timesteps,time_embedding_dim=time_dim,base_dim=base_dim,dim_mults=[2,4])
        self.unet = UNet(
            image_channels=channels,
            n_channels=base_dim,
            n_blocks=n_blocks,
            ch_mults=ch_mul,
            is_attn=is_attn,
            dropout=dropout,
        )
        """
        self.noise_scheduler = DDIMSampler(
            dif_timesteps,
            inf_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            var_schedule=var_sched,
        )
        """

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

        self.noise_scheduler.set_timesteps(inf_timesteps)

        self.sample_shape = (sample_amt, channels, height, width, depth)
        if mse_sum:
            self.mse = nn.MSELoss(reduction="sum")
        else:
            self.mse = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        imgs = batch

        sch = self.lr_schedulers()

        # sample noise
        noise = torch.randn_like(imgs)
        bs = imgs.shape[0]

        timesteps = torch.randint(
            0, self.hparams.dif_timesteps, (bs,), device=imgs.device
        ).long()

        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, timesteps)
        noise_pred = self.unet(noisy_imgs, timesteps)

        loss = self.mse(noise_pred.flatten(), noise.flatten())
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # variance = torch.mean(torch.std(noise-noise_pred),dim=[1,2,3])
        # self.log('variance',variance,on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        imgs = batch

        # sample noise
        noise = torch.randn_like(imgs)
        bs = imgs.shape[0]

        timesteps = torch.randint(
            0, self.hparams.dif_timesteps, (bs,), device=imgs.device
        ).long()

        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, timesteps)
        noise_pred = self.unet(noisy_imgs, timesteps)

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

        # variance = torch.mean(torch.std(noise-noise_pred),dim=[1,2,3])
        # self.log('val_variance',variance,on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def forward(self, noisy_img, timestep: int):
        """
        Noisy Image : (B,C,D,H,W)
        Timesteps : Integer
        """

        ## Make sure the devices are correct
        noisy_img = noisy_img.to(next(self.unet.parameters()).device)
        timesteps = noisy_img.new_full(
            (noisy_img.shape[0],), timestep, dtype=torch.long
        )

        return self.unet(noisy_img, timesteps)

    @torch.no_grad()
    def generate(self, inf_timesteps=None, sample_shape=(5, 1, 96, 96, 96)):

        if inf_timesteps is not None:
            self.noise_scheduler.set_timesteps(inf_timesteps)

        shape = sample_shape

        x = torch.randn(shape).to(next(self.unet.parameters()).device)

        for i in tqdm(self.noise_scheduler.timesteps):
            i = i.item()
            residual_noise = self(x, i)
            x = self.noise_scheduler.step(residual_noise, i, x).prev_sample

        x = x.cpu().numpy().squeeze()
        # x = ((x+1)/2)*255
        # x = np.round(x,0)

        return x

    def configure_optimizers(self):

        lr = self.hparams.lr
        optimizer = AdamW(self.unet.parameters(), lr=lr, weight_decay=0.000001)

        sched_gamma = self.hparams.scheduler_gamma
        sched = ExponentialLR(optimizer, sched_gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "frequency": 10, "interval": "epoch"},
        }

    @torch.no_grad()
    def training_epoch_end(self, training_step_outputs):

        if (self.current_epoch + 1) % self.save_freq == 0:
            sample_imgs = self.generate(sample_shape=self.sample_shape)
            np.save(f"{self.logger.log_dir}/{self.current_epoch}.npy", sample_imgs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data Saving Parameters
    parser.add_argument(
        "--dir", type=str, default="./results", help="directory that saves all the logs"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/denoised_grey_ints.npz",
        help="file name where the data belongs",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="lightning checkpoint from where training can be restarted",
    )
    # Model training
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--random_state", type=int, default=100, help="random state for everything"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.95,
        help="scheduler factor to reduce every 10 epochs",
    )

    parser.add_argument(
        "--clip_val",
        type=float,
        default=None,
        help="gradient clip value (norm)",
    )

    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpu per node to use during training",
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="number of nodes")
    # parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value")
    parser.add_argument(
        "--divide_batch",
        type=str,
        default="True",
        help="if batch_size needs to be divided for distributed training",
    )

    # Model and Data
    parser.add_argument(
        "--data_length", type=int, default=20000, help="number of random data points"
    )
    parser.add_argument(
        "--img_size", type=int, default=96, help="generated image size cubic dimension"
    )
    parser.add_argument(
        "--val_indices",
        type=str,
        default="dataset/val_indices.npy",
        help="indices to get validation set from",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=":,150:1500,:",
        help="if training data needs to be split from entire microstructure, define the region",
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="number of image channels"
    )
    parser.add_argument(
        "--dif_timesteps", type=int, default=1000, help="number of diffusion timesteps"
    )
    parser.add_argument(
        "--inf_timesteps", type=int, default=50, help="number of denoising timesteps"
    )
    parser.add_argument("--n_blocks", type=int, default=1, help="number of unet blocks")
    parser.add_argument(
        "--base_dim", type=int, default=16, help="base dimension in the UNet"
    )
    parser.add_argument(
        "--sample_interval", type=int, default=100, help="interval betwen image samples"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=64,
        help="number of samples that are generated",
    )
    parser.add_argument(
        "--apply_sym",
        type=str,
        default="True",
        help="if symmetry operations need to be applied during sampling from data (True/False)",
    )
    parser.add_argument(
        "--mse_sum",
        type=str,
        default="False",
        help="MSE loss is sum reduced or mean reduced",
    )
    parser.add_argument(
        "--ch_mul",
        type=str,
        default="(1,2,2,4)",
        help="channel_multipliers in the Unet",
    )
    parser.add_argument(
        "--is_attn",
        type=str,
        default="(0,0,1,1)",
        help="Whether the block has self-attention",
    )

    parser.add_argument(
        "--dropout",
        default=0.0,
        help="How much dropout in each resolution",
    )

    parser.add_argument(
        "--var_schedule",
        type=str,
        default="linear",
        help="Diffusion variance schedule :- linear or cosine",
    )
    parser.add_argument(
        "--beta_start", type=float, default=0.0001, help="variance schedule start"
    )
    parser.add_argument(
        "--beta_end", type=float, default=0.02, help="variance schedule end"
    )

    args = parser.parse_args()

    if type(args.dropout) == float:
        pass
    else:
        args.dropout = eval(args.dropout)

    args.apply_sym = eval(args.apply_sym)
    args.mse_sum = eval(args.mse_sum)
    args.ch_mul = eval(args.ch_mul)
    args.is_attn = eval(args.is_attn)
    args.divide_batch = eval(args.divide_batch)

    main()
