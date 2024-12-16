import os
import argparse
import numpy as np
from tqdm.auto import tqdm
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
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from data import Microstructures

# from unet import UNet
from unet_attention import UNet

# from sampler import DDIMSampler
from diffusers import DDPMScheduler, DDIMScheduler
from skimage.filters import threshold_multiotsu

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
        cond_path=args.cond_path,
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
        args.cond_dim,
        args.cross_attn,
        args.train_base_model,
    )

    if args.uncond_path is not None and args.ckpt is None:
        model.load_unconditional_weights(args.uncond_path)

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

    swa_callback = StochasticWeightAveraging(
        swa_lrs=0.0001 if args.train_base_model else 0.001,
        swa_epoch_start=0.8,
        annealing_epochs=int(0.1 * args.n_epochs),
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
            TQDMProgressBar(
                refresh_rate=int(0.05 * (args.data_length // args.batch_size)) + 1
            ),
            checkpoint_callback,
            best_callback,
            best_val_callback,
            swa_callback,
        ],
        precision=16,
        resume_from_checkpoint=args.ckpt,
        gradient_clip_val=args.clip_val,
        track_grad_norm=2,
    )

    trainer.fit(model, dm)


class MicroData(LightningDataModule):
    def __init__(
        self,
        data_path: str = ".",
        cond_path: str = ".",
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
        self.cond_dir = cond_path
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

    def train_dataloader(self):

        return DataLoader(
            self.data_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):

        return DataLoader(
            self.data_val, batch_size=self.batch_size * 2, num_workers=self.num_workers
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
        condition_dim=None,
        cross_attn=False,
        train_base_model=False,
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
            condition_dim=condition_dim,
            cross_attn=cross_attn,
        )

        self.cross_attn = cross_attn
        self.condition_dim = condition_dim
        self.train_base_model = train_base_model

        # Make sure the base model is not saved
        if not self.train_base_model:
            for param in self.unet.parameters():
                param.requires_grad = False

        if hasattr(self.unet, "condition_emb"):
            if self.unet.condition_emb is not None:
                for param in self.unet.condition_emb.parameters():
                    param.requires_grad = True

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

    def forward(self, noisy_img, timestep: int, condition=None):
        """
        Noisy Image : (B,C,D,H,W)
        Timesteps : Integer
        """

        ## Make sure the devices are correct
        noisy_img = noisy_img.to(next(self.unet.parameters()).device)
        timesteps = noisy_img.new_full(
            (noisy_img.shape[0],), timestep, dtype=torch.long
        )

        return self.unet(noisy_img, timesteps, condition)

    def training_step(self, batch, batch_idx):

        if self.condition_dim is not None:
            imgs, condition = batch
        else:
            imgs, condition = batch, None

        if self.train_base_model:
            if np.random.choice([True, False], p=[0.2, 0.8]):
                condition = None

        # sample noise
        noise = torch.randn_like(imgs)
        bs = imgs.shape[0]

        timesteps = torch.randint(
            0, self.hparams.dif_timesteps, (bs,), device=imgs.device
        ).long()

        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, timesteps)
        noise_pred = self.unet(noisy_imgs, timesteps, condition)

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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        w = 0.2

        if self.condition_dim is not None:
            imgs, condition = batch

            true_percs = []

            for i in imgs:
                percs = self.get_exp_vf_otsu(i)
                percs = percs.view(-1, 3)
                true_percs.append(percs)

            true_percs = torch.cat(true_percs, dim=0).type_as(condition)
            del condition

            with torch.no_grad():
                # sample noise
                noise = torch.randn_like(imgs)
                del imgs

                for i in self.noise_scheduler.timesteps:
                    i = i.item()
                    residual_noise_cond = self(noise, i, true_percs)
                    residual_noise = self(noise, i, None)

                    residual_noise = ((1 + w) * residual_noise_cond) - (
                        w * residual_noise
                    )

                    noise = self.noise_scheduler.step(
                        residual_noise, i, noise
                    ).prev_sample

            estimated_percs = []
            for i in noise:
                percs = self.get_exp_vf_otsu(i)
                percs = percs.view(-1, 3)
                estimated_percs.append(percs)

            estimated_percs = torch.cat(estimated_percs, dim=0).type_as(true_percs)

            loss = self.mse(estimated_percs.flatten(), true_percs.flatten())
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        else:
            imgs, condition = batch, None

            # sample noise
            noise = torch.randn_like(imgs)
            bs = imgs.shape[0]

            timesteps = torch.randint(
                0, self.hparams.dif_timesteps, (bs,), device=imgs.device
            ).long()

            noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, timesteps)
            noise_pred = self.unet(noisy_imgs, timesteps, condition)

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

        return loss

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

    @torch.no_grad()
    def generate(
        self, inf_timesteps=None, sample_shape=(10, 1, 96, 96, 96), condition=None, w=0
    ):

        if inf_timesteps is not None:
            self.noise_scheduler.set_timesteps(inf_timesteps)

        if condition is not None:
            if np.array(condition).shape[0] != sample_shape[0]:
                # print(condition.shape)
                condition = np.array([condition] * sample_shape[0])
                # print(condition.shape)

            assert (
                condition.shape[0] == sample_shape[0]
            ), "number of conditions not matching the number of samples"

        x = torch.randn(sample_shape).to(next(self.unet.parameters()).device)
        if condition is not None:
            condition = torch.tensor(condition).type_as(x)
            w = w
        else:
            w = -1

        for i in tqdm(self.noise_scheduler.timesteps):
            i = i.item()
            with torch.cuda.amp.autocast():
                residual_noise_cond = self(x, i, condition)
                residual_noise = self(x, i, None)

            residual_noise = ((1 + w) * residual_noise_cond) - (w * residual_noise)

            x = self.noise_scheduler.step(residual_noise, i, x).prev_sample

        x = x.cpu().numpy().squeeze()

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
        "--cond_path",
        type=str,
        default="~/group/final_img.npy",
        help="file name where the conditional segmented data belongs",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="lightning checkpoint from where training can be restarted",
    )

    parser.add_argument(
        "--uncond_path",
        type=str,
        default=None,
        help="lightning checkpoint for the base unconditional model",
    )

    parser.add_argument(
        "--train_base_model",
        type=str,
        default="False",
        help="do you want to train the base unconditional model or not (True/False)",
    )

    parser.add_argument(
        "--cross_attn",
        type=str,
        default="False",
        help="do you want to have cross attention between x and condition",
    )

    parser.add_argument(
        "--cond_dim",
        type=int,
        default=3,
        help="number of conditioning vector dimension",
    )

    # Model training
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="number of epochs of training"
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
        default=None,
        help="indices to get validation set from",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
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

    args.cross_attn = eval(args.cross_attn)
    args.train_base_model = eval(args.train_base_model)
    args.apply_sym = eval(args.apply_sym)
    args.mse_sum = eval(args.mse_sum)
    args.ch_mul = eval(args.ch_mul)
    args.is_attn = eval(args.is_attn)
    args.divide_batch = eval(args.divide_batch)

    main()
