import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

from data import Microstructures
#from unet import UNet
from unet_attention import UNet
from sampler import DDIMSampler

import warnings
warnings.filterwarnings('ignore')


def main():
    
    global args
    
    if args.n_epochs < 5*args.sample_interval:
        args.sample_interval = args.n_epochs//6
    else:
        pass
    
    if (args.n_gpu*args.n_nodes>1) and args.divide_batch:
        args.batch_size = args.batch_size//(args.n_gpu*args.n_nodes)
    else:
        pass
    
    print(args)
    
    seed_everything(42, workers=True)
    image_size = [args.img_size]*3
    
    dm = MicroData(
                   data_path=args.data_path,
                   img_size=image_size,
                   data_length=args.data_length,
                   apply_sym = args.apply_sym,
                   batch_size=args.batch_size,
                   num_workers=args.n_cpu
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
                )

    if (args.n_gpu*args.n_nodes)>1:
        strategy = 'ddp'
    else:
        strategy = None

    checkpoint_callback = ModelCheckpoint(
                                            save_top_k=1,
                                            monitor="loss",
                                            mode="min",
                                            save_last=True,
                                            filename="best_loss-{epoch:03d}-{loss:.6f}",
                                        )

    trainer = Trainer(
        default_root_dir=args.dir,
        accelerator="gpu",
        devices=args.n_gpu,
        num_nodes = args.n_nodes,
        max_epochs=args.n_epochs,
        strategy = strategy,
        deterministic = True,
        callbacks=[TQDMProgressBar(refresh_rate=(args.data_length//(40*args.batch_size))),checkpoint_callback],
        precision=16
    )

    trainer.fit(model, dm)


class MicroData(LightningDataModule):
    def __init__(
        self,
        data_path: str = '.',
        img_size = (64,64,64),
        data_length = 10000,
        apply_sym = True,
        batch_size: int = 256,
        num_workers: int = 1,
    ):
        super().__init__()
        
        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.length = data_length
        self.apply_symmetry = apply_sym
        
    def setup(self,stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            
            data_full = Microstructures(self.data_dir,
                                        self.img_size,
                                        self.length,
                                        apply_symmetry=self.apply_symmetry)
            
            
            self.data_train, self.data_val = random_split(data_full, 
                                                          [int(self.length*(9/10)),int(self.length*(1/10))])

    def train_dataloader(self):
        
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        
        return DataLoader(self.data_val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)
    
    
class Diffusion(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        depth,
        base_dim,
        lr: float = 0.0001,
        save_freq : int = 20,
        dif_timesteps : int = 1000,
        inf_timesteps : int = 50,
        sample_amt : int = 36,
        scheduler_gamma : float = 0.8,
        n_blocks : int=1,
        mse_sum : bool = False,
        ch_mul : tuple = (1,2,2,4),
        is_attn : tuple = (0,0,1,1),
        beta_start = 0.0001,
        beta_end = 0.1,
        var_sched = 'linear',
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_freq = save_freq
        
        # network
        #self.unet = Unet(size=height,timesteps=den_timesteps,time_embedding_dim=time_dim,base_dim=base_dim,dim_mults=[2,4])
        self.unet = UNet(image_channels = channels,
                         n_channels=base_dim,n_blocks=n_blocks,ch_mults=ch_mul, is_attn = is_attn)
        
        self.noise_scheduler = DDIMSampler(dif_timesteps,inf_timesteps,beta_start=beta_start,beta_end=beta_end,var_schedule=var_sched)
        self.sample_shape = (sample_amt,channels,height,width,depth)
        if mse_sum:
            self.mse = nn.MSELoss(reduction='sum')
        else:
            self.mse = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        imgs = batch

        sch = self.lr_schedulers()
        
        # sample noise
        noise = torch.randn_like(imgs)
        bs = imgs.shape[0]
        
        timesteps = torch.randint(0,self.hparams.dif_timesteps,(bs,),device=imgs.device).long()
        
        noisy_imgs = self.noise_scheduler.add_noise(imgs,timesteps,noise)
        noise_pred = self.unet(noisy_imgs,timesteps)
        
        loss = self.mse(noise_pred,noise)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #variance = torch.mean(torch.std(noise-noise_pred),dim=[1,2,3])
        #self.log('variance',variance,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    '''
    def validation_step(self, batch, batch_idx):
        
        imgs = batch
        
        # sample noise
        noise = torch.randn_like(imgs)
        bs = imgs.shape[0]
        
        timesteps = torch.randint(0,self.hparams.dif_timesteps,(bs,),device=imgs.device).long()
        
        noisy_imgs = self.noise_scheduler.add_noise(imgs,timesteps,noise)
        noise_pred = self.unet(noisy_imgs,timesteps)
        
        loss = self.mse(noise_pred,noise)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #variance = torch.mean(torch.std(noise-noise_pred),dim=[1,2,3])
        #self.log('val_variance',variance,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    '''
    
    def configure_optimizers(self):
        
        lr = self.hparams.lr
        optimizer = AdamW(self.unet.parameters(), lr=lr, weight_decay=0.0001)
        
        sched_gamma = self.hparams.scheduler_gamma
        sched = ExponentialLR(optimizer,sched_gamma)
        
        return {"optimizer": optimizer,
                "lr_scheduler": { "scheduler": sched, "frequency": 10, "interval" : "epoch"}}

    @torch.no_grad()
    def training_epoch_end(self,training_step_outputs):
        
        if (self.current_epoch+1)%self.save_freq==0:
            sample_imgs = self.noise_scheduler.sample(self.unet,self.sample_shape).cpu().detach()
            sample_imgs = torch.clamp(sample_imgs,-1,1)
            sample_imgs = sample_imgs.numpy()
            np.save(f'{self.logger.log_dir}/{self.current_epoch}.npy',sample_imgs)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    #Data Saving Parameters
    parser.add_argument("--dir", type=str, default='./results', help="directory that saves all the logs")
    parser.add_argument("--data_path", type=str, default='greyscale.npz', help="file name where the data belongs")
    # Model training 
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--scheduler_gamma", type=float, default=0.8, help="scheduler factor to reduce every 10 epochs")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu per node to use during training")
    parser.add_argument("--n_nodes", type=int, default=1, help="number of nodes")
    #parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value")
    parser.add_argument("--divide_batch", type=str, default="True", help="if batch_size needs to be divided for distributed training")
    
    #Model and Data
    parser.add_argument("--data_length", type=int, default=20000, help="number of random data points")
    parser.add_argument("--img_size", type=int, default=64, help="generated image size cubic dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--dif_timesteps", type=int, default=1000, help="number of diffusion timesteps")
    parser.add_argument("--inf_timesteps", type=int, default=50, help="number of denoising timesteps")
    parser.add_argument("--n_blocks", type=int, default=1, help="number of unet blocks")
    parser.add_argument("--base_dim", type=int, default=16, help="base dimension in the UNet")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
    parser.add_argument("--sample_size", type=int, default=64, help="number of samples that are generated")
    parser.add_argument("--apply_sym", type=str, default="True", help="if symmetry operations need to be applied during sampling from data (True/False)")
    parser.add_argument("--mse_sum", type=str, default="False", help="MSE loss is sum reduced or mean reduced")
    parser.add_argument("--ch_mul", type=str, default="(1,2,2,4)", help="channel_multipliers in the Unet")
    parser.add_argument("--is_attn", type=str, default="(0,0,1,1)", help="Whether the block has self-attention")
    parser.add_argument("--var_schedule", type=str, default="linear", help="Diffusion variance schedule :- linear or cosine")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="variance schedule start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="variance schedule end")
    
    args = parser.parse_args()
    
    args.apply_sym = eval(args.apply_sym)
    args.mse_sum = eval(args.mse_sum)
    args.ch_mul = eval(args.ch_mul)
    args.is_attn = eval(args.is_attn)
    args.divide_batch=eval(args.divide_batch)
    
    main()