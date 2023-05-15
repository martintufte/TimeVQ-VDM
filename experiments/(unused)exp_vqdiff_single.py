# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 12:32:38 2023

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import copy
from pathlib import Path
import tempfile
from einops import rearrange
from typing import Callable
from tqdm import tqdm

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from experiments.exp_base import ExpBase, detach_the_unnecessary
from vector_quantization import VectorQuantize
from diffusion_model import Unet, VDM

from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from utils import exists, compute_downsample_rate, freeze, timefreq_to_time, time_to_timefreq, \
    quantize, get_root_dir, StandardScaler


class ExpVQDiffSingle(ExpBase):
    def __init__(self,
                 config: dict,
                 n_train_samples: int,
                 stage: int = 2):
        """
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        :param stage: training stage
        """
        super().__init__()
        self.config = config
        self.dataset_name = config['dataset']['dataset_name']
        dataset_summary = pd.read_csv('datasets/DataSummary_UCR.csv')
        self.ts_length = int(dataset_summary.loc[dataset_summary['Name']==self.dataset_name, 'Length'])
        self.n_classes = int(dataset_summary.loc[dataset_summary['Name']==self.dataset_name, 'Class'])
        self.T_max_stage1 = config['stage1']['max_epochs'] * (np.ceil(n_train_samples / config['stage1']['batch_size']) + 1)
        self.T_max_stage2 = config['stage2']['max_epochs'] * (np.ceil(n_train_samples / config['stage2']['batch_size']) + 1)
        self.stage = stage

        self.n_fft = config['VQ']['n_fft']
        dim = config['EncDec']['dim']
        in_channels = config['dataset']['in_channels']
        
        downsampled_width= config['EncDec']['downsampled_width']
        downsample_rate = compute_downsample_rate(self.ts_length, self.n_fft, downsampled_width)
        diff_length = int(self.ts_length//2**int(np.log2(downsample_rate)+1))
        
        # Encoder, decoder and vector quantizer
        self.encoder = VQVAEEncoder(dim, 2 * in_channels, downsample_rate, config['EncDec']['n_resnet_blocks'])
        self.decoder = VQVAEDecoder(dim, 2 * in_channels, downsample_rate, config['EncDec']['n_resnet_blocks'])
        self.vq_model = VectorQuantize(dim, config['VQ']['codebook_sizes'], **config['VQ'])

        
        # Scale inputs for diffusion models learned from the codebooks
        embed = nn.Parameter(copy.deepcopy(self.vq_model._codebook.embed))  # pretrained discrete tokens (LF)     
        self.scaler = StandardScaler().fit(embed)
        
        # Low frequency diffusion model
        self.unet = Unet(
            ts_length    = diff_length,
            n_classes    = self.n_classes,
            dim          = config['Unet']['dim'],
            dim_mults    = config['Unet']['dim_mults'],
            in_channels  = config['VQ']['codebook_dim'] * 5,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        self.diffusion = VDM(model = self.unet, scaler = self.scaler, **config['VDM'])
        
        
        self.modules = {
            'stage1' : {
                'encoder' : self.encoder,
                'decoder' : self.decoder,
                'vq_model' : self.vq_model
                },
            'stage2' : {
                'diffusion' : self.diffusion
                }
            }
        
        
        # load trained models, freeze and evaluation
        if self.stage > 1:
            for module_name, module in self.modules['stage1'].items():
                self.load(module, get_root_dir().joinpath('saved_models'), f'{module_name}-{self.dataset_name}.ckpt')
                freeze(module)
                module.eval()
        if self.stage > 2:
            for module_name, module in self.modules['stage2'].items():
                self.load(module, get_root_dir().joinpath('saved_models'), f'{module_name}-{self.dataset_name}.ckpt')
                freeze(module)
                module.eval()
        
        # pre-trained feature extractor in case the perceptual loss is used
        if config['VQ']['perceptual_loss_weight']:
            self.fcn = load_pretrained_FCN(config['dataset']['dataset_name']).to(self.device)
            self.fcn.eval()
            freeze(self.fcn)
    
    
    @torch.no_grad()
    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))

    
    @torch.no_grad()
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize, spectrogram_padding: Callable = None):
        """
        x: (B, C, L)
        """
        
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        if exists(spectrogram_padding):
            xf = spectrogram_padding(xf)
        z = encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(z, vq_model)  # (b c h w), (b (h w) h), ...
        
        return z_q, indices

    
    @torch.no_grad()
    def sample(self, n_samples: int, sampling_steps: int, class_index=None, batch_size=256, guidance_scale=1.):
        n_iters = int(n_samples/batch_size)
        
        
        X = []
        for i in tqdm(range(0, n_samples, batch_size), desc='Sampling:', total = n_iters):
            b = batch_size if i + batch_size <= n_samples else n_samples - i
            
            z = self.diffusion.sample(b, sampling_steps, class_index, guidance_scale) # (B C (H W))
            z = rearrange(z, 'b (c h) w -> b c h w', h=5)
            z_q, embed_ind, _, _ = quantize(z, self.vq_model)
            u = self.decoder(z_q)
            x = timefreq_to_time(u, self.n_fft, self.config['dataset']['in_channels'])  # (B, C, L)

            # append batch
            X.append(x)
    
        X = torch.cat(X)
    
        return X





    def forward(self, batch):
        """
        :param x: input time series (B, C, L)
        """
        if self.stage == 1:
            x, y = batch
            x = x.to(dtype=torch.float32)
            
            recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
            vq_losses = {'vq_loss': None}
            perplexities = {'perplexity': 0.}
    
            # time-frequency transformation: STFT(x)
            C = x.shape[1]
            xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
            
            # register `upsample_size` in the decoders
            for decoder in [self.decoder]:
                if not decoder.is_upsample_size_updated:
                    decoder.register_upsample_size(torch.IntTensor(np.array(xf.shape[2:])))
            
            
            z = self.encoder(xf)
            z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)
            xfhat = self.decoder(z_q)
            xhat = timefreq_to_time(xfhat, self.n_fft, C)  # (B, C, L)
    
            recons_loss['time'] = F.mse_loss(xhat, x)
            recons_loss['timefreq'] = F.mse_loss(xfhat, xf)
            perplexities['perplexity'] = perplexity
            vq_losses['vq_loss'] = vq_loss
    
            if self.config['VQ']['perceptual_loss_weight']:
                z_fcn = self.fcn(x.float(), return_feature_vector=True).detach()
                zhat_fcn = self.fcn(xhat.float(), return_feature_vector=True)
                recons_loss['perceptual'] = F.mse_loss(z_fcn, zhat_fcn)
    
            # plot `x` and `xhat`
            r = np.random.rand()
            if self.training and r <= 0.05:
                b = np.random.randint(0, x.shape[0])
                c = np.random.randint(0, x.shape[1])
    
                plt.title(f'ep_{self.current_epoch}')
                plt.plot(x[b, c].cpu(), label='x')
                plt.plot(xhat[b, c].detach().cpu(), label='xhat')
                plt.legend()
                plt.ylim(-4, 4)
    
                wandb.log({"x vs xhat (training)": wandb.Image(plt)})
                plt.close()
    
            return recons_loss, vq_losses, perplexities

        elif self.stage == 2:
            x, y = batch
            x = x.to(dtype=torch.float32)
            
            z, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (B C H W)
            
            # combine height with channels
            z = rearrange(z, 'B C H W -> B (C H) W')
            
            # Diffusion loss
            loss = self.diffusion(z, y)
            
            return loss


    def training_step(self, batch, batch_idx):
        if self.stage == 1:
            recons_loss, vq_losses, perplexities = self.forward(batch)
            
            loss = (recons_loss['time'] + recons_loss['timefreq'] + \
                    vq_losses['vq_loss']['loss'] + recons_loss['perceptual'])
    
            # lr scheduler
            sch = self.lr_schedulers()
            sch.step()
    
            # log
            loss_hist = {'loss': loss,
                         'recons_loss.time': recons_loss['time'],
                         'recons_loss.timefreq': recons_loss['timefreq'],
    
                         'commit_loss': vq_losses['vq_loss']['commit_loss'],
                         'perplexity': perplexities['perplexity'],
    
                         'perceptual': recons_loss['perceptual']
                         }
    
            detach_the_unnecessary(loss_hist)
            return loss_hist
        
        elif self.stage == 2:
            diffusion_loss = self.forward(batch)
            loss = diffusion_loss
            
            # lr scheduler
            sch = self.lr_schedulers()
            sch.step()
            
            # log
            loss_hist = {'loss': loss,
                         'diffusion_loss': diffusion_loss
                         }
            
            detach_the_unnecessary(loss_hist)
            return loss_hist



    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.stage == 1:
            recons_loss, vq_losses, perplexities = self.forward(batch)
            loss = (recons_loss['time'] + recons_loss['timefreq'] + \
                    vq_losses['vq_loss'] + recons_loss['perceptual'])
    
            # log
            loss_hist = {'loss': loss,
                         'recons_loss.time': recons_loss['time'],
                         'recons_loss.timefreq': recons_loss['timefreq'],
    
                         'commit_loss': vq_losses['commit_loss'],
                         'perplexity': perplexities['perplexity'],
    
                         'perceptual': recons_loss['perceptual']
                         }
    
            detach_the_unnecessary(loss_hist)
            return loss_hist
        
        elif self.stage == 2:
            diffusion_loss = self.forward(batch)
            loss = diffusion_loss
            
            # lr scheduler
            sch = self.lr_schedulers()
            sch.step()
            
            # log
            loss_hist = {'loss': loss,
                         'diffusion_loss': diffusion_loss
                         }
            
            detach_the_unnecessary(loss_hist)
            return loss_hist
        

    
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.stage == 1:
            recons_loss, vq_losses, perplexities = self.forward(batch)
            loss = (recons_loss['time'] + recons_loss['timefreq'] + \
                    vq_losses['vq_loss'] + recons_loss['perceptual'])
    
            # log
            loss_hist = {'loss': loss,
                         'recons_loss.time': recons_loss['time'],
                         'recons_loss.timefreq': recons_loss['timefreq'],
    
                         'commit_loss': vq_losses['commit_loss'],
                         'perplexity': perplexities['perplexity'],
    
                         'perceptual': recons_loss['perceptual']
                         }
    
            detach_the_unnecessary(loss_hist)
            return loss_hist
        
        elif self.stage == 2:
            diffusion_loss = self.forward(batch)
            loss = diffusion_loss
            
            # lr scheduler
            sch = self.lr_schedulers()
            sch.step()
            
            # log
            loss_hist = {'loss': loss,
                         'diffusion_loss': diffusion_loss
                         }
            
            detach_the_unnecessary(loss_hist)
            return loss_hist




    def configure_optimizers(self):
        if self.stage == 1:
            opt = torch.optim.AdamW([{'params': self.encoder.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.decoder.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.vq_model.parameters(), 'lr': self.config['stage1']['lr']},
                                     ],
                                    weight_decay = self.config['stage1']['weight_decay'])
            return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max_stage1)}
        
        elif self.stage == 2:
            opt = torch.optim.AdamW([{'params': self.diffusion.parameters(), 'lr': self.config['stage2']['lr']},
                                     ],
                                    weight_decay = self.config['stage2']['weight_decay'])
            return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max_stage2)}
    
    