# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:50:41 2023

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
    zero_pad_low_freq, zero_pad_high_freq, quantize, get_root_dir, StandardScaler


class ExpVQVAEDiffusion(ExpBase):
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
        
        downsampled_width_l = config['EncDec']['downsampled_width']['lf']
        downsampled_width_h = config['EncDec']['downsampled_width']['hf']
        downsample_rate_l = compute_downsample_rate(self.ts_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(self.ts_length, self.n_fft, downsampled_width_h)
        diff_length_l = int(self.ts_length//2**int(np.log2(downsample_rate_l)+1))
        diff_length_h = int(self.ts_length//2**int(np.log2(downsample_rate_h)+1))
        
        # Low frequency encoder, decoder and vector quantizer
        self.encoder_l = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_l, config['EncDec']['n_resnet_blocks'])
        self.decoder_l = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_l, config['EncDec']['n_resnet_blocks'])
        self.vq_model_l = VectorQuantize(dim, config['VQ']['codebook_sizes']['lf'], **config['VQ'])

        # High frequency encoder, decoder and vector quantizer
        self.encoder_h = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_h, config['EncDec']['n_resnet_blocks'])
        self.decoder_h = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_h, config['EncDec']['n_resnet_blocks'])
        self.vq_model_h = VectorQuantize(dim, config['VQ']['codebook_sizes']['hf'], **config['VQ'])
        
        
        # Scale inputs for diffusion models learned from the codebooks
        embed_l = nn.Parameter(copy.deepcopy(self.vq_model_l._codebook.embed))  # pretrained discrete tokens (LF)
        embed_h = nn.Parameter(copy.deepcopy(self.vq_model_h._codebook.embed))  # pretrained discrete tokens (HF)        
        self.scaler_l = StandardScaler().fit(embed_l)
        self.scaler_h = StandardScaler().fit(embed_h)
        
        # Low frequency diffusion model
        self.unet_l = Unet(
            ts_length    = diff_length_l,
            n_classes    = self.n_classes,
            dim          = config['Unet']['dim'],
            dim_mults    = (1,2),#config['Unet']['dim_mults']['lf'],
            in_channels  = config['VQ']['codebook_dim'] * 5,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        self.diffusion_l = VDM(model = self.unet_l, scaler = self.scaler_l, **config['VDM'])
        
        # High frequency diffusion model
        self.unet_h = Unet(
            ts_length    = diff_length_h,
            n_classes    = self.n_classes,
            dim          = config['Unet']['dim'],
            dim_mults    = (1,2,4,8),#config['Unet']['dim_mults']['hf'],
            in_channels  = config['VQ']['codebook_dim'] * 5,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        self.diffusion_h = VDM(model = self.unet_h, scaler = self.scaler_h, **config['VDM'])
        
        
        self.modules = {
            'stage1' : {
                'encoder_l' : self.encoder_l,
                'decoder_l' : self.decoder_l,
                'vq_model_l' : self.vq_model_l,
                'encoder_h' : self.encoder_h,
                'decoder_h' : self.decoder_h,
                'vq_model_h' : self.vq_model_h
                },
            'stage2' : {
                'diffusion_l' : self.diffusion_l,
                'diffusion_h' : self.diffusion_h
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
    def sample(self, n_samples: int, sampling_steps_lf: int, sampling_steps_hf: int, class_index=None, batch_size=256, guidance_scale=1.):
        n_iters = int(n_samples/batch_size)
        
        
        print('n_samples:', type(n_samples))
        print('batch_size:', batch_size)
        
        
        X_l, X_h = [], []
        
        for i in tqdm(range(0, n_samples, batch_size), desc='Sampling:', total = n_iters):
            b = batch_size if i + batch_size <= n_samples else n_samples - i
            
            # sample LF part
            z_l = self.diffusion_l.sample(b, sampling_steps_lf, class_index, guidance_scale) # (B C (H W))
            z_l = rearrange(z_l, 'b (c h) w -> b c h w', h=5)
            z_l_q, embed_ind_l, _, _ = quantize(z_l, self.vq_model_l)
            u_l = zero_pad_high_freq(self.decoder_l(z_l_q))
            x_l = timefreq_to_time(u_l, self.n_fft, self.config['dataset']['in_channels'])  # (B, C, L)

            # sample HF part
            z_h = self.diffusion_h.sample(b, sampling_steps_hf, class_index, guidance_scale) # (B C (H W))
            z_h = rearrange(z_h, 'b (c h) w -> b c h w', h=5)
            z_h_q, embed_ind_h, _, _ = quantize(z_h, self.vq_model_h)
            u_h = zero_pad_high_freq(self.decoder_h(z_h_q))
            x_h = timefreq_to_time(u_h, self.n_fft, self.config['dataset']['in_channels'])  # (B, C, L)
            
            # append batch
            X_l.append(x_l)
            X_h.append(x_h)
    
    
        X_l = torch.cat(X_l)
        X_h = torch.cat(X_h)
    
        return X_l, X_h





    def forward(self, batch):
        """
        :param x: input time series (B, C, L)
        """
        if self.stage == 1:
            x, y = batch
    
            recons_loss = {'LF.time': 0., 'HF.time': 0., 'LF.timefreq': 0., 'HF.timefreq': 0., 'perceptual': 0.}
            vq_losses = {'LF': None, 'HF': None}
            perplexities = {'LF': 0., 'HF': 0.}
    
            # time-frequency transformation: STFT(x)
            C = x.shape[1]
            xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
            u_l = zero_pad_high_freq(xf)  # (B, C, H, W)
            x_l = timefreq_to_time(u_l, self.n_fft, C)  # (B, C, L)
    
            # register `upsample_size` in the decoders
            for decoder in [self.decoder_l, self.decoder_h]:
                if not decoder.is_upsample_size_updated:
                    decoder.register_upsample_size(torch.IntTensor(np.array(xf.shape[2:])))
    
            # forward: low-freq
            z_l = self.encoder_l(u_l)
            z_q_l, indices_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
            xfhat_l = self.decoder_l(z_q_l)
            uhat_l = zero_pad_high_freq(xfhat_l)
            xhat_l = timefreq_to_time(uhat_l, self.n_fft, C)  # (B, C, L)
    
            recons_loss['LF.time'] = F.mse_loss(x_l, xhat_l)
            recons_loss['LF.timefreq'] = F.mse_loss(u_l, uhat_l)
            perplexities['LF'] = perplexity_l
            vq_losses['LF'] = vq_loss_l
    
            # forward: high-freq
            u_h = zero_pad_low_freq(xf)  # (B, C, H, W)
            x_h = timefreq_to_time(u_h, self.n_fft, C)  # (B, C, L)
    
            z_h = self.encoder_h(u_h)
            z_q_h, indices_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
            xfhat_h = self.decoder_h(z_q_h)
            uhat_h = zero_pad_low_freq(xfhat_h)
            xhat_h = timefreq_to_time(uhat_h, self.n_fft, C)  # (B, C, L)
    
            recons_loss['HF.time'] = F.l1_loss(x_h, xhat_h)
            recons_loss['HF.timefreq'] = F.mse_loss(u_h, uhat_h)
            perplexities['HF'] = perplexity_h
            vq_losses['HF'] = vq_loss_h
    
            if self.config['VQ']['perceptual_loss_weight']:
                z_fcn = self.fcn(x.float(), return_feature_vector=True).detach()
                zhat_fcn = self.fcn(xhat_l.float() + xhat_h.float(), return_feature_vector=True)
                recons_loss['perceptual'] = F.mse_loss(z_fcn, zhat_fcn)
    
            # plot `x` and `xhat`
            r = np.random.rand()
            if self.training and r <= 0.05:
                b = np.random.randint(0, x_h.shape[0])
                c = np.random.randint(0, x_h.shape[1])
    
                fig, axes = plt.subplots(3, 1, figsize=(6, 2*3))
                plt.suptitle(f'ep_{self.current_epoch}')
                axes[0].plot(x_l[b, c].cpu())
                axes[0].plot(xhat_l[b, c].detach().cpu())
                axes[0].set_title('x_l')
                axes[0].set_ylim(-4, 4)
    
                axes[1].plot(x_h[b, c].cpu())
                axes[1].plot(xhat_h[b, c].detach().cpu())
                axes[1].set_title('x_h')
                axes[1].set_ylim(-4, 4)
    
                axes[2].plot(x_l[b, c].cpu() + x_h[b, c].cpu())
                axes[2].plot(xhat_l[b, c].detach().cpu() + xhat_h[b, c].detach().cpu())
                axes[2].set_title('x')
                axes[2].set_ylim(-4, 4)
    
                plt.tight_layout()
                wandb.log({"x vs xhat (training)": wandb.Image(plt)})
                plt.close()
    
            return recons_loss, vq_losses, perplexities

        elif self.stage == 2:
            x, y = batch
            
            z_l, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (B C H W)
            z_h, s_h = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (B C H W)
            
            # combine height with channels
            z_l = rearrange(z_l, 'B C H W -> B (C H) W')
            z_h = rearrange(z_h, 'B C H W -> B (C H) W')
            
            # LF loss
            loss_l = self.diffusion_l(z_l, y)
            
            # HF loss
            loss_h = self.diffusion_h(z_h, y, s_l)
        
            
            return (loss_l, loss_h)


    def training_step(self, batch, batch_idx):
        if self.stage == 1:
            recons_loss, vq_losses, perplexities = self.forward(batch)
            loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                    recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                    vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                    recons_loss['perceptual']
    
            # lr scheduler
            sch = self.lr_schedulers()
            sch.step()
    
            # log
            loss_hist = {'loss': loss,
                         'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                         'recons_loss.LF.time': recons_loss['LF.time'],
                         'recons_loss.HF.time': recons_loss['HF.time'],
    
                         'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                         'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],
    
                         'commit_loss.LF': vq_losses['LF']['commit_loss'],
                         'commit_loss.HF': vq_losses['HF']['commit_loss'],
                         'perplexity.LF': perplexities['LF'],
                         'perplexity.HF': perplexities['HF'],
    
                         'perceptual': recons_loss['perceptual']
                         }
    
            detach_the_unnecessary(loss_hist)
            return loss_hist
        
        elif self.stage == 2:
            diffusion_loss_l, diffusion_loss_h = self.forward(batch)
            loss = diffusion_loss_l + diffusion_loss_h
            
            # lr scheduler
            sch = self.lr_schedulers()
            sch.step()
            
            # log
            loss_hist = {'loss': loss,
                         'diffusion_loss.LF': diffusion_loss_l,
                         'diffusion_loss.HF': diffusion_loss_h
                         }
            
            detach_the_unnecessary(loss_hist)
            return loss_hist



    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.stage == 1:
            recons_loss, vq_losses, perplexities = self.forward(batch)
            loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                    recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                    vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                    recons_loss['perceptual']
    
            # log
            loss_hist = {'loss': loss,
                         'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                         'recons_loss.LF.time': recons_loss['LF.time'],
                         'recons_loss.HF.time': recons_loss['HF.time'],
    
                         'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                         'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],
    
                         'commit_loss.LF': vq_losses['LF']['commit_loss'],
                         'commit_loss.HF': vq_losses['HF']['commit_loss'],
                         'perplexity.LF': perplexities['LF'],
                         'perplexity.HF': perplexities['HF'],
    
                         'perceptual': recons_loss['perceptual']
                         }
    
            detach_the_unnecessary(loss_hist)
            return loss_hist
        
        elif self.stage == 2:
            diffusion_loss_l, diffusion_loss_h = self.forward(batch)
            loss = diffusion_loss_l + diffusion_loss_h

            loss_hist = {'loss': loss,
                         'diffusion_loss.LF': diffusion_loss_l,
                         'diffusion_loss.HF': diffusion_loss_h
                         }
            
            detach_the_unnecessary(loss_hist)
            return loss_hist
        

    
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.stage == 1:
            recons_loss, vq_losses, perplexities = self.forward(batch)
            loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                    recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                    vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                    recons_loss['perceptual']
    
            # log
            loss_hist = {'loss': loss,
                         'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                         'recons_loss.LF.time': recons_loss['LF.time'],
                         'recons_loss.HF.time': recons_loss['HF.time'],
    
                         'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                         'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],
    
                         'commit_loss.LF': vq_losses['LF']['commit_loss'],
                         'commit_loss.HF': vq_losses['HF']['commit_loss'],
                         'perplexity.LF': perplexities['LF'],
                         'perplexity.HF': perplexities['HF'],
    
                         'perceptual': recons_loss['perceptual']
                         }
    
            detach_the_unnecessary(loss_hist)
            return loss_hist
        
        elif self.stage == 2:
            diffusion_loss_l, diffusion_loss_h = self.forward(batch)
            loss = diffusion_loss_l + diffusion_loss_h

            # log
            loss_hist = {'loss': loss,
                         'diffusion_loss.LF': diffusion_loss_l,
                         'diffusion_loss.HF': diffusion_loss_h
                         }
            
            detach_the_unnecessary(loss_hist)
            return loss_hist




    def configure_optimizers(self):
        if self.stage == 1:
            opt = torch.optim.AdamW([{'params': self.encoder_l.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.decoder_l.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.vq_model_l.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.encoder_h.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.decoder_h.parameters(), 'lr': self.config['stage1']['lr']},
                                     {'params': self.vq_model_h.parameters(), 'lr': self.config['stage1']['lr']},
                                     ],
                                    weight_decay = self.config['stage1']['weight_decay'])
            return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max_stage1)}
        
        elif self.stage == 2:
            opt = torch.optim.AdamW([{'params': self.diffusion_l.parameters(), 'lr': self.config['stage2']['lr']},
                                     {'params': self.diffusion_h.parameters(), 'lr': self.config['stage2']['lr']},
                                     ],
                                    weight_decay = self.config['stage2']['weight_decay'])
            return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max_stage2)}
    
    