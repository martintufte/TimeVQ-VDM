# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:50:41 2023

@author: martigtu@stud.ntnu.no
"""

from math import floor, ceil
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
import random

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from experiments.exp_base import ExpBase, detach_the_unnecessary
from vector_quantization import VectorQuantize
from diffusion_model import Unet, VDM
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN

from utils import exists, default, compute_downsample_rate, freeze, timefreq_to_time, time_to_timefreq, \
    zero_pad_low_freq, zero_pad_high_freq, quantize, get_root_dir, StandardScaler


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class ExpVQVAEDiffusion(ExpBase):
    def __init__(self,
                 config: dict,
                 stage: int):
        """
        :param config: configs/config.yaml
        :param stage: training stage
        """
        
        # --- STAGE 0 ---

        super().__init__()
        self.config = config
        self.stage = stage
        
        self.dataset_name = config['dataset']['dataset_name']
        self.in_channels = config['dataset']['in_channels']
        dataset_summary = pd.read_csv('datasets/DataSummary_UCR.csv')
        self.ts_length = int(dataset_summary.loc[dataset_summary['Name']==self.dataset_name, 'Length'])
        self.n_classes = int(dataset_summary.loc[dataset_summary['Name']==self.dataset_name, 'Class'])
        self.n_train_samples = int(dataset_summary.loc[dataset_summary['Name']==self.dataset_name, 'Train'])
        self.T_max_stage1 = config['stage1']['max_epochs'] * (np.ceil(self.n_train_samples / config['stage1']['batch_size']) + 1)
        self.T_max_stage2 = config['stage2']['max_epochs'] * (np.ceil(self.n_train_samples / config['stage2']['batch_size']) + 1)
        self.n_fft = config['VQ']['n_fft']
        self.hop_length = config['VQ']['hop_length']
        
        
        # --- STAGE 1 ---
        
        dim = config['EncDec']['dim']
        downsampled_width_l = config['EncDec']['downsampled_width']['lf']
        downsampled_width_h = config['EncDec']['downsampled_width']['hf']
        downsample_rate_l = compute_downsample_rate(self.ts_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(self.ts_length, self.n_fft, downsampled_width_h)
        
        # Low frequency encoder, decoder and vector quantizer
        self.encoder_l = VQVAEEncoder(dim, 2 * self.in_channels, downsample_rate_l, config['EncDec']['n_resnet_blocks'])
        self.decoder_l = VQVAEDecoder(dim, 2 * self.in_channels, downsample_rate_l, config['EncDec']['n_resnet_blocks'])
        self.vq_model_l = VectorQuantize(dim, config['VQ']['codebook_sizes']['lf'], **config['VQ'])

        # High frequency encoder, decoder and vector quantizer
        self.encoder_h = VQVAEEncoder(dim, 2 * self.in_channels, downsample_rate_h, config['EncDec']['n_resnet_blocks'])
        self.decoder_h = VQVAEDecoder(dim, 2 * self.in_channels, downsample_rate_h, config['EncDec']['n_resnet_blocks'])
        self.vq_model_h = VectorQuantize(dim, config['VQ']['codebook_sizes']['hf'], **config['VQ'])
        
        # Scalers for latent space
        self.scaler_l = StandardScaler()
        self.scaler_h = StandardScaler()
        
        # stage 1 modules
        self.modules = {'stage1': {
            'encoder_l' : self.encoder_l,
            'decoder_l' : self.decoder_l,
            'vq_model_l' : self.vq_model_l,
            'encoder_h' : self.encoder_h,
            'decoder_h' : self.decoder_h,
            'vq_model_h' : self.vq_model_h,
            'scaler_l' : self.scaler_l,
            'scaler_h' : self.scaler_h
            }
        }
        
        if self.stage > 1:
            for module_name, module in self.modules['stage1'].items():
                self.load(module, get_root_dir().joinpath(f'saved_models/{self.dataset_name}'), f'{module_name}-{self.n_fft}.ckpt')
                freeze(module)
                module.eval()
                
        # pre-trained feature extractor in case the perceptual loss is used
        if config['VQ']['perceptual_loss_weight']:
            self.fcn = load_pretrained_FCN(config['dataset']['dataset_name']).to(self.device)
            self.fcn.eval()
            freeze(self.fcn)
        
        
        # --- STAGE 2 ---
        
        diff_length_l       = int(self.ts_length//2**int(np.log2(downsample_rate_l)+self.hop_length-1))
        diff_length_h       = int(self.ts_length//2**int(np.log2(downsample_rate_h)+self.hop_length-1))
        height_spectrogram  = floor(1 + self.n_fft/2)
        in_channels_l       = config['VQ']['codebook_dim'] * height_spectrogram
        in_channels_h       = config['VQ']['codebook_dim'] * height_spectrogram
        
        
        # Scale inputs for diffusion models learned from the codebooks
        #embed_l = nn.Parameter(copy.deepcopy(self.vq_model_l._codebook.embed))  # pretrained discrete tokens (LF)
        #embed_h = nn.Parameter(copy.deepcopy(self.vq_model_h._codebook.embed))  # pretrained discrete tokens (HF)        
        #self.scaler_l = StandardScaler(mean=float(embed_l.mean()), std=float(embed_l.std()))
        #self.scaler_h = StandardScaler(mean=float(embed_h.mean()), std=float(embed_h.std()))
        print(self.scaler_l.mean, self.scaler_l.std)
        print(self.scaler_h.mean, self.scaler_h.std)
        
        
        
        # Low frequency diffusion model
        self.unet_l = Unet(
            ts_length    = diff_length_l,
            n_classes    = self.n_classes,
            dim          = config['Unet_lf']['dim'],
            dim_mults    = config['Unet_lf']['dim_mults'],
            in_channels  = in_channels_l,
            resnet_block_groups = config['Unet_lf']['resnet_block_groups'],
            time_dim     = config['Unet_lf']['time_dim'],
            class_dim    = config['Unet_lf']['class_dim'],
            in_cond_channels = 0
        )
        self.diffusion_l = VDM(model = self.unet_l, scaler = self.scaler_l, **config['VDM'])
         
        
        # LF to HF conditional embedder
        self.lf_to_hf = nn.Sequential(
                    nn.Upsample(scale_factor=ceil(diff_length_h/diff_length_l), mode='nearest'),
                    Interpolate(size=diff_length_h, mode='linear'),
                    nn.Conv1d(in_channels_l, in_channels_l, kernel_size=3, padding=1)
                )
        
        # High frequency diffusion model
        self.unet_h = Unet(
            ts_length    = diff_length_h,
            n_classes    = self.n_classes,
            dim          = config['Unet_hf']['dim'],
            dim_mults    = config['Unet_hf']['dim_mults'],
            in_channels  = in_channels_h,
            resnet_block_groups = config['Unet_hf']['resnet_block_groups'],
            time_dim     = config['Unet_hf']['time_dim'],
            class_dim    = config['Unet_hf']['class_dim'],
            in_cond_channels = in_channels_l
        )
        self.diffusion_h = VDM(model = self.unet_h, scaler = self.scaler_h, **config['VDM'])
        
        
        self.modules.update({'stage2' : {
            'diffusion_l' : self.diffusion_l,
            'diffusion_h' : self.diffusion_h,
            'lf_to_hf' : self.lf_to_hf
            }
        })
        
        
        # load trained models, freeze and evaluation
        if self.stage > 2:
            for module_name, module in self.modules['stage2'].items():
                self.load(module, get_root_dir().joinpath(f'saved_models/{self.dataset_name}'), f'{module_name}-{self.n_fft}.ckpt')
                freeze(module)
                module.eval()
    
    
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
    def encode_to_z(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize, spectrogram_padding: Callable = None):
        """
        x: (B, C, L)
        """
        
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, self.hop_length, C)  # (B, C, H, W)
        if exists(spectrogram_padding):
            xf = spectrogram_padding(xf)
        z = encoder(xf)  # (b c h w)
        
        return z

    
    @torch.no_grad()
    def sample(self,
               n_samples: int,
               sampling_steps_lf: int,
               sampling_steps_hf: int,
               class_index: int = None,
               batch_size: int = 128,
               guidance_scale: float = 1.0):
        
        X_l, X_h = [], []
        for i in range(0, n_samples, batch_size):
            b = batch_size if i + batch_size <= n_samples else n_samples - i
            print('Generating sample', i, 'to', i+b, 'of', n_samples)
            
            # sample LF part with the diffusion model
            z_l = self.diffusion_l.sample(n_samples = b,
                                          sampling_steps = sampling_steps_lf,
                                          class_condition = class_index,
                                          input_condition = None,
                                          guidance_scale = guidance_scale,
                                          prior_distribution = None,
                                          start_time = 0.0,
                                          end_time = 1.0,
                                          desc = 'LF')
            
            print('z_l:', z_l.mean(), z_l.std())
            
            # hf condition
            hf_cond = self.lf_to_hf(z_l)
            
            z_l = rearrange(z_l, 'b (c h) w -> b c h w', h=5)
            z_l_q, embed_ind_l, _, _ = quantize(z_l, self.vq_model_l)
            u_l = zero_pad_high_freq(self.decoder_l(z_l_q))
            x_l = timefreq_to_time(u_l, self.n_fft, self.hop_length, self.config['dataset']['in_channels'])  # (B, C, L)
            
            
            # sample HF part
            z_h = self.diffusion_h.sample(n_samples = b,
                                          sampling_steps = sampling_steps_hf,
                                          class_condition = class_index,
                                          guidance_scale = guidance_scale,
                                          input_condition = hf_cond,
                                          prior_distribution = None,
                                          start_time = 0.0,
                                          end_time = 1.0,
                                          desc ='HF')
            
            print('z_h:', z_h.mean(), z_h.std())
            
            z_h = rearrange(z_h, 'b (c h) w -> b c h w', h=5)
            z_h_q, embed_ind_h, _, _ = quantize(z_h, self.vq_model_h)
            u_h = zero_pad_low_freq(self.decoder_h(z_h_q))
            x_h = timefreq_to_time(u_h, self.n_fft, self.hop_length, self.config['dataset']['in_channels'])  # (B, C, L)
            
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
            xf = time_to_timefreq(x, self.n_fft, self.hop_length, C)  # (B, C, H, W)
            u_l = zero_pad_high_freq(xf)  # (B, C, H, W)
            x_l = timefreq_to_time(u_l, self.n_fft, self.hop_length, C)  # (B, C, L)
    
            # register `upsample_size` in the decoders
            for decoder in [self.decoder_l, self.decoder_h]:
                if not decoder.is_upsample_size_updated:
                    decoder.register_upsample_size(torch.IntTensor(np.array(xf.shape[2:])))
    
            # forward: low-freq
            z_l = self.encoder_l(u_l)
            z_q_l, indices_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
            xfhat_l = self.decoder_l(z_q_l)
            uhat_l = zero_pad_high_freq(xfhat_l)
            xhat_l = timefreq_to_time(uhat_l, self.n_fft, self.hop_length, C)  # (B, C, L)
    
            recons_loss['LF.time'] = F.mse_loss(x_l, xhat_l)
            recons_loss['LF.timefreq'] = F.mse_loss(u_l, uhat_l)
            perplexities['LF'] = perplexity_l
            vq_losses['LF'] = vq_loss_l
    
            # forward: high-freq
            u_h = zero_pad_low_freq(xf)  # (B, C, H, W)
            x_h = timefreq_to_time(u_h, self.n_fft, self.hop_length, C)  # (B, C, L)
    
            z_h = self.encoder_h(u_h)
            z_q_h, indices_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
            xfhat_h = self.decoder_h(z_q_h)
            uhat_h = zero_pad_low_freq(xfhat_h)
            xhat_h = timefreq_to_time(uhat_h, self.n_fft, self.hop_length, C)  # (B, C, L)
    
            recons_loss['HF.time'] = F.l1_loss(x_h, xhat_h)
            recons_loss['HF.timefreq'] = F.mse_loss(u_h, uhat_h)
            perplexities['HF'] = perplexity_h
            vq_losses['HF'] = vq_loss_h
    
            if self.config['VQ']['perceptual_loss_weight']:
                z_fcn = self.fcn(x.float(), return_feature_vector=True).detach()
                zhat_fcn = self.fcn(xhat_l.float() + xhat_h.float(), return_feature_vector=True)
                recons_loss['perceptual'] = F.mse_loss(z_fcn, zhat_fcn)
    
            # plot `x` and `xhat`
            if self.training and self.current_epoch % 10 == 0:
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
            
            C = x.shape[1]
            xf = time_to_timefreq(x, self.n_fft, self.hop_length, C)  # (B, C, H, W)
            
            # LF
            xf_l = zero_pad_high_freq(xf)
            z_l = self.encoder_l(xf_l)  # (b c h w)
            
            # HF
            xf_h = zero_pad_low_freq(xf)
            z_h = self.encoder_h(xf_h)  # (b c h w)
            
            # combine height with channels
            z_l = rearrange(z_l, 'B C H W -> B (C H) W')
            z_h = rearrange(z_h, 'B C H W -> B (C H) W')
            
            # LF loss
            loss_l = self.diffusion_l(z_l, y)
            
            # hf condition
            hf_cond = self.lf_to_hf(z_l)
            
            # HF loss
            loss_h = self.diffusion_h(z_h, y, hf_cond)
            
            # plot `x_l` and `x_h`
            if self.training and self.current_epoch % 10 == 0:
                y = y.flatten()
                
                # noise
                noise_l = torch.randn_like(z_l)
                noise_h = torch.randn_like(z_h)
                
                b = noise_l.shape[0]
                
                # time
                t = random.random()
                
                # scale input to diffusion model
                z_l_scaled = self.scaler_l.transform(z_l)
                z_h_scaled = self.scaler_h.transform(z_h)
                
                # diffuse
                z_l_diffused = self.diffusion_l.q_sample(z_l_scaled, t, noise_l) # TODO: Fix this!!!
                z_h_diffused = self.diffusion_h.q_sample(z_h_scaled, t, noise_h)
                
                
                
                # LF loss
                pred_noise_l = self.diffusion_l.model(z_l_diffused, t, y)
                
                # hf condition
                hf_cond = self.lf_to_hf(z_l)
                
                # HF loss
                pred_noise_h = self.diffusion_h.model(z_h_diffused, t, y, hf_cond)
                
                b = np.random.randint(0, z_l.shape[0])
                c = np.random.randint(0, z_l.shape[1])
                
                fig, axes = plt.subplots(2, 1, figsize=(6, 2*2))
                plt.suptitle(f'ep_{self.current_epoch}')
                axes[0].plot(noise_l[b, c].cpu())
                axes[0].plot(pred_noise_l[b, c].detach().cpu())
                axes[0].set_title(f'noise_l (t={round(t,2)})')
                axes[0].set_ylim(-4, 4)
    
                b = np.random.randint(0, z_h.shape[0])
                c = np.random.randint(0, z_h.shape[1])
                
                axes[1].plot(noise_h[b, c].cpu())
                axes[1].plot(pred_noise_h[b, c].detach().cpu())
                axes[1].set_title(f'noise_h (t={round(t,2)})')
                axes[1].set_ylim(-4, 4)
    
                plt.tight_layout()
                wandb.log({"noise vs noise_hat (training)": wandb.Image(plt)})
                plt.close()
            
            
            return loss_l, loss_h


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
    
    