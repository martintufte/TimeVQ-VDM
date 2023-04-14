# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:09:28 2023

@author: martigtu@stud.ntnu.no
"""

### Alternative model

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tempfile
from typing import Union
from tqdm import tqdm
from math import floor

from einops import repeat, rearrange
from typing import Callable

# alternative imports
from generators.unet_alt import Unet
from generators.vdm_alt import VDM

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization.vq import VectorQuantize

from utils import compute_downsample_rate, get_root_dir, freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq, count_parameters, StandardScaler


def load_pretrained_tok_emb(pretrained_tok_emb, tok_emb, freeze_pretrained_tokens: bool):
    """
    :param pretrained_tok_emb: pretrained token embedding from stage 1
    :param tok_emb: token embedding of the transformer
    :return:
    """
    with torch.no_grad():
        if pretrained_tok_emb != None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb
            if freeze_pretrained_tokens:
                tok_emb.weight[:-1, :].requires_grad = False


class QDM(nn.Module):
    """
    Diffusion model with quantization steps.
    """

    def __init__(self,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 **kwargs):
        super().__init__()
        self.config = config
        self.n_classes = n_classes

        self.mask_token_ids = {'LF': config['VQ-VAE']['codebook_sizes']['lf'],
                               'HF': config['VQ-VAE']['codebook_sizes']['hf']}

        # define encoder, decoder, vq_models
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']
        downsampled_width_l = config['encoder']['downsampled_width']['lf']
        downsampled_width_h = config['encoder']['downsampled_width']['hf']
        self.n_fft = config['VQ-VAE']['n_fft']
        downsample_rate_l = compute_downsample_rate(input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(input_length, self.n_fft, downsampled_width_h)
        self.encoder_l = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_l, config['encoder']['n_resnet_blocks'])
        self.decoder_l = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_l, config['decoder']['n_resnet_blocks'])
        self.vq_model_l = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['lf'], **config['VQ-VAE'])
        self.encoder_h = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_h, config['encoder']['n_resnet_blocks'])
        self.decoder_h = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_h, config['decoder']['n_resnet_blocks'])
        self.vq_model_h = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['hf'], **config['VQ-VAE'])

        # load trained models for encoder, decoder, and vq_models
        dataset_name = self.config['dataset']['dataset_name']
        self.load(self.encoder_l, get_root_dir().joinpath('saved_models'), f'encoder_l-{dataset_name}.ckpt')
        self.load(self.decoder_l, get_root_dir().joinpath('saved_models'), f'decoder_l-{dataset_name}.ckpt')
        self.load(self.vq_model_l, get_root_dir().joinpath('saved_models'), f'vq_model_l-{dataset_name}.ckpt')
        self.load(self.encoder_h, get_root_dir().joinpath('saved_models'), f'encoder_h-{dataset_name}.ckpt')
        self.load(self.decoder_h, get_root_dir().joinpath('saved_models'), f'decoder_h-{dataset_name}.ckpt')
        self.load(self.vq_model_h, get_root_dir().joinpath('saved_models'), f'vq_model_h-{dataset_name}.ckpt')

        # freeze the models for encoder, decoder, and vq_models
        freeze(self.encoder_l)
        freeze(self.decoder_l)
        freeze(self.vq_model_l)
        freeze(self.encoder_h)
        freeze(self.decoder_h)
        freeze(self.vq_model_h)

        # evaluation model for encoder, decoder, and vq_models
        self.encoder_l.eval()
        self.decoder_l.eval()
        self.vq_model_l.eval()
        self.encoder_h.eval()
        self.decoder_h.eval()
        self.vq_model_h.eval()

        # token lengths
        self.num_tokens_l = self.encoder_l.num_tokens.item()
        self.num_tokens_h = self.encoder_h.num_tokens.item()

        # latent space dim
        self.H_prime_l, self.H_prime_h = self.encoder_l.H_prime, self.encoder_h.H_prime
        self.W_prime_l, self.W_prime_h = self.encoder_l.W_prime, self.encoder_h.W_prime

        # pretrained discrete tokens
        embed_l = nn.Parameter(copy.deepcopy(self.vq_model_l._codebook.embed))  # pretrained discrete tokens (LF)
        embed_h = nn.Parameter(copy.deepcopy(self.vq_model_h._codebook.embed))  # pretrained discrete tokens (HF)
        
        #codebook_sizes = config['VQ-VAE']['codebook_sizes']
        #codebook_size_l = codebook_sizes['lf']
        #codebook_size_h = codebook_sizes['hf']
        
        self.embed_l = embed_l
        self.embed_h = embed_h
        
        # diffusion model
        self.unet = Unet(
            ts_length    = config['encoder']['downsampled_width']['lf'],
            n_classes    = n_classes,
            dim          = 64,
            dim_mults    = (1, 2, 4),
            in_channels  = config['dataset']['in_channels'],
            out_channels = config['dataset']['in_channels'],
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        print('Trainable parameters for Diffusion Model:', count_parameters(self.unet))

        self.diffusion = VDM(denoising_model = self.unet, **config['VDM'])


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
        
        if spectrogram_padding is not None:
            xf = spectrogram_padding(xf)
            
        z = encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(z, vq_model)  # (b c h w), (b (h w) h), ...
        
        return z_q, indices


    def forward(self, x, y):
        """
        x: (B, C, L)
        y: (B, 1)
        """
        
        #z_l, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (B C H W)
        #z_h, s_h = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (B C H W)
        
        # combine height with channels
        #z_l = rearrange(z_l, 'B C H W -> B (C H) W')
        #z_h = rearrange(z_h, 'B C H W -> B (C H) W')
        
        # Diffusion loss
        diffusion_loss = self.diffusion(x, y)
    
        
        return diffusion_loss
    
    
    def vq_distr(self, x, y):
        B, C, L = x.shape 
        
        # STFT
        u = time_to_timefreq(x, self.n_fft, C) # (B C=2 H W)
        
        # zero-pad
        u_l = zero_pad_high_freq(u) # (B C=2 H=5 W)
        u_h = zero_pad_low_freq(u) # (B C=2 H=5 W)
        
        # encode
        z_l = self.encoder_l(u_l)  # (B C H W)
        z_h = self.encoder_h(u_h) # (B C H W)
        
        # quantize            
        z_l_q, s_l, _, _ = quantize(z_l, self.vq_model_l)
        z_h_q, s_h, _, _ = quantize(z_h, self.vq_model_h)
        
        
        return z_l_q, z_h_q
        
    

    def sample(self, n_samples: int, class_index=None, batch_size=256, guidance_scale=1.):
        n_iters = floor(n_samples/batch_size)
        sampling_steps = 100
        
        X_l, X_h = [], []
        
        for i in tqdm(range(0, n_samples, batch_size), desc='Sampling:', total = n_iters):
            b = batch_size if i + batch_size <= n_samples else n_samples - i
            
            # sample LF part
            z_l = self.diffusion_l.sample(b, sampling_steps, class_index, guidance_scale) # (B C (H W))
            z_l = rearrange(z_l, 'b (c h) w -> b c h w', h=5)
            z_l_q, embed_ind_l, _, _ = quantize(z_l, self.vq_model_l)
            u_l = zero_pad_high_freq(self.decoder_l(z_l_q))
            x_l = timefreq_to_time(u_l, self.n_fft, self.config['dataset']['in_channels'])  # (B, C, L)

            # sample HF part
            z_h = self.diffusion_h.sample(b, sampling_steps, class_index, guidance_scale) # (B C (H W))
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

