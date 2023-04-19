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
from diffusion_model.unet import Unet
from diffusion_model.vdm import VDM

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization.vq import VectorQuantize

from utils import compute_downsample_rate, get_root_dir, freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq, count_parameters, StandardScaler

import pandas as pd


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


class VQVDM(nn.Module):
    """
    ref: None
    """

    def __init__(self,
                 input_length: int,
                 config: dict,
                 **kwargs):
        super().__init__()
        self.config = config
        self.name = config['dataset']['dataset_name']
        dataset_summary = pd.read_csv('datasets/DataSummary_UCR.csv')
        self.ts_length = int(dataset_summary.loc[dataset_summary['Name']==self.name, 'Length'])
        self.n_classes = int(dataset_summary.loc[dataset_summary['Name']==self.name, 'Class'])
        

        self.mask_token_ids = {'LF': config['VQ']['codebook_sizes']['lf'], 'HF': config['VQ']['codebook_sizes']['hf']}

        # define encoder, decoder, vq_models
        dim = config['EncDec']['dim']
        in_channels = config['dataset']['in_channels']
        downsampled_width_l = config['EncDec']['downsampled_width']['lf']
        downsampled_width_h = config['EncDec']['downsampled_width']['hf']
        self.n_fft = config['VQ']['n_fft']
        downsample_rate_l = compute_downsample_rate(input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(input_length, self.n_fft, downsampled_width_h)
        self.encoder_l = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_l, config['EncDec']['n_resnet_blocks'])
        self.decoder_l = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_l, config['EncDec']['n_resnet_blocks'])
        self.vq_model_l = VectorQuantize(dim, config['VQ']['codebook_sizes']['lf'], **config['VQ'])
        self.encoder_h = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_h, config['EncDec']['n_resnet_blocks'])
        self.decoder_h = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_h, config['EncDec']['n_resnet_blocks'])
        self.vq_model_h = VectorQuantize(dim, config['VQ']['codebook_sizes']['hf'], **config['VQ'])

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
        
        # scaler for diffusion models learned from the codebooks
        self.scaler_l = StandardScaler().fit(embed_l)
        self.scaler_h = StandardScaler().fit(embed_h)
        
        self.embed_l = embed_l
        self.embed_h = embed_h
        
        # compute length for the diffusion models based on downsampling width
        rate_l = compute_downsample_rate(self.ts_length, config['VQ']['n_fft'], config['EncDec']['downsampled_width']['lf'])
        rate_h = compute_downsample_rate(self.ts_length, config['VQ']['n_fft'], config['EncDec']['downsampled_width']['hf'])
        length_l = int(self.ts_length//2**int(np.log2(rate_l)+1))
        length_h = int(self.ts_length//2**int(np.log2(rate_h)+1))
        
        
        # diffusion model LF
        self.unet_l = Unet(
            ts_length    = length_l,
            n_classes    = self.n_classes,
            dim          = config['Unet']['dim'],
            dim_mults    = (1,2),#config['Unet']['dim_mults']['lf'],
            in_channels  = config['VQ']['codebook_dim'] * 5,
            out_channels = config['VQ']['codebook_dim'] * 5,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        print('Trainable parameters for LF:', count_parameters(self.unet_l))

        self.diffusion_l = VDM(
            model = self.unet_l,
            scaler = self.scaler_l,
            **config['VDM']
        )
        
        # diffusion model HF
        self.unet_h = Unet(
            ts_length    = length_h,
            n_classes    = self.n_classes,
            dim          = config['Unet']['dim'],
            dim_mults    = (1,2,4,8),#config['Unet']['dim_mults']['hf'],
            in_channels  = config['VQ']['codebook_dim'] * 5,
            out_channels = config['VQ']['codebook_dim'] * 5,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        print('Trainable parameters for HF:', count_parameters(self.unet_h))

        self.diffusion_h = VDM(
            model = self.unet_h,
            scaler = self.scaler_h,
            **config['VDM'],
        )


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


    def forward(self, x, y, verbose=False):
        """
        x: (B, C, L)
        y: (B, 1)
        """
        
        if verbose:
            '''
            def forward(self, batch):
                """
                :param x: input time series (B, C, L)
                """
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
            '''
            B, C, L = x.shape 
            
            # STFT
            u = time_to_timefreq(x, self.n_fft, C) # (B C=2 H W)
            
            # zero-pad
            u_l = zero_pad_high_freq(u) # (B C=2 H=5 W)
            u_h = zero_pad_low_freq(u) # (B C=2 H=5 W)
            
            # LF and HF signals
            x_l = timefreq_to_time(u_l, self.n_fft, C)
            x_h = timefreq_to_time(u_h, self.n_fft, C)
            
            # encode
            z_l = self.encoder_l(u_l)  # (B C H W)
            z_h = self.encoder_h(u_h) # (B C H W)
            
            # quantize            
            z_l_q, s_l, _, _ = quantize(z_l, self.vq_model_l)
            z_h_q, s_h, _, _ = quantize(z_h, self.vq_model_h)
            
            # decode
            u_l_hat = self.decoder_l(z_l_q)
            u_h_hat = self.decoder_h(z_h_q)
            
            # zero-pad
            u_l_hat = zero_pad_high_freq(u_l_hat)
            u_h_hat = zero_pad_low_freq(u_h_hat)
            
            # inverse STFT
            x_l_hat = timefreq_to_time(u_l_hat, self.n_fft, self.config['dataset']['in_channels'])
            x_h_hat = timefreq_to_time(u_h_hat, self.n_fft, self.config['dataset']['in_channels'])
            
            
            return x, x_l, x_h, u, u_l, u_h, z_l, z_h, u_l_hat, u_h_hat, x_l_hat, x_h_hat


        else:
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
        
    

    def sample(self, n_samples: int, sampling_steps: int, class_index=None, batch_size=256, guidance_scale=1.):
        n_iters = int((n_samples/batch_size))
        
        
        print('n_samples:', type(n_samples))
        print('batch_size:', batch_size)
        
        
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

