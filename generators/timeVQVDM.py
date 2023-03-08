import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tempfile
from typing import Union

from einops import repeat, rearrange
from typing import Callable
from generators.unet import Unet
from generators.vdm import VDM

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization.vq import VectorQuantize

from utils import compute_downsample_rate, get_root_dir, freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq, count_parameters


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
                 n_classes: int,
                 **kwargs):
        super().__init__()
        self.config = config
        self.n_classes = n_classes

        self.mask_token_ids = {'LF': config['VQ-VAE']['codebook_sizes']['lf'], 'HF': config['VQ-VAE']['codebook_sizes']['hf']}

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
        
        codebook_sizes = config['VQ-VAE']['codebook_sizes']
        codebook_size_l = codebook_sizes['lf']
        codebook_size_h = codebook_sizes['hf']
        
        # diffusion model LF
        self.unet_l = Unet(
            ts_length    = config['VQ-VAE']['codebook_dim'],
            n_classes    = n_classes,
            dim          = 64,
            dim_mults    = (1,),
            in_channels  = 64,#self.num_tokens_l,
            out_channels = 64,#self.num_tokens_l,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        print('Trainable parameters for LF:', count_parameters(self.unet_l))

        self.diffusion_l = VDM(
            kind = 'LF',
            model = self.unet_l,
            **config['VDM']
        )
        
        # diffusion model HF
        self.unet_h = Unet(
            ts_length    = config['VQ-VAE']['codebook_dim'],
            n_classes    = n_classes,
            dim          = 64,
            dim_mults    = (1, 2, 4),
            in_channels  = 64,#self.num_tokens_h,
            out_channels = 64,#self.num_tokens_h,
            resnet_block_groups = config['Unet']['resnet_block_groups'],
            time_dim     = config['Unet']['time_dim'],
            class_dim    = config['Unet']['class_dim']
        )
        print('Trainable parameters for HF:', count_parameters(self.unet_h))

        self.diffusion_h = VDM(
            kind = 'HF',
            model = self.unet_h,
            num_tokens_l = self.num_tokens_l,
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


    def forward(self, x, y):
        """
        x: (B, C, L)
        y: (B, 1)
        """
        
        z_l, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (B C H W)
        z_h, _ = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (B C H W)
        
        # combine height and width of z_l
        z_l = torch.flatten(z_l, start_dim=-2) # (B C H*W)
        z_h = torch.flatten(z_h, start_dim=-2) # (B C H*W)
        
        # LF loss
        loss_l = self.diffusion_l(z_l, y)
        
        # HF loss
        loss_h = self.diffusion_h(z_h, y, s_l)
        
        
        return (loss_l, loss_h)


    def first_pass(self,
                   n_samples: int = 1,
                   sampling_steps: int = 16,
                   class_condition: Union[None, int, torch.Tensor] = None,
                   guidance_scale: float = 1.0,
                   ):
        
        z_l = self.diffusion_l.sample(n_samples, sampling_steps, class_condition, guidance_scale)        
  
        return z_l

    def second_pass(self,
                    s_l: torch.Tensor,
                    n_samples: int = 1,
                    sampling_steps: int = 16,
                    class_condition: Union[None, int, torch.Tensor] = None,
                    guidance_scale: float = 1.0,
                    ):
        
        # TODO: Include z_l
        
        z_h = self.diffusion_h.sample(n_samples, sampling_steps, class_condition, guidance_scale)
        
        return z_h
    
    
    def decode_token_ind_to_timeseries(self, s: torch.Tensor, frequency: str, return_representations: bool = False):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        :param frequency:
        :param return_representations:
        :return:
        """
        assert frequency in ['LF', 'HF']

        vq_model = self.vq_model_l if frequency == 'LF' else self.vq_model_h
        decoder = self.decoder_l if frequency == 'LF' else self.decoder_h
        zero_pad = zero_pad_high_freq if frequency == 'LF' else zero_pad_low_freq

        quantize = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        quantize = vq_model.project_out(quantize)  # (b n c)
        quantize = rearrange(quantize, 'b n c -> b c n')  # (b c n) == (b c (h w))
        H_prime = self.H_prime_l if frequency == 'LF' else self.H_prime_h
        W_prime = self.W_prime_l if frequency == 'LF' else self.W_prime_h
        quantize = rearrange(quantize, 'b c (h w) -> b c h w', h=H_prime, w=W_prime)

        xfhat = decoder(quantize)

        uhat = zero_pad(xfhat)
        xhat = timefreq_to_time(uhat, self.n_fft, self.config['dataset']['in_channels'])  # (B, C, L)

        if return_representations:
            return xhat, quantize
        else:
            return xhat
