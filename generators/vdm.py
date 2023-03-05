# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:51:03 2023

@author: martigtu@stud.ntnu.no
"""


from generators.unet_lightning import Unet

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from einops import reduce
from math import pi, prod
from tqdm import tqdm
from typing import Union

from utils import exists, default


class VDM(pl.LightningModule):
    def __init__(
        self,
        model : Unet,
        objective : str           = 'pred_noise',
        loss_type : str           = 'l2',
        lr : float                = 1e-5,
        adam_betas : tuple        = (0.9, 0.99),
        p_unconditional : float   = 0.1
    ):
        super().__init__()
        
        # --- assert inputs ---
        assert loss_type in {'l1', 'l2'}, 'loss_type must be either l1 or l2'
        assert objective in {'pred_noise', 'pred_x', 'pred_v'}, f'objective {objective} is not supported'
        
        # --- architecture ---
        self.model          = model                            # Unet model
        self.channels       = model.in_channels                # number of input channels
        self.ts_length      = model.ts_length                  # time series length
        self.loss_type      = loss_type                        # loss type
        self.objective      = objective                        # prediction objective
        self.lr             = lr                               # learning rate
        self.adam_betas     = adam_betas                       # adam parameters
        self.p_unconditional= p_unconditional                  # conditional dorpout parameter
        self.loss_fn        = {'l1': F.l1_loss, 'l2': F.mse_loss}[self.loss_type]
    
    
    # --- Variance schedule ---
    def alpha(self, t):
        return torch.cos(t * pi/2).view(-1,1,1).to(self.device)
    
    def sigma(self, t):
        return torch.sin(t * pi/2).view(-1,1,1).to(self.device)
    
    
    # --- Forward diffusion process ---
    def q_sample(self, x, t=None, noise=None):
        """
        return a sample from q(z_t | x_0) = alpha_t*x + sigma_t*noise, noise ~ N(0,1)
        """
        
        b, c, l = x.shape # (batch, channels, length)
        
        # time
        t = default(t, lambda: torch.rand((b), device=self.device))
        if t.numel() == 1:
           t = t.repeat(b)
           
        # noise
        noise = default(noise, lambda: torch.randn_like(x))
        
        return self.alpha(t) * x + self.sigma(t) * noise
    
    
    # --- Model predictions ---
    def x_from_noise(self, z, t, noise):
        """
        return x = (z_t - sigma_t * noise) / alpha_t
        """
        return (z - self.sigma(t) * noise) / self.alpha(t)


    def noise_from_x(self, z, t, x):
        """
        return noise = (z_t - alpha_t * x) / sigma_t
        """
        return (z - self.alpha(t) * x) / self.sigma(t)


    def x_from_v(self, z, t, v):
        """
        return start = alpha_t * z_t - sigma_t * v
        """
        return self.alpha(t) * z - self.sigma(t) * v


    def model_predictions(self, z, t, condition=None):
        """
        return the predicted noise and predicted x
        """
        b, _, _ = z.shape
        
        if t.numel() == 1:
            t = t.repeat(b)
        
        pred = self.model(z, t, condition)
        
        if self.objective == 'pred_noise':
            pred_noise = pred
            pred_x = self.x_from_noise(z, t, pred_noise)

        elif self.objective == 'pred_x':
            pred_x = pred
            pred_noise = self.noise_from_x(z, t, pred_x)
            
        elif self.objective == 'pred_v':
            pred_v = pred
            pred_x = self.x_from_v(z, t, pred_v)
            pred_noise = self.noise_from_x(z, t, pred_x)

        return pred_noise, pred_x


    # --- Sampling ---
    @torch.no_grad()
    def q_posterior(self, z_t, s, t, pred_noise):
        """
        return the mean and variance from
        q(z_s | z_t, x) = N(mean, var)
        
        mean = 1/alpha_(t|s) * (z_t + (1-e**lambda(t) - lambda(s)) * noise)
             = 1/alpha_(t|s) * (z_t + sigma_t * (1 - (alpha_t * sigma_s / (alpha_s * sigma_t))**2 ) * noise )
             
        var  = (1 - (alpha_t * sigma_s / (alpha_s * sigma_t))**2 ) sigma_s**2
        """
        
        b, c, l = z_t.shape # (batch, channels, length)
        
        
        alpha_ts   = self.alpha(t) / self.alpha(s)
        sigma_t   = self.sigma(t)
        sigma2_s  = self.sigma(s)**2
        expr      = 1 - sigma2_s * (alpha_ts / sigma_t)**2

        # change Tensors to correct shape and device
        alpha_ts = alpha_ts.view(-1,1,1).to(self.device)
        sigma_t  = sigma_t.view(-1,1,1).to(self.device)
        sigma2_s = sigma2_s.view(-1,1,1).to(self.device)
        expr     = expr.view(-1,1,1).to(self.device)
        
        # calculate model mean
        posterior_mean = 1/alpha_ts * (z_t - sigma_t * expr * pred_noise)
        
        # calculate model variance
        posterior_variance  = expr * sigma2_s
        
        return posterior_mean, posterior_variance


    @torch.no_grad()
    def p_sample(self, z, s, t, condition=None, guidance_weight=1.0):
        """
        single sample loop p_theta(z_s | z_t)
        """
        b, *_ = z.shape
        
        if exists(condition):
            # conditional / unconditional sampling
            cond_noise, cond_x     = self.model_predictions(z, t, condition)
            uncond_noise, uncond_x = self.model_predictions(z, t, None)
            
            # classifier-free predictions
            pred_noise = (1-guidance_weight) * uncond_noise + guidance_weight * cond_noise
            pred_x = (1-guidance_weight) * uncond_x + guidance_weight * cond_x
        else:
            # normal (unconditional sampling)
            pred_noise, pred_x = self.model_predictions(z, t, None)

        # get conditional mean and variance
        model_mean, model_variance = self.q_posterior(z, s, t, pred_noise)
        
        delta = torch.randn_like(z) if t>0 else 0.0
        
        # sample x from a previous time step
        pred_z = model_mean + model_variance * delta
        
        # Z-normalize the variance
        #pred_z /= torch.std(pred_z)
        
        return pred_z, pred_x


    @torch.no_grad()
    def sample(self, n_samples=1, sampling_steps=16, condition=None, guidance_weight=1.0):
        """ 
        Ancestral sampling from the diffusion model
        """
        
        # time discretization
        tau = torch.linspace(1, 0, sampling_steps+1, device=self.device).view(-1,1)
        
        # sample from prior N(0, I)
        z = torch.randn((n_samples, self.channels, self.ts_length), device=self.device)
        
        # sample from p_theta(z_s | z_t, t, condition)
        for s, t in tqdm(zip(tau[1:], tau[:-1]), desc='Sampling', total=sampling_steps):
            z, _ = self.p_sample(z, s, t, condition, guidance_weight)
            
        return z
    

    # --- Training ---
    def quasi_rand(self, shape : Union[int, tuple] = 1):
        '''
        return a quasi uniform distribution on [0, 1].
        u = [r + 0/k (mod 1), r + 1/k (mod 1), ... , r + (k-1)/k (mod 1)]
        '''
        if type(shape) == int:
            shape = (shape,)
        numel = prod(shape)
        
        return (torch.rand(1, device=self.device) + torch.linspace(0, 1-1/numel, numel, device=self.device).view(shape)) % 1
    
    
    def p_losses(self, x, t=None, class_condition : Union[None, torch.Tensor] = None, noise=None):
        """
        calculate the batch loss
        """
        
        b, c, l = x.shape # (batch, channels, length)
        
        # sample quasi-random time
        t = default(t, lambda: self.quasi_rand((b), device=self.device))
            
        # random noise
        noise = default(noise, lambda: torch.randn_like(x, device=self.device))

        # diffused sample
        z_t = self.q_sample(x, t, noise)
        
        # class condition
        if isinstance(class_condition, torch.Tensor):
            conditional_ind = torch.rand(class_condition.shape, device=self.device) > self.p_unconditional
            class_uncondition = torch.Tensor([self.n_classes]).type(torch.int32).repeat(b).to(self.device)
            class_condition = torch.where(conditional_ind, class_condition, class_uncondition)  
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = torch.Tensor([self.n_classes]).type(torch.int32).repeat(b).to(self.device)
            class_condition = class_uncondition
            
        pred = self.model(z_t, t, class_condition)
        
        
        # target
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x':
            target = x
        elif self.objective == 'pred_v':
            target = - self.sigma(t) * x + self.alpha(t) * noise
        else:
            raise ValueError('unknown objective!')
        
        
        # calculate loss
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        
        return loss.mean()

    
    # --- Overwriting PyTorch Lightning in-build methods ---
    def forward(self, x, *args, **kwargs):
        print('This is not how the variational diffusion model should be used! \
               Use the sample method instead!')
               
        '''
        def forward_lf(self, embed_ind_l, class_condition: Union[None, torch.Tensor] = None):
            device = embed_ind_l.device

            token_embeddings = self.tok_emb_l(embed_ind_l)  # (b n dim)
            cls_emb = self.class_embedding(class_condition, embed_ind_l.shape[0], device)  # (b 1 dim)

            n = token_embeddings.shape[1]
            position_embeddings = self.pos_emb.weight[:n, :]
            embed = self.drop(self.ln(token_embeddings + position_embeddings))  # (b, n, dim)
            embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
            embed = self.blocks(embed)  # (b, 1+n, dim)
            embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, n, dim)

            logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
            logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
            return logits

        def forward_hf(self, embed_ind_l, embed_ind_h, class_condition=None):
            """
            embed_ind_l (b n)
            embed_ind_h (b m); m > n
            """
            device = embed_ind_l.device

            token_embeddings_l = self.tok_emb_l(embed_ind_l)  # (b n dim)
            token_embeddings_l = self.projector(token_embeddings_l)  # (b m dim)
            token_embeddings_h = self.tok_emb_h(embed_ind_h)  # (b m dim)
            token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)  # (b m 2*dim)
            cls_emb = self.class_embedding(class_condition, embed_ind_l.shape[0], device)  # (b 1 2*dim)

            n = token_embeddings.shape[1]
            position_embeddings = self.pos_emb.weight[:n, :]
            embed = self.drop(self.ln(token_embeddings + position_embeddings))  # (b, m, 2*dim)
            embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+m, 2*dim)
            embed = self.blocks(embed)  # (b, 1+m, 2*dim)
            embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, m, dim)

            logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
            logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
            return logits

        def forward(self, embed_ind_l, embed_ind_h=None, class_condition: Union[None, torch.Tensor] = None):
            """
            embed_ind: indices for embedding; (b n)
            class_condition: (b 1); if None, unconditional sampling is operated.
            """
            if self.kind == 'LF':
                logits = self.forward_lf(embed_ind_l, class_condition)
            elif self.kind == 'HF':
                logits = self.forward_hf(embed_ind_l, embed_ind_h, class_condition)
            else:
                raise ValueError
            return logits
        
        '''
        
        return x
        
    
    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.flatten() # nescessary if Y is 2D
        
        t = torch.rand([X.shape[0]], device=self.device)
        loss = self.p_losses(X, t, Y)
        self.log("train/loss", loss)
        
        # update number of classes seen
        self.n_samples_seen = self.n_samples_seen.to(self.device)
        self.n_samples_seen += torch.bincount(Y, minlength=self.model.n_classes)
        
        return loss


    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.flatten() # nescessary if Y is 2D
        
        t = torch.rand([X.shape[0]], device=self.device)
        val_loss = self.p_losses(X, t, Y)
        self.log("val/loss", val_loss)
        
        return val_loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, betas=self.adam_betas)
        
        return optimizer
