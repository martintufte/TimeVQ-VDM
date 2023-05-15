# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:51:03 2023

@author: martigtu@stud.ntnu.no
"""


from diffusion_model.unet import Unet
from utils import exists, default, StandardScaler

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import reduce
from math import pi, prod
from tqdm import tqdm
from typing import Union
import random

class VDM(pl.LightningModule):
    def __init__(
        self,
        model : Unet,
        scaler : StandardScaler   = None,
        objective : str           = 'pred_noise',
        p_unconditional : float   = 0.1,
        loss_type : str           = 'l2',
#        lr : float                = 1e-5,
#        adam_betas : tuple        = (0.9, 0.99),
#        gamma: float              = None
    ):
        super().__init__()
        
        # --- assert inputs ---
        assert objective in {'pred_noise', 'pred_x', 'pred_v'}, f'objective {objective} is not supported!'
        assert loss_type in {'l1', 'l2'}, 'loss_type must be either l1 or l2'
        
        # --- architecture ---
        self.model          = model                            # Unet model
        self.scale_input    = exists(scaler)
        self.scaler         = default(scaler, StandardScaler(0.0, 1.0))
        self.channels       = model.in_channels                # number of input channels
        self.ts_length      = model.ts_length                  # time series length
        self.n_classes      = model.n_classes                  # number of classes
        self.loss_type      = loss_type                        # loss type
        self.objective      = objective                        # prediction objective
        self.p_unconditional= p_unconditional                  # unconditional probability
        self.loss_fn        = {'l1': F.l1_loss, 'l2': F.mse_loss}[self.loss_type]
#        self.lr             = lr                               # learning rate
#        self.adam_betas     = adam_betas                       # adam parameters
#        self.gamma          = gamma                            # minSNR-{gamma}
        
    
    # --- Variance schedule ---
    def alpha(self, t):
        return torch.cos(t * pi/2).view(-1,1,1).to(self.device)
    
    def sigma(self, t):
        return torch.sin(t * pi/2).view(-1,1,1).to(self.device)
    
    

#    def SNR(self, t):
#        return (torch.cos(t * pi/2)**2 / torch.sin(t * pi/2)**2).to(self.device)
#    
#    def minSNR_gamma(self, t):
#        return torch.minimum(torch.full((1,), self.gamma, device=self.device), self.SNR(t)).view(-1,1)
    
    
    # --- Forward diffusion process ---
    def q_sample(self,
                 z0,
                 t: Union[None, float, torch.Tensor] = None,
                 noise: Union[None, torch.Tensor] = None):
        """
        sample from q(z_t | z_0) = alpha_t*z_0 + sigma_t*noise, noise ~ N(0,1)
        """
        
        b, c, l = z0.shape # (batch, channels, length)
        
        # time
        t = default(t, lambda: torch.quasi_rand((b,), device=self.device))
        if type(t) == float:
            t = torch.full((b,), t, device=self.device)
        elif t.numel() == 1:
           t = t.repeat(b)
           
        # noise
        noise = default(noise, lambda: torch.randn_like(z0, device=self.device))
        
        return self.alpha(t) * z0 + self.sigma(t) * noise
    
    
    # --- Model predictions ---
    def z0_from_noise(self, z, t, noise):
        """ z_0 = (z_t - sigma_t * noise) / alpha_t """
        return (z - self.sigma(t) * noise) / self.alpha(t)


    def noise_from_z0(self, z, t, z0):
        """ noise = (z_t - alpha_t * z_0) / sigma_t """
        return (z - self.alpha(t) * z0) / self.sigma(t)


    def z0_from_v(self, z, t, v):
        """ z0 = alpha_t * z_t - sigma_t * v """
        return self.alpha(t) * z - self.sigma(t) * v
    
    
    def v_from_z0_noise(self, t, z0, noise):
        """ - self.sigma(t) * z0 + self.alpha(t) * noise """
        return - self.sigma(t) * z0 + self.alpha(t) * noise


    def model_predictions(self,
                          z: torch.Tensor,
                          t: Union[float, torch.Tensor],
                          class_condition: Union[None, torch.Tensor] = None,
                          input_condition: Union[None, torch.Tensor] = None):
        """
        return the predicted noise and z0
        """
        b, c, l = z.shape # (batch, channels, length)
        
        # time        
        if type(t) == float:
            t = torch.full((b,), t, device=self.device)
        elif t.numel() == 1:
            t = t.repeat(b)
        
        # prediction
        pred = self.model(z, t, class_condition, input_condition)
        
        if self.objective == 'pred_noise':
            pred_noise = pred
            pred_z0 = self.z0_from_noise(z, t, pred_noise)

        elif self.objective == 'pred_x':
            pred_z0 = pred
            pred_noise = self.noise_from_z0(z, t, pred_z0)
            
        elif self.objective == 'pred_v':
            pred_v = pred
            pred_z0 = self.z0_from_v(z, t, pred_v)
            pred_noise = self.noise_from_z0(z, t, pred_z0)

        return pred_noise, pred_z0


    # --- Sampling ---
    @torch.no_grad()
    def q_posterior(self, z_t, s, t, pred_noise):
        """
        return the mean and variance from
        q(z_s | z_t, z_0) = N(mean, var)
        
        mean = 1/alpha_(t|s) * (z_t + (1-e**(lambda(t) - lambda(s)) * noise))
             = 1/alpha_(t|s) * (z_t + sigma_t * (1 - (alpha_t * sigma_s / (alpha_s * sigma_t))**2 ) * noise )
             
        var  = (1 - (alpha_t * sigma_s / (alpha_s * sigma_t))**2 ) sigma_s**2
        """
        alpha_ts   = self.alpha(t) / self.alpha(s)
        sigma_t   = self.sigma(t)
        sigma2_s  = self.sigma(s)**2
        expr      = 1 - sigma2_s * (alpha_ts / sigma_t)**2
        
        # calculate model mean
        posterior_mean = 1/alpha_ts * (z_t - sigma_t * expr * pred_noise)
        
        # calculate model standard deviation
        posterior_std  = torch.sqrt(expr * sigma2_s)
        
        return posterior_mean, posterior_std


    @torch.no_grad()
    def p_sample(self,
                 z: torch.Tensor,
                 s: torch.Tensor,
                 t: torch.Tensor,
                 class_condition: Union[None, torch.Tensor] = None,
                 input_condition: Union[None, torch.Tensor] = None,
                 guidance_scale: float = 1.0,
                 z_normalize: bool = True):
        """
        single sample loop p_theta(z_s | z_t)
        """
        b, *_ = z.shape
        
        if exists(class_condition):
            # conditional sampling
            cond_noise, cond_z0     = self.model_predictions(z, t, class_condition, input_condition)
            uncond_noise, uncond_z0 = self.model_predictions(z, t, None, input_condition)
            
            # classifier-free predictions
            pred_noise = (1-guidance_scale) * uncond_noise + guidance_scale * cond_noise
            pred_z0 = (1-guidance_scale) * uncond_z0 + guidance_scale * cond_z0
        else:
            # unconditional sampling
            pred_noise, pred_z0 = self.model_predictions(z, t, None, input_condition)

        # get conditional mean and variance
        model_mean, model_std = self.q_posterior(z, s, t, pred_noise)
        
        # sample z from a previous time step
        delta = torch.randn_like(z) if t>0 else 0.0
        pred_z = model_mean + model_std * delta
        
        if z_normalize:
            pred_z /= torch.std(pred_z)
        
        return pred_z, pred_z0


    @torch.no_grad()
    def sample(self,
               n_samples: int = 1,
               sampling_steps: int = 16,
               class_condition: Union[None, int, torch.Tensor] = None,
               input_condition: Union[None, torch.Tensor] = None,
               guidance_scale: float = 1.0,
               prior_distribution : Union[None, torch.Tensor] = None,
               start_time: float = 0.0,
               end_time: float = 1.0,
               z_normalize: bool = True,
               desc: str ='Sampling'):
        """ 
        Ancestral sampling from the diffusion model
        """
        
        # class condition
        if type(class_condition) == int:
            class_condition = torch.full((n_samples,), class_condition, device=self.device)
        
        # time discretization
        tau = torch.linspace(end_time, start_time, sampling_steps+1, device=self.device).view(-1,1)
        
        # sample from prior distribtuion, usually N(0, I)
        z = default(prior_distribution, torch.randn((n_samples, self.channels, self.ts_length), device=self.device))
        
        # iterative denoising using p_theta(z_s | z_t, t, class_condition)
        for s, t in tqdm(zip(tau[1:], tau[:-1]), desc=desc, total=sampling_steps):
            z, _ = self.p_sample(z, s, t, class_condition, input_condition, guidance_scale, z_normalize)
        
        # scale generated samples
        if self.scale_input:
            z = self.scaler.inverse_transform(z)
            
        return z
    

    # --- Training ---
    @torch.no_grad()
    def quasi_rand(self, shape : Union[int, tuple] = 1):
        '''
        return a quasi uniform distribution on [0, 1].
        u = [r + 0/k (mod 1), r + 1/k (mod 1), ... , r + (k-1)/k (mod 1)]
        '''
        if type(shape) == int:
            shape = (shape,)
        numel = prod(shape)
        
        return (torch.rand(1, device=self.device) + torch.linspace(0, 1-1/numel, numel, device=self.device).view(shape)) % 1
    
    
    def p_losses(self,
                 z0,
                 class_condition: Union[None, torch.Tensor] = None,
                 input_condition: Union[None, torch.Tensor] = None,
                 t:               Union[None, torch.Tensor] = None,
                 noise:           Union[None, torch.Tensor] = None):
        """
        calculate the batch loss
        """
        
        b, c, l = z0.shape # (batch, channels, length)
        
        # quasi-uniform time
        t = default(t, lambda: self.quasi_rand(b))
        
        # gaussian noise
        noise = default(noise, lambda: torch.randn_like(z0, device=self.device))

        # diffuse sample
        z_t = self.q_sample(z0, t, noise)
        
        # class condition
        if isinstance(class_condition, torch.Tensor):
            conditional_idx = torch.rand(class_condition.shape, device=self.device) > self.p_unconditional
            class_uncondition = torch.full((b,), self.n_classes, device=self.device)
            class_condition = torch.where(conditional_idx, class_condition, class_uncondition)  
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = torch.full((b,), self.n_classes, device=self.device)
            class_condition = class_uncondition
        
        pred = self.model(z_t, t, class_condition, input_condition)
        
        
        # target
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x':
            target = z0
        elif self.objective == 'pred_v':
            target = self.v_from_z0_noise(t, z0, noise)
        else:
            raise ValueError('unknown objective!')
        
        
        # calculate loss
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        
#        # weight loss function
#        if self.gamma:
#            loss *= self.minSNR_gamma(t) / 2.02985
            
        return loss.mean()
    
    
    def forward(self,
                z0,
                class_condition: Union[None, torch.Tensor] = None,
                input_condition: Union[None, torch.Tensor] = None,
                t:               Union[None, torch.Tensor] = None,
                noise:           Union[None, torch.Tensor] = None):
        
        b, c, l = z0.shape # (batch, channels, length)

        # scale input z0
        if self.scale_input:
            z0 = self.scaler.transform(z0)
        
        # print
        if random.random() < 0.01:
            print('mean =', z0.mean(), 'var = ', z0.var())
        
        # flatten class condition
        class_condition = class_condition.flatten()
        
        # input condition
        input_condition = input_condition.to(device=self.device) if exists(input_condition) else None
        
        return self.p_losses(z0, class_condition, input_condition, t, noise)


    '''
    # --- Overwriting PyTorch Lightning in-build methods ---
    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.flatten() # nescessary if Y is 2D
        
        t = self.quasi_rand(X.shape[0])
        loss = self.p_losses(X, t, Y)
        self.log("train/loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.flatten() # nescessary if Y is 2D
        
        t = self.quasi_rand(X.shape[0])
        val_loss = self.p_losses(X, t, Y)
        self.log("val/loss", val_loss)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, betas=self.adam_betas)
        
        return optimizer
    '''