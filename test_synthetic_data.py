# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:51:03 2023

@author: martigtu@stud.ntnu.no
"""

# models
from unet_lightning import Unet
from vdm_lightning import VDM

# data
from synthetic_data import SyntheticData, SyntheticLabeledData

# metrics
from metrics import calculate_fid, count_parameters

# torch
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# logger
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from pathlib import Path

# plotting
from plot import nice_plot

import matplotlib.pyplot as plt

import math

def synthetic1():
    # setup
    train       = 1000
    test        = 1000
    n_classes   = 1
    ts_length   = 100
    
    # create data sets
    dataset_train = SyntheticLabeledData([
        SyntheticData(('sin', 'sin'), (70, 25), (0.5, 0.5), train, ts_length)
    ])
    dataset_test = SyntheticLabeledData([
        SyntheticData(('sin', 'sin'), (70, 25), (0.5, 0.5), test, ts_length)
    ])
    
    # create data loader
    batch_size  = 64
    loader_train  = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
    loader_test   = DataLoader(dataset_test, batch_size=batch_size, num_workers=0, shuffle=True)

    return train, test, n_classes, ts_length, batch_size, dataset_train, dataset_test, loader_train, loader_test


def synthetic2():
    # setup
    train       = 1000
    test        = 1000
    n_classes   = 2
    ts_length   = 100
    
    # create data sets
    dataset_train = SyntheticLabeledData([
        SyntheticData(('sin',), (70,), (1.0,), int(0.5*train), ts_length),
        SyntheticData(('sin',), (25,), (1.0,), int(0.5*train), ts_length)
    ])
    dataset_test = SyntheticLabeledData([
        SyntheticData(('sin',), (70,), (1.0,), int(0.5*test), ts_length),
        SyntheticData(('sin',), (25,), (1.0,), int(0.5*test), ts_length)
    ])
    
    # create data loader
    batch_size  = 64
    loader_train  = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
    loader_test   = DataLoader(dataset_test, batch_size=batch_size, num_workers=0, shuffle=True)

    return train, test, n_classes, ts_length, batch_size, dataset_train, dataset_test, loader_train, loader_test



def dim_from_length(ts_length, downsampling_limit=10):
    '''
    Get the dim multipliers for the downsampling layers in the Unet model.
    The downsampling limit is the maximum length of the time series at the lowest layer in the Unet.
    '''
    assert ts_length/downsampling_limit < 512, f'ts_length is to long to be downsampled to length {downsampling_limit}'
    
    d = math.ceil(math.log2(ts_length/downsampling_limit))
    
    dim = 2**(10-d)
    dim_mults = tuple([2**i for i in range(d)])
    
    return dim, dim_mults



if __name__()=='__main__':
    
    ### create synthetic time series data
    path = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')
    model_folder = 'models_synthetic'
    
    
    
    ### --- Sample quality wrt. to number of sampling steps ---
    train, test, n_classes, ts_length, batch_size, dataset_train, dataset_test, loader_train, loader_test = synthetic1()
    
    # unet
    unet = Unet(
        ts_length    = ts_length,
        n_classes    = n_classes,
        dim          = 64,
        dim_mults    = (1, 2, 4, 8),
        time_dim     = 256,
        class_dim    = 128,
        padding_mode = 'replicate'
    )
    print('Trainable parameters:', count_parameters(unet))

    # diffusion model
    diffusion_model = VDM(
        unet,
        loss_type  = 'l2',
        objective  = 'pred_noise',
        train_lr   = 1e-5,
        adam_betas = (0.9, 0.99),
        cond_drop  = 0.1
    )
    
    # fit
    trainer = Trainer(
        default_root_dir = path.joinpath(model_folder),
        max_epochs = 15,
        log_every_n_steps = 10,
        accelerator = 'gpu',
        devices = 1,
        logger = WandbLogger(project='Synthetic', name='synthetic1_plotting')
    )
    trainer.fit(diffusion_model, loader_train, loader_test)
    wandb.finish()

    print('Training Finished!')
    
    
    
    n_samples = 5
        
    c = 0
    fig, axes = plt.subplots(n_samples, 1, figsize=(5.5, 1.5*n_samples))
    for i, ax in enumerate(axes):
        x = diffusion_model.sample(5, [1, 2, 4, 8, 16][i], None)
        for j in range(5):
            ax.plot(x[j, c], linewidth=1)
        ax.grid(color='gray', which='major', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('T='+str([1, 2, 4, 8, 16][i]))
        if False:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            
        #ax.spines['left'].set_visible(False)
        #ax.legend([y[i].numpy()], loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/synthetic1_Tall'+'.pdf')
    plt.show()    
    
    
    
    ### --- Sample quality wrt. to guidance weight ---
    train, test, n_classes, ts_length, batch_size, dataset_train, dataset_test, loader_train, loader_test = synthetic2()
    
    # unet
    unet = Unet(
        ts_length    = ts_length,
        n_classes    = n_classes,
        dim          = 64,
        dim_mults    = (1, 2, 4, 8),
        time_dim     = 256,
        class_dim    = 128,
        padding_mode = 'replicate'
    )
    print('Trainable parameters:', count_parameters(unet))

    # diffusion model
    diffusion_model = VDM(
        unet,
        loss_type  = 'l2',
        objective  = 'pred_noise',
        train_lr   = 1e-5,
        adam_betas = (0.9, 0.99),
        cond_drop  = 0.1
    )
    
    # fit
    trainer = Trainer(
        default_root_dir = path.joinpath(model_folder),
        max_epochs = 15,
        log_every_n_steps = 10,
        accelerator = 'gpu',
        devices = 1,
        logger = WandbLogger(project='Synthetic', name='synthetic2_plotting')
    )
    trainer.fit(diffusion_model, loader_train, loader_test)
    wandb.finish()

    print('Training Finished!')
    
    
    
    n_samples = 10
        
    c = 0
    fig, axes = plt.subplots(5, 1, figsize=(5.5, 1.5*5))
    for i, ax in enumerate(axes):
        x = diffusion_model.sample(5, 32, torch.Tensor([1,1,1,1,1]), [0.0, 0.5, 1.0, 2.0, 3.0][i])
        for j in range(5):
            ax.plot(x[j, c], linewidth=1)
        ax.grid(color='gray', which='major', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('gw='+str([0.0, 0.5, 1.0, 2.0, 3.0][i]))
        if False:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            
        #ax.spines['left'].set_visible(False)
        #ax.legend([y[i].numpy()], loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/synthetic2_Tall1'+'.pdf')
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    import pandas as pd
    df_synthetic = pd.DataFrame({
        'Name' : ['synthetic1', 'synthetic2', 'synthetic3'],
        'train': [1000, 1000, 500],
        'test' : [1000, 1000, 500],
        'class': [1, 2, 5],
        'length': [200, 128, 160],
        'padding_mode': ['zeros', 'zeros', 'zeros'],
        'objective': ['pred_x','pred_x','pred_x'],
        'FID unc': [0,0,0],
    })
    
    
    for i, s in enumerate([synthetic1, synthetic2, synthetic3]):
        train, test, n_classes, ts_length, batch_size, dataset_train, dataset_test, loader_train, loader_test = synthetic1()
        
        name = ['synthetic1', 'synthetic2', 'synthetic3'][i]
        
        
        ### Plot a mini-batch of samples
        for batch in loader_train:
            x, y = batch
            break
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
        
        n_samples = 5
        c = 0
        fig, axes = plt.subplots(n_samples, 1, figsize=(5.5, 1.5*n_samples))
        for i, ax in enumerate(axes):
            ax.plot(x[i, c], linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if False:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                
            #ax.spines['left'].set_visible(False)
            ax.grid(color='gray', which='major', linestyle='--', linewidth=0.5)
            ax.legend([y[i].numpy()], loc='upper right')
        plt.tight_layout()
        plt.savefig('figures/synthetic1_ground_truth.pdf')
        plt.show()    
        
        
        ### Plot a mini-batch of generated samples.
        
        
        
        
        
        # try different parameterizations
        for objective in ['pred_x', 'pred_noise', 'pred_v']:
            for padding_mode in ['zeros', 'replicate']:
                
                dim, dim_mults = dim_from_length(ts_length)
                print('\n Architecture:')
                print('Name', name)
                print('dim:', dim)
                print('dim_mults:', dim_mults)
                print('objective', objective)
                print('padding_mode', padding_mode)
                
                ### unet model
                unet = Unet(
                    ts_length    = ts_length,
                    n_classes    = n_classes,
                    dim          = dim,
                    dim_mults    = dim_mults,
                    time_dim     = 128,
                    class_dim    = 128,
                    padding_mode = padding_mode
                )
                print('Trainable parameters:', count_parameters(unet))
        
                ### diffusion model
                diffusion_model = VDM(
                    unet,
                    loss_type  = 'l2',
                    objective  = objective,
                    train_lr   = 1e-5,
                    adam_betas = (0.9, 0.99),
                    cond_drop  = 0.1
                )
            
                ### fit
                trainer = Trainer(
                    default_root_dir = path.joinpath(model_folder),
                    max_epochs = 15,
                    log_every_n_steps = 10,
                    accelerator = 'gpu',
                    devices = 1,
                    logger = WandbLogger(project='Synthetic', name=str(i)+'_'+objective+'_'+padding_mode)
                )
                trainer.fit(diffusion_model, loader_train, loader_test)
                wandb.finish()
        
                print('Training Finished!')
                
                # sample 1000 samples (unconditionally) and calculate FID score
                n_samples = max(1000, train)
                sampling_steps = 25
                
                # ground truth (training set)
                ground_truth = dataset_train.X.reshape(-1, ts_length)
                
                unc_samples = diffusion_model.sample(n_samples, sampling_steps, None)
                unc_samples = unc_samples.reshape(-1, ts_length)
            
                fid = calculate_fid(unc_samples, ground_truth).numpy()
                
                print('Unconditional FID score is {}.'.format(round(float(fid),4)))
                
                df_synthetic = df_synthetic.append({
                    'Name' : name + '_' + padding_mode + '_' + objective,
                    'train' : train,
                    'test' : test,
                    'class' : n_classes,
                    'length' : ts_length,
                    'padding_mode' : padding_mode,
                    'objective' : objective,
                    'FID unc' : fid
                }, ignore_index=True)
    
    
    
    
    
    '''
    ### sample new data
    n_samples = 10
    sampling_steps = 25
    condition = torch.Tensor([1]).type(torch.int32).repeat(n_samples)
        
    samples = diffusion_model.sample(n_samples, sampling_steps, condition, guidance_weight=1.0)
    samples = diffusion_model.sample(n_samples, sampling_steps, None)
    
    # plot generated samples
    for i in range(10):plt.plot(samples[i, 0, :].cpu())
    plt.plot(samples[4, 0, :].cpu())
    plt.plot(samples[5, 0, :].cpu())
    
    
    ### calculate FID score
    
    # z1: ground truth, concatenation of the training and validation data
    # z2: generated samples
    
    z1 = torch.concat((dataset_train.data, dataset_val.data)).reshape(-1, ts_length)
    z2 = samples[0].reshape(-1, ts_length)
    
    fid = calculate_fid(z1, z2)
    print('FID score is {}.'.format(round(float(fid),4)))
    '''








