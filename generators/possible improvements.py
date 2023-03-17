# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:18:24 2023

@author: marti
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from generators.timeVQVDM import VQVDM
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings, quantize, split_signal

from preprocessing.preprocess_ucr import DatasetImporterUCR

from typing import Union

from math import floor


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    dataset_importer = DatasetImporterUCR(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage2']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    
    # plot sample along with reconstruction
    for x, y in train_data_loader:
        # plot x
        plt.plot(x[0,0,:])
        
        xf, xf_l, xf_h, x_l, x_h = split_signal(x, n_fft = 4, C = 1)
        
        xf2, xf_l2, xf_h2, x_l2, x_h2 = split_signal(x_l, n_fft = 4, C = 1)
        
        print('x:', x.shape)
        print('xf:', xf.shape)
        print('xf_l:', xf_h.shape)
        print('xf_h:', xf_l.shape)
        print('x_l:', x_l.shape)
        print('x_h:', x_h.shape)
        
        # plot x_L and x_H
        plt.plot(x_l[0,0,:])
        plt.plot(x_h[0,0,:])
        plt.plot(x_l2[0,0,:])
        plt.plot(x_h2[0,0,:])
        
        plt.savefig('figures/test.pdf')
        
        break
    
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    dataset_importer = DatasetImporterUCR(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage2']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]


    # plot sample along with reconstruction
    for x, y in train_data_loader:
        
        ts_length = x.shape[0]
        
        # plot x
        plt.plot(x[0,0,:])
        
        L = floor(ts_length/32)
        
        xf = torch.fft.fft(x)
        
        xf_l = torch.zeros_like(xf)
        xf_h = torch.zeros_like(xf)
        xf_l[:,:,:L] = 2*xf[:,:,:L]
        xf_h[:,:,L:(1-L)] = xf[:,:,L:(1-L)]
        
        xf_l = torch.real(torch.fft.ifft(xf_l))
        xf_h = torch.real(torch.fft.ifft(xf_h))
        
        plt.plot(xf_l[0,0,:])
        plt.plot(xf_h[0,0,:])
        #plt.plot((xf_l + xf_h)[0,0,:])
        
        plt.savefig('figures/test.pdf')
        
        break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    