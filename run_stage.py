# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:50:41 2023

@author: martigtu@stud.ntnu.no
"""

from argparse import ArgumentParser

import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange

from experiments.exp_vq_vae_diffusion import ExpVQVAEDiffusion

from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, count_parameters, save_model, load_yaml_param_settings, \
    zero_pad_high_freq, zero_pad_low_freq, timefreq_to_time, quantize

from utils.metrics import calculate_fid

from datetime import datetime


def load_args(config):
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', config))
    return parser.parse_args()


def train_stage(config: dict,
                experiment: pl.LightningModule,
                stage: int,
                train_data_loader: DataLoader,
                test_data_loader: DataLoader = None,
                wandb_project_case_idx: str = ''
                ):
    
    assert stage in {1, 2}, 'Wrong stage entered, need to be 1 or 2!'
    stage_name = f'stage{stage}'
    
    project_name = f'TimeVQVDM-{stage_name}'
    if wandb_project_case_idx:
        f'-{wandb_project_case_idx}'

    # --- fit ---
    train_exp = experiment(config, stage)
    
    # logger
    wandb_logger = WandbLogger(project = project_name,
                               name = config['dataset']['dataset_name']+'-'+datetime.now().strftime('%D - %H:%M:%S'),
                               config = config
                               )
    # trainer
    trainer = pl.Trainer(logger = wandb_logger,
                         enable_checkpointing = False,
                         callbacks = [LearningRateMonitor(logging_interval='epoch')],
                         max_epochs = config[stage_name]['max_epochs'],
                         devices = [config[stage_name]['gpus']],
                         accelerator = 'gpu'
                         )
    trainer.fit(train_exp,
                train_dataloaders = train_data_loader,
                val_dataloaders = test_data_loader)

    # additional log
    n_trainable_params = count_parameters(train_exp)
    wandb.log({'n_trainable_params:': n_trainable_params})
    
    # training scalers for latent space representations
    if stage == 1:
        n_samples = 0
        sum_latents_l, sum_latents_h = 0, 0
        sqsum_latents_l, sqsum_latents_h = 0, 0
        for step, (x, y) in enumerate(train_data_loader):
            x = x.to(torch.float32)
            z_l = train_exp.encode_to_z(x, train_exp.encoder_l, train_exp.vq_model_l, zero_pad_high_freq)
            z_h = train_exp.encode_to_z(x, train_exp.encoder_h, train_exp.vq_model_h, zero_pad_low_freq)
            
            n_batch = x.shape[0]
            n_samples += n_batch
            sum_latents_l += torch.mean(z_l)*n_batch
            sum_latents_h += torch.mean(z_h)*n_batch
            sqsum_latents_l += torch.mean(torch.square(z_l))*n_batch
            sqsum_latents_h += torch.mean(torch.square(z_h))*n_batch
            
        # compute mean and std
        mean_latents_l = sum_latents_l / n_samples
        mean_latents_h = sum_latents_h / n_samples
        
        std_latents_l = torch.sqrt( (sqsum_latents_l / n_samples) - mean_latents_l**2 )
        std_latents_h = torch.sqrt( (sqsum_latents_h / n_samples) - mean_latents_h**2 )
        
        # store mean and standard deviations
        train_exp.scaler_l.mean = torch.nn.Parameter(torch.full((1,), mean_latents_l))
        train_exp.scaler_l.std = torch.nn.Parameter(torch.full((1,), std_latents_l))
        train_exp.scaler_h.mean = torch.nn.Parameter(torch.full((1,), mean_latents_h))
        train_exp.scaler_h.std = torch.nn.Parameter(torch.full((1,), std_latents_h))        
        
    
    print('closing...')
    wandb.finish()

    # saving models
    print('saving the models...')
    save_model(train_exp.modules[stage_name],
               dirname = f'saved_models/{config["dataset"]["dataset_name"]}',
               id=str(config['VQ']['n_fft']))
    
    wandb.finish()



def sample_stage(config, experiment, train_data_loader):
    model = experiment(config, stage)
    
    # sampler
    n_samples           = config['sampler']['n_samples']
    sampling_steps_lf   = config['sampler']['sampling_steps']['lf']
    sampling_steps_hf   = config['sampler']['sampling_steps']['hf']
    class_index         = None
    batch_size          = config['sampler']['batch_size']
    guidance_scale      = config['sampler']['guidance_scale']
    
    alpha = 1/n_samples**0.5
    
    # plot generated samples
    x_l_gen, x_h_gen = model.sample(n_samples, sampling_steps_lf, sampling_steps_hf, class_index, batch_size, guidance_scale)
    for i in range(n_samples):
        #plt.plot(x_l_gen[i,0,:].cpu(), color='green', alpha=alpha, linestyle='-')
        #plt.plot(x_h_gen[i,0,:].cpu(), color='silver', alpha=alpha, linestyle='-')
        plt.plot(x_h_gen[i, 0, :].cpu() + x_l_gen[i, 0, :].cpu(), color='green', alpha=alpha, linestyle='-')
    plt.title('Generated samples from model')
    plt.ylim(-2.5, 2.5)
    plt.savefig('figures/' + config['dataset']['dataset_name'] + '_generated.pdf')
    plt.show()
    
    # plot examples from dataset
    n_samples_left = n_samples
    for x, y in train_data_loader:
        
        C = x.shape[1]
        z_l = model.encode_to_z(x, model.encoder_l, model.vq_model_l, zero_pad_high_freq)  # (B C H W)
        z_q_l, _, _, _ = quantize(z_l, model.vq_model_l)
        xfhat_l = model.decoder_l(z_q_l)
        uhat_l = zero_pad_high_freq(xfhat_l)
        x_l = timefreq_to_time(uhat_l, model.n_fft, model.hop_length, C)
        
        z_h = model.encode_to_z(x, model.encoder_h, model.vq_model_h, zero_pad_low_freq)  # (B C H W)
        z_q_h, _, _, _ = quantize(z_h, model.vq_model_h)
        xfhat_h = model.decoder_h(z_q_h)
        uhat_h = zero_pad_low_freq(xfhat_h)
        x_h = timefreq_to_time(uhat_h, model.n_fft, model.hop_length, C)
        
        for i in range(min(x_l.shape[0], n_samples_left)):
            #plt.plot(x_l[i, 0, :].cpu(), color='green', alpha=alpha, linestyle='-')
            #plt.plot(x_h[i, 0, :].cpu(), color='silver', alpha=alpha, linestyle='-')
            plt.plot(x_h[i, 0, :].cpu()+x_l[i, 0, :].cpu(), color='green', alpha=alpha, linestyle='-')
            #plt.plot(x[i, 0, :].cpu(), color='red', alpha=alpha, linestyle='-')
            
        n_samples_left -= batch_size
        if n_samples_left <= 0:
            break
    plt.title('Samples from dataset')
    plt.ylim(-2.5, 2.5)
    plt.savefig('figures/' + config['dataset']['dataset_name'] + '_examples.pdf')
    plt.show()


def eval_stage(config, experiment, train_data_loader, test_data_laoder):
    model = experiment(config, len(train_data_loader.dataset), stage)
    
    # evaluation
    n_samples_gen       = config['evaluation']['n_samples']
    sampling_steps_lf   = config['evaluation']['sampling_steps']['lf']
    sampling_steps_hf   = config['evaluation']['sampling_steps']['hf']
    batch_size          = config['evaluation']['batch_size']
    guidance_scale      = config['evaluation']['guidance_scale']
    
    # training data set
    n_samples_train = len(dataset_importer.X_train)
    n_classes = len(np.unique(dataset_importer.Y_train))
    
    # update number of generated samples
    n_samples_gen = min(n_samples_train, n_samples_gen)
    
    # get number of training samples for each class
    n_samples_per_class = [np.count_nonzero(dataset_importer.Y_train == k) for k in range(n_classes)]
    
    # generate samples
    samples_LF, samples_HF = [], []
    for class_index, n_samples in enumerate(n_samples_per_class, start=0):    
        gen_LF, gen_HF = model.sample(n_samples, sampling_steps_lf, sampling_steps_hf, class_index, batch_size, guidance_scale)
        samples_LF.append(gen_LF)
        samples_HF.append(gen_HF)
    samples_LF, samples_HF = torch.cat(samples_LF), torch.cat(samples_HF)
    
    # train samples
    x = torch.from_numpy(dataset_importer.X_train)
    x = rearrange(x, 'b (c l) -> b c l', c=1).to(torch.float32)
    
    C = x.shape[1]
    z_l, s_l = model.encode_to_z_q(x, model.encoder_l, model.vq_model_l, zero_pad_high_freq)  # (B C H W)
    z_q_l, _, _, _ = quantize(z_l, model.vq_model_l)
    xfhat_l = model.decoder_l(z_q_l)
    uhat_l = zero_pad_high_freq(xfhat_l)
    train_LF = timefreq_to_time(uhat_l, model.n_fft, C)
    
    z_h, s_h = model.encode_to_z_q(x, model.encoder_h, model.vq_model_h, zero_pad_low_freq)  # (B C H W)
    z_q_h, _, _, _ = quantize(z_h, model.vq_model_h)
    xfhat_h = model.decoder_h(z_q_h)
    uhat_h = zero_pad_high_freq(xfhat_h)
    train_HF = timefreq_to_time(uhat_h, model.n_fft, C)
    
    # plot samples
    for i in range(n_samples_gen):
        plt.plot(samples_LF[i, 0, :].cpu(), color='green', alpha=0.05, linestyle='-')
    plt.title('Generated samples from model')
    plt.savefig('figures/' + config['dataset']['dataset_name'] + '_generated.pdf')
    plt.show()
    
    for i in range(n_samples_gen):
        plt.plot(train_LF[i, 0, :].cpu(), color='green', alpha=0.05, linestyle='-')
    plt.title('Samples from dataset')
    plt.savefig('figures/' + config['dataset']['dataset_name'] + '_examples.pdf')
    plt.show()
    
    # calculate FID score
    z1 = rearrange(samples_LF, 'b c l -> b (c l)')
    z2 = rearrange(train_LF, 'b c l -> b (c l)')
    fid = calculate_fid(z1, z2)
    print(fid)

    # fix variance
    z1 = (z1-z1.mean())/z1.std()
    z2 = (z2-z2.mean())/z2.std()
    fid = calculate_fid(z1, z2)
    print(fid)
    
    '''
    # TODO: inplement this!
    # evaluation
    print('evaluating...')
    input_length = train_data_loader.dataset.X.shape[-1]
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    
    
    
    # sampler
    n_samples           = config['sampler']['n_samples']
    sampling_steps_lf   = config['sampler']['sampling_steps']['lf']
    sampling_steps_hf   = config['sampler']['sampling_steps']['hf']
    class_index         = None
    batch_size          = config['sampler']['batch_size']
    guidance_scale      = config['sampler']['guidance_scale']
    
    # plot generated samples
    samples = train_exp.sample(n_samples, sampling_steps_lf, sampling_steps_hf, class_index, batch_size, guidance_scale)
    
    z_test, z_gen = evaluation.compute_z(x_gen)
    
    fid, (z_test, z_gen) = evaluation.fid_score(z_test, z_gen)
    IS_mean, IS_std = evaluation.inception_score(x_gen)
    wandb.log({'FID': fid})

    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)
    evaluation.log_pca(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    evaluation.log_tsne(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    '''
    
    
    

if __name__ == '__main__':
    
    # experiment
    experiment = ExpVQVAEDiffusion
    stage = 3

    # load config
    config_name = 'config.yaml'
    args = load_args(config_name)
    config = load_yaml_param_settings(args.config)

    # data pipeline
    dataset_importer = DatasetImporterUCR(**config['dataset'])
    stage_name = {1: 'stage1', 1.5: 'stage1', 2: 'stage2', 3: 'sampler'}[stage]
    batch_size = config[stage_name]['batch_size']
    train_data_loader = build_data_pipeline(batch_size, dataset_importer, config, kind='train')
    test_data_loader = build_data_pipeline(batch_size, dataset_importer, config, kind='test')

    # stage 1 and 2
    if stage == 1 or stage == 2:
        train_stage(config, experiment, stage, train_data_loader)
    
    # sampler
    elif stage == 3:
        sample_stage(config, experiment, train_data_loader)
    
    # evaluation: fid score
    elif stage == 4:
        eval_stage(config, experiment, train_data_loader, test_data_loader)
