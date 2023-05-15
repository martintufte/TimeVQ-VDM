"""
Stage 1: VQ training
Stage 2: Diffusion model training
"""
from argparse import ArgumentParser

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from experiments.exp_vq_vae_diffusion import ExpVQVAEDiffusion

from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, count_parameters, save_model, load_yaml_param_settings, \
    zero_pad_high_freq, zero_pad_low_freq, timefreq_to_time, quantize

from datetime import datetime


def load_args(config):
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', config))
    return parser.parse_args()


def train_stage(config: dict,
                experiment: 'model',
                stage: int,
                train_data_loader: DataLoader,
                test_data_loader: DataLoader = None,
                do_validate: bool = False,
                wandb_project_case_idx: str = ''
                ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    assert stage in {1, 2}, 'Wrong stage entered, need to be 1 or 2!'
    stage_name = 'stage'+str(stage)
    
    project_name = 'TimeVQVDM-'+stage_name
    if wandb_project_case_idx != '':
        project_name += f'-{wandb_project_case_idx}'

    # fit
    train_exp = experiment(config, len(train_data_loader.dataset), stage)
        
    wandb_logger = WandbLogger(project=project_name,
                               name=config['dataset']['dataset_name']+'-'+datetime.now().strftime('%D - %H:%M:%S'),
                               config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config[stage_name]['max_epochs'],
                         devices=[config[stage_name]['gpus']],
                         accelerator='gpu')
    trainer.fit(train_exp,
                train_dataloaders = train_data_loader,
                val_dataloaders = test_data_loader if do_validate else None)

    # additional log
    n_trainable_params = count_parameters(train_exp)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('closing...')
    wandb.finish()

    # saving models
    print('saving the models...')
    if stage == 1:
        save_model(
            {'encoder_l': train_exp.encoder_l,
             'decoder_l': train_exp.decoder_l,
             'vq_model_l': train_exp.vq_model_l,
             'encoder_h': train_exp.encoder_h,
             'decoder_h': train_exp.decoder_h,
             'vq_model_h': train_exp.vq_model_h,
             }, id=config['dataset']['dataset_name']
        )
    else:
        save_model(
            {'diffusion_l': train_exp.diffusion_l,
             'diffusion_h': train_exp.diffusion_h,
             }, id=config['dataset']['dataset_name'])


    # evaluation
    '''
    print('evaluating...')
    input_length = train_data_loader.dataset.X.shape[-1]
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    evaluation = Evaluation(config['dataset']['dataset_name'], config['trainer_params']['gpus'][0], config)
    _, _, x_gen = evaluation.sample(max(evaluation.X_test.shape[0], config['dataset']['batch_sizes']['stage2']),
                                    input_length,
                                    n_classes,
                                    'unconditional')
    z_test, z_gen = evaluation.compute_z(x_gen)
    fid, (z_test, z_gen) = evaluation.fid_score(z_test, z_gen)
    IS_mean, IS_std = evaluation.inception_score(x_gen)
    wandb.log({'FID': fid, 'IS_mean': IS_mean, 'IS_std': IS_std})

    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)
    evaluation.log_pca(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    evaluation.log_tsne(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    '''
    wandb.finish()



if __name__ == '__main__':
    
    # experiment
    experiment = ExpVQDiffSingle
    config_name = 'config_single.yaml'
    stage = 1

    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    dataset_importer = DatasetImporterUCR(**config['dataset'])
    batch_size = config['stage'+str(stage)]['batch_size']
    train_data_loader = build_data_pipeline(batch_size, dataset_importer, config, kind='train')
    test_data_loader = build_data_pipeline(batch_size, dataset_importer, config, kind='test')
    
    # train
    if stage < 3:
        train_stage(config, experiment, stage, train_data_loader)
    else:
        model = experiment(config, len(train_data_loader.dataset), stage)
    
        # sampler
        n_samples           = config['stage3']['n_samples']
        sampling_steps_lf   = config['stage3']['sampling_steps']['lf']
        sampling_steps_hf   = config['stage3']['sampling_steps']['hf']
        class_index         = None
        batch_size          = config['stage3']['batch_size']
        guidance_scale      = config['stage3']['guidance_scale']
        
        
        # plot generated samples
        samples = model.sample(n_samples, sampling_steps_lf, sampling_steps_hf, class_index, batch_size, guidance_scale)
        x_l_gen, x_h_gen = samples
        for i in range(n_samples):
            plt.plot(x_l_gen[i,0,:].cpu(), color='green', alpha=0.2, linestyle='-')
            plt.plot(x_h_gen[i,0,:].cpu(), color='silver', alpha=0.2, linestyle='-')
        plt.title('Generated samples from model')
        plt.savefig('figures/' + config['dataset']['dataset_name'] + '_generated.pdf')
        plt.show()
    
    
        # plot examples from dataset
        for x, y in train_data_loader:
            C = x.shape[1]
            z_l, s_l = model.encode_to_z_q(x, model.encoder_l, model.vq_model_l, zero_pad_high_freq)  # (B C H W)
            z_q_l, _, _, _ = quantize(z_l, model.vq_model_l)
            xfhat_l = model.decoder_l(z_q_l)
            uhat_l = zero_pad_high_freq(xfhat_l)
            x_l = timefreq_to_time(uhat_l, model.n_fft, C)
            
            z_h, s_h = model.encode_to_z_q(x, model.encoder_h, model.vq_model_h, zero_pad_low_freq)  # (B C H W)
            z_q_h, _, _, _ = quantize(z_h, model.vq_model_h)
            xfhat_h = model.decoder_h(z_q_h)
            uhat_h = zero_pad_high_freq(xfhat_h)
            x_h = timefreq_to_time(uhat_h, model.n_fft, C)
            
            for i in range(n_samples):
                plt.plot(x_l[i,0,:].cpu(), color='green', alpha=0.2, linestyle='-')
                plt.plot(x_h[i,0,:].cpu(), color='silver', alpha=0.2, linestyle='-')
            break
        plt.title('Samples from dataset')
        plt.savefig('figures/' + config['dataset']['dataset_name'] + '_examples.pdf')
        plt.show()
    
    
    
    
    









