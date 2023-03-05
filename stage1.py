"""
Stage 1: VQ training

run `python stage1.py`
"""
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_vq_vae import ExpVQVAE
from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.data_pipeline import build_data_pipeline
from utils import *


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args()


def train_stage1(config: dict,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader = None,
                 do_validate: bool = False,
                 wandb_project_case_idx: str = ''
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-stage1'
    if wandb_project_case_idx != '':
        project_name += f'-{wandb_project_case_idx}'

    # fit
    input_length = train_data_loader.dataset.X.shape[-1]
    train_exp = ExpVQVAE(input_length, config, len(train_data_loader.dataset))
    wandb_logger = WandbLogger(project=project_name, name=None, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['stage1'],
                         devices=config['trainer_params']['gpus'],
                         accelerator='gpu')
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader if do_validate else None
                )

    # additional log
    n_trainable_params = count_parameters(train_exp)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    save_model({'encoder_l': train_exp.encoder_l,
                'decoder_l': train_exp.decoder_l,
                'vq_model_l': train_exp.vq_model_l,
                'encoder_h': train_exp.encoder_h,
                'decoder_h': train_exp.decoder_h,
                'vq_model_h': train_exp.vq_model_h,
                }, id=config['dataset']['dataset_name'])


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    dataset_importer = DatasetImporterUCR(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage1']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    # train
    train_stage1(config, train_data_loader)
