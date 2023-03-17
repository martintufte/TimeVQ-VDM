"""
`python sample.py`

sample
    1) unconditional sampling
    2) class-conditional sampling
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt

from generators.timeVQVDM import VQVDM
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings, quantize

from preprocessing.preprocess_ucr import DatasetImporterUCR

from typing import Union


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args()



def plot_generated_samples(x_new_l, x_new_h, x_new, title: str, max_len=20):
    """
    x_new: (n_samples, c, length); c=1 (univariate)
    """
    n_samples = x_new.shape[0]
    if n_samples > max_len:
        print(f"`n_samples` is too large for visualization. maximum is {max_len}")
        return None

    try:
        fig, axes = plt.subplots(1*n_samples, 1, figsize=(3.5, 1.7*n_samples))
        alpha = 0.5
        if n_samples > 1:
            for i, ax in enumerate(axes):
                ax.set_title(title)
                ax.plot(x_new[i, 0, :])
                ax.plot(x_new_l[i, 0, :], alpha=alpha)
                ax.plot(x_new_h[i, 0, :], alpha=alpha)
        else:
            axes.set_title(title)
            axes.plot(x_new[0, 0, :])
            axes.plot(x_new_l[0, 0, :], alpha=alpha)
            axes.plot(x_new_h[0, 0, :], alpha=alpha)

        plt.tight_layout()
        plt.show()
    except ValueError:
        print(f"`n_samples` is too large for visualization. maximum is {max_len}")


def save_generated_samples(x_new: np.ndarray, save: bool, fname: str = None):
    if save:
        fname = 'generated_samples.npy' if not fname else fname
        with open(get_root_dir().joinpath('generated_samples', fname), 'wb') as f:
            np.save(f, x_new)
            print("numpy matrix of the generated samples are saved as `generated_samples/generated_samples.npy`.")

class Sampler(object):
    def __init__(self, real_train_data_loader, config, device):
        self.config = config
        self.device = device
        self.guidance_scale = self.config['class_guidance']['guidance_scale']

        # build VQVAE
        # train_data_loader = build_data_pipeline(self.config, 'train')
        n_classes = len(np.unique(real_train_data_loader.dataset.Y))
        input_length = real_train_data_loader.dataset.X.shape[-1]
        self.vqvdm = VQVDM(input_length, **self.config['VDM'], config=self.config, n_classes=n_classes).to(device)

        # load
        dataset_name = self.config['dataset']['dataset_name']
        ckpt_fname = os.path.join('saved_models', f'vqvdm-{dataset_name}.ckpt')
        saved_state = torch.load(ckpt_fname)
        try:
            self.vqvdm.load_state_dict(saved_state)
        except:
            saved_state_renamed = {}  # need it to load the saved model from the odin server.
            for k, v in saved_state.items():
                if '.ff.' in k:
                    saved_state_renamed[k.replace('.ff.', '.net.')] = v
                else:
                    saved_state_renamed[k] = v
            saved_state = saved_state_renamed
            self.vqvdm.load_state_dict(saved_state)

        # inference mode
        self.vqvdm.eval()

    @torch.no_grad()
    def unconditional_sample(self, n_samples: int, class_index : Union[int, None] = None, batch_size=256):
        return self.vqvdm.sample(n_samples, None, batch_size, self.guidance_scale)

    @torch.no_grad()
    def conditional_sample(self, n_samples: int, class_index: int, batch_size=256):
        """
        class_index: starting from 0. If there are two classes, then `class_index` ∈ {0, 1}.
        """
        return self.vqvdm.sample(n_samples, class_index, batch_size, self.guidance_scale)
    
    
    @torch.no_grad()
    def random_sample(self, n_samples: int, class_index: int, batch_size=256):
        """
        class_index: starting from 0. If there are two classes, then `class_index` ∈ {0, 1}.
        """
        return self.vqvdm.sample(n_samples, class_index, batch_size, self.guidance_scale)
    

    def sample(self, kind: str, n_samples: int, class_index: int, batch_size: int, guidance_scale: int = 1.0):
        if kind == 'unconditional':
            x_new_l, x_new_h = self.unconditional_sample(n_samples, None, batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'conditional':
            x_new_l, x_new_h = self.conditional_sample(n_samples, class_index, batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'random':
            x_new_l, x_new_h = self.random_sample(n_samples, class_index, batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError
        return x_new_l, x_new_h




if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    dataset_importer = DatasetImporterUCR(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage2']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    # load VQVDM
    from generators.timeVQVDM import VQVDM
    sampler = Sampler(train_data_loader, config, device='cpu')
    
    # plot sample along with reconstruction
    for x, y in train_data_loader:
        x, u, u_l, u_h, z_l, z_h, u_l_hat, u_h_hat, x_l_hat, x_h_hat = sampler.vqvdm(x, y, verbose=True)
        plt.plot(x_l_hat[0,0,:].cpu(), color='silver', linestyle='--', label='recon LF')
        plt.plot(x_h_hat[0,0,:].cpu(), color='gray', linestyle='--', label='recon HF')
        plt.plot(x_h_hat[0,0,:].cpu()+x_l_hat[0,0,:].cpu(), color='red', label='recon')
        plt.plot(x[0,0,:].cpu(), label='True')
        plt.legend()
        plt.show()
        break
    
    # plot distribution of tokens
    for x, y in train_data_loader:
        v_l_q, v_h_q = sampler.vqvdm.vq_distr(x, y)
        vql = v_l_q.flatten()
        vqh = v_h_q.flatten()
        plt.hist(vql.cpu(), density=True, bins=50, color='silver', alpha=0.5, label='LF')
        plt.hist(vqh.cpu(), density=True, bins=50, color='red', alpha=0.3, label='HF')
        plt.title('Distribution of tokens in "'+config['dataset']['dataset_name']+'"')
        plt.legend()
        plt.show()
        break
    
    # sample unconditionally
    conditional_samples = sampler.sample(kind='unconditional', n_samples=5, class_index=0, batch_size=10)
    x_l_gen, x_h_gen = conditional_samples
    
    for i in range(5):
        plt.plot(x_l_gen[i,0,:].cpu(), color='silver', linestyle='--', label='gen LF')
        #plt.plot(x_h_gen[i,0,:].cpu(), color='gray', linestyle='--', label='gen HF')
        #plt.plot(x_h_gen[i,0,:].cpu()+x_l_gen[i,0,:].cpu(), color='red', label='generation')
        plt.legend()
        plt.show()
    
    
    
    
    
    
    
    
    