import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.timeVQVDM import VQVDM


class ExpVQVDM(ExpBase):
    def __init__(self,
                 input_length: int,
                 config: dict,
                 n_train_samples: int,
                 n_classes: int):
        super().__init__()
        self.config = config
        self.vqvdm = VQVDM(input_length, config=config, n_classes=n_classes)
        self.T_max = config['trainer_params']['max_epochs']['stage2'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage2']) + 1)


    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.vqvdm.parameters(), 'lr': self.config['exp_params']['lr']},],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # log
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist
