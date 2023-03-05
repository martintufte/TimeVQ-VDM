# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:14:49 2023

@author: martigtu@stud.ntnu.no
"""

import math
from typing import Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import repeat

# Variational diffusion model
from generators.unet import Unet
from generators.vdm import VDM

from utils.metrics import count_parameters


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


class LatentDiffusionModel(pl.LightningModule):
    def __init__(self,
                 kind: str,
                 num_tokens: int,
                 codebook_sizes: dict,
                 embed_dim: int,
                 hidden_dim: int,
                 p_unconditional: float,
                 n_classes: int,
                 pretrained_tok_emb_l: nn.Parameter = None,
                 pretrained_tok_emb_h: nn.Parameter = None,
                 freeze_pretrained_tokens: bool = False,
                 num_tokens_l: int = None,
                 **kwargs):
        """
        :param kind:
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param p_unconditional:
        :param n_classes:
        :param pretrained_tok_emb_l: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; low-frequency
        :param pretrained_tok_emb_h: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; high-frequency
        :param freeze_pretrained_tokens:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()
        assert kind in ['LF', 'HF']
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == 'LF' else 2 * embed_dim
        out_dim = embed_dim

        # token embeddings
        self.tok_emb_l = nn.Embedding(codebook_sizes['lf'] + 1, embed_dim)  # `+1` is for mask-token
        load_pretrained_tok_emb(pretrained_tok_emb_l, self.tok_emb_l, freeze_pretrained_tokens)
        if kind == 'HF':
            self.tok_emb_h = nn.Embedding(codebook_sizes['hf'] + 1, embed_dim)  # `+1` is for mask-token
            load_pretrained_tok_emb(pretrained_tok_emb_h, self.tok_emb_h, freeze_pretrained_tokens)

        # diffusion model
        codebook_size = codebook_sizes['lf'] if kind == 'LF' else codebook_sizes['hf']
        unet = Unet(
            ts_length    = codebook_size,
            n_classes    = n_classes,
            dim          = 64,
            dim_mults    = (1, 2, 4, 8),
            in_channels  = in_dim,
            out_channels = out_dim,
            time_dim     = 128,
            class_dim    = 128
        )
        print('Trainable parameters:', count_parameters(unet))

        diffusion_model = VDM(
            unet,
            loss_type  = 'l2',
            objective  = 'pred_noise',
            train_lr   = 1e-5,
            adam_betas = (0.9, 0.99),
            cond_drop  = 0.1
        )
        

        if kind == 'HF':
            self.projector = nn.Conv1d(num_tokens_l, self.num_tokens, kernel_size=1)

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        if isinstance(class_condition, torch.Tensor):
            # if condition is given (conditional sampling)
            conditional_ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = torch.where(conditional_ind, class_condition.long(), class_uncondition)  # (b 1)
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = class_uncondition
        cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
        return cls_emb

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
