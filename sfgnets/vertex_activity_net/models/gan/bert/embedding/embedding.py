"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Embedding module for conditional WGAN-GP (both generator and critic) input.

Note: This script is based on the "Pytorch implementation of Google AI's 2018 BERT" project.
      Original GitHub Repository: https://github.com/codertimo/BERT-pytorch
      File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/bert.py
      This is a modified version of the original script to suit specific needs.
"""

import torch
import torch.nn as nn
from .label import LabelEmbedding
from .position import PositionalEmbedding
from .noise import NoiseEmbedding


class Embedding(nn.Module):

    def __init__(self, input_size, label_size, noise_size,
                 embed_size, dropout=0.1):
        """
        Embedding module for conditional WGAN-GP (both generator and critic) input.

        Args:
            input_size (int): The size of the input data.
            label_size (int): The size of label information.
            noise_size (int): The size of noise vector.
            embed_size (int): The size of the embedding vector.
            dropout (float): The dropout probability for regularisation.
        """
        super().__init__()
        # Embedding of input image (energy loss 1D voxels)
        self.input = nn.Linear(input_size, embed_size)
        # Embedding of input kinematic parameters
        self.label = LabelEmbedding(label_dim=label_size, embed_size=embed_size)
        # Embedding of input voxel positions
        self.position = PositionalEmbedding(embed_size=embed_size)
        # Embedding of input noise
        self.noise = NoiseEmbedding(noise_dim=noise_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, charge, label, noise, cls_tokens):
        """
        Forward pass of the embedding module.

        Args:
            charge (Tensor): The input charge data tensor.
            label (Tensor): The label information tensor.
            noise (Tensor): The noise tensor.
            cls_tokens (Tensor or None): The classification tokens for the critic model.

        Returns:
            Tensor: The output embedding tensor.
        """
        if cls_tokens is None:
            # Generator
            dim = self.position.dim - 1  # 125 for generator
            x = self.label(label) + self.position(dim) + self.noise(noise)
        else:
            # Critic
            x = self.input(charge)  # input embedding
            x = torch.cat((cls_tokens, x), dim=1)  # add cls tokens
            dim = x.size(1)  # 126 for critic
            x = x + self.label(label) + self.position(dim)  # + self.noise(noise)

        return self.dropout(x)
