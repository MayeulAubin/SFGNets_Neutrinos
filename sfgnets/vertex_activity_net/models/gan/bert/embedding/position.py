"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description:

Note: This script is based on the "Pytorch implementation of Google AI's 2018 BERT" project.
      Original GitHub Repository: https://github.com/codertimo/BERT-pytorch
      File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py
      This is a modified version of the original script to suit specific needs.
"""

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_size=512, dim=125+1):
        super().__init__()
        self.dim = dim
        self.register_buffer('vol_idx', torch.arange(0, self.dim).long())
        self.embedding = nn.Embedding(self.dim, embed_size)

    def forward(self, dim):
        return self.embedding(self.vol_idx[:dim]).unsqueeze(0)
