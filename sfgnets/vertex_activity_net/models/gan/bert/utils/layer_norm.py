"""
This script is part of the "Pytorch implementation of Google AI's 2018 BERT" project.
GitHub Repository: https://github.com/codertimo/BERT-pytorch
File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/layer_norm.py

Please note that this script remains unchanged from its original version in the repository.
"""

import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2