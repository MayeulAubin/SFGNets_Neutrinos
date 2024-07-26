"""
This script is part of the "Pytorch implementation of Google AI's 2018 BERT" project.
GitHub Repository: https://github.com/codertimo/BERT-pytorch
File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/sublayer.py

Please note that this script remains unchanged from its original version in the repository.
"""

import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
