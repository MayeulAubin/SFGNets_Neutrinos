"""
This script is part of the "Pytorch implementation of Google AI's 2018 BERT" project.
GitHub Repository: https://github.com/codertimo/BERT-pytorch
File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/feed_forward.py

Please note that this script remains unchanged from its original version in the repository.
"""

import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
