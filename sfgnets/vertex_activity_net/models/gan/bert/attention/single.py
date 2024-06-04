"""
This script is part of the "Pytorch implementation of Google AI's 2018 BERT" project.
GitHub Repository: https://github.com/codertimo/BERT-pytorch
File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/attention/single.py

Please note that this script remains unchanged from its original version in the repository.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn