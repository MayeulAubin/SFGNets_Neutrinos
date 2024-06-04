"""
This script is part of the "Pytorch implementation of Google AI's 2018 BERT" project.
GitHub Repository: https://github.com/codertimo/BERT-pytorch
File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py

Please note that this script remains unchanged from its original version in the repository.
"""

import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))