"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Applies a standard sinusoidal positional encoding to the target sequence.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor


# Positional encoding for the target sequence
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int = 10):
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size

        # Calculate the sine and cosine positional encodings
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        # Saving buffer (same as parameter without gradients needed)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tgt: Tensor):
        """
        Applies positional encoding to the target sequence.

        Args:
            tgt (Tensor): Target sequence.

        Returns:
            Tensor: Target sequence with positional encoding added (and dropout).
        """
        return self.dropout(tgt * math.sqrt(self.emb_size) + self.pos_embedding[:tgt.size(0), :])
