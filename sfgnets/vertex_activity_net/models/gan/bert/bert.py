"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: BERT model for conditional WGAN-GP (both generator and critic).

Note: This script is based on the "Pytorch implementation of Google AI's 2018 BERT" project.
      Original GitHub Repository: https://github.com/codertimo/BERT-pytorch
      This is a modified version of the original script to suit specific needs.
"""

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from .embedding import Embedding


class BERT(nn.Module):

    def __init__(self, input_size=1, label_size=7, noise_size=100, hidden=768,
                 n_layers=12, attn_heads=12, dropout=0.1):
        """
        Initialise the BERT model for conditional WGAN-GP (both generator and critic).

        Args:
            input_size (int): The size of the input data.
            label_size (int): The size of label information.
            noise_size (int): The size of noise vector.
            hidden (int): The hidden size of the BERT model.
            n_layers (int): The number of transformer layers in BERT.
            attn_heads (int): The number of attention heads in the multi-head self-attention mechanism.
            dropout (float): The dropout probability for regularisation.
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # Paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # Classification token for discriminator (critic)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))

        # Embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedding(input_size=input_size, label_size=label_size, noise_size=noise_size,
                                   embed_size=hidden, dropout=dropout)

        # Multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, input, label, noise):
        """
        Forward pass of the BERT model.

        Args:
            input (Tensor or None): The input data tensor. If None, the model is a generator.
            label (Tensor): The label information tensor.
            noise (Tensor): The noise tensor.

        Returns:
            Tensor: The output tensor after passing through the BERT model.
        """
        batch_size = noise.size(0)
        if input is not None:
            # Add pad token at the beginning (for the critic output_params)
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        else:
            cls_tokens = None

        # Embed the indexed sequence
        x = self.embedding(input, label, noise, cls_tokens)

        # Run over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask=None)

        return x
