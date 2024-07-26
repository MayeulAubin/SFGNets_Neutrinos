"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description:

Note: This script is based on the "Pytorch implementation of Google AI's 2018 BERT" project.
      Original GitHub Repository: https://github.com/codertimo/BERT-pytorch
      File: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/segment.py
      This is a modified version of the original script to suit specific needs.
"""

import torch.nn as nn


class NoiseEmbedding(nn.Linear):
    def __init__(self, noise_dim=100, embed_size=512):
        super().__init__(noise_dim, embed_size)
