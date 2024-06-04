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


class LabelEmbedding(nn.Module):
    def __init__(self, label_dim=7, embed_size=512):
        super().__init__()
        self.embedding = nn.Linear(label_dim, embed_size)

    def forward(self, label):
        return self.embedding(label).unsqueeze(1)
