"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Concatenates the source hits (x, y, z, c), the vertex regression token, and the
             muon exiting information; it embeds them together with the token type
             (0: vertex regression token, 1: muon exiting information, 2: regular hit)
"""

import torch
import torch.nn as nn


# Source sequence embedding
class EmbeddingSource(nn.Module):
    def __init__(self, emb_size=512, img_size=7, dropout=0.1, device=None):
        super().__init__()
        self.vertex_token = nn.Parameter(torch.randn(1, 1))  # vertex token
        self.token_type = torch.tensor([[0, 1, 2]]).long().to(device)  # token type
        self.vertex_emb = nn.Linear(1, emb_size)  # embed vertex token
        self.muon_emb = nn.Linear(6, emb_size)  # embed muon information
        self.embedding_charge = nn.Linear(1, emb_size)  # embed charge
        self.embedding_x = nn.Embedding(img_size, emb_size)  # embed coordinate X
        self.embedding_y = nn.Embedding(img_size, emb_size)  # embed coordinate Y
        self.embedding_z = nn.Embedding(img_size, emb_size)  # embed coordinate Z
        self.embedding_token = nn.Embedding(3, emb_size)  # embed coordinate Z
        self.dropout = nn.Dropout(dropout)

    def forward(self, indexes, charges, exit_muon):
        """
        Forward pass of the source sequence embedding.

        Args:
            indexes (Tensor): Indexes of hits.
            charges (Tensor): Charges (energy loss) of hits.
            exit_muon (Tensor): Exiting muon information.

        Returns:
            src_emb (Tensor): Embedded source sequence.
        """
        # Token for vertex regression
        vertex = self.vertex_token.repeat(1, indexes.size(1), 1)
        vertex_emb = self.vertex_emb(vertex)

        # Embedding of reconstructed exiting muon
        exit_muon_emb = self.muon_emb(exit_muon.reshape(1, -1, 6))

        # Embedding of the particle images
        particles_emb = self.embedding_x(indexes[:, :, 0]) + self.embedding_y(indexes[:, :, 1]) + \
                        self.embedding_z(indexes[:, :, 2]) + self.embedding_charge(charges)

        # Concatenate all the embeddings
        src_emb = torch.cat((vertex_emb, exit_muon_emb, particles_emb), dim=0)

        # Add token (type) embedding
        src_emb[0:1] += self.embedding_token(self.token_type[:, 0:1])
        src_emb[1:2] += self.embedding_token(self.token_type[:, 1:2])
        src_emb[2:] += self.embedding_token(self.token_type[:, 2:3])

        return self.dropout(src_emb)
