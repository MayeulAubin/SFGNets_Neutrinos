"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Mask generation functions for sequences.
"""

import torch


def generate_square_subsequent_mask(size, device):
    """
    Generate a square subsequent mask.

    Parameters:
    - size (int): The size of the mask.
    - device: The device to create the mask on (e.g., 'cuda' or 'cpu').

    Returns:
    - torch.Tensor: A square subsequent mask with appropriate padding.
    """
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_value, device):
    """
    Create masks for source and target sequences.

    Parameters:
    - src (torch.Tensor): Source sequence.
    - tgt (torch.Tensor): Target sequence.
    - pad_value: The padding value.
    - device: The device to create masks on.

    Returns:
    - (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
      Returns source mask, target mask, source padding mask, and target padding mask.
    """
    src_seq_len = src.shape[0] + 2
    tgt_seq_len = tgt.shape[0] + 1

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = torch.zeros(size=(src.size(1), src.size(0)+2), dtype=torch.bool).to(device)
    src_padding_mask[:, 2:] = (src[:, :, 0] == pad_value).transpose(0, 1)

    tgt_padding_mask = torch.zeros(size=(tgt.size(1), tgt.size(0)+1), dtype=torch.bool).to(device)
    tgt_padding_mask[:, 1:] = (tgt[:, :, 0] == pad_value).transpose(0, 1)

    src[src == pad_value] = 0

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def create_mask_src(src, pad_value, device):
    """
    Create masks for source sequences.

    Parameters:
    - src (torch.Tensor): Source sequence.
    - pad_value: The padding value.
    - device: The device to create masks on.

    Returns:
    - (torch.Tensor, torch.Tensor):
      Returns source mask and source padding mask.
    """
    src_seq_len = src.shape[0] + 2

    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    src_padding_mask = torch.zeros(size=(src.size(1), src.size(0)+1), dtype=torch.bool).to(device)
    src_padding_mask[:, 1:] = (src[:, :, 0] == pad_value).transpose(0, 1)

    src[src == pad_value] = 0

    return src_mask, src_padding_mask


def create_mask_tgt(tgt, pad_value, device):
    """
    Create masks for target sequences.

    Parameters:
    - tgt (torch.Tensor): Target sequence.
    - pad_value: The padding value.

    Returns:
    - (torch.Tensor, torch.Tensor):
      Returns target mask and target padding mask.
    """
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt[:, :, 0] == pad_value).transpose(0, 1)

    return tgt_mask, tgt_padding_mask




