import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from ..utils import minkunet
from .dataset import PGunEvent


x_in_channels=2
y_out_channels=np.sum(PGunEvent.TARGETS_LENGTHS)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


#################### TRANSFORMER MODEL #####################

class FittingTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1  # dropout value
                 ):
        super(FittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        # weights initialisation
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self,
                src: Tensor,
                src_mask: Tensor,
                src_padding_mask: Tensor):
        # linear projection of the input
        src_emb = self.proj_input(src)
        # transformer encoder
        memory = self.transformer_encoder(src=src_emb, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        # dropout
        memory = self.dropout(memory)
        # linear projection of the output
        output = self.decoder(memory)
        # output[:, :, :3] += src[:, :, :3]  # learn residuals for x,y,z position
        return output
    
    


def create_baseline_model(x_in_channels:int=x_in_channels,
                          y_out_channels:int=y_out_channels,
                          device:torch.device=device):

    return minkunet.MinkUNet34B(in_channels=x_in_channels, out_channels=y_out_channels, D=3).to(device)



def create_transformer_model(x_in_channels:int=x_in_channels,
                            y_out_channels:int=y_out_channels,
                            device:torch.device=device,
                            D_MODEL:int = 64,
                            N_HEAD:int = 8,
                            DIM_FEEDFORWARD:int = 128,
                            NUM_ENCODER_LAYERS:int = 5):

    return FittingTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                 d_model=D_MODEL,
                                 n_head=N_HEAD,
                                 input_size=3+x_in_channels,
                                 output_size=y_out_channels,
                                 dim_feedforward=DIM_FEEDFORWARD).to(device)