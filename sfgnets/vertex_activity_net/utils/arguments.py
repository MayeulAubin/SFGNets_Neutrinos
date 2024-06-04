"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: argparse arguments for each neural network.
"""

import argparse


def args_transformer(version=1):
    """
    Create an argument parser for Transformer model configuration.

    Args:
        version (int): The version of the Transformer model configuration (1 or 2).

    Returns:
        argparse.ArgumentParser: An argument parser with options for configuring the Transformer model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cp", "--config_path", type=str,
                        default="config/decomposing_transformer_v{}.json".format(version),
                        help="path of configuration file")
    parser.add_argument("-hs", "--hidden", type=int, default=192, help="hidden size of transformer model")
    parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="dropout of the model")
    parser.add_argument("-el", "--encoder_layers", type=int, default=10, help="number of encoder layers")
    parser.add_argument("-dl", "--decoder_layers", type=int, default=10, help="number of decoder layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=12420, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=32, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate of the optimiser")
    parser.add_argument("-lrd", "--lr_decay", type=float, default=0.9, help="learning rate decay of the scheduler")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=4, help="batches for gradient accumulation")
    parser.add_argument("-st", "--scheduler_steps", type=int, default=400, help="scheduler steps in one cycle")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-2, help="weight_decay of the optimiser")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--eps", type=float, default=1e-9, help="value to prevent division by zero")
    parser.add_argument('-ws', '--warmup_steps', type=int, default=20, help='Maximum number of warmup steps')

    return parser


def args_gan():
    """
    Create an argument parser for GAN (generative adversarial network) model configuration.

    Returns:
        argparse.ArgumentParser: An argument parser with options for configuring the GAN model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cp", "--config_path", type=str, default="config/gan_p_conf1.json",
                        help="path of configuration file")
    parser.add_argument("-is", "--input_size", type=int, default=1, help="input dimension (per cube)")
    parser.add_argument("-ls", "--label_size", type=int, default=6, help="number of labels (kinematic parameters)")
    parser.add_argument("-ns", "--noise_size", type=int, default=512, help="size of the noise")
    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="dropout of the model")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of the optimiser")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="weight_decay of the optimiser")
    parser.add_argument('-cr', '--crit_repeats', type=int, default=5, help='Critic iterations per generator')
    parser.add_argument('-lgp', '--lambda_gp', type=int, default=10, help='Lambda value for gradient penalty')

    return parser

