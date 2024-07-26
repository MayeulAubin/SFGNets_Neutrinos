"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Classes for the conditional Wasserstein GAN with gradient penalty (cWGAN-GP).
"""

import torch
import torch.nn as nn
from .bert import BERT


class Generator(nn.Module):

    def __init__(self, input_size, label_size, noise_size, hidden,
                 n_layers, attn_heads, dropout):
        """
        BERT-based Generator for cWGAN-GP.

        Args:
            input_size (int): The size of the input data.
            label_size (int): The size of label information.
            noise_size (int): The size of noise vector.
            hidden (int): The hidden size of the BERT model.
            n_layers (int): The number of transformer layers in BERT.
            attn_heads (int): The number of attention heads in the multi-head self-attention mechanism.
            dropout (float): The dropout probability for regularization.
        """
        super().__init__()
        self.bert = BERT(input_size=input_size,
                         label_size=label_size,
                         noise_size=noise_size,
                         hidden=hidden,
                         n_layers=n_layers,
                         attn_heads=attn_heads,
                         dropout=dropout)
        self.decoder = Decoder(self.bert.hidden, 1, activation=True)

    def _init_weights(self):
        """
        Initialise the model weights
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, label, noise):
        """
        Forward pass of the Generator.

        Args:
            label (Tensor): The kinematic parameters tensor.
            noise (Tensor): The noise tensor.

        Returns:
            Tensor: The generated synthetic particle image.
        """
        x = self.bert(input=None, label=label, noise=noise)
        return self.decoder(x).view(x.shape[0], -1)


class Critic(nn.Module):

    def __init__(self, input_size, label_size, noise_size, hidden,
                 n_layers, attn_heads, dropout):
        """
        BERT-based Critic for cWGAN-GP.

        Args:
            input_size (int): The size of the input data.
            label_size (int): The size of label information.
            noise_size (int): The size of noise vector.
            hidden (int): The hidden size of the BERT model.
            n_layers (int): The number of transformer layers in BERT.
            attn_heads (int): The number of attention heads in the multi-head self-attention mechanism.
            dropout (float): The dropout probability for regularization.
        """
        super().__init__()
        self.bert = BERT(input_size=input_size,
                         label_size=label_size,
                         noise_size=noise_size,
                         hidden=hidden,
                         n_layers=n_layers,
                         attn_heads=attn_heads,
                         dropout=dropout)
        self.decoder = Decoder(self.bert.hidden, 1, activation=False)

    def _init_weights(self):
        """
        Initialise the model weights
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, label, noise):
        """
        Forward pass of the Critic.

        Args:
            x (Tensor): The input image (hits).
            label (Tensor): The kinematic parameters tensor.
            noise (Tensor): The noise tensor.

        Returns:
            Tensor: The output tensor after passing through the Critic.
        """
        x = x.view(x.shape[0], -1, 1)
        x = self.bert(input=x, label=label, noise=noise)
        return self.decoder(x[:, 0]).squeeze()


class Decoder(nn.Module):

    def __init__(self, hidden, out_size, activation=False):
        """
        Decoder module for cWGAN-GP.
    
        Args:
            hidden (int): The output size of the BERT model.
            out_size (int): The size of the output.
            activation (str): The activation function to use (e.g., "sigmoid" or None).
        """
        super().__init__()
        self.linear = nn.Linear(hidden, out_size)
        if activation is not None:
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        """
        Forward pass of the Decoder.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the Decoder.
        """
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class WGAN_GP_Loss(nn.Module):
    def __init__(self, lambda_gp=10):
        """
        Wasserstein GAN with Gradient Penalty (WGAN-GP) loss for cWGAN-GP.

        Args:
            lambda_gp (float): The weight of the gradient penalty term.

        """
        super(WGAN_GP_Loss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, critic, real_output, fake_output, real_data, fake_data, real_labels, z):
        """
        Calculate the cWGAN-GP loss.

        Args:
            critic (nn.Module): The critic model.
            real_output (Tensor): The output of the critic for real data.
            fake_output (Tensor): The output of the critic for fake data.
            real_data (Tensor): The real data samples.
            fake_data (Tensor): The generated fake data samples.
            real_labels (Tensor): The real data labels.
            z (Tensor): The random noise input for generating data samples.

        Returns:
            Tensor: The cWGAN-GP loss value, real loss, fake loss.
        """
        # Compute the Wasserstein distance between real and fake data
        real_loss = -torch.mean(real_output)
        fake_loss = torch.mean(fake_output)
        wd = real_loss + fake_loss

        # Compute the gradient penalty
        alpha = torch.rand(real_data.size(0), 1)
        alpha = alpha.expand(real_data.size()).to(real_output.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        interpolated_output = critic(interpolated, real_labels, z)
        grad = torch.autograd.grad(outputs=interpolated_output, inputs=interpolated,
                                   grad_outputs=torch.ones_like(interpolated_output),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        grad_penalty = self.lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()

        # Return the final loss
        return wd + grad_penalty, real_loss, fake_loss
