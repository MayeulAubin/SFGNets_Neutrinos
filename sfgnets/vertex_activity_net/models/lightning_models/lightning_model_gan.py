"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch Lightning model for the GAN.
"""

import torch
import pytorch_lightning as pl


class LightningModelGAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        critic,
        noise_size,
        crit_repeats,
        adversarial_loss,
        lr: float = 0.00005,
        wd: float = 0.,
    ):
        super().__init__()
        self.noise_size = noise_size  # size of the noise vector
        self.crit_repeats = crit_repeats  # critic iterations for each generator one
        self.lr = lr  # learning rate
        self.wd = wd  # weight decay
        self.automatic_optimization = False  # disable automatic optimisation
        self.adversarial_loss = adversarial_loss  # adversarial loss for critic
        self.generator = generator  # generator model
        self.critic = critic  # critic model

    def on_train_start(self):
        """
        Callback method called when the training starts.
        This method fixes the parameter groups for training.

        Note: This method is used to address an issue related to parameter groups in PyTorch Lightning.
        """
        self.optimizers()[0].param_groups = self.optimizers()[0]._optimizer.param_groups
        self.optimizers()[1].param_groups = self.optimizers()[1]._optimizer.param_groups

    def forward(self, labels, z):
        """
        Forward pass of the generator.

        Args:
            labels (Tensor): The input kinematic parameters.
            z (Tensor): The noise vector.

        Returns:
            Tensor: The generated fake images.
        """
        return self.generator(labels, z)

    def generator_train_step(self, optimizer_g, batch_size, real_data, labels):
        """
        Training step for the generator.

        Args:
            optimizer_g: The optimiser for the generator.
            batch_size (int): The batch size.
            real_data (Tensor): Real data samples.
            labels (Tensor): The input labels.
        """
        # generate fake images
        z = torch.normal(0, 1, size=(batch_size, 1, self.noise_size)).to(real_data.device)
        fake_data = self(labels, z)
        fake_output = self.critic(fake_data, labels, z)

        # loss
        g_loss = -torch.mean(fake_output)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()

        return

    def critic_train_step(self, optimizer_c, batch_size, real_data, labels):
        """
        Training step for the critic.

        Args:
            optimizer_c: The optimiser for the critic.
            batch_size (int): The batch size.
            real_data (Tensor): Real data samples.
            labels (Tensor): The input labels.

        Returns:
            float: Wasserstein GP loss.
            float: Critic loss on real data.
            float: Critic loss on fake data.
        """
        # Run critic on generated fake images
        z = torch.normal(0, 1, size=(batch_size, 1, self.noise_size)).to(real_data.device)
        fake_data = self(labels, z)
        fake_output = self.critic(fake_data.detach(), labels, z)

        # Run critic on real images
        real_output = self.critic(real_data, labels, z)

        # Wasserstein GP loss
        w_loss, c_loss_real, c_loss_fake = self.adversarial_loss(self.critic, real_output, fake_output,
                                                                 real_data, fake_data, labels, z)

        # Manual backward propagation
        self.manual_backward(w_loss)
        optimizer_c.step()

        return w_loss, c_loss_real, c_loss_fake

    def training_step(self, batch):
        """
        Training step for the model.

        Args:
            batch: The input batch.
        """
        real_data, labels = batch

        batch_size = real_data.size(0)
        optimizer_c, optimizer_g = self.optimizers()

        # Train critic
        self.toggle_optimizer(optimizer_c, 0)
        optimizer_c.zero_grad()
        mean_w_loss, mean_c_loss_real, mean_c_loss_fake = 0, 0, 0
        for _ in range(self.crit_repeats):
            w_loss, c_loss_real, c_loss_fake = self.critic_train_step(optimizer_c, batch_size, real_data, labels)
            mean_w_loss += w_loss.item() / self.crit_repeats
            mean_c_loss_real += c_loss_real.item() / self.crit_repeats
            mean_c_loss_fake += c_loss_fake.item() / self.crit_repeats
        self.log("w_loss", mean_w_loss, prog_bar=True)
        self.log("c_loss_real", mean_c_loss_real, prog_bar=True)
        self.log("c_loss_fake", mean_c_loss_fake, prog_bar=True)
        self.untoggle_optimizer(optimizer_c)

        # Train generator
        self.toggle_optimizer(optimizer_g, 1)
        optimizer_g.zero_grad()
        self.generator_train_step(optimizer_g, batch_size, real_data, labels)
        self.untoggle_optimizer(optimizer_g)

    def configure_optimizers(self):
        """
        Configure optimisers for the model.

        Returns:
            list: List of optimisers for the critic and generator.
            list: List of learning rate schedulers (empty in this case).
        """
        opt_c = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr, weight_decay=self.wd)
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr, weight_decay=self.wd)

        return [opt_c, opt_g], []
