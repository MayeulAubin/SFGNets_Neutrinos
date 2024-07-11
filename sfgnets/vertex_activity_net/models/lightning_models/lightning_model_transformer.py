"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch Lightning model for the second configuration of the decomposing transformer.
"""

import pytorch_lightning as pl
import torch_optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ...utils import create_mask, CustomLambdaLR, CombinedScheduler


# Define the PyTorch Lightning model
class LightningModelTransformer(pl.LightningModule):
    def __init__(self, model, loss_fn1, loss_fn2, loss_fn3, loss_fn4, pad_value, warmup_steps,
                 cosine_annealing_steps, lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=0.01, lr_decay=0.9, eps=1e-9,
                 loss_weights:list[float]|None = None, log_weighted_loss:bool=False):
        """
        Initialise the Lightning Model for the first configuration of the decomposing transformer.

        Args:
            model: The transformer model.
            loss_fn1: Loss function 1 (vertex position).
            loss_fn2: Loss function 2 (kinematic parameters).
            loss_fn3: Loss function 3 (particle identification).
            loss_fn4: Loss function 4 (keep iterating).
            pad_value: Padding value for tokens.
            warmup_steps: Number of warmup steps for learning rate.
            cosine_annealing_steps: Number of steps for cosine annealing of learning rate.
            lr: Initial learning rate.
            beta1: Beta1 hyperparameter for the optimiser.
            beta2: Beta2 hyperparameter for the optimiser.
            weight_decay: Weight decay (L2 regularization) applied to the model parameters.
            lr_decay: Learning rate decay after each cycle of the cosine annealing scheduler.
        """
        super().__init__()

        self.model = model
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.loss_fn3 = loss_fn3
        self.loss_fn4 = loss_fn4
        self.pad_value = pad_value
        self.warmup_steps = warmup_steps
        self.cosine_annealing_steps = cosine_annealing_steps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.lr_decay = lr_decay
        
        if loss_weights is None:
            self.loss_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.loss_weights = loss_weights
        
        self.log_weighted_loss = log_weighted_loss
        

    def on_train_start(self):
        """
        Callback method called when the training starts.
        This method fixes the parameter groups for training.

        Note: This method is used to address an issue related to parameter groups in PyTorch Lightning.
        """
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def common_step(self, batch):
        """
        Common processing steps for training and validation.

        Args:
            batch: Batch of data.

        Returns:
            loss1: Loss for vertex position.
            loss2: Loss for kinematic parameters.
            loss3: Loss for PID.
            loss4: Loss for keep iterating.
            loss: Total loss.
            batch_size: Batch size.
            lr: Current learning rate.
        """
        # Unpack values from the batch
        hits, exit_muon, vtx_true, params_true, keep_iter_true, pid_true, _ = batch

        # Slice 'params_true' to exclude the last item of each sequence
        params_true_input = params_true[:-1, :]
        pid_true_input = pid_true[:-1, :]

        # Create masks for source and target sequences
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(hits, params_true_input, self.pad_value,
                                                                             self.device)

        # Pass the inputs and masks to the model to get predictions
        vtx_pred, params_pred, pid_pred, keep_iter_pred = self.model(hits, exit_muon, params_true_input,
                                                                     pid_true_input, src_mask, tgt_mask,
                                                                     src_padding_mask, tgt_padding_mask)

        # Mask out all the padding tokens
        padding_mask = (params_true[:, :, 0] != self.pad_value)
        params_true = params_true[padding_mask]
        params_pred = params_pred[padding_mask]
        pid_true = pid_true[padding_mask]
        pid_pred = pid_pred[padding_mask]
        keep_iter_true = keep_iter_true[padding_mask]
        keep_iter_pred = keep_iter_pred[padding_mask]

        # Calculate loss
        loss1 = self.loss_fn1(vtx_pred, vtx_true)
        loss2 = self.loss_fn2(params_pred, params_true)
        loss3 = self.loss_fn3(pid_pred, pid_true)
        loss4 = self.loss_fn4(keep_iter_pred, keep_iter_true)
        loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 + self.loss_weights[2]*loss3 + self.loss_weights[3]*loss4

        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        if self.log_weighted_loss:
            return self.loss_weights[0]*loss1, self.loss_weights[1]*loss2, self.loss_weights[2]*loss3, self.loss_weights[3]*loss4, loss, hits.shape[1], lr
        else: 
            return loss1, loss2, loss3, loss4, loss, hits.shape[1], lr

    def training_step(self, batch, batch_idx):
        """
        Training step for a batch of data.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the batch.

        Returns:
            loss: Training loss for the batch.
        """
        loss1, loss2, loss3, loss4, loss, batch_size, lr = self.common_step(batch)

        self.log("train_loss1", loss1.item(), batch_size=batch_size)
        self.log("train_loss2", loss2.item(), batch_size=batch_size)
        self.log("train_loss3", loss3.item(), batch_size=batch_size)
        self.log("train_loss4", loss4.item(), batch_size=batch_size)
        self.log("train_loss", loss.item(), batch_size=batch_size)
        self.log("lr", lr, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a batch of data.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the batch.

        Returns:
            loss: Validation loss for the batch.
        """
        loss1, loss2, loss3, loss4, loss, batch_size, lr = self.common_step(batch)

        self.log("val_loss1", loss1.item(), batch_size=batch_size)
        self.log("val_loss2", loss2.item(), batch_size=batch_size)
        self.log("val_loss3", loss3.item(), batch_size=batch_size)
        self.log("val_loss4", loss4.item(), batch_size=batch_size)
        self.log("val_loss", loss.item(), batch_size=batch_size)
        self.log("lr", lr, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        """
        Configure and initialise the optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and learning rate scheduler.
        """
        # Set the minimum learning rate for the learning rate schedule
        min_lr = self.lr / 1000

        # Create an optimizer using the LAMB (Layer-wise Adaptive Moments optimizer)
        optimizer = torch_optimizer.Lamb(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                         eps=self.eps, weight_decay=self.weight_decay)

        # Create a warm-up scheduler that gradually increases the learning rate
        warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)

        # Create a cosine annealing scheduler that reduces learning rate in a cosine-like manner
        cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.cosine_annealing_steps,
                                                       T_mult=2, eta_min=min_lr)

        # Create a combined scheduler that combines the warm-up and cosine annealing schedulers
        scheduler = CombinedScheduler(optimizer=optimizer,
                                      scheduler1=warmup_scheduler,
                                      scheduler2=cosine_scheduler,
                                      warmup_steps=self.warmup_steps,
                                      lr_decay=self.lr_decay)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def lr_scheduler_step(self, scheduler, *args):
        """
        Perform a learning rate scheduler step.

        Args:
            scheduler: Learning rate scheduler.
        """
        scheduler.step()
