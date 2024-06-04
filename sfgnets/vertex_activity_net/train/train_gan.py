"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Training script for the proton GAN considering the first transformer configuration.
"""

import os
import json
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import GANDataset
from models import Generator, Critic, WGAN_GP_Loss, LightningModelGAN
from utils import args_gan
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    # Manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Arguments
    parser = args_gan()
    args = parser.parse_args()

    # Configuration file
    with open(args.config_path) as config_file:
        config = json.load(config_file)

    # Training set and loader
    train_set = GANDataset(config=config, split="train")
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)

    # Geneator and critic models
    generator = Generator(input_size=args.input_size, label_size=args.label_size, noise_size=args.noise_size,
                          hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
    critic = Critic(input_size=args.input_size, label_size=args.label_size, noise_size=args.noise_size,
                    hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
    generator._init_weights()
    critic._init_weights()
    gen_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    cri_total_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(generator)
    print(critic)
    print("Total trainable params: {} (generator), {} (discriminator).".format(gen_total_params, cri_total_params))

    # Loss function (for critic)
    adv_loss = WGAN_GP_Loss(args.lambda_gp)

    # Create lightning model
    lightning_model = LightningModelGAN(generator=generator,
                                        critic=critic,
                                        noise_size=args.noise_size,
                                        crit_repeats=args.crit_repeats,
                                        adversarial_loss=adv_loss, lr=args.lr, wd=args.weight_decay)

    # Define logger and checkpoint
    logger = CSVLogger(save_dir="logs/", name=config["log_path"])
    checkpoint_callback = ModelCheckpoint(dirpath=config["save_path"], every_n_train_steps=5000)

    # Create trainer module
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        precision="bf16",
        devices=[0],
        logger=logger,
        log_every_n_steps=100,
        deterministic=True,
    )

    # Run the training
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
    )


if __name__ == "__main__":
    main()
