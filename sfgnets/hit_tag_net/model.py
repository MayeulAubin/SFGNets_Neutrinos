import torch
from ..utils import minkunet
from warmup_scheduler_pytorch import WarmUpScheduler


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = minkunet.MinkUNet34B(in_channels=4, out_channels=3, D=3).to(device)

# Optimizer and scheduler
lr = 0.01 # learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
len_train_loader=160000 # value to set with a real loader
num_steps_one_cycle = 25 # period of the Cosine Annealing learning rate scheduler (in terms of epochs)
num_warmup_steps = 10 # number of warm up epochs for the learning rate
cosine_annealing_steps = len_train_loader * num_steps_one_cycle  # period of the Cosine Annealing learning rate scheduler (in terms of optimizer steps)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr/100)

warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                   len_loader=1,
                                   warmup_steps=len_train_loader * num_warmup_steps,
                                   warmup_start_lr=lr/100,
                                   warmup_mode='linear')