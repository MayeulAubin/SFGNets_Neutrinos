import os
import re
import sys
import math
import random
import time
import pickle as pk
import torch.nn as nn
import tqdm
import pandas as pd
import numpy as np
import torch
import MinkowskiEngine as ME


from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, explained_variance_score, accuracy_score
from scipy.special import softmax

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sfgnets.dataset import *
from sfgnets.utils import minkunet
from warmup_scheduler_pytorch import WarmUpScheduler

from sfgnets.plotting import plots_ht as plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = minkunet.MinkUNet34B(in_channels=4, out_channels=3, D=3).to(device)

# Optimizer and scheduler
lr = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
len_train_loader=160000 # value to set with a real loader
num_steps_one_cycle = 25
num_warmup_steps = 10
cosine_annealing_steps = len_train_loader * num_steps_one_cycle
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr/100)

warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                   len_loader=1,
                                   warmup_steps=len_train_loader * num_warmup_steps,
                                   warmup_start_lr=lr/100,
                                   warmup_mode='linear')


def filtering_events(x):
    return x['c'] is not None

def collate_sparse_minkowski(batch):
    """
    Custom collate function for Sparse Minkowski network.

    Parameters:
    - batch: list, a list of dictionaries containing 'c' (coordinates), 'x' (features), and 'y' (labels) keys.

    Returns:
    - dict, a dictionary containing 'f' (features), 'c' (coordinates), and 'y' (labels) keys.
    """
    # Extract coordinates from the batch and convert to integer type
    coords = [d['c'].int() for d in filter(filtering_events,batch)]

    # Concatenate features from the batch
    feats = torch.cat([d['x'] for d in filter(filtering_events,batch)])

    # Concatenate labels from the batch
    y = torch.cat([d['y'] for d in filter(filtering_events,batch)])
    
    
    # Concatenate labels from the batch
    try:
        aux = torch.cat([d['aux'] for d in filter(filtering_events,batch)])
    except TypeError:
        aux = None

    # Create a dictionary to store the collated data
    ret = {'f': feats, 'c': coords, 'y': y, 'aux':aux}

    return ret


# Create and save a scaler for the features
def create_scalers(root,
                   scaler_file):
    features=[]
    for filen_name in glob(f'{root}/*.npz'):
        data = np.load(filen_name)
        
        # Extract raw data
        x_0 = data['x']  # HitTime, HitCharge
        
        if x_0.shape[0]!=0: # Checking if the event is empty
        
            c = data['c']  # 3D coordinates (cube raw positions)
            
            # True vertex position
            verPos = data['verPos']
            
            x=np.zeros(shape=(x_0.shape[0], 4))
            x[:,0]=x_0[:,1] # have to remove 'HitTime'
            # Add as features the distance to the vertex position
            x[:, -3] = c[:, 0] - verPos[0]
            x[:, -2] = c[:, 1] - verPos[1]
            x[:, -1] = c[:, 2] - verPos[2]
            
            features.append(x.copy()) # get the feature (here the Charge)
    features=np.vstack(features)
    
    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(features)

    scaler_stan = StandardScaler()
    scaler_stan.fit(features)

    with open(scaler_file, "wb") as fd:
        pk.dump([scaler_minmax, scaler_stan], fd)

    

# Training function
def train(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          loss_func:torch.nn.Module=nn.CrossEntropyLoss(),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,):
    
    
    model.train()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / (batch_size*world_size)))
    train_loop = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Train loop {device}", position=1, leave=False)
    
    time_load, time_arrange, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    sum_loss = 0.
    
    for i, data in train_loop:
        time_load+=time.perf_counter()-t0
        optimizer.zero_grad()
        
        t0=time.perf_counter()
        # Arrange input, output, and target
        batch_input = arrange_sparse_minkowski(data,device=device)
        batch_output = model(batch_input)
        batch_target = arrange_truth(data,device=device)
        # batch_target = torch.reshape(batch_target, (len(batch_target),)).to(device)
        time_arrange+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        # Compute loss and backpropagate
        batch_loss = loss_func(batch_output.F, batch_target.F[...,0])
        batch_loss.backward()
        
        
        # Update progress bar
        train_loop.set_postfix({"loss":  f"{batch_loss.item():.5f}"})
          
        sum_loss += batch_loss.item()
        optimizer.step()
        warmup_scheduler.step()
        time_steps+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        
    if benchmarking:
        print(f"Training: Loading: {time_load:.2f} s \t Arranging: {time_arrange:.2f} s \t Model steps: {time_steps:.2f} s \t Total: {time_load+time_arrange+time_steps:.2f} s")
        
    return sum_loss / n_batches




# Testing function
def test(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          loss_func:torch.nn.Module=nn.CrossEntropyLoss(),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,):
    
    model.eval()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / (batch_size*world_size)))
    test_loop = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test loop {device}", position=1, leave=False)
    
    sum_loss = 0.
    true = []
    pred = []
    
    
    time_load, time_arrange, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    for i, data in test_loop:
        time_load+=time.perf_counter()-t0
        optimizer.zero_grad()
        
        t0=time.perf_counter()
        # Arrange input, output, and target
        batch_input = arrange_sparse_minkowski(data,device=device)
        batch_output = model(batch_input)
        batch_target = arrange_truth(data,device=device)
        time_arrange+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        # Calculate predictions and store true labels
        pred += np.argmax(F.softmax(batch_output.F, dim=-1).cpu().detach().numpy(), axis=-1).tolist() # argmax solution
        # pred += torch.multinomial(F.softmax(batch_output.F, dim=-1), num_samples=1)[...,0].cpu().detach().numpy().tolist() # random choice solution
        true += batch_target.F.cpu().detach().numpy().tolist()
        
        # Compute loss
        batch_loss = loss_func(batch_output.F, batch_target.F[...,0])
        sum_loss += batch_loss.item()
        time_steps+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        
    if benchmarking:
        print(f"Validating: Loading: {time_load:.2f} s \t Arranging: {time_arrange:.2f} s \t Model test: {time_steps:.2f} s \t Total: {time_load+time_arrange+time_steps:.2f} s")
        
          
    return np.array(pred), np.array(true), sum_loss / n_batches


def ddp_setup(rank, world_size):
    """
    Initialise the DDP for multi_GPU application
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def training(device,
            model,
            dataset,
            optimizer,
            warmup_scheduler,
            loss_func,
            batch_size:int = 256,
            train_fraction:float=0.8,
            val_fraction:float=0.19,
            seed:int=7,
            epochs:int =200,
            stop_after_epochs: int = 30,
            progress_bar:bool =True,
            sub_progress_bars:bool = False,
            benchmarking:bool =False,
            multi_GPU:bool=False,
            world_size:int=1,
            save_model_path:str=None,):
        
    # creates the data loaders
    train_loader, valid_loader, test_loader=split_dataset(dataset,
                                                        batch_size = batch_size,
                                                        train_fraction=train_fraction,
                                                        val_fraction=val_fraction,
                                                        seed=seed,
                                                        multi_GPU=multi_GPU,
                                                        collate=collate_sparse_minkowski)
    
    LOSSES=[[],[]]
    METRICS=[[],[],[]]
    LR=[]
    max_val_acc = np.NINF
    epochs_since_last_improvement=0
    loss_func.to(device)
    
    # print("Starting training...")
    j=save_model_path.split("_")[-1].split(".")[0] # extract the #j div of the save path
    
    epoch_bar=tqdm.tqdm(range(0, epochs),
                            desc=f"Training Hit tag net {j}",
                            disable=(not progress_bar),
                            position= 0,
                            leave=True,)

    for epoch in epoch_bar:

        # Early stopping: finish training when validation results don't improve for 10 epochs
        if epochs_since_last_improvement >= stop_after_epochs:
            print("Early stopping: finishing....")
            break
        
        # If multi_GPU, we need to change the epoch of the samplers to shuffle
        if multi_GPU:
            train_loader.sampler.set_epoch(epoch)
            valid_loader.sampler.set_epoch(epoch)

        loss=0
        # Train
        t0=time.perf_counter()
        loss += train(model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    warmup_scheduler=warmup_scheduler,
                    loss_func=loss_func,
                    device=device, 
                    progress_bar=sub_progress_bars,
                    benchmarking=benchmarking,
                    world_size=world_size)
        
        if benchmarking:
            print(f"Effective train time: {time.perf_counter()-t0:.2f} s")
        
        LOSSES[0].append(loss)

        # Test on the validation set
        t0=time.perf_counter()
        val_pred, val_true, val_loss = test(model=model,
                                        loader=valid_loader,
                                        optimizer=optimizer,
                                        warmup_scheduler=warmup_scheduler,
                                        loss_func=loss_func,
                                        device=device, 
                                        progress_bar=sub_progress_bars,
                                        benchmarking=benchmarking,
                                        world_size=world_size)
        if benchmarking:
                print(f"Effective validation time: {time.perf_counter()-t0:.2f} s")
                
        LOSSES[1].append(val_loss)

        t0=time.perf_counter()
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            val_true, val_pred, average='macro')
        if benchmarking: 
            print(f"Metrics time: {time.perf_counter()-t0:.2f} s")
        
        METRICS[0].append(precision)
        METRICS[1].append(recall)
        METRICS[2].append(f1_score)

        lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        LR.append(lr)

        # print("Epoch: {0}, Train loss: {1:.4f}, lr: {2}, Prec: {3:.2f}, "
        #     "Reca: {4:.2f}, F1: {5:.2f}".format(epoch, loss, lr, precision, recall, f1_score))

        # Check results (check for improvement)
        if f1_score > max_val_acc:
            
            max_val_acc = f1_score
            
            # # Save model state dict
            # print('Saving model with val acc: {0:.3f}'.format(max_val_acc))
            # torch.save(model.state_dict(), "/scratch2/sfgd/model_best_jun2023_crossentropy")
            
            if save_model_path is not None and (not multi_GPU or device==0):
                torch.save(model.state_dict(), save_model_path)
            
            # reset the counting of staling to 0
            epochs_since_last_improvement = 0

            # # Print epoch results
            # target_names = ["multiple", "single", "other"]
            # print(classification_report(val_true, val_pred, digits=3, target_names=target_names))
            # print(confusion_matrix(val_pred, val_true))

        else:
            epochs_since_last_improvement += 1
            
        # Set postfix to print the current loss
        epoch_bar.set_postfix(
            {
                "Tloss": f"{LOSSES[0][-1]:.2e}",
                # "Vloss": f"{LOSSES[1][-1]:.2e}",
                "m5VL": f"{np.mean(LOSSES[1][-5:]):.2e}",
                "m5F1":f"{np.mean(METRICS[2][-5:])*100:.1f}",
                # "m5Prec": f"{np.mean(METRICS[0][-5:])*100:.1f}",
                # "m5Reca":f"{np.mean(METRICS[1][-5:])*100:.1f}",
                # "F1":f"{METRICS[2][-1]*100:.1f}",
                "Prec": f"{METRICS[0][-1]*100:.1f}",
                "Reca":f"{METRICS[1][-1]*100:.1f}",
                "lr":f"{lr:.1e}",
            }
        )
            
            
    return {"training_loss":LOSSES[0],
            "validation_loss":LOSSES[1],
            "precision":METRICS[0],
            "recall":METRICS[1],
            "f1_score":METRICS[2],
            "learning_rate":LR,}




# Full testing function
def test_full(loader:DataLoader,
              model:torch.nn.Module,
              progress_bar:bool=False,
              device=device):
    
    model.eval()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc="Testing Hit tag net")
    
    do_we_consider_aux=loader.dataset.aux
    
    sum_loss = 0.
    pred = []
    feat=[]
    coord=[]
    target=[]
    aux=None
    
    if do_we_consider_aux:
        aux=[]
    
    for i, data in t:
        # Arrange input
        batch_input = arrange_sparse_minkowski(data, device)
        batch_output = model(batch_input)
        batch_target = arrange_truth(data, device)
        
        if do_we_consider_aux:
            batch_aux = arrange_aux(data, device)
            
        # Get decomposed features
        outputs = batch_output.decomposed_features
        
        # Iterate over events and store predictions
        for event_output in outputs:
            event_output = event_output.cpu().detach().numpy()
            pred.append(softmax(event_output, axis=-1))
        
        # Iterate over events and store coordinates
        for co in batch_output.decomposed_coordinates:
            coord.append(co.cpu().detach().numpy())
            
        # Iterate over events and store features
        for input in batch_input.decomposed_features:
            feat.append(input.cpu().detach().numpy())
        
        # Iterate over events and store targets
        for targ in batch_target.decomposed_features:
            target.append(targ.cpu().detach().numpy())
        
        if do_we_consider_aux:
            for au in batch_aux.decomposed_features:
                aux.append(au.cpu().detach().numpy())
            
          
    return {'predictions':pred,'f':feat,'c':coord,'y':target, 'aux':aux}




    
    
    

# Function to arrange truth data (for the FGD 'old' dataset)
def arrange_truth_old(data):
    return data['y']    


# Dataset class for the FGD dataset ('old'), used in Saul's previous codes. DO NOT TOUCH IT!
class SparseEvent_old(Dataset):
    def __init__(self, root, shuffle=False, **kwargs):            
        '''
        Initializer for SparseEvent class.

        Parameters:
        - root: str, root directory
        - shuffle: bool, whether to shuffle data_files
        '''
        self.root = root
        self.data_files = self.processed_file_names
        if shuffle:
            random.shuffle(self.data_files)
        self.total_events = len(self.data_files)

        with open("/scratch2/sfgd/scalers.p", "rb") as fd:
            self.scaler_minmax, self.scaler_stan = pk.load(fd)
    
    @property
    def processed_dir(self):
        return f'{self.root}'

    @property
    def processed_file_names(self):
        return natural_sort(glob(f'{self.processed_dir}/*.npz'))
    
    def __len__(self):
        return self.total_events
    
    def __getitem__(self, idx):
        # Load data from the file
        data = np.load(self.data_files[idx])

        # Extract raw data
        x = data['x']  # HitTime, HitCharge, HitMPPCXY, HitMPPCXZ, HitMPPCYZ,
                       # HitNumberXY, HitNumberXZ, HitNumberYZ,
                       # HitAdjacent, HitDiagonal, HitCorner, HitDistance2Vertex
        c = data['c']  # 3D coordinates (cube raw positions)
        c_new = c.copy()

        # True vertex position
        verPos = data['verPos']

        # Convert cube raw positions to cubes
        c_new=transform_cube(c_new)

        # Remove time ('HitTime') and distance to vertex ('HitDistance2Vertex')
        x = x[:, 1:-1]

        # Add delta X, Y, Z (with respect to true vertex position)
        x_new = np.zeros(shape=(x.shape[0], x.shape[1] + 3))
        x_new[:, :-3] = x[:, :]
        x_new[:, -3] = c[:, 0] - verPos[0]
        x_new[:, -2] = c[:, 1] - verPos[1]
        x_new[:, -1] = c[:, 2] - verPos[2]

        # Standardize dataset
        x = self.scaler_minmax.transform(x_new)

        # Keep only HitCharge and deltas
        x = x[:, [0, -3, -2, -1]]

        # Create tensors
        x = torch.FloatTensor(x)
        y = torch.LongTensor(data['y'] - 1)
        c = torch.FloatTensor(c_new)

        # Clean up and return the data
        del data
        return {'x': x, 'c': c, 'y': y}





class SparseEvent_for_noise_filtering(SparseEvent):
    
    def gety(self,data):
        # We set the Vertex Activity (-1) and Single P (0) to 0 (no noise), while Noise (1) to 1
        return torch.clamp(torch.LongTensor(data['y'] - 2),0,None)
    
    
class HitTagNetTwo(torch.nn.Module):
    
    def __init__(self, modelA, modelB, modelA_already_trained=True):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelA_already_trained = modelA_already_trained
    
    
    def forward(self, x):
        if self.modelA_already_trained:
            # if the modelA is already trained, just use it without computing gradients
            self.modelA.eval()
        
        # modelA predicts if the hit is Noise (1) or not (0)    
        y_noise_filtering=ME.MinkowskiFunctional.softmax(self.modelA(x),dim=-1)
        
        # We get the indices were the hits are noise
        is_the_point_noise=torch.argmax(y_noise_filtering.F,dim=-1,keepdim=False).to(bool) # argmax solution
        # is_the_point_noise=torch.multinomial(y_noise_filtering.F, num_samples=1)[...,0].to(bool) # random choice solution
        
        # We select only hits that are not noise
        x_new=ME.SparseTensor(features=x.F[~is_the_point_noise],coordinates=x.C[~is_the_point_noise])
        
        # modelB predicts if the hit is Vertex Activity (0) or Single P (1)    
        y_VA=ME.MinkowskiFunctional.softmax(self.modelB(x_new),dim=-1)
        
        # Then we reconstruct the final output:
        y_F=torch.zeros(list(y_noise_filtering.shape[:-1])+[3]).to(device)
        y_F[~is_the_point_noise,0]=y_VA.F[...,0]*y_noise_filtering.F[~is_the_point_noise,0] # if the hit is not noise, we set the probability of VA to VA*(not noise)
        y_F[~is_the_point_noise,1]=y_VA.F[...,1]*y_noise_filtering.F[~is_the_point_noise,0] # if the hit is not noise, we set the probability of Single P to (not VA)*(not noise)
        y_F[is_the_point_noise,1]=y_noise_filtering.F[is_the_point_noise,0] # if the hit is noise, we set the probability of VA to 0 and of Single P to (not noise)
        y_F[...,2]=y_noise_filtering.F[...,1] # we set the output probability of noise to the predicted probability of noise by modelA
        
        # All sparse tensors have the same coordinates, so we can construct the output easily
        y=ME.SparseTensor(features=y_F,coordinates=x.C)
        return y
    
    
    

def dice_loss_fn_multiclass(pred, target, num_classes, eps=1e-5):
    pred = F.softmax(pred, dim=1)
    
    dice_loss = 0.0
    for class_idx in range(num_classes):
        class_true = (target == class_idx).float()
        class_pred = pred[:, class_idx, ...]
        
        intersection = torch.sum(class_true * class_pred)
        union = torch.sum(class_true) + torch.sum(class_pred)
        
        class_dice = (2.0 * intersection + eps) / (union + eps)
        dice_loss += 1.0 - class_dice
    
    return dice_loss / num_classes  # average over classes



class DiceLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        
    def forward(self, pred, target):
        return dice_loss_fn_multiclass(pred, target, self.num_classes, self.eps)
    
    
FocalLoss = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=None,
                gamma=2,
                reduction='mean',
                force_reload=False,
            )

class SumLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses=torch.nn.ModuleList(losses)
        self.weights=weights
        if weights is None:
            self.weights=[1. for _ in self.losses]
    
    def forward(self, pred, target):
        loss=0.0
        for l,w in zip(self.losses,self.weights):
            loss+=w*l(pred,target)
        return loss/sum(self.weights)
    
    

    
    
    
    
    
    
    
################ SCRIPT FOR TRAINING ####################
## This is copied and cleaned from hit_tag_training_script.py
## it allows to run hit_tag_neet as a script directly


def main_worker(device,
                dataset,
                args,
                world_size=1,
                multi_GPU=False):

    #### Get the variables from args
    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    ## Get the multi_GPU flag
    multi_GPU=args.multi_GPU

    ## Get the benchmarking flag
    benchmarking=args.benchmarking
    sub_progress_bars=args.sub_tqdm

    multi_pass=args.multi_pass
            
    
    print(f"Starting main worker on device {device}")
    
    # if we are in a multi_GPU training setup, we need to initialise the DDP (Distributed Data Parallel)
    if multi_GPU:
        ddp_setup(rank=device, world_size=world_size)
    
    
    model = minkunet.MinkUNet34B(in_channels=4, out_channels=3, D=3).to(device)
    print(model.__str__().split('\n')[0][:-1]) # Print the model name
    
    if multi_GPU:
        model=torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[device])
        # For multi-GPU training, DDP requires to change BatchNorm layers to SyncBatchNorm layers
        model=ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    # Print the total number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {pytorch_total_params}")
        
    # Loss function
    loss_func= SumLoss([FocalLoss, DiceLoss(num_classes=3)])

    # Loaders parameters
    batch_size=args.batch_size
    train_fraction=0.9
    val_fraction=0.099
    seed=7
    len_train_loader=int(len(dataset)*train_fraction/(multi_pass*batch_size))

    # Optimizer and scheduler
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
    num_steps_one_cycle = 25
    num_warmup_steps = 10
    cosine_annealing_steps = len_train_loader * num_steps_one_cycle
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr*(1e-2))
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                    len_loader=1,
                                    warmup_steps=len_train_loader * num_warmup_steps,
                                    warmup_start_lr=lr/100,
                                    warmup_mode='linear')

    num_epochs=args.num_epochs
    stop_after_epochs=args.stop_after_epochs
    
    training_dict=training(device=device,
                            model=model,
                            dataset=dataset,
                            optimizer=optimizer,
                            warmup_scheduler=warmup_scheduler,
                            loss_func=loss_func,
                            batch_size=batch_size,
                            train_fraction=train_fraction,
                            val_fraction=val_fraction,
                            seed=seed,
                            epochs=num_epochs,
                            stop_after_epochs= stop_after_epochs,
                            progress_bar=True,
                            benchmarking=benchmarking,
                            sub_progress_bars=sub_progress_bars,
                            multi_GPU=multi_GPU,
                            world_size=world_size,
                            save_model_path=f"{args.save_path}models/hittag_model_{j}.torch")
    
    if multi_GPU:
        torch.distributed.destroy_process_group()
    
    return model, training_dict



def _test_model(device,
                dataset,
                args,
                model):
    
    full_loader=full_dataset(dataset,
                                     collate=collate_sparse_minkowski)
    all_results = test_full(full_loader,
                            model,
                            progress_bar=True,
                            )

    print("Saving results...")
    with open(f'{args.save_path}results/hit_tagging_model_{args.j}_pred.pkl', 'wb') as file:
        pk.dump(all_results,file)
    
    print("Flattening and extracting...")
    ## Get a flat array of true and predicted labels
    target_names = ["Vertex activity", "Single P", "Noise"]
    val_true=np.vstack(all_results['y']).flatten()

    y_pred=np.vstack(all_results['predictions'])
    val_pred=np.argmax(y_pred,axis=-1) # argmax solution
    # val_pred=torch.multinomial(torch.Tensor(y_pred).to(device),num_samples=1)[...,0].cpu().numpy() # random choice solution
    
    print("Printing classification report...")
    print(classification_report(val_true, val_pred, digits=3, target_names=target_names))
    
    
    
    




if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
                        prog='HitTagTraining',
                        description='Trains a model for Hit Tagging in SFG',)

    parser.add_argument('j', metavar='j', type=int, help='#j div of the model')
    parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for training")
    parser.add_argument('scaler_file',metavar='Scaler_File', type=str, help="File storing the dataset features scalers")
    parser.add_argument('save_path',metavar='Save_Path', type=str, help="Path to save results and models")
    parser.add_argument('-T', '--test', action='store_true', help='runs test after training (classification report,...)')
    parser.add_argument('--test_only', action='store_true', help='runs only the test (classification report,...)')
    parser.add_argument('-m', '--multi_GPU', action='store_true', help='runs the script on multi GPU')
    parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the duration of the different parts of the code')
    parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('-n','--num_epochs', type=int, default=200, help='number of epochs for the training')
    parser.add_argument('--multi_pass', type=int, default=1, help='how many times the whole dataset is gone through in an epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='maximum learning rate for training (defines the scale of the learning rate)')
    parser.add_argument('--stop_after_epochs', type=int, default=40, help='maximum number of epochs without improvement before stopping the training (early termination)')
    parser.add_argument('-f', '--filter', action='store_true', help='filter the hits to keep only the ones close to the vertex, use --cut to control the distance of the cut')
    parser.add_argument('--cut', type=float, default=8., help='when filtering, maximum distance of the hits to the vertex to be kept')
    parser.add_argument('-r', '--retag', action='store_true', help='retag the hit tags according to the retag_cut function, use --rcut to control the distance of the cut')
    parser.add_argument('--rcut', type=float, default=40., help='when filtering, maximum distance of the hits to the vertex to be kept')
    args = parser.parse_args()

    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    ## Get the multi_GPU flag
    multi_GPU=args.multi_GPU

    ## Get the benchmarking flag
    benchmarking=args.benchmarking
    sub_progress_bars=args.sub_tqdm

    multi_pass=args.multi_pass

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    torch.multiprocessing.set_sharing_strategy('file_system')
    
    
    print(args.dataset_folder[:-6]+"test/")



    def main():
        global multi_GPU, benchmarking
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if (not args.test_only):
        
            if multi_GPU:
                print(f"Training on Multi GPU...")
            else:
                print(f"Training on Single GPU {device}...")
            if benchmarking:
                print("Benchmarking...")
            if multi_pass!=1:
                print(f"Multi pass {multi_pass}...")

            # generate dataset
            dataset=SparseEvent(args.dataset_folder,
                                scaler_file=args.scaler_file,
                                multi_pass=multi_pass,
                                filtering_func=select_hits_near_vertex(cut=args.cut,dist_type="cube") if args.filter else None,
                                center_event=args.filter,
                                retagging_func=retag_cut(args.rcut) if args.retag else None) 

            t0=time.perf_counter()

            if multi_GPU:
                world_size = torch.cuda.device_count()
                model, training_dict=torch.multiprocessing.spawn(main_worker, args=(
                                                                        dataset,
                                                                        args,
                                                                        world_size,
                                                                        multi_GPU), nprocs=world_size)
                
            else:
                model, training_dict=main_worker(device,
                                    dataset,
                                    args,)
                
            t0=t0-time.perf_counter()

            torch.save(training_dict,f"{args.save_path}results/training_dict_{j}.torch")    

        ## Runs the tests
        if args.test or args.test_only:
            testdataset_folder=args.dataset_folder[:-6]+"test/" ## we are assuming that the dataset_folder is of type "*_train/" whereas the testdataset folder will be "*_test/"
            testdataset=SparseEvent(testdataset_folder,
                            scaler_file=args.scaler_file,
                            multi_pass=multi_pass,
                            filtering_func=select_hits_near_vertex(cut=args.cut,dist_type="cube") if args.filter else None,
                            center_event=args.filter,
                            retagging_func=retag_cut(args.rcut) if args.retag else None) 
            
            model = minkunet.MinkUNet34B(in_channels=4, out_channels=3, D=3).to(device)
            model.load_state_dict(torch.load(f"{args.save_path}models/hittag_model_{j}.torch"))
            
            _test_model(device=device,
                        dataset=testdataset,
                        args=args,
                        model=model)
    
        
    main()

