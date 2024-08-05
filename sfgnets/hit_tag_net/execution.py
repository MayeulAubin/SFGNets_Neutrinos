import torch
import tqdm
import time
import numpy as np
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from warmup_scheduler_pytorch import WarmUpScheduler
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import softmax

from .model import device
from .dataset import arrange_sparse_minkowski, arrange_truth, arrange_aux, collate_sparse_minkowski, split_dataset, SparseEvent





# Training function
def train(model:torch.nn.Module,
          loader:DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          loss_func:torch.nn.Module=nn.CrossEntropyLoss(),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,) -> float:
    """
    Runs the training loop once over the dataset.
    
    Parameters:
    - model: torch.nn.Module, the neural network model to train
    - loader: DataLoader, the training data loader that provides the data to train on
    - optimizer: torch.optim.Optimizer, the optimizer to use for training
    - warmup_scheduler: WarmUpScheduler, the scheduler to use for warming-up the learning rate in the firsts epochs
    - loss_func: torch.nn.Module, the loss function to use for training
    - device: torch.device, the device to run the training on
    - progress_bar: bool, whether to display a progress bar
    - benchmarking: bool, whether to benchmark the training loop by printing the execution times of each step
    - world_size: int, the number of processes participating in the training (for distributed training, multi GPU)
    
    Returns:
    - average_loss: float, the average loss over the training loop
    """
    
    
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




# Validation function
def test(model:torch.nn.Module,
          loader:DataLoader,
          optimizer:torch.optim.Optimizer|None=None,
          warmup_scheduler:WarmUpScheduler|None=None,
          loss_func:torch.nn.Module=nn.CrossEntropyLoss(),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,):
    """
    Runs the validation loop once over the dataset.
    
    Parameters:
    - model: torch.nn.Module, the neural network model to validate
    - loader: DataLoader, the validation data loader that provides the data to validate on
    - optimizer: torch.optim.Optimizer, the optimizer (not used, there so that the test and train function have the same parameters)
    - warmup_scheduler: WarmUpScheduler, the warm-up scheduler (not used, there so that the test and train function have the same parameters)
    - loss_func: torch.nn.Module, the loss function to use for validation
    - device: torch.device, the device to run the validation on
    - progress_bar: bool, whether to display a progress bar
    - benchmarking: bool, whether to benchmark the training loop by printing the execution times of each step
    - world_size: int, the number of processes participating in the validation (for distributed validation, multi GPU)
    
    Returns:
    - predictions: np.ndarray, the predictions of the model over the batch
    - targets: np.ndarray, the targets over the batch
    - average_loss: float, the average loss over the validation loop
    """
    
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


def training(device: torch.device,
            model:nn.Module,
            dataset:SparseEvent,
            optimizer:torch.optim.Optimizer,
            warmup_scheduler:WarmUpScheduler,
            loss_func:nn.Module,
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
            save_model_path:str|None=None,) -> dict[str,list[float]]:
    """
    Trains and validates a neural network for the Hit tagging task. Runs the epochs loop.
    
    Parameters:
    - device: torch.device, the device to run the training on
    - model: nn.Module, the neural network model to train and validate
    - dataset: SparseEvent, the dataset to train and validate on
    - optimizer: torch.optim.Optimizer, the optimizer to use for training
    - warmup_scheduler: WarmUpScheduler, the scheduler to use for warming-up the learning rate in the firsts epochs
    - loss_func: torch.nn.Module, the loss function to use for training and validation
    - batch_size: int, the batch size for training and validation
    - train_fraction: float, the fraction of the dataset to use for training
    - val_fraction: float, the fraction of the dataset to use for validation
    - seed: int, the random seed for reproducibility
    - epochs: int, the number of epochs to train (the epoch loop can stop before reaching this number)
    - stop_after_epochs: int, the maximum number of epochs without improvements of the validation loss before the training is stopped
    - progress_bar: bool, whether to display the epoch loop progress bar
    - sub_progress_bars: bool, whether to display the sub-progress bars for training and validation loops
    - benchmarking: bool, whether to benchmark the training and validation loops by printing the execution times of each step
    - multi_GPU: bool, whether to run the training on several GPUs (distributed training)
    - world_size: int, the number of processes participating (in case of multi_GPU)
    - save_model_path: str, the file path to save the model weights
    
    Returns:
    - training_loss: list[float], the average losses of each training loop of each epoch
    - validation_loss: list[float], the average losses of each validation loop of each epoch
    - precision: list[float], the macro average precision of the model on the validation data
    - recall: list[float], the macro average recall of the model on the validation data
    - f1_score: list[float], the macro average f1 score of the model on the validation data
    - learning_rate: list[float], the learning rate after each training loop
    """
        
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
    if save_model_path is not None:
        j=save_model_path.split("_")[-1].split(".")[0] # extract the #j div of the save path
    else:
        j=0
    
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
              device:torch.device=device) -> dict[str,list[np.ndarray]|None]:
    """
    Test a neural network model on a given test set for the Hit tagging task.
    
    Parameters:
    - loader: DataLoader, the test data loader
    - model: torch.nn.Module, the neural network model to be tested
    - progress_bar: bool, whether to display a progress bar
    - device: torch.device, the device on which to run the model
    
    Returns:
    - predictions: list[np.ndarray], the predictions of the model on the test data
    - f: list[np.ndarray], the features of the test data
    - coord: list[np.ndarray], the coordinates of the test data
    - y: list[np.ndarray], the targets of the test data
    - aux: list[np.ndarray]|None, the auxiliary variables of the test data
    """
    
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
