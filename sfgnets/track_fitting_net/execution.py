import torch
import tqdm
import time
import numpy as np
import math
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from warmup_scheduler_pytorch import WarmUpScheduler
from torch import Tensor

from .model import device
from .dataset import PGunEvent
from .dataset import arrange_sparse_minkowski, arrange_truth, arrange_aux, collate_minkowski, collate_transformer, split_dataset, create_mask_src, transform_inverse_cube, transform_cube
from .losses import SumLoss, SumPerf




#################### TRAINING FUNCTIONS #####################    

def execute_model(model:torch.nn.Module,
                data:dict,
                model_type:str='minkowski', # or 'transformer'
                device:torch.device=device,
                do_we_consider_aux:bool=False,
                do_we_consider_coord:bool=False,
                do_we_consider_feat:bool=False,
                do_we_consider_event_id:bool=False,
                last_event_id:int|None=None,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """
    Execute the model on some data.
    Adapts whether it is 'minkowski' type (SparseTensors) or 'transformer' type (PaddedSequence).
    Many options to retrieve the useful information from the data.
    
    Parameters:
    - model: torch.nn.Module, model to use
    - data: dict, input data to the model, comes from a DataLoader with a PGunEvent Dataset and collate_transformer or collate_minkowski function.
    - model_type: str, type of the model ('minkowski' or 'transformer')
    - device: torch.device, device to run the model on
    - do_we_consider_aux: bool, whether to consider auxiliary variables in the output
    - do_we_consider_coord: bool, whether to consider coordinates in the output
    - do_we_consider_feat: bool, whether to consider features in the output
    - do_we_consider_event_id: bool, whether to consider event id in the output
    - last_event_id: int, the starting point for counting events
    
    Return:
    - pred:  torch.Tensor, predicted values
    - targ:  torch.Tensor, target values
    - mask:  torch.Tensor, mask for data (noise hits, secondary trajectories, ...)
    - aux:   torch.Tensor, auxiliary variables
    - coord: torch.Tensor, coordinates
    - feats: torch.Tensor, features
    - event_id: torch.Tensor, event id
    """
    
    if data['f'] is not None:
        aux=None
        coord=None
        feats=None
        event_id=None
        
        if model_type=='minkowski':
            # arranging the data to be into Sparse Minkowski Tensors
            data={'f':arrange_sparse_minkowski(data,device),'y':arrange_truth(data,device),'aux':arrange_aux(data,device),'c':data['c'], 'mask':arrange_mask(data,device)}
            # run the model
            batch_output=model(data['f'])
            # flatten the data
            pred=batch_output.F
            targ=data['y'].F
            # masking the data (noise hits)
            mask=data['mask'].F.float()
            targ=targ*mask
            pred=pred*mask
            # if necessary extract the auxiliary variables
            if do_we_consider_aux:
                aux=data['aux'].F
            # if necessary extract the coordinates
            if do_we_consider_coord:
                coord=data['f'].C[:,1:] # the first index is the batch index, which we remove
            # if necessary extract the features
            if do_we_consider_feat:
                feats=data['f'].F
            # if necessary extract the event id
            if do_we_consider_event_id:
                event_id=data['f'].C[:,[0]]+last_event_id
                
        elif model_type=='transformer':
            features=data['f'].to(device)
            # create masks
            src_mask, src_padding_mask = create_mask_src(features)
            # run model
            batch_output = model(features, src_mask, src_padding_mask)
            # masking the data (noise hits)
            mask=data['mask'].to(device).float()[...,None] # adds an extra dimension to match that of the features/targets
            targ=data['y'].to(device)*mask
            pred=batch_output*mask
            # packing the data
            pred = torch.nn.utils.rnn.pack_padded_sequence(pred, data['lens'], batch_first=False, enforce_sorted=False)
            targ = torch.nn.utils.rnn.pack_padded_sequence(targ, data['lens'], batch_first=False, enforce_sorted=False)
            # flattening the data
            pred=pred.data
            targ=targ.data
            # if necessary extract the auxiliary variables
            if do_we_consider_aux:
                aux=torch.nn.utils.rnn.pack_padded_sequence(data['aux'], data['lens'], batch_first=False, enforce_sorted=False).data
            # if necessary extract the coordinates
            if do_we_consider_coord:
                coord=torch.nn.utils.rnn.pack_padded_sequence(data['c'], data['lens'], batch_first=False, enforce_sorted=False).data
            # if necessary extract the features
            if do_we_consider_feat:
                feats=torch.nn.utils.rnn.pack_padded_sequence(features, data['lens'], batch_first=False, enforce_sorted=False).data
            # if necessary extract the event id
            if do_we_consider_event_id:
                event_id=torch.nn.utils.rnn.pack_padded_sequence(torch.ones(features.shape[:-1])[...,None]*torch.arange(features.shape[1])[None,:,None]+last_event_id, data['lens'], batch_first=False, enforce_sorted=False).data
            # change the mask data to fit the format of pred, targs, ...
            mask=torch.nn.utils.rnn.pack_padded_sequence(mask, data['lens'], batch_first=False, enforce_sorted=False).data
            
        else:
            raise ValueError(f"Wrong model type {model_type}")
        
        return pred, targ, mask, aux, coord, feats, event_id
    
    else:
        return Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        



# Training function
def train(model:torch.nn.Module,
          loader:DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          model_type:str='minkowski', # or 'transformer'
          loss_func:SumLoss=SumLoss([torch.nn.MSELoss()]),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,
          notebook_tqdm:bool=False,):
    """
    Trains the model for one epoch.
    
    Parameters:
    - model: torch.nn.Module, model to train
    - loader: DataLoader, data loader for training
    - optimizer: torch.optim.Optimizer, optimizer to use for training
    - warmup_scheduler: WarmUpScheduler, scheduler for learning rate warmup
    - model_type: str, type of the model ('minkowski' or 'transformer')
    - loss_func: SumLoss, loss function to use for training (it has to be a SumLoss, that is a loss with components to track)
    - device: torch.device, device to run the model on
    - progress_bar: bool, whether to show or not the tqdm progress bar for this epoch
    - benchmarking: bool, whether to benchmark the training loop (show computation time)
    - world_size: int, the number of GPUs used
    - notebook_tqdm: bool, whether to use tqdm progress bar for a Jupyter notebook
    
    Returns:
    - loss: float, average training loss over the epoch
    - comp_loss: np.ndarray, array of training loss components, averaged over the epoch
    """
    
    
    model.train()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / (batch_size*world_size)))
    train_loop = tqdm.notebook.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Train loop {device}", position=1, leave=False) if notebook_tqdm else tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Train loop {device}", position=1, leave=False)
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    sum_loss = 0.
    comp_loss=np.zeros(len(loss_func))
    
    for i, data in train_loop:
        time_load+=time.perf_counter()-t0
        optimizer.zero_grad()
        
        t0=time.perf_counter()
        
        pred,targ, _m, _a, _c, _f, _e = execute_model(model=model,
                                data=data,
                                model_type=model_type,
                                device=device)
        
        if pred.numel()>0:
        
            time_model+=time.perf_counter()-t0
            t0=time.perf_counter()
            
            loss,loss_composition=loss_func(pred,targ)
            loss.backward()
            
            # Update progress bar
            train_loop.set_postfix({"loss":  f"{loss.item():.5f}","lr":f"{optimizer.param_groups[0]['lr']:.2e}"})
            
            sum_loss += loss.item()
            comp_loss += loss_composition
            
            optimizer.step()
            warmup_scheduler.step()
            time_steps+=time.perf_counter()-t0
            
            t0=time.perf_counter()
        
    if benchmarking:
        print(f"Training: Loading: {time_load:.2f} s \t Arranging: {time_model:.2f} s \t Model steps: {time_steps:.2f} s \t Total: {time_load+time_model+time_steps:.2f} s")
        
    return sum_loss / n_batches, comp_loss / n_batches




# Validation function
def test(model:torch.nn.Module,
          loader:DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          model_type:str='minkowski', # or 'transformer'
          loss_func:SumLoss=SumLoss([torch.nn.MSELoss()]),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,
          notebook_tqdm:bool=False,):
    """
    Validates the model over an epoch.
    
    Parameters:
    - model: torch.nn.Module, model to validate
    - loader: DataLoader, data loader for validation
    - optimizer: torch.optim.Optimizer, optimizer (not used)
    - warmup_scheduler: WarmUpScheduler, scheduler (not used)
    - model_type: str, type of the model ('minkowski' or 'transformer')
    - loss_func: SumLoss, loss function to use for training (it has to be a SumLoss, that is a loss with components to track)
    - device: torch.device, device to run the model on
    - progress_bar: bool, whether to show or not the tqdm progress bar for this epoch
    - benchmarking: bool, whether to benchmark the training loop (show computation time)
    - world_size: int, the number of GPUs used
    - notebook_tqdm: bool, whether to use tqdm progress bar for a Jupyter notebook
    
    Returns:
    - predictions: np.ndarray, predictions
    - true_targets: np.ndarray, true targets
    - loss: float, average validation loss over the epoch
    - comp_loss: np.ndarray, array of validation loss components, averaged over the epoch
    """
    
    model.eval()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / (batch_size*world_size)))
    test_loop = tqdm.notebook.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test loop {device}", position=1, leave=False) if notebook_tqdm else tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test loop {device}", position=1, leave=False)
    
    sum_loss = 0.
    comp_loss=np.zeros(len(loss_func))
    true_targets = []
    predictions = []
    
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    for i, data in test_loop:
        time_load+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        
        pred,targ, _m, _a, _c, _f, _e = execute_model(model=model,
                                data=data,
                                model_type=model_type,
                                device=device)
        
        if pred.numel()>0:
            
            time_model+=time.perf_counter()-t0
            t0=time.perf_counter()
            
            loss,loss_composition=loss_func(pred,targ)
            sum_loss += loss.item()
            comp_loss += loss_composition
            
            # Update progress bar
            test_loop.set_postfix({"loss":  f"{loss.item():.5f}"})
            
            true_targets+=targ.tolist()
            predictions+=pred.tolist()
            
            time_steps+=time.perf_counter()-t0
            
            t0=time.perf_counter()
        
    if benchmarking:
        print(f"Validating: Loading: {time_load:.2f} s \t Model run: {time_model:.2f} s \t Loss computation: {time_steps:.2f} s \t Total: {time_load+time_model+time_steps:.2f} s")
        
          
    return np.array(predictions), np.array(true_targets), sum_loss / n_batches, comp_loss / n_batches 


def ddp_setup(rank, world_size):
    """
    Initialise the DDP for multi_GPU application
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    

# Full testing function
def test_full(model:torch.nn.Module,
            loader:DataLoader,
            model_type:str='minkowski', # or 'transformer'
            device:torch.device=device, 
            progress_bar:bool=False,
            do_we_consider_aux:bool=False,
            do_we_consider_coord:bool=False,
            do_we_consider_feat:bool=True,
            do_we_consider_event_id:bool=True,
            max_batches:int|None=None,) -> dict[str, list[np.ndarray]|None]:
    """
    Tests the model over the test dataset.
    
    Parameters:
    - model: torch.nn.Module, model to test
    - loader: DataLoader, data loader for testing
    - model_type: str, type of the model ('minkowski' or 'transformer')
    - device: torch.device, device to run the model on
    - progress_bar: bool, whether to show or not the tqdm progress bar for testing
    - do_we_consider_aux: bool, whether to consider auxiliary variables
    - do_we_consider_coord: bool, whether to consider coordinate 
    - do_we_consider_feat: bool, whether to consider features
    - do_we_consider_event_id: bool, whether to consider event IDs
    - max_batches: int|None, maximum number of batches to test, if None, all batches will be tested
    
    Returns:
    - results: dict[str, list[np.ndarray]], dictionary with test results, each value is a list of arrays, one per event
    
    """
    
    model.eval()
    
    batch_size = loader.batch_size
    
    if max_batches is not None:
        n_batches=max_batches
    else:
       n_batches = int(math.ceil(len(loader.dataset) / batch_size))
       
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc="Testing Track fitting net")
    
    
    sum_loss = 0.
    last_event_id=0
    pred = []
    feat=[] if do_we_consider_feat else None
    coord=[] if do_we_consider_coord else None
    target=[]
    aux=[] if do_we_consider_aux else None
    mask=[]
    event_id=[] if do_we_consider_event_id else None
    
    for i, data in t:
        
        _pred,_targ, _mask, _aux, _coord, _feat, _event_id = execute_model(model=model,
                                                    data=data,
                                                    model_type=model_type,
                                                    device=device,
                                                    do_we_consider_aux=do_we_consider_aux,
                                                    do_we_consider_coord=do_we_consider_coord,
                                                    do_we_consider_feat=do_we_consider_feat,
                                                    do_we_consider_event_id=do_we_consider_event_id,
                                                    last_event_id=last_event_id,
                                                    )
        
        if _pred.numel()>0:
            
            pred.append(_pred.cpu().detach().numpy())
            target.append(_targ.cpu().detach().numpy())
            mask.append(_mask.cpu().detach().numpy())
            if do_we_consider_aux:
                aux.append(_aux.cpu().detach().numpy())
            if do_we_consider_coord:
                coord.append(_coord.cpu().detach().numpy())
            if do_we_consider_feat:
                feat.append(_feat.cpu().detach().numpy())
            if do_we_consider_event_id:
                event_id.append(_event_id.cpu().detach().numpy())
                last_event_id=event_id[-1].max()
            
            if i>n_batches:
                break
        
    
        
    torch.cuda.empty_cache() # release the GPU memory      
    return {'predictions':pred,'f':feat,'c':coord,'y':target, 'aux':aux, 'mask':mask, 'event_id':event_id}
    


def measure_performances(results_from_test_full:dict,
                        dataset:PGunEvent,
                        perf_func:SumPerf=SumPerf([torch.nn.MSELoss()]),
                        device:torch.device=device,
                        model_type:str='minkowski', 
                        do_we_consider_aux:bool=False,
                        do_we_consider_coord:bool=False,
                        do_we_consider_feat:bool=True,
                        do_we_consider_event_id:bool=True,
                        mom_spherical_coord:bool=False,) -> dict[str,np.ndarray]:
    """
    Analyse the results from testing. 
    Gives some performances and convert back predictions, features and targets to their unscaled values.
    
    Parameters:
    - results_from_test_full: dict[str, list[np.ndarray]], results from the test_full function
    - dataset: PGunEvent, dataset to use for scaling and know which inputs/targets were used
    - perf_func: SumPerf, function to calculate the performance (not used)
    - device: torch.device, device to run the model on
    - model_type: str, type of the model ('minkowski' or 'transformer')
    - do_we_consider_aux: bool, whether to consider auxiliary variables
    - do_we_consider_coord: bool, whether to consider coordinate
    - do_we_consider_feat: bool, whether to consider features
    - do_we_consider_event_id: bool, whether to consider event IDs
    - mom_spherical_coord: bool, whether spherical coordinates were used for predicting the momentum
    
    Returns:
    - results: dict[str, np.ndarray],  unscaled results from the test_full function 
    
    """
    aux=None
    coord=None
    features=None
    event_id=None
    
    
    if do_we_consider_feat:
        features=np.vstack(results_from_test_full['f'])
        if model_type=='minkowski':
            features=features # in minkowski models the features do not contain the coordinates
        elif model_type=='transformer':
            features=features[:,3:] # saul order # in transformer models we must remove the coordinates from the features
            # features=features[:,:-3] # order for the first 5 models # in transformer models we must remove the coordinates from the features
        else:
            raise ValueError(f"Wrong model type {model_type}")
        
        if dataset.inputs is None:
            ## Use the full scaler
            features=dataset.scaler_x.inverse_transform(features)
        else:
            ## We are using only some parts of the input
            _min=dataset.scaler_x.min_[dataset.inputs]
            _scale=dataset.scaler_x.scale_[dataset.inputs]
            features-=_min
            features/=_scale
    
    if do_we_consider_aux:
        aux=np.vstack(results_from_test_full['aux'])
    
    if do_we_consider_coord:
        coord=np.vstack(results_from_test_full['c'])
        if dataset.scale_coordinates:
            coord=dataset.scaler_c.inverse_transform(coord)
        else:
            coord=transform_inverse_cube(coord)
    
    mask=np.vstack(results_from_test_full['mask'])
    targets=np.vstack(results_from_test_full['y'])
    _pred=np.vstack(results_from_test_full['predictions'])
    
    if dataset.y_indexes_with_scale is None:
        ## If all targets are considered, we aplly usual inverse transformation
        targets=dataset.scaler_y.inverse_transform(targets)
        pred=dataset.scaler_y.inverse_transform(_pred)
        
        ### In case of a classification task in index 10 (and after), use the following code
        # targets[:,:10]=dataset.scaler_y.inverse_transform(targets[:,:10])
        # _pred[:,:10]=dataset.scaler_y.inverse_transform(_pred[:,:10])
        # pred=np.zeros_like(targets)
        # pred[:,:10]=_pred[:,:10]
        # pred[:,10]=np.argmax(_pred[:,10:],axis=-1)
        
    else:
        ## If only some targets are considered, we restrict the inverse transformation to those
        _min=dataset.scaler_y.min_[dataset.y_indexes_with_scale]
        _scale=dataset.scaler_y.scale_[dataset.y_indexes_with_scale]
        targets[:,np.arange(len(dataset.y_indexes_with_scale))]-=_min
        targets[:,np.arange(len(dataset.y_indexes_with_scale))]/=_scale
        
        if not mom_spherical_coord:
            _pred[:,np.arange(len(dataset.y_indexes_with_scale))]-=_min
            _pred[:,np.arange(len(dataset.y_indexes_with_scale))]/=_scale
        else:
            ## we are assuming that there are both the position and the momentum in the predictions, so that the momentum is in indexes 3:6  
            _pred[:,0:3]-=_min[0:3]
            _pred[:,0:3]/=_scale[0:3]
            pred_norm=_pred[:,3]/_scale[3]
            pred_theta=_pred[:,4]
            pred_phi=_pred[:,5]
            pred_momx=pred_norm*np.sin(pred_theta)*np.cos(pred_phi)
            pred_momy=pred_norm*np.sin(pred_theta)*np.sin(pred_phi)
            pred_momz=pred_norm*np.cos(pred_theta)
            _pred[:,3]=pred_momx
            _pred[:,4]=pred_momy
            _pred[:,5]=pred_momz
             
        if dataset.targets_n_classes[-1]>0:
            ## If we have some class predictions, we need to argmax
            pred=np.zeros_like(targets)
            pred[:,:(targets.shape[-1]-1)]=_pred[:,:(targets.shape[-1]-1)]
            pred[:,(targets.shape[-1]-1)]=np.argmax(_pred[:,(targets.shape[-1]-1):],axis=-1)
        else:
            pred=_pred
    
    targets_=torch.Tensor(targets*mask).to(device)
    pred_=torch.Tensor(_pred*mask).to(device)
    del _pred
    
    scores=None

    # scores=perf_func(pred_,targets_)
    del pred_, targets_
    
    # scores=scores.cpu().numpy()
    
    if do_we_consider_event_id:
        event_id=np.vstack(results_from_test_full['event_id'])
    
    return {'predictions':pred, 'f':features, 'c':coord, 'y':targets, 'aux':aux, 'mask':mask, 'scores':scores, 'event_id':event_id}    
        


def training(device:torch.device,
            model:torch.nn.Module,
            dataset:PGunEvent,
            optimizer:torch.optim.Optimizer,
            warmup_scheduler:WarmUpScheduler,
            model_type:str='minkowski', # or 'transformer'
            loss_func:SumLoss=SumLoss([torch.nn.MSELoss()]),
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
            save_model_path:str|None=None,
            num_workers:int=24,
            notebook_tqdm:bool=False,
            ) -> dict[str,list[float]]:
    """
    Training function.
    
    Parameters:
    - device: torch.device, device to run the model on
    - model: torch.nn.Module, the model to train
    - dataset: PGunEvent, the dataset to use for training
    - optimizer: torch.optim.Optimizer, optimizer to use for training
    - warmup_scheduler: WarmUpScheduler, scheduler for learning rate warmup
    - model_type: str, type of model ('minkowski' or 'transformer')
    - loss_func: SumLoss, loss function to use
    - batch_size: int, batch size for training
    - train_fraction: float, fraction of data to use for training
    - val_fraction: float, fraction of data to use for validation
    - seed: int, seed for random number generation
    - epochs: int, number of epochs to train
    - stop_after_epochs: int, stop training after this number of epochs if validation loss does not improve
    - progress_bar: bool, whether to display a epochs progress bar during training
    - sub_progress_bars: bool, whether to display a progress bars of each epoch during training
    - benchmarking: bool, whether to display benchmark (computation time)
    - multi_GPU: bool, whether multi GPUs are used (parrallelization)
    - world_size: int, number of GPUs used
    - save_model_path: str|None, path to save the trained model
    - num_workers: int, number of workers for data loading
    - notebook_tqdm: bool, whether to use notebook-like tqdm progress bars
    
    Returns:
    - dict, dictionary with training results containing training and validation losses and loss composition, with learning rate and loss weights.
    """
    
    # Select the correct collate function
    if model_type=='minkowski':
        collate_fn=collate_minkowski
    elif model_type=='transformer':
        collate_fn=collate_transformer
    else:
        raise ValueError(f"Wrong model type {model_type}")
    
        
    # creates the data loaders
    train_loader, valid_loader, test_loader=split_dataset(dataset,
                                                        batch_size = batch_size,
                                                        train_fraction=train_fraction,
                                                        val_fraction=val_fraction,
                                                        seed=seed,
                                                        multi_GPU=multi_GPU,
                                                        collate=collate_fn,
                                                        num_workers=num_workers)
    
    LOSSES=[[],[]]
    COMP_LOSSES=[[],[]]
    LR=[]
    max_val_acc = np.inf
    epochs_since_last_improvement=0
    loss_func.to(device)
    
    # print("Starting training...")
    if save_model_path is not None:
        j=save_model_path.split("_")[-1].split(".")[0] # extract the #j div of the save path
    else:
        j=str(0)
    
    epoch_bar=tqdm.notebook.tqdm(range(0, epochs),
                            desc=f"Training Track fitting net {j} {model_type}",
                            disable=(not progress_bar),
                            position= 0,
                            leave=True,) if notebook_tqdm else tqdm.tqdm(range(0, epochs),
                                                                        desc=f"Training Track fitting net {j} {model_type}",
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

        # Train
        t0=time.perf_counter()
        loss, comp_loss = train(model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    model_type=model_type,
                    warmup_scheduler=warmup_scheduler,
                    loss_func=loss_func,
                    device=device, 
                    progress_bar=sub_progress_bars,
                    benchmarking=benchmarking,
                    world_size=world_size,
                    notebook_tqdm=notebook_tqdm)
        
        if benchmarking:
            print(f"Effective train time: {time.perf_counter()-t0:.2f} s")
        
        LOSSES[0].append(loss)
        COMP_LOSSES[0].append(comp_loss)

        # Test on the validation set
        t0=time.perf_counter()
        val_pred, val_true, val_loss, val_comp_loss = test(model=model,
                                        loader=valid_loader,
                                        optimizer=optimizer,
                                        model_type=model_type,
                                        warmup_scheduler=warmup_scheduler,
                                        loss_func=loss_func,
                                        device=device, 
                                        progress_bar=sub_progress_bars,
                                        benchmarking=benchmarking,
                                        world_size=world_size,
                                        notebook_tqdm=notebook_tqdm)
        if benchmarking:
                print(f"Effective validation time: {time.perf_counter()-t0:.2f} s")
                
        LOSSES[1].append(val_loss)
        COMP_LOSSES[1].append(val_comp_loss)

        lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        LR.append(lr)

        # Check results (check for improvement)
        if val_loss < max_val_acc:
            
            max_val_acc = val_loss
            
            if save_model_path is not None and (not multi_GPU or device==0):
                torch.save(model.state_dict(), save_model_path)
            
            # reset the counting of staling to 0
            epochs_since_last_improvement = 0

        else:
            epochs_since_last_improvement += 1
            
        # Set postfix to print the current loss
        epoch_bar.set_postfix(
            {
                "Tloss": f"{LOSSES[0][-1]:.2e}",
                "Vloss": f"{LOSSES[1][-1]:.2e}",
                # "m5VL": f"{np.mean(LOSSES[1][-5:]):.2e}",
            }
        )

    COMP_LOSSES=np.array(COMP_LOSSES)
    
    return {"training_loss":LOSSES[0],
            "validation_loss":LOSSES[1],
            "learning_rate":LR,
            "training_loss_composition":COMP_LOSSES[0],
            "validation_loss_composition":COMP_LOSSES[1],
            "loss_weights":loss_func.weights}


