import numpy as np
import torch
import time

from .utils import track_recon
from .. import track_fitting_net
from ..datasets import *
from sklearn.metrics import classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ChargeIdDataset(EventDataset):
    
    def __init__(self, 
                 root:str, 
                 shuffle:bool=False,
                 **kwargs):
        
        super().__init__(root=root, shuffle=shuffle, **kwargs)
        self.variables=['x','y','aux', 'event_id', 'traj_id']
        
    def getx(self,data:np.lib.npyio.NpzFile):
        return torch.FloatTensor(data['x'])
    
    def gety(self,data:np.lib.npyio.NpzFile):
        return torch.IntTensor(data['y'])

    def getaux(self,data:np.lib.npyio.NpzFile):
        return torch.FloatTensor(data['aux'])
    
    def getevent_id(self,data:np.lib.npyio.NpzFile):
        return torch.IntTensor(data['event_id'])
    
    def gettraj_id(self,data:np.lib.npyio.NpzFile):
        return torch.IntTensor(data['traj_id'])
    
    def getc(self,data:np.lib.npyio.NpzFile):
        return None


def generate_dataset(
                    j_model:int=22,
                    use_sparse_cnn:bool=False,
                    masking_scheme:str='primary',
                    batch_size:int=16,
                    save_path:str="/scratch4/maubin/",
                    dataset_folder:str="/scratch4/maubin/data/pgun_train",
                    scaler_file:str="/scratch4/maubin/data/scaler_trackfit2.p",
                    ):
    
    if use_sparse_cnn:
        model=track_fitting_net.create_sparse_cnn_model(y_out_channels=6, x_in_channels=1)
    else:
        model=track_fitting_net.create_transformer_model(y_out_channels=6, x_in_channels=1)
        
    model.load_state_dict(torch.load(f"{save_path}models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j_model}.torch"))
    

    track_fitting_train_dataset=PGunEvent(root=dataset_folder,
                                            shuffle=True,
                                            files_suffix='npz',
                                            scaler_file=scaler_file,
                                            use_true_tag=True,
                                            scale_coordinates=(not use_sparse_cnn),
                                            targets=None,
                                            inputs=[0],
                                            masking_scheme=masking_scheme)
    
    track_fitting_full_loader=full_dataset(track_fitting_train_dataset,
                        collate=track_fitting_net.collate_minkowski if use_sparse_cnn else track_fitting_net.collate_transformer,
                        batch_size=batch_size,)
    
    print("Predicting the track points position using the track fitting network...")
    track_fitting_all_results=track_fitting_net.test_full(loader=track_fitting_full_loader,
                                                            model=model,
                                                            model_type="minkowski" if use_sparse_cnn else "transformer",
                                                            progress_bar=True,
                                                            device=device,
                                                            do_we_consider_aux=True,
                                                            do_we_consider_coord=True,
                                                            do_we_consider_feat=True,
                                                            # max_batches=10,
                                                            )
    del model
    
    print("Updating the results dictionnary to invert the data transformations...")
    track_fitting_all_results=track_fitting_net.measure_performances(track_fitting_all_results,
                                            track_fitting_train_dataset,
                                            track_fitting_net.perf_fn,
                                            device, 
                                            model_type="minkowski" if use_sparse_cnn else "transformer",
                                            do_we_consider_aux=True,
                                            do_we_consider_coord=True,
                                            do_we_consider_feat=True,
                                            mom_spherical_coord=False)
    
    print("Sort the event, the trajectories and the hits...")
    track_fitting_all_results=track_recon.sort_event_from_all_results(track_fitting_all_results)
    
    torch.cuda.empty_cache()
    print("Reconstruct the direction and curvature of each trajectory points...")
    charge_pred,mom_pred,curv_pred,points_pred,point_coordinate_order_pred=track_recon.charge_and_momentum_fit(track_fitting_all_results,
                                                                                    'pred',
                                                                                    mode='node_d',
                                                                                    n=4,
                                                                                    device=device,
                                                                                    chargeID_mode="curv_estimate",
                                                                                    )
    
    print("Store the data in a dictionary...")
    mom_norm=np.linalg.norm(mom_pred,axis=-1,keepdims=True)+1e-9
    curv_norm=np.linalg.norm(curv_pred,axis=-1,keepdims=True)+1e-9
    
    chargeID_dataset_dict={}
    chargeID_dataset_dict['x']=np.concatenate([points_pred,mom_pred/mom_norm,curv_pred/curv_norm,mom_norm,curv_norm,point_coordinate_order_pred[:,None]],axis=1)
    chargeID_dataset_dict['y']=(np.round(track_fitting_all_results['true_charge'][:,None]+1)).astype(int)
    chargeID_dataset_dict['aux']=track_fitting_all_results['aux']
    chargeID_dataset_dict['event_id']=track_fitting_all_results['event_id']
    chargeID_dataset_dict['traj_id']=track_fitting_all_results['traj_id']
    
    return chargeID_dataset_dict



def save_dataset(chargeID_dataset_dict:dict,
                 save_path:str,
                 show_progress_bar:bool=True):
    
    length=int(chargeID_dataset_dict['event_id'].max())
    
    for index in tqdm.tqdm(range(length),disable=(not show_progress_bar), desc="Saving to npz files"):
        i,j=np.searchsorted(chargeID_dataset_dict['event_id'][:,0],[index,index+1],side='left')
        indexes=np.arange(i,np.clip(j,i,length-1))
        return_dict={}
        
        for key in chargeID_dataset_dict.keys():
            return_dict[key]=chargeID_dataset_dict[key][indexes]
            
        np.savez_compressed(f"{save_path}event_{index}",**return_dict)



model = track_fitting_net.create_transformer_model(x_in_channels=9,
                                                    y_out_channels=3,
                                                    device=device,
                                                    D_MODEL = 32,
                                                    N_HEAD = 4,
                                                    DIM_FEEDFORWARD = 128,
                                                    NUM_ENCODER_LAYERS = 3)

loss_fn=torch.nn.CrossEntropyLoss()
PAD_IDX=-1.


def collate_fn(batch):
    
    lens = [len(d['x']) for d in batch]
    
    try:
        x = torch.nn.utils.rnn.pad_sequence([batch[k]['x'] for k in filter(lambda k: lens[k]>0, range(len(lens)))], batch_first=False, padding_value=PAD_IDX)
        y = torch.nn.utils.rnn.pad_sequence([batch[k]['y'] for k in filter(lambda k: lens[k]>0, range(len(lens)))], batch_first=False, padding_value=PAD_IDX)
        aux = torch.nn.utils.rnn.pad_sequence([batch[k]['aux'] for k in filter(lambda k: lens[k]>0, range(len(lens)))], batch_first=False, padding_value=PAD_IDX)
        event_id = torch.nn.utils.rnn.pad_sequence([batch[k]['event_id'] for k in filter(lambda k: lens[k]>0, range(len(lens)))], batch_first=False, padding_value=PAD_IDX)
        traj_id = torch.nn.utils.rnn.pad_sequence([batch[k]['traj_id'] for k in filter(lambda k: lens[k]>0, range(len(lens)))], batch_first=False, padding_value=PAD_IDX)
        lens = [lens[k] for k in filter(lambda k: lens[k]>0, range(len(lens)))]

        return {"x":x, "y":y, "aux":aux, "event_id":event_id, "traj_id":traj_id, "lens":lens}
    
    except RuntimeError as E:
        
        return {"x":torch.Tensor(), "y":torch.Tensor(), "aux":torch.Tensor(), "event_id":torch.Tensor(), "traj_id":torch.Tensor()}


def execute_model(model:torch.nn.Module,
                data:dict,
                device:torch.device=device,
                ):
    if data['x'] is not None and data['x'].numel()>0:
        features=data['x'].to(device)
        # create masks
        src_mask, src_padding_mask = track_fitting_net.create_mask_src(features)
        # run model
        pred = model(features, src_mask, src_padding_mask)
        # masking the data (noise hits)
        targ=data['y'].to(device)
        # packing the data
        pred = torch.nn.utils.rnn.pack_padded_sequence(pred, data['lens'], batch_first=False, enforce_sorted=False)
        targ = torch.nn.utils.rnn.pack_padded_sequence(targ, data['lens'], batch_first=False, enforce_sorted=False)
        # flattening the data
        pred=pred.data
        targ=targ.data
        aux=torch.nn.utils.rnn.pack_padded_sequence(data['aux'], data['lens'], batch_first=False, enforce_sorted=False).data
        event_id=torch.nn.utils.rnn.pack_padded_sequence(data['event_id'], data['lens'], batch_first=False, enforce_sorted=False).data
        traj_id=torch.nn.utils.rnn.pack_padded_sequence(data['traj_id'], data['lens'], batch_first=False, enforce_sorted=False).data
        feats=torch.nn.utils.rnn.pack_padded_sequence(features, data['lens'], batch_first=False, enforce_sorted=False).data
        
        return pred, targ, aux, feats, event_id, traj_id
    else:
        return torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()


# Training function
def train(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:track_fitting_net.WarmUpScheduler,
          loss_func=loss_fn,
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          notebook_tqdm:bool=False,):
    
    
    model.train()
    
    batch_size = loader.batch_size
    n_batches = int(np.ceil(len(loader.dataset) / (batch_size)))
    n_effective = 0 
    train_loop = tqdm.notebook.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Train loop {device}", position=1, leave=False) if notebook_tqdm else tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Train loop {device}", position=1, leave=False)
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    sum_loss = 0.
    
    for i, data in train_loop:
        time_load+=time.perf_counter()-t0
        optimizer.zero_grad()
        
        t0=time.perf_counter()
        
        pred, targ,  _a,  _f, _e, _ti = execute_model(model=model,
                                data=data,
                                device=device)
        
        time_model+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        if pred.numel()>0:
            
            loss=loss_func(pred,targ[:,0].to(int))
            loss.backward()
            
            # Update progress bar
            train_loop.set_postfix({"loss":  f"{loss.item():.5f}","lr":f"{optimizer.param_groups[0]['lr']:.2e}"})
            
            sum_loss += loss.item()
            n_effective += 1
            
            optimizer.step()
            warmup_scheduler.step()
            
            time_steps+=time.perf_counter()-t0
            t0=time.perf_counter()
        
    if benchmarking:
        print(f"Training: Loading: {time_load:.2f} s \t Arranging: {time_model:.2f} s \t Model steps: {time_steps:.2f} s \t Total: {time_load+time_model+time_steps:.2f} s")
      
        
    return sum_loss / n_effective



# Testing function
def test(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:track_fitting_net.WarmUpScheduler,
          loss_func=loss_fn,
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          notebook_tqdm:bool=False,):
    
    model.eval()
    
    batch_size = loader.batch_size
    n_batches = int(np.ceil(len(loader.dataset) / (batch_size)))
    n_effective = 0
    test_loop = tqdm.notebook.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test loop {device}", position=1, leave=False) if notebook_tqdm else tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test loop {device}", position=1, leave=False)
    
    sum_loss = 0.
    true_targets = []
    predictions = []
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    
    for i, data in test_loop:
        time_load+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        pred, targ,  _a,  _f, _e, _ti = execute_model(model=model,
                                data=data,
                                device=device)
        
        time_model+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        if pred.numel()>0:
            
            
            loss=loss_func(pred,targ[:,0].to(int))
            sum_loss += loss.item()
            n_effective += 1
            
            # Update progress bar
            test_loop.set_postfix({"loss":  f"{loss.item():.5f}"})
            
            # true_targets+=targ.tolist()
            # predictions+=pred.tolist()
            time_steps+=time.perf_counter()-t0
            t0=time.perf_counter()
        
    if benchmarking:
        print(f"Testing: Loading: {time_load:.2f} s \t Arranging: {time_model:.2f} s \t Model steps: {time_steps:.2f} s \t Total: {time_load+time_model+time_steps:.2f} s")
      
        
          
    return np.array(predictions), np.array(true_targets), sum_loss / n_effective, 




def training(device:torch.device,
            model:torch.nn.Module,
            dataset:PGunEvent,
            optimizer:torch.optim.Optimizer,
            warmup_scheduler:track_fitting_net.WarmUpScheduler,
            loss_func=loss_fn,
            batch_size:int = 256,
            train_fraction:float=0.8,
            val_fraction:float=0.19,
            seed:int = 7,
            epochs:int = 200,
            stop_after_epochs: int = 30,
            progress_bar:bool =True,
            sub_progress_bars:bool = False,
            benchmarking:bool=False,
            save_model_path:str|None=None,
            num_workers:int=16,
            notebook_tqdm:bool=False,
            ):
    
    
        
    # creates the data loaders
    train_loader, valid_loader, test_loader=split_dataset(dataset,
                                                        batch_size = batch_size,
                                                        train_fraction=train_fraction,
                                                        val_fraction=val_fraction,
                                                        seed=seed,
                                                        multi_GPU=False,
                                                        collate=collate_fn,
                                                        num_workers=num_workers)
    
    LOSSES=[[],[]]
    LR=[]
    max_val_acc = np.inf
    epochs_since_last_improvement=0
    loss_func.to(device)
    
    # print("Starting training...")
    if save_model_path is not None:
        j=save_model_path.split("_")[-1].split(".")[0] # extract the #j div of the save path
    else:
        j=""
    
    epoch_bar=tqdm.notebook.tqdm(range(0, epochs),
                            desc=f"Training Track fitting net {j}",
                            disable=(not progress_bar),
                            position= 0,
                            leave=True,) if notebook_tqdm else tqdm.tqdm(range(0, epochs),
                                                                        desc=f"Training Track fitting net {j}",
                                                                        disable=(not progress_bar),
                                                                        position= 0,
                                                                        leave=True,)

    for epoch in epoch_bar:

        # Early stopping: finish training when validation results don't improve for 10 epochs
        if epochs_since_last_improvement >= stop_after_epochs:
            print("Early stopping: finishing....")
            break
        

        # Train
        loss = train(model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    warmup_scheduler=warmup_scheduler,
                    loss_func=loss_func,
                    device=device, 
                    progress_bar=sub_progress_bars,
                    notebook_tqdm=notebook_tqdm,
                    benchmarking=benchmarking)
        
        
        LOSSES[0].append(loss)

        # Test on the validation set
        val_pred, val_true, val_loss = test(model=model,
                                        loader=valid_loader,
                                        optimizer=optimizer,
                                        warmup_scheduler=warmup_scheduler,
                                        loss_func=loss_func,
                                        device=device, 
                                        progress_bar=sub_progress_bars,
                                        notebook_tqdm=notebook_tqdm,
                                        benchmarking=benchmarking)
                
        LOSSES[1].append(val_loss)

        lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        LR.append(lr)
        
        # Check results (check for improvement)
        if val_loss < max_val_acc:
            
            max_val_acc = val_loss
            
            if save_model_path is not None:
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

    
    return {"training_loss":LOSSES[0],
            "validation_loss":LOSSES[1],
            "learning_rate":LR,}
    




# Predicting function
def test_full(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          notebook_tqdm:bool=False,
          max_batches:int|None=None,):
    
    model.eval()
    
    batch_size = loader.batch_size
    if max_batches is not None:
        n_batches=max_batches
    else:
       n_batches = int(np.ceil(len(loader.dataset) / batch_size))
    test_loop = tqdm.notebook.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test full loop {device}", position=1, leave=False) if notebook_tqdm else tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test full loop {device}", position=1, leave=False)
    
    sum_loss = 0.
    true_targets = []
    predictions = []
    aux = []
    features = []
    event_id = []
    traj_id = []
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    
    for i, data in test_loop:
        time_load+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        pred, targ,  aux_,  feat, event_, traj_ = execute_model(model=model,
                                data=data,
                                device=device)
        
        time_model+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        if pred.numel()>0:
            
            predictions.append(pred.detach().cpu().numpy())
            true_targets.append(targ.detach().cpu().numpy()) 
            aux.append(aux_.detach().cpu().numpy())   
            features.append(feat.detach().cpu().numpy())  
            event_id.append(event_.detach().cpu().numpy())   
            traj_id.append(traj_.detach().cpu().numpy())   
        
            time_steps+=time.perf_counter()-t0
            t0=time.perf_counter()
        
    if benchmarking:
        print(f"Testing full: Loading: {time_load:.2f} s \t Arranging: {time_model:.2f} s \t Model steps: {time_steps:.2f} s \t Total: {time_load+time_model+time_steps:.2f} s")
      
        
          
    return {"predictions": np.concatenate(predictions,axis=0), "y": np.concatenate(true_targets,axis=0), "aux": np.concatenate(aux,axis=0), "x": np.concatenate(features,axis=0), "event_id": np.concatenate(event_id,axis=0), "traj_id": np.concatenate(traj_id,axis=0),}





    
    

def main_worker(device:torch.device,
                dataset:ChargeIdDataset,
                args,):
    
    global loss_fn
    
    #### Get the variables from args
    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    sub_progress_bars=args.sub_tqdm
    
    
    if args.resume:
        print("Loading model state dict save...")
        model.load_state_dict(torch.load(f"{args.save_path}models/chargeID_model_{j}.torch"))
    
    # Print the total number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable params: {pytorch_total_params}")
    

    # Loaders parameters
    batch_size=args.batch_size
    train_fraction=0.9
    val_fraction=0.099
    seed=7
    len_train_loader=int(len(dataset)*train_fraction/(batch_size))

    # Optimizer and scheduler
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
    num_steps_one_cycle = 30
    num_warmup_steps = 3
    cosine_annealing_steps = len_train_loader * num_steps_one_cycle
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr*(1e-2))
    warmup_scheduler = track_fitting_net.WarmUpScheduler(optimizer, lr_scheduler,
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
                            loss_func=loss_fn,
                            batch_size=batch_size,
                            train_fraction=train_fraction,
                            val_fraction=val_fraction,
                            seed=seed,
                            epochs=num_epochs,
                            stop_after_epochs= stop_after_epochs,
                            progress_bar=True,
                            sub_progress_bars=True,
                            save_model_path=f"{args.save_path}models/chargeID_model_{j}.torch",
                            benchmarking=args.benchmarking,
                            notebook_tqdm=False)
    
    
    return model, training_dict    





if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
                        prog='ChargeIDTraining',
                        description='Trains a model for Charge Identification in SFG',)

    parser.add_argument('j', metavar='j', type=int, help='#j div of the model')
    parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for training")
    parser.add_argument('dataset_folder_trackfit',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for generating the dataset")
    parser.add_argument('scaler_file',metavar='Scaler_File', type=str, help="File storing the dataset features scalers")
    parser.add_argument('save_path',metavar='Save_Path', type=str, help="Path to save results and models")
    parser.add_argument('--test_only', action='store_true', help='runs only the test (measure performances, plots, ...)')
    parser.add_argument('-T', '--test', action='store_true', help='runs test after training (measure performances, plots, ...)')
    parser.add_argument('-B', '--sparse_cnn', action='store_true', help='use the sparse_cnn model (MinkUNet) for the track fitting, otherwise use the transformer')
    parser.add_argument('-R', '--resume', action='store_true', help='resume the training by loading the saved model state dictionary')
    parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
    parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the execution time for benchmarking')
    parser.add_argument('-g', '--generate', action='store_true', help='generates the datasets for training and testing')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('-n','--num_epochs', type=int, default=100, help='number of epochs for the training')
    parser.add_argument('--lr', type=float, default=1e-3, help='maximum learning rate for training (defines the scale of the learning rate)')
    parser.add_argument('--stop_after_epochs', type=int, default=20, help='maximum number of epochs without improvement before stopping the training (early termination)')
    parser.add_argument('--j_trackfit', type=int, default=22, help='#j div of the trackfit model')
    parser.add_argument('--batch_size_trackfit', type=int, default=16, help='batch size for predictions with the trackfit model')
    parser.add_argument('--ms', type=str, default='distance', metavar='Masking Scheme', help='Which mask to use for the dataset (distance, primary, tag, ...)')

    args = parser.parse_args()
    

    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    sub_progress_bars=args.sub_tqdm


    torch.multiprocessing.set_sharing_strategy('file_system')
    
    



    def main():
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        
        if (not args.test_only):
        
            if args.generate:
                # generate dataset
                dataset_dict=generate_dataset(
                        j_model=args.j_trackfit,
                        use_sparse_cnn=args.sparse_cnn,
                        masking_scheme=args.ms,
                        batch_size=args.batch_size_trackfit,
                        save_path=args.save_path,
                        dataset_folder=args.dataset_folder_trackfit,
                        scaler_file=args.scaler_file,
                        )
                print("Saving the dataset into npz files...")
                save_dataset(dataset_dict, args.dataset_folder)
                del dataset_dict
            
            dataset=ChargeIdDataset(root=args.dataset_folder,shuffle=True)
            

            model, training_dict=main_worker(device,
                                dataset,
                                args,)
                

            # torch.save(model.state_dict(), f"/scratch4/maubin/models/hittag_model_{j}.torch")
            torch.save(training_dict,f"{args.save_path}results/chargeID_training_dict_{j}.torch")
        
        if args.test or args.test_only:
            print("Testing the model...")
            testdataset_folder=args.dataset_folder[:-6]+"test/" ## we are assuming that the dataset_folder is of type "*_train/" whereas the testdataset folder will be "*_test/"
            testdataset_folder_trackfit=args.dataset_folder_trackfit[:-6]+"test/"
            if args.generate:
                testdataset_dict=generate_dataset(
                        j_model=args.j_trackfit,
                        use_sparse_cnn=args.sparse_cnn,
                        masking_scheme=args.ms,
                        batch_size=args.batch_size_trackfit,
                        save_path=args.save_path,
                        dataset_folder= testdataset_folder_trackfit,
                        scaler_file=args.scaler_file,
                        )
                print("Saving the dataset into npz files...")
                save_dataset(testdataset_dict, testdataset_folder)
                del testdataset_dict
                
            testdataset=ChargeIdDataset(root=testdataset_folder,shuffle=False)
                        
            model = track_fitting_net.create_transformer_model(x_in_channels=9,
                                                                y_out_channels=3,
                                                                device=device,
                                                                D_MODEL = 32,
                                                                N_HEAD = 4,
                                                                DIM_FEEDFORWARD = 128,
                                                                NUM_ENCODER_LAYERS = 3)
            
            model.load_state_dict(torch.load(f"{args.save_path}models/chargeID_model_{j}.torch"))
            
            full_loader=full_dataset(testdataset,
                                    collate=collate_fn,
                                    batch_size=args.batch_size,)
            
            all_results=test_full(loader=full_loader,
                                model=model,
                                progress_bar=True,
                                device=device,
                                )

            with open(f'{args.save_path}results/chargeID_fitting_model_{j}_pred.pkl', 'wb') as file:
                pk.dump(all_results,file)
                
            print("Showing the results...")
            # print(all_results['predictions'][:30])
            print(np.mean(np.argmax(all_results['predictions'],axis=-1)))
            # print(classification_report(y_true=np.round(all_results['y'][:,0]).astype(int),
            #                             y_pred=np.argmax(all_results['predictions'],axis=-1),
            #                             target_names=['-1','0','1']))
                    
            

    main()
