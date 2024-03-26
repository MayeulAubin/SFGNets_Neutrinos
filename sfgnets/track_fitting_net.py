import torch
import os
from sfgnets.dataset import *
from sfgnets.utils import minkunet
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tqdm
from sklearn.metrics import precision_recall_fscore_support
from warmup_scheduler_pytorch import WarmUpScheduler
import time
import math
from copy import deepcopy

x_in_channels=2
y_out_channels=10+len(particles_classes.keys())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FittingTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1  # dropout value
                 ):
        super(FittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        # weights initialisation
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self,
                src: Tensor,
                src_mask: Tensor,
                src_padding_mask: Tensor):
        # linear projection of the input
        src_emb = self.proj_input(src)
        # transformer encoder
        memory = self.transformer_encoder(src=src_emb, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        # dropout
        memory = self.dropout(memory)
        # linear projection of the output
        output = self.decoder(memory)
        # output[:, :, :3] += src[:, :, :3]  # learn residuals for x,y,z position
        return output

class SumLoss(nn.Module):
    """
    Defines the sum of several losses. Will return both the total loss and the detail of each individual loss term (loss composition).
    """
    def __init__(self, losses:list[nn.Module], weights:list[float]=None):
        super().__init__()
        self.losses=torch.nn.ModuleList(losses)
        self.weights=weights
        if weights is None:
            self.weights=[1. for _ in self.losses]
    
    def __len__(self):
        return len(self.losses)
    
    def __getitem__(self, key: int | slice | list[int]):
        if key is list:
            losses=[self.losses[k] for k in key]
            weights=[self.weights[k] for k in key]
        else:
            losses=self.losses[key]
            weights=self.weights[key]
        return self.__class__(losses=losses,weights=weights)
    
    def forward(self, pred:Tensor, target:Tensor) -> tuple[Tensor,list[float]]:
        loss=0.0
        loss_composition=[]
        
        for l,w in zip(self.losses,self.weights):
            _loss=l(pred,target)
            loss+=w*_loss
            loss_composition.append(_loss.item())
            
        return loss/sum(self.weights), loss_composition
    
    
    
class PartialLoss(nn.Module):
    """
    Creates a loss that applies only to specific indexes of the predictions and targets. Allows to have a loss specific to some indexes.
    Example: PartialLoss(nn.MSELoss(),[0,1,2]) will aplly an MSELoss to the first 3 targets
    """
    def __init__(self, loss_func:nn.Module, indexes:list[int]):
        super().__init__()
        self.loss_func=loss_func
        self.indexes=indexes
    
    def forward(self, pred:Tensor, target:Tensor):
        return self.loss_func(pred[...,self.indexes],target[...,self.indexes])

class PartialClassificationLoss(nn.Module):
    """
    Creates a classification loss that applies only to specific indexes of the predictions and a specific index of the targets.
    Example: PartialLoss(nn.CrossEntropyLoss(),[0,1,2], 0) will aplly a Cross Entropy loss between the first three predictions (probabilities) and the first target (true class).
    """
    def __init__(self, loss_func:nn.Module, indexes_pred:list[int], index_target:int):
        super().__init__()
        self.loss_func=loss_func
        self.indexes_pred=indexes_pred
        self.index_target=index_target
    
    def forward(self, pred:Tensor, target:Tensor) -> Tensor:
        return self.loss_func(pred[...,self.indexes_pred],target[...,self.index_target].to(torch.int64))

    
class SumPerf(SumLoss):
    """
    Similar to SumLoss, but instead of returning a returning the reduced loss, return a full Tensor (for performances analysis)
    """
    def __init__(self, losses:list[nn.Module], weights:list[float]=None):
        super().__init__(losses=losses, weights=weights)
        for loss in self.losses:
            ## We ensure that the losses have a 'none' reduction 
            if type(loss) is PartialLoss or type(loss) is PartialClassificationLoss:
                loss.loss_func.reduction='none' # if the loss is one of these custom class, we must change the loss_func
            else:
                loss.reduction='none'

    def forward(self,  pred:Tensor, target:Tensor) -> Tensor:
        assert len(pred.shape)==2
        scores=torch.zeros((pred.shape[0], 1+len(self))).to(pred.device)
        for i, (l,w) in enumerate(zip(self.losses,self.weights)):
            _loss=l(pred,target)
            if len(_loss.shape)==2:
                _loss=_loss.mean(dim=-1)
            scores[:,0]+=w*_loss
            scores[:,i]=_loss
            
        return scores
    
    @staticmethod
    def from_SumLoss(sumloss:SumLoss):
        weights=deepcopy(sumloss.weights)
        losses=[deepcopy(l) for l in sumloss.losses]
        return SumPerf(losses=losses, weights=weights)

## Transformer parameters
D_MODEL = 64
N_HEAD = 8
DIM_FEEDFORWARD = 128
NUM_ENCODER_LAYERS = 5


baseline_model=minkunet.MinkUNet34B(in_channels=x_in_channels, out_channels=y_out_channels, D=3).to(device)

transformer_model = FittingTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                 d_model=D_MODEL,
                                 n_head=N_HEAD,
                                 input_size=3+x_in_channels,
                                 output_size=y_out_channels,
                                 dim_feedforward=DIM_FEEDFORWARD).to(device)


PAD_IDX=-1.


loss_fn=SumLoss(losses=[
                            PartialLoss(nn.MSELoss(), [0,1,2]), # node position
                            PartialLoss(nn.MSELoss(), [3,4,5]), # node direction
                            PartialLoss(nn.MSELoss(), [6]), # number of particles
                            PartialLoss(nn.MSELoss(), [7]), # energy deposited
                            PartialLoss(nn.MSELoss(), [8]), # particle charge
                            PartialLoss(nn.MSELoss(), [9]), # particle mass
                            PartialClassificationLoss(nn.CrossEntropyLoss(), [10+i for i in range(len(particles_classes.keys()))], 10), # particle class
                        ],
                weights=[
                            2.e5, # node position
                            1.e2, # node direction
                            2.e3, # number of particles
                            5.e4, # energy deposited
                            1.e1, # particle charge
                            5.e2, # particle mass
                            5.e-1, # particle class
                        ])

perf_fn=SumPerf.from_SumLoss(loss_fn)

# Optimizer and schedulers
lr = 0.01
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
len_train_loader=120000 # value to set with a real loader
num_steps_one_cycle = 25
num_warmup_steps = 10
cosine_annealing_steps = len_train_loader * num_steps_one_cycle
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr/100)
warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                   len_loader=1,
                                   warmup_steps=len_train_loader * num_warmup_steps,
                                   warmup_start_lr=lr/100,
                                   warmup_mode='linear')



# Create and save a scaler for the features and targets
def create_scalers(root:str,
                   scaler_file:str):
    features=[]
    targets=[]
    coordinates=[]
    
    dataset=PGunEvent(root=root,
              scaler_file=None)
    
    for j in tqdm.tqdm(range(len(dataset)), desc="Loading data"):
        event=dataset[j]
        features.append(event['x'].cpu().numpy())
        targets.append(event['y'].cpu().numpy()[:,:10]) # all but particle class
        coordinates.append(event['c'].cpu().numpy())
    
    features=np.vstack(features)
    targets=np.vstack(targets)
    coordinates=np.vstack(coordinates)
    
    print("Creating scalers...")
    scaler_x = MinMaxScaler()
    scaler_x.fit(features)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(targets)
    
    scaler_c = MinMaxScaler()
    scaler_c.fit(coordinates)

    with open(scaler_file, "wb") as fd:
        pk.dump([scaler_x, scaler_y, scaler_c], fd)


def filtering_events(x):
    return x['c'] is not None and len(x['c'])<1000



# function to collate data samples for a transformer
def collate_transformer(batch):
    """
    Custom collate function for Transformers
    """
    
    device=batch[0]['x'].device
   
    coords = [d['c']for d in filter(filtering_events,batch)]

    feats = [torch.cat([d['x'],d['c']],dim=-1) for d in filter(filtering_events,batch)]

    targets = [d['y'] for d in filter(filtering_events,batch)]

    aux = [d['aux'] for d in filter(filtering_events,batch)]
    
    masks = [d['mask']for d in filter(filtering_events,batch)]
    

    lens = [len(x) for x in feats]

    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=False, padding_value=PAD_IDX)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=False, padding_value=PAD_IDX)
    coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=False, padding_value=PAD_IDX)
    try:
        aux = torch.nn.utils.rnn.pad_sequence(aux, batch_first=False, padding_value=PAD_IDX)
    except TypeError: # if the aux variables are None, the cat will raise a type error
        aux = None
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=False, padding_value=False)

    return {'f':feats, 'y':targets, 'aux':aux, 'c':coords, 'mask':masks, 'lens':lens,}


# function to collate data samples for a MinkUNet
def collate_minkowski(batch):
    """
    Custom collate function for Sparse Minkowski network.

    Parameters:
    - batch: list, a list of dictionaries containing 'c' (coordinates), 'x' (features), and 'y' (labels) keys.

    Returns:
    - dict, a dictionary containing 'f' (features), 'c' (coordinates), and 'y' (labels) keys already arranged for a ME sparse tensor
    """
    device=batch[0]['x'].device
    # Extract coordinates from the batch and convert to integer type
    coords = [d['c'].int() for d in filter(filtering_events,batch)]

    # Concatenate features from the batch
    feats = torch.cat([d['x'] for d in filter(filtering_events,batch)])

    # Concatenate labels from the batch
    y = torch.cat([d['y'] for d in filter(filtering_events,batch)])
    
    # Concatenate aux from the batch
    try:
        aux = torch.cat([d['aux'] for d in filter(filtering_events,batch)])
    except TypeError: # if the aux variables are None, the cat will raise a type error
        aux = None
    
    # Concatenate masks from the batch
    masks = torch.cat([d['mask'] for d in filter(filtering_events,batch)])

    # Create a dictionary to store the collated data
    ret = {'f': feats, 'c': coords, 'y': y, 'aux':aux, 'mask':masks}

    return ret


# Transformer masks
def create_mask_src(src):
    """
    Creates the source mask (in our case it is total because we do not hide anything) and the padding mask
    """
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
    src_padding_mask = (src[:, :, 0] == PAD_IDX).transpose(0, 1).to(src.device)

    return src_mask, src_padding_mask


def execute_model(model:torch.nn.Module,
                data:dict,
                model_type:str='minkowski', # or 'transformer'
                device:torch.device=device,
                do_we_consider_aux:bool=False,
                do_we_consider_coord:bool=False,
                do_we_consider_feat:bool=False,):
    """
    Returns the prediction, target and mask for a given model. Adapts whether it is 'minkowski' type (SparseTensors) or 'transformer' type (PaddedSequence).
    """
    aux=None
    coord=None
    feats=None
    
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
        # change the mask data to fit the format of pred, targs, ...
        mask=torch.nn.utils.rnn.pack_padded_sequence(mask, data['lens'], batch_first=False, enforce_sorted=False).data
        
    else:
        raise ValueError(f"Wrong model type {model_type}")
    
    return pred, targ, mask, aux, coord, feats
    



# Training function
def train(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          model_type:str='minkowski', # or 'transformer'
          loss_func:SumLoss=SumLoss([torch.nn.MSELoss()]),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,):
    
    
    model.train()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / (batch_size*world_size)))
    train_loop = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Train loop {device}", position=1, leave=False)
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    sum_loss = 0.
    comp_loss=np.zeros(len(loss_func))
    
    for i, data in train_loop:
        time_load+=time.perf_counter()-t0
        optimizer.zero_grad()
        
        t0=time.perf_counter()
        
        pred,targ, _m, _a, _c, _f = execute_model(model=model,
                                data=data,
                                model_type=model_type,
                                device=device)
        
        time_model+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        loss,loss_composition=loss_func(pred,targ)
        loss.backward()
        
        # Update progress bar
        train_loop.set_postfix({"loss":  f"{loss.item():.5f}"})
          
        sum_loss += loss.item()
        comp_loss += loss_composition
        
        optimizer.step()
        warmup_scheduler.step()
        time_steps+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        
    if benchmarking:
        print(f"Training: Loading: {time_load:.2f} s \t Arranging: {time_model:.2f} s \t Model steps: {time_steps:.2f} s \t Total: {time_load+time_model+time_steps:.2f} s")
        
    return sum_loss / n_batches, comp_loss / n_batches




# Testing function
def test(model:torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          model_type:str='minkowski', # or 'transformer'
          loss_func:SumLoss=SumLoss([torch.nn.MSELoss()]),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,):
    
    model.eval()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / (batch_size*world_size)))
    test_loop = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc=f"Test loop {device}", position=1, leave=False)
    
    sum_loss = 0.
    comp_loss=np.zeros(len(loss_func))
    true_targets = []
    predictions = []
    
    
    time_load, time_model, time_steps, t0= 0., 0., 0., time.perf_counter()
    
    for i, data in test_loop:
        time_load+=time.perf_counter()-t0
        
        t0=time.perf_counter()
        
        pred,targ, _m, _a, _c, _f = execute_model(model=model,
                                data=data,
                                model_type=model_type,
                                device=device)
        
        time_model+=time.perf_counter()-t0
        t0=time.perf_counter()
        
        loss,loss_composition=loss_func(pred,targ)
        sum_loss += loss.item()
        comp_loss += loss_composition
        
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
            loader:torch.utils.data.DataLoader,
            model_type:str='minkowski', # or 'transformer'
            device:torch.device=device, 
            progress_bar:bool=False,
            do_we_consider_aux:bool=False,
            do_we_consider_coord:bool=False,
            do_we_consider_feat:bool=True,):
    
    model.eval()
    
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=(not progress_bar), desc="Testing Track fitting net")
    
    
    sum_loss = 0.
    pred = []
    feat=None
    coord=None
    target=[]
    aux=None
    mask=[]
    
    if do_we_consider_aux:
        aux=[]
    
    if do_we_consider_coord:
        coord=[]
    
    if do_we_consider_feat:
        feat=[]
    
    for i, data in t:
        
        _pred,_targ, _mask, _aux, _coord, _feat = execute_model(model=model,
                                                    data=data,
                                                    model_type=model_type,
                                                    device=device,
                                                    do_we_consider_aux=do_we_consider_aux,
                                                    do_we_consider_coord=do_we_consider_coord,
                                                    do_we_consider_feat=do_we_consider_feat,)
        
        pred.append(_pred.cpu().detach().numpy())
        target.append(_targ.cpu().detach().numpy())
        mask.append(_mask.cpu().detach().numpy())
        if do_we_consider_aux:
            aux.append(_aux.cpu().detach().numpy())
        if do_we_consider_coord:
            coord.append(_coord.cpu().detach().numpy())
        if do_we_consider_feat:
            feat.append(_feat.cpu().detach().numpy())
    
        
    torch.cuda.empty_cache() # release the GPU memory      
    return {'predictions':pred,'f':feat,'c':coord,'y':target, 'aux':aux, 'mask':mask}
    


def measure_performances(results_from_test_full:dict,
                        dataset:PGunEvent,
                        perf_func:SumPerf=SumPerf([torch.nn.MSELoss()]),
                        device:torch.device=device,
                        model_type:str='minkowski', 
                        do_we_consider_aux:bool=False,
                        do_we_consider_coord:bool=False,
                        do_we_consider_feat:bool=True,):
    aux=None
    coord=None
    features=None
    
    if do_we_consider_feat:
        features=np.vstack(results_from_test_full['f'])
        if model_type=='minkowski':
            features=dataset.scaler_x.inverse_transform(features) # in minkowski models the features do not contain the coordinates
        elif model_type=='transformer':
            features=dataset.scaler_x.inverse_transform(features[:,:-3]) # in transformer models we must remove the coordinates from the features
        else:
            raise ValueError(f"Wrong model type {model_type}")
    
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
    targets[:,:10]=dataset.scaler_y.inverse_transform(targets[:,:10])
    
    _pred=np.vstack(results_from_test_full['predictions'])
    _pred[:,:10]=dataset.scaler_y.inverse_transform(_pred[:,:10])
    
    targets_=torch.Tensor(targets*mask).to(device)
    pred_=torch.Tensor(_pred*mask).to(device)
    
    
    scores=perf_func(pred_,targets_)
    del pred_, targets_
    
    pred=np.zeros_like(targets)
    pred[:,:10]=_pred[:,:10]
    pred[:,10]=np.argmax(_pred[:,10:],axis=-1)
    del _pred
    
    scores=scores.cpu().numpy()
    
    return {'predictions':pred, 'f':features, 'c':coord, 'y':targets, 'aux':aux, 'mask':mask, 'scores':scores}    
        


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
            save_model_path:str=None,):
    
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
                                                        collate=collate_fn)
    
    LOSSES=[[],[]]
    COMP_LOSSES=[[],[]]
    LR=[]
    max_val_acc = np.inf
    epochs_since_last_improvement=0
    loss_func.to(device)
    
    # print("Starting training...")
    j=save_model_path.split("_")[-1].split(".")[0] # extract the #j div of the save path
    
    epoch_bar=tqdm.tqdm(range(0, epochs),
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
                    world_size=world_size)
        
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
                                        world_size=world_size)
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
                # "Vloss": f"{LOSSES[1][-1]:.2e}",
                "m5VL": f"{np.mean(LOSSES[1][-5:]):.2e}",
            }
        )

    COMP_LOSSES=np.array(COMP_LOSSES)
    
    return {"training_loss":LOSSES[0],
            "validation_loss":LOSSES[1],
            "learning_rate":LR,
            "training_loss_composition":COMP_LOSSES[0],
            "validation_loss_composition":COMP_LOSSES[1],
            "loss_weights":loss_func.weights}



   
    



    
    
################ SCRIPT FOR TRAINING ####################
## it allows to run track_fitting_net as a script directly    
    

    
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
    
    ## Replace the weights of the loss function
    loss_fn.weights=args.weights
    
    ## Get whether we will be using the baseline model or the transformer
    use_baseline=args.baseline
    
    
    print(f"Starting main worker on device {device}")
    
    # if we are in a multi_GPU training setup, we need to initialise the DDP (Distributed Data Parallel)
    if multi_GPU:
        ddp_setup(rank=device, world_size=world_size)
    
    
    if use_baseline:
        model=minkunet.MinkUNet34B(in_channels=x_in_channels, out_channels=y_out_channels, D=3).to(device)
    else:
        model = FittingTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                 d_model=D_MODEL,
                                 n_head=N_HEAD,
                                 input_size=3+x_in_channels,
                                 output_size=y_out_channels,
                                 dim_feedforward=DIM_FEEDFORWARD).to(device)
    
    print(model.__str__().split('\n')[0][:-1]) # Print the model name
    
    if multi_GPU:
        model=torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[device])
        # For multi-GPU training, DDP requires to change BatchNorm layers to SyncBatchNorm layers
        model=ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) if use_baseline else torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Print the total number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {pytorch_total_params}")
    

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
                            model_type= 'minkowski' if use_baseline else 'transformer',
                            warmup_scheduler=warmup_scheduler,
                            loss_func=loss_fn,
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
                            save_model_path=f"{args.save_path}models/trackfit_model_{'baseline_' if use_baseline else ''}{j}.torch")
    
    if multi_GPU:
        torch.distributed.destroy_process_group()
    
    return model, training_dict    
    
    
    


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
                        prog='TrackFittingTraining',
                        description='Trains a model for Track Fitting in SFG',)

    parser.add_argument('j', metavar='j', type=int, help='#j div of the model')
    parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for training")
    parser.add_argument('scaler_file',metavar='Scaler_File', type=str, help="File storing the dataset features scalers")
    parser.add_argument('save_path',metavar='Save_Path', type=str, help="Path to save results and models")
    parser.add_argument('-B', '--baseline', action='store_true', help='use the baseline model (MinkUNet), otherwise use the transformer')
    parser.add_argument('-m', '--multi_GPU', action='store_true', help='runs the script on multi GPU')
    parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the duration of the different parts of the code')
    parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('-n','--num_epochs', type=int, default=200, help='number of epochs for the training')
    parser.add_argument('--multi_pass', type=int, default=1, help='how many times the whole dataset is gone through in an epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='maximum learning rate for training (defines the scale of the learning rate)')
    parser.add_argument('--stop_after_epochs', type=int, default=40, help='maximum number of epochs without improvement before stopping the training (early termination)')
    parser.add_argument('-w','--weights', type=float, nargs=len(loss_fn), default=loss_fn.weights, help='weights for the loss functions')
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
    
    ## Replace the weights of the loss function
    loss_fn.weights=args.weights
    
    ## Get whether we will be using the baseline model or the transformer
    use_baseline=args.baseline



    def main():
        global multi_GPU, benchmarking
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if multi_GPU:
            print(f"Training on Multi GPU...")
        else:
            print(f"Training on Single GPU {device}...")
        if benchmarking:
            print("Benchmarking...")
        if multi_pass!=1:
            print(f"Multi pass {multi_pass}...")
        if use_baseline:
            print(f"Using baseline model...")

        # generate dataset
        dataset=PGunEvent(root=args.dataset_folder,
                        shuffle=True,
                        multi_pass=multi_pass,
                        files_suffix='npz',
                        scaler_file=args.scaler_file,
                        use_true_tag=True,
                        scale_coordinates=(not use_baseline),)

        t0=time.perf_counter()

        if multi_GPU:
            # from sfgnets.track_fitting_net import main_worker
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

        # torch.save(model.state_dict(), f"/scratch4/maubin/models/hittag_model_{j}.torch")
        torch.save(training_dict,f"{args.save_path}results/trackfit_training_dict_{'baseline_' if use_baseline else ''}{j}.torch")
        
    main()
