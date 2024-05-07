import torch
import os
os.environ["OMP_NUM_THREADS"]="16"
from sfgnets.dataset import *
from sfgnets.utils import minkunet
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import MinMaxScaler
import tqdm
from warmup_scheduler_pytorch import WarmUpScheduler
import time
import math
from copy import deepcopy

from sfgnets.plotting import plots_tf as plots

x_in_channels=2
y_out_channels=np.sum(PGunEvent.TARGETS_LENGTHS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################### TRANSFORMER MODEL #####################

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
    
    


def create_baseline_model(x_in_channels:int=x_in_channels,
                          y_out_channels:int=y_out_channels,
                          device:torch.device=device):

    return minkunet.MinkUNet34B(in_channels=x_in_channels, out_channels=y_out_channels, D=3).to(device)


def create_transformer_model(x_in_channels:int=x_in_channels,
                            y_out_channels:int=y_out_channels,
                            device:torch.device=device,
                            D_MODEL:int = 64,
                            N_HEAD:int = 8,
                            DIM_FEEDFORWARD:int = 128,
                            NUM_ENCODER_LAYERS:int = 5):

    return FittingTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                 d_model=D_MODEL,
                                 n_head=N_HEAD,
                                 input_size=3+x_in_channels,
                                 output_size=y_out_channels,
                                 dim_feedforward=DIM_FEEDFORWARD).to(device)
    



#################### LOSSES #####################    


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
        if type(key) is list:
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
    
    def rebuild_partial_losses(self):
        """
        In the case where a __getitem__ has been applied, the indexes of the partial losses might be wrong.
        This function redefines the indexes of the partial losses to make sure that they follow back to back
        """
        k_index,k_index_target=0,0
        for loss in self.losses:
            if type(loss) is PartialLoss:
                assert k_index==k_index_target, f"Mismatch of the indexes of prediction and targets for loss {loss} with values k_index: {k_index} and k_index_target: {k_index_target}"
                length=len(loss.indexes)
                loss.indexes=[k for k in range(k_index,k_index+length)]
                k_index+=length
                k_index_target=k_index
            elif type(loss) is PartialClassificationLoss:
                loss.index_target=k_index_target
                k_index_target+=1
                length=len(loss.indexes_pred)
                loss.indexes_pred=[k for k in range(k_index,k_index+length)]
                k_index+=length
    
    
    
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
    

class MomentumLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, angle_resolution:float = np.pi/100 , norm_resolution:float= np.sqrt(3)/2*1/10, reduction: str = 'mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.angle_resolution=angle_resolution
        self.norm_resolution=norm_resolution
    
    def forward(self, pred:Tensor, target:Tensor) -> Tensor:
        pred_momentum=pred-0.5
        target_momentum=target-0.5
        
        pred_norm=torch.linalg.norm(pred_momentum,dim=-1)
        target_norm=torch.linalg.norm(target_momentum,dim=-1)
        
        ## For the direction loss, we use the angle between the two vectors. First we compute the dot product of the two normalised vectors, then we compute its arccosinus to get the angle
        direction_loss=torch.acos(torch.sum(pred_momentum*target_momentum,dim=-1)/(pred_norm*target_norm+1e-9))
        
        ## For the norm loss, we compute the squared difference (MSE) of the two norms
        norm_loss=(pred_norm-target_norm)**2
        
        ## The total loss is the product of the two losses, with some constants to avoid one loss being neglected
        total_loss=((direction_loss**2+self.angle_resolution**2)*(norm_loss+self.norm_resolution**2)-(self.angle_resolution**2)*(self.norm_resolution**2))*1/(10*self.angle_resolution*10*self.norm_resolution)**2
    
        if self.reduction=='mean':
            return total_loss.mean()
        elif self.reduction=='sum':
            return total_loss.sum()
        else:
            return total_loss

class MomDirLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, dir_weight:float=25. , norm_weight:float=1e-2, reg_weight:float=1., reduction: str = 'mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.dir_weight=dir_weight
        self.norm_weight=norm_weight
        self.reg_weight=reg_weight
        self.loss_fn=torch.nn.MSELoss(reduction=reduction)
    
    def forward(self, pred:Tensor, target:Tensor) -> Tensor:
        pred_momentum=2.*(pred-0.5)
        target_momentum=2.*(target-0.5)
        
        pred_norm=torch.linalg.norm(pred_momentum,dim=-1)
        target_norm=torch.linalg.norm(target_momentum,dim=-1)
        dir_loss=(1.-torch.sum(pred_momentum*target_momentum,dim=-1)/(pred_norm*target_norm+1e-9)) if self.dir_weight>0. else torch.zeros_like(pred_norm)
        regularisation_loss=self.reg_weight*100.*torch.relu(pred_norm**2-4.)+torch.relu(1./pred_norm**2-1e8) if self.reg_weight>0. else torch.zeros_like(pred_norm)
        
        if self.reduction=='mean':
            dir_loss=dir_loss.mean()
            regularisation_loss=regularisation_loss.mean()
        elif self.reduction=='sum':
            dir_loss=dir_loss.sum()
            regularisation_loss=regularisation_loss.sum()

        norm_loss=self.norm_weight*self.loss_fn(pred,target) if self.norm_weight>0. else 0.
        
        return dir_loss+norm_loss+regularisation_loss
    
class MomSphLoss(MomDirLoss):
    
    def forward(self, pred:Tensor, target:Tensor) -> Tensor:
        
        pred_norm=pred[...,0]
        pred_theta=pred[...,1]
        pred_phi=pred[...,2]
        
        target_momentum=2.*(target-0.5)
        target_norm=torch.linalg.norm(target_momentum,dim=-1)
        target_2d_norm=torch.linalg.norm(target_momentum[...,0:2],dim=-1) if self.dir_weight>0. else torch.zeros_like(pred_norm)
        
        target_theta=torch.acos(target_momentum[...,2]/(target_norm+1e-9)) if self.dir_weight>0. else torch.zeros_like(pred_norm)
        target_phi=torch.sign(target_momentum[...,1])*torch.acos(target_momentum[...,0]/(target_2d_norm+1e-9)) if self.dir_weight>0. else torch.zeros_like(pred_norm)
        
        norm_loss=self.norm_weight*self.loss_fn(pred_norm,target_norm) if self.norm_weight>0. else 0.
        
        dir_loss=self.dir_weight*(1-torch.cos(target_theta-pred_theta))*(target_norm>1e-9) if self.dir_weight>0. else torch.zeros_like(pred_norm)
        dir_loss+=self.dir_weight*(1-torch.cos(target_phi-pred_phi))*(target_2d_norm>1e-9) if self.dir_weight>0. else torch.zeros_like(pred_norm)
        
        regularisation_loss=(pred_theta**2)*(pred_theta<0.)+((pred_theta-torch.pi)**2)*(pred_theta>torch.pi) if self.reg_weight>0. else torch.zeros_like(pred_norm)
        regularisation_loss+=(pred_phi**2)*(pred_phi<0.)+((pred_phi-2*torch.pi)**2)*(pred_phi>2*torch.pi) if self.reg_weight>0. else torch.zeros_like(pred_norm)
        regularisation_loss*=self.reg_weight*100
        
        if self.reduction=='mean':
            dir_loss=dir_loss.mean()
            regularisation_loss=regularisation_loss.mean()
        elif self.reduction=='sum':
            dir_loss=dir_loss.sum()
            regularisation_loss=regularisation_loss.sum()
        
        
        return dir_loss+norm_loss+regularisation_loss
        
        

    
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







loss_fn=SumLoss(losses=[
                            PartialLoss(nn.MSELoss(), [0,1,2]), # node position
                            PartialLoss(nn.MSELoss(), [3,4,5]), # node momentum
                            ],
                weights=[
                            1.e4, # node position
                            1.e2, # node momentum
                        ])

loss_fn_mom_loss=SumLoss(losses=[
                            PartialLoss(nn.MSELoss(), [0,1,2]), # node position
                            PartialLoss(MomentumLoss(), [3,4,5]), # node momentum
                            ],
                weights=[
                            1.e4, # node position
                            1.e2, # node momentum
                        ])

loss_fn_momdir_loss=SumLoss(losses=[
                            PartialLoss(nn.MSELoss(), [0,1,2]), # node position
                            PartialLoss(MomDirLoss(dir_weight=1.,norm_weight=0.,reg_weight=0.), [3,4,5]), # node momentum direction
                            PartialLoss(MomDirLoss(dir_weight=0.,norm_weight=0.01,reg_weight=0.), [3,4,5]), # node momentum norm
                            PartialLoss(MomDirLoss(dir_weight=0.,norm_weight=0.,reg_weight=1.), [3,4,5]), # node momentum regularisation
                            ],
                weights=[
                            1.e4, # node position
                            1.e2, # node momentum direction
                            1.e4, # node momentum norm
                            1.e2, # node momentum regularisation
                        ])

loss_fn_momsph=SumLoss(losses=[
                            PartialLoss(nn.MSELoss(), [0,1,2]), # node position
                            PartialLoss(MomSphLoss(dir_weight=1.,norm_weight=0.,reg_weight=0.), [3,4,5]), # node momentum direction
                            PartialLoss(MomSphLoss(dir_weight=0.,norm_weight=0.01,reg_weight=0.), [3,4,5]), # node momentum norm
                            PartialLoss(MomSphLoss(dir_weight=0.,norm_weight=0.,reg_weight=1.), [3,4,5]), # node momentum regularisation
                            ],
                weights=[
                            1.e4, # node position
                            1.e2, # node momentum direction
                            1.e2, # node momentum norm
                            1.e2, # node momentum regularisation
                        ])

perf_fn=SumPerf.from_SumLoss(loss_fn)



#################### TRAINING DEFAULT PARAMETERS #####################    


baseline_model=create_baseline_model()
transformer_model=create_transformer_model()

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




#################### DATALOADER FUNCTIONS #####################    



PREMASKING=True

def filtering_events(x):
    if PREMASKING:
        return x['c'] is not None and len(x['c'][x['mask']>0])>2 and len(x['c'][x['mask']>0])<1000
    else:
        return x['c'] is not None and len(x['c'])<1000

PAD_IDX=-1.

# function to collate data samples for a transformer
def collate_transformer(batch):
    """
    Custom collate function for Transformers
    """
    
    device=batch[0]['x'].device
    
    if PREMASKING:
        coords = [d['c'][d['mask']>0] for d in filter(filtering_events,batch)]

        feats = [torch.cat([d['c'][d['mask']>0],d['x'][d['mask']>0]],dim=-1) for d in filter(filtering_events,batch)] # saul's order
        
        targets = [d['y'][d['mask']>0] for d in filter(filtering_events,batch)]
        
        aux = [d['aux'][d['mask']>0] for d in filter(filtering_events,batch)]
        
        # try:
        #     aux = [d['aux'][d['mask']>0] for d in filter(filtering_events,batch)]
        # except IndexError: # if aux are None, we will get an IndexError here
        #     aux = [None for d in filter(filtering_events,batch)]
        
        masks = [d['mask'][d['mask']>0] for d in filter(filtering_events,batch)]
        
    
    else:
        coords = [d['c'] for d in filter(filtering_events,batch)]

        # feats = [torch.cat([d['x'],d['c']],dim=-1) for d in filter(filtering_events,batch)] # order for the first 5 models
        feats = [torch.cat([d['c'],d['x']],dim=-1) for d in filter(filtering_events,batch)] # saul's order

        targets = [d['y'] for d in filter(filtering_events,batch)]

        aux = [d['aux'] for d in filter(filtering_events,batch)]
        
        masks = [d['mask'] for d in filter(filtering_events,batch)]
    

    lens = [len(x) for x in feats]
    try:
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=False, padding_value=PAD_IDX)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=False, padding_value=PAD_IDX)
        coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=False, padding_value=PAD_IDX)
        try:
            aux = torch.nn.utils.rnn.pad_sequence(aux, batch_first=False, padding_value=PAD_IDX)
        except TypeError: # if the aux variables are None, the cat will raise a type error
            aux = None
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=False, padding_value=False)

        return {'f':feats, 'y':targets, 'aux':aux, 'c':coords, 'mask':masks, 'lens':lens,}
    
    except RuntimeError:
        return{'f':None, 'y':None, 'aux':None, 'c':None, 'mask':None, 'lens':None,}


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
    
    if PREMASKING:
        # Extract coordinates from the batch and convert to integer type
        coords = [d['c'][d['mask']>0].int() for d in filter(filtering_events,batch)]

        # Concatenate features from the batch
        feats = torch.cat([d['x'][d['mask']>0] for d in filter(filtering_events,batch)])

        # Concatenate labels from the batch
        y = torch.cat([d['y'][d['mask']>0] for d in filter(filtering_events,batch)])
        
        # Concatenate aux from the batch
        try:
            aux = torch.cat([d['aux'][d['mask']>0] for d in filter(filtering_events,batch)])
        except (TypeError, IndexError): # if the aux variables are None, the cat will raise a type error
            aux = None
        
        # Concatenate masks from the batch
        masks = torch.cat([d['mask'][d['mask']>0] for d in filter(filtering_events,batch)])
    
    else:
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



#################### TRAINING FUNCTIONS #####################    

def execute_model(model:torch.nn.Module,
                data:dict,
                model_type:str='minkowski', # or 'transformer'
                device:torch.device=device,
                do_we_consider_aux:bool=False,
                do_we_consider_coord:bool=False,
                do_we_consider_feat:bool=False,
                do_we_consider_event_id:bool=False,
                last_event_id:int=None,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """
    Returns the prediction, target and mask for a given model. Adapts whether it is 'minkowski' type (SparseTensors) or 'transformer' type (PaddedSequence).
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
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          warmup_scheduler:WarmUpScheduler,
          model_type:str='minkowski', # or 'transformer'
          loss_func:SumLoss=SumLoss([torch.nn.MSELoss()]),
          device:torch.device=device, 
          progress_bar:bool=False,
          benchmarking:bool=False,
          world_size:int=1,
          notebook_tqdm:bool=False,):
    
    
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
          world_size:int=1,
          notebook_tqdm:bool=False,):
    
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
            loader:torch.utils.data.DataLoader,
            model_type:str='minkowski', # or 'transformer'
            device:torch.device=device, 
            progress_bar:bool=False,
            do_we_consider_aux:bool=False,
            do_we_consider_coord:bool=False,
            do_we_consider_feat:bool=True,
            do_we_consider_event_id:bool=True,
            max_batches:int=None,) -> dict[list[np.ndarray]]:
    
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
            save_model_path:str=None,
            num_workers:int=24,
            notebook_tqdm:bool=False,
            ):
    
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
    j=save_model_path.split("_")[-1].split(".")[0] # extract the #j div of the save path
    
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



   
    



    
    
################ SCRIPT FOR TRAINING ####################
## it allows to run track_fitting_net as a script directly    
    

    
def main_worker(device:torch.device,
                dataset:PGunEvent,
                args:dict,
                world_size:int=1,
                multi_GPU:bool=False):
    global loss_fn
    
    #### Get the variables from args
    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    ## Get the multi_GPU flag
    multi_GPU=args.multi_GPU

    ## Get the benchmarking flag
    benchmarking=args.benchmarking
    sub_progress_bars=args.sub_tqdm

    multi_pass=args.multi_pass
    
    if args.mom_loss:
        # loss_fn.losses[1].loss_func=MomentumLoss()
        loss_fn=loss_fn_mom_loss
    elif args.momdir_loss:
        # loss_fn.losses[1].loss_func=MomDirLoss()
        loss_fn=loss_fn_momdir_loss
    elif args.momsph:
        # loss_fn.losses[1].loss_func=MomSphLoss(dir_weight=1.,norm_weight=1e-2)
        loss_fn=loss_fn_momsph
    
    if args.targets is not None:
        ## Select only the targets for the loss function
        loss_fn=loss_fn[dataset.targets]
        
        ## Replace the weights of the loss function
        loss_fn.weights=[args.weights[k] for k in dataset.targets]
    
    # ## Reconstruct the partial loss indexes, was useful when using one loss per target
    # if args.targets is not None:
    #     loss_fn.rebuild_partial_losses()
    
    ## Get whether we will be using the baseline model or the transformer
    use_baseline=args.baseline
    
    
    print(f"Starting main worker on device {device}")
    
    # if we are in a multi_GPU training setup, we need to initialise the DDP (Distributed Data Parallel)
    if multi_GPU:
        ddp_setup(rank=device, world_size=world_size)
    
    
    if use_baseline:
        model=create_baseline_model(y_out_channels=sum(dataset.targets_lengths)+sum(dataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
    else:
        model=create_transformer_model(y_out_channels=sum(dataset.targets_lengths)+sum(dataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
    
    print(model.__str__().split('\n')[0][:-1]) # Print the model name
    
    if args.resume:
        print("Loading model state dict save...")
        model.load_state_dict(torch.load(f"{args.save_path}models/trackfit_model_{'baseline_' if use_baseline else ''}{j}.torch"))
    
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
    num_steps_one_cycle = 120
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
    parser.add_argument('--test_only', action='store_true', help='runs only the test (measure performances, plots, ...)')
    parser.add_argument('-T', '--test', action='store_true', help='runs test after training (measure performances, plots, ...)')
    parser.add_argument('-B', '--baseline', action='store_true', help='use the baseline model (MinkUNet), otherwise use the transformer')
    parser.add_argument('-R', '--resume', action='store_true', help='resume the training by loading the saved model state dictionary')
    parser.add_argument('-m', '--multi_GPU', action='store_true', help='runs the script on multi GPU')
    parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the duration of the different parts of the code')
    parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('-n','--num_epochs', type=int, default=200, help='number of epochs for the training')
    parser.add_argument('--multi_pass', type=int, default=1, help='how many times the whole dataset is gone through in an epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='maximum learning rate for training (defines the scale of the learning rate)')
    parser.add_argument('--stop_after_epochs', type=int, default=40, help='maximum number of epochs without improvement before stopping the training (early termination)')
    parser.add_argument('-w','--weights', type=float, nargs=len(loss_fn), default=loss_fn.weights, help='weights for the loss functions')
    parser.add_argument('-t','--targets', type=int, nargs="*", default=None, help='the target indices to include')
    parser.add_argument('-i','--inputs', type=int, nargs="*", default=None, help='the input indices to include')
    parser.add_argument('--ms', type=str, default='distance', metavar='Masking Scheme', help='Which mask to use for the dataset (distance, primary, tag, ...)')
    parser.add_argument('--mom_loss', action='store_true', help='use the momentum loss instead of the simple MSE')
    parser.add_argument('--momdir_loss', action='store_true', help='use the momentum-direction loss instead of the simple MSE')  
    parser.add_argument('--momsph', action='store_true', help='use spherical coordinates for the momentum')  
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
    
    ## Get whether we will be using the baseline model or the transformer
    use_baseline=args.baseline
    



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
            if use_baseline:
                print(f"Using baseline model...")

            # generate dataset
            dataset=PGunEvent(root=args.dataset_folder,
                            shuffle=True,
                            multi_pass=multi_pass,
                            files_suffix='npz',
                            scaler_file=args.scaler_file,
                            use_true_tag=True,
                            scale_coordinates=(not use_baseline),
                            targets=args.targets,
                            inputs=args.inputs,
                            masking_scheme=args.ms)
            
            if args.targets is not None or args.weights!=loss_fn.weights:
                print(f"Selected targets are: "+"".join([f"{dataset.targets_names[k]} ({args.weights[dataset.targets[k]]:.1e})  " for k in range(len(dataset.targets_names))]))

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
        
        if args.test or args.test_only:
            print("Testing the model...")
            testdataset_folder=args.dataset_folder[:-6]+"test/" ## we are assuming that the dataset_folder is of type "*_train/" whereas the testdataset folder will be "*_test/"
            
            testdataset=PGunEvent(root=testdataset_folder,
                        shuffle=False,
                        multi_pass=multi_pass,
                        files_suffix='npz',
                        scaler_file=args.scaler_file,
                        use_true_tag=True,
                        scale_coordinates=(not use_baseline),
                        targets=args.targets,
                        inputs=args.inputs,
                        masking_scheme=args.ms)
            
            if use_baseline:
                model=create_baseline_model(y_out_channels=sum(testdataset.targets_lengths)+sum(testdataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
            else:
                model=create_transformer_model(y_out_channels=sum(testdataset.targets_lengths)+sum(testdataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
            
            model.load_state_dict(torch.load(f"{args.save_path}models/trackfit_model_{'baseline_' if use_baseline else ''}{j}.torch"))
            
            full_loader=full_dataset(testdataset,
                                    collate=collate_minkowski if use_baseline else collate_transformer,
                                    batch_size=args.batch_size,)
            
            all_results=test_full(loader=full_loader,
                                                    model=model,
                                                    model_type="minkowski" if use_baseline else "transformer",
                                                    progress_bar=True,
                                                    device=device,
                                                    do_we_consider_aux=True,
                                                    do_we_consider_coord=True,
                                                    do_we_consider_feat=True,
                                                    # max_batches=10,
                                                    )
            perf_fn=SumPerf.from_SumLoss(loss_fn)

            if args.targets is not None:
                if args.mom_loss:
                    loss_fn.losses[1].loss_func=MomentumLoss()
                perf_fn=perf_fn[args.targets]
                # perf_fn.rebuild_partial_losses()

            all_results=measure_performances(all_results,
                                            testdataset,
                                            perf_fn,
                                            device, 
                                            model_type="minkowski" if use_baseline else "transformer",
                                            do_we_consider_aux=True,
                                            do_we_consider_coord=True,
                                            do_we_consider_feat=True,
                                            mom_spherical_coord=args.momsph)

            with open(f'{args.save_path}results/track_fitting_model_{"baseline_" if use_baseline else ""}{j}_pred.pkl', 'wb') as file:
                    pk.dump(all_results,file)
                    
            plots.plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"baseline_" if use_baseline else ""}{j}.png',
                        model_name=str(j),
                        show=False,
                        )
            
            if args.targets is None or 1 in args.targets:
                plots.plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"baseline_" if use_baseline else ""}{j}.png',
                        model_name=str(j),
                        mode='mom',
                        show=False,
                        )
                
                plots.plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"baseline_" if use_baseline else ""}{j}.png',
                        model_name=str(j),
                        mode='mom_d',
                        show=False,
                        )
                
                plots.plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"baseline_" if use_baseline else ""}{j}.png',
                        model_name=str(j),
                        mode='mom_n',
                        show=False,
                        )

    main()
