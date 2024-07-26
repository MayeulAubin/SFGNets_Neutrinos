import numpy as np
import pickle as pk
from sklearn.preprocessing import MinMaxScaler
import torch
from glob import glob
import MinkowskiEngine as ME
from numpy.lib import npyio
import tqdm

from ..datasets.dataclass import EventDataset
from ..datasets.constants import RANGES, CUBE_SIZE
from ..datasets.utils import full_dataset, transform_cube, transform_inverse_cube, filtering_events, split_dataset
from ..datasets.utils_minkowski import arrange_sparse_minkowski, arrange_truth, arrange_aux


particles_classes={0:0, # no particle, or low energy neutrons
                   11:1, # e+
                   -11:2, # e-
                   22:3, # gamma
                   13:4, # mu+
                   -13:5, # mu-
                   2212:6, # p
                   211:7, # pi+
                   -211:8, # pi-
                   2112:9, # n
                   999:10, # particle different from the above ones
                   }


class PGunEvent(EventDataset):
    
    TARGETS_NAMES=["position", "momentum"]
    TARGETS_LENGTHS=[3,3]
    TARGETS_N_CLASSES=[0,0]
    INPUT_NAMES=["charge","hittag"]
    
    def __init__(self,
                 scaler_file:str|None=None,
                 use_true_tag:bool=True,
                 scale_coordinates:bool=False,
                 targets:int|slice|list[int]|None=None,
                 inputs:int|slice|list[int]|None=None,
                 masking_scheme:str='distance',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.variables=['x','y','c','mask','aux']
        
        self.use_true_tag=use_true_tag
        self.scale_coordinates=scale_coordinates
        self.targets=targets
        self.inputs=[inputs] if type(inputs) is int else inputs
        self.masking_scheme=masking_scheme
        
        self.def_local_targets_variables(targets)
        
        if scaler_file is not None:
            self.use_scaler=True
            with open(scaler_file, "rb") as fd:
                self.scaler_x, self.scaler_y, self.scaler_c = pk.load(fd)
        else:
            self.use_scaler=False
            
            
            
            
    def def_local_targets_variables(self, targets:int|slice|list[int]|None):
        """
        This function was particularly helpful when using a lot of different targets, some of regression type other of classifications.
        It builds the local targets_names, targets_lengths, targets_n_classes (if classification task), and the y_indexes
        """
        
        if targets is None: # then all targets are used, defined by the class variables
            self.targets_names=self.__class__.TARGETS_NAMES # names of the targets
            self.targets_lengths=self.__class__.TARGETS_LENGTHS # length of each target (1 if scalar, 3 if 3D vector, ...)
            self.targets_n_classes=self.__class__.TARGETS_N_CLASSES # number of classes of the targets (non zero for classification targets)
            self.y_indexes=None # the indexes of the targets of interest out of the full target vector
            self.y_indexes_with_scale=None # the indexes of the targets of interest requirering a scale (not classification) out of the full target vector
        else: # else the chosen targets are used
            if targets is int:
                targets=[targets]
            self.targets_names=list(np.array(self.__class__.TARGETS_NAMES)[targets])
            self.targets_lengths=list(np.array(self.__class__.TARGETS_LENGTHS)[targets])
            self.targets_n_classes=list(np.array(self.__class__.TARGETS_N_CLASSES)[targets])
            indices_cumsum=[0]+list(np.cumsum(self.__class__.TARGETS_LENGTHS)) # this list we count the start indexes of all targets
            self.y_indexes,self.y_indexes_with_scale=[],[]
            for i in targets:
                self.y_indexes+=list(range(indices_cumsum[i],indices_cumsum[i+1]))
                if self.__class__.TARGETS_N_CLASSES[i]==0:
                    self.y_indexes_with_scale+=list(range(indices_cumsum[i],indices_cumsum[i+1]))
                    
    
    def getx(self, data:npyio.NpzFile):
        
        charge=data['x'][:,1]
        
        if self.use_true_tag:
            tag=data['tag']
        else:
            tag=data['tag_pred']
            
        x=np.zeros((charge.shape[0],2))
        x[:,0]=charge
        x[:,1]=tag-1
        
        if self.use_scaler:
            x = self.scaler_x.transform(x)
        
        if self.inputs is not None:
            x=x[:,self.inputs]
        
        return torch.FloatTensor(x)
    
    def gety(self, data:npyio.NpzFile):
        
        position=data['node_c']
        direction=data['node_d']
        momentum=data["node_m"]
        
        y=np.zeros((len(position),3+3))
        y[:,0:3]=position-data['c'] # it will be the relative position of the 'node' compared to the cube
        y[:,3:6]=momentum
        # y[:,3:6]=direction
        
        if self.use_scaler:    
            y[:,:10]=self.scaler_y.transform(y[:,:10]) # all but particle class (no scaling for classes)
        
        if self.y_indexes is not None:
            y=y[:,self.y_indexes] # we keep only the targets of interest selected with y_indexes
        
        return torch.FloatTensor(y)
    
    def getc(self,data:npyio.NpzFile):
        c = data['c']  # 3D coordinates (cube raw positions)
        
        if c.shape[0]==0:
            return None
        
        if self.scale_coordinates:
            if self.use_scaler:
                c=self.scaler_c.transform(c) # scale coordinates for the Transformer
        else:
            # Convert cube raw positions to cubes (for Minkowksi convolution)
            c=transform_cube(c)
        
        return torch.FloatTensor(c)
    
    def getmask(self, data:npyio.NpzFile):
        
        true_tag=data['tag']
        number_of_segments=data['node_n']
        distance_node_point=data["distance_node_point"]
        threshold_distance=40 # threshold distance between center of cube and trajectory point of 40 mm 

        if self.masking_scheme=='distance':
            ## The mask is defined as the hits which have a segment inside, or the hits whithout segments but closer than the threshold distance from their associated trajectory point
            return torch.BoolTensor(~((number_of_segments==0)*(distance_node_point>threshold_distance))) 
        elif self.masking_scheme=='tag':
            ## Remove noise hits
            return torch.BoolTensor(true_tag!=3)
        elif self.masking_scheme=='primary':
            ## Keeps only the primary trakectory
            return torch.BoolTensor(data["traj_parentID"]==0)
        elif self.masking_scheme=='segmented':
            ## Select randomly a trajectory and mask others, a weighting by the number of hits is embedded in the selection
            selected_traj_ID=np.random.choice(data['traj_ID'])
            return torch.BoolTensor(data["traj_ID"]==selected_traj_ID)
        else:
            raise ValueError(f"Masking scheme not recognized {self.masking_scheme}")
    
    def getaux(self, data:npyio.NpzFile):
        
        input_particle=data["input_particle"]
        # NTraj=data["NTraj"]
        traj_ID=data["traj_ID"]
        traj_parentID=data["traj_parentID"]
        # distance_node_point=data["distance_node_point"]
        distance_node_point=np.linalg.norm(data["node_c"][:,0:3]-data["c"],axis=-1)
        momentum=data["node_m"]
        tag=data['tag']
        number_of_particles=data['node_n']
        energy_deposited=data['Edepo']
        particle_pdg=data['pdg']
        direction=data['node_d']
        traj_length=data['traj_length']
        event_entry=data['event_entry']
        recon_c=data['recon_c']
        recon_d=data['recon_d']
        order_index=data['order_index']
        particle_charge=data['p_charge']
        
        aux=np.zeros((tag.shape[0],25))
        aux[:,0]=input_particle
        aux[:,1]=traj_ID
        aux[:,2]=traj_parentID
        aux[:,3]=distance_node_point
        aux[:,4:7]=momentum
        aux[:,7]=tag
        aux[:,8]=number_of_particles
        aux[:,9]=energy_deposited
        aux[:,10]=particle_pdg
        aux[:,11:14]=direction
        aux[:,15]=traj_length
        aux[:,16]=event_entry
        aux[:,17:20]=recon_c
        aux[:,20:23]=recon_d
        aux[:,23]=order_index
        aux[:,24]=particle_charge
        
        return torch.FloatTensor(aux)




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

def filtering_events(x:dict) -> bool:
    """
    Filters events based on the following criteria:
        1. Not a "None" event
        2. Has at least 2 hits after masking (if PREMASKING) 
        3. Has less than 1000 hits
    
    Parameters:
    - x: dict, event dictionnary with 'c' (coordinates) and 'mask' (mask) keys
    
    Returns:
    - bool, True if the event passes the filters
    """
    if PREMASKING:
        return x['c'] is not None and len(x['c'][x['mask']>0])>2 and len(x['c'][x['mask']>0])<1000
    else:
        return x['c'] is not None and len(x['c'])<1000

PAD_IDX=-1.

# function to collate data samples for a transformer
def collate_transformer(batch):
    """
    Custom collate function for Transformers.
    
    Parameters:
    - batch: list, list of event dictionaries
    
    Returns:
    - batch_dict: dict, dictionnary of torch Tensors padded for the transformer
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
    
    Parameters:
    - src: torch.Tensor, source tensor
    
    Returns:
    - src_mask: torch.Tensor, source mask
    - src_padding_mask: torch.Tensor, padding mask
    """
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
    src_padding_mask = (src[:, :, 0] == PAD_IDX).transpose(0, 1).to(src.device)

    return src_mask, src_padding_mask

