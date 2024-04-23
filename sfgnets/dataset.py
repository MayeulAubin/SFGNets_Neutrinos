
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle as pk
import random
import re
import torch
import MinkowskiEngine as ME
from torch.utils.data import random_split
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
import tqdm


RANGES = np.array([[ -985.92 ,   985.92 ],
                   [ -257.56 ,   317.56 ],
                   [-2888.776,  -999.096]])  # detector ranges (X, Y, Z)
CUBE_SIZE = 10.27 / 2  # half of cube size in mm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def natural_sort(l:list):
    """
    Perform a natural sort on the given list.

    Parameters:
    - l: list, input list to be sorted

    Returns:
    - list, sorted list
    """
    # Function to convert text to a numeric value for sorting
    convert = lambda text: int(text) if text.isdigit() else text.lower()

    # Function to create a key for sorting using the converted values
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    # Sort the list using the created key
    return sorted(l, key=alphanum_key)


# convert raw coordinates to cubes
def transform_cube(X:np.ndarray):
    """
    Convert raw coordinates to cubes.
    
    Parameters:
    - X: numpy array
    - dim: int, dimension (2 or 3)
    """
    X-=RANGES[None,:,0]
    X/=(CUBE_SIZE * 2)
    return X.astype(int)

def transform_inverse_cube(X:np.ndarray):
    """
    Convert cubes to raw coordinates.
    
    Parameters:
    - X: numpy array
    """
    X=X.astype(float)
    X+=0.5
    X = X * (CUBE_SIZE * 2) + RANGES[None,:,0]
    return X
        

def collate_default(batch):
    """
    Custom collate function for Sparse Minkowski network.

    Parameters:
    - batch: list, a list of dictionaries

    Returns:
    - dict, a dictionary for the batch
    """
    ret={}
    
    for key in batch[0]:
        ret[key]=torch.cat([d[key] for d in filter(lambda x: x['c'] is not None,batch)])

    return ret


def retag_hits_based_on_distance(data:np.lib.npyio.NpzFile,cut:float):
    y=data['y']-1
    indexes=(np.linalg.norm(data['c']-data['verPos'][None,:], axis=1)>cut)*(y==0) # indexes of VA hits further than the cut distance
    indexes_sp=indexes*(data['pdg']!=0)
    indexes_no=indexes*(data['pdg']==0)
    y[indexes_sp]=1
    y[indexes_no]=2
    return y

def retag_cut(cut:float):
    return lambda data: retag_hits_based_on_distance(data, cut=cut)



class EventDataset(Dataset, metaclass=ABCMeta):
    
    def __init__(self, 
                 root:str, 
                 shuffle:bool=False,
                 multi_pass:int=1,
                 files_suffix:str='npz',
                 filtering_func=None,
                 **kwargs):            
        '''
        Initializer for EventDataset class.

        Parameters:
        - root: str, root directory
        - shuffle: bool, whether to shuffle data_files
        - multi_pass: int, how many times should the files be gone through, to allow better performance when multi_GPU by reducing the number of epochs accordingly
        - files_suffix: str, the file type to load ('npz', 'pk', ...)
        '''
        
        self.variables=['x','y','c','aux']
        
        self.root = root
        self.files_suffix=files_suffix
        self.data_files = self.processed_file_names
        self.multi_pass=multi_pass
        
        if shuffle:
            random.shuffle(self.data_files)
            
        self.total_events = len(self.data_files)
        
        self.filtering=(filtering_func is not None)
        self.filtering_func=filtering_func
    
    @property
    def processed_dir(self):
        return f'{self.root}'

    @property
    def processed_file_names(self):
        return natural_sort(glob(f'{self.processed_dir}/*.{self.files_suffix}'))
    
    def __len__(self):
        return self.total_events*self.multi_pass
    
    @abstractmethod
    def getx(self,data:np.lib.npyio.NpzFile):
        return None
    
    @abstractmethod
    def gety(self,data:np.lib.npyio.NpzFile):
        return None
    
    @abstractmethod
    def getc(self,data:np.lib.npyio.NpzFile):
        return None
    
    @abstractmethod
    def getaux(self,data:np.lib.npyio.NpzFile):
        return None
    

    def __getitem__(self, idx:int):
        
        # Load data from the file
        data = np.load(self.data_files[idx%self.total_events])
        
        return_dict={}
        
        try:
            # Extract the data
            for name in self.variables: # iterate over the variables 'x', 'y', 'c', ...
                return_dict[name]=getattr(self,'get'+name)(data) # get the variable (features x, targets y, coordinates c, auxiliary aux)
        except Exception as E:
            print(f"Error extracting data for index {idx}, file {self.data_files[idx%self.total_events]}")
            raise E
        
        # If necessary, filter the variables based on the filtering function
        if self.filtering:
            for name in self.variables: # iterate over the variables 'x', 'y', 'c', ...
                try:
                    if return_dict[name] is not None: # check that the variable is not None, and is actually a tensor/array
                        return_dict[name]=return_dict[name][self.filtering_func(data)] # use self.filtering_func(data) as indexes to be kept
                except Exception as E:
                    print(f"Error filtering data for variable {name}, index {idx}, file {self.data_files[idx%self.total_events]}")

        # Clean upthe data
        del data
        
        # Check that none of the data is None, if it is return None for all variables
        if return_dict['x'] is None or return_dict['y'] is None or return_dict['c'] is None:
            for key in return_dict.key():
                return_dict[key]=None # then set all variables to None
        
        return return_dict
    
    def all_data(self,
                 progress_bar:bool=True,
                 max_index:int=None,):
        
        if max_index is None:
            max_index=len(self)
        
        return_dict={}
        
        for i in tqdm.tqdm(range(max_index), disable=not progress_bar):
            dat_dict=self[i]
            
            if i==0:
                for key in dat_dict.keys():
                    return_dict[key]=[dat_dict[key]]
            else:
                for key in dat_dict.keys():
                    return_dict[key].append(dat_dict[key])
        
        for key in return_dict.keys():
            return_dict[key]=np.concatenate(return_dict[key],axis=0)
        
        return return_dict
    
    @property
    def data(self):
        return self.all_data(progress_bar=False)
    
    
    
# Dataset class for SFG dataset ('new')
class SparseEvent(EventDataset):
    def __init__(self, 
                 root:str, 
                 shuffle:bool=False, 
                 scaler_file:str=None, 
                 multi_pass:int=1, 
                 recon_ver:bool=False,
                 aux:bool=False,
                 center_event:bool=False,
                 scale_coordinates:bool=True,
                 retagging_func=None,
                 **kwargs):            
        '''
        Initializer for SparseEvent class.

        Parameters:
        - root: str, root directory
        - shuffle: bool, whether to shuffle data_files
        - multi_pass: int, how many times should the files be gone through, to allow better performance when multi_GPU by reducing the number of epochs accordingly
        - recon_ver: bool, whether to use the reconstructed vertex position (if True) or the true vertex position (if False)
        - aux: bool, whether to load the auxiliary variables in the dataset
        - center_event: bool, whether the event should be centered around the vertex (coordinates 0 0 0 afterwards)
        - scale_coordinates: bool, whether to transform the coordinates XYZ of the hit to cube indices XYZ (int)
        - retagging_func: function, if provided, function applied on data that returns the new tags y
        '''
        super().__init__(root=root,
                         shuffle=shuffle,
                         multi_pass=multi_pass,
                         **kwargs)
        self.aux=aux
        self.recon_ver=recon_ver
        self.center_event=center_event
        self.scale_coordinates=scale_coordinates

        if scaler_file is not None:
            self.use_scaler=True
            with open(scaler_file, "rb") as fd:
                self.scaler_minmax, self.scaler_stan = pk.load(fd)
        else:
            self.use_scaler=False
            
        if retagging_func is not None:
            self.retagging=True
            self.retagging_func=retagging_func
        else:
            self.retagging=False
        
    
    def getx(self,data):
        # Extract raw data
        x_0 = data['x']  # HitTime, HitCharge
        
        if x_0.shape[0]==0: # Checking if the event is empty
            return None
        
        c = data['c']  # 3D coordinates (cube raw positions)
        
        # True vertex position
        if self.recon_ver:
            verPos = data['recon_verPos'] # use the reconstructed vertex position
        else:
            verPos = data['verPos'] # use the true vertex position
        
        x=np.zeros(shape=(x_0.shape[0], 4))
        x[:,0]=x_0[:,1] # have to remove 'HitTime'
        # Add as features the distance to the vertex position
        x[:, -3] = c[:, 0] - verPos[0]
        x[:, -2] = c[:, 1] - verPos[1]
        x[:, -1] = c[:, 2] - verPos[2]
        
        # Standardize dataset
        if self.use_scaler:
            x = self.scaler_minmax.transform(x) 
        
        return torch.FloatTensor(x)
    
    
    def getc(self,data:np.lib.npyio.NpzFile):
        c = data['c']  # 3D coordinates (cube raw positions)
        
        if c.shape[0]==0:
            return None
        
        # Convert cube raw positions to cubes
        if self.scale_coordinates:
            c=transform_cube(c) 
        
        # center the event if necessary
        if self.center_event:
            verPos = data['recon_verPos'] if self.recon_ver else data['verPos']
            verPos=transform_cube(verPos) # beware this changes the vertex position array, might change it in the data dict too, beware to not apply this before getx
            c-=verPos
        
        return torch.FloatTensor(c)
    
    def gety(self,data:np.lib.npyio.NpzFile):
        if self.retagging:
            return torch.LongTensor(self.retagging_func(data))
        return torch.LongTensor(data['y'] - 1)
    
    def getaux(self,data:np.lib.npyio.NpzFile):
        if self.aux:
            pdg=data['pdg']
            reaction_code=int(data['reaction_code'])*np.ones_like(pdg)
            aux=np.zeros((len(pdg),2))
            aux[:,0]=pdg
            aux[:,1]=reaction_code
            return torch.Tensor(aux)
        else:
            return None
        

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
                 scaler_file:str=None,
                 use_true_tag:bool=True,
                 scale_coordinates:bool=False,
                 targets:int|slice|list[int]=None,
                 inputs:int|slice|list[int]=None,
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
                    
    
    def getx(self, data:np.lib.npyio.NpzFile):
        
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
    
    def gety(self, data:np.lib.npyio.NpzFile):
        
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
    
    def getc(self,data:np.lib.npyio.NpzFile):
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
    
    def getmask(self, data:np.lib.npyio.NpzFile):
        
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
    
    def getaux(self, data:np.lib.npyio.NpzFile):
        
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
    
    
    

def split_dataset(dataset:Dataset,
                  batch_size:int = 32,
                  train_fraction:float=0.8,
                  val_fraction:float=0.19,
                  seed:int=7,
                  multi_GPU:bool=False,
                  num_workers:int=24,
                  collate=collate_default,):

    # Get the length of the dataset
    fulllen = len(dataset)

    # Split the dataset into train, validation, and test sets
    train_len = int(fulllen * train_fraction)
    val_len = int(fulllen * val_fraction)
    test_len = fulllen - train_len - val_len

    # Use random_split to create DataLoader datasets
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoader instances for train, validation, and test sets
    if multi_GPU:
        train_loader = DataLoader(
            train_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(train_set)
        )
        valid_loader = DataLoader(
            val_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(val_set)
        )
        test_loader = DataLoader(
            test_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(test_set)
        )
        
    else:
        train_loader = DataLoader(
            train_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        valid_loader = DataLoader(
            val_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        test_loader = DataLoader(
            test_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
    
    return  train_loader, valid_loader, test_loader




def full_dataset(dataset:Dataset,
                  batch_size = 512,
                  multi_GPU=False,
                  num_workers=24,
                  collate=collate_default,
                 ):
    
    if multi_GPU:
        full_loader = DataLoader(
            dataset, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset),
            )
    else:
        full_loader = DataLoader(
            dataset, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False
            )
    
    return full_loader
        

# Function to arrange sparse minkowski data
def arrange_sparse_minkowski(data:dict,
                             device=device):
    return ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    
# Function to arange the truth in sparse Minkowski
def arrange_truth(data:dict,
                  device=device):
    if len(data['y'].shape)==2: # checking if the data has already been expanded of one dimension
        y=data['y']
    elif len(data['y'].shape)==1: # if not, expand the data
        y=data['y'][:,None]
    else: 
        raise ValueError(f"y has not a compatible shape: {y.shape}")
        
    return ME.SparseTensor(
        features=y,
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )


def arrange_aux(data:dict,
                device=device):
    
    if data['aux'] is not None:
    
        return ME.SparseTensor(
            features=data['aux'],
            coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
            device=device
        )
        
    else:
        return None

def arrange_mask(data:dict,
                device=device):
    
    
    return ME.SparseTensor(
        features=data['mask'][:,None],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    
    
    
    

## Filtering function for hits, to remove hits too far away from the vertex    
def select_hits_near_vertex(cut:float=8.,
                            dist_type:str="cube"):
    
    def f(data, cut, dist_type):
        if dist_type=="cube":
            return (np.max(np.abs(data['c']-data['verPos'][None,:]), axis=1)/CUBE_SIZE)<cut
        elif dist_type=="euclidian":
            return np.linalg.norm(data['c']-data['verPos'][None,:], axis=1)<cut
        else:
            raise NotImplementedError(f"The distance type {dist_type} is not implemented for selecting hits near a vertex")
    
    return lambda data: f(data, cut=cut, dist_type=dist_type)
    