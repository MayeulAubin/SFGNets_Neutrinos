import numpy as np
import pickle as pk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from glob import glob
import MinkowskiEngine as ME
from numpy.lib import npyio

from ..datasets.dataclass import EventDataset
from ..datasets.constants import RANGES, CUBE_SIZE
from ..datasets.utils import full_dataset, transform_cube, transform_inverse_cube, filtering_events, split_dataset
from ..datasets.utils_minkowski import arrange_sparse_minkowski, arrange_truth, arrange_aux



def collate_sparse_minkowski(batch:list[dict]) -> dict:
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


def retag_hits_based_on_distance(data:npyio.NpzFile,
                                 cut:float) -> np.ndarray:
    """
    Changes the hit tags based on the distance to the vertex and the pdg.
    Hits previously tagged as vertex activity further than cut to the vertex are retagged as tracks or noise based on the pdg.
    
    Parameters:
    - data: npyio.NpzFile, event file.
    - cut: float, cut distance.
    
    Returns:
    - y: np.ndarray, new tags.
    """
    y=data['y']-1
    indexes=(np.linalg.norm(data['c']-data['verPos'][None,:], axis=1)>cut)*(y==0) # indexes of VA hits further than the cut distance
    indexes_sp=indexes*(data['pdg']!=0)
    indexes_no=indexes*(data['pdg']==0)
    y[indexes_sp]=1
    y[indexes_no]=2
    return y


def retag_cut(cut:float):
    """
    Creates a retagging function for a specified cut distance, based on retag_hits_based_on_distance.
    
    Parameters:
    - cut: float, cut distance.
    
    Returns:
    - retag: function, retagging the hits.
    """
    return lambda data: retag_hits_based_on_distance(data, cut=cut)




class SparseEvent(EventDataset):
    def __init__(self, 
                 root:str, 
                 shuffle:bool=False, 
                 scaler_file:str|None=None, 
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
    
    
    def getc(self,data:npyio.NpzFile):
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
    
    def gety(self,data:npyio.NpzFile):
        if self.retagging:
            return torch.LongTensor(self.retagging_func(data))
        return torch.LongTensor(data['y'] - 1)
    
    def getaux(self,data:npyio.NpzFile):
        if self.aux:
            pdg=data['pdg']
            reaction_code=int(data['reaction_code'])*np.ones_like(pdg)
            aux=np.zeros((len(pdg),2))
            aux[:,0]=pdg
            aux[:,1]=reaction_code
            return torch.Tensor(aux)
        else:
            return None
        
        


# Create and save a scaler for the features if necessary
# Not used anymore because the scalers are manually specified with known ranges.
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
    
    

## Filtering function for hits, to remove hits too far away from the vertex    
def select_hits_near_vertex(cut:float=8.,
                            dist_type:str="cube"):
    """
    Creates a filtering function removing hits far away from the vertex based on a cut distance and a distance type.
    
    Parameters:
    - cut: float, cut distance
    - dist_type: str, type of distance to use, either "cube" or "euclidian"
    
    Returns:
    - f: function, filtering function, a function that returns a boolean array indicating which hits are close enough.
    """
    
    def f(data, cut, dist_type):
        if dist_type=="cube":
            return (np.max(np.abs(data['c']-data['verPos'][None,:]), axis=1)/CUBE_SIZE)<cut
        elif dist_type=="euclidian":
            return np.linalg.norm(data['c']-data['verPos'][None,:], axis=1)<cut
        else:
            raise NotImplementedError(f"The distance type {dist_type} is not implemented for selecting hits near a vertex")
    
    return lambda data: f(data, cut=cut, dist_type=dist_type)