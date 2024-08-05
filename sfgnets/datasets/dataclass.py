
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import random
from abc import ABC, abstractmethod, ABCMeta
import tqdm
from numpy.lib import npyio

from .utils import natural_sort


class EventDataset(Dataset, metaclass=ABCMeta):
    """
    Data class representing a list of hits inside SFGD constituing an event. Each hit has features 'x', targets 'y', coordinates 'c' and auxiliary variables 'aux'.
    The data is stored as numpy arrays (one per aforementioned variables), which rows correspond to hits. The arrays are saved into a numpy 'npz' file, one per event.
    Only the directory in which the files are stored is necessary.
    """
    
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
        - filtering_func: function, function to filter the indexes of the arrays in the data file
        '''
        
        self.variables=['x','y','c','aux']
        
        self.root = root
        self.files_suffix=files_suffix
        self.data_files = self.processed_file_names
        self.multi_pass=multi_pass
        
        if shuffle:
            random.shuffle(self.data_files)
            
        self.total_events = len(self.data_files)
        
        self.filtering_func=filtering_func
    
    @property
    def processed_dir(self) -> str:
        return f'{self.root}'

    @property
    def processed_file_names(self) -> list[str]:
        return natural_sort(glob(f'{self.processed_dir}/*.{self.files_suffix}'))
    
    def __len__(self) -> int:
        return self.total_events*self.multi_pass
    
    @abstractmethod
    def getx(self,data:npyio.NpzFile) -> None|np.ndarray:
        return None
    
    @abstractmethod
    def gety(self,data:npyio.NpzFile) -> None|np.ndarray:
        return None
    
    @abstractmethod
    def getc(self,data:npyio.NpzFile) -> None|np.ndarray:
        return None
    
    @abstractmethod
    def getaux(self,data:npyio.NpzFile) -> None|np.ndarray:
        return None
    

    def __getitem__(self, idx:int) -> dict[str,None|np.ndarray]:
        """
        Returns a dictionary corresponding to the data of the event of index idx
        """
        
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
        if self.filtering_func is not None:
            for name in self.variables: # iterate over the variables 'x', 'y', 'c', ...
                try:
                    if return_dict[name] is not None: # check that the variable is not None, and is actually a tensor/array
                        return_dict[name]=return_dict[name][self.filtering_func(data)] # use self.filtering_func(data) as indexes to be kept
                except Exception as E:
                    print(f"Error filtering data for variable {name}, index {idx}, file {self.data_files[idx%self.total_events]}")

        # Clean upthe data
        del data
        
        # Check that none of the data is None, if it is return None for all variables
        if return_dict['x'] is None or return_dict['y'] is None:
            for key in return_dict.keys():
                return_dict[key]=None # then set all variables to None
        
        return return_dict
    
    
    def all_data(self,
                 progress_bar:bool=True,
                 max_index:int|None=None,) -> dict[str,None|np.ndarray]:
        """
        Returns all data concatenated into single arrays (no separation between events).
        """
        
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
    def data(self) -> dict[str,None|np.ndarray]:
        return self.all_data(progress_bar=False)
