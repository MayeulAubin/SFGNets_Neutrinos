
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import numpy as np
from .constants import RANGES, CUBE_SIZE


def natural_sort(l:list[str]) -> list[str]:
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
def transform_cube(X:np.ndarray) -> np.ndarray:
    """
    Convert raw coordinates to cubes.
    
    Parameters:
    - X: numpy array
    - dim: int, dimension (2 or 3)
    """
    X-=RANGES[None,:,0]
    X/=(CUBE_SIZE * 2)
    return X.astype(int)


def transform_inverse_cube(X:np.ndarray) -> np.ndarray:
    """
    Convert cubes to raw coordinates.
    
    Parameters:
    - X: numpy array
    """
    X=X.astype(float)
    X+=0.5
    X = X * (CUBE_SIZE * 2) + RANGES[None,:,0]
    return X
        



def filtering_events(x:dict) -> bool:
    """
    Filtering function for events used in collate function building batches.
    
    Parameters:
    - x: dict, a dictionary containing 'c' (coordinates), 'x' (features), and 'y' (labels) keys.
    
    Returns:
    - bool, True if the dictionary is not empty, False otherwise.
    """
    return x['c'] is not None



def collate_default(batch:list[dict[str,Tensor]]) -> dict[str,Tensor]:
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
    

def split_dataset(dataset:Dataset,
                  batch_size:int = 32,
                  train_fraction:float=0.8,
                  val_fraction:float=0.19,
                  seed:int=7,
                  multi_GPU:bool=False,
                  num_workers:int=24,
                  collate=collate_default,) -> tuple[DataLoader,DataLoader,DataLoader]:
    """
    Create training, validation and test data loaders based on a dataset and a collate function.
    
    Parameters:
    - dataset: Dataset, the dataset to split
    - batch_size: int, the batch size for DataLoader
    - train_fraction: float, the fraction of the dataset for training
    - val_fraction: float, the fraction of the dataset for validation
    - seed: int, the seed for the random split
    - multi_GPU: bool, whether to use multi-GPU for data loading
    - num_workers: int, the number of workers for DataLoader
    - collate: function, the aggregation function for creating a batch from a list of data elements
    
    Returns:
    - tuple, (train_loader, valid_loader, test_loader)
    """
    

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
                 ) -> DataLoader:
    """
    Create a data loader for a full dataset. Used when the dataset is a test dataset to have a single test data loader.
    
    Parameters:
    - dataset: Dataset, the dataset to turn into a data loader
    - batch_size: int, the batch size for DataLoader
    - multi_GPU: bool, whether to use multi-GPU
    - num_workers: int, the number of workers for DataLoader
    - collate: function, the aggregation function for creating a batch from a list of data elements
    
    Returns:
    - DataLoader, a DataLoader for the full dataset
    """
    
    if multi_GPU:
        full_loader = DataLoader(
            dataset, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset),
            )
    else:
        full_loader = DataLoader(
            dataset, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, shuffle=False
            )
    
    return full_loader