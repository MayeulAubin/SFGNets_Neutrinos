
import torch
import MinkowskiEngine as ME
        

# Function to arrange sparse minkowski data
def arrange_sparse_minkowski(data:dict,
                             device:torch.device) -> ME.SparseTensor:
    """
    Converts a dictionary containing 'f' (features) and'c' (coordinates) keys into Minkowski Engine Sparse Tensor.
    
    Parameters:
    - data: dict, dictionary containing 'f' (features) and 'c' (coordinates) keys
    
    Returns:
    - features: SparseTensor, sparse tensor containing 'f' (features) at 'c' (coordinates)
    """
    return ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    
# Function to arange the truth in sparse Minkowski
def arrange_truth(data:dict,
                  device:torch.device) -> ME.SparseTensor:
    """
    Converts a dictionary containing 'y' (labels) and'c' (coordinates) keys into Minkowski Engine Sparse Tensor.
    
    Parameters:
    - data: dict, dictionary containing 'y' (labels) and 'c' (coordinates) keys
    
    Returns:
    - features: SparseTensor, sparse tensor containing 'y' (labels) at 'c' (coordinates)
    """
    if len(data['y'].shape)==2: # checking if the data has already been expanded of one dimension
        y=data['y']
    elif len(data['y'].shape)==1: # if not, expand the data
        y=data['y'][:,None]
    else: 
        raise ValueError(f"y has not a compatible shape: {data['y'].shape}")
        
    return ME.SparseTensor(
        features=y,
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )


def arrange_aux(data:dict,
                device:torch.device) -> ME.SparseTensor|None:
    """
    Converts a dictionary containing 'aux' (auxiliary) and'c' (coordinates) keys into Minkowski Engine Sparse Tensor.
    
    Parameters:
    - data: dict, dictionary containing 'aux' (auxiliary) and 'c' (coordinates) keys
    
    Returns:
    - features: SparseTensor or None, sparse tensor containing 'aux' (auxiliary) at 'c' (coordinates) if 'aux' is available, else None
    """
    if data['aux'] is not None:
    
        return ME.SparseTensor(
            features=data['aux'],
            coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
            device=device
        )
        
    else:
        return None


def arrange_mask(data:dict,
                device:torch.device) -> ME.SparseTensor:
    """
    Converts a dictionary containing 'mask' (mask) and'c' (coordinates) keys into Minkowski Engine Sparse Tensor.
    
    Parameters:
    - data: dict, dictionary containing 'mask' (mask) and 'c' (coordinates) keys
    
    Returns:
    - features: SparseTensor, sparse tensor containing 'mask' (mask) at 'c' (coordinates)
    """
    
    return ME.SparseTensor(
        features=data['mask'][:,None],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )