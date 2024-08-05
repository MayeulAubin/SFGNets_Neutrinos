import numpy as np
import tqdm
import torch
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import precision_recall_fscore_support, classification_report

np.set_printoptions(precision=3)

B_FIELD_INTENSITY = 0.2 # magnitude of the B field (supposed to be along X) in Teslas
C_SPEED_OF_LIGHT = 2.99792e8 # speed of light in m/s

DEFAULT_MOM_NORM= 100.


def sort_event_from_all_results(all_results:dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """
    Sort the results by event_id, traj_id and hit_order so that they are prepared for further processing.
    Notably it is possible to use choose_direction_sign with mode 'order' thanks to this.
    
    Parameters:
    - all_results: dict[str,np.ndarray], dictionary containing all the unscaled results of the testing of a model (it should be the return of track_fitting_net.execution.measure_performances)
    
    Returns:
    - all_results_sorted: dict[str,np.ndarray], dictionary containing all the unscaled results sorted by event_id, traj_id and hit_order; of the testing of a model
    """
    
    order_index=all_results['aux'][:,[23]].astype(int)
    exclude_not_attributed_hits=(order_index[:,0]<1001)
    order_index=order_index[exclude_not_attributed_hits]
    
    event_id=all_results['event_id'].astype(int)[exclude_not_attributed_hits]
    traj_id=all_results['aux'][:,[1]].astype(int)[exclude_not_attributed_hits]
    
    order_array=np.concatenate([event_id,traj_id,order_index],axis=1)
    
    indexes=np.lexsort(order_array[:, ::-1].T)
    
    ret_dict={}
    for key in all_results.keys():
        ret_dict[key]=all_results[key][exclude_not_attributed_hits][indexes] if all_results[key] is not None else None
    
    ret_dict['node_c_absolute']=(all_results['predictions'][:,:3]+all_results['c'])[exclude_not_attributed_hits][indexes]
    
    if all_results['predictions'].shape[-1]==6:
        ret_dict['node_d']=all_results['predictions'][:,3:6][exclude_not_attributed_hits][indexes]
        ret_dict['node_d']/=(np.linalg.norm(ret_dict['node_d'],axis=-1,keepdims=True)+1e-9)
    else:
        ret_dict['node_d']=None
    
    ret_dict['event_id']=event_id[indexes]
    ret_dict['traj_id']=traj_id[indexes]
    ret_dict['order_index']=order_index[indexes]
    
    ret_dict['true_node_c_absolute']=(all_results['y'][:,:3]+all_results['c'])[exclude_not_attributed_hits][indexes]
    # ret_dict['true_node_d']=all_results['aux'][:,11:14][exclude_not_attributed_hits][indexes]
    
    ret_dict['true_node_d']=all_results['aux'][:,4:7][exclude_not_attributed_hits][indexes]
    ret_dict['true_node_d']/=(np.linalg.norm( ret_dict['true_node_d'],axis=-1,keepdims=True)+1e-9)
    
    ret_dict['true_momentum']=all_results['aux'][:,4:7][exclude_not_attributed_hits][indexes]
    ret_dict['true_charge']=all_results['aux'][:,24][exclude_not_attributed_hits][indexes]
    ret_dict['y']=np.concatenate([all_results['y'][:,:3],all_results['aux'][:,4:7]],axis=-1)[exclude_not_attributed_hits][indexes]
    
    ret_dict['recon_node_c_absolute']=(all_results['aux'][:,17:20]+all_results['c'])[exclude_not_attributed_hits][indexes]
    ret_dict['recon_node_d']=all_results['aux'][:,20:23][exclude_not_attributed_hits][indexes]
    
    ret_dict['exclude_not_attributed_hits']=exclude_not_attributed_hits
    ret_dict['indexes']=indexes
    
    return ret_dict


def get_charge(curvature:np.ndarray|torch.Tensor, 
               direction:np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor|float:
    """
    Computes the charge from the curvature and direction given the fixed magnetic B field.
    Definition from Curvature_to_MomentumAndCharge.
    
    Parameters:
    - curvature: array-like, shape (...,3), the curvature of the trajectory points (defined as the vector pointing to the center with norm the inverse of the radius of the local fitting circle)
    - direction: array-like, shape (...,3), the direction of the trajectory points.
    
    Returns:
    - charge: array-like, shape (...,), the charge of the trajectory points. It is positive for right-handed tracks, negative for left-handed tracks, and zero for tracks that are parallel to the magnetic field.
    """
    
    if isinstance(curvature, np.ndarray):
        charge = np.cross(curvature, direction)[...,0]/B_FIELD_INTENSITY # B is along X, the dot product is just the first index times B intensity
        return np.sign(charge)
    elif isinstance(curvature, torch.Tensor):
        charge = torch.cross(curvature, direction)[...,0]/B_FIELD_INTENSITY # B is along X, the dot product is just the first index times B intensity
        return torch.sign(charge)
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.") 
    

def get_momentum_magnitude(curvature:np.ndarray|torch.Tensor, direction:np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor|float:
    """
    Computes the momentum magnitude from the curvature and direction given the fixed magnetic B field.
    Definition from Curvature_to_MomentumAndCharge
    
    Parameters:
    - curvature: array-like, shape (...,3), the curvature of the trajectory points (defined as the vector pointing to the center with norm the inverse of the radius of the local fitting circle)
    - direction: array-like, shape (...,3), the direction of the trajectory points.
    
    Returns:
    - momentum_magnitude: array-like, shape (...,), the momentum magnitude at each trajectory point.
    """
    
    if isinstance(curvature, np.ndarray):
        factor= -0.3 * B_FIELD_INTENSITY / np.sqrt(1.-direction[...,0]**2)
        return np.clip(np.abs(factor/(np.cross(curvature, direction)[...,0]+1e-9)),0,4000) 
    elif isinstance(curvature, torch.Tensor):
        factor= -0.3 * B_FIELD_INTENSITY / torch.sqrt(1.-direction[...,0]**2)
        return torch.clip(torch.abs(factor/(torch.cross(curvature, direction)[...,0]+1e-9)),0,4000) 
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.") 


def _handle_single_hit_cases(same_event_traj_matrix:np.ndarray|torch.Tensor, 
                             node_d:np.ndarray|torch.Tensor, 
                             node_c_absolute:np.ndarray|torch.Tensor) -> None|tuple[np.ndarray|torch.Tensor,np.ndarray|torch.Tensor,np.ndarray|torch.Tensor,np.ndarray|torch.Tensor]:
    """
    Handles cases where an event/trajectory has less than two hits when estimating the direction and curvature.
    If the matrix of hits has only one hit, we return a default value. 
    If the matrix has several hits (but for a given trajectory/event there is only one hit), we allow this hit to reach any other hit of other trajectories/events.
    TODO: in the latter case, we should allow the hit to reach only hits of the same event.
    
    Parameters:
    - same_event_traj_matrix: array-like, shape (n_points, n_points), a matrix indicating which hits belong to the same event and trajectory.
    - node_d: array-like, shape (n_points, 3), the direction of each trajectory point.
    - node_c_absolute: array-like, shape (n_points, 3), the absolute coordinate of each trajectory point.
    
    Returns:
    - construct_direction_and_curvature_return: None|tuple, None if the data can be further processed, else default values for pathological cases.
    """
    
    if isinstance(same_event_traj_matrix, np.ndarray):
        # NumPy array operations
        sum_hits = np.sum(same_event_traj_matrix, axis=1)
        sum_condition = (sum_hits < 2).any()

        if sum_condition:
            if len(same_event_traj_matrix) == 1:
                if node_d is not None:
                    return np.array([0.]), DEFAULT_MOM_NORM * node_d, np.array([[0., 0., 0.]]), node_c_absolute, np.array([0.])
                else:
                    return np.array([0.]), DEFAULT_MOM_NORM * np.array([[0., 0., 1.]]), np.array([[0., 0., 0.]]), node_c_absolute, np.array([0.])
            else:
                same_event_traj_matrix[(sum_hits < 2)[...,0]] = 1.0  # use all hits available if the event
                
    elif isinstance(same_event_traj_matrix, torch.Tensor):
        # PyTorch tensor operations
        sum_hits = torch.sum(same_event_traj_matrix, dim=1)
        sum_condition = torch.any(sum_hits < 2).item()

        if sum_condition:
            if same_event_traj_matrix.size(0) == 1:
                if node_d is not None:
                    return torch.tensor([0.]), DEFAULT_MOM_NORM * node_d, torch.tensor([[0., 0., 0.]]), node_c_absolute, torch.tensor([0.])
                else:
                    return torch.tensor([0.]), DEFAULT_MOM_NORM * torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., 0.]]), node_c_absolute, torch.tensor([0.])
            else:
                same_event_traj_matrix[(sum_hits < 2)[...,0]] = 1.0  # use all hits available if the event
    

def choose_direction_sign(node_c_absolute:np.ndarray|torch.Tensor,
                          matrix_of_differences:np.ndarray|torch.Tensor,
                          weights_for_differences:np.ndarray|torch.Tensor,
                          node_d:np.ndarray|torch.Tensor|None,
                          mode:str) -> np.ndarray|torch.Tensor:
    """
    Indicates whether the other points are after or before the reference point in the trajectory.
    For a given matrix of differences M_{i,j} returns the antisymmetric matrix dirorder_{i,j} := +1 if the point j is after the point i, -1 if the point j is before the point i.
    Supports various modes to determine the order of the points:
        - 'order': We suppose that the points are already ordered, so dirorder_{i,j} = sign(j-i)
        - 'node_d': We rely on the infered node directions to determine the order: dirorder_{i,j} = sign(M_{i,j}.node_d_{i})
        - 'PCA': We first select an axis based on the most dispersion of the points (PCA analysis), then we need to choose the sign of this axis:
            - 'PCAo': the sign is given by the order of the points: we make sure that on average mean_j(dirorder_{i,j})*mean_j(sign(j-i))=1
            - 'PCAnd': the sign is given by the node_d 
            - 'PCAl': the sign is set so that the directions are consistently pointing in the same direction.
            
    Parameters:
    - node_c_absolute: array-like, shape (n_points, 3), the absolute coordinates of each trajectory point.
    - matrix_of_differences: array-like, shape (n_points, n_points, 3), the matrix of difference vectors between the points.
    - weights_for_differences: array-like, shape (n_points, n_points), the weights for the differences for the choice of the direction sign.
    - node_d: array-like, shape (n_points, 3), the direction of each trajectory point.
    - mode: str, the mode to determine the order of the points (see above).
    
    Returns:
    - dirorder_matrix: array-like, shape (n_points, n_points), the antisymmetric sign matrix indicating the order of the points relative to each other (+1 if j after i, -1 if before).
    """
    
    
    if isinstance(node_c_absolute, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(node_c_absolute, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=node_c_absolute.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    ## Now we need to know if the difference vectors need to be counted positively or negatively, depending on their sign along the direction of the track
    ## Since we don't know yet the direction of the track (that's the goal of this algorithm), we need an alternative way to get their sign
    ## The following options are available:
    
    if mode=='order':
        ## We use the order of the hits: hits before are counted negatively, hits after positively
        ## Nodes_c is supposed to be ordered by their position along the track
        dir_order_matrix=module_.ones((node_c_absolute.shape[0], node_c_absolute.shape[0]))
        if use_torch:
            dir_order_matrix=dir_order_matrix.to(device)
        dir_order_matrix=module_.triu(dir_order_matrix)-module_.tril(dir_order_matrix)
    
    elif mode=='node_d':
        ## We use the predicted direction 'node_d' as an approximation: differences with a positive scalar product with node_d are counted positively, differences with a negative scalar product negatively
        if node_d is not None:
            dir_order_matrix=module_.sign(module_.sum(node_d[None,:,:]*matrix_of_differences,**axis_arg))
        else:
            raise ValueError("node_d is None while mode is 'node_d'")
        
    elif mode[:3]=='PCA':
        ## We compute a local PCA to get the local axis of most elongation, then we sign it using a submode order or node_d, and then we use this approximate direction in the same way as 'node_d'
        M=matrix_of_differences*weights_for_differences[:,:,None] # shape (N,N,3)
        M_mean=M.mean(**axis_arg,**keepdim_arg)
        # Compute covariance using einsum
        covariance_M = module_.einsum('ijk,ijl->ikl', M - M_mean, M - M_mean) / (M.shape[0] - 1) # shape (N,3,3), covariance_M[i]=Cov(M[i])
        
        try:
            # Compute eigenvalues and eigenvectors for each covariance matrix
            eigenval, eigenvect = module_.linalg.eigh(covariance_M)
            # Find the indices of the eigenvectors with the largest eigenvalues
            max_eigenvector_indices = module_.argmax(eigenval, *axis_arg)
            # Use fancy indexing to get the corresponding eigenvectors
            approx_dir = eigenvect[module_.arange(M.shape[0]), max_eigenvector_indices]
            # Compute the scalar product of the difference vectors on the approximate direction to get the sign
            dir_order_matrix=module_.sign(module_.sum(approx_dir[None,:,:]*matrix_of_differences, *axis_arg))
            PCA_failed=False
            
        except (np.linalg.LinAlgError,torch.linalg.LinAlgError):
            print('PCA failed, defaulting to order mode')
            # If the eigenvectors do not converge, default to order mode:
            PCA_failed=True
            dir_order_matrix=module_.ones((node_c_absolute.shape[0], node_c_absolute.shape[0]))
            if use_torch:
                dir_order_matrix=dir_order_matrix.to(device)
            dir_order_matrix=module_.triu(dir_order_matrix)-module_.tril(dir_order_matrix)
        
        if mode=='PCAo' or mode=='PCA':
            if not PCA_failed:
                ## Here we use the 'order' mode to get the correct sign of the approximate direction provided by the eigenvector.
                ## To do this we check that, on average, hits after are counted positively and before negatively
                hit_order=module_.ones((M.shape[0], M.shape[0]))
                if use_torch:
                    hit_order=hit_order.to(device)
                hit_order=(module_.triu(hit_order)-module_.tril(hit_order))
                hit_order*=dir_order_matrix*weights_for_differences
                dir_order_matrix*=module_.sign(module_.sum(hit_order,**axis_arg,**keepdim_arg))
            
        elif mode=='PCAnd':
            if not PCA_failed:
                if node_d is not None:
                    ## Here we use the 'node_d' mode to get the correct sign of the approximate direction provided by the eigenvector.
                    ## To do this we check that the scalar product between the preidcted node_d and the approximate direction is positive.
                    eigenvect_sign=module_.sum(node_d*approx_dir,**axis_arg)
                    dir_order_matrix*=module_.sign(eigenvect_sign)
                else:
                    raise ValueError("node_d is None while mode is 'PCAnd'")
                
        elif mode=='PCAl':
            if not PCA_failed:
                ## Here we make sure that the eigenvectors are locally along the same axis.
                ## To do that we check that the scalar product between approximate directions is positive locally
                eigenvect_sign=module_.sum(approx_dir[None,:,:]*approx_dir[:,None,:]*weights_for_differences[:,:,None],**{list(axis_arg.keys())[0]:(1,2)})
                dir_order_matrix*=module_.sign(eigenvect_sign)
        
        else:
            raise ValueError(f"mode {mode} not recognized")
    else:
        raise ValueError(f"mode {mode} not recognized")
    
    return dir_order_matrix



def _estimate_direction(node_c_absolute:np.ndarray|torch.Tensor,
                        matrix_of_differences:np.ndarray|torch.Tensor,
                        matrix_of_distances:np.ndarray|torch.Tensor,
                        node_d:np.ndarray|torch.Tensor|None,
                        args_node_d:dict[str,float],
                        args_dir:dict[str,float],
                        mode:str,
                        ) -> tuple[np.ndarray|torch.Tensor,np.ndarray|torch.Tensor]:
    
    """
    Constructs the estimated direction locally, 
    and the direction order matrix that tells whether the other points are after or before along the trajectory path.
    Inspired from sfgRecon -> SFGCreateTrack.hxx : CreateTrackFromClusters 
    Uses a kernel to average the direction locally. The kernel is w_{i,j} = (M_{i,j}/d)*e^{-(M_{i,j}/d)} where d is a scale parameter.
    Can use the information provided by node_d to estimate the direction.
    The parameters are given through args_node_d and args_dir
    
    Parameters:
    - node_c_absolute: array-like, shape (n_points, 3), the absolute coordinates of each trajectory point.
    - matrix_of_differences: array-like, shape (n_points, n_points, 3), the matrix of difference vectors between the points.
    - matrix_of_distances: array-like, shape (n_points, n_points), the symmetric positive matrix of the distances between the points.
    - node_d: array-like, shape (n_points, 3), the direction of each trajectory point.
    - args_node_d: dict, dictionary containing parameters for node_d usage.
    - args_dir: dict, dictionnary containing parameters for the estimation of the direction from the trajectory points coordinates.
    - mode: str, the mode for the estimation of the direction sign ('order', 'node_d', 'PCAo', 'PCAnd')
    
    Returns:
    - estimated_direction: array-like, shape (n_points, 3), the estimated direction of each trajectory point.
    - dirorder_matrix: array-like, shape (n_points, n_points), the antisymmetric sign matrix indicating the order of the points relative to each other (+1 if j after i, -1 if before).
    """
    
    if isinstance(node_c_absolute, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(node_c_absolute, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=node_c_absolute.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    
    weight_predicted_node_d = args_node_d['weight_predicted_node_d']
    scale_factor_node_d = args_node_d['scale_factor_node_d']
    
    dirScale = args_dir["dirScale"]
    
    ## Computes the weights for direction
    ## wd_{ij} := D_{ij}/d * e^{-D_{ij}/d} where d is an arbitrary scale of points to consider
    weights_differences_for_directions=(matrix_of_distances/dirScale)*module_.exp(-(matrix_of_distances/dirScale))
    
    ## Now we need to know if the difference vectors need to be counted positively or negatively, depending on their sign along the direction of the track
    ## Since we don't know yet the direction of the track (that's the goal of this algorithm), we need an alternative way to get their sign
    ## dirorder_{ij} = +1 or -1 such that M_{ij} * dirorder_{ij} is in the correct direction
    dir_order_matrix=choose_direction_sign(node_c_absolute,matrix_of_differences,weights_differences_for_directions,node_d,mode)
    
    ## Vector of reconstructed directions
    ## dir_i := mean_{along j}(M_{ij} * dirorder_{ij})_{weighted by wd_{ij}}
    dir_from_differences=module_.sum(weights_differences_for_directions[:,:,None]*matrix_of_differences*dir_order_matrix[:,:,None]/(matrix_of_distances[:,:,None]+1e-9),**{list(axis_arg.keys())[0]:1})/(module_.sum(weights_differences_for_directions,**axis_arg)[:,None]+1e-9)
    
        
    # if node_d is not None:
    #     ## If node_d is available, we use it by averaging it with the reconstructed direction.
    #     ## We compute the average locally, using the same kind of weighting, but with a scale that can be different
        
    #     weights_node_d_for_directions=(matrix_of_distances/(dirScale*scale_factor_node_d))*module_.exp(-(matrix_of_distances/(dirScale*scale_factor_node_d)))
        
    #     dir_from_node_d=module_.sum(weights_node_d_for_directions[:,:,None]*node_d[None,:,:],**{list(axis_arg.keys())[0]:1})/(module_.sum(weights_node_d_for_directions,**axis_arg)[:,None]+1e-9)
        
    #     direrrors = module_.linalg.norm(dir_from_node_d,**axis_arg)
    #     direrrors = 1.*(module_.abs(direrrors-1.)<1e-2)*module_.isfinite(direrrors)
        
    #     direction=(dir_from_differences+weight_predicted_node_d*dir_from_node_d*direrrors[:,None])/(1.+weight_predicted_node_d*direrrors.mean())
    #     assert weight_predicted_node_d==0., f"weight_predicted_node_d is {weight_predicted_node_d}"
    #     if ((direction-dir_from_differences)**2>1e-5 + ~module_.isfinite(direction)).any():
    #         print(direrrors.mean())
    #         print((weight_predicted_node_d*dir_from_node_d*direrrors[:,None]).mean())
    #         print(f"discordance {((direction-dir_from_differences)**2).mean()}")
    #         assert False
    # else:
    #     direction=dir_from_differences
        
    direction=dir_from_differences
    direction/=(module_.linalg.norm(direction,**axis_arg,**keepdim_arg)+1e-9)
    
    return direction, dir_order_matrix



def _shift_points_along_trajectory(node_c_absolute:np.ndarray|torch.Tensor,
                                   direction:np.ndarray|torch.Tensor,
                                   same_event_traj_matrix:np.ndarray|torch.Tensor,
                                   matrix_of_differences:np.ndarray|torch.Tensor,
                                   matrix_of_distances:np.ndarray|torch.Tensor,
                                   args_dir:dict[str,float],
                                   ) -> np.ndarray|torch.Tensor:
    """
    Translate the points positions so that they fall on the trajectory line provided by the estimated directions.
    To do that we translate them orthogonally to their direction towards the local average position.
    
    Parameters:
    - node_c_absolute: array-like, shape (n_points, 3), the absolute coordinates of each trajectory point.
    - direction: array-like, shape (n_points, 3), the estimated direction of each trajectory point.
    - same_event_traj_matrix: array-like, shape (n_points, n_points), the boolean matrix indicating which points are in the same event and trajectory.
    - matrix_of_differences: array-like, shape (n_points, n_points, 3), the matrix of difference vectors between the points.
    - matrix_of_distances: array-like, shape (n_points, n_points), the symmetric positive matrix of the distances between the points.
    - args_dir: dict, dictionnary containing parameters for the estimation of the direction from the trajectory points coordinates.
    
    Returns:
    - points_updated: array-like, shape (n_points, 3), the updated absolute coordinates of each trajectory point.
    """
    
    if isinstance(node_c_absolute, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(node_c_absolute, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=node_c_absolute.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    
    dirScale = args_dir["dirScale"]

    ## Shift the trajectory points to place them on the trajectory
    ## Computes the center of the neighbouring points, shifts the points towards it but perpendicularly to the direction of the trajectory
    ## wp_{ij} := e^{-D_{ij}/(2*d)}, where d is the dirScale
    weights_for_points_shift=module_.exp(-(matrix_of_distances/(2*dirScale)))
    points_shift=module_.sum(weights_for_points_shift[:,:,None]*matrix_of_differences/(matrix_of_distances[:,:,None]+1e-9),**{list(axis_arg.keys())[0]:1})/(module_.sum(weights_for_points_shift,**axis_arg)[:,None]+1e-9)
    points_shift=points_shift - module_.sum(points_shift*direction,**axis_arg,**keepdim_arg)*direction
    points_updated=node_c_absolute+points_shift
    
    ## Computes the updated matrix of differences between the points positions
    ## Modifies in place the matrix
    ## M_{ij} := p_j - p_i
    matrix_of_differences[...]=points_updated[None,:,:]*same_event_traj_matrix-points_updated[:,None,:]*same_event_traj_matrix
    
    
    ## Computes the matrix of distances updated
    ## Modifies in place the matrix
    ## D_{ij} :=  ||M_{ij}|| = ||p_j - p_i||
    matrix_of_distances[...]=module_.linalg.norm(matrix_of_differences,**axis_arg)
    
    return points_updated



def _estimate_curvature(direction:np.ndarray|torch.Tensor,
                        dir_order_matrix:np.ndarray|torch.Tensor,
                        matrix_of_differences:np.ndarray|torch.Tensor,
                        matrix_of_distances:np.ndarray|torch.Tensor,
                        args_curv:dict[str,float],
                        ) -> np.ndarray|torch.Tensor :
    """
    Constructs the locally estimated curvature.
    Inspired from sfgRecon -> SFGCreateTrack.hxx : CreateTrackFromClusters 
    Uses a kernel to estimate the curvature locally. The kernel is w_{i,j} = (M_{i,j}/d)*e^{-(M_{i,j}/d)}*inclusion_weight_{i,j} where d is a scale parameter
    Uses an inclusion_weight to enforce the approximation requirements of the estimation.
    The parameters are given through args_curv.
    
    Parameters:
    - direction: array-like, shape (n_points, 3), the estimated direction of each trajectory point.
    - dir_order_matrix: array-like, shape (n_points, n_points), the antisymmetric sign matrix indicating the order of the points relative to each other (+1 if j after i, -1 if before).
    - matrix_of_differences: array-like, shape (n_points, n_points, 3), the matrix of difference vectors between the points.
    - matrix_of_distances: array-like, shape (n_points, n_points), the symmetric positive matrix of the distances between the points.
    - args_curv: dict, dictionnary containing parameters for the estimation of the curvature
    
    Returns:
    - curvature: array-like, shape (n_points, 3), the estimated curvature of each trajectory point (defined as the vector pointing to the center with norm the inverse of the radius of the local fitting circle).
    """
    
    
    if isinstance(matrix_of_differences, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(matrix_of_differences, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=matrix_of_differences.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    

    
    curvScale = args_curv['curvScale']
    
    ## Computes the weights for getting the average direction after the point
    ## wda_{ij} := D_{ij}/d * e^{-D_{ij}/d} where d is an arbitrary scale of points to consider
    ## The kernel is such that close points are avoided and only points at the distance around the scale ~d are considered.
    ## This allows to get the average direction but quite far away from the point.
    weight_dir_after_for_curv=(matrix_of_distances/curvScale)*module_.exp(-(matrix_of_distances/curvScale))*(dir_order_matrix>0.)
    
    ## Computes the weights for getting the average direction before the point
    ## wda_{ij} := D_{ij}/d * e^{-D_{ij}/d} where d is an arbitrary scale of points to consider
    weight_dir_before_for_curv=(matrix_of_distances/curvScale)*module_.exp(-(matrix_of_distances/curvScale))*(dir_order_matrix<0.)
    
    ## Get the average direction after and before the point.
    other_dir_after=module_.sum(direction[None,:,:]*weight_dir_after_for_curv[:,:,None],**{list(axis_arg.keys())[0]:1})/(module_.sum(weight_dir_after_for_curv,**{list(axis_arg.keys())[0]:1})[:,None]+1e-9)
    other_dir_before=module_.sum(direction[None,:,:]*weight_dir_before_for_curv[:,:,None],**{list(axis_arg.keys())[0]:1})/(module_.sum(weight_dir_before_for_curv,**{list(axis_arg.keys())[0]:1})[:,None]+1e-9)
    
    ## The distance factor allows to account for the distance between the point and where the other directions are computed.
    distance_factor=module_.sum(1/(matrix_of_distances+1e-9)*(weight_dir_before_for_curv+weight_dir_after_for_curv),**axis_arg)/(module_.sum(weight_dir_before_for_curv+weight_dir_after_for_curv,**axis_arg)+1e-9)
    
    ## The curvature is obtained as a cross product between the directions at different points along the trajectory.
    ## We compute the cross product between the other direction before the point and the direction at the point, and the same for the direction after the point.
    ## We multiply by the distance factor to get the correct scale for the momentum norm.
    curvature=-distance_factor[:,None]*(module_.cross(other_dir_before,direction,**axis_arg)+module_.cross(direction,other_dir_after,**axis_arg))/2
    
    ## Now the "curvature" vector is not the curvature, but a vector mostly along the magnetic B field containing the charge and momentum norm information.
    ## To get back the real curvature that points toward the center of the arc circle, we need to cross product with the direction
    curvature=module_.cross(direction,curvature,**axis_arg)
    
    return curvature




def trace_ideal_trajectory(s:np.ndarray|torch.Tensor,
                           d_x:float|np.ndarray|torch.Tensor,
                           p_0:float|np.ndarray|torch.Tensor,
                           K:float|np.ndarray|torch.Tensor,
                           q:float|np.ndarray|torch.Tensor,
                           theta_0:float|np.ndarray|torch.Tensor,
                           X_0:np.ndarray|torch.Tensor,)-> np.ndarray|torch.Tensor:
    
    """
    Predicts the ideal trajectory of a particle in the detector (where the magnetic B field is assumed to be along the x axis).
    We assume that the stopping power of the material dE/dX is constant of value K over the trajectory (this could be relaxed to include Bethe-Bloch formula).
    
    Parameters:
    - s : the (signed) distance (in mm) between the predicted point and the reference point
    - d_x : the fraction of the momentum along the x axis (between 0. and 1.)
    - p_0 : the momentum (in MeV)
    - K : the stopping power (in MeV/mm)
    - q : the charge (+1,0,-1)
    - theta_0 : the angle in the YZ plane of the direction of the momentum at the reference point
    - X_0 : the 3D offset of the reference point
    
    Returns:
    - points : the 3D coordinates of the predicted trajectory
    """
    
    if isinstance(s, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(s, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=s.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    
    points=module_.zeros((s.shape[0],3))
    if use_torch:
        points=points.to(device)
    
    energy_reduction_factor = module_.clip(1 - K/p_0*s,1e-9,None)
    log_energy_reduction_factor = module_.log(energy_reduction_factor)
    B_field_factor = q*B_FIELD_INTENSITY*C_SPEED_OF_LIGHT*1e-9
    circle_factor = 1/(K+1j*B_field_factor+1e-9)
    x_axis_factor = module_.sqrt(1-d_x**2)
    initial_angle_factor = module_.exp(1j*theta_0)
    gyro_energy_factor = B_field_factor/K
    rotation = module_.exp(1j*gyro_energy_factor*log_energy_reduction_factor)
    
    a = x_axis_factor*initial_angle_factor*p_0*(1-energy_reduction_factor*rotation)*circle_factor
    
    points[:,1]=module_.real(a)
    points[:,2]=module_.imag(a)
    
    return points+X_0[None,:]



def trace_ideal_direction(s:np.ndarray|torch.Tensor,
                           d_x:float|np.ndarray|torch.Tensor,
                           p_0:float|np.ndarray|torch.Tensor,
                           K:float|np.ndarray|torch.Tensor,
                           q:float|np.ndarray|torch.Tensor,
                           theta_0:float|np.ndarray|torch.Tensor,)-> np.ndarray|torch.Tensor:
    
    """
    Predicts the ideal direction of the trajectory of a particle in the detector (where the magnetic B field is assumed to be along the x axis).
    We assume that the stopping power of the material dE/dX is constant of value K over the trajectory (this could be relaxed to include Bethe-Bloch formula).
    
    Parameters:
    - s : the (signed) distance (in mm) between the predicted point and the reference point
    - d_x : the fraction of the momentum along the x axis (between 0. and 1.)
    - p_0 : the momentum (in MeV)
    - K : the stopping power (in MeV/mm)
    - q : the charge (+1,0,-1)
    - theta_0 : the angle in the YZ plane of the direction of the momentum at the reference point
    
    Returns:
    - directions : the 3D coordinates of the predicted direction of the trajectory
    """
    
    
    if isinstance(s, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(s, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=s.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    theta=theta_0+q*B_FIELD_INTENSITY*C_SPEED_OF_LIGHT/K*1e-9*module_.log(module_.clip((1-K/p_0*s),1e-9,None))
    directions=module_.ones((s.shape[0],3))
    if use_torch:
        directions=directions.to(device)
    directions[:,0]*=d_x
    coeff_=module_.sqrt(1-d_x**2)
    u=coeff_*module_.exp(1j*theta)
    directions[:,1]=module_.real(u)
    directions[:,2]=module_.imag(u)
    
    return directions



def trace_simple_trajectory(s:np.ndarray|torch.Tensor,
                           d_x:float|np.ndarray|torch.Tensor,
                           p_0:float|np.ndarray|torch.Tensor,
                           q:float|np.ndarray|torch.Tensor,
                           theta_0:float|np.ndarray|torch.Tensor,
                           X_0:np.ndarray|torch.Tensor,)-> np.ndarray|torch.Tensor:
    
    """
    Predicts the ideal trajectory of a particle in the detector (where the magnetic B field is assumed to be along the x axis).
    We assume no stopping power (no absorption). The trajectory is a perfect circle/helix.
    
    Parameters:
    - s : the (signed) distance (in mm) between the predicted point and the reference point
    - d_x : the fraction of the momentum along the x axis (between 0. and 1.)
    - p_0 : the momentum (in MeV)
    - q : the charge (+1,0,-1)
    - theta_0 : the angle in the YZ plane of the direction of the momentum at the reference point
    - X_0 : the 3D offset of the reference point
    
    Returns:
    - points : the 3D coordinates of the predicted trajectory
    """
    
    if isinstance(s, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(s, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=s.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    
    points=module_.zeros(list(s.shape)+[3])
    if use_torch:
        points=points.to(device)
    
    ## Along the X direction, the trajectory is a straight line so we have:    
    points[...,0]=d_x*s
    
    ## The following coefficient is in mm
    coeff_=module_.sqrt(1-d_x**2)*(p_0*1e9/(q*B_FIELD_INTENSITY*C_SPEED_OF_LIGHT))
    a=coeff_*(1-module_.exp(-1j*q*B_FIELD_INTENSITY*C_SPEED_OF_LIGHT/p_0*1e-9*s))*module_.exp(-1j*(module_.pi/2-theta_0))
    points[...,1]=module_.real(a)
    points[...,2]=module_.imag(a)
    
    return points+X_0


def jac_ideal_trajectory(s:np.ndarray|torch.Tensor,
                        d_x:float|np.ndarray|torch.Tensor,
                        p_0:float|np.ndarray|torch.Tensor,
                        K:float|np.ndarray|torch.Tensor,
                        q:float|np.ndarray|torch.Tensor,
                        theta_0:float|np.ndarray|torch.Tensor,
                        X_0:np.ndarray|torch.Tensor,)-> np.ndarray|torch.Tensor:
    
    """
    Give the jacobian matrix of the ideal trajectory with respect to the parameters of the ideal trajectory
    We assume that the stopping power of the material dE/dX is constant of value K over the trajectory (this could be relaxed to include Bethe-Bloch formula).
    
    Parameters:
    - s : the (signed) distance (in mm) between the predicted point and the reference point
    - d_x : the fraction of the momentum along the x axis (between 0. and 1.)
    - p_0 : the momentum (in MeV)
    - K : the stopping power (in MeV/mm)
    - q : the charge (+1,0,-1)
    - theta_0 : the angle in the YZ plane of the direction of the momentum at the reference point
    - X_0 : the initial position
    
    Returns:
    - points : the 3D coordinates of the predicted trajectory
    """
    
    if isinstance(s, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
    elif isinstance(s, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=s.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    

    energy_reduction_factor = module_.clip(1 - K/p_0*s,1e-9,None)
    log_energy_reduction_factor = module_.log(energy_reduction_factor)
    B_field_factor = q*B_FIELD_INTENSITY*C_SPEED_OF_LIGHT*1e-9
    circle_factor = 1/(K+1j*B_field_factor+1e-9)
    x_axis_factor = module_.sqrt(1-d_x**2)
    initial_angle_factor = module_.exp(1j*theta_0)
    gyroradius = B_field_factor/p_0
    gyro_energy_factor = B_field_factor/K
    rotation = module_.exp(1j*gyro_energy_factor*log_energy_reduction_factor)
    
    default_a = x_axis_factor*initial_angle_factor*p_0*(1-energy_reduction_factor*rotation)*circle_factor
    
    jacobian_matrix=module_.zeros(list(s.shape)+[3,7])
    
    if use_torch:
        jacobian_matrix=jacobian_matrix.to(device)
    
    jacobian_matrix[...,0,0]=s
    jacobian_matrix[...,0,1:]=0.
    
    d_x_jacobian = -1.*default_a*d_x/(1-d_x**2+1e-9)
    jacobian_matrix[...,1,0]=module_.real(d_x_jacobian)
    jacobian_matrix[...,2,0]=module_.imag(d_x_jacobian)
    
    p_0_jacobian = x_axis_factor*initial_angle_factor*(1-(1+gyroradius*s)*rotation)*circle_factor
    jacobian_matrix[...,1,1]=module_.real(p_0_jacobian)
    jacobian_matrix[...,2,1]=module_.imag(p_0_jacobian)
    
    K_jacobian = x_axis_factor*initial_angle_factor*(p_0*(rotation*(2j*gyroradius*s-gyroradius*gyro_energy_factor+1.)-1.)+1j*gyro_energy_factor*p_0/K*energy_reduction_factor*log_energy_reduction_factor*rotation)*circle_factor
    jacobian_matrix[...,1,2]=module_.real(K_jacobian)
    jacobian_matrix[...,2,2]=module_.imag(K_jacobian)
    
    theta_0_jacobian = 1j*default_a
    jacobian_matrix[...,1,3]=module_.real(theta_0_jacobian)
    jacobian_matrix[...,2,3]=module_.imag(theta_0_jacobian)
    
    jacobian_matrix[...,0,4]=1.
    
    jacobian_matrix[...,1,5]=1.
    
    jacobian_matrix[...,2,6]=1.
    
    return jacobian_matrix
    
    

def _fit_one_traj_with_scipy(s:np.ndarray,
                            X:np.ndarray,
                            guess:list=[0.,1000.,1e-4,0.,0.,0.,0.],
                            bounds:tuple[list,list]=([-1.,1e-9,1e-9,-np.pi,-10.,-10.,-10.],[1.,4e3,1e-3,np.pi,10.,10.,10.]),):
    """
    Fit the parameters of an ideal trajectory on the data of a given trajectory.
    Relies on scipy.optimize.curve_fit to fit the parameters of trace_ideal_trajectory.
    Fits 3 different curves (one for each charge +1, 0, -1), and returns the best one.
    Quite slow and sometimes weird results.
    
    Parameters:
    - s : the (signed) distance (in mm) between the predicted point and the reference point
    - X : the 3D coordinates of the trajectory
    - guess : initial parameters values
    - bounds : the bounds for the parameters
    
    Returns:
    - params: the parameters of the best fit
    - err: the average error on the fit
    - X_pred: the predicted points of the ideal trajectory fitted
    """
    
    ## Base function for tracing the trajectory based on trace_ideal_trajectory
    def f_base(q,d_x,p_0,K_over_p0,theta_0,x_0,y_0,z_0):
        points=trace_ideal_trajectory(s,d_x,p_0,K=K_over_p0*p_0,q=q,theta_0=theta_0,X_0=np.array([x_0,y_0,z_0]))
        return X-points
    
    def fqP(x):
        print(x)
        return f_base(+1,*x).reshape((-1,))
    
    def fqN(x):
        return f_base(-1,*x).reshape((-1,))
    
    def fq0(x):
        return f_base(0,*x).reshape((-1,))
    
    
    def jac_base(q,d_x,p_0,K_over_p0,theta_0,x_0,y_0,z_0):
        jacobian=jac_ideal_trajectory(s,d_x,p_0,K=K_over_p0*p_0,q=q,theta_0=theta_0,X_0=np.array([x_0,y_0,z_0]))
        return jacobian
    
    def jac_fqP(x):
        return jac_base(+1,*x).reshape((-1,7))
    
    def jac_fqN(x):
        return jac_base(-1,*x).reshape((-1,7))
    
    def jac_fq0(x):
        return jac_base(0,*x).reshape((-1,7))
    
    
    try:
        optP=least_squares(fqP,x0=guess,bounds=bounds,jac=jac_fqP)
        params_qP=optP['x']
        X_pred_qP=X-f_base(+1,*params_qP)
        err_qP=np.mean((X-X_pred_qP)**2)
    except RuntimeError as R:
        err_qP=1e9
        print("ERROR qP")
        
    try:
        optN=least_squares(fqN,x0=guess,bounds=bounds,jac=jac_fqN)
        params_qN=optN['x']
        X_pred_qN=X-f_base(-1,*params_qN)
        err_qN=np.mean((X-X_pred_qN)**2)
    except RuntimeError as R:
        err_qN=1e9
        print("ERROR qN")
    
    try:
        opt0=least_squares(fq0,x0=guess,bounds=bounds,jac=jac_fq0)
        params_q0=opt0['x']
        X_pred_q0=X-f_base(0,*params_q0)
        err_q0=np.mean((X-X_pred_q0)**2)
    except RuntimeError as R:
        err_q0=1e9
        print("ERROR q0")
        
    ## Handle case where the three possible charge failed to converge
    if err_qP>1e8 and err_qN>1e8 and err_q0>1e8:
        print("TRIPLE ERROR")
        scale_=np.array(bounds[1]-bounds[0])
        return guess, 1e9, X
    
    if err_qP<err_qN:
        if err_qP<err_q0:
            q,params,err,X_pred=1,params_qP,err_qP,X_pred_qP
        else:
            q,params,err,X_pred=0,params_q0,err_q0,X_pred_q0
    else:
        if err_qN<err_q0:
            q,params,err,X_pred=-1,params_qN,err_qN,X_pred_qN
        else:
            q,params,err,X_pred=0,params_q0,err_q0,X_pred_q0
            
    params=[params[0],params[1],params[2]*params[1],q,params[3],np.array([params[4],params[5],params[6]])]
    print(err)
    print(params)
    return params, err, X_pred
    
    
def _fit_local_arc_circle(direction:torch.Tensor,
                        dir_order_matrix:torch.Tensor,
                        matrix_of_differences:torch.Tensor,
                        matrix_of_distances:torch.Tensor,
                        args_node_d:dict[str,float],
                        args_curv:dict[str,float],
                        guess:torch.Tensor=torch.Tensor([0.5,1000.,np.pi/2,0.,0.,0.]),
                        bounds:torch.Tensor=torch.Tensor([[-1.,1e-9,-np.pi,-10.,-10.,-10.],[1.,4e3,np.pi,10.,10.,10.]])):
    """
    Fit locally a simple trajectory model to the trajectory points.
    Relies on torch.optim.LBGFS to fit the parameters of trace_simple_trajectory
    The fit is dones locally, meaning that there is one fit per trajectory point.
    
    Parameters:
    - direction: array-like, shape (n_points, 3), the estimated direction of each trajectory point.
    - dir_order_matrix: array-like, shape (n_points, n_points), the antisymmetric sign matrix indicating the order of the points relative to each other (+1 if j after i, -1 if before).
    - matrix_of_differences: array-like, shape (n_points, n_points, 3), the matrix of difference vectors between the points.
    - matrix_of_distances: array-like, shape (n_points, n_points), the symmetric positive matrix of the distances between the points.
    - args_curv: dict, dictionnary containing parameters for the estimation of the curvature
    - guess : initial parameters values
    - bounds : the bounds for the parameters
     
    
    Returns:
    - charge: the charge of the particle
    - momentum norm
    - direction
    - curvature
    - parameters
    - error
    """
    
    
    curvScale_d = args_curv['curvScale_d']
    curvScale_h = args_curv['curvScale_h']
    p_scale = args_curv['p_scale']
    
    bounds=bounds.to(direction.device)
    guess=guess.to(direction.device)
    
    s=dir_order_matrix*matrix_of_distances
    parameters=torch.zeros((s.shape[0],guess.shape[0],3)).to(s.device)
    parameters[...]=guess[None,:,None]
    parameters[:,0,:]=direction[:,0,None]
    parameters[:,2,:]=torch.acos(direction[:,1,None])*torch.sign(direction[:,2,None])
    parameters=parameters.requires_grad_(True)
    
    
    
    ## Computes the distances projected along the estimated direction
    ## Ld_{ij} := M_{ij}.dir_i
    matrix_of_distances_along_the_direction=torch.sum(matrix_of_differences*direction[:,None,:],dim=-1)
    
    ## Computes the distances orthogonaly to the estimated direction
    ## Lh_{ij} := || M_{ij} - (M_{ij}.dir_i)*dir_i ||
    matrix_of_distances_orthogonally_to_the_direction=torch.linalg.norm(matrix_of_differences-matrix_of_distances_along_the_direction[:,:,None]*direction[:,None,:],dim=-1)
    
    weights_ortho=torch.exp(-(matrix_of_distances_orthogonally_to_the_direction**2)/(curvScale_h**2))
    
    
    def compute_loss(parameters_):
        params_=torch.clip(parameters_,min=bounds[[0],:,None],max=bounds[[1],:,None])
        d_x=params_[:,0]
        p_0=params_[:,1]
        theta_0=params_[:,2]
        x_0=params_[:,3]
        y_0=params_[:,4]
        z_0=params_[:,5]
        q=torch.Tensor([1.,0.,-1.]).to(s.device)[None,:]
        weights_along=torch.exp(-(matrix_of_distances_along_the_direction[:,:,None]**2)/((1+p_0/p_scale)*curvScale_d**2))
        weights=weights_ortho[:,:,None]*weights_along
        
        differences_predicted=trace_simple_trajectory(s[:,:,None]*torch.ones(list(s.shape)+[3]).to(s.device),
                                                d_x[:,None,:],
                                                p_0[:,None,:],
                                                theta_0[:,None,:],
                                                q[:,None,:],
                                                torch.cat([x_0[:,None,:,None],
                                                            y_0[:,None,:,None],
                                                            z_0[:,None,:,None],],
                                                        dim=-1),
                                                )
        
        err=((matrix_of_differences[:,:,None,:]-differences_predicted)**2).mean(dim=-1)
        err*=weights/weights.sum(dim=1,keepdim=True)
        err=err.mean(dim=1)
        loss_=err.mean()
        constrain_matrix=torch.ones_like(parameters_)
        constrain_loss=torch.nn.HuberLoss()(parameters_,constrain_matrix*bounds[[0],:,None])*(parameters_<bounds[[0],:,None])
        constrain_loss+=torch.nn.HuberLoss()(parameters_,constrain_matrix*bounds[[1],:,None])*(parameters_>bounds[[1],:,None])
        return loss_, err
    
    def closure():
        nonlocal parameters
        lbfgs.zero_grad()
        # print(parameters)
        loss,_ = compute_loss(parameters)
        loss.backward()
        # print(loss.item())
        # print(parameters.grad)
        return loss
    
    lbfgs=torch.optim.LBFGS([parameters],lr=1e-1,max_iter=200)
    
    lbfgs.step(closure)
    
    # optimizer = torch.optim.Adam([parameters],lr=1e0)
    
    # for k in range(100):
    #     optimizer.zero_grad()
    #     print(parameters)
    #     loss,_ = compute_loss(parameters)
    #     loss.backward()
    #     print(loss.item())
    #     print(parameters.grad)
    #     optimizer.step()
        
        
    _,err=compute_loss(parameters)
    charge_index=torch.argmin(err,dim=-1)
    params=parameters[torch.arange(s.shape[0]),:,charge_index]
    charge=1-charge_index
    dir_=torch.zeros((s.shape[0],3)).to(s.device)
    dir_[:,0]=params[:,0]
    dir_[:,1]=torch.cos(params[:,2])
    dir_[:,2]=torch.sin(params[:,2])
    mom_n=params[:,1]
    curv_=torch.zeros_like(dir_).to(s.device)
    curv_[:,1]=dir_[:,2]
    curv_[:,2]=dir_[:,1]
    curv_*=charge[:,None]*0.3*B_FIELD_INTENSITY/(mom_n[:,None]*torch.sqrt(1-dir_[:,[0]]**2))
    
    return charge, mom_n, dir_, curv_, params, err
        
    



def construct_direction_and_curvature(node_c_absolute:np.ndarray|torch.Tensor,
                                     node_d:np.ndarray|torch.Tensor|None=None,
                                     event_id:np.ndarray|torch.Tensor|None=None,
                                     traj_ID:np.ndarray|torch.Tensor|None=None,
                                     args_node_d:dict[str,float]=dict(weight_predicted_node_d=0.,scale_factor_node_d=1.),
                                     args_dir:dict[str,float]=dict(dirScale=70.),
                                     args_curv:dict[str,float]=dict(curvScale=300.),
                                     mode:str='order',
                                     chargeID_mode:str='curv_estimate',
                                    **kwargs
                                     ) -> tuple[np.ndarray|torch.Tensor, np.ndarray|torch.Tensor, np.ndarray|torch.Tensor, np.ndarray|torch.Tensor]:
    """
    Main function to estimate the charge, momentum, direction and curvature of trajectory points.
    
    Parameters:
    - node_c_absolute : array-like, shape (n_points, 3), the absolute coordinates of the trajectory points.
    - node_d : array-like, shape (n_points, 3), a first prediction of directions of the trajectory points (useful notably for choosing the direction sign).
    - event_id : array-like, shape (n_points,), the event ID of the trajectory points (to group points per event)
    - traj_ID : array-like, shape (n_points,), the trajectory ID of the trajectory points (to group points per trajectory)
    - args_node_d: dict, dictionary containing parameters for node_d usage in direction estimation (not used by default)
    - args_dir: dict, dictionnary containing parameters for the estimation of the direction from the trajectory points coordinates.
    - args_curv: dict, dictionnary containing paramters for the estimation of the curvature.
    - mode: str, the mode for the estimation of the direction sign ('order', 'node_d', 'PCAo', 'PCAnd')
    - chargeID_mode: str, the mode for the estimation of the charge ("curv_estimate", "simple_curv", "fit_whole_traj", "fit_local_traj")
    
    Returns:
    - charge : array-like, shape (n_points,), the estimated charge of the trajectory points.
    - momentum : array-like, shape (n_points, 3), the estimated momentum of the trajectory points (with momentum norm and direction).
    - curvature : array-like, shape (n_points, 3), the estimated curvature of the trajectory points
    - points_updated : array-like, shape (n_points, 3), the trajectory points shifted towards the estimated trajectory line
    - point_order_coord : array-like, shape (n_points,), the estimated coordinates of the points along the trajectory, ranging from -1 (the first point) to +1 (the last point)
    """
    
    
    
    if isinstance(node_c_absolute, np.ndarray):
        module_=np
        axis_arg={"axis":-1}
        keepdim_arg={"keepdims":True}
        use_torch=False
        
        
    elif isinstance(node_c_absolute, torch.Tensor):
        module_=torch
        axis_arg={"dim":-1}
        keepdim_arg={"keepdim":True}
        use_torch=True
        device=node_c_absolute.device
        
    else:
        raise ValueError("Unsupported data type. Please provide either NumPy array or PyTorch Tensor.")
    
    if event_id is None:
        event_id = module_.zeros((node_c_absolute.shape[0],1))
        if use_torch:
            event_id=event_id.to(device)
    
    if traj_ID is None:
        traj_ID = module_.zeros((node_c_absolute.shape[0],1))
        if use_torch:
            traj_ID=traj_ID.to(device)
        
    if node_d is not None:
        node_d_norm=module_.linalg.norm(node_d,**axis_arg)
        node_d/=node_d_norm[...,None]
        
    ## Selecting hits of same event and trajectory
    same_event_traj_matrix=(event_id[None,:]==event_id[:,None])
    same_event_traj_matrix*=(traj_ID[None,:]==traj_ID[:,None])
    
    ## Handling cases where we have only one hit in the event and trajectory
    _handle=_handle_single_hit_cases(same_event_traj_matrix,node_d,node_c_absolute)
    if _handle is not None:
        return _handle
        
    ## Computes the matrix of differences between the points positions
    ## M_{ij} := p_j - p_i
    matrix_of_differences = node_c_absolute[None,:,:]*same_event_traj_matrix-node_c_absolute[:,None,:]*same_event_traj_matrix
    
    ## Computes the matrix of distances
    ## D_{ij} :=  ||M_{ij}|| = ||p_j - p_i||
    matrix_of_distances = module_.linalg.norm(matrix_of_differences,**axis_arg)
    
    ## Get the estimated direction and direction order matrix
    ## dir_i and dirorder_{ij}
    direction,  dir_order_matrix = _estimate_direction(node_c_absolute,
                                                    matrix_of_differences,
                                                    matrix_of_distances,
                                                    node_d,
                                                    args_node_d,
                                                    args_dir,
                                                    mode,
                                                    )
    
        
    # assert np.allclose(np.linalg.norm(direction,axis=-1),1.), f"Direction must be a unit vector: {(~np.isclose(np.linalg.norm(direction,axis=-1),1.)).mean()}, {np.linalg.norm(direction,axis=-1).mean()}"
    point_order_coord = - module_.mean(1.*dir_order_matrix*same_event_traj_matrix[:,:,0],**axis_arg)/module_.mean(1.*same_event_traj_matrix[:,:,0],**axis_arg)
    
    
    if chargeID_mode == "curv_estimate":
        ## We use the curvature estimator to compute the curvature, and then deduce the charge and the momentum norm
        
        ## Shift the trajectory points to place them on the trajectory
        ## Computes the center of the neighbouring points, shifts the points towards it but perpendicularly to the direction of the trajectory
        ## Modifies in place matrix_of_differences and matrix_of_distances accordingly
        points_updated=_shift_points_along_trajectory(node_c_absolute,
                                                    direction,
                                                    same_event_traj_matrix,
                                                    matrix_of_differences,
                                                    matrix_of_distances,
                                                    args_dir,)
        
        ## Get the estimated curvature
        ## curv_i
        curvature = _estimate_curvature(direction,
                                        dir_order_matrix,
                                        matrix_of_differences,
                                        matrix_of_distances,
                                        args_curv,
                                        )

            
        charge=get_charge(curvature,direction)
        mom_n=get_momentum_magnitude(curvature,direction)
        
        
    elif chargeID_mode == "simple_curv":
        
        k=min(3, max(1, point_order_coord.shape[0]-4))
        if use_torch:
            first_points=torch.topk(point_order_coord,k,largest=False,sorted=False,).indices
            middle_points=torch.topk(torch.abs(point_order_coord),k,largest=False,sorted=False,).indices
            last_points=torch.topk(-point_order_coord,k,largest=False,sorted=False,).indices
        else:
            first_points=np.argpartition(point_order_coord,k)[:k]
            middle_points=np.argpartition(np.abs(point_order_coord),k)[:k]
            last_points=np.argpartition(-point_order_coord,k)[:k]
        
        beggining_average=node_c_absolute[first_points].mean(**{list(axis_arg.keys())[0]:0})
        middle_average=node_c_absolute[middle_points].mean(**{list(axis_arg.keys())[0]:0})
        end_average=node_c_absolute[last_points].mean(**{list(axis_arg.keys())[0]:0})
        
        dir_1=middle_average-beggining_average
        dir_2=end_average-middle_average
        
        
        dir_1/=(module_.linalg.norm(dir_1)**2+1e-9)
        dir_2/=(module_.linalg.norm(dir_2)**2+1e-9)
        
        cross=module_.cross(dir_2,dir_1)
        charge=module_.sign(cross)[0]*module_.ones_like(event_id)[:,0]
        avg_dir=direction[middle_points].mean(**{list(axis_arg.keys())[0]:0})
        curvature=module_.cross(avg_dir,cross)*module_.ones_like(direction)
        charge=module_.sign(module_.cross(direction,curvature,**axis_arg)[:,0])
        
        mom_n=get_momentum_magnitude(curvature,avg_dir)*module_.ones_like(event_id)[:,0]
        
        points_updated=node_c_absolute
        
        
        
    elif chargeID_mode == "fit_whole_traj":
        ## We fit the whole trajectory to a particle trajectory model and infer the charge and momentum norm from the fit
        
        ## Get the point at the center of the trajectory
        reference_point_index=np.argmin(np.abs(np.sum(dir_order_matrix,axis=1)))
        X=matrix_of_differences[reference_point_index]
        s=matrix_of_distances[reference_point_index]*dir_order_matrix[reference_point_index]
        guess=[0.,1000.,1e-4,0.,0.,0.,0.]
        guess[0]=np.mean(direction[...,0])
        guess[3]=np.mean(np.arccos(direction[...,1])*np.sign(direction[...,2]))
        parameters,err,points_updated=_fit_one_traj_with_scipy(s,X,guess=guess)
        points_updated+=node_c_absolute[[reference_point_index]]
        direction=trace_ideal_direction(s,*parameters[:-1])
        p_0=parameters[1]
        K=parameters[2]
        mom_n=(p_0-K*s)
        curvature=np.zeros_like(direction)
        charge=parameters[3]*np.ones_like(mom_n)
    
    
    elif chargeID_mode == "fit_local_traj":
        ## We fit locally the trajectory to a particle trajectory model and infer the charge and momentum norm from the fit 
        
        ## Shift the trajectory points to place them on the trajectory
        ## Computes the center of the neighbouring points, shifts the points towards it but perpendicularly to the direction of the trajectory
        ## Modifies in place matrix_of_differences and matrix_of_distances accordingly
        points_updated=_shift_points_along_trajectory(node_c_absolute,
                                                    direction,
                                                    same_event_traj_matrix,
                                                    matrix_of_differences,
                                                    matrix_of_distances,
                                                    args_dir,)
        
        ## Get the estimated curvature
        ## curv_i
        charge,mom_n,direction,curvature,parameters,err=_fit_local_arc_circle(direction,
                            dir_order_matrix,
                            matrix_of_differences,
                            matrix_of_distances,
                            args_node_d,
                            args_curv,
                            )
        
        points_updated[:,0]+=parameters[:,-3]
        points_updated[:,1]+=parameters[:,-2]
        points_updated[:,2]+=parameters[:,-1]
        
    
    else:
        raise ValueError(f"Unsupported chargeID_mode: {chargeID_mode}")
        
    # if node_d is not None:
    #     ## If the node_d is not a unit vector, but contains information about the momentum norm prediction, we use it
    #     ## To do that, we check that the node_d norm is above 1 MeV for 99% of the cases, if not, it means that node_d are unit vectors
    #     if (1.*(node_d_norm>1)).mean()>0.99:
    #         weight_predicted_node_d=args_node_d['weight_predicted_node_d']
    #         mom_n+=weight_predicted_node_d*node_d_norm
    #         mom_n/=(1.+weight_predicted_node_d)
    
    
    
    return charge, mom_n[:,None]*direction, curvature, points_updated, point_order_coord
    
   


def charge_and_momentum_fit(ret_dict:dict,
                        values:str='pred',
                        show_progressbar:bool=True,
                        N:int|None=None,
                        n:int=1,
                        device:torch.device|None=None,
                        **kwargs) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Execute construct_direction_and_curvature batch per batch to get the estimation of the charge, momentum, direction and curvature of trajectory points.
    
    Parameters:
    - ret_dict: dict, dictionary containing all the unscaled results sorted by event_id, traj_id and hit_order; of the testing of a model (the return of sort_event_from_all_results)
    - values: str, which trajectories to use ('pred' for the predictions of the model, 'recon' for the predictions of the Bayesian filter, 'from_truth' for the targets).
    - show_progressbar: bool, whether to display a progress bar or not
    - N: int, number of events to process (default is the maximum event_id)
    - n: int, number of events to process at once (batch size, default is 1)
    - device: torch.device, device to run the computations on
    
    Returns:
    - charge : array-like, shape (n_points,), the estimated charge of the trajectory points.
    - momentum : array-like, shape (n_points, 3), the estimated momentum of the trajectory points (with momentum norm and direction).
    - curvature : array-like, shape (n_points, 3), the estimated curvature of the trajectory points
    - points_updated : array-like, shape (n_points, 3), the trajectory points shifted towards the estimated trajectory line
    - point_order_coord : array-like, shape (n_points,), the estimated coordinates of the points along the trajectory, ranging from -1 (the first point) to +1 (the last point)
    """
    
    Charge=[]
    Momentum=[]
    Curvature=[]
    Points_updated=[]
    Point_order_coord=[]
    
    if device is not None:
        use_torch=True
        module_=torch
        for key in ret_dict.keys():
            if not isinstance(ret_dict[key], torch.Tensor):
                ret_dict[key]=torch.Tensor(ret_dict[key]).to(device) if ret_dict[key] is not None else None
    else:
        use_torch=False
        module_=np
    
    if values=='pred':
        node_c_absolute=ret_dict['node_c_absolute']
        node_d=ret_dict['node_d']
    elif values=='recon':
        node_c_absolute=ret_dict['recon_node_c_absolute']
        node_d=ret_dict['recon_node_d']
    elif values=='from_truth':
        node_c_absolute=ret_dict['true_node_c_absolute']
        node_d=ret_dict['true_node_d']
    elif values=='truth':
        return ret_dict['true_charge'],ret_dict['true_momentum']
    else:
        raise ValueError(f"Mode {values} not recognized")
    
    
    N=int(module_.max(ret_dict['event_id']) if N is None else N)
    i=0
    progress_bar=tqdm.tqdm(range(N//n+2),disable=(not show_progressbar))
    
    for k in progress_bar:
        
        j=module_.searchsorted(ret_dict['event_id'][:,0],k*n,side='right')
        if use_torch:
            j=j.item()
            
        if i==j:
            continue
        
        ind=module_.arange(i,j)
        if use_torch:
            ind=ind.to(device)
        
        progress_bar.set_postfix({"size":str(j-i)})
         
        i=j
        
        charge,mom,curv,points_u,p_o_c=construct_direction_and_curvature(node_c_absolute=node_c_absolute[ind],
                                                       node_d=node_d[ind] if node_d is not None else None,
                                                       event_id=ret_dict['event_id'][ind],
                                                       traj_ID=ret_dict['traj_id'][ind],
                                                       **kwargs)
        assert len(node_c_absolute[ind]) == len(charge)
        assert len(node_c_absolute[ind]) == len(mom)
        
        
        if use_torch:
            Charge.append(charge.detach().cpu().numpy())
            Momentum.append(mom.detach().cpu().numpy())
            Curvature.append(curv.detach().cpu().numpy())
            Points_updated.append(points_u.detach().cpu().numpy())
            Point_order_coord.append(p_o_c.detach().cpu().numpy())
            
        else:
            Charge.append(charge.copy())
            Momentum.append(mom.copy())
            Curvature.append(curv.copy())
            Points_updated.append(points_u.copy())
            Point_order_coord.append(p_o_c.copy())
        
        
    charge=np.concatenate(Charge,axis=0)
    mom=np.concatenate(Momentum,axis=0)
    curv=np.concatenate(Curvature,axis=0)
    points_u=np.concatenate(Points_updated,axis=0)
    Point_order_coord=np.concatenate(Point_order_coord,axis=0)
    
    if use_torch:
        for key in ret_dict.keys():
            if isinstance(ret_dict[key], torch.Tensor):
                ret_dict[key]=ret_dict[key].detach().cpu().numpy() if ret_dict[key] is not None else None
    
    
    return charge, mom, curv, points_u, Point_order_coord



def update_dict(ret_dict:dict,
                charge:np.ndarray,
                momentum:np.ndarray,
                curvature:np.ndarray,
                points_updated:np.ndarray,
                **kwargs) -> dict:
    """
    Copy a dictionnary of results and add the charge, momentum, curvature and points_updated to it.
    """
    
    ret2={}
    for key in ret_dict.keys():
        ret2[key]=ret_dict[key]
    
    ret2['charge']=charge
    ret2['curvature']=curvature
    ret2['predictions']=np.concatenate([points_updated-ret2['c'],momentum],axis=-1)
    ret2['node_c_absolute']=points_updated
    
    for key in kwargs.keys():
        ret2[key]=kwargs[key]
    
    return ret2



def group_per_traj(ret_dict_updated:dict,
                        show_progressbar:bool=True,
                        N:int|None=None,
                        null_charge_threshold:float=0.5) -> dict:
    """
    Re estimate the charge and momentum at the trajectory level.
    
    Parameters:
    - ret_dict_updated: dict, dictionary containing all the results (return of update_dict)
    - show_progressbar: bool, whether to display a progress bar or not
    - N: int, number of events to process (default is the maximum event_id)
    - null_charge_threshold: float, value of the estimated charge below which it is assigned to a null charge
    
    Returns:
    - ret2 : dict, dictionary containing the reestimated charge and momentum at the trajectory level
    """
    
    N=int(np.max(ret_dict_updated['event_id']) if N is None else N)
    i=0
    
    count_of_discarded_traj=0
    count_of_traj=0
    
    ret2={}
    ret2['charge']=[]
    ret2['true_charge']=[]
    ret2['true_momentum_norm']=[]
    ret2['traj_length']=[]
    keys=[]
    for key in ret_dict_updated.keys():
        if ret_dict_updated[key] is not None and key not in ret2.keys():
            ret2[key]=[]
            keys.append(key)
            
    
    for k in tqdm.tqdm(range(N), desc="Event loop", disable=(not show_progressbar)) :
        j=np.searchsorted(ret_dict_updated['event_id'][:,0],k,side='right')
        
        if i==j:
            continue
        
        indexes=np.arange(i,j)
        i=j
        
        Ntraj=int(np.max(ret_dict_updated['traj_id'][indexes,0]))
        
        if Ntraj>1:
            itraj=0
            for traj_id in range(Ntraj):
                
                jtraj=np.searchsorted(ret_dict_updated['traj_id'][indexes,0],traj_id,side='right')
                if itraj==jtraj:
                    continue
                
                sub_indexes=np.arange(itraj,jtraj)
                itraj=jtraj
                
                
                true_charge_average=np.mean(ret_dict_updated['true_charge'][indexes][sub_indexes])
                true_charge=np.round(true_charge_average)
                
                if (np.abs(true_charge_average-true_charge)>1e-2).any():
                    print(f"ERROR: incorrect averaging of the charge for event {k} trajectory {traj_id} with average {true_charge_average}")
                    count_of_discarded_traj+=1
                    continue
                
                for key in keys:
                    ret2[key].append(np.mean(ret_dict_updated[key][indexes][sub_indexes],axis=0,keepdims=True))
                    
                direction=ret2['predictions'][indexes][sub_indexes][:,3:]
                direction/=(np.linalg.norm(direction,axis=1,keepdims=True)+1e-9)
                direction=direction.mean(axis=0)
                first_point=np.argmin(ret2['point_order_coordinate'][indexes][sub_indexes])
                last_point=np.argmax(ret2['point_order_coordinate'][indexes][sub_indexes])
                ret2['traj_length'].append(np.abs(np.sum((ret2['node_c_absolute'][indexes][sub_indexes][last_point]-ret2['node_c_absolute'][indexes][sub_indexes][first_point])*direction)))
                
                ret2['true_charge'].append(true_charge)
                
                charge_average=np.mean(ret_dict_updated['charge'][indexes][sub_indexes])
                charge=np.round(charge_average)*(charge_average>null_charge_threshold)
                
                ret2['charge'].append(charge)
                
                true_momentum_norm=np.mean(np.linalg.norm(ret_dict_updated['true_momentum'][indexes][sub_indexes],axis=-1))
                
                ret2['true_momentum_norm'].append(true_momentum_norm)
                count_of_traj+=1
        
        else:
            
            
            true_charge_average=np.mean(ret_dict_updated['true_charge'][indexes])
            true_charge=np.round(true_charge_average)
            
            if (np.abs(true_charge_average-true_charge)>1e-2).any():
                # print(f"ERROR: incorrect averaging of the charge for event {k} trajectory 0 with average {true_charge_average}")
                count_of_discarded_traj+=1
                continue
            
            for key in keys:
                ret2[key].append(np.mean(ret_dict_updated[key][indexes],axis=0,keepdims=True))
                
            direction=ret2['predictions'][indexes][:,3:]
            direction/=(np.linalg.norm(direction,axis=1,keepdims=True)+1e-9)
            direction=direction.mean(axis=0)
            first_point=np.argmin(ret2['point_order_coordinate'][indexes])
            last_point=np.argmax(ret2['point_order_coordinate'][indexes])
            ret2['traj_length'].append(np.abs(np.sum((ret2['node_c_absolute'][indexes][last_point]-ret2['node_c_absolute'][indexes][first_point])*direction)))
            
            ret2['true_charge'].append(true_charge)
            
            charge_average=np.mean(ret_dict_updated['charge'][indexes])
            charge=np.sign(charge_average)*(np.abs(charge_average)>null_charge_threshold)
            
            ret2['charge'].append(charge)
            
            true_momentum_norm=np.mean(np.linalg.norm(ret_dict_updated['true_momentum'][indexes],axis=-1))
            
            ret2['true_momentum_norm'].append(true_momentum_norm)
            count_of_traj+=1
            
        
    for key in ret2.keys():
        try:
            ret2[key]=np.concatenate(ret2[key],axis=0)
        except ValueError:
            ret2[key]=np.array(ret2[key])
        
    print(f"Number of discarded trajectories: {count_of_discarded_traj} | {100*count_of_discarded_traj/count_of_traj:.2f}%")
        
    return ret2
            
            


def test_dirScale(ret_dict:dict,
                values:str='from_truth',
                N:int|None=4999,
                n:int=1,
                device:torch.device|None=None,
                min_dirScale:float=2.,
                max_dirScale:float=200.,
                num_points:int=100,
                **kwargs) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Test different values for the scale parameter of the weighting kernel in the direction estimation.
    
    Parameters:
    - ret_dict: dict, dictionary containing all the results (return of sort_event_from_all_results)
    - values: str, which trajectories to use ('pred' for the predictions of the model, 'recon' for the predictions of the Bayesian filter, 'from_truth' for the targets).
    - N: int, number of events to process (None is the maximum event_id)
    - n: int, batch size
    - device: torch.device, device on which the computations are performed (None for CPU)
    - min_dirScale: float, minimum value of the scale parameter for the direction estimation.
    - max_dirScale: float, maximum value of the scale parameter for the direction estimation.
    - num_points: int, number of points to test the direction scales.
    
    Returns:
    - dirScales: np.ndarray, the scale parameter for the direction estimation tested.
    - Mean: np.ndarray, the mean direction anglular distances for the tested scale parameters.
    - Res: np.ndarray, the 68% of the direction angular distances
    - Inverted: np.ndarray, the percentage of the direction angular distances that are inverted (i.e., angle > pi/2).
    """
        
        
    dirScales = np.linspace(min_dirScale,max_dirScale,num_points)
    Mean=[]
    Res=[]
    Inverted=[]
    
    for l in tqdm.tqdm(range(len(dirScales)),desc="Testing the dir scales",leave=True,position=0):
        
        dirScale=dirScales[l]
        
        
        _,mom_pred,_,_,_=charge_and_momentum_fit(ret_dict=ret_dict,
                                                                            values=values,
                                                                            show_progressbar=False,
                                                                            N=N,
                                                                            n=n,
                                                                            device=device,
                                                                            args_dir=dict(dirScale=dirScale),
                                                                            **kwargs)
        
        len_=mom_pred.shape[0]
        true_momentum=ret_dict['true_momentum'][:len_]
        
        
        scalar_prod = np.sum(mom_pred*true_momentum,axis=-1)/(np.linalg.norm(mom_pred,axis=-1)*np.linalg.norm(true_momentum,axis=-1)+1e-6)
        scalar_prod = np.clip(scalar_prod,-1.,1.)
        direction_angle = np.arccos(scalar_prod)
        
    
        
        mean_ = np.mean(direction_angle)
        res_ = np.quantile(direction_angle,q=0.68)
        inverted_ = (direction_angle > np.pi/2).mean()
        
        Mean.append(mean_)
        Res.append(res_)
        Inverted.append(inverted_)
        
    return dirScales,np.array(Mean),np.array(Res),np.array(Inverted)




def test_curvScale(ret_dict:dict,
                values:str='from_truth',
                N:int|None=4999,
                n:int=1,
                device:torch.device|None=None,
                min_curvScale:float=5.,
                max_curvScale:float=500.,
                num_points:int=100,
                **kwargs) -> tuple[np.ndarray,np.ndarray]:
    """
    Test different values for the scale parameter of the weighting kernel in the curvature estimation.
    
    Parameters:
    - ret_dict: dict, dictionary containing all the results (return of sort_event_from_all_results)
    - values: str, which trajectories to use ('pred' for the predictions of the model, 'recon' for the predictions of the Bayesian filter, 'from_truth' for the targets).
    - N: int, number of events to process (None is the maximum event_id)
    - n: int, batch size
    - device: torch.device, device on which the computations are performed (None for CPU)
    - min_dirScale: float, minimum value of the scale parameter for the direction estimation.
    - max_dirScale: float, maximum value of the scale parameter for the direction estimation.
    - num_points: int, number of points to test the direction scales.
    
    Returns:
    - curvScales: np.ndarray, the scale parameter for the curvature estimation tested.
    - Acc: np.ndarray, the accuracy of the charge prediction for the tested scale parameters.
    """
        
        
    curvScales = np.linspace(min_curvScale,max_curvScale,num_points)
    # Prec=[]
    # Rec=[]
    # F1=[]
    Acc=[]
    
    for l in tqdm.tqdm(range(len(curvScales)),desc="Testing the curv scales",leave=True,position=0):
        
        curvScale=curvScales[l]
        
        
        charge_pred,_,_,_,_=charge_and_momentum_fit(ret_dict=ret_dict,
                                                                            values=values,
                                                                            show_progressbar=False,
                                                                            N=N,
                                                                            n=n,
                                                                            device=device,
                                                                            args_curv=dict(curvScale=curvScale),
                                                                            **kwargs)
        
        len_=charge_pred.shape[0]
        # prec_,rec_,f1_,_=precision_recall_fscore_support(y_true=np.round(ret_dict['true_charge'][:len_]).astype(int),
        #                                  y_pred=np.round(charge_pred).astype(int),
        #                                  average='weighted',
        #                                  zero_division=0.)
        # Prec.append(prec_)
        # Rec.append(rec_)
        # F1.append(f1_)
        Acc.append((ret_dict['true_charge'][:len_].astype(int)==charge_pred.astype(int)).mean())
        
    # return curvScales,np.array(Prec),np.array(Rec),np.array(F1)
    return curvScales,Acc
    


def optimize_smoothing_parameters_gpu(ret_dict:dict,
                        values:str='pred',
                        show_progressbar:bool=True,
                        device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        lr:float=1e-3,
                        parameters:np.ndarray|torch.Tensor|list=[56.06 ,  2.39 ,  2.52],
                        **kwargs
                        ):
    
    if values=='pred':
        node_c_absolute=ret_dict['node_c_absolute']
        node_d=ret_dict['node_d']
    elif values=='recon':
        node_c_absolute=ret_dict['recon_node_c_absolute']
        node_d=ret_dict['recon_node_d']
    elif values=='from_truth':
        node_c_absolute=ret_dict['true_node_c_absolute']
        node_d=ret_dict['true_node_d']
    elif values=='truth':
        return ret_dict['true_charge'],ret_dict['true_momentum']
    else:
        raise ValueError(f"Mode {values} not recognized")
    
    
    node_c_absolute=torch.Tensor(node_c_absolute).to(device)
    node_d=torch.Tensor(node_d).to(device) if node_d is not None else None
    event_id=torch.Tensor(ret_dict['event_id']).to(device)
    traj_id=torch.Tensor(ret_dict['traj_id']).to(device)
    true_mom=torch.Tensor(ret_dict['true_momentum']).to(device)
    
    n=8
    N=np.max(ret_dict['event_id'])
    i=0
    
    parameters=torch.Tensor(parameters).to(device).requires_grad_(True)
    optimizer=torch.optim.Adam([parameters],lr=lr)
    
    train_loop=tqdm.tqdm(range(N//(2*n)),disable=(not show_progressbar),desc="Tuning the parameters")
    for k in train_loop:
        
        j=np.searchsorted(ret_dict['event_id'][:,0],k*n,side='right')
        if i==j:
            continue
        ind=torch.arange(i,j)
        i=j
        
        charge,mom,_=construct_direction_and_curvature_gpu(node_c_absolute=node_c_absolute[ind],
                                                       node_d=node_d[ind] if node_d is not None else None,
                                                       event_id=event_id[ind],
                                                       traj_ID=traj_id[ind],
                                                       device=device,
                                                       dirScale=parameters[0],
                                                       scale_factor_node_d=parameters[1],
                                                       weight_predicted_node_d=parameters[2],
                                                        **kwargs
                                                       )
        
        target=true_mom[ind]
        loss=(1.-torch.sum(mom*target,dim=-1)/(torch.linalg.norm(target,dim=-1)*torch.linalg.norm(mom,dim=-1)+1e-9)).mean()
        regularisation=((parameters<1e-3)*(parameters-1e-3)**2).mean()*3e6
        regularisation+=((parameters[[1,2]]>20)*(parameters[[1,2]]-20)**2).mean()
        regularisation+=((parameters[0]>200)*(parameters[0]-200)**2).mean()
        loss+=regularisation
        try:
            loss.backward()
        except RuntimeError as E:
            print(parameters.cpu().detach().tolist())
            raise E
        
        if k%10==0:
            train_loop.set_postfix({"loss":  f"{loss.item():.5f}", "params":parameters.cpu().detach().tolist()})
            
        optimizer.step()
        
    test_loop=tqdm.tqdm(range(N//(2*n),N//n+2),disable=(not show_progressbar),desc="Testing the parameters")
    loss_average=0.
    count=0
    for k in test_loop:
        
        j=np.searchsorted(ret_dict['event_id'][:,0],k*n,side='right')
        if i==j:
            continue
        ind=torch.arange(i,j)
        i=j
        
        charge,mom,_=construct_direction_and_curvature_gpu(node_c_absolute=node_c_absolute[ind],
                                                       node_d=node_d[ind] if node_d is not None else None,
                                                       event_id=event_id[ind],
                                                       traj_ID=traj_id[ind],
                                                       device=device,
                                                       dirScale=parameters[0],
                                                       scale_factor_node_d=parameters[1],
                                                       weight_predicted_node_d=parameters[2],
                                                        **kwargs
                                                       )
        target=true_mom[ind]
        loss=(1.-torch.sum(mom*target,dim=-1)/(torch.linalg.norm(target,dim=-1)*torch.linalg.norm(mom,dim=-1)+1e-9)).mean()
        if k%10==0:
            test_loop.set_postfix({"loss":  f"{loss.item():.5f}",})
        loss_average+=loss.item()
        count+=1
    
    return parameters.cpu().detach().numpy(), loss_average/count