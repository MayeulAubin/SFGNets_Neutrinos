import numpy as np
import tqdm
import torch

B_FIELD_INTENSITY = 0.2 # magnitude of the B field (supposed to be along X) in Teslas

DEFAULT_MOM_NORM= 100.

def get_charge(curvature:np.ndarray|float, direction:np.ndarray) -> np.ndarray|float:
    charge = np.cross(curvature, direction)[...,0]/B_FIELD_INTENSITY # B is along X, the dot product is just the first index times B intensity
    return np.sign(charge)
    

def get_momentum_magnitude(curvature:np.ndarray|float, direction:np.ndarray) -> np.ndarray|float:
    factor= -0.3 * B_FIELD_INTENSITY / np.sqrt(1.-direction[...,0]**2)
    return np.clip(np.abs(factor/(curvature+1e-9)),0,4000) 




def choose_direction_sign(node_c_absolute:np.ndarray,
                          matrix_of_differences:np.ndarray,
                          weights_for_differences:np.ndarray,
                          node_d:np.ndarray|None,
                          mode:str) -> np.ndarray:
    
    ## Now we need to know if the difference vectors need to be counted positively or negatively, depending on their sign along the direction of the track
    ## Since we don't know yet the direction of the track (that's the goal of this algorithm), we need an alternative way to get their sign
    ## The following options are available:
    
    if mode=='order':
        ## We use the order of the hits: hits before are counted negatively, hits after positively
        ## Nodes_c is supposed to be ordered by their position along the track
        dir_order_matrix=np.ones((node_c_absolute.shape[0], node_c_absolute.shape[0]))
        dir_order_matrix=np.triu(dir_order_matrix)-np.tril(dir_order_matrix)
    
    elif mode=='node_d':
        ## We use the predicted direction 'node_d' as an approximation: differences with a positive scalar product with node_d are counted positively, differences with a negative scalar product negatively
        if node_d is not None:
            dir_order_matrix=np.sign(np.sum(node_d[None,:,:]*matrix_of_differences,axis=-1))
        else:
            raise ValueError("node_d is None while mode is 'node_d'")
        
    elif mode[:3]=='PCA':
        ## We compute a local PCA to get the local axis of most elongation, then we sign it using a submode order or node_d, and then we use this approximate direction in the same way as 'node_d'
        M=matrix_of_differences*weights_for_differences[:,:,None] # shape (N,N,3)
        M_mean=M.mean(axis=1,keepdims=True)
        # Compute covariance using einsum
        covariance_M = np.einsum('ijk,ijl->ikl', M - M_mean, M - M_mean) / (M.shape[0] - 1) # shape (N,3,3), covariance_M[i]=Cov(M[i])
        
        try:
            # Compute eigenvalues and eigenvectors for each covariance matrix
            eigenval, eigenvect = np.linalg.eigh(covariance_M)
            # Find the indices of the eigenvectors with the largest eigenvalues
            max_eigenvector_indices = np.argmax(eigenval, axis=1)
            # Use fancy indexing to get the corresponding eigenvectors
            approx_dir = eigenvect[np.arange(M.shape[0]), max_eigenvector_indices]
            # Compute the scalar product of the difference vectors on the approximate direction to get the sign
            dir_order_matrix=np.sign(np.sum(approx_dir[None,:,:]*matrix_of_differences,axis=-1))
            PCA_failed=False
            
        except np.linalg.LinAlgError:
            print('PCA failed, defaulting to order mode')
            # If the eigenvectors do not converge, default to order mode:
            PCA_failed=True
            dir_order_matrix=np.ones((node_c_absolute.shape[0], node_c_absolute.shape[0]))
            dir_order_matrix=np.triu(dir_order_matrix)-np.tril(dir_order_matrix)
        
        if mode=='PCAo' or mode=='PCA':
            if not PCA_failed:
                ## Here we use the 'order' mode to get the correct sign of the approximate direction provided by the eigenvector.
                ## To do this we check that, on average, hits after are counted positively and before negatively
                hit_order=np.ones((M.shape[0], M.shape[0]))
                hit_order=(np.triu(hit_order)-np.tril(hit_order))
                hit_order*=dir_order_matrix*weights_for_differences
                dir_order_matrix*=np.sign(np.sum(hit_order,axis=1,keepdims=True))
            
        elif mode=='PCAnd':
            if not PCA_failed:
                if node_d is not None:
                    ## Here we use the 'node_d' mode to get the correct sign of the approximate direction provided by the eigenvector.
                    ## To do this we check that the scalar product between the preidcted node_d and the approximate direction is positive.
                    eigenvect_sign=np.sum(node_d*approx_dir,axis=-1)
                    dir_order_matrix*=np.sign(eigenvect_sign)
                else:
                    raise ValueError("node_d is None while mode is 'PCAnd'")
                
        elif mode=='PCAl':
            if not PCA_failed:
                ## Here we make sure that the eigenvectors are locally along the same axis.
                ## To do that we check that the scalar product between approximate directions is positive locally
                eigenvect_sign=np.sum(approx_dir[None,:,:]*approx_dir[:,None,:]*weights_for_differences[:,:,None],axis=(1,2))
                dir_order_matrix*=np.sign(eigenvect_sign)
        
        else:
            raise ValueError(f"mode {mode} not recognized")
    else:
        raise ValueError(f"mode {mode} not recognized")
    
    return dir_order_matrix




def construct_direction_and_curvature(node_c_absolute:np.ndarray,
                                     node_d:np.ndarray|None=None,
                                     event_id:np.ndarray|None=None,
                                     traj_ID:np.ndarray|None=None,
                                     weight_predicted_node_d:float=0.,
                                     dirScale:float=30.,
                                     curvScale:float=60.,
                                     curvCutoff:float=10.,
                                     scale_factor_node_d:float=1.,
                                     mode:str='order',
                                    **kwargs
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    
    
    
    if event_id is None:
        event_id = np.zeros(node_c_absolute.shape[0])
    
    if traj_ID is None:
        traj_ID = np.zeros(node_c_absolute.shape[0])
        
    if node_d is not None:
        node_d_norm=np.linalg.norm(node_d,axis=-1)
        node_d/=node_d_norm[...,None]
        
    
    same_event_traj_matrix=(event_id[None,:]==event_id[:,None])
    same_event_traj_matrix*=(traj_ID[None,:]==traj_ID[:,None])
    
    if (np.sum(same_event_traj_matrix,axis=1)<2).any():
        ## We have at least one hit alone
        if len(same_event_traj_matrix)==1:
            ## We have only one hit in the event
            if node_d is not None:
                return np.array([0.]), DEFAULT_MOM_NORM*node_d, np.array([[0.,0.,0.]]) # defaulting to 0. charge, node_d momentum of norm default, and 0. curvature
            else:
                return np.array([0.]), DEFAULT_MOM_NORM*np.array([[0.,0.,1.]]), np.array([[0.,0.,0.]]) # defaulting to 0. charge, momentum of norm default along Z, and 0. curvature
        else:
            same_event_traj_matrix[(np.sum(same_event_traj_matrix,axis=1)<2)]=1. # use all hits available if the event
    
    matrix_of_differences=np.zeros((node_c_absolute.shape[0], node_c_absolute.shape[0], 3))
    matrix_of_differences[:,:,:]=node_c_absolute[None,:,:]*same_event_traj_matrix-node_c_absolute[:,None,:]*same_event_traj_matrix
    
    matrix_of_distances=np.linalg.norm(matrix_of_differences,axis=-1)
    
    weights_differences_for_directions=(matrix_of_distances/dirScale)*np.exp(-(matrix_of_distances/dirScale))
    weights_differences_for_curvature=(matrix_of_distances/curvScale)*np.exp(-(matrix_of_distances/curvScale))*(1-np.exp(-5.*(matrix_of_distances**2/curvCutoff**2)))
    
    ## Now we need to know if the difference vectors need to be counted positively or negatively, depending on their sign along the direction of the track
    ## Since we don't know yet the direction of the track (that's the goal of this algorithm), we need an alternative way to get their sign
    
    dir_order_matrix=choose_direction_sign(node_c_absolute,matrix_of_differences,weights_differences_for_directions,node_d,mode)
    
    
    dir_from_differences=np.sum(weights_differences_for_directions[:,:,None]*matrix_of_differences*dir_order_matrix[:,:,None]/(matrix_of_distances[:,:,None]+1e-9),axis=1)/(np.sum(weights_differences_for_directions,axis=1)[:,None]+1e-9)
    
    curv_from_differences=np.sum(weights_differences_for_curvature[:,:,None]*matrix_of_differences/(matrix_of_distances[:,:,None]**2+1e-9),axis=1)/(np.sum(weights_differences_for_curvature,axis=1)[:,None]+1e-9)
    
    if node_d is not None:
        ## If node_d is available, we use it by averaging it with the reconstructed direction.
        ## We compute the average locally, using the same kind of weighting, but with a scale that can be different
        
        weights_node_d_for_directions=(matrix_of_distances/(dirScale*scale_factor_node_d))*np.exp(-(matrix_of_distances/(dirScale*scale_factor_node_d)))
        weights_node_d_for_curvature=(matrix_of_distances/(curvScale*scale_factor_node_d))*np.exp(-(matrix_of_distances/(curvScale*scale_factor_node_d)))*(1-np.exp(-5.*(matrix_of_distances**2/curvCutoff**2)))
        
        
        dir_from_node_d=np.sum(weights_node_d_for_directions[:,:,None]*node_d[None,:,:],axis=1)/(np.sum(weights_node_d_for_directions,axis=1)[:,None]+1e-9)
        curv_from_node_d=np.sum(weights_node_d_for_curvature[:,:,None]*node_d[None,:,:]*dir_order_matrix[:,:,None]/(matrix_of_distances[:,:,None]+1e-9),axis=1)/(np.sum(weights_node_d_for_curvature,axis=1)[:,None]+1e-9)
        direction=(dir_from_differences+weight_predicted_node_d*dir_from_node_d)/(1.+weight_predicted_node_d)
        curvature=(curv_from_differences+weight_predicted_node_d*curv_from_node_d)/(1.+weight_predicted_node_d)
    else:
        direction=dir_from_differences
        curvature=curv_from_differences
        
    direction/=(np.linalg.norm(direction,axis=-1,keepdims=True)+1e-9)
        
    assert np.allclose(np.linalg.norm(direction,axis=-1),1.), f"Direction must be a unit vector: {(~np.isclose(np.linalg.norm(direction,axis=-1),1.)).mean()}, {np.linalg.norm(direction,axis=-1).mean()}"
    
    curvature = curvature - np.sum(curvature*direction,axis=-1,keepdims=True)*direction
        
    charge=get_charge(curvature,direction)
    mom_n=get_momentum_magnitude(np.linalg.norm(curvature,axis=-1),direction)
    
    if node_d is not None:
        ## If the node_d is not a unit vector, but contains information about the momentum norm prediction, we use it
        ## To do that, we check that the node_d norm is above 1 MeV for 99% of the cases, if not, it means that node_d are unit vectors
        if (node_d_norm>1).mean()>0.99:
            mom_n+=weight_predicted_node_d*node_d_norm
            mom_n/=(1.+weight_predicted_node_d)
    
    return charge, mom_n[:,None]*direction, curvature
    


def sort_event_from_all_results(all_results:dict) -> dict:
    
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
        ret_dict['node_d']/=np.linalg.norm(ret_dict['node_d'],axis=-1,keepdims=True)
    else:
        ret_dict['node_d']=None
    
    ret_dict['event_id']=event_id[indexes]
    ret_dict['traj_id']=traj_id[indexes]
    ret_dict['order_index']=order_index[indexes]
    
    ret_dict['true_node_c_absolute']=(all_results['y'][:,:3]+all_results['c'])[exclude_not_attributed_hits][indexes]
    ret_dict['true_node_d']=all_results['aux'][:,11:14][exclude_not_attributed_hits][indexes]
    ret_dict['true_momentum']=all_results['aux'][:,4:7][exclude_not_attributed_hits][indexes]
    ret_dict['true_charge']=all_results['aux'][:,24][exclude_not_attributed_hits][indexes]
    ret_dict['y']=np.concatenate([all_results['y'][:,:3],all_results['aux'][:,4:7]],axis=-1)[exclude_not_attributed_hits][indexes]
    
    ret_dict['recon_node_c_absolute']=(all_results['aux'][:,17:20]+all_results['c'])[exclude_not_attributed_hits][indexes]
    ret_dict['recon_node_d']=all_results['aux'][:,20:23][exclude_not_attributed_hits][indexes]
    
    ret_dict['exclude_not_attributed_hits']=exclude_not_attributed_hits
    ret_dict['indexes']=indexes
    
    return ret_dict


def charge_and_momentum(ret_dict:dict,
                        values:str='pred',
                        show_progressbar:bool=True,
                        **kwargs):
    
    Charge=[]
    Momentum=[]
    Curvature=[]
    
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
    
    n=1
    N=np.max(ret_dict['event_id'])
    i=0
    
    for k in tqdm.tqdm(range(N//n+1),disable=(not show_progressbar)):
        
        j=np.searchsorted(ret_dict['event_id'][:,0],k*n,side='right')
        if i==j:
            continue
        ind=np.arange(i,j)
        i=j
        
        charge,mom,curv=construct_direction_and_curvature(node_c_absolute=node_c_absolute[ind],
                                                       node_d=node_d[ind] if node_d is not None else None,
                                                       event_id=ret_dict['event_id'][ind],
                                                       traj_ID=ret_dict['traj_id'][ind],
                                                       **kwargs)
        assert len(node_c_absolute[ind]) == len(charge)
        assert len(node_c_absolute[ind]) == len(mom)
        
        
        Charge.append(charge.copy())
        Momentum.append(mom.copy())
        Curvature.append(curv.copy())
        
        
    charge=np.concatenate(Charge,axis=0)
    mom=np.concatenate(Momentum,axis=0)
    curv=np.concatenate(Curvature,axis=0)
    
    
    return charge, mom, curv







def get_charge_gpu(curvature:torch.Tensor, direction:torch.Tensor) -> torch.Tensor:
    charge = torch.cross(curvature, direction)[...,0]/B_FIELD_INTENSITY # B is along X, the dot product is just the first index times B intensity
    return torch.sign(charge)
    

def get_momentum_magnitude_gpu(curvature:torch.Tensor, direction:torch.Tensor) -> torch.Tensor:
    factor= -0.3 * B_FIELD_INTENSITY / torch.sqrt(1.-direction[...,0]**2)
    return torch.clip(torch.abs(factor/(curvature+1e-9)),0,4000) 




def choose_direction_sign_gpu(node_c_absolute:torch.Tensor,
                          matrix_of_differences:torch.Tensor,
                          weights_for_differences:torch.Tensor,
                          node_d:torch.Tensor|None,
                          mode:str,
                          device:torch.device) -> torch.Tensor:
    
    ## Now we need to know if the difference vectors need to be counted positively or negatively, depending on their sign along the direction of the track
    ## Since we don't know yet the direction of the track (that's the goal of this algorithm), we need an alternative way to get their sign
    ## The following options are available:
    
    if mode=='order':
        ## We use the order of the hits: hits before are counted negatively, hits after positively
        ## Nodes_c is supposed to be ordered by their position along the track
        dir_order_matrix=torch.ones((node_c_absolute.shape[0], node_c_absolute.shape[0])).to(device)
        dir_order_matrix=torch.triu(dir_order_matrix)-torch.tril(dir_order_matrix)
    
    elif mode=='node_d':
        ## We use the predicted direction 'node_d' as an approximation: differences with a positive scalar product with node_d are counted positively, differences with a negative scalar product negatively
        if node_d is not None:
            dir_order_matrix=torch.sign(torch.sum(node_d[None,:,:]*matrix_of_differences,dim=-1))
        else:
            raise ValueError("node_d is None while mode is 'node_d'")
        
    elif mode[:3]=='PCA':
        ## We compute a local PCA to get the local axis of most elongation, then we sign it using a submode order or node_d, and then we use this approximate direction in the same way as 'node_d'
        M=matrix_of_differences*weights_for_differences[:,:,None] # shape (N,N,3)
        M_mean=M.mean(dim=1,keepdim=True)
        # Compute covariance using einsum
        covariance_M = torch.einsum('ijk,ijl->ikl', M - M_mean, M - M_mean) / (M.shape[0] - 1) # shape (N,3,3), covariance_M[i]=Cov(M[i])
        
        try:
            # Compute eigenvalues and eigenvectors for each covariance matrix
            eigenval, eigenvect = torch.linalg.eigh(covariance_M)
            # Find the indices of the eigenvectors with the largest eigenvalues
            max_eigenvector_indices = torch.argmax(eigenval, dim=1)
            # Use fancy indexing to get the corresponding eigenvectors
            approx_dir = eigenvect[torch.arange(M.shape[0]).to(device), max_eigenvector_indices]
            # Compute the scalar product of the difference vectors on the approximate direction to get the sign
            dir_order_matrix=torch.sign(torch.sum(approx_dir[None,:,:]*matrix_of_differences,dim=-1))
            PCA_failed=False
            
        except np.linalg.LinAlgError:
            print('PCA failed, defaulting to order mode')
            # If the eigenvectors do not converge, default to order mode:
            PCA_failed=True
            dir_order_matrix=torch.ones((node_c_absolute.shape[0], node_c_absolute.shape[0])).to(device)
            dir_order_matrix=torch.triu(dir_order_matrix)-torch.tril(dir_order_matrix)
        
        if mode=='PCAo' or mode=='PCA':
            if not PCA_failed:
                ## Here we use the 'order' mode to get the correct sign of the approximate direction provided by the eigenvector.
                ## To do this we check that, on average, hits after are counted positively and before negatively
                hit_order=torch.ones((M.shape[0], M.shape[0])).to(device)
                hit_order=(torch.triu(hit_order)-torch.tril(hit_order))
                hit_order*=dir_order_matrix*weights_for_differences
                dir_order_matrix*=torch.sign(torch.sum(hit_order,dim=1,keepdim=True))
            
        elif mode=='PCAnd':
            if not PCA_failed:
                if node_d is not None:
                    ## Here we use the 'node_d' mode to get the correct sign of the approximate direction provided by the eigenvector.
                    ## To do this we check that the scalar product between the preidcted node_d and the approximate direction is positive.
                    eigenvect_sign=torch.sum(node_d*approx_dir,dim=-1)
                    dir_order_matrix*=torch.sign(eigenvect_sign)
                else:
                    raise ValueError("node_d is None while mode is 'PCAnd'")
                
        elif mode=='PCAl':
            if not PCA_failed:
                ## Here we make sure that the eigenvectors are locally along the same axis.
                ## To do that we check that the scalar product between approximate directions is positive locally
                eigenvect_sign=torch.sum(approx_dir[None,:,:]*approx_dir[:,None,:]*weights_for_differences[:,:,None],dim=(1,2))
                dir_order_matrix*=torch.sign(eigenvect_sign)
        
        else:
            raise ValueError(f"mode {mode} not recognized")
    else:
        raise ValueError(f"mode {mode} not recognized")
    
    return dir_order_matrix




def construct_direction_and_curvature_gpu(node_c_absolute:torch.Tensor,
                                     node_d:torch.Tensor|None=None,
                                     event_id:torch.Tensor|None=None,
                                     traj_ID:torch.Tensor|None=None,
                                     weight_predicted_node_d:float=0.,
                                     dirScale:float=30.,
                                     curvScale:float=60.,
                                     curvCutoff:float=10.,
                                     scale_factor_node_d:float=1.,
                                     mode:str='order',
                                     device=torch.device,
                                    **kwargs
                                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    
    
    
    if event_id is None:
        event_id = torch.zeros(node_c_absolute.shape[0]).to(device)
    
    if traj_ID is None:
        traj_ID = torch.zeros(node_c_absolute.shape[0]).to(device)
        
    if node_d is not None:
        node_d_norm=torch.linalg.norm(node_d,dim=-1)
        node_d/=node_d_norm[...,None]
        
    
    same_event_traj_matrix=(event_id[None,:]==event_id[:,None]).to(device)
    same_event_traj_matrix*=(traj_ID[None,:]==traj_ID[:,None]).to(device)
    
    if (torch.sum(same_event_traj_matrix,dim=1)<2).any():
        ## We have at least one hit alone
        if len(same_event_traj_matrix)==1:
            ## We have only one hit in the event
            if node_d is not None:
                return torch.Tensor([0.]).to(device), DEFAULT_MOM_NORM*node_d, torch.Tensor([[0.,0.,0.]]).to(device) # defaulting to 0. charge, node_d momentum of norm default, and 0. curvature
            else:
                return torch.Tensor([0.]).to(device), DEFAULT_MOM_NORM*torch.Tensor([[0.,0.,1.]]), torch.Tensor([[0.,0.,0.]]).to(device) # defaulting to 0. charge, momentum of norm default along Z, and 0. curvature
        else:
            same_event_traj_matrix[(torch.sum(same_event_traj_matrix,dim=1)<2)[:,0]]=1. # use all hits available if the event
    
    matrix_of_differences=torch.zeros((node_c_absolute.shape[0], node_c_absolute.shape[0], 3)).to(device)
    matrix_of_differences[:,:,:]=node_c_absolute[None,:,:]*same_event_traj_matrix-node_c_absolute[:,None,:]*same_event_traj_matrix
    
    matrix_of_distances=torch.linalg.norm(matrix_of_differences,dim=-1)
    
    weights_differences_for_directions=(matrix_of_distances/dirScale)*torch.exp(-(matrix_of_distances/dirScale))
    weights_differences_for_curvature=(matrix_of_distances/curvScale)*torch.exp(-(matrix_of_distances/curvScale))*(1-torch.exp(-5.*(matrix_of_distances**2/curvCutoff**2)))
  
    ## Now we need to know if the difference vectors need to be counted positively or negatively, depending on their sign along the direction of the track
    ## Since we don't know yet the direction of the track (that's the goal of this algorithm), we need an alternative way to get their sign
    
    dir_order_matrix=choose_direction_sign_gpu(node_c_absolute,matrix_of_differences,weights_differences_for_directions,node_d,mode,device)
    
    
    dir_from_differences=torch.sum(weights_differences_for_directions[:,:,None]*matrix_of_differences*dir_order_matrix[:,:,None]/(matrix_of_distances[:,:,None]+1e-9),dim=1)/(torch.sum(weights_differences_for_directions,dim=1)[:,None]+1e-9)
    
    curv_from_differences=torch.sum(weights_differences_for_curvature[:,:,None]*matrix_of_differences/(matrix_of_distances[:,:,None]**2+1e-9),dim=1)/(torch.sum(weights_differences_for_curvature,dim=1)[:,None]+1e-9)
    
    if node_d is not None:
        ## If node_d is available, we use it by averaging it with the reconstructed direction.
        ## We compute the average locally, using the same kind of weighting, but with a scale that can be different
        
        weights_node_d_for_directions=(matrix_of_distances/(dirScale*scale_factor_node_d))*torch.exp(-(matrix_of_distances/(dirScale*scale_factor_node_d)))
        weights_node_d_for_curvature=(matrix_of_distances/(curvScale*scale_factor_node_d))*torch.exp(-(matrix_of_distances/(curvScale*scale_factor_node_d)))*(1-torch.exp(-5.*(matrix_of_distances**2/curvCutoff**2)))
        
        dir_from_node_d=torch.sum(weights_node_d_for_directions[:,:,None]*node_d[None,:,:],dim=1)/(torch.sum(weights_node_d_for_directions,dim=1)[:,None]+1e-9)
        curv_from_node_d=torch.sum(weights_node_d_for_curvature[:,:,None]*node_d[None,:,:]*dir_order_matrix[:,:,None]/(matrix_of_distances[:,:,None]+1e-9),dim=1)/(torch.sum(weights_node_d_for_curvature,dim=1)[:,None]+1e-9)
        direction_=(dir_from_differences+weight_predicted_node_d*dir_from_node_d)/(1.+weight_predicted_node_d)
        curvature=(curv_from_differences+weight_predicted_node_d*curv_from_node_d)/(1.+weight_predicted_node_d)
    else:
        direction_=dir_from_differences
        curvature=curv_from_differences
        
    direction=direction_/(torch.linalg.norm(direction_,dim=-1,keepdim=True)+1e-9)
        
    # assert np.allclose(np.linalg.norm(direction,axis=-1),1.), f"Direction must be a unit vector: {(~np.isclose(np.linalg.norm(direction,axis=-1),1.)).mean()}, {np.linalg.norm(direction,axis=-1).mean()}"
    
    curvature = curvature - torch.sum(curvature*direction,dim=-1,keepdim=True)*direction
        
    charge=get_charge_gpu(curvature,direction)
    mom_n=get_momentum_magnitude_gpu(torch.linalg.norm(curvature,dim=-1),direction)
    
    if node_d is not None:
        ## If the node_d is not a unit vector, but contains information about the momentum norm prediction, we use it
        ## To do that, we check that the node_d norm is above 1 MeV for 99% of the cases, if not, it means that node_d are unit vectors
        if (1.*(node_d_norm>1)).mean()>0.99:
            mom_n+=weight_predicted_node_d*node_d_norm
            mom_n/=(1.+weight_predicted_node_d)
    
    return charge, mom_n[:,None]*direction, curvature



def charge_and_momentum_gpu(ret_dict:dict,
                        values:str='pred',
                        show_progressbar:bool=True,
                        device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        **kwargs):
    
    Charge=[]
    Momentum=[]
    Curvature=[]
    
    
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
    
    n=8
    N=np.max(ret_dict['event_id'])
    i=0
    
    
    for k in tqdm.tqdm(range(N//n+2),disable=(not show_progressbar)):
        
        j=np.searchsorted(ret_dict['event_id'][:,0],k*n,side='right')
        if i==j:
            continue
        ind=torch.arange(i,j)
        i=j
        
        charge,mom,curv=construct_direction_and_curvature_gpu(node_c_absolute=node_c_absolute[ind],
                                                       node_d=node_d[ind] if node_d is not None else None,
                                                       event_id=event_id[ind],
                                                       traj_ID=traj_id[ind],
                                                       device=device,
                                                       **kwargs)
        assert len(node_c_absolute[ind]) == len(charge)
        assert len(node_c_absolute[ind]) == len(mom)
        
        
        Charge.append(charge.cpu().detach().numpy())
        Momentum.append(mom.cpu().detach().numpy())
        Curvature.append(curv.cpu().detach().numpy())
        
        
    charge=np.concatenate(Charge,axis=0)
    mom=np.concatenate(Momentum,axis=0)
    curv=np.concatenate(Curvature,axis=0)
    
    
    return charge, mom, curv



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