import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
import numpy as np


#################### LOSSES #####################    


class SumLoss(nn.Module):
    """
    Defines the sum of several losses. Will return both the total loss and the detail of each individual loss term (loss composition).
    """
    def __init__(self, losses:list[nn.Module], weights:list[float]|None=None):
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
    """
    Loss for the Momentum as a L2 norm: 3D distance between the predicted an true momentum vectors.
    """
    
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
    """
    Loss for the direction of the momentum and its norm, separated.
    The direction loss is the cosine loss (scalar product between true and predicted directions)
    """
    
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
    """
    Loss for the direction of the momentum and its norm, separated.
    The direction loss is cosine of the theta and phi angles
    """
    
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
    def __init__(self, losses:list[nn.Module], weights:list[float]|None=None):
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