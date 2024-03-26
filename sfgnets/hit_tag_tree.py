import numpy as np
import sklearn as sk
from sklearn import tree

from sfgnets.dataset import EventDataset

## Dataset for the decision tree 
class HitTagTreeDataset(EventDataset):
    
    
    @property
    def data(self):
        x,y,c,aux=[],[],[],[]
        for k in range(len(self)):
            dat=self[k]
            if dat['x'] is not None and dat['c'] is not None:
                x.append(dat['x'])
                y.append(dat['y'][:,None])
                c.append(dat['c'])
                aux.append(dat['aux'])
                
        x=np.vstack(x)
        y=np.vstack(y)
        c=np.vstack(c)
        aux=np.vstack(aux) if self.aux else None
        
        return {'x':x, 'y':y, 'c':c, 'aux':aux}
    
    @property
    def XY(self):
        dat=self.data
        return (dat['x'], dat['y'])

    
    def getx(self,data):
        # Extract raw data
        x_0 = data['x']  # HitTime, HitCharge
        
        if x_0.shape[0]==0: # Checking if the event is empty
            return None
        
        c = data['c']  # 3D coordinates (cube raw positions)
        
        # Checking if the recon_ver is defined or not 
        if not hasattr(self, 'recon_ver'):
            print("recon_ver not defined, defaulting to False")
            self.recon_ver=False
        
        # True vertex position
        if self.recon_ver:
            verPos = data['recon_verPos'] # use the reconstructed vertex position
        else:
            verPos = data['verPos'] # use the true vertex position
        
        x=np.zeros(shape=(x_0.shape[0], 2))
        x[:,0]=x_0[:,1] # have to remove 'HitTime', just keep the charge
        # Add as features the distance to the vertex position
        
        # Checking if the recon_ver is defined or not 
        if not hasattr(self, 'L2Norm'):
            print("L2Norm not defined, defaulting to True")
            self.L2Norm=True
            
        if self.L2Norm:
            x[:,1]=np.linalg.norm(c-verPos[None,:], axis=1) # distance to the vertex
        else:
            x[:,1]=np.max(np.abs(c-verPos[None,:]), axis=1)/10.27 # distance in cubes
        
        return x # hit charge and distance to vertex
    
    
    def getc(self,data:np.lib.npyio.NpzFile):
        c = data['c']  # 3D coordinates (cube raw positions)
        
        if c.shape[0]==0:
            return None
        
        return c
    
    def gety(self,data:np.lib.npyio.NpzFile):
        return data['y'] - 1
    
    def getaux(self,data:np.lib.npyio.NpzFile):
        # Checking if the aux is defined or not 
        if not hasattr(self, 'aux'):
            print("aux not defined, defaulting to False")
            self.aux=False
        
        if self.aux:
            pdg=data['pdg']
            reaction_code=int(data['reaction_code'])*np.ones_like(pdg)
            aux=np.zeros((len(pdg),2))
            aux[:,0]=pdg
            aux[:,1]=reaction_code
            return aux
        else:
            return None
        
weight=None
# weight='balanced'
        
DecisionTreeClassifier=tree.DecisionTreeClassifier(max_depth=3,
                                                      min_samples_leaf=10,
                                                      class_weight=weight)




