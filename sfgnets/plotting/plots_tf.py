
################## TRACK FITTING PLOTS ####################

import os
import sys
import pickle as pk
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sfgnets.dataset import PGunEvent, RANGES, CUBE_SIZE
import corner
import matplotlib
import particle
from sfgnets.plotting.plot_3D_events import plotly_event_nodes as _plotly_event_nodes
from sfgnets.plotting.plot_3D_events import plotly_event_general as _plotly_event_general

# plt.rcParams['text.usetex'] = True

target_names = ["Vertex activity", "Single P", "Noise"]
use_log_scale=False
x_lim_dist=(0.,40.) if use_log_scale else (0.5,40.)
x_lim_dist_m=(0.,3500.) if use_log_scale else (0.,3500.)
# x_lim_dist=(0.5,80) if use_log_scale else (0.5,80)
x_lim_charge=(0.5,2000) if use_log_scale else (0.5, 300)
delfault_dpi=200
Nbins=1000
y_log_scale=True


reaction_codes=[0, 1, 11, 12, 13, 16, 17, 21, 22, 23, 26, 31, 32, 33, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 51, 52]
reaction_codes_short_labels=['tot', 'ccqe', 'ccppip', 'ccppi0', 'ccnpip', 'cccoh', 'ccgam', 'ccmpi', 'cceta', 'cck', 'ccdis', 'ncnpi0', 'ncppi0', 'ncppim', 'ncnpip', 'nccoh', 'ncngam', 'ncpgam', 'ncmpi', 'ncneta', 'ncpeta', 'nck0', 'nckp', 'ncdis', 'ncqep', 'ncqen']
reaction_codes_long_labels=['Total', 'CCQE: $\\nu_{l} n \\rightarrow l^{-} p$', 'CC 1$\\pi: \\nu_{l} p \\rightarrow l^{-} p \\pi^{+}$', 'CC 1$\\pi: \\nu_{l} n \\rightarrow l^{-} p \\pi^{0}$', 'CC 1$\\pi: \\nu_{l} n \\rightarrow l^{-} n \\pi^{+}$', 'CC coherent-$\\pi: \\nu_{l} ^{16}O \\rightarrow l^{-} ^{16}O \\pi^{+}$', '1$\\gamma from \\Delta: \\nu_{l} n \\rightarrow l^{-} p \\gamma$', "CC (1.3 < W < 2 GeV): $\\nu_{l} N \\rightarrow l^{-} N' multi-\\pi", 'CC 1$\\eta: \\nu_{l} n \\rightarrow l^{-} p \\eta$', 'CC 1K: $\\nu_{l} n \\rightarrow l^{-} \\Lambda K^{+}$', "CC DIS (2 GeV $\leq W$): $\\nu_{l} N \\rightarrow l^{-} N' mesons$", 'NC 1$\\pi: \\nu_{l} n \\rightarrow \\nu_{l} n \\pi^{0}$', 'NC 1$\\pi: \\nu_{l} p \\rightarrow \\nu_{l} p \\pi^{0}$', 'NC 1$\\pi: \\nu_{l} n \\rightarrow \\nu_{l} p \\pi^{-}$', 'NC 1$\\pi: \\nu_{l} p \\rightarrow \\nu_{l} n \\pi^{+}$', 'NC coherent-\\pi: $\\nu_{l} ^{16}O \\rightarrow \\nu_{l} ^{16}O \\pi^{0}$', '1\\gamma from \\Delta: $\\nu_{l} n \\rightarrow \\nu_{l} n \\gamma$', '1\\gamma from \\Delta: $\\nu_{l} p \\rightarrow \\nu_{l} p \\gamma$', 'NC (1.3 < W < 2 GeV): $\\nu_{l} N \\rightarrow \\nu_{l} N multi-\\pi$', 'NC 1\\eta: $\\nu_{l} n \\rightarrow \\nu_{l} n \\eta$', 'NC 1\\eta: $\\nu_{l} p \\rightarrow \\nu_{l} p \\eta$', 'NC 1K: $\\nu_{l} n \\rightarrow \\nu_{l} \\Lambda K^{0}$', 'NC 1K: $\\nu_{l} n \\rightarrow \\nu_{l} \\Lambda K^{+}$', "NC DIS (2 GeV < W): $\\nu_{l} N \\rightarrow \\nu_{l} N' mesons", 'NC elastic: $\\nu_{l} n \\rightarrow \\nu_{l} n','NC elastic: $\\nu_{l} p \\rightarrow \\nu_{l} p']
reaction_codes_short_labels_dict=dict(zip(reaction_codes,reaction_codes_short_labels))
reaction_codes_long_labels_dict=dict(zip(reaction_codes,reaction_codes_long_labels))

list_of_particles=['e+','e-','gamma','mu+','mu-','n','p','pi+','pi-']
list_of_pdgs=[-11,11,22,-13,13,2112,2212,211,-211]
particle_dict=dict(zip(list_of_pdgs,list_of_particles))


color_list = ["#3f7dae","#ae703f",'#117238', '#4e6206', '#6e4d00','#823312','#851433']



class DataContainer:
    def __init__(self, all_results:dict[str,np.ndarray], dataset:PGunEvent):
        self.all_results=all_results
        self.dataset=dataset
        
        self.aux=all_results['aux']
        
        # ## Distance between the truth and predicted nodes
        # self.euclidian_distance=np.linalg.norm(all_results['y'][all_results["mask"][...,0]>0.][...,:3]-all_results['predictions'][all_results["mask"][...,0]>0.][...,:3],axis=1)
        
        difference_vector=(all_results['y'][all_results["mask"][...,0]>0.][...,:3]-all_results['predictions'][all_results["mask"][...,0]>0.][...,:3])
        true_direction=self.aux[all_results["mask"][...,0]>0.][:,11:14]
        self.euclidian_distance=np.linalg.norm(difference_vector-np.sum(difference_vector*true_direction,axis=-1,keepdims=True)*true_direction,axis=-1)
        
        
        self.input_particle=all_results['aux'][:,0]
        self.NTraj = all_results['aux'][:,1]
        self.traj_parentID = all_results['aux'][:,2]
        self.distance_node_point = all_results['aux'][:,3]
        self.momentum = all_results['aux'][:,4:7]
        self.tag = all_results['aux'][:,7]
        self.number_of_particles = all_results['aux'][:,8]
        self.energy_deposited = all_results['aux'][:,9]
        self.particle_pdg = all_results['aux'][:,10]
        self.direction = all_results['aux'][:,11:14]
        self.traj_length = all_results['aux'][:,15]
        self.event_entry = all_results['aux'][:,16]
        
        self.mask=(all_results["mask"][...,0]>0.)
        self.f=all_results['f']
        self.y=all_results['y']
        self.c=all_results['c']
        self.predictions=all_results['predictions']
        self.event_id=all_results['event_id'][...,0]
        
        
        self.recon_c=all_results['aux'][:,17:20]-self.c
        self.recon_d=all_results['aux'][:,20:23]
        
        difference_vector=(all_results['y'][all_results["mask"][...,0]>0.][...,:3]-self.recon_c[all_results["mask"][...,0]>0.][...,:3])
        self.recon_euclidian_distance=np.linalg.norm(difference_vector-np.sum(difference_vector*true_direction,axis=-1,keepdims=True)*true_direction,axis=-1)
        
        if self.y.shape[-1]>5:
            self.momentum_is_available=True
            pred_momentum=all_results['predictions'][all_results["mask"][...,0]>0.][...,3:6]
            true_momentum=all_results['y'][all_results["mask"][...,0]>0.][...,3:6]
            pred_norm=np.linalg.norm(pred_momentum,axis=-1)
            true_norm=np.linalg.norm(true_momentum,axis=-1)
            self.m_euclidian_distance=np.linalg.norm(pred_momentum-true_momentum,axis=1)
            self.md_distance=np.arccos(np.sum(pred_momentum*true_momentum,axis=-1)/(pred_norm*true_norm+1e-3))
            self.md_distance[(self.md_distance<0)+(self.md_distance>np.pi)]=np.pi
            self.md_distance[np.isnan(self.md_distance)]=np.pi
            self.mn_distance=np.abs(true_norm-pred_norm)
            self.recon_mn_distance=np.empty_like(self.md_distance)
            self.recon_m_euclidian_distance=np.empty_like(self.m_euclidian_distance)
            self.recon_md_distance=np.arccos(np.sum(self.recon_d[all_results["mask"][...,0]>0.]*true_momentum,axis=-1)/(true_norm+1e-6))
        else:
            self.momentum_is_available=False
        
    def __len__(self):
        return self.aux.shape[0]
        
        

def plot_event_display(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                       i:int=0,
                        show:bool=True,
                        savefig_path:str=None,
                        pred_pdg_index:int=None,
                        show_target_momentum:bool=True,
                        show_pred_momentum:bool=False,
                        compare_to_bayesian_filter:bool=False,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        

    indexes=data.mask*(data.event_id==i)
    track=(data.y[...,:3]+data.c[...,:3])[indexes]
    
    fig=_plotly_event_nodes(track=track,
                            general_label="True track",
                            pdg=data.particle_pdg[indexes],
                            momentum=data.momentum[indexes] if show_target_momentum else None,
                            **kwargs)
    
    if compare_to_bayesian_filter:
        average_mom_norm=np.linalg.norm(data.momentum[indexes],axis=-1).mean()
        fig=_plotly_event_nodes(track=(data.recon_c+data.c)[indexes],
                                        general_label="Bayesian filter",
                                        momentum=data.recon_d[indexes]*average_mom_norm if show_pred_momentum else None,
                                        color="#117da3",
                                        fig=fig,
                                        **kwargs)
    
    fig=_plotly_event_nodes(track=(data.predictions[...,:3]+data.c)[indexes],
                                        general_label="Pred track",
                                        pdg=data.predictions[...,pred_pdg_index][indexes] if pred_pdg_index is not None else None,
                                        momentum=data.predictions[indexes][...,3:6] if show_pred_momentum else None,
                                        color="#a33711",
                                        fig=fig,
                                        **kwargs)

    fig=_plotly_event_nodes(track=(data.c)[indexes],
                                        general_label="Center of cubes",
                                        color="#541af8",
                                        fig=fig,
                                        **kwargs)

    fig=_plotly_event_general(pos3d=data.c[indexes],
                              energies=np.clip(data.f[indexes][:,0],0,500),
                              max_energy=500,
                              fig=fig,
                                **kwargs)
    if show:
        fig.show()
    
    try:
        print(f"Input particle: {particle_dict[int(data.input_particle[indexes].mean())]}")
    except KeyError:
        print(f"Particle not recognized. Input particle(s) are :{np.unique(data.input_particle[indexes].astype(int))}")
    
    return fig
    

def plot_pred_X(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                mom:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,ax=plt.subplots(figsize=(16,8),facecolor='white')
    ranges=(-15,15) if not mom else (-3500,3500)
    if not mom:
        ax.hist(data.predictions[data.mask,0],bins=500,label='predictions',alpha=0.6,range=ranges)
        ax.hist(data.y[data.mask,0],bins=500,label='truth',alpha=0.6,range=ranges)
        ax.set_xlabel("Relative node position in the X direction")
        ax.set_title("Distribution of predicted and true relative X position of nodes")
    else:
        ax.hist(data.predictions[data.mask,4],bins=500,label='predictions',alpha=0.6,range=ranges)
        ax.hist(data.y[data.mask,4],bins=500,label='truth',alpha=0.6,range=ranges)
        ax.set_xlabel("Momentum in the X direction")
        ax.set_title("Distribution of predicted and true momenta in the X direction of nodes")
    ax.set_xlim(*ranges)
    ax.set_xticks(np.linspace(ranges[0],ranges[1],ranges[1]-ranges[0]+1), minor=True)
    ax.set_ylabel("Number of hits")
    ax.legend()
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig


def _plot_euclidian_distance_dist(euclidian_distance:np.ndarray,
                             ax,
                             mom:bool=False,
                             y_log_scale=y_log_scale,
                             x_range:tuple[float,float]=None,
                             x_unit:str=None,
                             x_label:str=None,
                             ax_title:str=None,
                             ax_args:dict={},
                             show_summary_stats:bool=True,
                             **kwargs):
    if x_range is None:
        x_range=(0.,20.) if not mom else (0.,3000)
    if x_unit is None:
        x_unit="mm" if not mom else "MeV"
    if x_label is None:
        x_label="Euclidian distance (mm)" if not mom else "Euclidian distance (MeV)"
    if ax_title is None:
        ax_title="Distribution of the euclidian distance between predicted and true nodes" if not mom else "Distribution of the euclidian distance between predicted and true momenta"
    
    if len(euclidian_distance)>0:
    
        ret=ax.hist(euclidian_distance,bins=500,range=x_range,**ax_args)
        ax.set_ylabel("Number of hits")
        ax.set_yscale("log" if y_log_scale else 'linear')
        if show_summary_stats:
            ax.text(ret[1].max()*0.83,ret[0].max()*0.8, f"Mean:    {np.mean(euclidian_distance):.3f} {x_unit}\nMedian: {np.median(euclidian_distance):.3f} {x_unit}", multialignment='left', bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title(ax_title)
        ax.set_xlabel(x_label)
    
    
def _plot_euclidian_distance_cdf(euclidian_distance:np.ndarray,
                             ax2,
                             mom:bool=False,
                             x_range:tuple[float,float]=None,
                             x_unit:str=None,
                             x_label:str=None,
                             ax2_title:str=None,
                             ax2_args:dict=dict(label="CDF",alpha=1.,color="#bf7e02"),
                             show_summary_stats:bool=True,
                             **kwargs):    
    if x_range is None:
        x_range=(0.,20.) if not mom else (0.,3000)
    if x_unit is None:
        x_unit="mm" if not mom else "MeV"
    if x_label is None:
        x_label="Euclidian distance (mm)" if not mom else "Euclidian distance (MeV)"
    if ax2_title is None:
        ax2_title="Cumulative distribution function of the euclidian distance between predicted and true nodes" if not mom else "Cumulative distribution function of the euclidian distance between predicted and true momenta"
    
    if len(euclidian_distance)>0:
    
        # ax2=ax.twinx()
        ax2.plot(np.sort(euclidian_distance),np.arange(len(euclidian_distance))/len(euclidian_distance),**ax2_args)
        ax2.set_ylabel("Cumulative distribution")
        ax2.set_ylim(0,1)
        ax2.set_xlim(*x_range)
        ax2.set_yticks(np.linspace(0.,1.,11))
        ax2.set_yticks(np.linspace(0.,1.,21), minor=True)
        ax2.grid(visible=True)
        ax2.set_xticks(np.linspace(x_range[0],x_range[1],11))
        ax2.set_xticks(np.linspace(x_range[0],x_range[1],21), minor=True)
        ax2.set_xlabel("Euclidian distance (mm)")
        if show_summary_stats:
            ax2.text(0.808*(x_range[1]-x_range[0]),0.115, f"68%:    {np.quantile(euclidian_distance,q=0.68):.2f} {x_unit}\n95%:    {np.quantile(euclidian_distance,q=0.95):.2f} {x_unit}\n99%:    {np.quantile(euclidian_distance,q=0.99):.2f} {x_unit}", multialignment='left', bbox=dict(facecolor='white', alpha=0.5))
        ax2.set_title(ax2_title)
        
    

def plot_euclidian_distance(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                mom:bool=False,
                compare_to_bayesian_filter:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,(ax,ax2)=plt.subplots(ncols=2,nrows=1,figsize=(24,8),facecolor='white')
    
    if not mom:
        eucl=data.euclidian_distance
        recon_eucl=data.recon_euclidian_distance
    else:
        eucl=data.m_euclidian_distance
        recon_eucl=data.recon_m_euclidian_distance
    
    if not compare_to_bayesian_filter:
        _plot_euclidian_distance_dist(eucl,ax,mom=mom,**kwargs)
        _plot_euclidian_distance_cdf(eucl,ax2,mom=mom,**kwargs)
    else:
        _plot_euclidian_distance_dist(eucl,ax,mom=mom,ax_args=dict(label="Model",histtype='step'),show_summary_stats=False,**kwargs)
        _plot_euclidian_distance_dist(recon_eucl,ax,mom=mom,ax_args=dict(label="Bayesian filter",histtype='step'),show_summary_stats=False,**kwargs)
        _plot_euclidian_distance_cdf(eucl,ax2,mom=mom,ax2_args=dict(label="Model"),show_summary_stats=False,**kwargs)
        _plot_euclidian_distance_cdf(recon_eucl,ax2,mom=mom,ax2_args=dict(label="Bayesian filter"),show_summary_stats=False,**kwargs)
        ax.legend(loc="upper right")
        ax2.legend(loc="upper right")
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig



def plot_euclidian_distance_by_input_particle(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                mom:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=len(list_of_particles),figsize=(24,8*len(list_of_particles)),facecolor='white')
    
    if not mom:
        eucl=data.euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the input particle",y=0.9)
    else:
        eucl=data.m_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the momentum predicted and the true momentum depending on the input particle",y=0.9)
    
    for k, part in enumerate(list_of_particles):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(eucl)
        euclidian_distance=eucl[data.input_particle==list_of_pdgs[k]]
        _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,**kwargs)
        ax.annotate(f"{part}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig
    



def plot_euclidian_distance_by_node_n(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                mom:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=3,figsize=(24,8*3),facecolor='white')
    
    indexes=[(data.number_of_particles==0),(data.number_of_particles==1),(data.number_of_particles>1)]
    labels=["No hit segment", "One hit segment", "Multiple hit segments"]
    
    if not mom:
        eucl=data.euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the number of hit segments")
    else:
        eucl=data.m_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the momentum predicted and the true momentum depending on the number of hit segments")
    
    for k, ind in enumerate(indexes):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(eucl)
        euclidian_distance=eucl[ind]
        _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,**kwargs)
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    
    
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig





def plot_euclidian_distance_by_tag(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                mom:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=2,figsize=(24,8*2),facecolor='white')
    
    indexes=[(data.tag==2),(data.tag==3)]
    labels=["Track hit", "Noise", ]
    
    
    if not mom:
        eucl=data.euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the hit tag")
    else:
        eucl=data.m_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the momentum predicted and the true momentum depending on the hit tag")
    
    
    for k, ind in enumerate(indexes):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(eucl)
        euclidian_distance=eucl[ind]
        _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,**kwargs)
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig





def plot_euclidian_distance_by_primary(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                mom:bool=False,
                compare_to_bayesian_filter:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=2,figsize=(24,8*2),facecolor='white')
    
    indexes=[(data.traj_parentID==0),(data.traj_parentID!=0)]
    labels=["Primary trajectory", "Secondary trajectory", ]
    
    if not mom:
        eucl=data.euclidian_distance
        recon_eucl=data.recon_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node for primary and secondary trajectories")
    else:
        eucl=data.m_euclidian_distance
        recon_eucl=data.recon_m_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the momentum predicted and the true momentum for primary and secondary trajectories")
    
    for k, ind in enumerate(indexes):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(eucl)
        euclidian_distance=eucl[ind]
        recon_euclidian_distance=recon_eucl[ind]
        
        if not compare_to_bayesian_filter:
            _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,**kwargs)
            _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,**kwargs)
        else:
            _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,ax_args=dict(label="Model",alpha=0.5),show_summary_stats=True,**kwargs)
            _plot_euclidian_distance_dist(recon_euclidian_distance,ax,mom=mom,ax_args=dict(label="Bayesian filter",alpha=0.5),show_summary_stats=False,**kwargs)
            _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,ax2_args=dict(label="Model"),show_summary_stats=True,**kwargs)
            _plot_euclidian_distance_cdf(recon_euclidian_distance,ax2,mom=mom,ax2_args=dict(label="Bayesian filter"),show_summary_stats=False,**kwargs)
            ax.legend()
            ax2.legend()
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig



def plot_euclidian_distance_by_pdg(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                particles_to_consider_pdg:list[int]=[-11,11,22, -13, 13, 2112, 2212, 211, -211],
                include_pdg_0:bool=True,
                mom:bool=False,
                compare_to_bayesian_filter:bool=False,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    
    labels=[particle.Particle.from_pdgid(part).name for part in particles_to_consider_pdg]+['other']
    if include_pdg_0:
        labels=["pdg 0"]+labels
        particles_to_consider_pdg=[0]+particles_to_consider_pdg

    fig,axes=plt.subplots(ncols=2,nrows=len(particles_to_consider_pdg)+1,figsize=(24,8*(len(particles_to_consider_pdg)+1)),facecolor='white')
    
    
    if not mom:
        eucl=data.euclidian_distance
        recon_eucl=data.recon_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the hit particle",y=0.9)
    else:
        eucl=data.m_euclidian_distance
        recon_eucl=data.recon_m_euclidian_distance
        fig.suptitle("Distribution of the euclidian distance between the momentum predicted and the true momentum depending on the hit particle",y=0.9)
    
    for k, part in enumerate(particles_to_consider_pdg):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(eucl)
        euclidian_distance=eucl[data.particle_pdg==part]
        recon_euclidian_distance=recon_eucl[data.particle_pdg==part]
        
        if not compare_to_bayesian_filter:
            _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,**kwargs)
            _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,**kwargs)
        else:
            _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,ax_args=dict(label="Model",alpha=0.5),show_summary_stats=True,**kwargs)
            _plot_euclidian_distance_dist(recon_euclidian_distance,ax,mom=mom,ax_args=dict(label="Bayesian filter",alpha=0.5),show_summary_stats=False,**kwargs)
            _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,ax2_args=dict(label="Model"),show_summary_stats=True,**kwargs)
            _plot_euclidian_distance_cdf(recon_euclidian_distance,ax2,mom=mom,ax2_args=dict(label="Bayesian filter"),show_summary_stats=False,**kwargs)
            ax.legend()
            ax2.legend()
            
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        
    ax=axes[-1][0]
    ax2=axes[-1][1]
    total_size=len(eucl)
    euclidian_distance=eucl[np.isin(data.particle_pdg,particles_to_consider_pdg,invert=True)]
    _plot_euclidian_distance_dist(euclidian_distance,ax,mom=mom,**kwargs)
    _plot_euclidian_distance_cdf(euclidian_distance,ax2,mom=mom,**kwargs)
    ax.annotate(f"{labels[-1]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig


def _get_perf(data:DataContainer,
              indexes:list[np.ndarray],
              show_progress_bar:bool=True,
              mom:bool=False,
              **kwargs):
    
    perfs=[[],[]]
    for k in tqdm.tqdm(range(len(indexes)),desc="Computing performances",disable=(not show_progress_bar)):
        if indexes[k].any():
            if not mom:
                eucl=data.euclidian_distance[indexes[k]]
            else:
                eucl=data.m_euclidian_distance[indexes[k]]
            perfs[0].append(np.mean(eucl))
            perfs[1].append(np.quantile(eucl,q=0.95))
        else:
            perfs[0].append(np.nan)
            perfs[1].append(np.nan)
    return perfs


def _plot_perf_dist(data:DataContainer,
                    features:np.ndarray,
                    lims:tuple[float,float],
                    feature_name:str,
                    xlabel:str,
                    Nbins:int=100,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    color_list:list[str]=color_list,
                    y_log_scale:bool=y_log_scale,
                    figsize:tuple[float]=(16,12),
                    use_log_scale=use_log_scale,
                    mom:bool=False,
                    x_label:str=None,
                    **kwargs):
    
    if x_label is None:
        ylabel="Euclidian distance (mm)" if not mom else "Euclidian distance (MeV)"
    else:
        ylabel=x_label
    
    bins=np.logspace(np.log10(lims[0]),np.log10(lims[1]),Nbins) if use_log_scale else np.linspace(lims[0],lims[1],Nbins) #log xscale
    
    indexes=[(features>=bins[k])*(features<bins[k+1]) for k in range(len(bins)-1)]
    
    perf=_get_perf(data,indexes,mom=mom,**kwargs)
    
    fig,ax=plt.subplots(figsize=figsize, facecolor="white")
    ax2=ax.twinx()
    ax2.hist(features, bins=bins, alpha=0.3, label="Support")
    
    ax.plot(bins[:-1],perf[0],label="Mean",  color=color_list[3], marker="^")
    ax.plot(bins[:-1],perf[1],label="q=95%", color=color_list[4], marker="+")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log' if use_log_scale else 'linear')
    # ax.set_ylim(0.,1.)
    # ax.set_xlim(lims)

    ax2.set_ylabel('Support')
    if y_log_scale:
        ax2.set_yscale('log')
    

    ax.set_title(f'Euclidian distance between truth and prediction depending on the {feature_name} for model {model_name}')
    ax.legend()
    ax2.legend(loc="lower right")

    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig


def plot_perf_charge(data:DataContainer,
                    x_lim_charge:tuple[float,float]=x_lim_charge,
                    Nbins:int=100,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    mom:bool=False,
                    **kwargs):
    
    return _plot_perf_dist(data=data,
                    features=data.f[:,0],
                    lims=x_lim_charge,
                    feature_name="Hit charge",
                    xlabel="Charge",
                    Nbins=Nbins,
                    model_name=model_name,
                    show=show,
                    savefig_path=savefig_path,
                    mom=mom,
                    **kwargs)
    

def plot_perf_distance(data:DataContainer,
                    x_lim_dist:tuple[float,float]=x_lim_dist,
                    Nbins:int=100,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    mom:bool=False,
                    **kwargs):
    
    return _plot_perf_dist(data=data,
                    features=data.distance_node_point,
                    lims=x_lim_dist,
                    feature_name="cube centre distance to closest projected trajectory point",
                    xlabel="Distance between cube centre and trajectory point (mm)",
                    Nbins=Nbins,
                    model_name=model_name,
                    show=show,
                    savefig_path=savefig_path,
                    mom=mom,
                    **kwargs)
    
    

def plot_perf_traj_length(data:DataContainer,
                    x_lim_dist:tuple[float,float]=(20.,1020.),
                    Nbins:int=100,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    mom:bool=False,
                    **kwargs):
    
    return _plot_perf_dist(data=data,
                    features=data.traj_length,
                    lims=x_lim_dist,
                    feature_name="trajectory length",
                    xlabel="Trajectory length (mm)",
                    Nbins=Nbins,
                    model_name=model_name,
                    show=show,
                    savefig_path=savefig_path,
                    mom=mom,
                    **kwargs)
    

def plot_perf_kin_ener(data:DataContainer,
                    x_lim_mom:tuple[float,float]=(0.,3500.),
                    Nbins:int=100,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    mom:bool=False,
                    **kwargs):
    
    return _plot_perf_dist(data=data,
                    features=np.linalg.norm(data.momentum,axis=-1),
                    lims=x_lim_mom,
                    feature_name="kinetic energy",
                    xlabel="Kinetic energy (MeV)",
                    Nbins=Nbins,
                    model_name=model_name,
                    show=show,
                    savefig_path=savefig_path,
                    mom=mom,
                    **kwargs)
    
    
def plots(data:DataContainer|tuple[dict,PGunEvent],
          plots_chosen:list[str]=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance"],
        show:bool=True,
        savefig_path:str=None,
        model_name:str='',
        mode:str=None,
        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    args_={}
    if data.momentum_is_available and mode=='mom':
        args_['mom']=True
        
    elif data.momentum_is_available and mode=='mom_d':
        args_['mom']=True
        args_['x_unit']="rad"
        args_['x_range']=(0,np.pi)
        args_['x_label']="Angle between true and predicted momentum directions (rad)"
        args_['ax_title']="Distribution of the angle between predicted and true momenta"
        data.m_euclidian_distance=data.md_distance
        data.recon_m_euclidian_distance=data.recon_md_distance
        
    elif data.momentum_is_available and mode=='mom_n':
        args_['mom']=True
        args_['x_unit']="MeV"
        args_['x_range']=(0,3000)
        args_['x_label']="Absolute difference of momentum norm between truth and prediction (MeV)"
        args_['ax_title']="Distribution of the absolute difference of norm between predicted and true momenta"
        data.m_euclidian_distance=data.mn_distance
        data.recon_m_euclidian_distance=data.recon_mn_distance
        
    else:
        mode=None
        
    

    figs=[]
    
    if savefig_path is not None:
        root=".".join(savefig_path.split('.')[:-1])
        suffix=savefig_path.split('.')[-1]
    
    for func in plots_chosen:
        print(f"Plotting {func}{' with mode '+mode if mode is not None else ''}...  ")
        figs.append(globals()["plot_"+func](data=data,
                        show=show,
                        model_name=model_name,
                        savefig_path=root+'_'+func+f'{"_"+mode if mode is not None else ""}'+'.'+suffix if savefig_path is not None else None,
                        **args_,
                        **kwargs))
    
    
    return figs