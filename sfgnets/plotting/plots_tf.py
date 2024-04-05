
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

list_of_particles=['e+','e-','gamma','mu+','mu-','p','pi+','pi-']
list_of_pdgs=[-11,11,22,-13,13,2212,211,-211]
particle_dict=dict(zip(list_of_pdgs,list_of_particles))


color_list = ["#3f7dae","#ae703f",'#117238', '#4e6206', '#6e4d00','#823312','#851433']



class DataContainer:
    def __init__(self, all_results:dict[str,np.ndarray], dataset:PGunEvent):
        self.all_results=all_results
        self.dataset=dataset
        
        self.aux=all_results['aux']
        self.euclidian_distance=np.linalg.norm(all_results['y'][all_results["mask"][...,0]>0.][...,:3]-all_results['predictions'][all_results["mask"][...,0]>0.][...,:3],axis=1)
        
        self.input_particle=all_results['aux'][:,0]
        self.NTraj = all_results['aux'][:,1]
        self.traj_parentID = all_results['aux'][:,2]
        self.distance_node_point = all_results['aux'][:,3]
        self.momentum = all_results['aux'][:,4:7]
        self.tag = all_results['aux'][:,7]
        self.number_of_particles = all_results['aux'][:,8]
        self.energy_deposited = all_results['aux'][:,9]
        self.particle_pdg = all_results['aux'][:,10]
        self.direction = all_results['aux'][:,11]
        
        self.mask=(all_results["mask"][...,0]>0.)
        self.f=all_results['f']
        self.y=all_results['y']
        self.c=all_results['c']
        self.predictions=all_results['predictions']
        self.event_id=all_results['event_id'][...,0]
        
        

def plot_event_display(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                       i:int=0,
                        show:bool=True,
                        savefig_path:str=None,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        

    indexes=data.mask*(data.event_id==i)
    track=(data.y[...,:3]+data.c[...,:3])[indexes]
    fig=_plotly_event_nodes(track=track,
                                        general_label="True track")

    fig=_plotly_event_nodes(track=(data.predictions[...,:3]+data.c)[indexes],
                                        general_label="Pred track",
                                        color="#a33711",
                                        fig=fig)

    fig=_plotly_event_nodes(track=(data.c)[indexes],
                                        general_label="Center of cubes",
                                        color="#541af8",
                                        fig=fig)

    fig=_plotly_event_general(pos3d=data.c[indexes],
                                            fig=fig)
    if show:
        fig.show()
        
    print(f"Input particle: {particle_dict[int(data.input_particle[indexes].mean())]}")
    
    return fig
    

def plot_pred_X(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,ax=plt.subplots(figsize=(16,8),facecolor='white')
    ranges=(-15,15)
    ax.hist(data.predictions[data.mask,0],bins=500,label='predictions',alpha=0.6,range=ranges)
    ax.hist(data.y[data.mask,0],bins=500,label='truth',alpha=0.6,range=ranges)
    ax.set_xlim(*ranges)
    ax.set_xticks(np.linspace(ranges[0],ranges[1],ranges[1]-ranges[0]+1), minor=True)
    ax.set_xlabel("Relative node position in the X direction")
    ax.set_ylabel("Number of hits")
    ax.set_title("Distribution of predicted and true relative X position of nodes")
    ax.legend()
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig


def _plot_euclidian_distance_dist(euclidian_distance:np.ndarray,
                             ax,
                             y_log_scale=y_log_scale,
                             **kwargs):
    
    if len(euclidian_distance)>0:
    
        ret=ax.hist(euclidian_distance,bins=500,range=(0.,20.),)
        ax.set_xlabel("Euclidian distance (mm)")
        ax.set_ylabel("Number of hits")
        ax.set_yscale("log" if y_log_scale else 'linear')
        ax.text(ret[1].max()*0.83,ret[0].max()*0.8, f"Mean:    {np.mean(euclidian_distance):.3f} mm\nMedian: {np.median(euclidian_distance):.3f} mm", multialignment='left', bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title("Distribution of the euclidian distance between predicted and true nodes")
    
    
def _plot_euclidian_distance_cdf(euclidian_distance:np.ndarray,
                             ax2,
                             **kwargs):    
    
    if len(euclidian_distance)>0:
    
        # ax2=ax.twinx()
        ax2.plot(np.sort(euclidian_distance),np.arange(len(euclidian_distance))/len(euclidian_distance),label="CDF",alpha=1.,color="#bf7e02")
        ax2.set_ylabel("Cumulative distribution")
        ax2.set_xlabel("Euclidian distance (mm)")
        ax2.set_ylim(0,1)
        ax2.set_xlim(0,10)
        ax2.set_xticks(np.linspace(0.,10.,11))
        ax2.set_xticks(np.linspace(0.,10.,21), minor=True)
        ax2.set_yticks(np.linspace(0.,1.,11))
        ax2.set_yticks(np.linspace(0.,1.,21), minor=True)
        ax2.grid()
        ax2.text(8.08,0.115, f"90%:    {np.quantile(euclidian_distance,q=0.9):.2f} mm\n95%:    {np.quantile(euclidian_distance,q=0.95):.2f} mm\n99%:    {np.quantile(euclidian_distance,q=0.99):.2f} mm", multialignment='left', bbox=dict(facecolor='white', alpha=0.5))
        ax2.set_title("Cumulative distribution function of the euclidian distance between predicted and true nodes")
    

def plot_euclidian_distance(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,(ax,ax2)=plt.subplots(ncols=2,nrows=1,figsize=(24,8),facecolor='white')
    
    _plot_euclidian_distance_dist(data.euclidian_distance,ax,**kwargs)
    _plot_euclidian_distance_cdf(data.euclidian_distance,ax2,**kwargs)
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig



def plot_euclidian_distance_by_input_particle(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=len(list_of_particles),figsize=(24,8*len(list_of_particles)),facecolor='white')
    
    for k, part in enumerate(list_of_particles):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(data.euclidian_distance)
        euclidian_distance=data.euclidian_distance[data.input_particle==list_of_pdgs[k]]
        _plot_euclidian_distance_dist(euclidian_distance,ax,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,**kwargs)
        ax.annotate(f"{part}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the input particle",y=0.9)
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig
    



def plot_euclidian_distance_by_node_n(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=3,figsize=(24,8*3),facecolor='white')
    
    indexes=[(data.number_of_particles==0),(data.number_of_particles==1),(data.number_of_particles>1)]
    labels=["No hit segment", "One hit segment", "Multiple hit segments"]
    
    for k, ind in enumerate(indexes):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(data.euclidian_distance)
        euclidian_distance=data.euclidian_distance[ind]
        _plot_euclidian_distance_dist(euclidian_distance,ax,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,**kwargs)
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    
    fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the number of hit segments")
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig





def plot_euclidian_distance_by_tag(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=2,figsize=(24,8*2),facecolor='white')
    
    indexes=[(data.tag==2),(data.tag==3)]
    labels=["Track hit", "Noise", ]
    
    for k, ind in enumerate(indexes):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(data.euclidian_distance)
        euclidian_distance=data.euclidian_distance[ind]
        _plot_euclidian_distance_dist(euclidian_distance,ax,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,**kwargs)
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the hit tag")
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig





def plot_euclidian_distance_by_primary(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    fig,axes=plt.subplots(ncols=2,nrows=2,figsize=(24,8*2),facecolor='white')
    
    indexes=[(data.traj_parentID==0),(data.traj_parentID!=0)]
    labels=["Primary trajectory", "Secondary trajectory", ]
    
    for k, ind in enumerate(indexes):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(data.euclidian_distance)
        euclidian_distance=data.euclidian_distance[ind]
        _plot_euclidian_distance_dist(euclidian_distance,ax,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,**kwargs)
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    
    fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node for primary and secondary trajectories")
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig



def plot_euclidian_distance_by_pdg(data:DataContainer|tuple[dict[str,np.ndarray],PGunEvent],
                show:bool=True,
                savefig_path:str=None,
                particles_to_consider_pdg:list[int]=[-11,11,22, -13, 13, 2212, 211, -211],
                include_pdg_0:bool=True,
                **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    
    labels=[particle.Particle.from_pdgid(part).name for part in particles_to_consider_pdg]+['other']
    if include_pdg_0:
        labels=["pdg 0"]+labels
        particles_to_consider_pdg=[0]+particles_to_consider_pdg

    fig,axes=plt.subplots(ncols=2,nrows=len(particles_to_consider_pdg)+1,figsize=(24,8*(len(particles_to_consider_pdg)+1)),facecolor='white')
    
    for k, part in enumerate(particles_to_consider_pdg):
        ax=axes[k][0]
        ax2=axes[k][1]
        total_size=len(data.euclidian_distance)
        euclidian_distance=data.euclidian_distance[data.particle_pdg==part]
        _plot_euclidian_distance_dist(euclidian_distance,ax,**kwargs)
        _plot_euclidian_distance_cdf(euclidian_distance,ax2,**kwargs)
        ax.annotate(f"{labels[k]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        
    ax=axes[-1][0]
    ax2=axes[-1][1]
    total_size=len(data.euclidian_distance)
    euclidian_distance=data.euclidian_distance[np.isin(data.particle_pdg,particles_to_consider_pdg,invert=True)]
    _plot_euclidian_distance_dist(euclidian_distance,ax,**kwargs)
    _plot_euclidian_distance_cdf(euclidian_distance,ax2,**kwargs)
    ax.annotate(f"{labels[-1]}",xy=(-0.12, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
    ax2.annotate(f"Support: {len(euclidian_distance)/total_size*100:.1f}%",xy=(1.05, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)
        
    
    fig.suptitle("Distribution of the euclidian distance between the node predicted and the true node depending on the hit particle",y=0.9)
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
    
    if show:
        fig.show()
        
    return fig


def _get_perf(data:DataContainer,
              indexes:list[np.ndarray],
              show_progress_bar:bool=True,
              **kwargs):
    
    perfs=[[],[]]
    for k in tqdm.tqdm(range(len(indexes)),desc="Computing performances",disable=(not show_progress_bar)):
        if indexes[k].any():
            eucl=data.euclidian_distance[indexes[k]]
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
                    **kwargs):
    
    bins=np.logspace(np.log10(lims[0]),np.log10(lims[1]),Nbins) if use_log_scale else np.linspace(lims[0],lims[1],Nbins) #log xscale
    
    indexes=[(features>=bins[k])*(features<bins[k+1]) for k in range(len(bins)-1)]
    
    perf=_get_perf(data,indexes,**kwargs)
    
    fig,ax=plt.subplots(figsize=figsize, facecolor="white")
    ax2=ax.twinx()
    ax2.hist(features, bins=bins, alpha=0.3, label="Support")
    
    ax.plot(bins[:-1],perf[0],label="Mean",  color=color_list[3], marker="^")
    ax.plot(bins[:-1],perf[1],label="q=95%", color=color_list[4], marker="+")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Euclidian distance between true and predicted nodes (mm)')
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
                    **kwargs)
    

def plot_perf_distance(data:DataContainer,
                    x_lim_dist:tuple[float,float]=x_lim_dist,
                    Nbins:int=100,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    **kwargs):
    
    return _plot_perf_dist(data=data,
                    features=data.distance_node_point,
                    lims=x_lim_dist,
                    feature_name="true node (or cube center) distance to closest trajectory point (or projected)",
                    xlabel="Distance between true node and trajectory point (mm)",
                    Nbins=Nbins,
                    model_name=model_name,
                    show=show,
                    savefig_path=savefig_path,
                    **kwargs)