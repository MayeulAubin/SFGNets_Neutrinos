
################## HIT TAGGING PLOTS ####################

import os
import sys
import pickle as pk
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sfgnets.dataset import SparseEvent, RANGES, CUBE_SIZE, transform_inverse_cube
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, explained_variance_score, accuracy_score, ConfusionMatrixDisplay
import corner
import matplotlib
import particle
from sfgnets.plotting.plot_3D_events import plotly_event_hittag as _plotly_event_hittag

plt.rcParams['text.usetex'] = True

target_names = ["Vertex activity", "Single P", "Noise"]
use_log_scale=False
x_lim_dist=(0.5,2000) if use_log_scale else (0.5,200)
# x_lim_dist=(0.5,80) if use_log_scale else (0.5,80)
x_lim_charge=(0.5,2000) if use_log_scale else (0.5, 300)
delfault_dpi=200
Nbins=100
y_log_scale=False


particles_to_consider_pdg=[11,-11,22, 13, -13, 2212, 2112, 211, -211]

reaction_codes=[0, 1, 11, 12, 13, 16, 17, 21, 22, 23, 26, 31, 32, 33, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 51, 52]
reaction_codes_short_labels=['tot', 'ccqe', 'ccppip', 'ccppi0', 'ccnpip', 'cccoh', 'ccgam', 'ccmpi', 'cceta', 'cck', 'ccdis', 'ncnpi0', 'ncppi0', 'ncppim', 'ncnpip', 'nccoh', 'ncngam', 'ncpgam', 'ncmpi', 'ncneta', 'ncpeta', 'nck0', 'nckp', 'ncdis', 'ncqep', 'ncqen']
reaction_codes_long_labels=['Total', 'CCQE: $\\nu_{l} n \\rightarrow l^{-} p$', 'CC 1$\\pi: \\nu_{l} p \\rightarrow l^{-} p \\pi^{+}$', 'CC 1$\\pi: \\nu_{l} n \\rightarrow l^{-} p \\pi^{0}$', 'CC 1$\\pi: \\nu_{l} n \\rightarrow l^{-} n \\pi^{+}$', 'CC coherent-$\\pi: \\nu_{l} ^{16}O \\rightarrow l^{-} ^{16}O \\pi^{+}$', '1$\\gamma from \\Delta: \\nu_{l} n \\rightarrow l^{-} p \\gamma$', "CC (1.3 < W < 2 GeV): $\\nu_{l} N \\rightarrow l^{-} N' multi-\\pi", 'CC 1$\\eta: \\nu_{l} n \\rightarrow l^{-} p \\eta$', 'CC 1K: $\\nu_{l} n \\rightarrow l^{-} \\Lambda K^{+}$', "CC DIS (2 GeV $\leq W$): $\\nu_{l} N \\rightarrow l^{-} N' mesons$", 'NC 1$\\pi: \\nu_{l} n \\rightarrow \\nu_{l} n \\pi^{0}$', 'NC 1$\\pi: \\nu_{l} p \\rightarrow \\nu_{l} p \\pi^{0}$', 'NC 1$\\pi: \\nu_{l} n \\rightarrow \\nu_{l} p \\pi^{-}$', 'NC 1$\\pi: \\nu_{l} p \\rightarrow \\nu_{l} n \\pi^{+}$', 'NC coherent-\\pi: $\\nu_{l} ^{16}O \\rightarrow \\nu_{l} ^{16}O \\pi^{0}$', '1\\gamma from \\Delta: $\\nu_{l} n \\rightarrow \\nu_{l} n \\gamma$', '1\\gamma from \\Delta: $\\nu_{l} p \\rightarrow \\nu_{l} p \\gamma$', 'NC (1.3 < W < 2 GeV): $\\nu_{l} N \\rightarrow \\nu_{l} N multi-\\pi$', 'NC 1\\eta: $\\nu_{l} n \\rightarrow \\nu_{l} n \\eta$', 'NC 1\\eta: $\\nu_{l} p \\rightarrow \\nu_{l} p \\eta$', 'NC 1K: $\\nu_{l} n \\rightarrow \\nu_{l} \\Lambda K^{0}$', 'NC 1K: $\\nu_{l} n \\rightarrow \\nu_{l} \\Lambda K^{+}$', "NC DIS (2 GeV < W): $\\nu_{l} N \\rightarrow \\nu_{l} N' mesons", 'NC elastic: $\\nu_{l} n \\rightarrow \\nu_{l} n','NC elastic: $\\nu_{l} p \\rightarrow \\nu_{l} p']
reaction_codes_short_labels_dict=dict(zip(reaction_codes,reaction_codes_short_labels))
reaction_codes_long_labels_dict=dict(zip(reaction_codes,reaction_codes_long_labels))


color_list = ["#3f7dae","#ae703f",'#117238', '#4e6206', '#6e4d00','#823312','#851433']


def _get_vals(all_results:dict[str,np.ndarray]) -> tuple[np.ndarray]:
    val_true=np.vstack(all_results['y']).flatten()
    y_pred=np.vstack(all_results['predictions'])
    val_pred=np.argmax(y_pred,axis=-1) # argmax solution
    # val_pred=torch.multinomial(torch.Tensor(y_pred).to(device),num_samples=1)[...,0].cpu().numpy() # random choice solution
    
    return val_true, val_pred, y_pred

def _get_unscaled_features(all_results:dict[str,np.ndarray],
                           dataset:SparseEvent) -> np.ndarray:
    ## Get the features and unscale them (to obtain the meaningful features)
    scaled_features=np.vstack(all_results['f'])
    unscaled_features=dataset.scaler_minmax.inverse_transform(scaled_features)
    return unscaled_features


class DataContainer:
    def __init__(self, all_results:dict[str,np.ndarray], dataset:SparseEvent):
        self.val_true,self.val_pred,self.y_pred=_get_vals(all_results,)
        self.unscaled_features=_get_unscaled_features(all_results,dataset)
        self.all_results=all_results
        self.dataset=dataset
        
        self.distance_to_vertex=np.linalg.norm(self.unscaled_features[:,1:],axis=-1)
        self.charge=self.unscaled_features[:,0]
        
        self.aux=np.vstack(all_results['aux'])
        
    def __len__(self):
        return self.aux.shape[0]


def print_classification_report(data:DataContainer|tuple[dict,SparseEvent],**kwargs):
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    print(classification_report(data.val_true, data.val_pred, digits=3, target_names=target_names))
    

def plot_pred_proba(data:DataContainer|tuple[dict,SparseEvent],
                     show:bool=True,
                     savefig_path:str=None,
                     model_name:str='',
                     y_log_scale:bool=y_log_scale,
                    labels_pred_proba=["Pred Vertex activity", "Pred Single P", "Pred Noise"],
                    titles_pred_proba=["True Vertex activity", "True Single P", "True Noise"],
                     **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    
    labels=labels_pred_proba
    titles=titles_pred_proba
        
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(32, 8), facecolor="white")
    l=3
    for k in range(3):
        logits_pred=data.y_pred[data.val_true==k]
        for i in range(l-1,-1,-1):
            # sns.kdeplot(logits_pred[:,i], alpha=0.5, label=labels[i], ax=axes[k])
            axes[k].hist(logits_pred[:,i], bins=100, alpha=0.5, label=labels[i], density=True)
        axes[k].set_title(titles[k])
        axes[k].legend()
        axes[k].set_xlabel("Predicted probabilities")
        axes[k].set_ylabel("Frequency")
        if y_log_scale:
            axes[k].set_yscale('log')

    fig.suptitle(f'Hit-Tagging Model {model_name} Predicted probabilities distributions')
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig


def plot_crossed_distribution(data:DataContainer|tuple[dict,SparseEvent],
                                show:bool=True,
                                savefig_path:str=None,
                                model_name:str='',
                                use_log_scale:bool=use_log_scale,
                                **kwargs):
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    
    colors=["#003f5c", "#1e955a", "#ef5675", "#ffa600"]
    titles=["True Vertex activity", "True Single P", "True Noise"]


    total_dat=np.vstack([data.distance_to_vertex,data.charge,data.y_pred[:,0],data.y_pred[:,1],data.y_pred[:,2]]).swapaxes(0,1)

    fig=plt.figure(facecolor="white", figsize=(24,24))
    legend_lines=[]

    for k in range(3):
        if k in (0,1,2):
            fig=corner.corner(total_dat[data.val_true==k],
                        range=[x_lim_dist, x_lim_charge, (0,1), (0,1), (0,1),],
                        labels=["Vertex distance", "Charge", "Probability VA", "Probability SP", "Probability No"],
                        bins=100,
                            smooth=5.0,
                            color=colors[k],
                            fig=fig,
                            plot_contours =True,
                            axes_scale=["log" if use_log_scale else "linear", "log" if use_log_scale else "linear", "linear", "linear", "linear"],
                            labelpad=-0.2)
            legend_lines.append(matplotlib.lines.Line2D([0], [0], color=colors[k], lw=6, label=titles[k]))

    plt.figlegend(handles=legend_lines, loc="upper right", fontsize=12, bbox_to_anchor=(0.95,0.95))
    fig.suptitle(f"Crossed distributions of predicted probabilities and distant to vertex, charge and true label for Hit-Tagging Model {model_name}")
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig


def _get_perf_binning(data:DataContainer,
                 features:np.array,
                 lims:tuple[float,float],
                 Nbins:int=100,
                 use_log_scale:bool=use_log_scale,
                 show_progress_bar:bool=True,
                 **kwargs)-> tuple[np.ndarray]:
    
    bins=np.logspace(np.log10(lims[0]),np.log10(lims[1]),Nbins) if use_log_scale else np.linspace(lims[0],lims[1],Nbins) #log xscale

    prec=[]
    reca=[]
    f1=[]
    for i in tqdm.tqdm(range(len(bins)-1),desc="Computing performances over bins",disable=(not show_progress_bar)):
        indexes=(bins[i]<=features)*(features<bins[i+1])
        precision, recall, f1_score, _ = precision_recall_fscore_support(
                data.val_true[indexes], data.val_pred[indexes], average='macro')
        prec.append(precision)
        reca.append(recall)
        f1.append(f1_score)
    
    return np.array(prec), np.array(reca), np.array(f1), bins,


def _plot_perf_dist(precision:np.array,
                    recall:np.array,
                    f1_score:np.array,
                    bins:np.array,
                    features:np.array,
                    feature_name:str,
                    xlabel:str,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    color_list:list[str]=color_list,
                    y_log_scale:bool=y_log_scale,
                    figsize:tuple[float]=(16,12),
                    **kwargs):
    
    fig,ax=plt.subplots(figsize=figsize, facecolor="white")
    ax.plot(bins[:-1],precision,label="Precision",  color=color_list[3], marker="^")
    ax.plot(bins[:-1],recall,label="Recall", color=color_list[4], marker="+")
    ax.plot(bins[:-1],f1_score,label="F1 score",color=color_list[5], marker="s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Performance')
    ax.set_xscale('log' if use_log_scale else 'linear')
    ax.set_ylim((0.,1.))
    # ax.set_xlim(lims)

    ax2=ax.twinx()
    ax2.hist(features, bins=bins, alpha=0.3, label="Support")
    ax2.set_ylabel('Support')
    if y_log_scale:
        ax2.set_yscale('log')
    

    ax.set_title(f'Performances depending on the {feature_name} for model {model_name}')
    ax.legend()
    ax2.legend(loc="lower right")

    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig


def plot_perf_distance(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        use_log_scale:bool=use_log_scale,
                        x_lim_dist:tuple[float]=x_lim_dist,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    prec,reca,f1,bins=_get_perf_binning(data,data.distance_to_vertex,lims=x_lim_dist,Nbins=Nbins,use_log_scale=use_log_scale,**kwargs)
    
    fig=_plot_perf_dist(prec,reca,f1,bins,
                        features=data.distance_to_vertex,
                        feature_name="distance to the vertex",
                        xlabel="Distance to vertex (mm)",
                        show=show,
                        savefig_path=savefig_path,
                        model_name=model_name,
                        **kwargs,)
    return fig


def plot_perf_charge(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        use_log_scale:bool=use_log_scale,
                        x_lim_charge:tuple[float]=x_lim_charge,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    prec,reca,f1,bins=_get_perf_binning(data,data.charge,lims=x_lim_charge,Nbins=Nbins,use_log_scale=use_log_scale,**kwargs)
    
    fig=_plot_perf_dist(prec,reca,f1,bins,
                        features=data.charge,
                        feature_name="hit charge",
                        xlabel="Hit charge",
                        show=show,
                        savefig_path=savefig_path,
                        model_name=model_name,
                        **kwargs,)
    return fig


def _get_perf_bars(data:DataContainer,
                 features:np.array,
                 bars:list[float],
                 show_progress_bar:bool=True,
                 **kwargs)-> tuple[np.ndarray]:
    
    prec=[]
    reca=[]
    f1=[]
    count=[]
    for part in tqdm.tqdm(bars,desc="Computing performances over bars", disable=(not show_progress_bar)):
        indexes=(features==part)
        count.append(np.sum(indexes))
        precision, recall, f1_score, _ = precision_recall_fscore_support(
                data.val_true[indexes], data.val_pred[indexes], average='weighted')
        prec.append(precision)
        reca.append(recall)
        f1.append(f1_score)
        
    # Now events with unrecognised interaction type
    indexes=np.isin(features,bars,invert=True)
    count.append(np.sum(indexes))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
            data.val_true[indexes], data.val_pred[indexes], average='macro')
    prec.append(precision)
    reca.append(recall)
    f1.append(f1_score)
        
    return np.array(prec), np.array(reca), np.array(f1), count,



def _plot_perf_bars(precision:np.array,
                    recall:np.array,
                    f1_score:np.array,
                    count:np.array,
                    bar_names:list[str],
                    feature_name:str,
                    xlabel:str,
                    model_name:str='',
                    show:bool=True,
                    savefig_path:str=None,
                    color_list:list[str]=color_list,
                    y_log_scale:bool=y_log_scale,
                    figsize:tuple[float]=(16,12),
                    **kwargs):
    
    fig,ax=plt.subplots(figsize=figsize, facecolor="white")
    ax.plot(bar_names,precision,label="Precision",  color=color_list[3], marker="^")
    ax.plot(bar_names,recall,label="Recall", color=color_list[4], marker="+")
    ax.plot(bar_names,f1_score,label="F1 score",color=color_list[5], marker="s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Performance')
    ax.set_ylim((0.,1.))

    ax2=ax.twinx()
    rect=ax2.bar(x=bar_names, height=count, alpha=0.3, label="Support")
    ax2.bar_label(rect,padding=3)
    ax2.set_ylabel('Support')
    
    if y_log_scale:
        ax2.set_yscale('log')

    ax.set_title(f'Performances depending on the {feature_name} model {model_name}')
    ax.legend()
    ax2.legend(loc="lower right")

    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig


def plot_perf_particle(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        particles_to_consider_pdg:list[int]=particles_to_consider_pdg,
                        include_pdg_0:bool=True,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    labels=[particle.Particle.from_pdgid(part).name for part in particles_to_consider_pdg]+['other']
    if include_pdg_0:
        labels=["pdg 0"]+labels
        particles_to_consider_pdg=[0]+particles_to_consider_pdg
    prec,reca,f1,count=_get_perf_bars(data=data,features=data.aux[:,0],bars=particles_to_consider_pdg,**kwargs)
    fig=_plot_perf_bars(prec,reca,f1,count,labels,"particle type","Particle",model_name,show,savefig_path,**kwargs)
    return fig


def plot_perf_interaction(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        reaction_codes:list[int]=reaction_codes[1:],
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    labels=[reaction_codes_short_labels_dict[key] for key in reaction_codes]+['other']
    prec,reca,f1,count=_get_perf_bars(data=data,features=np.abs(data.aux[:,1]),bars=reaction_codes,**kwargs)
    fig=_plot_perf_bars(prec,reca,f1,count,labels,"interaction type","Neutrino interaction type",model_name,show,savefig_path,figsize=(24,12),**kwargs)
    return fig



def _get_perf_binning_by_tag(data:DataContainer,
                            features:np.array,
                            lims:tuple[float,float],
                            Nbins:int=100,
                            use_log_scale:bool=use_log_scale,
                            show_progress_bar:bool=True,
                            **kwargs)-> tuple[np.ndarray]:
    
    bins=np.logspace(np.log10(lims[0]),np.log10(lims[1]),Nbins) if use_log_scale else np.linspace(lims[0],lims[1],Nbins) #log xscale
    
    eff=[[[],[]],[[],[]],[[],[]]]
    vals=[data.val_true,data.val_pred]
    for i in tqdm.tqdm(range(len(bins)-1),desc="Computing performances over bins by tag",disable=(not show_progress_bar)):
        _indexes=(bins[i]<=features)*(features<bins[i+1])
        for l in range(2):
            for k in range(len(target_names)):
                indexes=_indexes*(vals[l]==k)
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                        data.val_true[indexes], data.val_pred[indexes], average=None, labels=[0,1,2])
                eff[k][l].append(recall[k] if l==0 else precision[k])
    
    return np.array(eff), bins, vals,



def _plot_perf_dist_by_tag(eff:np.array,
                            bins:np.array,
                            vals:list[np.ndarray],
                            features:np.array,
                            feature_name:str,
                            xlabel:str,
                            model_name:str='',
                            show:bool=True,
                            savefig_path:str=None,
                            color_list:list[str]=color_list,
                            y_log_scale:bool=y_log_scale,
                            face_colors:list[str]=["#cae8c8","#e7d4c6"],
                            figsize:tuple[float]=(36,18),
                            **kwargs):
    
    ## Plot the efficiency depending on the true/pred label and a feature
    fig,axes=plt.subplots(nrows=2,ncols=len(target_names),figsize=figsize, facecolor="white")
    ax2=[[0,0,0],[0,0,0]]
    for l,axs in enumerate(axes):
        for k, ax in enumerate(axs):
            ax.plot(bins[:-1],eff[k][l],label="Recall" if l==0 else "Precision",color=color_list[5], marker="s")
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Performance')
            ax.set_xscale('log' if use_log_scale else 'linear')
            # ax.set_xlim(x_lim_dist)
            ax.set_ylim(0.,1.02)
            # ax.set_ylim(min(min(prec[k]),min(reca[k]),min(f1[k]))-0.05,max(max(prec[k]),max(reca[k]),max(f1[k]))+0.05)

            ax2[l][k]=ax.twinx()
            ax2[l][k].hist(features[(vals[l]==k)], bins=bins, alpha=0.4, label="Support")
            ax2[l][k].set_ylabel('Support')
            if y_log_scale:
                ax2.set_yscale('log')        
            ax.legend()
            ax2[l][k].legend(loc="lower right")
            
            ax.set_facecolor(face_colors[0] if l==0 else face_colors[1])
            if l==0:
                ax.set_title(f'{target_names[k]} hits', fontsize=18, pad=20)
                
            if k==0:
                ax.annotate(f"{'True' if l==0 else'Predicted'} label",xy=(-0.10, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)

            
    fig.suptitle(f"Performances depending on the {feature_name} and on the labels for model {model_name}",fontsize=25,y=0.95)
    fig.show()

    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig


def plot_perf_distance_by_tag(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        use_log_scale:bool=use_log_scale,
                        x_lim_dist:tuple[float]=x_lim_dist,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    eff,bins,vals=_get_perf_binning_by_tag(data,data.distance_to_vertex,lims=x_lim_dist,Nbins=Nbins,use_log_scale=use_log_scale,**kwargs)
    
    fig=_plot_perf_dist_by_tag(eff,bins,vals,
                        features=data.distance_to_vertex,
                        feature_name="distance to the vertex",
                        xlabel="Distance to vertex (mm)",
                        show=show,
                        savefig_path=savefig_path,
                        model_name=model_name,
                        **kwargs,)
    return fig


def plot_perf_charge_by_tag(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        use_log_scale:bool=use_log_scale,
                        x_lim_charge:tuple[float]=x_lim_charge,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    eff,bins,vals=_get_perf_binning_by_tag(data,data.charge,lims=x_lim_charge,Nbins=Nbins,use_log_scale=use_log_scale,**kwargs)
    
    fig=_plot_perf_dist_by_tag(eff,bins,vals,
                        features=data.charge,
                        feature_name="hit charge",
                        xlabel="Hit charge",
                        show=show,
                        savefig_path=savefig_path,
                        model_name=model_name,
                        **kwargs,)
    return fig



def _get_perf_bars_by_tag(data:DataContainer,
                 features:np.array,
                 bars:list[float],
                 show_progress_bar:bool=True,
                 **kwargs)-> tuple[np.ndarray]:
    
    
    eff=[[[],[]],[[],[]],[[],[]]]
    count=[[[],[]],[[],[]],[[],[]]]
    vals=[data.val_true,data.val_pred]
    for part in tqdm.tqdm(bars,desc="Computing performances over bins bars",disable=(not show_progress_bar)):
        _indexes=(features==part)
        for l in range(2):
            for k in range(len(target_names)):
                indexes=_indexes*(vals[l]==k)
                count[k][l].append(np.sum(indexes))
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                        data.val_true[indexes], data.val_pred[indexes], average=None, labels=[0,1,2])
                eff[k][l].append(recall[k] if l==0 else precision[k])
                
    _indexes=np.isin(features,bars,invert=True)
    for l in range(2):
        for k in range(len(target_names)):
            # Now events with unrecognised interaction type
            indexes=_indexes*(vals[l]==k)
            count[k][l].append(np.sum(indexes))
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                        data.val_true[indexes], data.val_pred[indexes], average=None, labels=[0,1,2])
            eff[k][l].append(recall[k] if l==0 else precision[k])
    
    return np.array(eff), count, vals,




def _plot_perf_bars_by_tag(eff:np.array,
                            count:np.array,
                            vals:list[np.ndarray],
                            bar_names:list[str],
                            feature_name:str,
                            xlabel:str,
                            model_name:str='',
                            show:bool=True,
                            savefig_path:str=None,
                            color_list:list[str]=color_list,
                            y_log_scale:bool=y_log_scale,
                            face_colors:list[str]=["#cae8c8","#e7d4c6"],
                            figsize:tuple[float]=(36,18),
                            **kwargs):
    
    ## Plot the efficiency depending on the true/pred label and a feature
    fig,axes=plt.subplots(nrows=2,ncols=len(target_names),figsize=figsize, facecolor="white")
    ax2=[[0,0,0],[0,0,0]]
    for l,axs in enumerate(axes):
        for k, ax in enumerate(axs):
            ax.plot(bar_names,eff[k][l],label="Recall" if l==0 else "Precision",color=color_list[5], marker="s")
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Performance')
            ax.set_xscale('log' if use_log_scale else 'linear')
            # ax.set_xlim(x_lim_dist)
            ax.set_ylim(0.,1.02)
            # ax.set_ylim(min(min(prec[k]),min(reca[k]),min(f1[k]))-0.05,max(max(prec[k]),max(reca[k]),max(f1[k]))+0.05)

            ax2[l][k]=ax.twinx()
            rect=ax2[l][k].bar(x=bar_names, height=count[k][l], alpha=0.3, label="Support")
            ax2[l][k].set_ylabel('Support')
            if y_log_scale:
                ax2.set_yscale('log')        
            ax.legend()
            ax2[l][k].legend(loc="lower right")
            
            ax.set_facecolor(face_colors[0] if l==0 else face_colors[1])
            if l==0:
                ax.set_title(f'{target_names[k]} hits', fontsize=18, pad=20)
                
            if k==0:
                ax.annotate(f"{'True' if l==0 else'Predicted'} label",xy=(-0.10, 0.5),xycoords='axes fraction',horizontalalignment='center', verticalalignment='center', fontsize=18, rotation=90)

            
    fig.suptitle(f"Performances depending on the {feature_name} and on the labels for model {model_name}",fontsize=25,y=0.95)
    fig.show()

    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig



def plot_perf_particle_by_tag(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        particles_to_consider_pdg:list[int]=particles_to_consider_pdg,
                        include_pdg_0:bool=True,
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    labels=[particle.Particle.from_pdgid(part).name for part in particles_to_consider_pdg]+['other']
    if include_pdg_0:
        labels=["pdg 0"]+labels
        particles_to_consider_pdg=[0]+particles_to_consider_pdg
    eff,count,vals=_get_perf_bars_by_tag(data=data,features=data.aux[:,0],bars=particles_to_consider_pdg,**kwargs)
    fig=_plot_perf_bars_by_tag(eff,count,vals,labels,"particle type","Particle",model_name,show,savefig_path,**kwargs)
    return fig


def plot_perf_interaction_by_tag(data:DataContainer|tuple[dict,SparseEvent],
                        show:bool=True,
                        savefig_path:str=None,
                        model_name:str='',
                        reaction_codes:list[int]=reaction_codes[1:],
                        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
        
    labels=[reaction_codes_short_labels_dict[key] for key in reaction_codes]+['other']
    eff,count,vals=_get_perf_bars_by_tag(data=data,features=np.abs(data.aux[:,1]),bars=reaction_codes,**kwargs)
    fig=_plot_perf_bars_by_tag(eff,count,vals,labels,"interaction type","Neutrino interaction type",model_name,show,savefig_path,figsize=(40,10),**kwargs)
    return fig


def plot_event_display(data:DataContainer|tuple[dict,SparseEvent],
                       i:int=0,
                     show:bool=True,
                     savefig_path:str=None,
                     model_name:str='',
                     **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    
    X=data.all_results['c'][i].copy()
    transform_inverse_cube(X)
    f=data.dataset.scaler_minmax.inverse_transform(data.all_results['f'][i].copy())
    fig=_plotly_event_hittag(pos3d=X,
                            energies=np.clip(f[:,0],0,150),
                            max_energies=150,
                            hittags=data.all_results['y'][i][:,0],
                            general_label="True ",
                            xyz_ranges=RANGES,)

    hit_tag_pred=data.all_results['predictions'][i].copy()
    hittag=np.argmax(hit_tag_pred,axis=1)
    verPos=(X-f[:,1:]).mean(axis=0)

    fig=_plotly_event_hittag(pos3d=X,
                            energies=np.clip(f[:,0],0,150),
                            max_energies=150,
                            hittags=hittag,
                            general_label="Pred ",
                            xyz_ranges=RANGES,
                            cmaps=["autumn","copper","summer"],
                            fig=fig,
                            vertex=verPos[:,None],
                            )
    
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig



def plot_distance_by_interaction(data:DataContainer|tuple[dict,SparseEvent],
                                show:bool=True,
                                savefig_path:str=None,
                                reaction_codes:list[int]=[1,12,26],
                                **kwargs):

    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data
    
    fig,axes=plt.subplots(nrows=1,ncols=len(reaction_codes),figsize=(10*len(reaction_codes),8), facecolor="white")
    for k,ax in enumerate(axes):
        indexes=(np.abs(data.aux[:,1])==reaction_codes[k])*(data.val_true==0)
        distance_to_vertex=np.sort(data.distance_to_vertex[indexes])
        ax.plot(distance_to_vertex,np.arange(len(distance_to_vertex))/len(distance_to_vertex),label="CDF of VA hits")
        ax.set_xlabel("Distance to the vertex (mm)")
        ax.set_ylabel("Cumulative distribution")
        ax.set_title(reaction_codes_long_labels_dict[reaction_codes[k]])
        # ax.set_title(reaction_codes_short_labels_dict[reaction_codes[k]])
        ax.set_xlim((0.,100.))
        ax.set_ylim((0.,1.))
        ax.set_xticks(np.linspace(0.,100.,11))
        ax.set_xticks(np.linspace(0.,100.,21), minor=True)
        ax.set_yticks(np.linspace(0.,1.,11))
        ax.set_yticks(np.linspace(0.,1.,21), minor=True)
        ax.grid()
        ax.grid(which = 'minor', alpha = 0.2)
        ax.legend()
    
    fig.suptitle("Cumulative distribution of hit distance to the vertex for VA hits depending on the neutrino interaction")
        
    if savefig_path is not None:
        fig.savefig(savefig_path,dpi=delfault_dpi)
        
    if show:
        fig.show()
        
    return fig



def plots(data:DataContainer|tuple[dict,SparseEvent],
          plots_chosen:list[str]=["pred_proba","crossed_distribution","perf_charge","perf_distance","perf_charge_by_tag","perf_distance_by_tag","perf_interaction","perf_particle","perf_interaction_by_tag","perf_particle_by_tag","event_display"],
        show:bool=True,
        savefig_path:str=None,
        model_name:str='',
        **kwargs):
    
    if (not data.__class__.__name__=="DataContainer"): # if the data is not a DataContainer, it must be the input of the class to construct the object
        data=DataContainer(*data) # construct the DataContainer out of the provided data

    figs=[]
    
    if savefig_path is not None:
        root=".".join(savefig_path.split('.')[:-1])
        suffix=savefig_path.split('.')[-1]
    
    for func in plots_chosen:
        print(f"Plotting {func}...")
        figs.append(globals()["plot_"+func](data=data,
                        show=show,
                        model_name=model_name,
                        savefig_path=root+func+suffix if savefig_path is not None else None,
                        **kwargs))
    
    return figs