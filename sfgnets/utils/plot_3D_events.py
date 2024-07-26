## Defines the general plot function for plotting in 3D events

import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import particle

pdg_names_dict=dict(zip([0,-11,11,22,-13,13,2212,211,-211],['pdg0','e+','e-','gamma','mu+','mu-','p','pi+','pi-']))

def rgb_to_hex(rgb):
    """
    Convert RGB values to a hexadecimal color code.

    Args:
        rgb (tuple): A tuple containing the RGB values as floats between 0 and 1.

    Returns:
        str: Hexadecimal color code representing the RGB values.
    """
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def plotly_event_general(pos3d:np.ndarray|None=None,
                    image:np.ndarray|None=None,
                    track:np.ndarray|None=None, 
                    max_energy:float=200,
                    cube_size:float = 10,
                    energies:np.ndarray|None=None,
                    xyz_ranges:list[tuple[float,float]]|np.ndarray=np.array([[-985.92, +985.92],[-257.56, +317.56],[-2888.78, -999.1]]),
                    fig=None,
                    cmap:str="Wistia",
                    track_color:str="#11a337",
                    title:str="Particles interacting in SFG detector",
                    label:str='Hit',
                    track_label:str|list[str]='Track',
                    pdg:np.ndarray|None=None,
                    track_pdg:np.ndarray|None=None,
                    vertex:tuple[list[float],list[float],list[float]]|None=None,
                    center:bool=False,
                    focus:bool=False,
                    momentum:np.ndarray|None=None,
                    cone_scale:float=1.,
                    darkmode:bool=False,
                    figsize:tuple[int,int]|None=None,
                    hit_opacity_factor:float=1.,
                    **kwargs
                    ) -> go.Figure:
    """
    Generate a 3D plot using Plotly for visualising a voxelised image.

    Args:
        pos3d (numpy.ndarray): The 3D coordinates of the hit as a numpy array.
        image (numpy.ndarray): The voxelised image as a numpy array.
        max_energy (float, optional): The maximum energy value for color scaling. Default is 1.
        cube_size (float, optional): The size of the cube in the plot. Default is 1.
    """
    
    if figsize is None:
        ## Figure size
        # full screen
        width, height = 1500, 800  # Adjust these values as needed
        # small screen
        width, height = 1000, 562  # Adjust these values as needed
    else:
        width, height = figsize
        
    
    if darkmode:
        ## Blue theme
        gridcolor='blue'
        backgroundcolor="#07042F"
        legend_color="white"
        paper_bgcolor='rgba(0, 0, 0, 0)'
        plot_bgcolor='rgba(0, 0, 0, 0)'
    else:
        ## White theme
        gridcolor='#505050'
        backgroundcolor="#ececec"
        legend_color="black"
        paper_bgcolor='rgba(255, 255, 255, 255)'
        plot_bgcolor='rgba(255, 255, 255, 255)'
    
    
    
    if image is not None:
        if pos3d is not None:
            raise ValueError("Cannot provide both pos3d and image.")
        # Convert the image to coordinates
        image_int = image.astype(int)
        x1, y1, z1 = np.nonzero(image_int)
        opacity = np.clip(image / (max_energy * 0.5), 0, 1)
        
    elif pos3d is not None:
        if pos3d.shape[-1]!= 3:
            raise ValueError("pos3d must be a 3D array")
        reshaped_pos3d = np.reshape(pos3d, (-1,3))
        x1, y1, z1 = reshaped_pos3d[...,0], reshaped_pos3d[...,1], reshaped_pos3d[...,2]
        if energies is not None:
            reshaped_energies=np.reshape(energies, (-1,))
    
    elif track is not None:
        if track.shape[-1]!= 3:
            raise ValueError("track must be a 3D array")
        
    else:
        raise ValueError("Must provide either pos3d or image or track")

    # Create a 3D figure if not provided
    if fig is None:
        fig = go.Figure()
    

    base_cmap = plt.get_cmap(cmap)
    num_colors = 100
    hex_colors = [rgb_to_hex(base_cmap(i / (num_colors - 1))) for i in range(num_colors)]
    custom_cmap = ListedColormap(hex_colors)
    N = int(max_energy) + 1  # Adjust N to your desired range
    values = np.linspace(0, N, num=N)
    colors = [rgb_to_hex(custom_cmap(value / N)) for value in values]

    # Create a mesh3d trace for each voxel (cube)
    if pos3d is not None or image is not None:
        for j, (x, y, z) in enumerate(zip(x1, y1, z1)):
            
            name=label
            
            if image is not None:
                op=opacity[x,y,z]
                cl=colors[int(image[x, y, z])]
            else:
                op=0.6
                if energies is not None:
                    cl=colors[int(reshaped_energies[j])]
                else:
                    cl=np.random.choice(colors)
            
            if pdg is not None:
                if pos3d is not None:
                    # old_index=np.unravel_index(j, pos3d.shape[:-1])
                    particle_pdg=pdg[j]
                    if j:
                        # if the pdg is already a number, then it should be converted to its string representation
                        if type(particle_pdg) is int or type(particle_pdg) is float or type(particle_pdg) is np.float32: 
                            particle_pdg=int(particle_pdg)
                            # we first try to use the dictionary
                            try: 
                                name=f"{pdg_names_dict[particle_pdg]}"
                            # otherwise we extend the dictionary
                            except KeyError:
                                try:
                                    pdg_names_dict[particle_pdg]=particle.Particle.from_pdgid(particle_pdg).name
                                    name=f"{pdg_names_dict[particle_pdg]}"
                                except particle.ParticleNotFound:
                                    name="not found"
                        # if the pdg is a list/array (as when using hit segments pdg), use it as it is
                        elif len(particle_pdg)>0:
                            name=f"{particle_pdg}"
            
            cube = go.Mesh3d(
                x=[x - cube_size/2, x - cube_size/2, x + cube_size/2, x + cube_size/2, x - cube_size/2, x - cube_size/2, x + cube_size/2, x + cube_size/2],
                y=[y - cube_size/2, y + cube_size/2, y + cube_size/2, y - cube_size/2, y - cube_size/2, y + cube_size/2, y + cube_size/2, y - cube_size/2],
                z=[z - cube_size/2, z - cube_size/2, z - cube_size/2, z - cube_size/2, z + cube_size/2, z + cube_size/2, z + cube_size/2, z + cube_size/2],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                color=cl,
                flatshading=True,
                opacity=np.clip(op*hit_opacity_factor,0.,1.),
                legendgroup=label,
                name=name,
                showlegend=(j==0),
                colorbar={"title":"Energy deposited",
                          "tickmode":"array",
                          "tickvals":np.linspace(0, N, num=10),
                          },
                colorscale=[vc for vc in zip(values/N,colors)],
                showscale=(j==0) and (energies is not None),
            )
            fig.add_trace(cube)
    
    # Adds the vertex point
    if vertex is not None:
        point=go.Scatter3d(x=vertex[0],y=vertex[1],z=vertex[2],
                           mode="markers",
                           marker={"size":10.,"color":"#860021"},
                           name="Vertex")
        fig.add_trace(point)
        
    # Adds the tracks:
    if track is not None:
        if len(track.shape)==2:
            
            track_=track[None,...]
            
            if track_pdg is not None:
                track_pdg_=track_pdg[None,...]
            else:
                track_pdg_=None
                
            if momentum is not None:
                momentum_=momentum[None,...]
                
            if type(track_label) is not list:
                track_label=[track_label]
        else:
            track_=track
            track_pdg_=track_pdg
            momentum_=momentum
        
        for k,t in enumerate(track_):
            if track_pdg_ is None:
                ## If no momenta is provided, plot the tracks as points
                if momentum is None:
                    trk=go.Scatter3d(x=t[:,0],y=t[:,1],z=t[:,2],
                                    mode="markers", # can be changed to "lines+markers" to include lines connecting points
                                    marker={"size":4.,"color":track_color},
                                    name=track_label[k],
                                    )
                ## If the momentum is provided, plot the tracks+momentum as cones
                else:
                    cone_rescaling=(np.linalg.norm(t[1:]-t[:-1],axis=1)*2/(np.linalg.norm(momentum_[k][1:],axis=-1)+np.linalg.norm(momentum_[k][:-1],axis=-1))).min()*100 if len(t)>1 else 1.
                    # cone_rescaling*=np.max(np.linalg.norm(momentum_[k],axis=-1))/1000 if np.max(np.linalg.norm(momentum_[k],axis=-1))<200 else 1.
                    cone_rescaling*=1.+1e-6 -np.exp(-np.mean(np.linalg.norm(momentum_[k],axis=-1))/100)
                    trk=go.Cone(x=t[:,0],y=t[:,1],z=t[:,2],
                                u=momentum_[k,:,0],v=momentum_[k,:,1],w=momentum_[k,:,2],
                                    # sizemode="absolute",
                                    sizeref=cone_scale/cone_rescaling,
                                    name=track_label[k],
                                    colorbar={"title":"Momentum",
                                              "tickmode":"array",
                                              "tickvals":[0,500,1000,1500,2000,2500,3000, 3500],
                                              "ticktext":["0","500 MeV","1 GeV", "1500 MeV", "2 GeV", "2500 MeV", "3 GeV", "3500 MeV"],
                                              "ticks":"outside"},
                                    cmin=0.,
                                    cmax=3500.,
                                    colorscale=[(0.0,track_color),(1.0,"#ffffff")],
                                    showscale=True,
                                    showlegend=True,
                                    
                                    )
                fig.add_trace(trk)
            else:
                # if provided pdgs, we filter by pdg and use each of them separately to have the names of the points set accordingly
                pdgs_of_interest=np.unique(track_pdg_)
                for l,pdg_ in enumerate(pdgs_of_interest):
                    pdg_=int(pdg_)
                    # we first try to use the dictionary
                    try: 
                        name=f"{pdg_names_dict[pdg_]}"
                    # otherwise we extend the dictionary
                    except KeyError:
                        try:
                            pdg_names_dict[pdg_]=particle.Particle.from_pdgid(pdg_).name
                            name=f"{pdg_names_dict[pdg_]}"
                        except particle.ParticleNotFound:
                            name="not found"
                        
                    if momentum is None:
                        trk=go.Scatter3d(x=t[track_pdg_[k]==pdg_,0],y=t[track_pdg_[k]==pdg_,1],z=t[track_pdg_[k]==pdg_,2],
                                        mode="markers", # can be changed to "lines+markers" to include lines connecting points
                                        marker={"size":4.,"color":track_color},
                                        # legendgroup=track_label[k],
                                        # legendgrouptitle_text=track_label[k],
                                        # showlegend=(l==0),
                                        name=name,
                                        legend="legend2",
                                        )
                    else:
                        cone_rescaling=(np.linalg.norm(t[track_pdg_[k]==pdg_][1:]-t[track_pdg_[k]==pdg_][:-1],axis=1)*2/(np.linalg.norm(momentum_[k,track_pdg_[k]==pdg_][1:],axis=-1)+np.linalg.norm(momentum_[k,track_pdg_[k]==pdg_][:-1],axis=-1))).min()*100 if len(t[track_pdg_[k]==pdg_])>1 else 1.
                        # cone_rescaling*=np.max(np.linalg.norm(momentum_[k,track_pdg_[k]==pdg_],axis=-1))/1000 if np.max(np.linalg.norm(momentum_[k,track_pdg_[k]==pdg_],axis=-1))<200 else 1.
                        cone_rescaling*=1.+1e-6 -np.exp(-np.mean(np.linalg.norm(momentum_[k,track_pdg_[k]==pdg_],axis=-1))/100)
                        trk=go.Cone(x=t[track_pdg_[k]==pdg_,0],y=t[track_pdg_[k]==pdg_,1],z=t[track_pdg_[k]==pdg_,2],
                                    u=momentum_[k,track_pdg_[k]==pdg_,0],v=momentum_[k,track_pdg_[k]==pdg_,1],w=momentum_[k,track_pdg_[k]==pdg_,2],
                                        # sizemode="absolute",
                                        sizeref=cone_scale/cone_rescaling,
                                        # legendgroup=track_label[k],
                                        # legendgrouptitle_text=track_label[k],
                                        colorbar={"title":"Momentum",
                                              "tickmode":"array",
                                              "tickvals":[0,500,1000,1500,2000,2500,3000, 3500],
                                              "ticktext":["0","500 MeV","1 GeV", "1500 MeV", "2 GeV", "2500 MeV", "3 GeV", "3500 MeV"],
                                              "ticks":"outside"},
                                        cmin=0.,
                                        cmax=3500.,
                                        colorscale=[(0.0,track_color),(1.0,"#ffffff")],
                                        showscale=(l==0),
                                        showlegend=True,
                                        name=name,
                                        legend="legend2",
                                        )
                    fig.add_trace(trk)
                    
                fig.update_layout(legend2={"title":track_label[k],"yanchor":"bottom","xanchor":"right"})
                    
                # trk=go.Scatter3d(x=t[np.isin(track_pdg_[k],pdgs_of_interest,invert=True),0],y=t[np.isin(track_pdg_[k],pdgs_of_interest,invert=True),1],z=t[np.isin(track_pdg_[k],pdgs_of_interest,invert=True),2],
                #                     mode="markers", # can be changed to "lines+markers" to include lines connecting points
                #                     marker={"size":4.,"color":track_color},
                #                     legendgroup=track_label[k],
                #                     showlegend=False,
                #                     name="other",
                #                     )
                # fig.add_trace(trk)
                

    
    # Adds the vertex point

    # Set the margin to remove all margins
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Set the image size based on the aspect ratio
    fig.update_layout(width=width, height=height,title=title)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xyz_ranges[0],
                       title='X',
                       showticklabels=True,
                       gridcolor=gridcolor, backgroundcolor=backgroundcolor),
            yaxis=dict(range=xyz_ranges[1],
                       title='Y',
                       showticklabels=True,
                       gridcolor=gridcolor, backgroundcolor=backgroundcolor),
            zaxis=dict(range=xyz_ranges[2],
                       title='Z',
                       showticklabels=True,
                       gridcolor=gridcolor, backgroundcolor=backgroundcolor),
        )
    )
    
    # Set the aspect ratio to be equal
    aspect_ratio_x = 1
    aspect_ratio_y = (xyz_ranges[1][1]-xyz_ranges[1][0])/(xyz_ranges[0][1]-xyz_ranges[0][0])
    aspect_ratio_z = (xyz_ranges[2][1]-xyz_ranges[2][0])/(xyz_ranges[0][1]-xyz_ranges[0][0])
    aspect_ratio = np.array([aspect_ratio_x,aspect_ratio_y,aspect_ratio_z])
    
    fig.update_layout(scene=dict(aspectmode='manual',
                  aspectratio=dict(x=aspect_ratio_x, y=aspect_ratio_y, z=aspect_ratio_z)))

    # Set the camera parameters for initial zoom
    eye= dict(x=0., y=0.1, z=0.) if focus else dict(x=0., y=1., z=0.)
    if vertex is not None and center:
        # _eye=np.array([eye['x'],eye['y'],eye['z']])
        _center = (vertex[:,0]-np.mean(xyz_ranges,axis=1))/(xyz_ranges[:,1]-xyz_ranges[:,0])*aspect_ratio
        # _center = vertex[:,0]
        _center = dict(x=_center[0],y=_center[1],z=_center[2])
        # print(center)
        # center= dict(x=0, y=0.2, z=-0.3)
    else:
        _center= dict(x=0, y=0, z=0)
    fig.update_layout(
        scene=dict(
            camera=dict(
                center=_center,  # adjust the center of the view
                eye=eye,  # adjust the camera's initial position
            )
        )
    )

    fig.update_layout(paper_bgcolor=paper_bgcolor, plot_bgcolor=plot_bgcolor)
    fig.update_layout(legend=dict(font=dict(color=legend_color),orientation="h"))

    return fig


def plotly_event_hittag(pos3d:np.ndarray=None,
                        image:np.ndarray=None,
                        energies:np.ndarray=None,
                        hittags:np.ndarray=None,
                        pdg:np.ndarray=None,
                        cmaps:list[str]=["spring","Wistia","cool"],
                        general_label:str="",
                        fig=None,
                        vertex:tuple[list[float],list[float],list[float]]=None,
                        **kwargs):
    """
    Plots 3D events with different labelgroups and cmaps for each tag: 1 is Multiparticle, 2 is Single particle, 3 is noise
    """
    
    if hittags is None:
        raise ValueError("Can't plot with no hittags")
    
    for i,label in enumerate(["Vertex activity", "Single particle", "Noise"]):
        
        
        fig=plotly_event_general(pos3d=pos3d[hittags==i] if pos3d is not None else None,
                                 image=image[hittags==i] if image is not None else None, 
                                energies=energies[hittags==i] if energies is not None else None,
                                fig=None if i==0 and fig is None else fig,
                                cmap=cmaps[i],
                                label=general_label+label,
                                pdg=pdg[hittags==i] if pdg is not None else None,
                                vertex=vertex if i==2 else None,
                                **kwargs)
        
    return fig


def plotly_event_nodes(track:np.ndarray,
                        general_label:str="Track",
                        color:str="#11a337",
                        fig=None,
                        vertex:tuple[list[float],list[float],list[float]]=None,
                        pdg:np.ndarray=None,
                        **kwargs
                        ):
    """
    Plots the tracks of particles in SFG given their nodes positions.
    """
    
    fig=plotly_event_general(track=track,
                             track_label=general_label,
                             track_color=color,
                             vertex=vertex,
                             fig=fig,
                             track_pdg=pdg,
                             **kwargs)
    
    return fig