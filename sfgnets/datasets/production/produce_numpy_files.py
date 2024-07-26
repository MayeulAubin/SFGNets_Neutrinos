from tkinter import Radiobutton
import ROOT as rt
import numpy as np
import sys
import tqdm
import particle
import pickle as pk

from ..constants import RANGES, CUBE_SIZE, ORIGIN


def prod_numpy_hittag_dataset(root_input_file,
                              output_dir,
                              verbose=True,
                              aux=False):
    """
    Starts the production of numpy event_#.npz files to be used as a dataset for the hittagging model (Vertex activity, tracks, noise).
    The root_input_file is a loaded root file that has been applied MakeProject
    Output_dir is the directory in which the event_#.npz files are created.
    The created files have the following entries:
        x:              features of interest for each hit (Time, Charge)
        c:              3D coordinates of each hit
        y:              target feature for each hit (TrkHitTag)
        verPos:         true vertex 3D position
        reconverPos:    reconstructed vertex 3D position
    If aux we add also some auxiliary variables:
        pdg:            PDG code of the highest energy depositing particle
        reaction_code:  vertex reaction code, corresponding to the interaction type
    """
        
    recon_dir=root_input_file.Get("ReconDir") # Get the Reconstruction directory of the root file
    sfg_tree=recon_dir.Get("SFG") # Get the SFG tree of the Reconstruction directory

    truth_dir=root_input_file.Get("TruthDir") # Get the Truth directory of the root file
    vtx_tree=truth_dir.Get("Vertices") # Get the vertex tree of the 
    
    failed_events=[0,0,0]

    ## Iterate over the entries (each entry is an event) of the SFG tree
    ## Use tqdm to have a nice progress bar
    for entry in tqdm.tqdm(range(sfg_tree.GetEntries()),total=sfg_tree.GetEntries(),desc="Creating numpy files"):
        
        sfg_tree.GetEntry(entry) # Get the current entry (event) of the SFG tree
        vtx_tree.GetEntry(entry) # Get the current entry (event) of the vertex tree
        
        ## Assert that we have exactly one primary vertex in SFG
        NVtx=vtx_tree.NVtxSFG
        if NVtx!=1:
            if verbose:
                print(f"ERROR event {entry}: Wrong number of vertices in SFG : {NVtx}")
            failed_events[0]+=1
        ## Assert that we have at least one hit recorded
        elif len(sfg_tree.Hits)==0:
            if verbose:
                print(f"ERROR event {entry}: Hits array is empty")
            failed_events[1]+=1
        ## Assert that we have at least one reconstructed vertex
        elif len(sfg_tree.Vertices)==0:
            if verbose:
                print(f"ERROR event {entry}: Reconstructed vertices array is empty")
            failed_events[2]+=1
        else:
            ## Extract the true vertex position
            verPos=np.array([vtx_tree.Vertices[0].Position[0],vtx_tree.Vertices[0].Position[1],vtx_tree.Vertices[0].Position[2]])
        
            ## Extract the data from the SFG detector
            xyz_pos=[] # 3D coordinates of the hits
            features=[] # features of each hit for our model (time, charge, ...)
            target=[] # parameter of each hit to be predicted by our model (vertex activity or not)
            pdg=[]
            
            ## Get the correct hits to be kept with sfg_tree.AlgoResults[0].Hits
            indices=sfg_tree.AlgoResults[0].Hits
            
            ## Loops over all hits in SFG to be kept
            for index in indices:
                hit=sfg_tree.Hits[index]
                xyz_pos.append(np.array(hit.Position))
                features.append([hit.Time,
                                hit.Charge,
                                ])
                target.append(hit.TrkHitTag)
                
                if aux:
                    if len(hit.HitSegTrueEdepo)>0:
                        max_energy_deposition=max(range(len(hit.HitSegTrueEdepo)), key=hit.HitSegTrueEdepo.__getitem__)
                        pdg.append(hit.HitSegTruePDG[int(max_energy_deposition)])
                    else:
                        # pdg.append(0) # previously the PDG code was set to 0 for empty HitSeg arrays, but it is in conflict with the 'low energy neutrons' that have a 0 pdg code
                        pdg.append(888) # instead we should now use a 888 pdg code for empty HitSeg arrays
                
            ## Extract the closest (in spatial distance) reconstructed vertex
            recon_vertices=np.array([v.Position.Vect() for v in sfg_tree.Vertices])
            i_vert=np.argmin(np.linalg.norm(verPos[None,:]-recon_vertices,axis=-1))
            recon_verPos=recon_vertices[i_vert]
                
            
            ## Convert to numpy arrays
            xyz_pos=np.array(xyz_pos)
            features=np.array(features)
            target=np.array(target)
            
            if not aux:
                ## Save the event to a file
                np.savez_compressed(f"{output_dir}/event_{entry}",
                                    x=features,
                                    y=target,
                                    c=xyz_pos,
                                    verPos=verPos,
                                    recon_verPos=recon_verPos)
            else:
                ## Save the event to a file with auxiliary variables added
                np.savez_compressed(f"{output_dir}/event_{entry}",
                                    x=features,
                                    y=target,
                                    c=xyz_pos,
                                    verPos=verPos,
                                    recon_verPos=recon_verPos,
                                    pdg=pdg,
                                    reaction_code=int(str(vtx_tree.Vertices[0].ReactionCode)),)

    print(f"Failed events due to wrong number of true vertices in SFG: \t {failed_events[0]:>8} | {100*failed_events[0]/sfg_tree.GetEntries():.2f}%")
    print(f"Failed events due to empty Hit array: \t\t\t\t {failed_events[1]:>8} | {100*failed_events[1]/sfg_tree.GetEntries():.2f}%")
    print(f"Failed events due to empty reconstructed Vertices array: \t {failed_events[2]:>8} | {100*failed_events[2]/sfg_tree.GetEntries():.2f}%")
    print(f"Total failed events: \t\t\t\t\t\t {sum(failed_events):>8} | {100*sum(failed_events)/sfg_tree.GetEntries():.2f}%")
    
    
    
    
    
    
def prod_numpy_pgun_dataset(root_input_file:rt.TFile,
                              output_dir:str,
                              particle_name:str,
                              verbose:bool=True,
                              start:int=0,
                              **kwargs):
    """
    Starts the production of numpy event_#_{particle}.npz files to be used as a dataset for the pgun model.
    The root_input_file is a loaded root file that has been applied MakeProject
    Output_dir is the directory in which the event_#.npz files are created.
    The created files have the following entries:
        x:                      features (Energy deposition in charge measured)
        c:                      3D coordinates of the hit in mm
        tag:                    tag of the hit (TrkHitTag: Single particle, multi particle, noise)
        node_c:                 offset of the true trajectory point (associated to the hit) compared to the hit cube centre
        node_d:                 true direction of the trajectory point (in 3D)
        node_n:                 number of Hit Segments inside the hit, is 0 if there are no segment (in which case the other values are interpolated)
        node_m:                 momentum (in 3D) of the closest trajectory point to the Hit Seg Point of the same PDG
        Edepo:                  true energy deposited
        p_charge:               particle charge
        p_mass:                 particle mass
        pdg:                    particle PDG code
        traj_parentID:          parent ID of the trajectory associated with the hit
        traj_ID:                ID of the trajectory associated with the hit
        distance_node_point:    distance between the trajectory point (node_c) and the cube centre (c)
        NTraj:                  number of trajectories of the event
        traj_length:            estimated trajectory length (not reliable due to ordering issues)
        input_particle:         PDG code of the PGUN particle
        recon_c:                reconstructed offset of the trajectory point (reconstructed with the implemented Bayesian Filter)
        recon_d:                reconstructed direction of the trajectory point (reconstructed with the implemented Bayesian Filter)
        event_entry:            ID of the event
        order_index:            order of the hit along the track
    """
    
    list_of_particles=['e+','e-','gamma','mu+','mu-','n','p','pi+','pi-']
    list_of_pdgs=[-11,11,22,-13,13,2112,2212,211,-211]
    input_pdg_dict=dict(zip(list_of_particles,list_of_pdgs))


    recon_dir = root_input_file.Get(
        "ReconDir"
    )  # Get the Reconstruction directory of the root file
    sfg_tree = recon_dir.Get("SFG")  # Get the SFG tree of the Reconstruction directory

    truth_dir = root_input_file.Get(
        "TruthDir"
    )  # Get the Truth directory of the root file
    traj_tree = truth_dir.Get("Trajectories")  # Get the trajectory tree

    failed_events = [0, 0, 0]
    charge_dict = {}
    mass_dict = {}

    ## Iterate over the entries (each entry is an event) of the SFG tree
    ## Use tqdm to have a nice progress bar
    progess_bar=tqdm.tqdm(
        range(start,sfg_tree.GetEntries()), initial=start, total=sfg_tree.GetEntries(), desc=f"Creating numpy files for {particle_name}"
    )
    for entry in progess_bar:

        
        defaulting_hits = 0

        sfg_tree.GetEntry(entry)  # Get the current entry (event) of the SFG tree
        traj_tree.GetEntry(
            entry
        )  # Get the current entry (event) of the trajectory tree

        ## Assert that we have at least one hit recorded
        if len(sfg_tree.Hits) == 0:
            if verbose:
                print(f"ERROR event {entry}: Hits array is empty")
            failed_events[1] += 1
        ## Assert that we have at least one true trajectory
        elif traj_tree.NTraj == 0:
            if verbose:
                print(f"ERROR event {entry}: True trajectory array is empty")
            failed_events[0] += 1
        else:

            ## Construct the relevant arrays for the Trajectory points
            traj_pdg = []
            trajID = []
            parentID = []
            point = []
            momentum = []
            is_valid_traj = []
            t_length=[]
            
            ####### True Trajectories section #######

            ## Loops over the trajectories to construct the arrays
            for k in range(len(traj_tree.Trajectories)):

                Traj = traj_tree.Trajectories[k]

                ## First let's decide whether the trajectory is valid or not
                if (
                    len(Traj.Points) < 2
                ):  # if the trajectory has less than 2 points, it is not valid
                    is_valid_traj_ = 0
                elif (
                    len(Traj.Points) == 2
                ):  # if the trajectory has exactly two points, it is not valid if:
                    if 1e-2 > np.linalg.norm(
                        [
                            Traj.Points[0].PositionX - Traj.Points[1].PositionX,
                            Traj.Points[0].PositionY - Traj.Points[1].PositionY,
                            Traj.Points[0].PositionZ - Traj.Points[1].PositionZ,
                        ]
                    ):
                        is_valid_traj_ = 0  # the two points are close (norm of the difference below 0.01 mm)
                    elif 20 < np.linalg.norm(
                        [
                            Traj.Points[0].PositionX - Traj.Points[1].PositionX,
                            Traj.Points[0].PositionY - Traj.Points[1].PositionY,
                            Traj.Points[0].PositionZ - Traj.Points[1].PositionZ,
                        ]
                    ):
                        is_valid_traj_ = 0  # the two points are far (norm of the difference above 20 mm)
                    else:  # else the trajectory is valid
                        is_valid_traj_ = 1
                else:  # the trajectory is valid if it has more than two points
                    is_valid_traj_ = 1

                if is_valid_traj_:
                    point_=[]
                    ## Loops over the points of a trajectory
                    for j in range(len(Traj.Points)):
                        is_valid_traj.append(is_valid_traj_)
                        traj_pdg.append(Traj.PDG)
                        trajID.append(Traj.ID)
                        parentID.append(Traj.ParentID)

                        P = Traj.Points[j]
                        point_.append(
                            [
                                P.PositionX,
                                P.PositionY,
                                P.PositionZ,
                            ]
                        )
                        momentum.append(
                            [
                                P.MomentumX,
                                P.MomentumY,
                                P.MomentumZ,
                            ]
                        )
                    ## Appends the points    
                    point+=point_
                    ## Computes the trajectory length
                    point_=np.array(point_)    
                    traj_length_=np.sum(np.linalg.norm(point_[1:]-point_[:-1],axis=1))
                    ## Appends the trajectory length
                    t_length+=[traj_length_ for r in point_]

            if (
                len(traj_pdg) == 0
            ):  # if there are no valid true trajectory, skip the event
                if verbose:
                    print(f"ERROR event {entry}: No valid true trajectory")
                failed_events[2] += 1
                continue

            ## Convert to numpy arrays the trajectory arrays
            # is_valid_traj = np.array(is_valid_traj)
            traj_pdg = np.array(traj_pdg)
            trajID = np.array(trajID)
            parentID = np.array(parentID)
            point = np.array(point)
            momentum = np.array(momentum)
            t_length=np.array(t_length)
            
            
            ####### Reconstruced Trajectories section #######
            
            ## Construct the relevant arrays for the reconstructed trajectory points
            rec_point = []
            rec_dir = []
            hit_node_indx = []
            
            ## Get the correct reconstructed tracks to be kept with sfg_tree.AlgoResults[0].Particles
            indices = sfg_tree.AlgoResults[0].Particles
            
            if len(indices)==0:
                if verbose:
                    print(f"ERROR event {entry}: Particles array is empty.")
                no_reconstructed_trajectories=True
                
            else:
                no_reconstructed_trajectories=False
            
                ## Loops over all reconstructed particles in SFG to be kept
                for index in indices:

                    ## Get the reconstructed particles
                    rec_part = sfg_tree.Particles[index]
                    
                    
                    try:
                        ## Checks that there are not too many hits in this particle (to prevent memory errors)
                        if len(rec_part.Hits)<1000:
                            ## Get the indices of the hits corresponding to this trajectory
                            hit_node_indx.append(np.array(rec_part.Hits))
                        else:
                            ## If there are too many hits, no matching is done
                            hit_node_indx.append(np.array([-1]))
                            if verbose:
                                print(f"ERROR event {entry}: Too many hits in particle {index} with number of hits {len(rec_part.Hits)}")
                                
                    except Exception as E:
                        return entry, E
                        
                        
                    ## Extract the position and direction of the nodes
                    rec_point.append(np.array([sfg_tree.Nodes[k].Position.Vect() for k in rec_part.Nodes]))
                    rec_dir.append(np.array([sfg_tree.Nodes[k].Direction for k in rec_part.Nodes]))
                
                max_length_hits=max(len(h) for h in hit_node_indx)
                max_length_nodes=max(len(n) for n in rec_point)
                
                rec_point = np.array([np.pad(r,((0,max_length_nodes-len(r)),(0,0)),mode='constant',constant_values=np.inf) for r in rec_point])
                rec_dir = np.array([np.pad(r,((0,max_length_nodes-len(r)),(0,0)),mode='constant',constant_values=np.nan) for r in rec_dir])
                hit_node_indx = np.array([np.pad(r,(0,max_length_hits-len(r)),mode='constant',constant_values=-1) for r in hit_node_indx])
                    
                    
            ####### Hits section #######

            ## Extract the data from the Trajectory true directory
            traj_parentID = []  # 0 if primary track
            traj_ID = [] 
            distance_node_point = []  # should be quite low for non cross talk hits
            NTraj = traj_tree.NTraj
            traj_length = []
            
            ## Extract the data from the Reconstructed trajectory
            recon_c=[] # Reconstructed trajectory point interpolation coordinates
            recon_d=[] # Reconstructed trajectory direction

            ## Extract the data from the SFG detector
            xyz_pos = []  # 3D coordinates of the hits
            features = []  # features of each hit for our model (time, charge, ...)
            tag = (
                []
            )  # parameter of each hit to be predicted by our model (vertex activity or not)
            node_c = (
                []
            )  # True trajectory point interpolation coordinates, or hit segment of the highest energy depositing particle
            node_d = (
                []
            )  # True trajectory direction, or hit segment of the highest energy depositing particle
            node_n = (
                []
            )  # Number of Hit Seg inside the hit, is 0 if there are no segment (in which case the other values are interpolated)
            node_m = (
                []
            )  # Momentum of the closest trajectory point to the Hit Seg Point of the same PDG
            Edepo = []  # Total energy deposition in the hit
            p_charge = []  # Charge of the particle of maximum energy deposition
            p_mass = []  # Mass of the particle of maximum energy deposition
            pdg = []  # PDG code of the particle of maximum energy deposition
            
            order_index = []

            ## Get the correct hits to be kept with sfg_tree.AlgoResults[0].Hits
            indices = sfg_tree.AlgoResults[0].Hits

            ## Loops over all hits in SFG to be kept
            for index in indices:

                ## Get the hit
                hit = sfg_tree.Hits[index]

                xyz_pos.append(np.asarray(hit.Position))

                features.append(
                    [
                        hit.Time,
                        hit.Charge,
                    ]
                )

                tag.append(hit.TrkHitTag)

                node_n.append(len(hit.HitSegTrueP))

                energy_deposition = list(hit.HitSegTrueEdepo)
                energy_deposition = np.asarray(energy_deposition)

                ####### Signal Hits subsection #######
                
                ## Check that the hit has some segments ('node') in it, should be true for all hits with a tag != 3
                if len(energy_deposition) > 0:

                    Edepo.append(np.sum(energy_deposition))

                    # ## Uses the hit segments weighted by energy deposition to construct the node position and direction
                    # node_c.append(
                    #     np.sum(
                    #         np.asarray([pos.Vect() for pos in hit.HitSegPosition])
                    #         * energy_deposition[:, None],
                    #         axis=0,
                    #     )
                    #     / Edepo[-1]
                    # )
                    # node_d.append(
                    #     np.sum(
                    #         np.asarray(hit.HitSegDirection)
                    #         * energy_deposition[:, None],
                    #         axis=0,
                    #     )
                    #     / Edepo[-1]
                    # )

                    ## Get particle with highest energy deposition
                    # max_energy_deposition=np.argmax(energy_deposition)
                    max_energy_deposition = max(
                        range(len(hit.HitSegTrueEdepo)),
                        key=hit.HitSegTrueEdepo.__getitem__,
                    )
                    particle_PDG_with_highest_energy_deposition = hit.HitSegTruePDG[
                        max_energy_deposition
                    ]
                    pdg.append(particle_PDG_with_highest_energy_deposition)

                    ## Extract particle mass and charge from PDG, use a dict to avoid loading a Particle each time
                    try:
                        p_charge.append(
                            charge_dict[particle_PDG_with_highest_energy_deposition]
                        )
                        p_mass.append(
                            mass_dict[particle_PDG_with_highest_energy_deposition]
                        )
                    except KeyError:
                        charge_dict[particle_PDG_with_highest_energy_deposition] = (
                            particle.Particle.from_pdgid(
                                particle_PDG_with_highest_energy_deposition
                            ).charge
                        )
                        mass_dict[particle_PDG_with_highest_energy_deposition] = (
                            particle.Particle.from_pdgid(
                                particle_PDG_with_highest_energy_deposition
                            ).mass
                        )
                        p_charge.append(
                            charge_dict[particle_PDG_with_highest_energy_deposition]
                        )
                        p_mass.append(
                            mass_dict[particle_PDG_with_highest_energy_deposition]
                        )

                    ## Match the hit 'node' (Hit Seg Point) to the closest valid trajectory point of the same PDG
                    valid_indexes = (
                        traj_pdg == particle_PDG_with_highest_energy_deposition
                    )

                    ## If no trajectory of the same PDG code as the hit segment particle with highest energy deposition is found, default to hit segments
                    if not valid_indexes.any():
                        if verbose:
                            print(
                                f"ERROR event {entry}: Hit {index}: No point matching PDG {particle_PDG_with_highest_energy_deposition} defaulting to hit segments"
                            )
                        defaulting_hits += 1
                        node_c.append(np.array(hit.HitSegPosition[max_energy_deposition].Vect()))
                        node_d.append(np.array(hit.HitSegDirection[max_energy_deposition]))
                        ## Compute the distance between trajectory points and cube center
                        node_distances_to_trajectory_points = np.linalg.norm(
                            point - xyz_pos[-1][None, :], axis=1
                        )
                        ## Finds the closest point among the valid trajectories
                        index_of_closest_point = np.argmin(
                            node_distances_to_trajectory_points
                        )
                        mom_norm=np.linalg.norm(momentum[index_of_closest_point])
                        node_m.append(node_d[-1]*mom_norm) # set the momentum to be along the direction with a norm of the closest found momentum
                        traj_parentID.append(parentID[index_of_closest_point])
                        traj_ID.append(trajID[index_of_closest_point])
                        traj_length.append(t_length[index_of_closest_point])
                    
                    ## If a trajectory of same PDG could be found, proceed
                    else:
                        
                        
                        ## Compute the distance between trajectory points and cube center
                        node_distances_to_trajectory_points = np.linalg.norm(
                            point[valid_indexes] - xyz_pos[-1][None, :], axis=1
                        )

                        ## Finds the closest point among the valid trajectories
                        index_of_closest_point = np.argmin(
                            node_distances_to_trajectory_points
                        )
                        ## Extract its momentum
                        node_m.append(momentum[valid_indexes][index_of_closest_point])
                        ## Extract its trajectory length
                        traj_length.append(t_length[valid_indexes][index_of_closest_point])
                        ## Extract the ID of the parent trajectory (0 if primary)
                        traj_parentID.append(
                            parentID[valid_indexes][index_of_closest_point]
                        )
                        traj_ID.append(
                            trajID[valid_indexes][index_of_closest_point]
                        )
                        
                        
                        point_1_c = point[valid_indexes][index_of_closest_point]
                        
                        ## Match the hit to the second closest valid trajectory point of the same pdg and further than 0.01 mm of the first point
                        valid_indexes_2 = (
                            np.linalg.norm(point[valid_indexes] - point_1_c[None, :], axis=1) >= 1e-2
                        )
                        
                        ## Check if any points matched the previous condition, if not default to hit segments information
                        if not valid_indexes_2.any():
                            if verbose:
                                print(
                                    f"ERROR event {entry}: Hit {index}: No second point of the same trajectory, defaulting to hit segments"
                                )
                            defaulting_hits += 1
                            node_c.append(np.array(hit.HitSegPosition[max_energy_deposition].Vect()))
                            node_d.append(np.array(hit.HitSegDirection[max_energy_deposition]))
                            
                        ## If a second trajectory point could be found, proceed
                        else:
                            ## Get the second closest point of the same trajectory
                            index_of_closest_point_2 = np.argmin(
                                                                    node_distances_to_trajectory_points[valid_indexes_2]
                                                                )
                            
                            point_2_c = point[valid_indexes][valid_indexes_2][index_of_closest_point_2]
                            ## Get the direction vector between the two selected points
                            unit_dir_vector = point_2_c - point_1_c
                            unit_dir_vector /= np.linalg.norm(unit_dir_vector)
                            ## Get the projection of the hit coordinate onto the trajectory segment composed of the two points
                            proj = np.dot(xyz_pos[-1] - point_1_c, unit_dir_vector)
                            node_c.append(point_1_c + proj * unit_dir_vector)
                            ## Use the unit vector joining the two trajectory points as direction
                            node_d.append(unit_dir_vector)
                            
                    distance_node_point.append(np.linalg.norm(node_c[-1] - xyz_pos[-1]))
                            
                
                ####### Noise Hits subsection #######            

                ## Hit without any segment in it, will be constructing values based on the closest trajectory points
                else:

                    ## Match the hit to the closest valid trajectory point
                    # valid_indexes = is_valid_traj == 1
                    node_distances_to_trajectory_points = np.linalg.norm(
                        point - xyz_pos[-1][None, :], axis=1
                    )
                    index_of_closest_point = np.argmin(
                        node_distances_to_trajectory_points
                    )
                    pdg_of_closest_point = traj_pdg[index_of_closest_point]
                    momentum_of_closest_point = momentum[index_of_closest_point]
                    point_1_c = point[index_of_closest_point]

                    ## Match the hit to the second closest valid trajectory point of the same pdg and further than 0.01 mm of the first point
                    valid_indexes_2 = (traj_pdg == pdg_of_closest_point) * (
                        np.linalg.norm(point - point_1_c[None, :], axis=1) >= 1e-2
                    )

                    if not valid_indexes_2.any():
                        if verbose:
                            print(
                                f"ERROR event {entry}: Hit {index}: No point matching PDG {pdg_of_closest_point} for no segment hit, defaulting to all points"
                            )
                        defaulting_hits += 1
                        valid_indexes_2 = (
                            np.linalg.norm(point - point_1_c[None, :], axis=1) >= 1e-2
                        )

                    index_of_closest_point_2 = np.argmin(
                        node_distances_to_trajectory_points[valid_indexes_2]
                    )
                    point_2_c = point[valid_indexes_2][index_of_closest_point_2]

                    ## Get the direction vector between the two selected points
                    unit_dir_vector = point_2_c - point_1_c
                    unit_dir_vector /= np.linalg.norm(unit_dir_vector)

                    ## Get the projection of the hit coordinate onto the trajectory segment composed of the two points
                    proj = np.dot(xyz_pos[-1] - point_1_c, unit_dir_vector)
                    node_c.append(point_1_c + proj * unit_dir_vector)

                    ## Appends the other interpolated parameters
                    node_d.append(unit_dir_vector)
                    node_m.append(momentum_of_closest_point)
                    pdg.append(pdg_of_closest_point)
                    traj_parentID.append(parentID[index_of_closest_point])
                    traj_ID.append(trajID[index_of_closest_point])
                    ## Extract its trajectory length
                    traj_length.append(t_length[index_of_closest_point])
                    distance_node_point.append(np.linalg.norm(node_c[-1] - xyz_pos[-1]))

                    ## Interpolates the energy deposition from the charge from a linear regression (that was fitter on some training data)
                    Edepo.append(
                        0.0124058 * hit.Charge
                    )  # the coefficient 0.012 comes from a linear regression I did on the training dataset of hits with segments
                    # Edepo.append(0.) # other possibility

                    ## Extract particle mass and charge from PDG, use a dict to avoid loading a Particle each time
                    try:
                        p_charge.append(charge_dict[pdg_of_closest_point])
                        p_mass.append(mass_dict[pdg_of_closest_point])
                    except KeyError:
                        mass_dict[pdg_of_closest_point] = particle.Particle.from_pdgid(
                            pdg_of_closest_point
                        ).mass
                        charge_dict[pdg_of_closest_point] = particle.Particle.from_pdgid(
                            pdg_of_closest_point
                        ).charge
                        p_charge.append(charge_dict[pdg_of_closest_point])
                        p_mass.append(mass_dict[pdg_of_closest_point])

                    # ## If the hit has no segment in it, all values are put to zero except the node coordinates that is set to the hit position
                    # Edepo.append(0.)
                    # node_c.append(xyz_pos[-1])
                    # node_d.append(np.zeros(3))
                    # p_charge.append(0)
                    # p_mass.append(0.)
                    # # pdg.append(0)
                    # pdg.append(888)
                    
                    
                    
                ####### Reconstructed Trajectory matching subsection #######

                if (not no_reconstructed_trajectories):
                    ## Boolean array of matching the hit index
                    matching_hit_recon=(hit_node_indx==index) # shape (n_particles, n_hits)
                    ## Compute the distances between the cube center and the reconstructed trajectories points
                    node_distances_to_recon_trajectory_points = np.linalg.norm(
                                                                            rec_point - xyz_pos[-1][None,None, :], axis=-1
                                                                        )
                    
                    ## If we have exactly one matching, we use the associated particle
                    if np.sum(matching_hit_recon)==1:
                        
                        index_of_closest_recon_point=[0,0]
                        ## Select the particle that has the matching hit
                        index_of_closest_recon_point[0]=np.argmax(np.sum(matching_hit_recon,axis=-1)) # we first collapse the second dimension corresponding to the matching hits of each particle to extract the particle of interest
                        index_of_closest_recon_point[1]=np.argmin(node_distances_to_recon_trajectory_points[index_of_closest_recon_point[0]]) # we find the closest point of this selected particle
                        index_of_closest_recon_point=tuple(index_of_closest_recon_point)
                        
                        ## Extract the hit order in the reconstructed track
                        order_index.append(np.argmax(matching_hit_recon[index_of_closest_recon_point[0]])) # the hits are order by their position along the track in the sfgtree.Particles.Hits vector that we use
                    
                    ## Otherwise, we look among all particles
                    else:
                        if verbose:
                            print(
                                f"ERROR event {entry}: Hit {index}: {np.sum(matching_hit_recon)} matching among the reconstructed trajectories"
                            )
                        defaulting_hits += 1
                        
                        ## Find the closest point among the reconstructed trajectories
                        index_of_closest_recon_point = np.unravel_index(np.argmin(node_distances_to_recon_trajectory_points),
                                                                         node_distances_to_recon_trajectory_points.shape)
                        
                        order_index.append(1010+np.sum(matching_hit_recon)) # the indexes above 1000 correspond to hits not matched, above 1010 means that the wrong number of reconstructed trajectories was found 
                        
                    point_1_c=rec_point[index_of_closest_recon_point]
            
                    
                    ## Set the selected point to infinite distance to discard it for the second closest point
                    node_distances_to_recon_trajectory_points[index_of_closest_recon_point]=np.inf
                    
                    ## Find the second closest point of the already selected reconstructed trajectory
                    index_of_closest_recon_point_2 = np.argmin(node_distances_to_recon_trajectory_points[index_of_closest_recon_point[0]])
                    
                    point_2_c=rec_point[index_of_closest_recon_point[0]][index_of_closest_recon_point_2]
                    
                    
                    ## Get the direction vector between the two selected points
                    unit_dir_vector = point_2_c - point_1_c
                    unit_dir_vector /= np.linalg.norm(unit_dir_vector)
                    
                    ## Get the projection of the hit coordinate onto the trajectory segment composed of the two points
                    proj = np.dot(xyz_pos[-1] - point_1_c, unit_dir_vector)
                    recon_c.append(point_1_c + proj * unit_dir_vector)
                    
                    recon_d.append(rec_dir[index_of_closest_recon_point])
                    
            
            
            ####### Saving data #######

            ## Convert to numpy arrays
            xyz_pos = np.array(xyz_pos)
            features = np.array(features)
            # target=np.array(target)
            tag = np.array(tag)
            node_c = np.array(node_c)
            node_d = np.array(node_d)
            node_n = np.array(node_n)
            node_m = np.array(node_m)
            Edepo = np.array(Edepo)
            pdg=np.array(pdg)
            p_charge = np.array(p_charge)
            p_mass = np.array(p_mass)
            traj_parentID = np.array(traj_parentID)
            traj_ID = np.array(traj_ID)
            distance_node_point = np.array(distance_node_point)
            traj_length=np.array(traj_length)
            
            ## If there are no reconstructed trajectories, default the reconstruction to the cube centers and null directions
            if no_reconstructed_trajectories:
                recon_c=xyz_pos
                recon_d=np.array(node_d)*0.
                order_index=np.ones_like(traj_ID)*1001 # the indexes above 1000 correspond to hits not matched, 1001 correspond to no reconstructed trajectory was found
            else:
                recon_c=np.array(recon_c)
                recon_d=np.array(recon_d)
                order_index=np.array(order_index)
                
                
            ####### Assertion of the correct arrays length #######
            
            assert len(xyz_pos) == len(features), f"Features or xyz_pos have the wrong length: {len(features)} and {len(xyz_pos)}"
            assert len(xyz_pos) == len(tag), f"Wrong length: {len(tag)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(node_c), f"Wrong length: {len(node_c)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(node_d), f"Wrong length: {len(node_d)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(node_n), f"Wrong length: {len(node_n)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(node_m), f"Wrong length: {len(node_m)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(Edepo), f"Wrong length: {len(Edepo)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(p_charge), f"Wrong length: {len(p_charge)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(p_mass), f"Wrong length: {len(p_mass)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(pdg), f"Wrong length: {len(pdg)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(traj_parentID), f"Wrong length: {len(traj_parentID)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(traj_ID), f"Wrong length: {len(traj_ID)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(distance_node_point), f"Wrong length: {len(distance_node_point)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(traj_length), f"Wrong length: {len(traj_length)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(recon_c), f"Wrong length: {len(recon_c)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(recon_d), f"Wrong length: {len(recon_d)} instead of {len(xyz_pos)}"
            assert len(xyz_pos) == len(order_index), f"Wrong length: {len(order_index)} instead of {len(xyz_pos)}"
            

            ## Save the event to a file
            np.savez_compressed(
                f"{output_dir}/event_{entry}_{particle_name}",
                x=features,
                # y=target,
                c=xyz_pos,
                tag=tag,
                node_c=node_c,
                node_d=node_d,
                node_n=node_n,
                node_m=node_m,
                Edepo=Edepo,
                p_charge=p_charge,
                p_mass=p_mass,
                pdg=pdg,
                traj_parentID=traj_parentID,
                traj_ID=traj_ID,
                distance_node_point=distance_node_point,
                NTraj=NTraj,
                traj_length=traj_length,
                input_particle=input_pdg_dict[particle_name],
                recon_c=recon_c,
                recon_d=recon_d,
                event_entry=entry,
                order_index=order_index,
            )
            
            progess_bar.set_postfix({"Unmatched":f"{100*defaulting_hits/len(indices):.1f}%"})

    print(
        f"Failed events due to wrong number of true trajectories in SFG: \t {failed_events[0]:>8} | {100*failed_events[0]/sfg_tree.GetEntries():.2f}%"
    )
    print(
        f"Failed events due to empty Hit array: \t\t\t\t {failed_events[1]:>8} | {100*failed_events[1]/sfg_tree.GetEntries():.2f}%"
    )
    # print(f"Failed events due to empty reconstructed Vertices array: \t {failed_events[2]:>8} | {100*failed_events[2]/sfg_tree.GetEntries():.2f}%")
    print(
        f"Total failed events: \t\t\t\t\t\t {sum(failed_events):>8} | {100*sum(failed_events)/sfg_tree.GetEntries():.2f}%"
    )
    


    
    
    
    
def prod_numpy_pbomb_dataset(root_input_file:rt.TFile,
                              output_dir:str,
                              particle_name:str,
                              verbose:bool=True,
                              start:int=0,
                              **kwargs):
    """
    Starts the production of numpy event_#_{particle}.npz files to be used as a dataset for the VA model.
    The root_input_file is a loaded root file that has been applied MakeProject
    Output_dir is the directory in which the event_#.npz files are created.
    The created files have the following entries:
        sparse_image:       array containing both the coordinates of the hit cube centre (c), and the features (charge measured, tag)
        pos_ini:            initial position of the PGUN particle
        pos_fin:            final position of the PGUN particle
        pdg:                PDG code of the PGUN particle
        ke:                 kinetic energy (momentum norm) of the PGUN particle
        theta:              spherical theta coordinate of the initial direction of the PGUN particle
        phi:                spherical phi coordinate of the initial direction of the PGUN particle
        exit:               boolean indicating whether the particle exited the VA region
        adjacent_cube:      boolean indicating if a hit has been recorded adjacent to the VA region
        ke_exit:            kinetic energy (momentum norm) of the PGUN particle at exit
        theta_exit:         spherical theta coordinate of the exiting direction of the PGUN particle
        phi_exit:           spherical phi coordinate of the exiting direction of the PGUN particle
        pos_exit:           exiting position of the PGUN particle
        ke_exit_reduce:     kinetic energy (momentum norm) of the PGUN particle at exit
        theta_exit_reduce:  spherical theta coordinate of the initial direction of the PGUN particle
        phi_exit_reduce:    spherical phi coordinate of the initial direction of the PGUN particle
        pos_exit_reduce:    exiting position of the PGUN particle
        file_index:         ID of the file
        event_index:        ID of the event
    """
    
    list_of_particles=['e+','e-','gamma','mu+','mu-','n','p','pi+','pi-']
    list_of_pdgs=[-11,11,22,-13,13,2112,2212,211,-211]
    list_of_masses=[0.51,0.51,0.,105.66,105.66,939.57,938.27,139.57,139.57]
    list_of_charges=[+1,-1,0,+1,-1,0,+1,+1,-1]
    input_pdg_dict=dict(zip(list_of_particles,list_of_pdgs))
    mass_dict=dict(zip(list_of_particles,list_of_masses))
    charge_dict=dict(zip(list_of_particles,list_of_charges))
    
    eps = 1e-4
    cube_size = 2*CUBE_SIZE
    bins = 30
    bin_edges = np.linspace(start=-cube_size/2.-eps, stop=cube_size/2.+eps, num=bins+1)
    
    
    origin_point = ORIGIN
    VA_region_full_size = 9*cube_size
    VA_region_reduced_size = 7*cube_size


    recon_dir = root_input_file.Get(
        "ReconDir"
    )  # Get the Reconstruction directory of the root file
    sfg_tree = recon_dir.Get("SFG")  # Get the SFG tree of the Reconstruction directory

    truth_dir = root_input_file.Get(
        "TruthDir"
    )  # Get the Truth directory of the root file
    traj_tree = truth_dir.Get("Trajectories")  # Get the trajectory tree

    failed_events = [0, 0, 0, 0, 0]
    last_nb_failed_events = 0
    
    exit_s = 0
    adjacent_cube_s = 0
    total_exit_s = 0
    total_adjacent_cube_s = 0
    
    lookup_table = {}
    charges = []
    ini_pos = [[], [], []]
    kes = []
    thetas = []
    phis = []

    ## Iterate over the entries (each entry is an event) of the SFG tree
    ## Use tqdm to have a nice progress bar
    progess_bar=tqdm.tqdm(
        range(start,sfg_tree.GetEntries()), initial=start, total=sfg_tree.GetEntries(), desc=f"Creating numpy files for {particle_name}"
    )
    for entry in progess_bar:

        

        sfg_tree.GetEntry(entry)  # Get the current entry (event) of the SFG tree
        traj_tree.GetEntry(
            entry
        )  # Get the current entry (event) of the trajectory tree

        ## Assert that we have at least one hit recorded, otherwise skip the event
        if len(sfg_tree.Hits) == 0:
            if verbose:
                print(f"ERROR event {entry}: Hits array is empty")
            failed_events[1] += 1
        ## Assert that we have at least one true trajectory, otherwise skip the event
        elif traj_tree.NTraj == 0:
            if verbose:
                print(f"ERROR event {entry}: True trajectory array is empty")
            failed_events[0] += 1
        ## Assert that the PGUN particle trajectory has at least two points, otherwise skip the event
        elif len(traj_tree.Trajectories[0].Points) < 2:
            if verbose:
                print(f"ERROR event {entry}: PGUN particle trajectory has {len(traj_tree.Trajectories[0].Points)} points")
            failed_events[2] += 1   
        ## Assert that the first trajectory is that of the input particle, otherwise skip the event
        elif traj_tree.Trajectories[0].PDG != input_pdg_dict[particle_name]:
            if verbose:
                print(f"ERROR event {entry}: The first trajectory is of PDG {traj_tree.Trajectories[0].PDG} instead of {input_pdg_dict[particle_name]}")
            failed_events[4] += 1  
        ## Consider the event if all assertions are verified 
        else:

            ####### True Trajectories section #######

            ## Use the first trajectory (the one of the PGUN particle)

            Traj = traj_tree.Trajectories[0]
            
            point_=[]
            momentum=[]
            ## Loops over the points of a trajectory
            for j in range(len(Traj.Points)):

                P = Traj.Points[j]
                point_.append(
                    [
                        P.PositionX,
                        P.PositionY,
                        P.PositionZ,
                    ]
                )
                momentum.append(
                    [
                        P.MomentumX,
                        P.MomentumY,
                        P.MomentumZ,
                    ]
                )
            
            point_=np.array(point_)
            momentum=np.array(momentum)
            
            non_zero_momentum_indexes = (np.max(np.abs(momentum),axis=-1) > 0.)
            
            if not non_zero_momentum_indexes.any():
                if verbose:
                    print(f"ERROR event {entry}: Null energy at all points ")
                failed_events[3] += 1
                continue
            
            point_ = point_[non_zero_momentum_indexes]
            momentum = momentum[non_zero_momentum_indexes]
            
            L1_distance_to_the_origin = np.max(np.abs(point_-origin_point), axis=-1) # L1 distance, that is used to see if the particle exited the VA region
            L2_distance_to_the_origin = np.linalg.norm(point_-origin_point, axis=-1) # L2 distance, that is used to select the closest and furthest points
            
            exit_ = (L1_distance_to_the_origin > VA_region_full_size/2).any()
            exit_s += int(exit_)
            
            first_point_index = np.argmin(L2_distance_to_the_origin)
            last_point_index = np.argmax(L2_distance_to_the_origin)
            
            
            if (L1_distance_to_the_origin <= VA_region_full_size/2).any(): # check if there is at least a point in the VA region
                exiting_point_index = np.argmax(L2_distance_to_the_origin[(L1_distance_to_the_origin <= VA_region_full_size/2)])
            else: # if not use the closest point
                exiting_point_index = first_point_index
            
            if (L1_distance_to_the_origin <= VA_region_reduced_size/2).any(): # check if there is at least a point in the reduced VA region
                exiting_reduce_point_index = np.argmax(L2_distance_to_the_origin[(L1_distance_to_the_origin <= VA_region_reduced_size/2)])
            else: # if not use the closest point
                exiting_reduce_point_index = first_point_index

            pos_ini = point_[first_point_index]
            pos_fin = point_[last_point_index]
            pos_exit = point_[exiting_point_index]
            pos_exit_reduce = point_[exiting_reduce_point_index]
            
            mass = mass_dict[particle_name]
            
            ke = np.sqrt(np.sum(momentum[first_point_index]**2)+mass**2)-mass # kinetic energy defined has KE = E - m^2 = sqrt(p^2+m^2)-m^2
            xy_norm = momentum[first_point_index,0]**2 + momentum[first_point_index,1]**2
            theta = np.arctan2(np.sqrt(xy_norm),momentum[first_point_index,2])*180/np.pi
            phi = (np.arctan2(momentum[first_point_index,1],momentum[first_point_index,0])*180/np.pi)%360 # angle between 0° and 360°
            
            ke_exit = np.sqrt(np.sum(momentum[exiting_point_index]**2)+mass**2)-mass
            xy_norm_exit = momentum[exiting_point_index,0]**2 + momentum[exiting_point_index,1]**2
            theta_exit = np.arctan2(np.sqrt(xy_norm_exit),momentum[exiting_point_index,2])*180/np.pi
            phi_exit = (np.arctan2(momentum[exiting_point_index,1],momentum[exiting_point_index,0])*180/np.pi)%360
            
            ke_exit_reduce = np.sqrt(np.sum(momentum[exiting_reduce_point_index]**2)+mass**2)-mass
            xy_norm_exit_reduce = momentum[exiting_reduce_point_index,0]**2 + momentum[exiting_reduce_point_index,1]**2
            theta_exit_reduce = np.arctan2(np.sqrt(xy_norm_exit_reduce),momentum[exiting_reduce_point_index,2])*180/np.pi
            phi_exit_reduce = (np.arctan2(momentum[exiting_reduce_point_index,1],momentum[exiting_reduce_point_index,0])*180/np.pi)%360
            
            ## Assert that all energies are non zero
            if ke == 0 or ke_exit ==0 or ke_exit_reduce ==0:
                if verbose:
                    print(f"ERROR event {entry}: Null energy : ke: {ke:.2e}  ke_exit:{ke_exit:.2e}  ke_exit_reduce:{ke_exit_reduce:.2e} ")
                failed_events[3] += 1
                continue
            
            
                    
            ####### Hits section #######

            ## Extract the data from the SFG detector
            xyz_pos = []  # 3D coordinates of the hits
            features = []  # features of each hit for our model (charge, tag)

            ## Get the correct hits to be kept with sfg_tree.AlgoResults[0].Hits
            indices = sfg_tree.AlgoResults[0].Hits

            ## Loops over all hits in SFG to be kept
            for index in indices:

                ## Get the hit
                hit = sfg_tree.Hits[index]

                xyz_pos.append(np.asarray(hit.Position))

                features.append(
                    [
                        hit.Charge,
                        hit.TrkHitTag,
                    ]
                )
            
            
            ## Convert to numpy arrays
            xyz_pos = np.array(xyz_pos)
            features = np.array(features)
            sparse_image = np.concatenate([xyz_pos,features],axis=-1)
            
            
            L1_distance_to_the_origin = np.max(np.abs(xyz_pos-origin_point), axis=-1) # L1 distance, that is used to see if the a hit is adjacent to the VA region
            
            adjacent_cube = ((VA_region_full_size/2-eps <= L1_distance_to_the_origin)*(L1_distance_to_the_origin <= VA_region_full_size/2+cube_size+eps)).any()
            # adjacent_cube_reduce = (VA_region_reduced_size/2-eps <= L1_distance_to_the_origin <= VA_region_reduced_size/2+1+eps).any()
            
            adjacent_cube_s += int(adjacent_cube)
            
            event_index = entry
            file_index = 0
            
            # gather statistics
            charges.extend(sparse_image[:,3])
            ini_pos[0].append(pos_ini[0])
            ini_pos[1].append(pos_ini[1])
            ini_pos[2].append(pos_ini[2])
            kes.append(ke)
            thetas.append(theta)
            phis.append(phi)
            
            # Retrieve the bin of each value
            bin_indices = np.digitize(pos_ini-origin_point, bin_edges, right=False)[0]
            bin_indices = tuple(bin_indices)
            
            if bin_indices not in lookup_table:
                lookup_table[bin_indices] = [entry]
            else:
                lookup_table[bin_indices].append(entry)
                
            
            
            ####### Saving data #######
            

            ## Save the event to a file
            np.savez_compressed(
                f"{output_dir}/event_{entry}_{particle_name}",
                sparse_image = sparse_image,
                pos_ini = pos_ini,
                pos_fin = pos_fin,
                pdg = input_pdg_dict[particle_name],
                ke = ke,
                theta = theta,
                phi = phi,
                exit = exit_,
                adjacent_cube = adjacent_cube,
                ke_exit = ke_exit,
                theta_exit = theta_exit,
                phi_exit = phi_exit,
                pos_exit = pos_exit,
                ke_exit_reduce = ke_exit_reduce,
                theta_exit_reduce = theta_exit_reduce,
                phi_exit_reduce = phi_exit_reduce,
                pos_exit_reduce = pos_exit_reduce,
                file_index = file_index,
                event_index = event_index,
            )
        
        
        if entry%100 == 99:
            progess_bar.set_postfix({"Failed events":f"{np.sum(failed_events)-last_nb_failed_events}%", "Exits":f"{exit_s}%", "Adj":f"{adjacent_cube_s}%"})
            total_exit_s+=exit_s
            total_adjacent_cube_s+=adjacent_cube_s
            exit_s = 0
            adjacent_cube_s = 0
            last_nb_failed_events = np.sum(failed_events)
            
            
    charges = np.array(charges)
    ini_pos = np.array(ini_pos)
    kes = np.array(kes)
    thetas = np.array(thetas)
    phis = np.array(phis)
    
    with open(f"{output_dir}/data_stats_{particle_name}.p", "wb") as fd:
        pk.dump([charges, ini_pos, kes, thetas, phis, lookup_table, bin_edges], fd)

    print(
        f"Failed events due to wrong number of true trajectories in SFG: \t {failed_events[0]:>8} | {100*failed_events[0]/sfg_tree.GetEntries():.2f}%"
    )
    print(
        f"Failed events due to empty Hit array: \t\t\t\t {failed_events[1]:>8} | {100*failed_events[1]/sfg_tree.GetEntries():.2f}%"
    )
    print(
        f"Failed events due to wrong number of trajectory points: \t {failed_events[2]:>8} | {100*failed_events[2]/sfg_tree.GetEntries():.2f}%"
    )
    print(
        f"Failed events due to some null kinetic energy: \t\t\t {failed_events[3]:>8} | {100*failed_events[3]/sfg_tree.GetEntries():.2f}%"
    )
    print(
        f"Failed events due to wrong trajectory PDG: \t\t\t {failed_events[4]:>8} | {100*failed_events[4]/sfg_tree.GetEntries():.2f}%"
    )
    # print(f"Failed events due to empty reconstructed Vertices array: \t {failed_events[2]:>8} | {100*failed_events[2]/sfg_tree.GetEntries():.2f}%")
    print(
        f"Total failed events: \t\t\t\t\t\t {sum(failed_events):>8} | {100*sum(failed_events)/sfg_tree.GetEntries():.2f}%"
    )
    print()
    print(f"Events with exit: \t\t {total_exit_s:>8} | {100*total_exit_s/sfg_tree.GetEntries():.2f}%")
    print(f"Events with adjacency: \t\t {total_adjacent_cube_s:>8} | {100*total_adjacent_cube_s/sfg_tree.GetEntries():.2f}%")
    