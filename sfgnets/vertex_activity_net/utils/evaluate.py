import numpy as np
import torch
import tqdm

from ...datasets.constants import CUBE_SIZE, ORIGIN
from .data_functions import rescale
from .mask_generation import create_mask, create_mask_src, create_mask_tgt
from ..models.decomposing_transformer.transformer import VATransformer



def eval_event(event_n:int, 
               model, 
               test_set, 
               device:str|torch.device="cpu", 
               use_truth:bool=False):
    """
    Evaluate an event using the given transformer model and test set (configuration 2).

    Args:
        event_n (int): The index of the event to be evaluated.
        model: The transformer neural network model for event evaluation.
        test_set: The dataset containing the events for evaluation.
        device (str, optional): The device to run the evaluation on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        tuple: A tuple containing the following elements:
            - particle_images (numpy.ndarray): Rescaled particle images.
            - sfgd_images (list[numpy.ndarray]): Images from the detector for event display.
            - muon_exit (numpy.ndarray): Rescaled muon exit parameters.
            - kin_true (numpy.ndarray): True event kinematic parameters.
            - kin_pred (numpy.ndarray): Predicted event kinematic parameters.
            - pid_true (numpy.ndarray): True event particle types.
            - pid_pred (numpy.ndarray): Predicted event particle types.
            - vtx_true (float): True vertex position.
            - vtx_pred (float): Predicted vertex position.
    """
    # Evaluation mode
    model = model.to(device)
    model.eval()

    # Retrieve event
    event = test_set[event_n]
    muon_exit, kin_true, vtx_true = event['exit_muon'], event['params'], event['ini_pos']
    sfgd_images = event['sfgd_images']
    event = test_set.collate_fn([event])
    
    if use_truth:
        # Unpack values from the batch
        hits, exit_muon, vtx_true, params_true, keep_iter_true, pid_true, _, particle_images = event

        # Slice 'params_true' to exclude the last item of each sequence
        params_true_input = params_true[:-1, :]
        pid_true_input = pid_true[:-1, :]

        # Create masks for source and target sequences
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(hits, params_true_input, test_set.pad_value,
                                                                             device)

        # Pass the inputs and masks to the model to get predictions
        vtx_pred, params_pred, pid_pred, keep_iter_pred = model(hits.to(device), exit_muon.to(device), params_true_input.to(device),
                                                                     pid_true_input.to(device), src_mask.to(device), tgt_mask.to(device),
                                                                     src_padding_mask.to(device), tgt_padding_mask.to(device))
        
        keep_iter_pred = keep_iter_pred[:,0,:].argmax(axis=1)
        first_index_to_stop = ((keep_iter_pred == 0)*torch.arange(len(keep_iter_pred)).to(device)).argmin()+1
        
        
        # kin_true = params_true
        kin_pred = params_pred[:first_index_to_stop]
        pid_pred = pid_pred[:first_index_to_stop].argmax(dim=-1)
        
        
    
    else:
        src, muon, vtx_true, tgt, next_tgt, pid_true, _, particle_images = event
        src = src.to(device)
        muon = muon.to(device)

        # Run encoder
        memory, vtx_pred = model.encode(src, muon, None, None)
        memory = memory.to(device)

        # Starting token
        kin_pred = model.first_token_kin.reshape(1, 1, 3)
        pid_pred = model.first_token_pid.reshape(1, -1)

        # Iterate over the decoder
        # max_len = len(tgt) # this limits the number of particles predicted to the current number of true particles => bias
        max_len = sum(test_set.max_add_part.values())
        for i in range(max_len):
            kin_out, pid_out, keep_iterating_out = model.decode(kin_pred, pid_pred, memory, None, None, None)
            kin_last = kin_out[-1].reshape(1, kin_out.shape[1], kin_out.shape[2])
            pid_last = pid_out[-1].reshape(1, pid_out.shape[1], pid_out.shape[2])
            pid_last = torch.nn.functional.softmax(pid_last.detach(), dim=2).argmax(dim=2)
            keep_iterating_last = keep_iterating_out[-1, 0]

            # Increase size of prediction
            kin_pred = torch.cat([kin_pred, kin_last], dim=0)
            pid_pred = torch.cat([pid_pred, pid_last], dim=0)

            if keep_iterating_last.argmax(0) == 0:
                break
    
    # Retrieve output
    pid_true = pid_true.cpu().numpy()[:, 0] # no need to remove the muon, already done in the collate function
    pid_pred = pid_pred.detach().cpu().numpy()[1:, 0]
    kin_true = kin_true[1:, :]
    kin_pred = kin_pred.detach().cpu().numpy()[1:, 0, :]
    vtx_true = vtx_true.cpu().numpy()
    vtx_pred = vtx_pred.detach().cpu().numpy()
    particle_images = particle_images[0]
    
    

    # Rescale to original values
    rescale(test_set, particle_images, muon_exit, kin_true, kin_pred, vtx_true, vtx_pred)

    return particle_images, sfgd_images, muon_exit, kin_true, kin_pred, pid_true, pid_pred, vtx_true[0], vtx_pred[0]



def eval_batch(batch, model, test_set, smearing=False, device="cpu"):
    model.eval()
    PAD_IDX=test_set.pad_value
   
    src, muon, vtx, tgt, next_tgt, pid, _, _ = batch
    tgt_input = tgt[:-1, :]
    pid_input = pid[:-1, :]
    
    src = src.clone()
    src_mask, src_padding_mask = create_mask_src(src, PAD_IDX, device)
    
    if smearing:
        
        # interpolate back
        exit_pos_muon = np.interp(muon[:,:3].ravel(), test_set.source_range, 
                       (test_set.min_exit_pos_muon, test_set.max_exit_pos_muon)).reshape(muon[:,:3].shape)
        KE_muon = np.interp(muon[:,3].ravel(), test_set.source_range, 
                           (test_set.min_KE_muon, test_set.max_KE_muon)).reshape(muon[:,3].shape)
        theta_muon = np.interp(muon[:,4].ravel(), test_set.source_range, 
                           (test_set.min_theta, test_set.max_theta)).reshape(muon[:,4].shape)
        phi_muon = np.interp(muon[:,5].ravel(), test_set.source_range, 
                           (test_set.min_phi, test_set.max_phi)).reshape(muon[:,5].shape)
        
        # smearing values
        smear_pos = np.random.normal(loc=0.0, scale=1, size=exit_pos_muon.shape) # +-3 mm
        smear_ke = np.random.normal(loc=0.0, scale=3, size=KE_muon.shape) # +-5 MeV
        smear_theta = np.random.normal(loc=0.0, scale=1, size=theta_muon.shape) # +-5 degrees
        smear_phi = np.random.normal(loc=0.0, scale=2, size=theta_muon.shape) # +-10 degrees
        
        # smear muon info
        exit_pos_muon = np.clip(exit_pos_muon+smear_pos, test_set.min_exit_pos_muon, test_set.max_exit_pos_muon) 
        KE_muon = np.clip(KE_muon+smear_ke, test_set.min_KE_muon, test_set.max_KE_muon) 
        theta_muon = np.clip(theta_muon+smear_theta, test_set.min_theta, test_set.max_theta) 
        phi_muon = np.clip(phi_muon+smear_phi, test_set.min_phi, test_set.max_phi) 
        
        # interpolate
        exit_pos_muon = np.interp(exit_pos_muon.ravel(), (test_set.min_exit_pos_muon, test_set.max_exit_pos_muon),
                                 test_set.source_range).reshape(muon[:,:3].shape)
        exit_pos_muon /= np.abs(exit_pos_muon).max(axis=1).reshape(-1,1)  # make sure the exiting position touches the volume
        KE_muon = np.interp(KE_muon.ravel(), (test_set.min_KE_muon, test_set.max_KE_muon),
                            test_set.source_range).reshape(muon[:,3].shape)
        theta_muon = np.interp(theta_muon.ravel(), (test_set.min_theta, test_set.max_theta),
                               test_set.source_range).reshape(muon[:,4].shape)
        phi_muon = np.interp(phi_muon.ravel(), (test_set.min_phi, test_set.max_phi),
                             test_set.source_range).reshape(muon[:,5].shape)
        
        # retrieve original values
        muon[:,:3] = torch.tensor(exit_pos_muon)
        muon[:,3] = torch.tensor(KE_muon)
        muon[:,4] = torch.tensor(theta_muon)
        muon[:,5] = torch.tensor(phi_muon)

    src = src.to(device)
    muon = muon.to(device)
    src_mask = src_mask.to(device)
    src_padding_mask = src_padding_mask.to(device)
 
    memory, vtx_pred = model.encode(src, muon, src_mask, src_padding_mask)
    memory = memory.to(device)
    
    max_len = len(tgt)
    
    ys_kin = torch.zeros(max_len+1, tgt_input.shape[1], 3).fill_(PAD_IDX).type(torch.float).to(device)
    ys_pid = torch.zeros(max_len+1, tgt_input.shape[1]).fill_(0).type(torch.long).to(device)
    ys_pid_prob = torch.zeros(max_len, tgt_input.shape[1], 3).fill_(PAD_IDX).type(torch.float)
    
    #ys_first = tgt[0]
    ys_kin_first = model.first_token_kin.repeat(1, tgt_input.shape[1], 1)
    ys_pid_first = model.first_token_pid.repeat(1, tgt_input.shape[1])
    ys_kin[0, :, :] = ys_kin_first
    ys_pid[0, :] = ys_pid_first
    
    # keep track of predictions that finished (none before starting)
    prev_info = torch.ones(size=(src.shape[1],)).bool().to(device)
    
    del src, src_mask
    
    for i in range(max_len):
        # create masks and run model
        tgt_mask, tgt_padding_mask = create_mask_tgt(ys_kin[:i+1], PAD_IDX, device)
        out, pid_pred, is_next = model.decode(ys_kin[:i+1], ys_pid[:i+1], memory, 
                                              tgt_mask, tgt_padding_mask, src_padding_mask)
        
        # reshape output
        out_last = out[-1].reshape(1, out.shape[1], out.shape[2])
        pid_last = pid_pred[-1].reshape(1, pid_pred.shape[1], pid_pred.shape[2])
        pid_last = torch.nn.functional.softmax(pid_last.detach(), dim=2) # probabilities
        is_next_last = is_next[-1].argmax(1).bool()
        
        # only update results of predictions that haven't finished
        ys_kin[i+1, prev_info, :] = out_last.detach()[:,prev_info,:]
        ys_pid[i+1, prev_info] = pid_last.argmax(dim=2)[:,prev_info]
        ys_pid_prob[i, prev_info] = pid_last[:, prev_info, :]
        
        # update the information with predictions that just finished
        prev_info = torch.logical_and(prev_info, is_next_last) 
    
    return ys_kin[1:], ys_pid_prob, vtx_pred



def approximation_formula_va_energy(total_charge:np.ndarray|float,
                                    trajectory_length_longest_proton:np.ndarray|float,
                                    calibration_coeff:float=67.,
                                    birks_coeff:float=0.126) -> np.ndarray|float :
    """
    Estimate the vertex activity region energy using the approximation formula

    Args:
        total_charge (np.ndarray|float): total hit charge (number of photo electrons) summed over the vertex activity hits (all but muons hits)
        trajectory_length_longest_proton (np.ndarray|float): length of the trajectory of the proton travelling the furthest, in mm
        calibration_coeff (float): calibration coefficient of the detector, in photo electrons per MeV
        birks_coeff (float): Birks coefficient of the detector, in mm/MeV

    Returns:
        approximated_vertex_activity_energy (np.ndarray|float): approximated energy of the vertex activity region
    """
    
    return (total_charge/calibration_coeff)/np.clip((1-birks_coeff*(total_charge/calibration_coeff)/trajectory_length_longest_proton),1e-2,1e3)



def approximation_formula_va_energy_standard(hits, threshold = 100, correction = False):
    c_b = 0.0126
    # c_cali = 100
    c_cali = 85.82
    min_length = 0.5
    max_length = 4.5*np.sqrt(3)*CUBE_SIZE*2
    
    hits_aux = hits.copy()
    # hits_aux[:,:3]= (hits_aux[:,:3]-4)*CUBE_SIZE*2
    hits_aux[:,:3]-=ORIGIN

    mask = (hits_aux[:, 3] >= threshold)
    
    if mask.any():
        # Calculate the Euclidean distances
        distances = np.linalg.norm(hits_aux[mask][:,:3], axis=1)
        longest_proton = distances.max()
        
        # Avoid unphysical distances (notably division by 0)
        longest_proton = np.clip(longest_proton,min_length,max_length)
            
    else:
        longest_proton = min_length
    
    distances = np.linalg.norm(hits_aux[:,:3], axis=1)
    
    if (distances  <= longest_proton).any():
        E = hits_aux[distances <= longest_proton][:,3].sum() / c_cali
    else:
        E = 0.
        
    E_reco = (1 / (1 - c_b*(E/longest_proton))) * E
    
    if correction:
        E_reco = 0.869*E_reco-26.25
    
    
    return E_reco, E, longest_proton


def analyze_event(model,
                  test_set,
                  event_n:int,
                  device:str|torch.device='cuda',
                  use_truth:bool=False,
                  use_formula_correction:bool=False):
    
    # Run the decomposing transformer and get particle information
    images, sfgd_image, exit_muon, params_true, params_pred, pids_true, pids_pred, vtx_true, vtx_pred = \
                            eval_event(event_n, model, test_set, device=device, use_truth=use_truth)
                            
    event = test_set[event_n]
                            
    # Extract muon exit information
    exit_x, exit_y, exit_z, exit_ke, exit_theta, exit_phi = exit_muon
    
    ke, theta, phi = params_true[:,0], params_true[:,1], params_true[:,2] 
    nb_part = len(params_true[:])
    nb_pred = len(params_pred)
    
    min_nb = min(nb_part,nb_pred)
    
    ke_pred, theta_pred, phi_pred = params_pred[:,0], params_pred[:,1], params_pred[:,2]
    
    ke_diff = ke[:min_nb]-ke_pred[:min_nb]
    
    deg_to_rad = np.pi/180.
    
    true_direction = np.zeros((nb_part,3))
    true_direction[:,0] = np.sin(theta*deg_to_rad)*np.cos(phi*deg_to_rad)
    true_direction[:,1] = np.sin(theta*deg_to_rad)*np.sin(phi*deg_to_rad)
    true_direction[:,2] = np.cos(theta*deg_to_rad)
    
    pred_direction = np.zeros((nb_pred,3))
    pred_direction[:,0] = np.sin(theta_pred*deg_to_rad)*np.cos(phi_pred*deg_to_rad)
    pred_direction[:,1] = np.sin(theta_pred*deg_to_rad)*np.sin(phi_pred*deg_to_rad)
    pred_direction[:,2] = np.cos(theta_pred*deg_to_rad)
    
    angle_distance = np.arccos(np.sum(true_direction[:min_nb]*pred_direction[:min_nb],axis=-1))
    angle_distance /= deg_to_rad
    
    vtx_distance = np.linalg.norm(vtx_true-vtx_pred)
    
    pid_accuracy = (pids_true[:min_nb] == pids_pred[:min_nb])
    
    ## Previous function for estimating the vertex activity energy with the formula
    # if images[1:].any():
    #     trajectory_length_longest = np.max(np.linalg.norm(np.array(np.where(np.sum(images[1:],axis=0)))-4,axis=0))*CUBE_SIZE*2.
    #     trajectory_length_longest = np.clip(trajectory_length_longest, 1e-1, 1e3) # avoid division by zero
    # else:
    #     trajectory_length_longest = 1000.
    # total_charge = np.sum(images[1:])
    # approximated_vertex_activity_energy = approximation_formula_va_energy(total_charge,trajectory_length_longest)
    
    approximated_vertex_activity_energy, E_cal, longest_proton = approximation_formula_va_energy_standard(np.vstack(sfgd_image[1:]),
                                                                                   correction=use_formula_correction)
    
    
    
    return {
            ## True information
            "ke":ke,
            "pid":pids_true,
            "muon_ke":exit_ke,
            "nb_part":nb_part,
            "lengths":np.array(event["lengths"][1:]), # remove the muon (first index)
            "true_theta":theta,
            "true_phi":phi,
            
            ## Performances of the model
            "ke_diff":ke_diff,
            "total_ke_diff":np.sum(ke)-np.sum(ke_pred),
            "angle_distance":angle_distance,
            "vtx_distance":vtx_distance,
            "pid_accuracy":pid_accuracy,
            "nb_pred":nb_pred,
            "pred_theta":theta_pred,
            "pred_phi":phi_pred,
            
            "min_nb":min_nb,
            "formula_total_ke":approximated_vertex_activity_energy,
            
            "E_cal":E_cal,
            "longest_proton":longest_proton,
            }



def analyze_testset(model,
                    test_set,
                    N_max:int|None=None,
                    device:str|torch.device='cuda',
                    use_truth:bool=False,
                    use_formula_correction:bool=False):
    
    if N_max is None:
        N_max = len(test_set)
    N_max = min(N_max,len(test_set))
    
    per_event_analysis = {
                            ## True information
                             "muon_ke":[],   
                             "nb_part":[],
                             "nb_protons":[],
                             "nb_neutrons":[],
                             "total_ke":[],
                             "ke_protons":[],
                             "ke_neutrons":[],
                             
                             ## Analysis information
                             "total_ke_diff":[],
                             "total_ke_res_%":[],
                             "total_ke_abs_res_%":[],
                             "vtx_distance":[],
                             "nb_pred":[],
                             "nb_part_diff":[],
                             "nb_part_abs_diff":[],
                             "pid_accuracy":[],
                             "ke_diff_avg":[],
                             "angle_distance_avg":[],
                             "formula_total_ke":[],
                             "formula_total_ke_diff":[],
                             "formula_total_ke_res_%":[],
                             "formula_total_ke_abs_res_%":[],
                             
                             "E_cal":[],
                             "longest_proton":[],

                            }
    
    
    
    per_particle_analysis = {
                             ## True information
                             "pid":[],
                             "ke":[],
                             "muon_ke":[],   
                             "nb_part":[],
                             "nb_protons":[],
                             "nb_neutrons":[],
                             "total_ke":[],
                             "total_ke_diff":[],
                             "part_length":[],
                             "true_theta":[],
                             "true_phi":[],
                             
                             ## Analysis information
                             "ke_diff":[],
                             "ke_res_%":[],
                             "ke_abs_diff":[],
                             "angle_distance":[],
                             "pred_theta":[],
                             "pred_phi":[],
                             "diff_theta":[],
                             "diff_phi":[],
                             "pid_accuracy":[],
                             "nb_pred":[],
                             "vtx_distance":[],
                             }
    
    
    
    for event_n in tqdm.tqdm(range(N_max), desc="Analyze test set"):
        
        analysis_event = analyze_event(model=model,
                       test_set=test_set,
                       event_n=event_n,
                       device=device,
                       use_truth=use_truth,
                       use_formula_correction=use_formula_correction)
        
        ## True information per event
        per_event_analysis["muon_ke"].append(analysis_event["muon_ke"])
        per_event_analysis["nb_part"].append(analysis_event["nb_part"])
        per_event_analysis["nb_protons"].append(np.sum(analysis_event["pid"]==0))
        per_event_analysis["nb_neutrons"].append(np.sum(analysis_event["pid"]==1))
        per_event_analysis["total_ke"].append(np.sum(analysis_event["ke"]))
        per_event_analysis["ke_protons"].append(np.sum(analysis_event["ke"]*(analysis_event["pid"]==0)))
        per_event_analysis["ke_neutrons"].append(np.sum(analysis_event["ke"]*(analysis_event["pid"]==1)))
        
        ## Performances per event
        per_event_analysis["total_ke_diff"].append(analysis_event["total_ke_diff"])
        per_event_analysis["total_ke_res_%"].append(100*analysis_event["total_ke_diff"]/np.clip(np.sum(analysis_event["ke"]),1.,None))
        per_event_analysis["total_ke_abs_res_%"].append(np.abs(per_event_analysis["total_ke_res_%"][-1]))
        per_event_analysis["nb_pred"].append(analysis_event["nb_pred"])
        per_event_analysis["nb_part_diff"].append(analysis_event["nb_part"]-analysis_event["nb_pred"])
        per_event_analysis["nb_part_abs_diff"].append(np.abs(analysis_event["nb_part"]-analysis_event["nb_pred"]))
        per_event_analysis["pid_accuracy"].append(np.mean(analysis_event["pid_accuracy"]))
        per_event_analysis["vtx_distance"].append(analysis_event["vtx_distance"])
        per_event_analysis["ke_diff_avg"].append(np.mean(np.abs(analysis_event["ke_diff"])))
        per_event_analysis["angle_distance_avg"].append(np.mean(analysis_event["angle_distance"]))
        per_event_analysis["formula_total_ke"].append(analysis_event["formula_total_ke"])
        per_event_analysis["formula_total_ke_diff"].append(per_event_analysis["total_ke"][-1]-analysis_event["formula_total_ke"])
        per_event_analysis["formula_total_ke_res_%"].append(100*(per_event_analysis["total_ke"][-1]-analysis_event["formula_total_ke"])/np.clip(per_event_analysis["total_ke"][-1],1.,None))
        per_event_analysis["formula_total_ke_abs_res_%"].append(np.abs(per_event_analysis["formula_total_ke_res_%"][-1]))
        per_event_analysis["E_cal"].append(analysis_event["E_cal"])
        per_event_analysis["longest_proton"].append(analysis_event["longest_proton"])
        
        ## True information per particle
        min_nb = analysis_event["min_nb"]
        per_particle_analysis["muon_ke"].extend([analysis_event["muon_ke"] for k in range(min_nb)])
        per_particle_analysis["nb_part"].extend([analysis_event["nb_part"] for k in range(min_nb)])
        per_particle_analysis["nb_protons"].extend([per_event_analysis["nb_protons"][-1] for k in range(min_nb)])
        per_particle_analysis["nb_neutrons"].extend([per_event_analysis["nb_neutrons"][-1] for k in range(min_nb)])
        per_particle_analysis["total_ke"].extend([per_event_analysis["total_ke"][-1] for k in range(min_nb)])
        per_particle_analysis["total_ke_diff"].extend([per_event_analysis["total_ke_diff"][-1] for k in range(min_nb)])
        per_particle_analysis["ke"].extend(list(analysis_event["ke"][:min_nb]))
        per_particle_analysis["pid"].extend(list(analysis_event["pid"][:min_nb]))
        per_particle_analysis["part_length"].extend(list(analysis_event["lengths"][:min_nb]))
        per_particle_analysis["true_theta"].extend(list(analysis_event["true_theta"][:min_nb]))
        per_particle_analysis["true_phi"].extend(list(analysis_event["true_phi"][:min_nb]))
        
        ## Performances per particle
        per_particle_analysis["nb_pred"].extend([analysis_event["nb_pred"] for k in range(min_nb)])
        per_particle_analysis["pid_accuracy"].extend(list(analysis_event["pid_accuracy"]))
        per_particle_analysis["vtx_distance"].extend([analysis_event["vtx_distance"] for k in range(min_nb)])
        per_particle_analysis["ke_diff"].extend(list(analysis_event["ke_diff"][:min_nb]))
        per_particle_analysis["ke_res_%"].extend(list(100*analysis_event["ke_diff"][:min_nb]/np.clip(analysis_event["ke"][:min_nb],1.,None)))
        per_particle_analysis["ke_abs_diff"].extend(list(np.abs(analysis_event["ke_diff"][:min_nb])))
        per_particle_analysis["angle_distance"].extend(list(analysis_event["angle_distance"]))
        per_particle_analysis["pred_theta"].extend(list(analysis_event["pred_theta"][:min_nb]))
        per_particle_analysis["pred_phi"].extend(list(analysis_event["pred_phi"][:min_nb]))
        per_particle_analysis["diff_theta"].extend(list(analysis_event["true_theta"][:min_nb]-analysis_event["pred_theta"][:min_nb]))
        per_particle_analysis["diff_phi"].extend(list(analysis_event["true_phi"][:min_nb]-analysis_event["pred_phi"][:min_nb]))
        
        
    for dict_ in [per_event_analysis,per_particle_analysis]:
        for key in dict_.keys():
            dict_[key] = np.array(dict_[key])
    
    return per_event_analysis,per_particle_analysis
    
        