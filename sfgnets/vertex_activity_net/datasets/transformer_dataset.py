"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch dataset that dynamically generates vertex-activity
             images depicting the overlap of particles, including one muon,
             0-4 protons, 0-1 deuterium, and 0-1 tritium, all originating from
             a common starting point within the detector.
"""

import numpy as np
import pickle as pk
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from ..utils import set_random_seed, shift_image, shift_particle, fix_exit_shift, fix_empty_particles



class TransformerDataset(Dataset):

    def __init__(self, config: dict, split: str = "train"):
        """
        Dataset initialiser.

        Args:
            config (dict): Dictionary with JSON configuration entries.
            split (str): String indicating the purpose of the dataset ("train", "val", "test").

        Returns:
            None
        """
        
        self.mu_particles = config["mu_particles"]
        self.additional_particles = config["additional_particles"]
        self.particles = self.mu_particles + self.additional_particles
        
        charges = {}
        lookup_table = {}
        bin_edges = {}
        lookups = {}
        
        for part in self.particles:
            with open(config["dataset_metadata"].format(part), "rb") as fd:
                charges_p, _, _, _, _, lookup_table_p, bin_edges_p = pk.load(fd)
            charges[part] = charges_p
            lookup_table[part] = lookup_table_p
            bin_edges[part] = bin_edges_p
            lookups[part] = set(lookup_table_p.keys())
        
        # Common indices (minicube indexes) of all particles
        self.indices = list(set(lookup_table_p.keys()).intersection(*lookups.values()))  # indices (keys) of the lookup tables

        # # Make sure the lookup tables are the same for all particles
        # assert all(s == lookups[0] for s in lookups)

        self.dataset = config["dataset"]  # dataset path
        self.pad_value = config["pad_value"]  # padding value
        self.cube_size = config["cube_size"]  # cube (voxel) size in mm (one side)
        self.min_theta = config["min_theta"]  # min theta (in degrees)
        self.max_theta = config["max_theta"]  # max theta (in degrees)
        self.min_phi = config["min_phi"]  # min phi (in degrees)
        self.max_phi = config["max_phi"]  # max phi (in degrees)
        self.img_size = config["img_size"]  # img_size x img_size x img_size
        self.origin_point = np.array(config["origin_point"]) # origin point (to center the events)
        self.PID_FROM_PARTICLE = config["PID_FROM_PARTICLE"] # dictionary to map the particles names to an ID
        
        self.max_add_part = {} # maximum number of additional particles per event
        for part in self.additional_particles:
            self.max_add_part[part] = config[f"max_{part}"]
            
        self.min_charge = config["min_charge"]  # min charge (energy loss) per cube
        self.max_charge = max([charge_.max() for charge_ in charges.values()])  # max charge per cube
        
        self.min_ini_pos = -self.cube_size - self.cube_size / 2.  # min initial particle 1D position
        self.max_ini_pos = self.cube_size + self.cube_size / 2.  # max initial particle 1D position
        
        self.min_ke = {}
        self.max_ke = {}
        for part in self.particles:
            self.min_ke[part] = config[f"min_ke_{part}"]  # min initial particle kinetic energy
            self.max_ke[part] = config[f"max_ke_{part}"]  # max initial particle kinetic energy
            
        self.cube_shift = config["cube_shift"]  # random shift (in cubes) of the particle position
        self.min_exit_pos_mu = -(self.cube_size * self.img_size) / 2.  # min muon exiting 1D position
        self.max_exit_pos_mu = (self.cube_size * self.img_size) / 2.  # max muon exiting 1D position
        
        self.lookup_table = lookup_table
        self.bin_edges = bin_edges  # bin edges for lookup table
        
        self.source_range = config["source_range"]  # range for input values
        self.target_range = config["target_range"]  # range for target values
        
        
        self.total_events = len(self.indices)  # total number of different input particle positions
        self.split = split  # "train", "val", or "test"
        set_random_seed(config["random_seed"], random=random, numpy=np)  # for reproducibility

        # Number of particles in each event of the validation and test sets
        self.add_part_val = {}
        self.add_part_test = {}
        for part in self.additional_particles:
            self.add_part_val[part] = np.random.randint(0, self.max_add_part[part] + 1, self.total_events)
            self.add_part_test[part] = np.random.randint(0, self.max_add_part[part] + 1, self.total_events)

        # check if val/test event events have 0 particles and fix (add 1 particle to those events)
        fix_empty_particles(list(self.add_part_val.values()), np.random)
        fix_empty_particles(list(self.add_part_test.values()), np.random)

        # Shuffle all the lists (particles starting from the same position (minicube)) in the dictionary
        for lookup_table_p in self.lookup_table.values():
            for key in lookup_table_p:
                random.shuffle(lookup_table_p[key])
                

    def __len__(self) -> int:
        """
        Returns the total number of events in the dataset.

        Returns:
            int: Total number of events in the dataset.
        """
        return self.total_events



    def __getitem__(self, idx:int) -> dict:
        """
        Construct on-the-fly events with 1 muon, 0 to max_add_part any additional particle

        Args:
            idx (int): Dataset index (from 0 to the length of the lookup tables).

        Returns:
            event (dict): Dictionary with: 
                                            (1) the images of each particle, 
                                            (2) the initial positions of each particle, 
                                            (3) kinematic parameters of each particle, 
                                            (4) type of each particle, 
                                            (5) exiting muon information, 
                                            (6) length of each particle.
        """
        # Get particle candidates from index (particles starting from the same position)
        index = self.indices[idx]
        candidates = {}
        for part in self.particles:
            candidates[part] = self.lookup_table[part][index]
            
        for part in self.mu_particles:
            if self.split == "train":
                # Removes the candidates for testing and validation purposes
                candidates[part] = candidates[part][:-2]
            elif self.split == "val":
                # Keeps the second to last candidate for validation purposes
                candidates[part] = candidates[part][-2:-1]
            else:
                # Keeps the last candidate for test purposes
                candidates[part] = candidates[part][-1:]
                
        for part in self.additional_particles:
            if self.split == "train":
                # Removes the candidates for testing and validation purposes
                candidates[part] = candidates[part][:len(candidates[part]) - (self.add_part_val[part][idx] + self.add_part_test[part][idx])]
            elif self.split == "val":
                # Keeps some candidates for validation purposes
                candidates[part] = candidates[part][len(candidates[part]) - (self.add_part_val[part][idx] + self.add_part_test[part][idx]):len(candidates[part]) - self.add_part_test[part][idx]]
            else:
                # Keeps the last candidates for test purposes
                candidates[part] = candidates[part][len(candidates[part]) - self.add_part_test[part][idx]:]
                

        if self.split == "train":
            # Candidates per particle. Make sure there's at least one particle
            nb = {mu:0 for mu in self.mu_particles}
            for part in self.additional_particles:
                nb[part] = np.random.randint(0, min(self.max_add_part[part] + 1, len(candidates[part])))
            # Selectes the muon type
            mu_selected = np.random.randint(0, len(self.mu_particles))
            nb[self.mu_particles[mu_selected]] = 1
                
            if sum(nb.values()) == 1:
                rand_labels = np.random.randint(0, len(self.additional_particles))
                nb[self.additional_particles[rand_labels]] += 1

            for part in self.particles:
                candidates[part] = random.sample(candidates[part], nb[part])
                
        else:
            set_random_seed(idx, random=random, numpy=np)  # for reproducibility

        # Retrieve the particle candidates
        particles, parts, pids = [], [], []
        for part in self.particles:
            for cand_id in candidates[part]:
                filepath = self.dataset+f"event_{cand_id}_{part}.npz"
                loaded_cand = np.load(filepath)  # load particle
                particles.append(loaded_cand)
                parts.append(part)

        # Random shift (same for all the particles)
        shift_x, shift_y, shift_z = np.random.randint(-self.cube_shift, self.cube_shift + 1, 3)

        # Prepare event
        images, params, lens, muon_exit = [], [], [], []
        for i, particle in enumerate(particles):
            ## Shift the coordinates to have the center point at 0 0 0
            sparse_image = particle['sparse_image']
            sparse_image = sparse_image[~((sparse_image[:,:3]<self.cube_size*(self.img_size + 2)).prod(axis=-1))]
            sparse_image = sparse_image[~((-sparse_image[:,:3]<self.cube_size*(self.img_size + 2)).prod(axis=-1))]
            sparse_image[:,:3] -= self.origin_point[None,:]
            
            hits = np.round(sparse_image).astype(int)  # array of shape (Nx5) [points vs (x, y, z, charge, tag)]
            pos_ini = particle['pos_ini'] - self.origin_point  # particle initial 3D position
            pos_fin = particle['pos_fin'] - self.origin_point   # particle final 3D position
            length = np.linalg.norm(pos_fin - pos_ini)  # particle length
            ke = particle['ke']  # particle initial kinetic energy
            theta = particle['theta']  # particle initial theta (dir. in spherical coordinates)
            phi = particle['phi']  # particle initial theta (dir. in spherical coordinates)
            pdg = particle['pdg'] # particle PDG

            assert hits.shape[0] > 0

            if parts[i] in self.mu_particles:
                # Muon case
                # assert particle['exit']  # all muons must escape
                if not particle['exit']:
                    return {'images': None,
                            'ini_pos': None,
                            'params': None,
                            'pids': None,
                            'exit_muon': None,
                            'lens': None
                            }

                # Exiting reconstructed kinematics on outer
                # (img_size+2) x (img_size+2) x (img_size+2) cube VA volume
                ke_exit = particle['ke_exit']
                theta_exit = particle['theta_exit']
                phi_exit = particle['phi_exit']
                pos_exit = particle['pos_exit'] - self.origin_point 

                # Exiting reconstructed on the inner
                # img_size x img_size x img_size cube VA sub-volume
                ke_exit_reduce = particle['ke_exit_reduce']
                theta_exit_reduce = particle['theta_exit_reduce']
                phi_exit_reduce = particle['phi_exit_reduce']
                pos_exit_reduce = particle['pos_exit_reduce'] - self.origin_point 

                # Adjust the exit point of a muon particle considering a potential random shift
                pos_exit_target, shift_plane = fix_exit_shift(pos_exit, pos_exit_reduce, shift_x, shift_y, shift_z)
                if not shift_plane:
                    ke_exit = ke_exit_reduce
                    theta_exit = theta_exit_reduce
                    phi_exit = phi_exit_reduce

                # Shift muon exiting position
                shift_particle(pos_exit_target, shift_x, shift_y, shift_z, self.cube_size)

            # Reconstruct the image from sparse points to a NxNxN volume
            dense_image = np.zeros(shape=(self.img_size + 2, self.img_size + 2, self.img_size + 2))
            dense_image[hits[:, 0], hits[:, 1], hits[:, 2]] = hits[:, 3]

            # Shift image
            shifted_image = shift_image(dense_image, shift_x, shift_y, shift_z, self.img_size)

            # Shift particle initial position
            shift_particle(pos_ini, shift_x, shift_y, shift_z, self.cube_size)

            # Rescale values of particle image and kinematics
            shifted_image = np.interp(shifted_image.ravel(), (self.min_charge, self.max_charge),
                                      self.target_range).reshape(shifted_image.shape)
            pos_ini = np.interp(pos_ini.ravel(), (self.min_ini_pos, self.max_ini_pos),
                                self.source_range).reshape(pos_ini.shape)
            theta = np.interp(theta, (self.min_theta, self.max_theta), self.source_range).reshape(1)
            phi = np.interp(phi, (self.min_phi, self.max_phi), self.source_range).reshape(1)
            
            if parts[i] in self.mu_particles:
                # Muon case
                ke = np.interp(ke, (self.min_ke[part], self.max_ke[part]), self.source_range).reshape(1)
                ke_exit = np.interp(ke_exit, (self.min_ke[part], self.max_ke[part]), self.source_range).reshape(1)
                theta_exit = np.interp(theta_exit, (self.min_theta, self.max_theta), self.source_range).reshape(1)
                phi_exit = np.interp(phi_exit, (self.min_phi, self.max_phi), self.source_range).reshape(1)
                pos_exit = np.interp(pos_exit_target, (self.min_exit_pos_mu, self.max_exit_pos_mu),
                                     self.source_range).reshape(pos_exit.shape)
                pos_exit /= np.abs(pos_exit).max()  # make sure the exiting position touches the volume
            else:
                # Other case (proton, deuterium, tritium)
                ke = np.interp(ke, (self.min_ke[part], self.max_ke[part]), self.source_range).reshape(1)

            # Store particle information
            images.append(shifted_image)
            params.append(np.concatenate((pos_ini, ke, theta, phi)))
            lens.append(length)
            pids.append(self.PID_FROM_PARTICLE[part])

            del particle

        if len(images) == 0:
            return {'images': None,
                    'ini_pos': None,
                    'params': None,
                    'pids': None,
                    'exit_muon': None,
                    'lens': None
                    }

        # Lists to numpy arrays
        images = np.array(images)
        params = np.array(params)
        lens = np.array(lens)
        pids = np.array(pdgs)  # particle identification

        # Sort additional particles by kinetic energy in descendent order (don't order the muon)
        order = params[1:, 3].argsort()[::-1]
        images[1:] = images[1:][order]
        params[1:] = params[1:][order]
        lens[1:] = lens[1:][order]
        pids[1:] = pids[1:][order]

        # Exiting muon information
        exit_muon = np.concatenate((pos_exit, ke_exit, theta_exit, phi_exit))

        # Create a dictionary with the information of the constructed input event
        event = {'images': images,
                 'ini_pos': params[:, :3].mean(axis=0),  # mean of initial positions -> vertex position
                 'params': params[:, 3:],  # KE, theta, phi
                 'pids': pids,  # particle type (defined with PID_FROM_PDG)
                 'exit_muon': exit_muon,  # exiting point, KE, theta, phi
                 'lens': lens,
                 }

        return event



    def collate_fn(self, batch:list[dict]):
        """
        Collates and preprocesses a batch of data samples for the dataloader.

        Args:
            batch (list): A list of events, where each sample is a dictionary containing
                various fields including 'images', 'exit_muon', 'ini_pos', 'params', and 'lens'.

        Returns:
            Tuple: A tuple of torch tensors containing the processed data, including 'img_batch',
            'exit_muons', 'ini_pos', 'params_batch', 'is_next_batch', and 'lens_batch'. If the split
            is 'test', it also returns 'X' containing the test images.
        """
        img_batch, exit_muons, ini_pos, params_batch, pids_batch, \
            is_next_batch, lens_batch = [], [], [], [], [], [], []

        if self.split == "test":
            test_images = []

        for event in batch:
            # Discard the "None" events (the events that are causing issues in the __getitem__)
            if event['images'] is None:
                continue

            # Aggregate voxels from event particles (inner subvolume)
            charge_sum = event['images'][:, 1:-1, 1:-1, 1:-1].sum(0)
            indexes = np.where(charge_sum)  # indexes of non-zero values
            charges = charge_sum[indexes].reshape(-1, 1)  # retrieve non-zero charges
            indexes = np.stack(indexes, axis=1)  # retrieve non-zero indexes

            # Overlapping_img: particles x (x, y, z, c)
            overlapping_img = torch.tensor(np.concatenate((indexes, charges), axis=1))
            exit_muon = torch.tensor(event['exit_muon'])
            pos_ini = torch.tensor(event['ini_pos'])
            params = torch.tensor(event['params'][1:])  # exclude muon params
            pids = torch.tensor(event['pids'][1:] - 1)
            lens = torch.tensor(event['lens'])

            # Set the transformer-decoder ending condition
            is_next = torch.ones(size=(params.shape[0],))
            is_next[-1] = 0

            if self.split == "test":
                test_images.append(event['images'])

            # Append data to respective lists
            img_batch.append(overlapping_img)
            exit_muons.append(exit_muon)
            ini_pos.append(pos_ini)
            params_batch.append(params)
            is_next_batch.append(is_next)
            pids_batch.append(pids)
            lens_batch.append(lens)

        assert len(img_batch) > 0

        # Convert lists to torch tensors and pad sequences
        img_batch = pad_sequence(img_batch, padding_value=self.pad_value).float()
        exit_muons = torch.stack(exit_muons).float()
        ini_pos = torch.stack(ini_pos).float()
        params_batch = pad_sequence(params_batch, padding_value=self.pad_value).float()
        pids_batch = pad_sequence(pids_batch, padding_value=0).long()
        is_next_batch = pad_sequence(is_next_batch, padding_value=self.pad_value).long()
        lens_batch = pad_sequence(lens_batch, padding_value=self.pad_value).float()

        if self.split == "test":
            return img_batch, exit_muons, ini_pos, params_batch, is_next_batch, pids_batch, lens_batch, test_images
        return img_batch, exit_muons, ini_pos, params_batch, is_next_batch, pids_batch, lens_batch
    
    
    

