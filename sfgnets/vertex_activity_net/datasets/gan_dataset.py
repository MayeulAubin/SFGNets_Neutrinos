"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch dataset that for the generative-adversarial network (GAN).
"""

import numpy as np
import pickle as pk
import torch
from torch.utils.data import Dataset


class GANDataset(Dataset):

    def __init__(self, config: dict, split: str = "train"):
        """
        Dataset initialiser.

        Args:
            config (dict): Dictionary with JSON configuration entries.
            split (str): String indicating the purpose of the dataset ("train", "val", "test").

        Returns:
            None
        """
        with open(config["dataset_metadata"], "rb") as fd:
            charge, ini_pos, _, _, _, _, _ = pk.load(fd)
        with open(config["indices_path"], "rb") as fd:
            indices = pk.load(fd)

        self.particle = config["particle"]  # particle type ("p", "mu", "D", or "T")
        self.config = config["config"]  # first or seconf configuration
        indices = indices["indices_{}_{}".format(self.particle, self.config)]  # file id
        self.indices = indices["{}_indices".format(split)]  # file id
        self.dataset = config["dataset"]  # proton dataset path
        self.img_size = config["img_size"]  # img_size x img_size x img_size
        self.min_charge = config["min_charge"]  # min charge (energy loss) per cube
        self.max_charge = charge.max()  # max charge per cube
        self.min_pos = ini_pos.min()  # min initial particle 1D position
        self.max_pos = ini_pos.max()  # max initial particle 1D position
        self.min_ke = config["min_ke"]  # min initial proton kinetic energy
        self.max_ke = config["max_ke"]  # max initial proton kinetic energy
        self.min_theta = config["min_theta"]  # min theta (in degrees)
        self.max_theta = config["max_theta"]  # max theta (in degrees)
        self.min_phi = config["min_phi"]  # min phi (in degrees)
        self.max_phi = config["max_phi"]  # max phi (in degrees)
        self.source_range = config["source_range"]  # range for input values
        self.target_range = config["target_range"]  # range for target values
        self.total_events = len(self.indices)  # total number of different particles
        if self.particle == "mu":
            self.cube_size = config["cube_size"]  # cube (voxel) size in mm (one side)
            self.min_exit_pos_mu = -(self.cube_size * self.img_size) / 2.  # min muon exiting 1D position
            self.max_exit_pos_mu = (self.cube_size * self.img_size) / 2.  # max muon exiting 1D position

    def __len__(self):
        """
        Returns the total number of events in the dataset.

        Returns:
            int: Total number of events in the dataset.
        """
        return self.total_events

    def collate_fn(self, batch):
        """
        Collates a batch of data into tensors.

        Args:
            batch (list): A list of particles.

        Returns:
            tuple: A tuple containing two tensors - image batch and parameters batch.
        """
        img_batch = np.array([event['image'] for event in batch if event['image'] is not None])
        if self.particle == "mu":
            params_batch = np.array([np.concatenate([event['pos_ini'], event['pos_exit'],
                                                     event['ke_exit'], event['theta_exit'], event['phi_exit']])
                                     for event in batch if event['pos_ini'] is not None])
        else:
            params_batch = np.array([np.concatenate([event['pos_ini'], event['ke'], event['theta'], event['phi']])
                                     for event in batch if event['pos_ini'] is not None])
        img_batch = torch.tensor(img_batch).float()
        params_batch = torch.tensor(params_batch).float()

        return img_batch, params_batch

    def __getitem__(self, idx):
        """
        Retrieves a data sample at the given index.

        Args:
            idx (int): Index of the particle to retrieve.

        Returns:
            dict: A dictionary containing information about the particle image and kinematics.
        """
        # Retrieve particle data
        index = self.indices[idx]
        filepath = self.dataset.format(index)
        particle = np.load(filepath)

        # Retrieve input
        hits = particle['sparse_image'].astype(int)  # array of shape (Nx5) [points vs (x, y, z, charge, tag)]
        pos_ini = particle['pos_ini']  # array with initial position (x1, y1, z1)
        ke = particle['ke']  # kinetic energy
        theta, phi = particle['theta'], particle['phi']  # theta and phi (spherical coordinates)

        if hits.shape[0] == 0:
            del particle
            return {'image': None,
                    'pos_ini': None,
                    'ke': None,
                    'theta': None,
                    'phi': None}

        # Reconstruct the image to a (self.img_size-2)x(self.img_size-2)x(self.img_size-2) flat volume
        dense_image = np.zeros(shape=(self.img_size + 2, self.img_size + 2, self.img_size + 2))
        dense_image[hits[:, 0], hits[:, 1], hits[:, 2]] = hits[:, 3]
        dense_image = dense_image[2:-2, 2:-2, 2:-2]
        dense_image = dense_image.reshape(-1)

        # Rescale values of particle image and kinematics
        dense_image = np.interp(dense_image.ravel(), (self.min_charge, self.max_charge),
                                self.target_range).reshape(dense_image.shape)
        pos_ini = np.interp(pos_ini.ravel(), (self.min_pos, self.max_pos), self.source_range).reshape(pos_ini.shape)
        if self.particle == "mu":
            # Exiting reconstructed kinematics on outer
            ke_exit = particle['ke_exit']
            theta_exit = particle['theta_exit']
            phi_exit = particle['phi_exit']
            pos_exit = particle['pos_exit']
            # Rescale
            pos_exit = np.interp(pos_exit.ravel(), (self.min_exit_pos_mu, self.max_exit_pos_mu),
                                 self.source_range).reshape(pos_exit.shape)
            pos_exit /= np.abs(pos_exit).max()
            ke_exit = np.interp(ke_exit, (self.min_ke, self.max_ke), self.source_range).reshape(1)
            theta_exit = np.interp(theta_exit, (self.min_theta, self.max_theta), self.source_range).reshape(1)
            phi_exit = np.interp(phi_exit, (self.min_phi, self.max_phi), self.source_range).reshape(1)

        else:
            ke = np.interp(ke, (self.min_ke, self.max_ke), self.source_range).reshape(1)
            theta = np.interp(theta, (self.min_theta, self.max_theta), self.source_range).reshape(1)
            phi = np.interp(phi, (self.min_phi, self.max_phi), self.source_range).reshape(1)

        del particle

        # Create a dictionary with the information of the particle
        if self.particle == "mu":
            particle = {'image': dense_image,
                        'pos_ini': pos_ini,
                        'pos_exit': pos_exit,
                        'ke_exit': ke_exit,
                        'theta_exit': theta_exit,
                        'phi_exit': phi_exit}
        else:
            particle = {'image': dense_image,
                        'pos_ini': pos_ini,
                        'ke': ke,
                        'theta': theta,
                        'phi': phi
                        }

        return particle
