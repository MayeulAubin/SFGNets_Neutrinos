"""
For each event of the dataset stored in event_{n}_{part}.npz, check if it is valid and then recreate the dataset metadata accodingly.
Check the following:
    - That there are hits in the VA region
    - That muons exit the VA region
    - That additional particles don't have adjacent cubes
"""

import numpy as np
import pickle as pk
import tqdm
from glob import glob
import re
import argparse


def check_event(event_file, 
                is_muon:bool,
                VA_region_size:float,
                origin_point:np.ndarray) -> bool:
    
    ## Check if the event has hits in the VA region
    hit_coordinates = event_file["sparse_image"][:,:3] - origin_point[None,:]
    if not ((np.abs(hit_coordinates)<=VA_region_size/2+1e-2).prod(axis=-1)).any():
        return False
    
    ## Check if muons exit the VA region
    if is_muon:
        if not event_file["exit"]:
            return False
    
    ## Check if additional particles don't have adjacent cubes
    else:
        if event_file["adjacent_cube"]:
            return False
    
    return True



def natural_sort(l:list) -> list:
    """
    Perform a natural sort on the given list.

    Parameters:
    - l: list, input list to be sorted

    Returns:
    - list, sorted list
    """
    # Function to convert text to a numeric value for sorting
    convert = lambda text: int(text) if text.isdigit() else text.lower()

    # Function to create a key for sorting using the converted values
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    # Sort the list using the created key
    return sorted(l, key=alphanum_key)



def rebuild_metadata(dataset_path:str,
                     metadata_file:str,
                     mu_particles:list[str],
                     additional_particles:list[str],
                     VA_region_size:float,
                     origin_point:np.ndarray) -> None:
    
    particles = mu_particles + additional_particles
    are_muons = [True for part in mu_particles] + [False for part in additional_particles]
    
    for k in tqdm.tqdm(range(len(particles)),desc="Rebuilding metadatas", position=0, leave=True):
        
        part = particles[k]
        is_muon = are_muons[k]
        
        ## Get the previous metadatas
        with open(f"{dataset_path}/{metadata_file}_{part}.p", "rb") as fd:
            charges, ini_pos, kes, thetas, phis, lookup_table, bin_edges = pk.load(fd)
        
        ## Get all the event files
        event_files = natural_sort(glob(f'{dataset_path}/event_*_{part}.npz'))
        
        ## Reset the metadatas
        charges, ini_pos, kes, thetas, phis = [], [[],[],[]], [], [], []
        lookup_table = {}
        
        failed_events = 0
        last_nb_failed_events = 0
        progess_bar = tqdm.tqdm(range(len(event_files)), desc=f"Events {part}", position=1, leave=False)
        for n in progess_bar:
            
            event = np.load(event_files[n])
            entry = event["event_index"]
            
            if check_event(event,
                           is_muon=is_muon,
                           VA_region_size=VA_region_size,
                           origin_point=origin_point):
                
                # Retrieve the bin of the initial position
                bin_indices = np.digitize(event["pos_ini"]-origin_point, bin_edges, right=False)
                bin_indices = tuple(bin_indices)
                
                if bin_indices not in lookup_table:
                    lookup_table[bin_indices] = [entry]
                else:
                    lookup_table[bin_indices].append(entry)
                    
                # gather statistics
                charges.extend(event["sparse_image"][:,3])
                ini_pos[0].append(event["pos_ini"][0])
                ini_pos[1].append(event["pos_ini"][1])
                ini_pos[2].append(event["pos_ini"][2])
                kes.append(event["ke"])
                thetas.append(event["theta"])
                phis.append(event["phi"])
            
            else:
                failed_events += 1
                
            if n%1000 == 999:
                progess_bar.set_postfix({"Failed events":f"{(failed_events-last_nb_failed_events)/10:.1f}%"})
                last_nb_failed_events = failed_events
        
        
        charges = np.array(charges)
        ini_pos = np.array(ini_pos)
        kes = np.array(kes)
        thetas = np.array(thetas)
        phis = np.array(phis)
        
        ## Save the metadatas
        with open(f"{dataset_path}/{metadata_file}_filtered_{part}.p", "wb") as fd:
            pk.dump([charges, ini_pos, kes, thetas, phis, lookup_table, bin_edges], fd)
            



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                        prog='FilteringVADataset',
                        description='Filters the events of the vertex activity dataset and recreates the metadatas.',)

    parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#_part.npz files and the metadatas")
    parser.add_argument('-m','--meta',metavar='Metadata file', type=str, default="data_stats", help="Metadata file names root")
    parser.add_argument('-mu','--mupart',metavar='Mu particles', type=str, nargs="*", default=["mu+","mu-"], help="Muons like particles")
    parser.add_argument('-ad','--addpart',metavar='Additional particles', type=str, nargs="*", default=["p","n"], help="Additional particles that remain in the VA region")
    parser.add_argument('-va','--vasize',metavar='Vertex activity size', type=float,  default=10.27*9, help="Size of the vertex activity region to consider")
    parser.add_argument('-or','--originpoint',metavar='Origin point', type=float, nargs=3, default=[5.13, 35.13, -1938.80], help="Central hit coordinates of the VA region")
    args = parser.parse_args()
    
    
    rebuild_metadata(dataset_path=args.dataset_folder,
                     metadata_file=args.meta,
                     mu_particles=args.mupart,
                     additional_particles=args.addpart,
                     VA_region_size=args.vasize,
                     origin_point=np.array(args.originpoint))