import ROOT as rt
import numpy as np
import os
import argparse
import sys
import tqdm
import shutil
import gc

parser = argparse.ArgumentParser(
                    prog='ConvertRootToNumpy',
                    description='Converts a Root file containing SFG events to a list of numpy files, one per event',)
parser.add_argument('input', metavar='input', type=str, help='input root, files of the type input_{particle}.root or input_{particle}_test.root will be used for the production')
parser.add_argument('output_dir', metavar='output_dir', type=str, help='output directory')
parser.add_argument('-v', '--verbose', action='store_true', help='prints the errors in the production')
parser.add_argument('--test', action='store_true', help='use the test suffix for the input file')
parser.add_argument('--aux', action='store_true', help='store also the auxiliary variables (pdg, reaction_code, ...)')
parser.add_argument('-p', '--particle', type=str, default=None, help='runs the conversion for a specific particle')
parser.add_argument('-s', '--start', type=int, default=0, help='in case of a specific particle, starting point of the data conversion')
args = parser.parse_args()

from sfgnets.datasets.production.produce_numpy_files import prod_numpy_pgun_dataset

list_of_particles=['e+','e-','gamma','mu+','mu-','n','p','pi+','pi-']

## Get the second argument passed to the script (output directory)
output_dir=args.output_dir

if args.particle is None:    
    for part in tqdm.tqdm(list_of_particles,desc="Prod numpy all particles"):
        ## Get the first argument passed to the script (the name of the root files)
        input=args.input
        input_file_name=f"{input}_{part}{'_test' if args.test else ''}.root"
        
        ## Load the root file
        root_input_file = rt.TFile(input_file_name)
        
        ## Make Project (it seems to be necessary to convert ND280 Objects to vectors, arrays,...)
        root_input_file.MakeProject(f"prod_numpy{'_test' if args.test else ''}","ND::TSFGReconModule","recreate++")
        
        prod_numpy_pgun_dataset(root_input_file,
                                output_dir,
                                part,
                                verbose=args.verbose,
                                aux=args.aux)

else:
    part=args.particle
    ## Get the first argument passed to the script (the name of the root files)
    input=args.input
    input_file_name=f"{input}_{part}{'_test' if args.test else ''}.root"
    
    start=args.start
    
    ## Load the root file
    root_input_file = rt.TFile(input_file_name)
    
    
    ## Make Project (it seems to be necessary to convert ND280 Objects to vectors, arrays,...)
    root_input_file.MakeProject(f"prod_numpy_{part}{'_test' if args.test else ''}","ND::TSFGReconModule","recreate++")
    
    
    
    ret=prod_numpy_pgun_dataset(root_input_file,
                            output_dir,
                            part,
                            verbose=args.verbose,
                            aux=args.aux,
                            start=start)
    
    if ret is not None:
        with open(f"prod_numpy_{part}{'_test' if args.test else ''}.txt",'w') as file:
            file.write(str(ret[0]))
    
    # raise ret[1]


