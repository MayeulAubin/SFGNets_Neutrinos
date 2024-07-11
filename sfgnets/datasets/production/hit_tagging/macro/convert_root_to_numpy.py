import ROOT as rt
import numpy as np
import os
import argparse
import sys
import tqdm

parser = argparse.ArgumentParser(
                    prog='ConvertRootToNumpy',
                    description='Converts a Root file containing SFG events to a list of numpy files, one per event',)
parser.add_argument('input_file', metavar='input_file', type=str, help='input root file')
parser.add_argument('output_dir', metavar='output_dir', type=str, help='output directory')
parser.add_argument('-v', '--verbose', action='store_true', help='prints the errors in the production')
parser.add_argument('--aux', action='store_true', help='store also the auxiliary variables (pdg, reaction_code, ...)')
args = parser.parse_args()

from sfgnets.datasets.production.produce_numpy_files import prod_numpy_hittag_dataset

## Get the first argument passed to the script (the name of the root file)
input_file_name=args.input_file

## Get the second argument passed to the script (output directory)
output_dir=args.output_dir

## Load the root file
root_input_file = rt.TFile(input_file_name)


## Make Project (it seems to be necessary to convert ND280 Objects to vectors, arrays,...)
root_input_file.MakeProject("prod_numpy","ND::TSFGReconModule","recreate++")

prod_numpy_hittag_dataset(root_input_file,output_dir, args.verbose, args.aux)