import json
import torch
import numpy as np
from ..datasets import TransformerDataset
from ..models import VATransformer
from .arguments import args_transformer
from .evaluate import analyze_testset
import pickle as pk
import os
import argparse

parser = argparse.ArgumentParser(prog='VertexActivityAnalysis',
                        description='Run the analysis of a Vertex Activity model',)
    
parser.add_argument("-cp", "--config_path", type=str,
                  default="sfgnets/vertex_activity_net/config/decomposing_transformer_v{}.json",
                  help="path of configuration file")
parser.add_argument("-hs", "--hidden", type=int, default=192, help="hidden size of transformer model")
parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="dropout of the model")
parser.add_argument("-el", "--encoder_layers", type=int, default=10, help="number of encoder layers")
parser.add_argument("-dl", "--decoder_layers", type=int, default=10, help="number of decoder layers")
parser.add_argument("-a", "--attn_heads", type=int, default=16, help="number of attention heads")
parser.add_argument("-G", "--gpu", type=int, default=1, help="GPU ID (cuda) to be used")
parser.add_argument("-vc", "--version_config", type=int, default=1, help="Version of the config to use")
parser.add_argument("-T", "--use_truth", action="store_true", help="Use the truth information in the iteration of the transformer")
parser.add_argument("-fc", "--formula_correction", action="store_true", help="Use the approximation formula correction")


args_trans = parser.parse_args()

# Set the version
args_trans.config_path = args_trans.config_path.format(args_trans.version_config)

# Configuration files
with open(args_trans.config_path) as config_file:
    config_trans = json.load(config_file)
    
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_trans.gpu)
    
    
test_set_trans = TransformerDataset(config=config_trans, split="test")
transformer = VATransformer(num_encoder_layers=args_trans.encoder_layers,
                               num_decoder_layers=args_trans.decoder_layers,
                               emb_size=args_trans.hidden,
                               num_head=args_trans.attn_heads,
                               img_size=config_trans["img_size"],
                               kin_tgt_size=config_trans["target_size"],
                            #    pid_tgt_size=len(config_trans["additional_particles"]),
                              #  pid_tgt_size=3, # default value
                               pid_tgt_size=2, # default value
                               dropout=args_trans.dropout,
                               max_len=sum([config_trans[f"max_{part}"] for part in config_trans["additional_particles"]]),
                               device="cuda",
                               )
checkpoint_trans = torch.load(config_trans["checkpoint_test_path"], map_location='cuda')
transformer.load_state_dict(checkpoint_trans, strict=True)
transformer.eval()

model_version = config_trans["checkpoint_test_path"].split('/')[-1].split('.')[0]

print(f"Model used : {model_version}")

per_event_analysis,per_particle_analysis = analyze_testset(model=transformer, test_set=test_set_trans,
                                                           device="cuda",
                                                           use_truth=args_trans.use_truth, 
                                                        #    N_max=1000,
                                                            use_formula_correction=args_trans.formula_correction,
                                                           )

with open(f"/scratch4/maubin/results/{model_version}_analysis.pk","wb") as f:
    pk.dump((per_event_analysis,per_particle_analysis),f)