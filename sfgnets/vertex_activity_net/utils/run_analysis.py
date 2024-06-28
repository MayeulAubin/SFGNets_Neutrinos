import json
import torch
import numpy as np
from ..datasets import TransformerDataset
from ..models import VATransformer
from ..utils import args_transformer
from .my_utils import analyze_testset
import pickle as pk

parser_trans= args_transformer()
args_trans = parser_trans.parse_args(["-cp", "../modules/sfgnets/vertex_activity_net/config/decomposing_transformer_v1.json",
                         ])

# Configuration files
with open(args_trans.config_path) as config_file:
    config_trans = json.load(config_file)
    
    
test_set_trans = TransformerDataset(config=config_trans, split="test")
transformer = VATransformer(num_encoder_layers=args_trans.encoder_layers,
                               num_decoder_layers=args_trans.decoder_layers,
                               emb_size=args_trans.hidden,
                               num_head=args_trans.attn_heads,
                               img_size=config_trans["img_size"],
                               tgt_size=config_trans["target_size"],
                               dropout=args_trans.dropout,
                               max_len=config_trans["max_p"]+config_trans["max_n"],
                              #  max_len=6,
                               device="cpu",
                               )
checkpoint_trans = torch.load(config_trans["checkpoint_test_path"], map_location='cpu')
transformer.load_state_dict(checkpoint_trans, strict=True)
transformer.eval()

per_event_analysis,per_particle_analysis = analyze_testset(model=transformer, test_set=test_set_trans, 
                                                        #    N_max=1000
                                                           )


with open("/scratch4/maubin/results/vatransformer_v9_analysis.pk","wb") as f:
    pk.dump((per_event_analysis,per_particle_analysis),f)