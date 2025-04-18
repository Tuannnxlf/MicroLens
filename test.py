import os
import torch


best_ckpt = '/opt/data/private/vllm2rec/MicroLens/Code/VideoRec/SASRec/checkpoint/checkpoint_MicroLens-100k_pairs_id/cpt_v1_sasrec_blocknum_2_tau_0.07_bs_512_ed_1024_lr_0.0001_l2_0.1_maxLen_10/epoch-12.pt'
checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
print("Checkpoint keys:", checkpoint['model_state_dict'].keys())