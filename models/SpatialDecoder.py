import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules




class SpatialDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # decoder
        # v1.0: transformer
        dim_in = params.dim_embed_loc
        dim_linear_list = [dim_in] + params.dim_LD_linear + [params.lid_size+3]
        self.loc_decoder = modules.DecoderTF(params, dim_in, params.head_num_loc, params.dim_dec_loc, params.num_dec_loc, params.traj_len+2, dim_linear_list)
        # v1.1: GRU
        # dim_in = params.dim_embed_loc * 2
        # dim_linear_list = [params.dim_dec_loc] + params.dim_LD_linear + [params.lid_size+3]
        # self.loc_decoder = modules.DecoderGRU(params, dim_in, params.dim_dec_loc, params.num_dec_loc, 'spatial', dim_linear_list)


    def forward(self, loc_tgt, loc_tgt_embed, zs):

        # decode location chain in the next day
        # v1.0: transformer
        loc_chain = self.loc_decoder(loc_tgt, loc_tgt_embed, zs)
        # v1.1: GRU
        # X_hid = torch.cat([loc_tgt_embed, zs.unsqueeze(1).expand(-1, loc_tgt_embed.shape[1], -1)], dim=-1)    # X_hid:(B, S, E*2), zs:(B,E)
        # loc_chain = self.loc_decoder(X_hid)

        return loc_chain
        


        