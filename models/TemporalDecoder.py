import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules



class TemporalDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # params
        self.traj_len = params.traj_len
        # decoder
        # dim_in = params.dim_com_list[-1] + 2 * params.dim_embed_tim
        dim_in =  2 * params.dim_embed_tim
        # dim_in = params.dim_com_list[-1] + params.dim_embed_tim
        dim_linear_list = [params.dim_dec_tim] + params.dim_TD_linear
        self.tim_decoder = modules.DecoderGRU(params, dim_in, params.dim_dec_tim, params.num_dec_tim, 'temporal', dim_linear_list)

    def forward(self, zt, loc_info):

        # concat time and predicted loc info
        # X_hid = torch.cat([zt.unsqueeze(1).expand(-1, loc_info.shape[1], -1), loc_info], dim=-1)   # (B, S=24, Ht+Hs)
        X_hid = zt.unsqueeze(1).expand(-1, loc_info.shape[1], -1)
        # decode duration chain in the next day
        tim_chain = self.tim_decoder(X_hid) * self.traj_len  # tim_chain:(B,S,1)
        tim_chain = tim_chain.squeeze(-1)

        return tim_chain
        