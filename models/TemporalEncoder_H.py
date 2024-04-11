import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules

'''
M-v1

v1.1: add mask day
v1: initial.

'''


class TemporalEncoder_H(nn.Module):
    def __init__(self, params):
        super().__init__()

        # parameters
        self.win_len = params.win_len   # default=7
        self.device = params.device
        
        # time embedder
        self.tim_embedder = modules.nnEmbedder(params.traj_len+1, params.dim_embed_tim)
        
        # day encoder
        self.day_encoder = modules.EncoderTF(params, params.dim_embed_tim, params.head_num_tim, params.dim_enc_tim, params.num_enc_tim, params.traj_len)

        # # week encoder
        # self.week_encoder = modules.EncoderTF(params, 2 * params.dim_embed_tim, params.head_num_tim, params.dim_enc_tim, params.num_enc_tim, params.traj_len)

        # loc compresssor
        dim_compress = [params.dim_embed_loc, params.dim_embed_tim]
        self.loc_compressor = modules.LocCompressor(dim_compress)


    def forward(self, tim_batch, loc_day_all, mask_day, mask_traj):

        tim_day_all = torch.tensor([]).to(self.device)
        # encode daily trajectory
        for i in range(self.win_len):
            # time embedding
            tim_embed = self.tim_embedder(tim_batch[:, i, :])       # tim_batch:(B, 7, S=24), tim_embed:(B,S,E)
            # encode daily trajectory in week
            tim_day = self.day_encoder(tim_embed, mask_traj[:, i, 1:-1]).unsqueeze(1)    # tim_day:(B, 1, H), mask_traj:(B,7,S=26)
            # for k in range(tim_day.size()[0]):
            #     print('k',tim_day[k])
            tim_day_all = torch.cat([tim_day_all, tim_day], dim=1)
        
        # encode whole week infomation
        loc_info = self.loc_compressor(loc_day_all)                 # loc_day_all:(B, 7, E)
        info_cat = torch.cat([loc_info, tim_day_all], dim=-1)       # tim_day_all:(B, 7, H), loc_info:(B, 7, H), info_cat:(B,7,2H)
        tim_week = torch.mean(info_cat, dim=1)            # tim_week:(B, 2H)
        # for i in range(tim_week.size()[0]):
        #     print('week',tim_week[i])
        
        return tim_week
    