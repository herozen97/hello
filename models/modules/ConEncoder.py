import torch.nn as nn
import torch


class ConEncoder(nn.Module):
    '''
    Module in LocGenerator
    '''
    def __init__(self, params):
        super().__init__()

        self.home_encoder = nn.Linear(params.embed_dim_lid, params.CE_dim_attr[0])
        self.commuter_encoder = nn.Linear(1, params.CE_dim_attr[1])
        self.rg_encoder = nn.Linear(1, params.CE_dim_attr[2])
        self.motif_encoder = nn.Linear(params.dim_motif, params.CE_dim_attr[3])


    def forward(self, X):

        # encode
        attr_home, (attr_commuter, attr_rg, attr_motif, _) = X
        # t1 = time.time()
        attr_home_en = self.home_encoder(attr_home)
        # t2 = time.time()
        attr_commuter_en = self.commuter_encoder(attr_commuter)
        # t3 = time.time()
        attr_rg_en = self.rg_encoder(attr_rg)
        # t4 = time.time()
        attr_motif_en = self.motif_encoder(attr_motif)
        # t5 = time.time()

        # concat 
        hc = torch.cat([attr_home_en, attr_commuter_en, attr_rg_en, attr_motif_en], dim=-1)    # (B, Hc)
       
        return hc