import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import TokenChannelEmbedding
import numpy as np


class Embeddings(nn.Module):
    def __init__(self, configs):
        super(Embeddings, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        # Embedding
        if configs.no_temporal_block and configs.no_channel_block:
            raise ValueError("At least one of the two blocks should be True")
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        if configs.no_channel_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def forward(self, x_enc):
        # Embedding
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
       
        # if enc_out_t is None:
        #     enc_out = enc_out_c
        # elif enc_out_c is None:
        #     enc_out = enc_out_t
        # else:
        #     enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)       
        return (enc_out_t,enc_out_c)

    
