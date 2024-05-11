import random

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from point_4d_convolution_segment import *
from transformer import *
from transformer_mask import *
from pst_convolutions import *
from pointnet import pointnet
import torchvision.transforms as transforms
from . import data_utils as d_utils
from einops import rearrange, reduce, repeat
from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 150):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(0)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)
        # return x


class fusion_net(nn.Module):
    def __init__(self ):                                            
        super().__init__()


        self.project1 = nn.Sequential(
                nn.LayerNorm(2048),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                ) 


        self.project2 = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Linear(1024, 2048),
                ) 

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

        self.mlp_out = nn.Sequential(
            nn.LayerNorm(512),  #1024  2560
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 19),
        )

        self.mlp_out_new = nn.Sequential(
            nn.LayerNorm(2048),  #1024  2560 2176
            nn.Linear(2048, 512),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 19),
        )


        self.config = BertConfig(
                 hidden_size=2048,  #1024 2560 2176
                 num_hidden_layers=8,  #4
                 num_attention_heads=8, #16 
                 type_vocab_size=2)

        self.encoder = BertEncoder(self.config)

        self.pos_embedding = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)

        self.vit_positions = nn.Parameter(torch.randn((150, 2048)))

        self.image_drop = nn.Dropout(0.1)

        self.transformer3 = Transformer_mask(2048, 5, 8, 128, 1024)

        self.transformer_2d = Transformer(2048, 5, 8, 128, 1024)
        self.transformer_3d = Transformer(2048, 5, 8, 128, 1024)

        self.learning_position = LearnedPositionEncoding(d_model=512,max_len=150)
        self.layer_norm_2d = nn.LayerNorm(2048)
        self.layer_norm_3d = nn.LayerNorm(2048)
        self.cls_token = nn.Parameter(torch.randn(1,1,2048))

    def create_1d_absolute_sin_cos_embedding(self,pos_len, dim):
        assert dim % 2 == 0, "wrong dimension!"
        position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
        # i矩阵
        i_matrix = torch.arange(dim//2, dtype=torch.float)
        i_matrix /= dim / 2
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1 / i_matrix
        i_matrix = i_matrix.to(torch.long)
        # pos矩阵
        pos_vec = torch.arange(pos_len).to(torch.long)
        # 矩阵相乘，pos变成列向量，i_matrix变成行向量
        out = pos_vec[:, None] @ i_matrix[None, :]
        # 奇/偶数列
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        # 赋值
        position_emb[:, 0::2] = emb_sin
        position_emb[:, 1::2] = emb_cos
        return position_emb 


    def append_cls_token(patch_embedding):
        bs, _, model_dim = patch_embedding.shape
        cls_token_embedding = t.randn(bs, 1, model_dim, requires_grad=True)
        # 把cls放到第一个位置上
        token_embedding = t.cat([cls_token_embedding, patch_embedding], dim=1)
        return token_embedding

    def forward(self, input_3d, input_2d,label,xyzt,mode):

        # xyzt  (8,150*64,4)
        # input_3d = self.mlp_head(input_3d)

        # xyzt = xyzt.reshape((xyzt.shape[0],150,64,xyzt.shape[2]))
        # pe_2d = torch.mean(xyzt,2)

        #### 绝对位置pe
        # pe_2d = self.create_1d_absolute_sin_cos_embedding(150,2048)
         #### vit pe  
        pe_2d = self.vit_positions
        pe_3d = self.vit_positions
        # pe_all = self.vit_positions
        pe_2d = pe_2d.to(input_3d.device)
        pe_3d = pe_3d.to(input_3d.device)
        # pe_all = pe_all.to(input_3d.device)
        ######## CLS token
        # b = input_3d.shape[0]  # 单独先将batch缓存起来
        # # 将cls_token 扩展b次
        # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # # 将cls token在维度1扩展到输入上
       
        input_image = self.project2(input_2d) # 8， 150， 2048

        # input_image = torch.cat([cls_tokens, input_image], dim=1)
        # input_3d = torch.cat([cls_tokens, input_3d], dim=1)

        # input_image = input_2d

        ######
        # input_image = self.layer_norm_2d(input_image + pe_2d)
        # input_image = self.transformer_2d(input_image)
        # input_image = self.layer_norm_2d(input_image + pe_all)

        # input_3d = self.layer_norm_3d(input_3d + pe_3d)
        # input_3d = self.transformer_3d(input_3d)
        # input_3d = self.layer_norm_3d(input_3d + pe_all)

        #######

        # input_image = input_2d


        # # ###
        # # input_3d = self.project1(input_3d)

        # # input_image = self.project1(input_2d)
        # input_image = self.learning_position(input_image)

        # input_3d = self.learning_position(input_3d)
        # pe_3d = pe_3d.to(input_3d.device)

        input_image = input_image + pe_2d
        # input_image = self.learning_position(input_image)

        input_image = self.image_drop(input_image)

        input_3d = input_3d + pe_3d
        # input_3d = self.learning_position(input_3d)

        input_3d = self.image_drop(input_3d)


        # if mode == 'train':

        fusion_input = torch.cat( [input_3d ,input_image] , 1 )   ## 8 ， 300 ，2048

        # fusion_input = self.learning_position(fusion_input)



        head_mask = ([None] * self.config.num_hidden_layers)

        attention_mask = torch.ones(fusion_input.shape[0] ,1 , fusion_input.shape[1] , fusion_input.shape[1]).to(fusion_input.device)
        attention_mask[:, :, :150, 150:] = 0.
            
        extended_attention_mask = (1.0 - attention_mask) * -10000.0




        # mask = torch.ones((fusion_input.shape[0],fusion_input.shape[1])).to(fusion_input.device)
        # B, L = mask.shape
        # mask = mask.unsqueeze(dim = 1).repeat(1,L,1)
        # mask = mask.unsqueeze(dim = 1).repeat(1,8,1,1)

        # # if mode == 'train':
            
        # mask[:, :, :150, 150:] = 0.

        # extended_attention_mask = (1.0 - mask) * -10000.0
        # out = self.transformer3(fusion_input,extended_attention_mask)

        encoder_outputs = self.encoder(
            fusion_input,
            extended_attention_mask,
            head_mask=head_mask
        )

        out = encoder_outputs[0]
        
        # ### only 2d
        # data_3d = torch.zeros((input_2d.shape[0],150,512)).to(input_3d.device)
        # input_fu = torch.cat( [data_3d ,input_2d] , 1 )
        # fusion_feature = input_2d
       
        # output = self.project1(input_fu)
        # ####
        output = self.mlp_out_new(out)
        # output = self.mlp_out(out)
        fusion_feature = out

        return output , fusion_feature


class PrimitiveTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu3 = nn.ReLU()
        self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.pos_embedding_seg = nn.Conv1d(in_channels=48, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )


    def forward(self, input , segment_feature ): 



        # 4d BACKBONE
        # [B, L, N, 3]
        xyzs, features,anchor_idx= self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        # xyzs = torch.zeros(8,150,64,3)
        # features = torch.zeros(8,150,64,2048)
        # segment_feature （B,150,2048,48)

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        device = raw_feat.get_device()
        # xyzts = []
        # xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        # xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        # for t, xyz in enumerate(xyzs):
        #     t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
        #     # t = torch.zeros((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device)
        #     xyzt = torch.cat(tensors=(xyz, t), dim=2)
        #     xyzts.append(xyzt)

        # xyzts = torch.stack(tensors=xyzts, dim=1)
        # xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]
                                                                                      # [B, L,   n, C]
        features = torch.reshape(input=raw_feat, shape=(raw_feat.shape[0], raw_feat.shape[1]*raw_feat.shape[2], raw_feat.shape[3]))         # [B, L*n, C]

        # # xyzts_before = xyzts.clone()

        # xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        # xyzts_before = xyzts.clone()
        xyzts_before = features
        # # embedding = xyzts + features     # (8,150,64,2048)
        anchor_idx = (anchor_idx).to(torch.long)
        se_l = []

        for ij in range (segment_feature.shape[0]):
        # seg_ccc = segment_feature.permute(2, 0, 1,3)
            seg_ccc = segment_feature
            se = seg_ccc[ij,:,anchor_idx[ij,:],:]
            se_l.append(se)

        se_l = torch.stack(se_l, axis=0)

        seg = se_l.reshape((features.shape[0],features.shape[1],48))

        seg = self.pos_embedding_seg(seg.permute(0, 2, 1)).permute(0, 2, 1)


        embedding = features + seg 
        # embedding = xyzts + embedding
        # embedding = features + seg
        # embedding = features

        point_feat = torch.reshape(input=embedding, shape=(B * L * 8, -1, C))  # [B*L*4, n', C]

        point_feat = self.emb_relu1(point_feat)
        
        point_feat = self.transformer1(point_feat)  # [B*L*4, n', C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L * 8, C))  # [B, L*4, C]


        # primitive_feature = torch.reshape(input=primitive_feature, shape=(B * L , -1, C))  # [B*L*4, n', C]
        # primitive_feature = self.emb_relu3(primitive_feature)
        # primitive_feature = self.transformer3(primitive_feature)  # [B*L*4, n', C]

        anchor_feature = torch.reshape(input=primitive_feature, shape=(B*L, 8, C))
        anchor_feature = anchor_feature.permute(0, 2, 1)
        anchor_feature = F.adaptive_max_pool1d(anchor_feature, (1))
        anchor_feature = torch.reshape(input=anchor_feature, shape=(B, L, C))

        primitive_feature = self.emb_relu2(anchor_feature)
        primitive_feature = self.transformer2(primitive_feature) # B. L, C

        # primitive_feature = primitive_feature.reshape(B*L, 8, C)
        # primitive_feature = primitive_feature.permute(0,2,1)
        # primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))
        # primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L, C))  

        # output = torch.max(input=primitive_feature, dim=1, keepdim=False, out=None)[0]

        # output =  self.mlp_head(primitive_feature)


        # B = segment_feature.shape[0]
        # seg_feature = torch.reshape(input=segment_feature, shape=(B*150, 2048, 48))
        # seg_feature = seg_feature.permute(0, 2, 1)
        # seg_feature = F.adaptive_max_pool1d(seg_feature, (1))
        # seg_feature = torch.reshape(input=seg_feature, shape=(B, 150, 48))


        # seg = self.pos_embedding_seg(seg_feature.permute(0, 2, 1)).permute(0, 2, 1)

        # output = primitive_feature + seg
        output = primitive_feature
        # output =  self.mlp_head(primitive_feature)
        # output = anchor_feature
        return output , xyzts_before
