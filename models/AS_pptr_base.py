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
# from point_4d_convolution import *
from point_4d_convolution_segment import *
from transformer import *
from pst_convolutions import *
from pointnet import pointnet
import torchvision.transforms as transforms
from . import data_utils as d_utils



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


    def forward(self, input,segment_feature):



        # 4d BACKBONE
        # [B, L, N, 3]
        # xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]
        xyzs, features,anchor_idx = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        device = raw_feat.get_device()
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]
                                                                                      # [B, L,   n, C]
        features = torch.reshape(input=raw_feat, shape=(raw_feat.shape[0], raw_feat.shape[1]*raw_feat.shape[2], raw_feat.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)


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



        embedding = xyzts + features
        
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
        primitive_feature = self.transformer2(primitive_feature) # B. L*4, C

        # primitive_feature = primitive_feature.reshape(B*L, 8, C)
        # primitive_feature = primitive_feature.permute(0,2,1)
        # primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))
        # primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L, C))  

        # output = torch.max(input=primitive_feature, dim=1, keepdim=False, out=None)[0]

        output =  self.mlp_head(primitive_feature)

        return output




class LongPrimitiveTransformer(nn.Module):
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

        self.pointnet = pointnet()

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):


       # 3d BACKBON

        input_copy = input.clone()     #b * 150 * 2048 * 3
        B1, L1, N1, C1 = input_copy.size()

        S = 30
        index = torch.tensor(random.sample(range(L1), S), device=device)
        index = index.long()
        input_copy = torch.index_select(input_copy, 1, index)   #b * 20 * 2048 * 3


        B1, L1, N1, C1 = input_copy.size()

        pointnet_input = torch.reshape(input=input_copy, shape=(B1 * L1, N1, C1))   # 40 * 2048 * 3


        pointnet_output = self.pointnet(pointnet_input.transpose(1, 2))             #b*l  * 128 * 2048

        BL, N2, C2 = pointnet_output.size()                                     # 40 , 128 , 2048 （120）

        pointnet_output = torch.reshape(input=pointnet_output, shape=(BL*5, -1, C2))  # [B*L*4, n', C] #BL4 n 2048
        # pointnet_output = self.emb_relu1(pointnet_output)
        # pointnet_output = self.transformer1(pointnet_output)  # [B*L*4, n', C]

        pointnet_output = pointnet_output.permute(0, 2, 1)
        pointnet_output = F.adaptive_max_pool1d(pointnet_output, (1))  # B*l*4, C, 1 . 
        pointnet_output = torch.reshape(input=pointnet_output, shape=(B1, L1 * 5, C2))  # [B, L*4, C] 



       # 4d BACKBONE
        # [B, L, N, 3]
        # xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]
        xyzs, features,anchor_idx= self.tube_embedding(input)

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features  # 150,64,2048

        point_feat = torch.reshape(input=raw_feat, shape=(B * L * 4, -1, C))  # [B*L*4, n', C]  64,2048
        point_feat = self.emb_relu1(point_feat)
        point_feat = self.transformer1(point_feat)  # [B*L*4, n', C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L * 4, C))  # [B, L*4, C] 


        # Integrate
        primitive_feature = torch.cat((primitive_feature, pointnet_output), dim=1)


        anchor_feature = torch.reshape(input=primitive_feature, shape=(B*L, 5, C))
        anchor_feature = anchor_feature.permute(0, 2, 1)
        anchor_feature = F.adaptive_max_pool1d(anchor_feature, (1))
        anchor_feature = torch.reshape(input=anchor_feature, shape=(B, L, C))


        primitive_feature = self.emb_relu2(anchor_feature)
        output = self.transformer2(primitive_feature)
        # output = self.transformer2(primitive_feature)[:,:L*4,:]  # B. L*4, C

        # output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output

# 750 900 1050
# 600
# 150 300 