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
from point_4d_convolution import *
from transformer import *
from pst_convolutions import *
from pointnet import pointnet
import torchvision.transforms as transforms
from . import data_utils as d_utils

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class fusion_net(nn.Module):
    def __init__(self ):                                            
        super().__init__()


        self.project1 = nn.Sequential(
                nn.Linear(2048, 1024, bias = False),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512, bias = False)
                ) 


        self.project2 = nn.Sequential(
                nn.Linear(512, 256, bias = False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128, bias = False)
                ) 

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

        self.mlp_out = nn.Sequential(
            nn.LayerNorm(1024),  #1024  2560
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 19),
        )

        self.mlp_out_new = nn.Sequential(
            nn.LayerNorm(2176),  #1024  2560 2176
            nn.Linear(2176, 512),
            nn.GELU(),
            nn.Linear(512, 19),
        )


        self.config = BertConfig(
                 hidden_size=2176,  #1024 2560 2176
                 num_hidden_layers=12,
                 num_attention_heads=16, #16 40 32
                 type_vocab_size=2)

        self.encoder = BertEncoder(self.config)


    def forward(self, input_3d, input_2d,label):

        a_list = []
        # clip = self.mlp_head(input_3d)
        clip = input_3d

        im_task = []

        for idx in range (input_3d.shape[0]):

            image = input_2d[idx]

            out_image = self.project2(image)


        # for image in input_2d:
            
        #     # print('new',idx)
        #     out_image = self.project2(image)


            task = []
            task1 = []
            idx_img = -1

            label_x = label[idx]

            for idj in range (label_x.shape[0]):

                aa = -1
                j = label_x[idj]
                # print('j',j)
                # task1.append(out_image[j])
                if j != aa:
                    idx_img += 1
                    if idx_img > (image.shape[0] - 1):

                        idx_img = idx_img -1

                    task.append(out_image[idx_img])
                    aa = j
                elif j == aa:
                    task.append(out_image[idx_img])

            img_stack = torch.stack(task,0)
            im_task.append(img_stack)
            # a_list.append(out_image)

        
        final_image_2d = torch.stack(im_task,0)

        final_fusion = torch.cat([clip,final_image_2d],2)
        # final_fusion = torch.stack([clip,final_image_2d],2)


        head_mask = ([None] * self.config.num_hidden_layers)


        # extended_attention_mask = torch.zeros(final_fusion.shape[0] ,1 , final_fusion.shape[1] , final_fusion.shape[1]  ).to(final_fusion.device)
        extended_attention_mask = torch.ones(final_fusion.shape[0] ,1 , final_fusion.shape[1] , final_fusion.shape[1]  ).to(final_fusion.device)


        encoder_outputs = self.encoder(
            final_fusion,
            extended_attention_mask,
            head_mask=head_mask
        )

        out = encoder_outputs[0]

        # output = self.mlp_out(out)
        output = self.mlp_out_new(out)

        return output


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

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )


    def forward(self, input):



        # 4d BACKBONE
        # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        # xyzs = torch.zeros(8,150,64,3)
        # features = torch.zeros(8,150,64,2048)

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

        # output =  self.mlp_head(primitive_feature)
        output = primitive_feature
        return output
