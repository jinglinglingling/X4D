import os
import sys
import numpy as np
import random
# import open3d as o3d
# from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py
import torchvision.transforms as transforms
from . import data_utils as d_utils

import cv2 as cv
from pyquaternion import Quaternion
from math import *
import torch

# import gif
# import matplotlib.pyplot as plt





class SegDataset(Dataset):
    def __init__(self, root=None, train=True,transform=None):
        super(SegDataset, self).__init__()

        self.train = train

        self.pcd = []
        self.center = []
        self.label = []
        self.transform = transform

        self.image_2d = []

        self.ind = []

        self.segment = []
        if self.train:
            for filename in ['train1.h5', 'train2.h5', 'train3.h5','train4.h5']:
            # for filename in ['train4.h5']:
                print(filename)

                with h5py.File('./HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:
                    
                    if filename == 'train4.h5':
                    # if filename == '1.h5':
                        aa = f.keys()
                        self.pcd.append(np.array(f['pcd'][:-500]))
                        self.center.append(np.array(f['center'][:-500]))
                        self.label.append(np.array(f['label'][:-500]))
                        # self.label.append(np.array(f['label']))
                        
                    else:
                        
                        aa = f.keys()
                        self.pcd.append(np.array(f['pcd']))
                        self.center.append(np.array(f['center']))
                        self.label.append(np.array(f['label']))

                        
        else:
            for filename in ['train4.h5']:
            # for filename in ['test_wolabel.h5']:
                print(filename)
                # with h5py.File(root+'/'+filename,'r') as f:
                with h5py.File('./Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:
                    bb = f.keys()
                    self.pcd.append(np.array(f['pcd'][-500:]))
                    self.center.append(np.array(f['center'][-500:]))
                    self.label.append(np.array(f['label'][-500:]))
                    
                    # aacc2 = (np.array(f['label'][-500:]))
                    # self.pcd.append(np.array(f['pcd']))

                    # self.center.append(np.array(f['center']))

                    # lab = np.zeros([f['center'].shape[0],f['center'].shape[1]])
                    # self.label.append(np.array(lab))

        self.pcd = np.concatenate(self.pcd, axis=0)
        self.center = np.concatenate(self.center,axis=0)
        self.label = np.concatenate(self.label,axis=0)

      

    def __len__(self):
        return len(self.pcd)

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 20
        
        # if flip:
        # pc = (pc - center)
        # pc[:,:,0] *= -1
        # pc += center
        # else:
        
        # if np.random.uniform(0, 1) > 0.5:
        #     pc = pc - center
        #     jittered_data = np.clip(0.01 * np.random.randn(150,2048,3), -1*0.05, 0.05)
        #     jittered_data += pc
        #     pc = pc + center

        # scale = np.random.uniform(0.8, 1.2)
        # pc = (pc - center) * scale + center
        
        # rot_axis = np.array([0, 1, 0])
        # rot_angle = np.random.uniform(np.pi * -0.01, np.pi * 0.01)
        # q = Quaternion(axis=rot_axis, angle=rot_angle)
        # R = q.rotation_matrix

        # pc = np.dot(pc - center, R) + center


        # coord_min = np.min(pc[:,:,:3], axis=0)
        # coord_max = np.max(pc[:,:,:3], axis=0)
        # coord_diff = coord_max - coord_min
        # translation = np.random.uniform(-0.01, 0.01, size=(3)) * coord_diff

        # pc = pc - center
        # pc+= translation
        # pc = pc + center


        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * -0.01, np.pi * 0.01)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center


        coord_min = np.min(pc[:,:,:3], axis=0)
        coord_max = np.max(pc[:,:,:3], axis=0)
        coord_diff = coord_max - coord_min
        translation = np.random.uniform(-0.01, 0.01, size=(3)) * coord_diff

        pc = pc - center
        pc+= translation
        pc = pc + center
        

        if np.random.uniform(0, 1) > 0.5:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(150,2048,3), -1*0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center
        





        return pc



    def __getitem__(self, index):
        pc = self.pcd[index]
        center_0 = self.center[index][0]
        label = self.label[index]

        # self.image_2d = []
        # image = self.image_2d[index]
        

        if self.train:
            img_index = index
        else:
            img_index = index + 2471
            # print('img_index',img_index)
     
        img_root = './Action_seg/HOI4D_ActionSeg-main/datasets/2D_image_stream/%s'%img_index

        img_l = []
        img_l2 = []
        for j in range (150):
            # files = sorted(glob.glob(os.path.join(root, '*/*%s' % 'png')))
            png_path = img_root + '/%s.png'%j
            color_img_0 =cv.imread(png_path,1)
            color_img = self.transform(color_img_0)
            color_img2 = self.transform(color_img_0)
            img_l.append(color_img)
            img_l2.append(color_img2)

        img = torch.stack(img_l, axis=0)
        img2 = torch.stack(img_l2, axis=0)
        img_final = torch.cat([img.unsqueeze(1), img2.unsqueeze(1)], dim=1)
        


        TG_feature = torch.zeros(img_final.shape)
        TG_f = torch.diff(img_final , dim = 0)
        TG_f += 255.0
        TG_f /= 2.0
        
        TG_feature[0] = TG_f[0]
        TG_feature[1:] = TG_f[0:]
        
        seg_root = './Action_seg/HOI4D_ActionSeg-main/datasets/segment_feature/%s.npy'%img_index

        file = np.load(seg_root,allow_pickle=True)
        seg_feat = file
        seg_feat = torch.tensor(seg_feat)
   
        if self.train:
            
            pc = self.augment(pc, center_0)
            # frame = viz_matplot(pc[0])
        




        return pc.astype(np.float32), label.astype(np.int64), img_final.to(torch.float32),TG_feature,seg_feat.to(torch.float32)


if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/AS_data_base_h5', train=False)

