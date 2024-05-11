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

#

def get_mapping(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict


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

        if self.train:
            for filename in ['train1.h5', 'train2.h5', 'train3.h5','train4.h5']:
            # for filename in ['train4.h5']:
                print(filename)
                # root = '/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/datasets/'
                # with h5py.File(root+'/'+filename,'r') as f:
                with h5py.File('/mnt/petrelfs/jinglinglin/4D_HOI/Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:
                    
                    # if filename == 'train4.h5':
                    if filename == '1.h5':
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
            # for filename in ['train4.h5']:
            for filename in ['test_wolabel.h5']:
                print(filename)
                # with h5py.File(root+'/'+filename,'r') as f:
                with h5py.File('/mnt/petrelfs/jinglinglin/4D_HOI/Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:
                    bb = f.keys()
                    # self.pcd.append(np.array(f['pcd'][-500:]))
                    # self.center.append(np.array(f['center'][-500:]))
                    # self.label.append(np.array(f['label'][-500:]))
                    
                    # aacc2 = (np.array(f['label'][-500:]))
                    self.pcd.append(np.array(f['pcd']))

                    self.center.append(np.array(f['center']))

                    lab = np.zeros([f['center'].shape[0],f['center'].shape[1]])
                    self.label.append(np.array(lab))

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
        
        if np.random.uniform(0, 1) > 0.5:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(150,2048,3), -1*0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center
        
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
        

        # if np.random.uniform(0, 1) > 0.5:
        #     pc = pc - center
        #     jittered_data = np.clip(0.01 * np.random.randn(150,2048,3), -1*0.05, 0.05)
        #     jittered_data += pc
        #     pc = pc + center

        # scale = np.random.uniform(0.8, 1.2)
        # pc = (pc - center) * scale + center
        





        return pc

    def connect(self, pc, train_img):

        a = 1 

    def __getitem__(self, index):
        pc = self.pcd[index]
        center_0 = self.center[index][0]
        label = self.label[index]

        # self.image_2d = []
        # image = self.image_2d[index]
        


        # img = torch.zeros((150,3,224,224))
        # # image = torch.tensor(image)
        # for i in range (image.shape[0]):

        #     # cv.imwrite('/mnt/petrelfs/jinglinglin/4D_HOI/before.png',image[i] )
        #     img[i] = self.transform(image[i])
        # # img = torch.tensor(img)
        #     # from  torchvision import utils as vutils
        #     # vutils.save_image(img[i], '/mnt/petrelfs/jinglinglin/4D_HOI/after.png', normalize=False)


            # img[i] = img[i].numpy()
            # cv.imwrite('/mnt/petrelfs/jinglinglin/4D_HOI/after.png',img[i] )
            # cv.imwrite('/mnt/petrelfs/jinglinglin/4D_HOI/after*255.png',img[i]*255 )


        # if self.train:
            # pc, img = self.connect(pc,self.train_img[index])
        # context_2d = np.load('/nvme/jinglinglin/Linglin_x/2D_images/%s.npy'%index,allow_pickle=True)
        # img = torch.FloatTensor(context_2d)

        # img = img.permute(0, 3 , 1, 2)
        # img = self.transform(img)
        # img = img.numpy()

        # self.image_2d.append(img)

        # print('index' , index)

        # frames = [ ]
        # for i in range (0,149):
        #     # plt.clf()
        #     frame1 = plott(pc[i])
        #     # frame = viz_matplot(clip[0][i])
        #     # frame = clip[2][145]
        #     # pcd = o3d.geometry.PointCloud()
        #     # pcd.points = o3d.utility.Vector3dVector(frame)
    #     # o3d.io.write_point_cloud("/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/sync4.ply", pcd)
            
        #     # exit()
        #     frames.append(frame1)
        # gif.save(frames,'/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/Visualization/before_normalize_v2.gif',duration=50)

   
        if self.train:
            
            pc = self.augment(pc, center_0)
            # frame = viz_matplot(pc[0])
        

            

            # pc = pc[0]
            # pc = pc.numpy()
        
        # pc = trans_1(point)
        # pc = trans_1(pc)
        # pc = pc.numpy()
        


        # for i in range (0,149):
        #     # plt.clf()
        #     frame1 = plott(pc[i])
        #     # frame = viz_matplot(clip[0][i])
        #     # frame = clip[2][145]
        #     # pcd = o3d.geometry.PointCloud()
        #     # pcd.points = o3d.utility.Vector3dVector(frame)
        #     # o3d.io.write_point_cloud("/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/sync4.ply", pcd)
            
        #     # exit()
        #     frames.append(frame1)
        # gif.save(frames,'/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/Visualization/Normalize_v2.gif',duration=50)




        return pc.astype(np.float32), label.astype(np.int64)

if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/AS_data_base_h5', train=False)

