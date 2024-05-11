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

trans_1 = transforms.Compose(
    [
        # d_utils.PointcloudToTensor(),
        # d_utils.PointcloudRotate(),
        # d_utils.PointcloudTranslate(0.5, p=1),
        # d_utils.PointcloudJitter(p=1),
        # d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
        # d_utils.PointcloudNormalize(),
        d_utils.PointcloudRandomInputDropout(p=1),
        
    ])



# @gif.frame#@gif.frame是GIF库用来创建帧序列的装饰器，紧接着的def gm(n)函数的输出就是一个PIL类
# def plott(points):

#     # plt.plot(x,y)
#     x = points[:, 0]  # x position of point
#     y = points[:, 1]  # y position of point
#     z = points[:, 2]  # z position of point
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x,   # x
#                y,   # y
#                z,   # z
#                c=z, # height data for color
#                cmap='rainbow',
#                marker="x")
#     ax.axis()
#     plt.show()

# def viz_matplot(points):
#     x = points[:, 0]  # x position of point
#     y = points[:, 1]  # y position of point
#     z = points[:, 2]  # z position of point
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     frame1 = ax.scatter(x,   # x
#                y,   # y
#                z,   # z
#                c=z, # height data for color
#                cmap='rainbow',
#                marker="x")
#     ax.axis()
#     plt.show()
#     plt.savefig('/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/Visualization/Flip_2.png')
#     return frame1


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
            # for filename in ['train1.h5', 'train2.h5', 'train3.h5','train4.h5']:
            for filename in ['train4.h5']:
                print(filename)
                # root = '/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/datasets/'
                # with h5py.File(root+'/'+filename,'r') as f:
                with h5py.File('/mnt/petrelfs/jinglinglin/4D_HOI/Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:
                    
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
                        # self.label.append(np.array(f['label']))

            self.label1 = np.load('/mnt/petrelfs/jinglinglin/4D_HOI/Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/new_label.npy')
                        
        else:
            for filename in ['train4.h5']:
            # for filename in ['test_wolabel.h5']:
                print(filename)
                # with h5py.File(root+'/'+filename,'r') as f:
                with h5py.File('/mnt/petrelfs/jinglinglin/4D_HOI/Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:
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

        # a = 1
        ##### load .png
        # if self.train:
        #     # for i in range (self.pcd.shape[0]):
        #     for i in range (10):

        #         img_root = '/mnt/petrelfs/jinglinglin/4D_HOI/2D_image_stream/%s'%i

        #         img_l = []
        #         for j in range (150):
        #             # files = sorted(glob.glob(os.path.join(root, '*/*%s' % 'png')))
        #             png_path = img_root + '/%s.png'%j
        #             color_img =cv.imread(png_path,1)
        #             img_l.append(color_img)
        #         immg = np.stack(img_l, axis=0)
        #         self.image_2d.append(immg)
        #     print('shape', len(self.image_2d))
        # else:

        #     # for i in range (2471,2971):
        #     for i in range (2970,2971):

        #         img_root = '/mnt/petrelfs/jinglinglin/4D_HOI/2D_image_stream/%s'%i
        #         img_l = []
        #         for j in range (150):
        #             # files = sorted(glob.glob(os.path.join(root, '*/*%s' % 'png')))
        #             png_path = img_root + '/%s.png'%j
        #             color_img =cv.imread(png_path,1)
        #             img_l.append(color_img)
        #         immg = np.stack(img_l, axis=0)
        #         self.image_2d.append(immg)


        # self.image_2d = np.concatenate(self.image_2d, axis=0)

        ############ Load .npy
        # if self.train:
        #     for index in range (self.pcd.shape[0]):
        #     # for index in range (100):
        #         context_2d = np.load('/mnt/petrelfs/jinglinglin/4D_HOI/image_2d/%s.npy'%index,allow_pickle=True)
        #         img = torch.FloatTensor(context_2d)

        #         img = img.permute(0, 3 , 1, 2)
        #         img = self.transform(img)
        #         img = img.numpy()

        #         if img.shape[0] < 25:
        #             dim = 25 - img.shape[0]
        #             pad_m = np.zeros([dim, img.shape[1] , img.shape[2] , img.shape[3]],dtype=np.float32 )
        #             img = np.vstack((img,pad_m))
        #             # img2 = img[[not np.all(img[i] == 0) for i in range(img.shape[0])], :]
        #         self.image_2d.append(img)
        #     print('train 2D_image')

        # else:
        #     for index in range (2471,2971):
        #     # for index in range (2):
        #         context_2d = np.load('/mnt/petrelfs/jinglinglin/4D_HOI/image_2d/%s.npy'%index,allow_pickle=True)
        #         img = torch.FloatTensor(context_2d)

        #         img = img.permute(0, 3 , 1, 2)
        #         img = self.transform(img)
        #         img = img.numpy()

        #         if img.shape[0] < 25:
        #             dim = 25 - img.shape[0]
        #             pad_m = np.zeros([dim, img.shape[1] , img.shape[2] , img.shape[3]],dtype=np.float32 )
        #             img = np.vstack((img,pad_m))

        #         self.image_2d.append(img)
        #     print('test 2D_image')




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
        
        if self.train:
            img_index = index
        else:
            img_index = index + 2471
     
        img_root = '/mnt/petrelfs/jinglinglin/4D_HOI/2D_image_stream/%s'%img_index

        img_l = []
        img_l2 = []
        for j in range (150):
            # files = sorted(glob.glob(os.path.join(root, '*/*%s' % 'png')))
            png_path = img_root + '/%s.png'%j
            color_img_0 =cv.imread(png_path,1)
            color_img = self.transform(color_img_0)
            color_img2 = self.transform(color_img_0)
            from  torchvision import utils as vutils
            vutils.save_image(color_img, '/mnt/petrelfs/jinglinglin/xueying/1.png', normalize=True)
            vutils.save_image(color_img2, '/mnt/petrelfs/jinglinglin/xueying/2.png', normalize=True)
            img_l.append(color_img)
            img_l2.append(color_img2)
        img = torch.stack(img_l, axis=0)
        img2 = torch.stack(img_l2, axis=0)
        img_final = torch.cat([img.unsqueeze(1), img2.unsqueeze(1)], dim=1)


        TG_feature = torch.zeros(img_final.shape)
        TG_f = torch.diff(img_final , dim = 0)
        TG_feature[0] = TG_f[0]
        TG_feature[1:] = TG_f[0:]
        # img = torch.zeros((150,3,224,224))
        # # image = torch.tensor(image)
        # for i in range (image.shape[0]):

        #     # cv.imwrite('/mnt/petrelfs/jinglinglin/4D_HOI/before.png',image[i] )
        #     img[i] = self.transform(image[i])
        #     # from  torchvision import utils as vutils
        #     # vutils.save_image(img[i], '/mnt/petrelfs/jinglinglin/4D_HOI/after_norm.png', normalize=True)

        #     # img[i] = img[i].numpy()
        #     # cv.imwrite('/mnt/petrelfs/jinglinglin/4D_HOI/after.png',img[i] )
        #     # cv.imwrite('/mnt/petrelfs/jinglinglin/4D_HOI/after*255.png',img[i]*255 )


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
        
        # pc = trans_1(point)        pc = trans_1(pc)
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




        return pc.astype(np.float32), label.astype(np.int64), img_final.to(torch.float32)


if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/AS_data_base_h5', train=False)

