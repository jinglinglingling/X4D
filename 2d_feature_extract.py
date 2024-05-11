
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from torchvision import transforms, datasets
import cv2 as cv
from torchsummary import summary


transforms = transforms.Compose([
    transforms.Resize(256),   
    transforms.CenterCrop(224),   

])



class MyDataset(Dataset):
    def __init__(self,input_data,transform=None):
        self.data=input_data
        self.transform = transform
    def __getitem__(self,index):
        
        img = self.data[index]

        img = torch.FloatTensor(img)

        img = img.permute(2, 0, 1)
        img = self.transform(img)

        # img = img.permute(1,2,0)

        # img = img.numpy()
        # cv.imwrite('/nvme/jinglinglin/Linglin_x/Video/read_test1.jpg',img)
# 
        return img
    
    def __len__(self):
        
        return len(self.data)#返回数据总数量

a = np.load('/nvme/jinglinglin/Linglin_x/Video/2d_frame_compose.npz', allow_pickle=True)
dd = a['arr_0']
# dd = a 
train_img = dd.reshape(-1,150,432,768,3)


train_data = MyDataset(train_img,transform=transforms)
my_train_feature=DataLoader(dataset=train_data,batch_size=8,shuffle=False)

resnet18 = models.resnet18(pretrained=True)
# resnet18.fc = nn.Linear(512, 10)
resnet18.fc = nn.Identity()

# resnet18.to("cuda0")
# resnet18.fc.add_module("fc",nn.Linear(10,20))
# x = torch.randn([1,3,224,224])
# feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
# output = feature_extractor(x) # output now has the features corresponding to input x
# print(output.shape)
# print(resnet18)
# output = resnet18(x)
# summary(resnet18,(3,224,224),device="cpu")
# exit()

for input_data in my_train_feature:

    output = resnet18(input_data.to(torch.float32))
    a = 1