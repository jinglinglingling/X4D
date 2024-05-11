from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import AutoProcessor, CLIPModel
from torchvision import transforms


import utils

from scheduler import WarmupMultiStepLR

# from datasets.AS_base_segment_temp import SegDataset
from datasets.AS_base_segment_v1 import SegDataset
# from datasets.AS_base_segment_submit import SegDataset
import models.AS_pptr_base_2d_new_segment3 as Models
from models.SnippetTransformer import SnippetEmbedding
from models.data_utils import *

from collections import Counter

from models.lovasz_losses import lovasz_softmax
import torchvision.models as models

from lightly.loss.ntx_ent_loss import NTXentLoss


from lr_scheduler import get_scheduler

version = 'act_clip_tempgra_resnet.pth'
exp_file = '/mnt/petrelfs/jinglinglin/xueying/Action_seg/HOI4D_ActionSeg-main/output/'
if not os.path.exists(exp_file):
    os.mkdir(exp_file)


transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
        transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
        # transforms.Resize([224,224]), 
        transforms.RandomHorizontalFlip(0.5),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
    ])


transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
        transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
        # transforms.Resize([224,224]), 
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
    ])


processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends
 
 
def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score
 
 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def contrastive_loss(feat1, feat2, obj_count, margin=0.1, max_margin=True, weight=3., reduction=True):
    sim_losses = 0. if reduction else []
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)
    for b_i in range(feat1.shape[0]):
        feat_2d, feat_3d, num_obj = feat1[b_i,:,:], feat2[b_i,:,:], obj_count[b_i]
        feat_2d, feat_3d = feat_2d[:num_obj,:], feat_3d[:num_obj,:]
        cos_scores = feat_2d.mm(feat_3d.t())
        diagonal = cos_scores.diag().view(feat_2d.size(0), 1)
        d1 = diagonal.expand_as(cos_scores)
        d2 = diagonal.t().expand_as(cos_scores)
        # feat_3d retrieval
        cost_3d = (margin + cos_scores - d1).clamp(min=0)
        # feat2d retrieval
        cost_2d = (margin + cos_scores - d2).clamp(min=0)
        cost_3d = cost_3d.to(feat1.device)
        cost_2d = cost_2d.to(feat1.device)
        # clear diagonals
        I = (torch.eye(cos_scores.size(0), device=torch.device('cpu')) > .5).to(feat1.device)
        cost_3d = cost_3d.masked_fill_(I, 0)
        cost_2d = cost_2d.masked_fill_(I, 0)
        topk = min(3,int(cost_3d.shape[0]))
        cost_3d = (torch.topk(cost_3d, topk, dim=1)[0])
        cost_2d = (torch.topk(cost_2d, topk, dim=0)[0])
        if reduction: 
            batch_loss = torch.sum(cost_3d) + torch.sum(cost_2d)
            sim_losses = sim_losses + batch_loss
        else: 
            batch_loss = torch.mean(cost_3d) + torch.mean(cost_2d)
            sim_losses.append(batch_loss)
    if reduction: 
        return weight * sim_losses/(torch.sum(obj_count))
    else:
        return weight * torch.tensor(sim_losses, device=torch.device('cuda'))

def train_one_epoch(model,CLIP,temporal_tf1,temporal_tf2, resnet18,resnet18_2, mix_model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq,weight):
    model.train()
    for param in CLIP.parameters():
        param.requires_grad = False
        
    temporal_tf2.train()
    temporal_tf1.train()
    resnet18.train()
    resnet18_2.train()
    mix_model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    f = []
    a_max = 0
    header = 'Epoch: [{}]'.format(epoch)

    import time
    # img_2d_list = []
    for clip, target,image_2d, tempgrad, segment_feature in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()

        image_con = []
        output_img = []
        
        a_time = time.time()

        B , L , C = image_2d.shape[0] , image_2d.shape[1] , image_2d.shape[2]
        image = image_2d.reshape((B*L,C,224,224))
        # inputs = processor(images=image, return_tensors="pt")
        # inputs = inputs.to(device)
        # output2 = CLIP.get_image_features(**inputs)
        image = image.to(device)
        output2 = resnet18_2(image)
        output3 = output2.reshape(B,L,512)
        output_img = temporal_tf1(output3, output3, output3)


        # print('aaaaa' , time.time() - a_time)
        b_time = time.time()

        tempgrad = tempgrad.reshape((B*L,C,224,224))
        tempgrad = tempgrad.to(device)
        output2 = resnet18(tempgrad)
        # output_tempgrad = output2.reshape(B,L,512)
        output3 = output2.reshape(B,L,512)
        output_tempgrad = temporal_tf2(output3, output3, output3)


        # print('bbbb' , time.time() - b_time)

        c_time = time.time()

        segment_feature = segment_feature.to(device)
        clip, target = clip.to(device), target.to(device)
        output_3d , xyzt = model(clip,segment_feature)

        output_fusion , fusion_feature , TG_output = mix_model(output_3d,output_img,output_tempgrad,target,xyzt,'train')

        # print('cccc' , time.time() - c_time)

        output = output_fusion[:,:150,:] 
        pre_2d = output_fusion[:,150:300,:] 
        # pre_temp = output_fusion[:,300:,:] 
        
        
        # Feature_3d = fusion_feature[:,:150,:] 
        # Featur_2d = fusion_feature[:,150:300,:] 
        # Featur_temp = fusion_feature[:,300:,:] 


        tempo_g = output_tempgrad.detach()
        # tempo_g = output_tempgrad
        cos = nn.CosineEmbeddingLoss(margin=0.2)

        output_img = output_img.reshape((output_3d.shape[0],150*512))
        tempo_g = tempo_g.reshape((output_3d.shape[0],150*512))

        loss_flag = torch.ones(output_img.shape[0]).cuda()
        COS_Loss = cos(output_img, tempo_g, loss_flag)



        #bat = [150,150,150,150,150,150,150,150]
        #bat = np.array(bat)
        #bat = torch.from_numpy(bat)
        #bat = bat.to(device)
        #sim2d3d_loss = contrastive_loss(output_tempgrad, output_3d, bat)

        #contras_loss =  F.mse_loss(output_3d, output_tempgrad)


        loss1 = lovasz_softmax(output.permute(0,2,1) ,target)
        loss1_1 = lovasz_softmax(pre_2d.permute(0,2,1) ,target)
        # loss1_2 = lovasz_softmax(pre_temp.permute(0,2,1) ,target)
        
        # loss = 0.8*criterion(target , output.permute(0,2,1))

        loss_tg = criterion(TG_output.permute(0,2,1), target)      
        loss2 = criterion(output.permute(0,2,1), target)      
        loss3 =  criterion(pre_2d.permute(0,2,1), target)   
        # loss4 =  criterion(pre_temp.permute(0,2,1), target)   
#####

        # loss =  0.25 * contras_loss + loss2 +  loss3# + 0.5 * loss1 #+ 0.5 * loss1_1 # +0.2 * sim2d3d_loss
        # loss =   loss2  +  loss3 #+ 0.25*sim2d3d_loss
        # loss = loss2 + loss3 + 0.5 * loss1 + 0.5 * loss1_1

        # loss = loss2 + loss3  + COS_Loss  + loss4 #+ loss_tg
        loss = loss2 + loss3  + COS_Loss  + loss_tg

        d_time = time.time()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=temporal_tf1.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=temporal_tf2.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=resnet18.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=mix_model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        
        output = torch.max(output,dim=-1)[1]
        # acc = torch.mean(torch.tensor(output==target,dtype=torch.float))
        output, target = output.cpu().numpy().astype(np.int32), target.cpu().numpy().astype(np.int32)
        acc = np.mean(output == target)
        # output, target = output.astype(torch.int32), target.astype(torch.int32)
        # acc = torch.mean(output == target)

        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))

        lr_scheduler.step()
        sys.stdout.flush()

        # print('dddd' , time.time() - d_time)


    state = {'3d':model.state_dict(),
         'temp1':temporal_tf1.state_dict(),
         '2d':resnet18.state_dict(),
         'merge':mix_model.state_dict()}
    torch.save(state, exp_file + version)



def evaluate(model,CLIP, temporal_tf1,temporal_tf2, resnet18,resnet18_2, mix_model, criterion, data_loader, device, len_test,pre,eval_max,epoch):
    model.eval()
    temporal_tf1.eval()
    temporal_tf2.eval()
    resnet18.eval()
    resnet18_2.eval()
    mix_model.eval()


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    acc_list = []
    acc_list_2d = []

    total_correct_class = [0] * 19
    total_class = [0] * 19

    with torch.no_grad():
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        edit = 0
        length = 0


        #checkpoint = torch.load('/mnt/petrelfs/jinglinglin/4D_HOI/Experiment/Sbatch_test/output/Last_day/s11_200.pth')
        #model.load_state_dict(checkpoint['3d']) # 这里checkpoint可以看为字典，和之前保存的state相对应
        #resnet18.load_state_dict(checkpoint['2d'])
        #mix_model.load_state_dict(checkpoint['merge'])
    
        for clip, target, image_2d, tempgrad, segment_feature in metric_logger.log_every(data_loader, 20, header):

            image_con = []
            output_img = []


            B , L , C = image_2d.shape[0] , image_2d.shape[1] , image_2d.shape[2]
            # image = image_2d.reshape((B*L,C,224,224))
            # image = image.to(device)
            # output2 = resnet18(image)
            # output_img = output2.reshape(B,L,512)
            output_img = torch.ones((B,L,512)).to(device)
            output_tempgrad = torch.ones((B,L,512)).to(device)
            # output_img = torch.ones((B,L,2048)).to(device)



            clip, target = clip.to(device), target.to(device)
            output_3d , xyzt = model(clip,segment_feature)


            output_fusion , fusion_feature, TG_output = mix_model(output_3d,output_img,output_tempgrad,target,xyzt,'test')

            output = output_fusion[:,:150,:] 

            # pre_2d = output_fusion[:,150:300,:] 
            # output_tempgrad = output_fusion[:,300:,:] 
            pre_2d = output


            loss_3d = criterion(output.permute(0,2,1), target)      

            # loss3_2d =  criterion(pre_2d.permute(0,2,1), target)   
            # loss_temp =  criterion(output_tempgrad.permute(0,2,1), target)    
            
            loss = loss_3d


            output = torch.max(output,dim=-1)[1]
            pre_2d = torch.max(pre_2d,dim=-1)[1]


            pre_2d = pre_2d.cpu().numpy().astype(np.int32)
            output, target = output.cpu().numpy().astype(np.int32), target.cpu().numpy().astype(np.int32)


            output_new = output.copy()
            
            for i in range (output.shape[0]):
                for j in range (1, output.shape[1]-1 ):
                    
                    a = output[i][j]
                    b = output[i][j-1]
                    c = output[i][j+1]

                    if a != b:
                        if b == c:
                            output_new[i][j] = b
            
            output_new2 = output_new.copy()
            output_new3 = output_new2.copy()

            for i in range (output_new2.shape[0]):
                for j in range (1, output_new2.shape[1]-1 ):
                    
                    a = output_new2[i][j]
                    b = output_new2[i][j-1]
                    c = output_new2[i][j+1]

                    if a != b:
                        if b == c:
                            output_new3[i][j] = b
            pre_2d = output
            output = output_new2

            pre.append(output)
            
            acc = np.mean(output == target)
            acc_2d = np.mean(pre_2d == target)
            # acc = torch.mean(torch.tensor(output==target,dtype=torch.float))
            acc_list.append(acc)
            acc_list_2d.append(acc_2d)
            for b in range(output.shape[0]):
                # print(output[b].shape)
                # print(target[b].shape)
                edit += edit_score(output[b], target[b])
            for b in range(output.shape[0]):
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(output[b], target[b], overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1


            batch_size = clip.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
    # pre = np.array(pre)
    

    metric_logger.synchronize_between_processes()
    total_acc = np.mean((np.array(acc_list)))
    total_acc_2d = np.mean((np.array(acc_list_2d)))
    edit = (1.0 * edit) / len_test
    print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1
    print("total acc:", total_acc)
    print("total acc_2d:", total_acc_2d)
    print('Max acc', eval_max)

    # if total_acc > eval_max:

    pre_out = np.concatenate(pre , 0)
    # np.save('/mnt/petrelfs/jinglinglin/4D_HOI/Experiment/Sbatch_test/output/Last_day/S11_output/%s.npy'%(epoch+40), pre_out)
    np.save(exp_file + str(epoch) + '.npy', pre_out)
    return total_acc

def top_ce_loss(gt_cls, pred_cls):


    # ce_loss = F.cross_entropy(pred_cls,gt_cls)
    
    loss = torch.nn.CrossEntropyLoss()
    ce_loss = loss(pred_cls,gt_cls)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) **2 * ce_loss).mean()
    loss = focal_loss 

    return loss


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    # device = torch.device('cpu') 

    print(device)
    
    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = SegDataset(root='/datasets/AS_data_base', train=True,transform =transforms_train)

    dataset_test = SegDataset(root='/datasets/AS_data_base', train=False,transform =transforms_test)

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    
    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                  mlp_dim=args.mlp_dim, num_classes=19)


    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    resnet18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18.fc = nn.Identity()

    resnet18_2 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18_2.fc = nn.Identity()


    fusion_model = getattr(Models, 'fusion_net')
    mix_model = fusion_model()
    temporal_tf1 = SnippetEmbedding(1, 512, 512, 512)
    temporal_tf2 = SnippetEmbedding(1, 512, 512, 512)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        temporal_tf1 = nn.DataParallel(temporal_tf1)
        temporal_tf2 = nn.DataParallel(temporal_tf2)
        resnet18 = nn.DataParallel(resnet18)
        resnet18_2 = nn.DataParallel(resnet18_2)
        mix_model = nn.DataParallel(mix_model)
    model.to(device)
    CLIP.to(device)
    temporal_tf1.to(device)
    temporal_tf2.to(device)
    resnet18.to(device)
    resnet18_2.to(device)
    mix_model.to(device)   
    


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pre_state = checkpoint['model'] 
        # for name in pre_state.keys():
        #     print(name)
        # update_dict = {k.replace('_encoder',''): v for k, v in pre_state.items() if k.startswith("tube_embedding.") or k.startswith("transformer1.") or k.startswith("transformer2.") or k.startswith("pos_embedding_encoder")} 
        # update_dict = {k.replace('_encoder',''): v for k, v in pre_state.items() if k.startswith("tube_embedding.") or k.startswith("transformer1.") or k.startswith("pos_embedding_encoder")} 
        update_dict = {k: v for k, v in pre_state.items() if k.startswith("module.tube_embedding.") or k.startswith("module.transformer1.") or k.startswith("module.transformer2.") or k.startswith("module.pos")}
        # update_dict = {k: v for k, v in pre_state.items() if k.startswith("tube_embedding.") or k.startswith("transformer1.") or k.startswith("transformer2.") or k.startswith("pos")}  
        for name in update_dict.keys():
            print(name)
        net_state_dict = model.state_dict()
        # for name in net_state_dict.keys():
        #     print(name)
        net_state_dict.update(update_dict)
        model.load_state_dict(net_state_dict)
        # print(pre_state['transformer1.layers.1.0.fn.fn.to_qkv.weight'])
        # print(model.state_dict()['transformer1.layers.1.0.fn.fn.to_qkv.weight'])

    criterion = nn.CrossEntropyLoss()

    lr = args.lr

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = torch.optim.SGD([
    {'params': model.parameters(),'lr': lr},
    {'params': temporal_tf1.parameters(),'lr': 0.05},
    {'params': temporal_tf2.parameters(),'lr': 0.05},
    {'params': resnet18.parameters(),'lr': 1e-4},
    {'params': resnet18_2.parameters(),'lr': 1e-4},
    {'params': mix_model.parameters(),'lr': 0.05}
    ],weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    # lr_scheduler = get_scheduler(optimizer)
    # temporal_tf1 = 1
    CLIP.eval()
 
    weight = 0
########
    print("Start training")
    start_time = time.time()
    acc = 0
    eval_max = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model,CLIP,temporal_tf1,temporal_tf2, resnet18, resnet18_2,mix_model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq, weight)
        # lr_scheduler.step()
        # if epoch > 55:
        pre = []
        acc = max(acc, evaluate(model,CLIP,temporal_tf1,temporal_tf2, resnet18,resnet18_2, mix_model, criterion, data_loader_test, device, len(dataset_test),pre,eval_max,epoch))
        eval_max = acc

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--seed', default=115, type=int, help='random seed')
    parser.add_argument('--model', default='PrimitiveTransformer', type=str, help='model')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=64, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=16, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=1, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=2048, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=1024, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 35,60], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # import h5py
    # with h5py.File('/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/datasets/AS_data_base/test_wolabel.h5','r') as f:
    #     a = 1
    # exit()
    # aaaa = np.load('/nvme/jinglinglin/Linglin_x/Action_seg/HOI4D_ActionSeg-main/pred.npy')
    # a = np.random.random((3,4))
    # idx = a[np.where( a < Max acc0.5)]
    
    # torch.set_num_threads(2)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    args = parse_args()
    main(args)
