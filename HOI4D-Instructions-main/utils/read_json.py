import json
import os
import subprocess
import sys
import glob


import numpy as np
from PIL import Image

import h5py
import cv2 as cv





target = []
for filename in ['train1.h5', 'train2.h5', 'train3.h5','train4.h5']:
    print(filename)
    with h5py.File('./HOI4D_ActionSeg-main/datasets/AS_data_base/'+filename,'r') as f:

        aa = f.keys()
        target.append(np.array(f['label']))
label_target = np.concatenate(target,axis=0)
aa =1 


root = './4D_HOI/HOI4D_annotations'
with open('./4D_HOI/HOI4D-Instructions-main/release.txt', 'r') as f:

    rgb_list = [os.path.join(root, i.strip(),'action') for i in f.readlines()]
    max_time = 1
    dur_time = 0
    duration = []

    idd_all = []
    label_all = []

    start_all = []
    end_all = []

    idd_max = 0

    # action = None
    action_name = []

    task_all = []

    ddd= [ ]

    for file_index in range (label_target.shape[0]):

        rgb = rgb_list[file_index]

        file_label = label_target[file_index]
        action = None

        with open(rgb + '/color.json','r',encoding = 'utf-8') as fp:
       
            data = json.load(fp)

        
        idd = []
        label = []
        start = []
        end = []
        action_n = []

        start_time = []
        end_time = []



        for k ,v in data.items():

            if k == 'markResult':
                for duc in v:
                    if duc == 'marks':
                        ff = v['marks']
                        for dr in ff:
                            idd.append(dr['id'])
                            if dr['id'] > idd_max:
                                idd_max = dr['id']
                                print(idd_max)

                            if dr['event'] != action:
                                action = dr['event']
                                action_n.append(dr['event'])
                                start_time.append(dr['hdTimeStart'])
                                end_time.append(dr['hdTimeEnd'])
                            # if idd_max == 25:
                            #     abc = 1
                            average_time = (dr['hdTimeEnd'] + dr['hdTimeStart'])/2

                            # a = 20/300
                            # average_time = np.rint(average_time / a )
                            # if average_time > 299:
                            #     average_time = 299

                            # if dur['event'] != action:
                            #     action = dur['event']
                            #     action_n.append(dur['event'])
                                # label.append(average_time)      

                                  


            elif k == 'events':
                for dur in v:
                    idd.append(dur['id'])

                    if dur['id'] > idd_max:
                        idd_max = dur['id']
                        # print(idd_max)
                    # if idd_max == 25:
                    #     abc = 1

                    # if dur['event'] != action:
                    #     action = dur['event']

                    # label.append(dur['event'])
                    # time = dur['endTime'] - dur['startTime']
                    average_time = (dur['endTime'] + dur['startTime'])/2



                    if dur['event'] != action:
                        action = dur['event']
                        action_n.append(dur['event'])
                        start_time.append(dur['startTime'])
                        end_time.append(dur['endTime'])

                    # a = 20/300
                    # average_time = np.rint(average_time / a )
                    # if average_time > 299:
                    #     average_time = 299

                    # # action_n.append(dur['event'])

                    # if dur['event'] != action:
                    #     action = dur['event']
                    #     action_n.append(dur['event'])
                    #     label.append(average_time)
                        # if time < max_time:
                        #     max_time = time
                        #     print('time', max_time)
        ddd.append(1)

        ind = 0
        idx_img = 0
        task= []
        label_name = []
        aa = -1
        for label_index in range (file_label.shape[0]):



                j = file_label[label_index]

                if j != aa:
                    if idx_img != 0:
                        task.append(idx_img)
                    label_name.append(j)
                    idx_img = 1
                    aa = j
                elif j == aa:
                    idx_img += 1
                
                if label_index == 149:
                    if file_label[label_index] != file_label[label_index-1]:

                        # task[-1] = task[-1]+1
                        if len(label_name) != len(action_n):

                            task[-1] = task[-1]+1
                            # print('aaa')
                        else:
                            task.append(idx_img)
                            # print('bbb')

                    else:
                        task.append(idx_img)

        if file_index == 2673:
            task[-2] = task[-2] + task[-1]
            task = task[:-1]
            

        if len(task) != len(start_time):
            a = 1

        a = sum(task)
        if a != 150:
            a = 1


            
        # label_all.append(label)
        idd_all.append(idd)
        action_name.append(action_n)
        task_all.append(task)
        start_all.append(start_time)
        end_all.append(end_time)

        fp.close()   

    if len(task_all) != 2971:
        a = 1



file_rate = 20/300
file_name = 'jpg'
# root = '/nvme/jinglinglin/Linglin_x/video_mp4/HOI4D_release'
root = './Video/HOI4D_release'
with open('./HOI4D-Instructions-main/release.txt', 'r') as f:
    rgb_list = [os.path.join(root, i.strip()) for i in f.readlines()]
    img_data = []
    label_idx = 0

    max_d = 0

    for image_index in range (len(task_all)):

        print(image_index)
        
        # image_index = 255
        rgb = rgb_list[image_index]

        task = task_all[image_index]

        if image_index > -1:

            pair_label = label_target[image_index]
            start_t = start_all[image_index]
            end_t = end_all[image_index]


            files = sorted(glob.glob(os.path.join(rgb, '*/*%s' % file_name)))

            image_idx = 1
            pair_idx = 0
            temporal_img = []

            if image_index == 114:
                a = 1
            abc = []
            for j in range (len(task)):

                tt = end_t[j] - start_t[j]
                task_time = tt / (task[j] + 1)

                for bc in range(task[j]):
                    
                    # start_id = (start_t[j] + start_t[j+1]) /2
                    splt_time = start_t[j] + task_time*bc + task_time
                    file_t = np.rint(splt_time / file_rate)
                    abc.append(file_t)

            # abc = 

            for i in range (len(abc)):

                indx = int(pair_label[i])

                img_no = int(abc[i])
                img_path = files[img_no]
                color_img =cv.imread(img_path,1)

                resized_img = cv.resize(color_img, (int(color_img.shape[1]/4),int(color_img.shape[0]/4)) , interpolation = cv.INTER_CUBIC)

                color_img = resized_img.reshape(1 , resized_img.shape[0] , resized_img.shape[1] , 3)

                # temporal_img.append(color_img)

                save_path = './2D_image_stream/%s'%image_index
                if os.path.exists(save_path):
                    # np.save(save_path+'/%s.png'%image_index,color_img)
                    cv.imwrite(save_path+'/%s.png'%i,resized_img )
                else:
                    os.mkdir(save_path)
                    # np.save(save_path+'/%s.png'%image_index,color_img)
                    cv.imwrite(save_path+'/%s.png'%i,resized_img )
                    

print('Over')
        

