# HOI4D competition
Please check out the HOI4D Challenge and related datasets (Including RGB videos and train.h5 files) on the website www.hoi4d.top and https://github.com/leolyliu/HOI4D-Instructions

# Pre-trained segmentation feature

In the same scenario, the semantic segmentation features may help the scene understanding of 4d tasks, we provide these features from pre-trained network, which can be downloaded on (https://cuhko365-my.sharepoint.com/personal/120090452_link_cuhk_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F120090452%5Flink%5Fcuhk%5Fedu%5Fcn%2FDocuments%2Ffeat&ga=1).

If this is not necessary, please comment the related segment_feature code, as this will only slightly affect the final performance and is not our contribution.

### Get Human-object Interaction 2D images

1. First you need to install [ffmpeg](https://ffmpeg.org/).
2. Then run ``` HOI4D-Instructions-main/python utils/decode.py``` to generate RGB and depth images from download videos.

### Select 3d aligned images 

  run ``` HOI4D-Instructions-main/python utils/read_json.py


### Train the action_seg network

1. Modify the load file path in HOI4D_ActionSeg-main/datasets/AS_base.py
2. Put train1,2,3,4.h5 and test data files in HOI4D_ActionSeg-main/datasets/AS_data_base 
3. Put aligned images in HOI4D_ActionSeg-main/datasets/2D_image_stream 
4. Put pre-trained segmentation features in HOI4D_ActionSeg-main/datasets/segment_feature
5. Run ``` HOI4D_ActionSeg-main/pptr+2d_new_segment_clip_temp_resnet_slidewindow_X4D.py```


#### Core file

The core files are
dataset process: AS_base.py 
model: AS_pptr_base_2d_new_segment3
train and inference: pptr+2d_new_segment_clip_temp_resnet_slidewindow_X4D.py