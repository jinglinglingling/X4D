#!/bin/bash

#SBATCH --quotatype=reserved
#SBATCH --job-name=X4D
#SBATCH --gres=gpu:1
#SBATCH -o 20210827-%j.out


python -u ./HOI4D_ActionSeg-main/pptr+2d_new_segment_clip_temp_resnet_slidewindow_X4D.py