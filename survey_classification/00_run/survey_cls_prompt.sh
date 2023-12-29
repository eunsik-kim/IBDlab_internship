#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # Cores per node
#SBATCH --partition=gpu1       # Partition Name (cpu1, gpu1, hgx)
##
#SBATCH --job-name=sv_cls
#SBATCH -o SLURM.%N.%j.out         # STDOUT
#SBATCH -e SLURM.%N.%j.err         # STDERR
##
#SBATCH --gres=gpu:1       # GPU 1~4 (you need)
#SBATCH --cpus-per-task=8

image_path=/home2/ENROOT_IMAGES/nvcr.io+nvidia+pytorch+21.09-py3.sqsh
image_name=${image_path##*/} 
container_name=experiment1
container_path="/scratch/enroot/$UID/data/" 
echo container_name: $container_name
enroot create -f -n $container_name $image_path 
enroot start \
--root \
--rw \
-m $HOME:$HOME \
$container_name \
/bin/bash -c 'pip install --upgrade pip && \
            pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 &&\
            pip install transformers tensorflow accelerate wandb datasets scipy scikit-learn openpyxl seaborn flax && \
            export TF_ENABLE_ONEDNN_OPTS=0 && \
            cd /home2/eunsik12/lab/KIHASA/02_code && \
            wandb login --relogin "96e213bec6e0b2c89e5254f9b8ab09d6270c24b3" && \
            python train_prompt.py --data_dir ../03_model/mentalbertavg_prompt20 --data_type eng --avg_mode --n_tokens 20'
enroot remove --force $(enroot list)

# 1. --data_dir ../03_model/mentalbertavg_prompt --data_type eng --avg_mode --n_tokens 20'
# 2. --data_dir ../03_model/mentalbertavg_prompt --data_type eng --avg_mode --n_tokens 100'


#/home2/ENROOT_IMAGES/ubuntu22.04-py3.10-cuda11.8-pytorch2.1.0.dev20230513+cu118-jupyter.sqsh
#/home2/ENROOT_IMAGES/nvcr.io+nvidia+pytorch+21.09-py3.sqsh
