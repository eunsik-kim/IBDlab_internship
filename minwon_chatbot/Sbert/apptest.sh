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
container_name=fastapitest
container_path="/scratch/enroot/$UID/data/" 
echo container_name: $container_name
enroot create -f -n $container_name $image_path 
enroot start \
--root \
--rw \
-m $HOME:$HOME \
$container_name \
/bin/bash -c 'pip install --upgrade pip && \
            pip install sentence_transformers pandas &&\
            pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
            cd /home2/eunsik12/Sbert && \
            python app.py'
enroot remove --force $(enroot list)
