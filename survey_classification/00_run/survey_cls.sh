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
container_name=experiment13
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
            python train.py --data_dir ../03_model/mentalbertavg_ft2_skip3_15 --data_type eng --avg_mode --drop_epoch 7 --n_drop 50 --skip_mode'
enroot remove --force $(enroot list)

# 1. --data_dir ../03_model/mentalbertast_ft_sparse --data_type eng --sparse_mode --ast_mode'
# 2. --data_dir ../03_model/mentalbertavg_ft_sparse --data_type eng --sparse_mode --avg_mode'
# 3. --data_dir ../03_model/mentalbertmax_ft_sparse --data_type eng --sparse_mode --max_mode'
# 4. --data_dir ../03_model/mentalbertcls_ft_sparse --data_type eng --sparse_mode --cls_mode'
# 5. --data_dir ../03_model/mentalbertavg_ft_recon --data_type eng --recon_mode --avg_mode'
# 6. --data_dir ../03_model/sentencebertavg_ft_recon --model_name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --data_type kor --recon_mode --avg_mode'
# 7. --data_dir ../03_model/sentencebertavg_ft_sparse --model_name 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' --data_type kor --sparse_mode --avg_mode'
# 8. --data_dir ../03_model/mentalbertavg_ft --data_type eng --avg_mode'
# 9. --data_dir ../03_model/mentalbertast_ft --data_type eng --ast_mode'
# 10. --data_dir ../03_model/mentalbertavg_ft_skip --data_type eng --avg_mode --skip_mode'
# 11. --data_dir ../03_model/sentencebertavg_ft_skip --model_name 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' --data_type kor --avg_mode'
# 12. --data_dir ../03_model/mentalbertavg_ft_skip3_15 --data_type eng --avg_mode --drop_epoch 3 --n_drop 15 --skip_mode'
# 13. --data_dir ../03_model/mentalbertavg_ft_skip7_50 --data_type eng --avg_mode --drop_epoch 7 --n_drop 50 --skip_mode'
# 14. --data_dir ../03_model/psychbertbertavg_ft --model_name 'mnaylor/psychbert-cased' --data_type eng --avg_mode'
# 15. --data_dir ../03_model/psychbertbertavg_ft_recon --model_name 'mnaylor/psychbert-cased' --data_type eng --avg_mode --recon_mode' 
# 16. --data_dir ../03_model/psychbertbertavg_ft_skip --model_name 'mnaylor/psychbert-cased' --data_type eng --avg_mode --skip_mode' 


#/home2/ENROOT_IMAGES/ubuntu22.04-py3.10-cuda11.8-pytorch2.1.0.dev20230513+cu118-jupyter.sqsh
#/home2/ENROOT_IMAGES/nvcr.io+nvidia+pytorch+21.09-py3.sqsh
