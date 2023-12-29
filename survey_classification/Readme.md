**survey classification training code**

0. pip install -r requirements.txt
1. wandb login --relogin "<put--your--wandb--token>"
2.1 python train.py --data_dir ../03_model/mentalbertavg_ft --data_type eng --avg_mode --token "<put--your--huggingface--token>"
2.2 python train_prompt.py --data_dir ../03_model/mentalbertavg_prompt20 --data_type eng --avg_mode --n_tokens 20 --token "<put--your--huggingface--token>"

여러 버전으로 실험을 할 수 있음
ast mode: answer token mean pooling
avg mode: average mean pooling
max mode: max token pooling
cls mode: cls token pooling
recon mode: input data augmentation 
sparse mode: input data augmentation
drop_epoch, n_drop : choose top-k disturbing question for classification accuracy and delete these questions in classifier head(잘 구현이 안됨)
n_tokens: prompt token length
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

참고사항 : 질문별로 prompt를 다르게 주능 기능을 구현못함