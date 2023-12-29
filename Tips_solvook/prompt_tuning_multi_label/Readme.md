**prompt_tuning_multi_label training code**

0. pip install -r requirements.txt
1. wandb login --relogin "<put--your--wandb--token>"
2. python prompt.py --pretrained_model_name bigscience/bloomz-560m --token "<put--your--huggingface--token>"

주의사항
1. excel 파일이 버전이 바뀌면 sheet name이나 colum name 오류 발생할 수 도 있음
2. solvook_handout te, tr, val 은 generated_data로 부터 생성됨


