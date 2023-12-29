import torch 
import os, ast, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.manifold import TSNE
from datasets import load_dataset

def get_dataset(arg):
    train_file_path, eval_file_path = os.path.join(arg.data_dir, arg.train_file), os.path.join(arg.data_dir, arg.eval_file)
    if os.path.exists(train_file_path) and os.path.exists(eval_file_path):
        FileNotFoundError(f'There are no train or test files in {train_file_path} or {eval_file_path}')
    dataset = load_dataset("csv", data_files={"train": train_file_path, "eval": eval_file_path},)
    return dataset

def get_test_dataset(arg):
    test_file_path = os.path.join(arg.data_dir, arg.test_file)
    if os.path.exists(test_file_path):
        FileNotFoundError(f'There are no train or test files in {test_file_path}')
    dataset = load_dataset('csv', data_files = {'test': test_file_path})
    return dataset

#  input_columns = ['질문', '본문', '조건', '선지', '정답']
def preprocess_function(examples, tokenizer, skill_dict, method_dict, is_testset = False, input_columns = ['질문'], label_columns = ['s&m'], pretrained_model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"):
    batch_size = len(examples['본문'])
    
    # input column concatenate
    prefix_text = '[INST] <<SYS>>\n영어문제가 주어진다. 지시사항에 답하여라. \n<<SYS>>\n\n 다음의 질문에 해당되는 \'skill\'과 \'method\'를 답하여라. '
    input_list = [prefix_text for _ in range(batch_size)]
    for col_name in input_columns:
        for idx, x in enumerate(examples[col_name]):
            if x == None:
                continue    # 조건, 선지는 없는경우 존재
            input_list[idx] += f'{col_name} : {x}, ' 
    
    # since OOM remove candidate        
    # skill_candidate = '다음 선택지에서 skill을 한가지 고르면 된다. 선택지:[1.vocabulary, 2.grammar, 3.expression, 4.content, 5.context]'
    # method_candidate = '다음 선택지에서 method를 한가지 고르면 된다. 선택지: [1. 내용 해석하기 (영-한 변환), 2. 어휘 쓰기 및 찾기, 3. 문장 쓰기, 4. 밑줄 친 부분 고쳐쓰기, 5. 선택지 내 요소들이 모두 맞는 것 찾기, 6. 내용과의 일치 여부 판단하기, 7. 요지 찾기, 8. 유추하기, 9. 주제문 찾기, 10. 순서 배열하기, 11. (글의 흐름에 맞게) 문장 배치하기, 12. 연결어 찾기, 13. 정오 여부 판단하기, 14. 잘못된 것 고치기, 15. 유사 여부 판단하기, 16. 관련 없는 문장 찾기]'
    # quiztype_candidate = '다음 선택지에서 문제유형을 고르면 된다. 선택지:[4. 어휘 쓰기, 5. 어휘 뜻 맞히기, 10. 글의 요지(요악문 포함), 11. 글의 순서, 12. 내용 일치, 13. 내용 불일치, 15. 내용 유추, 16. 다른 용법 찾기, 17. 한/영 해석(조건 제시), 18. 무관한 문장 찾기, 19. 단어 재배열, 20. 밑줄 친 부분 고쳐쓰기, 22. 서술형(영어), 24. 서술형(조건 영어작문), 25. 서술형(한글), 26. 선택지 2개 이상 중 맞는 것 고르기(어법), 27. 선택지 2개 이상 중 맞는 것 고르기 (어휘), 28. 어법 상 맞는 것 찾기, 29. 어법 상 틀린 것 찾기, 30. 연결어 찾기, 34. 적절한 어휘 찾기, 35. 적절한 제목/주제 찾기, 36. 주어진 문장 넣기, 38. 틀린 어휘 찾기, 40. 서술형(조건 한글작문), 41. 내용 일치 (영어 질문), 42. 같은 용법 찾기, 44. 다른 어휘 찾기, 45. 같은 어휘 찾기]\n'
    
    tail = '정답 = [/INST]'
    instruction_text = tail
    for idx in range(batch_size):
        input_list[idx] += instruction_text
    
    # make target list
    target_list = ['' for _ in range(batch_size)]
    for idx in range(batch_size):
        sm = ast.literal_eval(examples['s&m'][idx])
        skill_index = sm[::2]
        method_index = sm[1::2]
        target_list[idx] += 'skill-method는 '+ '또는 '.join(['skill: ' + skill_dict[si] + ', method: '+ method_dict[mi] for si, mi in zip(skill_index, method_index)]) +'이다. </s>'

    # tokenizing dataset input
    model_inputs = tokenizer(input_list)
    # if test mode, remove labels
    if is_testset:
        skills = []
        methods = []
        for idx in range(batch_size):
            sm = ast.literal_eval(examples['s&m'][idx])
            skills.append([skill_dict[si] for si in sm[::2]])
            methods.append([method_dict[mi] for mi in sm[1::2]])
        model_inputs["skill"] = skills
        model_inputs["method"] = methods
        return model_inputs
    
    labels = tokenizer(target_list)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids # input_ids ... labels
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i]) # input_ids + labels
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids # -100 ... labels
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def collate_fn(samples):
    batch_size = len(samples)
    max_length = max([len(samples[i]['input_ids']) for i in range(batch_size)])
    sample_input_ids = []
    label_input_ids = []
    attention_mask_ids = []
    for sample in samples:
        sample_input_ids.append(sample["input_ids"])
        label_input_ids.append(sample["labels"])
        attention_mask_ids.append(sample["attention_mask"])
    samples = {'input_ids': sample_input_ids, 'attention_mask': attention_mask_ids, 'labels': label_input_ids}
        
    for i in range(batch_size):
        sample_input_ids = samples["input_ids"][i]
        label_input_ids = samples["labels"][i]
        attention_mask_ids = samples["attention_mask"][i]
        # padding to left in batch_size
        samples["input_ids"][i] = [0] * (max_length - len(sample_input_ids)) + sample_input_ids
        samples["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + attention_mask_ids
        samples["labels"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
    
    # list -> tensor    
    samples["input_ids"] = torch.tensor(samples["input_ids"]).contiguous()
    samples["attention_mask"] = torch.tensor(samples["attention_mask"]).contiguous()
    samples["labels"] = torch.tensor(samples["labels"]).contiguous()
    return samples

def pred_parse(string , pats = ['vocabulary', 'grammar', 'expression', 'content', 'context']):
    # find answer parse from output 
    txt_preds, num_preds = [], []
    for pat in pats:
        for match in re.finditer(pat, string):
            txt_preds.append(match.group())
    txt_preds = np.unique(txt_preds) # 중복 처리
    if 3>= len(txt_preds)>= 1:
        for pred in txt_preds:       
            num_preds.append(pats.index(pred))
    else:                           # 예외 skill label: 5
        num_preds = len(pats)
    return txt_preds, num_preds

# deprecated tool

# def find_similar_index(query, value):    
#     cosine_sims = []
#     for row_A in query:
#         row_A = row_A.view(1, -1)  
#         cos_sim_row = F.cosine_similarity(row_A, value, dim=1)
#         cosine_sims.append(cos_sim_row)
#     return torch.argmax(torch.stack(cosine_sims), dim=1)

# def draw_tsne(matrix):
#     # Create a t-SNE model and transform the data
#     tsne = TSNE(n_components=2, perplexity = 1, random_state=42, init='random', learning_rate=200)
#     vis_dims = tsne.fit_transform(matrix)
#     x = [x for x,y in vis_dims]
#     y = [y for x,y in vis_dims]
#     plt.figure(figsize=(5, 5))
#     plt.scatter(x, y, c=np.arange(len(matrix)), alpha=0.3)
#     for idx in range(len(matrix)):
#         plt.annotate(f'{idx+1}: prompt', (x[idx], y[idx]))
#     plt.title("prompt embedding t-SNE")
#     return plt
