# KOR
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
# import deepl
# from openai import Openai

class surveyQA(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, recon=False, ast_mode = False, sparse_mode = False):
        if recon:
            assert not ast_mode, "recon or ast_mode 둘 중 하나만 선택해야됩니다."
            df['q-a'] = df['recon_question']
        self.df = df
        self.tokenizer = tokenizer
        self.id = df['ID'].unique().tolist()
        self.ast_mode = ast_mode
        self.sparse_mode = sparse_mode

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        qalist = self.df[(self.df['ID'] == self.id[index])]['q-a'].tolist()
        label = self.df[(self.df['ID'] == self.id[index])]['label'].unique()
        sample = self.tokenizer(qalist, return_tensors='pt', add_special_tokens=True, padding=True)
        if self.ast_mode:
            qlist = self.df[(self.df['ID'] == self.id[index])]['question'].tolist()
            question_data = self.tokenizer(qlist, return_tensors='pt', add_special_tokens=True, padding=True)
            question_length = torch.sum(torch.ne(question_data['input_ids'], 0), dim=1)
            
            alist = self.df[(self.df['ID'] == self.id[index])]['answer'].tolist()
            answer_data = self.tokenizer(alist, return_tensors='pt', add_special_tokens=True, padding=True)
            answer_length = torch.sum(torch.ne(answer_data['input_ids'], 0), dim=1) - 2 # [cls], [sep] token
            
            answer_mask = torch.zeros_like(sample['input_ids'])
            for idx in range(len(question_length)):
                if self.sparse_mode and qalist[idx] == ' ':
                    answer_mask[idx, :] = sample['attention_mask'][idx, :]
                else:
                    answer_mask[idx, question_length[idx]:question_length[idx] + answer_length[idx]] = 1
            sample['answer_mask'] = answer_mask
        return sample, torch.tensor(label)
    
def collate_fn(batch):
    # 각 배치에서 최대 길이 결정
    max_length = max([item['input_ids'].shape[-1] for item, label in batch])
    question_nums = batch[0][0]['input_ids'].shape[0]
    input_ids, attention_mask, answer_mask = [], [], []
    # 모든 텐서의 크기 및 모양을 일치시킴
    for item, label in batch:
        pad_length = max_length - item['input_ids'].shape[-1]
        input_ids.append(torch.cat([item['input_ids'], torch.zeros((question_nums, pad_length), dtype=torch.long)], dim = 1))
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((question_nums, pad_length), dtype=torch.long)], dim = 1))
        if 'answer_mask' in batch[0][0].keys():
            answer_mask.append(torch.cat([item['answer_mask'], torch.zeros((question_nums, pad_length), dtype=torch.long)], dim = 1))
            
    labels = [label for item, label in batch]
    if 'answer_mask' in batch[0][0].keys():
        return {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_mask), 'answer_mask' : torch.stack(answer_mask)}, torch.stack(labels)
    return {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_mask)}, torch.stack(labels)

def find_label(row):
    if row['S1'] == 1:
        return 0
    elif row['S5'] == 5:
        return 3
    elif row['S3'] == 5:
        return 2
    else:
        return 1
    
############################################ KOR ######################################################
def load_KOR_Dataset(sparse_mode = True):
    survey = pd.read_csv('../01_data/NMHSK2021_2210.csv', encoding='cp949', low_memory=False)
    data_test = pd.read_excel('../01_data/question-answer-after-fixed.xlsx',sheet_name=None, header=None,
                            names=['question_id', 'question', 'answer', 'score'])

    b, d, e, j, k = data_test['b'], data_test['d'], data_test['e'], data_test['j'], data_test['k']
    questiondf_list = [b, d, e, j, k]
    question_type = ['ID', 'S1', 'S2A', 'S3', 'S4A', 'S5']
    for q_df in questiondf_list:
        q_df.dropna(inplace=True)
        q_df.reset_index(drop= True, inplace= True)
        q_df['question'] = q_df['question'].replace({'\(': '', '/': ', ', '\)': ''}, regex=True)
        q_df.drop(['answer', 'score'], axis=1, inplace=True)
        question_type.extend(q_df['question_id'].to_list())

    all_id = survey.columns.to_list()
    not_used_id = list(set(all_id) - set(question_type))
    survey.drop(list(not_used_id), axis=1, inplace=True)

    questions = pd.concat(questiondf_list, ignore_index=True)
    questions = questions.transpose()
    questions.rename(columns=questions.iloc[0],inplace=True)
    questions = questions.drop(questions.index[0]).transpose()
        
    survey['label'] = survey.apply(find_label, axis=1)
    survey.drop(columns=['ID','S1', 'S2A', 'S3', 'S4A', 'S5'], axis=1, inplace=True)
    survey = survey.transpose()
    question_survey = pd.concat([questions, survey], axis = 1)
    question_survey.reset_index(inplace=True)

    df_list = []
    replacing_text = {1 : '아니요, 절대 없습니다.', 5 : '네, 맞습니다.', 6 : '대답 안함.', '1' : '아니요, 절대 없습니다.', '2': '아니요, 절대 없습니다.', '3': '시도 하지 않아서 모릅니다.', '5' : '네, 맞습니다.', '6' : '대답 안함.', ' ':'대답 안함.', '9':'대답 안함.'}
    recon_index = {1:'negative_qa', 5:'positive_qa'}
    recon_questions = pd.read_csv('../01_data/question_answer_reconstruction.csv')
    for idx in range(5511):
        series_answer = question_survey[idx][:-1]
        df_person = pd.DataFrame(question_survey['question'][:-1] + series_answer.apply(lambda x:replacing_text[x]), columns=['q-a'])
        df_person['ID'] = f'id {idx}'
        df_person['question'] = question_survey['question'][:-1]
        df_person['answer'] = series_answer.apply(lambda x: replacing_text[x])
        df_person['label'] = question_survey[idx][427]
        for i in range(427):
            answer = series_answer[i] if series_answer[i] in ['1', '5', 1, 5]  else False
            if answer:
                recon_text = recon_questions.loc[i, recon_index[int(answer)]]
                df_person.loc[i, 'recon_question'] = recon_text
            else:
                df_person.loc[i, 'recon_question'] = ''
        df_list.append(df_person)
    emb_id = pd.concat(df_list, ignore_index=True)
    if  sparse_mode:
        emb_id.loc[emb_id['answer'] == '대답 안함.', 'q-a'] = ' '
    return emb_id, np.array(question_type[6:])
'''
# input data reconstruction
def redesign_KOR_question(question, client, positive = True, irrelevant = False):
    if irrelevant:
        instruction = "다음 주어지는 문장은 설문조사의 질문에 해당된다. 제시된 문장을 보고 '(주어진 질문)~ 은 상관없는 질문이다. 혹은 나와 관련이 없다.'의 형태로 변환하여 출력해줘. 변경해야할 문장: "        
    elif positive:
        instruction = "다음 주어지는 문장은 설문조사의 질문에 해당된다. 제시된 문장을 보고 '나는 ~(주어진 질문)~ 그런적이 있다.'의 긍정의 형태로 변환하여 출력해줘. 변경해야할 문장: "
    else:
        instruction = "다음 주어지는 문장은 설문조사의 질문에 해당된다. 제시된 문장을 보고 '나는 ~(주어진 질문)~ 그런적이 없다.'의 부정의 형태로 변환하여 출력해줘. 변경해야할 문장: "
    messages = [{'role':'user', 'content': instruction + question}]
    res = client.chat.completions.create(model = 'gpt-4-1106-preview', messages = messages)
    return res.choices[0].message.content


def reconstruction_KOR_datatset(api_key):
    client = OpenAI(api_key=api_key)
    questions['positive_qa'] = questions['question'].apply(lambda x: redesign_KOR_question(x, client, True, False))
    questions['negative_qa'] = questions['question'].apply(lambda x: redesign_KOR_question(x, client, False, False))

    questions.to_csv('../01_data/question_answer_reconstruction.csv', index=False)
'''

############################################ ENG #####################################################
'''
# translation of korean question
def tlanslation_questions():
    data_test = pd.read_excel('../01_data/question-answer-after.xlsx',sheet_name=None, header=None,
                            names=['question_id', 'question', 'answer', 'score'])

    b = data_test['b']
    d = data_test['d']
    e = data_test['e']
    j = data_test['j']
    k = data_test['k']

    question_type = [b, d, e, j, k]
    for qtype in question_type:
        qtype.dropna(inplace=True)
        qtype.reset_index(inplace=True, drop=True)

    auth_key = "847b8235-1067-160f-487d-13c6b8171457:fx"
    translator = deepl.Translator(auth_key)

    for qtype in question_type:
        for idx, msg in enumerate(qtype['question']):
            qtype.loc[idx, 'translated_question'] = translator.translate_text(msg, target_lang="EN-US").text

    question_list = ['b', 'd', 'e', 'j', 'k']
    with pd.ExcelWriter('../01_data/question-answer-after-translated.xlsx') as writer:
        for i, df in enumerate(question_type):
            df.to_excel(writer, sheet_name=question_list[i], index=False)

# translate reconstruction data
def translation_reconstruction_questions(auth_key):
    recon_questions = pd.read_csv('../01_data/question_answer_reconstruction.csv')
    translator = deepl.Translator(auth_key)
    for idx, msg in enumerate(recon_questions['positive_qa']):
        recon_questions.loc[idx, 'positive_qa'] = translator.translate_text(msg, target_lang="EN-US").text

    for idx, msg in enumerate(recon_questions['negative_qa']):
        recon_questions.loc[idx, 'negative_qa'] = translator.translate_text(msg, target_lang="EN-US").text

    recon_questions.to_csv('../01_data/question_answer_reconstruction_translated.csv', index=False)

# input data reconstruction
def redesign_ENG_question(question, client, positive = True, irrelevant = False):
    if irrelevant:
        instruction = "The following sentences are questions from a survey. Look at the given sentence and rephrase it to read: '(The given question) is irrelevant or not relevant to me.' Sentence to change:"        
    elif positive:
        instruction = "The following sentences correspond to questions in a survey. Look at the given sentence and convert it to an affirmative form of 'I have done ~(given question)~ and print it out.' Sentence to change: "
    else:
        instruction = "The following sentences correspond to questions in a survey. Look at the given sentence and convert it into the negative form of 'I have never done ~(given question)~ and output it.' Sentence to change: "
    messages = [{'role':'user', 'content': instruction + question}]
    res = client.chat.completions.create(model = 'gpt-4-1106-preview', messages = messages)
    return res.choices[0].message.content


def reconstruction_ENG_datatset(api_key):
    client = OpenAI(api_key=api_key)
    questions['positive_qa'] = questions['question'].apply(lambda x: redesign_ENG_question(x, client, True, False))
    questions['negative_qa'] = questions['question'].apply(lambda x: redesign_ENG_question(x, client, False, False))

    questions.to_csv('../01_datat/question_answer_reconstruction.csv', index=False)
'''
# ENG
def load_ENG_Dataset(sparse_mode = True):
    survey = pd.read_csv('../01_data/NMHSK2021_2210.csv', encoding='cp949', low_memory=False)
    data_test = pd.read_excel('../01_data/question-answer-after-translated.xlsx',sheet_name=None, header=None)

    b, d, e, j, k = data_test['b'], data_test['d'], data_test['e'], data_test['j'], data_test['k']
    questiondf_list = [b, d, e, j, k]
    question_type = ['ID', 'S1', 'S2A', 'S3', 'S4A', 'S5']
    for q_df in questiondf_list:
        q_df.columns = q_df.iloc[0]
        q_df.reset_index(inplace= True)
        q_df.drop(index = [0], axis = 0, inplace=True)
        q_df['translated_question'] = q_df['translated_question'].replace({'\(': '', '/': ', ', '\)': ''}, regex=True)
        q_df.drop(['question','answer', 'score'], axis=1, inplace=True)
        question_type.extend(q_df['question_id'].to_list())

    all_id = survey.columns.to_list()
    not_used_id = list(set(all_id) - set(question_type))
    survey.drop(list(not_used_id), axis=1, inplace=True)

    questions = pd.concat(questiondf_list, ignore_index=True)
    questions = questions.transpose()
    questions.rename(columns=questions.iloc[0], inplace=True)
    questions = questions.drop(questions.index[0]).transpose()
    questions = questions.reset_index()

    survey['label'] = survey.apply(find_label, axis=1)
    survey.drop(columns=['ID','S1', 'S2A', 'S3', 'S4A', 'S5'], axis=1, inplace=True)
    survey = survey.transpose().reset_index()
    question_survey = pd.concat([questions, survey], axis = 1)

    df_list = []
    replacing_text = {1 : 'No, I have not.', 5 : 'Yes, I have', 6 : "It's not applicable.", '1' : "No, I have not.", '5' : 'Yes, I have', '6' : "It's not applicable.", ' ':"It's not applicable.", '6' : "It's not applicable.", '9':"It's not applicable.", '2': "No, absolutely not.", '3': "I don't know because I haven't tried."}
    recon_index = {1:'negative_qa', 5:'positive_qa'}
    recon_questions = pd.read_csv('../01_data/question_answer_reconstruction_translated.csv')
    for idx in range(5511):
        series_answer = question_survey[idx][:-1]
        df_person = pd.DataFrame(question_survey['translated_question'][:-1] + question_survey[idx][:-1].apply(lambda x:replacing_text[x] if pd.notnull(x) else replacing_text[6]), columns=['q-a'])
        df_person['ID'] = f'id {idx}'
        df_person['question'] = question_survey['translated_question'][:-1]
        df_person['answer'] = series_answer
        df_person['label'] = question_survey[idx][427]
        for i in range(427):
            answer = series_answer[i] if series_answer[i] in ['1', '5', 1, 5]  else False
            if answer:
                recon_text = recon_questions.loc[i, recon_index[int(answer)]]
                df_person.loc[i, 'recon_question'] = recon_text
            else:
                df_person.loc[i, 'recon_question'] = ''
        df_list.append(df_person)
    emb_id = pd.concat(df_list, ignore_index=True)
    if sparse_mode:
        sparse_answer_list=[6, '6', '9', ' ']
        emb_id.loc[emb_id['answer'].isin(sparse_answer_list), 'q-a'] = ' '
    emb_id['answer'] = emb_id['answer'].apply(lambda x: replacing_text[x])
    return emb_id, np.array(question_type[6:])

