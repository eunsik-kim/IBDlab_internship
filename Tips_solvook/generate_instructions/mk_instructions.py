import pandas as pd
import numpy as np
import json, os, argparse
import re, random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split

def load_dataset(data_path = './100_Solvook_handout_DB_english.xlsx'):
    all_sheet = pd.read_excel(data_path, sheet_name= None)
    sheet_name_list = list(all_sheet.keys())
    handout_db = all_sheet[sheet_name_list[0]]
    paragraph_db = all_sheet[sheet_name_list[1]]
    paragraph_list = all_sheet[sheet_name_list[3]]
    handout_type = all_sheet[sheet_name_list[4]]
    return handout_db, paragraph_db, paragraph_list, handout_type

# remove prefix numbering 
def remove_non_alphanumeric_prefix(input_string):
    pattern = '[가-힣a-zA-Z]'
    match = re.search(pattern, input_string)
    if match: cleaned_string = input_string[match.start():]
    else:     cleaned_string = input_string
    return cleaned_string

# { "skill": [3], "method": [3] } format에서 skill값과 method 값을 추출
def extract_skill_method(s_and_m):
    try:
        s_and_m_dict = json.loads(s_and_m.replace("'", "\""))
        skill = s_and_m_dict['skill'][0] if 'skill' in s_and_m_dict else None
        method = s_and_m_dict['method'][0] if 'method' in s_and_m_dict else None
        return [skill, method]
    except:
        return []

def preprocessing_handout_db(handout_db):
    handout_db = handout_db.dropna(subset=['질문', '본문'])
    selected_columns = ['본문', '조건', '선지', '정답', '질문',  
                        '분류체계 시트 > \n문제유형_01', '분류체계 시트\n문제유형_02', '분류체계 시트\n문제유형_03', 
                        '분류체계 시트 > \nskill # , method_01', '분류체계 시트 > \nskill # , method_02',
                        '분류체계 시트 > \nskill # , method_03', 'story\nid', 'paragraph\nID', 'unit_name', 'story_name', '교과서명']
    new_df = handout_db[selected_columns]
    new_df = new_df.reset_index(drop=True)
    new_column_names = {
        '분류체계 시트 > \n문제유형_01' : '문제유형1',
        '분류체계 시트\n문제유형_02' : '문제유형2',
        '분류체계 시트\n문제유형_03' : '문제유형3',
        '분류체계 시트 > \nskill # , method_01' : 's&m1',
        '분류체계 시트 > \nskill # , method_02' : 's&m2', 
        '분류체계 시트 > \nskill # , method_03' : 's&m3',
        'story\nid': 'story id', 'paragraph\nID' : 'paragraph ID'
    }
    new_df = new_df.rename(columns=new_column_names)

    # replace _____(long) > __(short)
    for col in ['본문','질문','선지','조건','정답']:
        for idx in range(len(new_df['질문'])):
            try:
                new_df.loc[idx, col]= re.sub('___+','___', new_df.loc[idx, col]) 
            except:
                pass
    new_df['질문'] = new_df['질문'].apply(remove_non_alphanumeric_prefix)

    # concatenate multi-label-skills & methods & 문제유형
    for idx in range(1,4):
        new_df['s&m'+str(idx)] = new_df['s&m'+str(idx)].apply(extract_skill_method) 
        new_df['문제유형'+str(idx)] = new_df['문제유형'+str(idx)].apply(lambda x: [int(x)] if not pd.isna(x) else [])
    new_df['s&m'] = new_df['s&m1'] + new_df['s&m2'] + new_df['s&m3']
    new_df['문제유형'] = new_df['문제유형1'] + new_df['문제유형2'] + new_df['문제유형3']
    new_df = new_df.reset_index(drop=True)
    return new_df

# pragraph_db와 paragrph_list를 textbook_id와 unit_id를 기준으로 merge
def preprocess_paragraph_db(paragraph_db, paragraph_list):
    grouped_paragraph_db = paragraph_db.groupby(['textbook_id', 'unit_id'], as_index=False)['paragraphs'].apply(', '.join)
    df_paragraph_db = pd.merge(grouped_paragraph_db, paragraph_list[['textbook_id', 'unit_id', '교과서명',
                                                                 '출판사', 'unit_title', 'unit_name', 'story name', 'story type']], on=['textbook_id', 'unit_id'])
    return df_paragraph_db

# 문제 유형, skill, metho의 number에 따른 dictionary
def get_sm_dict(handout_type):
    handout_type.columns = handout_type.iloc[0]
    handout_type = handout_type[1:]
    handout_type.reset_index(drop= True, inplace=True)
    quiztype_dict = {}
    skill_dict = {}
    method_dict = {}
    for idx in range(len(handout_type)):
        row = handout_type.iloc[idx]
        quiztype_dict[row['quiz type']] = row['문제 유형 (영어)']
        skill_dict[row['skill #']] = row['skill (2depth)']
        method_dict[row['method #']] = row['method (2depth) 영어']
    return skill_dict, method_dict, quiztype_dict

# 2번 instruction을 위해 유사문장을 뽑음(셔플하는데 시간이 걸려서 약 15m분 소요)
def shuffle_get_similar_query(handout_db):
    random.seed(0)
    vectorizer = TfidfVectorizer()
    query_list = handout_db['질문'].tolist()
    mat = vectorizer.fit_transform(query_list)
    sim_mat = cosine_similarity(mat)
    type_df = handout_db['문제유형']

    equal_pair_query = []
    diff_pair_query = []

    # 같은 유형일 경우
    for i, row in enumerate(sim_mat):
        if not type_df[i]: # except null value
            continue
        order_row = [[idx, value] for idx, value in enumerate(row)]
        random.shuffle(order_row)
        for j, sim in order_row:
            if 0.6 > sim > 0.4: # arbitrary threshold
                if set(type_df[i]).intersection(set(type_df[j])):
                    equal_pair_query.append([query_list[i], query_list[j]]) 
                    break
                
    # 다른 유형일 경우
    for i, row in enumerate(sim_mat):
        if not type_df[i]: # except null value
            continue
        order_row = [[idx, value] for idx, value in enumerate(row)]
        random.shuffle(order_row)
        for j, sim in order_row:
            if 0.4 > sim > 0.3: # arbitrary threshold
                if not set(type_df[i]).intersection(set(type_df[j])):
                    diff_pair_query.append([query_list[i], query_list[j]]) 
                    break
    return equal_pair_query, diff_pair_query

def save_split_data(handout_db, arg):
    tr, te_0 = train_test_split(handout_db, 
                                stratify=handout_db['문제유형'].apply(lambda x: x[0] if x[0] != 39 else 41),
                                test_size=0.2,
                                random_state=1)
    # val / te
    val, te  = train_test_split(te_0,
                                stratify=te_0['문제유형'].apply(lambda x: x[0] if x[0] != 5 else 41),
                                test_size=0.5,
                                random_state=1)
    dataset = [tr, val, te]
    for data in dataset:
        data.reset_index(drop = True, inplace=True)
    os.makedirs(arg.saving_path, exist_ok=True)
    tr.to_csv(os.path.join(arg.saving_path,'./solvook_handout_tr.csv'), encoding='utf-8-sig')
    val.to_csv(os.path.join(arg.saving_path,'./solvook_handout_val.csv'), encoding='utf-8-sig')
    te.to_csv(os.path.join(arg.saving_path,'./solvook_handout_te.csv'), encoding='utf-8-sig')
    return tr
    
def save_json(prompt, filename = 'example.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(prompt, f, ensure_ascii=False, indent=4)

def main(arg):
    print('loading dataset')
    raw_handout_db, raw_paragraph_db, raw_paragraph_list, handout_type = load_dataset(arg.data_path)   
    handout_db = preprocessing_handout_db(raw_handout_db)
    paragraph_db = preprocess_paragraph_db(raw_paragraph_db, raw_paragraph_list)
    skill_dict, method_dict, quiztype_dict = get_sm_dict(handout_type)
    print('split and saving files')
    handout_db = save_split_data(handout_db, arg)
    print('get similar query')
    equal_pair_query, diff_pair_query = shuffle_get_similar_query(handout_db)
    print('finished for pairing queries')
    
    prompt_style = {"prompt":"다음은 문제를 설명하는 지침입니다. 지침에 따라 문제를 적절하게 완료하는 응답을 작성하십시오.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
                    "response":"{output}",
                    "meta":{"source":"solvook","language":"kor"}}
    prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10 = [], [], [], [], [], [], [], [], [], []

    print('making instructions')
    for idx in range(len(handout_db['질문'])):
        query = handout_db.loc[idx, '질문']
        passage = handout_db.loc[idx, '본문'] if not pd.isnull(handout_db.loc[idx, '본문']) else '없음'
        condition = handout_db.loc[idx, '조건'] if not pd.isnull(handout_db.loc[idx, '조건']) else '없음'
        choices = handout_db.loc[idx, '선지'] if not pd.isnull(handout_db.loc[idx, '선지']) else '없음'
        answer = handout_db.loc[idx, '정답'] if not pd.isnull(handout_db.loc[idx, '정답']) else False # except null value
        try: # except null value
            quiztype = ' 또는 '.join([quiztype_dict[qi] for qi in handout_db.loc[idx, '문제유형']])
        except:
            quiztype = False
        skill_index = handout_db.loc[idx, 's&m'][::2]
        method_index = handout_db.loc[idx, 's&m'][1::2]
        sm = ' 또는 '.join(['skill: ' + skill_dict[si] + ', method: '+ method_dict[mi] for si, mi in zip(skill_index, method_index)])
        s = ' 또는 '.join(np.unique([skill_dict[si] for si in skill_index]))
        m = ' 또는 '.join(np.unique([method_dict[mi] for mi in method_index]))

        # 주어진 instruction example에 맞게 instruction 생성
        if quiztype:
            instruction1 = "다음의 질문이 어떤 문제 유형에 해당하는지, 고등 영어 교육과정 중 어떤 skill, method과 관련되어 있는지 답하라. 질문: {질문}".format(질문 = query)
            output1 = "문제 유형은 {문제_유형_상세}]이고, skill-method는 {sm}와 관련되어 있다.".format(문제_유형_상세= quiztype, sm = sm)
            prompt1.append({'prompt':prompt_style['prompt'].format(instruction = instruction1), 'response':prompt_style['response'].format(output = output1),'meta':prompt_style['meta']})
            
        instruction2 = "다음 질문에 답하기 위해서는 어떤 method가 필요한가? method 후보: {ml}, 질문: {질문}".format(ml = list(method_dict.values()), 질문 = query)
        output2 = "제시된 질문에 답을 하기 위해 {method}를 할줄 알면 질문을 풀 수 있다".format(method= m)
        prompt2.append({'prompt':prompt_style['prompt'].format(instruction = instruction2), 'response':prompt_style['response'].format(output = output2),'meta':prompt_style['meta']})
        
        instruction3 = "다음 질문에 답하기 위해서는 어떤 skill이 필요한가? skill 후보: {sl}, 질문: {질문}".format(sl = list(skill_dict.values()), 질문 = query)
        output3 = "제시된 질문에 답을 하기 위해 {skill}를 알아야 한다".format(skill = s)
        prompt3.append({'prompt':prompt_style['prompt'].format(instruction = instruction3), 'response':prompt_style['response'].format(output = output3),'meta':prompt_style['meta']})
        
        if quiztype:
            instruction4 = "다음 본문에 대해 출제된 질문의 문제유형은 무엇인가? 문제유형 후보: {qyl}, 본문: {본문}, 질문: {질문}".format(qyl = list(quiztype_dict.values()), 본문 = passage, 질문 = query)
            output4 = "주어진 문제의 문제유형은 {문제_유형_상세}이다.".format(문제_유형_상세 = quiztype)
            prompt4.append({'prompt':prompt_style['prompt'].format(instruction = instruction4), 'response':prompt_style['response'].format(output = output4),'meta':prompt_style['meta']})
            
        if answer:
            instruction5 = "문제가 제시된다. 다음의 문제의 정답을 맞춰라. 본문: {본문}, 조건: {조건}, 질문: {질문}, 선지: {선지}".format(본문 = passage, 조건 = condition, 질문 = query, 선지 = choices)
            output5 = "문제의 정답은 {정답}이다.".format(정답 = answer)
            prompt5.append({'prompt':prompt_style['prompt'].format(instruction = instruction5), 'response':prompt_style['response'].format(output = output5),'meta':prompt_style['meta']})
                
        instruction6 = "다음과 같은 형태의 본문에 어울리는 질문은 무엇인가? 본문: {본문}".format(본문 = passage)
        output6 = "어울리는 질문은 {질문}이다.".format(질문 = query)    
        prompt6.append({'prompt':prompt_style['prompt'].format(instruction = instruction6), 'response':prompt_style['response'].format(output = output6),'meta':prompt_style['meta']})

        if not pd.isnull(handout_db.loc[idx, 'paragraph ID']):
            paragraph_connection_list = ['story id', 'paragraph ID','unit_name', 'story_name', '교과서명']
            paragraph_connection = []
            for connect in paragraph_connection_list:
                paragraph_connection.append(str(handout_db.loc[idx, connect]))
            paragraph_connection = '-'.join(paragraph_connection)

            instruction10 = "다음으로 제시되는 지문을 보고 어떤 교과서의 본문과 연관되는지 답하여라. 지문: {본문}".format(본문 = passage)
            output10 = "주어진 지문과 연관되어 있는 교과서는 {paragraph_connection}이다.".format(paragraph_connection = paragraph_connection)
            prompt10.append({'prompt':prompt_style['prompt'].format(instruction = instruction10), 'response':prompt_style['response'].format(output = output10),'meta':prompt_style['meta']})

    equal_pair_query = np.unique(equal_pair_query)
    diff_pair_query = np.unique(diff_pair_query)
    for sim_idx in range(len(equal_pair_query)):
        equal_querys = equal_pair_query[sim_idx]

        instruction7_1 = "다음 두 질문의 문제 유형이 같은지 다른지 답하라. 질문1: {질문1}, 질문2: {질문2}".format(질문1 = equal_querys[0], 질문2 = equal_querys[1])
        output7_1 = "두 질문의 문제 유형이 같다"
        prompt7.append({'prompt':prompt_style['prompt'].format(instruction = instruction7_1), 'response':prompt_style['response'].format(output = output7_1),'meta':prompt_style['meta']})

    for sim_idx in range(len(diff_pair_query)):
        diff_querys = diff_pair_query[sim_idx]

        instruction7_2 = "다음 두 질문의 문제 유형이 같은지 다른지 답하라. 질문1: {질문1}, 질문2: {질문2}".format(질문1 = diff_querys[0], 질문2 = diff_querys[1])
        output7_2 = "두 질문의 문제 유형이 다르다"
        prompt7.append({'prompt':prompt_style['prompt'].format(instruction = instruction7_2), 'response':prompt_style['response'].format(output = output7_2),'meta':prompt_style['meta']})

    for par_idx in range(len(paragraph_db['paragraphs'])):
        paragraph = paragraph_db.loc[par_idx, 'paragraphs']
        story_name = paragraph_db.loc[par_idx, 'story name']
        story_type = paragraph_db.loc[par_idx, 'story type']

        paragraph_info_list = ['교과서명', '출판사', 'unit_title', 'unit_name']
        paragraph_info = []
        for para in paragraph_info_list:
            paragraph_info.append(paragraph_db.loc[par_idx, para])
        paragraph_info = '-'.join(paragraph_info)

        instruction8 = "다음으로 제시되는 지문은 {paragraph_info}의 한 본문이다. 본문의 글의 종류는 무엇인가? 본문: {본문}".format(paragraph_info= paragraph_info, 본문 = paragraph)
        output8 = "주어진 본문의 글의 종류는 {story_type}이다.".format(story_type = story_type)
        prompt8.append({'prompt':prompt_style['prompt'].format(instruction = instruction8), 'response':prompt_style['response'].format(output = output8),'meta':prompt_style['meta']})

        instruction9 = "다음으로 제시되는 지문은 {paragraph_info}의 한 본문이다. 다음 본문의 제목은 무엇인가? 본문: {본문}".format(paragraph_info= paragraph_info, 본문 = paragraph)
        output9 = "주어진 본문의 글의 제목은 {story_name}이다.".format(story_name = story_name)
        prompt9.append({'prompt':prompt_style['prompt'].format(instruction = instruction9), 'response':prompt_style['response'].format(output = output9),'meta':prompt_style['meta']})

    # save
    print('saving instructions')
    prompt = []
    promptlist = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]
    if arg.example:
        for idx, p in enumerate(promptlist):
            save_json(p, os.path.join(arg.saving_path,'example'+str(idx+1)+'.json'))
            print(f'{idx+1} instruction의 총 갯수는 {len(p)}입니다.')
            prompt.extend(p)
    save_json(prompt, os.path.join(arg.saving_path,'Solvook_instruction.json'))
    print(f'모든 생성된 instruction 갯수는 {len(prompt)}개 입니다.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()                
    parser.add_argument("--data_path", type=str, default='./100_Solvook_handout_DB_english.xlsx')
    parser.add_argument("--saving_path", type=str, default='./generated_data')
    parser.add_argument("--example", action='store_true')

    arg = parser.parse_args()    
    main(arg)
