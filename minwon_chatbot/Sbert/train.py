import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from tqdm import tqdm
from torch.utils.data import DataLoader

class sbert_dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self, ):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            if type(idx) == int:
                return [self.df.loc[idx, '질문'], self.df.loc[idx, '업무']]
            return [[a,b] for a, b in zip(self.df.loc[idx, '질문'].tolist(), self.df.loc[idx, '업무'].tolist())]

        except KeyError as e:
            raise IndexError()

def make_sts_input_example(dataset):
    input_examples = []
    for i, data in enumerate(dataset):
        sentence1 = data[0]
        sentence2 = data[1]
        score = 1.
        input_examples.append(InputExample(texts=[sentence1, sentence2], label=score))

    return input_examples

train_df = pd.read_csv('./질문_업무_db.csv',  encoding='utf-8', index_col=[0]) #cp949
train_df.dropna(inplace=True)
train_df.reset_index(drop=True, inplace=True)
train_dataset = sbert_dataset(train_df)
train_dataset = make_sts_input_example(train_dataset)

tqdm.pandas()
train_df['input'] = train_df.progress_apply(lambda x: InputExample(texts=[x.질문, x.업무]), axis = 1)
test_df = train_df[-41610:].copy()
train_df = train_df[:-41610] 
train_examples = list(train_df['input'])
test_examples = list(test_df['input'])

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=16)

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

sentences1 = ['상수도사업본부 중부수도사업소 요금과 요금1팀중구용산구 부서의 업무는 차량관리 69오6382 팀 휴대폰 관리 010 6428 요금심사 및 체납징수 신당2동 신당4동 청구동 신당3동 남영동', 
              '행정국 인력개발과 교육팀 부서의 업무는 직장교육 멘토링 포함 운영에 관한 사항 학습동아리 운영 및 연구 저술 공무원 지원에 관한 사항 독도아카데미에 관한 사항 대학생 아르바이트 운영 인턴십 경력증명 발급 등 5급 승진자 교육 관련 사항 9급 신임자 교육 관리 공무직 교육 관련 사항 자발적 학습조직 활성화 추진 지원 대직자 이하진']
sentences2 = ['수도 요금 누수 감액을 받았는데 장기간 누수가 의심되어 문의', '서울시 대학생 아르바이트 선발 결과 문의']

vec1 = model.encode('어떻게 이럴수가 있지?')
vec2 = model.encode(sentences2[0])
vec3 = model.encode(sentences1[0])
print('Negative example cosine_sim :', util.cos_sim(vec1.reshape(1,-1), vec2.reshape(1,-1)))
print('Positive example cosine_sim :', util.cos_sim(vec3.reshape(1,-1), vec2.reshape(1,-1)))

print('training start')
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=300)
print('training ends')
model.save('MNR_loss')
print('save complete')

vec1 = model.encode('어떻게 이럴수가 있지?')
vec2 = model.encode(sentences2[0])
vec3 = model.encode(sentences1[0])
print('After training, Negative example cosine_sim :', util.cos_sim(vec1.reshape(1,-1), vec2.reshape(1,-1)))
print('After training, Positive example cosine_sim :', util.cos_sim(vec3.reshape(1,-1), vec2.reshape(1,-1)))
