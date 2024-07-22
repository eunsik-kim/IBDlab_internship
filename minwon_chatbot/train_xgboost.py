import pandas as pd
import sklearn
import re

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

tqdm.pandas()

def preprocessing_text_data(text):
    #strip을 통한 문자열 선행,후행 줄바꿈 제거
    x = text.strip()
        
    #replace를 통한 문자열 내 구분자 제거
    x = x.replace("\n", " ")
    x = x.replace("\t", " ")

    #URL 제거
    x = re.sub(re_url, " ", x)

    #단일 자모음 제거
    x = re.sub("[ㄱ-ㅣ]", " ", x)

    #개인정보 마스킹
    x = re.sub(ph_num, " ", x)
    x = re.sub(call_num, " ", x)
    x = re.sub(num_plate, " ", x)

    x = x.replace("&quot;", '"')
    x = x.replace("&upos;", "'")
    x = x.replace("&gt;", '>')
    x = x.replace("&lt;", "<")
    x = x.replace("&amp;", "&")
    x = re.sub("[^가-힣A-Za-z\s.,?!\(\)\'\"\&\;\<\>]", ' ', x)

    #다중 공백 제거
    x = re.sub(" +", " ", x)
    x = x.strip()
    
    return x

df1 = pd.read_csv("classification_datas/qa_sim_under_50.csv")
df1.head(3)

# 전처리 모음
re_han = re.compile('[一-龥]+')
re_url = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$\-@\.&+:/?=]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
ph_num = re.compile("(01\d\s{0,}\-{0,}\d{3,4}\s{0,}\-{0,}\d{4})")
call_num = re.compile("(0[2-6]\d?\s{0,}\-{0,}\d{3,4}\s{0,}\-{0,}\d{4})")
num_plate = re.compile("(\d{2,3}[가-힣]\d{4})")

df1['질문'] = df1['질문'].progress_apply(lambda x: preprocessing_text_data(x))
train_dataset = df1['질문'].copy()
train_dataset['label'] = train_dataset['상담유형(대)'].factorize(sort=True)[0]

model = SentenceTransformer("marigold334/KR-SBERT-V40K-klueNLI-augSTS-ft")
train_dataset['text'] = model.encode(train_dataset['질문'].values, show_progress_bar=True)

X = model.encode(train_dataset['질문'].values, show_progress_bar=True)
y = train_dataset['label'].values

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1004, stratify=y)
xgbc = XGBClassifier(verbose=True, random_state = 32)
xgbc.fit(X_train, y_train)

y_pred = xgbc.predict(X_test) # 예측 라벨(0과 1로 예측)
# 예측 라벨과 실제 라벨 사이의 정확도 측정
accuracy_score(y_pred, y_test) # 0.7847533632286996