### 시립대학교 [IDB Lab](https://intelligent-big-data-lab.notion.site) 인턴

---
요약: 자연어 데이터 분석 업무, 민원 응답 챗봇으로 [해커톤 2등 수상](https://kdatascience.kr/conference/event)
기간: 2023.06.01 ~ 2023.12.31
업무: 영어 교재 언어 모델, 설문조사 분류, 민원 응대 챗봇
---

### [영어 교재 언어 모델](https://github.com/eunsik-kim/IBDlab_internship/tree/main/survey_classification)

- 약 6달(2023.07.01 ~ 2023.12.20)
- 영어 교재 제작 보조용 언어 모델 학습에 참여
- 교과서 및 평가원 문제들을 pandas를 활용해 전처리하고, 이를 통해 instruction tuning dataset 제작
- 분류 테스트 과제(Tips)를 수행하기 위해 모델을 학습하고 비교 평가를 수행
    - 언어모델을 Hugging Face의 PEFT(Parameter Efficient Tuning) 방법을 사용하여 튜닝
    - GPT-4.0 API와 프롬프트 튜닝한 모델의 multi-label(17 class) 분류 정확도를 비교

### [설문조사 분류](https://github.com/eunsik-kim/IBDlab_internship/tree/main/survey_classification)

- 약 2달(2023.10.01 ~ 2023.11.31)
- 설문조사 응답 데이터를 언어 모델에 적용해보려는 연구 과제
    - BERT, SentenceTransformer, MentalBERT 모델을 사용하여 설문 결과를 예측(분류 성능으로 평가)
    - 모델 학습 후 설문조사 질문별로 언어 모델 예측치에 영향을 주는 기여도 분석(Top-N 비교, t-SNE 작성)
- GPU 메모리 제한(OOM)의 한계를 극복하기 위해, PEFT의 LoRA(Low-Rank Adaptation) 방법과 유사하게 학습 진행

### 민원 응대 챗봇

- 해커톤 참가 준비 포함 약 10일 (2023.11.20 ~ 2023.11.30)
- 민원 응대를 위해 기존에 학습된 챗봇에 RAG(Retrieval-Augmented Generation) 기법을 적용하여 응답 성능 향상 시도
    - 민원 처리 부서 데이터와 민원 질문 간의 관계를 추출하기 위한 [모델을 학습시킴](https://huggingface.co/marigold334/KR-SBERT-V40K-klueNLI-augSTS-ft)
    - 민원 질문 분류를 위해 XGBoost 모델을 사용
    - 위의 두 모델을 활용하여 민원 질문 프롬프트를 작성하고, LangChain 라이브러리를 이용해 챗봇 응답 파이프라인을 구축
