import gradio as gr
from xgboost import XGBClassifier
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.utils.math import cosine_similarity
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 48  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

test_xgb_model = XGBClassifier()
test_xgb_model.load_model("text_xgboost_classifier.json")
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="llama2_with_mk3_q4.gguf",       # path to gguf
    n_gpu_layers=n_gpu_layers,                  # gpu layer 수?
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx= 2000,
    max_tokens=2000,
    temperature=0.4,
    verbose=True,  # Verbose is required to pass to the callback manager
    )

embeddings = SentenceTransformerEmbeddings(model_name="marigold334/KR-SBERT-V40K-klueNLI-augSTS-ft")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def find_text_class(question):
    # xgboost 
    text_class_dictionary = ['개인과가정', '경제산업', '교통', '구정일반', '기업과경제', '기타', '도시기반', '도시환경',
       '문화/체육/관광', '문화와여가', '부동산과재산', '사회보장과복지', '사회복지', '상하수도', '서비스와생활',
       '세금과재정', '시정일반', '유관기관', '자연과환경', '재난과안전', '제조건설과개발', '주요사업소', '주택/건축']
    
    question_embedding = embeddings.embed_query(question)
    text_class = text_class_dictionary[test_xgb_model.predict([question_embedding])[0]]
    return text_class

def prompt_router(question, prompt_embeddings, prompt_templates):
    query_embedding = embeddings.embed_query(question)
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    return most_similar

def template_choice(question):
    persona_prompt = "당신은 {text_class} 담당 AI 챗봇 상담사입니다. 민원인에게 친절한 말투의 한국어로 명확한 업무 안내를 진행해야 하며, 필요한 경우 상세한 설명을 포함합니다. "
    ask_prompt = "질문 또는 문의 형식의 민원이 접수되었으며, 주어진 자료를 참고하여 적절한 답변을 생성합니다. "
    recieve_prompt = "제안 혹은 접수 형식의 민원이 접수되었으며, 적절한 답변을 생성합니다. "
    
    prompt_templates = [ask_prompt + question, recieve_prompt+ question]
    prompt_embeddings = embeddings.embed_documents(prompt_templates)
    choiced = prompt_router(question, prompt_embeddings, prompt_templates)
    if choiced == ask_prompt:
        docs = db.similarity_search(question, k= 3)
        doc_prompt = f"다음 자료를 참고하여 안내합니다.\n 자료: {docs}"
        choiced += doc_prompt
    complete_template = persona_prompt + choiced
    return complete_template
    
def yuuung(question, history):
    text_class = find_text_class(question)
    template = template_choice(question)
    tail_prompt = "\\n\n###민원인: {Question} 반드시 한국어로 답변을 해주세요 \n\n###상담사: " 
    template += tail_prompt  
    
    prompt = PromptTemplate(template=template, input_variables=["Question", "text_class"])
    chain = prompt | llm.bind(stop=["\n"]) | StrOutputParser()
    # llm_chain = LLMChain(prompt=prompt, llm=llm,  memory=memory)
    # llm_chain.run(Question = question, text_class=text_class)
    response = chain.invoke({'Question': question, 'text_class': text_class}).split('###')[0]
    return response

demo = gr.ChatInterface(
    yuuung,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="불편 사항이나 문의 사항 전달주시면 친절히 답변드리도록 하겠습니다!", container=False, scale=7),
    title="도비봇",
    description="도비봇은 서울시 다산120콜센터 민원 데이터를 통해 학습된 챗봇으로, 시민의 복지 및 편리성 향상을 위해 노력합니다.",
    theme="soft",
    examples=['강동구 OO아파트 앞 도로에 많은 차량이 불법 주정차되어 있습니다. 누구한테 연락해야 해결가능한가요','종로구 수도세가 많이 나와 문의드립니다 수도세는 어디에 문의해야하나요'],
    cache_examples=False,
    retry_btn=None,
    clear_btn="Clear",
)

if __name__ == '__main__':
    demo.launch(inline = True,) # share=True