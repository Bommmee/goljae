from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

load_dotenv()
async def setup_rag_chain():
    try:
        # Hugging Face에서 파일 다운로드
        index_path = hf_hub_download(
            repo_id="Bommmmee/faiss-index",
            filename="index.faiss",
            repo_type="dataset",
            local_dir="./temp_index"  # 임시 디렉토리 지정
        )
        
        pkl_path = hf_hub_download(
            repo_id="Bommmmee/faiss-index",
            filename="index.pkl",
            repo_type="dataset",
            local_dir="./temp_index"  # 임시 디렉토리 지정
        )
        
        # OpenAI 임베딩 초기화
        embeddings = OpenAIEmbeddings()
        
        # FAISS 인덱스 로드
        index_dir = os.path.dirname(index_path)
        print(f"Loading from directory: {index_dir}")  # 디버깅용 출력
        
        vectorstore = FAISS.load_local(
            folder_path=index_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # LLM 설정
        llm = ChatOpenAI(temperature=0)
        
        # 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        # 프롬프트 템플릿
        template = """
        당신은 문서 내용을 이해하고 질문에 답변하는 전문가입니다.

        규칙:
        1. 문서에서 찾을 수 있는 정보만 사용하여 답변하세요.
        2. 찾을 수 없는 경우 "죄송합니다. 다시 구체적으로 입력해 주세요"라고 답변하세요.
        3. 추측하지 마세요.

        문서:
        {context}

        질문: {question}

        답변:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        # QA 체인 설정
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=False,
            verbose=True
        )

        return qa_chain
        
    except Exception as e:
        print(f"RAG 체인 설정 중 오류 발생: {e}")
        raise