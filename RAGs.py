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
    """Hugging Face에서 인덱스를 로드하여 RAG 체인 설정"""
    try:
        # Hugging Face에서 파일 다운로드 - 임시 디렉토리 사용
        index_path = hf_hub_download(
            repo_id="Bommmmee/faiss-index",
            filename="index.faiss",
            repo_type="dataset",
            local_dir="/tmp"  # 임시 디렉토리 사용
        )
        
        # OpenAI 임베딩 초기화
        embeddings = OpenAIEmbeddings()
        
        # FAISS 인덱스 로드 - 메모리 최적화 설정
        vectorstore = FAISS.load_local(
            folder_path=os.path.dirname(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # QA 체인 설정 - 메모리 최적화
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0),
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 2}  # k 값을 줄여서 메모리 사용량 감소
            ),
            return_source_documents=False,
            verbose=False
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"RAG 체인 설정 중 오류 발생: {e}")
        raise
