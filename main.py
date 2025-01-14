from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging
from RAGs import setup_rag_chain

# FastAPI 앱 생성
app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 로컬 Next.js 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list] = None

# 전역 변수로 QA 체인 저장
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG 체인 초기화"""
    global qa_chain
    try:
        logger.info("RAG 체인 초기화 시작")
        qa_chain = await setup_rag_chain()
        logger.info("RAG 체인 초기화 성공")
    except Exception as e:
        logger.error(f"RAG 체인 초기화 실패: {e}")
        qa_chain = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    global qa_chain
    
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    try:
        result = await qa_chain.acall({"question": request.question})
        
        return ChatResponse(
            answer=result.get("answer", "답변을 생성할 수 없습니다."),
            sources=None  # sources 정보를 반환하지 않음
        )
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
