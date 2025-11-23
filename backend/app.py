from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os

from chatbot_engine import ChatbotEngine

app = FastAPI(title="RIO Shop Chatbot API")

# Cấu hình CORS để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo chatbot engine
PERSIST_DIR = os.getenv("PERSIST_DIR", "../db")
try:
    chatbot = ChatbotEngine(persist_dir=PERSIST_DIR)
    print(f"✅ Chatbot engine initialized successfully from {PERSIST_DIR}")
except Exception as e:
    print(f"❌ Error initializing chatbot: {e}")
    chatbot = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    history: List[str]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "RIO Shop Chatbot API is running",
        "chatbot_ready": chatbot is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint xử lý chat với khách hàng.
    """
    if not chatbot:
        raise HTTPException(
            status_code=503,
            detail="Chatbot engine is not initialized"
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )
    
    try:
        answer, history = chatbot.chat(request.message)
        return ChatResponse(answer=answer, history=history)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@app.post("/reset")
async def reset_chat():
    """Reset lịch sử hội thoại."""
    if not chatbot:
        raise HTTPException(
            status_code=503,
            detail="Chatbot engine is not initialized"
        )
    
    try:
        chatbot.reset_chat()
        return {"status": "ok", "message": "Chat history reset successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting chat: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)