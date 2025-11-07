from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage
import traceback

from src.flight_booking_agent.graph.workflow import app as graph_app

fastapi_app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str  
    
class ChatResponse(BaseModel):
    response: str
    thread_id: str

@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint xử lý chat với checkpoint.
    Mỗi thread_id đại diện cho 1 cuộc hội thoại riêng biệt.
    """
    try:
        # Config với thread_id để LangGraph biết load state nào
        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }
        
        # Invoke graph với checkpoint
        result = graph_app.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        # Lấy response cuối cùng
        last_message = result["messages"][-1]
        
        return ChatResponse(
            response=last_message.content,
            thread_id=request.thread_id
        )
        
    except Exception as e:
        # 2. Đảm bảo bạn có những dòng này để in lỗi chi tiết
        print("\n--- TRACEBACK LỖI CHI TIẾT ---")
        traceback.print_exc()
        print("------------------------------\n")
        
        # 3. Dòng raise HTTPException phải nằm cuối cùng
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/history/{thread_id}")
async def get_conversation_history(thread_id: str):
    """
    Lấy lịch sử hội thoại của một thread_id
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Lấy state hiện tại từ checkpoint
        state = graph_app.get_state(config)
        
        if state and state.values.get("messages"):
            messages = state.values["messages"]
            return {
                "thread_id": thread_id,
                "messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content
                    }
                    for msg in messages
                ]
            }
        else:
            return {"thread_id": thread_id, "messages": []}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.delete("/history/{thread_id}")
async def clear_conversation(thread_id: str):
    """
    Xóa lịch sử hội thoại của một thread_id
    """
    try:
        # Trong SQLite checkpoint, không có API trực tiếp để xóa
        # Bạn có thể tạo thread_id mới thay vì xóa
        return {"message": f"Để bắt đầu cuộc hội thoại mới, hãy dùng thread_id khác"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))