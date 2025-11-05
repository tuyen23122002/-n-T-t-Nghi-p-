# src/flight_booking_agent/agents/information_gatherer.py
import json
from datetime import datetime, timedelta
from typing import Optional
# SỬA LỖI PYDANTIC: Import trực tiếp từ pydantic
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from ..config import llm
from ..graph.state import AgentState

from .utils import get_iata_code, convert_relative_date


# === THÊM HÀM MỚI TẠI ĐÂY ===
def general_node(state: AgentState) -> dict:
    """
    NODE GIẢ LẬP: Xử lý các câu hỏi chung.
    Trong tương lai, node này sẽ chứa logic để gọi RAG hoặc các tool kiến thức chung.
    """
    print("---NODE: GENERAL (Placeholder)---")
    
    # Hiện tại, chỉ trả về một tin nhắn thông báo chung
    response_message = AIMessage(content="Dạ, em có thể giúp gì khác cho anh/chị ạ?")
    
    return {
        "messages": [response_message],
        "previous_agent": "general_agent"
    }