# src/flight_booking_agent/config.py
import os
from dotenv import load_dotenv
# Thay đổi import
from langchain_google_vertexai import ChatVertexAI
from .tools.flight_tools import search_flights_tool


load_dotenv()
""" 
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# LLM thông thường để trò chuyện
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base=VLLM_BASE_URL,
    temperature=0.7
)

"""
llm = ChatVertexAI(
    model_name="gemini-2.5-pro", # Tên model vẫn giữ nguyên
    temperature=0.7,
    project="end-to-end-agentic-rag",  # Thay bằng ID dự án GCP của bạn
    location="us-central1",      # Ví dụ: "us-central1"
)

# LLM đặc biệt có khả năng sử dụng tools
llm_with_tools = llm.bind_tools([search_flights_tool])