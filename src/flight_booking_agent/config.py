# src/flight_booking_agent/config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Tải các biến môi trường từ file .env
load_dotenv()

# Lấy base URL của vLLM (mặc định là http://localhost:8000/v1 nếu bạn không đổi)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# Khởi tạo mô hình LLM
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-1.5B-Instruct",   # Tên model bạn đã load trên vLLM
    openai_api_key="EMPTY",               # vLLM không cần key, cứ để "EMPTY"
    openai_api_base=VLLM_BASE_URL,        # URL trỏ đến server vLLM
    temperature=0.7
)
